"""Deterministic capture of MCP tool artifacts produced inside a FEDOT.MAS run.

Why (F010.A3): mol-gen / ML MCP tools (`generate_mols`, `generate_case_mols`,
`predict_ml`, …) are remote and do not return their data inline — they upload it to S3
and return a *presigned URL* to a results CSV. The FEDOT.MAS sub-agent (an ADK
``LlmAgent``) keeps only its free-text paraphrase under ``output_key``, so the raw
structured link is dropped — and the sub-agent can hallucinate SMILES that were never in
the tool output.

Alembic / synspace-style tools return the payload *inline*
(``structuredContent.molecules`` / JSON in ``content[].text``). Those are captured
here as durable workspace CSV/JSON **before** the LLM paraphrase, for the same reason.

Fix seam (no fedotmas fork): ``MAS(plugins=[...])`` is threaded to the ADK ``Runner``.
This ``BasePlugin.after_tool_callback`` fires at the tool-call boundary, BEFORE the LLM
paraphrase, and stashes every S3 link / inline molecule table it sees into its own
``captured`` list (owned by the caller — no reliance on cross-stage session-state merge).

ADK passes ``after_tool_callback`` the result of
``CallToolResult.model_dump(exclude_none=True, mode="json")`` (mcp_tool.py) — i.e. the
WRAPPED MCP envelope ``{content:[{text:<json>}], structuredContent:{...}, isError}``,
NOT the tool's top-level dict (F010.A4). The link therefore lives under
``structuredContent`` (or as a JSON string inside ``content[].text``), so the extractor
searches RECURSIVELY for any dict carrying a ``*presigned_url`` key, parsing JSON-looking
strings on the way down.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable

from google.adk.plugins import BasePlugin


def _dedup_key(art: dict) -> tuple:
    location = art.get("s3_key") or art.get("workspace_path") or art.get("url")
    return (art.get("tool"), location)


def merge_artifacts(existing: list[dict], new: Iterable[dict]) -> list[dict]:
    """Single source of artifact de-duplication (by ``tool`` + ``s3_key``/``url``).

    Shared between the plugin (within one FEDOT run) and ``fedotmas_tools``
    (across sequential ``fedot_tool`` calls in the same session) so both use
    the exact same identity rule.
    """
    merged = list(existing)
    seen = {_dedup_key(a) for a in merged}
    for art in new:
        if not (art.get("url") or art.get("s3_key") or art.get("workspace_path")):
            continue
        key = _dedup_key(art)
        if key not in seen:
            merged.append(art)
            seen.add(key)
    return merged


def _is_artifact_dict(d: dict) -> bool:
    return any(
        k.endswith("presigned_url") and isinstance(v, str) and v
        for k, v in d.items()
    )


def _walk_for_s3(obj: Any, found: list) -> None:
    """Recursively collect dicts that carry a ``*presigned_url`` key. Handles ADK's
    CallToolResult envelope and JSON-string content blocks."""
    if isinstance(obj, dict):
        if _is_artifact_dict(obj):
            found.append(obj)
        for v in obj.values():
            _walk_for_s3(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _walk_for_s3(v, found)
    elif isinstance(obj, str) and "presigned_url" in obj:
        try:
            _walk_for_s3(json.loads(obj), found)
        except Exception:
            pass


def _is_molecule_payload(d: dict) -> bool:
    molecules = d.get("molecules")
    return isinstance(molecules, list) and bool(molecules) and all(
        isinstance(item, str) and item.strip() for item in molecules
    )


def _walk_for_molecules(obj: Any, found: list) -> None:
    """Collect dicts with a non-empty ``molecules: list[str]`` (synspace-style MCP)."""
    if isinstance(obj, dict):
        if _is_molecule_payload(obj):
            found.append(obj)
        for v in obj.values():
            _walk_for_molecules(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _walk_for_molecules(v, found)
    elif isinstance(obj, str) and "molecules" in obj:
        try:
            _walk_for_molecules(json.loads(obj), found)
        except Exception:
            pass


def _normalize(d: dict, tool_name: str | None) -> dict:
    url = next((v for k, v in d.items() if k.endswith("presigned_url") and v), None)
    return {
        "url": url,
        "s3_key": d.get("results_s3_key") or d.get("s3_key"),
        "bucket": d.get("bucket_name") or d.get("bucket"),
        "columns": d.get("columns"),
        "generated_count": d.get("generated_count"),
        "case": d.get("case"),
        "tool": tool_name,
    }


def molecules_to_csv(payload: dict) -> str:
    """Serialize synspace-style ``{molecules, properties}`` to CSV text."""
    molecules = list(payload.get("molecules") or [])
    properties = payload.get("properties") if isinstance(payload.get("properties"), list) else []
    rows: list[dict[str, Any]] = []
    for i, smiles in enumerate(molecules):
        prop = properties[i] if i < len(properties) and isinstance(properties[i], dict) else {}
        row: dict[str, Any] = {"smiles": smiles}
        if "similarity" in prop:
            row["score"] = prop["similarity"]
        elif "score" in prop:
            row["score"] = prop["score"]
        if prop.get("rxn-name") or prop.get("rxn_name"):
            row["rxn_name"] = prop.get("rxn-name") or prop.get("rxn_name")
        if prop.get("rxn"):
            row["rxn"] = prop["rxn"]
        rows.append(row)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames or ["smiles"])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _write_molecule_csv(
    payload: dict,
    *,
    tool_name: str | None,
    tool_context: Any,
) -> dict | None:
    """Persist inline MCP molecules as candidates.csv in the experiment workspace."""
    state = getattr(tool_context, "state", None)
    if not isinstance(state, dict):
        return None
    runtime = state.get("experiment_runtime") or {}
    task_id = runtime.get("active_task_id")
    attempt_id = runtime.get("active_attempt_id")
    if not task_id or not attempt_id:
        return None

    csv_text = molecules_to_csv(payload)
    raw = csv_text.encode("utf-8")
    if not raw:
        return None

    from CoScientist.config import get_settings

    folder = (
        Path(get_settings().code_exec.workspace_root)
        / "experiment_artifacts"
        / str(task_id)
        / str(attempt_id)
    )
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "candidates.csv"
    destination.write_bytes(raw)
    return {
        "name": "candidates.csv",
        "workspace_path": str(destination.resolve()),
        "media_type": "text/csv",
        "size_bytes": len(raw),
        "checksum_sha256": hashlib.sha256(raw).hexdigest(),
        "durability": "workspace",
        "tool": tool_name or "mcp_inline_molecules",
        "generated_count": len(payload.get("molecules") or []),
    }


class ArtifactCapturePlugin(BasePlugin):
    """Capture S3 links and inline molecule tables before the LLM can drop them."""

    def __init__(self, name: str = "artifact_capture") -> None:
        super().__init__(name)
        self.captured: list[dict] = []
        self.inline_molecule_payloads: list[dict] = []

    def _add(self, art: dict) -> None:
        self.captured = merge_artifacts(self.captured, [art])

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):  # noqa: ANN001
        tool_name = getattr(tool, "name", None)
        found: list = []
        _walk_for_s3(result, found)
        for d in found:
            self._add(_normalize(d, tool_name))

        molecule_payloads: list = []
        _walk_for_molecules(result, molecule_payloads)
        for payload in molecule_payloads:
            self.inline_molecule_payloads.append(
                {"tool": tool_name, "payload": payload, "tool_args": tool_args}
            )
            art = _write_molecule_csv(payload, tool_name=tool_name, tool_context=tool_context)
            if art is not None:
                self._add(art)
        return None  # never mutate the tool result itself
