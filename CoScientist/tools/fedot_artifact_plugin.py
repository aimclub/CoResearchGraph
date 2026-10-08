"""Deterministic capture of S3 artifact links produced by MCP tools inside a FEDOT.MAS run.

Why (F010.A3): mol-gen / ML MCP tools (`generate_mols`, `generate_case_mols`,
`predict_ml`, …) are remote and do not return their data inline — they upload it to S3
and return a *presigned URL* to a results CSV. The FEDOT.MAS sub-agent (an ADK
``LlmAgent``) keeps only its free-text paraphrase under ``output_key``, so the raw
structured link is dropped — and the sub-agent can hallucinate SMILES that were never in
the tool output.

Fix seam (no fedotmas fork): ``MAS(plugins=[...])`` is threaded to the ADK ``Runner``.
This ``BasePlugin.after_tool_callback`` fires at the tool-call boundary, BEFORE the LLM
paraphrase, and stashes every S3 link it sees into its own ``captured`` list (owned by
the caller — no reliance on cross-stage session-state merge).

ADK passes ``after_tool_callback`` the result of
``CallToolResult.model_dump(exclude_none=True, mode="json")`` (mcp_tool.py) — i.e. the
WRAPPED MCP envelope ``{content:[{text:<json>}], structuredContent:{...}, isError}``,
NOT the tool's top-level dict (F010.A4). The link therefore lives under
``structuredContent`` (or as a JSON string inside ``content[].text``), so the extractor
searches RECURSIVELY for any dict carrying a ``*presigned_url`` key, parsing JSON-looking
strings on the way down.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.plugins import BasePlugin


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


class ArtifactCapturePlugin(BasePlugin):
    """Capture S3 artifact links from tool results before the LLM can drop them."""

    def __init__(self, name: str = "artifact_capture") -> None:
        super().__init__(name)
        self.captured: list[dict] = []

    def _add(self, art: dict) -> None:
        key = (art.get("tool"), art.get("s3_key") or art.get("url"))
        if art.get("url") and key not in {
            (a.get("tool"), a.get("s3_key") or a.get("url")) for a in self.captured
        }:
            self.captured.append(art)

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):  # noqa: ANN001
        tool_name = getattr(tool, "name", None)
        found: list = []
        _walk_for_s3(result, found)
        for d in found:
            self._add(_normalize(d, tool_name))
        return None  # never mutate the tool result itself
