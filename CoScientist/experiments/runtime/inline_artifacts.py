"""Materialize structured inline route results as workspace artifacts."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from CoScientist.config import get_settings

_FENCED_BLOCK = re.compile(r"```(?P<kind>[a-zA-Z0-9_-]*)\s*\n(?P<body>.*?)```", re.DOTALL)
_MAX_INLINE_BYTES = 10_000_000
_EXTENSIONS = {
    "text/csv": ".csv",
    "application/json": ".json",
    "text/plain": ".txt",
    "text/markdown": ".md",
}


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _valid_csv(text: str) -> bool:
    try:
        reader = csv.reader(io.StringIO(text))
        return len(next(reader)) >= 2 and next(reader, None) is not None
    except (csv.Error, StopIteration):
        return False


def _csv_payload(result: Any) -> str | None:
    for text in _strings(result):
        for match in _FENCED_BLOCK.finditer(text):
            if match.group("kind").lower() == "csv" and _valid_csv(body := match.group("body").strip()):
                return body + "\n"
        if _valid_csv(raw := text.strip()):
            return raw + "\n"
    return None


def _encode_payload(
    value: Any,
    *,
    name: str,
    media_type: str | None,
) -> tuple[bytes, str] | None:
    if media_type == "text/csv" or name.lower().endswith(".csv"):
        text = _csv_payload(value)
        return (text.encode("utf-8"), "text/csv") if text is not None else None
    try:
        return (
            json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
            media_type or "application/json",
        )
    except (TypeError, ValueError):
        return None


def has_structured_family_outputs(outputs: Mapping[str, Any] | None) -> bool:
    """True when MCP/Fedot returned computable family evidence, not a status string.

    A single scalar (``n_molecules=10``) is not enough — generate/dock still
    need an S3/file table. Two+ numeric fields or a non-empty list/object is.
    """
    if not isinstance(outputs, Mapping) or not outputs:
        return False
    # Diagnostics explain why a route could not execute; persisting them is
    # useful, but they are not the requested scientific output and must not
    # suppress fallback to another route.
    skip = {
        "mcp_url", "mcp_endpoint", "tool_limitation", "tool_limitations",
        "limitation", "limitations", "diagnostic", "diagnostics",
        "wrong_dataset", "dataset_mismatch",
    }
    items = {k: v for k, v in outputs.items() if str(k) not in skip}
    if not items:
        return False

    def _rich(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, (list, tuple, Mapping)) and value:
            return True
        return False

    rich = [v for v in items.values() if _rich(v)]
    if not rich:
        return False
    if any(isinstance(v, (list, tuple, Mapping)) and v for v in rich):
        return True
    return len(items) >= 2


def _write_artifact(
    *,
    task_id: str,
    attempt_id: str,
    name: str,
    media_type: str,
    payload: bytes,
    producer_tool: str,
    state: MutableMapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not payload or len(payload) > _MAX_INLINE_BYTES:
        return None
    ext = "" if Path(name).suffix else _EXTENSIONS.get(media_type, ".json")
    folder = Path(get_settings().code_exec.workspace_root) / "experiment_artifacts" / str(task_id) / str(attempt_id)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / (Path(name).name + ext)
    destination.write_bytes(payload)
    artifact = {
        "name": name,
        "role": "data",
        "workspace_path": str(destination.resolve()),
        "media_type": media_type,
        "size_bytes": len(payload),
        "checksum_sha256": hashlib.sha256(payload).hexdigest(),
        "durability": "workspace",
        "tool": producer_tool,
    }
    if state is not None:
        state["fedot_artifacts"] = [*(state.get("fedot_artifacts") or []), artifact]
    return artifact


def materialize_inline_result(
    state: MutableMapping[str, Any],
    result: Any,
    *,
    producer_tool: str = "fedot_inline_result",
) -> list[dict[str, Any]]:
    """Persist one unambiguous expected artifact from a structured route result."""
    runtime = state.get("experiment_runtime") or {}
    task_id, attempt_id = runtime.get("active_task_id"), runtime.get("active_attempt_id")
    expected = (((runtime.get("tasks") or {}).get(task_id) or {}).get("task") or {}).get("expected_artifacts") or []
    if not task_id or not attempt_id or result is None or len(expected) != 1:
        return []

    spec = expected[0]
    name = str(spec.get("name") or "route-result")
    media_type = spec.get("media_type")
    if media_type == "text/csv" or name.lower().endswith(".csv"):
        if (text := _csv_payload(result)) is None:
            return []
        payload, media_type = text.encode("utf-8"), "text/csv"
    else:
        if (encoded := _encode_payload(result, name=name, media_type=media_type)) is None:
            return []
        payload, media_type = encoded
    artifact = _write_artifact(
        task_id=str(task_id),
        attempt_id=str(attempt_id),
        name=name,
        media_type=media_type,
        payload=payload,
        producer_tool=producer_tool,
        state=state,
    )
    return [artifact] if artifact else []


def materialize_outputs_as_artifacts(
    *,
    task_id: str,
    attempt_id: str,
    expected_artifacts: list[Mapping[str, Any]],
    outputs: Mapping[str, Any] | None,
    existing: list[Mapping[str, Any]] | None = None,
    producer_tool: str = "record_result_outputs",
) -> list[dict[str, Any]]:
    """Persist expected artifacts already present under result.outputs."""
    if not outputs or not expected_artifacts:
        return []
    present = {str(item.get("name") or "") for item in (existing or []) if isinstance(item, Mapping)}
    created: list[dict[str, Any]] = []
    outputs = dict(outputs)
    if "mcp_endpoint" not in outputs and "mcp_url" in outputs:
        outputs["mcp_endpoint"] = outputs["mcp_url"]
    if "mcp_url" not in outputs and "mcp_endpoint" in outputs:
        outputs["mcp_url"] = outputs["mcp_endpoint"]
    for spec in expected_artifacts:
        name = str(spec.get("name") or "")
        if not name or name in present or name not in outputs:
            continue
        if (encoded := _encode_payload(outputs[name], name=name, media_type=spec.get("media_type"))) is None:
            continue
        payload, media_type = encoded
        if artifact := _write_artifact(
            task_id=task_id,
            attempt_id=attempt_id,
            name=name,
            media_type=media_type,
            payload=payload,
            producer_tool=producer_tool,
        ):
            created.append(artifact)
            present.add(name)
    # Persist the whole outputs blob when the planner invented a filename the
    # MCP never used (cluster_assignments.csv vs total_clusters_identified).
    family_name = "family_outputs.json"
    if family_name not in present and has_structured_family_outputs(outputs):
        encoded = _encode_payload(dict(outputs), name=family_name, media_type="application/json")
        if encoded is not None:
            payload, media_type = encoded
            if artifact := _write_artifact(
                task_id=task_id,
                attempt_id=attempt_id,
                name=family_name,
                media_type=media_type,
                payload=payload,
                producer_tool=producer_tool,
            ):
                created.append(artifact)
    return created


__all__ = [
    "materialize_inline_result",
    "materialize_outputs_as_artifacts",
]
