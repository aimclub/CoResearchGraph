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
    skip = {"mcp_url", "mcp_endpoint"}
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


_TOOL_RESULTS_KEY = "_em_tool_results"
_MAX_TOOL_RESULTS = 500
_MAX_TOOL_RESULT_BYTES = 200_000
_MATERIALIZABLE_ROLES = frozenset({"data", "report", "log"})


def _parse_structured(response: Any) -> Any:
    """A tool's structured payload: MCP content blocks or JSON text become dict/list."""
    if isinstance(response, Mapping) and isinstance(response.get("content"), list):
        text = "\n".join(
            str(block.get("text") or "")
            for block in response["content"]
            if isinstance(block, Mapping) and block.get("type", "text") == "text"
        ).strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError:
            return None
    if isinstance(response, (Mapping, list)):
        return response
    if isinstance(response, str):
        try:
            return json.loads(response)
        except ValueError:
            return None
    return None


def record_tool_result(
    state: MutableMapping[str, Any],
    *,
    attempt_id: str,
    tool: str,
    args: Mapping[str, Any] | None,
    response: Any,
) -> bool:
    """Keep a route agent's structured tool result for the attempt.

    MCP tools hand their numbers back in the response, not in files; without
    this record the only trace of a training run is the agent's prose.
    """
    parsed = _parse_structured(response)
    if not isinstance(parsed, (Mapping, list)) or not parsed:
        return False
    try:
        encoded = json.dumps(parsed, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return False
    if len(encoded) > _MAX_TOOL_RESULT_BYTES:
        return False
    existing = list(state.get(_TOOL_RESULTS_KEY) or [])
    if len(existing) >= _MAX_TOOL_RESULTS:
        return False
    existing.append({
        "attempt_id": str(attempt_id),
        "tool": str(tool),
        "args": dict(args or {}),
        "result": json.loads(encoded),
    })
    state[_TOOL_RESULTS_KEY] = existing
    return True


def _flatten(value: Any, prefix: str, out: dict[str, Any]) -> dict[str, Any]:
    """Nested mappings become dotted columns; lists stay as JSON text."""
    for key, item in value.items():
        column = f"{prefix}{key}"
        if isinstance(item, Mapping):
            _flatten(item, column + ".", out)
        elif isinstance(item, (list, tuple)):
            out[column] = json.dumps(item, ensure_ascii=False, default=str)
        else:
            out[column] = item
    return out


def _tool_result_rows(results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in results:
        row: dict[str, Any] = {"tool": entry.get("tool")}
        _flatten(entry.get("args") or {}, "", row)
        result = entry.get("result")
        if isinstance(result, Mapping):
            _flatten(result, "", row)
        else:
            row["result"] = json.dumps(result, ensure_ascii=False, default=str)
        rows.append(row)
    return rows


def _rows_csv(rows: list[dict[str, Any]]) -> str:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return buffer.getvalue()


def materialize_tool_results(
    state: MutableMapping[str, Any],
    *,
    task_id: str,
    attempt_id: str,
    expected_artifacts: list[Mapping[str, Any]],
    existing_names: set[str] | None = None,
    producer_tool: str = "route_tool_results",
) -> list[dict[str, Any]]:
    """Write the attempt's recorded tool results under the planner's artifact names.

    A react_tools attempt calls MCP tools that return metrics as JSON and has no
    tool to write files, while the planner names a table (``informer_metrics.csv``)
    and the evidence gate demands it. One row per tool call, flattened, as CSV
    or JSON by the expected name, so the gate, downstream ``task_artifact``
    inputs and the report all find what the run measured.
    """
    results = [
        r for r in (state.get(_TOOL_RESULTS_KEY) or [])
        if isinstance(r, Mapping) and str(r.get("attempt_id")) == str(attempt_id)
    ]
    if not results:
        return []
    present = set(existing_names or ())
    rows = _tool_result_rows(results)
    created: list[dict[str, Any]] = []
    for spec in expected_artifacts:
        if not isinstance(spec, Mapping):
            continue
        name = str(spec.get("name") or "").strip()
        role = str(spec.get("role") or "data")
        if not name or name in present or role not in _MATERIALIZABLE_ROLES:
            continue
        media_type = str(spec.get("media_type") or "")
        lowered = name.lower()
        if media_type == "text/csv" or lowered.endswith(".csv"):
            payload, media = _rows_csv(rows).encode("utf-8"), "text/csv"
        elif media_type in {"", "application/json", "text/plain", "text/markdown"} or lowered.endswith((".json", ".txt", ".md")):
            payload, media = json.dumps(rows, ensure_ascii=False, indent=2, default=str).encode("utf-8"), "application/json"
        else:
            continue
        artifact = _write_artifact(
            task_id=str(task_id),
            attempt_id=str(attempt_id),
            name=name,
            media_type=media,
            payload=payload,
            producer_tool=producer_tool,
            state=state,
        )
        if artifact:
            present.add(name)
            created.append({"name": name, "rows": len(rows), "workspace_path": artifact["workspace_path"]})
    return created


__all__ = [
    "materialize_inline_result",
    "materialize_outputs_as_artifacts",
    "materialize_tool_results",
    "record_tool_result",
]
