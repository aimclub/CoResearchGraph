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
    candidates: set[str] = set()
    for text in _strings(result):
        for match in _FENCED_BLOCK.finditer(text):
            if match.group("kind").lower() == "csv" and _valid_csv(body := match.group("body").strip()):
                candidates.add(body + "\n")
        if _valid_csv(raw := text.strip()):
            candidates.add(raw + "\n")
    return next(iter(candidates)) if len(candidates) == 1 else None


def _encode_payload(
    value: Any,
    *,
    name: str,
    media_type: str | None,
) -> tuple[bytes, str] | None:
    if media_type == "text/csv" or name.lower().endswith(".csv"):
        text = _csv_payload(value)
        return (text.encode("utf-8"), "text/csv") if text is not None else None
    if media_type == "application/json" or name.lower().endswith(".json"):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return None
        if not isinstance(value, (dict, list)):
            return None
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
    ext = "" if Path(name).suffix else _EXTENSIONS.get(media_type, ".json")
    folder = Path(get_settings().code_exec.workspace_root) / "experiment_artifacts" / str(task_id) / str(attempt_id)
    destination = folder / (Path(name).name + ext)
    # The executor's description must never replace bytes already delivered
    # by the tool or Coder for this exact task/attempt/output.
    if destination.is_file():
        payload = destination.read_bytes()
    else:
        if not payload or len(payload) > _MAX_INLINE_BYTES:
            return None
        folder.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(payload)
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
    output_name: str | None = None,
) -> list[dict[str, Any]]:
    """Persist one unambiguous expected artifact from a structured route result."""
    runtime = state.get("experiment_runtime") or {}
    task_id, attempt_id = runtime.get("active_task_id"), runtime.get("active_attempt_id")
    expected = (((runtime.get("tasks") or {}).get(task_id) or {}).get("task") or {}).get("expected_artifacts") or []
    if output_name is not None:
        expected = [item for item in expected if item.get("name") == output_name]
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


def capture_structured_tool_result(
    state: MutableMapping[str, Any], tool: str, data: Any,
    arguments: Mapping[str, Any] | None = None,
) -> None:
    """Preserve every response; bind the complete collection when the route returns."""
    runtime = state.get("experiment_runtime") or {}
    tid, aid = runtime.get("active_task_id"), runtime.get("active_attempt_id")
    task = ((runtime.get("tasks") or {}).get(tid) or {}).get("task") or {}
    declared = {t.get("name") for s in task.get("mcp_servers") or [] for t in s.get("tools") or []}
    if not tid or not aid or tool not in declared or not isinstance(data, (dict, list)):
        return
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    raw = _write_artifact(task_id=tid, attempt_id=aid,
        name="tool-response-" + hashlib.sha256(payload).hexdigest()[:16] + ".json",
        media_type="application/json", payload=payload, producer_tool=tool)
    if raw:
        raw["role"] = "log"  # An unbound response is not an ordered output.
        state["mcp_artifacts"] = [*(state.get("mcp_artifacts") or []), raw]
        attempt = runtime["tasks"][tid]["attempts"][aid]
        attempt.setdefault("tool_response_artifacts", []).append({
            "tool": tool, "arguments": dict(arguments or {}), "path": raw["workspace_path"],
        })


def materialize_bound_tool_results(state: MutableMapping[str, Any], attempt: Mapping[str, Any]) -> None:
    """An explicit output binding covers all its tool calls in this attempt."""
    runtime = state.get("experiment_runtime") or {}
    task = runtime["tasks"][runtime["active_task_id"]]["task"]
    bindings: dict[str, set[str]] = {}
    for binding in (task.get("design") or {}).get("analysis_artifacts") or []:
        bindings.setdefault(binding["name"], set()).add(binding.get("path_or_tool") or "")
    for name, tools in bindings.items():
        responses = []
        for row in attempt.get("tool_response_artifacts") or []:
            if row["tool"] in tools:
                responses.append({
                    "tool": row["tool"], "arguments": row["arguments"],
                    "response": json.loads(Path(row["path"]).read_text(encoding="utf-8")),
                })
        if responses:
            data = responses[0]["response"] if len(responses) == 1 else {"tool_responses": responses}
            materialize_inline_result(state, data, producer_tool="captured_tool_results", output_name=name)


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
    if not expected_artifacts:
        return []
    present = {str(item.get("name") or "") for item in (existing or []) if isinstance(item, Mapping)}
    created: list[dict[str, Any]] = []
    outputs = dict(outputs or {})
    if "mcp_endpoint" not in outputs and "mcp_url" in outputs:
        outputs["mcp_endpoint"] = outputs["mcp_url"]
    if "mcp_url" not in outputs and "mcp_endpoint" in outputs:
        outputs["mcp_url"] = outputs["mcp_endpoint"]
    for spec in expected_artifacts:
        name = str(spec.get("name") or "")
        if not name or name in present:
            continue
        encoded = _encode_payload(outputs[name], name=name, media_type=spec.get("media_type")) if name in outputs else None
        payload, media_type = encoded or (b"", spec.get("media_type") or "application/json")
        if artifact := _write_artifact(
            task_id=task_id,
            attempt_id=attempt_id,
            name=name,
            media_type=media_type,
            payload=payload,
            producer_tool=producer_tool,
        ):
            artifact.update(role=spec.get("role") or "data", output_id=spec.get("output_id"))
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
