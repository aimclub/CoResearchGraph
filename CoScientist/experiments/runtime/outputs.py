"""Logical outputs, the files that satisfy them, and repair of that binding.

A consumer names an output of a task. The artifact id and the filename are
facts discovered later. Success, readiness and launch all resolve through
``resolve_output`` so those three cannot disagree.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from CoScientist.experiments.runtime.errors import ExperimentRuntimeError
from CoScientist.experiments.schemas.models import logical_output_id

BINDINGS_KEY = "output_bindings"
PINS_KEY = "input_pins"
REPAIRS_KEY = "output_repairs"
RECOVERY_BUDGET = 3

REPAIRABLE_BLOCK_CODES = frozenset({
    "required_upstream_artifact_missing",
    "materialization_unavailable",
    "repairable_dependency_blocked",
})

_URL_FIELDS = frozenset({
    "url", "uri", "href", "presigned_url", "external_url", "s3_url",
    "download_url", "results_url",
})
_TABLE_SUFFIXES = frozenset({".csv", ".tsv"})
_JSON_SUFFIXES = frozenset({".json"})


def is_repairable_block(task_runtime: Mapping[str, Any] | None) -> bool:
    reason = (task_runtime or {}).get("blocked_reason") or {}
    return (
        str((task_runtime or {}).get("status") or "") == "blocked"
        and str(reason.get("code") or "") in REPAIRABLE_BLOCK_CODES
    )


def _binding_key(task_id: str, output_id: str) -> str:
    return f"{task_id}/{output_id}"


def _pin_key(consumer_task_id: str, data_id: str) -> str:
    return f"{consumer_task_id}/{data_id}"


def _artifacts_of(runtime: Mapping[str, Any], task_id: str | None) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for result in runtime.get("results") or []:
        if not isinstance(result, dict):
            continue
        if task_id and result.get("task_id") != task_id:
            continue
        if result.get("status") not in {"success", "partial"}:
            continue
        for artifact in result.get("artifacts") or []:
            if isinstance(artifact, dict):
                found.append(artifact)
    return found


def _by_id(runtime: Mapping[str, Any], artifact_id: str) -> dict[str, Any] | None:
    for result in runtime.get("results") or []:
        if not isinstance(result, dict):
            continue
        for artifact in result.get("artifacts") or []:
            if isinstance(artifact, dict) and artifact.get("artifact_id") == artifact_id:
                return artifact
    return None


def _expects_json(expected: Any) -> bool:
    name = str(getattr(expected, "name", "") or "")
    media = str(getattr(expected, "media_type", "") or "").lower()
    return Path(name).suffix.lower() in _JSON_SUFFIXES or media == "application/json"


def _expects_table(expected: Any) -> bool:
    name = str(getattr(expected, "name", "") or "")
    media = str(getattr(expected, "media_type", "") or "").lower()
    role = str(getattr(expected, "role", "") or "")
    suffix = Path(name).suffix.lower()
    if suffix in _TABLE_SUFFIXES or media in {"text/csv", "text/tab-separated-values"}:
        return True
    return role == "data" and bool(getattr(expected, "columns", None))


def _local_path(artifact: Mapping[str, Any]) -> Path | None:
    text = str(artifact.get("workspace_path") or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_file() else None


def json_is_pointer(payload: Any) -> bool:
    """A URL, or an object that only points at one, is not the dataset."""
    if isinstance(payload, str):
        return payload.strip().startswith(("http://", "https://"))
    if not isinstance(payload, dict) or not payload:
        return False
    keys = {str(key).lower() for key in payload}
    urls = [
        value for key, value in payload.items()
        if str(key).lower() in _URL_FIELDS
        and isinstance(value, str)
        and value.strip().startswith(("http://", "https://"))
    ]
    if not urls:
        return False
    return keys <= (_URL_FIELDS | {"bucket", "key", "s3_key", "name", "filename"})


def _read_table(path: Path, *, suffix: str) -> tuple[list[str], list[list[str]]]:
    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as stream:
        try:
            payload = json.load(stream)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, (dict, list)):
            raise ValueError("a JSON document is not a delimited table")
        stream.seek(0)
        rows = list(csv.reader(stream, delimiter=delimiter))
    if not rows:
        return [], []
    if any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError("table rows do not match the header width")
    return [cell.strip() for cell in rows[0]], rows[1:]


def _required_fields(expected: Any) -> list[str]:
    return [str(item) for item in (getattr(expected, "columns", None) or []) if str(item).strip()]


def _json_structure_problem(payload: Any, expected: Any) -> str | None:
    """Opened JSON must be the declared object or table, not a pointer to one."""
    if json_is_pointer(payload):
        return "JSON pointer is not the required table"
    fields = _required_fields(expected)
    allow_empty = bool(getattr(expected, "allow_empty", False))
    if isinstance(payload, list):
        if not payload and not allow_empty:
            return "table is empty"
        if not fields or not payload:
            return None
        if any(not isinstance(row, dict) for row in payload):
            return "JSON array items are not objects with the required fields"
        missing = [field for field in fields if any(field not in row for row in payload)]
        if missing:
            return f"missing fields: {', '.join(missing)}"
        return None
    if isinstance(payload, dict):
        if not fields:
            return None
        missing = [field for field in fields if field not in payload]
        if missing:
            return f"missing fields: {', '.join(missing)}"
        return None
    return "JSON is not an object or array"


def contract_problem(artifact: Mapping[str, Any], expected: Any) -> str | None:
    """Why this file does not satisfy the output, or None when it does.

    A managed ``bucket/key`` without bytes is a location. It does not prove
    the data was read. A bare URL is not a table. A JSON document that only
    holds a URL is a manifest, not the table it points at. Declared JSON is
    checked for object-or-array structure and for the required fields.
    """
    wants_json = _expects_json(expected)
    wants_table = _expects_table(expected) or wants_json
    has_s3 = bool(artifact.get("bucket") and artifact.get("s3_key"))
    path = _local_path(artifact)
    url_only = bool(artifact.get("external_url")) and path is None and not has_s3
    if wants_table and url_only:
        return "a URL is not the required table"
    fields = _required_fields(expected)
    if path is None:
        # Structural contracts must be opened. A storage key is only a place.
        if has_s3 and (fields or wants_json):
            return "bucket/key is a location, not verified data"
        if fields:
            return "declared columns were not read from the artifact"
        if wants_json:
            return "JSON was not read from the artifact"
        return None
    suffix = path.suffix.lower()
    try:
        # A serialized MCP endpoint is a URL, not a dataset. Only apply the
        # inferred JSON data contract to data artifacts (or declared JSON).
        if wants_json or (suffix in _JSON_SUFFIXES and getattr(expected, "role", None) == "data"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _json_structure_problem(payload, expected)
        if wants_table or suffix in _TABLE_SUFFIXES:
            header, body = _read_table(path, suffix=suffix or ".csv")
            if not header:
                return "table has no header"
            if not body and not getattr(expected, "allow_empty", False):
                return "table is empty"
            if fields:
                missing = [col for col in fields if col not in header]
                if missing:
                    return f"missing columns: {', '.join(missing)}"
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, ValueError):
        return "artifact content is not readable as the declared contract"
    return None


def content_verified(artifact: Mapping[str, Any], expected: Any) -> bool:
    """True only when bytes were opened and matched the contract."""
    path = _local_path(artifact)
    if path is None:
        return False
    try:
        with path.open("rb") as stream:
            stream.read(1)
    except OSError:
        return False
    return contract_problem(artifact, expected) is None


def convert_csv_workspace_to_json(workspace_path: str, *, dest_name: str) -> str | None:
    """Write a real JSON array. Renaming ``.csv`` to ``.json`` is not this."""
    source = Path(workspace_path)
    if not source.is_file():
        return None
    try:
        header, body = _read_table(source, suffix=source.suffix.lower() or ".csv")
    except (OSError, UnicodeError, csv.Error, ValueError):
        return None
    if not header:
        return None
    records = [
        dict(zip(header, row, strict=True))
        for row in body
    ]
    dest = source.with_name(dest_name if dest_name.endswith(".json") else f"{dest_name}.json")
    dest.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return str(dest)


def publish_bindings(
    runtime: dict[str, Any],
    *,
    task_id: str,
    expected: list[Any],
    artifacts: list[Mapping[str, Any]],
    result_id: str,
    result_version: int,
    attempt_id: str,
) -> list[str]:
    """Bind each expected output to one artifact of this attempt.

    Two matches are an error. Bytes may be shared across artifacts; the
    binding of this result stays its own.
    """
    warnings: list[str] = []
    slot = runtime.setdefault(BINDINGS_KEY, {})
    claimed: set[str] = set()
    for item in expected:
        output_id = str(getattr(item, "output_id", "") or logical_output_id(getattr(item, "name", "")))
        hits = [
            art for art in artifacts
            if str(art.get("artifact_id") or "") not in claimed
            and (
                str(art.get("output_id") or "") == output_id
                or str(art.get("name") or "") == str(getattr(item, "name", ""))
            )
        ]
        verified = [art for art in hits if contract_problem(art, item) is None]
        if len(verified) > 1:
            raise ExperimentRuntimeError(
                "output_ambiguous",
                f"Output {task_id}/{output_id} matches {len(verified)} artifacts.",
            )
        if not verified:
            if getattr(item, "required", True):
                warnings.append(f"output {output_id} has no artifact")
            continue
        art = verified[0]
        claimed.add(str(art.get("artifact_id") or ""))
        key = _binding_key(task_id, output_id)
        previous = slot.get(key) if isinstance(slot.get(key), dict) else None
        if (
            isinstance(previous, dict)
            and previous.get("artifact_id") == art.get("artifact_id")
            and previous.get("attempt_id") == attempt_id
        ):
            continue
        history = list((previous or {}).get("history") or [])
        if isinstance(previous, dict) and previous.get("artifact_id"):
            history.append({
                "artifact_id": previous.get("artifact_id"),
                "result_version": previous.get("result_version"),
                "attempt_id": previous.get("attempt_id"),
            })
        slot[key] = {
            "task_id": task_id,
            "output_id": output_id,
            "preferred_name": str(getattr(item, "name", "") or ""),
            "artifact_id": art.get("artifact_id"),
            "result_id": result_id,
            "result_version": result_version,
            "attempt_id": attempt_id,
            "current": True,
            "history": history,
        }
    return warnings


def resolve_output(
    runtime: Mapping[str, Any],
    *,
    source_task_id: str | None,
    source_output_id: str | None = None,
    source_artifact_id: str | None = None,
    consumer_task_id: str | None = None,
    data_id: str | None = None,
) -> dict[str, Any]:
    """One lookup for success checks, readiness and task launch.

    A pin wins over a later producer attempt. Several unbound matches are an
    error; the previous 'take the last one' rule is not used.
    """
    if consumer_task_id and data_id:
        pin = (runtime.get(PINS_KEY) or {}).get(_pin_key(consumer_task_id, data_id))
        if isinstance(pin, dict) and pin.get("artifact_id"):
            pinned = _by_id(runtime, str(pin["artifact_id"]))
            if pinned is not None:
                return pinned

    token = str(source_artifact_id or "").strip()
    if token.startswith("ART-"):
        found = _by_id(runtime, token)
        if found is None:
            raise ExperimentRuntimeError("artifact_not_found", f"Required artifact {token!r} does not exist.")
        return found

    output_id = str(source_output_id or "").strip()
    if not output_id and token and token != "artifact":
        output_id = logical_output_id(token)

    bindings = runtime.get(BINDINGS_KEY) or {}
    bound_hits: list[dict[str, Any]] = []
    if isinstance(bindings, dict):
        for binding in bindings.values():
            if not isinstance(binding, dict) or not binding.get("current", True):
                continue
            if source_task_id and binding.get("task_id") != source_task_id:
                continue
            preferred = str(binding.get("preferred_name") or "")
            if output_id and binding.get("output_id") == output_id:
                bound_hits.append(binding)
            elif token and (preferred == token or binding.get("output_id") == logical_output_id(token)):
                bound_hits.append(binding)
            elif token:
                artifact = _by_id(runtime, str(binding.get("artifact_id") or ""))
                if artifact and token in {
                    artifact.get("name"), artifact.get("artifact_id"), artifact.get("session_artifact_id"),
                }:
                    bound_hits.append(binding)
    if len(bound_hits) > 1:
        distinct = {str(item.get("output_id") or "") for item in bound_hits}
        if len(distinct) != 1:
            raise ExperimentRuntimeError(
                "output_ambiguous",
                f"Input {token or output_id!r} matches outputs {sorted(distinct)}.",
            )
    if len(bound_hits) == 1:
        found = _by_id(runtime, str(bound_hits[0].get("artifact_id") or ""))
        if found is not None:
            return found

    matches: list[dict[str, Any]] = []
    for artifact in _artifacts_of(runtime, source_task_id):
        name = str(artifact.get("name") or "")
        art_output = str(artifact.get("output_id") or "")
        if token and (artifact.get("artifact_id") == token or name == token or Path(name).name == Path(token).name):
            matches.append(artifact)
        elif output_id and art_output == output_id:
            matches.append(artifact)
        elif token and art_output and art_output == logical_output_id(token):
            matches.append(artifact)
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for artifact in matches:
        aid = str(artifact.get("artifact_id") or "")
        if aid in seen:
            continue
        seen.add(aid)
        unique.append(artifact)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise ExperimentRuntimeError(
            "output_ambiguous",
            f"Input {token or output_id!r} matches {len(unique)} artifacts.",
        )
    label = token or output_id or ""
    raise ExperimentRuntimeError("artifact_not_found", f"Required artifact {label!r} does not exist.")


def pin_input(
    runtime: dict[str, Any],
    *,
    consumer_task_id: str,
    data_id: str,
    artifact: Mapping[str, Any],
    source_task_id: str | None,
    source_output_id: str | None,
) -> None:
    """Freeze the artifact version a consumer actually started with."""
    pins = runtime.setdefault(PINS_KEY, {})
    key = _pin_key(consumer_task_id, data_id)
    if key in pins:
        return
    pins[key] = {
        "consumer_task_id": consumer_task_id,
        "data_id": data_id,
        "source_task_id": source_task_id,
        "source_output_id": source_output_id or artifact.get("output_id"),
        "artifact_id": artifact.get("artifact_id"),
        "attempt_id": artifact.get("attempt_id"),
    }


def pinned_artifact_ids(runtime: Mapping[str, Any]) -> set[str]:
    pins = runtime.get(PINS_KEY) or {}
    if not isinstance(pins, Mapping):
        return set()
    return {
        str(pin.get("artifact_id"))
        for pin in pins.values()
        if isinstance(pin, Mapping) and pin.get("artifact_id")
    }


def _task_dump(row: Mapping[str, Any]) -> dict[str, Any]:
    task = row.get("task")
    return task if isinstance(task, dict) else {}


def _input_producers(row: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    for ref in _task_dump(row).get("input_data") or []:
        if isinstance(ref, Mapping) and str(ref.get("source_task_id") or "").strip():
            found.append(str(ref["source_task_id"]).strip())
    return found


def _dependency_ids(row: Mapping[str, Any]) -> list[str]:
    return [str(dep) for dep in (_task_dump(row).get("depends_on") or []) if str(dep).strip()]


def _completed_result(runtime: Mapping[str, Any], task_id: str) -> dict[str, Any] | None:
    found: dict[str, Any] | None = None
    for item in runtime.get("results") or []:
        if (
            isinstance(item, Mapping)
            and item.get("task_id") == task_id
            and item.get("status") in {"success", "partial"}
        ):
            found = dict(item)
    return found


def bindings_cover_required_outputs(runtime: Mapping[str, Any], task_id: str) -> bool:
    """True when every required output is already bound to a contract-valid file."""
    from CoScientist.experiments.schemas.models import ExperimentTask

    row = (runtime.get("tasks") or {}).get(task_id) or {}
    if not isinstance(row, Mapping):
        return False
    latest = _completed_result(runtime, task_id)
    if latest is None:
        return False
    try:
        task = ExperimentTask.model_validate(_task_dump(row))
    except Exception:  # noqa: BLE001 — a broken task dump is not a published output
        return False
    bindings = runtime.get(BINDINGS_KEY) or {}
    stored = [item for item in (latest.get("artifacts") or []) if isinstance(item, Mapping)]
    required = [item for item in task.expected_artifacts if item.required]
    if not required:
        return False
    for expected in required:
        binding = bindings.get(_binding_key(task_id, expected.output_id)) if isinstance(bindings, Mapping) else None
        if not isinstance(binding, Mapping) or not binding.get("artifact_id"):
            return False
        art = next(
            (item for item in stored if item.get("artifact_id") == binding.get("artifact_id")),
            None,
        )
        if art is None or contract_problem(art, expected) is not None:
            return False
    return True


def _producer_ids_for_block(
    runtime: Mapping[str, Any],
    task_id: str,
    row: Mapping[str, Any],
    seen: set[str],
) -> list[str]:
    if task_id in seen:
        return []
    seen.add(task_id)
    reason = row.get("blocked_reason") or {}
    code = str(reason.get("code") or "")
    tasks = runtime.get("tasks") or {}
    found: list[str] = []
    if code == "required_upstream_artifact_missing":
        for art in reason.get("artifacts") or []:
            if isinstance(art, Mapping) and str(art.get("source_task_id") or "").strip():
                found.append(str(art["source_task_id"]).strip())
        if not found:
            found.extend(_input_producers(row))
    elif code == "materialization_unavailable":
        found.append(task_id)
    elif code == "repairable_dependency_blocked":
        for dep in _dependency_ids(row):
            dep_row = tasks.get(dep) or {}
            if isinstance(dep_row, Mapping) and is_repairable_block(dep_row):
                found.extend(_producer_ids_for_block(runtime, dep, dep_row, seen))
    return found


def recovery_producer_ids(runtime: Mapping[str, Any]) -> list[str]:
    """Producers to repair once, in plan order.

    Several consumers of one output share one repair. A producer whose
    required outputs are already bound is not listed. An exhausted producer
    stays listed until ``recover_task_outputs`` records the stop, so the
    executor makes that call instead of spinning on an empty action list.
    """
    tasks = runtime.get("tasks") or {}
    ordered: list[str] = []
    seen: set[str] = set()
    for task_id in runtime.get("task_order") or []:
        row = tasks.get(task_id) or {}
        if not isinstance(row, Mapping) or not is_repairable_block(row):
            continue
        for producer in _producer_ids_for_block(runtime, str(task_id), row, set()):
            if producer in seen:
                continue
            if bindings_cover_required_outputs(runtime, producer):
                continue
            if _completed_result(runtime, producer) is None:
                continue
            seen.add(producer)
            ordered.append(producer)
    return ordered


def seal_exhausted_consumers(runtime: dict[str, Any], producer_id: str, message: str) -> None:
    """Stop repairable waits on this producer without touching other blocks."""
    tasks = runtime.get("tasks") or {}
    for task_id in list(runtime.get("task_order") or []):
        row = tasks.get(task_id)
        if not isinstance(row, dict) or not is_repairable_block(row):
            continue
        producers = _producer_ids_for_block(runtime, str(task_id), row, set())
        if producer_id not in producers and str(task_id) != producer_id:
            continue
        row["status"] = "blocked"
        row["blocked_reason"] = {
            "code": "output_recovery_exhausted",
            "producer_task_id": producer_id,
            "message": message,
        }
        row["last_message"] = message


def schema_outputs_verified(task: Any, artifacts: list[Mapping[str, Any]]) -> bool:
    """A schema criterion needs opened bytes, not a path or a URL."""
    required = [item for item in task.expected_artifacts if getattr(item, "required", True)]
    if not required:
        return False
    by_output = {str(art.get("output_id") or ""): art for art in artifacts}
    for expected in required:
        if str(getattr(expected, "role", "") or "") not in {"data", "model"}:
            continue
        art = by_output.get(str(getattr(expected, "output_id", "") or ""))
        if art is None or not content_verified(art, expected):
            return False
    return True
