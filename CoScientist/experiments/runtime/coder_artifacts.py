"""Seed Coder sandbox inputs and promote outputs into experiment lineage."""
from __future__ import annotations

import hashlib
import html
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, MutableMapping
from CoScientist.config import get_settings
from CoScientist.experiments.runtime.errors import ExperimentRuntimeError
from CoScientist.experiments.runtime.shared import artifact_name_key

_WORKSPACE_STATE_KEY = "coder_workspace_id"
_log = logging.getLogger(__name__)


def _mirror(state: Mapping[str, Any], destination: Path, payload: bytes,
            record: MutableMapping[str, Any]) -> None:
    """Copy a promoted artifact into the session store and note its id.

    ``promote_coder_workspace_artifacts`` has no ADK context, but the scope is
    pinned in state by ``session_key`` the first time anything resolves it —
    the same reason ``workspace_sync`` reads it from there rather than from a
    context it does not have.
    """
    try:
        from CoScientist.graph.session_scope import (
            GRAPH_SCOPE_SESSION_KEY,
            GRAPH_SCOPE_USER_KEY,
        )
        from CoScientist.reporting.mirror import mirror_artifact

        user_id = str(state.get(GRAPH_SCOPE_USER_KEY) or "")
        session_id = str(state.get(GRAPH_SCOPE_SESSION_KEY) or "")
        if not (user_id and session_id):
            return
        mirrored = mirror_artifact(
            user_id=user_id, session_id=session_id,
            payload=payload, path=destination,
            filename=Path(record["name"]).name,
            label=str(record.get("name") or ""),
            tool="CoderAgent", source_kind="coder",
        )
        if mirrored.get("artifact_id"):
            record["artifact_id"] = mirrored["artifact_id"]
        if mirrored.get("bucket"):
            record["bucket"], record["s3_key"] = mirrored["bucket"], mirrored["s3_key"]
    except Exception as exc:  # noqa: BLE001 — promotion must not fail on this
        _log.warning("coder artifact mirror failed for %s (%s)", destination, exc)


def _find_candidate(root: Path, expected_name: str) -> Path | None:
    """Locate a sandbox file by exact basename, then by normalized stem."""
    if not root.is_dir():
        return None
    target = Path(expected_name).name
    if (exact := root / target).is_file():
        return exact
    if matches := [p for p in root.rglob(target) if p.is_file()]:
        return min(matches, key=lambda p: len(p.parts))
    if not (want := artifact_name_key(target)):
        return None
    soft = [p for p in root.rglob("*") if p.is_file() and artifact_name_key(p.name) == want]
    return min(soft, key=lambda p: len(p.parts)) if soft else None


def ensure_coder_workspace_id(state: MutableMapping[str, Any]) -> str:
    """Pin a sandbox id before CoderAgent runs so start_task can drop files there."""
    existing = state.get(_WORKSPACE_STATE_KEY)
    if existing:
        return str(existing)
    ws = f"ws_{uuid.uuid4().hex[:12]}"
    state[_WORKSPACE_STATE_KEY] = ws
    return ws


def _producer_attempt_id(state: MutableMapping[str, Any], source_task_id: str) -> str | None:
    runtime = state.get("experiment_runtime")
    if not isinstance(runtime, dict):
        return None
    for result in reversed(runtime.get("results") or []):
        if isinstance(result, dict) and result.get("task_id") == source_task_id:
            attempt = result.get("attempt_id")
            return str(attempt) if attempt else None
    return None


def _read_resolved_bytes(item: Mapping[str, Any]) -> bytes | None:
    path = item.get("resolved_workspace_path") or item.get("workspace_path")
    if isinstance(path, str) and path.strip():
        candidate = Path(path.strip())
        if candidate.is_file():
            return candidate.read_bytes()
    url = item.get("resolved_url") or item.get("url") or item.get("external_url")
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        import requests
        response = requests.get(html.unescape(url.strip()), timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as exc:
        _log.info("coder seed: download failed for %s (%s)", url, exc)
        return None


def _safe_name(item: Mapping[str, Any]) -> str:
    raw = (
        item.get("source_artifact_id")
        or item.get("data_id")
        or item.get("name")
        or "input.bin"
    )
    name = Path(str(raw)).name
    if name in {"", ".", ".."}:
        return "input.bin"
    return name


def _copy_aliases(dest: Path, sandbox: Path, *, source_task: str, attempt_id: str | None, name: str) -> None:
    """Also place the file where the executor historically quotes paths."""
    att = attempt_id or "latest"
    aliases = (
        sandbox / "experiment_artifacts" / source_task / att / name,
        sandbox / "workspace" / "experiment_artifacts" / source_task / att / name,
    )
    for alias in aliases:
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.resolve() != dest.resolve():
            shutil.copy2(dest, alias)


def seed_coder_upstream_inputs(
    state: MutableMapping[str, Any],
    resolved_inputs: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Copy/download resolved inputs into the Coder sandbox.

    Mutates ``resolved_inputs`` with ``coder_sandbox_path``. Required inputs that
    cannot be materialized raise ``ExperimentRuntimeError`` so Coder is not
    started against an empty workspace.
    """
    if not resolved_inputs:
        return []
    ws_id = ensure_coder_workspace_id(state)
    sandbox = Path(get_settings().code_exec.workspace_root) / ws_id
    sandbox.mkdir(parents=True, exist_ok=True)
    seeded: list[dict[str, Any]] = []
    for item in resolved_inputs:
        if not isinstance(item, dict):
            continue
        payload = _read_resolved_bytes(item)
        required = item.get("required", True)
        if payload is None:
            if required:
                label = item.get("data_id") or item.get("source_artifact_id") or "input"
                raise ExperimentRuntimeError(
                    "coder_input_missing",
                    f"Required Coder input {label!r} is not in the sandbox and could not be fetched.",
                )
            continue
        source_task = str(item.get("source_task_id") or "upstream")
        name = _safe_name(item)
        dest = sandbox / "inputs" / source_task / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        rel = str(Path("inputs") / source_task / name)
        item["coder_sandbox_path"] = rel
        item["resolved_workspace_path"] = str(dest)
        _copy_aliases(
            dest,
            sandbox,
            source_task=source_task,
            attempt_id=_producer_attempt_id(state, source_task),
            name=name,
        )
        seeded.append({"data_id": item.get("data_id"), "coder_sandbox_path": rel, "bytes": len(payload)})
        _log.info("coder seed: wrote %s (%s bytes)", rel, len(payload))
    return seeded


def promote_coder_workspace_artifacts(
    state: MutableMapping[str, Any],
) -> list[dict[str, Any]]:
    """Copy expected coder outputs into experiment_artifacts and capture them."""
    runtime = state.get("experiment_runtime")
    if not isinstance(runtime, dict):
        return []
    task_id, attempt_id = runtime.get("active_task_id"), runtime.get("active_attempt_id")
    if not task_id or not attempt_id:
        return []
    expected = (((runtime.get("tasks") or {}).get(task_id) or {}).get("task") or {}).get("expected_artifacts") or []
    if not isinstance(expected, list) or not expected:
        return []

    workspace_id = state.get(_WORKSPACE_STATE_KEY) or ""
    root = Path(get_settings().code_exec.workspace_root)
    sandbox = root / str(workspace_id) if workspace_id else None
    if sandbox is None or not sandbox.is_dir():
        return []

    dest_dir = root / "experiment_artifacts" / str(task_id) / str(attempt_id)
    dest_dir.mkdir(parents=True, exist_ok=True)

    bucket = state.setdefault("coder_artifacts", [])
    if not isinstance(bucket, list):
        state["coder_artifacts"] = []
        bucket = state["coder_artifacts"]
    existing = {str(item.get("workspace_path") or "") for item in bucket if isinstance(item, dict)}

    promoted: list[dict[str, Any]] = []
    for item in expected:
        if not isinstance(item, dict) or not (name := str(item.get("name") or "").strip()):
            continue
        if (source := _find_candidate(sandbox, name)) is None:
            continue
        destination = dest_dir / Path(name).name
        if destination.resolve() != source.resolve():
            shutil.copy2(source, destination)
        payload = destination.read_bytes()
        record = {
            "name": name,
            "role": item.get("role") or "data",
            "media_type": item.get("media_type"),
            "workspace_path": str(destination),
            "size_bytes": len(payload),
            "checksum_sha256": hashlib.sha256(payload).hexdigest(),
            "producer_tool": "CoderAgent",
            "source": "coder_workspace",
        }
        # Until now a coder artifact carried nothing but a path on this machine.
        # The report collector needs a url or a bucket to pick something up, and
        # these files also sit outside the directory it walks — so every one of
        # them was produced, written to disk, and never seen by anyone. The
        # bytes and the digest are already in hand here, so the mirror costs a
        # copy and gives the file an address.
        _mirror(state, destination, payload, record)
        if record["workspace_path"] not in existing:
            bucket.append(record)
            existing.add(record["workspace_path"])
            promoted.append(record)
    return promoted


__all__ = [
    "ensure_coder_workspace_id",
    "promote_coder_workspace_artifacts",
    "seed_coder_upstream_inputs",
]
