"""Durable storage for Web UI pipeline checkpoints.

Each checkpoint is a standalone JSON file.  Atomic replacement makes a server
crash during a write harmless, and one corrupt checkpoint does not hide the
others.  Public listing functions intentionally omit the potentially large
session-state snapshot and the original user query.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic_core import to_jsonable_python

from CoScientist.web.session_store import state_dir

_LOCK = threading.RLock()
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")
_MAX_CHECKPOINTS_PER_SESSION = 200


def checkpoint_dir(user_id: str, session_id: str) -> Path:
    return (
        state_dir()
        / "checkpoints"
        / _SAFE.sub("_", user_id)
        / _SAFE.sub("_", session_id)
    )


def _path(user_id: str, session_id: str, checkpoint_id: str) -> Path:
    safe_id = _SAFE.sub("_", checkpoint_id)
    if not safe_id or safe_id != checkpoint_id:
        raise ValueError("Invalid checkpoint id.")
    return checkpoint_dir(user_id, session_id) / f"{safe_id}.json"


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "id", "stage_index", "stage_count", "agent", "title",
            "created_at", "run_version", "ui_event_index",
        )
    }


def save_checkpoint(
    user_id: str,
    session_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Persist one stage-boundary snapshot and return its public metadata."""
    checkpoint_id = str(payload.get("id") or f"cp_{uuid4().hex}")
    record = dict(payload)
    record["id"] = checkpoint_id
    # ADK state is executable resume data, not a display-only export.  The
    # converter preserves supported values such as pydantic models and
    # datetimes, but an unknown leaf must fail the checkpoint instead of being
    # silently replaced with a string of a different type.  The checkpoint
    # sink is observer-only, so this error is logged without poisoning the
    # canonical durable session.
    record = to_jsonable_python(record)

    with _LOCK:
        directory = checkpoint_dir(user_id, session_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = _path(user_id, session_id, checkpoint_id)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, target)

        files = sorted(
            directory.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for stale in files[_MAX_CHECKPOINTS_PER_SESSION:]:
            stale.unlink(missing_ok=True)
    return _public(record)


def load_checkpoint(
    user_id: str,
    session_id: str,
    checkpoint_id: str,
) -> dict[str, Any]:
    target = _path(user_id, session_id, checkpoint_id)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise KeyError(f"Unknown checkpoint {checkpoint_id!r}.") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Checkpoint {checkpoint_id!r} is corrupt.") from exc
    if not isinstance(value, dict) or not isinstance(value.get("state"), dict):
        raise ValueError(f"Checkpoint {checkpoint_id!r} has no session state.")
    return value


def list_checkpoints(user_id: str, session_id: str) -> list[dict[str, Any]]:
    directory = checkpoint_dir(user_id, session_id)
    if not directory.exists():
        return []
    records: list[dict[str, Any]] = []
    for target in directory.glob("*.json"):
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                records.append(_public(value))
        except (OSError, json.JSONDecodeError):
            continue
    records.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return records


def delete_checkpoints(user_id: str, session_id: str) -> None:
    directory = checkpoint_dir(user_id, session_id)
    if not directory.exists():
        return
    with _LOCK:
        for target in directory.glob("*.json*"):
            target.unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            pass


__all__ = [
    "checkpoint_dir",
    "delete_checkpoints",
    "list_checkpoints",
    "load_checkpoint",
    "save_checkpoint",
]
