"""Durable, session-owned FEDOT runs (also usable by a same-host A2A worker).

Each run has one writer and an append-only event journal. There is no global
"latest trace", retention limit, or dependency on a running Langfuse server.
WEB_STATE_DIR must be shared by the web process and local worker processes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

_LOCK = threading.RLock()
_RUN_ID = re.compile(r"[a-f0-9]{32}\Z")


def _directory(scope: tuple[str, str]) -> Path:
    base = Path(os.getenv("WEB_STATE_DIR", str(
        Path(os.getenv("RESEARCH_GRAPH_DIR", "./graph_runs")) / "web_state")))
    # Hash the tuple, not sanitised path components: a/b must not alias a_b.
    key = hashlib.sha256(json.dumps(list(scope), ensure_ascii=False).encode()).hexdigest()[:32]
    return base / "fedot_runs" / key


def _path(scope: tuple[str, str], run_id: str) -> Path:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid FEDOT run_id")
    return _directory(scope) / run_id


def _json(value: Any) -> str:
    def safe(item):
        if isinstance(item, float) and not math.isfinite(item):
            return str(item)
        if isinstance(item, dict):
            return {str(k): safe(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(v) for v in item]
        return item
    return json.dumps(safe(value), ensure_ascii=False, default=str, allow_nan=False)


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(uuid4().hex[:8] + ".tmp")
    try:
        temporary.write_text(_json(value), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_run(scope: tuple[str, str], **context: Any) -> dict:
    meta = {**context, "run_id": uuid4().hex, "user_id": scope[0],
            "session_id": scope[1], "started_at": time.time(),
            "status": "running", "event_count": 0, "version": 1}
    _atomic(_path(scope, meta["run_id"]) / "meta.json", meta)
    return meta


def get_run(scope: tuple[str, str], run_id: str) -> dict | None:
    path = _path(scope, run_id) / "meta.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if (meta.get("user_id"), meta.get("session_id")) != scope:
        return None
    return meta


def list_runs(scope: tuple[str, str]) -> list[dict]:
    runs = []
    for path in _directory(scope).glob("*/meta.json"):
        try:
            meta = get_run(scope, path.parent.name)
            if meta is not None:
                runs.append(meta)
        except (ValueError, OSError):
            continue
    return sorted(runs, key=lambda r: (r["started_at"], r["run_id"]), reverse=True)


def append_event(scope: tuple[str, str], run_id: str, payload: dict) -> dict:
    with _LOCK:
        meta = get_run(scope, run_id)
        if meta is None:
            raise KeyError(run_id)
        event = {**payload, "run_id": run_id, "user_id": scope[0],
                 "session_id": scope[1], "ts": time.time(),
                 "seq": meta["event_count"] + 1}
        directory = _path(scope, run_id)
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_json(event) + "\n")
        meta["event_count"] = event["seq"]
        if event["type"] == "config":
            meta["has_config"] = True
        elif event["type"] == "run_end":
            meta.update(status=event["status"], ended_at=event["ts"], error=event.get("error"))
        _atomic(directory / "meta.json", meta)
        return event


def read_events(scope: tuple[str, str], run_id: str, offset: int = 0) -> tuple[list[dict], int]:
    """Read complete lines only; leave a concurrently written tail for next poll."""
    events = []
    try:
        with (_path(scope, run_id) / "events.jsonl").open("rb") as stream:
            stream.seek(offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break
                offset = stream.tell()
                events.append(json.loads(line))
    except FileNotFoundError:
        pass
    return events, offset


def snapshot(scope: tuple[str, str]) -> dict:
    """One portable section of the session bundle, including every run/graph."""
    with _LOCK:
        runs = []
        for meta in reversed(list_runs(scope)):
            events, _ = read_events(scope, meta["run_id"])
            runs.append({"meta": meta, "events": events})
    return {"version": 1, "runs": runs}


def validate_snapshot(data: dict) -> None:
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("runs"), list):
        raise ValueError("Invalid FEDOT run history")
    seen = set()
    for run in data["runs"]:
        if not isinstance(run, dict) or not isinstance(run.get("meta"), dict):
            raise ValueError("Invalid FEDOT run record")
        meta, events = run.get("meta", {}), run.get("events")
        run_id = meta.get("run_id", "")
        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or run_id in seen:
            raise ValueError("Invalid or duplicate FEDOT run_id")
        if not isinstance(meta.get("started_at"), (int, float)) or not math.isfinite(meta["started_at"]):
            raise ValueError("Invalid FEDOT start time")
        if meta.get("status") not in {"running", "success", "error", "timeout", "cancelled", "snapshot"}:
            raise ValueError("Invalid FEDOT status")
        if not isinstance(events, list) or any(not isinstance(e, dict) or not isinstance(e.get("type"), str) for e in events):
            raise ValueError("Invalid FEDOT events")
        if any(not isinstance(e.get("ts"), (int, float)) or not math.isfinite(e["ts"]) for e in events):
            raise ValueError("Invalid FEDOT event time")
        seen.add(run_id)


def restore(scope: tuple[str, str], data: dict) -> None:
    validate_snapshot(data)
    for run in data["runs"]:
        meta = dict(run["meta"])
        run_id = meta["run_id"]
        directory = _path(scope, run_id)
        if directory.exists():
            raise ValueError("Cannot overwrite an existing FEDOT run")
        meta.setdefault("origin", {k: meta.get(k) for k in ("user_id", "session_id", "run_id")})
        meta.update(user_id=scope[0], session_id=scope[1], imported=True,
                    event_count=len(run["events"]))
        # An imported running snapshot is not a live computation on this server.
        if meta.get("status") == "running":
            meta["status"] = "snapshot"
        directory.mkdir(parents=True)
        with (directory / "events.jsonl").open("w", encoding="utf-8") as stream:
            for seq, event in enumerate(run["events"], 1):
                stream.write(_json({**event, "run_id": run_id, "user_id": scope[0],
                                    "session_id": scope[1], "seq": seq}) + "\n")
        _atomic(directory / "meta.json", meta)
