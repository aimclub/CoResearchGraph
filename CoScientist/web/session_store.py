"""On-disk persistence for the Web UI's users, sessions and chat history.

The ADK session service owns agent state, and the research/knowledge graphs are
already snapshotted per session — but the UI's own catalogue (who exists, which
sessions they have) and the chat/agent transcript lived only in process memory.
Restart the server and every past conversation disappeared from the UI, even
though its graphs and artifacts were still on disk.

Layout under ``WEB_STATE_DIR`` (default ``graph_runs/web_state``):

    registry.json                      users + session metadata
    sessions/<user_id>/<session_id>.jsonl   one JSON event per line

Events are append-only JSONL: a long run writes thousands of them, and appending
a line is cheap and crash-safe (a truncated last line is skipped on read). The
registry is small, so it is rewritten atomically.

Every function is best-effort: persistence must never break a run.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOCK = threading.RLock()
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def state_dir() -> Path:
    return Path(os.getenv("WEB_STATE_DIR",
                          str(Path(os.getenv("RESEARCH_GRAPH_DIR", "./graph_runs")) / "web_state")))


def registry_path() -> Path:
    return state_dir() / "registry.json"


def events_path(user_id: str, session_id: str) -> Path:
    return state_dir() / "sessions" / _SAFE.sub("_", user_id) / f"{_SAFE.sub('_', session_id)}.jsonl"


# ── registry ─────────────────────────────────────────────────────────────────
def save_registry(users: Dict[str, Any], sessions: Dict[Any, Any]) -> bool:
    """Persist the UI catalogue (atomic replace)."""
    try:
        with _LOCK:
            p = registry_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "users": list(users.values()),
                # tuple keys are not JSON-serialisable; the session dicts already
                # carry user_id + id, so the list round-trips losslessly.
                "sessions": list(sessions.values()),
            }
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, p)
        return True
    except Exception:  # noqa: BLE001
        return False


def load_registry() -> Dict[str, List[Dict[str, Any]]]:
    """Read the catalogue back; an absent or corrupt file yields an empty one."""
    try:
        p = registry_path()
        if not p.exists():
            return {"users": [], "sessions": []}
        data = json.loads(p.read_text(encoding="utf-8"))
        return {"users": list(data.get("users") or []),
                "sessions": list(data.get("sessions") or [])}
    except Exception:  # noqa: BLE001
        return {"users": [], "sessions": []}


# ── chat / agent events ──────────────────────────────────────────────────────
def append_event(user_id: str, session_id: str, event: Dict[str, Any]) -> bool:
    """Append one UI event to the session's transcript."""
    try:
        p = events_path(user_id, session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        return True
    except Exception:  # noqa: BLE001
        return False


def load_events(user_id: str, session_id: str, limit: int = 0) -> List[Dict[str, Any]]:
    """Read a session's transcript (last `limit` events when limit > 0)."""
    p = events_path(user_id, session_id)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn final line from a hard kill — skip it
    except Exception:  # noqa: BLE001
        return out
    return out[-limit:] if limit and limit > 0 else out


def has_events(user_id: str, session_id: str) -> bool:
    p = events_path(user_id, session_id)
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:  # noqa: BLE001
        return False


def delete_session(user_id: str, session_id: str) -> bool:
    try:
        events_path(user_id, session_id).unlink(missing_ok=True)
        return True
    except Exception:  # noqa: BLE001
        return False
