"""The account of an agent's run, kept past the end of the process.

`agent_summary.summarize` writes, for one agent node of the execution tree, what
that agent did — and kept it in an in-process LRU of 256 entries. So every
restart threw away every summary a study had ever paid a model to write, and
reopening a finished run re-bought them one card at a time. An imported session
bundle arrived with none at all.

    <GRAPH_SNAPSHOT_DIR>/sessions/<user>/<session>/agent_summaries.json

beside `execution.json` and `research_active.json`, following the shape
`reporting/artifact_index.py` already uses for the same problem.

Keyed by the node AND by a stamp of the trace it was written from. A running
agent's trace changes under it, and a summary of what an agent had done twenty
tool calls ago, presented as current, is worse than no summary: the stamp is
what lets a reader be told the account is stale instead of being shown a stale
one. It is stored as its own field rather than folded into the key, so
staleness can be read off the record without re-deriving anything.

The scope is NOT part of the key. The file already belongs to one session, and
keying on the scope as well would mean an imported bundle — which lands under a
new user and session id — matched nothing it had just been given.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from CoScientist.graph.session_scope import SessionKey, storage_dir

logger = logging.getLogger(__name__)

STORE_FILENAME = "agent_summaries.json"
VERSION = 1

#: A long study writes one per agent per turn. Far above any real run, low
#: enough that a runaway cannot grow the file without bound.
_MAX_ENTRIES = 500


def _root() -> str:
    return os.getenv("GRAPH_SNAPSHOT_DIR", "./graph_runs")


def store_path(key: SessionKey) -> Path:
    """Where one session keeps its agent summaries."""
    return storage_dir(_root(), key) / STORE_FILENAME


def stamp(trace: str) -> str:
    """A short digest of the trace a summary was written from.

    The account is only true of the run it describes. When the agent runs on,
    the trace changes and this changes with it — which is how the panel knows
    to offer a fresh one rather than present an old one as current.
    """
    return hashlib.sha1(str(trace or "").encode("utf-8")).hexdigest()[:16]


def _entry_key(node_id: str, lang: str, trace_stamp: str) -> str:
    return f"{node_id}|{lang}|{trace_stamp}"


def _read(path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 — a damaged file is not a failed run
        logger.warning("agent summaries: cannot read %s (%s)", path, exc)
        return {}
    entries = data.get("entries") if isinstance(data, dict) else None
    return entries if isinstance(entries, dict) else {}


def _write(path: Path, entries: Dict[str, Dict[str, Any]]) -> None:
    """Atomic, like every other per-session file here: a half-written index is
    indistinguishable from a corrupt one on the next read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp")
    try:
        with handle:
            json.dump({"version": VERSION, "entries": entries}, handle,
                      ensure_ascii=False, indent=2)
        os.replace(handle.name, path)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise


def get(key: SessionKey, node_id: str, *, lang: str,
        trace_stamp: str) -> Optional[Dict[str, Any]]:
    """The stored account of this node's run, if one matches this trace."""
    if not node_id or not trace_stamp:
        return None
    try:
        entry = _read(store_path(key)).get(_entry_key(node_id, lang, trace_stamp))
    except Exception:  # noqa: BLE001
        return None
    if not entry or not entry.get("summary"):
        return None
    return dict(entry)


def latest(key: SessionKey, node_id: str, *,
           lang: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The newest account of this node, whatever trace it was written from.

    For a reader who wants what is known rather than what is current — the node
    report cites these, and an account written two tool calls ago still says
    what the agent did. The caller compares `trace_stamp` to decide whether to
    say so.
    """
    try:
        entries = [e for e in _read(store_path(key)).values()
                   if e.get("node_id") == node_id
                   and (lang is None or e.get("lang") == lang)]
    except Exception:  # noqa: BLE001
        return None
    if not entries:
        return None
    return dict(max(entries, key=lambda e: e.get("written_at") or 0))


def put(key: SessionKey, node_id: str, *, lang: str, trace_stamp: str,
        summary: str, model: str = "") -> None:
    """Record one account. Never raises: this is a saving, not a result."""
    if not (node_id and trace_stamp and str(summary or "").strip()):
        return
    import time

    try:
        path = store_path(key)
        entries = _read(path)
        entries[_entry_key(node_id, lang, trace_stamp)] = {
            "node_id": node_id,
            "lang": lang,
            "trace_stamp": trace_stamp,
            "summary": summary,
            "model": model,
            "written_at": time.time(),
        }
        if len(entries) > _MAX_ENTRIES:
            oldest = sorted(entries.items(),
                            key=lambda kv: kv[1].get("written_at") or 0)
            for stale_key, _ in oldest[:len(entries) - _MAX_ENTRIES]:
                entries.pop(stale_key, None)
        _write(path, entries)
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent summaries: cannot record %s (%s)", node_id, exc)


def all_entries(key: SessionKey) -> Dict[str, Dict[str, Any]]:
    """Everything stored for one session. For export and for tests."""
    try:
        return _read(store_path(key))
    except Exception:  # noqa: BLE001
        return {}


class SessionSummaries:
    """One session's store, as an object `summarize` can be handed.

    Injected rather than reached for, so `agent_summary` keeps no path logic and
    its existing tests keep passing with no store at all.
    """

    def __init__(self, key: SessionKey) -> None:
        self.key = key

    def get(self, node_id: str, lang: str, trace_stamp: str) -> Optional[Dict[str, Any]]:
        return get(self.key, node_id, lang=lang, trace_stamp=trace_stamp)

    def put(self, node_id: str, lang: str, trace_stamp: str,
            summary: str, model: str = "") -> None:
        put(self.key, node_id, lang=lang, trace_stamp=trace_stamp,
            summary=summary, model=model)

    def latest(self, node_id: str, *,
               lang: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """What is KNOWN about this run, not what is current.

        The node report cites these, and an account written two tool calls ago
        still says what the agent did. The caller compares `trace_stamp` if it
        wants to say the account has been overtaken.
        """
        return latest(self.key, node_id, lang=lang)


def for_session(key: SessionKey) -> SessionSummaries:
    return SessionSummaries(key)


__all__ = ["STORE_FILENAME", "SessionSummaries", "all_entries", "for_session",
           "get", "latest", "put", "stamp", "store_path"]
