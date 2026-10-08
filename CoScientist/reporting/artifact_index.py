"""A durable, per-session index of the artifacts a run produced.

The capture plugin used to write artifact links into ADK session state only, and
``main.py`` runs an ``InMemorySessionService``. A restart before the report step
therefore lost every figure the run had made.

This module writes the same list to disk, beside the graph snapshots:

    <GRAPH_SNAPSHOT_DIR>/sessions/<user_id>/<session_id>/artifacts.json

The durable reference is ``bucket`` plus ``s3_key``. An entry may also carry the
presigned ``url`` it was captured with, and that URL is a cache, never the
reference. It expires, usually in an hour. The report collector uses it while it
is fresh, and once the vault client lands it will mint a new one from the key
instead.

The vault does not hold this list. Its session manifest is a derived query over
object listings, with no writable object, so it cannot accept an append.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from CoScientist.graph.session_scope import SessionKey, safe_component, session_key, storage_dir

logger = logging.getLogger(__name__)

INDEX_FILENAME = "artifacts.json"


def _root() -> str:
    return os.getenv("GRAPH_SNAPSHOT_DIR", "./graph_runs")


def index_path(key: SessionKey) -> Path:
    """The artifact index of one session, beside its graph snapshots."""
    return storage_dir(_root(), key) / INDEX_FILENAME


def _entry_id(entry: Dict[str, Any]) -> str:
    """What makes two entries the same artifact. The durable reference when the
    entry has one, so re-capturing the same object under a fresh presigned URL
    updates that entry instead of adding a duplicate."""
    bucket, key = entry.get("bucket"), entry.get("s3_key")
    if bucket and key:
        return f"s3://{bucket}/{key}"
    return str(entry.get("url") or "")


def load(session_id: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read the index of one session.

    ``user_id`` is optional because the report collector is called from a tool
    that knows only the session id. Without it, search every user directory —
    session ids are unique, so at most one matches.
    """
    if user_id:
        paths = [index_path((user_id, session_id))]
    else:
        root = Path(_root()) / "sessions"
        paths = sorted(root.glob(f"*/{safe_component(session_id)}/{INDEX_FILENAME}"))

    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001 — a broken index must not sink a report
            logger.warning("artifact index: cannot read %s (%s)", path, exc)
            continue
        entries = data.get("artifacts") if isinstance(data, dict) else data
        if isinstance(entries, list):
            return [e for e in entries if isinstance(e, dict)]
    return []


def record(entries: List[Dict[str, Any]], context: Any = None, *,
           user_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
    """Merge entries into the index of one session. Never raises.

    Reads the existing file, appends what is new, and replaces the file
    atomically. Concurrent tool calls in one session can still interleave, and
    the loser only loses its own append — the graph node fields from change 1
    hold a second copy of every reference.
    """
    if not entries:
        return
    try:
        key = session_key(context, user_id=user_id, session_id=session_id)
        path = index_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        merged = load(key[1], key[0])
        known = {_entry_id(e): e for e in merged if _entry_id(e)}
        added = changed = 0
        for entry in entries:
            eid = _entry_id(entry)
            if not eid:
                continue
            previous = known.get(eid)
            if previous is None:
                known[eid] = entry
                merged.append(entry)
                added += 1
                continue
            # The same object, captured again under a fresh presigned URL. Keep
            # the newer URL. The stored one is older and expires first, and the
            # report downloads with whatever sits here, maybe an hour later.
            url = entry.get("url")
            if url and url != previous.get("url"):
                previous["url"] = url
                changed += 1
        if not added and not changed:
            return

        payload = json.dumps({"artifacts": merged}, ensure_ascii=False, indent=2)
        # Write beside the target, then rename. A crash mid-write leaves the old
        # index intact instead of a truncated file.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise
        logger.info(
            "artifact index: +%d new, %d refreshed -> %s (%d total)",
            added, changed, path, len(merged),
        )
    except Exception as exc:  # noqa: BLE001 — capture must never break a tool call
        logger.warning("artifact index: cannot record (%s)", exc)


__all__ = ["INDEX_FILENAME", "index_path", "load", "record"]
