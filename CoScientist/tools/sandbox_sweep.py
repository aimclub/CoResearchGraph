"""Bring what the sandbox produced into the run, so the report can see it.

The report collector knows two places files come from: the artifact index (what
the capture plugin caught in tool results) and the local workspace directory on
this disk. The OpenHands sandbox is neither. It runs on another machine, and
the only files of its that ever reached a report were the ones the sandbox
agent itself chose to upload and reported back in ``s3_uploads`` — a decision
made by a model in the middle of a coding task, about files it produced last.
Everything else it made stayed in the container until the container went away.

So the aggregator was never withholding the artifacts; it never had them. This
sweep runs before collection and stages the workspace into the directory the
collector already walks, which means the classification, the caps, the pruning
of vendored trees and the markdown blocks all stay exactly where they were.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from CoScientist.reporting.collect import (
    _FILE_EXTS,
    _IMAGE_EXTS,
    _TABLE_EXTS,
    _WORKSPACE_SKIP_DIRS,
    _looks_like,
)

logger = logging.getLogger(__name__)

#: Where the sandbox keeps a run's working tree.
SANDBOX_ROOT = "/workspace"

#: Staged under the session's workspace, in a directory of its own so a file
#: from the container is never confused with one produced on this host.
STAGE_DIRNAME = "_sandbox"

#: A workspace is a working tree, not a result set: it holds a venv, a cloned
#: library, a pip cache, a conversation log per agent turn. Walking all of it
#: would cost minutes and bury the three files that matter, so the descent is
#: bounded on every axis.
MAX_DEPTH = 5
MAX_FILES = 80
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024

#: Project furniture that happens to share an extension with a result.
BOILERPLATE_NAMES = frozenset({
    "package.json", "package-lock.json", "tsconfig.json", "composer.json",
    "requirements.txt", "requirements-dev.txt", "pyproject.toml", "poetry.lock",
    "uv.lock", "readme.md", "license.md", "agents.md", "changelog.md",
    "meta.json", "base_state.json", "owner_lease.json", "tasks.json",
})

#: Directories the sandbox agent uses for its own bookkeeping — its transcript,
#: its lease files. Real to it, noise in a report.
EXTRA_SKIP_DIRS = frozenset({"conversations", "pdb", ".openhands", ".cache"})


def _wanted(name: str) -> bool:
    """Whether a file is the sort of thing a report is built out of."""
    if name.lower() in BOILERPLATE_NAMES or name.startswith("."):
        return False
    return _looks_like(name, _IMAGE_EXTS + _TABLE_EXTS + _FILE_EXTS)


def _walk(sandbox: Any, session_id: str, tool_context: Any) -> List[Dict[str, Any]]:
    """Breadth-first listing of the sandbox workspace, pruned as it goes."""
    found: List[Dict[str, Any]] = []
    queue: List[Tuple[str, int]] = [(SANDBOX_ROOT, 0)]
    while queue and len(found) < MAX_FILES:
        path, depth = queue.pop(0)
        listing = sandbox.list_sandbox_files(
            path, session_id=session_id, tool_context=tool_context,
        )
        if listing.get("status") != "ok":
            # The first failure is the whole sandbox being unreachable or gone;
            # deeper ones are a directory that vanished mid-run. Neither is
            # worth failing a report over.
            logger.info("sandbox sweep: %s — %s", path, listing.get("error"))
            if depth == 0:
                return []
            continue
        entries = listing.get("entries") or []
        # A coder step may clone a whole library into the workspace. Its bundled
        # example plots are not results of this run, and the giveaway is the
        # same one the collector uses on disk: the directory has a .git in it.
        if depth and any(str(e.get("name")) == ".git" for e in entries):
            continue
        for entry in entries:
            name = str(entry.get("name") or "")
            full = entry.get("path") or f"{path}/{name}"
            if str(entry.get("type", "")).lower() in ("dir", "directory"):
                if depth + 1 > MAX_DEPTH:
                    continue
                if name in _WORKSPACE_SKIP_DIRS or name in EXTRA_SKIP_DIRS:
                    continue
                if name.startswith(".") or name.endswith((".dist-info", ".egg-info")):
                    continue
                queue.append((full, depth + 1))
                continue
            if not _wanted(name):
                continue
            size = int(entry.get("size") or 0)
            if size > MAX_FILE_BYTES:
                logger.info("sandbox sweep: %s is %d bytes, left in place", full, size)
                continue
            found.append({"path": full, "name": name, "size": size})
            if len(found) >= MAX_FILES:
                break
    return found


def sweep_sandbox_workspace(
    tool_context: Any,
    state: Dict[str, Any],
    session_id: str,
    workspace_root: Path | str,
    already_named: Set[str] | None = None,
) -> Set[str]:
    """Stage the sandbox's results where the collector will find them.

    ``already_named`` holds the filenames that reached the artifact index by
    another route — the uploads the sandbox agent published itself. Those are
    collected from their own S3 reference, and staging them again would put the
    same figure in the report twice.

    Returns the workspace-relative paths staged, for the caller's logs.
    """
    from CoScientist.tools.coder_tools import openhands_sandbox as sandbox

    try:
        sandbox.resolve_sandbox_url()
    except Exception:  # noqa: BLE001 — no sandbox configured is the common case
        return set()

    session = sandbox.resolve_session_key(session_id, tool_context)
    if not sandbox.read_binding(session, tool_context):
        return set()  # this run never started one

    try:
        found = _walk(sandbox, session_id, tool_context)
    except Exception as exc:  # noqa: BLE001 — a report outranks its attachments
        logger.warning("sandbox sweep: listing failed (%s)", exc)
        return set()
    if not found:
        return set()

    published = {n.lower() for n in (already_named or set())}
    stage = Path(workspace_root) / f"ws_{session_id}" / STAGE_DIRNAME
    staged: Set[str] = set()
    total = 0
    for item in found:
        if item["name"].lower() in published:
            continue
        if total + item["size"] > MAX_TOTAL_BYTES:
            logger.info("sandbox sweep: stopping at %d bytes staged", total)
            break
        relative = item["path"][len(SANDBOX_ROOT):].lstrip("/") or item["name"]
        dest = stage / relative
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            got = sandbox.download_sandbox_file(
                item["path"], str(dest), session_id=session_id,
                tool_context=tool_context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("sandbox sweep: %s — %s", item["path"], exc)
            continue
        if got.get("status") != "ok":
            logger.info("sandbox sweep: %s — %s", item["path"], got.get("error"))
            continue
        total += int(got.get("size_bytes") or item["size"])
        staged.add(f"{STAGE_DIRNAME}/{relative}")

    logger.info("sandbox sweep: staged %d of %d file(s), %d bytes",
                len(staged), len(found), total)
    return staged


__all__ = ["sweep_sandbox_workspace", "SANDBOX_ROOT", "STAGE_DIRNAME"]
