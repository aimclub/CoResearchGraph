"""Stage a task's input files from the read-only data mount into the tool's
working tree.

Kept stdlib-only and separate from ``alembic.main`` so it is testable without
the agent stack, and importable under either package layout.

The correctness point is that a directory input must land as a *writable*
directory of symlinks, not as a bare ``dst -> src`` directory symlink. Tools
routinely write scratch files beside their inputs; through a directory symlink
those writes land on the read-only ``/mount/data`` volume and fail with
``OSError: [Errno 30] Read-only file system``. Mirroring the directories for
real and symlinking only the files keeps writes on the container layer while
still never copying the (tens of gigabytes of) input data.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Optional


def task_mounts(task: dict) -> list[tuple[str, str]]:
    """The ``(src, dst)`` pairs a task declares across its example and test cases."""
    pairs: list[tuple[str, str]] = []
    for invocation in [task.get("example") or {}, *(task.get("test_cases") or {}).values()]:
        if isinstance(invocation, dict):
            pairs += [(s, d) for s, d in (invocation.get("mount") or {}).items()]
    return pairs


def link_writable_tree(src: Path, dst: Path) -> None:
    """Mirror ``src`` at ``dst`` as a writable tree whose leaves link back to it.

    Directories are recreated for real so a tool can add scratch files anywhere
    in the tree; only the files are symlinked, so the bulk data is referenced
    rather than copied. A symlinked subdirectory is linked as-is instead of
    being walked into — following it could lead back out of the tree.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if target.exists() or target.is_symlink():
            continue
        if child.is_dir() and not child.is_symlink():
            link_writable_tree(child, target)
        else:
            target.symlink_to(child)


def stage_task_inputs(
    tasks: list[dict],
    data_root: Path,
    input_root: Path,
    *,
    warn: Optional[Callable[[str], Any]] = None,
) -> None:
    """Stage every task's declared mount files from ``data_root`` into ``input_root``.

    Directories become writable trees of symlinks (see :func:`link_writable_tree`);
    single files are symlinked directly. A missing source is reported and
    skipped, never fatal — testing degrades gracefully rather than aborting the
    run. ``warn`` is an optional log sink.
    """
    if not data_root.exists():
        if warn and tasks and any(task_mounts(t) for t in tasks):
            warn("[tasks] no /mount/data bind mount — task input files unavailable.")
        return
    for task in tasks:
        for src, dst in task_mounts(task):
            source, destination = data_root / src, input_root / dst
            if not source.exists():
                if warn:
                    warn(f"[tasks] mount source missing: {source}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                continue
            try:
                if source.is_dir():
                    link_writable_tree(source, destination)
                else:
                    destination.symlink_to(source)
            except OSError as exc:
                # Falling back to a copy can mean gigabytes of data, which is
                # exactly what linking avoids — say so rather than doing it
                # silently.
                if warn:
                    warn(f"[tasks] could not link {source} ({exc}) — copying instead.")
                (shutil.copytree if source.is_dir() else shutil.copy)(source, destination)
