#!/usr/bin/env python3
"""Build a set of tools from a manifest and register them, in one batch.

Converting repositories one at a time by hand does not scale past a handful,
and a tool that is built but never registered is invisible to the rest of the
system. This runs a whole list: each entry is converted by alembic and then
registered in the tool catalogue, so ``Retrieve_tools`` finds it in later runs
and on other machines.

A failure is recorded against its own entry and never aborts the batch — one
unbuildable repository out of twenty should cost you that one.

Manifest (JSON or YAML)::

    {
      "tools": [
        {"repo": "https://github.com/wallet-maker/cytopus", "name": "cytopus",
         "hint": "a tool returning the cell-type gene-set database for cell types"}
      ]
    }

  * ``repo`` (required) — repository to convert.
  * ``hint`` (optional) — free-text steer for the explorer (``ALEMBIC_HINTS``).
  * ``name`` (optional) — registry name (default: the repository basename).

Usage::

    python CoScientist/alembic/preconvert.py <manifest> --dry-run
    python CoScientist/alembic/preconvert.py <manifest> [--parallel N] [--output report.json]

``--dry-run`` prints what each entry would do — the resolved hint, the exact
build command, and a reachability check — without building or registering
anything. This module runs on the host; it shells out to ``start_chain.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# CoScientist/alembic/preconvert.py → project root is 2 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
START_CHAIN = PROJECT_ROOT / "CoScientist" / "alembic" / "start_chain.py"

# start_chain.py prints "url : http://localhost:<port>/mcp" once the serve
# container is up (same line alembic_tools parses).
_URL_RE = re.compile(r"url\s*:\s*(http://\S+/mcp)")
_REACH_TIMEOUT = 15  # git ls-remote precheck (seconds)


def _repo_name(repo_url: str) -> str:
    """Last path segment of a repo URL, without a trailing ``.git``."""
    return re.sub(r"\.git$", "", repo_url.rstrip("/").split("/")[-1])


# ── manifest ─────────────────────────────────────────────────────────────────


@dataclass
class ManifestItem:
    repo: str
    name: str
    hint: str = ""
    # Path to a gold task spec (e.g. ToolMaker's benchmark/tasks/<task>.yaml). When
    # set, it is threaded as ALEMBIC_TASKS so alembic builds a tool matching that
    # exact contract (REQUIRED-TASK gate) — gold-conformant by construction, unlike
    # the soft ``hint`` (ALEMBIC_HINTS). ``$VARS`` / ``~`` are expanded at build time.
    task_spec: str | None = None
    # Local data dir for the task's input mounts (start_chain --mount-dir); needed
    # for data-dependent gold tasks (e.g. tabpfn's train/test CSVs).
    mount_dir: str | None = None
    # Docker ``--gpus`` value (e.g. "all") forwarded to start_chain so the build +
    # serve containers see the GPU — required for the GPU foundation-model tools
    # (built on the bdgx daemon). ``None`` ⇒ CPU-only (flag omitted).
    gpus: str | None = None
    # Named daemon-side volume to reuse for the staged mount_dir (start_chain
    # --stage-volume): pre-stage a heavy dataset once, skip the copy on rebuilds.
    stage_volume: str | None = None


@dataclass
class Manifest:
    benchmark: str
    items: list[ManifestItem] = field(default_factory=list)


def load_manifest(path: str | Path) -> Manifest:
    """Parse + validate a JSON or YAML manifest into a :class:`Manifest`."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)

    if not isinstance(raw, dict) or "tools" not in raw:
        raise ValueError(f"{path}: manifest must be an object with a 'tools' list")
    tools = raw.get("tools") or []
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{path}: 'tools' must be a non-empty list")

    items: list[ManifestItem] = []
    for i, entry in enumerate(tools):
        if not isinstance(entry, dict) or not entry.get("repo"):
            raise ValueError(f"{path}: tools[{i}] must be an object with a 'repo'")
        repo = str(entry["repo"]).strip()
        items.append(
            ManifestItem(
                repo=repo,
                name=str(entry.get("name") or _repo_name(repo)),
                hint=str(entry.get("hint") or ""),
                task_spec=(str(entry["task_spec"]) if entry.get("task_spec") else None),
                mount_dir=(str(entry["mount_dir"]) if entry.get("mount_dir") else None),
                gpus=(str(entry["gpus"]) if entry.get("gpus") else None),
                stage_volume=(
                    str(entry["stage_volume"]) if entry.get("stage_volume") else None
                ),
            )
        )
    return Manifest(benchmark=str(raw.get("benchmark") or "unknown"), items=items)


def _resolve_task_spec(item: ManifestItem) -> str | None:
    """Expand ``$VARS``/``~`` in the item's ``task_spec`` path (or ``None``)."""
    if not item.task_spec:
        return None
    return os.path.expanduser(os.path.expandvars(item.task_spec))


def _task_spec_env(resolved: str) -> str:
    """The value to thread as ``ALEMBIC_TASKS``. When ``resolved`` is a readable
    local file, return its **content** (inline YAML/JSON) rather than the path:
    the conversion's ``server.py`` is generated inside a container on the build
    daemon (often remote), where a *local* path does not exist — ``_load_tasks``
    would then silently fall back to native mode and drop the REQUIRED-TASK gate.
    ``_load_tasks`` already accepts inline task text, so passing content makes the
    gold contract survive the daemon boundary. Non-file values (already-inline
    text, or an in-container path) pass through unchanged."""
    try:
        p = Path(resolved)
        if p.is_file():
            return p.read_text(encoding="utf-8")
    except OSError:
        pass
    return resolved


# ── the real per-item steps (injectable, so the batch is testable) ───────────


def check_repo_available(
    repo_url: str, timeout: int = _REACH_TIMEOUT
) -> tuple[bool, str]:
    """True if ``repo_url`` is reachable (``git ls-remote``) — a dead/private
    repo is caught in seconds instead of burning a full conversion."""
    try:
        r = subprocess.run(
            ["git", "ls-remote", "--exit-code", "--heads", repo_url],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return (
            r.returncode == 0,
            "" if r.returncode == 0 else "unreachable/empty repo",
        )
    except subprocess.TimeoutExpired:
        return (False, f"reachability check timed out after {timeout}s")
    except OSError as exc:
        return (False, f"git ls-remote failed: {exc}")


def build_command(item: ManifestItem) -> list[str]:
    """The exact start_chain build+serve command for an item (serving ON — no
    ``--no-serve`` — so an ``mcp_url`` is produced). A ``mount_dir`` is forwarded as
    ``--mount-dir`` so a data-dependent gold task can stage its input files."""
    cmd = [sys.executable, str(START_CHAIN), item.repo]
    if item.mount_dir:
        cmd += ["--mount-dir", os.path.expanduser(os.path.expandvars(item.mount_dir))]
    if item.gpus:
        cmd += ["--gpus", item.gpus]
    if item.stage_volume:
        cmd += ["--stage-volume", item.stage_volume]
    return cmd


def convert_repo(item: ManifestItem, *, timeout: float | None = None) -> dict[str, Any]:
    """Convert one repo via alembic, feeding a gold ``task_spec`` (``ALEMBIC_TASKS``,
    a REQUIRED-TASK gate → gold-conformant tool) and/or a soft ``hint``
    (``ALEMBIC_HINTS``) on the child env; serving is on, so the served ``mcp_url``
    is parsed from stdout. Returns ``{returncode, mcp_url, log_tail, error?}`` —
    never raises for a build failure (surfaced via ``returncode``/``error``)."""
    env = os.environ.copy()
    if item.hint:
        env["ALEMBIC_HINTS"] = item.hint
    task_spec = _resolve_task_spec(item)
    if task_spec:
        env["ALEMBIC_TASKS"] = _task_spec_env(task_spec)
    try:
        proc = subprocess.run(
            build_command(item),
            env=env,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "mcp_url": None,
            "error": f"build timed out ({timeout}s)",
        }

    out = (proc.stdout or "") + (proc.stderr or "")
    m = _URL_RE.search(out)
    result: dict[str, Any] = {
        "returncode": proc.returncode,
        "mcp_url": m.group(1) if m else None,
        "log_tail": "\n".join(out.splitlines()[-15:]),
    }
    if proc.returncode != 0:
        result["error"] = f"build exited {proc.returncode}"
    elif not result["mcp_url"]:
        result["error"] = "build finished but no mcp_url was printed (serving skipped?)"
    return result


async def register_result(mcp_url: str, name: str, *, manager=None):
    """Register a built server in the tool catalogue (the default registrar)."""
    from CoScientist.tools.registry_bridge import register_mcp_server

    return await register_mcp_server(mcp_url, name, manager=manager)


async def build_manager(local_embedder: bool = False):
    """Build the shared rag_tools manager for a real batch run.

    Default: the APIEmbedder stack ``registry_bridge._default_manager`` uses (so
    what we index matches ``Retrieve_tools``). ``local_embedder=True`` swaps in an
    in-process sentence-transformers ``LocalEmbedder`` — no embedding API server
    needed — for offline/local runs (its vector space differs from the API
    embedder, so use it only when both index and query use it)."""
    if not local_embedder:
        from CoScientist.tools.registry_bridge import _default_manager

        return await _default_manager()

    from rag_tools import create_manager
    from rag_tools.config.settings import get_settings
    from rag_tools.retrieval import APIReranker, BM25Reranker, HybridReranker
    from rag_tools.retrieval.embedder import LocalEmbedder

    s = get_settings()
    reranker = HybridReranker(
        [APIReranker(s.api_reranker), BM25Reranker(s.bm_reranker)], s.hybrid_reranker
    )
    return await create_manager(s, LocalEmbedder(s.embedding), reranker)


def _plan(item: ManifestItem) -> dict[str, Any]:
    """The dry-run plan for one item — what a real run WOULD do."""
    task_spec = _resolve_task_spec(item)
    env: dict[str, str] = {}
    if item.hint:
        env["ALEMBIC_HINTS"] = item.hint
    if task_spec:
        env["ALEMBIC_TASKS"] = _task_spec_env(task_spec)
    return {
        "repo": item.repo,
        "name": item.name,
        "hint": item.hint,
        "task_spec": task_spec,
        "gpus": item.gpus,
        "command": " ".join(build_command(item)),
        "env": env,
    }


# ── the batch ────────────────────────────────────────────────────────────────


async def _process_item(
    item: ManifestItem,
    *,
    dry_run: bool,
    sem: asyncio.Semaphore,
    converter: Callable[..., dict],
    registrar: Callable[..., Any],
    precheck: Callable[[str], tuple[bool, str]],
    convert_timeout: float | None,
    manager,
    reg_lock: asyncio.Lock | None = None,
) -> dict[str, Any]:
    """One manifest item end-to-end, isolated: any failure becomes a recorded
    result, never propagates (so one bad repo can't abort the batch).

    ``reg_lock`` serializes the registry write: all items share one ``rag_tools``
    manager (a single asyncpg connection), which is NOT safe for concurrent use —
    without the lock, two parallel items racing the shared session raise
    ``asyncpg InterfaceError: another operation is in progress``.
    """
    rec: dict[str, Any] = {"repo": item.repo, "name": item.name, "hint": item.hint}
    async with sem:
        try:
            ok, reason = await asyncio.to_thread(precheck, item.repo)
            if not ok:
                return {**rec, "status": "skipped", "error": reason}

            if dry_run:
                return {**rec, "status": "planned", "plan": _plan(item)}

            conv = await asyncio.to_thread(converter, item, timeout=convert_timeout)
            mcp_url = conv.get("mcp_url")
            if not mcp_url:
                return {
                    **rec,
                    "status": "failed",
                    "error": conv.get("error", "no mcp_url"),
                }

            if reg_lock is not None:
                async with reg_lock:
                    server = await registrar(mcp_url, item.name, manager=manager)
            else:
                server = await registrar(mcp_url, item.name, manager=manager)
            server_status = getattr(
                getattr(server, "status", None), "value", None
            ) or str(getattr(server, "status", ""))
            registered_ok = "error" not in server_status.lower()
            return {
                **rec,
                "status": "registered" if registered_ok else "registered_with_errors",
                "mcp_url": mcp_url,
                "server_id": getattr(server, "server_id", None),
                "server_status": server_status,
            }
        except Exception as exc:  # noqa: BLE001 — per-item isolation is the point
            return {**rec, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


async def run_batch_async(
    manifest: Manifest,
    *,
    dry_run: bool = False,
    parallel: int = 4,
    converter: Callable[..., dict] = convert_repo,
    registrar: Callable[..., Any] = register_result,
    precheck: Callable[[str], tuple[bool, str]] = check_repo_available,
    convert_timeout: float | None = None,
    manager=None,
) -> list[dict[str, Any]]:
    """Run the whole manifest; return one result record per item (order preserved).

    All steps are injectable so the batch is unit-testable without Docker / a
    live registry. A single ``rag_tools`` manager is shared across items (built
    by the caller / CLI); dry-run needs none.
    """
    sem = asyncio.Semaphore(max(1, parallel))
    # one lock (created in the running loop) serializes writes to the shared manager
    reg_lock = asyncio.Lock()
    tasks = [
        _process_item(
            item,
            dry_run=dry_run,
            sem=sem,
            converter=converter,
            registrar=registrar,
            precheck=precheck,
            convert_timeout=convert_timeout,
            manager=manager,
            reg_lock=reg_lock,
        )
        for item in manifest.items
    ]
    return await asyncio.gather(*tasks)


async def run_batch(
    manifest: Manifest,
    *,
    dry_run: bool = False,
    parallel: int = 4,
    convert_timeout: float | None = None,
    local_embedder: bool = False,
    converter: Callable[..., dict] = convert_repo,
    registrar: Callable[..., Any] = register_result,
    precheck: Callable[[str], tuple[bool, str]] = check_repo_available,
) -> list[dict[str, Any]]:
    """Build the shared ``rag_tools`` manager, run the batch, and close the manager
    — all awaited in a **single event loop**.

    The manager's asyncpg pool is bound to the loop it is created in; building it
    in one ``asyncio.run`` and using it in another (a separate ``asyncio.run``)
    raises ``RuntimeError: Future attached to a different loop`` / ``Event loop is
    closed``. Keeping build + run + close in one loop is the fix; the CLI calls
    this once. ``dry_run`` needs no manager.
    """
    manager = None
    if not dry_run:
        manager = await build_manager(local_embedder=local_embedder)
    try:
        return await run_batch_async(
            manifest,
            dry_run=dry_run,
            parallel=parallel,
            converter=converter,
            registrar=registrar,
            precheck=precheck,
            convert_timeout=convert_timeout,
            manager=manager,
        )
    finally:
        if manager is not None:
            await manager.close()


def summarize(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Alembic pre-conversion batch (manifest → convert → register)"
    )
    p.add_argument("manifest", help="path to the JSON/YAML manifest")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the per-item plan; build/register nothing",
    )
    p.add_argument(
        "--parallel", type=int, default=4, help="max concurrent items (default: 4)"
    )
    p.add_argument(
        "--convert-timeout",
        type=float,
        default=None,
        help="per-build timeout in seconds",
    )
    p.add_argument("--output", help="write the full JSON report to this path")
    p.add_argument(
        "--local-embedder",
        action="store_true",
        help="index with an in-process sentence-transformers LocalEmbedder "
        "instead of the APIEmbedder server (offline/local registration)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    args = _parse_args(argv)
    manifest = load_manifest(args.manifest)

    print(
        f"[preconvert] benchmark={manifest.benchmark} items={len(manifest.items)} "
        f"{'(dry-run)' if args.dry_run else ''}"
    )

    # build manager + run batch + close manager all in ONE event loop (the
    # asyncpg pool cannot cross loops — see run_batch).
    records = asyncio.run(
        run_batch(
            manifest,
            dry_run=args.dry_run,
            parallel=args.parallel,
            convert_timeout=args.convert_timeout,
            local_embedder=args.local_embedder,
        )
    )

    for r in records:
        line = f"  [{r['status']}] {r['name']} ({r['repo']})"
        if r.get("mcp_url"):
            line += f" → {r['mcp_url']}"
        if r.get("error"):
            line += f" — {r['error']}"
        print(line)
    print(f"[preconvert] summary: {summarize(records)}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {"benchmark": manifest.benchmark, "records": records},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[preconvert] report → {args.output}")

    # Non-zero exit if nothing succeeded on a real run (dry-run always 0).
    if not args.dry_run and not any(
        r["status"].startswith("registered") for r in records
    ):
        return 1
    return 0


__all__ = [
    "Manifest",
    "ManifestItem",
    "build_command",
    "check_repo_available",
    "convert_repo",
    "load_manifest",
    "register_result",
    "run_batch_async",
    "summarize",
]


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.exit(main())
