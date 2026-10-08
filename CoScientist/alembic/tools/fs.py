"""Filesystem tools: clone/read/search the repo, read/write output and reports.

R8: the pipeline works on one repo, set once via ``clone_repo`` (or the
orchestrator) — no tool here takes a repo URL.
"""
import asyncio
import contextvars
import subprocess

from alembic.tools.paths import (
    IGNORE_EXTS, MAX_BYTES, output_dir, rel_or_ignored, repo_path,
    reports_dir, set_current_repo,
)

# ── read_file de-dup ───────────────────────────────────────────────────────────
# The Explorer was observed cycling read_file over the same handful of files
# many times, thrashing to the step ceiling. This tracks the paths already read
# *within one agent invocation* and stubs a repeat so the re-read is free and
# the model is nudged to move on. Scoped per-invocation, enabled only for the
# Explorer (agent_runtime.enable_read_dedup).
_read_seen: contextvars.ContextVar[set | None] = contextvars.ContextVar(
    "read_seen", default=None
)


def enable_read_dedup(enabled: bool) -> None:
    """Called by agent_runtime at each agent-invocation start: a fresh empty set
    when this agent should de-dup its own repeated reads, None to disable."""
    _read_seen.set(set() if enabled else None)


async def clone_repo(repo_url: str) -> dict:
    """Clone the GitHub repository to local disk (the one call that takes a URL).

    Returns the local path and a flat file list for you to select from.

    Example:
        clone_repo("https://github.com/Roestlab/massformer")
        # -> {"local_path": ".alembic/massformer/repos", "files": [...]}
    """
    set_current_repo(repo_url)
    # run on a worker thread — see bash()/bash_env() in shell.py for why.
    return await asyncio.to_thread(_clone_repo_sync, repo_url)


def _clone_repo_sync(repo_url: str, ref: str | None = None) -> dict:
    """Clone (idempotent). ``ref`` optionally pins a branch/tag/commit — used
    for TM-Bench task specs that fix ``repo.commit`` / ``repo.branch``. A ref
    that cannot be resolved is a warning, not a crash: fall back to the default
    branch so a mis-pinned task still runs."""
    set_current_repo(repo_url)
    dest = repo_path()
    # --recurse-submodules: some repos vendor real dependencies as git
    # submodules — without this the submodule dir clones empty and any import
    # touching it fails with a misleading ModuleNotFoundError.
    sub = ["--recurse-submodules", "--shallow-submodules"]
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)
        if ref and _looks_like_sha(ref):
            # A commit (esp. a short SHA) can't be shallow-fetched by name — do a
            # full clone (all branch history) and check the commit out of it.
            subprocess.run(["git", "clone", repo_url, str(dest)], check=True, capture_output=True)
            co = subprocess.run(["git", "-C", str(dest), "checkout", ref], capture_output=True, text=True)
            if co.returncode != 0:
                subprocess.run(["git", "-C", str(dest), "fetch", "origin", ref], capture_output=True)
                co = subprocess.run(["git", "-C", str(dest), "checkout", ref], capture_output=True, text=True)
            if co.returncode != 0:
                print(f"[clone] WARNING: could not checkout commit {ref!r} — using default branch")
            subprocess.run(["git", "-C", str(dest), "submodule", "update", "--init",
                            "--recursive", "--depth=1"], capture_output=True)
        elif ref:
            # A branch/tag: try a shallow branch clone; on failure fall back to
            # a plain default-branch clone.
            r = subprocess.run(["git", "clone", "--depth=1", *sub, "--branch", ref,
                                repo_url, str(dest)], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[clone] WARNING: branch/tag {ref!r} not found — using default branch")
                subprocess.run(["git", "clone", "--depth=1", *sub, repo_url, str(dest)],
                               check=True, capture_output=True)
        else:
            subprocess.run(["git", "clone", "--depth=1", *sub, repo_url, str(dest)],
                           check=True, capture_output=True)

    files = [rel for p in dest.rglob("*") if (rel := rel_or_ignored(p, dest))]
    return {"local_path": str(dest), "files": sorted(files)}


def _looks_like_sha(ref: str) -> bool:
    return len(ref) >= 7 and all(c in "0123456789abcdef" for c in ref.lower())


def read_file(path: str) -> dict:
    """Read a text file from the cloned repository (path relative to repo root).

    Returns up to 40 KB of content. Do NOT use this on data files (.csv,
    .parquet, .tsv, .json arrays) — use bash("head -n 20 <path>") instead.

    Example:
        read_file("README.md")
    """
    full = repo_path() / path
    if not full.exists():
        return {"error": f"File not found: {path}."}
    if full.is_dir():
        return {"error": f"'{path}' is a directory, not a file. Use search() or bash('ls') to list its contents."}
    if full.suffix in IGNORE_EXTS:
        return {"error": f"Binary/data file skipped: {path}."}

    seen = _read_seen.get()
    if seen is not None:
        if path in seen:
            return {
                "path": path,
                "already_read": True,
                "note": (
                    f"You already read '{path}' earlier in this session and its "
                    "content has not changed. Do NOT read it again — use what you "
                    "already have. If you are cycling back to files you have "
                    "already read, you are done exploring: write the report now."
                ),
            }
        seen.add(path)

    raw = full.read_bytes()[:MAX_BYTES]
    return {"path": path, "content": raw.decode("utf-8", errors="replace")}


def search(pattern: str) -> dict:
    """Find files in the cloned repo matching a glob pattern.

    Examples:
        search("**/*.yaml")
        search("*.sh")
    """
    dest = repo_path()
    matched = [rel for p in dest.glob(pattern) if (rel := rel_or_ignored(p, dest))]
    return {"pattern": pattern, "matches": sorted(matched)}


def read_report(report_name: str) -> dict:
    """Read a Markdown report from the reports directory.

    Args:
        report_name: Filename without the .md extension, e.g. "exploration".
    """
    path = reports_dir() / f"{report_name}.md"
    if not path.exists():
        return {"error": f"No report found at {path}."}
    return {"report_path": str(path), "content": path.read_text(encoding="utf-8")}


def _norm_out_rel(relative_path: str) -> str:
    """Normalize a path meant to live *under* the output dir. The tool base
    already IS output/, so a leading ``output/`` (or ``./`` / ``/``) is the
    common doubling mistake (``output/output/tools/x.py``) that would hide
    the file from the gates — strip it so read/write/update agree."""
    p = (relative_path or "").strip().lstrip("/")
    if p.startswith("./"):
        p = p[2:]
    while p.startswith("output/"):
        p = p[len("output/"):]
    return p


def write_file(relative_path: str, content: str) -> dict:
    """Write a source file to the output directory.

    Output lives at .alembic/<repo-name>/output/<relative_path>.

    Examples:
        write_file("tools/predict.py", "...")
        write_file("tests/test_predict.py", "...")
    """
    dest = output_dir() / _norm_out_rel(relative_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return {"written": str(dest)}


def read_output_file(relative_path: str) -> dict:
    """Read a file from the output directory.

    Examples:
        read_output_file("tools/predict.py")
        read_output_file("tests/test_predict.py")
    """
    full = output_dir() / _norm_out_rel(relative_path)
    if not full.exists():
        return {"error": f"File not found: {full}"}
    if full.is_dir():
        return {"error": f"'{relative_path}' is a directory, not a file."}
    raw = full.read_bytes()[:MAX_BYTES]
    return {"path": str(full), "content": raw.decode("utf-8", errors="replace")}


def update_file(relative_path: str, content: str) -> dict:
    """Overwrite a file in the output directory with corrected content.

    Always write the full file — not a patch.

    Examples:
        update_file("tools/predict.py", "...")
    """
    dest = output_dir() / _norm_out_rel(relative_path)
    if not dest.exists():
        return {"error": f"File not found: {dest}. Cannot update a file that does not exist."}
    dest.write_text(content, encoding="utf-8")
    return {"updated": str(dest)}


def write_report(report_name: str, content: str) -> dict:
    """Write a Markdown report to the reports directory.

    Args:
        report_name: Filename without the .md extension (e.g. "exploration").
        content:     Full Markdown content to write.

    Example:
        write_report("exploration", "# massformer...")
    """
    reports = reports_dir()
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"{report_name}.md"
    out.write_text(content, encoding="utf-8")
    return {"report_path": str(out)}
