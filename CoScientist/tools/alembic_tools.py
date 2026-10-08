"""ADK function tools wrapping the Alembic pipeline (GitHub repo → MCP server).

Alembic (CoScientist/alembic) builds a validated FastMCP tool server from a
scientific repository inside Docker. A full build takes tens of minutes, so the
tools here are job-based: ``build_mcp_server`` launches ``start_chain.py`` as a
host-side background subprocess and returns a ``job_id``; ``check_mcp_build``
reports progress (current pipeline stage, log tail) and, once the serve
container is up, the resulting MCP endpoint URL.

Jobs are process-wide (like the coder's local job registry), so over A2A —
where every orchestrator delegation is a fresh session — a later delegation can
find and continue an earlier build via ``list_mcp_builds``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google.adk.tools import ToolContext

# /<root>/CoScientist/tools/alembic_tools.py -> /<root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Hydrate os.environ from the project .env so subprocess builds inherit vars
# like A2A_HOST (advertise host for the served MCP). Idempotent; a no-op when
# the main app already called load_dotenv earlier.
load_dotenv(PROJECT_ROOT / ".env")
START_CHAIN = PROJECT_ROOT / "CoScientist" / "alembic" / "start_chain.py"
# Host-side stdout logs of the build subprocesses (the pipeline's own logs live
# inside the build container; this is the start_chain wrapper output).
LOG_DIR = PROJECT_ROOT / ".alembic" / "a2a_builds"

logger = logging.getLogger(__name__)

_LOG_TAIL_LINES = 15
_MAX_JOBS = 200  # cap registry size; evict oldest finished jobs past this
# Base for the absolute, clickable build-page link handed back to the agent.
# Empty when nothing set it, and then no absolute link is offered at all: the
# page only exists while the web UI is running, and a link that does not open
# sends the agent looking for the build somewhere else on the host.
_WEB_BASE_URL = os.environ.get("COSCIENTIST_WEB_BASE_URL", "").rstrip("/")

_JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

# Per-job identity file, alongside the log. Persisted so a process restart can
# rebuild `_JOBS` without reparsing docker chatter out of the log.
_META_FIELDS = ("job_id", "repo_url", "status", "started_at", "finished_at",
                "log_file", "workdir", "pid", "mcp_url", "image", "container",
                "error", "registered", "registration_error", "image_id",
                "server_id", "served_at", "image_deleted")


def _meta_path(job_id: str) -> Path:
    return LOG_DIR / f"{job_id}.json"


def _write_job_meta(rec: Dict[str, Any]) -> None:
    """Snapshot the fields we care about to <log_dir>/<job_id>.json."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        payload = {k: rec.get(k) for k in _META_FIELDS if rec.get(k) is not None}
        _meta_path(rec["job_id"]).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:  # noqa: BLE001 — best effort
        logger.warning("job meta write failed for %s: %s", rec.get("job_id"), exc)


def _read_job_meta(job_id: str) -> Optional[Dict[str, Any]]:
    p = _meta_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

# Patterns over start_chain.py / alembic.main output.
_URL_RE = re.compile(r"url\s*:\s*(http://\S+/mcp)")
# Anchored to the "MCP server up" summary lines: the earlier "building base
# image: docker build ..." line must not be read as the tool image.
_IMAGE_RE = re.compile(r"^\s*image\s*:\s*(\S+)", re.M)
_CONTAINER_RE = re.compile(r"^\s*container\s*:\s*(\S+)", re.M)
_STAGE_RE = re.compile(r"STAGE (\d) — (\S+)")


def _validator_counts(text: str) -> Optional[Dict[str, Any]]:
    """Tool and test counts from the validator's closing event in a build log."""
    for line in reversed(text.splitlines()):
        if '"validator"' in line and '"counts"' in line:
            event = parse_event_line(line)
            if (event and event.get("type") == "stage" and event.get("stage") == "validator"
                    and event.get("counts")):
                return event["counts"]
    return None


def _repo_name(repo_url: str) -> str:
    """Last path segment of a repo URL, without a trailing ``.git``
    (same rule as alembic.common.get_repo_name, kept local so importing this
    module never touches the alembic package's top-level path setup)."""
    return re.sub(r"\.git$", "", repo_url.rstrip("/").split("/")[-1])


def _evict_finished_jobs() -> None:
    if len(_JOBS) <= _MAX_JOBS:
        return
    for job_id, rec in list(_JOBS.items()):
        if rec["status"] != "running":
            del _JOBS[job_id]
        if len(_JOBS) <= _MAX_JOBS:
            return


def _read_log(rec: Dict[str, Any]) -> str:
    try:
        return Path(rec["log_file"]).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _finalize(rec: Dict[str, Any], returncode: int) -> None:
    """Parse the finished build's log into the job record (under _LOCK)."""
    text = _read_log(rec)
    rec["returncode"] = returncode
    rec["finished_at"] = time.time()
    if returncode == 0:
        url = _URL_RE.search(text)
        image = _IMAGE_RE.search(text)
        container = _CONTAINER_RE.search(text)
        rec["status"] = "done"
        rec["mcp_url"] = url.group(1) if url else None
        rec["image"] = image.group(1) if image else None
        rec["container"] = container.group(1) if container else None
    else:
        rec["status"] = "failed"


def _runner(rec: Dict[str, Any]) -> None:
    log_path = Path(rec["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Per-job workdir: every build gets its own <job_id>/workdir so artifacts of
    # earlier builds of the same repo are not overwritten. Alembic reads
    # ALEMBIC_WORKDIR at process start; each subprocess is fresh.
    workdir = Path(rec["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    # Persist the record's identity to disk so a restart of the web/agent
    # process can rebuild the registry from scratch. ``_JOBS`` is in-memory
    # only; without this file, the log alone cannot always tell us the repo
    # URL a job was launched against.
    _write_job_meta(rec)
    env = os.environ.copy()
    env["ALEMBIC_WORKDIR"] = str(workdir)
    # start_chain.py bind-mounts this dir into the build container at
    # /work/.alembic, so the pipeline's artifacts (exploration.md, plan.json,
    # generated tools/, server.py, setup.sh) land on the host for the web UI
    # to render — instead of dying with the container.
    env["ALEMBIC_HOST_WORKDIR"] = str(workdir)
    # start_chain tags the committed image alembic-tool:<job_id> as well, so this
    # build stays reachable after a newer build of the repo moves alembic-tool:<repo>.
    env["ALEMBIC_JOB_ID"] = rec["job_id"]
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, str(START_CHAIN), rec["repo_url"]],
                stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT, env=env,
            )
            with _LOCK:
                rec["pid"] = proc.pid
            returncode = proc.wait()
    except OSError as exc:  # docker/python missing, log dir unwritable, ...
        with _LOCK:
            rec["status"] = "failed"
            rec["error"] = f"could not launch the build subprocess: {exc}"
            rec["finished_at"] = time.time()
        return
    with _LOCK:
        _finalize(rec, returncode)
    if rec.get("status") == "done" and rec.get("image"):
        # The id survives the tag moving to a newer build of the same repo.
        # Best effort: a lookup that fails must not lose the finished build.
        try:
            found = _docker("image", "inspect", "-f", "{{.Id}}", rec["image"], timeout=30)
        except Exception as exc:  # noqa: BLE001
            logger.warning("image id lookup failed for %s: %s", rec["image"], exc)
        else:
            if found.returncode == 0 and found.stdout.strip():
                with _LOCK:
                    rec["image_id"] = found.stdout.strip()
    _write_job_meta(rec)
    # The catalogue entry is made here, when the build finishes, and not when
    # someone asks about it. An agent that starts a build and reports the job_id
    # back (which is what its prompt tells it to do) may never poll, and a tool
    # that exists but is in no catalogue is a tool the next run rebuilds from
    # scratch. Own thread, no event loop of its own, so asyncio.run is safe here.
    #
    # This thread is a daemon, so a host process that exits before the build
    # ends takes it down and nothing here runs — the build container finishes
    # regardless, but its result is lost. That is the same boundary the whole
    # job registry has (``_JOBS`` lives in memory), and it does not bite the
    # long-lived processes the system actually runs in: the web server, the A2A
    # services and the REPL all outlive their builds.
    try:
        asyncio.run(_register_in_catalogue(rec))
    except Exception as exc:  # noqa: BLE001 — the build itself succeeded
        logger.warning("catalogue registration thread failed: %s", exc)
    _write_job_meta(rec)  # the registration outcome has to outlive this process


def _snapshot(rec: Dict[str, Any], with_log_tail: bool = True) -> Dict[str, Any]:
    """The tool-facing view of one job record (call under _LOCK)."""
    out = {
        "job_id": rec["job_id"],
        "repo_url": rec["repo_url"],
        "status": rec["status"],
        "elapsed_seconds": round((rec.get("finished_at") or time.time()) - rec["started_at"]),
        "started_at": rec["started_at"],
        # Live build page in the CoScientist web UI (tails this build's log and
        # renders the streamed pipeline events). ``progress_page`` is relative
        # and always present, since the web layer resolves it itself.
        "progress_page": f"/alembic/builds/{rec['job_id']}",
        "workdir": rec.get("workdir"),
    }
    if _WEB_BASE_URL:
        out["progress_url"] = f"{_WEB_BASE_URL}/alembic/builds/{rec['job_id']}"
    text = _read_log(rec) if (with_log_tail or rec["status"] != "running") else ""
    stages = _STAGE_RE.findall(text)
    if stages:
        out["stage"] = f"{stages[-1][0]}/5 {stages[-1][1]}"
    counts = _validator_counts(text)
    if counts:
        out["tool_counts"] = counts
    if rec["status"] == "running":
        if with_log_tail:
            out["log_tail"] = "\n".join(text.splitlines()[-_LOG_TAIL_LINES:])
        out["note"] = ("The build is still running (a full build takes tens of "
                       "minutes). Do other work and call "
                       f"check_mcp_build('{rec['job_id']}') again later; do not "
                       "poll in a tight loop. That call is the only source of "
                       "this build's result. An MCP server found any other way "
                       "on this host belongs to some earlier build and says "
                       "nothing about this one.")
    elif rec["status"] == "done":
        out["mcp_url"] = rec.get("mcp_url")
        out["image"] = rec.get("image")
        out["container"] = rec.get("container")
        if "registered" in rec:
            out["registered"] = rec["registered"]
        for key in ("registration_error", "image_id", "image_deleted", "server_id"):
            if rec.get(key):
                out[key] = rec[key]
        if not rec.get("mcp_url"):
            out["note"] = ("Build finished but no MCP URL was printed — the image "
                           f"{rec.get('image') or 'alembic-tool:<repo>'} was likely "
                           "built with serving skipped; check the log.")
    else:
        out["error"] = rec.get("error") or "\n".join(text.splitlines()[-_LOG_TAIL_LINES:])
    return out


async def build_mcp_server(
    repo_url: str,
    force_rebuild: bool = False,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """Start an Alembic build: turn a GitHub repository into a served MCP tool
    server (clone → env → generated+validated tools → FastMCP server in Docker).

    The build runs in the background. This returns immediately with a job_id;
    track it with check_mcp_build(job_id).

    Args:
        repo_url: GitHub repository URL, e.g. "https://github.com/whitead/synspace".
        force_rebuild: Start a fresh build even if this repo already has a
            finished (or running) build in this process.

    Returns:
        status "running" with the job_id to check later; or the existing job for
        this repo (already running/done) unless force_rebuild is set.
    """
    repo_url = (repo_url or "").strip()
    if not re.match(r"^(https?://|git@)\S+/\S+", repo_url):
        return {"status": "error",
                "error": f"repo_url does not look like a git repository URL: {repo_url!r}"}

    with _LOCK:
        if not force_rebuild:
            # Prefer a live build; else the most recent finished one.
            same = [r for r in _JOBS.values() if r["repo_url"] == repo_url]
            for rec in reversed(same):
                if rec["status"] == "running":
                    snap = _snapshot(rec, with_log_tail=False)
                    snap["note"] = ("A build for this repository is already running — "
                                    f"reusing it. Track it with check_mcp_build('{rec['job_id']}').")
                    return snap
            for rec in reversed(same):
                if rec["status"] == "done":
                    snap = _snapshot(rec)
                    snap["note"] = ("This repository was already built in this process — "
                                    "reusing the result. Pass force_rebuild=true to rebuild.")
                    return snap

        job_id = f"{_repo_name(repo_url)}-{secrets.token_hex(3)}"
        rec: Dict[str, Any] = {
            "job_id": job_id,
            "repo_url": repo_url,
            "status": "running",
            "started_at": time.time(),
            "log_file": str(LOG_DIR / f"{job_id}.log"),
            "workdir": str(LOG_DIR / job_id / "workdir"),
        }
        _evict_finished_jobs()
        _JOBS[job_id] = rec

    threading.Thread(target=_runner, args=(rec,), daemon=True,
                     name=f"alembic-build-{job_id}").start()
    return {
        "status": "running",
        "job_id": job_id,
        "repo_url": repo_url,
        "note": ("Build started (base image → pipeline → docker commit → serve). "
                 "A full build takes tens of minutes: report the job_id back, do "
                 f"other work, and call check_mcp_build('{job_id}') later. That "
                 "call is the only source of this build's result; an MCP server "
                 "found any other way on this host belongs to some earlier "
                 "build."),
    }


async def check_mcp_build(
    job_id: str, tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """Check an Alembic build started by build_mcp_server.

    Args:
        job_id: The id returned by build_mcp_server.

    Returns:
        status "running" with the current pipeline stage and a log tail;
        "done" with the served MCP endpoint (mcp_url), image and container;
        or "failed" with the error tail of the build log.
    """
    with _LOCK:
        rec = _JOBS.get(job_id)
        if rec is None:
            return {"status": "error",
                    "error": f"unknown job_id {job_id!r} — use list_mcp_builds() "
                             "to see the builds known to this process."}
        out = _snapshot(rec)
    # Outside the lock: publishing talks to the registry over the network.
    await _publish_to_catalogue(rec, out, tool_context)
    return out


async def _register_in_catalogue(rec: Dict[str, Any]) -> None:
    """Ingest a finished build's MCP server into the rag_tools registry, once.

    This is the durable half: it needs nothing from the session, so it runs the
    moment the build finishes and does not wait for an agent to ask. Called
    again from a poll it is a no-op, because the attempt is recorded on the job.

    Never raises — a registry that is down must not turn a successful build into
    a failed tool call. The outcome is kept on the record and reported back to
    the agent as ``registered``.
    """
    if rec.get("status") != "done" or not rec.get("mcp_url") or "registered" in rec:
        return
    rec["registered"] = True  # one attempt per build, however often it is polled

    # A loopback URL in a shared catalogue is a broken entry: other machines
    # resolve it to their own localhost. Skip registration and say how to opt in.
    from urllib.parse import urlparse

    from CoScientist.alembic.remote import _LOCAL_HOSTS

    if urlparse(rec["mcp_url"]).hostname in _LOCAL_HOSTS:
        rec["registered"] = False
        rec["registration_error"] = (
            "served on a loopback address, so it was kept out of the shared "
            "catalogue; set A2A_HOST in .env to a host other machines can reach"
        )
        return

    from CoScientist.tools.registry_bridge import register_mcp_server

    name = _repo_name(rec["repo_url"])
    build = f" {rec['job_id']}" if rec.get("job_id") else ""
    try:
        server = await register_mcp_server(
            rec["mcp_url"], name, description=f"Alembic build{build} of {rec['repo_url']}"
        )
    except Exception as exc:  # noqa: BLE001 — the build itself succeeded
        logger.warning("catalogue registration failed for %s: %s", name, exc)
        rec["registered"] = False
        rec["registration_error"] = f"{type(exc).__name__}: {exc}"
        return
    # Kept so stopping the server can remove exactly this row; builds of one
    # repo share the name, and the row id also hashes the url.
    rec["server_id"] = getattr(server, "server_id", None)

    # A server row with no tools behind it is not a registration: retrieval
    # scores tools, so nothing will ever surface it. Say so instead of
    # reporting success the agent cannot act on.
    from rag_tools.storage.models import ToolStatus

    if getattr(server, "status", None) == ToolStatus.ERROR:
        rec["registered"] = False
        rec["registration_error"] = (
            f"{name} was added to the catalogue but its tools could not be "
            "indexed, so retrieval will not find it"
        )


async def _publish_to_catalogue(
    rec: Dict[str, Any], out: Dict[str, Any], tool_context: Optional[ToolContext]
) -> None:
    """Report the catalogue outcome and make the tool callable in this run.

    By the time a poll gets here the build thread has normally registered the
    server already; the call below only covers a record that never went through
    that thread. What is left is the run-scoped half: putting the url into
    ``deployed_mcps`` so the executor can call the tool without waiting for a
    retrieval round.
    """
    if out.get("status") != "done" or not out.get("mcp_url"):
        return
    await _register_in_catalogue(rec)
    out["registered"] = rec.get("registered", False)
    if rec.get("registration_error"):
        out["registration_error"] = rec["registration_error"]

    if tool_context is not None:
        from CoScientist.tools.registry_bridge import resolve_into_state

        # ADK records a state change on assignment, so the list is rebuilt and
        # put back whole instead of being appended to in place.
        state = {"deployed_mcps": list(
            (getattr(tool_context, "state", None) or {}).get("deployed_mcps") or []
        )}
        resolve_into_state(state, out["mcp_url"], _repo_name(rec["repo_url"]))
        tool_context.state["deployed_mcps"] = state["deployed_mcps"]


async def list_mcp_builds(tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """List every Alembic build known to this process (running and finished).

    Use this to recover a lost job_id or to find an MCP server that was already
    built for a repository in an earlier delegation/session.

    Returns:
        builds: one summary per job (job_id, repo_url, status, stage/mcp_url).
    """
    with _LOCK:
        return {"builds": [_snapshot(rec, with_log_tail=False)
                           for rec in _JOBS.values()]}


ALEMBIC_TOOLS = [build_mcp_server, check_mcp_build, list_mcp_builds]


# ── Web dashboard support ─────────────────────────────────────────────────────
# The CoScientist web UI (CoScientist/web/app.py) renders a live build page that
# tails a build's log and forwards the ``ALEMBIC_EVENT`` lines the container
# streams. These helpers are plain (non-ADK) functions the web layer calls.
#
# A build's log lives on disk at LOG_DIR/<job_id>.log regardless of which
# process started it, so the web helpers work even when the McpBuilderAgent runs
# in a separate A2A process: the in-memory _JOBS record is authoritative when
# present, and disk is the fallback (status re-derived from the log).
_EVENT_PREFIX = "ALEMBIC_EVENT "

_REPO_URL_RE = re.compile(
    r"(https?://[^\s\"'<>]+?(?:\.git|/[^\s\"'<>]+?))(?=[\s\"'<>]|$)"
)


def _recover_repo_url(text: str) -> Optional[str]:
    """Best-effort repo URL recovery from a build log. Docker echoes the run
    args with the URL near the end; the ``pipeline start`` ALEMBIC_EVENT also
    embeds it as ``\"repo_url\"``. Try the structured event first."""
    m = re.search(r'"repo_url"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1)
    m = _REPO_URL_RE.search(text)
    if m:
        return m.group(1)
    return None


def _status_from_log(text: str) -> str:
    """Best-effort status for a build we only know from its on-disk log (started
    by another process, so not in this process's _JOBS)."""
    if _URL_RE.search(text) or '"status": "complete"' in text:
        return "done"
    for marker in ("pipeline failed", "Traceback (most recent call last)",
                   "failed to connect to the docker API", '"status": "failed"'):
        if marker in text:
            return "failed"
    return "running"


def web_build_log_file(job_id: str) -> Optional[Path]:
    """Path to a build's log, or None if there is no such build."""
    with _LOCK:
        rec = _JOBS.get(job_id)
    if rec is not None:
        return Path(rec["log_file"])
    p = LOG_DIR / f"{job_id}.log"
    return p if p.exists() else None


_BUILD_CONTAINER_RE = re.compile(r"--name (alembic-build-\S+)")


def cancel_build(job_id: str) -> Dict[str, Any]:
    """Stop a running build from the web page.

    Removing the build container makes start_chain's ``docker run`` fail, so
    start_chain exits and the build thread records the job as failed. Before
    the container exists (the base image is still building) start_chain itself
    is stopped.
    """
    with _LOCK:
        rec = _JOBS.get(job_id)
    job = rec if rec is not None else (_read_job_meta(job_id) or {})
    if job.get("status") != "running":
        return {"ok": False, "error": f"build {job_id} is not running"}
    names = _BUILD_CONTAINER_RE.findall(_read_log(job)) if job.get("log_file") else []
    if names:
        _docker("rm", "-f", names[-1], timeout=60)
    elif job.get("pid"):
        try:
            os.kill(job["pid"], signal.SIGTERM)
        except OSError:
            pass
    if rec is not None:
        with _LOCK:
            rec["error"] = "cancelled from the web page"
    return {"ok": True}


def web_build_workdir(job_id: str) -> Optional[Path]:
    """The alembic workdir for a build, or None if it never ran / never
    persisted one (pre-per-job-workdir legacy builds)."""
    with _LOCK:
        rec = _JOBS.get(job_id)
    if rec is not None and rec.get("workdir"):
        p = Path(rec["workdir"])
        return p if p.exists() else None
    p = LOG_DIR / job_id / "workdir"
    return p if p.exists() else None


def web_build_repo_url(job_id: str) -> Optional[str]:
    """Best-effort repo_url for a build: in-memory record → persisted meta →
    parse the log's first ``start_chain`` invocation line."""
    with _LOCK:
        rec = _JOBS.get(job_id)
    if rec is not None:
        return rec.get("repo_url")
    meta = _read_job_meta(job_id)
    if meta and meta.get("repo_url"):
        return meta["repo_url"]
    log = LOG_DIR / f"{job_id}.log"
    if not log.exists():
        return None
    try:
        head = log.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return None
    url = _recover_repo_url(head)
    if url:
        # Backfill the meta file so subsequent lookups skip the regex.
        _write_job_meta({"job_id": job_id, "repo_url": url,
                         "log_file": str(log),
                         "workdir": str(LOG_DIR / job_id / "workdir")})
    return url


def web_build_snapshot(job_id: str) -> Optional[Dict[str, Any]]:
    """Status/result view of one build for the web page. In-memory record wins;
    otherwise reconstruct from the on-disk meta file + log tail."""
    with _LOCK:
        rec = _JOBS.get(job_id)
        if rec is not None:
            return _snapshot(rec)
    log = LOG_DIR / f"{job_id}.log"
    meta = _read_job_meta(job_id) or {}
    if not log.exists() and not meta:
        return None
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    status = meta.get("status") or _status_from_log(text)
    workdir_p = Path(meta["workdir"]) if meta.get("workdir") else LOG_DIR / job_id / "workdir"
    out: Dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "progress_page": f"/alembic/builds/{job_id}",
        "workdir": str(workdir_p) if workdir_p.exists() else None,
        "started_at": meta.get("started_at") or (log.stat().st_mtime if log.exists() else None),
    }
    if meta.get("repo_url"):
        out["repo_url"] = meta["repo_url"]
    else:
        # Legacy fallback: recover repo_url from the log body.
        url = _recover_repo_url(text)
        if url:
            out["repo_url"] = url
    if meta.get("started_at") and meta.get("finished_at"):
        out["elapsed_seconds"] = round(meta["finished_at"] - meta["started_at"])
    if _WEB_BASE_URL:
        out["progress_url"] = f"{_WEB_BASE_URL}/alembic/builds/{job_id}"
    if status == "done":
        # After a start or stop from the web page the meta file holds the
        # current container and address. Before that the log's last serve
        # summary does, and the meta file covers a build whose log is gone.
        acted = bool(meta.get("served_at"))
        for key, pattern in (("mcp_url", _URL_RE), ("image", _IMAGE_RE),
                             ("container", _CONTAINER_RE)):
            found = pattern.findall(text)
            out[key] = meta.get(key) if (acted and meta.get(key)) or not found else found[-1]
        for key in ("registered", "registration_error", "image_id", "image_deleted", "server_id"):
            if key in meta:
                out[key] = meta[key]
    elif status == "failed":
        out["error"] = "\n".join(text.splitlines()[-_LOG_TAIL_LINES:])
    stages = _STAGE_RE.findall(text)
    if stages:
        out["stage"] = f"{stages[-1][0]}/5 {stages[-1][1]}"
    counts = _validator_counts(text)
    if counts:
        out["tool_counts"] = counts
    return out


def web_list_builds() -> list:
    """Every build the web UI can show: in-memory records merged with all
    persisted on-disk builds (meta files + raw logs), newest first."""
    seen: Dict[str, Dict[str, Any]] = {}
    with _LOCK:
        for rec in _JOBS.values():
            seen[rec["job_id"]] = _snapshot(rec, with_log_tail=False)
    if LOG_DIR.exists():
        # Every persisted job identity — survives a process restart even
        # when no log file was written (very short-lived failure, wipe, etc.).
        job_ids = {p.stem for p in LOG_DIR.glob("*.log")}
        job_ids |= {p.stem for p in LOG_DIR.glob("*.json")}
        for jid in job_ids:
            if jid in seen:
                continue
            snap = web_build_snapshot(jid)
            if snap is not None:
                log = LOG_DIR / f"{jid}.log"
                if log.exists():
                    snap["mtime"] = log.stat().st_mtime
                    snap.setdefault("started_at", snap["mtime"])
                seen[jid] = snap
    return sorted(seen.values(),
                  key=lambda s: s.get("started_at") or s.get("mtime") or 0,
                  reverse=True)


def parse_event_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse one ``ALEMBIC_EVENT <json>`` log line into its event dict, or None
    if the line is not a structured event."""
    if not line.startswith(_EVENT_PREFIX):
        return None
    try:
        return json.loads(line[len(_EVENT_PREFIX):])
    except (ValueError, TypeError):
        return None


# ── Serving controls for the web builds list ─────────────────────────────────
# Start, restart and stop a finished build's MCP server, and delete its image.
# Docker is asked about the live state at the moment of the action; the meta
# file records the outcome and from then on wins over the build log.

_SERVE_SETTLE_SECONDS = 8
_TOOL_IMAGE = "alembic-tool"


def _docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(["docker", *args], 1, "", str(exc))


def docker_inventory() -> Dict[str, Any]:
    """Images (id to size), tags (tag to id) and containers (name to image id and
    running flag) on the local daemon, read in three docker calls."""
    inv: Dict[str, Any] = {"images": {}, "tags": {}, "containers": {}}
    r = _docker("images", "--no-trunc", "--format",
                "{{.ID}}|{{.Repository}}:{{.Tag}}|{{.Size}}", timeout=30)
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        image_id, tag, size = parts
        inv["images"][image_id] = size
        if "<none>" not in tag:
            inv["tags"][tag] = image_id
    names = _docker("ps", "-aq", timeout=30).stdout.split()
    if names:
        # docker inspect still prints the containers it found when one vanished
        # in between, so the output is read whatever the exit code.
        r = _docker("inspect", "-f", "{{.Name}}|{{.Image}}|{{.State.Running}}", *names,
                    timeout=30)
        for line in r.stdout.splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                inv["containers"][parts[0].lstrip("/")] = {
                    "image_id": parts[1], "running": parts[2] == "true"}
    return inv


def job_image(job: Dict[str, Any], inv: Dict[str, Any]) -> Optional[str]:
    """The id of the image a build serves from: its own, never alembic-tool:<repo>.

    That tag moves to every newer build of the repository, so starting from it
    would serve another build under this build's name.
    """
    if job.get("image_deleted"):
        return None
    if job.get("image_id") in inv["images"]:
        return job["image_id"]
    own = inv["tags"].get(f"{_TOOL_IMAGE}:{job.get('job_id')}")
    if own:
        return own
    container = inv["containers"].get(job.get("container") or "")
    if container and container["image_id"] in inv["images"]:
        return container["image_id"]
    return None


def _container_state(name: Optional[str]) -> Dict[str, Any]:
    if not name:
        return {"exists": False, "running": False}
    r = _docker("inspect", "-f", "{{.State.Running}}|{{.Image}}", name, timeout=30)
    if r.returncode != 0 or "|" not in r.stdout:
        return {"exists": False, "running": False}
    running, image_id = r.stdout.strip().split("|", 1)
    port = None
    if running == "true":
        p = _docker("port", name, "8000/tcp", timeout=30)
        if p.returncode == 0 and p.stdout.strip():
            port = p.stdout.strip().splitlines()[0].rsplit(":", 1)[-1]
    return {"exists": True, "running": running == "true", "image_id": image_id, "port": port}


def _stays_up(name: str) -> bool:
    """A server that cannot start (no server.py, broken venv) exits within a
    second or two; still running after the settle time rules that out."""
    deadline = time.monotonic() + _SERVE_SETTLE_SECONDS
    while True:
        if not _container_state(name)["running"]:
            return False
        if time.monotonic() >= deadline:
            return True
        time.sleep(1)


def _exited_error(name: str) -> str:
    logs = _docker("logs", "--tail", "20", name, timeout=30)
    return (f"MCP server container {name} exited right after start:\n"
            f"{(logs.stdout + logs.stderr).strip()[-1500:]}")


def _advertised_url(port: str) -> str:
    """The address other machines reach the server at, resolved the way
    start_chain does when it serves (A2A_HOST, else localhost)."""
    from CoScientist.alembic.remote import advertised_url, resolve_advertise_host

    return advertised_url(resolve_advertise_host(a2a_host=os.environ.get("A2A_HOST")), port)


def _unregister(server_id: str) -> Optional[str]:
    """Remove a catalogue row. Returns the error text, or None once it is gone."""
    from CoScientist.tools.registry_bridge import unregister_mcp_server

    try:
        asyncio.run(unregister_mcp_server(server_id))
    except Exception as exc:  # noqa: BLE001  reported to the page, never raised
        return f"{type(exc).__name__}: {exc}"
    return None


def _job(job_id: str) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """(the in-memory record or None, everything known about the build)."""
    snap = web_build_snapshot(job_id) or {}
    meta = _read_job_meta(job_id) or {}
    with _LOCK:
        rec = _JOBS.get(job_id)
        live = dict(rec) if rec is not None else {}
    return rec, {**snap, **meta, **live, "job_id": job_id}


def _update_job(job: Dict[str, Any], rec: Optional[Dict[str, Any]],
                fields: Dict[str, Any]) -> None:
    """Apply an action's outcome to the in-memory record, if any, and the meta file."""
    if rec is not None:
        with _LOCK:
            rec.update(fields)
            current = dict(rec)
        _write_job_meta(current)
        return
    meta = {k: job.get(k) for k in _META_FIELDS}
    meta.update(fields)
    _write_job_meta(meta)


def _log_event(job: Dict[str, Any], text: str) -> None:
    """Append to the build log, which is the build's history."""
    path = Path(job.get("log_file") or LOG_DIR / f"{job['job_id']}.log")
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"[web] {time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
    except OSError as exc:
        logger.warning("could not append to %s: %s", path, exc)


def _refresh_registration(job: Dict[str, Any], mcp_url: str) -> Dict[str, Any]:
    """The build-time catalogue check, run against the server's current address.

    server_id hashes the url, so a new address is a new catalogue row: the row
    for the previous address is removed first.
    """
    notes = []
    old_id = job.get("server_id")
    same_url = job.get("mcp_url") == mcp_url
    if old_id and not same_url:
        err = _unregister(old_id)
        if err:
            notes.append(f"the entry for {job.get('mcp_url')} was not removed: {err}")
    probe: Dict[str, Any] = {"status": "done", "mcp_url": mcp_url,
                             "repo_url": job["repo_url"], "job_id": job["job_id"]}
    try:
        asyncio.run(_register_in_catalogue(probe))
    except Exception as exc:  # noqa: BLE001  the server runs, only the catalogue step failed
        probe.update(registered=False, registration_error=f"{type(exc).__name__}: {exc}")
    errors = [e for e in (probe.get("registration_error"), *notes) if e]
    return {
        "mcp_url": mcp_url,
        "registered": bool(probe.get("registered")),
        # A failed re-registration at the same address leaves the old row in place.
        "server_id": probe.get("server_id") or (old_id if same_url else None),
        "registration_error": "; ".join(errors) or None,
    }


def _serve_new_container(job: Dict[str, Any], image_ref: str) -> tuple[Optional[str], str]:
    """Serve the image in a new container through start_chain --serve-only, so
    the port, env file, GPU flag and start check match a build's own serve."""
    cmd = [sys.executable, str(START_CHAIN), job["repo_url"], "--serve-only", "--image", image_ref]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT,
                           timeout=300, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"could not run start_chain: {exc}"
    output = (r.stdout + r.stderr).strip()
    _log_event(job, f"serving {image_ref} in a new container\n{output}")
    containers = _CONTAINER_RE.findall(r.stdout)
    if r.returncode != 0 or not containers:
        return None, output[-1500:] or f"start_chain exited with {r.returncode}"
    return containers[-1], ""


def _now_serving(job: Dict[str, Any], rec: Optional[Dict[str, Any]], name: str,
                 verb: str, **fields: Any) -> Dict[str, Any]:
    """Finish a start or restart: re-check registration at the container's
    current address, record the outcome and log it."""
    port = _container_state(name).get("port")
    if not port:
        return {"ok": False, "error": f"{name} is running but publishes no port for 8000"}
    fields.update(_refresh_registration(job, _advertised_url(port)), served_at=time.time())
    _update_job(job, rec, fields)
    _log_event(job, f"{verb} {name}; serving at {fields['mcp_url']}")
    return {"ok": True, **fields}


def start_build_server(job_id: str) -> Dict[str, Any]:
    """Start a finished build's MCP server and re-check its catalogue registration.

    The build's container comes back with ``docker start``, keeping its port
    and address. A new container is made from the build's image only when that
    container is gone or refuses to start (its port was taken meanwhile).
    """
    rec, job = _job(job_id)
    if job.get("status") != "done":
        return {"ok": False, "error": "only a finished build has a server to start"}
    inv = docker_inventory()
    image = job_image(job, inv)
    if not image:
        return {"ok": False, "error": "this build has no image of its own on this host"}
    name = job.get("container")
    state = _container_state(name)
    if state["running"]:
        return {"ok": False, "error": f"{name} is already running"}
    if state["exists"] and state.get("image_id") == image and _docker("start", name).returncode == 0:
        if not _stays_up(name):
            return {"ok": False, "error": _exited_error(name)}
    else:
        own_tag = f"{_TOOL_IMAGE}:{job_id}"
        new_name, error = _serve_new_container(job, own_tag if inv["tags"].get(own_tag) == image else image)
        if not new_name:
            return {"ok": False, "error": error}
        if state["exists"]:
            _docker("rm", name)  # the container this one replaces
        name = new_name
    return _now_serving(job, rec, name, "started", container=name, image_id=image)


def restart_build_server(job_id: str) -> Dict[str, Any]:
    """Restart a running build server and re-check its catalogue registration."""
    rec, job = _job(job_id)
    name = job.get("container")
    if not _container_state(name)["running"]:
        return {"ok": False, "error": "the server is not running; use Start"}
    r = _docker("restart", name)
    if r.returncode != 0:
        return {"ok": False, "error": f"docker restart failed: {(r.stderr or r.stdout).strip()[-500:]}"}
    if not _stays_up(name):
        return {"ok": False, "error": _exited_error(name)}
    return _now_serving(job, rec, name, "restarted")


def stop_build_server(job_id: str) -> Dict[str, Any]:
    """Stop a build server. Once the stop succeeds, a registered server leaves
    the shared catalogue; when that removal fails, the next Stop retries it."""
    rec, job = _job(job_id)
    name = job.get("container")
    if _container_state(name)["running"]:
        r = _docker("stop", name)
        if r.returncode != 0:
            return {"ok": False, "error": f"docker stop failed: {(r.stderr or r.stdout).strip()[-500:]}"}
    fields: Dict[str, Any] = {"served_at": time.time()}
    result: Dict[str, Any] = {"ok": True}
    server_id = job.get("server_id")
    if job.get("registered") and server_id:
        err = _unregister(server_id)
        if err:
            result["warning"] = fields["registration_error"] = (
                f"stopped, but the catalogue entry was not removed: {err}")
        else:
            fields.update(registered=False, server_id=None,
                          registration_error="stopped; removed from the shared catalogue")
    _update_job(job, rec, fields)
    _log_event(job, f"stopped {name}" + (f"; {result['warning']}" if "warning" in result else ""))
    return {**result, **{k: v for k, v in fields.items() if k != "served_at"}}


def _builds_sharing(image: str, job_id: str, inv: Dict[str, Any]) -> list:
    """Other finished builds whose image resolves to the same image id. Older
    builds without a job tag can share one: both served from alembic-tool:<repo>."""
    return [b["job_id"] for b in web_list_builds()
            if b.get("job_id") != job_id and b.get("status") == "done"
            and job_image(b, inv) == image]


def delete_build_image(job_id: str, include_shared: bool = False) -> Dict[str, Any]:
    """Delete a build's image together with its containers and every tag on it.

    When other builds use the same image, nothing happens unless
    ``include_shared`` confirms it; they all lose the image then. Registered
    servers leave the catalogue first, and while that fails the image stays, so
    no catalogue row is left pointing at a server that cannot come back.
    """
    rec, job = _job(job_id)
    inv = docker_inventory()
    image = job_image(job, inv)
    if not image:
        return {"ok": False, "error": "this build has no image of its own on this host"}
    shared = _builds_sharing(image, job_id, inv)
    if shared and not include_shared:
        return {"ok": False, "shared_with": shared,
                "error": f"the image is also the image of {', '.join(shared)}; "
                         "deleting it takes it from those builds too"}
    affected = [(rec, job)] + [_job(other) for other in shared]
    unregistered = []
    for owner_rec, owner in affected:
        if owner.get("registered") and owner.get("server_id"):
            err = _unregister(owner["server_id"])
            if err:
                for done_rec, done in unregistered:
                    _update_job(done, done_rec, {"registered": False, "server_id": None})
                return {"ok": False, "error": f"the catalogue entry of {owner['job_id']} could not "
                                              f"be removed, so the image was kept: {err}"}
            unregistered.append((owner_rec, owner))
    # Exact image id: `docker ps --filter ancestor=` also matches containers of
    # images built on top of this one.
    containers = [name for name, c in inv["containers"].items() if c["image_id"] == image]
    for container in containers:
        _docker("rm", "-f", container)
    tags = [tag for tag, image_id in inv["tags"].items() if image_id == image]
    r = _docker("rmi", *(tags or [image]))
    if r.returncode != 0:
        for done_rec, done in unregistered:
            _update_job(done, done_rec, {"registered": False, "server_id": None})
        return {"ok": False, "error": f"docker rmi failed: {(r.stderr or r.stdout).strip()[-500:]}"}
    now = time.time()
    for owner_rec, owner in affected:
        fields: Dict[str, Any] = {"image_deleted": True, "image_id": None,
                                  "registration_error": None, "served_at": now}
        if (owner_rec, owner) in unregistered:
            fields.update(registered=False, server_id=None)
        _update_job(owner, owner_rec, fields)
        together = "" if owner is job else f" together with {job_id}"
        _log_event(owner, f"deleted image {image}{together} (tags: {', '.join(tags) or 'none'}; "
                          f"containers: {', '.join(containers) or 'none'})")
    return {"ok": True, "removed_tags": tags, "removed_containers": containers,
            "also_deleted_for": shared}


def non_runnable_builds(inv: Dict[str, Any]) -> list:
    """Builds that cannot be served again from this host: failed ones, and
    finished ones without an image of their own. Running builds are left out."""
    return [b for b in web_list_builds()
            if b.get("status") != "running"
            and not (b.get("status") == "done" and job_image(b, inv))]


def _forget_build(job_id: str, inv: Dict[str, Any]) -> Optional[str]:
    """Remove one build from the history. Returns why it was kept, or None."""
    rec, job = _job(job_id)
    if job.get("status") == "running":
        return "the build is running"
    if job.get("registered") and job.get("server_id"):
        err = _unregister(job["server_id"])
        if err:
            return f"its catalogue entry could not be removed: {err}"
        _update_job(job, rec, {"registered": False, "server_id": None})
    container = job.get("container")
    state = inv["containers"].get(container or "")
    if state and state["running"]:
        return f"{container} is running"
    if state:
        _docker("rm", container)
        if state["image_id"] not in inv["tags"].values():
            # An untagged image that only this container kept alive is garbage
            # now; rmi refuses when another container or image still needs it.
            _docker("rmi", state["image_id"])
    artifacts = LOG_DIR / job_id
    if artifacts.exists():
        shutil.rmtree(artifacts, ignore_errors=True)
    if artifacts.exists():
        # Builds from before start_chain handed the workdir back to the user
        # left root-owned files, which a container can remove.
        _docker("run", "--rm", "-v", f"{LOG_DIR}:/builds", "--entrypoint", "rm",
                "alembic-base:latest", "-rf", f"/builds/{job_id}")
    if artifacts.exists():
        return f"could not delete {artifacts}"
    for path in {Path(job.get("log_file") or LOG_DIR / f"{job_id}.log"), _meta_path(job_id)}:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            return f"could not delete {path}: {exc}"
    with _LOCK:
        _JOBS.pop(job_id, None)
    return None


def clear_non_runnable_builds() -> Dict[str, Any]:
    """Remove every non-runnable build from the history: artifacts, stopped
    container, log and meta file. A build that cannot be removed completely
    stays, and the result says why."""
    inv = docker_inventory()
    removed, kept = [], []
    for b in non_runnable_builds(inv):
        reason = _forget_build(b["job_id"], inv)
        if reason:
            kept.append({"job_id": b["job_id"], "reason": reason})
        else:
            removed.append(b["job_id"])
    return {"ok": not kept, "removed": removed, "kept": kept}


# ── Session-bundle helpers ────────────────────────────────────────────────────
# Called by session_bundle.py to snapshot / restore the in-memory job registry
# when exporting or importing a .cossession.zip archive.

_SNAPSHOT_KEYS = ("job_id", "repo_url", "status", "mcp_url", "image",
                  "container", "started_at", "finished_at", "log_file")


def export_jobs_snapshot() -> list:
    """Serialisable snapshot of every job in the process-wide registry."""
    with _LOCK:
        return [{k: rec.get(k) for k in _SNAPSHOT_KEYS} for rec in _JOBS.values()]


def import_jobs_snapshot(jobs: list) -> None:
    """Restore job records from a previously exported snapshot.

    * ``running`` → ``failed`` (the original process is gone).
    * ``log_file`` is repointed to this process's LOG_DIR.
    * Existing records with the same job_id are NOT overwritten.
    """
    with _LOCK:
        for rec in jobs:
            jid = rec.get("job_id")
            if not jid or jid in _JOBS:
                continue
            entry = dict(rec)
            if entry.get("status") == "running":
                entry["status"] = "failed"
                entry["error"] = "Build was running when the session was exported."
                entry.setdefault("finished_at", entry.get("started_at"))
            entry["log_file"] = str(LOG_DIR / f"{jid}.log")
            _JOBS[jid] = entry


__all__ = ["ALEMBIC_TOOLS", "build_mcp_server", "check_mcp_build", "list_mcp_builds",
           "web_build_log_file", "web_build_snapshot", "web_build_workdir",
           "web_build_repo_url", "web_list_builds", "cancel_build", "parse_event_line",
           "docker_inventory", "job_image", "start_build_server", "restart_build_server",
           "stop_build_server", "delete_build_image", "non_runnable_builds",
           "clear_non_runnable_builds",
           "export_jobs_snapshot", "import_jobs_snapshot", "LOG_DIR"]
