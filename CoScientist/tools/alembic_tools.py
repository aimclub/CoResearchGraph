"""ADK function tools wrapping the Alembic pipeline (GitHub repo → MCP server).

Alembic (CoScientist/alembic) builds a validated FastMCP tool server from a
scientific repository inside Docker. A full build takes tens of minutes, so the
tools here are job-based: ``build_mcp_server`` launches ``start_chain.py`` as a
host-side background subprocess and returns a ``job_id``; ``check_mcp_build``
reports progress (current pipeline stage, log tail) and, once the serve
container is up, the resulting MCP endpoint URL.

Jobs are process-wide (like the coder's local job registry) and every state
change is written to disk, so over A2A — where every orchestrator delegation
may be a fresh process — a later delegation can find and continue an earlier
build via ``list_mcp_builds``, and the experiment module can poll a build with
``peek_mcp_build``/``wait_mcp_build`` from whichever process it runs in.
"""
from __future__ import annotations

import asyncio
import hashlib
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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback for local development
    fcntl = None  # type: ignore[assignment]

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
LOG_DIR = Path(
    os.environ.get(
        "COSCIENTIST_ALEMBIC_LOG_DIR",
        str(PROJECT_ROOT / ".alembic" / "a2a_builds"),
    )
)
# Durable job records the experiment module reads across processes. They
# follow LOG_DIR unless COSCIENTIST_ALEMBIC_JOB_DIR pins them elsewhere.
_JOB_DIR_OVERRIDE = os.environ.get("COSCIENTIST_ALEMBIC_JOB_DIR")
JOB_METADATA_DIR = Path(_JOB_DIR_OVERRIDE) if _JOB_DIR_OVERRIDE else LOG_DIR / "jobs"

logger = logging.getLogger(__name__)

_LOG_TAIL_LINES = 15
_MAX_JOBS = 200  # cap registry size; evict oldest finished jobs past this
_METADATA_VERSION = 1
# What the durable record carries. Session scopes and the tool list are kept
# so a restarted web process still attaches a reused server to its chat.
_RECORD_FIELDS = frozenset(
    {
        "job_id",
        "repo_url",
        "status",
        "started_at",
        "finished_at",
        "log_file",
        "workdir",
        "pid",
        "returncode",
        "mcp_url",
        "image",
        "container",
        "error",
        "idempotency_key",
        "run_id",
        "task_id",
        "attempt_id",
        "registered",
        "registration_error",
        "image_id",
        "server_id",
        "served_at",
        "image_deleted",
        "origin",
        "tool_counts",
        "hub",
        "hints",
        "task_spec",
        "max_tools",
        "resume_from",
        "scopes",
        "tools",
    }
)
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
                "server_id", "served_at", "image_deleted", "origin", "tool_counts", "hub",
                "hints", "task_spec", "max_tools", "resume_from", "idempotency_key", "run_id",
                "task_id", "attempt_id")


def _meta_path(job_id: str) -> Path:
    return LOG_DIR / f"{job_id}.json"


def _write_job_meta(rec: Dict[str, Any]) -> None:
    """Snapshot the fields we care about to <log_dir>/<job_id>.json, and the
    durable registry copy when the record is complete enough for it."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        payload = {k: rec.get(k) for k in _META_FIELDS if rec.get(k) is not None}
        _meta_path(rec["job_id"]).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:  # noqa: BLE001 — best effort
        logger.warning("job meta write failed for %s: %s", rec.get("job_id"), exc)
    _persist_quietly(rec)


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
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
# start_chain prints this banner only once the served container stayed up.
_SERVE_BANNER = "MCP server up."


# ── Durable job registry ──────────────────────────────────────────────────────
# The experiment module runs its state machine in a process that may not be
# the one that started the build. Every record is written atomically to
# JOB_METADATA_DIR on each state change, and a process loads them at import
# (or on demand via reload_mcp_builds / peek_mcp_build).

def _metadata_dir() -> Path:
    """The registry directory; follows LOG_DIR unless the env override pins it."""
    return Path(_JOB_DIR_OVERRIDE) if _JOB_DIR_OVERRIDE else LOG_DIR / "jobs"


@contextmanager
def _registry_file_lock() -> Iterator[None]:
    """Serialize read-check-create across processes sharing the registry."""
    directory = _metadata_dir()
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".registry.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _metadata_file(job_id: str) -> Path:
    return _metadata_dir() / f"{job_id}.json"


def _record_for_disk(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        key: value
        for key, value in rec.items()
        if key in _RECORD_FIELDS and value is not None
    }
    out["metadata_version"] = _METADATA_VERSION
    return out


def _persist_job(rec: Dict[str, Any]) -> None:
    """Atomically replace one job's durable JSON record."""
    directory = _metadata_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = _metadata_file(rec["job_id"])
    temporary = directory / (
        f".{rec['job_id']}.{os.getpid()}.{threading.get_ident()}."
        f"{secrets.token_hex(4)}.tmp"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(
                _record_for_disk(rec),
                stream,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        try:
            directory_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _persist_quietly(rec: Dict[str, Any]) -> None:
    """_persist_job for a record the registry would accept back; a write that
    fails is logged, since the caller's own outcome stands without it."""
    if not _valid_disk_record(_record_for_disk(rec)):
        return
    try:
        _persist_job(rec)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("job registry write failed for %s: %s", rec.get("job_id"), exc)


def _delete_job_metadata(job_id: str) -> None:
    try:
        _metadata_file(job_id).unlink()
    except FileNotFoundError:
        pass


def _valid_disk_record(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("metadata_version") == _METADATA_VERSION
        and isinstance(value.get("job_id"), str)
        and _JOB_ID_RE.fullmatch(value["job_id"]) is not None
        and isinstance(value.get("repo_url"), str)
        and isinstance(value.get("log_file"), str)
        and value.get("status") in {"running", "done", "failed"}
        and isinstance(value.get("started_at"), (int, float))
    )


def _load_jobs_from_disk(*, merge: bool = False) -> int:
    """Load valid records, skipping torn, malformed, and future-version files."""
    loaded: list[Dict[str, Any]] = []
    directory = _metadata_dir()
    if directory.exists():
        for path in directory.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if not _valid_disk_record(value):
                continue
            rec = {key: value[key] for key in _RECORD_FIELDS if key in value}
            rec["_recovered"] = True
            loaded.append(rec)
    loaded.sort(key=lambda item: (item["started_at"], item["job_id"]))
    if not merge:
        _JOBS.clear()
    for rec in loaded:
        existing = _JOBS.get(rec["job_id"])
        if existing is None:
            _JOBS[rec["job_id"]] = rec
    return len(loaded)


def reload_mcp_builds() -> int:
    """Reconstruct the in-memory registry from durable metadata.

    This is primarily useful to long-lived coordinators that replace workers.
    Normal process startup performs the same load automatically.
    """
    with _LOCK:
        return _load_jobs_from_disk()


def _pid_is_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _refresh_recovered_job(rec: Dict[str, Any]) -> None:
    """Reconcile a running record whose original watcher no longer exists."""
    if not rec.get("_recovered") or rec.get("status") != "running":
        return
    text = _read_log(rec)
    inferred = _status_from_log(text)
    if inferred == "done":
        _finalize(rec, returncode=0)
        return
    if inferred == "failed":
        _finalize(rec, returncode=1)
        return
    if _pid_is_alive(rec.get("pid")):
        return
    rec["status"] = "failed"
    rec["finished_at"] = time.time()
    rec["error"] = (
        "build process is no longer running after registry recovery; "
        "inspect the persisted log before retrying"
    )
    _persist_quietly(rec)


def _repo_exists(repo_url: str, timeout: int = 20) -> Tuple[bool, str]:
    """Is ``repo_url`` a reachable git repository? (checked with git ls-remote)"""
    try:
        proc = subprocess.run(
            ["git", "ls-remote", "--exit-code", "-h", repo_url],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true"},
        )
    except subprocess.TimeoutExpired:
        return False, f"git ls-remote timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode == 0:
        return True, "ok"
    lines = [l.strip() for l in (proc.stderr or proc.stdout or "").splitlines() if l.strip()]
    named = [l for l in lines if l.lower().startswith(("fatal:", "remote:", "error:"))]
    picked = (named or lines or [f"git exited {proc.returncode}"])[0]
    return False, picked[:200]


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


def _repo_identity(repo_url: str) -> str:
    return re.sub(r"\.git$", "", repo_url.strip().rstrip("/")).lower()


def _reuse_snapshot(rec: Dict[str, Any], *, idempotent: bool = False) -> Dict[str, Any]:
    """The snapshot handed back for a build that is reused instead of started."""
    snap = _snapshot(rec, with_log_tail=rec["status"] != "running")
    if idempotent:
        snap["note"] = (
            "The idempotency key already identifies this repository build — "
            f"reusing job {rec['job_id']}."
        )
    elif rec["status"] == "running":
        snap["note"] = (
            "A build for this repository is already running — reusing it. "
            f"Track it with check_mcp_build('{rec['job_id']}')."
        )
    elif rec["status"] == "done":
        snap["note"] = (
            "This repository was already built — reusing the result. "
            "Pass force_rebuild=true to rebuild."
        )
    return snap


def _first_reusable_same_repo(repo_url: str) -> Optional[Dict[str, Any]]:
    """The record to reuse for ``repo_url``: a live build, else the most recent
    finished one, else None. Recovered records are reconciled first.

    Caller must hold ``_LOCK``.
    """
    same = [
        value
        for value in _JOBS.values()
        if _repo_identity(value["repo_url"]) == _repo_identity(repo_url)
    ]
    for existing in same:
        _refresh_recovered_job(existing)
    for existing in reversed(same):
        if existing["status"] == "running":
            return existing
    for existing in reversed(same):
        if existing["status"] == "done":
            return existing
    return None


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
        rec["status"] = "done"
        # The last serve summary is this build's; an earlier line can echo a
        # repository file that happens to look like one.
        for key, pattern in (("mcp_url", _URL_RE), ("image", _IMAGE_RE),
                             ("container", _CONTAINER_RE)):
            found = pattern.findall(text)
            rec[key] = found[-1] if found else None
    else:
        rec["status"] = "failed"
    if _JOBS.get(rec["job_id"]) is rec:
        # The terminal state stays visible in this process either way; a later
        # recovered snapshot can derive it from the durable log and retry.
        _persist_quietly(rec)


def _resolve_env_file() -> Optional[Path]:
    override = os.environ.get("COSCIENTIST_ALEMBIC_ENV_FILE")
    if override:
        path = Path(override)
        return path if path.exists() else None
    for candidate in (
        PROJECT_ROOT / "CoScientist" / ".env",
        PROJECT_ROOT / ".env",
    ):
        if candidate.exists():
            return candidate
    return None


def _start_chain_env() -> Dict[str, str]:
    """Host env for start_chain, with a usable MODEL when OpenRouter is absent."""
    env = os.environ.copy()
    if not env.get("MODEL") and not env.get("OPENROUTER_API_KEY"):
        env["MODEL"] = (
            env.get("LLM__CODER_MODEL")
            or env.get("LLM__MAIN_MODEL")
            or "openai/gpt-4o-mini"
        )
    return env


def _start_chain_cmd(repo_url: str) -> list[str]:
    cmd = [sys.executable, str(START_CHAIN), repo_url]
    env_file = _resolve_env_file()
    if env_file is not None:
        cmd += ["--env-file", str(env_file)]
    return cmd


def alembic_preflight(
    *, timeout_s: float = 5.0, runner=subprocess.run,
) -> Dict[str, Any]:
    """Check the Docker daemon Alembic would use before offering a build.

    The probe inherits the same environment as ``start_chain.py`` (including
    ``DOCKER_HOST``).  It is deliberately read-only and bounded: a broken
    remote DNS/daemon must be discovered before a long-lived build job is
    created, not several minutes into that job.
    """
    try:
        result = runner(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
            env=_start_chain_env(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "reason": f"Docker preflight failed: {type(exc).__name__}: {exc}",
        }
    if result.returncode == 0:
        return {
            "available": True,
            "reason": "Docker daemon is reachable",
            "server_version": (result.stdout or "").strip() or None,
        }
    detail = (result.stderr or result.stdout or "docker info failed").strip()
    return {
        "available": False,
        "reason": f"Docker preflight failed: {detail[:500]}",
    }


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
    env = _start_chain_env()
    env["ALEMBIC_WORKDIR"] = str(workdir)
    # start_chain.py bind-mounts this dir into the build container at
    # /work/.alembic, so the pipeline's artifacts (exploration.md, plan.json,
    # generated tools/, server.py, setup.sh) land on the host for the web UI
    # to render — instead of dying with the container.
    env["ALEMBIC_HOST_WORKDIR"] = str(workdir)
    # start_chain tags the committed image alembic-tool:<job_id> as well, so this
    # build stays reachable after a newer build of the repo moves alembic-tool:<repo>.
    env["ALEMBIC_JOB_ID"] = rec["job_id"]
    # What the operator asked for beyond the repository: a soft steer for the
    # explorer, and a task spec that pins the tools the build must produce.
    # start_chain passes both through to the build container.
    for key, var in (("hints", "ALEMBIC_HINTS"), ("task_spec", "ALEMBIC_TASKS"),
                     ("max_tools", "ALEMBIC_MAX_TOOLS")):
        if rec.get(key):
            env[var] = str(rec[key])
    cmd = _start_chain_cmd(rec["repo_url"])
    if rec.get("resume_from"):
        # Same workdir, so the stages before this one are read from disk.
        cmd += ["--resume", str(rec["resume_from"])]
    try:
        # A resumed build appends: the earlier stages' log stays readable.
        with open(log_path, "a" if rec.get("resume_from") else "w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT, env=env,
            )
            with _LOCK:
                rec["pid"] = proc.pid
                _persist_quietly(rec)
            returncode = proc.wait()
    except OSError as exc:  # docker/python missing, log dir unwritable, ...
        with _LOCK:
            rec["status"] = "failed"
            rec["error"] = f"could not launch the build subprocess: {exc}"
            rec["finished_at"] = time.time()
            _persist_quietly(rec)
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
    # regardless, but its result is lost. The record on disk still says
    # "running" then, and _refresh_recovered_job settles it from the log and
    # the pid once another process picks it up.
    try:
        asyncio.run(_register_in_catalogue(rec))
    except Exception as exc:  # noqa: BLE001 — the build itself succeeded
        logger.warning("catalogue registration thread failed: %s", exc)
    _write_job_meta(rec)  # the registration outcome has to outlive this process
    if rec.get("status") == "done":
        _auto_upload(rec)


def _auto_upload(rec: Dict[str, Any]) -> None:
    """Upload a finished build to the MCP hub when the auto-upload setting is on."""
    try:
        from CoScientist.tools import alembic_hub

        if not alembic_hub.auto_upload_enabled():
            return
        result = alembic_hub.upload_build(rec["job_id"], keep_better=True)
    except Exception as exc:  # noqa: BLE001 - the build itself succeeded
        logger.warning("auto-upload of %s to the MCP hub failed: %s", rec.get("job_id"), exc)
        return
    if not result.get("ok") and not result.get("skipped"):
        logger.warning("auto-upload of %s to the MCP hub failed: %s", rec["job_id"], result.get("error"))


def _snapshot(rec: Dict[str, Any], with_log_tail: bool = True) -> Dict[str, Any]:
    """The tool-facing view of one job record (call under _LOCK)."""
    _refresh_recovered_job(rec)
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
    for key in ("idempotency_key", "run_id", "task_id", "attempt_id", "pid"):
        if rec.get(key) is not None:
            out[key] = rec[key]
    text = _read_log(rec) if (with_log_tail or rec["status"] != "running") else ""
    stages = _STAGE_RE.findall(text)
    if stages:
        out["stage"] = f"{stages[-1][0]}/5 {stages[-1][1]}"
    # A server taken from a tool image has no build log; its counts came from the image.
    counts = _validator_counts(text) or rec.get("tool_counts")
    if counts:
        out["tool_counts"] = counts
    if rec.get("origin"):
        out["origin"] = rec["origin"]
    if rec.get("hub"):
        out["hub"] = rec["hub"]
    if rec.get("origin") == "hub" and rec["status"] == "running":
        out["stage"] = "pulling from the MCP hub"
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


def _claim(rec: Dict[str, Any], scope: Optional[list]) -> None:
    """Record that a session built or reused this build, so its server is
    attached to that session's executor (live_build_servers). Call under _LOCK."""
    if scope and scope not in rec.setdefault("scopes", []):
        rec["scopes"].append(scope)


_RESTARTED_NOTE = ("The server of this build was stopped: started it again from the "
                   "build's image. Use the mcp_url below.")

# A container that stays up can still be loading its libraries: the address is
# handed out only once the server answers MCP.
_SERVE_READY_TIMEOUT = float(os.getenv("ALEMBIC_SERVE_READY_TIMEOUT", "60"))
_SERVE_READY_POLL = 2.0


def _wait_until_answering(mcp_url: str) -> Optional[list]:
    """The tools a just-started server lists, polled until it answers or
    _SERVE_READY_TIMEOUT passes (then None). Runs an event loop, so call it
    from a worker thread."""
    deadline = time.monotonic() + _SERVE_READY_TIMEOUT
    while True:
        tools = asyncio.run(_served_tools(mcp_url))
        if tools is not None or time.monotonic() >= deadline:
            return tools
        time.sleep(_SERVE_READY_POLL)


def _mark_unserved(out: Dict[str, Any], served: Dict[str, Any], job_id: str) -> None:
    """Take the address out of a result whose server cannot be called now."""
    out.pop("mcp_url", None)
    out.pop("tools", None)
    out["server_running"] = False
    if served.get("answering") is False:
        out["note"] = (f"The server of this build was started again but does not answer MCP "
                       f"after {int(_SERVE_READY_TIMEOUT)} s. Call check_mcp_build('{job_id}') "
                       "again in a minute.")
    else:
        out["note"] = ("The build finished, but its server is stopped and did not start "
                       f"again: {served.get('error')}. Call build_mcp_server with "
                       "force_rebuild=true to build it anew.")


def _await_answer(rec: Dict[str, Any], url: str) -> bool:
    """Wait until the server at ``url`` answers MCP and keep its tools on ``rec``."""
    tools = _wait_until_answering(url)
    with _LOCK:
        rec["serve_pending"] = tools is None
        if tools:
            rec["tools"] = tools
    return tools is not None


def _ensure_served(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Start a finished build's server again when it is down and wait until it answers.

    Returns {"ok": True, "started": bool}, {"ok": False, "error": ...} when it does
    not start, or {"ok": False, "answering": False}. Runs docker: call it outside _LOCK.
    """
    container = rec.get("container")
    if rec.get("status") != "done" or not container:
        return {"ok": True, "started": False}
    if _container_state(container)["running"]:
        # serve_pending: started earlier for an agent and not answering yet.
        if not rec.get("serve_pending"):
            return {"ok": True, "started": False}
        url = rec["mcp_url"]
    else:
        started = start_build_server(rec["job_id"])
        if not started.get("ok"):
            logger.warning("could not serve %s again: %s", rec["job_id"], started.get("error"))
            return {"ok": False, "error": started.get("error")}
        url = started["mcp_url"]
    if not _await_answer(rec, url):
        logger.warning("%s was started but does not answer MCP at %s", rec["job_id"], url)
        return {"ok": False, "answering": False}
    return {"ok": True, "started": True}


def _adopt(job: Dict[str, Any], scope: Optional[list],
           note: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Put a build found on the host into this process's registry.

    Returns (record, snapshot with ``note``), the pair build_mcp_server
    publishes. A build already in ``_JOBS`` only takes the current address.
    """
    with _LOCK:
        rec = _JOBS.get(job["job_id"])
        if rec is None:
            rec = {k: job[k] for k in (
                "job_id", "repo_url", "status", "started_at", "finished_at", "log_file",
                "workdir", "mcp_url", "image", "container", "image_id", "server_id",
                "registered", "registration_error", "origin", "tool_counts")
                if job.get(k) is not None}
            rec.setdefault("started_at", time.time())
            rec.setdefault("log_file", str(LOG_DIR / f"{job['job_id']}.log"))
            _evict_finished_jobs()
            _JOBS[rec["job_id"]] = rec
        else:
            rec.update({k: job[k] for k in ("mcp_url", "container", "image_id") if job.get(k)})
        _claim(rec, scope)
        snap = _snapshot(rec)
        current = dict(rec)
    # The registry copy, so peek_mcp_build finds the job from another process.
    _persist_quietly(current)
    snap["note"] = note
    return rec, snap


# What the builds page reads from a tool image: the reports and the generated
# code. The venvs (hundreds of MB) stay behind, and so does pipeline.log, the
# agents' output.
_IMAGE_ARTIFACTS = ("reports", "output/tools", "output/helpers", "output/tests",
                    "output/server.py", "output/setup.sh")


def _import_image_artifacts(job_id: str, container: str, repo: str
                            ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Copy the alembic artifacts of ``container``'s image into a workdir of its own.

    A server built outside the builds tool (start_chain by hand, later an image
    from the hub) keeps its reports and generated tools only inside the image,
    and the builds page reads them from a build's workdir. Returns (workdir,
    validation counts); (None, None) when the image has none.
    """
    workdir = LOG_DIR / job_id / "workdir"
    base = workdir / repo
    copied = False
    for rel in _IMAGE_ARTIFACTS:
        dest = base / rel
        if dest.exists():  # docker cp into an existing folder nests the source in it
            copied = True
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = _docker("cp", f"{container}:/work/.alembic/{repo}/{rel}", str(dest), timeout=300)
        copied = copied or r.returncode == 0
    if not copied:
        shutil.rmtree(LOG_DIR / job_id, ignore_errors=True)
        return None, None
    try:
        counts = json.loads((base / "reports" / "validation.json").read_text(encoding="utf-8"))["counts"]
    except (OSError, ValueError, KeyError, TypeError):
        counts = None
    return str(workdir), counts


def _reuse_from_host(repo_url: str, scope: Optional[list]
                     ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """A server for ``repo_url`` this host can provide without the pipeline.

    In order: a finished build whose container still runs; a finished build
    whose own image is still here, started again with start_build_server; a
    running alembic server for the repository with no build record (started
    with start_chain.py by hand). None when there is nothing to reuse.
    """
    inv = docker_inventory()
    builds = web_list_builds()
    wanted = _repo_identity(repo_url)
    done = [b for b in builds
            if _repo_identity(b.get("repo_url") or "") == wanted and b.get("status") == "done"]
    for build in done:
        name = build.get("container")
        port = _container_state(name).get("port") \
            if name and inv["containers"].get(name, {}).get("running") else None
        if port:
            job = {**_job(build["job_id"])[1], "container": name, "mcp_url": _advertised_url(port)}
            return _adopt(job, scope, (
                f"This repository was already built ({build['job_id']}) and its server "
                f"{name} is still running: reusing it. Pass force_rebuild=true to rebuild."))
    for build in done:
        if not job_image(_job(build["job_id"])[1], inv):
            continue
        started = start_build_server(build["job_id"])
        if started.get("ok"):
            rec, snap = _adopt({**_job(build["job_id"])[1], **started}, scope, (
                f"This repository was already built ({build['job_id']}) and its image is "
                "still here, but the server was down: started it again from that image "
                "without rerunning the pipeline. Pass force_rebuild=true to rebuild."))
            if not _await_answer(rec, started["mcp_url"]):
                _mark_unserved(snap, {"answering": False}, build["job_id"])
            return rec, snap
        logger.warning("could not serve %s again: %s", build["job_id"], started.get("error"))
    claimed = {b.get("container") for b in builds}
    own = re.compile(rf"alembic-serve-{re.escape(_repo_name(repo_url))}-[0-9a-f]+$")
    for name, container in inv["containers"].items():
        if not (container["running"] and own.match(name)) or name in claimed:
            continue
        port = _container_state(name).get("port")
        if port:
            job_id = f"{_repo_name(repo_url)}-external-{name.rsplit('-', 1)[-1]}"
            imported = _import_image_artifacts(job_id, name, _repo_name(repo_url))
            return _record_external(job_id, name, repo_url, port, container["image_id"],
                                    imported, scope, "reused")
    return None


def _web_flag(name: str, default: bool) -> bool:
    """A boolean web setting; ``default`` when the settings cannot be loaded."""
    try:
        from CoScientist.config import get_settings

        return bool(getattr(get_settings().web, name))
    except Exception:  # noqa: BLE001 - settings unavailable outside the app
        return default


def _agent_may_build() -> bool:
    """Whether an agent may start a conversion (tens of minutes) by itself.
    Reuse and hub pulls are not affected; the builds page always converts."""
    return _web_flag("alembic_agent_build_enabled", False)


def _pull_from_hub(repo_url: str, scope: Optional[list]) -> Optional[Dict[str, Any]]:
    """Start pulling a server converted from ``repo_url`` from the MCP hub; its snapshot, or None.

    None when the hub search setting is off, no hub is configured, the hub has
    no such server, or Docker Hub does not answer: the build goes ahead then.
    """
    try:
        from CoScientist.tools import alembic_hub

        if not alembic_hub.search_enabled():
            return None
        found = alembic_hub.find_for_repo(repo_url)
        if found is None:
            return None
        snap = alembic_hub.start_pull(found["name"], "latest", scope=scope, repo_url=repo_url)
    except Exception as exc:  # noqa: BLE001 - a hub problem must not block a build
        logger.warning("MCP hub lookup for %s failed: %s", repo_url, exc)
        return None
    return snap if snap.get("status") == "running" else None


def _record_external(job_id: str, name: str, repo_url: str, port: str,
                     image_id: Optional[str], imported: Tuple[Optional[str], Optional[Dict[str, Any]]],
                     scope: Optional[list], verb: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Register a running server started outside the builds tool as a build of its own.

    Kept on disk like any build, so it outlives this process, and its page has a
    history line saying where it came from. Returns what _adopt returns.
    """
    workdir, counts = imported
    job = {"job_id": job_id, "repo_url": repo_url, "status": "done",
           "finished_at": time.time(), "mcp_url": _advertised_url(port),
           "container": name, "image_id": image_id,
           "origin": "image" if workdir else "unknown",
           "workdir": workdir, "tool_counts": counts}
    rec, snap = _adopt(job, scope, (
        f"Found a running MCP server for this repository ({name}) that was "
        "started outside this tool: reusing it. Pass force_rebuild=true to rebuild."))
    if counts:
        summary = (f"alembic artifacts copied from its image, {counts.get('tools_passed')} of "
                   f"{counts.get('tools_total')} tools passed validation")
    elif workdir:
        summary = "alembic artifacts copied from its image, without a validation report"
    else:
        summary = "its image carries no alembic artifacts"
    with _LOCK:
        current = dict(rec)
    _write_job_meta(current)
    _log_event(current, f"{verb} {name}, a running server started outside the builds "
                        f"tool; {summary}")
    return rec, snap


# Serve containers carry start_chain's project label, and their name carries the
# repository: alembic-serve-<repo>-<hex>.
_PROJECT_LABEL = "project=coscientist"
_SERVE_NAME_RE = re.compile(r"^alembic-serve-(.+)-([0-9a-f]+)$")
_DISCOVERY_INTERVAL = 15.0
_last_discovery = 0.0


def _plan_repo_url(workdir: Optional[str], repo: str) -> Optional[str]:
    """The repository named in the plan among a server's copied artifacts."""
    if not workdir:
        return None
    try:
        plan = json.loads((Path(workdir) / repo / "reports" / "plan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return plan.get("repo_url") if isinstance(plan, dict) else None


def adopt_unclaimed_servers() -> list:
    """Record every running alembic server no build knows; returns their job ids.

    The repository comes from the image's alembic.repo_url label, else from the
    plan among its artifacts; a server with neither is skipped. The builds list
    polls, so a pass runs at most once per _DISCOVERY_INTERVAL seconds.
    """
    global _last_discovery
    with _LOCK:
        now = time.monotonic()
        if now - _last_discovery < _DISCOVERY_INTERVAL:
            return []
        _last_discovery = now
    r = _docker("ps", "--filter", f"label={_PROJECT_LABEL}", "--filter", "name=alembic-serve-",
                "--format", '{{.Names}}|{{.Label "alembic.repo_url"}}', timeout=30)
    if r.returncode != 0:
        return []
    builds = web_list_builds()
    claimed = {b.get("container") for b in builds}
    known_images = None
    added = []
    for line in r.stdout.splitlines():
        name, _, labelled = line.strip().partition("|")
        match = _SERVE_NAME_RE.match(name)
        if not match or name in claimed:
            continue
        state = _container_state(name)
        if not state.get("port"):
            continue
        # A new container of a known build (a Start that replaced one): its
        # record catches up within seconds, so it gets no second record.
        if known_images is None:
            inv = docker_inventory()
            known_images = {job_image(b, inv) for b in builds} - {None}
        if state.get("image_id") in known_images:
            continue
        repo, suffix = match.groups()
        job_id = f"{repo}-external-{suffix}"
        imported = _import_image_artifacts(job_id, name, repo)
        repo_url = labelled or _plan_repo_url(imported[0], repo)
        if not repo_url:
            shutil.rmtree(LOG_DIR / job_id, ignore_errors=True)
            logger.info("left %s out of the builds list: nothing names its repository", name)
            continue
        rec, _ = _record_external(job_id, name, repo_url, state["port"], state.get("image_id"),
                                  imported, None, "found")
        added.append(rec["job_id"])
    return added


_TASK_SPEC_MAX_CHARS = 100_000


def resolve_task_spec(value: Optional[str]) -> Optional[str]:
    """A task spec as text: a local file or an http(s) link is read here, since
    the build container, often on another daemon, cannot reach this host's paths."""
    value = (value or "").strip()
    if not value:
        return None
    if value.lower().startswith(("http://", "https://")):
        import requests

        resp = requests.get(value, timeout=30)
        resp.raise_for_status()
        text = resp.text
    else:
        try:
            path = Path(value).expanduser()
            text = path.read_text(encoding="utf-8") if path.is_file() else value
        except OSError:
            text = value
    if len(text) > _TASK_SPEC_MAX_CHARS:
        raise ValueError(f"the task spec is over {_TASK_SPEC_MAX_CHARS} characters")
    return text


def _keyed_job(idempotency_key: str, repo_url: str) -> Optional[Dict[str, Any]]:
    """The job a coordinator key already names, or None. Call under _LOCK.

    Raises ValueError when the key belongs to a build of another repository.
    """
    with _registry_file_lock():
        # Refresh only for coordinator-keyed requests: this makes idempotency
        # work across worker processes without changing the process-local
        # reuse behaviour of unkeyed calls.
        _load_jobs_from_disk(merge=True)
        keyed = [
            value for value in _JOBS.values()
            if value.get("idempotency_key") == idempotency_key
        ]
    for existing in reversed(keyed):
        if _repo_identity(existing["repo_url"]) != _repo_identity(repo_url):
            raise ValueError(
                f"idempotency_key {idempotency_key!r} is already "
                "associated with a different repository"
            )
        return existing
    return None


async def build_mcp_server(
    repo_url: str,
    force_rebuild: bool = False,
    hints: Optional[str] = None,
    task_spec: Optional[str] = None,
    max_tools: Optional[int] = None,
    tool_context: Optional[ToolContext] = None,
    idempotency_key: Optional[str] = None,
    run_id: Optional[str] = None,
    task_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Start an Alembic build: turn a GitHub repository into a served MCP tool
    server (clone → env → generated+validated tools → FastMCP server in Docker).

    The build runs in the background. This returns immediately with a job_id;
    track it with check_mcp_build(job_id).

    Args:
        repo_url: GitHub repository URL, e.g. "https://github.com/whitead/synspace".
        force_rebuild: Start a fresh build even if this repo already has a
            build or a running server on this host.
        hints: Free text saying what kind of tool is needed, to steer the
            explorer. It forces no tool name or signature; leave it unset to let
            the explorer propose tools on its own.
        task_spec: For an operator who already knows the exact tools the server
            must expose: their spec as JSON/YAML text, or a path or link to it.
            Those tools are then required, and the build fails without them.
        max_tools: The most tools the server may expose (1-30). Unset, the
            explorer proposes 2-5 and the build keeps at most 12. Applies to a
            new build only; a reused build keeps the tools it has.
        idempotency_key: Coordinator-supplied operation key. Repeating the same
            key for the same repository always returns the original job.
        run_id: Optional experiment run associated with this build.
        task_id: Optional experiment task associated with this build.
        attempt_id: Optional task attempt associated with this build.

    Returns:
        status "running" with the job_id to check later; or, unless force_rebuild
        is set, an existing build of this repo, a server already on this host, or
        a server being pulled from the MCP hub (also "running", checked the same way).
    """
    from CoScientist.capabilities import alembic_enabled, capability_disabled

    if not alembic_enabled():
        return capability_disabled("Alembic MCP builder", "EXPERIMENTS__ROUTE_ALEMBIC")

    repo_url = (repo_url or "").strip()
    if not re.match(r"^(https?://|git@)\S+/\S+", repo_url):
        return {"status": "error",
                "error": f"repo_url does not look like a git repository URL: {repo_url!r}"}
    if max_tools is not None and max_tools != "":
        from CoScientist.alembic.config import MAX_TOOLS_LIMIT as limit

        try:
            max_tools = int(max_tools)
        except (TypeError, ValueError):
            max_tools = 0
        if not 1 <= max_tools <= limit:
            return {"status": "error", "repo_url": repo_url,
                    "error": f"max_tools must be a whole number from 1 to {limit}"}
    else:
        max_tools = None
    # The daemon Alembic would use (DOCKER_HOST or the active context) must
    # answer before anything else: a build, a restart of a served container
    # and a hub pull all go through it. A DNS name that does not resolve used
    # to surface minutes into the job; the agent is told at once instead, with
    # the way out it actually has.
    preflight = await asyncio.to_thread(alembic_preflight)
    if not preflight.get("available"):
        docker_host = os.environ.get("DOCKER_HOST") or "local daemon"
        return {
            "status": "error",
            "error_code": "docker_unavailable",
            "repo_url": repo_url,
            "error": f"the Docker daemon Alembic uses is unreachable ({docker_host}): "
                     f"{preflight.get('reason') or 'docker info failed'}",
            "note": ("Nothing was built and nothing can be reused or pulled while the "
                     "daemon is down: this is the host's environment (DOCKER_HOST, VPN, "
                     "DNS), not the repository. Do not retry the build in this session. "
                     "If the computation itself is what matters, run the repository as "
                     "code through CoderAgent (clone it in the sandbox) and say in your "
                     "answer that no reusable MCP server was made."),
        }

    idempotency_key = (idempotency_key or "").strip() or None
    associations = {
        "idempotency_key": idempotency_key,
        "run_id": (run_id or "").strip() or None,
        "task_id": (task_id or "").strip() or None,
        "attempt_id": (attempt_id or "").strip() or None,
    }

    from CoScientist.graph.session_scope import session_key

    scope = list(session_key(tool_context)) if tool_context is not None else None

    if idempotency_key is not None:
        # A coordinator retrying its own operation gets its own job back, in
        # whatever state it is, before anything else is looked at.
        try:
            with _LOCK:
                keyed = _keyed_job(idempotency_key, repo_url)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        except OSError as exc:
            return {"status": "error",
                    "error": f"could not read the Alembic build registry: {exc}"}
        if keyed is not None:
            served = await asyncio.to_thread(_ensure_served, keyed)
            with _LOCK:
                _claim(keyed, scope)
                snap = _reuse_snapshot(keyed, idempotent=True)
            if served.get("started"):
                snap["note"] += " " + _RESTARTED_NOTE
            elif not served["ok"]:
                _mark_unserved(snap, served, keyed["job_id"])
            await _publish_to_catalogue(keyed, snap, tool_context)
            return snap

    # A guessed repository would fail only at the clone step, minutes into the build.
    ok, why = _repo_exists(repo_url)
    if not ok:
        return {
            "status": "error",
            "repo_url": repo_url,
            "error": f"repository is not reachable: {why}",
            "note": ("Nothing was built. Find the real repository of this "
                     "paper/library (search the web or the paper's text for its "
                     "code link) and call build_mcp_server again with the URL "
                     "you verified."),
        }

    try:
        task_spec = resolve_task_spec(task_spec)
    except Exception as exc:  # noqa: BLE001 - nothing is built on a spec we cannot read
        return {"status": "error", "repo_url": repo_url,
                "error": f"the task spec could not be read: {type(exc).__name__}: {exc}"}
    hints = (hints or "").strip() or None

    reuse = None
    done = None
    with _LOCK:
        if not force_rebuild:
            # Prefer a live build; else the most recent finished one.
            found = _first_reusable_same_repo(repo_url)
            if found is not None and found["status"] == "running":
                _claim(found, scope)
                return _reuse_snapshot(found)
            done = found
    if done is not None:
        # A server that is down and does not start leaves the build unusable:
        # the host cascade below and then a new build take over.
        served = await asyncio.to_thread(_ensure_served, done)
        if served["ok"] or served.get("answering") is False:
            with _LOCK:
                _claim(done, scope)
                snap = _reuse_snapshot(done)
            if not served["ok"]:
                # Started and still loading: a new build would only take longer.
                _mark_unserved(snap, served, done["job_id"])
                return snap
            if served["started"]:
                snap["note"] += " " + _RESTARTED_NOTE
            reuse = (done, snap)
    if reuse is None and not force_rebuild:
        # Nothing in this process's memory, e.g. after a restart. An earlier
        # build may still serve, or its image can be served again in seconds.
        reuse = await asyncio.to_thread(_reuse_from_host, repo_url, scope)
    if reuse is None and not force_rebuild:
        pulled = await asyncio.to_thread(_pull_from_hub, repo_url, scope)
        if pulled is not None:
            return pulled
    if reuse is not None:
        # A reused server must reach the session the same way a fresh one does,
        # or the executor cannot call it: publish it into `deployed_mcps`.
        await _publish_to_catalogue(reuse[0], reuse[1], tool_context)
        return reuse[1]

    # tool_context is the mark of an agent call; the builds page passes none.
    if tool_context is not None and not _agent_may_build():
        return {
            "status": "error",
            "repo_url": repo_url,
            "error": "converting a repository is turned off for agents in the settings "
                     "(Alembic MCP: agent may convert repositories)",
            "note": ("No server for this repository runs on this host, and the MCP hub has "
                     "none either. Say so in your answer and go on with the tools you have; "
                     "an operator can convert the repository on the MCP builder page."),
        }

    try:
        with _LOCK:
            with _registry_file_lock():
                repo_prefix = re.sub(r"[^A-Za-z0-9._-]", "-", _repo_name(repo_url))
                job_id = f"{repo_prefix}-{secrets.token_hex(3)}"
                rec: Dict[str, Any] = {
                    "job_id": job_id,
                    "repo_url": repo_url,
                    "status": "running",
                    "started_at": time.time(),
                    "log_file": str(LOG_DIR / f"{job_id}.log"),
                    # Per-job workdir: the build's artifacts land here on the
                    # host instead of dying with the container.
                    "workdir": str(LOG_DIR / job_id / "workdir"),
                    "scopes": [scope] if scope else [],
                    "origin": "builder",
                    "hints": hints,
                    "task_spec": task_spec,
                    "max_tools": max_tools,
                    **{
                        key: value
                        for key, value in associations.items()
                        if value is not None
                    },
                }
                _evict_finished_jobs()
                _JOBS[job_id] = rec
                try:
                    _persist_job(rec)
                except OSError:
                    del _JOBS[job_id]
                    raise
    except OSError as exc:
        return {
            "status": "error",
            "error": f"could not persist Alembic build metadata: {exc}",
        }

    threading.Thread(target=_runner, args=(rec,), daemon=True,
                     name=f"alembic-build-{job_id}").start()
    with _LOCK:
        result = _snapshot(rec, with_log_tail=False)
    result["note"] = ("Build started (base image → pipeline → docker commit → serve). "
                      "A full build takes tens of minutes: report the job_id back, do "
                      f"other work, and call check_mcp_build('{job_id}') later. That "
                      "call is the only source of this build's result; an MCP server "
                      "found any other way on this host belongs to some earlier "
                      "build.")
    return result


# ── Experiment-module API ─────────────────────────────────────────────────────
# Synchronous views of the registry for the state machine, which runs in
# callbacks with no tool context and may live in another process than the
# build (hence the reload from disk).

def _known_job(job_id: str) -> Optional[Dict[str, Any]]:
    """The record for ``job_id``, loading the registry when this process has
    not seen it. Call under _LOCK."""
    rec = _JOBS.get(job_id)
    if rec is None:
        try:
            _load_jobs_from_disk(merge=True)
        except OSError as exc:
            logger.warning("could not reload the Alembic build registry: %s", exc)
        rec = _JOBS.get(job_id)
    return rec


def peek_mcp_build(job_id: str) -> Dict[str, Any]:
    """Sync snapshot of one job (no wait). Reloads durable metadata if needed."""
    with _LOCK:
        rec = _known_job(job_id)
        if rec is None:
            return {
                "status": "error",
                "error": (
                    f"unknown job_id {job_id!r} — use list_mcp_builds() "
                    "to see the builds known to this registry."
                ),
            }
        return _snapshot(rec, with_log_tail=rec["status"] != "running")


def wait_mcp_build(
    job_id: str,
    *,
    timeout_s: float = 1800.0,
    poll_s: float = 5.0,
) -> Dict[str, Any]:
    """Block until a build is done/failed/error, or ``timeout_s`` elapses.

    Experiment Module uses this so the executor never records a premature
    failure while Docker is still building. Interactive McpBuilder turns
    keep the async protocol (return immediately, check later).
    """
    deadline = time.time() + max(0.0, float(timeout_s))
    poll = max(0.05, float(poll_s))
    last = peek_mcp_build(job_id)
    while last.get("status") == "running":
        if time.time() >= deadline:
            out = dict(last)
            out["wait_timed_out"] = True
            out["note"] = (
                f"Build still running after {timeout_s:.0f}s — "
                f"reuse job {job_id}; do not start a new build."
            )
            return out
        time.sleep(poll)
        last = peek_mcp_build(job_id)
    return last


def _recorded_tools(mcp_url: str = "", job_id: str = "") -> list[dict[str, Any]]:
    """The tool list a record already holds for this server (taken when it
    answered MCP), so the experiment module needs no second round trip."""
    with _LOCK:
        for rec in _JOBS.values():
            if job_id and rec["job_id"] != job_id:
                continue
            if mcp_url and rec.get("mcp_url") != mcp_url:
                continue
            tools = rec.get("tools")
            if isinstance(tools, list) and tools:
                return [dict(tool) for tool in tools]
    return []


def list_served_mcp_tools(mcp_url: str, timeout_s: float = 8.0) -> list[dict[str, Any]]:
    """``tools/list`` of a served FastMCP endpoint, as the bridge consumes it.

    The build record's own list is used when it has one; else the server is
    asked, which also yields each tool's input schema. Used after WAIT_DONE so
    post-build Fedot sees real tool names instead of the ``alembic_built_tool``
    placeholder. Never raises — empty list on miss.
    """
    url = (mcp_url or "").strip()
    if not url.startswith("http"):
        return []
    recorded = _recorded_tools(mcp_url=url)
    if recorded:
        return recorded
    timeout_s = max(1.0, float(timeout_s))

    async def _list() -> list[dict[str, Any]]:
        from mcp import ClientSession
        try:
            from mcp.client.streamable_http import streamable_http_client as _http_client
        except ImportError:  # older mcp SDK
            from mcp.client.streamable_http import streamablehttp_client as _http_client

        async with _http_client(url) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                out: list[dict[str, Any]] = []
                for tool in listed.tools or []:
                    name = str(getattr(tool, "name", "") or "").strip()
                    if not name:
                        continue
                    item: dict[str, Any] = {
                        "name": name,
                        "description": str(getattr(tool, "description", "") or "")[:500],
                    }
                    schema = getattr(tool, "inputSchema", None) or getattr(
                        tool, "input_schema", None
                    )
                    if schema:
                        item["input_schema"] = schema
                    out.append(item)
                return out

    async def _list_with_timeout() -> list[dict[str, Any]]:
        return await asyncio.wait_for(_list(), timeout=timeout_s)

    def _run_isolated() -> list[dict[str, Any]]:
        return asyncio.run(_list_with_timeout())

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return _run_isolated()
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run_isolated).result(timeout=timeout_s + 2)
    except Exception as exc:
        logger.warning("list_served_mcp_tools failed url=%s err=%s", url, exc)
        return []


def enrich_snapshot_with_tools(snap: dict[str, Any], timeout_s: float = 8.0) -> dict[str, Any]:
    """Attach ``tools`` from the build record or the live MCP when the build
    snapshot omitted them."""
    if not isinstance(snap, dict) or snap.get("status") != "done":
        return snap
    existing = snap.get("tools") or snap.get("mcp_tools")
    if isinstance(existing, list) and existing:
        return snap
    url = str(snap.get("mcp_url") or "").strip()
    tools = _recorded_tools(job_id=str(snap.get("job_id") or "")) or \
        list_served_mcp_tools(url, timeout_s=timeout_s)
    if not tools:
        return snap
    out = dict(snap)
    out["tools"] = tools
    return out


async def check_mcp_build(
    job_id: str, tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """Check an Alembic build started by build_mcp_server.

    Args:
        job_id: The id returned by build_mcp_server.

    Returns:
        status "running" with the current pipeline stage and a log tail;
        "done" with the served MCP endpoint (mcp_url), image and container;
        or "failed" with the error tail of the build log. A finished build
        whose server was stopped has its server started again first; when
        that fails, the result has no mcp_url and server_running is false.
    """
    from CoScientist.graph.session_scope import session_key

    # A long-lived process keeps builds across chats, and McpBuilderAgent reaches
    # a known one through list_mcp_builds + check_mcp_build, never through
    # build_mcp_server. Claim it here too, or its server is not attached to this
    # session's executor and the task falls through to CoderAgent.
    scope = list(session_key(tool_context)) if tool_context is not None else None
    with _LOCK:
        rec = _known_job(job_id)
        if rec is None:
            return {"status": "error",
                    "error": f"unknown job_id {job_id!r} — use list_mcp_builds() "
                             "to see the builds known to this registry."}
    served = await asyncio.to_thread(_ensure_served, rec)
    with _LOCK:
        _claim(rec, scope)
        out = _snapshot(rec)
    if served.get("started"):
        out["note"] = _RESTARTED_NOTE
    elif not served["ok"]:
        # An address of a server that cannot be called is a dead end for the executor.
        _mark_unserved(out, served, job_id)
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
        logger.info("catalogue registration skipped for %s: loopback address %s",
                    _repo_name(rec["repo_url"]), rec["mcp_url"])
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


_TOOL_LIST_TIMEOUT = 20


async def _served_tools(mcp_url: str) -> Optional[list]:
    """Names and one-line descriptions of the tools a served MCP server lists.

    The build result carries them so that the agents downstream call the tools
    by their real names. Without them the caller guessed a name ("sample_space"
    for synspace's chemical_space) and the call went nowhere. None when the
    server does not answer in time, so a server with no tools is told apart.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def _list():
        async with streamablehttp_client(mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return (await session.list_tools()).tools

    try:
        listed = await asyncio.wait_for(_list(), _TOOL_LIST_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - the build result stands without the list
        # Debug only: _wait_until_answering expects refusals while a server loads.
        logger.debug("could not list the tools of %s: %s", mcp_url, exc)
        return None
    return [{"name": tool.name,
             "description": ((tool.description or "").strip().splitlines() or [""])[0][:160]}
            for tool in listed]


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
    tools = rec.get("tools")
    if not tools:
        tools = await _served_tools(out["mcp_url"])
        if tools is None:
            logger.warning("could not list the tools of %s", out["mcp_url"])
    if tools:
        with _LOCK:
            rec["tools"] = tools
        out["tools"] = tools

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
        A finished build also has server_running; when it is false,
        check_mcp_build(job_id) starts the server again.
    """
    with _LOCK:
        builds = [(rec.get("container") if rec["status"] == "done" else None,
                   _snapshot(rec, with_log_tail=False)) for rec in _JOBS.values()]
    for container, snap in builds:
        if container:
            state = await asyncio.to_thread(_container_state, container)
            snap["server_running"] = state["running"]
    return {"builds": [snap for _, snap in builds]}


def live_build_servers(scope: Optional[Tuple[str, str]]) -> Dict[str, str]:
    """{url: name} of the running servers of finished builds ``scope`` built or reused.

    A fallback for ``deployed_mcps``, which the nested McpBuilderAgent invocation
    does not always pass on to the agent that calls the tool.
    """
    if not scope:
        return {}
    scope = list(scope)
    with _LOCK:
        recs = [dict(r) for r in _JOBS.values() if scope in r.get("scopes", [])]
    servers: Dict[str, str] = {}
    for rec in recs:
        url, container = rec.get("mcp_url"), rec.get("container")
        if rec.get("status") != "done" or not url or not container:
            continue
        if _container_state(container)["running"]:
            servers[url] = _repo_name(rec["repo_url"])
    return servers


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
    if _SERVE_BANNER in text or '"status": "complete"' in text:
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


_RESUMABLE_STAGES = ("explorer", "environment", "coder", "validator", "wrapper")


def resume_build(job_id: str, stage: str) -> Dict[str, Any]:
    """Run a finished build again from ``stage`` in its own workdir.

    The stages before ``stage`` are not repeated: their output (the plan, the
    venv, the generated tools) is read from the workdir the build left. A
    FEDOT build that failed in the coder stage kept a 3 GB environment that
    took 25 minutes to install; resuming at the coder reuses it.
    """
    if stage not in _RESUMABLE_STAGES:
        return {"ok": False, "error": f"stage must be one of {', '.join(_RESUMABLE_STAGES)}"}
    with _LOCK:
        rec = _JOBS.get(job_id)
        if rec is None:
            meta = _read_job_meta(job_id)
            if meta is None:
                return {"ok": False, "error": f"unknown build {job_id}"}
            rec = dict(meta)
            _JOBS[job_id] = rec
        if rec.get("status") == "running":
            return {"ok": False, "error": f"build {job_id} is still running"}
        if not rec.get("workdir") or not Path(rec["workdir"]).exists():
            return {"ok": False, "error": f"build {job_id} has no workdir to resume from"}
        rec.update({
            "status": "running",
            "resume_from": stage,
            "resumed_at": time.time(),
            "started_at": time.time(),
        })
        for key in ("finished_at", "returncode", "error", "mcp_url", "container", "image",
                    "image_id", "registered", "registration_error"):
            rec.pop(key, None)
        _persist_quietly(rec)
    threading.Thread(target=_runner, args=(rec,), daemon=True,
                     name=f"alembic-resume-{job_id}").start()
    return {"ok": True, "job_id": job_id, "resume_from": stage,
            "progress_page": f"/alembic/builds/{job_id}"}


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
    counts = _validator_counts(text) or meta.get("tool_counts")
    if counts:
        out["tool_counts"] = counts
    if meta.get("origin"):
        out["origin"] = meta["origin"]
    if meta.get("hub"):
        out["hub"] = meta["hub"]
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
    # `docker images` leaves out an image that lost its last tag, which happens
    # when a newer build of the repository takes alembic-tool:<repo>. A container
    # made from it still runs or can start again, so the image is added by id;
    # without it that server showed "no image" and no controls on the builds page.
    missing = sorted({c["image_id"] for c in inv["containers"].values()} - set(inv["images"]))
    if missing:
        r = _docker("image", "inspect", "-f", "{{.Id}}|{{.Size}}", *missing, timeout=30)
        for line in r.stdout.splitlines():
            image_id, _, size = line.partition("|")
            if image_id in missing and size.isdigit():
                inv["images"][image_id] = _docker_size(int(size))
    return inv


def _docker_size(size: int) -> str:
    """Bytes the way `docker images` prints them (decimal units, three digits)."""
    value = float(size)
    for unit in ("B", "kB", "MB", "GB"):
        if value < 1000:
            return f"{value:.3g}{unit}"
        value /= 1000
    return f"{value:.3g}TB"


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


def _serve_new_container(job: Dict[str, Any], image_ref: str,
                         port: Optional[str] = None) -> tuple[Optional[str], str]:
    """Serve the image in a new container through start_chain --serve-only, so
    the port, env file, GPU flag and start check match a build's own serve.
    ``port`` is the host port to publish on; by default a random one."""
    cmd = [sys.executable, str(START_CHAIN), job["repo_url"], "--serve-only", "--image", image_ref]
    if port:
        cmd += ["--port", str(port)]
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


def _s3_fingerprint(name: str, value: str) -> str:
    """Same rule as start_chain.s3_fingerprint: keys are compared by SHA-256."""
    if value and name.endswith(("ACCESS_KEY", "SECRET_KEY")):
        return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return value


def _container_env(name: str) -> Dict[str, str]:
    r = _docker("inspect", "-f", "{{json .Config.Env}}", name, timeout=30)
    try:
        return dict(item.split("=", 1) for item in json.loads(r.stdout) if "=" in item)
    except (ValueError, TypeError):
        return {}


def _container_host_port(name: str) -> Optional[str]:
    """The host port a (possibly stopped) container publishes its MCP port on."""
    r = _docker("inspect", "-f", "{{json .HostConfig.PortBindings}}", name, timeout=30)
    try:
        return json.loads(r.stdout)["8000/tcp"][0]["HostPort"] or None
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def _expected_s3_settings(repo_url: str) -> Optional[Dict[str, str]]:
    """The S3 settings start_chain would give a serve container now (--serve-env)."""
    try:
        r = subprocess.run([sys.executable, str(START_CHAIN), repo_url, "--serve-env"],
                           capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=60, check=False)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        logger.warning("could not read the S3 settings start_chain would use: %s", exc)
        return None


def _s3_settings_changed(name: str, repo_url: str) -> list:
    """Names of the S3 settings container ``name`` was created with that differ
    from what a new serve container would get now. Empty when they match or
    cannot be compared, so an unreadable setting never costs a container."""
    expected = _expected_s3_settings(repo_url)
    if not expected:
        return []
    current = _container_env(name)
    return sorted(key for key, value in expected.items()
                  if (_s3_fingerprint(key, current.get(key, "")) or "") != (value or ""))


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
    own = state["exists"] and state.get("image_id") == image
    # A container keeps the environment it was created with. After .env moves to
    # another S3 (a local MinIO to the shared vault), docker start would bring back
    # the old endpoint and keys, so such a container is replaced instead.
    changed = _s3_settings_changed(name, job["repo_url"]) if own else []
    verb = "started"
    if own and not changed and _docker("start", name).returncode == 0:
        if not _stays_up(name):
            return {"ok": False, "error": _exited_error(name)}
    else:
        port = None
        if changed:
            # The replacement takes its port, so the address and the catalogue row
            # stay; removing it first frees that port.
            port = _container_host_port(name)
            _docker("rm", name)
            verb = f"S3 settings changed ({', '.join(changed)}), so started a new container"
        own_tag = f"{_TOOL_IMAGE}:{job_id}"
        new_name, error = _serve_new_container(
            job, own_tag if inv["tags"].get(own_tag) == image else image, port)
        if not new_name:
            return {"ok": False, "error": error}
        if state["exists"] and not changed:
            _docker("rm", name)  # the container this one replaces
        name = new_name
    return _now_serving(job, rec, name, verb, container=name, image_id=image)


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


def remove_build_container(job_id: str) -> Dict[str, Any]:
    """Stop and remove a build's container, keeping the image. Other builds on
    the same image are untouched, and Start makes this one a new container."""
    _, job = _job(job_id)
    name = job.get("container")
    if not _container_state(name)["exists"]:
        return {"ok": False, "error": "this build has no container on this host"}
    result = stop_build_server(job_id)
    if not result["ok"]:
        return result
    r = _docker("rm", "-f", name)
    if r.returncode != 0:
        return {"ok": False, "error": f"docker rm failed: {(r.stderr or r.stdout).strip()[-500:]}"}
    _log_event(job, f"removed container {name}; the image stays")
    return {**result, "removed_container": name}


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
    for path in {Path(job.get("log_file") or LOG_DIR / f"{job_id}.log"),
                 _meta_path(job_id), _metadata_file(job_id)}:
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


try:
    _load_jobs_from_disk()
except OSError:
    # Tool import must remain available even if a mounted metadata directory is
    # temporarily unreadable. New builds fail explicitly when persistence is
    # attempted, rather than breaking unrelated agent assembly.
    pass


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


__all__ = ["ALEMBIC_TOOLS", "alembic_preflight", "build_mcp_server", "check_mcp_build", "list_mcp_builds",
           "peek_mcp_build", "wait_mcp_build", "list_served_mcp_tools",
           "enrich_snapshot_with_tools", "reload_mcp_builds", "live_build_servers",
           "web_build_log_file", "web_build_snapshot", "web_build_workdir",
           "web_build_repo_url", "web_list_builds", "cancel_build", "parse_event_line",
           "adopt_unclaimed_servers", "resolve_task_spec",
           "docker_inventory", "job_image", "start_build_server", "restart_build_server",
           "stop_build_server", "remove_build_container", "delete_build_image",
           "non_runnable_builds", "clear_non_runnable_builds",
           "export_jobs_snapshot", "import_jobs_snapshot", "LOG_DIR", "JOB_METADATA_DIR"]
