"""Tools for the CoderAgent: bash/code execution, file I/O, package install.

Execution is routed to a remote code-execution server (submit + poll), giving the
agent a real sandbox for long-running work — cloning repos, running scripts and
git, building/processing data — without blocking on a short subprocess timeout.
Each session gets its own workspace id so concurrent sessions don't clobber each
other's files.

If no code-exec server is configured (settings.code_exec.url is empty), the
toolset transparently falls back to local subprocess execution (background
threads) in a per-session workspace directory.
"""

import asyncio
import os
import re
import shlex
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
import time
import sys
from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from CoScientist.config import get_settings
from CoScientist.graph.session_scope import DEFAULT_SESSION_KEY, session_key

settings = get_settings()
_CFG = settings.code_exec

# Process-wide registry of background jobs for the local-fallback execution path
# (keyed by job_id). The remote path keeps job state on the code-exec server.
_LOCAL_JOBS: Dict[str, Dict[str, Any]] = {}
_MAX_LOCAL_JOBS = 500  # cap registry size; evict oldest finished jobs past this

# Session-state key pinning the per-ADK-session sandbox workspace. Shared with
# the orchestrator's seed callback (seed_coder_workspace) so every CoderAgent
# delegation in one ADK session — and any sub-agent that uses this toolset —
# lands in the SAME sandbox, across user messages.
_WORKSPACE_STATE_KEY = "coder_workspace_id"


def _evict_finished_jobs() -> None:
    """Drop oldest finished job records so _LOCAL_JOBS doesn't grow unbounded.

    Dicts preserve insertion order, so the front holds the oldest jobs. Running
    jobs are never evicted (their threads still write to the record).
    """
    overflow = len(_LOCAL_JOBS) - _MAX_LOCAL_JOBS
    if overflow <= 0:
        return
    for job_id, rec in list(_LOCAL_JOBS.items()):
        if overflow <= 0:
            break
        if rec.get("status") != "running":
            del _LOCAL_JOBS[job_id]
            overflow -= 1


# ─── Safety blocklist ────────────────────────────────────────────────────────
# Each entry is a regex pattern that, if matched, causes the command to be
# rejected before execution.

_BLOCKED: List[re.Pattern] = [p for p in map(re.compile, [
    r"rm\s+.*--no-preserve-root",       # explicit override of root protection
    r"rm\s+(-[a-z]*f[a-z]*\s+|--force\s+).*[/~]\s*$",  # rm -rf / or ~/
    r"rm\s+-[a-z]*r[a-z]*\s+/",         # rm -r /...
    r"mkfs",                             # reformat filesystem
    r"dd\s+.*if\s*=\s*/dev/",           # read raw disk
    r"dd\s+.*of\s*=\s*/dev/[hs]d",     # write to raw disk device
    r">\s*/dev/[hs]d[a-z]",             # redirect into disk device
    r"chmod\s+.*-R.*\s+/\s*$",          # chmod 777 /
    r"chown\s+.*-R.*\s+/\s*$",          # chown -R /
    r":\(\)\s*\{[^}]*\|[^}]*&[^}]*\}",  # fork bomb  :(){ :|:& };:
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\bpoweroff\b",
    r"\binit\s+[06]\b",
    r"mv\s+/\s",                         # mv / somewhere
    r"(curl|wget)\s+.*\|\s*(ba|da|z|fi)?sh",  # curl|bash piping
    r"python[23]?\s+-c.*os\.system.*rm", # sneaky python rm
])]


def _is_dangerous(command: str) -> Optional[str]:
    """Return the matching pattern string if the command is blocked, else None."""
    for pattern in _BLOCKED:
        if pattern.search(command):
            return pattern.pattern
    return None


# ─── Approval list ───────────────────────────────────────────────────────────
# Commands that are allowed but outward-facing or hard to reverse: they require
# human approval (HITL) before running when a handler is configured. This is a
# deterministic gate — it does not depend on the model remembering to ask.

_REQUIRES_APPROVAL: List[re.Pattern] = [p for p in map(re.compile, [
    r"\bgit\s+push\b",                      # publishing to a remote
    r"\bgit\s+remote\s+(add|set-url)\b",    # repointing a remote
    r"\bpip\s+install\b", r"\bpip3\s+install\b",
    r"\b(apt|apt-get|yum|dnf|brew|conda)\s+(install|remove|update|upgrade)\b",
    r"\brm\s+-[a-z]*r",                      # recursive delete
    r"\brm\s+-[a-z]*f",                      # force delete
    r"\bmv\s+",                              # moves can clobber
    r"(curl|wget)\b.*\|\s*(ba|da|z|fi)?sh", # pipe-to-shell
    r"(curl|wget)\b",                        # network fetches
    r"\bdocker\s+(run|rm|rmi|build|push)\b",
    r"\bgh\s+",                              # GitHub CLI (PRs, releases, etc.)
])]


def _requires_approval(command: str) -> Optional[str]:
    """Return the matching pattern if the command needs human approval, else None."""
    for pattern in _REQUIRES_APPROVAL:
        if pattern.search(command):
            return pattern.pattern
    return None


# ─── Toolset ─────────────────────────────────────────────────────────────────

class CoderToolset(BaseToolset):
    """Toolset for software development tasks: bash, file I/O, package install."""

    def __init__(self, prefix: str = "coder_", hitl_handler=None):
        super().__init__()
        self.tool_name_prefix = prefix
        # When set, outward-facing / hard-to-reverse commands (see
        # _REQUIRES_APPROVAL) must be approved by a human before they run.
        self._hitl_handler = hitl_handler

    # ── workspace ──────────────────────────────────────────────────────────────

    @staticmethod
    def _session_id(tool_context: Optional[ToolContext]) -> Optional[str]:
        """Best-effort session id, robust across ADK versions."""
        if tool_context is None:
            return None
        scope = session_key(tool_context)
        if scope != DEFAULT_SESSION_KEY:
            return scope[1]
        inv = (
            getattr(tool_context, "_invocation_context", None)
            or getattr(tool_context, "invocation_context", None)
        )
        session = getattr(inv, "session", None) if inv is not None else None
        sid = getattr(session, "id", None)
        return str(sid) if sid else None

    @staticmethod
    def _workspace_id(tool_context: Optional[ToolContext]) -> str:
        """One sandbox workspace per ADK session, reused by ALL tool calls.

        Every CoderAgent delegation runs in its OWN ephemeral AgentTool
        sub-session with a random session id, so anchoring on the session id
        directly would spawn a fresh, empty workspace on every call (a clone in
        one call, `cd repo` in the next → fails). Instead we pin the workspace
        id in SESSION STATE: AgentTool copies the parent session state into each
        sub-session and forwards state deltas back, so the id written on first
        use is reused by every later CoderAgent call in the same top-level ADK
        session — and persists across user messages (the session state is
        persisted between messages).

        Priority: an explicit CODER_WORKSPACE_ID pin (e.g. the A2A shared
        workspace) wins; otherwise the state-pinned id; otherwise we mint one
        (from the session id, else random) and store it. Read at call time so it
        doesn't depend on import order.
        """
        fixed = settings.web.coder_workspace_id or os.getenv("CODER_WORKSPACE_ID")
        if fixed:
            safe = re.sub(r"[^A-Za-z0-9_\-]", "", fixed)[:48] or "shared"
            return f"ws_{safe}"

        if tool_context is None:
            return "default"

        existing = tool_context.state.get(_WORKSPACE_STATE_KEY)
        if existing:
            return existing

        # First use in this session: mint a stable id and pin it in state.
        seed = CoderToolset._session_id(tool_context) or uuid.uuid4().hex[:12]
        ws = f"ws_{re.sub(r'[^A-Za-z0-9_-]', '', str(seed))[:48] or 'default'}"
        tool_context.state[_WORKSPACE_STATE_KEY] = ws
        return ws

    def _local_workspace(self, tool_context: Optional[ToolContext]) -> Path:
        path = Path(_CFG.workspace_root) / self._workspace_id(tool_context)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_tools(self, readonly_context: Optional[ReadonlyContext]) -> List[BaseTool]:
        return [
            self.execute_bash,
            self.check_job,
            self.read_file,
            self.write_file,
            self.list_directory,
            self.install_package,
        ]

    async def close(self) -> None:
        await asyncio.sleep(0)

    # ── bash ─────────────────────────────────────────────────────────────────

    async def _maybe_request_approval(
        self,
        command: str,
        tool_context: ToolContext,
    ) -> Optional[Dict[str, Any]]:
        """If the command needs approval and a handler is set, ask the human.

        Returns a denial dict when the human rejects (so execute_bash should
        return it and skip execution), or None to proceed.
        """
        matched = _requires_approval(command)
        if matched is None or self._hitl_handler is None or not settings.web.hitl_enabled:
            return None

        # Imported lazily to keep the toolset usable without the HITL package.
        from CoScientist.hitl.models import HITLRequest, HITLAction

        user_id, session_id = session_key(tool_context)
        request = HITLRequest(
            agent_name="CoderAgent",
            action_type=HITLAction.APPROVE,
            message=(
                "CoderAgent wants to run a command that is outward-facing or hard "
                f"to reverse:\n\n    {command}\n\nApprove execution?"
            ),
            context={
                "command": command,
                "matched_rule": matched,
                "_session": {
                    "user_id": user_id,
                    "session_id": session_id,
                },
            },
            invoked_via="callback",
        )
        response = await self._hitl_handler.handle_request(request)
        if response.approved:
            return None
        return {
            "status": "denied",
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "message": (
                "Command requires human approval and was rejected "
                f"(matched: {matched!r}). "
                + (response.instructions or response.free_input or "")
            ).strip(),
        }

    async def execute_bash(
        self,
        command: str,
        timeout: int = None,
        tool_context: ToolContext = None,
    ) -> Dict[str, Any]:
        """
        Run a shell command in this session's isolated sandbox workspace and
        return its result.

        This WAITS for the command to finish and returns its stdout, stderr and
        exit_code directly — for almost everything (git clone, pip install, a
        script, data processing) you get the result in this single call and do
        NOT need check_job. Only if the command is still running after an inline
        wait (a genuinely long build or training run) does it return status
        "running" with a `job_id`; call `check_job(job_id)` ONCE later for it —
        never poll in a tight loop, that just wastes turns.

        Use this to run scripts, build and test code, run git commands (clone,
        commit, push), and process data.

        Working directory is always the session sandbox (do NOT pass absolute host
        paths); all relative paths resolve inside it and persist across calls.

        Dangerous commands (rm -rf /, mkfs, dd to disk, fork bomb, etc.) are
        blocked and will never be executed. Outward-facing or hard-to-reverse
        commands (git push, package installs, recursive/force deletes, network
        fetches) may require human approval before running.

        Args:
            command: Shell command to run.
            timeout: Maximum seconds the job may run before it is killed
                (default: server's long-job timeout, ~30 min).

        Returns:
            Dict with status ("success" | "error" | "timeout" | "blocked"),
            stdout, stderr, exit_code, and workspace_id. For a long job that
            outlives the inline wait: status "running" with a `job_id` to
            check_job later.
        """
        blocked_by = _is_dangerous(command)
        if blocked_by:
            return {
                "status": "blocked",
                "blocked_by": blocked_by,
                "job_id": None,
                "message": f"Command blocked by safety policy (matched: {blocked_by!r}). "
                           "Refused to execute.",
            }

        # Human approval gate for outward-facing / hard-to-reverse commands.
        approval = await self._maybe_request_approval(command, tool_context)
        if approval is not None:
            return approval

        timeout = timeout or _CFG.default_timeout
        workspace_id = self._workspace_id(tool_context)

        if _CFG.url:
            started = await self._submit_remote(command, timeout, workspace_id)
        else:
            started = await self._submit_local(command, timeout, workspace_id, tool_context)

        job_id = started.get("job_id")
        if not job_id or started.get("status") != "running":
            return started  # blocked / denied / submit error — nothing to await

        # Wait for the command INSIDE the tool (a Python poll loop, no LLM
        # round-trips) so the common case returns its result in a single call.
        # Only a job that outlives exec_wait comes back as "running" + job_id.
        res = await self._await_job(job_id, _CFG.exec_wait)
        if res.get("status") == "running":
            res = dict(res)
            res["job_id"] = job_id
            res["message"] = (
                f"Still running after {_CFG.exec_wait}s — this is a long job. "
                f"Do other work and call check_job('{job_id}') once later; "
                "do NOT poll it in a tight loop."
            )
        return res

    async def _await_job(self, job_id: str, max_wait: float) -> Dict[str, Any]:
        """Poll a job internally until it finishes or `max_wait` seconds elapse.

        No LLM involvement — this is a Python loop. Returns the final result, or
        the last "running" status if the job is still going when max_wait is hit.
        """
        elapsed = 0.0
        interval = 0.25
        while True:
            res = await self._check_remote(job_id) if _CFG.url else self._check_local(job_id)
            if res["status"] != "running" or elapsed >= max_wait:
                return res
            await asyncio.sleep(interval)
            elapsed += interval
            interval = min(interval * 1.5, 3.0)

    async def check_job(self, job_id: str, tool_context: ToolContext = None) -> Dict[str, Any]:
        """
        Check a long job that execute_bash handed back as still "running".

        You normally do NOT need this: execute_bash already waits for the command
        and returns its result directly. Use check_job ONLY when execute_bash
        returned status "running" with a job_id (a long build/training run). It
        waits inline for the job to finish before returning; if it is still
        running you get status "running" — call check_job again, but do not poll
        in a tight loop.

        Args:
            job_id: The id returned by execute_bash.

        Returns:
            Dict with status ("running" | "success" | "error" | "timeout" |
            "blocked"), stdout, stderr, exit_code, and workspace_id.
        """
        return await self._await_job(job_id, _CFG.check_wait)

    # ── remote (code-exec server) ──────────────────────────────────────────────

    @staticmethod
    def _normalize(res: Dict[str, Any]) -> Dict[str, Any]:
        """Map a server/local job record to the tool's response shape."""
        status = res.get("status")
        if status in ("running", "pending"):
            norm = "running"
        elif res.get("exit_code") == 0:
            norm = "success"
        elif status in ("timeout", "blocked"):
            norm = status
        else:
            norm = "error"
        return {
            "status": norm,
            "stdout": res.get("stdout", ""),
            "stderr": res.get("stderr", ""),
            "exit_code": res.get("exit_code"),
            "workspace_id": res.get("workspace_id"),
            "job_id": res.get("job_id"),
        }

    async def _submit_remote(
        self, command: str, timeout: int, workspace_id: str,
    ) -> Dict[str, Any]:
        """Submit the command to the code-exec server; return its job_id at once."""
        submit_url = _CFG.url.rstrip("/") + _CFG.submit_path
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                submit = await client.post(submit_url, json={
                    "command": command,
                    "workspace_id": workspace_id,
                    "timeout": timeout,
                })
                submit.raise_for_status()
                job_id = submit.json()["job_id"]
        except Exception as e:
            return {"status": "error", "job_id": None, "workspace_id": workspace_id,
                    "stderr": f"Code-exec server error: {e}", "exit_code": -1}
        return {
            "status": "running",
            "job_id": job_id,
            "workspace_id": workspace_id,
            "message": f"Job started. Call check_job('{job_id}') to get its result.",
        }

    async def _check_remote(self, job_id: str) -> Dict[str, Any]:
        result_url = _CFG.url.rstrip("/") + _CFG.result_path
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                poll = await client.get(result_url, params={"job_id": job_id})
                poll.raise_for_status()
                res = poll.json()
        except Exception as e:
            return {"status": "error", "job_id": job_id,
                    "stderr": f"Code-exec server error: {e}", "exit_code": -1}
        return self._normalize(res)

    async def _run_sync(
        self, command: str, workspace_id: str, max_wait: int = 30,
    ) -> Dict[str, Any]:
        """Run a SHORT internal command and wait for its result.

        Used only for the toolset's own fast operations (file read/write,
        directory listing) — never for user commands, which are fire-and-forget.
        Submits a job and polls it internally for up to `max_wait` seconds.
        """
        started = await self._submit_remote(command, max_wait, workspace_id)
        job_id = started.get("job_id")
        if not job_id:
            return started  # error/blocked — nothing to poll
        # Tight ramping interval (like check_job) so fast file ops return almost
        # immediately instead of waiting a full poll_interval.
        elapsed, interval = 0.0, 0.25
        while elapsed < max_wait:
            res = await self._check_remote(job_id)
            if res["status"] != "running":
                return res
            await asyncio.sleep(interval)
            elapsed += interval
            interval = min(interval * 1.5, 3.0)
        return {"status": "timeout", "job_id": job_id,
                "stderr": f"Internal command did not finish within {max_wait}s.",
                "exit_code": -1}

    # ── local fallback (in-process background jobs) ────────────────────────────

    async def _submit_local(
        self, command: str, timeout: int, workspace_id: str,
        tool_context: Optional[ToolContext],
    ) -> Dict[str, Any]:
        """Fallback: launch a background subprocess and return its job_id at once.

        Jobs run concurrently; their records live in the process-wide
        `_LOCAL_JOBS` registry and are read back by `_check_local`.
        """
        cwd = str(self._local_workspace(tool_context))
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        record: Dict[str, Any] = {
            "job_id": job_id, "workspace_id": workspace_id, "status": "running",
            "stdout": "", "stderr": "", "exit_code": None,
        }
        _evict_finished_jobs()
        _LOCAL_JOBS[job_id] = record

        def _run() -> None:
            # Runs in a dedicated OS thread, independent of the event loop, and
            # writes the outcome straight into the shared record. Using a real
            # thread (not asyncio.create_task) avoids two failure modes: the task
            # being garbage-collected before it finishes, and the result update
            # being stranded when the event loop is busy in a blocking LLM call —
            # either of which would leave the job stuck "running" forever.
            try:
                r = subprocess.run(command, shell=True, capture_output=True,
                                   text=True, timeout=timeout, cwd=cwd)
                outcome = {"status": "success" if r.returncode == 0 else "error",
                           "stdout": r.stdout, "stderr": r.stderr,
                           "exit_code": r.returncode}
            except subprocess.TimeoutExpired:
                outcome = {"status": "timeout", "stderr": f"Timed out after {timeout}s.",
                           "exit_code": -1}
            except Exception as e:  # pragma: no cover - defensive
                outcome = {"status": "error", "stderr": str(e), "exit_code": -1}
            # Write status LAST: a reader that sees a terminal status is then
            # guaranteed to see the other fields already set (avoids a race where
            # check_job reads status="success" but exit_code is still None and
            # mis-normalizes it to "error").
            record["stdout"] = outcome.get("stdout", "")
            record["stderr"] = outcome.get("stderr", "")
            record["exit_code"] = outcome.get("exit_code")
            record["status"] = outcome["status"]

        threading.Thread(target=_run, daemon=True).start()
        return {
            "status": "running",
            "job_id": job_id,
            "workspace_id": workspace_id,
            "message": f"Job started. Call check_job('{job_id}') to get its result.",
        }

    def _check_local(self, job_id: str) -> Dict[str, Any]:
        record = _LOCAL_JOBS.get(job_id)
        if record is None:
            return {"status": "error", "job_id": job_id, "exit_code": -1,
                    "stderr": f"Unknown job_id: {job_id}"}
        return self._normalize(record)

    # ── file I/O ──────────────────────────────────────────────────────────────

    async def read_file(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        tool_context: ToolContext = None,
    ) -> Dict[str, Any]:
        """
        Read the contents of a file from the session workspace.

        Args:
            file_path: Path to the file. Relative paths resolve inside this
                session's sandbox workspace.
            start_line: First line to return, 1-indexed (default: beginning).
            end_line: Last line to return, inclusive (default: end of file).

        Returns:
            Dict with content, total_lines, and the actual line range returned.
        """
        # In remote-exec mode, read via the sandbox so paths match where commands run.
        if _CFG.url:
            res = await self._run_sync(
                f"sed -n '{(start_line or 1)},{end_line or '$'}p' {shlex.quote(file_path)}",
                self._workspace_id(tool_context),
            )
            if res["status"] == "success":
                return {"status": "success", "content": res["stdout"]}
            return {"status": "error", "error": res.get("stderr", "read failed")}

        ws = self._local_workspace(tool_context)

        def _read() -> Dict[str, Any]:
            path = Path(file_path)
            if not path.is_absolute():
                path = ws / path
            if not path.exists():
                return {"status": "error", "error": f"File not found: {file_path}"}
            if not path.is_file():
                return {"status": "error", "error": f"Not a file: {file_path}"}
            try:
                lines = path.read_text(errors="replace").splitlines(keepends=True)
                total = len(lines)
                lo = max((start_line or 1) - 1, 0)
                hi = min(end_line or total, total)
                # Keep the success shape identical to the remote path: {status, content}.
                return {"status": "success", "content": "".join(lines[lo:hi])}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        return await asyncio.to_thread(_read)

    async def write_file(
        self,
        file_path: str,
        content: str,
        overwrite: bool = True,
        tool_context: ToolContext = None,
    ) -> Dict[str, Any]:
        """
        Write content to a file in the session workspace, creating parent
        directories as needed.

        Args:
            file_path: Path for the file. Relative paths resolve inside this
                session's sandbox workspace.
            content: Text content to write.
            overwrite: Whether to overwrite if the file already exists (default True).

        Returns:
            Dict with status and bytes_written.
        """
        # In remote-exec mode, write via the sandbox using a base64 payload.
        if _CFG.url:
            import base64
            b64 = base64.b64encode(content.encode()).decode()
            qpath = shlex.quote(file_path)
            cmd = (
                f"mkdir -p \"$(dirname {qpath})\" && "
                f"echo {b64} | base64 -d > {qpath}"
            )
            res = await self._run_sync(cmd, self._workspace_id(tool_context))
            if res["status"] == "success":
                return {"status": "success", "file_path": file_path,
                        "bytes_written": len(content.encode())}
            return {"status": "error", "error": res.get("stderr", "write failed")}

        ws = self._local_workspace(tool_context)

        def _write() -> Dict[str, Any]:
            path = Path(file_path)
            if not path.is_absolute():
                path = ws / path
            if path.exists() and not overwrite:
                return {
                    "status": "skipped",
                    "message": f"{file_path} already exists and overwrite=False.",
                }
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                return {
                    "status": "success",
                    "file_path": str(path.resolve()),
                    "bytes_written": len(content.encode()),
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}

        return await asyncio.to_thread(_write)

    async def list_directory(
        self,
        path: str = ".",
        recursive: bool = False,
        tool_context: ToolContext = None,
    ) -> Dict[str, Any]:
        """
        List the contents of a directory in the session workspace.

        Args:
            path: Directory path, relative to the session sandbox (default: ".").
            recursive: Whether to list recursively (default False).

        Returns:
            Dict with stdout listing of the directory.
        """
        # Remote: list via the sandbox so the view matches where commands run.
        if _CFG.url:
            flag = "-Ra" if recursive else "-la"
            cmd = f"ls {flag} {shlex.quote(path)}"
            res = await self._run_sync(cmd, self._workspace_id(tool_context))
            return {"status": res["status"], "listing": res.get("stdout", ""),
                    "stderr": res.get("stderr", "")}

        # Local: walk the filesystem directly (no shell — avoids path injection).
        ws = self._local_workspace(tool_context)

        def _ls() -> Dict[str, Any]:
            base = Path(path)
            if not base.is_absolute():
                base = ws / base
            if not base.exists():
                return {"status": "error", "listing": "", "stderr": f"Path not found: {path}"}
            if not base.is_dir():
                return {"status": "error", "listing": "", "stderr": f"Not a directory: {path}"}
            try:
                entries = base.rglob("*") if recursive else base.iterdir()
                lines = []
                for p in sorted(entries):
                    kind = "d" if p.is_dir() else "f"
                    size = p.stat().st_size if p.is_file() else "-"
                    lines.append(f"{kind} {size}\t{p.relative_to(base)}")
                return {"status": "success", "listing": "\n".join(lines), "stderr": ""}
            except Exception as e:
                return {"status": "error", "listing": "", "stderr": str(e)}

        return await asyncio.to_thread(_ls)

    # ── package management ────────────────────────────────────────────────────

    async def install_package(
        self,
        package_name: str,
        upgrade: bool = False,
        tool_context: ToolContext = None,
    ) -> Dict[str, Any]:
        """
        Install a Python package with pip inside the session sandbox.

        Like execute_bash this WAITS for the install and returns its result
        directly; only a very slow install outlives the inline wait and comes
        back as status "running" with a `job_id` for check_job. Installs may
        require human approval before starting.

        Args:
            package_name: Package name (e.g. "numpy" or "numpy==1.26.0"), or a
                          space-separated list. Must contain only alphanumeric
                          chars, hyphens, underscores, dots, brackets, version
                          specifiers, and spaces.
            upgrade: Pass --upgrade flag (default False).

        Returns:
            Dict with status ("success" | "error" | "timeout" | "blocked" |
            "denied"), stdout, stderr and exit_code — or status "running" with
            a `job_id` for a long install.
        """
        # Basic package name validation — reject shell injection attempts.
        # Note: ';' is intentionally NOT allowed (it would enable command
        # chaining once interpolated into the shell command below).
        if not re.match(r'^[\w\-\.\[\]<>=!,~ ]+$', package_name):
            return {
                "status": "error",
                "error": f"Invalid package name: {package_name!r}",
            }

        cmd = f"pip install {'--upgrade ' if upgrade else ''}{package_name}"
        return await self.execute_bash(cmd, timeout=600, tool_context=tool_context)


# Gate outward-facing/destructive commands behind human approval when HITL is active.
from CoScientist.hitl.handler import ConsoleHITLHandler, DelegatingHITLHandler
_hitl_handler = DelegatingHITLHandler(ConsoleHITLHandler())

coder_toolset = CoderToolset(hitl_handler=_hitl_handler)
coder_toolset_instance = coder_toolset.get_tools(None)


def seed_coder_workspace(callback_context, llm_request=None):
    """before_model callback (wired on the orchestrator): pin the coder sandbox
    to THIS ADK session before any CoderAgent delegation.

    The orchestrator's callback context carries the top-level (user) session id,
    which is stable across the whole session and across user messages. Seeding
    the workspace id from it — once, before delegating — guarantees every
    CoderAgent delegation (even several fanned out in one turn) shares one
    sandbox deterministically. The CoderToolset self-seeds as a fallback when no
    orchestrator runs (e.g. standalone A2A), and an explicit CODER_WORKSPACE_ID
    pin still wins over both.
    """
    if settings.web.coder_workspace_id or os.getenv("CODER_WORKSPACE_ID"):
        return None
    state = callback_context.state
    if state.get(_WORKSPACE_STATE_KEY):
        return None
    scope = session_key(callback_context)
    sid = scope[1] if scope != DEFAULT_SESSION_KEY else None
    if sid is None:
        inv = (
            getattr(callback_context, "_invocation_context", None)
            or getattr(callback_context, "invocation_context", None)
        )
        session = getattr(inv, "session", None) if inv is not None else None
        sid = getattr(session, "id", None)
    if sid:
        safe = re.sub(r"[^A-Za-z0-9_-]", "", str(sid))[:48] or "session"
        state[_WORKSPACE_STATE_KEY] = f"ws_{safe}"
    return None

