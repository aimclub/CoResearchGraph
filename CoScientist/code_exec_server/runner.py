"""Job runner for the code-exec server.

Each command runs as a detached subprocess inside a per-workspace directory, so
files (cloned repos, datasets, checkpoints) persist across calls within the same
ADK session (`workspace_id`). Jobs run in the background; the client submits and
then polls for completion — this is what lets a training run or a long
optimization survive past any single short HTTP timeout.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional


# ─── Safety blocklist (mirrors CoderToolset's client-side guard, defense in depth) ──
_BLOCKED = [re.compile(p) for p in [
    r"rm\s+.*--no-preserve-root",
    r"rm\s+(-[a-z]*f[a-z]*\s+|--force\s+).*[/~]\s*$",
    r"rm\s+-[a-z]*r[a-z]*\s+/",
    r"mkfs",
    r"dd\s+.*if\s*=\s*/dev/",
    r"dd\s+.*of\s*=\s*/dev/[hs]d",
    r">\s*/dev/[hs]d[a-z]",
    r"chmod\s+.*-R.*\s+/\s*$",
    r"chown\s+.*-R.*\s+/\s*$",
    r":\(\)\s*\{[^}]*\|[^}]*&[^}]*\}",
    r"\bshutdown\b", r"\breboot\b", r"\bhalt\b", r"\bpoweroff\b",
    r"\binit\s+[06]\b",
    r"mv\s+/\s",
]]

# Workspace ids come from the agent; keep them filesystem-safe.
_WS_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _is_dangerous(command: str) -> Optional[str]:
    for pat in _BLOCKED:
        if pat.search(command):
            return pat.pattern
    return None


class JobStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"     # exit_code == 0
    ERROR = "error"         # exit_code != 0
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


@dataclass
class Job:
    job_id: str
    workspace_id: str
    command: str
    timeout: int
    status: JobStatus = JobStatus.RUNNING
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "workspace_id": self.workspace_id,
            "status": self.status.value,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class JobRunner:
    """In-memory job registry backed by on-disk per-workspace directories."""

    def __init__(self, workspace_root: str = "./workspace", max_output_bytes: int = 1_000_000,
                 max_jobs: int = 1000):
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.max_output_bytes = max_output_bytes
        self.max_jobs = max_jobs
        self._jobs: Dict[str, Job] = {}

    def _evict_finished(self) -> None:
        """Drop oldest finished jobs so the registry doesn't grow unbounded."""
        overflow = len(self._jobs) - self.max_jobs
        if overflow <= 0:
            return
        for job_id, job in list(self._jobs.items()):
            if overflow <= 0:
                break
            if job.status != JobStatus.RUNNING:
                del self._jobs[job_id]
                overflow -= 1

    def _workspace_dir(self, workspace_id: str) -> Path:
        if not _WS_RE.match(workspace_id):
            workspace_id = "default"
        path = self.workspace_root / workspace_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def submit(self, command: str, workspace_id: str = "default", timeout: int = 1800) -> Job:
        job = Job(
            job_id=f"job_{uuid.uuid4().hex[:16]}",
            workspace_id=workspace_id or "default",
            command=command,
            timeout=int(timeout),
        )
        self._evict_finished()
        self._jobs[job.job_id] = job

        blocked_by = _is_dangerous(command)
        if blocked_by:
            job.status = JobStatus.BLOCKED
            job.stderr = f"Command blocked by safety policy (matched: {blocked_by!r})."
            job.exit_code = None
            job.finished_at = time.time()
            return job

        asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: Job) -> None:
        cwd = str(self._workspace_dir(job.workspace_id))
        # Pin HOME/TMP into the workspace so tooling that writes to $HOME stays
        # inside the sandbox.
        env = dict(os.environ)
        env["HOME"] = cwd
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            proc = await asyncio.create_subprocess_shell(
                job.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=job.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                job.status = JobStatus.TIMEOUT
                job.stderr = f"Command timed out after {job.timeout}s."
                job.exit_code = -1
                job.finished_at = time.time()
                return

            job.stdout = self._cap(out.decode(errors="replace"))
            job.stderr = self._cap(err.decode(errors="replace"))
            job.exit_code = proc.returncode
            job.status = JobStatus.SUCCESS if proc.returncode == 0 else JobStatus.ERROR
        except Exception as e:  # pragma: no cover - defensive
            job.status = JobStatus.ERROR
            job.stderr = f"Runner error: {e}"
            job.exit_code = -1
        finally:
            job.finished_at = time.time()

    def _cap(self, text: str) -> str:
        b = text.encode()
        if len(b) <= self.max_output_bytes:
            return text
        head = b[: self.max_output_bytes].decode(errors="replace")
        return head + f"\n...[truncated {len(b) - self.max_output_bytes} bytes]"

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)
