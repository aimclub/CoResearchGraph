"""Shell-execution tools exposed to the agents: bash / bash_env.

``bash_env`` additionally records every SUCCESSFUL command while recording is
armed (the Environment stage) — the transcript becomes ``setup.sh`` (R5),
ToolMaker's ``installed_state.bash()`` approach: the artefact is what actually
ran, not what an agent claims ran.
"""
import asyncio
import os
import signal
import subprocess
from pathlib import Path

from alembic.config import BASH_ENV_TIMEOUT, BASH_TIMEOUT, MAX_BYTES

_recorded: list[str] = []
_recording = False


def start_env_recording() -> None:
    global _recording
    _recorded.clear()
    _recording = True


def stop_env_recording() -> list[str]:
    global _recording
    _recording = False
    return list(_recorded)


def record_env_command(cmd: str) -> None:
    if _recording:
        _recorded.append(cmd)


def _glob_command(stripped: str) -> dict | None:
    """Handle the custom ``glob <pattern>`` shortcut. Returns None if not glob."""
    first = stripped.split()[0] if stripped else ""
    if first != "glob":
        return None
    parts = stripped.split(None, 1)
    if len(parts) < 2:
        return {"error": "glob requires a pattern argument."}
    pattern = parts[1]
    if pattern.startswith("/"):
        root, pat = Path("/"), pattern.lstrip("/")
    else:
        root, pat = Path("."), pattern
    matched = sorted(str(p) for p in root.glob(pat))
    return {"matches": matched}


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the command's whole process group, so any grandchild the shell
    spawned (a ``sudo``/``apt-get``/download it kicked off) dies with it instead
    of lingering as an orphan waiting on a prompt."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def _run_shell(command: str, timeout: int, record: bool = False) -> dict:
    """Shared body for ``bash`` / ``bash_env``: glob shortcut + shell run.

    Commands run in a NEW SESSION with stdin closed: detached from any
    controlling terminal, a command that would otherwise block on a prompt
    (``sudo`` asking for a password, an interactive installer) fails fast with
    "no tty present" instead of hanging until the timeout. On timeout the whole
    process group is killed so nothing is left orphaned.
    """
    stripped = command.strip()
    if not stripped:
        return {"error": "empty command"}

    glob_result = _glob_command(stripped)
    if glob_result is not None:
        return glob_result

    proc = subprocess.Popen(
        stripped, shell=True, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return {"error": f"Command timed out after {timeout} seconds."}

    output = stdout or ""
    if proc.returncode != 0 and stderr:
        output += "\n[stderr] " + stderr
    if record and proc.returncode == 0:
        record_env_command(stripped)
    return {"output": output[:MAX_BYTES], "exit_code": proc.returncode}


async def bash(command: str) -> dict:
    """Run a shell command with a 15 s timeout.

    The pipeline runs inside an ephemeral container, so any command line is
    accepted — the container is the security boundary. The custom
    ``glob <pattern>`` shortcut is recognised as a convenience.

    Examples:
        bash("ls .alembic/massformer/repos")
        bash("grep -rn 'def train' .alembic/massformer/repos -l")
        bash("head -n 30 .alembic/massformer/repos/README.md")
        bash("glob .alembic/massformer/repos/**/*.yaml")
    """
    # Offload the blocking subprocess.run to a worker thread — ADK calls
    # non-async tools synchronously, which would freeze the event loop and
    # silently defeat any asyncio-based timeout wrapping this turn.
    return await asyncio.to_thread(_run_shell, command, BASH_TIMEOUT)


async def bash_env(command: str) -> dict:
    """Run a shell command with a 900 s timeout — for slow installs and downloads.

    Same semantics as ``bash``, just a longer timeout so package managers
    (pip / uv / apt-get) have time to build, and a pretrained-weight download
    (potentially multi-GB) has room to finish. Inherits this process's full
    environment, so HF_TOKEN (if set) is automatically visible to any
    huggingface_hub/huggingface-cli call — never pass it on the command line.

    Examples:
        bash_env("uv venv .alembic/massformer/output/.venv --python 3.9")
        bash_env("uv pip install --python .alembic/massformer/output/.venv/bin/python torch")
        bash_env("apt-get update && apt-get install -y --no-install-recommends libpoppler-cpp-dev")
        bash_env("huggingface-cli download MahmoodLab/UNI2-h --local-dir .alembic/UNI/repos/checkpoints")
    """
    return await asyncio.to_thread(_run_shell, command, BASH_ENV_TIMEOUT, True)
