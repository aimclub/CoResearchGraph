"""Workdir layout: current-repo state, per-repo path helpers, script paths.

R8: one pipeline works on exactly one repo, so tools no longer take repo_url —
``set_current_repo`` is called once (by the pipeline and by ``clone_repo``) and
every path helper resolves against it. Helpers still accept an explicit
``repo_url`` for host-side callers (benchmark, unit checks).
"""
import os
from pathlib import Path

from alembic.common import get_repo_name
from alembic.config import MAX_BYTES  # re-exported for tools that import it from here

WORKDIR = Path(os.environ.get("ALEMBIC_WORKDIR", ".alembic"))

# Standalone scripts run inside a repo's venv (see venv.py / invoke.py).
_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
COMPAT_CHECK_SCRIPT = _SCRIPTS_DIR / "compat_check.py"
RUN_FUNCTION_SCRIPT = _SCRIPTS_DIR / "run_function.py"
S3_TRANSFER_SCRIPT = _SCRIPTS_DIR / "s3_transfer.py"

# Where TM-Bench-style input data lands inside the container (R4).
MOUNT_DATA  = Path("/mount/data")    # host benchmark/data, bind-mounted ro
MOUNT_INPUT = Path("/mount/input")   # per-task staged inputs

IGNORE = {
    ".git", "__pycache__", ".eggs", "*.egg-info", "dist", "build",
    "node_modules", ".tox", ".mypy_cache", ".pytest_cache",
    "checkpoints", "wandb", "mlruns", ".ipynb_checkpoints",
}
IGNORE_EXTS = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".h5", ".hdf5",
    ".pt", ".pth", ".ckpt", ".pkl", ".npy", ".npz", ".parquet",
}

_current_repo: str | None = None


def set_current_repo(repo_url: str) -> None:
    global _current_repo
    _current_repo = repo_url


def current_repo() -> str:
    if _current_repo is None:
        raise RuntimeError("current repo not set — call set_current_repo() first")
    return _current_repo


def _url(repo_url: str | None) -> str:
    return repo_url if repo_url else current_repo()


def _repo_base(repo_url: str | None = None) -> Path:
    """Root dir for everything related to this repo: <WORKDIR>/<repo-name>/"""
    return WORKDIR / get_repo_name(_url(repo_url))


def repo_path(repo_url: str | None = None) -> Path:
    """Where the repo is cloned: <WORKDIR>/<repo-name>/repos/"""
    return _repo_base(repo_url) / "repos"


def output_dir(repo_url: str | None = None) -> Path:
    """Where tools/, tests/, server.py, .venv live: <WORKDIR>/<repo-name>/output/"""
    return _repo_base(repo_url) / "output"


def reports_dir(repo_url: str | None = None) -> Path:
    """Where reports + structured run data live: <WORKDIR>/<repo-name>/reports/"""
    return _repo_base(repo_url) / "reports"


def venv_python(out_dir: Path) -> str:
    """The MAIN venv python (``.venv``) — the repo + its deps + pytest, where
    tool functions and tests run. 'python' if it doesn't exist yet.

    Uses the venv symlink path directly — do NOT resolve(), as that follows the
    symlink to the bare uv Python binary, which lacks the venv's site-packages
    (and, on a PEP-668 host, is externally-managed so installs into it fail).
    """
    candidate = out_dir / ".venv" / "bin" / "python"
    return str(candidate.absolute()) if candidate.exists() else "python"


def tools_python(out_dir: Path) -> str:
    """Where tool functions and tests run — always the main venv (``.venv``)."""
    return venv_python(out_dir)


def server_python(out_dir: Path) -> str:
    """The SERVER venv python (``.venv-server``) — fastmcp only, runs server.py;
    built at the wrapper stage. Symlink path directly (see ``venv_python``)."""
    candidate = out_dir / ".venv-server" / "bin" / "python"
    return str(candidate.absolute()) if candidate.exists() else "python"


def rel_or_ignored(path: Path, root: Path) -> str | None:
    """Relative path string for an indexable file, or None if it's ignored."""
    if not path.is_file() or path.suffix in IGNORE_EXTS:
        return None
    rel = path.relative_to(root)
    if any(part in IGNORE for part in rel.parts):
        return None
    return str(rel)
