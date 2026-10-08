"""Venv creation and dependency-compatibility checking."""
import asyncio
import json
import os
import shlex
import subprocess
from pathlib import Path

from alembic.config import BASH_ENV_TIMEOUT, VENV_COMPAT_TIMEOUT, VENV_SETUP_TIMEOUT
from alembic.tools.paths import COMPAT_CHECK_SCRIPT, output_dir, repo_path, venv_python
from alembic.tools.shell import record_env_command

# The MCP runtime the generated server.py needs at serve time. `pytest` lives in
# the *tools* venv (it runs the tests), so it is ensured separately; fastmcp+mcp
# must be importable inside the *server* venv or the committed image cannot serve.
# `boto3` backs helpers/s3_transfer.py (lazy-imported there, so it costs nothing
# when S3 is unconfigured) — this list is duplicated in
# docker/alembic/serve.Dockerfile's `uv pip install`; keep the two in sync.
SERVER_PACKAGES = ("fastmcp", "mcp", "boto3")


def _pip_install(use_uv: bool, python: str, venv_dir: Path, *args: str) -> None:
    """Install into the venv via uv (when available) or the venv's own pip.

    Every subprocess call is bounded — a stalled resolver or a build that
    drops into an interactive prompt must not hang forever.
    """
    cmd = (["uv", "pip", "install", "--python", python, *args]
           if use_uv
           else [str(venv_dir / "bin" / "pip"), "install", *args])
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=VENV_SETUP_TIMEOUT)
    record_env_command(shlex.join(cmd))


async def setup_venv(packages: list[str] | None = None,
                     requirements_file: str | None = None,
                     python_version: str | None = None) -> dict:
    """Create the MAIN ``.venv`` (where tools + tests run) and install deps.

    Uses `uv` when available, falls back to `python -m venv` + `pip`. Installs
    `pytest` automatically. The repo package ITSELF is installed deterministically
    by the pipeline (``install_repo``: editable `pip install -e .`, or a `.pth`
    for non-package repos) — you do not need to install it here; just get the
    Python version and runtime deps right. fastmcp/mcp are NOT installed here
    (they live in the separate ``.venv-server`` built at wrapper time).

    Args:
        packages:          Extra pip-installable package names.
        requirements_file: Path to requirements.txt relative to cloned repo root.
        python_version:    Python version string, e.g. "3.11".

    Examples:
        setup_venv(requirements_file="requirements.txt", python_version="3.11")
        setup_venv(packages=["numpy", "scipy"])
    """
    # Run on a worker thread — see bash()/bash_env() in shell.py for why.
    return await asyncio.to_thread(
        _setup_venv_sync, packages, requirements_file, python_version)


def _setup_venv_sync(packages: list[str] | None,
                     requirements_file: str | None,
                     python_version: str | None) -> dict:
    # LLMs often pass a single package as a bare string, or a comma/space list —
    # coerce so `["fastmcp",...] + packages` never TypeErrors and half-builds the venv.
    if isinstance(packages, str):
        packages = [p for p in packages.replace(",", " ").split() if p]
    out_dir  = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = out_dir / ".venv"
    python   = str(venv_dir / "bin" / "python")

    use_uv = subprocess.run(["which", "uv"], capture_output=True).returncode == 0

    try:
        if use_uv:
            cmd = ["uv", "venv", str(venv_dir)]
            if python_version:
                cmd += ["--python", python_version]
        else:
            py_bin = f"python{python_version}" if python_version else "python"
            cmd = [py_bin, "-m", "venv", str(venv_dir)]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=VENV_SETUP_TIMEOUT)
        record_env_command(shlex.join(cmd))
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": f"venv creation failed: {e.stderr.strip()}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"venv creation timed out after {VENV_SETUP_TIMEOUT}s"}

    errors = []
    if requirements_file:
        req_path = repo_path() / requirements_file
        if req_path.exists():
            try:
                _pip_install(use_uv, python, venv_dir, "-r", str(req_path))
            except subprocess.CalledProcessError as e:
                errors.append(f"requirements install failed: {e.stderr.strip()}")
            except subprocess.TimeoutExpired:
                errors.append(f"requirements install timed out after {VENV_SETUP_TIMEOUT}s")
        else:
            errors.append(f"requirements file not found: {req_path}")

    # The MAIN venv only needs pytest on top of the repo's deps; fastmcp/mcp go
    # into the separate .venv-server at wrapper time.
    install_pkgs = ["pytest"] + (packages or [])
    try:
        _pip_install(use_uv, python, venv_dir, *install_pkgs)
    except subprocess.CalledProcessError as e:
        errors.append(f"package install failed: {e.stderr.strip()}")
    except subprocess.TimeoutExpired:
        errors.append(f"package install timed out after {VENV_SETUP_TIMEOUT}s")

    if errors:
        return {"success": False, "venv": str(venv_dir), "error": "; ".join(errors)}
    return {"success": True, "venv": str(venv_dir), "python": python}


def _add_repo_pth(out: Path, repo: Path) -> bool:
    """Put the repo root on the main venv's import path via a ``.pth`` file — the
    fallback for repos with no installable project (pure-Python script repos)."""
    sps = sorted((out / ".venv").glob("lib/python*/site-packages"))
    if not sps:
        return False
    try:
        (sps[0] / "_alembic_repo.pth").write_text(str(repo) + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


def install_repo() -> dict:
    """Deterministically make the cloned repo importable in the MAIN venv — one
    fixed procedure per repo, keyed on which files exist (no LLM decision):

      1. ``requirements.txt``        → ``uv pip install -r requirements.txt``
      2. ``pyproject.toml``/``setup.py`` → ``uv pip install -e .`` (package + its
         deps + builds any Cython/C extension); on editable failure, retry
         non-editable ``uv pip install .``.
      3. neither (script-only repo)  → drop a ``.pth`` pointing at the repo root
         into the venv's site-packages.

    Idempotent (safe to re-run). Successful commands are recorded into setup.sh
    (R5). Returns ``{"ok", "steps", "note"}`` — ``ok`` means the procedure ran;
    the env gate's repo-import smoke is the source of truth for whether imports
    actually resolve."""
    out = output_dir().resolve()
    repo = repo_path().resolve()
    python = venv_python(out)
    if not Path(python).exists():
        return {"ok": False, "steps": [], "note": "main venv (.venv) not built yet"}

    # setup.sh runs `cd /work` then replays these commands, so what we RECORD must
    # be workdir-relative (portable into a container), interpreter-targeted, and
    # NOT the absolute host paths / bare `uv pip install` we execute locally.
    rel_python = str(output_dir() / ".venv" / "bin" / "python")   # .alembic/<name>/output/.venv/bin/python
    rel_repo   = str(repo_path())                                 # .alembic/<name>/repos

    steps: list[str] = []
    notes: list[str] = []

    def _uv(exec_args: list[str], record_args: list[str]) -> subprocess.CompletedProcess:
        cmd = ["uv", "pip", "install", "--python", python, *exec_args]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               cwd=str(out), timeout=BASH_ENV_TIMEOUT)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, 124, "", "install timed out")
        if r.returncode == 0:
            record_env_command("uv pip install --python " + rel_python + " " + " ".join(record_args))
        return r

    req = repo / "requirements.txt"
    if req.exists():
        r = _uv(["-r", str(req)], ["-r", str(repo_path() / "requirements.txt")])
        steps.append(f"requirements.txt rc={r.returncode}")
        if r.returncode != 0:
            notes.append(f"requirements.txt install failed: {r.stderr[-300:]}")

    if (repo / "pyproject.toml").exists() or (repo / "setup.py").exists():
        r = _uv(["-e", str(repo)], ["-e", rel_repo])
        if r.returncode != 0:
            notes.append("editable install failed — retrying non-editable")
            r = _uv([str(repo)], [rel_repo])
        steps.append(f"package rc={r.returncode}")
        if r.returncode != 0:
            notes.append(f"package install failed: {r.stderr[-400:]}")
    else:
        ok = _add_repo_pth(out, repo)
        steps.append(f"pth={'ok' if ok else 'fail'}")
        if not ok:
            notes.append("no installable project and could not add a .pth")

    return {"ok": True, "steps": steps, "note": "; ".join(notes)}


def _venv_root(python: str) -> Path | None:
    """The venv directory for a ``.../<venv>/bin/python`` interpreter, or None
    when ``python`` is not a venv interpreter (system python)."""
    p = Path(python)
    return p.parent.parent if p.parent.name == "bin" else None


def _clean_probe_env() -> dict:
    """Environment for an import probe with ``PYTHONPATH`` stripped — a leaked
    PYTHONPATH can otherwise satisfy the probe from a copy that is not in the
    venv and will not survive into the committed image."""
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def _resolves_in_venv(python: str, import_name: str) -> bool:
    """True iff ``import_name`` imports under ``python`` AND its module file
    physically lives inside that interpreter's venv. The probe strips PYTHONPATH
    and runs from the venv root (not the output dir) so neither a system/PYTHONPATH
    leak nor a stray ``<name>.py`` in the working dir can satisfy it — only a real
    install in the venv counts. For a non-venv python (no venv root) mere
    importability is accepted. Version-agnostic (no 3.11-only flags)."""
    root = _venv_root(python)
    code = (f"import importlib.util as u; s=u.find_spec({import_name!r}); "
            "print(s.origin if s and s.origin else '')")
    r = subprocess.run([python, "-c", code], capture_output=True, text=True,
                       env=_clean_probe_env(), cwd=str(root) if root else None)
    origin = r.stdout.strip()
    if r.returncode != 0 or not origin:
        return False
    if root is None:
        return True
    try:
        return Path(origin).resolve().is_relative_to(root.resolve())
    except (ValueError, OSError):
        return False


def ensure_pkg(python: str, import_name: str, pip_name: str | None = None) -> str | None:
    """Deterministically make sure ``import_name`` really lives inside ``python``'s
    venv, installing ``pip_name`` (default = import_name) if not. Called by the env
    gate, not the LLM — a debugger that rebuilds a venv by hand (bare ``uv venv``,
    no pip) often drops pytest (tools venv) or fastmcp (server venv) that setup_venv
    would add. Returns an error string or None.

    The check verifies the package resolves *inside the venv* (not via a
    PYTHONPATH/system leak) and re-verifies after installing — an ``uv pip
    install`` that reports success but does not land must not pass silently, which
    is exactly what let a fastmcp-less server reach the wrapper and get shimmed."""
    if _resolves_in_venv(python, import_name):
        return None
    cmd = ["uv", "pip", "install", "--python", python, pip_name or import_name]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=VENV_SETUP_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"{import_name} install timed out"
    if r.returncode != 0:
        return f"{import_name} install failed: {r.stderr.strip()[:300]}"
    record_env_command(shlex.join(cmd))
    if not _resolves_in_venv(python, import_name):
        return (f"{import_name} still not importable inside {python}'s venv after "
                f"install — the venv may lack pip or the install did not land")
    return None


def ensure_pytest(python: str) -> str | None:
    return ensure_pkg(python, "pytest")


def ensure_server_packages(server_python: str) -> str | None:
    """Guarantee the MCP server runtime (fastmcp + mcp + boto3, the S3
    pass-through's lazy dependency) is importable inside the *server* venv.
    Returns the first blocking error, or None. The wrapper's deterministic
    codegen and G4 both assume this holds."""
    for pkg in SERVER_PACKAGES:
        err = ensure_pkg(server_python, pkg)
        if err:
            return err
    return None


async def check_venv_compat(venv_name: str = ".venv") -> dict:
    """Check compatibility by replaying the repo's own import statements in a venv.

    Scans the cloned repo's Python files with AST, collects every unique
    `import X` and `from X import Y` where X is an installed package, then
    executes each statement in the venv. This catches both ABI conflicts
    (numpy 1 vs 2) and removed-API errors (e.g. `from transformers import AdamW`
    removed in transformers>=4.38).

    Args:
        venv_name: Directory name of the venv to check, relative to the repo
                   output dir. Default ".venv" (the main venv — repo + deps,
                   where tools run).

    Returns only failures; successful imports are omitted to keep output small.

    Example:
        check_venv_compat()
    """
    # Run on a worker thread — see bash()/bash_env() in shell.py for why.
    return await asyncio.to_thread(_check_venv_compat_sync, venv_name)


def _check_venv_compat_sync(venv_name: str) -> dict:
    out_dir  = output_dir().resolve()
    repo_dir = repo_path().resolve()
    venv_py  = out_dir / venv_name / "bin" / "python"
    if not venv_py.exists():
        return {"error": f"venv python not found at {venv_py}"}

    try:
        r = subprocess.run(
            [str(venv_py.absolute()), str(COMPAT_CHECK_SCRIPT), str(repo_dir)],
            capture_output=True, text=True, timeout=VENV_COMPAT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"compat check timed out after {VENV_COMPAT_TIMEOUT}s"}
    if r.returncode != 0:
        return {"error": f"compat check script failed: {r.stderr.strip()[:500]}"}

    try:
        data = json.loads(r.stdout.strip())
    except Exception:
        return {"error": f"could not parse compat output: {r.stdout[:300]}"}

    conflicts = data.get("conflicts", {})
    return {
        "conflicts": conflicts,
        "checked": data.get("checked", 0),
        "has_conflicts": bool(conflicts),
    }
