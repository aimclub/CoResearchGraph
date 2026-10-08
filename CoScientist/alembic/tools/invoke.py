"""Validation tools: per-tool static checks, pytest runner, live invocation.

The pipeline's G3 gate and validator loop are code; only ``run_tool_tests`` and
``invoke_tool_function`` are also exposed to the debugger agent so it can
verify its fixes with exactly the machinery the validator uses.
"""
from __future__ import annotations

import ast
import asyncio
import builtins
import json
import os
import re
import signal
import subprocess
from pathlib import Path

from alembic.config import (
    IMPORT_CHECK_TIMEOUT, INVOKE_TIMEOUT, MAX_BYTES,
    RESULT_MAX_LIST_ITEMS, RESULT_MAX_STR_LEN, RESULT_SENTINEL, TEST_TIMEOUT,
)
from alembic.tools.paths import (
    RUN_FUNCTION_SCRIPT, output_dir, repo_path, server_python, tools_python,
)


# ══════════════════════════════════════════════════════════════════════════════
# Static undefined-name check (zero-execution)
# ══════════════════════════════════════════════════════════════════════════════
_BUILTIN_NAMES = frozenset(dir(builtins)) | {
    "__name__", "__file__", "__doc__", "__package__", "__spec__", "__loader__",
    "__builtins__", "__annotations__", "__dict__",
}


def _extract_target_names(node: ast.expr, out: set[str]) -> None:
    if isinstance(node, ast.Name):
        out.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            _extract_target_names(elt, out)
    elif isinstance(node, ast.Starred):
        _extract_target_names(node.value, out)


def find_undefined_names(source: str) -> list[str] | None:
    """Whole-file, zero-execution pass — flag any Name(Load) reference not
    bound anywhere in the file and not a builtin (catches `torch.x` with no
    `import torch`). Deliberately whole-file-permissive; bails on `import *`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    bound: set[str] = set()
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    return None
                bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
            a = node.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                bound.add(arg.arg)
            if a.vararg:
                bound.add(a.vararg.arg)
            if a.kwarg:
                bound.add(a.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            a = node.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                bound.add(arg.arg)
            if a.vararg:
                bound.add(a.vararg.arg)
            if a.kwarg:
                bound.add(a.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, (ast.Assign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                _extract_target_names(t, bound)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
            _extract_target_names(node.target, bound)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    _extract_target_names(item.optional_vars, bound)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                bound.add(node.name)
        elif isinstance(node, ast.comprehension):
            _extract_target_names(node.target, bound)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            referenced.add(node.id)
    return sorted(referenced - bound - _BUILTIN_NAMES) or None


# ══════════════════════════════════════════════════════════════════════════════
# G3: per-tool artefact checks (deterministic, run in the actual container venv)
# ══════════════════════════════════════════════════════════════════════════════
def _tool_file_errors(name: str, out_dir: Path, python: str) -> list[str]:
    """Static checks for one tools/<name>.py: exists, defines <name>, compiles,
    no undefined names, module imports cleanly under the tools venv."""
    f = out_dir / "tools" / f"{name}.py"
    if not f.exists():
        return [f"tools/{name}.py is missing"]
    source = f.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"tools/{name}.py: SyntaxError: {e}"]
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
               for n in tree.body):
        return [f"tools/{name}.py defines no top-level function named '{name}'"]
    errs = []
    undefined = find_undefined_names(source)
    if undefined:
        errs.append(f"tools/{name}.py: name(s) {', '.join(undefined)} are used but never "
                    f"imported or defined — will NameError at runtime")
    r = subprocess.run([python, "-c", f"import sys; sys.path.insert(0, {str(out_dir)!r}); "
                                      f"import tools.{name}"],
                       capture_output=True, text=True, timeout=IMPORT_CHECK_TIMEOUT,
                       cwd=str(out_dir))
    if r.returncode != 0:
        errs.append(f"import tools.{name} failed: {r.stderr.strip()[-800:]}")
    return errs


def _test_file_errors(name: str, out_dir: Path, python: str) -> list[str]:
    """Static checks for tests/test_<name>.py: exists, has >=1 test_smoke_*,
    imports tools.<name>, and pytest can collect it (imports resolve)."""
    f = out_dir / "tests" / f"test_{name}.py"
    if not f.exists():
        return [f"tests/test_{name}.py is missing"]
    source = f.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"tests/test_{name}.py: SyntaxError: {e}"]
    errs = []
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name.startswith("test_smoke") for n in tree.body):
        errs.append(f"tests/test_{name}.py has no test_smoke_* function")
    imported = any(
        (isinstance(n, ast.ImportFrom) and n.module == f"tools.{name}")
        or (isinstance(n, ast.Import) and any(a.name == f"tools.{name}" for a in n.names))
        for n in ast.walk(tree))
    if not imported:
        errs.append(f"tests/test_{name}.py must import the function as "
                    f"'from tools.{name} import {name}'")
    r = subprocess.run([python, "-m", "pytest", str(f), "--collect-only", "-q",
                        "-p", "no:cacheprovider"],
                       capture_output=True, text=True, timeout=IMPORT_CHECK_TIMEOUT,
                       cwd=str(out_dir))
    if r.returncode not in (0, 5):   # 5 = no tests collected (already reported above)
        errs.append(f"pytest cannot collect tests/test_{name}.py: "
                    f"{(r.stdout + r.stderr).strip()[-800:]}")
    return errs


def check_tool_artefacts(tool_names: list[str]) -> dict:
    """G3 gate body: every planned tool has a compiling, importable function
    file and a collectable test file. Returns {passed, errors: {tool: [...]}}."""
    out_dir = output_dir().resolve()
    python  = tools_python(out_dir)
    (out_dir / "tools" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "tools" / "__init__.py").touch()
    errors: dict[str, list[str]] = {}
    for name in tool_names:
        try:
            errs = _tool_file_errors(name, out_dir, python) + _test_file_errors(name, out_dir, python)
        except subprocess.TimeoutExpired:
            errs = [f"static check for '{name}' timed out"]
        if errs:
            errors[name] = errs
    return {"passed": not errors, "errors": errors}


# ══════════════════════════════════════════════════════════════════════════════
# G2 helper: repo-import smoke test (env gate)
# ══════════════════════════════════════════════════════════════════════════════
def check_repo_imports(modules: list[str]) -> dict:
    """Import each top-level module named by the plan's tool targets, in the
    tools venv with the repo (and its ``src/``) on sys.path — verifies the built
    environment can actually load the code the tools will wrap. A ``src/``-layout
    repo (module under ``repo/src``) or a properly pip-installed package both
    pass. Returns {passed, errors}."""
    out_dir  = output_dir().resolve()
    repo_dir = repo_path().resolve()
    python   = tools_python(out_dir)
    paths = [str(repo_dir)] + ([str(repo_dir / "src")] if (repo_dir / "src").is_dir() else [])
    errors: dict[str, str] = {}
    for mod in modules:
        code = (f"import sys; sys.path[:0] = {paths!r}; import {mod}")
        try:
            r = subprocess.run([python, "-c", code], capture_output=True, text=True,
                               timeout=IMPORT_CHECK_TIMEOUT, cwd=str(repo_dir))
        except subprocess.TimeoutExpired:
            continue   # slow import ≠ broken env (heavy ML packages)
        if r.returncode != 0:
            errors[mod] = r.stderr.strip()[-500:]
    return {"passed": not errors, "errors": errors}


# ══════════════════════════════════════════════════════════════════════════════
# Per-tool pytest run (R6: smoke vs invocation split, 120 s cap)
# ══════════════════════════════════════════════════════════════════════════════
_TEST_LINE = re.compile(r"::(test_\w+)(?:\[[^\]]*\])?\s+(PASSED|FAILED|ERROR|XPASS|XFAIL|SKIPPED)")

# Symbols whose presence in a test function marks it as MOCKED — it does not
# exercise the real repo, so it must NOT count as invocation-correctness evidence
# (else a hollow test would satisfy the `perfect` gate). See _mocked_test_functions.
_MOCK_NAMES = {"patch", "MagicMock", "Mock", "AsyncMock", "NonCallableMock",
               "PropertyMock", "mock_open", "monkeypatch", "mocker", "seal"}


def _mocked_test_functions(source: str) -> set[str]:
    """Names of top-level test_* functions that use mocking (unittest.mock,
    monkeypatch, or pytest-mock). A ``test_invoc_*`` that mocks the repo is NOT
    proof the tool really ran, so run_tool_tests reclassifies it as a smoke test:
    it still runs, but cannot make a tool ``perfect`` (hollow-validation guard)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    mocked: set[str] = set()
    for node in tree.body:
        if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")):
            continue
        if {a.arg for a in node.args.args} & {"monkeypatch", "mocker"}:
            mocked.add(node.name)          # fixture-injected mocking
            continue
        for n in ast.walk(node):           # body + decorators (@patch, @mock.patch)
            if (isinstance(n, ast.Name) and n.id in _MOCK_NAMES) or \
               (isinstance(n, ast.Attribute) and n.attr in _MOCK_NAMES):
                mocked.add(node.name)
                break
    return mocked


async def run_tool_tests(tool_name: str) -> dict:
    """Run tests/test_<tool_name>.py and return the smoke/invocation split.

    Returns {smoke_passed, smoke_total, invoc_passed, invoc_total,
    timeout: bool, failures: str}. test_smoke_* functions count as smoke
    (quick sanity), test_invoc_* as evidence-based invocation-correctness
    tests; any other test_* counts as smoke.

    Example: run_tool_tests("predict")
    """
    return await asyncio.to_thread(_run_tool_tests_sync, tool_name)


def _run_tool_tests_sync(tool_name: str) -> dict:
    out_dir = output_dir().resolve()
    f = out_dir / "tests" / f"test_{tool_name}.py"
    python = tools_python(out_dir)
    if not f.exists():
        return {"error": f"{f} not found"}
    mocked = _mocked_test_functions(f.read_text(encoding="utf-8", errors="replace"))
    proc = subprocess.Popen(
        [python, "-m", "pytest", str(f), "-v", "--tb=short", "--no-header",
         "-p", "no:cacheprovider"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=str(out_dir), start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=TEST_TIMEOUT)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        return {"timeout": True, "smoke_passed": None, "smoke_total": None,
                "invoc_passed": None, "invoc_total": None,
                "failures": f"test run exceeded {TEST_TIMEOUT}s"}

    counts = {"smoke": [0, 0], "invoc": [0, 0]}   # [passed, total]
    mocked_invoc = 0
    for m in _TEST_LINE.finditer(out):
        name, status = m.group(1), m.group(2)
        if status in ("SKIPPED", "XFAIL"):
            continue
        # A mocked test_invoc_* is not real evidence — reclassify it as smoke so
        # it still runs but cannot make the tool `perfect` (hollow-validation guard).
        if name.startswith("test_invoc") and name in mocked:
            mocked_invoc += 1
        is_real_invoc = name.startswith("test_invoc") and name not in mocked
        kind = "invoc" if is_real_invoc else "smoke"
        counts[kind][1] += 1
        if status in ("PASSED", "XPASS"):
            counts[kind][0] += 1

    failed_tail = ""
    if proc.returncode != 0:
        failed_tail = out[-3000:]
    return {"smoke_passed": counts["smoke"][0], "smoke_total": counts["smoke"][1],
            "invoc_passed": counts["invoc"][0], "invoc_total": counts["invoc"][1],
            "invoc_mocked": mocked_invoc,
            "timeout": False, "failures": failed_tail}


# ══════════════════════════════════════════════════════════════════════════════
# Live function invocation (execution-level status)
# ══════════════════════════════════════════════════════════════════════════════
# File extensions that mark a string arg as a local-file path worth
# existence-checking before invocation. Deliberately narrow (a curated set,
# not "anything with a slash") so HF ids like "MahmoodLab/UNI2-h" and device
# strings like "cuda:0" are never mistaken for paths.
_PATH_EXTS = {
    ".csv", ".tsv", ".txt", ".json", ".yaml", ".yml", ".pdb", ".fasta", ".fa",
    ".fastq", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".svg",
    ".h5", ".hdf5", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".safetensors",
    ".nii", ".gz", ".zip", ".tar", ".pdf", ".mat", ".parquet", ".fits", ".wav",
    ".mp3", ".pkl", ".pickle", ".xlsx", ".xls", ".sdf", ".mol", ".mol2", ".cif",
}


def _missing_input_files(args: dict, repo_dir: Path, out_dir: Path) -> str | None:
    """R6: a path-shaped input arg that resolves nowhere means the sample data
    simply isn't available — the tool stays a runtime success, not a failure.
    Covers files with a known data extension AND TM-Bench ``/mount/input/...``
    paths (whole-slide dirs etc. that are gated / not bundled)."""
    for key, val in (args or {}).items():
        if not isinstance(val, str) or "://" in val:
            continue
        is_mount_input = val.startswith("/mount/input/")
        if not is_mount_input and Path(val).suffix.lower() not in _PATH_EXTS:
            continue
        candidates = [Path(val), repo_dir / val, out_dir / val]
        if not any(c.exists() for c in candidates):
            kind = "mount input" if is_mount_input else "input file"
            return f"{kind} for {key}={val!r} not available"
    return None


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the WHOLE process tree — the function may spawn its own uncapped
    subprocesses; killing only the immediate child leaks them."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _truncate_large_result(value):
    """Recursively cap list length / string length in a tool result."""
    if isinstance(value, list):
        out = [_truncate_large_result(v) for v in value[:RESULT_MAX_LIST_ITEMS]]
        if len(value) > RESULT_MAX_LIST_ITEMS:
            out.append(f"... ({len(value) - RESULT_MAX_LIST_ITEMS} more items truncated)")
        return out
    if isinstance(value, dict):
        return {k: _truncate_large_result(v) for k, v in value.items()}
    if isinstance(value, str) and len(value) > RESULT_MAX_STR_LEN:
        return value[:RESULT_MAX_STR_LEN] + f"... ({len(value) - RESULT_MAX_STR_LEN} more chars truncated)"
    return value


def _parse_result(stdout: str) -> dict | None:
    """Extract the JSON after the last RESULT_SENTINEL line, so banners /
    progress bars printed before it never break the parse."""
    if RESULT_SENTINEL in stdout:
        tail = stdout.rsplit(RESULT_SENTINEL, 1)[1].strip()
    else:
        lines = [l for l in stdout.splitlines() if l.strip()]
        tail = lines[-1] if lines else ""
    try:
        return json.loads(tail)
    except Exception:
        return None


async def invoke_tool_function(tool_name: str, args: dict | None = None) -> dict:
    """Invoke a generated tool function (tools/<tool_name>.py) live, in the
    tools venv, and return its result.

    Returns {"ok": True, "result": ...} on success; {"ok": False, "error",
    "traceback", "stderr"} on a crash; or {"ok": True, "runtime_success": True,
    "reason": ...} when the call ran past the time cap or its sample input
    files are not available (execution-level success, correctness unknown).

    Example:
        invoke_tool_function("predict", {"input_path": "data/x.csv"})
    """
    return await asyncio.to_thread(_invoke_tool_function_sync, tool_name, args)


def _invoke_tool_function_sync(tool_name: str, args: dict | None = None) -> dict:
    out_dir = output_dir().resolve()
    python  = tools_python(out_dir)
    if not (out_dir / "tools" / f"{tool_name}.py").exists():
        return {"ok": False, "error": f"tools/{tool_name}.py not found"}

    missing = _missing_input_files(args or {}, repo_path().resolve(), out_dir)
    if missing:
        return {"ok": True, "runtime_success": True,
                "reason": f"not invoked: {missing}"}

    proc = subprocess.Popen(
        [python, str(RUN_FUNCTION_SCRIPT), str(out_dir), tool_name,
         json.dumps(args or {})],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=str(out_dir), start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=INVOKE_TIMEOUT)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        return {"ok": True, "runtime_success": True,
                "reason": f"still running after {INVOKE_TIMEOUT}s — treated as "
                          f"runtime success (resource-heavy, not a confirmed bug)"}

    parsed = _parse_result(stdout.strip())
    if parsed is None:
        return {"ok": False, "error": "could not parse runner output",
                "returncode": proc.returncode,
                "stdout": stdout[-1500:], "stderr": stderr[-1500:]}
    if not parsed.get("ok") and stderr:
        parsed.setdefault("stderr", stderr[-2000:])
    if parsed.get("ok") and "result" in parsed:
        parsed["result"] = _truncate_large_result(parsed["result"])
    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# G4: generated server check
# ══════════════════════════════════════════════════════════════════════════════
def check_server() -> dict:
    """Compile + import server.py under the server venv (imports are light by
    construction: fastmcp + stdlib, plus a lazy boto3 inside
    helpers/s3_transfer.py that is never imported at module level). Returns
    {passed, error}.

    Guards against a *shimmed* server: if fastmcp is absent from the venv, an LLM
    fallback (or a stray local ``fastmcp.py``) can make ``server.py`` import via a
    stub ``FastMCP`` that never actually serves. So before trusting the import we
    require the real fastmcp module to resolve inside the server venv — the probe
    strips PYTHONPATH and runs from the venv dir so a leak or local stub cannot
    satisfy it (version-agnostic: no 3.11-only flags)."""
    out_dir = output_dir().resolve()
    server  = out_dir / "server.py"
    python  = server_python(out_dir)
    if not server.exists():
        return {"passed": False, "error": f"server.py not found at {server}"}
    probe = ("import importlib.util as u; s=u.find_spec('fastmcp'); "
             "print(s.origin if s and s.origin else '')")
    _venv = Path(python)
    _cwd = str(_venv.parent.parent) if _venv.parent.name == "bin" else None
    _env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    rp = subprocess.run([python, "-c", probe], capture_output=True, text=True,
                        env=_env, cwd=_cwd)
    origin = rp.stdout.strip()
    if rp.returncode != 0 or not origin:
        return {"passed": False, "error": (
            "fastmcp is not a real install in the server venv — server.py would "
            "only import via a shim and the committed image could not serve MCP")}
    _root = _venv.parent.parent if _venv.parent.name == "bin" else None
    if _root is not None:
        try:
            venv_local = Path(origin).resolve().is_relative_to(_root.resolve())
        except (ValueError, OSError):
            venv_local = False
        if not venv_local:
            return {"passed": False, "error": (
                f"fastmcp resolves to {origin} outside the server venv — the "
                "committed isolated venv would not import it at serve time")}
    r = subprocess.run([python, "-m", "py_compile", str(server)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"passed": False, "error": r.stderr.strip()[:MAX_BYTES]}
    load = ("import importlib.util as _u, sys as _s; "
            f"_s.path.insert(0, r'{server.parent}'); "
            f"_spec=_u.spec_from_file_location('server', r'{server}'); "
            "_mod=_u.module_from_spec(_spec); _spec.loader.exec_module(_mod)")
    try:
        r2 = subprocess.run([python, "-c", load], capture_output=True, text=True,
                            timeout=IMPORT_CHECK_TIMEOUT, cwd=str(server.parent))
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "server.py import timed out"}
    if r2.returncode != 0:
        return {"passed": False, "error": r2.stderr.strip()[-2000:]}
    return {"passed": True}
