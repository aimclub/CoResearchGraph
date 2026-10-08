"""Deterministic code generation: server.py, setup.sh, TM-Bench code.py.

The MCP wrapper is a pure function of the verified tool files (Q&A decision):
signatures and docstrings are extracted by AST and rendered into a FastMCP
server whose every tool shells through ``helpers/run_function.py`` — the same
runner the validator used, so serving and validation share one execution path
and the two-venv layout needs no special casing. An LLM touches server.py only
if the compile gate fails.
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

from alembic.tools.paths import MOUNT_DATA, RUN_FUNCTION_SCRIPT, S3_TRANSFER_SCRIPT, output_dir

# Builtins + typing names the generated server is guaranteed to know (it does
# ``from typing import *`` of these). Capitalised typing generics are included so
# an annotation like ``List[str]`` survives instead of being dropped to untyped.
_SAFE_TYPES = {
    "str",
    "int",
    "float",
    "bool",
    "dict",
    "list",
    "tuple",
    "set",
    "frozenset",
    "bytes",
    "None",
    "Optional",
    "Union",
    "Any",
    "List",
    "Dict",
    "Tuple",
    "Set",
    "FrozenSet",
    "Sequence",
    "Iterable",
    "Mapping",
    "Literal",
}

# typing names emitted as an import in the generated server so preserved
# capitalised annotations (List[str], Dict[str, Any], …) resolve at import time.
_TYPING_IMPORTS = (
    "Any",
    "Dict",
    "FrozenSet",
    "Iterable",
    "List",
    "Literal",
    "Mapping",
    "Optional",
    "Sequence",
    "Set",
    "Tuple",
    "Union",
)


def _safe_annotation(node: ast.expr | None) -> str | None:
    """Unparse an annotation only if every Name in it is a builtin/typing type
    the server venv is guaranteed to know — anything repo-specific is dropped."""
    if node is None:
        return None
    if any(isinstance(n, ast.Name) and n.id not in _SAFE_TYPES for n in ast.walk(node)):
        return None
    if any(isinstance(n, ast.Attribute) for n in ast.walk(node)):
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _type_name(value: object) -> str | None:
    """Builtin type annotation for a concrete value (bool before int — bool is an
    int subclass). None for values we can't map (e.g. ``None``)."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, set):
        return "set"
    if isinstance(value, dict):
        return "dict"
    return None


def _type_from_default(node: ast.expr | None) -> str | None:
    """Infer a param type from its literal default (``['all-cells']`` → ``list``,
    ``True`` → ``bool``, ``1`` → ``int``). None for non-literals or ``None``."""
    if node is None:
        return None
    try:
        return _type_name(ast.literal_eval(node))
    except (ValueError, SyntaxError):
        return None


def _literal_default(node: ast.expr | None) -> str | None:
    """Unparse a default only when it is a literal; a non-literal default makes
    the wrapper param required (the coder is instructed to use literals)."""
    if node is None:
        return None
    try:
        ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    return ast.unparse(node)


def _find_def(source: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None


def _param_annotation(
    arg: ast.arg, default_node: ast.expr | None, sample_args: dict | None
) -> str | None:
    """Best available type for a param, in precedence order (E1.4): (a) the
    function's own server-safe annotation, else (b) its literal default's type,
    else (c) the recorded ``sample_args`` value's type. Untyped params make an LLM
    pass mis-serialised args (a JSON string for a list), so typing them is what
    lets the tool actually be called."""
    ann = _safe_annotation(arg.annotation)
    if ann is None:
        ann = _type_from_default(default_node)
    if ann is None and sample_args and arg.arg in sample_args:
        ann = _type_name(sample_args[arg.arg])
    return ann


def tool_signature(
    name: str, out_dir: Path | None = None, sample_args: dict | None = None
) -> dict | None:
    """Extract {name, params: [(name, annotation|None, default|None)], doc}
    from tools/<name>.py. None if the file/function is missing or unparseable.

    ``sample_args`` (the plan's recorded invocation for this tool) is used to type
    un-annotated, un-defaulted params so the served schema is not typeless."""
    out = out_dir or output_dir()
    f = out / "tools" / f"{name}.py"
    if not f.exists():
        return None
    fn = _find_def(f.read_text(encoding="utf-8", errors="replace"), name)
    if fn is None:
        return None
    a = fn.args
    pos = [*a.posonlyargs, *a.args]
    defaults: list[ast.expr | None] = [None] * (len(pos) - len(a.defaults)) + list(
        a.defaults
    )
    params = [
        (arg.arg, _param_annotation(arg, d, sample_args), _literal_default(d))
        for arg, d in zip(pos, defaults)
    ]
    params += [
        (
            arg.arg,
            _param_annotation(arg, d, sample_args),
            _literal_default(d) if d else None,
        )
        for arg, d in zip(a.kwonlyargs, a.kw_defaults)
    ]
    return {
        "name": name,
        "params": params,
        "doc": ast.get_docstring(fn) or f"Run {name}.",
    }


def function_param_names(
    name: str, out_dir: Path | None = None
) -> tuple[set[str] | None, bool]:
    """(accepted param names, has **kwargs) for the generated tools/<name>.py
    function — the ground truth for which kwargs it actually accepts. Returns
    (None, False) if the file/function is missing/unparseable. Used to filter
    sample args against the REAL wrapper signature (task tools rename the repo's
    params, so the repo symbol's params are the wrong thing to filter against)."""
    out = out_dir or output_dir()
    f = out / "tools" / f"{name}.py"
    if not f.exists():
        return None, False
    fn = _find_def(f.read_text(encoding="utf-8", errors="replace"), name)
    if fn is None:
        return None, False
    a = fn.args
    names = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    return names, a.kwarg is not None


def _render_param(p: tuple[str, str | None, str | None]) -> str:
    name, ann, default = p
    s = name + (f": {ann}" if ann else "")
    return s + (f" = {default}" if default is not None else "")


def _escape_doc(doc: str) -> str:
    escaped = doc.replace("\\", "\\\\").replace('"""', r"\"\"\"")
    # A doc ending in a bare quote would glue onto the closing triple-quote.
    return escaped + " " if escaped.endswith('"') else escaped


def render_server(repo_name: str, signatures: list[dict]) -> str:
    """Render the FastMCP server from extracted tool signatures."""
    blocks = []
    for sig in signatures:
        # required params first — a literal-defaulted param may follow one
        # without a default in the original, which Python forbids; reorder.
        params = sorted(sig["params"], key=lambda p: p[2] is not None)
        declared = {n for n, _, _ in params}
        # Every tool additionally accepts an optional user_id/session_id pair
        # — SessionScopePlugin (CoScientist/tools/session_scope_plugin.py)
        # fills them in at the ADK tool-call boundary when a tool declares
        # them and the caller left them blank. Declared unconditionally, not
        # only when S3 is configured: whether S3 is on is a runtime env
        # check _call() makes per call, but this signature is baked in at
        # codegen time — the params exist (and are silently ignored by _call)
        # even on a server with no S3 configured at all. Skipped for a tool
        # whose own signature already names one of them — no duplicate
        # parameter — but in that case the declared value flows ONLY into
        # the helper kwargs (via payload below, unchanged); "" is passed to
        # _call() for that scope slot instead (see call_user_id/
        # call_session_id below), because the tool's OWN user_id/session_id
        # is domain data (e.g. a database row id) with nothing to do with
        # the S3 namespace — mixing it in would leak that value into the
        # bucket layout. "" falls back to the request headers / local-
        # default, same as a caller that never sends the scope params.
        scope_extras = [
            (n, "str", '""') for n in ("user_id", "session_id") if n not in declared
        ]
        args = ", ".join(_render_param(p) for p in (*params, *scope_extras))
        payload = ", ".join(f'"{n}": {n}' for n, _, _ in sig["params"])
        doc = _escape_doc(sig["doc"])
        if scope_extras:
            names = " / ".join(n for n, *_ in scope_extras)
            hint_word = "hints" if len(scope_extras) > 1 else "hint"
            pronoun = "them" if len(scope_extras) > 1 else "it"
            doc += (
                f"\n\n    {names}: optional S3 vault scope {hint_word}, filled in "
                "automatically by the framework (SessionScopePlugin) when left "
                f"blank — leave {pronoun} empty unless you deliberately need "
                "another session's objects."
            )
        call_user_id = "user_id" if "user_id" not in declared else '""'
        call_session_id = "session_id" if "session_id" not in declared else '""'
        blocks.append(
            f"@mcp.tool()\n"
            f"def {sig['name']}({args}) -> dict:\n"
            f'    """{doc}"""\n'
            f'    return _call("{sig["name"]}", {{{payload}}}, '
            f"{call_user_id}, {call_session_id})\n"
        )
    tools_src = "\n\n".join(blocks)
    typing_imports = ", ".join(_TYPING_IMPORTS)
    return f'''"""FastMCP server for {repo_name} — generated by alembic.

Each tool shells through the tools venv via helpers/run_function.py — the same
runner that validated the tool functions, so serving and validation share one
execution path (two-venv layouts work unchanged).

When S3 is configured (ENDPOINT_URL/ACCESS_KEY/SECRET_KEY/BUCKET_NAME — see
helpers/s3_transfer.py) a *_path/*_file kwarg given as an s3:// or http(s)://
URI is downloaded to a per-call scratch dir before the tool runs, and any
*_path/*_file the tool returns as an existing local file (outside the mounted
data, and inside the cloned repo only when this call wrote it) is uploaded and
presigned after it. Without those four variables set
the RUNTIME behaves exactly as before S3 support existed — but the tool
SCHEMA does not: every tool below always declares the trailing
user_id/session_id params regardless of whether S3 is configured, since that
decision is baked in at codegen time, not made at runtime.

Every tool below also declares optional user_id/session_id scope params
(filled in automatically by SessionScopePlugin at the ADK boundary when left
blank) that take priority over the X-Coscientist-* request headers for the
S3 key prefix; both fall back to ("local", "default") when neither is set.
"""
import importlib.util
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import {typing_imports}

from fastmcp import FastMCP

_OUT = Path(__file__).resolve().parent
_PYTHON = str(_OUT / ".venv" / "bin" / "python")   # main venv: repo + deps
_RUNNER = str(_OUT / "helpers" / "run_function.py")
_REPOS_DIR = _OUT.parent / "repos"                  # published only when the call wrote the file
_MOUNT_DATA = Path("{MOUNT_DATA}")                  # read-only benchmark data, never published
_SENTINEL = "<<<ALEMBIC_RESULT>>>"

mcp = FastMCP("{repo_name}")


class _S3Unavailable:
    """No-op stand-in for helpers/s3_transfer.py when it is missing or fails
    to load — S3 support is simply off, exactly as if it were never
    configured. A broken/absent helper must never take down the base
    tool-calling path (which owes nothing to S3 at all)."""

    @staticmethod
    def s3_enabled() -> bool:
        return False

    @staticmethod
    def scope_from_headers(headers):
        return ("local", "default")

    @staticmethod
    def safe_id(v, default):
        # No-op counterpart to helpers/s3_transfer.py:safe_id — kept so this
        # stub stays a COMPLETE stand-in: s3_enabled() being False already
        # short-circuits _call() before _s3_scope() would reach this, but a
        # future reordering of that call sequence must not turn a successful
        # subprocess run into an AttributeError against this shim. Always
        # returns `default` — this shim does NOT sanitize/pass through `v`
        # the way the real safe_id() does, since S3 being off means nothing
        # downstream ever looks at the result anyway.
        return default


# Stdlib-only at import time (boto3 lives inside its functions), so loading it
# costs nothing on a server with S3 unconfigured — see helpers/s3_transfer.py.
# Guarded: a missing/broken helper file degrades to S3-off rather than
# breaking every tool call.
try:
    _s3_spec = importlib.util.spec_from_file_location(
        "s3_transfer", _OUT / "helpers" / "s3_transfer.py")
    _s3 = importlib.util.module_from_spec(_s3_spec)
    _s3_spec.loader.exec_module(_s3)
except Exception:
    _s3 = _S3Unavailable()


def _s3_scope(user_id: str = "", session_id: str = "") -> tuple[str, str]:
    """(user, session) for one call's S3 key prefix. The user_id/session_id
    tool params (filled in by SessionScopePlugin at the ADK boundary — see
    CoScientist/tools/session_scope_plugin.py) take priority when BOTH are
    non-empty; otherwise this falls back to the request's X-Coscientist-*
    headers (cheap and compatible: a caller that bypasses the plugin, stdio
    transport, or an old fastmcp with no header access), which in turn fall
    back to ("local", "default")."""
    if user_id and session_id:
        return _s3.safe_id(user_id, "local"), _s3.safe_id(session_id, "default")
    headers = None
    try:
        from fastmcp.server.dependencies import get_http_headers
        headers = get_http_headers()
    except Exception:
        pass
    return _s3.scope_from_headers(headers)


def _call(tool: str, kwargs: dict, user_id: str = "", session_id: str = "") -> dict:
    scratch = None
    try:
        # prepare_kwargs() runs INSIDE the try — a failed input download (S3
        # unreachable, a typo'd URI, an S3_HTTP_MAX_BYTES overrun) must still
        # hit the finally below, or every failed call leaks its scratch dir.
        if _s3.s3_enabled():
            scratch = _OUT / ".scratch" / uuid.uuid4().hex
            kwargs = _s3.prepare_kwargs(kwargs, scratch)
        started = time.time()
        r = subprocess.run([_PYTHON, _RUNNER, str(_OUT), tool, json.dumps(kwargs)],
                           cwd=str(_OUT), capture_output=True, text=True)
        parts = r.stdout.rsplit(_SENTINEL, 1)
        if len(parts) == 2:
            out = json.loads(parts[1].strip())
            if out.get("ok"):
                res = out.get("result")
                result = res if isinstance(res, dict) else {{"result": res}}
                if scratch is not None:
                    prefix = _s3.call_prefix(
                        _s3_scope(user_id, session_id), "{repo_name}", tool)
                    # scratch is denied too: a tool echoing its downloaded
                    # input_path must not re-upload it or return a local path
                    # the finally below is about to delete. The mounted data
                    # is the same case for an input given as a local path. In
                    # the repo only what this call wrote is an output.
                    result = _s3.publish_result(
                        result, prefix, (_MOUNT_DATA, scratch), _REPOS_DIR, started)
                return result
            raise RuntimeError(out.get("error") or "tool failed")
        raise RuntimeError((r.stderr or r.stdout)[-2000:] or "runner produced no output")
    finally:
        # Runs after a successful publish_result() above (so its uploads see
        # the file first) AND on every failure/timeout/exception path — a
        # long-lived server must never accumulate scratch dirs from failed
        # calls.
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


{tools_src}

if __name__ == "__main__":
    mcp.run()
'''


def write_server(
    repo_name: str, tool_names: list[str], sample_args: dict | None = None
) -> dict:
    """Generate output/server.py + output/helpers/run_function.py and
    output/helpers/s3_transfer.py for every tool whose signature extracts
    cleanly. Returns {written, tools, skipped}.

    ``sample_args`` maps tool name → its recorded invocation args (the plan's
    ``ToolSpec.sample_args``); it types un-annotated params so the served MCP
    schema is not typeless (E1.4)."""
    out = output_dir()
    sample_args = sample_args or {}
    sigs, skipped = [], []
    for name in tool_names:
        sig = tool_signature(name, out, sample_args=sample_args.get(name))
        (sigs if sig else skipped).append(sig or name)
    helpers = out / "helpers"
    helpers.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUN_FUNCTION_SCRIPT, helpers / "run_function.py")
    shutil.copy(S3_TRANSFER_SCRIPT, helpers / "s3_transfer.py")
    server = out / "server.py"
    server.write_text(render_server(out.parent.name, sigs), encoding="utf-8")
    return {
        "written": str(server),
        "tools": [s["name"] for s in sigs],
        "skipped": skipped,
    }


# The transcript is recorded in a live build where each command runs with
# whatever cwd and venv state the previous one left behind. Replayed cold on a
# clean image those two assumptions break, so the commands are normalised as
# they are written — not patched afterwards by whoever replays them.
_PORTABILITY_FIXES = (
    # `uv venv` does not install pip, but transcripts routinely go on to call
    # `.venv/bin/pip` directly. Seed it so those commands resolve.
    (re.compile(r"\buv venv\b(?!\s+--seed)"), "uv venv --seed"),
    # A relative `cd .alembic/...` resolves against wherever an earlier `cd`
    # left us. Anchor it to /work, where setup.sh starts.
    (re.compile(r"(^|&&\s*)cd \.alembic/"), r"\1cd /work/.alembic/"),
)


def portable_command(command: str) -> str:
    """One recorded env-stage command, rewritten so a cold replay reproduces it."""
    for pattern, replacement in _PORTABILITY_FIXES:
        command = pattern.sub(replacement, command)
    return command


def render_setup_sh(commands: list[str]) -> str:
    """setup.sh from the recorded transcript of successful env-stage commands."""
    commands = [portable_command(c) for c in commands]
    body = (
        "\n".join(commands) if commands else "# (no environment commands were recorded)"
    )
    return (
        "#!/usr/bin/env bash\n"
        "# Environment setup transcript — the commands that actually succeeded\n"
        "# during the alembic Environment stage, in order. Recorded by code.\n"
        "set -euo pipefail\n"
        "cd /work\n\n"
        f"{body}\n"
    )


def write_setup_sh(commands: list[str]) -> Path:
    out = output_dir() / "setup.sh"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    rendered = render_setup_sh(commands)
    if commands or not existing:
        out.write_text(rendered, encoding="utf-8")
        out.chmod(0o755)
    return out


def render_code_py(tool_name: str, out_dir: Path | None = None) -> str | None:
    """TM-Bench export: the tool function copied verbatim (self-contained by
    construction — imports live inside the function body)."""
    out = out_dir or output_dir()
    f = out / "tools" / f"{tool_name}.py"
    if not f.exists():
        return None
    source = f.read_text(encoding="utf-8", errors="replace")
    fn = _find_def(source, tool_name)
    if fn is None:
        return None
    segment = ast.get_source_segment(source, fn)
    return (segment + "\n") if segment else None
