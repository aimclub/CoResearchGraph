"""Disk-only readers over an alembic build's on-disk state.

Every reader takes an explicit ``workdir`` (the value of ``ALEMBIC_WORKDIR`` that
the build ran under) and ``repo_url`` so the same functions work for the live
run (workdir = ``.alembic``) and for any past job (workdir =
``.alembic/a2a_builds/<job_id>/workdir``). No module state, no dependency on
``alembic.tools.paths._current_repo``.
"""
from __future__ import annotations

import ast
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Optional


# ── path layout (mirrors alembic.tools.paths but explicit) ──────────────────
def _repo_name(repo_url: str) -> str:
    return re.sub(r"\.git$", "", repo_url.rstrip("/").split("/")[-1])


def output_dir(workdir: Path, repo_url: str) -> Path:
    return workdir / _repo_name(repo_url) / "output"


def reports_dir(workdir: Path, repo_url: str) -> Path:
    return workdir / _repo_name(repo_url) / "reports"


def repo_path(workdir: Path, repo_url: str) -> Path:
    return workdir / _repo_name(repo_url) / "repos"


# ── file readers ────────────────────────────────────────────────────────────
_MAX_FILE_CHARS = 14_000
IGNORE_DIRS = {".git", "__pycache__", ".eggs", "dist", "build",
               "node_modules", ".tox", ".mypy_cache", ".pytest_cache",
               "checkpoints", "wandb", "mlruns", ".ipynb_checkpoints",
               ".venv", ".venv-server"}
IGNORE_EXTS = {".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
               ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
               ".pdf", ".zip", ".tar", ".gz", ".h5", ".hdf5",
               ".pt", ".pth", ".ckpt", ".pkl", ".npy", ".npz", ".parquet"}


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _split_sections(md: str) -> dict:
    """Split a markdown report into ``{h2-title: body}`` on ``## `` headers."""
    sections: dict = {}
    current = "_intro"
    buf: list = []
    for line in md.splitlines():
        if line.startswith("## "):
            sections[current] = "\n".join(buf).strip()
            current = line[3:].strip()
            buf = []
        elif line.startswith("# ") and current == "_intro" and not buf:
            sections["_title"] = line[2:].strip()
        else:
            buf.append(line)
    sections[current] = "\n".join(buf).strip()
    return {k: v for k, v in sections.items() if v or k == "_title"}


def build_report(workdir: Path, repo_url: str) -> Optional[dict]:
    """Exploration report as ``{raw, sections}``, or None if not written yet."""
    p = reports_dir(workdir, repo_url) / "exploration.md"
    try:
        content = p.read_text(encoding="utf-8")
    except OSError:
        return None
    return {"raw": content, "sections": _split_sections(content)}


def build_setup(workdir: Path, repo_url: str) -> Optional[str]:
    p = output_dir(workdir, repo_url) / "setup.sh"
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def build_files(workdir: Path, repo_url: str) -> list:
    """server.py + tools/, helpers/, tests/ .py files as {path, lang, content}."""
    out = output_dir(workdir, repo_url)
    files: list = []

    def add(p: Path, lang: str) -> None:
        if not p.is_file():
            return
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        if len(txt) > _MAX_FILE_CHARS:
            txt = txt[:_MAX_FILE_CHARS] + "\n… (truncated)"
        files.append({"path": str(p.relative_to(out)), "lang": lang, "content": txt})

    add(out / "server.py", "python")
    for sub in ("tools", "helpers", "tests"):
        d = out / sub
        if d.is_dir():
            for f in sorted(d.glob("*.py")):
                add(f, "python")
    return files


def _load_plan(workdir: Path, repo_url: str) -> Optional[dict]:
    return _read_json(reports_dir(workdir, repo_url) / "plan.json")


# The @mcp.tool functions of server.py, for builds and old images without a plan.
def _tools_from_server(workdir: Path, repo_url: str) -> list:
    try:
        tree = ast.parse((output_dir(workdir, repo_url) / "server.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any("mcp.tool" in ast.unparse(d) for d in node.decorator_list):
            doc = (ast.get_docstring(node) or "").strip().splitlines()
            out.append({"name": node.name, "purpose": doc[0] if doc else "", "node": node})
    return out


def _tool_def(workdir: Path, repo_url: str, name: str):
    """The generated function ``name`` from ``tools/<name>.py``, if it parses."""
    try:
        tree = ast.parse((output_dir(workdir, repo_url) / "tools" / f"{name}.py")
                         .read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def declared_params(workdir: Path, repo_url: str, name: str) -> set[str] | None:
    """The parameter names of the generated function ``name``, or None when it
    takes ``**kwargs`` or its code cannot be read."""
    fn = _tool_def(workdir, repo_url, name)
    if fn is None or fn.args.kwarg is not None:
        return None
    a = fn.args
    return {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}


def call_args_for(workdir: Path, repo_url: str, name: str, args: dict) -> dict:
    """``args`` without the keys the generated function does not accept.

    The plan's sample args describe the repo function, and the coder may wrap
    it with other parameters. Kept as they are when the function takes
    ``**kwargs`` or cannot be read.
    """
    names = declared_params(workdir, repo_url, name)
    if names is None:
        return args
    return {k: v for k, v in args.items() if k in names}


def build_tools(workdir: Path, repo_url: str) -> dict:
    """Right-panel tool cards: names and purpose from plan.json, the signature
    from the generated code (the coder may deviate from the plan), verdicts
    from validation.json."""
    plan = _load_plan(workdir, repo_url)
    from_server = [] if plan else _tools_from_server(workdir, repo_url)
    if not plan and not from_server:
        return {"tools": [], "title": ""}
    validation = _read_json(reports_dir(workdir, repo_url) / "validation.json") or {}
    from CoScientist.tools.alembic_hub import tool_verdicts

    by_name = ({t.get("name"): t for t in validation.get("tools", [])}
               or tool_verdicts(reports_dir(workdir, repo_url)))
    tools = []
    for t in plan["tools"] if plan else from_server:
        v = by_name.get(t.get("name"), {})
        status = v.get("status")            # perfect | passed | failed | untested
        badge = {"perfect": "pass", "passed": "pass",
                 "failed": "fail"}.get(status)   # None -> pending in the UI
        fn = _tool_def(workdir, repo_url, t.get("name") or "") or t.get("node")
        tools.append({
            "name": t.get("name"),
            "sig": ast.unparse(fn.args) if fn else ", ".join(t.get("params") or []),
            "ret": "dict",
            "desc": t.get("purpose", ""),
            "target": t.get("target"),
            "status": badge,
            "verdict": status,
            "exec_ok": v.get("exec_ok"),
            "invoc_passed": v.get("invoc_passed"),
            "invoc_total": v.get("invoc_total"),
            "perfect": bool(v.get("perfect")),
            "error": v.get("error") or None,
            "runs": [{"input": i.get("args") or {}, "passed": bool(i.get("ok")),
                      "error": i.get("error")} for i in (v.get("invocations") or [])],
        })
    plan_repo = (plan or {}).get("repo_url") or repo_url
    return {"tools": tools, "title": f"{_repo_name(plan_repo)} · MCP server"}


def build_examples(workdir: Path, repo_url: str) -> dict:
    plan = _load_plan(workdir, repo_url)
    if not plan:
        return {"examples": []}
    out = []
    for t in plan.get("tools", []):
        sa = t.get("sample_args")
        if sa is None:
            continue
        out.append({"name": t.get("name"),
                    "args": call_args_for(workdir, repo_url, t.get("name") or "", sa),
                    "evidence": t.get("evidence") or ""})
    return {"examples": out}


def build_syntax_check(workdir: Path, repo_url: str) -> Optional[dict]:
    status = _read_json(reports_dir(workdir, repo_url) / "stage_status.json") or {}
    coder = status.get("coder")
    if not coder:
        return None
    passed = coder.get("status") == "passed"
    gate = coder.get("gate", {})
    detail = "" if passed else json.dumps(gate.get("errors", gate),
                                          ensure_ascii=False)[:2000]
    return {"name": "syntax", "passed": passed, "detail": detail}


def build_tests_check(workdir: Path, repo_url: str) -> Optional[dict]:
    validation = _read_json(reports_dir(workdir, repo_url) / "validation.json")
    if not validation:
        return None
    c = validation.get("counts", {})
    tp, tt = c.get("tests_passed") or 0, c.get("tests_total") or 0
    passed = bool(tt) and tp >= tt
    detail = (f"smoke tests {tp}/{tt}; tools passed "
              f"{c.get('tools_passed', 0)}/{c.get('tools_total', 0)}")
    return {"name": "tests", "passed": passed, "detail": detail}


def build_checks(workdir: Path, repo_url: str) -> list:
    """All available quality-gate badges (syntax + tests) as a list."""
    return [c for c in (build_syntax_check(workdir, repo_url),
                        build_tests_check(workdir, repo_url)) if c]


# ── bundle: a `docker build`-able serving zip ───────────────────────────────
_ARTIFACT_INCLUDE = ("tools", "tests", "helpers", "tmbench", "server.py", "setup.sh")
_ARTIFACT_SKIP = {".venv", ".venv-server", "__pycache__"}
# artifacts.py lives at CoScientist/alembic/web/ — parents[3] is repo root.
_DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker" / "alembic"


def _rel_or_ignored(path: Path, root: Path) -> Optional[str]:
    if not path.is_file() or path.suffix in IGNORE_EXTS:
        return None
    rel = path.relative_to(root)
    if any(part in IGNORE_DIRS for part in rel.parts):
        return None
    return str(rel)


def _bundle_dockerfile(name: str, repo_url: str) -> str:
    return f'''# Self-contained serving image for {name} — generated by alembic.
# Build & run:
#   docker build -t {name}-mcp .
#   docker run -p 8000:8000 {name}-mcp        # MCP (streamable-http) on :8000
FROM python:3.11
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \\
    ALEMBIC_WORKDIR=/work/.alembic MCP_PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends \\
        git curl ca-certificates build-essential pkg-config \\
        libcairo2 libfontconfig1 libx11-6 libxext6 libxrender1 \\
        libgl1 libglib2.0-0 libsm6 libxcb1 \\
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /work
COPY .alembic /work/.alembic
RUN bash /work/.alembic/{name}/output/setup.sh
RUN uv venv /work/.alembic/{name}/output/.venv-server --python 3.11 \\
 && uv pip install --python /work/.alembic/{name}/output/.venv-server/bin/python fastmcp mcp
COPY serve.py      /usr/local/bin/serve.py
COPY entrypoint.py /usr/local/bin/entrypoint.py
EXPOSE 8000
ENTRYPOINT ["python", "/usr/local/bin/entrypoint.py"]
CMD ["serve", "{repo_url}"]
'''


def _bundle_readme(name: str, repo_url: str) -> str:
    return (
        f"# {name} — MCP server bundle\n\n"
        f"Generated by **alembic** from `{repo_url}`. Self-contained: builds a "
        "container that serves the generated tools as a FastMCP (streamable-http) "
        "server.\n\n"
        "## Build & run\n```\n"
        f"docker build -t {name}-mcp .\n"
        f"docker run -p 8000:8000 {name}-mcp\n```\n"
        "The MCP server listens on `http://localhost:8000`.\n"
    )


def _portable_setup_sh(out: Path, repo: Path, name: str) -> str:
    txt = (out / "setup.sh").read_text(encoding="utf-8")
    txt = txt.replace(str(out.resolve()), f".alembic/{name}/output")
    txt = txt.replace(str(repo.resolve()), f".alembic/{name}/repos")
    venvpy = f".alembic/{name}/output/.venv/bin/python"
    lines = []
    for line in txt.splitlines():
        if line.lstrip().startswith("uv pip install ") and "--python" not in line:
            line = line.replace("uv pip install ",
                                f"uv pip install --python {venvpy} ", 1)
        lines.append(line)
    return "\n".join(lines) + "\n"


def bundle_zip(workdir: Path, repo_url: str) -> Optional[bytes]:
    """A `docker build`-able bundle for the build, or None if server not built."""
    name = _repo_name(repo_url)
    out = output_dir(workdir, repo_url)
    if not out.exists() or not (out / "server.py").exists():
        return None
    repo = repo_path(workdir, repo_url)
    prefix = f".alembic/{name}"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for item in _ARTIFACT_INCLUDE:
            p = out / item
            if item == "setup.sh" and p.is_file():
                z.writestr(f"{prefix}/output/setup.sh",
                           _portable_setup_sh(out, repo, name))
            elif p.is_file():
                z.write(p, f"{prefix}/output/{p.name}")
            elif p.is_dir():
                for f in p.rglob("*"):
                    rel = f.relative_to(out)
                    if f.is_file() and not any(part in _ARTIFACT_SKIP for part in rel.parts):
                        z.write(f, f"{prefix}/output/{rel}")
        if repo.is_dir():
            for f in repo.rglob("*"):
                if f.is_file() and _rel_or_ignored(f, repo) is not None:
                    z.write(f, f"{prefix}/repos/{f.relative_to(repo)}")
        for fn in ("serve.py", "entrypoint.py"):
            src = _DOCKER_DIR / fn
            if src.exists():
                z.writestr(fn, src.read_text(encoding="utf-8"))
        z.writestr("Dockerfile", _bundle_dockerfile(name, repo_url))
        z.writestr("README.md", _bundle_readme(name, repo_url))
    return buf.getvalue()


__all__ = [
    "output_dir", "reports_dir", "repo_path",
    "build_report", "build_setup", "build_files", "build_tools",
    "build_examples", "build_syntax_check", "build_tests_check", "build_checks",
    "bundle_zip",
]
