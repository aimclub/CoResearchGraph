"""Deterministic repo analysis for the Plan gate — no LLM.

Two jobs, both pure functions of the cloned source:
  * ``symbol_table`` + ``verify_target`` — build a module→symbol map by AST and
    confirm each Explorer-proposed tool target actually exists (extracting its
    real parameter names). This is the single strongest anti-hallucination
    mechanism, independently confirmed by ToolMaker and ToolRosella.
  * ``decide_layout`` — one-venv vs two-venv from the repo's declared Python
    constraint, so the Environment agent is handed the answer instead of
    re-deriving a prose decision tree every run (audit N15).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_SKIP_DIRS = {".git", "__pycache__", "build", "dist", "node_modules",
              ".tox", ".mypy_cache", ".pytest_cache", ".venv", ".venv-repo"}


def _module_name(py: Path, root: Path) -> str:
    rel = py.relative_to(root).with_suffix("")
    parts = [p for p in rel.parts]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _func_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = node.args
    names = [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
    return [n for n in names if n not in ("self", "cls")]


def symbol_table(repo_dir: Path) -> dict:
    """Return {"modules": {dotted: {"functions": {name:[params]}, "classes":[..]}},
    "names": {symbol: [{"module","kind","params"}]}} for every top-level def."""
    modules: dict[str, dict] = {}
    names: dict[str, list] = {}
    for py in repo_dir.rglob("*.py"):
        rel = py.relative_to(repo_dir)   # check parts RELATIVE to the repo — the
        # absolute path often lives under a dotted workdir like .alembic/, which
        # a startswith(".") check on absolute parts would wrongly skip entirely.
        if any(part in _SKIP_DIRS or part.startswith(".") for part in rel.parts):
            continue
        if re.search(r"(^|/)test", str(rel)):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        mod = _module_name(py, repo_dir)
        funcs: dict[str, list] = {}
        classes: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                funcs[node.name] = _func_params(node)
                names.setdefault(node.name, []).append(
                    {"module": mod, "kind": "function", "params": _func_params(node)})
            elif isinstance(node, ast.ClassDef):
                if node.name.startswith("_"):
                    continue
                classes.append(node.name)
                names.setdefault(node.name, []).append(
                    {"module": mod, "kind": "class", "params": []})
                # Index public methods so a "Class.method" tool target can be
                # verified precisely (not dropped as if hallucinated).
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and not item.name.startswith("_"):
                        mp = _func_params(item)
                        for key in (f"{node.name}.{item.name}", item.name):
                            names.setdefault(key, []).append(
                                {"module": mod, "kind": "method", "params": mp})
        modules[mod] = {"functions": funcs, "classes": classes}
    return {"modules": modules, "names": names}


def verify_target(target: str, table: dict, repo_dir: Path) -> dict:
    """Verify one tool target against the AST table / filesystem.

    ``target`` is "module.path:symbol", "script:relpath", or a bare symbol.
    Returns {"ok": bool, "params": [...], "reason": str}. "ok" is True when the
    symbol clearly exists (exact module match preferred, any-module match
    accepted as a likely re-export); False only when it appears nowhere.
    """
    target = (target or "").strip()
    if target.startswith("script:"):
        rel = target[len("script:"):].strip()
        exists = (repo_dir / rel).is_file()
        return {"ok": exists, "params": [],
                "reason": "" if exists else f"script '{rel}' not found in repo"}

    module, _, symbol = target.rpartition(":")
    symbol = (symbol or module).strip()
    module = module.strip()

    # The Explorer sometimes names a CLI script as the target but omits the
    # "script:" prefix (e.g. "MedSAM_Inference.py" or "src/run.py"). If the
    # bare target looks like a file path and that file exists, accept it as a
    # script — don't drop a real entry point as a hallucinated symbol.
    if not module and (symbol.endswith(".py") or "/" in symbol):
        if (repo_dir / symbol).is_file():
            return {"ok": True, "params": [], "reason": f"script:{symbol}"}

    hits = table["names"].get(symbol, [])
    if not hits and "." in symbol:
        # "Class.method" whose exact pair wasn't indexed — accept if the base
        # class exists (method may be inherited/dynamic; the coder/static gate
        # will catch a truly wrong name later). Only drop when neither exists.
        base = symbol.split(".")[0]
        if table["names"].get(base):
            return {"ok": True, "params": [],
                    "reason": f"'{symbol}' not statically found but class '{base}' exists"}
    if not hits:
        return {"ok": False, "params": [],
                "reason": f"symbol '{symbol}' defined nowhere in the repo (possible hallucination)"}
    exact = [h for h in hits if h["module"] == module] if module else []
    chosen = exact[0] if exact else hits[0]
    note = "" if exact or not module else f"'{symbol}' found in '{chosen['module']}', not '{module}' (re-export?)"
    return {"ok": True, "params": chosen["params"], "reason": note}


def target_top_modules(targets: list[str]) -> list[str]:
    """Top-level module names referenced by plan tool targets — the env gate
    imports these in the tools venv as a repo-import smoke test (R3)."""
    mods = set()
    for target in targets:
        target = (target or "").strip()
        if target.startswith("script:") or ":" not in target:
            continue
        module = target.rpartition(":")[0]
        if module:
            mods.add(module.split(".")[0])
    return sorted(mods)


# ── Layout decision ──────────────────────────────────────────────────────────
_SERVER_CANDIDATES = ["3.11", "3.12", "3.13", "3.10"]   # preference order, all ≥3.10
_OLD_CANDIDATES    = ["3.9", "3.8", "3.7"]


def _read_python_constraint(repo_dir: Path) -> str | None:
    py = repo_dir / "pyproject.toml"
    if py.exists():
        txt = py.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"requires-python\s*=\s*[\"']([^\"']+)", txt)
        if m:
            return m.group(1).strip()
        m = re.search(r"^\s*python\s*=\s*[\"']([^\"']+)", txt, re.MULTILINE)  # poetry
        if m:
            return _poetry_to_pep440(m.group(1).strip())
    for fn in ("setup.py", "setup.cfg"):
        f = repo_dir / fn
        if f.exists():
            m = re.search(r"python_requires\s*=\s*[\"']([^\"']+)", f.read_text(encoding="utf-8", errors="replace"))
            if m:
                return m.group(1).strip()
    return None


def _poetry_to_pep440(spec: str) -> str:
    """Best-effort convert a poetry caret/tilde spec to a PEP 440 range."""
    m = re.match(r"[\^~](\d+)\.(\d+)", spec)
    if not m:
        return spec
    major, minor = int(m.group(1)), int(m.group(2))
    if spec[0] == "^":
        return f">={major}.{minor},<{major + 1}.0"
    return f">={major}.{minor},<{major}.{minor + 1}"


def decide_layout(repo_dir: Path) -> dict:
    """Return {"layout","server_python","repo_python","source"}.

    Always two-venv: the server venv (``.venv-server``, fastmcp) is isolated from
    the main venv (``.venv`` — the repo + its deps + pytest), so fastmcp's
    dependency tree can never conflict with the repo's. ``server_python`` is a
    fixed ≥3.10 version for fastmcp; ``repo_python`` is what the repo's own
    ``requires-python`` asks for (a modern version it admits, else the newest
    older one, else a modern default)."""
    SERVER = "3.11"
    default = {"layout": "two-venv", "server_python": SERVER,
               "repo_python": "3.11", "source": "default (no constraint declared)"}
    spec = _read_python_constraint(repo_dir)
    if not spec:
        return default
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        S = SpecifierSet(spec)
    except Exception:
        return {**default, "source": f"unparseable constraint {spec!r}"}

    modern = [v for v in _SERVER_CANDIDATES if Version(v) in S]
    if modern:
        repo_py, src = ("3.11" if "3.11" in modern else modern[0]), f"requires-python {spec!r}"
    else:
        old = [v for v in _OLD_CANDIDATES if Version(v) in S]
        repo_py = old[0] if old else "3.11"
        src = f"requires-python {spec!r} (older Python)"
    return {"layout": "two-venv", "server_python": SERVER,
            "repo_python": repo_py, "source": src}
