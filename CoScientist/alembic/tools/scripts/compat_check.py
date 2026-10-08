"""Replay a repo's own import statements inside a venv to surface conflicts.

Run by ``check_venv_compat`` with the *venv's* python:
    python compat_check.py <repo_path>
Prints one JSON line: {"conflicts": {...}, "checked": <int>}.
"""
import sys, ast, importlib.metadata, json
from pathlib import Path

repo_path = Path(sys.argv[1])

# Build the set of top-level module names that are actually installed,
# so we skip repo-internal imports and uninstalled optional deps.
installed_roots = set()
for dist in importlib.metadata.distributions():
    top = dist.read_text("top_level.txt")
    if top:
        for n in top.strip().splitlines():
            n = n.strip()
            if n and not n.startswith("_"):
                installed_roots.add(n)
    else:
        record = dist.read_text("RECORD") or ""
        for line in record.splitlines():
            part = line.split(",")[0].strip().split("/")[0]
            if (not part
                    or part.endswith((".dist-info", ".data"))
                    or part.startswith(("_", "."))
                    or "." in part):
                continue
            installed_roots.add(part[:-3] if part.endswith(".py") else part)

# Collect unique import statements from repo source, filtered to
# installed packages only (avoids false positives from repo-internal code).
stmts = {}  # dedup key -> statement string

for py_file in repo_path.rglob("*.py"):
    try:
        source = py_file.read_text(encoding="utf-8", errors="replace")
        tree   = ast.parse(source, filename=str(py_file))
    except Exception:
        continue

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in installed_roots:
                    stmt = f"import {alias.name}"
                    stmts[stmt] = stmt
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                root = node.module.split(".")[0]
                if root in installed_roots:
                    names = ", ".join(a.name for a in node.names)
                    stmt  = f"from {node.module} import {names}"
                    key   = node.module + ":" + ",".join(
                        sorted(a.name for a in node.names)
                    )
                    stmts[key] = stmt

# Execute each statement; sys.modules caches modules so heavy packages
# (torch, transformers) are only loaded once.
conflicts = {}
for stmt in stmts.values():
    try:
        exec(stmt)  # noqa: S102
    except ImportError as e:
        conflicts[stmt] = {"error": str(e)}
    except Exception as e:
        conflicts[stmt] = {"error": type(e).__name__ + ": " + str(e)}

print(json.dumps({"conflicts": conflicts, "checked": len(stmts)}))
