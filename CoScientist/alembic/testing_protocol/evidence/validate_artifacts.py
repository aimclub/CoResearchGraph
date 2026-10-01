#!/usr/bin/env python3
"""Structural validity of alembic's machine-readable artifacts.

For every repository processed in a benchmark run, the pipeline emits four JSON
artifacts under ``output/<repo>/reports/`` plus a generated FastMCP ``server.py``.
This script validates each of them against an explicit JSON Schema (draft 2020-12)
and, for ``server.py``, against the MCP tool contract (every tool the wrapper
stage reports as wrapped must exist as an ``@mcp.tool()``-decorated function with
a docstring).

Usage:
    python nirsii/evidence/validate_artifacts.py [<run_dir>]

Default run dir: benchmarks/alembic/runs/2026-07-10_tmbench-all-v2
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "benchmarks/alembic/runs/2026-07-10_tmbench-all-v2"

_INT = {"type": "integer", "minimum": 0}
_STAGES = ["explorer", "environment", "coder", "validator", "wrapper"]

PLAN_SCHEMA = {
    "type": "object",
    "required": ["repo_url", "env", "tools", "tasks"],
    "properties": {
        "repo_url": {"type": "string", "pattern": r"^https?://"},
        "env": {
            "type": "object",
            "required": ["layout", "server_python", "dependencies"],
            "properties": {
                "layout": {"type": "string", "enum": ["one-venv", "two-venv"]},
                "dependencies": {"type": "array", "items": {"type": "string"}},
            },
        },
        "tools": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["name", "target", "purpose", "params", "sample_args", "verified"],
                "properties": {
                    "name": {"type": "string", "pattern": r"^[A-Za-z_][A-Za-z0-9_]*$"},
                    "params": {"type": "array"},
                    "sample_args": {"type": "object"},
                    "verified": {"type": "boolean"},
                },
            },
        },
        "tasks": {"type": "array"},
    },
}

STAGE_STATUS_SCHEMA = {
    "type": "object",
    "required": _STAGES,
    "properties": {
        s: {
            "type": "object",
            "required": ["status"],
            "properties": {
                "status": {"type": "string",
                           "enum": ["passed", "failed", "written", "skipped", "partial"]}
            },
        }
        for s in _STAGES
    },
}

METRICS_SCHEMA = {
    "type": "object",
    "required": ["actions_per_stage", "tokens_per_stage", "durations_per_stage",
                 "tool_calls_per_stage", "total_actions", "total_tokens"],
    "properties": {
        "actions_per_stage": {"type": "object", "additionalProperties": _INT},
        "tokens_per_stage": {"type": "object", "additionalProperties": _INT},
        "durations_per_stage": {"type": "object",
                                "additionalProperties": {"type": "number", "minimum": 0}},
        "total_actions": _INT,
        "total_tokens": _INT,
    },
}

VALIDATION_SCHEMA = {
    "type": "object",
    "required": ["tools", "counts", "debugger_rounds"],
    "properties": {
        "tools": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["name", "tests_passed", "tests_total", "exec_ok",
                             "invoc_passed", "invoc_total", "passed", "perfect", "status"],
                "properties": {
                    "name": {"type": "string"},
                    "tests_passed": _INT, "tests_total": _INT,
                    "invoc_passed": _INT, "invoc_total": _INT,
                    "exec_ok": {"type": "boolean"},
                    "passed": {"type": "boolean"}, "perfect": {"type": "boolean"},
                    "status": {"type": "string"},
                },
            },
        },
        "counts": {
            "type": "object",
            "required": ["tools_total", "tools_passed", "tools_perfect", "tests_passed",
                         "tests_total", "invoc_passed", "invoc_total", "exec_ok",
                         "exec_attempted"],
            "additionalProperties": _INT,
        },
        "debugger_rounds": _INT,
    },
}

SCHEMAS = {
    "plan.json": PLAN_SCHEMA,
    "stage_status.json": STAGE_STATUS_SCHEMA,
    "metrics.json": METRICS_SCHEMA,
    "validation.json": VALIDATION_SCHEMA,
}


def _relax(schema):
    """Contract profile: fields the pipeline documents as tri-state accept null.

    ``exec_ok`` is ``bool | None`` (contract.py:152) where None means "no cheap
    real invocation exists" (main.py:691); ``sample_args`` and ``invoc_*`` are
    absent the same way when the plan records no sample invocation. The strict
    profile rejects those nulls, the contract profile accepts them.
    """
    import copy
    sc = copy.deepcopy(schema)
    tools = sc.get("properties", {}).get("tools", {}).get("items", {}).get("properties", {})
    for field, base in (("sample_args", "object"), ("exec_ok", "boolean"),
                        ("invoc_passed", "integer"), ("invoc_total", "integer"),
                        ("tests_passed", "integer"), ("tests_total", "integer")):
        if field in tools:
            tools[field] = {"type": [base, "null"]}
    return sc


CONTRACT_SCHEMAS = {name: _relax(sc) for name, sc in SCHEMAS.items()}


def check_json(path: Path, schema: dict) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        data = json.loads(path.read_text())
    except Exception as exc:                                  # noqa: BLE001
        return False, f"not parseable: {exc}"
    errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: e.path)
    if errors:
        e = errors[0]
        return False, f"schema: {'/'.join(map(str, e.path)) or '<root>'}: {e.message[:80]}"
    return True, "ok"


def check_server(out_dir: Path) -> tuple[bool, str]:
    """server.py must parse and expose every wrapped tool as an MCP tool."""
    server = out_dir / "server.py"
    if not server.exists():
        return False, "missing"
    try:
        tree = ast.parse(server.read_text())
    except SyntaxError as exc:
        return False, f"syntax error line {exc.lineno}"

    exposed = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            (isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "tool")
            or getattr(d, "attr", "") == "tool"
            for d in node.decorator_list
        )
    }
    if not exposed:
        return False, "no @mcp.tool functions"

    status = out_dir / "reports" / "stage_status.json"
    wrapped = []
    if status.exists():
        try:
            wrapped = json.loads(status.read_text()).get("wrapper", {}).get("gate", {}).get(
                "tools_wrapped", []) or []
        except Exception:                                     # noqa: BLE001
            wrapped = []
    missing = [t for t in wrapped if t not in exposed]
    if missing:
        return False, f"tools not exposed: {', '.join(missing[:3])}"
    undocumented = [n for n, f in exposed.items() if not ast.get_docstring(f)]
    if undocumented:
        return False, f"no docstring: {', '.join(undocumented[:3])}"
    return True, f"ok ({len(exposed)} tools)"


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN
    out = run / "output"
    if not out.is_dir():
        sys.exit(f"no output/ under {run}")

    repos = sorted(p.name for p in out.iterdir() if p.is_dir())
    rows, ok_total, all_total = [], 0, 0
    contract_ok, contract_all = 0, 0
    for repo in repos:
        results = []
        for fname, schema in SCHEMAS.items():
            ok, msg = check_json(out / repo / "reports" / fname, schema)
            results.append((fname, ok, msg))
            c_ok, _ = check_json(out / repo / "reports" / fname, CONTRACT_SCHEMAS[fname])
            contract_ok += c_ok
            contract_all += 1
        srv = ("server.py",) + check_server(out / repo)
        results.append(srv)
        contract_ok += srv[1]
        contract_all += 1
        ok_n = sum(1 for _, ok, _ in results if ok)
        ok_total += ok_n
        all_total += len(results)
        rows.append((repo, ok_n, len(results), results))

    print(f"Run: {run}")
    print(f"Repositories: {len(repos)}\n")
    header = f"| {'Repo':<14} | " + " | ".join(f"{n:<16}" for n in list(SCHEMAS) + ["server.py"]) + " |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for repo, ok_n, tot, results in rows:
        cells = " | ".join(f"{('valid' if ok else 'INVALID: ' + msg):<16}" for _, ok, msg in results)
        print(f"| {repo:<14} | {cells} |")

    pct = 100.0 * ok_total / all_total if all_total else 0.0
    cpct = 100.0 * contract_ok / contract_all if contract_all else 0.0
    print(f"\nStrict profile   (no nulls anywhere): {ok_total}/{all_total} = {pct:.1f}%")
    print(f"Contract profile (tri-state nulls ok): {contract_ok}/{contract_all} = {cpct:.1f}%")
    for repo, ok_n, tot, results in rows:
        for fname, ok, msg in results:
            if not ok:
                print(f"  - {repo}/{fname}: {msg}")


if __name__ == "__main__":
    main()
