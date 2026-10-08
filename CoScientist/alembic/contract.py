"""The structured inter-stage contract + run-data records.

Machine-critical data (tool list, signatures, sample args, env layout, per-tool
verdicts, stage/gate status) travels between stages and out to the benchmark as
structured JSON parsed and written by code — never as free-text a later stage
or the harness must re-read correctly. ``validation.md`` is still rendered, but
purely for humans: nothing parses it anymore (R2).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

from alembic.tools.paths import reports_dir

# ── Fenced-block extraction ──────────────────────────────────────────────────
# Models frequently forget the closing ``` fence, so we do NOT require it: grab
# everything after the opening fence, cut a closing fence only if present, and
# recover the object structurally (balanced braces, trailing-comma tolerant).
def _after_fence(text: str, lang: str) -> str | None:
    """Text following the first ```<lang> fence, up to a closing ``` if any."""
    m = re.search(rf"```[ \t]*{lang}[ \t]*\n?(.*)", text or "", re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1).split("```", 1)[0]


def _brace_slice(s: str) -> str | None:
    """The first balanced {...} object in ``s`` (quote/escape aware), or None."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def parse_json_block(text: str) -> dict | None:
    """First ```json object in ``text`` as a dict — tolerant of a missing
    closing fence and trailing commas; falls back to any balanced object."""
    text = text or ""
    for chunk in (_after_fence(text, "json"), text):
        if not chunk:
            continue
        blob = _brace_slice(chunk)
        if not blob:
            continue
        for attempt in (blob, re.sub(r",(\s*[}\]])", r"\1", blob)):
            try:
                val = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(val, dict):
                return val
    return None


# ── Plan (Explorer proposal → verified by the Plan gate) ─────────────────────
@dataclass
class ToolSpec:
    name: str
    target: str                       # "module.path:symbol" or "script:relpath"
    purpose: str = ""
    params: list[str] = field(default_factory=list)   # real names (Plan gate)
    sample_args: dict | None = None   # cheap real invocation args (Explorer)
    evidence: str = ""                # documented basis for correctness tests (R6)
    verified: bool = False            # AST-verified against the clone
    note: str = ""                    # why unverified / demoted


@dataclass
class EnvSpec:
    layout: str = "two-venv"          # always two-venv: .venv (main) + .venv-server
    server_python: str = "3.11"       # .venv-server python (fastmcp)
    repo_python: str | None = "3.11"  # .venv (main) python: repo + deps + pytest
    requirements_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    system_libs: list[str] = field(default_factory=list)
    weights: list[dict] = field(default_factory=list)


@dataclass
class Plan:
    repo_url: str
    env: EnvSpec
    tools: list[ToolSpec]
    tasks: list[dict] = field(default_factory=list)   # TM-Bench task specs (R4)

    def to_json(self) -> str:
        return json.dumps(
            {"repo_url": self.repo_url, "env": asdict(self.env),
             "tools": [asdict(t) for t in self.tools], "tasks": self.tasks},
            indent=2, ensure_ascii=False,
        )


def save_plan(plan: Plan) -> None:
    d = reports_dir(plan.repo_url)
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(plan.to_json(), encoding="utf-8")


def load_plan(repo_url: str | None = None) -> Plan | None:
    p = reports_dir(repo_url) / "plan.json"
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return Plan(
        repo_url=raw.get("repo_url", ""),
        env=EnvSpec(**raw.get("env", {})),
        tools=[ToolSpec(**t) for t in raw.get("tools", [])],
        tasks=raw.get("tasks", []),
    )


# ── Per-tool validation record (R6: two-level verdicts) ──────────────────────
@dataclass
class ToolReport:
    """Execution status and test statistics for one tool.

    Semantics (maintainer-specified):
      * ``tests_*``  — the coder's regular/smoke tests (test_smoke_*). None = none exist.
      * ``exec_ok``  — did a direct invocation crash? True also covers "runtime
        success": a timeout or missing input files is NOT a crash (R6).
        None = never invoked (no sample args).
      * ``invoc_*``  — evidence-based correctness tests (test_invoc_*).
        None = no grounds for one existed.
      * passed  = all tests green AND never crashed (and at least one signal exists).
      * perfect = passed AND all invocation-correctness tests green (≥1 exists).
    """
    name: str
    tests_passed: int | None = None
    tests_total: int | None = None
    exec_ok: bool | None = None
    exec_note: str = ""
    invoc_passed: int | None = None
    invoc_total: int | None = None
    error: str = ""                   # last failure detail (debugger input)

    @property
    def passed(self) -> bool:
        if self.exec_ok is False:
            return False
        if self.tests_total and (self.tests_passed or 0) < self.tests_total:
            return False
        return bool(self.tests_total) or self.exec_ok is not None

    @property
    def perfect(self) -> bool:
        return (self.passed and bool(self.invoc_total)
                and (self.invoc_passed or 0) == self.invoc_total)

    @property
    def status(self) -> str:
        if self.perfect:
            return "perfect"
        if self.passed:
            return "passed"
        if self.exec_ok is None and not self.tests_total and not self.invoc_total:
            return "untested"
        return "failed"


@dataclass
class Validation:
    tools: list[ToolReport] = field(default_factory=list)
    debugger_rounds: int = 0
    debugger_actions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def counts(self) -> dict:
        def _sum(attr):
            vals = [getattr(t, attr) for t in self.tools if getattr(t, attr) is not None]
            return sum(vals) if vals else None
        return {
            "tools_total":    len(self.tools),
            "tools_passed":   sum(t.passed for t in self.tools),
            "tools_perfect":  sum(t.perfect for t in self.tools),
            "tests_passed":   _sum("tests_passed"),
            "tests_total":    _sum("tests_total"),
            "invoc_passed":   _sum("invoc_passed"),
            "invoc_total":    _sum("invoc_total"),
            "exec_ok":        sum(1 for t in self.tools if t.exec_ok),
            "exec_attempted": sum(1 for t in self.tools if t.exec_ok is not None),
        }


def _fmt_frac(passed: int | None, total: int | None) -> str:
    return "-" if not total else f"{passed or 0}/{total}"


def render_validation_md(repo_name: str, v: Validation) -> str:
    """Human-readable validation summary. Nothing parses this file (R2)."""
    c = v.counts()
    L = [f"# {repo_name} Validation Report", "",
         f"**Tools: {c['tools_passed']}/{c['tools_total']} passed, "
         f"{c['tools_perfect']} perfect** — tests {_fmt_frac(c['tests_passed'], c['tests_total'])}, "
         f"exec ok {c['exec_ok']}/{c['exec_attempted']}, "
         f"invocation tests {_fmt_frac(c['invoc_passed'], c['invoc_total'])}", "",
         "| Tool | Smoke tests | Exec | Invocation tests | Verdict |",
         "|---|---|---|---|---|"]
    for t in v.tools:
        if t.exec_ok is None:
            ex = "not invoked"
        else:
            ex = "ok" if t.exec_ok else "CRASHED"
        if t.exec_note:
            ex += f" ({t.exec_note})"
        L.append(f"| {t.name} | {_fmt_frac(t.tests_passed, t.tests_total)} | {ex} "
                 f"| {_fmt_frac(t.invoc_passed, t.invoc_total)} | {t.status} |")
    L += ["", f"## Debugger — {v.debugger_rounds} round(s)"]
    L += [f"- {a}" for a in v.debugger_actions] or ["None required."]
    if v.notes:
        L += ["", "## Notes"] + [f"- {n}" for n in v.notes]
    return "\n".join(L) + "\n"


def write_validation(repo_url: str | None, repo_name: str, v: Validation) -> None:
    """Persist validation.json (the harness contract) + validation.md (humans).
    Called incrementally during validation so a crash still leaves valid data."""
    d = reports_dir(repo_url)
    d.mkdir(parents=True, exist_ok=True)
    (d / "validation.json").write_text(json.dumps({
        "tools": [{**asdict(t), "passed": t.passed, "perfect": t.perfect,
                   "status": t.status} for t in v.tools],
        "counts": v.counts(),
        "debugger_rounds": v.debugger_rounds,
        "debugger_actions": v.debugger_actions,
        "notes": v.notes,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (d / "validation.md").write_text(render_validation_md(repo_name, v), encoding="utf-8")


# ── Stage status (R2: run data, written incrementally) ───────────────────────
def update_stage_status(stage: str, repo_url: str | None = None, **fields) -> None:
    """Merge ``fields`` into reports/stage_status.json[stage]. The file is the
    harness's source of truth for how far a run got and what each gate saw."""
    d = reports_dir(repo_url)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "stage_status.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        data = {}
    data.setdefault(stage, {}).update(fields)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
