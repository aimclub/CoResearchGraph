#!/usr/bin/env python3
"""Alembic pipeline orchestration.

Five stage names the harness knows (explorer/environment/coder/validator/
wrapper), interleaved with deterministic gates — the LLM proposes, code
disposes. Every LLM stage runs inside a reset loop (R1): a failed gate rolls
the stage's files back to its start-of-stage checkpoint and reruns it with a
short note about the previous failure.

    clone → EXPLORER ─▶ [G1 plan] ─▶ ENVIRONMENT ─▶ [G2 env] ─▶ CODER
          ─▶ [G3 artefacts] ─▶ VALIDATOR (code loop + batched debugger)
          ─▶ WRAPPER (codegen; LLM only on G4 failure) ─▶ export

All run data (stage_status.json, validation.json, metrics.json) is written
incrementally by code — the benchmark parses no reports (R2).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import asyncio
import json
import os
import shutil
import subprocess
import time

import yaml
from loguru import logger
from google.adk.sessions import InMemorySessionService

from alembic import config
from alembic import events
from alembic.events import emit
from alembic.agents import (
    coder_agent, debugger_agent, environment_agent, explorer_agent, wrapper_agent,
)
from alembic.agent_runtime import classify_error, run_agent
from alembic.contract import (
    EnvSpec, Plan, ToolReport, ToolSpec, Validation,
    load_plan, parse_json_block, save_plan, update_stage_status, write_validation,
)
from alembic.tools import (
    WORKDIR, check_server, check_tool_artefacts, ensure_pytest, get_repo_name,
    invoke_tool_function, run_tool_tests, set_current_repo,
    start_env_recording, stop_env_recording,
)
from alembic.tools.venv import ensure_server_packages, install_repo
from alembic.tools.analysis import decide_layout, symbol_table, target_top_modules, verify_target
from alembic.staging import stage_task_inputs, task_mounts
from alembic.tools.codegen import function_param_names, render_code_py, write_server, write_setup_sh
from alembic.tools.fs import _clone_repo_sync
from alembic.tools.invoke import check_repo_imports
from alembic.tools.paths import MOUNT_DATA, MOUNT_INPUT, output_dir, repo_path, reports_dir, server_python, tools_python
from alembic.tools.shell import record_env_command
from alembic.tools.venv import _check_venv_compat_sync

STAGES = config.STAGES


# ══════════════════════════════════════════════════════════════════════════════
# Target tasks (TM-Bench, R4)
# ══════════════════════════════════════════════════════════════════════════════
def _load_tasks(cli_value: str | None) -> list[dict]:
    """Task spec(s): JSON/YAML text, a path, or comma-separated paths. Each task
    is {name, description, arguments, returns, example, ...}. [] = native mode."""
    raw = cli_value or config.TASKS
    if not raw:
        return []
    texts: list[str] = []
    if all(Path(p.strip()).exists() for p in raw.split(",") if p.strip()):
        texts = [Path(p.strip()).read_text(encoding="utf-8") for p in raw.split(",") if p.strip()]
    else:
        texts = [raw]
    tasks: list[dict] = []
    for text in texts:
        try:
            v = yaml.safe_load(text)   # YAML is a JSON superset
        except yaml.YAMLError:
            logger.warning("[tasks] unparseable task spec — ignoring one entry.")
            continue
        if isinstance(v, dict):
            tasks.append(v)
        elif isinstance(v, list):
            tasks.extend(t for t in v if isinstance(t, dict))
    for t in tasks:
        t.setdefault("name", "task")
    return tasks


def _tasks_prompt(tasks: list[dict], propose_extras: bool = False) -> str:
    if not tasks:
        return ""
    lines = ["", "", "REQUIRED TASKS — your plan MUST include one tool per task, "
             "named EXACTLY as given, with EXACTLY these argument names. These are "
             "the priority: verify them first and keep them at the FRONT of the "
             "tools list:"]
    for t in tasks:
        lines += [f"- name: {t.get('name')}",
                  f"  description: {t.get('description', '')}",
                  f"  arguments: {json.dumps(t.get('arguments', {}))}",
                  f"  returns: {json.dumps(t.get('returns', {}))}",
                  f"  example invocation: {json.dumps(t.get('example', {}))}"]
    if propose_extras:   # explorer only — the coder implements what the plan lists
        lines += ["",
                  "BEYOND the required task(s): the goal is to turn this whole repo "
                  "into an MCP server, so ALSO propose the repo's other most important "
                  "workflow tools (its key train / predict / evaluate / preprocess / "
                  "featurize entry points), best first, each with the same evidence + "
                  "sample_args rigor. Add them AFTER the required tool(s). Only propose "
                  "tools you can ground in real repo code — skip the rest."]
    return "\n".join(lines)


def _hints_prompt(hints: str | None) -> str:
    """A SOFT nudge for the explorer only: free-text describing tool(s) the
    caller hopes to see mined among the others. No forced name/signature, no
    plan-gate enforcement — the opposite end of the spectrum from _tasks_prompt
    (REQUIRED TASKS, exact + gated). Composes with it: required tools stay
    pinned, this just steers what else gets proposed."""
    if not hints or not hints.strip():
        return ""
    return (
        "\n\nSUGGESTED DIRECTION (soft, not a requirement): among the tools "
        "you propose, the caller would especially value something along these "
        "lines:\n"
        + hints.strip()
        + "\nTreat this only as a hint about what would be useful. Do NOT invent "
        "a tool the repo cannot support, do NOT force any particular name or "
        "argument signature to match it, and do NOT let it crowd out the repo's "
        "other important workflow tools. If the repo has real code for it, lean "
        "toward surfacing that tool (best-first); otherwise ignore the hint. "
        "Ground every tool in real repo code as usual."
    )


_DATA_POLICY = ("\n\nDATA POLICY (absolute): do NOT download any datasets, even if the "
                "exploration report or the repo README asks for one. Allowed downloads: "
                "package dependencies, configs, and pretrained model weights/checkpoints "
                "needed so the tools work standalone.")


def _stage_task_inputs(tasks: list[dict]) -> None:
    """Stage each task's example/test_case mount files from the bind-mounted
    data dir into /mount/input. See :mod:`alembic.staging`."""
    stage_task_inputs(tasks, MOUNT_DATA, MOUNT_INPUT, warn=logger.warning)


def _task_mounts(task: dict) -> list[tuple[str, str]]:
    return task_mounts(task)


def _task_ref(tasks: list[dict]) -> str | None:
    """The commit/branch to pin the clone to. Tasks that pin different refs on
    one repo (the STAMP dual case) can't all be honoured — use the first and
    warn; a single-task run pins faithfully."""
    refs = []
    for t in tasks:
        repo = t.get("repo") or {}
        r = repo.get("commit") or repo.get("branch")
        if r:
            refs.append(str(r))
    if not refs:
        return None
    if len(set(refs)) > 1:
        logger.warning(f"[tasks] tasks pin different refs {set(refs)} — cloning at {refs[0]}; "
                       f"tools for the other ref may not resolve (run separately if so).")
    return refs[0]


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════
async def run_pipeline(repo_url: str, resume_from: str | None = None,
                       stop_after: str | None = None, tasks_cli: str | None = None,
                       hints_cli: str | None = None):
    set_current_repo(repo_url)
    name = get_repo_name(repo_url)
    base = WORKDIR / name
    session_service = InMemorySessionService()
    tasks = _load_tasks(tasks_cli)
    hints = hints_cli or config.HINTS
    await emit({"type": "pipeline", "status": "start", "repo": name,
                "repo_url": repo_url})

    for stg in (resume_from, stop_after):
        if stg is not None and stg not in STAGES:
            logger.error(f"Unknown stage '{stg}'. Valid: {', '.join(STAGES)}")
            return

    if resume_from is None:
        _clean_workdir(name)
    else:
        logger.info(f"[Resume] from stage: {resume_from} (workdir preserved)")

    log_file = base / "pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                         level="DEBUG", encoding="utf-8")
    logger.info(f"[Run] {name} — log → {log_file}"
                + (f"  ({len(tasks)} target task(s))" if tasks else "")
                + ("  (soft hint active)" if hints and hints.strip() else ""))

    metrics = _new_metrics()

    def _should_run(stage: str) -> bool:
        idx = STAGES.index(stage)
        if resume_from is not None and idx < STAGES.index(resume_from):
            return False
        if stop_after is not None and idx > STAGES.index(stop_after):
            return False
        return True

    try:
        _clone_repo_sync(repo_url, ref=_task_ref(tasks))  # idempotent; explorer reuses it
        _stage_task_inputs(tasks)

        # ── 1. Explorer + plan gate ─────────────────────────────────────────
        plan_ok = True
        if _should_run("explorer"):
            _banner(1, f"Explorer ({repo_url})")
            plan_ok = await _staged_llm(
                "explorer", explorer_agent, name, metrics, session_service,
                message_fn=lambda note: repo_url + _tasks_prompt(tasks, propose_extras=True)
                                        + _hints_prompt(hints) + note,
                gate_fn=lambda: _plan_gate(repo_url, tasks),
                owned=[reports_dir() / "exploration.md", reports_dir() / "plan.json"],
                required_report="exploration",
                post_fn=lambda final: _salvage_report("exploration", final, needs_plan=True),
            )
            if not plan_ok:
                logger.error("[explorer] no usable plan after resets — downstream stages skipped.")

        # ── 2. Environment + env gate ───────────────────────────────────────
        if plan_ok and _should_run("environment"):
            _banner(2, f"Environment ({repo_url})")
            start_env_recording()
            await _staged_llm(
                "environment", environment_agent, name, metrics, session_service,
                message_fn=lambda note: _environment_message(repo_url, tasks) + note,
                gate_fn=lambda: _env_gate(name, session_service, metrics),
                owned=[output_dir() / ".venv"],
                on_reset=start_env_recording,
            )
            write_setup_sh(stop_env_recording())

        # ── 3. Coder + artefact gate ────────────────────────────────────────
        if plan_ok and _should_run("coder"):
            _banner(3, f"Coder ({repo_url})")
            await _staged_llm(
                "coder", coder_agent, name, metrics, session_service,
                message_fn=lambda note: _coder_message(repo_url, tasks) + note,
                gate_fn=lambda: _coder_gate(),
                owned=[output_dir() / "tools", output_dir() / "tests"],
            )

        # ── 4. Validator (deterministic loop + batched debugger) ────────────
        if plan_ok and _should_run("validator"):
            _banner(4, f"Validator ({repo_url})")
            await _validate(repo_url, name, session_service, metrics)

        # ── 5. Wrapper (codegen + gate + LLM fallback) + export ─────────────
        if plan_ok and _should_run("wrapper"):
            _banner(5, f"Wrapper ({repo_url})")
            await _wrap(name, session_service, metrics)
            _export_task_code(tasks)
            _report_completion(name, base)

        await emit({"type": "pipeline", "status": "complete", "repo": name,
                    "metrics": metrics})

    finally:
        _finalize_metrics(base, metrics)
        logger.remove(sink_id)


# ── LLM stage runner with reset loop (R1) ─────────────────────────────────────
async def _staged_llm(stage, agent, name, metrics, session_service, message_fn,
                      gate_fn, owned, required_report=None, post_fn=None,
                      on_reset=None) -> bool:
    """Run one LLM stage; verify its exit gate; on failure roll back the
    stage-owned paths and rerun with a note (≤ STAGE_RESET extra loops).
    Returns whether the gate ever passed."""
    await emit({"type": "stage", "stage": stage, "status": "running"})
    note = ""
    for attempt in range(config.STAGE_RESET + 1):
        sid = f"{name}_{stage}_r{attempt}"
        await session_service.create_session(
            app_name=config.APP_NAME, user_id=config.USER_ID, session_id=sid)
        final = await _run_llm_stage(stage, agent, sid, message_fn(note),
                                     metrics, session_service, required_report)
        if post_fn:
            post_fn(final)
        gate = gate_fn()
        if asyncio.iscoroutine(gate):
            gate = await gate
        update_stage_status(stage, status="passed" if gate["ok"] else "failed",
                            resets=attempt, gate=gate.get("info", {}))
        if gate["ok"]:
            logger.info(f"[{stage}] gate passed"
                        + (f" after {attempt} reset(s)" if attempt else ""))
            await emit({"type": "stage", "stage": stage, "status": "done",
                        "resets": attempt})
            return True
        note = ("\n\nNOTE — previous attempt failed its exit gate; do better this time:\n"
                + gate.get("note", "unknown failure"))
        if attempt < config.STAGE_RESET:
            _rollback(owned)
            if on_reset:
                on_reset()
            logger.warning(f"[{stage}] gate FAILED — reset "
                           f"{attempt + 1}/{config.STAGE_RESET}: {gate.get('note','')[:300]}")
    logger.error(f"[{stage}] gate still failing after {config.STAGE_RESET} reset(s) — moving on.")
    await emit({"type": "stage", "stage": stage, "status": "failed"})
    return False


async def _run_llm_stage(stage, agent, session_id, message, metrics,
                         session_service, required_report=None) -> str:
    started = time.monotonic()
    timeout = config.STAGE_TIMEOUT.get(stage)          # None = no wall clock (R1)
    deadline = started + timeout * config.REPORT_GRACE_FRACTION if timeout else None
    coro = run_agent(agent, session_service, session_id, message,
                     required_report=required_report, deadline=deadline)
    try:
        if timeout:
            final, steps, tokens, sm = await asyncio.wait_for(coro, timeout=timeout)
        else:
            final, steps, tokens, sm = await coro
    except asyncio.TimeoutError:
        logger.error(f"[{stage}] STAGE TIMEOUT after {timeout}s — continuing.")
        _accumulate_stage(metrics, stage, round(time.monotonic() - started, 1), 0, 0, None)
        metrics["abort_reason_per_stage"][stage] = "stage_timeout"
        return ""
    _accumulate_stage(metrics, stage, round(time.monotonic() - started, 1), steps, tokens, sm)
    return final


def _rollback(paths: list[Path]) -> None:
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink(missing_ok=True)


def _salvage_report(report_name: str, final_text: str,
                    needs_plan: bool = False) -> None:
    """Persist the agent's final response text when write_report wasn't called
    (a common, intermittent LLM slip) — or, for the plan-bearing exploration
    report, when the file holds no parseable JSON plan but the response does."""
    path = reports_dir() / f"{report_name}.md"
    existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    final_text = final_text or ""
    if final_text.strip() in ("", "Agent did not produce a final response."):
        return
    if needs_plan and not parse_json_block(existing) and parse_json_block(final_text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(final_text, encoding="utf-8")
        logger.warning(f"[{report_name}] salvaged plan from the agent's final response")
        return
    if not existing.strip() and final_text.strip():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(final_text, encoding="utf-8")
        logger.warning(f"[{report_name}] salvaged the agent's final response")


# ══════════════════════════════════════════════════════════════════════════════
# G1 — Plan gate (deterministic)
# ══════════════════════════════════════════════════════════════════════════════
def _plan_gate(repo_url: str, tasks: list[dict]) -> dict:
    """Verify the Explorer's proposed tools against the real repo AST, extract
    real parameter names, drop clear hallucinations, decide the venv layout,
    and (task mode) require one tool per task. Writes plan.json."""
    repo_dir = repo_path().resolve()
    exploration = reports_dir() / "exploration.md"
    proposal = parse_json_block(exploration.read_text(encoding="utf-8", errors="replace")) \
        if exploration.exists() else None
    if not proposal:
        return {"ok": False, "note": "exploration.md contains no parseable ```json plan block — "
                                     "end the report with the required fenced JSON plan."}

    layout = decide_layout(repo_dir)
    env_raw = proposal.get("env", {}) if isinstance(proposal.get("env"), dict) else {}
    env = EnvSpec(
        layout=layout["layout"], server_python=layout["server_python"], repo_python=layout["repo_python"],
        requirements_files=env_raw.get("requirements_files", []),
        dependencies=env_raw.get("dependencies", []),
        system_libs=env_raw.get("system_libs", []),
        weights=env_raw.get("weights", []),
    )

    table = symbol_table(repo_dir)
    task_by_name = {t.get("name"): t for t in tasks}
    tools: list[ToolSpec] = []
    dropped: list[str] = []
    for t in (proposal.get("tools") or [])[: config.MAX_TOOLS + 4]:
        if not isinstance(t, dict) or not t.get("name") or not t.get("target"):
            continue
        v = verify_target(t["target"], table, repo_dir)
        if not v["ok"]:
            dropped.append(f"{t['name']} ({t['target']}: {v['reason']})")
            continue
        sample = t.get("sample_args") if isinstance(t.get("sample_args"), dict) else None
        task = task_by_name.get(t["name"])
        if task and isinstance((task.get("example") or {}).get("arguments"), dict):
            sample = task["example"]["arguments"]   # authoritative for task tools
        tools.append(ToolSpec(name=t["name"], target=t["target"], purpose=t.get("purpose", ""),
                              params=v["params"], sample_args=sample,
                              evidence=str(t.get("evidence") or ""), verified=True, note=v["reason"]))
        if len(tools) >= config.MAX_TOOLS:
            break

    missing_tasks = [n for n in task_by_name if n not in {t.name for t in tools}]
    plan = Plan(repo_url=repo_url, env=env, tools=tools, tasks=tasks)
    save_plan(plan)
    info = {"layout": env.layout, "verified": len(tools), "dropped": len(dropped),
            "missing_tasks": missing_tasks}
    logger.info(f"[plan gate] layout={env.layout} ({layout['source']}); "
                f"verified {len(tools)} tool(s), dropped {len(dropped)}."
                + (f" MISSING required task tools: {missing_tasks}" if missing_tasks else ""))
    if dropped:
        logger.info(f"[plan gate] dropped: {'; '.join(dropped)}")

    if not tools:
        return {"ok": False, "info": info,
                "note": "none of the proposed tool targets exist in the repo: "
                        + "; ".join(dropped[:6])}
    if missing_tasks:
        return {"ok": False, "info": info,
                "note": f"the plan must include tool(s) named exactly {missing_tasks} "
                        f"implementing the required task(s), targeting real repo code."}
    return {"ok": True, "info": info}


# ══════════════════════════════════════════════════════════════════════════════
# G2 — Env gate (deterministic compat + repo-import smoke; one debugger round)
# ══════════════════════════════════════════════════════════════════════════════
def _uv_venv(path: Path, python_version: str) -> str | None:
    """Create a venv at ``path`` on ``python_version`` via uv. Returns an error
    string or None."""
    try:
        r = subprocess.run(["uv", "venv", str(path), "--python", python_version],
                           capture_output=True, text=True, timeout=config.VENV_SETUP_TIMEOUT)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return f"venv creation failed: {e}"
    return None if r.returncode == 0 else f"venv creation failed: {r.stderr[-300:]}"


def _ensure_main_venv(plan) -> None:
    """Guarantee the main ``.venv`` exists on the repo's Python (build it if the
    env agent didn't) so the deterministic ``install_repo`` has somewhere to go."""
    out = output_dir()
    if (out / ".venv" / "bin" / "python").exists():
        return
    py = (plan.env.repo_python if plan and plan.env.repo_python else "3.11")
    err = _uv_venv(out / ".venv", py)
    if err:
        logger.warning(f"[env gate] could not pre-build .venv: {err}")


def _ensure_server_venv(plan) -> str | None:
    """Build the isolated ``.venv-server`` (fastmcp only) — deferred to the
    wrapper stage so the environment stage never touches server concerns.
    Returns an error string or None."""
    out = output_dir()
    if not (out / ".venv-server" / "bin" / "python").exists():
        py = (plan.env.server_python if plan else "3.11")
        err = _uv_venv(out / ".venv-server", py)
        if err:
            return err
    return ensure_server_packages(server_python(out.resolve()))


async def _env_gate(name: str, session_service, metrics) -> dict:
    """Deterministic env verification. HARD failures fail the gate (main venv
    missing, pytest absent, or a planned tool's target module won't import).
    fastmcp is NOT checked here — the server venv is a wrapper-stage concern.
    check_venv_compat conflicts are SOFT — a broken numpy ABI would
    already fail the repo-import smoke, so a surviving conflict is in peripheral
    code the tools don't touch; surface it to the debugger but don't loop a
    version-pinned repo forever on it."""
    plan = load_plan()
    out = output_dir()

    async def _checks() -> tuple[list[str], list[str]]:
        hard: list[str] = []
        soft: list[str] = []
        if not (out / ".venv" / "bin" / "python").exists():
            return [f"main venv missing: {out}/.venv was never created"], []
        pytest_err = await asyncio.to_thread(ensure_pytest, tools_python(out.resolve()))
        if pytest_err:
            hard.append(pytest_err)
        # Compat check only on the main venv — the server venv (fastmcp) is
        # built and probed at the wrapper stage, isolated from the repo's deps.
        r = await asyncio.to_thread(_check_venv_compat_sync, ".venv")
        if r.get("error"):
            soft.append(f"compat check on .venv: {r['error']}")
        else:
            for stmt, detail in list(r.get("conflicts", {}).items())[:8]:
                soft.append(f"[.venv] `{stmt}` fails: {detail.get('error', '')[:200]}")
        mods = target_top_modules([t.target for t in (plan.tools if plan else [])])
        if mods:
            ri = await asyncio.to_thread(check_repo_imports, mods)
            for mod, err in ri["errors"].items():
                hard.append(f"repo-import smoke: `import {mod}` fails in the main venv: {err[:300]}")
        return hard, soft

    # Deterministic first (R3, 'code disposes'): guarantee the main venv exists
    # and install the repo + its deps into it BEFORE checking — a repo-import
    # smoke failure is almost always just the un-installed repo package, and one
    # unified install path (requirements → editable → .pth fallback) beats
    # spending an LLM debugger round rediscovering the same command every run.
    await asyncio.to_thread(_ensure_main_venv, plan)
    ir = await asyncio.to_thread(install_repo)
    if ir.get("steps") or ir.get("note"):
        logger.info(f"[env gate] deterministic install_repo: "
                    f"{'; '.join(ir.get('steps', [])) or ir.get('note', '')}")

    hard, soft = await _checks()

    if hard or soft:
        # one bounded debugger round before deciding (R3)
        logger.warning(f"[env gate] {len(hard)} hard + {len(soft)} soft problem(s) — "
                       f"one debugger round.")
        await _call_debugger(
            name, session_service, metrics,
            "The built Python environment fails its deterministic checks. Fix ONLY "
            "the environment — do NOT write any application code. Everything installs "
            "into the single main venv `.venv`: `uv pip install --python "
            "<output>/.venv/bin/python <pkg-or-repo-dir>`, and do NOT create "
            "additional venvs (the fastmcp server venv is built separately, later). "
            "A 'repo-import smoke' failure means the repo package itself is not "
            "importable in `.venv` (usually a missing system lib or an unresolved "
            "dependency the automatic install could not satisfy) — the repo dir was "
            "already installed for you, so fix the underlying cause; it is NOT a tool "
            "to create. There are NO tool or test files yet (the Coder stage has not "
            "run): do NOT create tools/, tests/, or any .py files, and do NOT call "
            "run_tool_tests / invoke_tool_function. Only shell/install commands. "
            "Problems:\n- " + "\n- ".join(hard + soft))
        hard, soft = await _checks()

    info = {"hard": hard[:8], "soft": soft[:8]}
    if hard:
        return {"ok": False, "info": info, "note": "\n".join(hard[:8])}
    if soft:
        logger.info(f"[env gate] passing with {len(soft)} residual compat conflict(s) in "
                    f"peripheral code (tool modules import cleanly): {soft[0][:160]}")
    return {"ok": True, "info": info}


# ══════════════════════════════════════════════════════════════════════════════
# G3 — Coder artefact gate
# ══════════════════════════════════════════════════════════════════════════════
def _coder_gate() -> dict:
    plan = load_plan()
    names = [t.name for t in (plan.tools if plan else [])]
    if not names:
        return {"ok": False, "note": "no planned tools to check"}
    r = check_tool_artefacts(names)
    if r["passed"]:
        return {"ok": True, "info": {"tools": names}}
    note = "\n".join(f"{tool}: {'; '.join(errs)}" for tool, errs in r["errors"].items())
    return {"ok": False, "info": {"errors": r["errors"]}, "note": note}


# ══════════════════════════════════════════════════════════════════════════════
# Validator — deterministic loop, batched debugger (R3/R6)
# ══════════════════════════════════════════════════════════════════════════════
async def _validate(repo_url, name, session_service, metrics):
    started = time.monotonic()
    await emit({"type": "stage", "stage": "validator", "status": "running"})
    plan = load_plan()
    if not plan or not plan.tools:
        logger.error("[validator] no plan/tools — nothing to validate.")
        update_stage_status("validator", status="failed", gate={"note": "no plan"})
        await emit({"type": "stage", "stage": "validator", "status": "failed"})
        return

    v = Validation()
    reports = {t.name: ToolReport(name=t.name) for t in plan.tools}
    v.tools = list(reports.values())
    failures_by_class: dict[str, int] = {}
    n_actions = 0

    frozen: set[str] = set()          # tools whose failure didn't change → give up
    last_sig: dict[str, str] = {}
    for rnd in range(config.DEBUGGING_ROUNDS + 1):
        failures: list[str] = []
        for t in plan.tools:
            rep = reports[t.name]
            if (rnd and rep.passed) or t.name in frozen:
                continue                      # skip green + given-up tools
            fails = await _check_tool(t, rep)
            n_actions += 1 + bool(t.sample_args)
            if fails and rnd:
                sig = fails[0][:200]
                if last_sig.get(t.name) == sig:
                    # a debugger round changed nothing for this tool — stop
                    # burning rounds on it (e.g. a genuinely unfixable dep).
                    frozen.add(t.name)
                    v.notes.append(f"{t.name}: unchanged failure after debug — frozen")
                    continue
                last_sig[t.name] = sig
            elif fails:
                last_sig[t.name] = fails[0][:200]
            for f in fails:
                failures_by_class[classify_error(f)] = failures_by_class.get(classify_error(f), 0) + 1
            failures += fails
            write_validation(repo_url, name, v)   # incremental (R2)
            # The args the tool was actually invoked with: plan keys the
            # generated signature does not accept are dropped, as in _check_tool.
            shown = _clean_sample_args(t) if t.sample_args is not None else {}
            await emit({"type": "validation", "tool": t.name,
                        "passed": rep.passed, "status": rep.status,
                        "exec_ok": rep.exec_ok, "input": shown,
                        "error": (rep.error or None) if not rep.passed else None})
        if not failures:
            break
        if rnd >= config.DEBUGGING_ROUNDS:
            v.notes.append(f"debugger budget exhausted with {len(failures)} failure(s) left")
            break
        v.debugger_rounds += 1
        logger.info(f"[validator] round {rnd + 1}: {len(failures)} failure(s) → batched debugger call.")
        summary = await _call_debugger(
            name, session_service, metrics,
            f"Repository: {repo_url}\n\nThe following tool failures were observed. "
            f"Look for shared root causes first; fix ALL of them, then verify with "
            f"run_tool_tests / invoke_tool_function:\n\n" + "\n\n".join(failures[:12]),
            memory=v.debugger_actions)
        v.debugger_actions.append(summary[:300])

    write_validation(repo_url, name, v)
    c = v.counts()
    update_stage_status("validator", status="passed", counts=c,
                        debugger_rounds=v.debugger_rounds)
    await emit({"type": "stage", "stage": "validator", "status": "done",
                "counts": c})
    logger.info(f"[Validator done] tools passed {c['tools_passed']}/{c['tools_total']} "
                f"(perfect {c['tools_perfect']}); tests {c['tests_passed']}/{c['tests_total']}; "
                f"exec ok {c['exec_ok']}/{c['exec_attempted']}")

    metrics["durations_per_stage"]["validator"] = round(time.monotonic() - started, 1)
    metrics["actions_per_stage"]["validator"] = n_actions
    metrics["total_actions"] += n_actions
    for label, cnt in failures_by_class.items():
        metrics["failures_by_class"][label] = metrics["failures_by_class"].get(label, 0) + cnt


def _clean_sample_args(t: ToolSpec) -> dict:
    """Drop sample-arg keys the tool's generated function doesn't accept —
    otherwise `fn(**args)` TypeErrors on an unexpected kwarg and the debugger
    (which can't rewrite plan sample_args) loops on it. Filter against the REAL
    signature of the written ``tools/<name>.py`` — NOT the repo target's params,
    which task tools deliberately rename (e.g. task ``slide_dir`` → repo
    ``wsi_dir``). Skip filtering when the function takes ``**kwargs`` or its
    signature can't be read."""
    args = dict(t.sample_args or {})
    names, has_kwargs = function_param_names(t.name)
    if not names or has_kwargs:
        return args
    dropped = [k for k in args if k not in names]
    if dropped:
        logger.info(f"[validator] {t.name}: dropping sample args not in the "
                    f"function signature {sorted(names)}: {dropped}")
        for k in dropped:
            args.pop(k)
    return args


async def _check_tool(t: ToolSpec, rep: ToolReport) -> list[str]:
    """Run one tool's exec check + pytest file; update its report; return
    failure descriptions for the batched debugger."""
    fails: list[str] = []

    if t.sample_args is not None:
        args = _clean_sample_args(t)
        r = await invoke_tool_function(t.name, args)
        if r.get("ok"):
            rep.exec_ok = True
            rep.exec_note = r.get("reason", "")        # runtime-success detail (R6)
        else:
            rep.exec_ok = False
            rep.error = (r.get("error") or "")[:300]
            fails.append(f"invoke_tool_function('{t.name}', {json.dumps(args)}) crashed:\n"
                         f"{r.get('error','')}\n{(r.get('traceback') or r.get('stderr') or '')[-1200:]}")
    else:
        rep.exec_ok, rep.exec_note = None, "no cheap real invocation exists"

    tr = await run_tool_tests(t.name)
    if tr.get("error"):
        rep.tests_passed = rep.tests_total = None
        fails.append(f"tests for '{t.name}': {tr['error']}")
    elif tr.get("timeout"):
        rep.tests_passed = rep.tests_total = None
        rep.exec_note = (rep.exec_note + "; " if rep.exec_note else "") + tr["failures"]
    else:
        rep.tests_passed, rep.tests_total = tr["smoke_passed"], tr["smoke_total"]
        rep.invoc_passed, rep.invoc_total = tr["invoc_passed"], tr["invoc_total"]
        if tr["failures"]:
            fails.append(f"pytest tests/test_{t.name}.py has failures:\n{tr['failures'][-1500:]}")
    return fails


async def _call_debugger(name, session_service, metrics, message, memory=None) -> str:
    sid = f"{name}_debugger_{time.monotonic_ns()}"
    await session_service.create_session(app_name=config.APP_NAME, user_id=config.USER_ID,
                                         session_id=sid)
    mem = ("\n\nPrevious fix attempts this run (do NOT repeat them):\n- "
           + "\n- ".join(m[:200] for m in memory)) if memory else ""
    try:
        final, steps, tokens, sm = await asyncio.wait_for(
            run_agent(debugger_agent, session_service, sid, message + mem),
            timeout=config.DEBUGGER_CALL_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(f"[debugger] call timed out after {config.DEBUGGER_CALL_TIMEOUT}s.")
        return "debugger timed out"
    metrics["total_actions"] += steps
    metrics["total_tokens"] += tokens
    metrics["total_cost_usd"] = metrics.get("total_cost_usd", 0.0) + sm.get("cost_usd", 0.0)
    return final


# ══════════════════════════════════════════════════════════════════════════════
# Wrapper — deterministic codegen; LLM only when G4 fails (Q&A)
# ══════════════════════════════════════════════════════════════════════════════
async def _wrap(name, session_service, metrics):
    started = time.monotonic()
    await emit({"type": "stage", "stage": "wrapper", "status": "running"})
    plan = load_plan()
    out = output_dir()
    names = [t.name for t in (plan.tools if plan else [])
             if (out / "tools" / f"{t.name}.py").exists()]
    if not names:
        update_stage_status("wrapper", status="failed", gate={"note": "no tool files exist"})
        logger.error("[wrapper] no tool files to wrap.")
        await emit({"type": "stage", "stage": "wrapper", "status": "failed"})
        return

    # Build the isolated fastmcp server venv (.venv-server) now — the only place
    # that touches server concerns, kept out of the environment stage so the
    # repo's deps and fastmcp's can never conflict (two-venv model).
    server_err = await asyncio.to_thread(_ensure_server_venv, plan)
    if server_err:
        logger.warning(f"[wrapper] server venv setup problem (continuing): {server_err}")

    # The plan's recorded invocation for each tool: the last resort for typing
    # a param the repo function itself leaves un-annotated and un-defaulted.
    sample_args = {t.name: t.sample_args for t in plan.tools if t.sample_args}
    res = write_server(name, names, sample_args=sample_args)
    gate = check_server()
    used_fallback = False
    if not gate["passed"]:
        used_fallback = True
        logger.warning(f"[wrapper] G4 failed — calling fallback wrapper agent: {gate['error'][:200]}")
        sid = f"{name}_wrapper"
        await session_service.create_session(app_name=config.APP_NAME, user_id=config.USER_ID,
                                             session_id=sid)
        try:
            _, steps, tokens, sm = await asyncio.wait_for(
                run_agent(wrapper_agent, session_service, sid,
                          f"The generated server.py fails its compile/import gate.\n"
                          f"Output dir: {out}\nError:\n{gate['error']}"),
                timeout=config.WRAPPER_CALL_TIMEOUT)
            metrics["total_actions"] += steps
            metrics["total_tokens"] += tokens
            metrics["total_cost_usd"] = metrics.get("total_cost_usd", 0.0) + sm.get("cost_usd", 0.0)
        except asyncio.TimeoutError:
            logger.warning("[wrapper] fallback agent timed out.")
        gate = check_server()

    update_stage_status("wrapper", status="passed" if gate["passed"] else "failed",
                        gate={"tools_wrapped": res["tools"], "skipped": res["skipped"],
                              "llm_fallback": used_fallback,
                              "error": "" if gate["passed"] else gate.get("error", "")[:500]})
    metrics["durations_per_stage"]["wrapper"] = round(time.monotonic() - started, 1)
    await emit({"type": "stage", "stage": "wrapper",
                "status": "done" if gate["passed"] else "failed",
                "tools_wrapped": res["tools"], "llm_fallback": used_fallback})
    if gate["passed"]:
        logger.info(f"[wrapper] server.py OK — {len(res['tools'])} tool(s) exposed"
                    + (" (via LLM fallback)" if used_fallback else ""))
    else:
        logger.error(f"[wrapper] server.py still failing: {gate.get('error','')[:300]}")


def _export_task_code(tasks: list[dict]) -> None:
    """TM-Bench export: code.py per task (verbatim self-contained function)."""
    for t in tasks:
        code = render_code_py(t["name"])
        dest = output_dir() / "tmbench" / t["name"] / "code.py"
        if code:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(code, encoding="utf-8")
            logger.info(f"[export] {dest}")
        else:
            logger.warning(f"[export] no function source for task '{t['name']}' — code.py skipped.")


# ══════════════════════════════════════════════════════════════════════════════
# Stage messages built from the plan (exploration report appended, R3)
# ══════════════════════════════════════════════════════════════════════════════
def _exploration_text() -> str:
    p = reports_dir() / "exploration.md"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def _environment_message(repo_url: str, tasks: list[dict]) -> str:
    plan = load_plan()
    lines = [repo_url, "", "Computed environment layout — trust this, it is authoritative:"]
    if plan:
        e = plan.env
        if e.repo_python:
            lines.append(f"  python: {e.repo_python}  (the main venv .venv is built on this)")
        if e.requirements_files:
            lines.append(f"  requirements files: {', '.join(e.requirements_files)}")
        if e.system_libs:
            lines.append(f"  likely system libs: {', '.join(e.system_libs)}")
        if e.weights:
            lines.append(f"  external weights to download: {json.dumps(e.weights)}")
    msg = "\n".join(lines)
    if tasks:
        msg += ("\n\nThis run targets TM-Bench task(s): "
                + ", ".join(t.get("name", "?") for t in tasks)
                + ". The tools for these tasks must work standalone." + _DATA_POLICY)
    return msg + "\n\n--- EXPLORATION REPORT ---\n" + _exploration_text()


def _coder_message(repo_url: str, tasks: list[dict]) -> str:
    plan = load_plan()
    lines = [repo_url]
    if plan and plan.tools:
        lines += ["", "Verified tools to implement (targets confirmed to exist; params are "
                  "the REAL signature; sample_args/evidence guide your tests):"]
        for t in plan.tools:
            lines.append(f"  - {t.name}  ->  {t.target}  params={t.params}")
            lines.append(f"      purpose: {t.purpose}")
            lines.append(f"      sample_args: {json.dumps(t.sample_args)}")
            lines.append(f"      evidence: {t.evidence or '(none — smoke tests only)'}")
    msg = "\n".join(lines) + _tasks_prompt(tasks)
    if tasks:
        msg += ("\n\nFor each required task, the function signature must use EXACTLY the "
                "task's argument names, and the returned dict must contain EXACTLY the "
                "declared return keys.")
    return msg + "\n\n--- EXPLORATION REPORT ---\n" + _exploration_text()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _banner(n: int, label: str) -> None:
    sep = "=" * 60
    logger.info(f"\n{sep}\n  STAGE {n} — {label}\n{sep}")


def _clean_workdir(name: str) -> None:
    d = WORKDIR / name
    if d.exists():
        shutil.rmtree(d)
        logger.debug(f"[clean] removed {d}")


def _new_metrics() -> dict:
    return {"actions_per_stage": {}, "tokens_per_stage": {}, "durations_per_stage": {},
            "tool_calls_per_stage": {}, "guard_retries_per_stage": {},
            "transient_fault_retries_per_stage": {}, "abort_reason_per_stage": {},
            "failures_by_class": {}, "total_actions": 0, "total_tokens": 0,
            "cost_usd_per_stage": {}, "total_cost_usd": 0.0}


def _accumulate_stage(metrics, stage, duration, steps, tokens, sm) -> None:
    """Accumulate across reset attempts (a stage may run several times, R1)."""
    metrics["durations_per_stage"][stage] = metrics["durations_per_stage"].get(stage, 0) + duration
    metrics["actions_per_stage"][stage] = metrics["actions_per_stage"].get(stage, 0) + steps
    metrics["tokens_per_stage"][stage] = metrics["tokens_per_stage"].get(stage, 0) + tokens
    metrics["total_actions"] += steps
    metrics["total_tokens"] += tokens
    cost = sm.get("cost_usd", 0.0) if sm else 0.0
    metrics["cost_usd_per_stage"][stage] = metrics["cost_usd_per_stage"].get(stage, 0.0) + cost
    metrics["total_cost_usd"] += cost
    if sm:
        for key, mkey in [("tool_calls", "tool_calls_per_stage"),
                          ("guard_retries", "guard_retries_per_stage"),
                          ("transient_fault_retries", "transient_fault_retries_per_stage")]:
            val = sm[key]
            if isinstance(val, dict):
                cur = metrics[mkey].setdefault(stage, {})
                for k, c in val.items():
                    cur[k] = cur.get(k, 0) + c
            else:
                metrics[mkey][stage] = metrics[mkey].get(stage, 0) + val
        if sm["abort_reason"]:
            metrics["abort_reason_per_stage"][stage] = sm["abort_reason"]
        for label, c in sm["failures_by_class"].items():
            metrics["failures_by_class"][label] = metrics["failures_by_class"].get(label, 0) + c


def _finalize_metrics(base: Path, metrics: dict) -> None:
    import sys as _sys, traceback as _tb
    exc_type, exc_val, exc_tb = _sys.exc_info()
    d = base / "reports"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if exc_val is not None:
        (d / "error.json").write_text(json.dumps({
            "exception": type(exc_val).__name__, "message": str(exc_val),
            "traceback": "".join(_tb.format_exception(exc_type, exc_val, exc_tb)),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.error(f"[pipeline] error saved → {d}/error.json")


def _report_completion(name: str, base: Path) -> None:
    sep = "=" * 60
    if (base / "reports" / "validation.json").exists():
        logger.success(f"\n{sep}\n  Pipeline complete: {name}\n  Run data: {base}/reports/\n{sep}")
    else:
        logger.error(f"\n{sep}\n  Pipeline incomplete: {name} — no validation.json written.\n{sep}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python -m alembic.main <repo_url> [--resume <stage>] "
                     "[--until <stage>] [--tasks <spec>] [--hints <text>]")
        logger.error(f"       stages: {', '.join(STAGES)}")
        sys.exit(1)

    def _arg(flag: str) -> str | None:
        if flag not in sys.argv:
            return None
        i = sys.argv.index(flag)
        if i + 1 >= len(sys.argv):
            logger.error(f"{flag} requires a value")
            sys.exit(1)
        return sys.argv[i + 1]

    # Bridge live events out of the (isolated) build container: when
    # ALEMBIC_EMIT_STDOUT is set (the container entrypoint sets it), stream every
    # pipeline/stage/validation event as a single ``ALEMBIC_EVENT <json>`` line on
    # stdout. That stdout is captured by start_chain -> the host build log, which
    # the CoScientist web UI tails to render a live build page. No sink => no-op,
    # so the plain CLI and the benchmark runner are unaffected.
    if os.environ.get("ALEMBIC_EMIT_STDOUT"):
        async def _stdout_sink(msg: dict) -> None:
            print("ALEMBIC_EVENT " + json.dumps(events.safe(msg), ensure_ascii=False),
                  flush=True)
        events.set_sink(_stdout_sink)

    try:
        asyncio.run(run_pipeline(sys.argv[1], resume_from=_arg("--resume"),
                                 stop_after=_arg("--until"),
                                 tasks_cli=_arg("--tasks") or _arg("--target-task"),
                                 hints_cli=_arg("--hints")))
    except Exception:
        logger.exception("Pipeline error:")
        raise
