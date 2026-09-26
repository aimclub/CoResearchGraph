"""Fail-closed plan/result review agents."""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from typing import Any, AsyncGenerator, Literal

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types

from CoScientist.config import get_settings
from CoScientist.experiments.critique import (
    PlanValidationError,
    json_validation_errors,
    validate_and_critique_plan,
)
from CoScientist.agents.callbacks.report_language import session_report_language
from CoScientist.experiments.plan_view import plan_to_view
from CoScientist.experiments.runtime import (
    approve_plan,
    approve_plan_with_human_override,
    initialize_runtime,
    mark_result_review,
)
from CoScientist.experiments.runtime.execution_bridge import (
    close_plan_record,
    record_plan_proposed,
)
from CoScientist.experiments.runtime.shared import audit
from CoScientist.experiments.runtime.state_machine import (
    REPLAN_ROUNDS_KEY,
    fedot_route_available,
    medical_route_available,
    session_route_agents,
)
from CoScientist.experiments.schemas import (
    CodeRequirement,
    ExecutionRoute,
    ExperimentPlan,
    PlanCritique,
)
from CoScientist.graph.session_scope import session_key
from CoScientist.hitl.handler import AbstractHITLHandler, DelegatingHITLHandler
from CoScientist.hitl.models import (
    HITLAction,
    HITLDecisionSource,
    HITLRequest,
    HITLResponse,
)
from CoScientist.hitl.resolver import resolve_auto, resolve_timeout
from CoScientist.hitl.session_agent import SessionAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_AUTO_APPROVE_TRUTHY = frozenset({"1", "true", "yes", "on"})
_OK_TASK_STATUSES = frozenset({"done", "done_with_warnings", "skipped", "success", "completed", "partial", "partial_success"})
_EVIDENCE_ROUTES = frozenset({"research", "medical"})


def _publish_approved_plan_to_graph(ctx: InvocationContext, state: Any) -> None:
    """Best-effort: mirror the approved plan into the research graph — the
    method per task (VerificationMethod + Hypothesis —tested_by→ VM), and the
    task itself as the plan wrote it (ExperimentTask —elaborates→ PlanStep).
    Two commits, because they are two different claims about the study: what
    was intended, and by what means it will be tested. A graph failure must
    never break the approve itself."""
    try:
        from CoScientist.experiments.runtime.graph_bridge import (
            publish_plan_detail_to_graph,
            publish_plan_to_graph,
        )
        from CoScientist.graph.research.store import get_research_graph

        graph = get_research_graph(ctx)
        publish_plan_to_graph(graph, state)
        publish_plan_detail_to_graph(graph, state)
    except Exception as exc:  # noqa: BLE001 — approve wins over graph mirroring
        audit(logger, f"EXPERIMENT_GRAPH_PLAN_PUBLISH_FAILED error={exc}",
              level=logging.WARNING)


def result_tasks_ok(runtime: dict[str, Any] | None) -> bool:
    """Compute tasks must succeed. Failed literature/medical is ok if unused as input."""
    if not isinstance(runtime, dict):
        return False
    if runtime.get("tasks_ok") is True:
        return True
    raw_tasks = runtime.get("tasks") or {}
    if isinstance(raw_tasks, list):
        tasks = {str(t.get("id") or i): t for i, t in enumerate(raw_tasks) if isinstance(t, dict)}
    elif isinstance(raw_tasks, dict):
        tasks = raw_tasks
    else:
        tasks = {}

    rows = [row for row in tasks.values() if isinstance(row, dict)]
    if not rows:
        return False
    consumers: set[str] = set()
    for row in rows:
        dump = row.get("task") if isinstance(row.get("task"), dict) else {}
        for item in dump.get("input_data") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("kind") or "") != "task_artifact":
                continue
            src = str(item.get("source_task_id") or "").strip()
            if src:
                consumers.add(src)
    for task_id, row in tasks.items():
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        if status in _OK_TASK_STATUSES:
            continue
        route = str(row.get("planned_route") or (row.get("task") or {}).get("route") or "")
        tid = str((row.get("task") or {}).get("id") or task_id)
        if route in _EVIDENCE_ROUTES and tid not in consumers:
            continue
        return False
    return True


_audit = functools.partial(audit, logger)


def _window_word(window: float | None) -> str:
    """The window an audit line reports, when there may not be one.

    `None` is a real outcome now — it is how the operator's "wait for me" is
    passed down — and a handler can still answer `timed_out` under it: the
    fail-closed handler does exactly that when no interactive reviewer is
    connected. So the line has to be able to say there was no deadline.
    """
    return f"{window:g}" if window else "none"


class FailClosedExperimentHITLHandler(AbstractHITLHandler):
    """Pause review when no interactive reviewer is connected."""

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        return resolve_timeout(reason="no_interactive_reviewer")


def fail_closed_handler() -> DelegatingHITLHandler:
    """Delegating handler so Web runtime can attach its UI handler."""
    return DelegatingHITLHandler(FailClosedExperimentHITLHandler())


def _headless_auto_approve() -> bool:
    """The one-switch override: both reviews, from the environment.

    Kept as the headless entry point (scripts/test_lanes.py, smoke runs) —
    one variable, no UI, both kinds.
    """
    return os.getenv("COSCIENTIST_EXPERIMENT_HITL_AUTO_APPROVE", "").strip().lower() in _AUTO_APPROVE_TRUTHY


def _auto_approve(kind: str) -> bool:
    """Whether the single global HITL mode approves this review."""
    try:
        from CoScientist.hitl.mode import auto_approves
        return auto_approves()
    except Exception:  # noqa: BLE001 — an unreadable mode still asks the human
        return False


def _approval_mode() -> str:
    """For the audit line: the global mode made the decision."""
    return "mode_auto"


def _auto_approve_response() -> HITLResponse:
    return resolve_auto(HITLRequest(
        agent_name="ExperimentReview",
        action_type=HITLAction.APPROVE,
        message="Automatic experiment review",
    ))


def _context_invariant_errors(plan: ExperimentPlan, context: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if (rid := context.get("experiment_run_id")) and plan.experiment_run_id != rid:
        errors.append({
            "type": "context_invariant", "loc": ["experiment_run_id"], "input": plan.experiment_run_id,
            "msg": f"experiment_run_id must equal experiment_context.experiment_run_id ({rid!r})",
        })
    if (req := context.get("source_request")) and plan.source_request != req:
        errors.append({
            "type": "context_invariant", "loc": ["source_request"], "input": plan.source_request,
            "msg": "source_request must equal experiment_context.source_request",
        })
    return errors


def _json_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if not isinstance(value, str):
        return value
    from CoScientist.experiments.runtime.shared import parse_fenced_json

    return parse_fenced_json(value)


def _plan_digest(plan: ExperimentPlan | dict[str, Any]) -> str:
    payload = plan.model_dump(mode="json") if isinstance(plan, ExperimentPlan) else plan
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _execution_blockers(critique: Any) -> list[dict[str, Any]]:
    """Issues an operator cannot turn into an executable route by consent."""
    issues = getattr(critique, "issues", None) or []
    blocked: list[dict[str, Any]] = []
    for issue in issues:
        if not getattr(issue, "is_blocking", False):
            continue
        # Every blocker is hard. Major completeness/consistency/relevance and
        # complexity findings describe scientific scope and may be accepted as
        # an explicitly partial plan; feasibility/security/schema findings do
        # not create a missing tool, route or permission when accepted.
        if issue.severity == "blocker" or issue.category in {
            "feasibility", "security", "schema"
        }:
            blocked.append(issue.model_dump(mode="json"))
    return blocked


def _stamp_context_invariants(
    payload: Any, context: dict[str, Any], previous: ExperimentPlan | None = None,
) -> Any:
    """Authoritative context wins for run-id / source_request / plan identity.

    ``plan_id`` and ``revision`` are runtime bookkeeping, not planner output. The
    model has no reliable way to recall the previous plan_id across a revision
    round, and a wrong guess is a BLOCKING critique ("plan_id changed between
    revisions" / "A revised plan must increment revision") that costs one round
    of a budget only a few rounds deep. Observed 2026-09-04: a human HITL edit
    re-entered planning, the regenerated plan came back as
    ``PLAN-EXRUN-<uuid>`` where the runtime held ``PLAN-<uuid>``, and the whole
    experiment stopped without a single MCP call.

    The two validator checks stay where they are: stamping here makes them
    unreachable for model formatting while they still catch real programming
    errors, and it keeps the "validator repairs nothing" contract intact.
    """
    if not isinstance(payload, dict):
        return payload
    stamped = dict(payload)
    # Plan identity comes from the runtime, not from context, so it is stamped
    # even when experiment_context is empty - otherwise the two blocking checks
    # this defuses would quietly come back in exactly that corner.
    if previous is not None:
        stamped["plan_id"] = previous.plan_id
        stamped["revision"] = previous.revision + 1
    if not context:
        return stamped
    if run_id := context.get("experiment_run_id"):
        stamped["experiment_run_id"] = run_id
    if request := context.get("source_request"):
        stamped["source_request"] = request
    return stamped


def _esc(text: str, n: int | None = None) -> str:
    out = text.replace("|", "/")
    return out[:n] if n is not None else out


def _design_cell(value: Any, n: int | None = None) -> str:
    from CoScientist.experiments.schemas import is_design_placeholder
    if is_design_placeholder(value):
        return "—"
    return _esc(str(value).replace("\n", " "), n)


# The frame around the planner's own words. The goal, the questions and the
# task names arrive in the session's language; these labels used to be English
# regardless, so a Russian study opened its plan on "# Experiment plan".
_PLAN_WORDS = {
    "en": {
        "title": "Experiment plan", "revision": "revision", "goal": "Goal",
        "hypothesis_summary": "Hypothesis summary", "unspecified": "not specified",
        "methods": "Methods", "duration": "Total duration", "min": "min",
        "hypotheses": "Hypotheses",
        "matrix": "Design matrix (hypothesis → experiment → data → baseline → metrics)",
        "col": ("Task", "Hypothesis", "Question", "Dataset", "Baselines", "Metrics",
                "Tools", "Analysis artifacts", "Route"),
        "route": "Route", "repo": "Repo URL", "post_build": "Post-build route",
        "code_assessment": "Code assessment", "entrypoints": "entrypoints",
        "hypothesis": "Hypothesis", "question": "Question", "dataset": "Dataset",
        "baselines": "Baselines", "metrics": "Metrics",
        "analysis": "Analysis artifacts", "task": "Task", "rationale": "Rationale",
        "tools": "MCP/tools", "params": "Launch params", "inputs": "Inputs",
        "criteria": "Success criteria", "expected": "Expected artifacts",
        "task_duration": "Duration", "warnings": "Warnings", "none": "none",
        "risks": "Risks", "operation": "research task",
    },
    "ru": {
        "title": "План эксперимента", "revision": "ревизия", "goal": "Цель",
        "hypothesis_summary": "Проверяемая гипотеза", "unspecified": "не задана",
        "methods": "Методы", "duration": "Общая оценка", "min": "мин",
        "hypotheses": "Гипотезы",
        "matrix": "Матрица плана (гипотеза → эксперимент → данные → базлайн → метрики)",
        "col": ("Задача", "Гипотеза", "Вопрос", "Данные", "Базлайны", "Метрики",
                "Инструменты", "Анализ", "Маршрут"),
        "route": "Маршрут", "repo": "Репозиторий", "post_build": "Маршрут после сборки",
        "code_assessment": "Оценка кода", "entrypoints": "точки входа",
        "hypothesis": "Гипотеза", "question": "Вопрос", "dataset": "Данные",
        "baselines": "Базлайны", "metrics": "Метрики",
        "analysis": "Артефакты анализа", "task": "Что выполняется", "rationale": "Зачем",
        "tools": "MCP / инструменты", "params": "Параметры запуска",
        "inputs": "Входные данные", "criteria": "Критерии успеха",
        "expected": "Ожидаемые артефакты", "task_duration": "Длительность",
        "warnings": "Предупреждения", "none": "нет", "risks": "Риски",
        "operation": "задача исследования",
    },
}


def _plan_words(lang) -> dict:
    from CoScientist.agents.callbacks.report_language import normalize_report_language

    return _PLAN_WORDS[normalize_report_language(lang)]


def render_experiment_plan(plan: ExperimentPlan, lang: str = "en") -> str:
    w = _plan_words(lang)
    L = [
        f"# {w['title']} · {w['revision']} {plan.revision}", f"{w['goal']}: {plan.goal}",
        f"{w['hypothesis_summary']}: {plan.hypothesis or w['unspecified']}",
        f"{w['methods']}: {', '.join(plan.methods)}",
        f"{w['duration']}: {plan.total_est_duration_min} {w['min']}",
    ]
    if plan.hypotheses:
        L += ["", f"## {w['hypotheses']}"] + [f"- `{h.hypothesis_id}`: {h.statement}" for h in plan.hypotheses]
    L += [
        "", f"## {w['matrix']}",
        "| " + " | ".join(w["col"]) + " |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in plan.tasks:
        d = t.design
        bl = "; ".join(f"{b.name} ({b.kind})" for b in d.baselines) if d.baselines else ""
        mt = "; ".join(f"{m.name}/{m.direction}" + (f" [{m.test}]" if m.test else "") for m in d.metrics) if d.metrics else ""
        ar = "; ".join(f"{a.name} ({a.role}/{a.prepare_via})" for a in d.analysis_artifacts) if d.analysis_artifacts else ""
        tools_summary = "; ".join(
            f"{s.name}:{','.join(x.name for x in s.tools)}" for s in t.mcp_servers
        ) if t.mcp_servers else ""
        L.append(
            f"| {t.id} | `{d.hypothesis_ref}` | {_design_cell(d.experiment_question, 120)} "
            f"| {_design_cell(d.dataset.name)} | {_design_cell(bl, 100)} "
            f"| {_design_cell(mt, 100)} | {_design_cell(tools_summary, 80)} "
            f"| {_design_cell(ar, 100)} | `{t.route.value}` |"
        )
    for t in plan.tasks:
        d = t.design
        tools = [
            f"{s.name} ({s.url}): {', '.join(x.name for x in s.tools)}"
            if s.url else f"{s.name}: {', '.join(x.name for x in s.tools)}"
            for s in t.mcp_servers
        ]
        criteria_parts = []
        for c in t.success_criteria:
            crit_text = f"{c.criterion_id}: {c.description}"
            if c.metric and c.operator is not None and c.target is not None:
                crit_text += f" [{c.metric} {c.operator} {c.target}]"
            criteria_parts.append(crit_text)
        criteria = "; ".join(criteria_parts)

        arts = "; ".join(
            f"{a.name} ({a.role})" if not a.description or a.description == a.name
            else f"{a.name} ({a.role}: {a.description})"
            for a in t.expected_artifacts
        )
        inputs_list = []
        for inp in t.input_data:
            loc = inp.url or inp.workspace_path or inp.s3_key or ""
            if loc:
                inputs_list.append(f"{inp.data_id} [{inp.kind}: {loc}]")
            else:
                inputs_list.append(f"{inp.data_id} [{inp.kind}]")
        inputs_str = "; ".join(inputs_list) if inputs_list else w["none"]

        also = f" (+{', '.join(d.also_tests)})" if d.also_tests else ""
        # Named, not coded. The operator reads this card beside the research
        # frame, and «OP-1» told them nothing there either.
        op_no = str(d.operation_ref or "").rsplit("-", 1)[-1]
        op_str = (f" [{w['operation']} {op_no} · `{d.operation_ref}`]"
                  if d.operation_ref else "")
        notes = f" — {d.dataset.notes}" if d.dataset.notes else ""
        L += ["", f"## {t.id} · {t.name}", f"{w['route']}: `{t.route.value}`"]
        if t.code_assessment.requirement != CodeRequirement.UNKNOWN or t.repo_url:
            assessment = t.code_assessment
            detail = assessment.evidence or w["unspecified"]
            entrypoints = ", ".join(f"`{item}`" for item in assessment.entrypoints)
            if entrypoints:
                detail += f"; {w['entrypoints']}: {entrypoints}"
            L.append(
                f"{w['code_assessment']}: `{assessment.requirement.value}` — {detail}"
            )
        if t.route.value == "alembic_build":
            L += [f"{w['repo']}: {t.repo_url}", f"{w['post_build']}: `{t.post_build_route}`"]
        L += [
            f"{w['hypothesis']}: `{d.hypothesis_ref}`{also}{op_str}",
            f"{w['question']}: {_design_cell(d.experiment_question)}",
            f"{w['dataset']}: {_design_cell(d.dataset.name)}{notes if d.dataset.name else ''}",
            f"{w['baselines']}: {_design_cell('; '.join(f'{b.name} ({b.kind})' for b in d.baselines))}",
            f"{w['metrics']}: {_design_cell('; '.join(f'{m.name} ({m.direction})' for m in d.metrics))}",
            f"{w['analysis']}: {_design_cell('; '.join(f'{a.name} [{a.role}]' for a in d.analysis_artifacts))}",
            f"{w['task']}: {t.description}",
        ]
        if t.rationale and t.rationale != t.description:
            L.append(f"{w['rationale']}: {t.rationale}")
        L.append(f"{w['tools']}: {'; '.join(tools) if tools else w['none']}")
        if t.launch_params:
            params_str = ", ".join(f"{k}={v}" for k, v in t.launch_params.items())
            L.append(f"{w['params']}: {params_str}")
        L += [
            f"{w['inputs']}: {inputs_str}",
            f"{w['criteria']}: {criteria}",
            f"{w['expected']}: {arts}",
            f"{w['task_duration']}: {t.est_duration_min} {w['min']}",
            f"{w['warnings']}: {'; '.join(t.warnings) if t.warnings else w['none']}",
        ]
    if plan.risks:
        L += ["", f"## {w['risks']}"] + [f"- {r}" for r in plan.risks]
    return "\n".join(L)


def _artifact_canonical_location(a: dict[str, Any]) -> str:
    """Prefer real http(s) URL, then s3://bucket/key, then workspace path."""
    url = str(a.get("external_url") or "").strip()
    if url.startswith(("http://", "https://")):
        return url
    bucket, key = a.get("bucket"), a.get("s3_key")
    if bucket and key:
        return f"s3://{bucket}/{key}"
    if wp := a.get("workspace_path"):
        return str(wp)
    if url:
        return url
    return "(location missing)"


def build_experiment_artifacts_manifest(state: Any) -> list[dict[str, str]]:
    """Flat list of real ArtifactRef locations for prompts / reports."""
    rows: list[dict[str, str]] = []
    for r in state.get("experiment_task_results") or []:
        if not isinstance(r, dict):
            continue
        tid = str(r.get("task_id") or "")
        for a in r.get("artifacts") or []:
            if not isinstance(a, dict):
                continue
            rows.append({
                "task_id": tid,
                "artifact_id": str(a.get("artifact_id") or ""),
                "name": str(a.get("name") or ""),
                "location": _artifact_canonical_location(a),
                "media_type": str(a.get("media_type") or ""),
            })
    return rows


_RESULT_WORDS = {
    "en": {
        "title": "Experiment results", "count": "Task results",
        "locations": "Canonical artifact locations (do not invent URLs)",
        "none": "(none captured)", "route": "Route", "artifact": "Artifact",
        "summary": "Summary",
    },
    "ru": {
        "title": "Результаты эксперимента", "count": "Результатов задач",
        "locations": "Канонические адреса артефактов (не придумывать URL)",
        "none": "(ничего не собрано)", "route": "Маршрут", "artifact": "Артефакт",
        "summary": "Итог",
    },
}


def _openable(state: Any, location: str) -> str:
    """A URL a reader can press, or "" when this session does not hold the file.

    Read-only on the state: the scope keys are taken directly rather than
    through `session_key`, which WRITES the resolved pair back and must not run
    from a renderer.
    """
    text = str(location or "").strip()
    if not text:
        return ""
    try:
        from CoScientist.graph.session_scope import (
            GRAPH_SCOPE_SESSION_KEY,
            GRAPH_SCOPE_USER_KEY,
        )
        from CoScientist.utils.report_links import resolve_ref

        user = str((state or {}).get(GRAPH_SCOPE_USER_KEY) or "")
        session = str((state or {}).get(GRAPH_SCOPE_SESSION_KEY) or "")
        if not (user and session):
            return ""
        scope = (user, session)
        from CoScientist.reporting import session_files

        mirrored = session_files.artifact_id_for_url(scope, text)
        if mirrored:
            return resolve_ref(f"cos-artifact:{mirrored}", scope) or ""
        return resolve_ref(text, scope) or ""
    except Exception:  # noqa: BLE001 — a link is not worth a failed render
        return ""


def _located(state: Any, label: str, location: str) -> str:
    """The canonical address, and a link beside it when there is one.

    Beside, never instead. This section is headed "do not invent URLs" and the
    backticked address is what stops a model doing exactly that — it is the
    string the runtime will accept back. The link is for the human reading the
    same page, who otherwise has to copy an S3 key by hand.
    """
    address = f"`{location}`"
    href = _openable(state, location)
    return f"{address} → [{label}]({href})" if href else address


def render_experiment_results(state: Any) -> str:
    """The run's results. Reads its own language: it already has the state."""
    from CoScientist.agents.callbacks.report_language import normalize_report_language

    w = _RESULT_WORDS[normalize_report_language(session_report_language(state))]
    results = state.get("experiment_task_results") or []
    manifest = build_experiment_artifacts_manifest(state)
    if isinstance(state, dict):
        state["experiment_artifacts_manifest"] = manifest
    L = [
        f"# {w['title']}",
        f"{w['count']}: {len(results)}",
        "",
        f"## {w['locations']}",
    ]
    if manifest:
        for m in manifest:
            L.append(
                f"- `{m['task_id']}` / `{m['name']}` (`{m['artifact_id']}`): "
                + _located(state, m["name"] or m["artifact_id"], m["location"])
            )
    else:
        L.append(f"- {w['none']}")
    for r in results:
        L += [
            "",
            f"## {r.get('task_id')} · {r.get('status')}",
            str(r.get("summary") or ""),
            f"{w['route']}: `{r.get('route_used')}`",
        ]
        for a in r.get("artifacts") or []:
            if not isinstance(a, dict):
                continue
            L.append(
                f"- {w['artifact']} `{a.get('artifact_id')}` ({a.get('name')}): "
                + _located(state, str(a.get("name") or a.get("artifact_id") or ""),
                           _artifact_canonical_location(a))
            )
    if summary := state.get("experiment_summary"):
        L += ["", f"## {w['summary']}", str(summary)]
    return "\n".join(L)


def _is_refusal(response: HITLResponse) -> bool:
    """A human saying NO — as opposed to asking for a revision, or not answering.

    The UI keeps these apart already: «Доработать» sends `edit`, «Отклонить»
    sends `reject`. Downstream they were the same, because neither set
    `stop_review_loop` and `SessionAgent` feeds anything unapproved back to the
    author. A reject WITH a note is still a reject: the note is the reason, not
    a request to try again.

    A timeout is not a refusal by the operator even though it now arrives as
    one — nobody decided anything — so it keeps its own branch and its own
    pause reason.
    """
    if response.approved or response.timed_out or response.stop_review_loop:
        return False
    return response.action == HITLAction.REJECT


def _plan_outcome(response: HITLResponse) -> str:
    """How the human left this round of the plan, in the record's vocabulary."""
    if response.approved:
        return "approved"
    if response.timed_out or response.stop_review_loop:
        return "paused"
    if response.action == HITLAction.EDIT:
        return "revision_requested"
    return "rejected"


def _clip(text: Any, limit: int = 400) -> str | None:
    out = " ".join(str(text or "").split())
    if not out:
        return None
    return out if len(out) <= limit else out[: limit - 1] + "…"


#: Why the last review did not end in an approval: "plan_review_timeout",
#: "result_review_timeout", "max_plan_revisions" or
#: "inventory_blocker_repeated". Read by the orchestrator's suppressor to say
#: so in the final report, and by the planner context builder.
PAUSE_REASON_STATE_KEY = "experiment_review_pause_reason"
ROUTE_SELECTIONS_STATE_KEY = "experiment_route_selections"
_ROUTE_CODER_OPTION = "Coder — execute repository code directly"
_ROUTE_ALEMBIC_OPTION = "Alembic — wrap the unchanged entrypoint as MCP"


def _alembic_preflight() -> dict[str, Any]:
    """Lazy import keeps normal experiment review independent of Alembic."""
    try:
        from CoScientist.tools.alembic_tools import alembic_preflight

        return alembic_preflight()
    except Exception as exc:  # noqa: BLE001 — an unavailable builder means Coder
        return {
            "available": False,
            "reason": f"Alembic preflight could not run: {type(exc).__name__}: {exc}",
        }


# State this reviewer owns and must hand back to whoever invoked the module.
#
# The module runs as an ADK AgentTool, and AgentTool gives it a FRESH in-memory
# session seeded from a copy of the caller's state; the only thing that travels
# back is ``event.actions.state_delta`` (agent_tool.py: "Forward state delta to
# parent session"). This agent, like every SessionAgent, writes through
# ``ctx.session.state`` — a plain dict, not the ADK ``State`` wrapper — so its
# writes mutate the throwaway child session and are dropped when the tool
# returns. Callbacks are unaffected: ``callback_context.state`` IS a ``State``,
# so the planner context builder's writes DO travel back.
#
# That asymmetry is the re-planning loop. The builder's "new run" wipe
# (experiment_runtime = None) reached the caller; the approval that followed it
# — approve_plan, then mark_result_review's phase=completed — did not. The next
# module hop was therefore seeded with experiment_runtime=None, the completion
# gate had nothing to match on, and the whole experiment was planned and run
# again. Measured 2026-09-02: two full re-runs of the same three tasks in one
# 41-minute run, 19:50:46 phase=completed -> 19:50:59 gate sees NoneType.
_REVIEW_OWNED_STATE_KEYS = (
    "experiment_runtime",
    "experiment_plan",
    "experiment_plan_view",
    "experiment_plan_critique",
    "experiment_plan_validation_errors",
    "experiment_plan_candidate",
    "experiment_plan_fallback_pending",
    "experiment_module_outcome",
    "experiment_plan_review_paused",
    "experiment_plan_record_id",
    ROUTE_SELECTIONS_STATE_KEY,
    PAUSE_REASON_STATE_KEY,
    "experiment_plan_revision_count",
    "experiment_inventory_blocker_hits",
    "experiment_artifacts_manifest",
    "experiment_task_results",
    "experiment_summary",
    REPLAN_ROUNDS_KEY,
)


class ExperimentReviewSessionAgent(SessionAgent):
    """LLM plan/summary stage with deterministic validation and mandatory HITL."""

    review_kind: Literal["plan", "result"]
    max_inventory_blocker_hits: int = 2  # same inventory-absence blocker twice → pause

    def __init__(self, **data: Any):
        if data.get("hitl_handler") is None:
            data["hitl_handler"] = fail_closed_handler()
        super().__init__(**data)
        self._state_publish_pending = False

    def _should_run_review(self) -> bool:
        # Deterministic schema/critique + initialize_runtime live in
        # ``_review_plan``. They must run even when the global HITL switch is
        # off; headless auto-approve then skips the human console.
        return self.hitl_handler is not None

    def _review_output(self, output_text: Any) -> str:
        if self.review_kind != "plan":
            return str(output_text)
        try:
            return render_experiment_plan(ExperimentPlan.model_validate(_json_payload(output_text)))
        except Exception:
            return str(output_text)

    async def _revise(
        self, *, ctx: InvocationContext, detail: Any, pause_prefix: str, edit_prefix: str,
        inventory_blocker: bool = False,
        plan: ExperimentPlan | None = None,
        critique: Any = None,
        context: dict[str, Any] | None = None,
        previous: ExperimentPlan | None = None,
        route_agents: set[str] | None = None,
        **_kwargs: Any,
    ) -> HITLResponse:
        state = ctx.session.state
        try:
            revisions = int(state.get("experiment_plan_revision_count") or 0) + 1
        except (TypeError, ValueError):
            revisions = 1
        state["experiment_plan_revision_count"] = revisions

        try:
            hits = int(state.get("experiment_inventory_blocker_hits") or 0) + (1 if inventory_blocker else 0)
        except (TypeError, ValueError):
            hits = 1 if inventory_blocker else 0
        state["experiment_inventory_blocker_hits"] = hits

        candidate: dict[str, Any] | None = None
        recovered_previous = False
        if plan is not None and critique is not None:
            blockers = _execution_blockers(critique)
            candidate = {
                "status": "blocked" if blockers else "needs_revision",
                "reason": "deterministic_critique",
                "digest": _plan_digest(plan),
                "plan": plan.model_dump(mode="json"),
                "critique": critique.model_dump(mode="json"),
                "readiness": {
                    "executable": not blockers,
                    "execution_blockers": blockers,
                },
            }
            # Save every schema-valid version, not just the final attempt. If a
            # later model turn breaks JSON, the operator can still see and
            # deliberately select this last executable version.
            state["experiment_plan_candidate"] = candidate
        elif isinstance(state.get("experiment_plan_candidate"), dict):
            saved = state["experiment_plan_candidate"]
            readiness = saved.get("readiness") or {}
            try:
                saved_plan = ExperimentPlan.model_validate(saved.get("plan"))
                saved_critique = PlanCritique.model_validate(saved.get("critique"))
            except (ValueError, TypeError):
                saved_plan = None
                saved_critique = None
            expected_run = str((context or {}).get("experiment_run_id") or "")
            if (
                saved_plan is not None
                and saved_critique is not None
                and readiness.get("executable")
                and not readiness.get("execution_blockers")
                and (not expected_run or saved_plan.experiment_run_id == expected_run)
            ):
                plan, critique = saved_plan, saved_critique
                candidate = saved
                recovered_previous = True

        max_rev = get_settings().experiments.max_plan_revisions
        if not (
            revisions >= max_rev
            or hits >= self.max_inventory_blocker_hits
        ):
            return HITLResponse(action=HITLAction.EDIT, approved=False, instructions=f"{edit_prefix} {detail}")
        reason = (
            "inventory_blocker_repeated"
            if hits >= self.max_inventory_blocker_hits
            else "max_plan_revisions"
        )

        # A schema-valid current plan with only scope/quality findings is still
        # useful. Stop spending model turns and ask the operator whether to run
        # exactly this version with its visible issues.
        if plan is not None and critique is not None and candidate is not None:
            blockers = list((candidate.get("readiness") or {}).get("execution_blockers") or [])
            candidate["status"] = "blocked" if blockers else "awaiting_human"
            candidate["reason"] = reason
            state["experiment_plan_candidate"] = candidate
            if not blockers:
                return await self._review_exhausted_candidate(
                    ctx=ctx,
                    plan=plan,
                    critique=critique,
                    reason=reason,
                    context=context or {},
                    previous=previous,
                    route_agents=route_agents or set(),
                    recovered_previous=recovered_previous,
                )

        state["experiment_plan_review_paused"] = True
        state["experiment_plan_fallback_pending"] = False
        state[PAUSE_REASON_STATE_KEY] = reason
        state["experiment_module_outcome"] = {
            "status": "blocked",
            "stage": "plan_review",
            "reason": reason,
        }
        _audit(f"EXPERIMENT_PLAN_REVIEW_PAUSED reason={reason}")
        return HITLResponse(
            action=HITLAction.REJECT, approved=False, stop_review_loop=True,
            instructions=f"{pause_prefix} {detail}",
        )

    async def _review_exhausted_candidate(
        self,
        *,
        ctx: InvocationContext,
        plan: ExperimentPlan,
        critique: Any,
        reason: str,
        context: dict[str, Any],
        previous: ExperimentPlan | None,
        route_agents: set[str],
        recovered_previous: bool = False,
    ) -> HITLResponse:
        """Put the current executable draft before a human exactly once."""
        state, cfg = ctx.session.state, get_settings().experiments
        user_id, session_id = session_key(ctx)
        digest = _plan_digest(plan)
        review_id = (
            f"plan-fallback:{plan.experiment_run_id}:{plan.plan_id}:"
            f"r{plan.revision}:{digest}"
        )
        critique_json = critique.model_dump(mode="json")
        accepted_issue_ids = [
            issue.issue_id
            for issue in critique.issues
            if issue.is_blocking
        ]

        state["experiment_plan_review_paused"] = False
        state["experiment_plan_fallback_pending"] = True
        state[PAUSE_REASON_STATE_KEY] = "plan_revision_budget_exhausted_pending_human"
        state["experiment_module_outcome"] = {
            "status": "awaiting_human",
            "stage": "plan_review",
            "reason": reason,
            "plan_id": plan.plan_id,
            "revision": plan.revision,
            "digest": digest,
            "review_id": review_id,
        }
        candidate = state.get("experiment_plan_candidate") or {}
        candidate["review_id"] = review_id
        state["experiment_plan_candidate"] = candidate
        runtime = initialize_runtime(state, plan, critique=critique_json)
        view = plan_to_view(plan, critique_json, status="awaiting_human")
        view["review_exhausted"] = True
        view["review_exhausted_reason"] = reason
        view["recovered_previous_candidate"] = recovered_previous
        if recovered_previous:
            view["superseded_validation_errors"] = list(
                state.get("experiment_plan_validation_errors") or []
            )
        state["experiment_plan_view"] = view
        record_id = record_plan_proposed(ctx, self.name, view)
        state["experiment_plan_record_id"] = record_id

        lang = session_report_language(state)
        response = await self.hitl_handler.handle_request(self._hitl(
            message=(
                "Automatic plan revisions are exhausted. Review the last schema-valid "
                "executable plan and either approve it with the listed issues or reject it."
                if recovered_previous else
                "Automatic plan revisions are exhausted. Review the current executable "
                "plan and either approve it with the listed issues or reject it."
            ),
            kind="plan",
            plan_id=plan.plan_id,
            output=render_experiment_plan(plan, lang),
            user_id=user_id,
            session_id=session_id,
            timeout_seconds=cfg.plan_review_timeout_s,
            plan_view=view,
            requires_human=True,
            review_exhausted=True,
            review_id=review_id,
        ))

        source = getattr(response.decision_source, "value", response.decision_source)
        if response.approved and source == HITLDecisionSource.HUMAN.value:
            # Re-evaluate the exact candidate after the wait. Settings and the
            # attached route tree may have changed while the card was open.
            try:
                checked_plan, checked_critique = validate_and_critique_plan(
                    plan.model_dump(mode="json"),
                    settings=cfg,
                    available_tools=(
                        context.get("critique_mcp_capabilities")
                        or context.get("available_mcp_capabilities") or []
                    ),
                    preferred_tools=context.get("preferred_mcp_capabilities"),
                    previous_plan=previous,
                    hypothesis_refs=context.get("hypothesis_refs") or [],
                    repo_candidates=context.get("repo_candidates") or [],
                    operations=context.get("operations") or [],
                    pipeline_scope=context.get("pipeline_scope"),
                    fedot_on=fedot_route_available(cfg, route_agents=route_agents),
                    medical_on=medical_route_available(route_agents=route_agents),
                )
                invariant_errors = _context_invariant_errors(checked_plan, context)
                blockers = _execution_blockers(checked_critique)
                if invariant_errors or blockers or _plan_digest(checked_plan) != digest:
                    raise PlanValidationError(
                        "Plan is no longer executable after review",
                        errors=invariant_errors or blockers or [{
                            "type": "candidate_changed", "loc": [],
                            "msg": "Plan digest changed while awaiting approval",
                        }],
                    )
            except (PlanValidationError, ValueError, TypeError) as exc:
                errors = json_validation_errors(getattr(exc, "errors", None) or [str(exc)])
                state["experiment_plan_validation_errors"] = errors
                state["experiment_plan_review_paused"] = True
                state["experiment_plan_fallback_pending"] = False
                state[PAUSE_REASON_STATE_KEY] = "fallback_candidate_no_longer_executable"
                state["experiment_plan_candidate"]["status"] = "blocked"
                state["experiment_module_outcome"] = {
                    "status": "blocked",
                    "stage": "plan_review",
                    "reason": "fallback_candidate_no_longer_executable",
                    "plan_id": plan.plan_id,
                    "revision": plan.revision,
                    "digest": digest,
                    "review_id": review_id,
                }
                view["status"] = "blocked"
                close_plan_record(ctx, record_id, "blocked", reason=str(errors))
                return HITLResponse(
                    action=HITLAction.REJECT,
                    approved=False,
                    decision_source=HITLDecisionSource.SYSTEM,
                    system_reason="fallback_candidate_no_longer_executable",
                    stop_review_loop=True,
                )

            runtime["critique"] = checked_critique.model_dump(mode="json")
            state["experiment_runtime"] = runtime
            approve_plan_with_human_override(
                state,
                plan_digest=digest,
                accepted_issue_ids=accepted_issue_ids,
                decision_source=source,
            )
            state["experiment_plan_fallback_pending"] = False
            state["experiment_plan_review_paused"] = False
            state["experiment_plan_validation_errors"] = None
            state[PAUSE_REASON_STATE_KEY] = None
            state["experiment_module_outcome"] = {
                "status": "running",
                "stage": "execution",
                "reason": "human_approved_with_issues",
                "plan_id": plan.plan_id,
                "revision": plan.revision,
                "digest": digest,
                "review_id": review_id,
            }
            view["status"] = "approved_with_issues"
            _publish_approved_plan_to_graph(ctx, state)
            close_plan_record(
                ctx, record_id, "approved", reason="human accepted unresolved critique issues"
            )
            _audit(
                f"EXPERIMENT_REVIEW_APPROVED kind=plan mode=human_override "
                f"plan_id={plan.plan_id} digest={digest[:12]} phase=execution"
            )
            return response

        state["experiment_plan_review_paused"] = True
        state["experiment_plan_fallback_pending"] = False
        terminal_reason = (
            "fallback_review_timeout" if response.timed_out
            else "fallback_auto_decision_rejected" if source != HITLDecisionSource.HUMAN.value
            else "fallback_rejected_by_operator"
        )
        state[PAUSE_REASON_STATE_KEY] = terminal_reason
        state["experiment_module_outcome"] = {
            "status": "blocked",
            "stage": "plan_review",
            "reason": terminal_reason,
            "plan_id": plan.plan_id,
            "revision": plan.revision,
            "digest": digest,
            "review_id": review_id,
        }
        candidate = state.get("experiment_plan_candidate") or {}
        candidate["status"] = "timed_out" if response.timed_out else "rejected"
        state["experiment_plan_candidate"] = candidate
        view["status"] = candidate["status"]
        close_plan_record(ctx, record_id, view["status"], reason=terminal_reason)
        return response.model_copy(update={"stop_review_loop": True})

    def _review_window(self, configured: float) -> float | None:
        """The deadline this review waits under, under the run's HITL mode.

        These two windows are deliberately the reviews' own: they fail CLOSED,
        so a bounded wait is a safety property of this stage rather than a
        preference — and it was the one voice the operator's mode did not
        reach. `handle_request` prefers a request's own window, so someone who
        had turned every timeout off still had the plan review expire at 300 s
        and the run skip execution.

        The MODE decides, outright: `debug` waits for the human (`None`),
        `basic` waits the ten minutes it names, and `auto` never gets here
        because `_auto_approve` answered first. Taking the tighter of the two
        was the obvious thing and the wrong one — the default review window is
        300 s, so the plan card, of all cards, got half the wait the mode
        advertises, and an operator who set the field to an hour was capped to
        ten minutes with nothing saying so.

        `configured` is retained in the signature for compatibility only.
        """
        try:
            from CoScientist.hitl.mode import wait_seconds

            window = wait_seconds()
        except Exception:  # noqa: BLE001 — fall back to the safe global default
            from CoScientist.hitl.mode import BASIC_WAIT_S

            return BASIC_WAIT_S
        return window

    def _hitl(
        self, *, message: str, kind: str, plan_id: Any, output: str,
        user_id: str, session_id: str, timeout_seconds: float | None,
        plan_view: dict[str, Any] | None = None,
        requires_human: bool = False,
        review_exhausted: bool = False,
        review_id: str | None = None,
    ) -> HITLRequest:
        context: dict[str, Any] = {
            "output": output, "experiment_review_kind": kind, "experiment_plan_id": plan_id,
            "_session": {"user_id": user_id, "session_id": session_id},
        }
        if plan_view:
            # The structured plan next to the rendered one: the web UI draws the
            # design matrix and the task cards from this, while ``output`` stays
            # the console's (and any other client's) copy of the same plan.
            context["experiment_plan"] = plan_view
        if review_exhausted:
            context["experiment_review_exhausted"] = True
        if review_id:
            context["experiment_review_id"] = review_id
        return HITLRequest(
            agent_name=self.name, action_type=HITLAction.APPROVE, message=message,
            context=context,
            invoked_via="internal_loop", timeout_seconds=timeout_seconds,
            requires_human=requires_human,
        )

    @staticmethod
    def _route_selection_key(task: Any) -> str:
        assessment = task.code_assessment
        return json.dumps(
            {
                "task_id": task.id,
                "operation_ref": task.design.operation_ref,
                "repo_url": task.repo_url,
                "requirement": assessment.requirement.value,
                "entrypoints": assessment.entrypoints,
                "evidence": assessment.evidence,
            },
            ensure_ascii=True,
            sort_keys=True,
        )

    @staticmethod
    def _apply_repository_route(
        plan: ExperimentPlan,
        task_id: str,
        route: str,
        *,
        warning: str | None = None,
    ) -> ExperimentPlan:
        selected = ExecutionRoute(route)
        tasks = []
        for task in plan.tasks:
            if task.id != task_id:
                tasks.append(task)
                continue
            update: dict[str, Any] = {
                "route": selected,
                "mcp_servers": [],
                "post_build_route": (
                    ExecutionRoute.REACT_TOOLS.value
                    if selected == ExecutionRoute.ALEMBIC_BUILD else None
                ),
            }
            if warning:
                update["warnings"] = [
                    *task.warnings,
                    *([] if warning in task.warnings else [warning]),
                ]
            tasks.append(task.model_copy(update=update))
        return plan.model_copy(update={"tasks": tasks})

    async def _select_repository_routes(
        self,
        *,
        ctx: InvocationContext,
        plan: ExperimentPlan,
        route_alembic: bool,
        user_id: str,
        session_id: str,
        timeout_seconds: float | None,
    ) -> tuple[ExperimentPlan, HITLResponse | None]:
        """Resolve the one intentionally non-automatic Coder/Alembic fork."""
        if not route_alembic:
            return plan, None

        state = ctx.session.state
        raw_cached = state.get(ROUTE_SELECTIONS_STATE_KEY)
        cached = dict(raw_cached) if isinstance(raw_cached, dict) else {}
        current = plan
        preflight: dict[str, Any] | None = None

        for original in plan.tasks:
            if (
                original.code_assessment.requirement != CodeRequirement.REUSE
                or not original.repo_url
                or original.route not in {ExecutionRoute.CODER, ExecutionRoute.ALEMBIC_BUILD}
            ):
                continue

            key = self._route_selection_key(original)
            prior = cached.get(key)
            if isinstance(prior, dict) and prior.get("route") in {
                ExecutionRoute.CODER.value,
                ExecutionRoute.ALEMBIC_BUILD.value,
            } and not prior.get("recheck"):
                route = str(prior["route"])
                current = self._apply_repository_route(current, original.id, route)
                _audit(
                    f"EXPERIMENT_ROUTE_DECISION task={original.id} route={route} "
                    "source=cached"
                )
                continue

            if preflight is None:
                preflight = _alembic_preflight()
            if not preflight.get("available"):
                reason = str(preflight.get("reason") or "Alembic is unavailable")
                route = ExecutionRoute.CODER.value
                cached[key] = {
                    "task_id": original.id,
                    "route": route,
                    "source": "system",
                    "reason": reason,
                    # Infrastructure can recover without changing the plan;
                    # probe it again on a later review instead of pinning a
                    # transient outage as if it were an operator decision.
                    "recheck": True,
                }
                current = self._apply_repository_route(
                    current,
                    original.id,
                    route,
                    warning=f"Alembic unavailable; using Coder: {reason}",
                )
                _audit(
                    f"EXPERIMENT_ROUTE_DECISION task={original.id} route={route} "
                    f"source=system reason={_clip(reason)}"
                )
                continue

            request = HITLRequest(
                agent_name=self.name,
                action_type=HITLAction.SELECT,
                message=(
                    f"Task {original.id} can reuse {original.repo_url} unchanged. "
                    "Choose direct execution or build a reusable MCP tool. "
                    "Coder is the default because it avoids the container/build step."
                ),
                options=[_ROUTE_CODER_OPTION, _ROUTE_ALEMBIC_OPTION],
                default_option=_ROUTE_CODER_OPTION,
                context={
                    "experiment_review_kind": "repository_route",
                    "experiment_plan_id": plan.plan_id,
                    "task_id": original.id,
                    "repo_url": original.repo_url,
                    "operation_ref": original.design.operation_ref,
                    "assessment": original.code_assessment.model_dump(mode="json"),
                    "alembic_preflight": preflight,
                    "_session": {"user_id": user_id, "session_id": session_id},
                },
                invoked_via="internal_loop",
                timeout_seconds=timeout_seconds,
            )
            response = await self.hitl_handler.handle_request(request)
            selected = response.selected_option
            if response.approved and selected in {_ROUTE_CODER_OPTION, _ROUTE_ALEMBIC_OPTION}:
                route = (
                    ExecutionRoute.ALEMBIC_BUILD.value
                    if selected == _ROUTE_ALEMBIC_OPTION else ExecutionRoute.CODER.value
                )
                source = getattr(response.decision_source, "value", response.decision_source)
                cached[key] = {
                    "task_id": original.id,
                    "route": route,
                    "source": str(source),
                }
                current = self._apply_repository_route(current, original.id, route)
                _audit(
                    f"EXPERIMENT_ROUTE_DECISION task={original.id} route={route} "
                    f"source={source}"
                )
                continue

            state["experiment_plan_review_paused"] = True
            reason = "repository_route_timeout" if response.timed_out else "repository_route_rejected"
            state[PAUSE_REASON_STATE_KEY] = reason
            state["experiment_module_outcome"] = {
                "status": "blocked",
                "stage": "plan_review",
                "reason": reason,
                "task_id": original.id,
            }
            state[ROUTE_SELECTIONS_STATE_KEY] = cached
            _audit(f"EXPERIMENT_PLAN_REVIEW_PAUSED reason={reason} task={original.id}")
            return current, response.model_copy(update={"stop_review_loop": True})

        state[ROUTE_SELECTIONS_STATE_KEY] = cached
        return current, None

    def _publish_state(self, ctx: InvocationContext, event: Event) -> None:
        """Copy this reviewer's decisions into ``event``'s state delta.

        Only a delta on a yielded event survives the AgentTool boundary, so a
        decision that is not published here is invisible to the next module hop.
        Keys ADK already put on the event (the output_key it fills from the
        model turn) win — this fills in what nobody else reports.
        """
        state = ctx.session.state
        delta = event.actions.state_delta
        published = []
        for key in _REVIEW_OWNED_STATE_KEYS:
            if key in delta or key not in state:
                continue
            delta[key] = state[key]
            published.append(key)
        self._state_publish_pending = False
        if published:
            runtime = state.get("experiment_runtime")
            _audit(
                "EXPERIMENT_REVIEW_STATE_PUBLISHED kind=%s phase=%r keys=%d"
                % (
                    self.review_kind,
                    runtime.get("phase") if isinstance(runtime, dict) else None,
                    len(published),
                )
            )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if self.review_kind == "result":
            runtime = ctx.session.state.get("experiment_runtime") or {}
            if runtime.get("phase") not in {"reporting", "awaiting_result_review"}:
                _audit(f"EXPERIMENT_REVIEW_PAUSED kind=result phase={runtime.get('phase') or 'missing'}")
                outcome = ctx.session.state.get("experiment_module_outcome") or {}
                if outcome.get("stage") == "plan_review":
                    reason = str(outcome.get("reason") or "plan_not_approved")
                    if outcome.get("status") == "awaiting_human":
                        message = (
                            "Automatic plan revisions are exhausted and the current "
                            "plan is waiting for a human decision. The experiment has not started."
                        )
                    else:
                        message = (
                            f"Experiment planning stopped before execution ({reason}). "
                            "No experiment results were produced."
                        )
                else:
                    message = (
                        "Experiment result review is paused because execution has not reached reporting."
                    )
                yield Event(
                    invocation_id=ctx.invocation_id, author=self.name, branch=ctx.branch,
                    content=types.Content(role="model", parts=[types.Part(
                        text=message,
                    )]),
                )
                return
        self._state_publish_pending = False
        async for event in super()._run_async_impl(ctx):
            if self._state_publish_pending:
                self._publish_state(ctx, event)
            yield event
        if self._state_publish_pending:
            # The base loop can decide without yielding anything (empty final
            # output, or a break on a paused review). Carry the delta out on an
            # event of our own rather than lose the decision.
            carrier = Event(
                invocation_id=ctx.invocation_id, author=self.name, branch=ctx.branch,
            )
            self._publish_state(ctx, carrier)
            yield carrier

    async def _review_plan(self, ctx: InvocationContext, output_text: Any) -> HITLResponse:
        state, cfg = ctx.session.state, get_settings().experiments
        user_id, session_id = session_key(ctx)
        try:
            context = state.get("experiment_context") or {}
            runtime = state.get("experiment_runtime") or {}
            previous = ExperimentPlan.model_validate(runtime["plan"]) if runtime.get("plan") else None
            route_agents = session_route_agents(getattr(ctx, "agent", None))
            payload = _stamp_context_invariants(_json_payload(output_text), context, previous)
            # Asked of this session's executor, the one start_task hands work to:
            # a route switched on after the session was built must not be
            # approved here and then refused there.
            plan, critique = validate_and_critique_plan(
                payload, settings=cfg,
                available_tools=(
                    context.get("critique_mcp_capabilities")
                    or context.get("available_mcp_capabilities") or []
                ),
                preferred_tools=context.get("preferred_mcp_capabilities"), previous_plan=previous,
                hypothesis_refs=context.get("hypothesis_refs") or [],
                repo_candidates=context.get("repo_candidates") or [],
                operations=context.get("operations") or [],
                pipeline_scope=context.get("pipeline_scope"),
                fedot_on=fedot_route_available(cfg, route_agents=route_agents),
                medical_on=medical_route_available(route_agents=route_agents),
            )
            if errs := _context_invariant_errors(plan, context):
                raise PlanValidationError("ExperimentPlan context invariants failed", errors=errs)
        except (PlanValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors = json_validation_errors(getattr(exc, "errors", None) or [str(exc)])
            state["experiment_plan_validation_errors"] = errors
            _audit("EXPERIMENT_PLAN_REVISE reason=schema errors=" + json.dumps(errors, default=str, ensure_ascii=True))
            return await self._revise(
                ctx=ctx, detail=errors,
                pause_prefix="Plan validation failed repeatedly; experiment remains paused. Last errors:",
                edit_prefix="Deterministic schema validation failed. Return a complete corrected ExperimentPlan JSON. Errors:",
                context=context,
                previous=previous,
                route_agents=set(route_agents or ()),
            )

        # Diagnostics belong to the candidate that produced them.  A later
        # schema-valid candidate must not carry an old exception through the
        # AgentTool boundary merely because deterministic critique still asks
        # for a revision.
        state["experiment_plan_validation_errors"] = None
        critique_json = critique.model_dump(mode="json")
        state["experiment_plan_critique"] = critique_json
        if critique.verdict != "approve":
            issue_text = "; ".join(
                f"{i.severity}/{i.category}: {i.message} Suggestion: {i.suggestion}"
                for i in critique.issues if i.is_blocking
            )
            inv = any("absent from the capability inventory" in (i.message or "") for i in critique.issues)
            _audit("EXPERIMENT_PLAN_REVISE reason=critique issues=" + json.dumps(critique_json["issues"], ensure_ascii=True))
            return await self._revise(
                ctx=ctx, detail=issue_text,
                pause_prefix="PlanCritique kept rejecting the plan; experiment remains paused. Last issues:",
                edit_prefix=(
                    "Deterministic PlanCritique requires revision. "
                    "Use ONLY tools from available_mcp_capabilities, or route=coder "
                    "(or alembic_build when a repo_candidate fits). Issues:"
                    if inv else "Deterministic PlanCritique requires revision:"
                ),
                inventory_blocker=inv,
                plan=plan,
                critique=critique,
                context=context,
                previous=previous,
                route_agents=set(route_agents or ()),
            )

        # A proven unchanged repository entrypoint is the only ambiguous code
        # route. Resolve it before runtime initialisation, so Alembic can never
        # be selected by a planner guess or by a later orchestrator bypass.
        window = self._review_window(cfg.plan_review_timeout_s)
        before_route_selection = plan
        plan, route_response = await self._select_repository_routes(
            ctx=ctx,
            plan=plan,
            route_alembic=bool(cfg.route_alembic),
            user_id=user_id,
            session_id=session_id,
            timeout_seconds=window,
        )
        if route_response is not None:
            return route_response
        if plan != before_route_selection:
            try:
                plan, critique = validate_and_critique_plan(
                    plan.model_dump(mode="json"), settings=cfg,
                    available_tools=(
                        context.get("critique_mcp_capabilities")
                        or context.get("available_mcp_capabilities") or []
                    ),
                    preferred_tools=context.get("preferred_mcp_capabilities"),
                    previous_plan=previous,
                    hypothesis_refs=context.get("hypothesis_refs") or [],
                    repo_candidates=context.get("repo_candidates") or [],
                    operations=context.get("operations") or [],
                    pipeline_scope=context.get("pipeline_scope"),
                    fedot_on=fedot_route_available(cfg, route_agents=route_agents),
                    medical_on=medical_route_available(route_agents=route_agents),
                )
                if errs := _context_invariant_errors(plan, context):
                    raise PlanValidationError(
                        "ExperimentPlan context invariants failed", errors=errs
                    )
            except (PlanValidationError, ValueError, TypeError) as exc:
                errors = json_validation_errors(getattr(exc, "errors", None) or [str(exc)])
                state["experiment_plan_validation_errors"] = errors
                return await self._revise(
                    ctx=ctx, detail=errors,
                    pause_prefix="Selected route failed deterministic validation; experiment remains paused:",
                    edit_prefix="Return a corrected plan for the selected repository route:",
                    context=context,
                    previous=previous,
                    route_agents=set(route_agents or ()),
                )
            state["experiment_plan_validation_errors"] = None
            critique_json = critique.model_dump(mode="json")
            state["experiment_plan_critique"] = critique_json
            if critique.verdict != "approve":
                issue_text = "; ".join(
                    f"{i.severity}/{i.category}: {i.message} Suggestion: {i.suggestion}"
                    for i in critique.issues if i.is_blocking
                )
                return await self._revise(
                    ctx=ctx, detail=issue_text,
                    pause_prefix="Selected route failed deterministic critique; experiment remains paused:",
                    edit_prefix="Return a corrected plan for the selected repository route:",
                    plan=plan,
                    critique=critique,
                    context=context,
                    previous=previous,
                    route_agents=set(route_agents or ()),
                )

        state["experiment_plan_review_paused"] = False
        state["experiment_plan_fallback_pending"] = False
        state["experiment_plan_candidate"] = None
        state["experiment_module_outcome"] = {
            "status": "awaiting_human" if not _auto_approve("plan") else "running",
            "stage": "plan_review" if not _auto_approve("plan") else "execution",
            "reason": "standard_plan_review",
            "plan_id": plan.plan_id,
            "revision": plan.revision,
            "digest": _plan_digest(plan),
        }
        state[PAUSE_REASON_STATE_KEY] = None
        state["experiment_plan_validation_errors"] = None
        # The budget bounds CONSECUTIVE failures, not a whole session. Until this
        # reset it was only cleared in approve_plan, so a plan that validated but
        # was never approved left the count standing: a later human HITL edit then
        # re-entered planning with a partly spent budget and could exhaust it on
        # the first stumble.
        state["experiment_plan_revision_count"] = 0
        state["experiment_inventory_blocker_hits"] = 0
        runtime = initialize_runtime(state, plan, critique=critique_json)
        route_selections = state.get(ROUTE_SELECTIONS_STATE_KEY)
        if isinstance(route_selections, dict):
            for decision in route_selections.values():
                if not isinstance(decision, dict):
                    continue
                task_runtime = (runtime.get("tasks") or {}).get(decision.get("task_id"))
                if not isinstance(task_runtime, dict):
                    continue
                if decision.get("route") != task_runtime.get("planned_route"):
                    continue
                task_runtime["route_history"] = [{
                    "route": task_runtime["planned_route"],
                    "reason": "repository_route_selected",
                    "decision_source": decision.get("source"),
                    **({"detail": decision["reason"]} if decision.get("reason") else {}),
                }]

        # One structured plan, three readers: the web review card, the call
        # graph's record of this round, and anything later that wants the plan
        # without re-deriving it from the runtime.
        lang = session_report_language(state)
        view = plan_to_view(plan, critique_json)
        state["experiment_plan_view"] = view
        record_id = record_plan_proposed(ctx, self.name, view)
        state["experiment_plan_record_id"] = record_id

        if _auto_approve("plan"):
            approve_plan(state)
            _publish_approved_plan_to_graph(ctx, state)
            close_plan_record(ctx, record_id, "approved", reason="headless auto-approve")
            _audit(f"EXPERIMENT_REVIEW_APPROVED kind=plan mode={_approval_mode()} plan_id={plan.plan_id} phase=execution")
            _audit("EXPERIMENT_DESIGN_MATRIX\n" + render_experiment_plan(plan, "en"))
            return _auto_approve_response()

        response = await self.hitl_handler.handle_request(self._hitl(
            message="Review and explicitly approve the experiment plan.", kind="plan",
            plan_id=plan.plan_id, output=render_experiment_plan(plan, lang),
            user_id=user_id, session_id=session_id, timeout_seconds=window,
            plan_view=view,
        ))
        if response.approved:
            approve_plan(state)
            state["experiment_module_outcome"] = {
                "status": "running",
                "stage": "execution",
                "reason": "plan_approved",
                "plan_id": plan.plan_id,
                "revision": plan.revision,
                "digest": _plan_digest(plan),
            }
            _publish_approved_plan_to_graph(ctx, state)
            _audit(f"EXPERIMENT_REVIEW_APPROVED kind=plan mode=human plan_id={plan.plan_id} phase=execution")
            _audit("EXPERIMENT_DESIGN_MATRIX\n" + render_experiment_plan(plan, "en"))
        elif _is_refusal(response):
            # «Отклонить» and «Доработать» were the same thing: neither set
            # `stop_review_loop`, so `SessionAgent` fed the note back and the
            # planner wrote another plan. The operator pressed reject three
            # revisions in a row and the module kept going — there was no way
            # to say "stop", only "try again".
            #
            # A refusal is now terminal for this module. The run does not die:
            # the executor is skipped for want of an approved runtime, and the
            # report guard turns that into an honest note instead of a report.
            state["experiment_plan_review_paused"] = True
            state[PAUSE_REASON_STATE_KEY] = "plan_rejected_by_operator"
            state["experiment_module_outcome"] = {
                "status": "rejected",
                "stage": "plan_review",
                "reason": "plan_rejected_by_operator",
                "plan_id": plan.plan_id,
                "revision": plan.revision,
                "digest": _plan_digest(plan),
            }
            _audit(f"EXPERIMENT_REVIEW_REJECTED kind=plan plan_id={plan.plan_id} "
                   f"reason={_clip(response.instructions) or 'no reason given'}")
            view["status"] = "rejected"
            close_plan_record(ctx, record_id, "rejected",
                              reason=_clip(response.instructions))
            return response.model_copy(update={"stop_review_loop": True})
        view["status"] = _plan_outcome(response)
        if response.timed_out:
            # The record has said "paused" all along (_plan_outcome); state
            # said nothing, so the orchestrator blamed its attempt budget.
            state[PAUSE_REASON_STATE_KEY] = "plan_review_timeout"
            state["experiment_module_outcome"] = {
                "status": "blocked",
                "stage": "plan_review",
                "reason": "plan_review_timeout",
                "plan_id": plan.plan_id,
                "revision": plan.revision,
                "digest": _plan_digest(plan),
            }
            _audit(f"EXPERIMENT_REVIEW_TIMEOUT kind=plan plan_id={plan.plan_id} "
                   f"window_s={_window_word(window)}")
        close_plan_record(ctx, record_id, view["status"],
                          reason=_clip(response.instructions))
        return response

    async def _review_result(self, ctx: InvocationContext, _output_text: Any) -> HITLResponse:
        state, cfg = ctx.session.state, get_settings().experiments
        user_id, session_id = session_key(ctx)
        runtime = state.get("experiment_runtime") or {}
        runtime["phase"] = "awaiting_result_review"
        if runtime:
            # Same reason as in mark_result_review: ADK only records a state
            # delta on assignment, so mutating the nested dict leaves the phase
            # this reviewer just set invisible to everything downstream.
            state["experiment_runtime"] = runtime
        tasks_ok = result_tasks_ok(runtime)
        # Materialize canonical ArtifactRef locations before HITL / auto-approve.
        rendered = render_experiment_results(state)

        if _auto_approve("result"):
            result = mark_result_review(state, approved=True)
            state["experiment_module_outcome"] = {
                "status": "completed",
                "stage": "result_review",
                "reason": "result_auto_approved",
                "plan_id": runtime.get("plan_id"),
                "tasks_ok": tasks_ok,
            }
            _audit(
                f"EXPERIMENT_REVIEW_APPROVED kind=result mode={_approval_mode()} "
                f"plan_id={runtime.get('plan_id')} phase={result['phase']} tasks_ok={str(tasks_ok).lower()}"
            )
            return _auto_approve_response()

        window = self._review_window(cfg.result_review_timeout_s)
        state["experiment_module_outcome"] = {
            "status": "awaiting_human",
            "stage": "result_review",
            "reason": "result_review_pending",
            "plan_id": runtime.get("plan_id"),
            "tasks_ok": tasks_ok,
        }
        response = await self.hitl_handler.handle_request(self._hitl(
            message="Accept the experiment results, or reject with feedback to request a redesigned experiment.",
            kind="result", plan_id=runtime.get("plan_id"), output=rendered,
            user_id=user_id, session_id=session_id, timeout_seconds=window,
        ))
        if response.timed_out:
            state[PAUSE_REASON_STATE_KEY] = "result_review_timeout"
            state["experiment_module_outcome"] = {
                "status": "blocked",
                "stage": "result_review",
                "reason": "result_review_timeout",
                "plan_id": runtime.get("plan_id"),
                "tasks_ok": tasks_ok,
            }
            _audit(f"EXPERIMENT_REVIEW_TIMEOUT kind=result "
                   f"plan_id={runtime.get('plan_id')} "
                   f"window_s={_window_word(window)}")
            return response
        if response.approved:
            result = mark_result_review(state, approved=True)
            state["experiment_module_outcome"] = {
                "status": "completed",
                "stage": "result_review",
                "reason": "result_approved",
                "plan_id": runtime.get("plan_id"),
                "tasks_ok": tasks_ok,
            }
            _audit(
                f"EXPERIMENT_REVIEW_APPROVED kind=result mode=human "
                f"plan_id={runtime.get('plan_id')} phase={result['phase']} tasks_ok={str(tasks_ok).lower()}"
            )
            return response
        feedback = response.instructions or response.free_input or "Human requested experiment redesign."
        mark_result_review(state, approved=False, feedback=feedback)
        state["experiment_module_outcome"] = {
            "status": "rejected",
            "stage": "result_review",
            "reason": "result_rejected_by_operator",
            "plan_id": runtime.get("plan_id"),
            "tasks_ok": tasks_ok,
        }
        return response.model_copy(update={"stop_review_loop": True})

    async def _review_decision(self, ctx: InvocationContext, output_text: Any) -> HITLResponse:
        try:
            if self.review_kind == "plan":
                return await self._review_plan(ctx, output_text)
            return await self._review_result(ctx, output_text)
        finally:
            # Publish whatever the decision wrote, including on the raising and
            # pausing paths — a pause that does not reach the caller is a loop.
            self._state_publish_pending = True


__all__ = [
    "ExperimentReviewSessionAgent",
    "FailClosedExperimentHITLHandler",
    "build_experiment_artifacts_manifest",
    "fail_closed_handler",
    "render_experiment_plan",
    "render_experiment_results",
    "result_tasks_ok",
]
