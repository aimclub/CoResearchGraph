"""Deterministic task/attempt state machine (ADK session is the store)."""
from __future__ import annotations

import copy
import functools
import logging
import os
from datetime import timedelta
from typing import Any, Callable, Collection, Mapping, MutableMapping
from uuid import uuid4

from CoScientist.config import get_settings
from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.schemas import (
    CodeRequirement,
    CriterionCheck,
    ExecutionRoute,
    ExperimentPlan,
    ExperimentTask,
    TaskResult,
    utc_now,
)
from CoScientist.experiments.runtime.artifacts import (
    ARTIFACT_KEYS,
    EVIDENCE_AGENT_ROUTES,
    append_notes_artifact,
    attest_durable_criteria,
    captured_delta,
    criteria_valid,
    find_artifact,
    has_durable_family_evidence,
    normalise_artifacts,
    required_artifacts_present,
    route_response_text,
    runtime_has_durable_data_evidence,
    synthesize_mcp_server_artifacts,
    task_requires_managed_s3,
)
from CoScientist.experiments.runtime.errors import ExperimentRuntimeError
from CoScientist.experiments.runtime.readiness import TERMINAL_TASK_STATES, refresh_readiness
from CoScientist.experiments.runtime.routing import (
    match_session_inventory_tool,
    mcp_routes_tried,
    session_inventory_nonempty,
    task_coverage_blob,
)
from CoScientist.experiments.runtime.shared import FABRICATION_MARKERS, audit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_AMEND_FIELDS = frozenset({
    "route",
    "mcp_servers",
    "repo_url",
    "post_build_route",
    "input_data",
    "launch_params",
    "warnings",
    "success_criteria",
})
_CLEAR_ACTIVE_KEYS = (
    "experiment_active_envelope", "filtered_tools", "deployed_mcps", "upstream_artifact_inputs"
)
# LLM often emits synonyms outside the closed TaskResult.status enum.
_RESULT_STATUS_ALIASES = {
    "error": "failure",
    "failed": "failure",
    "fail": "failure",
    "partial_success": "partial",
    "partially_successful": "partial",
    "incomplete": "partial",
    "ok": "success",
    "succeeded": "success",
}


def _is_core_execution_failure(result: Mapping[str, Any]) -> bool:
    """True when a partial label would hide absence of the primary operation."""
    code = str(result.get("error_code") or "").strip().upper()
    text = " ".join(
        str(result.get(key) or "") for key in ("error_code", "error_message", "summary")
    ).upper()
    return (
        "NO_MATCHING_TOOL" in text
        or code.startswith("MISSING_")
        or code.startswith("REQUIRED_INPUT_")
        or code.startswith("PRIMARY_OPERATION_")
        or code in {"TOOL_UNAVAILABLE", "CAPABILITY_UNAVAILABLE", "CAPABILITY_MISSING"}
    )


def _result_text_blob(result: dict[str, Any]) -> str:
    parts: list[str] = [str(result.get("summary") or "")]
    for w in result.get("warnings") or []:
        parts.append(str(w))
    if result.get("error_message"):
        parts.append(str(result["error_message"]))
    for check in result.get("criteria_checks") or []:
        if isinstance(check, dict):
            parts.append(str(check.get("observed") or ""))
            parts.append(str(check.get("details") or ""))
    return "\n".join(parts)


def fabrication_signals(result: dict[str, Any]) -> list[str]:
    """Matched fabrication/simulation markers in a record_result payload."""
    blob = _result_text_blob(result)
    return sorted({m.group(0).lower() for m in FABRICATION_MARKERS.finditer(blob)})


def _downgrade_fabricated_success(result: dict[str, Any]) -> dict[str, Any]:
    """Force success→partial when the agent admits simulated/fabricated evidence."""
    if result.get("status") != "success":
        return result
    hits = fabrication_signals(result)
    if not hits:
        return result
    out = copy.deepcopy(result)
    out["status"] = "partial"
    warnings = list(out.get("warnings") or [])
    warnings.append(
        "downgraded_from_success: fabricated/simulated evidence detected "
        f"({', '.join(hits)})"
    )
    out["warnings"] = warnings
    return out


def _coerce_alembic_mcp_success(
    attempt: Mapping[str, Any], result: dict[str, Any],
    task: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """MCP URL means the build attempt succeeded — reopen post_build, never partial."""
    if str(attempt.get("route") or "") != ExecutionRoute.ALEMBIC_BUILD.value:
        return result
    from CoScientist.experiments.runtime.alembic_bridge import harvest_alembic_mcp_url

    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    snap = attempt.get("alembic_snapshot")
    mcp_url = harvest_alembic_mcp_url(
        outputs,
        result.get("summary"),
        snap,
        repo_url=str((attempt.get("task") or {}).get("repo_url") or "").strip() or None,
    )
    if not str(mcp_url).startswith("http"):
        return result
    checks = []
    for item in result.get("criteria_checks") or []:
        if isinstance(item, dict):
            checks.append({**item, "passed": True})
        else:
            checks.append(item)
    # The served address is the evidence of a build task. An executor that
    # reports it without a check per criterion used to trip the evidence gate
    # ("missing required evidence: criteria=['C1']"), and the downgrade the
    # tool then tried was coerced back here and raised through the module
    # (2026-09-23, Informer2020 reuse). Attest every task criterion on it.
    seen = {str(c.get("criterion_id") or "") for c in checks if isinstance(c, dict)}
    for item in (task or attempt.get("task") or {}).get("success_criteria") or []:
        cid = str(item.get("criterion_id") or "").strip() if isinstance(item, dict) else ""
        if cid and cid not in seen:
            checks.append({"criterion_id": cid, "passed": True, "details": f"mcp_url={mcp_url}"})
    warnings = [
        w for w in (result.get("warnings") or [])
        if "downgraded_from_success" not in str(w)
    ]
    warnings.append("coerced_alembic_mcp_success")
    return {
        **result,
        "status": "success",
        "outputs": {**outputs, "mcp_url": mcp_url, "mcp_endpoint": mcp_url},
        "criteria_checks": checks,
        "warnings": warnings,
    }


RUNTIME_KEY = "experiment_runtime"
# Replan rounds live OUTSIDE the runtime dict on purpose: a replan rebuilds the
# runtime, and build_experiment_context nulls RUNTIME_KEY through its
# _CLEAR_ON_NEW_RUN list before the planner runs. A counter kept inside the
# runtime is therefore zeroed by the very event it is meant to count, which
# left the budget bounding nothing. builder.py resets this key only when the
# ask itself changes, so it bounds the run rather than a single plan.
REPLAN_ROUNDS_KEY = "experiment_replan_rounds"
# The task as planned, before start_task moved a coder task onto the research or
# medical family its text names; restored if that family's route goes away.
_PRE_FAMILY_REWRITE_KEY = "task_before_family_rewrite"
# The agent the route agents are attached to: a route is live only while its
# agent is one of this executor's enabled subordinates.
EXECUTOR_AGENT = "ExperimentExecutorAgent"
ROUTE_AGENT_BY_ROUTE = {
    ExecutionRoute.FEDOT_MAS.value: "FedotAgent",
    ExecutionRoute.REACT_TOOLS.value: "ExperimentAgent",
    ExecutionRoute.CODER.value: "CoderAgent",
    ExecutionRoute.ALEMBIC_BUILD.value: "McpBuilderAgent",
    ExecutionRoute.RESEARCH.value: "ResearchAgent",
    ExecutionRoute.MEDICAL.value: "MedicalAgent",
}
# Defaults; prefer resolve_fallback_chains(settings) so EXPERIMENTS__FALLBACK_* apply.
FALLBACK_CHAINS = {
    ExecutionRoute.FEDOT_MAS.value: [
        ExecutionRoute.FEDOT_MAS.value,
        ExecutionRoute.REACT_TOOLS.value,
        ExecutionRoute.CODER.value,
    ],
    ExecutionRoute.REACT_TOOLS.value: [ExecutionRoute.REACT_TOOLS.value, ExecutionRoute.CODER.value],
    ExecutionRoute.CODER.value: [ExecutionRoute.CODER.value],
    ExecutionRoute.ALEMBIC_BUILD.value: [ExecutionRoute.ALEMBIC_BUILD.value, ExecutionRoute.CODER.value],
    ExecutionRoute.RESEARCH.value: [ExecutionRoute.RESEARCH.value],
    ExecutionRoute.MEDICAL.value: [ExecutionRoute.MEDICAL.value],
}


def _settings(value: ExperimentsSettings | None) -> ExperimentsSettings:
    return value or get_settings().experiments


def resolve_fallback_chains(settings: ExperimentsSettings | None = None) -> dict[str, list[str]]:
    """Route fallback chains from settings (EXPERIMENTS__FALLBACK_*)."""
    cfg = _settings(settings)
    return {
        ExecutionRoute.FEDOT_MAS.value: list(cfg.fallback_fedot_mas),
        ExecutionRoute.REACT_TOOLS.value: list(cfg.fallback_react_tools),
        ExecutionRoute.CODER.value: list(cfg.fallback_coder),
        ExecutionRoute.ALEMBIC_BUILD.value: list(cfg.fallback_alembic_build),
        ExecutionRoute.RESEARCH.value: list(cfg.fallback_research),
        ExecutionRoute.MEDICAL.value: list(cfg.fallback_medical),
    }


def _runtime(state: MutableMapping[str, Any]) -> dict[str, Any]:
    if not isinstance(runtime := state.get(RUNTIME_KEY), dict):
        raise ExperimentRuntimeError("runtime_missing", "No experiment runtime is active.")
    return runtime


def _task(runtime: dict[str, Any], task_id: str) -> dict[str, Any]:
    if not isinstance(task := (runtime.get("tasks") or {}).get(task_id), dict):
        raise ExperimentRuntimeError("task_not_found", f"Unknown experiment task {task_id!r}.")
    return task


_audit = functools.partial(audit, logger)


def _publish_active_tasks(state: MutableMapping[str, Any], runtime: dict[str, Any]) -> None:
    state["active_tasks"] = [
        {
            "id": task_id,
            "title": tr["task"]["name"],
            "description": tr["task"]["description"],
            "assignee": "ExperimentModuleAgent",
            "route": tr["current_route"],
            "status": tr["status"],
            "notes": tr.get("last_message", ""),
        }
        for task_id in runtime["task_order"]
        for tr in (runtime["tasks"][task_id],)
    ]


def _block_unstartable(
    state: MutableMapping[str, Any], task_id: str, exc: ExperimentRuntimeError,
) -> None:
    """A ready task that cannot resolve required inputs is terminal, not retryable-ready."""
    runtime = _runtime(state)
    task_runtime = _task(runtime, task_id)
    if task_runtime["status"] in TERMINAL_TASK_STATES:
        return
    task_runtime["status"] = "blocked"
    task_runtime["last_message"] = str(exc)
    _sync_after_mutation(state, runtime)
    _audit(f"EXPERIMENT_TASK_BLOCKED task_id={task_id} reason={exc.code}")


def _clear_active(state: MutableMapping[str, Any], runtime: dict[str, Any]) -> None:
    runtime["active_task_id"] = runtime["active_attempt_id"] = None
    for key in _CLEAR_ACTIVE_KEYS:
        state[key] = None


def _finish_if_terminal(runtime: dict[str, Any]) -> None:
    if all(runtime["tasks"][tid]["status"] in TERMINAL_TASK_STATES for tid in runtime["task_order"]):
        runtime["phase"] = "reporting"


def _sync_after_mutation(
    state: MutableMapping[str, Any], runtime: dict[str, Any], *, clear_active: bool = False
) -> None:
    if clear_active:
        _clear_active(state, runtime)
    refresh_readiness(runtime)
    _finish_if_terminal(runtime)
    _publish_active_tasks(state, runtime)


def initialize_runtime(
    state: MutableMapping[str, Any],
    plan: ExperimentPlan,
    *,
    critique: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create task-scoped runtime for one reviewed plan draft."""
    tasks = {
        task.id: {
            "status": "pending",
            "planned_route": task.route.value,
            "current_route": task.route.value,
            "route_history": [{"route": task.route.value, "reason": "planned"}],
            "task": task.model_dump(mode="json"),
            "attempts": {},
            "attempt_order": [],
            "last_message": "",
        }
        for task in plan.tasks
    }
    # The round count is read from the durable key, not from the previous runtime:
    # by the time a replan reaches here the builder has already nulled RUNTIME_KEY.
    carried_rounds = int(state.get(REPLAN_ROUNDS_KEY) or 0)
    previous = state.get(RUNTIME_KEY)
    carried_feedback = (
        previous.get("result_review_feedback") if isinstance(previous, dict) else None
    )

    runtime = {
        "run_id": plan.experiment_run_id,
        "plan_id": plan.plan_id,
        "phase": "awaiting_review",
        "approved": False,
        "plan": plan.model_dump(mode="json"),
        "critique": critique,
        "active_task_id": None,
        "active_attempt_id": None,
        "task_order": [task.id for task in plan.tasks],
        "tasks": tasks,
        "results": [],
        # Why this plan exists, so the executor and the report can say so.
        "result_review_feedback": carried_feedback if carried_rounds else None,
        "replan_rounds": carried_rounds,
    }
    refresh_readiness(runtime)
    state[RUNTIME_KEY] = runtime
    state["experiment_plan"] = runtime["plan"]
    state["experiment_task_results"] = []
    state["experiment_artifacts_manifest"] = []
    state["experiment_summary"] = None
    _publish_active_tasks(state, runtime)
    return runtime


def approve_plan(state: MutableMapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime(state)
    if runtime["phase"] != "awaiting_review":
        raise ExperimentRuntimeError("invalid_phase", f"Plan approval requires awaiting_review, got {runtime['phase']!r}.")
    if (runtime.get("critique") or {}).get("verdict") != "approve":
        raise ExperimentRuntimeError("critique_revise", "Plan cannot be approved while deterministic critique requires revision.")
    runtime["approved"] = True
    runtime["phase"] = "execution"
    state["experiment_plan_revision_count"] = 0
    state["experiment_inventory_blocker_hits"] = 0
    refresh_readiness(runtime)
    # Same reason as mark_result_review: ADK records a state delta on assignment
    # to a top-level key, never on a nested mutation. Without this line the
    # approval above stays invisible to the caller and the plan is re-approved.
    state[RUNTIME_KEY] = runtime
    _publish_active_tasks(state, runtime)
    return {"status": "success", "phase": runtime["phase"], "plan_id": runtime["plan_id"]}


_TASK_VIEW_FIELDS = ("status", "current_route", "planned_route", "last_message")


def get_experiment_plan(state: MutableMapping[str, Any]) -> dict[str, Any]:
    """The executor's control-plane view: what exists, what state it is in, what
    can start now.

    The executor prompt orders a get_experiment_plan after every record_result,
    and this used to hand back a deep copy of the whole plan and the whole task
    runtime — every attempt and every attempt's tool scope with full JSON
    schemas. In a long run that is the single largest repeated payload in the
    conversation, and none of it is what the executor decides on: measured
    2026-09-01, the executor's prompt grew from 7 312 to 112 134 tokens across
    one run, and the module accounted for 92% of the run's input tokens.
    """
    runtime = _runtime(state)
    plan = runtime.get("plan") or {}
    tasks = runtime.get("tasks") or {}
    rounds = int(state.get(REPLAN_ROUNDS_KEY) or 0)
    budget = int(get_settings().experiments.max_replan_rounds)

    tasks_view: dict[str, Any] = {}
    ready: list[str] = []
    for task_id in runtime.get("task_order") or []:
        task_runtime = tasks.get(task_id)
        if not isinstance(task_runtime, dict):
            continue
        task = task_runtime.get("task") or {}
        row = {key: task_runtime.get(key) for key in _TASK_VIEW_FIELDS}
        row["name"] = task.get("name")
        row["depends_on"] = list(task.get("depends_on") or [])
        row["optional"] = bool(task.get("optional"))
        row["attempts"] = len(task_runtime.get("attempt_order") or [])
        tasks_view[task_id] = row
        if row["status"] == "ready":
            ready.append(task_id)

    view: dict[str, Any] = {
        "status": "success",
        "phase": runtime["phase"],
        "approved": runtime["approved"],
        # The executor is the only agent that can act on a replan, so it is the
        # one that has to see the budget: `replan_requested` used to be written
        # and read by nobody, which made a redo indistinguishable from a first
        # pass and hid how many rounds had already been spent.
        "replan_rounds": rounds,
        "max_replan_rounds": budget,
        "replan_rounds_remaining": max(budget - rounds, 0),
        # Plan header only: the task bodies are in the plan the human approved
        # and come back per task from start_task.
        "plan": {
            key: plan.get(key)
            for key in ("plan_id", "experiment_run_id", "revision", "goal",
                        "hypothesis", "total_est_duration_min")
            if plan.get(key) is not None
        },
        "tasks": tasks_view,
        "ready": ready,
    }
    if rounds:
        view["replan_reason"] = runtime.get("result_review_feedback")
    if runtime.get("replan_exhausted"):
        view["replan_exhausted"] = True
    return view


def generate_presigned_s3_url(bucket: str, s3_key: str, expiration: int) -> str:
    """Fresh input URL without persisting it in the approved plan."""
    app = get_settings()
    if not app.s3.use_s3:
        raise ExperimentRuntimeError("s3_unavailable", f"Cannot resolve required S3 input s3://{bucket}/{s3_key}: S3 is disabled.")
    import boto3
    client = boto3.client(
        "s3",
        endpoint_url=app.s3.endpoint_url,
        aws_access_key_id=app.s3.access_key,
        aws_secret_access_key=app.s3.secret_key,
    )
    return client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": s3_key}, ExpiresIn=expiration)


def _route_timeout(settings: ExperimentsSettings, route: str) -> float:
    return {
        ExecutionRoute.FEDOT_MAS.value: settings.fedot_timeout_s,
        ExecutionRoute.REACT_TOOLS.value: settings.react_timeout_s,
        ExecutionRoute.CODER.value: settings.coder_timeout_s,
        ExecutionRoute.ALEMBIC_BUILD.value: settings.coder_timeout_s,
        ExecutionRoute.RESEARCH.value: settings.research_timeout_s,
        ExecutionRoute.MEDICAL.value: settings.medical_timeout_s,
    }[route]


_TREE_CACHE: dict[str, Any] = {}


def _config_tree() -> Any:
    """The profile the next session is built from, reparsed only when it changes.

    Not get_config(): build_for_mode() loads the YAML afresh for every session,
    so a copy loaded once per process would miss an agent removed while the
    server runs - the very edit the route check exists to see. Parsing costs
    ~0.1 s, so the parse is keyed on the files' mtimes. ``enabled`` references
    are resolved on each read, not here, so a settings change is seen at once.
    """
    from CoScientist.assembly.schema import CONFIG_DIR, load_config, resolve_config_path

    path = resolve_config_path()
    stamp = (
        str(path),
        path.stat().st_mtime_ns if path.exists() else 0,
        tuple(sorted((p.name, p.stat().st_mtime_ns) for p in CONFIG_DIR.glob("*.yaml"))),
    )
    if _TREE_CACHE.get("stamp") != stamp:
        _TREE_CACHE.update(stamp=stamp, system=load_config(path))
    return _TREE_CACHE["system"]


def _route_agent_attached(agent_name: str, system: Any = None) -> bool:
    """Whether ``agent_name`` is an enabled subordinate of the executor.

    A profile without the executor (the main system.yaml) has no Experiment
    Module to attach to, so there the agent's own ``enabled`` decides.
    """
    if system is None:
        system = _config_tree()
    if EXECUTOR_AGENT in system.agents:
        return any(a.name == agent_name for a in system.enabled_subordinates(EXECUTOR_AGENT))
    return agent_name in system.agents and system.agent(agent_name).is_enabled()


def fedot_route_available(
    settings: ExperimentsSettings | None = None, *, system: Any = None,
    route_agents: Collection[str] | None = None,
) -> bool:
    """The one answer to "may fedot_mas be planned, validated or run?".

    EXPERIMENTS__ROUTE_FEDOT is the switch, and FedotAgent has to be listed and
    enabled under ExperimentExecutorAgent. Every consumer asks this: the planner
    prompt (with ``system`` = the config being built), the planner context, the
    critique, start_task, the fallback chains and the Alembic post-build route.
    When they asked different switches, the planner went on writing fedot_mas
    for an agent the YAML had removed, and start_task handed back
    ``route_agent=FedotAgent`` for an agent that was never attached, which
    ``enforce_continue_until_reporting`` then demanded until the run stalled.
    ``route_agents``: the AgentTools of the executor the session actually runs,
    when the caller can see them - the YAML and the switch may have changed
    since that tree was built, and it is the tree that gets the work.
    Fails closed: react_tools is always there to take the task.
    """
    if not _settings(settings).route_fedot:
        return False
    agent = ROUTE_AGENT_BY_ROUTE[ExecutionRoute.FEDOT_MAS.value]
    if route_agents is not None and agent not in route_agents:
        return False
    try:
        return _route_agent_attached(agent, system)
    except Exception as exc:  # noqa: BLE001 - an unreadable config must not stop a run
        logger.warning("FEDOT.MAS route: agent tree unreadable (%s) - treating it as off", exc)
        return False


def _fedot_live(settings: ExperimentsSettings, route_agents: Collection[str] | None) -> bool:
    return fedot_route_available(settings, route_agents=route_agents)


def alembic_route_available(
    settings: ExperimentsSettings | None = None, *, system: Any = None,
    route_agents: Collection[str] | None = None,
) -> bool:
    """Whether this run may delegate an Alembic build.

    The feature switch and the assembled tree must agree.  Checking both
    prevents an approved plan or an older cached tree from bypassing a setting
    changed after that tree was built.
    """
    if not _settings(settings).route_alembic:
        return False
    agent = ROUTE_AGENT_BY_ROUTE[ExecutionRoute.ALEMBIC_BUILD.value]
    if route_agents is not None and agent not in route_agents:
        return False
    if system is None and route_agents is None:
        return True
    try:
        return _route_agent_attached(agent, system)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Alembic route: agent tree unreadable (%s) - treating it as off", exc)
        return False


def _alembic_live(
    settings: ExperimentsSettings, route_agents: Collection[str] | None,
) -> bool:
    return alembic_route_available(settings, route_agents=route_agents)


def medical_route_available(
    *, system: Any = None, route_agents: Collection[str] | None = None,
) -> bool:
    """The one answer to "may the medical route be planned, validated or run?".

    MedicalAgent listed and enabled under ExperimentExecutorAgent. The switch
    itself (MEDICAL__ENABLED, also in the web settings) is the agent's own
    ``enabled`` in the YAML, so the tree already carries it. Same consumers and
    the same ``route_agents`` as fedot_route_available, for the same reason:
    when only the runtime read the switch, the planner went on writing medical
    tasks, the critique approved them, and start_task refused them with the
    task left 'ready' for ever. Fails closed: see start_task for what then
    happens to a task planned on it.
    """
    agent = ROUTE_AGENT_BY_ROUTE[ExecutionRoute.MEDICAL.value]
    if route_agents is not None and agent not in route_agents:
        return False
    try:
        return _route_agent_attached(agent, system)
    except Exception as exc:  # noqa: BLE001 - an unreadable config must not stop a run
        logger.warning("medical route: agent tree unreadable (%s) - treating it as off", exc)
        return False


def _medical_live(route_agents: Collection[str] | None) -> bool:
    return medical_route_available(route_agents=route_agents)


def agent_tool_names(agent: Any) -> frozenset[str] | None:
    """Names of the agents attached to ``agent`` as AgentTools, or None."""
    tools = getattr(agent, "tools", None)
    if not isinstance(tools, list):
        return None
    return frozenset(
        name for tool in tools
        if isinstance(name := getattr(getattr(tool, "agent", None), "name", None), str)
    )


def session_route_agents(agent: Any) -> frozenset[str] | None:
    """The route agents on the ExperimentExecutorAgent of the tree ``agent`` runs in.

    The planner and its critique run beside the executor, not inside it. This
    lets them ask about the tree the session was built with - the one start_task
    hands work to - instead of the YAML and switches as they are now: a route
    switched on after the session was built would otherwise be planned and
    approved, then refused at start_task. None when there is no tree to read.
    """
    try:
        root = getattr(agent, "root_agent", None) or agent
        return agent_tool_names(root.find_agent(EXECUTOR_AGENT))
    except Exception:  # noqa: BLE001 - no tree means the YAML decides
        return None


def _route_enabled(route: str, settings: ExperimentsSettings) -> bool:
    if route == ExecutionRoute.FEDOT_MAS.value:
        return fedot_route_available(settings)
    if route == ExecutionRoute.ALEMBIC_BUILD.value:
        return alembic_route_available(settings)
    if route == ExecutionRoute.MEDICAL.value:
        return medical_route_available()
    return route in {
        ExecutionRoute.REACT_TOOLS.value,
        ExecutionRoute.CODER.value,
        ExecutionRoute.RESEARCH.value,
    }


def _resolve_attempt_id(runtime: dict[str, Any], task_id: str, attempt_id: str) -> str:
    """Accept verbatim ids; repair common LLM truncations of the active ATT-*."""
    active_task = runtime.get("active_task_id")
    active_attempt = runtime.get("active_attempt_id")
    if active_task == task_id and active_attempt == attempt_id:
        return attempt_id
    # Near-miss: executor often drops the last hex char of ATT-<uuid.hex>.
    if (
        active_task == task_id
        and isinstance(active_attempt, str)
        and isinstance(attempt_id, str)
        and active_attempt.startswith("ATT-")
        and attempt_id.startswith("ATT-")
        and (
            active_attempt.startswith(attempt_id)
            or attempt_id.startswith(active_attempt)
            or (
                len(active_attempt) == len(attempt_id)
                and sum(a != b for a, b in zip(active_attempt, attempt_id)) == 1
            )
        )
    ):
        _audit(
            f"EXPERIMENT_ATTEMPT_ID_REPAIRED task_id={task_id} "
            f"provided={attempt_id} active={active_attempt}"
        )
        return active_attempt
    raise ExperimentRuntimeError(
        "attempt_mismatch",
        "task_id/attempt_id do not match the active attempt."
        + (f" active_attempt_id={active_attempt!r}" if active_attempt else ""),
    )


def _resolve_inputs(
    runtime: dict[str, Any],
    task: ExperimentTask,
    *,
    route: str,
    settings: ExperimentsSettings,
    presign: Callable[[str, str, int], str],
) -> list[dict[str, Any]]:
    expiration = max(60, int(_route_timeout(settings, route) + 60))
    expires_at = (utc_now() + timedelta(seconds=expiration)).isoformat()
    resolved: list[dict[str, Any]] = []
    for data_ref in task.input_data:
        item = data_ref.model_dump(mode="json")
        try:
            if data_ref.kind == "s3":
                item["resolved_url"], item["expires_at"] = presign(str(data_ref.bucket), str(data_ref.s3_key), expiration), expires_at
            elif data_ref.kind == "task_artifact":
                artifact = find_artifact(
                    runtime,
                    str(data_ref.source_artifact_id),
                    source_task_id=str(data_ref.source_task_id) if data_ref.source_task_id else None,
                )
                if artifact.get("bucket") and artifact.get("s3_key"):
                    item["resolved_url"], item["expires_at"] = presign(artifact["bucket"], artifact["s3_key"], expiration), expires_at
                elif artifact.get("workspace_path"):
                    item["resolved_workspace_path"] = artifact["workspace_path"]
                elif artifact.get("external_url"):
                    item["resolved_url"] = artifact["external_url"]
            elif data_ref.kind == "url":
                item["resolved_url"] = str(data_ref.url)
            elif data_ref.kind == "workspace":
                item["resolved_workspace_path"] = data_ref.workspace_path
        except Exception as exc:
            if data_ref.required:
                if isinstance(exc, ExperimentRuntimeError):
                    raise
                raise ExperimentRuntimeError("input_resolution_failed", f"Could not resolve required input {data_ref.data_id!r}: {exc}") from exc
            item["resolution_warning"] = str(exc)
        resolved.append(item)
    return resolved


def force_managed_s3_launch_params(launch_params: dict[str, Any] | None, *, require: bool) -> dict[str, Any]:
    """Ensure tools whose schema offers S3 upload persist a managed artifact."""
    params = copy.deepcopy(launch_params or {})
    if require:
        params["upload_results_to_s3"] = True
        params.setdefault("output_s3_prefix", "generated")
    return clamp_generate_launch_num(params)


def generate_num_cap() -> int:
    """Max generator ``num`` from ``EXPERIMENTS__MAX_GENERATE_NUM`` (0 = off)."""
    raw = os.getenv("EXPERIMENTS__MAX_GENERATE_NUM", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def clamp_generate_launch_num(launch_params: dict[str, Any] | None) -> dict[str, Any]:
    """Cap or fill ``num`` so Fedot cannot request 100 CVAE molecules."""
    params = copy.deepcopy(launch_params or {})
    cap = generate_num_cap()
    if cap <= 0:
        return params
    current = params.get("num")
    try:
        n = int(current) if current is not None else cap
    except (TypeError, ValueError):
        n = cap
    params["num"] = min(n, cap)
    return params


def _scope_tools(task: ExperimentTask) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filtered = [
        {
            "tool": tool.name,
            "server_id": server.server_id,
            "server_name": server.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "url": str(server.url) if server.url else None,
        }
        for server in task.mcp_servers for tool in server.tools
    ]
    deployed = [
        {
            "name": server.name,
            "url": str(server.url),
            "description": "; ".join(tool.description for tool in server.tools),
            "tools": [tool.model_dump(mode="json") for tool in server.tools],
        }
        for server in task.mcp_servers if server.url
    ]
    return filtered, deployed


def start_task(
    state: MutableMapping[str, Any],
    task_id: str,
    *,
    settings: ExperimentsSettings | None = None,
    presign: Callable[[str, str, int], str] = generate_presigned_s3_url,
    route_agents: Collection[str] | None = None,
) -> dict[str, Any]:
    """Open a fresh attempt on the task's route and return its envelope.

    ``route_agents``: the route agents attached to the calling executor, when
    known. The envelope names ``route_agent`` for the executor to call, so a
    route whose agent that executor does not hold must not be handed out.
    """
    cfg = _settings(settings)
    runtime = _runtime(state)
    task_runtime = _task(runtime, task_id)

    if task_runtime["status"] in TERMINAL_TASK_STATES:
        raise ExperimentRuntimeError("task_terminal", f"Task {task_id} is already terminal.")
    if runtime["phase"] != "execution" or not runtime["approved"]:
        raise ExperimentRuntimeError("plan_not_approved", "Only an approved plan in execution may start tasks.")
    if runtime.get("active_attempt_id"):
        raise ExperimentRuntimeError("task_already_running", "v0 permits only one running task at a time.")

    refresh_readiness(runtime)
    if task_runtime["status"] != "ready":
        raise ExperimentRuntimeError("task_not_ready", f"Task {task_id} must be ready, got {task_runtime['status']!r}.")
    route = task_runtime["current_route"]
    if _attempts_for_route(task_runtime, route) >= cfg.task_max_attempts:
        raise ExperimentRuntimeError(
            "attempt_budget_exhausted",
            f"Task {task_id} exhausted its {cfg.task_max_attempts} attempts on route {route!r}.",
        )

    # After the budget check, on purpose. Checked against react_tools instead, a
    # task whose react_tools attempts were spent would raise here with status
    # still 'ready' - which neither retry_task nor fallback_task accepts, and
    # the executor would be driven to start_task until the run gave out. The
    # price of this order is at most one react_tools attempt over budget, and
    # only when FEDOT went away after the fallback chose it.
    if route == ExecutionRoute.FEDOT_MAS.value and not _fedot_live(cfg, route_agents):
        route = ExecutionRoute.REACT_TOOLS.value
        task_runtime["current_route"] = route
        task_runtime["route_history"].append({
            "route": route,
            "reason": "FEDOT unavailable (EXPERIMENTS__ROUTE_FEDOT off or FedotAgent "
                      "not attached to ExperimentExecutorAgent)",
        })
    if route == ExecutionRoute.ALEMBIC_BUILD.value and not _alembic_live(cfg, route_agents):
        exc = ExperimentRuntimeError(
            "route_disabled",
            "Route 'alembic_build' is switched off: McpBuilderAgent is not in "
            "this run (EXPERIMENTS__ROUTE_ALEMBIC off, or not attached to "
            "ExperimentExecutorAgent).",
        )
        _block_unstartable(state, task_id, exc)
        raise exc
    if route == ExecutionRoute.MEDICAL.value and not _medical_live(route_agents):
        planned = task_runtime["planned_route"]
        if planned != ExecutionRoute.MEDICAL.value and _route_live(planned, cfg, route_agents):
            # The runtime put it on medical (a coder task naming a medical tool,
            # or a fallback), not the plan: go back to the route that was planned.
            if (original := task_runtime.pop(_PRE_FAMILY_REWRITE_KEY, None)) is not None:
                task_runtime["task"] = original
            route = planned
            task_runtime["current_route"] = route
            task_runtime["route_history"].append({
                "route": route,
                "reason": "medical unavailable (MEDICAL__ENABLED off or MedicalAgent not "
                          "attached to ExperimentExecutorAgent); back to the planned route",
            })
        else:
            # Planned on medical, and nothing stands in for MedicalAgent (PICO,
            # DICOM): blocked - terminal, reported - rather than left 'ready' for
            # a start_task that can never succeed.
            exc = ExperimentRuntimeError(
                "route_disabled",
                "Route 'medical' is switched off: MedicalAgent is not in this run "
                "(MEDICAL__ENABLED off, or not attached to ExperimentExecutorAgent).",
            )
            _block_unstartable(state, task_id, exc)
            raise exc
    if not _route_enabled(route, cfg):
        raise ExperimentRuntimeError("route_disabled", f"Route {route!r} is disabled for Experiment Module v0.")

    task_model = ExperimentTask.model_validate(task_runtime["task"])
    allowed_hypotheses = {
        str(row.get("hypothesis_id") or "").strip().upper()
        for row in ((state.get("experiment_context") or {}).get("hypothesis_refs") or [])
        if isinstance(row, dict) and row.get("hypothesis_id")
    }
    task_hypotheses = {
        task_model.design.hypothesis_ref.strip().upper(),
        *(str(item).strip().upper() for item in task_model.design.also_tests),
    }
    blocked_hypotheses = sorted(task_hypotheses - allowed_hypotheses) if allowed_hypotheses else []
    if blocked_hypotheses:
        raise ExperimentRuntimeError(
            "hypothesis_not_eligible",
            "Task references hypotheses that are not eligible in this experiment turn: "
            + ", ".join(blocked_hypotheses),
        )
    if (
        route == ExecutionRoute.CODER.value
        and task_model.code_assessment.requirement == CodeRequirement.UNKNOWN
        and not task_model.repo_url
        and not mcp_routes_tried(task_runtime)
    ):
        from CoScientist.experiments.capabilities.inventory import (
            FAMILY_MEDICAL,
            FAMILY_RESEARCH,
            match_named_family_capability,
        )

        blob = task_coverage_blob(state, task_model)
        # A switched-off medical family is no rewrite target: the rewrite would
        # only be refused as route_disabled one line later.
        families = {FAMILY_RESEARCH} | ({FAMILY_MEDICAL} if _medical_live(route_agents) else set())
        if family_hit := match_named_family_capability(blob, families=families):
            route = str(family_hit["family"])
            if not _route_enabled(route, cfg):
                raise ExperimentRuntimeError(
                    "route_disabled", f"Route {route!r} is disabled for Experiment Module v0.",
                )
            dumped = task_model.model_dump(mode="json")
            dumped["route"] = route
            dumped["mcp_servers"] = []
            dumped.pop("post_build_route", None)
            for art in (dumped.get("design") or {}).get("analysis_artifacts") or []:
                if isinstance(art, dict):
                    art["prepare_via"] = route
                    if family_hit.get("tool"):
                        art["path_or_tool"] = family_hit["tool"]
            task_model = ExperimentTask.model_validate(dumped)
            # Kept so the rewrite can be undone if its route goes away later.
            task_runtime.setdefault(_PRE_FAMILY_REWRITE_KEY, copy.deepcopy(task_runtime["task"]))
            task_runtime["task"] = task_model.model_dump(mode="json")
            task_runtime["current_route"] = route
            task_runtime["route_history"].append({
                "route": route,
                "reason": f"named_family_rewrote_coder:{family_hit.get('tool')}",
            })
            _audit(f"EXPERIMENT_CODER_REWRITTEN_TO_FAMILY task_id={task_id} route={route}")
        elif session_inventory_nonempty(state) and (
            matched := match_session_inventory_tool(state, task_model, blob)
        ):
            # One bound tool is ExperimentAgent's job. FEDOT.MAS is never a
            # default route, so an implicit rewrite never picks it.
            route = ExecutionRoute.REACT_TOOLS.value
            if not _route_enabled(route, cfg):
                raise ExperimentRuntimeError("route_disabled", f"Route {route!r} is disabled for Experiment Module v0.")
            task_runtime["current_route"] = route
            task_runtime["route_history"].append({"route": route, "reason": "inventory_rewrote_coder"})
            url = str(matched.get("url") or "").strip() or None
            dumped = task_model.model_dump(mode="json")
            dumped["route"] = route
            dumped["mcp_servers"] = [{
                "name": matched["server_id"],
                "server_id": matched["server_id"],
                "url": url,
                "tools": [{"name": matched["tool"], "input_schema": matched.get("input_schema")}],
                "source": "registry",
                "health": "unknown",
            }]
            task_model = ExperimentTask.model_validate(dumped)
            task_runtime["task"] = task_model.model_dump(mode="json")
            _audit(f"EXPERIMENT_CODER_REWRITTEN_TO_INVENTORY task_id={task_id} route={route}")
    if route == ExecutionRoute.CODER.value and task_runtime["planned_route"] == ExecutionRoute.CODER.value and task_model.mcp_servers and not cfg.route_coder_mcp:
        raise ExperimentRuntimeError("route_disabled", "Direct MCP-to-Coder mode is disabled.")

    launch_params = task_model.launch_params
    if task_requires_managed_s3(task_model):
        launch_params = force_managed_s3_launch_params(launch_params, require=True)
    else:
        launch_params = clamp_generate_launch_num(launch_params)
    if launch_params != (task_model.launch_params or {}):
        task_model = task_model.model_copy(update={"launch_params": launch_params})
        task_runtime["task"] = task_model.model_dump(mode="json")

    attempt_no = len(task_runtime["attempt_order"]) + 1
    attempt_id = f"ATT-{uuid4().hex}"
    filtered_tools, deployed_mcps = _scope_tools(task_model)
    if route == ExecutionRoute.CODER.value and not cfg.route_coder_mcp:
        filtered_tools, deployed_mcps = [], []
    if route in {ExecutionRoute.RESEARCH.value, ExecutionRoute.MEDICAL.value}:
        filtered_tools, deployed_mcps = [], []
    try:
        resolved_inputs = _resolve_inputs(runtime, task_model, route=route, settings=cfg, presign=presign)
        if route == ExecutionRoute.CODER.value and not resolved_inputs and task_model.depends_on:
            from CoScientist.experiments.schemas import DataRef
            synthetic: list[Any] = []
            for dep in task_model.depends_on:
                for result in reversed(runtime.get("results") or []):
                    if result.get("task_id") != dep:
                        continue
                    for art in result.get("artifacts") or []:
                        if not isinstance(art, dict):
                            continue
                        name = str(art.get("name") or "").strip()
                        if not name:
                            continue
                        synthetic.append(DataRef(
                            data_id=name,
                            kind="task_artifact",
                            description=f"Upstream artifact from {dep}",
                            source_task_id=dep,
                            source_artifact_id=name,
                            required=True,
                        ))
                    break
            if synthetic:
                task_model = task_model.model_copy(update={"input_data": synthetic})
                resolved_inputs = _resolve_inputs(
                    runtime, task_model, route=route, settings=cfg, presign=presign,
                )
    except ExperimentRuntimeError as exc:
        if exc.code in {"artifact_not_found", "input_resolution_failed"}:
            _block_unstartable(state, task_id, exc)
        raise
    from CoScientist.tools.fedot_artifact_handoff import seed_upstream_from_resolved_inputs

    upstream_bindings = seed_upstream_from_resolved_inputs(
        state, resolved_inputs, filtered_tools
    )
    if route == ExecutionRoute.CODER.value:
        from CoScientist.experiments.runtime.coder_artifacts import seed_coder_upstream_inputs
        try:
            seed_coder_upstream_inputs(state, resolved_inputs)
        except ExperimentRuntimeError as exc:
            if exc.code == "coder_input_missing":
                _block_unstartable(state, task_id, exc)
            raise
    started_at = utc_now().isoformat()
    attempt = {
        "attempt_id": attempt_id,
        "attempt_no": attempt_no,
        "status": "running",
        "route": route,
        "route_returned": False,
        "started_at": started_at,
        "artifact_cursor": {key: len(state.get(key) or []) for key in ARTIFACT_KEYS} | {"workspace_started_at": started_at},
        "tool_scope": {"filtered_tools": copy.deepcopy(filtered_tools), "deployed_mcps": copy.deepcopy(deployed_mcps)},
    }
    task_runtime["attempts"][attempt_id] = attempt
    task_runtime["attempt_order"].append(attempt_id)
    task_runtime["status"] = "running"
    runtime["active_task_id"] = task_id
    runtime["active_attempt_id"] = attempt_id

    envelope = {
        "plan_id": runtime["plan_id"],
        "experiment_run_id": runtime["run_id"],
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_no": attempt_no,
        "route": route,
        "route_agent": ROUTE_AGENT_BY_ROUTE[route],
        "task": task_model.model_dump(mode="json"),
        "resolved_inputs": resolved_inputs,
        "upstream_bindings": upstream_bindings,
    }
    state["experiment_active_envelope"] = envelope
    state["filtered_tools"] = filtered_tools
    state["deployed_mcps"] = deployed_mcps
    _publish_active_tasks(state, runtime)
    _audit(f"EXPERIMENT_TASK_STARTED task_id={task_id} attempt_id={attempt_id} route={route}")
    return {"status": "success", **copy.deepcopy(envelope)}


def active_attempt(state: MutableMapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime = _runtime(state)
    task_id, attempt_id = runtime.get("active_task_id"), runtime.get("active_attempt_id")
    if not task_id or not attempt_id:
        raise ExperimentRuntimeError("attempt_missing", "No route attempt is active.")
    task_runtime = _task(runtime, task_id)
    if not isinstance(attempt := task_runtime["attempts"].get(attempt_id), dict):
        raise ExperimentRuntimeError("attempt_missing", "Active attempt is missing.")
    return runtime, task_runtime, attempt


def mark_route_returned(state: MutableMapping[str, Any], route_agent: str) -> None:
    runtime, _, attempt = active_attempt(state)
    if route_agent != (expected := ROUTE_AGENT_BY_ROUTE.get(attempt["route"])):
        raise ExperimentRuntimeError("route_mismatch", f"Attempt expects {expected}, not {route_agent}.")
    attempt["route_returned"] = True
    attempt["route_agent"] = route_agent
    runtime["last_route_agent"] = route_agent


def _route_live(route: str, settings: ExperimentsSettings, route_agents: Collection[str] | None) -> bool:
    """_route_enabled, with FEDOT and medical narrowed to the executor that is running."""
    if route == ExecutionRoute.FEDOT_MAS.value:
        return _fedot_live(settings, route_agents)
    if route == ExecutionRoute.MEDICAL.value:
        return _medical_live(route_agents)
    return _route_enabled(route, settings)


def _next_fallback(
    task_runtime: dict[str, Any],
    settings: ExperimentsSettings | None = None,
    route_agents: Collection[str] | None = None,
) -> str | None:
    cfg = _settings(settings)
    chain = resolve_fallback_chains(cfg)[task_runtime["planned_route"]]
    if (index := chain.index(task_runtime["current_route"]) if task_runtime["current_route"] in chain else -1) < 0:
        return None
    used = {entry["route"] for entry in task_runtime["route_history"]}
    # A switched-off route is skipped, not offered: fallback_task would refuse
    # it as route_disabled and leave the task stuck in fallback_pending.
    return next(
        (r for r in chain[index + 1 :] if r not in used and _route_live(r, cfg, route_agents)), None,
    )


def _attempts_for_route(task_runtime: dict[str, Any], route: str) -> int:
    """Count attempts already spent on one route (task_max_attempts is per-route)."""
    attempts = task_runtime.get("attempts") or {}
    return sum(
        1
        for aid in task_runtime.get("attempt_order") or []
        if str((attempts.get(aid) or {}).get("route") or "") == route
    )


def _store_result(
    state: MutableMapping[str, Any],
    runtime: dict[str, Any],
    result: TaskResult,
) -> dict[str, Any]:
    result_json = result.model_dump(mode="json")
    runtime["results"].append(result_json)
    state["experiment_task_results"] = copy.deepcopy(runtime["results"])
    # Lazy import: review → runtime at module load; avoid cycle.
    from CoScientist.experiments.review import build_experiment_artifacts_manifest

    state["experiment_artifacts_manifest"] = build_experiment_artifacts_manifest(state)
    return result_json


def record_result(
    state: MutableMapping[str, Any],
    task_id: str,
    attempt_id: str,
    result: dict[str, Any],
    *,
    settings: ExperimentsSettings | None = None,
    route_agents: Collection[str] | None = None,
) -> dict[str, Any]:
    cfg = _settings(settings)
    runtime, task_runtime, attempt = active_attempt(state)
    attempt_id = _resolve_attempt_id(runtime, task_id, attempt_id)
    if attempt["status"] != "running":
        raise ExperimentRuntimeError("attempt_terminal", "Attempt is already terminal.")
    if not attempt.get("route_returned"):
        raise ExperimentRuntimeError("route_not_returned", "The route agent must return before record_result.")

    # Coerce common LLM synonyms before the closed-enum check.
    raw_status = str(result.get("status") or "").strip().lower().replace("-", "_")
    if raw_status in _RESULT_STATUS_ALIASES:
        coerced = _RESULT_STATUS_ALIASES[raw_status]
        patch: dict[str, Any] = {"status": coerced}
        if coerced == "failure":
            patch["retryable"] = bool(result.get("retryable", True))
        result = {**result, **patch}

    if (status := result.get("status")) not in {"success", "partial", "failure"}:
        raise ExperimentRuntimeError("result_status", "Result status must be success, partial, or failure.")

    if str(attempt.get("route") or "") == ExecutionRoute.ALEMBIC_BUILD.value:
        job_id = str(attempt.get("alembic_job_id") or "").strip()
        if job_id:
            from CoScientist.tools.alembic_tools import peek_mcp_build

            snap = peek_mcp_build(job_id)
            if snap.get("status") == "running":
                raise ExperimentRuntimeError(
                    "alembic_build_running",
                    f"Alembic job {job_id} is still running; do not record_result "
                    "until the build is done or failed. Reuse this job_id — do not fallback to coder.",
                )

    if str(attempt.get("route") or "") != ExecutionRoute.ALEMBIC_BUILD.value:
        result = _downgrade_fabricated_success(result)
    result = _coerce_alembic_mcp_success(attempt, result, task=task_runtime.get("task"))
    status = result["status"]

    task = ExperimentTask.model_validate(task_runtime["task"])
    known_criteria = {c.criterion_id for c in task.success_criteria}
    raw_checks = [
        dict(item) if isinstance(item, dict) else item
        for item in (result.get("criteria_checks") or [])
    ]
    for check in raw_checks:
        if not isinstance(check, dict):
            continue
        cid = str(check.get("criterion_id") or "").strip()
        if cid in known_criteria:
            continue
        prefixed = f"{task.id}-{cid}"
        if prefixed in known_criteria:
            check["criterion_id"] = prefixed
        elif len(known_criteria) == 1:
            check["criterion_id"] = next(iter(known_criteria))
    result = {**result, "criteria_checks": raw_checks}
    checks = [CriterionCheck.model_validate(item) for item in raw_checks]
    raw_artifacts = captured_delta(state, attempt)
    raw_artifacts.extend(copy.deepcopy(item) for item in (result.get("artifacts") or []) if isinstance(item, dict))
    outputs = result.get("outputs") or {}
    if isinstance(outputs, dict) and outputs:
        from CoScientist.experiments.runtime.inline_artifacts import materialize_outputs_as_artifacts
        raw_artifacts.extend(
            materialize_outputs_as_artifacts(
                task_id=task_id,
                attempt_id=attempt_id,
                expected_artifacts=[item.model_dump(mode="json") for item in task.expected_artifacts],
                outputs=outputs,
                existing=raw_artifacts,
            )
        )

    attempt_route = str(attempt.get("route") or task_runtime.get("current_route") or "")
    if attempt_route in EVIDENCE_AGENT_ROUTES and (
        attempt.get("family_tool_called") or status in {"success", "partial"}
    ):
        append_notes_artifact(
            task=task, attempt=attempt, raw_artifacts=raw_artifacts,
            text=route_response_text(state, result),
        )

    artifacts, artifact_warnings = normalise_artifacts(raw_artifacts, runtime=runtime, task_runtime=task_runtime,
                                                        attempt=attempt, state=state)
    if (
        status in {"success", "partial"}
        and str(task_runtime.get("planned_route") or "") == ExecutionRoute.ALEMBIC_BUILD.value
    ):
        # The served address is the build's deliverable: name it the way the
        # planner did, so consumers that list it as task_artifact input resolve.
        from CoScientist.experiments.runtime.alembic_bridge import (
            harvest_alembic_mcp_url,
            mcp_url_from_task_runtime,
        )

        served = harvest_alembic_mcp_url(
            outputs if isinstance(outputs, dict) else {},
            result.get("summary"),
            attempt.get("alembic_snapshot"),
        ) or mcp_url_from_task_runtime(task_runtime)
        artifacts = synthesize_mcp_server_artifacts(
            task, artifacts, mcp_url=served, runtime=runtime, attempt=attempt,
        )
    artifacts_ok, missing_artifacts = required_artifacts_present(task, artifacts, route=attempt_route)
    criteria_ok, failed_criteria = criteria_valid(task, checks, route=attempt_route)
    core_failure = _is_core_execution_failure(result)
    material_partial = status == "partial" and (
        core_failure or not artifacts_ok or not criteria_ok
    )
    if material_partial:
        # Partial means the primary operation completed and only non-core gaps
        # remain.  If required evidence is absent, keep the evidence we did get
        # but drive the normal retry/fallback chain instead of closing green.
        status = "failure"
        result = {
            **result,
            "status": "failure",
            "error_code": result.get("error_code") or "partial_missing_core_evidence",
            "error_message": result.get("error_message") or (
                "partial result lacks required primary-operation evidence"
            ),
            "retryable": bool(result.get("retryable", False)),
            "warnings": [
                *(result.get("warnings") or []),
                "partial_promoted_to_failure: retry/fallback required for missing core output",
            ],
        }
    durable_ok = has_durable_family_evidence(
        task, artifacts, route=attempt_route,
        outputs=outputs if isinstance(outputs, dict) else {},
    )
    if durable_ok:
        checks = attest_durable_criteria(task, checks)
        result = {**result, "criteria_checks": [c.model_dump(mode="json") for c in checks]}
        if not artifacts_ok:
            artifacts_ok, missing_artifacts = True, []
            artifact_warnings.append(
                "accepted_via_durable_family_evidence: S3/file/mcp_url present; "
                "planner artifact names are not required."
            )
        criteria_ok, failed_criteria = criteria_valid(task, checks, route=attempt_route)
        if status == "failure" and not core_failure and not material_partial and criteria_ok and artifacts_ok:
            # Ярлык оправдан: доказательство действительно есть, и называть это
            # полным провалом неверно. Но retryable=False здесь был отдельной,
            # незаметной потерей: провал получал право не повторяться.
            # Измерено 04.09.2026 на восьми прогонах — докинг падал по таймауту,
            # к попытке прилагался артефакт-заглушка family_outputs.json, статус
            # переписывался в partial, ретрай глушился, headless-ревью закрывало
            # прогон зелёным. Ни один из восьми докингов не оценил собственные
            # молекулы прогона, и ни один не был повторён.
            # Ярлык оставляем, право на повтор — за исходным результатом.
            status = "partial"
            result = {
                **result,
                "status": "partial",
                "error_code": None,
                "retryable": result.get("retryable", True),
                "warnings": [
                    *(result.get("warnings") or []),
                    "accepted_via_durable_family_evidence: relabeled failure after real evidence",
                ],
            }
            artifact_warnings.append(
                "accepted_via_durable_family_evidence: relabeled failure after real evidence"
            )
    if status == "success" and any(c.passed is not True for c in checks):
        status = "failure"
        result = {
            **result,
            "status": "failure",
            "error_code": result.get("error_code") or "criteria_failed",
            "error_message": result.get("error_message") or (
                "success requires all supplied criteria checks to pass"
            ),
            "retryable": True,
        }
    if status in {"success", "partial"} and (not criteria_ok or not artifacts_ok):
        raise ExperimentRuntimeError(
            "result_incomplete",
            f"A successful/partial result is missing required evidence: criteria={failed_criteria}, artifacts={missing_artifacts}.",
        )

    task_result = TaskResult.model_validate({
        "schema_version": "task-result/0.1",
        "result_id": result.get("result_id") or f"RES-{uuid4().hex}",
        "plan_id": runtime["plan_id"],
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_no": attempt["attempt_no"],
        "status": status,
        "planned_route": task_runtime["planned_route"],
        "route_used": attempt["route"],
        "started_at": attempt["started_at"],
        "finished_at": utc_now(),
        "summary": result.get("summary") or f"{task.name}: {status}",
        "outputs": outputs if isinstance(outputs, dict) else {},
        "artifacts": artifacts,
        "criteria_checks": checks,
        "error_code": result.get("error_code"),
        "error_message": result.get("error_message"),
        "retryable": bool(result.get("retryable", False)),
        "warnings": [*(result.get("warnings") or []), *artifact_warnings],
    })
    result_json = _store_result(state, runtime, task_result)

    attempt["status"] = status
    attempt["result_id"] = task_result.result_id
    task_runtime["last_message"] = task_result.summary
    post_build: dict[str, Any] | None = None
    if status == "success":
        if attempt["route"] == ExecutionRoute.ALEMBIC_BUILD.value:
            from CoScientist.experiments.runtime.alembic_bridge import (
                apply_alembic_success,
                harvest_alembic_mcp_url,
            )

            mcp_url = harvest_alembic_mcp_url(
                outputs if isinstance(outputs, dict) else {},
                result.get("summary"),
                attempt.get("alembic_snapshot"),
                repo_url=str(task.repo_url or "").strip() or None,
            )
            if not mcp_url:
                raise ExperimentRuntimeError(
                    "alembic_mcp_url_missing",
                    "Alembic success requires outputs.mcp_url before post_build_route can continue.",
                )
            if not task.post_build_route:
                raise ExperimentRuntimeError(
                    "alembic_post_build_missing",
                    "Alembic success requires post_build_route on the task.",
                )
            post_build = apply_alembic_success(
                state,
                runtime,
                task_runtime,
                mcp_url=mcp_url,
                outputs=outputs if isinstance(outputs, dict) else {},
                settings=cfg,
            )
        else:
            task_runtime["status"] = "done"
    elif status == "partial":
        task_runtime["status"] = "done_with_warnings"
    else:
        route = str(task_runtime.get("current_route") or "")
        attempts_left = _attempts_for_route(task_runtime, route) < cfg.task_max_attempts
        next_fb = _next_fallback(task_runtime, cfg, route_agents)
        # Same-route retries first; else next route in resolve_fallback_chains().
        if task_result.retryable and attempts_left:
            task_runtime["status"] = "retry_pending"
        elif next_fb is not None:
            task_runtime["status"] = "fallback_pending"
        else:
            task_runtime["status"] = "failed"

    _sync_after_mutation(state, runtime, clear_active=True)
    if post_build:
        # clear_active nulls deployed_mcps; restore Alembic servers for post_build start_task.
        state["deployed_mcps"] = copy.deepcopy(
            (task_runtime.get("task") or {}).get("mcp_servers") or []
        )
    managed = sum(1 for a in result_json["artifacts"] if a.get("bucket") and a.get("s3_key"))
    _audit(
        f"EXPERIMENT_RECORD_RESULT_SUCCESS task_id={task_id} attempt_id={attempt_id} "
        f"result_status={status} phase={runtime['phase']} artifacts={len(result_json['artifacts'])} "
        f"managed_artifacts={managed}"
        + (f" post_build_route={post_build.get('post_build_route')}" if post_build else "")
    )
    response = {"status": "success", "task_result": result_json, "phase": runtime["phase"]}
    if material_partial:
        response.update({
            "downgraded_from": "partial",
            "downgrade_reason": "partial_missing_core_evidence",
        })
    if post_build:
        response["post_build"] = post_build
    return response


def retry_task(
    state: MutableMapping[str, Any],
    task_id: str,
    *,
    settings: ExperimentsSettings | None = None,
) -> dict[str, Any]:
    cfg = _settings(settings)
    runtime = _runtime(state)
    task_runtime = _task(runtime, task_id)
    if task_runtime["status"] != "retry_pending":
        raise ExperimentRuntimeError("retry_not_allowed", "retry_task requires a retryable failed attempt.")
    route = str(task_runtime.get("current_route") or "")
    if _attempts_for_route(task_runtime, route) >= cfg.task_max_attempts:
        raise ExperimentRuntimeError("attempt_budget_exhausted", f"Retry budget exhausted on route {route!r}.")
    task_runtime["status"] = "ready"
    task_runtime["last_message"] = "Retry approved; start_task will create a new attempt."
    _publish_active_tasks(state, runtime)
    return {"status": "success", "task_id": task_id, "route": task_runtime["current_route"]}


def fallback_task(
    state: MutableMapping[str, Any],
    task_id: str,
    reason: str,
    *,
    settings: ExperimentsSettings | None = None,
    route_agents: Collection[str] | None = None,
) -> dict[str, Any]:
    cfg = _settings(settings)
    runtime = _runtime(state)
    task_runtime = _task(runtime, task_id)
    if task_runtime["status"] != "fallback_pending":
        raise ExperimentRuntimeError("fallback_not_allowed", "fallback_task requires fallback_pending state.")
    if str(task_runtime.get("current_route") or "") == ExecutionRoute.ALEMBIC_BUILD.value:
        job_id = ""
        for aid in reversed(task_runtime.get("attempt_order") or []):
            att = (task_runtime.get("attempts") or {}).get(aid) or {}
            if str(att.get("alembic_job_id") or "").strip():
                job_id = str(att["alembic_job_id"]).strip()
                break
        if job_id:
            from CoScientist.tools.alembic_tools import peek_mcp_build

            snap = peek_mcp_build(job_id)
            if snap.get("status") == "running":
                raise ExperimentRuntimeError(
                    "alembic_build_running",
                    f"Alembic job {job_id} is still running; stay on alembic_build "
                    "(retry_task), do not fallback to coder.",
                )
            live = str(snap.get("mcp_url") or "").strip()
            if snap.get("status") == "done" and live.startswith("http"):
                from CoScientist.experiments.runtime.alembic_bridge import apply_alembic_success

                post = apply_alembic_success(
                    state, runtime, task_runtime, mcp_url=live,
                    outputs={"mcp_url": live, "mcp_endpoint": live},
                    settings=cfg,
                )
                _sync_after_mutation(state, runtime, clear_active=True)
                state["deployed_mcps"] = copy.deepcopy(
                    (task_runtime.get("task") or {}).get("mcp_servers") or []
                )
                return {
                    "status": "success",
                    "task_id": task_id,
                    "route": post["post_build_route"],
                    "next_action": "start_task",
                    "must_start_task_id": task_id,
                    "post_build": post,
                    "message": (
                        f"Alembic MCP ready at {live}; continuing via "
                        f"{post['post_build_route']}. Call start_task('{task_id}') next."
                    ),
                }
    route = _next_fallback(task_runtime, cfg, route_agents)
    if route is None or route == ExecutionRoute.CODER.value:
        from CoScientist.experiments.runtime.alembic_bridge import mcp_url_from_task_runtime

        if mcp_url := mcp_url_from_task_runtime(task_runtime):
            raise ExperimentRuntimeError(
                "alembic_mcp_ready",
                f"Alembic MCP is already served at {mcp_url}; do not fallback to "
                "coder. Retry the post-build route or record an honest failure.",
            )
    if route is None:
        raise ExperimentRuntimeError("fallback_exhausted", "No acyclic fallback route remains.")
    if route == ExecutionRoute.CODER.value:
        if runtime_has_durable_data_evidence(runtime, task_id):
            raise ExperimentRuntimeError(
                "evidence_already_present",
                "Durable family evidence already exists for this task; "
                "do not fallback to coder. record_result(success) instead.",
            )
    if not _route_enabled(route, cfg):
        raise ExperimentRuntimeError("route_disabled", f"Fallback route {route!r} is disabled.")
    task_runtime["current_route"] = route
    task_runtime["route_history"].append({"route": route, "reason": reason})
    task_runtime["status"] = "ready"
    task_runtime["last_message"] = f"Fallback to {route}: {reason}"
    _publish_active_tasks(state, runtime)
    return {
        "status": "success",
        "task_id": task_id,
        "route": route,
        "next_action": "start_task",
        "must_start_task_id": task_id,
        "message": f"Fallback ready on {route}. Call start_task({task_id!r}) next — same task only.",
    }


def _complete_as_skipped(
    state: MutableMapping[str, Any], task_id: str, reason: str,
) -> dict[str, Any]:
    runtime = _runtime(state)
    task_runtime = _task(runtime, task_id)
    attempt_id = f"ATT-{uuid4().hex}"
    now = utc_now()
    result = TaskResult(
        schema_version="task-result/0.1",
        result_id=f"RES-{uuid4().hex}",
        plan_id=runtime["plan_id"],
        task_id=task_id,
        attempt_id=attempt_id,
        attempt_no=len(task_runtime["attempt_order"]) + 1,
        status="skipped",
        planned_route=task_runtime["planned_route"],
        route_used=task_runtime["current_route"],
        started_at=now,
        finished_at=now,
        summary=reason,
        criteria_checks=[],
    )
    task_runtime["attempt_order"].append(attempt_id)
    task_runtime["attempts"][attempt_id] = {
        "attempt_id": attempt_id,
        "attempt_no": result.attempt_no,
        "status": "skipped",
        "route": task_runtime["current_route"],
        "route_returned": False,
        "started_at": now.isoformat(),
        "result_id": result.result_id,
    }
    task_runtime["status"] = "skipped"
    task_runtime["last_message"] = reason
    result_json = _store_result(state, runtime, result)
    _sync_after_mutation(state, runtime)
    return {"status": "success", "task_result": result_json}


def skip_task(
    state: MutableMapping[str, Any], task_id: str, reason: str
) -> dict[str, Any]:
    runtime = _runtime(state)
    task_runtime = _task(runtime, task_id)
    task = ExperimentTask.model_validate(task_runtime["task"])
    if not task.optional:
        raise ExperimentRuntimeError("skip_required", "Only optional v0 tasks may be skipped without human amendment.")
    if task_runtime["status"] not in {"pending", "ready"}:
        raise ExperimentRuntimeError("skip_not_allowed", f"Cannot skip task in {task_runtime['status']!r} state.")
    return _complete_as_skipped(state, task_id, reason)


def amend_task(
    state: MutableMapping[str, Any],
    task_id: str,
    patch: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    runtime = _runtime(state)
    task_runtime = _task(runtime, task_id)
    if task_runtime["status"] not in {"pending", "ready"}:
        raise ExperimentRuntimeError("amend_not_allowed", "Only pending/ready v0 tasks may be amended.")
    if unknown := set(patch) - _AMEND_FIELDS:
        raise ExperimentRuntimeError("amend_fields", f"Unsupported amendment fields: {sorted(unknown)}.")
    amended = copy.deepcopy(task_runtime["task"])
    amended.update(copy.deepcopy(patch))
    task = ExperimentTask.model_validate(amended)
    task_runtime["task"] = task.model_dump(mode="json")
    original = task_runtime.get(_PRE_FAMILY_REWRITE_KEY)
    if original is not None and "route" not in patch:
        # The runtime's family rewrite stays in force and the planned route with
        # it; the amendment is carried onto the planned copy, so a later return
        # to that route keeps it.
        try:
            task_runtime[_PRE_FAMILY_REWRITE_KEY] = ExperimentTask.model_validate(
                {**copy.deepcopy(original), **copy.deepcopy(patch)}
            ).model_dump(mode="json")
        except ValueError:
            original = None
    if original is None or "route" in patch:
        # A new route is a new plan for this task: nothing earlier to go back to.
        task_runtime.pop(_PRE_FAMILY_REWRITE_KEY, None)
        task_runtime["planned_route"] = task.route.value
    task_runtime["current_route"] = task.route.value
    task_runtime["route_history"].append({"route": task.route.value, "reason": f"amend: {reason}"})
    requires_review = "success_criteria" in patch
    if requires_review:
        runtime["approved"] = False
        runtime["phase"] = "awaiting_review"
    task_runtime["last_message"] = f"Amended: {reason}"
    _publish_active_tasks(state, runtime)
    return {
        "status": "success",
        "task_id": task_id,
        "requires_review": requires_review,
        "phase": runtime["phase"],
    }


def mark_result_review(
    state: MutableMapping[str, Any],
    *,
    approved: bool,
    feedback: str | None = None,
) -> dict[str, Any]:
    runtime = _runtime(state)
    if runtime["phase"] not in {"reporting", "awaiting_result_review"}:
        raise ExperimentRuntimeError("invalid_phase", "Result review requires a reported experiment.")

    rounds = int(state.get(REPLAN_ROUNDS_KEY) or 0)
    budget = int(get_settings().experiments.max_replan_rounds)
    exhausted = False

    if approved:
        runtime["phase"] = "completed"
    elif rounds >= budget:
        # Out of replan budget: finish honestly rather than loop. The reviewer's
        # objection is preserved so the report can say the run ended unresolved.
        exhausted = True
        runtime["phase"] = "completed"
        runtime["replan_exhausted"] = True
        runtime["result_review_feedback"] = (
            f"{feedback or 'Result redesign requested.'} "
            f"(replan budget exhausted after {rounds} round(s); stopping instead of replanning)"
        )
    else:
        rounds += 1
        state[REPLAN_ROUNDS_KEY] = rounds
        runtime["replan_rounds"] = rounds
        runtime["phase"] = "replan_requested"
        runtime["result_review_feedback"] = feedback or "Result redesign requested."
        # A replan is a fresh planning round, so the per-round revision budget
        # starts over. Without this the next planner hop inherits a spent budget
        # and can pause the experiment on its first stumble.
        state["experiment_plan_revision_count"] = 0
        state["experiment_inventory_blocker_hits"] = 0

    # Reassign, do not just mutate: ADK records a state delta on assignment, so
    # an in-place edit of the nested dict never reaches session state. That is
    # what defeated build_experiment_context's `phase == "completed"` gate and
    # let a finished stage be re-delegated as a whole new plan+HITL+execute
    # cycle — five times in one observed run.
    state[RUNTIME_KEY] = runtime
    _audit(
        f"EXPERIMENT_RESULT_REVIEW approved={str(approved).lower()} "
        f"phase={runtime['phase']} replan_rounds={runtime.get('replan_rounds', 0)}/{budget}"
        + (" replan_exhausted=true" if exhausted else "")
    )
    return {
        "status": "success",
        "phase": runtime["phase"],
        "replan_rounds": int(runtime.get("replan_rounds") or 0),
        "max_replan_rounds": budget,
        "replan_exhausted": exhausted,
    }


__all__ = [
    "ExperimentRuntimeError",
    "FALLBACK_CHAINS",
    "ROUTE_AGENT_BY_ROUTE",
    "RUNTIME_KEY",
    "active_attempt",
    "amend_task",
    "approve_plan",
    "fallback_task",
    "clamp_generate_launch_num",
    "force_managed_s3_launch_params",
    "generate_num_cap",
    "generate_presigned_s3_url",
    "get_experiment_plan",
    "initialize_runtime",
    "mark_result_review",
    "mark_route_returned",
    "record_result",
    "resolve_fallback_chains",
    "retry_task",
    "skip_task",
    "start_task",
]

