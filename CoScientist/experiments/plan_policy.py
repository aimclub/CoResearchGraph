"""One bounded plan-size policy, with explicit per-run human exceptions."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from CoScientist.experiments.scope import partition_operations

CAPACITY_OVERRIDE_KEY = "experiment_plan_capacity_override"
MAX_SCHEMA_TASKS = 20


def operations_digest(operations: Any) -> str:
    compute, _ = partition_operations(operations)
    return hashlib.sha256(json.dumps(compute, ensure_ascii=True, sort_keys=True).encode()).hexdigest()


def effective_plan_settings(settings: Any, context: Mapping[str, Any]) -> Any:
    """An approval applies only to this exact run and operation set."""
    grant = context.get("plan_capacity_override") or {}
    if not isinstance(grant, Mapping) or grant.get("decision_source") != "human":
        return settings
    if not context.get("experiment_run_id") or grant.get("run_id") != context.get("experiment_run_id"):
        return settings
    if grant.get("operations_digest") != operations_digest(context.get("operations") or []):
        return settings
    value = grant.get("max_plan_tasks")
    if type(value) is not int or not settings.max_plan_tasks < value <= MAX_SCHEMA_TASKS:
        return settings
    return settings.model_copy(update={"max_plan_tasks": value})


def refresh_plan_limits(state: Any, settings: Any) -> None:
    from CoScientist.experiments.context.builder import PLANNER_CONTEXT_KEY, _prompt_context

    context = dict(state.get("experiment_context") or {})
    context["plan_capacity_override"] = state.get(CAPACITY_OVERRIDE_KEY)
    effective = effective_plan_settings(settings, context)
    context["plan_limits"] = {
        "max_tasks": effective.max_plan_tasks,
        "schema_max_tasks": MAX_SCHEMA_TASKS,
        "max_revisions": effective.max_plan_revisions,
        "operation_ref_cardinality": "one",
    }
    state["experiment_context"] = context
    state[PLANNER_CONTEXT_KEY] = _prompt_context(context)


async def check_experiment_plan_capacity(callback_context: Any) -> Any:
    """Before planner LLM: a scalar OP reference cannot cover N ops in <N tasks."""
    from google.genai import types
    from CoScientist.config import get_settings
    from CoScientist.graph.session_scope import session_key
    from CoScientist.hitl.models import HITLAction, HITLDecisionSource, HITLRequest

    state = callback_context.state
    settings = get_settings().experiments
    refresh_plan_limits(state, settings)
    context = state.get("experiment_context") or {}
    compute, _ = partition_operations(context.get("operations") or [])
    effective = effective_plan_settings(settings, context)
    if len(compute) <= effective.max_plan_tasks:
        if state.get("experiment_review_pause_reason") == "plan_capacity_conflict":
            state["experiment_plan_review_paused"] = False
            state["experiment_review_pause_reason"] = None
            state["experiment_module_outcome"] = None
        return None

    reason = "plan_capacity_conflict"
    limit = min(MAX_SCHEMA_TASKS, len(compute))
    message = (
        f"В вычислительном этапе {len(compute)} обязательных операций, а лимит плана — "
        f"{effective.max_plan_tasks} задач. Каждая задача покрывает одну операцию. "
        "Переписывание плана не устранит это противоречие. "
        f"Разрешить для этого плана до {limit} задач? Общий бюджет обращений к моделям "
        "не увеличивается. Отказ оставит планирование на паузе."
    )
    if len(compute) > MAX_SCHEMA_TASKS:
        message = (
            f"Обязательных вычислительных операций: {len(compute)}; схема допускает "
            f"не более {MAX_SCHEMA_TASKS} задач. Нужна явная разбивка исследования "
            "на этапы или изменение объёма. Операции не удалены; планирование на паузе."
        )
    outcome = {
        "status": "awaiting_human", "stage": "preflight", "reason": reason,
        "required_operations": [op["operation_id"] for op in compute],
        "max_plan_tasks": effective.max_plan_tasks, "requested_max_plan_tasks": limit,
        "can_raise_limit": len(compute) <= MAX_SCHEMA_TASKS,
        "accepted": False,
    }
    state["experiment_module_outcome"] = outcome
    state["experiment_plan_review_paused"] = True
    state["experiment_review_pause_reason"] = reason
    invocation = getattr(callback_context, "_invocation_context", None)
    handler = getattr(getattr(invocation, "agent", None), "hitl_handler", None)
    response = None
    if handler is not None and len(compute) <= MAX_SCHEMA_TASKS:
        user_id, session_id = session_key(callback_context)
        response = await handler.handle_request(HITLRequest(
            agent_name="ExperimentPlannerAgent", action_type=HITLAction.APPROVE,
            message=message, requires_human=True, invoked_via="callback", trigger="before_agent",
            timeout_seconds=settings.plan_review_timeout_s,
            context={"_session": {"user_id": user_id, "session_id": session_id},
                     "kind": reason, "output": message,
                     "required_operations": compute,
                     "review_id": f"capacity:{context.get('experiment_run_id')}:{operations_digest(compute)}"},
        ))
    if (response is not None and response.approved
            and response.action == HITLAction.APPROVE
            and response.decision_source == HITLDecisionSource.HUMAN):
        state[CAPACITY_OVERRIDE_KEY] = {
            "run_id": context.get("experiment_run_id"),
            "operations_digest": operations_digest(compute),
            "max_plan_tasks": limit, "decision_source": "human",
        }
        refresh_plan_limits(state, settings)
        state["experiment_plan_review_paused"] = False
        state["experiment_review_pause_reason"] = None
        state["experiment_module_outcome"] = {
            **outcome, "status": "running", "reason": "plan_capacity_approved", "accepted": True,
        }
        return None
    state["experiment_module_outcome"] = {**outcome, "status": "blocked"}
    return types.Content(role="model", parts=[types.Part(text=message)])
