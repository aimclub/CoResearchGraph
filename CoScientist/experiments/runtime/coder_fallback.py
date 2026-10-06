"""Automatic MCP → Coder when an MCP route is technically down.

This is not ``route_coder_mcp``. That switch lets a Coder task call MCP.
This one lets a task whose MCP route cannot be reached run once through Coder,
on the same logical step and inside the existing attempt budget.
"""
from __future__ import annotations

from typing import Any, Mapping

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.schemas import ExecutionRoute
from CoScientist.requirements.execution import is_infrastructure_failure

MCP_ROUTES = frozenset({
    ExecutionRoute.FEDOT_MAS.value,
    ExecutionRoute.REACT_TOOLS.value,
})
_HANDOFF = frozenset({
    "output_unbound",
    "artifact_not_found",
    "required_upstream_artifact_missing",
    "materialization_unavailable",
    "output_ambiguous",
    "output_recovery_exhausted",
    "contract_problem",
    "result_incomplete",
})
_TRANSIENT = frozenset({
    "timeout",
    "rate_limited",
    "http_429",
    "temporarily_unavailable",
})
_LIMITS = (
    "Coder reproduces the contracted outputs from the verified inputs. "
    "Calling the unavailable MCP again is not an alternative method."
)


def classify_failure(result: Mapping[str, Any], *, assessment_only: bool) -> str:
    """transient, technical, handoff, scientific, or other."""
    if assessment_only:
        return "scientific"
    code = str(result.get("error_code") or "").strip().lower()
    if code in _HANDOFF:
        return "handoff"
    if code in _TRANSIENT:
        return "transient"
    if is_infrastructure_failure(result):
        return "technical"
    return "other"


def meaning_preserved(task: Any) -> tuple[bool, str]:
    """The reviewed alternative method authorizes substitution, not a repo URL."""
    method = str(getattr(task, "coder_fallback_method", None) or "").strip()
    return (True, "") if method else (False, "No alternative Coder method is approved for this task.")


def _inputs_ready(runtime: Mapping[str, Any], task: Any) -> tuple[bool, str]:
    for item in getattr(task, "input_data", None) or []:
        if not getattr(item, "required", False):
            continue
        if getattr(item, "kind", "") != "task_artifact":
            if not (getattr(item, "url", None) or getattr(item, "s3_key", None) or getattr(item, "workspace_path", None)):
                return False, f"required input {getattr(item, 'data_id', '')} has no location"
            continue
        from CoScientist.experiments.runtime.outputs import resolve_output
        from CoScientist.experiments.runtime.errors import ExperimentRuntimeError
        try:
            resolve_output(runtime, source_task_id=item.source_task_id,
                source_artifact_id=item.source_artifact_id or "", source_output_id=item.source_output_id,
                consumer_task_id=task.id, data_id=item.data_id)
        except ExperimentRuntimeError as exc:
            return False, str(exc)
    return True, ""


def _brief(task: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    design = task.design
    return {
        "task_id": task.id,
        "target_refs": list(design.target_refs),
        "method": task.coder_fallback_method,
        "output_contract": [item.model_dump(mode="json") for item in task.expected_artifacts],
        "mcp_diagnostics": {
            "error_code": result.get("error_code"),
            "error_message": result.get("error_message") or result.get("summary"),
        },
        "limits": _LIMITS,
        "route": ExecutionRoute.CODER.value,
    }


def coder_fallback_decision(
    *,
    result: Mapping[str, Any],
    task_runtime: Mapping[str, Any],
    task: Any,
    runtime: Mapping[str, Any],
    settings: ExperimentsSettings,
    route_agents: Any = None,
    attempts_on_route: int,
    total_left: bool,
    assessment_only: bool,
    next_fallback: str | None = None,
    technical_failure: bool | None = None,
) -> dict[str, Any] | None:
    """Choose the sole next step after failure, including the ordinary chain.

    With no recorded technical_failure, only inspect a known MCP outage before
    start_task. That preflight may return None; a recorded failure always gets
    a decision. connection_refused retains its direct, bounded Coder fallback.
    """
    route = str(task_runtime.get("current_route") or "")
    kind = classify_failure(result, assessment_only=assessment_only)
    attempts_left = total_left and attempts_on_route < settings.task_max_attempts
    if route in MCP_ROUTES and kind == "scientific":
        return {
            "status": "failed",
            "message": "A missed scientific criterion is not an automatic technical retry.",
        }
    if route in MCP_ROUTES and kind == "handoff":
        return {
            "status": "failed",
            "message": "An output handoff is recovered from the existing file, not recomputed.",
        }
    if kind == "transient" and not attempts_left and next_fallback not in {None, ExecutionRoute.CODER.value}:
        return {"status": "fallback_pending"}
    if route not in MCP_ROUTES or kind == "other" or not settings.allow_coder_fallback:
        if technical_failure is None:
            return None
        if result.get("retryable") and attempts_left:
            return {"status": "retry_pending"}
        if technical_failure and total_left and next_fallback is not None:
            return {"status": "fallback_pending"}
        return {"status": "failed"}
    if kind == "transient" and attempts_left:
        return {"status": "retry_pending", "message": "Transient MCP error; retrying the same route."}
    history = {str(entry.get("route") or "") for entry in task_runtime.get("route_history") or []}
    if ExecutionRoute.CODER.value in history:
        why = "Coder was already used for this step and is not tried again."
    elif not total_left:
        why = "Attempt budget is exhausted, so Coder is not started."
    elif route_agents is not None and "CoderAgent" not in set(route_agents):
        why = "Coder is not connected, so the MCP failure stays failed."
    else:
        permitted, why = meaning_preserved(task)
        if permitted:
            _, why = _inputs_ready(runtime, task)
    if why:
        return {
            "status": "failed",
            "message": why,
            "stop": {"task_id": task.id, "kind": kind, "reason": why},
        }
    return {
        "status": "fallback_pending",
        "message": "MCP is unavailable; one Coder attempt is allowed.",
        "coder_fallback": _brief(task, result),
    }
