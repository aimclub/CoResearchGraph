"""Reconcile durable state into one deterministic final disposition.

Agent prose and a normal coroutine return are not completion evidence.  This
module deliberately uses only typed state written by the workflow: review
outcomes, experiment runtime state, the selected pipeline lanes, and the root
task roadmap.  It is shared by the web controller, report gate and pipeline
scope callback so those boundaries cannot disagree about whether work ended.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class DispositionKind(str, Enum):
    COMPLETED = "completed"
    COMPLETED_NEGATIVE = "completed_negative"
    COMPLETED_LIMITED = "completed_limited"
    PAUSED = "paused"


@dataclass(frozen=True)
class RunDisposition:
    kind: DispositionKind
    reason: str
    stage: str = ""
    pause_cause: str | None = None
    pending_decision: dict[str, Any] | None = None
    accepted: bool = False
    module_engaged: bool = False
    unfinished_task_ids: tuple[str, ...] = ()

    @property
    def terminal(self) -> bool:
        return self.kind is not DispositionKind.PAUSED

    @property
    def report_allowed(self) -> bool:
        return self.terminal and self.accepted

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        value["terminal"] = self.terminal
        value["report_allowed"] = self.report_allowed
        value["unfinished_task_ids"] = list(self.unfinished_task_ids)
        return value


_ROOT_TERMINAL = frozenset({"DONE", "CANCELLED", "FAILED", "SKIPPED"})
_ROOT_SUCCESS = frozenset({"DONE"})
_RUNTIME_TERMINAL = frozenset({"done", "done_with_warnings", "failed", "skipped", "blocked"})
_BLOCKED_OUTCOMES = frozenset({"blocked", "rejected", "awaiting_human", "paused"})
_ACCEPTED_RESULT_REASONS = frozenset({"result_approved", "result_auto_approved"})
_PLANNING_REASONS = frozenset({
    "max_plan_revisions",
    "plan_revision_budget_exhausted_pending_human",
    "fallback_review_timeout",
    "fallback_rejected_by_operator",
    "fallback_auto_decision_rejected",
    "fallback_candidate_no_longer_executable",
    "plan_review_timeout",
    "plan_rejected_by_operator",
    "repository_route_timeout",
    "repository_route_rejected",
    "inventory_blocker_repeated",
    "plan_capacity_conflict",
})


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _root_tasks(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    # _master_active_tasks is authoritative.  Do not combine it with the
    # per-agent projection: doing so turns a complete task into two conflicting
    # rows when active_tasks is stale.
    raw = state.get("_master_active_tasks")
    if not isinstance(raw, (list, tuple)):
        raw = state.get("active_tasks")
    return [row for row in (raw or []) if isinstance(row, Mapping)]


def _task_id(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("id") or row.get("task_id") or f"task-{index + 1}")


def _root_task_state(state: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    unfinished: list[str] = []
    non_success: list[str] = []
    for index, row in enumerate(_root_tasks(state)):
        task_id = _task_id(row, index)
        status = str(row.get("status") or "").strip().upper()
        if status not in _ROOT_TERMINAL:
            unfinished.append(task_id)
        if status not in _ROOT_SUCCESS:
            non_success.append(task_id)
    return tuple(unfinished), tuple(non_success)


def roadmap_digest(state: Mapping[str, Any]) -> str:
    """Bind a limited-scope decision to the exact roadmap it reviewed."""
    rows = [
        {
            "id": _task_id(row, index),
            "status": str(row.get("status") or ""),
            "assignee": str(row.get("assignee") or ""),
            "title": str(row.get("title") or row.get("name") or ""),
        }
        for index, row in enumerate(_root_tasks(state))
    ]
    payload = {
        "tasks": rows,
        "pipeline_scope": dict(_mapping(state.get("pipeline_scope"))),
        "pipeline_scope_done": list(state.get("pipeline_scope_done") or []),
    }
    wire = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()


def _runtime_unfinished(state: Mapping[str, Any]) -> tuple[str, ...]:
    runtime = _mapping(state.get("experiment_runtime"))
    tasks = _mapping(runtime.get("tasks"))
    order = runtime.get("task_order")
    task_ids = list(order) if isinstance(order, (list, tuple)) else list(tasks)
    return tuple(
        str(task_id)
        for task_id in task_ids
        if str(_mapping(tasks.get(task_id)).get("status") or "").strip().lower()
        not in _RUNTIME_TERMINAL
    )


def _pipeline_unfinished(state: Mapping[str, Any]) -> tuple[str, ...]:
    scope = _mapping(state.get("pipeline_scope"))
    if not scope:
        return ()
    selected = []
    for key, name in (
        ("research", "ResearchAgent"),
        ("hypotheses", "HypothesesAgent"),
        ("experiments", "ExperimentModuleAgent"),
    ):
        if scope.get(key):
            selected.append(name)
    done = {str(item) for item in (state.get("pipeline_scope_done") or [])}
    return tuple(name for name in selected if name not in done)


def _module_engaged(state: Mapping[str, Any]) -> bool:
    return bool(
        _mapping(state.get("experiment_module_outcome"))
        or _mapping(state.get("experiment_runtime"))
        or state.get("experiment_module_dispatched")
        or state.get("experiment_plan_review_paused")
        or state.get("experiment_plan_fallback_pending")
        or state.get("experiment_plan")
        or state.get("experiment_plan_candidate")
        or state.get("experiment_plan_view")
    )


def _scientific_negative(state: Mapping[str, Any]) -> bool:
    """A measured target miss is terminal science, not an execution failure."""
    results = [
        row for row in (state.get("experiment_task_results") or [])
        if isinstance(row, Mapping)
    ]
    if not results:
        return False
    executed = [
        row for row in results
        if str(row.get("execution_status") or "").lower() == "completed"
        or (
            not row.get("execution_status")
            and str(row.get("status") or "").lower() == "success"
        )
    ]
    return bool(executed) and any(
        str(row.get("assessment_status") or "").lower() == "not_met"
        for row in executed
    )


def _execution_has_limits(state: Mapping[str, Any]) -> bool:
    for row in state.get("experiment_task_results") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("execution_status") or "").lower() in {
            "completed_with_limits", "failed", "skipped",
        }:
            return True
        if str(row.get("status") or "").lower() in {"partial", "failure", "skipped"}:
            return True
    runtime = _mapping(state.get("experiment_runtime"))
    for row in _mapping(runtime.get("tasks")).values():
        if isinstance(row, Mapping) and str(row.get("status") or "").lower() in {
            "done_with_warnings", "failed", "skipped", "blocked",
        }:
            return True
    return False


def _accepted_limited_scope(
    state: Mapping[str, Any], outstanding: tuple[str, ...], current_run_id: str | None,
) -> bool:
    """Whether a human explicitly accepted these exact remaining root items."""
    outcome = _mapping(state.get("experiment_module_outcome"))
    record = _mapping(
        state.get("scientific_limited_scope_acceptance")
        or outcome.get("limited_scope_acceptance")
    )
    if not record or record.get("accepted") is not True:
        return False
    if str(record.get("decision_source") or "").lower() != "human":
        return False
    if not current_run_id or record.get("run_id") != current_run_id:
        return False
    if record.get("roadmap_digest") != roadmap_digest(state):
        return False
    covered = {
        str(item) for item in (
            record.get("accepted_task_ids")
            or record.get("outstanding_item_ids")
            or []
        )
    }
    return bool(outstanding) and set(outstanding).issubset(covered)


def _pause(
    *,
    reason: str,
    stage: str,
    module_engaged: bool,
    unfinished: tuple[str, ...] = (),
    status: str = "",
    details: Mapping[str, Any] | None = None,
) -> RunDisposition:
    planning = stage in {"plan_review", "planning", "preflight"} or reason in _PLANNING_REASONS
    cause = "planning_review" if planning else (
        "result_review" if stage == "result_review" else "scientific_work_incomplete"
    )
    if reason in {
        "max_plan_revisions",
        "inventory_blocker_repeated",
        "fallback_review_timeout",
        "fallback_rejected_by_operator",
    }:
        pending = {
            "kind": "experiment_plan_recovery",
            "reason": reason,
            "action": "resume_saved_plan_review",
            "message": (
                "Resume to review the saved executable plan candidate; the "
                "plan will not be regenerated and the revision budget will not reset."
            ),
        }
    elif reason == "plan_capacity_conflict":
        can_raise = bool(_mapping(details).get("can_raise_limit"))
        pending = {
            "kind": "plan_capacity_conflict",
            "reason": reason,
            "options": (
                ["approve_run_task_limit", "reject_and_pause"]
                if can_raise else ["revise_or_split_scope", "reject_and_pause"]
            ),
            "can_raise_limit": can_raise,
            "maximum_supported_tasks": 20,
        }
        for key in (
            "requested_task_count", "configured_task_limit", "operation_digest",
            "requested_max_plan_tasks", "max_plan_tasks", "required_operations",
        ):
            if key in _mapping(details):
                pending[key] = _mapping(details)[key]
        if "max_plan_tasks" in pending:
            pending["current_max_plan_tasks"] = pending["max_plan_tasks"]
    else:
        pending = {
            "kind": "scientific_outcome",
            "reason": reason,
            "stage": stage,
            "status": status,
            "unfinished_task_ids": list(unfinished),
        }
    return RunDisposition(
        kind=DispositionKind.PAUSED,
        reason=reason,
        stage=stage,
        pause_cause=cause,
        pending_decision=pending,
        accepted=False,
        module_engaged=module_engaged,
        unfinished_task_ids=unfinished,
    )


def accepted_stage_outcome(state: Mapping[str, Any], agent_name: str) -> bool:
    """Whether a returned pipeline AgentTool actually closed its named lane."""
    if agent_name != "ExperimentModuleAgent":
        assigned = [
            row for row in _root_tasks(state)
            if str(row.get("assignee") or "") == agent_name
        ]
        # Older plans did not retain an assignee.  A successful AgentTool return
        # remains the compatibility signal only when there is no contradictory
        # typed task state for that lane.
        return not assigned or all(
            str(row.get("status") or "").upper() == "DONE" for row in assigned
        )

    outcome = _mapping(state.get("experiment_module_outcome"))
    status = str(outcome.get("status") or "").lower()
    reason = str(outcome.get("reason") or "")
    if status == "not_applicable":
        return True
    if status != "completed" or reason not in _ACCEPTED_RESULT_REASONS:
        return False
    runtime = _mapping(state.get("experiment_runtime"))
    return (
        str(runtime.get("phase") or "").lower() == "completed"
        and not _runtime_unfinished(state)
    )


def reconcile_scientific_outcome(
    state: Mapping[str, Any] | None, *, current_run_id: str | None = None,
) -> RunDisposition:
    """Return the only disposition the controller/reporting boundaries may use."""
    state = _mapping(state)
    outcome = _mapping(state.get("experiment_module_outcome"))
    runtime = _mapping(state.get("experiment_runtime"))
    module_engaged = _module_engaged(state)
    status = str(outcome.get("status") or "").strip().lower()
    stage = str(outcome.get("stage") or "").strip()
    reason = str(
        outcome.get("reason")
        or state.get("experiment_review_pause_reason")
        or ""
    ).strip()

    root_unfinished, root_non_success = _root_task_state(state)
    runtime_unfinished = _runtime_unfinished(state)
    pipeline_unfinished = _pipeline_unfinished(state)
    unfinished = tuple(dict.fromkeys(
        (*root_unfinished, *runtime_unfinished, *pipeline_unfinished)
    ))

    if (
        status in _BLOCKED_OUTCOMES
        or state.get("experiment_plan_review_paused")
        or state.get("experiment_plan_fallback_pending")
    ):
        return _pause(
            reason=reason or status or "experiment_review_pending",
            stage=stage or "plan_review",
            module_engaged=module_engaged,
            unfinished=unfinished,
            status=status,
            details=outcome,
        )

    runtime_phase = str(runtime.get("phase") or "").strip().lower()
    accepted_result = (
        status == "completed"
        and stage == "result_review"
        and reason in _ACCEPTED_RESULT_REASONS
        and runtime_phase == "completed"
    )

    if module_engaged and not accepted_result and status != "not_applicable":
        return _pause(
            reason=reason or (
                "experiment_runtime_incomplete" if runtime_phase
                else "experiment_outcome_unaccepted"
            ),
            stage=stage or runtime_phase or "experiment",
            module_engaged=True,
            unfinished=unfinished,
            status=status,
        )

    # No review decision may waive a still-running experiment task.  Explicit
    # limited-scope acceptance applies only after runtime tasks are terminal.
    if runtime_unfinished:
        return _pause(
            reason="experiment_runtime_incomplete",
            stage="execution",
            module_engaged=module_engaged,
            unfinished=runtime_unfinished,
        )

    if unfinished or root_non_success:
        outstanding = tuple(dict.fromkeys((*unfinished, *root_non_success)))
        if _accepted_limited_scope(state, outstanding, current_run_id):
            return RunDisposition(
                kind=DispositionKind.COMPLETED_LIMITED,
                reason="operator_accepted_limits",
                stage=stage,
                accepted=True,
                module_engaged=module_engaged,
                unfinished_task_ids=outstanding,
            )
        return _pause(
            reason="root_roadmap_incomplete" if unfinished else "root_roadmap_not_successful",
            stage="roadmap",
            module_engaged=module_engaged,
            unfinished=outstanding,
        )

    if accepted_result and _execution_has_limits(state):
        return RunDisposition(
            kind=DispositionKind.COMPLETED_LIMITED,
            reason="accepted_scientific_limits",
            stage=stage,
            accepted=True,
            module_engaged=True,
        )

    if accepted_result and _scientific_negative(state):
        return RunDisposition(
            kind=DispositionKind.COMPLETED_NEGATIVE,
            reason="accepted_scientific_negative",
            stage=stage,
            accepted=True,
            module_engaged=True,
        )

    if accepted_result or status == "not_applicable":
        return RunDisposition(
            kind=DispositionKind.COMPLETED,
            reason=reason or "accepted_stage_outcome",
            stage=stage,
            accepted=True,
            module_engaged=module_engaged,
        )

    # Ordinary chats and legacy non-experiment lanes have no experiment state.
    # Their normal return remains terminal, provided no selected pipeline lane
    # or root roadmap says otherwise.
    return RunDisposition(
        kind=DispositionKind.COMPLETED,
        reason="untracked_chat_completed",
        accepted=True,
        module_engaged=False,
    )


__all__ = [
    "DispositionKind",
    "RunDisposition",
    "accepted_stage_outcome",
    "reconcile_scientific_outcome",
    "roadmap_digest",
]
