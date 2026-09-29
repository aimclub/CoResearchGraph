from __future__ import annotations

import json
from pathlib import Path

from CoScientist.experiments.outcome.reconciliation import (
    DispositionKind,
    reconcile_scientific_outcome,
)


def _approved_state(**extra):
    state = {
        "experiment_runtime": {"phase": "completed", "task_order": [], "tasks": {}},
        "experiment_module_outcome": {
            "status": "completed",
            "stage": "result_review",
            "reason": "result_approved",
            "accepted": True,
        },
    }
    state.update(extra)
    return state


def test_test1_snapshot_is_a_recoverable_pause_not_completion():
    root = Path(__file__).resolve().parents[3]
    snapshot = root / (
        "graph_runs/web_state/adk_sessions/coscientist_app/"
        "user_6f7b3af231cc48029f87ddc901a2facd/"
        "session_6496a804c45b43858c015a7c9e2ca026.json"
    )
    state = json.loads(snapshot.read_text(encoding="utf-8"))["state"]

    disposition = reconcile_scientific_outcome(state, current_run_id="run-limited")

    assert disposition.kind is DispositionKind.PAUSED
    assert disposition.reason == "max_plan_revisions"
    assert disposition.pause_cause == "planning_review"
    assert disposition.pending_decision["kind"] == "experiment_plan_recovery"
    assert {"TASK-1", "TASK-3"}.issubset(disposition.unfinished_task_ids)


def test_approved_experiment_does_not_waive_incomplete_root_task_mapping():
    state = _approved_state(_master_active_tasks=[
        {"id": "TASK-1", "status": "DONE"},
        {"id": "TASK-2", "status": "TODO"},
    ])

    disposition = reconcile_scientific_outcome(state)

    assert disposition.kind is DispositionKind.PAUSED
    assert disposition.reason == "root_roadmap_incomplete"
    assert disposition.unfinished_task_ids == ("TASK-2",)


def test_only_exact_human_limited_scope_acceptance_can_close_root_todos():
    state = _approved_state(
        _master_active_tasks=[{"id": "TASK-2", "status": "TODO"}],
        scientific_limited_scope_acceptance={
            "accepted": True,
            "decision_source": "human",
            "run_id": "run-limited",
            "accepted_task_ids": ["TASK-2"],
        },
    )

    from CoScientist.experiments.outcome.reconciliation import roadmap_digest
    state["scientific_limited_scope_acceptance"]["roadmap_digest"] = roadmap_digest(state)

    disposition = reconcile_scientific_outcome(state, current_run_id="run-limited")

    assert disposition.kind is DispositionKind.COMPLETED_LIMITED
    assert disposition.report_allowed is True


def test_live_runtime_task_cannot_be_waived_by_generic_result_approval():
    state = _approved_state()
    state["experiment_runtime"] = {
        "phase": "completed",
        "task_order": ["EXP-1"],
        "tasks": {"EXP-1": {"status": "running"}},
    }

    disposition = reconcile_scientific_outcome(state)

    assert disposition.kind is DispositionKind.PAUSED
    assert disposition.reason == "experiment_runtime_incomplete"


def test_accepted_scientific_negative_is_distinct_from_execution_failure():
    state = _approved_state(experiment_task_results=[{
        "task_id": "EXP-1",
        "status": "success",
        "execution_status": "completed",
        "assessment_status": "not_met",
    }])

    disposition = reconcile_scientific_outcome(state)

    assert disposition.kind is DispositionKind.COMPLETED_NEGATIVE
    assert disposition.report_allowed is True


def test_accepted_execution_failure_is_terminal_but_clearly_limited():
    state = _approved_state(experiment_task_results=[{
        "task_id": "EXP-1",
        "status": "partial",
        "execution_status": "failed",
        "assessment_status": "not_met",
    }])
    state["experiment_runtime"] = {
        "phase": "completed", "task_order": ["EXP-1"],
        "tasks": {"EXP-1": {"status": "failed"}},
    }

    disposition = reconcile_scientific_outcome(state)

    assert disposition.kind is DispositionKind.COMPLETED_LIMITED
    assert disposition.reason == "accepted_scientific_limits"
    assert disposition.report_allowed is True


def test_capacity_pause_exposes_requested_and_current_limits_without_impossible_approval():
    state = {
        "experiment_plan_review_paused": True,
        "experiment_review_pause_reason": "plan_capacity_conflict",
        "experiment_module_outcome": {
            "status": "blocked", "stage": "preflight",
            "reason": "plan_capacity_conflict", "accepted": False,
            "can_raise_limit": False, "requested_max_plan_tasks": 20,
            "max_plan_tasks": 8, "required_operations": [f"OP-{i}" for i in range(21)],
        },
    }

    pending = reconcile_scientific_outcome(state).pending_decision

    assert pending["requested_max_plan_tasks"] == 20
    assert pending["current_max_plan_tasks"] == 8
    assert "approve_run_task_limit" not in pending["options"]
    assert "revise_or_split_scope" in pending["options"]


def test_normal_chat_without_scientific_execution_state_is_unchanged():
    disposition = reconcile_scientific_outcome({"final_report": "ordinary answer"})

    assert disposition.kind is DispositionKind.COMPLETED
    assert disposition.reason == "untracked_chat_completed"
    assert disposition.report_allowed is True


def test_completed_approved_flow_is_terminal():
    disposition = reconcile_scientific_outcome(_approved_state())

    assert disposition.kind is DispositionKind.COMPLETED
    assert disposition.report_allowed is True
