"""Bounded execution completion, separate assessment, and targeted redo."""
from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.review import (
    ExperimentReviewSessionAgent,
    result_review_identity,
)
from CoScientist.experiments.runtime import (
    ATTEMPT_LEDGER_KEY,
    ExperimentRuntimeError,
    approve_plan,
    experiment_next_actions,
    experiment_state_revision,
    fallback_task,
    get_experiment_plan,
    initialize_runtime,
    logical_operation_key,
    mark_result_review,
    mark_route_returned,
    preview_task_redo,
    record_result,
    request_task_redo,
    result_redo_context,
    start_task,
)
from CoScientist.experiments.schemas import SuccessCriterion
from CoScientist.hitl.handler import AbstractHITLHandler
from CoScientist.hitl.models import HITLAction, HITLRequest, HITLResponse

from .helpers import _approved_state, _plan, _success_result, _task


def _artifact(state: dict, task_id: str) -> None:
    state.setdefault("fedot_artifacts", []).append({
        "name": f"{task_id.lower()}-result.csv",
        "bucket": "managed-experiments",
        "s3_key": f"results/{task_id}/result.csv",
        "role": "data",
        "media_type": "text/csv",
    })


class _CapturingResultReviewHandler(AbstractHITLHandler):
    def __init__(self, response: HITLResponse | None = None) -> None:
        self.requests: list[HITLRequest] = []
        self.response = response or HITLResponse(
            action=HITLAction.REJECT,
            approved=False,
            instructions="Keep the result; do not rerun anything.",
        )

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        self.requests.append(request)
        return self.response


def test_criterion_purpose_defaults_and_explicit_override():
    threshold = SuccessCriterion.model_validate({
        "criterion_id": "quality",
        "description": "AUROC >= .8",
        "kind": "threshold",
        "metric": "auroc",
        "operator": ">=",
        "target": 0.8,
        "verification": "Read the metrics output.",
    })
    delivery = SuccessCriterion.model_validate({
        "criterion_id": "delivery",
        "description": "Output delivered",
        "kind": "artifact_exists",
        "verification": "Open the output.",
    })
    overridden = threshold.model_copy(update={"purpose": "execution"})
    assert threshold.purpose == "assessment"
    assert delivery.purpose == "execution"
    assert overridden.purpose == "execution"


def test_failed_quality_threshold_is_terminal_success_not_retry():
    task = _task("EXP-1")
    task["success_criteria"].append({
        "criterion_id": "EXP-1-Q1",
        "description": "Score reaches target.",
        "kind": "threshold",
        "metric": "score",
        "operator": ">=",
        "target": 0.9,
        "verification": "Compare score.",
    })
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    _artifact(state, "EXP-1")
    payload = _success_result("EXP-1")
    payload["criteria_checks"].append({
        "criterion_id": "EXP-1-Q1",
        "passed": False,
        "observed": 0.62,
        "details": "The run completed but missed the target.",
    })

    stored = record_result(state, "EXP-1", started["attempt_id"], payload)

    result = stored["task_result"]
    assert result["status"] == "success"
    assert result["execution_status"] == "completed"
    assert result["assessment_status"] == "not_met"
    assert result["retryable"] is False
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "done"


def test_quality_labeled_failure_is_reclassified_partial_and_not_retried():
    task = _task("EXP-1")
    task["success_criteria"].append({
        "criterion_id": "EXP-1-Q1",
        "description": "Expert accepts the result.",
        "kind": "expert",
        "verification": "Expert review.",
    })
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    _artifact(state, "EXP-1")
    payload = {
        "status": "failure",
        "summary": "Execution completed; expert rejected the quality.",
        "criteria_checks": [
            {"criterion_id": "EXP-1-C1", "passed": True, "details": "completed"},
            {"criterion_id": "EXP-1-Q1", "passed": False, "details": "quality too low"},
        ],
        "error_code": "criteria_failed",
        "error_message": "expert criterion failed",
        "retryable": True,
    }
    result = record_result(state, "EXP-1", started["attempt_id"], payload)["task_result"]
    assert result["status"] == "partial"
    assert result["execution_status"] == "completed_with_limits"
    assert result["assessment_status"] == "not_met"
    assert result["retryable"] is False
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "done_with_warnings"


def test_scientific_refutation_is_preserved_as_a_valid_execution():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    _artifact(state, "EXP-1")
    payload = _success_result("EXP-1")
    payload["scientific_check"] = {
        "hypothesis_ref": "H1",
        "status": "refuted",
        "details": "The measured effect had the opposite sign.",
    }
    result = record_result(state, "EXP-1", started["attempt_id"], payload)["task_result"]
    assert result["execution_status"] == "completed"
    assert result["scientific_check"]["status"] == "refuted"


def test_record_result_is_idempotent_but_rejects_conflicting_overwrite():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    _artifact(state, "EXP-1")
    payload = _success_result("EXP-1")
    first = record_result(state, "EXP-1", started["attempt_id"], payload)
    duplicate = record_result(state, "EXP-1", started["attempt_id"], payload)
    assert duplicate["idempotent"] is True
    assert duplicate["task_result"]["result_id"] == first["task_result"]["result_id"]
    assert len(state["experiment_runtime"]["results"]) == 1

    with pytest.raises(ExperimentRuntimeError) as raised:
        record_result(
            state, "EXP-1", started["attempt_id"],
            {**payload, "summary": "conflicting replacement"},
        )
    assert raised.value.code == "result_conflict"
    assert len(state["experiment_runtime"]["results"]) == 1


def test_total_attempt_ledger_survives_cosmetic_replan_and_llm_budget_change():
    settings = ExperimentsSettings(route_fedot=True, task_max_total_attempts=1)
    task = _task("EXP-1")
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1", settings=settings)
    key = started["operation_key"]
    mark_route_returned(state, "FedotAgent")
    record_result(state, "EXP-1", started["attempt_id"], {
        "status": "failure",
        "summary": "Timed out.",
        "criteria_checks": [],
        "error_code": "timeout",
        "error_message": "timeout",
        "retryable": True,
    }, settings=settings)
    assert state[ATTEMPT_LEDGER_KEY][key]["attempts"] == 1

    renamed = copy.deepcopy(task)
    renamed["id"] = "EXP-2"
    renamed["name"] = "Cosmetically renamed task"
    renamed["description"] = "Same operation, rewritten prose."
    assert logical_operation_key(task) == logical_operation_key(renamed)
    state["orchestrator_llm_calls_max"] = 999
    initialize_runtime(
        state, _plan(renamed),
        critique={"verdict": "approve", "issues": [], "summary": "approved"},
    )
    approve_plan(state)
    with pytest.raises(ExperimentRuntimeError) as raised:
        start_task(state, "EXP-2", settings=settings)
    assert raised.value.code == "attempt_budget_exhausted"
    assert state[ATTEMPT_LEDGER_KEY][key]["attempts"] == 1


def test_wrong_dataset_diagnostic_falls_back_and_is_not_delivery_evidence():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    record_result(state, "EXP-1", started["attempt_id"], {
        "status": "failure",
        "summary": "Tool only works with its own dataset.",
        "outputs": {"tool_limitation": {"dataset": "built-in"}},
        "criteria_checks": [],
        "error_code": "wrong_dataset",
        "error_message": "provided dataset is unsupported",
        "retryable": False,
    })
    task_runtime = state["experiment_runtime"]["tasks"]["EXP-1"]
    assert task_runtime["status"] == "fallback_pending"
    assert fallback_task(state, "EXP-1", "wrong dataset")["route"] == "react_tools"


def test_terminal_failure_does_not_block_an_independent_task():
    state = _approved_state(_plan(_task("EXP-1"), _task("EXP-2")))
    failed = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    record_result(state, "EXP-1", failed["attempt_id"], {
        "status": "failure",
        "summary": "The requested analysis was not applicable.",
        "criteria_checks": [],
        "error_code": "not_applicable",
        "error_message": "The method does not apply to this sample.",
        "retryable": False,
    })

    runtime = state["experiment_runtime"]
    assert runtime["tasks"]["EXP-1"]["status"] == "failed"
    assert runtime["tasks"]["EXP-2"]["status"] == "ready"
    assert start_task(state, "EXP-2")["task_id"] == "EXP-2"


def test_durable_artifact_does_not_override_explicit_failed_execution_check():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    _artifact(state, "EXP-1")
    stored = record_result(state, "EXP-1", started["attempt_id"], {
        "status": "failure",
        "summary": "The output file exists, but validation failed.",
        "criteria_checks": [{
            "criterion_id": "EXP-1-C1",
            "passed": False,
            "details": "Validator rejected the execution output.",
        }],
        "error_code": "criteria_failed",
        "error_message": "Execution validation failed.",
        "retryable": False,
    })

    result = stored["task_result"]
    assert result["status"] == "failure"
    assert result["criteria_checks"][0]["passed"] is False
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "failed"


def test_assessment_only_failure_without_delivery_is_terminal_not_retryable():
    task = _task("EXP-1")
    task["success_criteria"].append({
        "criterion_id": "EXP-1-Q1",
        "description": "Score reaches target.",
        "kind": "threshold",
        "metric": "score",
        "operator": ">=",
        "target": 0.9,
        "verification": "Compare score.",
    })
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    stored = record_result(state, "EXP-1", started["attempt_id"], {
        "status": "failure",
        "summary": "The score did not reach the scientific target.",
        "outputs": {"diagnostics": {"score": 0.62}},
        "criteria_checks": [
            {"criterion_id": "EXP-1-C1", "passed": True, "details": "ran"},
            {
                "criterion_id": "EXP-1-Q1", "passed": False,
                "observed": 0.62, "details": "below target",
            },
        ],
        "error_code": "criteria_failed",
        "error_message": "Quality threshold was not met.",
        "retryable": True,
    })

    result = stored["task_result"]
    assert result["status"] == "failure"
    assert result["execution_status"] == "failed"
    assert result["assessment_status"] == "not_met"
    assert result["retryable"] is False
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "failed"
    assert get_experiment_plan(state)["next_actions"] == []


def test_known_local_format_gets_a_basic_syntax_check(tmp_path):
    task = _task("EXP-1", artifact_name="result.json")
    task["expected_artifacts"][0]["media_type"] = "application/json"
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    broken = tmp_path / "result.json"
    broken.write_text("{not-json", encoding="utf-8")
    state["fedot_artifacts"] = [{
        "name": "result.json",
        "workspace_path": str(broken),
        "role": "data",
        "media_type": "application/json",
    }]
    with pytest.raises(ExperimentRuntimeError) as raised:
        record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))
    assert raised.value.code == "result_incomplete"


def test_targeted_redo_reopens_only_selection_and_dependency_closure():
    state = _approved_state(_plan(
        _task("EXP-1"),
        _task("EXP-2", depends_on=["EXP-1"]),
        _task("EXP-3"),
    ))
    runtime = state["experiment_runtime"]
    for row in runtime["tasks"].values():
        row["status"] = "done"
    runtime["phase"] = "reporting"

    preview = preview_task_redo(state, ["EXP-1"])
    assert preview["affected_task_ids"] == ["EXP-1", "EXP-2"]
    assert preview["preserved_task_ids"] == ["EXP-3"]
    ui = result_redo_context(state)
    assert [row["task_id"] for row in ui["task_choices"]] == ["EXP-1", "EXP-2", "EXP-3"]
    assert ui["affected_by_task"]["EXP-1"] == ["EXP-1", "EXP-2"]
    assert ui["selection_required_for_redo"] is True
    requested = request_task_redo(state, ["EXP-1"], "Use a stronger model.")
    assert requested["phase"] == "execution"
    assert runtime["tasks"]["EXP-1"]["status"] == "ready"
    assert runtime["tasks"]["EXP-2"]["status"] == "pending"
    assert runtime["tasks"]["EXP-3"]["status"] == "done"
    assert runtime["tasks"]["EXP-1"]["operation_revision"] == 1


def test_targeted_redo_preserves_prior_result_and_versions_the_new_one():
    state = _approved_state(_plan(_task("EXP-1")))
    first = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    _artifact(state, "EXP-1")
    prior = record_result(state, "EXP-1", first["attempt_id"], _success_result("EXP-1"))[
        "task_result"
    ]
    base_operation_key = first["operation_key"]
    base_attempts = state[ATTEMPT_LEDGER_KEY][base_operation_key]["attempts"]
    mark_result_review(
        state, approved=False, feedback="Repeat with revised parameters.",
        selected_task_ids=["EXP-1"],
    )
    second = start_task(state, "EXP-1")
    assert second["operation_key"] != base_operation_key
    assert state[ATTEMPT_LEDGER_KEY][base_operation_key]["attempts"] == base_attempts
    assert state[ATTEMPT_LEDGER_KEY][second["operation_key"]]["attempts"] == 1
    mark_route_returned(state, "FedotAgent")
    _artifact(state, "EXP-1")
    latest = record_result(state, "EXP-1", second["attempt_id"], _success_result("EXP-1"))[
        "task_result"
    ]
    assert len(state["experiment_runtime"]["results"]) == 2
    assert prior["result_version"] == 1
    assert latest["result_version"] == 2
    assert latest["supersedes_result_id"] == prior["result_id"]


def test_result_review_id_is_stable_for_replay_and_changes_for_new_result_revision(
    monkeypatch,
):
    monkeypatch.delenv("COSCIENTIST_EXPERIMENT_HITL_AUTO_APPROVE", raising=False)
    state = _approved_state(_plan(_task("EXP-1")))
    first = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    _artifact(state, "EXP-1")
    record_result(state, "EXP-1", first["attempt_id"], _success_result("EXP-1"))
    runtime = state["experiment_runtime"]

    stable_id, stable_signature = result_review_identity(runtime)
    assert result_review_identity(copy.deepcopy(runtime)) == (stable_id, stable_signature)

    handler = _CapturingResultReviewHandler()
    agent = ExperimentReviewSessionAgent(
        name="ResultReviewer", review_kind="result", hitl_handler=handler,
    )
    ctx = SimpleNamespace(
        session=SimpleNamespace(state=state), invocation_id="review-invocation",
    )
    asyncio.run(agent._review_result(ctx, ""))
    assert handler.requests[0].context["experiment_review_id"] == stable_id
    assert handler.requests[0].context["experiment_targeted_redo"][
        "selection_required_for_redo"
    ] is True
    assert state["experiment_result_review_id"] == stable_id

    request_task_redo(state, ["EXP-1"], "Explicitly revise the scientific run.")
    second = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    _artifact(state, "EXP-1")
    record_result(state, "EXP-1", second["attempt_id"], _success_result("EXP-1"))
    revised_id, revised_signature = result_review_identity(runtime)
    assert revised_id != stable_id
    assert revised_signature != stable_signature


def test_approved_result_review_preserves_operator_notes(monkeypatch):
    monkeypatch.delenv("COSCIENTIST_EXPERIMENT_HITL_AUTO_APPROVE", raising=False)
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    _artifact(state, "EXP-1")
    record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))

    handler = _CapturingResultReviewHandler(HITLResponse(
        action=HITLAction.APPROVE,
        approved=True,
        instructions="Accepted; mention the small sample in the final report.",
    ))
    agent = ExperimentReviewSessionAgent(
        name="ResultReviewer", review_kind="result", hitl_handler=handler,
    )
    ctx = SimpleNamespace(
        session=SimpleNamespace(state=state), invocation_id="review-with-notes",
    )
    response = asyncio.run(agent._review_result(ctx, ""))

    runtime = state["experiment_runtime"]
    assert response.approved is True
    assert runtime["phase"] == "completed"
    assert runtime["result_review_disposition"] == "accepted"
    assert runtime["result_review_notes"] == (
        "Accepted; mention the small sample in the final report."
    )
    assert state["experiment_module_outcome"]["reason"] == "result_approved"


def test_unselected_quality_feedback_finishes_without_replan():
    state = _approved_state(_plan(_task("EXP-1")))
    state["experiment_runtime"]["phase"] = "reporting"
    outcome = mark_result_review(state, approved=False, feedback="Metric is too low.")
    assert outcome["phase"] == "completed"
    assert outcome["result_review_disposition"] == "changes_suggested"
    assert outcome["targeted_redo_available"] is True
    assert state["experiment_runtime"]["replan_rounds"] == 0


def test_plan_view_exposes_revision_and_only_valid_next_actions():
    state = _approved_state(_plan(_task("EXP-1")))
    initial = get_experiment_plan(state)
    assert initial["next_actions"] == [
        {"tool": "start_task", "arguments": {"task_id": "EXP-1"}}
    ]
    started = start_task(state, "EXP-1")
    running = get_experiment_plan(state)
    assert running["state_revision"] != initial["state_revision"]
    assert running["next_actions"][0]["tool"] == "FedotAgent"
    mark_route_returned(state, "FedotAgent")
    returned = get_experiment_plan(state)
    assert returned["next_actions"] == [{
        "tool": "record_result",
        "arguments": {"task_id": "EXP-1", "attempt_id": started["attempt_id"]},
        "required_arguments": ["result"],
    }]
    state["experiment_manual_pause"] = True
    paused = get_experiment_plan(state)
    assert paused["next_actions"] == []


def test_selector_contract_accepts_adk_dict_like_state():
    state = _approved_state(_plan(_task("EXP-1")))

    class DictLikeState:
        def __init__(self, values: dict) -> None:
            self.values = values

        def get(self, key, default=None):
            return self.values.get(key, default)

    wrapped = DictLikeState(state)
    assert experiment_state_revision(wrapped).startswith("STATE-")
    assert experiment_next_actions(wrapped) == [
        {"tool": "start_task", "arguments": {"task_id": "EXP-1"}}
    ]
