"""MCP technical failure may run once through Coder when that is allowed."""
from __future__ import annotations

import json

import pytest

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.review import render_experiment_results
from CoScientist.experiments.runtime import fallback_task, record_result, start_task
from CoScientist.experiments.runtime.state_machine import operation_attempt_count
from CoScientist.requirements.coverage import projection_from_state

from .helpers import _approved_state, _design, _plan, _route_return, _success_result, _task

_ON = ExperimentsSettings(route_fedot=True, allow_coder_fallback=True)
_OFF = ExperimentsSettings(route_fedot=True, allow_coder_fallback=False)


@pytest.fixture(autouse=True)
def _fedot_attached(monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().web, "fedot_fallback_enabled", True)
    monkeypatch.setattr(get_settings().experiments, "route_fedot", True)


def _coded(task_id: str = "EXP-1", **extra):
    design = _design("H1")
    design["target_refs"] = ["Q-1", "H1"]
    raw = _task(task_id, design=design)
    raw["code_assessment"] = {
        "requirement": "unknown",
        "evidence": "score.py computes the same table the MCP would have returned.",
        "entrypoints": ["score.py"],
    }
    raw.update(extra)
    return raw


def _outage(retryable: bool = False, **extra) -> dict:
    payload = {
        "status": "failure",
        "error_code": "connection_refused",
        "error_message": "connection refused",
        "summary": "connection refused",
        "retryable": retryable,
        "criteria_checks": [{
            "criterion_id": "EXP-1-C1",
            "passed": False,
            "details": "the server did not answer",
        }],
    }
    payload.update(extra)
    return payload


def _run(plan, task_id: str, payload: dict, settings: ExperimentsSettings):
    state = _approved_state(plan)
    state["experiment_context"] = {
        "requirement_refs": [{
            "id": "Q-1", "kind": "question", "formulation": "Which compounds bind?",
        }],
        "hypothesis_refs": [{"hypothesis_id": "H1", "statement": "Compound X inhibits the target."}],
    }
    state["report_language"] = "en"
    started = start_task(state, task_id, settings=settings)
    _route_return(state, "FedotAgent")
    stored = record_result(
        state, task_id, started["attempt_id"], payload, settings=settings,
    )
    return state, started, stored


def test_flag_off_does_not_call_coder():
    state, _started, _stored = _run(_plan(_coded()), "EXP-1", _outage(), _OFF)
    task = state["experiment_runtime"]["tasks"]["EXP-1"]
    assert task["status"] == "failed"
    assert "coder_fallback" not in task
    assert task["current_route"] == "fedot_mas"


@pytest.mark.parametrize("route,code,retryable,attempts,total_left,next_route,status,direct_coder", [
    ("react_tools", "connection_refused", False, 1, True, "coder", "fallback_pending", True),
    ("fedot_mas", "connection_refused", True, 1, True, "react_tools", "fallback_pending", True),
    ("fedot_mas", "timeout", False, 1, True, "react_tools", "retry_pending", False),
    ("fedot_mas", "timeout", False, 2, True, "react_tools", "fallback_pending", False),
    ("react_tools", "connection_refused", False, 1, False, None, "failed", False),
    ("coder", "tool_error", True, 1, True, None, "retry_pending", False),
    ("coder", "tool_error", True, 2, True, None, "failed", False),
    ("react_tools", "tool_error", False, 1, True, "coder", "fallback_pending", False),
    ("fedot_mas", "output_unbound", True, 1, True, "react_tools", "failed", False),
])
def test_one_decision_preserves_retry_chain_and_direct_outage_fallback(
    route, code, retryable, attempts, total_left, next_route, status, direct_coder,
):
    from CoScientist.experiments.runtime.coder_fallback import coder_fallback_decision

    task = _plan(_coded()).tasks[0]
    decision = coder_fallback_decision(
        result=_outage(retryable=retryable, error_code=code),
        task_runtime={"current_route": route, "route_history": [{"route": route}]},
        task=task, runtime={}, settings=_ON, route_agents={"CoderAgent", "ExperimentAgent", "FedotAgent"},
        attempts_on_route=attempts, total_left=total_left, assessment_only=False,
        next_fallback=next_route, technical_failure=True,
    )
    assert decision["status"] == status
    assert bool(decision.get("coder_fallback")) is direct_coder


def test_new_plan_must_explicitly_decide_fallback_but_stored_plans_remain_readable():
    from CoScientist.experiments.critique.validator import PlanValidationError, validate_and_critique_plan
    from CoScientist.experiments.schemas import ExperimentPlan

    payload = _plan(_coded()).model_dump(mode="json")
    del payload["tasks"][0]["coder_fallback_method"]
    assert ExperimentPlan.model_validate(payload).tasks[0].coder_fallback_method is None
    with pytest.raises(PlanValidationError, match="Fallback decision is missing"):
        validate_and_critique_plan(payload, settings=_ON)


@pytest.mark.parametrize("code", ["service_unavailable", "UPSTREAM_SERVICE_UNAVAILABLE", "INFRASTRUCTURE_FAILURE"])
def test_outage_with_saved_diagnostics_is_not_a_negative_scientific_result(code):
    raw = _coded()
    raw["success_criteria"] = [
        {"criterion_id": "saved", "kind": "execution", "description": "Response saved", "purpose": "execution", "verification": "Read response"},
        {"criterion_id": "score", "kind": "expert", "description": "Scientific target", "purpose": "assessment", "verification": "Assess score"},
    ]
    payload = _outage(error_code=code, criteria_checks=[
        {"criterion_id": "saved", "passed": True, "details": "Error response saved"},
        {"criterion_id": "score", "passed": False, "details": "No score because service failed"},
    ])
    state, _, stored = _run(_plan(raw), "EXP-1", payload, _ON)
    assert stored["task_result"]["status"] == "failure"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "fallback_pending"


def test_flag_on_switches_to_coder_and_keeps_the_step_and_the_contract():
    plan = _plan(_coded())
    state, started, _stored = _run(plan, "EXP-1", _outage(), _ON)
    task = state["experiment_runtime"]["tasks"]["EXP-1"]
    operation_key = task["operation_key"]
    assert task["status"] == "fallback_pending"
    assert operation_attempt_count(state, operation_key) == 1
    moved = fallback_task(state, "EXP-1", "MCP connection refused", settings=_ON)
    assert moved["route"] == "coder"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["operation_key"] == operation_key
    assert "coder_fallback" not in state["experiment_runtime"]["tasks"]["EXP-1"]
    envelope = start_task(state, "EXP-1", settings=_ON)
    assert envelope["route"] == "coder"
    assert envelope["operation_key"] == operation_key
    assert envelope["coder_brief"]["target_refs"] == ["Q-1", "H1"]
    assert envelope["coder_brief"]["mcp_diagnostics"]["error_code"] == "connection_refused"
    assert "unavailable MCP" in envelope["coder_brief"]["limits"]
    assert operation_attempt_count(state, operation_key) == 2

    resumed = json.loads(json.dumps(state))
    assert operation_attempt_count(resumed, operation_key) == 2
    assert resumed["experiment_runtime"]["tasks"]["EXP-1"]["coder_brief"]["target_refs"] == ["Q-1", "H1"]

    _route_return(resumed, "CoderAgent")
    resumed["fedot_artifacts"] = [{
        "name": "exp-1-result.csv",
        "bucket": "managed-experiments",
        "s3_key": "experiments/run/EXP-1/result.csv",
        "tool": "score.py",
    }]
    recorded = record_result(
        resumed, "EXP-1", envelope["attempt_id"],
        _success_result("EXP-1") | {"summary": "Coder wrote the binding table.", "outputs": {"verdict": "confirmed"}},
        settings=_ON,
    )
    assert recorded["task_result"]["status"] == "success"
    assert recorded["task_result"]["route_used"] == "coder"
    rows = {row["id"]: row for row in projection_from_state(resumed)}
    assert rows["Q-1"]["status"] == "met"
    assert rows["H1"]["status"] == "met"
    report = render_experiment_results(resumed)
    assert "`fedot_mas` → `coder`" in report
    assert "unavailable MCP" in report


def test_coder_is_not_substituted_when_it_cannot_preserve_the_step():
    other = _task("EXP-2")
    state, _started, _stored = _run(_plan({**_task("EXP-1"), "coder_fallback_method": None}, other), "EXP-1", _outage(), _ON)
    failed = state["experiment_runtime"]["tasks"]["EXP-1"]
    assert failed["status"] == "failed"
    assert "coder_fallback" not in failed
    assert "No alternative Coder method" in failed["last_message"]
    assert state["experiment_runtime"]["tasks"]["EXP-2"]["status"] == "ready"
    assert state["experiment_runtime"]["coder_fallback_stops"]


def test_a_file_handoff_does_not_recompute_and_a_timeout_retries_before_coder():
    state, _started, _stored = _run(
        _plan(_coded()), "EXP-1",
        _outage(retryable=True, error_code="output_unbound", error_message="output handoff failed"),
        _ON,
    )
    task = state["experiment_runtime"]["tasks"]["EXP-1"]
    assert task["status"] == "failed"
    assert "not recomputed" in task["last_message"]
    assert len(task["attempt_order"]) == 1

    transient, started, _stored = _run(
        _plan(_coded()), "EXP-1",
        _outage(error_code="timeout", error_message="timed out"),
        _ON,
    )
    assert transient["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "retry_pending"
    from CoScientist.experiments.runtime import retry_task

    retry_task(transient, "EXP-1", settings=_ON)
    again = start_task(transient, "EXP-1", settings=_ON)
    _route_return(transient, "FedotAgent")
    record_result(
        transient, "EXP-1", again["attempt_id"],
        _outage(error_code="timeout", error_message="timed out"),
        settings=_ON,
    )
    assert transient["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "fallback_pending"
    assert operation_attempt_count(transient, again["operation_key"]) == 2
    assert started["attempt_id"] != again["attempt_id"]


def test_a_missed_scientific_criterion_is_not_a_coder_retry():
    raw = _coded()
    raw["success_criteria"].append({
        "criterion_id": "EXP-1-C2",
        "description": "The score clears the bar.",
        "kind": "threshold",
        "purpose": "assessment",
        "metric": "score",
        "operator": ">=",
        "target": 0.8,
        "verification": "Read the score from the table.",
    })
    payload = {
        "status": "failure",
        "error_code": "criteria_failed",
        "error_message": "score 0.2 is below 0.8",
        "summary": "The assay ran and the score missed the bar.",
        "retryable": False,
        "criteria_checks": [
            {"criterion_id": "EXP-1-C1", "passed": True, "details": "the tool returned a table"},
            {"criterion_id": "EXP-1-C2", "passed": False, "details": "score 0.2 < 0.8"},
        ],
    }
    state, _started, stored = _run(_plan(raw), "EXP-1", payload, _ON)
    assert stored["task_result"]["status"] == "failure"
    task = state["experiment_runtime"]["tasks"]["EXP-1"]
    assert task["status"] == "failed"
    assert "coder_fallback" not in task
    rows = {row["id"]: row for row in projection_from_state(state)}
    assert rows["H1"]["status"] == "open"


def test_repository_does_not_authorize_substitution():
    raw = _coded(coder_fallback_method=None, repo_url="https://github.com/aimclub/FEDOT")
    state, _, _ = _run(_plan(raw), "EXP-1", _outage(), _ON)
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "failed"


def test_fallback_requires_the_requested_output_not_any_upstream_artifact():
    from CoScientist.experiments.runtime.coder_fallback import _inputs_ready
    from CoScientist.experiments.schemas import ExperimentTask
    raw = _coded()
    raw["input_data"] = [{"data_id": "measurements", "kind": "task_artifact",
        "description": "The measurements table", "required": True,
        "source_task_id": "PREP", "source_output_id": "measurements"}]
    runtime = {"results": [{"task_id": "PREP", "status": "success", "artifacts": [
        {"artifact_id": "log-1", "name": "execution.log", "role": "log", "workspace_path": "/tmp/execution.log"},
    ]}]}
    ready, reason = _inputs_ready(runtime, ExperimentTask.model_validate(raw))
    assert ready is False
    assert reason


def test_disabling_coder_does_not_turn_handoff_into_recomputation():
    state, _, _ = _run(_plan(_coded()), "EXP-1",
        _outage(retryable=True, error_code="output_unbound", error_message="Missing output binding"), _OFF)
    task = state["experiment_runtime"]["tasks"]["EXP-1"]
    assert task["status"] == "failed"
    assert len(task["attempt_order"]) == 1
