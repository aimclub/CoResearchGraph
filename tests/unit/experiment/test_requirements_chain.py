"""Question and deliverable plans run without a hypothesis.

The executor is the existing runtime with a canned route result. This is not
a live chemistry run.
"""
from __future__ import annotations

import pytest

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.critique.validator import critique_plan
from CoScientist.experiments.runtime import (
    ExperimentRuntimeError,
    logical_operation_key,
    operation_attempt_count,
    record_result,
    start_task,
)
from CoScientist.requirements.execution import is_infrastructure_failure

from .helpers import _approved_state, _design, _inventory, _plan, _route_return, _success_result, _task

_SETTINGS = ExperimentsSettings(route_fedot=True)


@pytest.fixture(autouse=True)
def _fedot_attached(monkeypatch):
    """The chemistry route is on for these plans even if another test built the app."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().experiments, "route_fedot", True)
    monkeypatch.setattr(get_settings().web, "fedot_fallback_enabled", True)


def _bare(task_id: str, **design_extra):
    design = _design("")
    design["hypothesis_ref"] = ""
    design.update(design_extra)
    task = _task(task_id, hypothesis_ref="", design=design)
    return task


def test_b05_a_plan_with_no_hypothesis_passes_critique_and_runtime():
    task = _bare("EXP-1", target_refs=["Q-1"], step_key="answer")
    plan = _plan(task, hypotheses=[])
    assert plan.hypotheses == []
    assert plan.tasks[0].design.hypothesis_ref == ""
    critique = critique_plan(
        plan,
        settings=_SETTINGS,
        hypothesis_refs=[],
        requirement_refs=[{"id": "Q-1", "obligation": True, "kind": "question"}],
        available_tools=_inventory(),
    )
    majors = [i.message for i in critique.issues if i.severity in {"blocker", "major"}]
    assert majors == []
    state = _approved_state(plan)
    started = start_task(state, "EXP-1")
    _route_return(state, "FedotAgent")
    state["fedot_artifacts"] = [{
        "name": "exp-1-result.csv",
        "bucket": "managed-experiments",
        "s3_key": "experiments/run/plan/EXP-1/attempt/result.csv",
        "tool": "estimate_property",
    }]
    stored = record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))
    assert stored["task_result"]["status"] == "success"
    assert state["experiment_runtime"]["phase"] == "reporting"


def test_b06_coverage_requires_the_obligation_not_every_preparatory_step():
    prep = _bare("EXP-1", step_key="prep")
    final = _bare("EXP-2", target_refs=["DL-1"], step_key="deliver")
    final["depends_on"] = ["EXP-1"]
    plan = _plan(prep, final, hypotheses=[])
    critique = critique_plan(
        plan,
        settings=_SETTINGS,
        hypothesis_refs=[],
        requirement_refs=[{"id": "DL-1", "obligation": True, "kind": "deliverable"}],
        available_tools=_inventory(),
    )
    assert not any("DL-1" in i.message and i.severity == "major" for i in critique.issues)

    only_report = _bare("EXP-1", step_key="report")
    missed = critique_plan(
        _plan(only_report, hypotheses=[]),
        settings=_SETTINGS,
        hypothesis_refs=[],
        requirement_refs=[{"id": "DL-1", "obligation": True}],
        available_tools=_inventory(),
    )
    assert any("DL-1" in i.message and i.severity == "major" for i in missed.issues)


def test_b07_steps_of_one_operation_do_not_share_an_attempt_budget():
    first = _bare("EXP-1", operation_ref="OP-1", step_key="screen", target_refs=["DL-1"])
    second = _bare("EXP-2", operation_ref="OP-1", step_key="dock", target_refs=["DL-1"])
    renamed = {**first, "name": "renamed step", "route": "coder"}
    assert logical_operation_key(first) == logical_operation_key(renamed)
    assert logical_operation_key(first) != logical_operation_key(second)
    plan = _plan(first, second, hypotheses=[])
    state = _approved_state(plan)
    started = start_task(state, "EXP-1")
    key_first = state["experiment_runtime"]["tasks"]["EXP-1"]["operation_key"]
    key_second = state["experiment_runtime"]["tasks"]["EXP-2"]["operation_key"]
    assert key_first != key_second
    assert operation_attempt_count(state, key_first) == 1
    assert operation_attempt_count(state, key_second) == 0
    assert started["attempt_id"]


def test_b08_service_unavailable_is_not_a_verdict_and_blocks_the_sibling():
    first = _bare("EXP-1", step_key="a")
    second = _bare("EXP-2", step_key="b")
    state = _approved_state(_plan(first, second, hypotheses=[]))
    started = start_task(state, "EXP-1")
    _route_return(state, "FedotAgent")
    state["experiment_runtime"]["tasks"]["EXP-1"]["attempts"][started["attempt_id"]]["last_tool_observation"] = {"tool": "estimate_property", "is_error": True}
    payload = {
        "status": "failure",
        "error_code": "service_unavailable",
        "error_message": "service unavailable",
        "summary": "service unavailable",
        "retryable": False,
        "criteria_checks": [{
            "criterion_id": "EXP-1-C1",
            "passed": False,
            "details": "the server did not answer",
        }],
    }
    assert is_infrastructure_failure(payload)
    stored = record_result(state, "EXP-1", started["attempt_id"], payload)
    assert stored["task_result"]["status"] == "failure"
    assert stored["task_result"].get("scientific_check") in (None, {})
    with pytest.raises(ExperimentRuntimeError) as caught:
        start_task(state, "EXP-2", settings=ExperimentsSettings(route_fedot=True, allow_coder_fallback=False))
    assert caught.value.code == "service_unavailable"


def test_known_outage_opens_coder_before_skip_when_fallback_is_allowed():
    first = _bare("EXP-1", step_key="a")
    second = _bare("EXP-2", step_key="b")
    second["code_assessment"] = {"entrypoints": ["analysis.py"]}
    state = _approved_state(_plan(first, second, hypotheses=[]))
    started = start_task(state, "EXP-1")
    _route_return(state, "FedotAgent")
    state["experiment_runtime"]["tasks"]["EXP-1"]["attempts"][started["attempt_id"]]["last_tool_observation"] = {"tool": "estimate_property", "is_error": True}
    record_result(state, "EXP-1", started["attempt_id"], {
        "status": "failure",
        "error_code": "service_unavailable",
        "error_message": "service unavailable",
        "summary": "service unavailable",
        "retryable": False,
        "criteria_checks": [{
            "criterion_id": "EXP-1-C1",
            "passed": False,
            "details": "the server did not answer",
        }],
    })
    cfg = ExperimentsSettings(route_fedot=True, allow_coder_fallback=True)
    opened = start_task(
        state, "EXP-2", settings=cfg, route_agents={"CoderAgent", "FedotAgent"},
    )
    assert opened["route"] == "coder"


def test_b16_contract_check_ignores_the_tool_name():
    from CoScientist.experiments.capabilities.contracts import dataset_mismatch

    task = {
        "design": {"dataset": {"dataset_scope": "binding-set"}},
        "input_data": [],
    }
    same = {
        "tool": "alpha_tool",
        "data_contract": {"input_mode": "fixed_dataset", "dataset_scope": "binding-set"},
    }
    renamed = {**same, "tool": "totally_different_name"}
    incompatible = {
        "tool": "alpha_tool",
        "data_contract": {"input_mode": "fixed_dataset", "dataset_scope": "other-set"},
    }
    assert dataset_mismatch(task, same) is None
    assert dataset_mismatch(task, renamed) is None
    assert dataset_mismatch(task, incompatible) is not None


@pytest.mark.parametrize("scope,blocked", [("tool", False), ("server", True)])
def test_failure_scope_does_not_disable_unrelated_tools(scope, blocked):
    from CoScientist.requirements.execution import note_service_stop, service_block_reason
    first = _task("EXP-1", tool="dock")
    second = _task("EXP-2", tool="properties")
    state = {}
    note_service_stop(state, first, {"tool": "dock", "failure_scope": scope, "is_error": True})
    model = _plan(second).tasks[0]
    assert bool(service_block_reason(state, model)) is blocked
    same = _plan(_task("EXP-3", tool="dock")).tasks[0]
    assert service_block_reason(state, same)


@pytest.mark.parametrize("method,enabled,allowed", [
    (None, True, False), ("Compute the same measurements with the approved local method.", True, True),
    ("Compute the same measurements with the approved local method.", False, False),
])
def test_known_outage_uses_explicit_method_not_repository(method, enabled, allowed):
    from CoScientist.requirements.execution import note_service_stop
    first, second = _task("EXP-1"), _task("EXP-2")
    second["coder_fallback_method"] = method
    second["code_assessment"] = {"requirement": "unknown", "entrypoints": []}
    state = _approved_state(_plan(first, second))
    note_service_stop(state, first, {"tool": "estimate_property", "is_error": True})
    cfg = ExperimentsSettings(route_fedot=True, allow_coder_fallback=enabled)
    before = state["experiment_runtime"]["tasks"]["EXP-2"]["operation_key"]
    if allowed:
        opened = start_task(state, "EXP-2", settings=cfg, route_agents={"FedotAgent", "CoderAgent"})
        assert opened["route"] == "coder"
        assert opened["coder_brief"]["method"] == method
        assert operation_attempt_count(state, before) == 1
    else:
        with pytest.raises(ExperimentRuntimeError, match="Service unavailable"):
            start_task(state, "EXP-2", settings=cfg, route_agents={"FedotAgent", "CoderAgent"})
        assert operation_attempt_count(state, before) == 0
    assert logical_operation_key({**second, "coder_fallback_method": "another implementation"}) == logical_operation_key(second)


def test_diagnostic_prose_and_successful_observations_do_not_blacklist_tools():
    from CoScientist.requirements.execution import note_service_stop, service_block_reason
    assert not is_infrastructure_failure({"status": "success", "summary": "Recovered after connection refused"})
    state = {}
    note_service_stop(state, _task("EXP-1"), {"tool": "estimate_property", "is_error": False})
    assert not service_block_reason(state, _plan(_task("EXP-2")).tasks[0])
