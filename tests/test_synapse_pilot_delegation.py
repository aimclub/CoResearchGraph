"""The pilot verifies observed ADK delegation without directing tool choice."""

from types import SimpleNamespace

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from CoScientist.agents.callbacks.pilot_delegation import require_pilot_delegations


REQUIRED = ("retrieve_tools", "ResearchAgent", "TaskExecutorAgent")


def _event(name, *, invocation_id="run-1", result=None, include_call=True):
    call = types.FunctionCall(name=name, args={})
    response = types.FunctionResponse(
        name=name, response={"result": "ok"} if result is None else result
    )
    return SimpleNamespace(
        invocation_id=invocation_id,
        get_function_calls=lambda: [call] if include_call else [],
        get_function_responses=lambda: [response],
    )


def _context(*events):
    invocation = SimpleNamespace(
        invocation_id="run-1", session=SimpleNamespace(events=list(events))
    )
    return SimpleNamespace(_invocation_context=invocation)


def _model_response(*, tool=None, partial=False):
    part = (
        types.Part.from_function_call(name=tool, args={})
        if tool
        else types.Part(text="Pilot report")
    )
    return LlmResponse(
        content=types.Content(role="model", parts=[part]), partial=partial
    )


def test_pilot_accepts_extra_calls_and_required_delegations_in_any_order():
    context = _context(
        _event("retrieve_tools"),
        _event("HypothesesAgent"),
        _event("TaskExecutorAgent"),
        _event("ResearchAgent"),
    )
    assert require_pilot_delegations(context, _model_response()) is None


def test_pilot_allows_intermediate_tool_calls():
    context = _context(_event("retrieve_tools"))
    assert (
        require_pilot_delegations(
            context, _model_response(tool="HypothesesAgent")
        )
        is None
    )
    assert require_pilot_delegations(context, _model_response(partial=True)) is None


@pytest.mark.parametrize("missing", REQUIRED)
def test_pilot_rejects_final_report_without_real_call_and_response(missing):
    events = [_event(name) for name in REQUIRED if name != missing]
    with pytest.raises(RuntimeError, match=missing):
        require_pilot_delegations(_context(*events), _model_response())


def test_pilot_ignores_other_invocations_and_failed_results():
    context = _context(
        _event("retrieve_tools"),
        _event("ResearchAgent", result={"status": "error", "message": "failed"}),
        _event("ResearchAgent", invocation_id="old-run"),
        _event("TaskExecutorAgent"),
    )
    with pytest.raises(RuntimeError, match="ResearchAgent"):
        require_pilot_delegations(context, _model_response())


def test_pilot_requires_a_call_as_well_as_a_response():
    context = _context(
        _event("retrieve_tools"),
        _event("ResearchAgent", include_call=False),
        _event("TaskExecutorAgent"),
    )
    with pytest.raises(RuntimeError, match="ResearchAgent"):
        require_pilot_delegations(context, _model_response())


def test_pilot_profile_wires_observation_without_changing_regular_demo():
    from CoScientist.assembly import build_system
    from CoScientist.assembly.schema import load_config, resolve_config_path

    pilot = load_config(resolve_config_path("synapse_pilot"))
    regular = load_config(resolve_config_path("synapse_demo"))
    callbacks = pilot.agent("OrchestratorAgent").callbacks
    regular_callbacks = regular.agent("OrchestratorAgent").callbacks

    assert callbacks.after_model[0] == "require_pilot_delegations"
    assert "require_pilot_tool" not in callbacks.before_model
    assert "require_pilot_expected_tool" not in callbacks.before_tool
    assert "guard_unknown_tools" not in callbacks.after_model
    assert "require_pilot_delegations" not in regular_callbacks.after_model

    orchestrator = build_system(pilot, remote_subagents=True).root
    assert {tool.name for tool in orchestrator.tools if hasattr(tool, "name")} >= {
        "ResearchAgent",
        "TaskExecutorAgent",
    }
    assert require_pilot_delegations in orchestrator.canonical_after_model_callbacks


def test_pilot_reasoning_matches_gpt_oss_gateway_without_changing_regular_demo():
    from CoScientist.assembly.schema import load_config, resolve_config_path

    pilot = load_config(resolve_config_path("synapse_pilot"))
    regular = load_config(resolve_config_path("synapse_demo"))

    assert pilot.defaults.reasoning == "medium"
    assert pilot.agent("HypothesesAgent").reasoning == "medium"
    assert regular.defaults.reasoning is False
    assert regular.agent("HypothesesAgent").reasoning == "high"
