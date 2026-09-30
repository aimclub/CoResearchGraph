import asyncio
from types import SimpleNamespace

import pytest
from google.adk.agents import LlmAgent
from google.adk.models import LlmResponse
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import PrivateAttr

from CoScientist.agents.callbacks.pilot_exact_retrieval import (
    ensure_pilot_exact_tool_retrieved,
)
from CoScientist.assembly.schema import load_config, resolve_config_path


def _context(
    state=None,
    task="Target tool: dataset_overview_heracleum_tox (server_id=server-a)",
):
    return SimpleNamespace(
        state={} if state is None else state,
        invocation_id="run-1",
        user_content=types.Content(parts=[types.Part(text=task)]),
    )


def _final(text="Done"):
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


def test_missing_explicit_tool_requests_exact_retrieval():
    result = ensure_pilot_exact_tool_retrieved(_context(), _final())
    call = result.content.parts[0].function_call
    assert call.name == "retrieve_tools"
    assert dict(call.args) == {"query": "dataset_overview_heracleum_tox"}


def test_matching_tool_and_server_needs_no_extra_lookup():
    state = {"accumulated_tools": [
        {"tool": "dataset_overview_heracleum_tox", "server_id": "server-a"}
    ]}
    assert ensure_pilot_exact_tool_retrieved(_context(state), _final()) is None


def test_same_name_on_wrong_server_still_requires_lookup():
    state = {"accumulated_tools": [
        {"tool": "dataset_overview_heracleum_tox", "server_id": "server-b"}
    ]}
    result = ensure_pilot_exact_tool_retrieved(_context(state), _final())
    assert result.content.parts[0].function_call.name == "retrieve_tools"


def test_name_without_server_id_still_requires_lookup():
    state = {"accumulated_tools": [
        {"tool": "dataset_overview_heracleum_tox"}
    ]}
    result = ensure_pilot_exact_tool_retrieved(_context(state), _final())
    assert result.content.parts[0].function_call.name == "retrieve_tools"


def test_other_server_id_in_task_does_not_override_target_server():
    task = (
        "Context server_id=server-b. "
        "Target tool: dataset_overview_heracleum_tox (server_id=server-a)"
    )
    state = {"accumulated_tools": [
        {"tool": "dataset_overview_heracleum_tox", "server_id": "server-a"}
    ]}
    assert ensure_pilot_exact_tool_retrieved(_context(state, task), _final()) is None


def test_no_explicit_target_keeps_model_response():
    assert (
        ensure_pilot_exact_tool_retrieved(_context(task="Find tools"), _final())
        is None
    )


def test_partial_model_response_is_not_replaced():
    response = _final()
    response.partial = True
    assert ensure_pilot_exact_tool_retrieved(_context(), response) is None


def test_model_originated_function_call_is_not_replaced():
    response = LlmResponse(content=types.Content(role="model", parts=[
        types.Part.from_function_call(
            name="retrieve_tools", args={"query": "Heracleum"}
        )
    ]))
    assert ensure_pilot_exact_tool_retrieved(_context(), response) is None


def test_second_missing_result_raises_instead_of_looping():
    ctx = _context()
    ensure_pilot_exact_tool_retrieved(ctx, _final())
    with pytest.raises(RuntimeError, match="after exact lookup"):
        ensure_pilot_exact_tool_retrieved(ctx, _final())


def test_interleaved_targets_keep_independent_attempt_limits():
    state = {}
    first = _context(state)
    second = _context(state, "Target tool: predict_ld50 (server_id=server-a)")
    ensure_pilot_exact_tool_retrieved(first, _final())
    ensure_pilot_exact_tool_retrieved(second, _final())
    with pytest.raises(RuntimeError, match="after exact lookup"):
        ensure_pilot_exact_tool_retrieved(first, _final())


def test_callback_is_confined_to_synapse_pilot_profile():
    pilot = load_config(resolve_config_path("synapse_pilot"))
    demo = load_config(resolve_config_path("synapse_demo"))
    name = "ensure_pilot_exact_tool_retrieved"
    assert name in pilot.agent("ToolRetrieverAgent").callbacks.after_model
    assert name not in demo.agent("ToolRetrieverAgent").callbacks.after_model


class _FinalOnlyModel(BaseLlm):
    _calls: int = PrivateAttr(default=0)

    def __init__(self):
        super().__init__(model="scripted-final-only")

    async def generate_content_async(self, llm_request, stream=False):
        self._calls += 1
        yield _final()


def test_adk_executes_exact_lookup_once_before_failing_closed():
    seen = []

    async def retrieve_tools(query: str) -> dict:
        seen.append(query)
        return {"status": "ok", "result": []}

    async def run():
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="exact_retrieval", user_id="user", session_id="session"
        )
        agent = LlmAgent(
            name="ToolRetrieverProbe",
            model=_FinalOnlyModel(),
            instruction="Find tools.",
            tools=[retrieve_tools],
            after_model_callback=ensure_pilot_exact_tool_retrieved,
        )
        runner = Runner(
            agent=agent, app_name="exact_retrieval", session_service=sessions
        )
        events = []
        try:
            async for event in runner.run_async(
                user_id="user",
                session_id="session",
                new_message=types.Content(role="user", parts=[types.Part(text=(
                    "Target tool: dataset_overview_heracleum_tox (server_id=server-a)"
                ))]),
            ):
                events.append(event)
        except RuntimeError as exc:
            assert "after exact lookup" in str(exc)
        else:
            raise AssertionError("missing tool was accepted")
        return events, await sessions.get_session(
            app_name="exact_retrieval", user_id="user", session_id="session"
        )

    events, session = asyncio.run(run())
    assert seen == ["dataset_overview_heracleum_tox"]
    assert any(
        response.name == "retrieve_tools"
        for event in events for response in event.get_function_responses()
    )
    assert not session.state.get("accumulated_tools")


def test_adk_accepts_exact_tool_after_real_lookup():
    seen = []

    async def retrieve_tools(query: str, tool_context: ToolContext) -> dict:
        seen.append(query)
        tool_context.state["accumulated_tools"] = [{
            "tool": query, "server_id": "server-a"
        }]
        return {"status": "ok", "result": [{"tool": query}]}

    async def run():
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="exact_retrieval_success", user_id="user", session_id="session"
        )
        agent = LlmAgent(
            name="ToolRetrieverSuccessProbe",
            model=_FinalOnlyModel(),
            instruction="Find tools.",
            tools=[retrieve_tools],
            after_model_callback=ensure_pilot_exact_tool_retrieved,
        )
        runner = Runner(
            agent=agent, app_name="exact_retrieval_success", session_service=sessions
        )
        events = [event async for event in runner.run_async(
            user_id="user",
            session_id="session",
            new_message=types.Content(role="user", parts=[types.Part(text=(
                "Target tool: dataset_overview_heracleum_tox (server_id=server-a)"
            ))]),
        )]
        return events

    events = asyncio.run(run())
    assert seen == ["dataset_overview_heracleum_tox"]
    assert any(
        response.name == "retrieve_tools"
        for event in events for response in event.get_function_responses()
    )
    assert events[-1].content.parts[0].text == "Done"
