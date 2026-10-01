"""ToolResultTruncationPlugin must not hide a large tool result from the agent's
own after_tool callbacks.

ADK skips an agent's after_tool callbacks as soon as a plugin returns a
replacement result. The truncation plugin returns one for every result above
its cap, so the callbacks that file tool answers into the state never ran for
large answers — ``collect_economics_result`` never wrote ``economics_ranking``
from a multi-route ``rank_routes_by_cost`` answer and the experiment hand-off
failed with "economics_ranking: nonempty object required".
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from CoScientist.agents.truncation_plugin import ToolResultTruncationPlugin

CAP = 50


def _context(*callbacks):
    agent = SimpleNamespace(canonical_after_tool_callbacks=list(callbacks))
    return SimpleNamespace(_invocation_context=SimpleNamespace(agent=agent), state={})


def _run(plugin, context, result, tool_name="rank_routes_by_cost"):
    return asyncio.run(plugin.after_tool_callback(
        tool=SimpleNamespace(name=tool_name), tool_args={"a": 1},
        tool_context=context, result=result,
    ))


def test_large_result_reaches_agent_callbacks_in_full_then_is_truncated():
    seen = []

    def callback(tool, args, tool_context, tool_response):
        seen.append((tool.name, args, tool_response))
        tool_context.state["filed"] = tool_response["text"]

    context = _context(callback)
    result = {"text": "x" * 500}
    returned = _run(ToolResultTruncationPlugin(max_chars=CAP), context, result)

    assert seen == [("rank_routes_by_cost", {"a": 1}, {"text": "x" * 500})]
    assert context.state["filed"] == "x" * 500          # the callback saw the FULL text
    assert returned["text"].startswith("x" * CAP)       # the model gets the bounded copy
    assert "truncated 450 chars" in returned["text"]


def test_small_result_leaves_callbacks_to_adk():
    calls = []
    context = _context(lambda **kw: calls.append(kw))

    assert _run(ToolResultTruncationPlugin(max_chars=CAP), context, {"text": "short"}) is None
    assert calls == []  # ADK itself runs them when the plugin returns None


def test_callback_answer_replaces_result_and_is_truncated():
    async def replaces(tool, args, tool_context, tool_response):
        return {"text": "y" * 200}

    never = []
    returned = _run(
        ToolResultTruncationPlugin(max_chars=CAP),
        _context(replaces, lambda **kw: never.append(kw)),
        {"text": "x" * 500},
    )
    assert returned["text"].startswith("y" * CAP)
    assert never == []  # stops at the first non-None answer, as ADK does


def test_short_callback_answer_is_still_returned():
    returned = _run(
        ToolResultTruncationPlugin(max_chars=CAP),
        _context(lambda **kw: {"ok": True}),
        {"text": "x" * 500},
    )
    assert returned == {"ok": True}


def test_context_without_agent_only_truncates():
    context = SimpleNamespace(state={})
    returned = _run(ToolResultTruncationPlugin(max_chars=CAP), context, ["z" * 100])
    assert returned[0].startswith("z" * CAP)


# ── Through the real ADK tool pipeline ──────────────────────────────────────

@pytest.fixture
def adk(monkeypatch):
    monkeypatch.setenv("OPIK_TRACK_DISABLE", "true")
    from google.adk.agents import LlmAgent
    from google.adk.apps.app import App
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    class OneToolCall(BaseLlm):
        """Calls ``tool_name`` once with ``args``, then answers; records what
        the model was shown as the tool's answer."""
        model: str = "scripted"
        tool_name: str = ""
        args: dict = {}
        shown: list = []

        async def generate_content_async(self, llm_request, stream=False):
            answers = [part.function_response for content in llm_request.contents
                       for part in content.parts or [] if part.function_response]
            if not answers:
                call = types.FunctionCall(name=self.tool_name, args=self.args)
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=call)]))
                return
            self.shown.append(answers[-1].response)
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="done")]))

    def run(tool, args, *, state, callbacks, plugins):
        llm = OneToolCall(tool_name=tool.__name__, args=args, shown=[])
        agent = LlmAgent(name="EconomicsAgent", model=llm, tools=[tool], after_tool_callback=callbacks)
        service = InMemorySessionService()
        runner = Runner(app=App(name="t", root_agent=agent, plugins=plugins), session_service=service)

        async def go():
            session = await service.create_session(app_name="t", user_id="u", state=state)
            message = types.Content(role="user", parts=[types.Part(text="go")])
            async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=message):
                pass
            return await service.get_session(app_name="t", user_id="u", session_id=session.id)

        session = asyncio.run(go())
        return session.state, llm.shown

    return run


@pytest.mark.parametrize("size", [10, 50_000])
def test_agent_callback_runs_exactly_once_on_full_result_in_adk(adk, size):
    seen = []

    def big_tool(n: int) -> dict:
        """fixture"""
        return {"text": "x" * n}

    def callback(tool, args, tool_context, tool_response):
        seen.append(len(tool_response["text"]))

    _, shown = adk(big_tool, {"n": size}, state={}, callbacks=[callback],
                   plugins=[ToolResultTruncationPlugin(max_chars=1000)])
    assert seen == [size]
    assert len(shown[-1]["text"]) <= 1000 + 100


def test_rank_routes_answer_above_the_cap_still_fills_economics_ranking(adk):
    """The production defect, end to end through ADK: four routes with their
    breakdown make the answer far larger than the 12k cap."""
    from CoScientist.microfluidics.economics import collect_economics_result

    route_ids = ["LIT-ROUTE-01", "LIT-ROUTE-02", "LIT-ROUTE-03", "LIT-ROUTE-04"]

    def rank_routes_by_cost(routes: list, target_qty: float, target_unit: str) -> dict:
        """fixture: the server's answer shape (JSON inside a text block)"""
        answer = {"preferred_currency": "RUB", "rank_by": "per_unit", "routes": [
            {"route_id": route["route_id"], "status": "ok", "rank": rank, "currency": "RUB",
             "cost_per_unit": f"{10 * rank}.00", "cost_packs": f"{12 * rank}.00",
             "starting_materials": [], "missing": [], "warnings": [],
             "resolved_inputs": [{"input": "filler", "hint": "h" * 4000}]}
            for rank, route in enumerate(routes, 1)
        ]}
        return {"content": [{"type": "text", "text": json.dumps(answer)}], "isError": False}

    state, shown = adk(
        rank_routes_by_cost,
        {"routes": [{"route_id": r} for r in route_ids], "target_qty": 1, "target_unit": "g"},
        state={"qualified_routes": {"status": "ok", "routes": [{"route_id": r} for r in route_ids]}},
        callbacks=[collect_economics_result],
        plugins=[ToolResultTruncationPlugin()],
    )
    assert "truncated" in json.dumps(shown[-1])  # the model saw the bounded copy
    ranking = state["economics_ranking"]
    assert sorted(ranking["routes"]) == route_ids
    assert ranking["routes"]["LIT-ROUTE-01"]["cost_per_unit"] == "10.00"
    assert ranking["target_qty"] == 1 and ranking["target_unit"] == "g"
