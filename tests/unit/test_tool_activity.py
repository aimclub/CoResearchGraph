from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from CoScientist.graph.session_scope import (
    GRAPH_SCOPE_SESSION_KEY,
    GRAPH_SCOPE_USER_KEY,
)
from CoScientist.logging import tool_activity


@pytest.fixture
def sink(monkeypatch):
    received: list[tuple[tuple[str, str], dict]] = []

    async def collect(key, payload):
        received.append((key, payload))

    monkeypatch.setattr(tool_activity, "_sink", collect)
    return received


def agent_tool_context(agent_name: str = "ResearchAgent", call_id: str = "fc_1"):
    """A sub-agent context as AgentTool builds it: child session, parent state."""
    return SimpleNamespace(
        agent_name=agent_name,
        function_call_id=call_id,
        state={
            GRAPH_SCOPE_USER_KEY: "user_1",
            GRAPH_SCOPE_SESSION_KEY: "session_1",
        },
        session=SimpleNamespace(user_id="child", id="child_session_xyz"),
    )


def test_subagent_tool_use_is_reported_under_the_parent_session(sink):
    plugin = tool_activity.ToolActivityPlugin()
    tool = SimpleNamespace(name="tavily_search")
    context = agent_tool_context()

    async def scenario():
        assert await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "ibuprofen"}, tool_context=context
        ) is None
        assert await plugin.after_tool_callback(
            tool=tool, tool_args={}, tool_context=context, result={"answer": "206.29"}
        ) is None

    asyncio.run(scenario())

    # The transient AgentTool child session must not leak: both records belong
    # to the public web session that owns the run.
    assert [key for key, _ in sink] == [("user_1", "session_1")] * 2
    call, result = (payload for _, payload in sink)
    assert call["phase"] == "call"
    assert call["author"] == "ResearchAgent"
    assert call["tool"] == "tavily_search"
    assert call["args"] == {"query": "ibuprofen"}
    assert result["phase"] == "result"
    assert result["result"] == {"answer": "206.29"}
    # Both records carry the call id, so a consumer pairs them exactly instead
    # of matching on tool name (which breaks when a tool is called twice).
    assert call["call_id"] == result["call_id"] == "fc_1"


def test_call_id_is_optional(sink):
    plugin = tool_activity.ToolActivityPlugin()
    context = agent_tool_context()
    del context.function_call_id

    asyncio.run(plugin.before_tool_callback(
        tool=SimpleNamespace(name="tavily_search"), tool_args={}, tool_context=context,
    ))

    (_, payload), = sink
    assert payload["call_id"] is None


def test_tool_error_is_reported(sink):
    plugin = tool_activity.ToolActivityPlugin()

    asyncio.run(plugin.on_tool_error_callback(
        tool=SimpleNamespace(name="tavily_search"),
        tool_args={},
        tool_context=agent_tool_context(),
        error=RuntimeError("rate limited"),
    ))

    (_, payload), = sink
    assert payload["phase"] == "error"
    assert payload["error"] == "rate limited"


def test_oversized_payload_degrades_to_truncated_text(sink):
    plugin = tool_activity.ToolActivityPlugin()

    asyncio.run(plugin.after_tool_callback(
        tool=SimpleNamespace(name="read_file"),
        tool_args={},
        tool_context=agent_tool_context("CoderAgent"),
        result={"blob": "x" * 50_000},
    ))

    (_, payload), = sink
    assert isinstance(payload["result"], str)
    assert len(payload["result"]) <= tool_activity._PREVIEW_LIMIT + 2


def test_a_failing_sink_cannot_break_the_run(monkeypatch):
    async def broken(key, payload):
        raise RuntimeError("sink down")

    monkeypatch.setattr(tool_activity, "_sink", broken)
    plugin = tool_activity.ToolActivityPlugin()

    assert asyncio.run(plugin.before_tool_callback(
        tool=SimpleNamespace(name="tavily_search"),
        tool_args={},
        tool_context=agent_tool_context(),
    )) is None


def test_plugin_is_inert_without_a_sink(monkeypatch):
    monkeypatch.setattr(tool_activity, "_sink", None)
    plugin = tool_activity.ToolActivityPlugin()

    assert asyncio.run(plugin.before_tool_callback(
        tool=SimpleNamespace(name="tavily_search"),
        tool_args={},
        tool_context=agent_tool_context(),
    )) is None
