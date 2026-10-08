from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from CoScientist.graph.session_scope import (
    GRAPH_SCOPE_SESSION_KEY,
    GRAPH_SCOPE_USER_KEY,
)
from CoScientist.logging import agent_output


@pytest.fixture
def sink(monkeypatch):
    received: list[tuple[tuple[str, str], dict]] = []

    async def collect(key, payload):
        received.append((key, payload))

    monkeypatch.setattr(agent_output, "_sink", collect)
    return received


@pytest.fixture
def reported(monkeypatch):
    """Pin the reported set so the test does not depend on system.yaml."""
    monkeypatch.setattr(
        agent_output, "reported_agents", lambda: frozenset({"HypothesesAgent"})
    )


def caller_context(agent_name: str = "OrchestratorAgent", call_id: str = "fc_1"):
    """The DELEGATING agent's context — an AgentTool runs in its parent's."""
    return SimpleNamespace(
        agent_name=agent_name,
        function_call_id=call_id,
        state={
            GRAPH_SCOPE_USER_KEY: "user_1",
            GRAPH_SCOPE_SESSION_KEY: "session_1",
        },
        session=SimpleNamespace(user_id="user_1", id="session_1"),
    )


def run_tool(plugin, name: str, result, context=None):
    return asyncio.run(plugin.after_tool_callback(
        tool=SimpleNamespace(name=name),
        tool_args={"request": "propose approaches"},
        tool_context=context or caller_context(),
        result=result,
    ))


def test_key_agent_answer_is_reported_as_its_own_message(sink, reported):
    plugin = agent_output.AgentOutputPlugin()

    assert run_tool(plugin, "HypothesesAgent", "H1: ...\nH2: ...") is None

    (key, payload), = sink
    assert key == ("user_1", "session_1")
    # The delegated agent authors the message; the caller is only context.
    assert payload["agent"] == "HypothesesAgent"
    assert payload["caller"] == "OrchestratorAgent"
    assert payload["content"] == "H1: ...\nH2: ..."
    assert payload["call_id"] == "fc_1"
    assert payload["timestamp"]


def test_unflagged_agents_and_plain_tools_stay_out_of_the_chat(sink, reported):
    plugin = agent_output.AgentOutputPlugin()

    run_tool(plugin, "ToolReranker", "0.42")
    run_tool(plugin, "tavily_search", {"answer": "206.29"})

    assert sink == []


def test_structured_output_is_rendered_as_json(sink, reported):
    plugin = agent_output.AgentOutputPlugin()

    run_tool(plugin, "HypothesesAgent", {"hypotheses": ["H1", "H2"]})

    (_, payload), = sink
    assert '"hypotheses"' in payload["content"]
    assert "H2" in payload["content"]


def test_empty_answer_produces_no_message(sink, reported):
    plugin = agent_output.AgentOutputPlugin()

    run_tool(plugin, "HypothesesAgent", "   ")
    run_tool(plugin, "HypothesesAgent", None)

    assert sink == []


def test_runaway_output_is_truncated(sink, reported):
    plugin = agent_output.AgentOutputPlugin()

    run_tool(plugin, "HypothesesAgent", "x" * 200_000)

    (_, payload), = sink
    assert len(payload["content"]) < agent_output._OUTPUT_LIMIT + 100
    assert payload["content"].endswith("(output truncated)")


def test_a_failing_sink_cannot_break_the_run(monkeypatch, reported):
    async def broken(key, payload):
        raise RuntimeError("sink down")

    monkeypatch.setattr(agent_output, "_sink", broken)

    assert run_tool(agent_output.AgentOutputPlugin(), "HypothesesAgent", "H1") is None


def test_plugin_is_inert_without_a_sink(monkeypatch, reported):
    monkeypatch.setattr(agent_output, "_sink", None)

    assert run_tool(agent_output.AgentOutputPlugin(), "HypothesesAgent", "H1") is None


def test_every_reported_agent_is_reachable_as_an_agent_tool():
    """The plugin matches on the TOOL name, so an AgentTool must carry it.

    ``AgentTool`` is named after the agent it wraps; a flagged agent that no
    parent delegates to would silently never report anything.
    """
    from google.adk.tools.agent_tool import AgentTool
    from CoScientist.assembly import build_system

    system = build_system()
    delegated = {
        tool.name
        for agent in system.agents.values()
        for tool in (getattr(agent, "tools", None) or [])
        if isinstance(tool, AgentTool)
    }

    assert agent_output.reported_agents() <= delegated


def test_configured_agents_come_from_the_system_config():
    """The default profile reports the deliverables, not pipeline internals."""
    names = agent_output.reported_agents()

    assert "HypothesesAgent" in names
    assert "ResearchAgent" in names
    assert "ToolReranker" not in names
    # A disabled agent (PlannerAgent by default) cannot answer, so it is out.
    from CoScientist.assembly.schema import get_config
    assert all(get_config().agent(n).is_enabled() for n in names)
