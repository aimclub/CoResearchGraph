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


def test_every_reported_agent_is_reachable_in_the_built_system():
    """A flagged name must belong to an agent this build actually runs.

    Delegated ones report through their ``AgentTool`` result (the plugin
    matches on the TOOL name, and an ``AgentTool`` is named after the agent it
    wraps); the rest report when the agent itself ends, which needs them to be
    in the system at all — a typo would silently never report anything.
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

    pipeline = agent_output.reported_agents() - delegated
    assert pipeline == agent_output.pipeline_reported_agents()
    assert pipeline <= set(system.agents)


def test_configured_agents_come_from_the_system_config():
    """The default profile reports the deliverables, not pipeline internals."""
    names = agent_output.reported_agents()

    assert "HypothesesAgent" in names
    assert "ResearchAgent" in names
    assert "ToolReranker" not in names
    # A disabled agent (PlannerAgent by default) cannot answer, so it is out.
    from CoScientist.assembly.schema import get_config
    assert all(get_config().agent(n).is_enabled() for n in names)


# ── The pipeline path: a flagged step of a sequential module ────────────────
@pytest.fixture
def pipeline(monkeypatch):
    """Pin the pipeline-reported set and its parent lookup."""
    monkeypatch.setattr(
        agent_output, "pipeline_reported_agents", lambda: frozenset({"EconomicsAgent"})
    )
    monkeypatch.setattr(agent_output, "_parent_of", lambda name: "ModuleB_Design")


def step_context(agent_name: str = "EconomicsAgent", state: dict | None = None):
    """The context of the agent ITSELF — a pipeline step has no caller frame."""
    return SimpleNamespace(
        agent_name=agent_name,
        invocation_id="inv_1",
        state={
            GRAPH_SCOPE_USER_KEY: "user_1",
            GRAPH_SCOPE_SESSION_KEY: "session_1",
            **(state or {}),
        },
        session=SimpleNamespace(user_id="user_1", id="session_1"),
    )


def model_response(*parts, partial: bool = False):
    return SimpleNamespace(
        partial=partial,
        content=SimpleNamespace(parts=[SimpleNamespace(**p) for p in parts]),
    )


def text_part(text: str, thought: bool = False):
    return {"text": text, "thought": thought, "function_call": None}


def call_part(name: str):
    return {"text": None, "thought": False, "function_call": SimpleNamespace(name=name)}


def run_step(plugin, *responses, context=None, agent_name="EconomicsAgent"):
    """Feed model responses to the plugin, then end the agent."""
    ctx = context or step_context(agent_name)

    async def drive():
        for response in responses:
            await plugin.after_model_callback(callback_context=ctx, llm_response=response)
        return await plugin.after_agent_callback(
            agent=SimpleNamespace(name=agent_name), callback_context=ctx
        )

    return asyncio.run(drive())


def test_pipeline_step_reports_its_last_message(sink, pipeline):
    plugin = agent_output.AgentOutputPlugin()

    assert run_step(
        plugin,
        model_response(text_part("Данные собраны. Собираю итоговый отчёт.")),
        model_response(call_part("economics_mcp")),
        model_response(text_part("# Экономика LIT-ROUTE-01\n\n1200 ₽/кг")),
    ) is None

    (key, payload), = sink
    assert key == ("user_1", "session_1")
    assert payload["agent"] == "EconomicsAgent"
    assert payload["caller"] == "ModuleB_Design"
    assert payload["content"].startswith("# Экономика LIT-ROUTE-01")
    assert payload["timestamp"]


def test_thoughts_and_tool_turns_are_not_the_answer(sink, pipeline):
    plugin = agent_output.AgentOutputPlugin()

    run_step(
        plugin,
        model_response(text_part("Отчёт по маршруту")),
        model_response(text_part("prices look odd", thought=True)),
        # Text alongside a tool call is an aside, not the final answer.
        model_response(text_part("сверю цену"), call_part("websearch")),
        # A streamed chunk is only part of a message.
        model_response(text_part("…хвост"), partial=True),
    )

    (_, payload), = sink
    assert payload["content"] == "Отчёт по маршруту"


def test_unflagged_step_stays_out_of_the_chat(sink, pipeline):
    plugin = agent_output.AgentOutputPlugin()

    run_step(
        plugin,
        model_response(text_part("candidates")),
        agent_name="MolDesignAgent",
        context=step_context("MolDesignAgent"),
    )

    assert sink == []


def test_silent_step_falls_back_to_its_output_key(sink, pipeline, monkeypatch):
    monkeypatch.setattr(agent_output, "_output_key_text", lambda a, s: s.get("economics", ""))
    plugin = agent_output.AgentOutputPlugin()

    run_step(plugin, context=step_context(state={"economics": "стоимость: 1200 ₽/кг"}))

    (_, payload), = sink
    assert payload["content"] == "стоимость: 1200 ₽/кг"


def test_nothing_answered_produces_no_message(sink, pipeline, monkeypatch):
    monkeypatch.setattr(agent_output, "_output_key_text", lambda a, s: "")
    plugin = agent_output.AgentOutputPlugin()

    run_step(plugin, model_response(text_part("   ")))

    assert sink == []


def test_a_reported_step_returns_no_content_to_adk(sink, pipeline):
    """Content returned here would be saved through the agent's output_schema."""
    plugin = agent_output.AgentOutputPlugin()

    assert run_step(plugin, model_response(text_part("отчёт"))) is None
    assert plugin._pipeline_text == {}  # nothing leaks into the next run
