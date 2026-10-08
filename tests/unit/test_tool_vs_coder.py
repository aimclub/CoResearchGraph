"""Did a run use a tool from the catalogue, or write the code from scratch?

The system is meant to look for an existing tool first. Nothing else records
whether it did, so a catalogue that has quietly stopped being used is
indistinguishable from one that works. The signal is read off the execution
graph; the agent sets come from the config, so renaming an agent cannot
silently break it.
"""

from CoScientist.assembly.schema import get_config
from CoScientist.graph.projection import _agents_bound_to, tool_vs_coder

_MCP = {"ExperimentAgent"}
_CODER = {"CoderAgent"}


def _graph(*nodes):
    return {"nodes": list(nodes)}


def _node(kind, who, label="x"):
    return {"id": label, "kind": kind, "executor_agent": who, "label": label}


def test_a_catalogue_tool_call_is_the_tool_path():
    graph = _graph(_node("tool_call", "ExperimentAgent", "segment_image"))

    out = tool_vs_coder(graph, mcp_agents=_MCP, coder_agents=_CODER)

    assert out["path"] == "mcp"
    assert out["mcp_tool_calls"] == ["segment_image"]


def test_delegating_to_the_executor_is_not_enough_on_its_own():
    """The lookup may have come up empty — only an actual call counts."""
    graph = _graph(_node("agent_call", "ExperimentAgent", "ExperimentAgent"))

    assert tool_vs_coder(graph, mcp_agents=_MCP, coder_agents=_CODER)["path"] == "none"


def test_delegating_to_the_coder_is_enough():
    """For the coder the delegation itself means code was written."""
    graph = _graph(_node("agent_call", "CoderAgent", "CoderAgent"))

    out = tool_vs_coder(graph, mcp_agents=_MCP, coder_agents=_CODER)

    assert out["path"] == "coder"
    assert out["coder_calls"] == ["CoderAgent"]


def test_both_paths_in_one_run_read_as_mixed():
    graph = _graph(
        _node("tool_call", "ExperimentAgent", "segment_image"),
        _node("tool_call", "CoderAgent", "execute_bash"),
    )

    assert tool_vs_coder(graph, mcp_agents=_MCP, coder_agents=_CODER)["path"] == "mixed"


def test_unrelated_activity_is_ignored():
    graph = _graph(
        _node("tool_call", "ResearchAgent", "tavily_search"),
        _node("goal", None, "do the thing"),
    )

    assert tool_vs_coder(graph, mcp_agents=_MCP, coder_agents=_CODER)["path"] == "none"


def test_an_empty_graph_is_not_an_error():
    assert tool_vs_coder({}, mcp_agents=_MCP, coder_agents=_CODER)["path"] == "none"


# ── the sets come from the config, not from names written in the code ────────


def test_the_tool_path_is_whoever_holds_the_registered_mcp_toolset():
    agents = _agents_bound_to(frozenset({"dynamic_tools"}))

    assert agents
    for name in agents:
        assert "dynamic_tools" in get_config().agent(name).tools


def test_the_coder_path_is_whoever_can_run_code():
    agents = _agents_bound_to(frozenset({"coder", "sandbox"}))

    assert agents
    for name in agents:
        assert {"coder", "sandbox"} & set(get_config().agent(name).tools)


def test_the_default_sets_are_read_from_the_live_config():
    """No arguments: the projection still knows both paths."""
    graph = _graph(_node("tool_call", "ExperimentAgent", "segment_image"))

    assert tool_vs_coder(graph)["path"] == "mcp"


def test_the_knowledge_graph_exposes_the_signal(tmp_path):
    """Callers hold a KnowledgeGraph, not a raw dict."""
    from CoScientist.graph.memory import KnowledgeGraph

    graph = KnowledgeGraph(run_id="run-x", snapshot_dir=str(tmp_path))

    assert graph.tool_vs_coder()["path"] == "none"
