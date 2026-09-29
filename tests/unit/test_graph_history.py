"""What an agent is told about the run so far.

``history`` is what ``root_summary`` renders and what ``inject_graph_root``
puts in the orchestrator prompt, so a node kind missing from it is a run the
agents cannot see. #358 renamed the per-run activation node from ``agent_call``
to ``agent`` and the list kept the old name, which left every locally-run agent
out of the very summary meant to describe them.
"""
from __future__ import annotations

import pytest

from CoScientist.graph.memory import KnowledgeGraph


@pytest.fixture
def graph(tmp_path):
    return KnowledgeGraph(run_id="execution", snapshot_dir=str(tmp_path))


def _run(graph) -> None:
    graph.add_node(id="goal:inv-1", kind="goal", label="соберите датасет",
                   status="success", t_start=100.0)
    graph.add_node(id="agent:CoderAgent@inv-1", kind="agent", label="CoderAgent",
                   executor_agent="CoderAgent", status="success",
                   output="собрано 812 строк", t_start=101.0)
    graph.add_node(id="tool:1", kind="tool_call", label="execute_bash",
                   executor_agent="CoderAgent", status="success", t_start=102.0)
    graph.add_node(id="plan:P1@r1", kind="decision", label="plan rev 1",
                   executor_agent="ExperimentPlannerAgent", status="success",
                   verdict="approved", t_start=103.0)


def test_history_carries_the_agents_that_ran(graph):
    _run(graph)

    kinds = {event["kind"] for event in graph.history()}
    assert "agent" in kinds, "the activation nodes the plugin writes are invisible"
    assert {"goal", "tool_call", "decision"} <= kinds

    activation = next(e for e in graph.history() if e["kind"] == "agent")
    assert activation["agent"] == "CoderAgent"
    assert activation["output"] == "собрано 812 строк"


def test_the_roster_stays_out_of_the_history(graph):
    """The system hub is configuration, not something that happened."""
    _run(graph)

    assert "system" not in {event["kind"] for event in graph.history()}


def test_the_root_summary_names_what_actually_ran(graph):
    """This is the text inject_graph_root puts in front of the orchestrator."""
    _run(graph)

    summary = graph.root_summary()

    assert "Recent activity in this session:" in summary
    assert "CoderAgent" in summary
    assert "собрано 812 строк" in summary
