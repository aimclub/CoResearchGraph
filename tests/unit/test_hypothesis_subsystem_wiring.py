"""The hypothesis subsystem must not cost the study its research graph.

#326 attached the MooseChem generator by replacing HypothesesAgent with it:
`class: llm` became `class: custom:hypothesis_subsystem`, and the toolset
`[task_tracker, graph, research_graph]` became `[generate_via_moosechem,
run_critic_loop]`. HypothesesAgent is the only role permitted to create
Hypothesis, VerificationMethod and ConfirmationCriteria nodes, so the graph
kept the question and its constraints and never received a hypothesis, a way
to test one, or a bar to judge it by. #362 reverted the lot.

These tests pin the shape that lets the subsystem back in: it is a tool the
agent may call, not a replacement for the agent.
"""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_YAML = ROOT / "CoScientist" / "agents" / "system.yaml"


@pytest.fixture(scope="module")
def hypotheses_agent():
    doc = yaml.safe_load(SYSTEM_YAML.read_text(encoding="utf-8"))
    return doc["agents"]["HypothesesAgent"]


def test_the_agent_keeps_its_graph_tools(hypotheses_agent):
    """The three it had are what write the graph; losing them emptied it."""
    tools = hypotheses_agent.get("tools") or []
    for required in ("task_tracker", "graph", "research_graph"):
        assert required in tools, f"{required} is what #326 dropped"


def test_the_subsystem_is_attached_beside_them_not_instead_of_them(hypotheses_agent):
    tools = hypotheses_agent.get("tools") or []
    assert "hypothesis_subsystem" in tools
    # The agent itself stays an LLM agent driven by the hypotheses prompt — the
    # prompt is where the whole graph-writing contract lives.
    assert hypotheses_agent.get("class") == "llm"
    assert hypotheses_agent.get("prompt") == "hypotheses"


def test_the_agent_keeps_the_human_in_the_loop(hypotheses_agent):
    """#326 dropped `hitl: true` with everything else."""
    assert hypotheses_agent.get("hitl") is True


def test_the_link_callbacks_survive(hypotheses_agent):
    """Dropping these left tool results without resolvable links."""
    callbacks = hypotheses_agent.get("callbacks") or {}
    assert "user_links" in (callbacks.get("before_agent") or [])
    assert "register_tool_result_links" in (callbacks.get("after_tool") or [])
    assert "resolve_link_refs" in (callbacks.get("before_tool") or [])


def test_the_prompt_tells_the_agent_the_tool_does_not_write_the_graph():
    """Attaching the tool is half of it; the agent has to know whose job it is."""
    templates = (ROOT / "CoScientist" / "agents" / "prompts" / "templates.py").read_text(
        encoding="utf-8")
    start = templates.index('@_register("hypotheses")')
    end = templates.index('@_register(', start + 10)
    prompt = templates[start:end]

    assert "HypothesisGenerator" in prompt, "the tool is never mentioned"
    lowered = prompt.lower()
    assert "never touches the graph" in lowered or "does not touch" in lowered
    # And the contract it already carried is still there.
    for node_type in ("Hypothesis", "VerificationMethod", "ConfirmationCriteria"):
        assert node_type in prompt
    assert "research_commit" in prompt


def test_hypotheses_agent_is_still_the_only_writer_of_its_node_types():
    """The permission model is what makes the agent irreplaceable here."""
    schema = (ROOT / "CoScientist" / "graph" / "research" / "schema.py").read_text(
        encoding="utf-8")
    start = schema.index('"HypothesesAgent": AgentPerm(')
    block = schema[start:start + 900]
    for node_type in ("Hypothesis", "VerificationMethod", "ConfirmationCriteria"):
        assert node_type in block


def test_the_subsystem_package_is_present():
    """Restored whole; a missing module would surface only at assembly time."""
    package = ROOT / "CoScientist" / "hypothesis_subsystem"
    names = {p.name for p in package.glob("*.py")}
    for required in ("__init__.py", "generator_agent.py", "critic_agent.py",
                     "loop_coordinator.py", "moosechem_mcp_tool.py",
                     "tool_registry.py", "models.py", "prompts.py"):
        assert required in names, required


def test_the_subsystem_entry_is_optional_so_a_deployment_without_it_still_builds():
    """No MooseChem MCP server configured must leave today's behaviour alone."""
    bindings = (ROOT / "CoScientist" / "assembly" / "bindings.py").read_text(
        encoding="utf-8")
    start = bindings.index('key="hypothesis_subsystem"')
    entry = bindings[start:start + 500]
    assert "optional=True" in entry
    factory = bindings[bindings.index("def _hypothesis_subsystem("):]
    factory = factory[:factory.index("\ndef ", 10)]
    assert "moosechem_url" in factory and "return None" in factory
