from pathlib import Path

from CoScientist.assembly.schema import load_config


def test_synapse_demo_excludes_medical_and_mcp_builder():
    config = load_config(Path("CoScientist/agents/synapse_demo.yaml"))
    assert {agent.a2a.key for agent in config.a2a_agents()} == {
        "orchestrator",
        "planner",
        "hypotheses",
        "research",
        "task_execution",
        "coder",
    }
    subordinates = {
        agent.name for agent in config.enabled_subordinates("OrchestratorAgent")
    }
    assert "MedicalAgent" not in subordinates
    assert "McpBuilderAgent" not in subordinates
    assert config.agent("MedicalAgent").a2a is None
    assert config.agent("McpBuilderAgent").a2a is None
    assert config.agent("PlanningPipelineAgent").a2a is None
