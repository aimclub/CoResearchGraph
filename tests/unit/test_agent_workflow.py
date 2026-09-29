"""Ordering, identity and localization contracts for the configuration diagram."""
import pytest

from CoScientist.assembly.registry import REGISTRY
from CoScientist.assembly.schema import SystemConfig
from CoScientist.config import get_settings, settings_scope
from CoScientist.tools.mcp_catalog import load_presentation, resolve_tool_presentation
from CoScientist.web.agent_tree import project_agent_tree
from CoScientist.web.agent_workflow import project_workflow


def blocks(block):
    yield block
    if block.get("body"):
        yield from blocks(block["body"])
    for child in block.get("children", []):
        yield from blocks(child)


def projection(monkeypatch, profile="hypotheses", mode="planner"):
    monkeypatch.setenv("COSCIENTIST_CONFIG", profile)
    snapshot = get_settings().model_copy(deep=True)
    snapshot.web.start_mode = mode
    snapshot.context_init.enabled = True
    snapshot.nir.enabled = True
    snapshot.mcp.normcontrol_url = "http://normcontrol.invalid/mcp"
    return snapshot, project_agent_tree(snapshot, desired_revision=2, active_revision=1, running=True)


@pytest.mark.parametrize("profile", ["system", "experiments", "hypotheses", "microfluidics"])
@pytest.mark.parametrize("mode", ["planner", "orchestrator", "orchestrator_planner"])
def test_all_visible_agents_have_instances_and_russian_copy(monkeypatch, profile, mode):
    snapshot, data = projection(monkeypatch, profile, mode)
    expected = {n["id"] for n in data["nodes"] if n["kind"] == "agent" and n["selected"]}
    instances = data["workflow"]["instances"]
    assert {i["agentId"] for i in instances} == expected
    assert len({i["id"] for i in instances}) == len(instances)
    assert "PlanningPipelineAgent" not in expected
    assert "ToolPreparerAgent" not in expected
    for node in data["nodes"]:
        assert node["title"]["ru"] != "Агент исследования"
        assert "пока не добавлено" not in node["descriptionLocalized"]["ru"]
        assert any("а" <= c <= "я" for c in node["descriptionLocalized"]["ru"].lower())
    again = project_agent_tree(snapshot, desired_revision=2, active_revision=1, running=True)
    assert data == again


def test_experiment_order_and_report_call_boundaries(monkeypatch):
    _, data = projection(monkeypatch)
    all_blocks = list(blocks(data["workflow"]["root"]))
    module = next(b for b in all_blocks if b.get("agentId") == "ExperimentModuleAgent")
    assert module["body"]["type"] == "sequence"
    assert [b["agentId"] for b in module["body"]["children"]] == [
        "ExperimentPlannerAgent", "ExperimentExecutorAgent", "ExperimentResultReviewAgent",
    ]
    executor = module["body"]["children"][1]
    assert executor["body"]["type"] == "delegates"
    assert executor["body"]["relation"] == "delegate"
    report = data["workflow"]["root"]["children"][-1]
    assert report["agentId"] == "ResultAggregatorAgent"
    assert report["body"]["children"][0]["agentId"] == "NirReportAgent"
    shared = [i for i in data["workflow"]["instances"] if i["agentId"] == "ResearchAgent"]
    assert len(shared) == 2 and all(i["reused"] for i in shared)
    owner = data["workflow"]["configuration"]["ownerInstanceId"]
    assert next(i for i in data["workflow"]["instances"] if i["id"] == owner)["agentId"] == "OrchestratorAgent"


@pytest.mark.parametrize("mode,initial", [("planner", True), ("orchestrator", False)])
def test_research_planner_location_tracks_start_mode(monkeypatch, mode, initial):
    _, data = projection(monkeypatch, mode=mode)
    planner = next(i for i in data["workflow"]["instances"] if i["agentId"] == "PlannerAgent")
    assert ("PlanningPipelineAgent/child/PlannerAgent" in planner["id"]) == initial
    if not initial:
        assert "OrchestratorAgent/call/PlannerAgent" in planner["id"]


def test_disabling_optional_agents_removes_every_call_site(monkeypatch):
    snapshot, _ = projection(monkeypatch)
    snapshot.nir.enabled = False
    snapshot.web.medical_agent_enabled = False
    data = project_agent_tree(snapshot, desired_revision=3, active_revision=2, running=True)
    names = {i["agentId"] for i in data["workflow"]["instances"]}
    assert "NirReportAgent" not in names
    assert "MedicalAgent" not in names
    assert data["pending"] and data["running"]


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_execution_routes_follow_session_settings(monkeypatch, enabled):
    snapshot, _ = projection(monkeypatch)
    snapshot.experiments.route_fedot = enabled
    snapshot.experiments.route_alembic = enabled
    data = project_agent_tree(snapshot, desired_revision=3, active_revision=2, running=True)
    names = {i["agentId"] for i in data["workflow"]["instances"]}
    assert ("FedotAgent" in names) is enabled
    assert ("McpBuilderAgent" in names) is enabled


def test_internal_sequences_parallel_and_custom_choices_survive_projection():
    config = SystemConfig.model_validate({"agents": {
        "Hidden": {"class": "sequential", "root": True, "internal": True, "children": ["Plan", "Fork", "Switch", "Review"]},
        "Plan": {}, "Review": {}, "A": {}, "B": {}, "C": {}, "D": {},
        "Fork": {"class": "parallel", "internal": True, "children": ["A", "B"]},
        "Switch": {"class": "custom:executor_switch", "internal": True, "children": ["C", "D"]},
    }})
    workflow = project_workflow(config, nir_enabled=False)
    main = workflow["root"]["children"][0]
    assert main["type"] == "sequence"
    assert [b["type"] for b in main["children"]] == ["agent", "parallel", "choice", "agent"]
    assert [b["agentId"] for b in main["children"][1]["children"]] == ["A", "B"]
    assert [b["agentId"] for b in main["children"][2]["children"]] == ["C", "D"]
    assert {i["agentId"] for i in workflow["instances"]} == {"Plan", "Review", "A", "B", "C", "D"}


def test_disabled_composite_and_custom_children_match_assembly():
    config = SystemConfig.model_validate({"agents": {
        "Root": {"class": "sequential", "root": True, "children": ["Off", "Custom", "Leaf"]},
        "Off": {"class": "sequential", "enabled": False, "internal": True, "children": ["Unreachable"]},
        "Unreachable": {}, "Leaf": {"enabled": False},
        "Custom": {"class": "custom:executor_switch", "children": ["ChildOff", "ChildOn"]},
        "ChildOff": {"enabled": False}, "ChildOn": {},
    }})
    names = {i["agentId"] for i in project_workflow(config, nir_enabled=False)["instances"]}
    # Enabled composites include even a disabled declared leaf; custom children
    # are filtered, and a disabled composite executes no children.
    assert names == {"Root", "Leaf", "Custom", "ChildOn"}


@pytest.mark.parametrize("profile", ["system", "experiments", "hypotheses", "microfluidics"])
def test_visible_builtin_tools_have_real_russian_translations(monkeypatch, profile):
    from CoScientist.agents import config_for_mode

    snapshot, data = projection(monkeypatch, profile)
    presentation = load_presentation()
    with settings_scope(snapshot):
        config = config_for_mode()
        for node in data["nodes"]:
            if node["kind"] != "agent":
                continue
            for key in config.agent(node["name"]).tools:
                if key not in REGISTRY.tools:
                    continue
                for doc in REGISTRY.tools[key].resolved_docs():
                    if doc.name.startswith("<"):
                        continue
                    resolved = resolve_tool_presentation(doc.name, doc.purpose, registry_key=key, presentation=presentation)
                    assert resolved["display_name"]["ru"] != "Инструмент", (key, doc.name)
                    assert "пока не добавлено" not in resolved["summary"]["ru"], (key, doc.name)
