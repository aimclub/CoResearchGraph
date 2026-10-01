import pytest

from CoScientist.assembly.schema import SystemConfig
from CoScientist.config import get_settings, settings_scope
from CoScientist.config.settings import AgentOverride
from CoScientist.web.agent_configuration_policy import (
    agent_control_policy,
    validate_agent_configuration_transition,
    validate_raw_agent_configuration_request,
)


def _policy(monkeypatch, profile: str, mode: str = "orchestrator"):
    monkeypatch.setenv("COSCIENTIST_CONFIG", profile)
    snapshot = get_settings().model_copy(deep=True)
    snapshot.web.start_mode = mode
    with settings_scope(snapshot):
        return agent_control_policy()


@pytest.mark.parametrize("mode", ["planner", "orchestrator", "orchestrator_planner"])
def test_system_profile_protects_main_pipeline(monkeypatch, mode):
    policy = _policy(monkeypatch, "system", mode)

    assert policy["TaskExecutorAgent"]["availableInProfile"] is True
    assert policy["TaskExecutorAgent"]["requiredForPipeline"] is True
    assert policy["TaskExecutorAgent"]["canDisable"] is False
    for name in ("ContextInitAgent", "TZSpecAgent", "ResultAggregatorAgent"):
        assert policy[name]["requiredForPipeline"] is True
        assert policy[name]["canDisable"] is False


@pytest.mark.parametrize("profile", ["experiments", "hypotheses"])
@pytest.mark.parametrize("mode", ["planner", "orchestrator", "orchestrator_planner"])
def test_experiment_profiles_exclude_orphan_executor_and_protect_sequence(
    monkeypatch, profile, mode,
):
    policy = _policy(monkeypatch, profile, mode)

    assert policy["TaskExecutorAgent"]["availableInProfile"] is False
    assert policy["TaskExecutorAgent"]["canEnable"] is False
    assert policy["ExperimentModuleAgent"]["requiredForPipeline"] is True
    assert policy["ExperimentModuleAgent"]["canDisable"] is False
    for name in (
        "ToolPreparerAgent",
        "ExperimentPlannerAgent",
        "ExperimentExecutorAgent",
        "ExperimentResultReviewAgent",
    ):
        assert policy[name]["requiredForPipeline"] is True
        assert policy[name]["canDisable"] is False


def test_microfluidics_profile_protects_root_sequence_and_report(monkeypatch):
    policy = _policy(monkeypatch, "microfluidics")

    assert policy["RootOrchestrator"]["requiredForPipeline"] is True
    assert policy["ReportAgent"]["requiredForPipeline"] is True
    assert policy["ReportAgent"]["canDisable"] is False
    assert policy["TZAgent"]["requiredForPipeline"] is True


def test_disabled_optional_agent_remains_available_for_reconnect(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_CONFIG", "experiments")
    snapshot = get_settings().model_copy(deep=True)
    snapshot.web.start_mode = "orchestrator"
    snapshot.agents.overrides["ResearchAgent"] = AgentOverride(enabled=False)
    with settings_scope(snapshot):
        control = agent_control_policy()["ResearchAgent"]

    assert control["availableInProfile"] is True
    assert control["effectiveEnabled"] is False
    assert control["canEnable"] is True


def test_child_reports_disabled_parent_without_enabling_it():
    config = SystemConfig.model_validate({
        "agents": {
            "Root": {"root": True, "subordinates": ["Branch"]},
            "Branch": {"class": "sequential", "children": ["Child"]},
            "Child": {},
        },
    })
    snapshot = get_settings().model_copy(deep=True)
    snapshot.agents.overrides["Branch"] = AgentOverride(enabled=False)
    with settings_scope(snapshot):
        policy = agent_control_policy(config)

    child = policy["Child"]
    assert child["availableInProfile"] is True
    assert child["canEnable"] is False
    assert child["controlCode"] == "parentDisabled"
    assert "родительскую ветку" in child["controlReason"]
    assert policy["Branch"]["effectiveEnabled"] is False


def test_recursive_subordinate_link_does_not_recurse_forever():
    config = SystemConfig.model_validate({
        "agents": {
            "Root": {"root": True, "subordinates": ["Branch"]},
            "Branch": {},
        },
    })
    # SystemConfig rejects cycles in YAML.  The projection still has to be
    # defensive because a custom mode transform can introduce a repeated link.
    config.agents["Branch"].subordinates.append("Root")

    policy = agent_control_policy(config)

    assert policy["Root"]["availableInProfile"] is True
    assert policy["Branch"]["availableInProfile"] is True


def test_raw_mode_controlled_flag_does_not_block_matching_mode_change(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_CONFIG", "system")
    before = get_settings().model_copy(deep=True)
    before.web.start_mode = "orchestrator"

    validate_raw_agent_configuration_request(before, {
        "general": {"startMode": "orchestrator_planner"},
        "agents": {"overrides": {"PlannerAgent": {"enabled": False}}},
    })


def test_legacy_disabled_required_agent_can_only_be_restored(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_CONFIG", "system")
    before = get_settings().model_copy(deep=True)
    before.web.start_mode = "orchestrator"
    before.agents.overrides["TaskExecutorAgent"] = AgentOverride(enabled=False)
    restored = before.model_copy(deep=True)
    restored.agents.overrides["TaskExecutorAgent"] = AgentOverride(enabled=True)

    validate_agent_configuration_transition(before, restored)
    with pytest.raises(ValueError, match="Обязательный участник"):
        validate_agent_configuration_transition(restored, before)
