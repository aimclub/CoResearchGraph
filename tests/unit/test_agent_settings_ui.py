"""Structural regressions for the dependency-free settings modal."""

from pathlib import Path


SETTINGS_JS = Path("CoScientist/web/static/js/modals/settings.js")


def test_agent_model_control_is_a_repeatable_select_with_custom_escape_hatch():
    source = SETTINGS_JS.read_text(encoding="utf-8")
    assert 'data-agent-model-choice="${escHtml(agent.name)}"' in source
    assert "agentsWithCustomModel.add(name)" in source
    assert "agentsWithCustomModel.delete(name)" in source
    assert "<datalist id=\"settings-agent-models\">" not in source
    assert "settings.agents.modelScope" in source


def test_start_mode_locked_agents_follow_the_current_settings_draft():
    source = SETTINGS_JS.read_text(encoding="utf-8")
    assert "agent.lock === 'startMode'" in source
    assert "agent.name === 'PlannerAgent'" in source
    assert "mode !== 'orchestrator_planner'" in source
    assert "agent.name === 'PlanningPipelineAgent'" in source
