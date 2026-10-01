"""Settings → Agents: the operator's per-agent changes over system.yaml.

``settings.agents`` holds them (web UI, or ``AGENTS__OVERRIDES`` /
``AGENTS__DEFAULT_REASONING`` in .env) and they are read where the YAML's own
values are: ``AgentConfig.is_enabled`` / ``resolved_reasoning`` and the
assembler's ``_resolve_model``, i.e. when a session's agent tree is built.
"""
from __future__ import annotations

import json

import pytest

from CoScientist.assembly import assembler
from CoScientist.assembly.schema import load_config
from CoScientist.config import get_settings
from CoScientist.config.settings import AgentOverride, Settings
from CoScientist.web.agent_settings import (
    agents_catalog,
    apply_agent_settings,
    current_agent_settings,
)


@pytest.fixture(autouse=True)
def _clean_overrides():
    block = get_settings().agents
    saved = (block.default_reasoning, dict(block.overrides))
    block.default_reasoning, block.overrides = None, {}
    yield block
    block.default_reasoning, block.overrides = saved


def _override(name, **fields):
    get_settings().agents.overrides[name] = AgentOverride(**fields)


def test_an_override_switches_an_ordinary_agent_off_and_back_on():
    cfg = load_config()
    research = cfg.agent("ResearchAgent")
    assert research.is_enabled()
    _override("ResearchAgent", enabled=False)
    assert not research.is_enabled()
    # The YAML's own answer is still there for the catalog to show.
    assert research.declared_enabled()
    assert "ResearchAgent" not in [a.name for a in cfg.enabled_subordinates("OrchestratorAgent")]


@pytest.mark.parametrize("name", [
    "OrchestratorAgent", "ToolRetrieverAgent", "PlannerAgent", "MedicalAgent", "ContextInitAgent",
])
def test_root_internal_start_mode_and_setting_agents_ignore_an_enabled_override(name):
    """The root is the run's only entry point, internal agents are a
    composite's stages, the start mode attaches or removes the planner
    pipeline itself, and an `enabled: ${setting}` agent is switched by that
    setting, which the runtime reads too: an override must not fight any of them."""
    agent = load_config().agent(name)
    declared = agent.declared_enabled()
    _override(name, enabled=not declared)
    assert agent.is_enabled() is declared


def test_a_reasoning_override_wins_over_the_yaml_and_a_bad_one_is_ignored():
    agent = load_config().agent("ResearchAgent")
    declared = agent.resolved_reasoning()
    _override("ResearchAgent", reasoning="HIGH")
    assert agent.resolved_reasoning() == "high"
    _override("ResearchAgent", reasoning="extreme")
    assert agent.resolved_reasoning() == declared


def test_the_assembler_builds_with_the_overridden_model_and_default_reasoning(monkeypatch):
    calls = []
    from CoScientist.agents import common

    monkeypatch.setattr(common, "make_llm", lambda *a, **kw: calls.append(("main", a, kw)))
    monkeypatch.setattr(common, "make_coder_llm", lambda *a, **kw: calls.append(("coder", a, kw)))

    system = load_config()
    # TaskExecutorAgent declares no reasoning of its own: it inherits the default.
    executor = system.agent("TaskExecutorAgent")
    assert executor.reasoning is None
    get_settings().agents.default_reasoning = "medium"
    _override("TaskExecutorAgent", model="coder")
    assembler._resolve_model(executor, system)
    kind, _, kwargs = calls[-1]
    assert kind == "coder"
    assert kwargs["reasoning"] == "medium"


def test_apply_keeps_only_valid_differences():
    apply_agent_settings({
        "defaultReasoning": "nonsense",
        "overrides": {
            "MedicalAgent": {"enabled": False, "reasoning": "low"},
            "ResearchAgent": {"enabled": False, "reasoning": "Low", "model": "  coder "},
            "HypothesesAgent": {"reasoning": "extreme"},
            "Empty": {},
        },
    })
    assert current_agent_settings() == {
        "defaultReasoning": "",
        "overrides": {
            # MedicalAgent is switched by MEDICAL__ENABLED: the dead `enabled` goes.
            "MedicalAgent": {"reasoning": "low"},
            "ResearchAgent": {"enabled": False, "reasoning": "low", "model": "coder"},
        },
    }


def test_the_env_carries_the_whole_map_as_json(monkeypatch):
    monkeypatch.setenv("AGENTS__OVERRIDES", json.dumps({"MedicalAgent": {"enabled": False}}))
    monkeypatch.setenv("AGENTS__DEFAULT_REASONING", "low")
    agents = Settings().agents
    assert agents.default_reasoning == "low"
    # Agent names keep their case: they are keys, not field names.
    assert agents.overrides["MedicalAgent"].enabled is False


def test_the_catalog_shows_declared_values_not_overridden_ones():
    _override("ResearchAgent", enabled=False, reasoning="high")
    entry = next(a for a in agents_catalog()["agents"] if a["name"] == "ResearchAgent")
    assert entry["enabled"] is True
    assert entry["reasoning"] == "medium"
    assert entry["hasModel"] and entry["lock"] is None
    root = next(a for a in agents_catalog()["agents"] if a["root"])
    assert root["lock"] == "root"


@pytest.mark.parametrize(
    ("mode", "planner_enabled", "pipeline_enabled"),
    [
        ("planner", True, True),
        ("orchestrator", True, False),
        ("orchestrator_planner", False, False),
    ],
)
def test_catalog_shows_start_mode_effective_planner_state(
    monkeypatch, mode, planner_enabled, pipeline_enabled,
):
    monkeypatch.setenv("COSCIENTIST_CONFIG", "experiments")
    monkeypatch.setattr(get_settings().web, "start_mode", mode)

    by_name = {a["name"]: a for a in agents_catalog()["agents"]}

    assert by_name["PlannerAgent"]["enabled"] is planner_enabled
    assert by_name["PlannerAgent"]["lock"] == "startMode"
    assert by_name["PlanningPipelineAgent"]["enabled"] is pipeline_enabled
    assert by_name["PlanningPipelineAgent"]["lock"] == "startMode"


def test_a_setting_backed_agent_names_the_field_its_switch_edits():
    """Settings → Agents is the one place agents are switched: for an agent
    whose `enabled` is a setting, the row edits that setting, not an override."""
    by_name = {a["name"]: a for a in agents_catalog()["agents"]}
    assert by_name["MedicalAgent"]["enabledSetting"] == "medicalAgent.enabled"
    assert by_name["MedicalAgent"]["lock"] is None
    assert by_name["ContextInitAgent"]["enabledSetting"] == "general.contextInitEnabled"
    assert by_name["FedotAgent"]["enabledSetting"] == "taskExecutorAgent.fedotFallback"
    assert by_name["McpBuilderAgent"]["enabledSetting"] == "experimentModule.routeAlembic"
    assert by_name["NirReportAgent"]["enabledSetting"] == "nirReport.enabled"
    assert by_name["ResearchAgent"]["enabledSetting"] is None


def test_a_model_alias_changes_only_the_llm_not_the_agent_contract(monkeypatch):
    from CoScientist.agents import common

    cfg = load_config().agent("ResearchAgent")
    contract = (cfg.prompt, tuple(cfg.tools), cfg.callbacks.model_dump())
    calls = []
    monkeypatch.setattr(common, "make_llm", lambda *a, **kw: calls.append((a, kw)))

    apply_agent_settings({"overrides": {"ResearchAgent": {"model": "nir"}}})
    assembler._resolve_model(cfg, load_config())

    assert calls[-1][0][0] == get_settings().llm.nir_model
    assert (cfg.prompt, tuple(cfg.tools), cfg.callbacks.model_dump()) == contract


# ── Tool budgets: a limiter callback's number, per agent ─────────────────────

def _limiter_of(callback, agent_name):
    """The limiter object a before_tool callback builds for one agent."""
    from types import SimpleNamespace

    from CoScientist.assembly.registry import REGISTRY

    ctx = SimpleNamespace(config=SimpleNamespace(name=agent_name))
    return REGISTRY.callback(callback).resolve(ctx).__self__


def test_the_catalog_offers_a_budget_for_agents_with_a_limiter():
    by_name = {a["name"]: a for a in agents_catalog()["agents"]}
    limit = by_name["ResearchAgent"]["toolLimit"]
    assert limit["kind"] == "searches"
    # Without an override the search cap is the common one from Settings → Tools.
    assert limit["setting"] == "researchAgent.maxSearches"
    assert limit["default"] == get_settings().web.max_searches
    assert by_name["OrchestratorAgent"]["toolLimit"] is None


def test_a_limit_override_reaches_the_search_limiter_of_that_agent_only():
    _override("ResearchAgent", limit=7)
    assert _limiter_of("WebSearchLimiter", "ResearchAgent").max_searches == 7
    assert _limiter_of("WebSearchLimiter", "OtherAgent").max_searches == get_settings().web.max_searches


def test_the_microfluidics_per_tool_budget_follows_the_override():
    import CoScientist.microfluidics.bindings  # noqa: F401 — registers the limiters

    assert _limiter_of("PerToolCallLimiter", "ResearchAgent").max_calls == 2
    _override("ResearchAgent", limit=4)
    limiter = _limiter_of("PerToolCallLimiter", "ResearchAgent")
    assert limiter.max_calls == 4
    # PaperRetriever's third explore_my_papers pass is a floor, not a cap.
    assert limiter.per_tool["explore_my_papers"] == 4


def test_apply_clamps_a_limit_and_drops_it_where_there_is_no_limiter():
    apply_agent_settings({"overrides": {
        "ResearchAgent": {"limit": 500},
        "OrchestratorAgent": {"limit": 3},
    }})
    assert current_agent_settings()["overrides"] == {"ResearchAgent": {"limit": 50}}
