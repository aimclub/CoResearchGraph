"""Structural tests for the YAML-driven agent assembly.

These build the real system from CoScientist/agents/system.yaml (no LLM calls)
and assert the invariants the assembler is supposed to guarantee — above all
that prompts and wiring cannot drift apart.

Run from the repo root:  pytest tests/unit/test_assembly.py -q
"""
import copy

import pytest
from dotenv import load_dotenv

load_dotenv()

from CoScientist.assembly import build_system, load_config  # noqa: E402
from CoScientist.assembly.prompting import PromptContext  # noqa: E402
from CoScientist.assembly.registry import REGISTRY  # noqa: E402
from CoScientist.assembly.schema import SystemConfig, get_config  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return get_config()


@pytest.fixture(scope="module")
def system(config):
    return build_system(config)


# ── config validation ────────────────────────────────────────────────────────

def test_config_loads_and_has_one_root(config):
    # The root is the delegation-tree orchestrator; lifecycle flow (plan/aggregate)
    # lives in the `pipeline` section, not in a SequentialAgent root.
    assert config.root.name == "OrchestratorAgent"
    order = config.build_order()
    assert order.index("ToolRetrieverAgent") < order.index("LocalToolsExtractorAgent")
    assert set(order) == set(config.agents)


def test_pipeline_stages_are_declared_agents_and_not_root(config):
    for stage in config.pipeline.stage_names():
        assert stage in config.agents, f"pipeline stage {stage!r} is not a declared agent"
        assert stage != config.root.name, "the root must not also be a pipeline stage"
    # The Result Aggregator is wired as a post stage (the report deliverable).
    assert "ResultAggregatorAgent" in config.pipeline.post


def test_run_root_is_one_sequential_run_ending_in_the_aggregator(monkeypatch):
    """The whole lifecycle is ONE ADK SequentialAgent (context-init pre-stage →
    orchestrator → aggregator) driven by a single run_async — so it is one
    invocation / one Opik trace with the Result Aggregator as the terminal stage
    (no separate static-directive run)."""
    from google.adk.agents.sequential_agent import SequentialAgent
    from CoScientist.assembly import build_system
    from CoScientist.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings.context_init, "enabled", True)
    system = build_system()
    run_root = system.run_root

    assert isinstance(run_root, SequentialAgent)
    names = [a.name for a in run_root.sub_agents]
    # The context-init pre-stage seeds the research frame before the orchestrator.
    assert names[0] == "ContextInitAgent", "context-init runs before the orchestrator"
    assert names.index("OrchestratorAgent") < names.index("ResultAggregatorAgent")
    assert names[-1] == "ResultAggregatorAgent", "aggregator must be the terminal stage"


def test_aggregator_is_graph_primary_and_read_only(config):
    """The aggregator reads the research graph (read-only surface, no commit) with
    no conversation history — it is grounded in the typed graph, not the transcript."""
    agg = config.agent("ResultAggregatorAgent")
    assert agg.include_contents == "none"
    assert "research_graph_readonly" in agg.tools
    assert "research_graph" not in agg.tools, "must use the read-only surface, not the worker one"
    assert "inject_research_context" in agg.callbacks.before_agent


def test_every_referenced_name_is_registered(config):
    for agent in config.agents.values():
        for tool in agent.tools:
            REGISTRY.tool(tool)  # raises on unknown
        for kind, names in agent.callbacks.items():
            for name in names:
                entry = REGISTRY.callback(name)
                assert entry.kind == kind, f"{agent.name}: {name} listed under {kind}"
        if agent.prompt:
            REGISTRY.prompt(agent.prompt)
        if agent.output_schema:
            REGISTRY.output_schema(agent.output_schema)
        if agent.planner:
            REGISTRY.planner(agent.planner)
        if agent.cls.startswith("custom:"):
            REGISTRY.agent_class(agent.cls.split(":", 1)[1])


def test_unknown_agent_reference_rejected(config):
    raw = copy.deepcopy(config.model_dump(by_alias=True))
    raw["agents"]["OrchestratorAgent"]["subordinates"].append("NoSuchAgent")
    with pytest.raises(Exception, match="NoSuchAgent"):
        SystemConfig.model_validate(raw)


def test_dependency_cycle_rejected(config):
    raw = copy.deepcopy(config.model_dump(by_alias=True))
    raw["agents"]["ToolRetrieverAgent"]["subordinates"] = ["TaskExecutorAgent"]
    with pytest.raises(Exception, match="cycle"):
        SystemConfig.model_validate(raw)


def test_duplicate_a2a_port_rejected(config):
    raw = copy.deepcopy(config.model_dump(by_alias=True))
    raw["agents"]["CoderAgent"]["a2a"]["port"] = raw["agents"]["ResearchAgent"]["a2a"]["port"]
    with pytest.raises(Exception, match="port"):
        SystemConfig.model_validate(raw)


# ── built system invariants ──────────────────────────────────────────────────

def test_all_agents_built_under_their_names(config, system):
    for name in config.agents:
        assert system.agent(name).name == name


def test_orchestrator_roster_matches_prompt_and_tools(config, system):
    orchestrator = system.agent("OrchestratorAgent")
    enabled = [a.name for a in config.enabled_subordinates("OrchestratorAgent")]
    attached = [t.agent.name for t in orchestrator.tools if hasattr(t, "agent")]
    assert attached == enabled
    for name in enabled:
        assert name in orchestrator.instruction, f"{name} wired but not in the prompt"
    for sub in config.agent("OrchestratorAgent").subordinates:
        if not config.agent(sub).is_enabled():
            assert sub not in orchestrator.instruction, f"{sub} disabled but still in the prompt"


def test_prompts_advertise_exactly_the_attached_function_tools(config, system):
    """Every attached function tool is named in the prompt and vice versa."""
    for name, cfg in config.agents.items():
        if cfg.cls != "llm" or not cfg.prompt:
            continue
        agent = system.agent(name)
        instruction = agent.instruction
        for tool in agent.tools:
            tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if tool_name is None or hasattr(tool, "get_tools") or hasattr(tool, "agent") or tool_name in ("request_approval", "request_selection"):
                continue  # toolsets resolve at runtime; AgentTools live in <<AGENTS>>; HITL tools are interactive
            assert tool_name in instruction, f"{name}: tool {tool_name} not in prompt"


def test_no_unfilled_placeholders(config, system):
    for name, cfg in config.agents.items():
        instruction = getattr(system.agent(name), "instruction", "") or ""
        assert "<<" not in instruction, f"{name}: unfilled placeholder in prompt"


def test_pipeline_state_injections_are_optional(config, system):
    """ADK {state_key} injections that depend on an upstream agent having called
    a tool must use the optional `{key?}` form, or a degenerate run (empty web
    search, no retrieval) crashes the agent with a KeyError mid-turn."""
    state_keys = (
        "accumulated_tools", "filtered_tools", "accumulated_web_mcps",
        "fedot_candidates", "reranker_candidates",
    )
    for name in config.agents:
        instruction = getattr(system.agent(name), "instruction", "") or ""
        for key in state_keys:
            assert "{" + key + "}" not in instruction, (
                f"{name}: bare ADK injection {{{key}}} — must be optional {{{key}?}}"
            )


def test_critic_prompt_embeds_current_roster(config):
    ctx = PromptContext(config=config.agent("OrchestratorAgent"), system=config)
    critic_prompt = REGISTRY.prompt("pre_action_critic")(ctx)
    for sub in config.enabled_subordinates("OrchestratorAgent"):
        assert sub.name in critic_prompt


def test_planner_roster_uses_real_agent_names(config, system):
    instruction = system.agent("PlannerAgent").instruction
    for sub in config.enabled_subordinates("OrchestratorAgent"):
        if sub.name != "PlannerAgent":
            assert sub.name in instruction
    # The old prose aliases must be gone.
    assert "Experiment Agent" not in instruction
    assert "Hypothesis Agent" not in instruction


def test_planner_optimizes_for_capability_coverage_not_step_count(monkeypatch, config):
    """MCP metadata compresses delegation units instead of expanding them.

    Built with the planner's retrieval tool pinned ON: the MCP guidance lives in
    the discovery block, which the switch removes by design, so reading the
    ambient system would assert whatever the developer's .env happens to say.
    """
    on = _build_with(monkeypatch, config, planner_retrieval_enabled=True)
    instruction = on.agent("PlannerAgent").instruction
    assert "SHORTEST executable roadmap" in instruction
    assert "one task per independent user deliverable" in instruction
    assert "run a compression pass" in instruction
    assert "Do not add an OrchestratorAgent task" in instruction
    assert "Prefer a ready direct generation/inference tool" in instruction
    assert "Never assume that TaskExecutorAgent can" in instruction
    assert "make ONE task" in instruction


def test_orchestrator_tool_discovery_gate(config, system):
    """Structural invariant: the retrieval tool is documented iff attached, and
    when attached the retrieve_tools gate is positioned BEFORE the routing roster
    so the orchestrator checks for ready-made tools before delegating. (The exact
    wording of the gate is content the prompt author may tune.)"""
    cfg = config.agent("OrchestratorAgent")
    instruction = system.agent("OrchestratorAgent").instruction
    has_retrieval = "retrieval" in cfg.tools
    assert ("retrieve_tools" in instruction) == has_retrieval
    if has_retrieval:
        # The gate must come before the routing roster (so it's read first).
        assert instruction.index("retrieve_tools") < instruction.index(
            "Delegate by the NATURE"
        )


def test_executor_sufficiency_and_discovery_guardrails(config, system):
    """Two routing guardrails are present when the relevant agents are wired:
    the ExperimentAgent must be able to abstain (NO_MATCHING_TOOL) when retrieved
    tools don't implement the task, and the orchestrator must do tool DISCOVERY
    itself rather than delegating "does a tool exist" to the Executor."""
    exp = system.agent("ExperimentAgent").instruction
    assert "NO_MATCHING_TOOL" in exp
    assert "Recommend CoderAgent" in exp

    cfg = config.agent("OrchestratorAgent")
    if "retrieval" in cfg.tools and "TaskExecutorAgent" in cfg.subordinates:
        orch = system.agent("OrchestratorAgent").instruction
        assert "retrieve_tools" in orch
        assert 'delegate "check if a tool exists"' in orch.lower() or \
               'Do NOT delegate "check if a tool exists"' in orch


def test_task_executor_is_a_router_over_both_execution_paths(config, system):
    """TaskExecutorAgent is an LLM ROUTER, not the tool pipeline itself: it picks
    between the ready-made MCP pipeline and the coder, both wired under it as
    AgentTools. The coder is therefore reached THROUGH it, not from the root."""
    executor = config.agent("TaskExecutorAgent")
    assert executor.cls == "llm"
    assert executor.subordinates == ["ToolPipelineAgent", "CoderAgent"]

    # The old sequential body moved to ToolPipelineAgent, reachable only via the
    # router (no A2A card of its own).
    pipeline = config.agent("ToolPipelineAgent")
    assert pipeline.cls == "sequential"
    # Prepare tools, then run ONE executor: the switch picks the ReAct executor,
    # or the FEDOT.MAS fallback when the reranker produced no usable ranking.
    assert pipeline.children == ["ToolPreparerAgent", "ExecutorSwitchAgent"]
    assert pipeline.a2a is None

    switch = config.agent("ExecutorSwitchAgent")
    assert switch.cls == "custom:executor_switch"
    assert switch.children == ["ExperimentAgent", "FedotAgent"]

    # Execution has ONE entry point on the orchestrator's roster.
    orch_subs = config.agent("OrchestratorAgent").subordinates
    assert "TaskExecutorAgent" in orch_subs
    assert "CoderAgent" not in orch_subs

    attached = [t.agent.name for t in system.agent("TaskExecutorAgent").tools
                if hasattr(t, "agent")]
    assert attached == ["ToolPipelineAgent", "CoderAgent"]


def test_router_prompt_absorbs_the_no_matching_tool_handoff(config, system):
    """The abstain verdict is resolved one level DOWN: the router re-issues the
    step to the coder itself, so the orchestrator is never asked to re-route it."""
    router = system.agent("TaskExecutorAgent").instruction
    assert "ToolPipelineAgent" in router and "CoderAgent" in router
    assert "NO_MATCHING_TOOL" in router

    orch = system.agent("OrchestratorAgent").instruction
    # The Executor-vs-Coder discriminator belongs to the router now.
    assert "re-route that step to" not in orch
    assert "Send ALL execution to TaskExecutorAgent" in orch


def test_dataset_collector_is_a_coder_subordinate_sharing_the_sandbox(monkeypatch, config):
    """The DatasetCollectorAgent is wired under CoderAgent, named in the coder's
    prompt, and uses the coder toolset — so it works in the same per-session
    sandbox workspace (the data it downloads lands where the coder builds on it).

    Built with the local coder tools pinned ON: without them the coder prompt is
    the thin sandbox relay, which deliberately names no subordinate.
    """
    coder = config.agent("CoderAgent")
    assert "DatasetCollectorAgent" in coder.subordinates
    collector = config.agent("DatasetCollectorAgent")
    # Both share the coder toolset -> same workspace-state anchor -> same sandbox.
    assert "coder" in collector.tools and "coder" in coder.tools
    on = _build_with(monkeypatch, config, coder_mode="local")
    # The coder prompt advertises the subordinate (rendered from config).
    coder_instruction = on.agent("CoderAgent").instruction
    assert "DatasetCollectorAgent" in coder_instruction
    assert "SAME sandbox" in coder_instruction
    # Reached only through the coder — not a standalone A2A service.
    assert collector.a2a is None
    # Built and attached as an AgentTool on the coder.
    attached = [t.agent.name for t in on.agent("CoderAgent").tools if hasattr(t, "agent")]
    assert attached == ["DatasetCollectorAgent"]


def test_research_graph_tools_match_prompt(monkeypatch, config):
    """Agents wired with a research_graph* tool document its write/read tools in
    their prompt (and agents without it don't) — the same consistency the
    assembler enforces for every tool, asserted explicitly for this feature.

    Built with the research graph pinned ON, because the roster below is read
    from `cfg.tools` (the YAML list), which still names the tool after the
    switch has dropped it. The switched-off half is its own test.
    """
    on = _build_with(monkeypatch, config, research_graph_enabled=True)

    for name, cfg in config.agents.items():
        if cfg.cls != "llm" or not cfg.prompt:
            continue
        instruction = on.agent(name).instruction
        has_worker = "research_graph" in cfg.tools
        has_orch = "research_graph_orchestrator" in cfg.tools
        if has_worker:
            assert "research_commit" in instruction, f"{name}: research_commit missing"
            assert "research_context_slice" in instruction, f"{name}: slice missing"
        elif not has_orch:
            assert "research_commit" not in instruction, f"{name}: research_commit leaked"
        # init/triggers/set_focus are orchestrator-only
        for orch_only in ("research_init", "research_triggers", "research_set_focus"):
            assert (orch_only in instruction) == has_orch, \
                f"{name}: {orch_only} presence != orchestrator-tool presence"


def test_orchestrator_prompt_documents_only_wired_critics(config, system):
    cfg = config.agent("OrchestratorAgent")
    instruction = system.agent("OrchestratorAgent").instruction
    assert ("Pre-action critic" in instruction) == (
        "pre_action_critique" in cfg.callbacks.after_model
    )
    assert ("Post-action critic" in instruction) == (
        "post_action_critique" in cfg.callbacks.after_tool
    )


def _planner_ctx(config, critic):
    """A PromptContext for the planner with the plan critic on or off."""
    cfg = config.agent("PlannerAgent").model_copy(update={"critic": critic})
    return PromptContext(config=cfg, system=config)


def test_plan_critic_prompt_lists_exactly_the_agents_a_plan_can_assign_to(config):
    """The plan critic checks assignees, so its roster must be the planner's
    SIBLINGS (what the orchestrator delegates to) — not the planner itself."""
    critic_prompt = REGISTRY.prompt("plan_critic")(_planner_ctx(config, True))
    for sub in config.enabled_subordinates("OrchestratorAgent"):
        if sub.name != "PlannerAgent":
            assert sub.name in critic_prompt
    assert "PlannerAgent" not in critic_prompt


def test_planner_prompt_documents_the_plan_review_only_when_the_critic_is_wired(config):
    planner_prompt = REGISTRY.prompt("planner")
    assert "PLAN REVIEW" in planner_prompt(_planner_ctx(config, True))
    assert "PLAN REVIEW" not in planner_prompt(_planner_ctx(config, False))


def test_plan_critic_is_wired_exactly_when_the_config_asks_for_it(config, system):
    """`critic:` is what puts a critic on the planner — nothing else does."""
    cfg = config.agent("PlannerAgent")
    assert (system.agent("PlannerAgent").plan_critic is not None) == cfg.uses_critic()
    # One round: a critic that keeps objecting must not loop the planner forever.
    assert system.agent("PlannerAgent").critic_max_rounds == 1


def test_turning_the_critic_on_builds_a_planner_that_has_one(config):
    raw = copy.deepcopy(config.model_dump(by_alias=True))
    raw["agents"]["PlannerAgent"]["critic"] = True
    planner = build_system(SystemConfig.model_validate(raw)).agent("PlannerAgent")

    assert planner.plan_critic is not None
    assert "PLAN REVIEW" in planner.instruction


def test_options_follow_settings_references(config, monkeypatch):
    """An `options:` value may be "${settings.path}" — that is how the plan
    critic's round budget reaches the agent from the web UI."""
    from CoScientist.config import get_settings

    cfg = config.agent("PlannerAgent")
    assert cfg.options["critic_max_rounds"] == "${web.planner_critic_rounds}"

    monkeypatch.setattr(get_settings().web, "planner_critic_rounds", 4)
    assert cfg.resolved_options() == {"critic_max_rounds": 4}


def test_the_critic_round_budget_reaches_the_built_planner(config, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().web, "planner_critic_rounds", 2)
    raw = copy.deepcopy(config.model_dump(by_alias=True))
    raw["agents"]["PlannerAgent"]["critic"] = True
    planner = build_system(SystemConfig.model_validate(raw)).agent("PlannerAgent")

    assert planner.critic_max_rounds == 2


def test_critic_on_a_plain_llm_agent_is_rejected(config):
    """Only the session agents own a review→revise loop a critic can drive."""
    raw = copy.deepcopy(config.model_dump(by_alias=True))
    raw["agents"]["ResearchAgent"]["critic"] = True
    with pytest.raises(Exception, match="critic"):
        SystemConfig.model_validate(raw)


def test_build_for_mode_planner(monkeypatch):
    from CoScientist.config import get_settings
    from CoScientist.agents import build_for_mode

    settings = get_settings()
    for mode in ("init", "planner"):
        monkeypatch.setattr(settings.web, "start_mode", mode)
        system = build_for_mode()
        assert system is not None
        assert system.root.name == "PlanningPipelineAgent"


def test_build_for_mode_orchestrator(monkeypatch):
    from CoScientist.config import get_settings
    from CoScientist.agents import build_for_mode

    settings = get_settings()
    monkeypatch.setattr(settings.web, "start_mode", "orchestrator")

    system = build_for_mode()
    assert system is not None
    assert system.root.name == "OrchestratorAgent"


def test_build_for_mode_orchestrator_planner(monkeypatch):
    from CoScientist.config import get_settings
    from CoScientist.agents import build_for_mode

    settings = get_settings()
    monkeypatch.setattr(settings.web, "start_mode", "orchestrator_planner")

    system = build_for_mode()
    assert system is not None
    assert system.root.name == "OrchestratorAgent"
    # PlannerAgent is disabled and removed from OrchestratorAgent's subordinates
    subs = [s.name for s in system.config.enabled_subordinates("OrchestratorAgent")]
    assert "PlannerAgent" not in subs

    # OrchestratorAgent has create_plan tool
    tools = _tool_names(system.root)
    assert "create_plan" in tools


def test_build_for_mode_run_root_includes_pipeline_pre_and_post(monkeypatch):
    from google.adk.agents.sequential_agent import SequentialAgent
    from CoScientist.config import get_settings
    from CoScientist.agents import build_for_mode

    settings = get_settings()
    monkeypatch.setattr(settings.web, "start_mode", "orchestrator")
    monkeypatch.setattr(settings.context_init, "enabled", True)

    system = build_for_mode()
    run_root = system.run_root
    assert isinstance(run_root, SequentialAgent)
    names = [a.name for a in run_root.sub_agents]
    assert names[0] == "ContextInitAgent"
    assert names[-1] == "ResultAggregatorAgent"
    assert "OrchestratorAgent" in names


def test_build_for_mode_planner_run_root(monkeypatch):
    from google.adk.agents.sequential_agent import SequentialAgent
    from CoScientist.config import get_settings
    from CoScientist.agents import build_for_mode

    settings = get_settings()
    monkeypatch.setattr(settings.context_init, "enabled", True)
    for mode in ("init", "planner"):
        monkeypatch.setattr(settings.web, "start_mode", mode)

        system = build_for_mode()
        run_root = system.run_root
        assert isinstance(run_root, SequentialAgent)
        names = [a.name for a in run_root.sub_agents]
        assert names[0] == "ContextInitAgent"
        assert names[1] == "PlanningPipelineAgent"
        assert names[-1] == "ResultAggregatorAgent"

        planning_agent = run_root.sub_agents[1]
        planning_children = [a.name for a in planning_agent.sub_agents]
        assert planning_children == ["PlannerAgent", "OrchestratorAgent"]






# ── web UI feature switches ──────────────────────────────────────────────────
# Knowledge Graph / Research Graph / Local Coder Tools each drop a tool entry
# out of every agent that lists it. The invariant under test is the one the
# assembler exists to protect: a switched-off tool must vanish from the agent
# AND from its prompt, so the model is never told to call something it lacks.

def _tool_names(agent) -> set:
    """Every identifier an attached tool goes by.

    Plain function tools carry ``__name__``, FunctionTool carries ``name``, and
    a toolset resolves its surface at runtime — so it is identified by class
    (GraphReaderToolset, ResearchGraphToolset).
    """
    names = set()
    for tool in getattr(agent, "tools", []):
        for candidate in (
            getattr(tool, "name", None),
            getattr(tool, "__name__", None),
            type(tool).__name__,
        ):
            if candidate:
                names.add(candidate)
    return names


def _build_with(monkeypatch, config, **flags):
    from CoScientist.config import get_settings

    settings = get_settings()
    for field, value in flags.items():
        if field == "research_graph_enabled":
            monkeypatch.setattr(settings.research_graph, "enabled", value)
        else:
            monkeypatch.setattr(settings.web, field, value)
    return build_system(config)


def test_knowledge_graph_switch_drops_graph_tools_and_prompt(monkeypatch, config):
    off = _build_with(monkeypatch, config, knowledge_graph_enabled=False)

    for name, cfg in config.agents.items():
        if "graph" not in cfg.tools:
            continue
        agent = off.agent(name)
        assert "GraphReaderToolset" not in _tool_names(agent), f"{name}: graph tool attached"

    # The orchestrator's KNOWLEDGE GRAPH section goes with it.
    assert "### KNOWLEDGE GRAPH" not in off.agent("OrchestratorAgent").instruction


def test_knowledge_graph_switch_on_attaches_graph_tools_and_prompt(monkeypatch, config):
    """The other half of the switch, pinned for the same reason as the coder's:
    a GRAPH__ENABLED=false in someone's .env must not fail this."""
    on = _build_with(monkeypatch, config, knowledge_graph_enabled=True)

    orchestrator = on.agent("OrchestratorAgent")
    assert "GraphReaderToolset" in _tool_names(orchestrator)
    assert "### KNOWLEDGE GRAPH" in orchestrator.instruction


def test_research_graph_switch_drops_tools_and_prompt(monkeypatch, config):
    off = _build_with(monkeypatch, config, research_graph_enabled=False)

    for name, cfg in config.agents.items():
        if not ({"research_graph", "research_graph_orchestrator"} & set(cfg.tools)):
            continue
        agent = off.agent(name)
        assert "ResearchGraphToolset" not in _tool_names(agent), f"{name}: research tool attached"
        if cfg.cls == "llm" and cfg.prompt:
            assert "research_commit" not in agent.instruction, f"{name}: prompt leaked"


def test_coder_local_tools_switch_leaves_only_the_sandbox(monkeypatch, config):
    """The CoderAgent keeps working — through the sandbox — and its prompt stops
    telling it to reach for execute_bash."""
    off = _build_with(monkeypatch, config, coder_mode="openhands")

    for name in ("CoderAgent", "DatasetCollectorAgent"):
        cfg = config.agent(name)
        assert "coder" in cfg.tools and "sandbox" in cfg.tools
        agent = off.agent(name)
        names = _tool_names(agent)
        assert "execute_bash" not in names, f"{name}: local coder tool still attached"
        assert "execute_bash" not in agent.instruction, f"{name}: prompt still names it"
        # Sandbox tools are the remaining execution surface (when configured).
        if "run_sandbox_task" in names:
            assert "run_sandbox_task" in agent.instruction


def test_coder_local_tools_switch_on_attaches_the_local_toolset(monkeypatch, config):
    """The other half of the switch. Pinned rather than read off the ambient
    settings: CODER__LOCAL_TOOLS_ENABLED=False in a developer's .env would
    otherwise turn this into a failure about their environment."""
    on = _build_with(monkeypatch, config, coder_mode="local")

    coder = on.agent("CoderAgent")
    assert "execute_bash" in _tool_names(coder)
    assert "execute_bash" in coder.instruction
