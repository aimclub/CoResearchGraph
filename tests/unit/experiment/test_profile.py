"""Experiment profile, settings, HITL, prompts, Fedot hard-stop."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from CoScientist.assembly.schema import load_config, resolve_config_path
from CoScientist.config.settings import ExperimentsSettings, Settings
from CoScientist.hitl.models import HITLAction, HITLRequest
from CoScientist.web.handler import WebHITLHandler

def test_experiments_settings_defaults_and_nested_env(monkeypatch):
    defaults = ExperimentsSettings()
    # FEDOT.MAS is off unless asked for: not reliable enough to be a default.
    assert defaults.route_fedot is False
    assert defaults.route_coder_mcp is False
    assert defaults.allow_coder_fallback is True
    assert defaults.route_alembic is False
    assert defaults.fallback_research == ["research"]
    assert defaults.fallback_medical == ["medical"]
    assert defaults.require_task_design is True
    assert defaults.task_max_attempts == 2
    assert defaults.max_plan_tasks == 8

    monkeypatch.setenv("EXPERIMENTS__ROUTE_FEDOT", "true")
    monkeypatch.setenv("EXPERIMENTS__MAX_PLAN_TASKS", "6")
    configured = Settings(_env_file=None)
    assert configured.experiments.route_fedot is True
    assert configured.experiments.max_plan_tasks == 6


def test_experiment_profile_is_an_overlay_not_a_copy():
    """The profile says only what belongs to the module.

    It was a standalone copy of system.yaml once, forked before the research
    graph existed, and it drifted for two months. Re-declaring a section here
    is how that starts again, so the absence of one is the assertion.
    """
    import yaml

    with open(resolve_config_path("experiments"), encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    assert raw["extends"] == "system"
    for section in ("defaults", "pipeline", "internal_tools"):
        assert section not in raw, f"{section} must be inherited, not restated"


def test_the_module_replaces_mains_execution_lane():
    config = load_config(resolve_config_path("experiments"))
    assert config.root.name == "OrchestratorAgent"
    orch = config.agent("OrchestratorAgent")
    assert orch.cls == "llm"
    assert orch.children == []
    # main's roster, with the module standing exactly where TaskExecutor stood.
    assert orch.subordinates == [
        "PlannerAgent",
        "HypothesesAgent",
        "ResearchAgent",
        "ExperimentModuleAgent",
        "MedicalAgent",
    ]
    # Three agents stand down, each for a structural reason: ADK allows an agent
    # exactly one `children` parent, and ToolPreparerAgent belongs to the module.
    for name in ("TaskExecutorAgent", "ToolPipelineAgent", "ExecutorSwitchAgent"):
        assert not config.agent(name).is_enabled(), name
    module = config.agent("ExperimentModuleAgent")
    assert module.children == [
        "ToolPreparerAgent",
        "ExperimentPlannerAgent",
        "ExperimentExecutorAgent",
        "ExperimentResultReviewAgent",
    ]
    # The A2A identity moves with the lane — two agents cannot claim one key.
    assert module.a2a.key == "task_execution"
    assert module.a2a.port == 8004
    assert config.agent("TaskExecutorAgent").a2a is None
    # The module's answer is a deliverable, like TaskExecutor's in main. It is
    # the only one of the four attached as an AgentTool, so the only one where
    # the flag does anything.
    assert module.report_output is True


def test_the_profile_inherits_mains_capabilities():
    """What the overlay does NOT say is the point: it arrives from system.yaml."""
    config = load_config(resolve_config_path("experiments"))
    main = load_config(resolve_config_path("system"))

    assert config.internal_tools and config.internal_tools == main.internal_tools
    # Whatever main enables, the profile enables: the point is that the overlay
    # does not say so. Read from main rather than frozen as a list, because
    # some of these are switchable — MedicalAgent has a narrow role and turns
    # off with MEDICAL__ENABLED=false — and a frozen list makes the switch look
    # like a regression.
    work_order = {a.name for a in config.agents.values()
                  if a.work_order and a.is_enabled()}
    assert work_order == {a.name for a in main.agents.values()
                          if a.work_order and a.is_enabled()}
    # The ones that are not switchable are still expected to be there.
    assert {"HypothesesAgent", "ResearchAgent", "DatasetCollectorAgent",
            "ExperimentAgent"} <= work_order
    # Plumbing the web UI hides, the S3 vault, the anti-fabrication checker and
    # the per-call sandbox approval: not one line of this file mentions them.
    assert len(config.internal_agent_names()) >= 12
    coder = config.agent("CoderAgent")
    assert {"vault", "verify"} <= set(coder.tools)
    assert coder.hitl and "hitl_before_tool" in coder.callbacks.before_tool
    # Resolved, not the raw declaration: the hypothesis generator's dial is a
    # "${settings.path}" reference so one run can turn it down without moving
    # the default. What the profile inherits is the value, either way.
    assert config.agent("HypothesesAgent").resolved_reasoning() == "high"
    assert config.agent("CoderAgent").resolved_reasoning() == "medium"


def test_route_agents_read_the_graph_and_the_bridge_writes_it():
    """A task must not end up with two independent Evidence nodes.

    The run record is written deterministically by runtime/graph_bridge from
    record_result — richer than an LLM commit and schema-correct — so the route
    agents get the research graph read-only. Lanes whose commit IS the science
    keep main's writing tool.
    """
    config = load_config(resolve_config_path("experiments"))
    for name in ("ExperimentAgent", "FedotAgent", "CoderAgent"):
        tools = config.agent(name).tools or []
        assert "research_graph_readonly" in tools, name
        assert "research_graph" not in tools, name
    for name in (
        "HypothesesAgent",
        "ResearchAgent",
        "MedicalAgent",
        "DatasetCollectorAgent",
    ):
        assert "research_graph" in (config.agent(name).tools or []), name
    # build_experiment_context reads state["research_context"]; without this
    # callback the planner inherited whatever slice ran last, or nothing.
    planner = config.agent("ExperimentPlannerAgent")
    assert planner.callbacks.before_agent.index("inject_research_context") < (
        planner.callbacks.before_agent.index("build_experiment_context")
    )


def test_the_modules_own_methods_survive_the_overlay():
    config = load_config(resolve_config_path("experiments"))

    hyp = config.agent("HypothesesAgent")
    assert hyp.prompt == "hypotheses"
    # Never a literal model string: that goes straight to the named provider and
    # ignores LLM__MAIN_MODEL, which is what routes this profile through
    # OpenRouter. Unset means defaults.model, which is `main`.
    assert hyp.model in (None, "main")
    assert "commit_experiment_hypotheses" in hyp.callbacks.after_agent
    assert "seed_hypotheses_from_em_request" in hyp.callbacks.before_model
    assert hyp.callbacks.after_model.index("enforce_hypothesis_research_commit") < (
        hyp.callbacks.after_model.index("normalize_em_hypothesis_commit")
    )
    assert "capture_hypotheses_after_research_commit" in hyp.callbacks.after_tool
    # Root must be bootstrapped before inject_research_context renders the
    # {research_context?} placeholder, so a fresh graph never reads EMPTY.
    assert hyp.callbacks.before_agent.index("bootstrap_research_question_if_empty") < (
        hyp.callbacks.before_agent.index("inject_research_context")
    )

    orch = config.agent("OrchestratorAgent")
    assert "coalesce_experiment_module_calls" in orch.callbacks.after_model
    assert "suppress_experiment_module_after_completed" in orch.callbacks.after_model
    assert "ask_pipeline_scope" in orch.callbacks.before_agent
    # The orchestrator must not rewrite ResearchAgent calls into experiments.
    # Literature belongs to the parent; inventory selects computational routes.
    assert "redirect_research_to_experiment_module" not in orch.callbacks.after_model
    assert "normalize_experiment_module_brief" not in orch.callbacks.after_model
    assert "inject_upstream_artifacts" not in orch.callbacks.before_agent

    preparer = config.agent("ToolPreparerAgent")
    assert "skip_when_experiment_stage_complete" in preparer.callbacks.before_agent
    assert "assess_experiment_inventory_feasibility" in preparer.callbacks.after_agent

    retriever = config.agent("ToolRetrieverAgent")
    assert retriever.prompt == "experiment_tool_retriever"
    assert "persist_experiment_em_request" in retriever.callbacks.before_agent
    assert "reset_experiment_retrieval_budget" in retriever.callbacks.before_agent
    assert "enforce_experiment_retrieval_budget" in retriever.callbacks.after_model
    assert "persist_experiment_retrieved_capabilities" in retriever.callbacks.after_agent

    reranker = config.agent("ToolReranker")
    # shortlist_reranker_tools narrows accumulated_tools to the top-K, and the
    # full pre-rerank set is what the planner falls back on when the reranker
    # abstains — so the module's snapshot has to be taken first.
    assert reranker.callbacks.before_agent.index(
        "stash_experiment_retrieved_capabilities"
    ) < reranker.callbacks.before_agent.index("shortlist_reranker_tools")
    assert reranker.callbacks.after_model == ["sanitize_json_output"]
    assert "collect_reranked_tools" in reranker.callbacks.after_agent
    assert "snapshot_experiment_discovered_capabilities" in reranker.callbacks.after_agent

    executor = config.agent("ExperimentExecutorAgent")
    assert config.agent("OrchestratorAgent").resolved_reasoning() == "low"
    assert config.agent("ExperimentPlannerAgent").resolved_reasoning() == "medium"
    assert executor.resolved_reasoning() == "low"
    assert config.agent("ExperimentResultReviewAgent").resolved_reasoning() == "medium"
    assert executor.callbacks.before_agent[0] == "skip_when_experiment_stage_complete"
    assert executor.callbacks.before_tool == ["guard_experiment_route"]
    assert executor.callbacks.after_tool == ["mark_experiment_route_returned"]
    # One selector owns all executor transitions.  Independent rewriters can
    # otherwise drive one another back into fallback/read loops.
    assert executor.callbacks.after_model == ["enforce_pending_record_result"]
    assert "CoderAgent" in executor.subordinates
    assert "McpBuilderAgent" in executor.subordinates
    assert "ResearchAgent" not in executor.subordinates
    assert "ResearchAgent" in config.agent("OrchestratorAgent").subordinates
    assert "MedicalAgent" in executor.subordinates

    fedot = config.agent("FedotAgent")
    assert fedot.prompt == "experiment_fedot_route"
    assert "pin_fedot_alembic_task" in fedot.callbacks.before_tool
    assert "refuse_when_fedot_deliverable" not in fedot.callbacks.before_agent
    assert config.agent("ExperimentAgent").prompt == "experiment_react_route"
    assert "task_tracker" not in config.agent("ExperimentAgent").tools
    assert "force_schema_s3_upload" in config.agent("ExperimentAgent").callbacks.before_tool

    mcp_builder = config.agent("McpBuilderAgent")
    assert "pin_alembic_build_args" in mcp_builder.callbacks.before_tool
    assert "await_alembic_job_if_experiment" in mcp_builder.callbacks.after_tool

    planner = config.agent("ExperimentPlannerAgent")
    assert planner.include_contents == "none"
    assert "skip_retriever_context" in planner.callbacks.before_model
    # ExperimentPlan is enforced by sanitize_json_output + deterministic critique.
    assert planner.output_schema is None
    assert config.agent("ResultAggregatorAgent").prompt == "experiment_result_aggregator"


def test_the_overlay_keeps_mains_link_and_artifact_hooks():
    """Re-declaring `callbacks:` replaces the whole block, so every hook main
    put on an agent has to be repeated here. This is the guard against losing
    one silently: uploaded-file links, tool-result links and the [[linkNNNN]]
    round trip all die quietly, not loudly."""
    config = load_config(resolve_config_path("experiments"))
    main = load_config(resolve_config_path("system"))
    hooks = {
        "user_links",
        "redact_link_urls",
        "resolve_link_refs",
        "register_tool_result_links",
        "expand_link_refs",
        "inject_report_language",
        "inject_dataset_context",
        "capture_mcp_artifacts",
        "save_uploaded_artifacts",
    }
    for name, agent in config.agents.items():
        if name not in main.agents or not agent.is_enabled():
            continue
        for kind in ("before_agent", "before_model", "before_tool", "after_tool", "after_model", "after_agent"):
            expected = [c for c in getattr(main.agent(name).callbacks, kind) if c in hooks]
            got = getattr(agent.callbacks, kind)
            missing = [c for c in expected if c not in got]
            assert not missing, f"{name}.{kind} lost {missing}"
    # Ordering rules the registry itself depends on.
    for name, agent in config.agents.items():
        if not agent.is_enabled():
            continue
        before_agent = agent.callbacks.before_agent
        if "user_links" in before_agent:
            assert before_agent[-1] == "user_links", f"{name}: user_links must be last"
        after_model = agent.callbacks.after_model
        if "expand_link_refs" in after_model:
            assert after_model[0] == "expand_link_refs", f"{name}: expand_link_refs must be first"


def test_the_experiment_profile_builds():
    """Validation is not enough: `python -m CoScientist.assembly` only checks the
    config, and ADK's one-parent rule for `children` fires at construction. A
    double-parented agent would sail through CI without this."""
    from CoScientist.assembly import build_system
    from CoScientist.experiments.review import ExperimentReviewSessionAgent

    config = load_config(resolve_config_path("experiments"))
    system = build_system(config)
    module = system.agent("ExperimentModuleAgent")
    from CoScientist.experiments.scope import skip_literature_only_experiment

    callbacks = module.before_agent_callback
    assert skip_literature_only_experiment in (callbacks if isinstance(callbacks, list) else [callbacks])
    for name, owns_research in (("OrchestratorAgent", True), ("ExperimentExecutorAgent", False)):
        attached = {getattr(getattr(tool, "agent", None), "name", None)
                    for tool in system.agent(name).tools}
        assert ("ResearchAgent" in attached) is owns_research
    assert [child.name for child in module.sub_agents] == [
        "ToolPreparerAgent",
        "ExperimentPlannerAgent",
        "ExperimentExecutorAgent",
        "ExperimentResultReviewAgent",
    ]
    assert system.agent("ToolPreparerAgent").parent_agent.name == "ExperimentModuleAgent"
    for name in ("ExperimentPlannerAgent", "ExperimentResultReviewAgent"):
        review_agent = system.agent(name)
        assert isinstance(review_agent, ExperimentReviewSessionAgent)
        assert review_agent.hitl_handler is not None  # fail-closed even headless
    assert system.agent("ExperimentPlannerAgent").include_contents == "none"
    from CoScientist.experiments.context.builder import build_experiment_context
    from CoScientist.experiments.plan_policy import check_experiment_plan_capacity

    callbacks = system.agent("ExperimentPlannerAgent").before_agent_callback
    assert callbacks.index(build_experiment_context) < callbacks.index(check_experiment_plan_capacity)


def test_planner_and_coder_prompts_cover_multi_h_and_anti_fabrication():
    from unittest.mock import MagicMock

    from CoScientist.experiments.prompts import templates as exp_prompts

    ctx = MagicMock()
    ctx.render_tools.return_value = ""
    ctx.render_agents.return_value = ""
    planner = exp_prompts.experiment_planner(ctx)
    coder = exp_prompts.experiment_coder_route(ctx)
    executor = exp_prompts.experiment_executor(ctx)
    retriever = exp_prompts.experiment_tool_retriever(ctx)
    assert "hypothesis_refs" in planner and "AUTHORITATIVE" in planner
    assert "operations is AUTHORITATIVE" in planner
    assert "design.operation_ref" in planner
    assert "Do NOT invent extra hypotheses" in planner
    assert "HypothesesAgent" in planner
    assert "deterministic review asks the operator" in planner
    assert "modify|unknown → route=coder" in planner
    assert "Bind exact inventory server_id+tool" in planner
    assert "risks/assumptions only at plan root" in planner
    assert "Mandatory markdown/HTML reports are forbidden" in planner
    assert "NEVER add a narrative task" in planner
    assert "Leftover MCP for a different operation is not coverage" in planner
    assert "Otherwise use route=coder" in planner
    assert "role=data" in planner
    assert "different-family" in planner
    assert "Cover every distinct operation" in retriever
    assert "one non-optional" in planner and "distinct target" in planner
    assert "also_tests" in planner
    assert "Uncovered hypothesis_refs" in planner
    assert "ANTI-FABRICATION" in coder
    assert "hardcoded" in coder.lower()
    assert "simulated/hardcoded" in executor.lower() or "fabricated" in executor.lower()
    assert "current phase" in executor and "reporting" in executor


def test_research_prompt_requires_both_literature_tools():
    from unittest.mock import MagicMock

    from CoScientist.agents.prompts.templates import research

    ctx = MagicMock()
    ctx.has_tool.side_effect = lambda key: key in {"paper_analysis", "papers_search"}
    ctx.render_tools.return_value = ""
    ctx.render_hitl.return_value = ""
    prompt = research(ctx)
    sci_at = prompt.find("`explore_scientific_database`")
    papers_at = prompt.find("ALWAYS call `search_papers`")
    assert sci_at != -1
    assert papers_at != -1
    assert sci_at < papers_at
    assert "Do not treat the RAG answer as the end of a literature review" in prompt
    assert "Never invent tool names" in prompt
    assert "fall back to `tavily_search`" in prompt
    assert "immediately fall back" not in prompt
    assert "argument `keywords`" in prompt


def test_web_hitl_timeout_is_fail_closed_for_every_request(monkeypatch):
    """Silence is a refusal now, everywhere — not only on the two experiment
    reviews. It used to approve for everything else, which is how thirty-four
    decisions in the recorded sessions were taken by the clock, and the response
    carried `timed_out=False` so nothing downstream could tell them from a human
    saying yes. A run that must proceed unattended has `HITL__MODE=auto`, and
    that is named out loud rather than inferred from nobody being there."""
    import CoScientist.hitl.mode as mode_mod

    monkeypatch.setattr(mode_mod, "wait_seconds", lambda: 0.001)

    async def scenario():
        handler = WebHITLHandler()
        experiment = await handler.handle_request(
            HITLRequest(
                agent_name="ExperimentPlannerAgent",
                action_type=HITLAction.APPROVE,
                message="Approve experiment plan",
                context={"experiment_review_kind": "plan"},
                timeout_seconds=0.001,
            )
        )
        assert experiment.approved is False
        assert experiment.timed_out is True

        other = await handler.handle_request(
            HITLRequest(
                agent_name="CoderAgent",
                action_type=HITLAction.APPROVE,
                message="Approve outward action",
                timeout_seconds=0.001,
            )
        )
        assert other.approved is False, "the clock does not approve a sandbox call"
        assert other.timed_out is True, "and it says plainly that nobody answered"

    asyncio.run(scenario())


def test_auto_mode_answers_without_drawing_a_card(monkeypatch):
    """`auto` is the mode for a run nobody is sitting with: no card, no wait."""
    monkeypatch.setenv("HITL__MODE", "auto")

    async def scenario():
        handler = WebHITLHandler()
        answered = await handler.handle_request(
            HITLRequest(agent_name="CoderAgent", action_type=HITLAction.APPROVE,
                        message="Approve outward action", timeout_seconds=600)
        )
        assert answered.approved is True
        assert answered.timed_out is False
        # Nothing was broadcast and nothing waited: the request never entered
        # the pending table.
        assert not handler._pending

    asyncio.run(scenario())


def test_glued_imperative_ask_splits_into_internal_ops():
    from CoScientist.context_init.operations import parse_glued_imperative_operations

    l2 = (
        "Generate GSK-3beta inhibitors with high activity. "
        "Suggest some small molecules that inhibit KRAS G12C - a target "
        "responsible for non-small cell lung cancer. "
        "Generate high activity tyrosine-protein kinase BTK inhibitors. "
        "Generate 2 molecules that would help with a blood lipid spectrum disorder."
    )
    ops = parse_glued_imperative_operations(l2)
    assert [op.operation_id for op in ops] == ["OP-1", "OP-2", "OP-3", "OP-4"]
    assert ops[0].statement.startswith("Generate GSK-3beta")
    assert "1." not in l2


# ── FEDOT.MAS: one switch, and the tree is the truth ─────────────────────────

def _experiments_without_fedot_in_executor():
    config = load_config(resolve_config_path("experiments"))
    config.agents["ExperimentExecutorAgent"].subordinates.remove("FedotAgent")
    return config


def _planner_prompt(config) -> str:
    from CoScientist.assembly.prompting import PromptContext
    from CoScientist.experiments.prompts import templates as exp_prompts

    ctx = PromptContext(config=config.agent("ExperimentPlannerAgent"), system=config)
    return exp_prompts.experiment_planner(ctx)


def test_fedot_agent_hangs_on_the_experiment_switch(monkeypatch):
    """Not on EXECUTOR__FEDOT_FALLBACK, which is main's reranker fallback: the
    agent, its roster line and every route decision follow one switch."""
    from CoScientist.config import get_settings

    config = load_config(resolve_config_path("experiments"))
    assert config.agent("FedotAgent").enabled == "${experiments.route_fedot}"
    roster = lambda: [a.name for a in config.enabled_subordinates("ExperimentExecutorAgent")]

    monkeypatch.setattr(get_settings().experiments, "route_fedot", False)
    assert "FedotAgent" not in roster()
    assert "ExperimentAgent" in roster()

    monkeypatch.setattr(get_settings().experiments, "route_fedot", True)
    monkeypatch.setattr(get_settings().web, "fedot_fallback_enabled", False)
    assert "FedotAgent" in roster()


def test_the_planner_never_hears_of_fedot_while_it_is_off(monkeypatch):
    """A route named in the prompt is a route the model uses: `react_tools if
    FEDOT off` as an aside was how every plan ended up on fedot_mas."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().experiments, "route_fedot", False)
    off = _planner_prompt(load_config(resolve_config_path("experiments")))
    assert "fedot" not in off.lower()
    assert '"route":"react_tools"' in off
    assert "post_build_route=react_tools" in off
    assert "<<" not in off

    # Switch on but the agent taken out of the executor: still off, because
    # the executor could not run it.
    monkeypatch.setattr(get_settings().experiments, "route_fedot", True)
    detached = _planner_prompt(_experiments_without_fedot_in_executor())
    assert "fedot" not in detached.lower()


def test_with_fedot_on_react_tools_stays_the_mcp_route(monkeypatch):
    """On, FEDOT.MAS is the narrow exception, not the default it used to be."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().experiments, "route_fedot", True)
    on = _planner_prompt(load_config(resolve_config_path("experiments")))
    assert "route: react_tools|fedot_mas|" in on
    assert '"route":"react_tools"' in on
    assert "→ react_tools (ExperimentAgent calls the bound tool directly)" in on
    assert "fedot_mas ONLY when ONE task must itself chain ≥2" in on
    assert "never the default" in on
    assert "FEDOT off" not in on
    assert "post_build_route=fedot_mas" not in on
    assert "<<" not in on


def test_the_planner_is_offered_the_medical_route_only_with_its_agent(monkeypatch):
    """MEDICAL__ENABLED used to reach only the runtime: the planner kept writing
    medical tasks for an agent that was not in the tree."""
    from CoScientist.config import get_settings

    config = load_config(resolve_config_path("experiments"))
    on = _planner_prompt(config)
    assert "alembic_build|medical" in on
    assert "|research" not in on
    assert "→ medical, mcp_servers=[]" in on

    monkeypatch.setattr(get_settings().web, "medical_agent_enabled", False)
    roster = [a.name for a in config.enabled_subordinates("ExperimentExecutorAgent")]
    assert "MedicalAgent" not in roster
    off = _planner_prompt(config)
    assert "medic" not in off.lower()
    # Rule 3 stays, so the rules cited by number still line up.
    assert "3) PubMed/PICO/DICOM asks: there is no clinical route in this run." in off
    # Falls through to Alembic (4) before coder (5), as with the route on.
    assert "other computational work falls through to routes 4-5" in off
    assert "<<" not in off


def test_the_profile_builds_without_fedot(monkeypatch):
    """Switched off, FedotAgent is neither a tool of the executor nor a line in
    its route roster, so nothing can hand work to it."""
    from CoScientist.assembly import build_system
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().experiments, "route_fedot", False)
    system = build_system(load_config(resolve_config_path("experiments")))
    executor = system.agent("ExperimentExecutorAgent")
    agent_tools = {getattr(getattr(t, "agent", None), "name", None) for t in executor.tools}
    assert "FedotAgent" not in agent_tools
    assert "ExperimentAgent" in agent_tools
    assert "FedotAgent" not in executor.instruction
