"""LLM Agents module — agents are assembled from CoScientist/agents/system.yaml.

The YAML is the single source of truth for the system layout (agents, tools,
callbacks, prompts, HITL, A2A exposure). This module builds the in-process
system once and re-exports the agent instances under their historical names so
existing imports keep working.
"""
import copy
import logging

from CoScientist.assembly import build_system
from CoScientist.assembly.schema import load_config
from CoScientist.logging import get_multi_agent_tracer
from CoScientist.agents.llm_repair import install_json_repair
from opik.integrations.adk import track_adk_agent_recursive

logger = logging.getLogger(__name__)

# Guard the LiteLlm tool-call JSON boundary process-wide BEFORE any runner executes:
# a malformed tool-call payload (qwen truncation / missing comma) must not kill the run.
# Idempotent; installed once at first import of the agents package (CLI + web both hit this).
install_json_repair()

_system = build_system()

# The full assembled system — for callers that need to iterate every agent of
# the ACTIVE profile (e.g. the web app wiring HITL handlers) instead of relying
# on the historical fixed names below.
agent_system = _system

# `orchestrator_agent` == the delegation-tree orchestrator (the real LLM). It is
# the config `root` (root: true in system.yaml) and stays the named export every
# caller (prompts, A2A) expects.
#
# Historical named exports. Alternative profiles ($COSCIENTIST_CONFIG, e.g.
# "microfluidics") declare only a subset of the agents — the missing ones
# resolve to None so importing this module keeps working for every profile.
orchestrator_agent = _system.root
root_agent = orchestrator_agent

# Agents that run as pipeline stages (pre/post) around the orchestrator.
pipeline_pre_agents = [_system.agent(n) for n in _system.config.pipeline.pre if _system.config.agent(n).is_enabled()]
pipeline_post_agents = [_system.agent(n) for n in _system.config.pipeline.post if _system.config.agent(n).is_enabled()]

# The RUN root: the whole lifecycle (pre → orchestrator → post/aggregator) is one
# ADK SequentialAgent, driven by a single Runner.run_async so it is ONE invocation
# = ONE trace, with the Result Aggregator as the terminal child (it reads the graph
# the orchestrator populated and writes the report). When no pipeline stages are
# declared, the orchestrator IS the run root (no needless wrapper).
run_root = _system.run_root

planner_agent = _system.agents.get("PlannerAgent")
hypotheses_agent = _system.agents.get("HypothesesAgent")
research_agent = _system.agents.get("ResearchAgent")
task_execution_agent = _system.agents.get("TaskExecutorAgent")
medical_agent = _system.agents.get("MedicalAgent")
coder_agent = _system.agents.get("CoderAgent")
tool_agent = _system.agents.get("ToolPreparerAgent")
tool_retriever_agent = _system.agents.get("ToolRetrieverAgent")
tool_reranker_agent = _system.agents.get("ToolReranker")
tool_websearcher_agent = _system.agents.get("ToolWebSearcherAgent")
fedot_agent = _system.agents.get("ExperimentAgent")
result_aggregator_agent = _system.agents.get("ResultAggregatorAgent")
tz_agent = _system.agents.get("TZAgent")

# Attach the Opik tracer only when tracing is enabled (see OPIK__ENABLED).
_tracer = get_multi_agent_tracer()
if _tracer is not None:
    track_adk_agent_recursive(run_root, _tracer)
    for _ag in _system.agents.values():
        if isinstance(getattr(_ag, "after_model_callback", None), list) and len(_ag.after_model_callback) > 1:
            _ag.after_model_callback.insert(0, _ag.after_model_callback.pop())
        if isinstance(getattr(_ag, "before_tool_callback", None), list) and len(_ag.before_tool_callback) > 1:
            _ag.before_tool_callback.insert(0, _ag.before_tool_callback.pop())


def build_for_mode():
    """Build an AgentSystem configured for the current start mode from settings.

    Reads ``settings.web.start_mode``:
      * ``"init"`` / ``"planner"`` — PlanningPipelineAgent is root (sequential: PlannerAgent →
        OrchestratorAgent).
      * ``"orchestrator"`` — OrchestratorAgent is root, with PlannerAgent
        added to its subordinates so it can be invoked on demand.
      * ``"orchestrator_planner"`` — OrchestratorAgent is root, provided with
        create_plan_tool directly, while PlannerAgent is disabled.

    Other runtime-tunable parameters (e.g. ``max_searches``) are read from
    ``settings.web`` by individual components at build time.

    Returns:
        An :class:`~CoScientist.assembly.assembler.AgentSystem`.
    """
    from CoScientist.config import get_settings
    start_mode = get_settings().web.start_mode

    if start_mode in ("init", "planner"):
        raw_config = load_config()
        patched = copy.deepcopy(raw_config)
        pipeline_agent_name = "PlanningPipelineAgent" if "PlanningPipelineAgent" in patched.agents else "InitAgent"
        if pipeline_agent_name in patched.agents:
            patched.agents[pipeline_agent_name].root = True
            patched.agents[pipeline_agent_name].enabled = True
            patched.agents["OrchestratorAgent"].root = False
            # In Planner mode the PlannerAgent runs first and its output replaces
            # the original user query; inject_original_query restores it so the
            # OrchestratorAgent sees the original request.
            # Must run before redact_link_urls so that any links in the restored
            # query are subsequently redacted into [[linkXXXX]] references.
            orch_cb = patched.agents["OrchestratorAgent"].callbacks.before_model
            if "inject_original_query" not in orch_cb:
                if "redact_link_urls" in orch_cb:
                    orch_cb.insert(orch_cb.index("redact_link_urls"), "inject_original_query")
                else:
                    orch_cb.insert(0, "inject_original_query")
            system = build_system(config=patched)
        else:
            logger.warning(
                "start_mode is set to %r but 'PlanningPipelineAgent' is not present in "
                "the system config; falling back to default build_system()",
                start_mode,
            )
            system = build_system()
        _tracer = get_multi_agent_tracer()
        if _tracer is not None:
            track_adk_agent_recursive(system.run_root, _tracer)
            for _ag in system.agents.values():
                if isinstance(getattr(_ag, "after_model_callback", None), list) and len(_ag.after_model_callback) > 1:
                    _ag.after_model_callback.insert(0, _ag.after_model_callback.pop())
                if isinstance(getattr(_ag, "before_tool_callback", None), list) and len(_ag.before_tool_callback) > 1:
                    _ag.before_tool_callback.insert(0, _ag.before_tool_callback.pop())
        return system

    if start_mode in ("orchestrator_planner", "orchestrator_plan"):
        raw_config = load_config()
        patched = copy.deepcopy(raw_config)

        # Make OrchestratorAgent the root.
        patched.agents["OrchestratorAgent"].root = True
        for name in ("PlanningPipelineAgent", "InitAgent"):
            if name in patched.agents:
                patched.agents[name].root = False
                patched.agents[name].enabled = False

        # Disable PlannerAgent and remove from Orchestrator's subordinates.
        if "PlannerAgent" in patched.agents:
            patched.agents["PlannerAgent"].root = False
            patched.agents["PlannerAgent"].enabled = False

        orch_subs = patched.agents["OrchestratorAgent"].subordinates
        if "PlannerAgent" in orch_subs:
            orch_subs.remove("PlannerAgent")

        # Give OrchestratorAgent the tool for creating/registering plans directly.
        orch_tools = patched.agents["OrchestratorAgent"].tools
        if "create_plan_tool" not in orch_tools:
            orch_tools.append("create_plan_tool")

        system = build_system(config=patched)
        _tracer = get_multi_agent_tracer()
        if _tracer is not None:
            track_adk_agent_recursive(system.run_root, _tracer)
            for _ag in system.agents.values():
                if isinstance(getattr(_ag, "after_model_callback", None), list) and len(_ag.after_model_callback) > 1:
                    _ag.after_model_callback.insert(0, _ag.after_model_callback.pop())
                if isinstance(getattr(_ag, "before_tool_callback", None), list) and len(_ag.before_tool_callback) > 1:
                    _ag.before_tool_callback.insert(0, _ag.before_tool_callback.pop())
        return system

    if start_mode != "orchestrator":
        raise ValueError(
            f"Unknown start_mode {start_mode!r}; expected 'init'/'planner', 'orchestrator', or 'orchestrator_planner'"
        )

    # Load a fresh config and patch it for orchestrator-as-root mode.
    raw_config = load_config()
    patched = copy.deepcopy(raw_config)

    # Make OrchestratorAgent the root.
    patched.agents["OrchestratorAgent"].root = True
    for name in ("PlanningPipelineAgent", "InitAgent"):
        if name in patched.agents:
            patched.agents[name].root = False
            patched.agents[name].enabled = False

    # Add PlannerAgent to OrchestratorAgent's subordinates (if not already).
    orch_subs = patched.agents["OrchestratorAgent"].subordinates
    if "PlannerAgent" not in orch_subs:
        orch_subs.insert(0, "PlannerAgent")

    # Re-validate the patched config and build.
    system = build_system(config=patched)
    _tracer = get_multi_agent_tracer()
    if _tracer is not None:
        track_adk_agent_recursive(system.run_root, _tracer)
        for _ag in system.agents.values():
            if isinstance(getattr(_ag, "after_model_callback", None), list) and len(_ag.after_model_callback) > 1:
                _ag.after_model_callback.insert(0, _ag.after_model_callback.pop())
            if isinstance(getattr(_ag, "before_tool_callback", None), list) and len(_ag.before_tool_callback) > 1:
                _ag.before_tool_callback.insert(0, _ag.before_tool_callback.pop())
    return system

__all__ = [
    "agent_system",
    "orchestrator_agent",
    "root_agent",
    "run_root",
    "planner_agent",
    "fedot_agent",
    "research_agent",
    "hypotheses_agent",
    "medical_agent",
    "coder_agent",
    "tool_retriever_agent",
    "tool_reranker_agent",
    "tool_websearcher_agent",
    "task_execution_agent",
    "tool_agent",
    "result_aggregator_agent",
    "pipeline_pre_agents",
    "pipeline_post_agents",
    "tz_agent",
    "build_for_mode",
]
