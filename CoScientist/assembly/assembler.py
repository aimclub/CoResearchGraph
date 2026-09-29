"""Builds the agent system from ``system.yaml``.

The assembler walks the validated :class:`SystemConfig` in dependency order and
constructs every declared agent:

  * ``llm``         -> google.adk LlmAgent — model, prompt (rendered from the
                       agent's PromptContext), tools, subordinate AgentTools,
                       callbacks, HITL tools, output_key/schema, planner
  * ``sequential``  -> SequentialAgent over ``children``
  * ``parallel``    -> ParallelAgent over ``children``
  * ``loop``        -> LoopAgent over ``children``, repeating them until one
                       escalates (``options.max_iterations`` bounds the loop)
  * ``custom:<x>``  -> the registered class (e.g. SessionAgent), passing
                       ``options`` through as constructor kwargs; ``critic:``
                       hands it a plan critic for its review loop

Disabled agents are still BUILT (so they can be served standalone over A2A);
``enabled`` only controls whether parents attach/advertise them.

With ``remote_subagents=True`` every subordinate that has an ``a2a`` section is
attached as a ``RemoteA2aAgent`` (HTTP) instead of the in-process instance —
prompts, rosters and critic wiring stay identical between the two modes by
construction.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools.agent_tool import AgentTool

# Populate the registry (tools/callbacks/classes + prompt templates).
import CoScientist.assembly.bindings  # noqa: F401  (registration side effect)
import CoScientist.agents.prompts.templates  # noqa: F401  (registration side effect)
import CoScientist.experiments.prompts.templates  # noqa: F401  (profile prompts)

from CoScientist.assembly.bindings import (
    HITL_TOOL_DOCS,
    WORK_ORDER_TOOL_DOCS,
    make_plan_critic,
)
from CoScientist.assembly.prompting import PromptContext
from CoScientist.agents.prompts.templates import _LANGUAGE_REQUIREMENT
from CoScientist.assembly.registry import REGISTRY, ToolEntry
from CoScientist.assembly.schema import (
    COMPOSITE_CLASSES,
    PIPELINE_ROOT_NAME,
    AgentConfig,
    SystemConfig,
    agent_override,
    default_reasoning_override,
    get_config,
    load_config,
    load_plugins,
)

_logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"<<[A-Z_]+>>")

_COMPOSITE_AGENT_CLASSES = {
    "sequential": SequentialAgent,
    "parallel": ParallelAgent,
    "loop": LoopAgent,
}


@dataclass
class AgentSystem:
    """The assembled system: every built agent by name, plus the root."""

    config: SystemConfig
    agents: Dict[str, BaseAgent] = field(default_factory=dict)
    _run_root: Optional[BaseAgent] = field(default=None, init=False, repr=False)

    @property
    def root(self) -> BaseAgent:
        return self.agents[self.config.root.name]

    @property
    def run_root(self) -> BaseAgent:
        if self._run_root is not None:
            return self._run_root
        pipeline_pre = [
            self.agents[n]
            for n in self.config.pipeline.pre
            if self.config.agent(n).is_enabled()
        ]
        pipeline_post = [
            self.agents[n]
            for n in self.config.pipeline.post
            if self.config.agent(n).is_enabled()
        ]
        if pipeline_pre or pipeline_post:
            from google.adk.agents.sequential_agent import SequentialAgent

            self._run_root = SequentialAgent(
                name=PIPELINE_ROOT_NAME,
                description=(
                    "Full research lifecycle: orchestrator run then report"
                    " synthesis."
                ),
                sub_agents=[*pipeline_pre, self.root, *pipeline_post],
            )
        else:
            self._run_root = self.root
        return self._run_root

    def agent(self, name: str) -> BaseAgent:
        if name not in self.agents:
            raise KeyError(f"Unknown agent {name!r}. Known: {sorted(self.agents)}")
        return self.agents[name]


def _resolve_model(cfg: AgentConfig, system: SystemConfig):
    from CoScientist.agents.common import make_coder_llm, make_llm

    # The operator's model for this agent (web UI → Agents) wins over the YAML.
    override = agent_override(cfg.name)
    override_model = (override.model or "").strip() if override is not None else ""
    ref = override_model or cfg.model or system.defaults.model
    deadline_s = cfg.llm_timeout
    # An agent that says nothing about reasoning inherits `defaults.reasoning`
    # (or the operator's system-wide replacement for it); an unset default
    # sends no reasoning kwargs at all. Resolved, not read raw: the
    # declaration may be a "${settings.path}" reference or an override.
    declared = cfg.resolved_reasoning()
    default = default_reasoning_override()
    if default is None:
        default = system.defaults.reasoning
    reasoning = declared if declared is not None else default
    if ref == "main":
        return make_llm(deadline_s=deadline_s, reasoning=reasoning)
    if ref == "coder":
        return make_coder_llm(deadline_s=deadline_s, reasoning=reasoning)
    if ref == "nir":
        # Long-form GOST authoring over a large evidence base. No dedicated
        # factory: make_llm already applies the reasoning and OpenRouter
        # provider kwargs, and this model needs no API key of its own.
        from CoScientist.config import get_settings

        nir_model = get_settings().llm.nir_model
        return make_llm(nir_model, deadline_s=deadline_s, reasoning=reasoning) \
            if nir_model else make_llm(deadline_s=deadline_s, reasoning=reasoning)
    return make_llm(ref, deadline_s=deadline_s, reasoning=reasoning)


def _resolve_tools(cfg: AgentConfig) -> List[ToolEntry]:
    """Resolve the agent's tool entries, dropping unavailable optional ones."""
    entries: List[ToolEntry] = []
    for key in cfg.tools:
        entry = REGISTRY.tool(key)
        if entry.factory() is None:
            if entry.optional:
                _logger.info("%s: optional tool %r not configured — skipped", cfg.name, key)
                continue
            raise ValueError(f"{cfg.name}: required tool {key!r} is not available")
        entries.append(entry)
    return entries


def _flatten(tool_obj) -> list:
    return list(tool_obj) if isinstance(tool_obj, list) else [tool_obj]


def _hitl_enabled() -> bool:
    from CoScientist.config import get_settings
    return get_settings().web.hitl_enabled


def _resolve_callback(name: str, expected_kind: str, ctx: PromptContext):
    entry = REGISTRY.callback(name)
    if entry.kind != expected_kind:
        raise ValueError(
            f"{ctx.config.name}: callback {name!r} is a {entry.kind} callback, "
            f"listed under {expected_kind}"
        )
    return entry.resolve(ctx)


def _callback_kwargs(cfg: AgentConfig, ctx: PromptContext) -> dict:
    kwargs = {}
    for kind, names in cfg.callbacks.items():
        if not names:
            continue
        resolved = [_resolve_callback(n, kind, ctx) for n in names]
        kwargs[f"{kind}_callback"] = resolved[0] if len(resolved) == 1 else resolved
    return kwargs


def _work_order_tool_names(
    cfg: AgentConfig, system: SystemConfig, tool_entries: List[ToolEntry]
) -> Optional[Set[str]]:
    """The tool names a Work Order may plan: the agent's documented tools plus
    its subordinate AgentTools. None when the surface is resolved at runtime (a
    placeholder doc such as "<dynamic MCP tools>") — names can't be checked."""
    docs = [d for e in tool_entries for d in e.resolved_docs()]
    if any(d.name.startswith("<") for d in docs):
        return None
    names = {d.name for d in docs}
    names |= {s.name for s in system.enabled_subordinates(cfg.name)}
    return names


def _attach_transient_error_refund(kwargs: dict) -> None:
    """First after_tool on every agent: a call that died on a dropped MCP
    session gives its attempt back to whichever limiter charged it."""
    from CoScientist.agents.callbacks.tool_callbacks import refund_transient_tool_error

    current = kwargs.get("after_tool_callback")
    rest = [] if current is None else list(current) if isinstance(current, list) else [current]
    kwargs["after_tool_callback"] = [refund_transient_tool_error] + rest


def _attach_work_order_callbacks(
    kwargs: dict, agent_name: str, internal_tools: List[str], step_review: bool = False
) -> None:
    """Reset the contract and enforce it FIRST: on agent start, before anything
    reads the state; before a tool, so a call the contract blocks never reaches
    the other callbacks (a WebSearchLimiter would count it against the quota).
    Link refs are not resolved yet then, which does not matter: the guard keys
    on tool names and shell verbs, not on URLs."""
    from CoScientist.hitl.work_order_guard import (
        make_reset_work_order,
        make_work_order_guard,
        make_work_report_fallback,
        make_work_step_journal,
    )

    def as_list(value) -> list:
        if value is None:
            return []
        return list(value) if isinstance(value, list) else [value]

    kwargs["before_agent_callback"] = (
        [make_reset_work_order(agent_name)] + as_list(kwargs.get("before_agent_callback"))
    )
    kwargs["before_tool_callback"] = (
        [make_work_order_guard(agent_name, internal_tools=internal_tools)] + as_list(kwargs.get("before_tool_callback"))
    )
    if step_review:
        # First after the tool: the journal records the answer as the tool gave
        # it, before any other callback could replace it.
        kwargs["after_tool_callback"] = (
            [make_work_step_journal(agent_name, internal_tools=internal_tools)]
            + as_list(kwargs.get("after_tool_callback"))
        )
    # Last after the agent: the other after_agent callbacks (e.g. collectors)
    # see the answer as the agent gave it; the human's verdict may replace it.
    kwargs["after_agent_callback"] = (
        as_list(kwargs.get("after_agent_callback")) + [make_work_report_fallback(agent_name)]
    )


def _render_instruction(cfg: AgentConfig, ctx: PromptContext) -> str:
    instruction = REGISTRY.prompt(cfg.prompt)(ctx)
    leftover = _PLACEHOLDER_RE.findall(instruction)
    if leftover:
        raise ValueError(
            f"{cfg.name}: prompt {cfg.prompt!r} left placeholders unfilled: {leftover}"
        )
    # The language rule, on every agent whose text a human can read, appended
    # here rather than written into each prompt. It used to be pasted into
    # three of the thirty-four, which is why the experiment executor — author
    # of the longest card the operator has to approve — wrote English into a
    # Russian study. At the END because a prompt is hundreds of lines of
    # English instructions, and a rule about language stated first and
    # contradicted by every line after it is a rule the model reads once.
    #
    # `internal:` agents are exempt: they are pipeline plumbing whose output is
    # JSON, ids and tool names, never prose, and is not shown in the UI.
    if not cfg.internal:
        instruction = instruction.rstrip("\n") + "\n\n" + _LANGUAGE_REQUIREMENT.strip("\n")
    # Empty placeholders (e.g. <<HITL>> when HITL is off) leave blank-line runs.
    return re.sub(r"\n{3,}", "\n\n", instruction).strip("\n") + "\n"


def _check_tool_consistency(cfg: AgentConfig, ctx: PromptContext, tools: list) -> None:
    """The attached function tools and the documented tool names must match.

    MCP toolsets resolve their tool surface at runtime — entries marked
    ``runtime_resolved`` are excluded (their docs are trusted as written).
    """
    documented: Set[str] = {
        d.name
        for e in ctx.tool_entries if not e.runtime_resolved
        for d in e.resolved_docs()
    }
    attached: Set[str] = set()
    for t in tools:
        name = getattr(t, "name", None) or getattr(t, "__name__", None)
        if name and not hasattr(t, "get_tools"):  # skip toolsets (runtime surface)
            attached.add(name)
    # AgentTools are documented through <<AGENTS>>, not <<TOOLS>>.
    attached -= {s.name for s in ctx.subordinates}
    missing_docs = attached - documented
    phantom_docs = documented - attached
    if missing_docs or phantom_docs:
        raise ValueError(
            f"{cfg.name}: prompt/tool mismatch — attached but undocumented: "
            f"{sorted(missing_docs)}; documented but not attached: {sorted(phantom_docs)}"
        )


def _build_llm_agent(
    cfg: AgentConfig,
    system: SystemConfig,
    built: Dict[str, BaseAgent],
    remote_subagents: bool,
) -> LlmAgent:
    tool_entries = _resolve_tools(cfg)
    hitl_attached = bool(cfg.hitl and _hitl_enabled())
    generic_hitl_attached = bool(hitl_attached and cfg.hitl_tools)
    work_order_attached = bool(cfg.work_order and hitl_attached)

    tools: list = []
    for entry in tool_entries:
        tools.extend(_flatten(entry.factory()))

    if work_order_attached:
        from CoScientist.hitl.work_order_tools import make_work_order_tools
        tools.extend(make_work_order_tools(
            cfg.name, _work_order_tool_names(cfg, system, tool_entries),
            internal_tools=system.internal_tools,
            step_review=cfg.work_order_step_review,
        ))

    if generic_hitl_attached:
        from CoScientist.hitl.tool import get_hitl_tools
        # Only the A2A ROOT can use the native pause: a pause inside a sub-agent
        # is swallowed by the parent's AgentTool (see get_hitl_tools).
        tools.extend(get_hitl_tools(a2a_root=bool(cfg.root)))
        tool_entries = tool_entries + [
            ToolEntry(key="hitl", factory=lambda: None, docs=HITL_TOOL_DOCS)
        ]
    if work_order_attached:
        tool_entries = tool_entries + [
            ToolEntry(key="work_order", factory=lambda: None, docs=WORK_ORDER_TOOL_DOCS)
        ]

    ctx = PromptContext(
        config=cfg,
        system=system,
        tool_entries=tool_entries,
        hitl_attached=generic_hitl_attached,
        work_order_attached=work_order_attached,
    )

    for sub in ctx.subordinates:
        tools.append(AgentTool(agent=_subordinate_instance(sub, built, remote_subagents)))

    _check_tool_consistency(cfg, ctx, tools)

    callbacks = _callback_kwargs(cfg, ctx)
    _attach_transient_error_refund(callbacks)
    if work_order_attached:
        _attach_work_order_callbacks(
            callbacks, cfg.name, system.internal_tools,
            step_review=cfg.work_order_step_review,
        )

    kwargs = dict(
        name=cfg.name,
        model=_resolve_model(cfg, system),
        description=cfg.description,
        tools=tools,
        **callbacks,
    )
    if cfg.prompt:
        kwargs["instruction"] = _render_instruction(cfg, ctx)
    if cfg.output_key:
        kwargs["output_key"] = cfg.output_key
    if cfg.include_contents:
        kwargs["include_contents"] = cfg.include_contents
    if cfg.mode:
        kwargs["mode"] = cfg.mode
    if cfg.output_schema:
        kwargs["output_schema"] = REGISTRY.output_schema(cfg.output_schema)
    if cfg.planner:
        kwargs["planner"] = REGISTRY.planner(cfg.planner)()
    kwargs.update(cfg.resolved_options())
    return LlmAgent(**kwargs)


def _subordinate_instance(
    sub: AgentConfig, built: Dict[str, BaseAgent], remote_subagents: bool
) -> BaseAgent:
    if remote_subagents and sub.a2a:
        from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
        from CoScientist.a2a.server import make_agent_card

        kwargs = {}
        from CoScientist.config import get_settings

        if get_settings().synapse.enabled:
            from CoScientist.a2a.synapse_tracing import make_remote_agent_config

            kwargs["config"] = make_remote_agent_config()

        return RemoteA2aAgent(
            name=sub.name,
            agent_card=make_agent_card(sub),
            description=sub.description,
            **kwargs,
        )
    return built[sub.name]


def _build_custom_agent(
    cfg: AgentConfig,
    system: SystemConfig,
    class_key: str,
    built: Optional[Dict[str, BaseAgent]] = None,
) -> BaseAgent:
    cls = REGISTRY.agent_class(class_key)
    kwargs = dict(name=cfg.name, description=cfg.description)
    if cfg.children and built is not None:
        kwargs["sub_agents"] = [
            built[c] for c in cfg.children if system.agent(c).is_enabled()
        ]
    if issubclass(cls, LlmAgent):
        tool_entries = _resolve_tools(cfg)
        hitl_attached = bool(cfg.hitl and _hitl_enabled())
        generic_hitl_attached = bool(hitl_attached and cfg.hitl_tools)
        tools = [t for e in tool_entries for t in _flatten(e.factory())]
        if generic_hitl_attached:
            from CoScientist.hitl.tool import get_hitl_tools
            tools.extend(get_hitl_tools(a2a_root=bool(cfg.root)))
            tool_entries = tool_entries + [
                ToolEntry(key="hitl", factory=lambda: None, docs=HITL_TOOL_DOCS)
            ]
        ctx = PromptContext(
            config=cfg,
            system=system,
            tool_entries=tool_entries,
            hitl_attached=generic_hitl_attached,
        )
        kwargs["model"] = _resolve_model(cfg, system)
        if tools:
            kwargs["tools"] = tools
        kwargs.update(_callback_kwargs(cfg, ctx))
        if cfg.prompt:
            kwargs["instruction"] = _render_instruction(cfg, ctx)
        if cfg.output_key:
            kwargs["output_key"] = cfg.output_key
        # Same wiring as plain LlmAgent — SessionAgent/custom planners honor
        # include_contents: none so ToolRetriever dumps do not pollute the prompt.
        if cfg.include_contents:
            kwargs["include_contents"] = cfg.include_contents
        if cfg.mode:
            kwargs["mode"] = cfg.mode
        if cfg.output_schema:
            kwargs["output_schema"] = REGISTRY.output_schema(cfg.output_schema)
        if cfg.planner:
            kwargs["planner"] = REGISTRY.planner(cfg.planner)()
        if hitl_attached:
            # Session-style agents take a review-loop handler instead of tools.
            from CoScientist.agents.common import hitl_handler
            kwargs["hitl_handler"] = hitl_handler
        if cfg.uses_critic():
            # An LLM critic reviews the agent's output inside that same loop,
            # independently of HITL (see agents/callbacks/critic.py).
            if "plan_critic" not in getattr(cls, "model_fields", {}):
                raise ValueError(
                    f"{cfg.name}: class {cls.__name__} declares no `plan_critic` "
                    f"field — it cannot run a critic review loop"
                )
            kwargs["plan_critic"] = make_plan_critic(ctx)
    elif cfg.uses_critic():
        raise ValueError(
            f"{cfg.name}: critic: needs an LlmAgent-based class, got {cls.__name__}"
        )
    kwargs.update(cfg.resolved_options())
    return cls(**kwargs)


def build_system(
    config: Optional[SystemConfig] = None,
    *,
    config_path: Optional[Path] = None,
    remote_subagents: bool = False,
) -> AgentSystem:
    """Assemble every agent declared in the config; return the full system."""
    if config is None:
        config = load_config(config_path) if config_path else get_config()
    load_plugins(config.plugins)

    built: Dict[str, BaseAgent] = {}
    for name in config.build_order():
        cfg = config.agent(name)
        if cfg.cls == "llm":
            agent = _build_llm_agent(cfg, config, built, remote_subagents)
        elif cfg.cls in COMPOSITE_CLASSES:
            cls = _COMPOSITE_AGENT_CLASSES[cfg.cls]
            # Workflow agents also honor before/after_agent callbacks (e.g. EM
            # ToolPreparer → assess_experiment_inventory_feasibility).
            sub_agents = [built[c] for c in cfg.children] if cfg.is_enabled() else []
            ctx = PromptContext(config=cfg, system=config)
            agent = cls(
                name=cfg.name,
                description=cfg.description,
                sub_agents=sub_agents,
                **_callback_kwargs(cfg, ctx),
                **cfg.resolved_options(),
            )
        else:  # custom:<key>
            agent = _build_custom_agent(
                cfg, config, cfg.cls.split(":", 1)[1], built
            )
        built[name] = agent

    return AgentSystem(config=config, agents=built)


def delegatable_agent_names() -> Set[str]:
    """Names of agents reachable via delegation (config-only; no agents built).

    Used by the execution-graph emitter to tell delegations apart from leaf
    tool calls.
    """
    return get_config().delegatable_names()


def load_config_cli() -> None:  # pragma: no cover — `python -m` helper
    """Validate the config and print a build summary (no LLM calls)."""
    config = get_config()
    print(f"OK: {len(config.agents)} agents, root={config.root.name}")
    if config.pipeline.pre or config.pipeline.post:
        print(f"  pipeline: pre={config.pipeline.pre} post={config.pipeline.post}")
    for name in config.build_order():
        cfg = config.agent(name)
        bits = [cfg.cls]
        if not cfg.is_enabled():
            bits.append("disabled")
        if cfg.tools:
            bits.append(f"tools={cfg.tools}")
        if cfg.subordinates:
            bits.append(f"subordinates={cfg.subordinates}")
        if cfg.children:
            bits.append(f"children={cfg.children}")
        if cfg.hitl:
            bits.append("hitl")
        if cfg.work_order:
            bits.append("work_order+step_review" if cfg.work_order_step_review else "work_order")
        if cfg.uses_critic():
            bits.append("critic")
        if cfg.a2a:
            bits.append(f"a2a={cfg.a2a.key}:{cfg.a2a.port}")
        print(f"  {name}: " + ", ".join(bits))


__all__ = [
    "AgentSystem",
    "build_system",
    "delegatable_agent_names",
    "load_config",
]
