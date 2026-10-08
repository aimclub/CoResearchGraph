"""Builds the agent system from ``system.yaml``.

The assembler walks the validated :class:`SystemConfig` in dependency order and
constructs every declared agent:

  * ``llm``         -> google.adk LlmAgent — model, prompt (rendered from the
                       agent's PromptContext), tools, subordinate AgentTools,
                       callbacks, HITL tools, output_key/schema, planner
  * ``sequential``  -> SequentialAgent over ``children``
  * ``parallel``    -> ParallelAgent over ``children``
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
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools.agent_tool import AgentTool

# Populate the registry (tools/callbacks/classes + prompt templates).
import CoScientist.assembly.bindings  # noqa: F401  (registration side effect)
import CoScientist.agents.prompts.templates  # noqa: F401  (registration side effect)

from CoScientist.assembly.bindings import HITL_TOOL_DOCS, make_plan_critic
from CoScientist.assembly.prompting import PromptContext
from CoScientist.assembly.registry import REGISTRY, ToolEntry
from CoScientist.assembly.schema import (
    AgentConfig,
    SystemConfig,
    get_config,
    load_config,
)

_logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"<<[A-Z_]+>>")


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
                name="ResearchPipeline",
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

    ref = cfg.model or system.defaults.model
    deadline_s = cfg.llm_timeout
    # An agent that says nothing about reasoning inherits `defaults.reasoning`;
    # an unset default sends no reasoning kwargs at all.
    reasoning = cfg.reasoning if cfg.reasoning is not None else system.defaults.reasoning
    if ref == "main":
        return make_llm(deadline_s=deadline_s, reasoning=reasoning)
    if ref == "coder":
        return make_coder_llm(deadline_s=deadline_s, reasoning=reasoning)
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


def _render_instruction(cfg: AgentConfig, ctx: PromptContext) -> str:
    instruction = REGISTRY.prompt(cfg.prompt)(ctx)
    leftover = _PLACEHOLDER_RE.findall(instruction)
    if leftover:
        raise ValueError(
            f"{cfg.name}: prompt {cfg.prompt!r} left placeholders unfilled: {leftover}"
        )
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

    tools: list = []
    for entry in tool_entries:
        tools.extend(_flatten(entry.factory()))

    if hitl_attached:
        from CoScientist.hitl.tool import get_hitl_tools
        # Only the A2A ROOT can use the native pause: a pause inside a sub-agent
        # is swallowed by the parent's AgentTool (see get_hitl_tools).
        tools.extend(get_hitl_tools(a2a_root=bool(cfg.root)))
        tool_entries = tool_entries + [
            ToolEntry(key="hitl", factory=lambda: None, docs=HITL_TOOL_DOCS)
        ]

    ctx = PromptContext(
        config=cfg,
        system=system,
        tool_entries=tool_entries,
        hitl_attached=hitl_attached,
    )

    for sub in ctx.subordinates:
        tools.append(AgentTool(agent=_subordinate_instance(sub, built, remote_subagents)))

    _check_tool_consistency(cfg, ctx, tools)

    kwargs = dict(
        name=cfg.name,
        model=_resolve_model(cfg, system),
        description=cfg.description,
        tools=tools,
        **_callback_kwargs(cfg, ctx),
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
        from CoScientist.a2a.config import AGENT_CARD_URLS

        return RemoteA2aAgent(
            name=sub.name,
            agent_card=AGENT_CARD_URLS[sub.a2a.key],
            description=sub.description,
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
        ctx = PromptContext(config=cfg, system=system, tool_entries=tool_entries)
        tools = [t for e in tool_entries for t in _flatten(e.factory())]
        kwargs["model"] = _resolve_model(cfg, system)
        if tools:
            kwargs["tools"] = tools
        kwargs.update(_callback_kwargs(cfg, ctx))
        if cfg.prompt:
            kwargs["instruction"] = _render_instruction(cfg, ctx)
        if cfg.output_key:
            kwargs["output_key"] = cfg.output_key
        if cfg.output_schema:
            kwargs["output_schema"] = REGISTRY.output_schema(cfg.output_schema)
        if cfg.planner:
            kwargs["planner"] = REGISTRY.planner(cfg.planner)()
        if cfg.hitl:
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

    built: Dict[str, BaseAgent] = {}
    for name in config.build_order():
        cfg = config.agent(name)
        if cfg.cls == "llm":
            agent = _build_llm_agent(cfg, config, built, remote_subagents)
        elif cfg.cls in ("sequential", "parallel"):
            cls = SequentialAgent if cfg.cls == "sequential" else ParallelAgent
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
