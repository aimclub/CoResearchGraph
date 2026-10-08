"""Pydantic schema + loader for ``system.yaml``.

The YAML declares every agent of the system in one place. Per agent:

  class:        llm | sequential | parallel | custom:<registered name>
  enabled:      bool, or "${settings.path}" resolved against app settings —
                a disabled agent is still BUILT (so it can be served standalone
                over A2A) but is not attached to / advertised by its parents
  model:        "main" | "coder" | a literal litellm model string
  reasoning:    model "thinking" for this agent — false/"off" to switch it off,
                or "minimal"|"low"|"medium"|"high"; unset inherits defaults
  prompt:       name of a registered prompt template
  tools:        registered tool names
  subordinates: agents attached as AgentTool (and rendered into <<AGENTS>>/<<ROUTING>>)
  children:     composite children (sequential/parallel execution order)
  callbacks:    {before_model|after_model|before_tool|after_tool|before_agent|after_agent: [names]}
  hitl:         whether the agent uses human-in-the-loop (tools + prompt section
                for llm agents, review-loop handler for session agents)
  critic:       an LLM critic reviews the agent's output once and it rewrites
                on request (session agents only; bool or "${settings.path}")
  report_output: the agent's final answer is a deliverable — show it in the chat
  output_key / output_schema / planner / options: passthrough constructor config
                (an ``options`` value may be "${settings.path}" too)
  a2a:          how the agent is exposed as an A2A service (key, port, skill, env)

A config may also start with ``extends: <name-or-path>``, inheriting another
config and overriding only the agents and fields it names — so a variant of the
system (a limited profile, a deployment with one agent off) is a short overlay
rather than a copy that drifts from the original. See :func:`_merge_raw`.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from CoScientist.config import get_settings

CONFIG_DIR = Path(__file__).resolve().parent.parent / "agents"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "system.yaml"

# Env var selecting an alternative system profile for the whole process.
# Accepts a bare profile name ("microfluidics" -> CoScientist/agents/
# microfluidics.yaml) or a filesystem path to a YAML file. Every entry point
# that builds the system through get_config() (CLI, web server, A2A serving)
# honours it, so one deployment can run a differently-shaped CoScientist
# without touching the default system.yaml.
CONFIG_ENV_VAR = "COSCIENTIST_CONFIG"


def resolve_config_path(ref: Optional[str] = None) -> Path:
    """Resolve a config reference: explicit ref, $COSCIENTIST_CONFIG, or default."""
    ref = ref or os.environ.get(CONFIG_ENV_VAR)
    if not ref:
        return DEFAULT_CONFIG_PATH
    path = Path(ref)
    if path.suffix in (".yaml", ".yml"):
        return path
    return CONFIG_DIR / f"{ref}.yaml"


def _is_setting_ref(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.strip().startswith("${")
        and value.strip().endswith("}")
    )


def _setting_value(ref: str) -> Any:
    """The live value behind a "${dotted.settings.path}" reference."""
    obj: Any = get_settings()
    for part in ref.strip()[2:-1].split("."):
        obj = getattr(obj, part)
    return obj


def _resolve_setting_ref(value: Union[bool, str]) -> bool:
    """Resolve an ``enabled`` value: a bool, or "${dotted.settings.path}"."""
    if isinstance(value, bool):
        return value
    if not _is_setting_ref(value):
        raise ValueError(
            f"enabled must be a bool or '${{settings.path}}', got {value!r}"
        )
    return bool(_setting_value(value))


# Accepted ``reasoning:`` values (mirrors CoScientist.agents.common, which
# turns them into provider kwargs — validated here so a typo fails at config
# load, not on the first model call).
REASONING_EFFORTS = ("minimal", "low", "medium", "high")
REASONING_OFF = ("off", "none", "disabled")


def _validate_reasoning(value: Any) -> Any:
    """A ``reasoning:`` declaration: unset, a bool, or an effort/off keyword."""
    if value is None or isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in REASONING_OFF or normalized in REASONING_EFFORTS:
        return normalized
    raise ValueError(
        f"reasoning must be a bool, one of {REASONING_OFF} or {REASONING_EFFORTS}, "
        f"got {value!r}"
    )


class SkillConfig(BaseModel):
    """One A2A AgentSkill advertised on the agent card."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)


class A2AConfig(BaseModel):
    """How the agent is exposed as a standalone A2A service."""

    model_config = ConfigDict(extra="forbid")

    key: str  # snake key: env prefix ("<KEY>_PORT") and serve-module argument
    port: int  # default port; overridable via the <KEY>_PORT env var
    skill: SkillConfig
    # Env defaults applied (setdefault) before the serving process builds the
    # system — for settings that must exist before tool modules import.
    env: Dict[str, str] = Field(default_factory=dict)


class CallbacksConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    before_model: List[str] = Field(default_factory=list)
    after_model: List[str] = Field(default_factory=list)
    before_tool: List[str] = Field(default_factory=list)
    after_tool: List[str] = Field(default_factory=list)
    before_agent: List[str] = Field(default_factory=list)
    after_agent: List[str] = Field(default_factory=list)

    def items(self):
        return self.model_dump().items()


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = ""  # filled from the mapping key
    cls: str = Field("llm", alias="class")
    enabled: Union[bool, str] = True
    root: bool = False
    model: Optional[str] = None
    llm_timeout: Optional[float] = None
    # Model reasoning ("thinking") for THIS agent: false / "off" to switch it
    # off entirely (fastest), or "minimal"|"low"|"medium"|"high" to turn it
    # down. Unset inherits `defaults.reasoning`, and an unset default leaves the
    # provider's own behaviour alone. Only hybrid models can be silenced — one
    # that always reasons (deepseek-r1, o-series) ignores the request.
    reasoning: Optional[Union[bool, str]] = None
    description: str = ""
    # How a PARENT's prompt routes work to this agent (one routing bullet).
    routing: str = ""
    # How the planner's roster describes this agent (defaults to description).
    planning: str = ""
    prompt: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    subordinates: List[str] = Field(default_factory=list)
    children: List[str] = Field(default_factory=list)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    hitl: bool = False
    # An LLM critic reviews my proposed output once before it is accepted, and
    # I rewrite it if the critic asks (session-style custom agents only — the
    # review loop is theirs). Independent of the orchestrator's pre/post-action
    # critic callbacks. Accepts "${settings.path}" like `enabled`.
    critic: Union[bool, str] = False
    # My final answer is a deliverable in its own right (hypotheses, a research
    # summary): report it to the chat instead of leaving it buried in the
    # delegation's function_response. See logging/agent_output.py.
    report_output: bool = False
    include_contents: Optional[str] = "default"
    mode: Optional[str] = None
    output_key: Optional[str] = None
    output_schema: Optional[str] = None
    planner: Optional[str] = None
    # Extra constructor kwargs for custom agent classes (e.g. plan_file_path).
    options: Dict[str, Any] = Field(default_factory=dict)
    a2a: Optional[A2AConfig] = None

    _check_reasoning = field_validator("reasoning")(
        classmethod(lambda cls, v: _validate_reasoning(v))
    )

    @field_validator("cls")
    @classmethod
    def _known_class(cls, v: str) -> str:
        if v in ("llm", "sequential", "parallel") or v.startswith("custom:"):
            return v
        raise ValueError(
            f"class must be llm | sequential | parallel | custom:<name>, got {v!r}"
        )

    @model_validator(mode="after")
    def _shape(self) -> "AgentConfig":
        composite = self.cls in ("sequential", "parallel")
        if composite:
            if not self.children:
                raise ValueError(f"{self.cls} agent needs non-empty children")
            for forbidden in ("tools", "subordinates", "prompt", "model", "llm_timeout"):
                if getattr(self, forbidden):
                    raise ValueError(
                        f"{self.cls} agent cannot have {forbidden} (got {getattr(self, forbidden)!r})"
                    )
            # Checked separately: `reasoning: false` is a real declaration but
            # a falsy one, so the truthiness loop above would let it through.
            if self.reasoning is not None:
                raise ValueError(
                    f"{self.cls} agent cannot have reasoning (it has no model of its own)"
                )
        elif self.children and not self.cls.startswith("custom:"):
            # custom: classes may take children too (e.g. an executor switch that
            # runs exactly one of them); everything else is a leaf.
            raise ValueError(f"{self.cls} agent cannot have children")
        return self

    def is_enabled(self) -> bool:
        return _resolve_setting_ref(self.enabled)

    def uses_critic(self) -> bool:
        return _resolve_setting_ref(self.critic)

    def resolved_options(self) -> Dict[str, Any]:
        """``options`` with every "${settings.path}" value replaced by the value.

        Lets a constructor kwarg follow a runtime setting the way ``enabled``
        and ``critic`` do — e.g. the plan critic's round budget, which the web
        UI writes to settings — instead of being frozen in the YAML. Values
        keep their own type (int stays int); only strings are inspected.
        """
        return {
            key: _setting_value(value) if _is_setting_ref(value) else value
            for key, value in self.options.items()
        }


class DefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "main"
    # System-wide reasoning default; per-agent `reasoning:` overrides it.
    reasoning: Optional[Union[bool, str]] = None

    _check_reasoning = field_validator("reasoning")(
        classmethod(lambda cls, v: _validate_reasoning(v))
    )


class PipelineConfig(BaseModel):
    """System lifecycle flow around the root orchestrator.

    ``pre`` stages run (in order) BEFORE the root agent, ``post`` stages run
    AFTER it — each as its own pass over the SAME session, so state flows
    between them. Stage agents are ordinary agents declared in ``agents``; they
    need not be attached to any parent (the assembler builds every declared
    agent regardless). This keeps the delegation tree (root + subordinates)
    separate from the run lifecycle instead of forcing flow through a
    ``SequentialAgent`` root.
    """

    model_config = ConfigDict(extra="forbid")

    pre: List[str] = Field(default_factory=list)
    post: List[str] = Field(default_factory=list)

    def stage_names(self) -> List[str]:
        return list(self.pre) + list(self.post)


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    agents: Dict[str, AgentConfig]

    @model_validator(mode="after")
    def _validate_graph(self) -> "SystemConfig":
        for key, agent in self.agents.items():
            if agent.name and agent.name != key:
                raise ValueError(f"Agent key {key!r} != name {agent.name!r}")
            agent.name = key

        roots = [a.name for a in self.agents.values() if a.root]
        if len(roots) != 1:
            raise ValueError(f"Exactly one agent must have root: true, got {roots}")

        for stage in self.pipeline.stage_names():
            if stage not in self.agents:
                raise ValueError(f"pipeline references unknown agent {stage!r}")
            if stage == roots[0]:
                raise ValueError(
                    f"pipeline stage {stage!r} is the root — the root runs on its "
                    "own, do not list it as a pre/post stage"
                )

        for agent in self.agents.values():
            # The critic's review→revise round needs an agent that can re-run
            # itself; only the session-style custom classes have that loop.
            if agent.critic and not agent.cls.startswith("custom:"):
                raise ValueError(
                    f"{agent.name}: critic: is supported only by custom "
                    f"session agents, not by a {agent.cls!r} agent"
                )
            for ref in agent.subordinates + agent.children:
                if ref not in self.agents:
                    raise ValueError(f"{agent.name}: unknown agent reference {ref!r}")
            dupes = {r for r in agent.subordinates if agent.subordinates.count(r) > 1}
            if dupes:
                raise ValueError(f"{agent.name}: duplicate subordinates {sorted(dupes)}")

        # No cycles through children/subordinates (also guarantees a build order).
        self.build_order()

        keys = [a.a2a.key for a in self.agents.values() if a.a2a]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate a2a keys: {sorted(keys)}")
        ports = [a.a2a.port for a in self.agents.values() if a.a2a]
        if len(ports) != len(set(ports)):
            raise ValueError(f"Duplicate a2a ports: {sorted(ports)}")
        return self

    # ── queries ──────────────────────────────────────────────────────────────
    @property
    def root(self) -> AgentConfig:
        return next(a for a in self.agents.values() if a.root)

    def agent(self, name: str) -> AgentConfig:
        if name not in self.agents:
            raise KeyError(f"Unknown agent {name!r}")
        return self.agents[name]

    def deps(self, name: str) -> List[str]:
        agent = self.agent(name)
        return agent.children + agent.subordinates

    def build_order(self) -> List[str]:
        """Dependency-first topological order over children + subordinates."""
        order: List[str] = []
        state: Dict[str, int] = {}  # 0 visiting, 1 done

        def visit(name: str, chain: tuple) -> None:
            if state.get(name) == 1:
                return
            if state.get(name) == 0:
                cycle = " -> ".join(chain + (name,))
                raise ValueError(f"Agent dependency cycle: {cycle}")
            state[name] = 0
            for dep in self.deps(name):
                visit(dep, chain + (name,))
            state[name] = 1
            order.append(name)

        for name in self.agents:
            visit(name, ())
        return order

    def enabled_subordinates(self, name: str) -> List[AgentConfig]:
        return [
            self.agent(s) for s in self.agent(name).subordinates
            if self.agent(s).is_enabled()
        ]

    def parents_of(self, name: str) -> List[AgentConfig]:
        return [
            a for a in self.agents.values()
            if name in a.subordinates or name in a.children
        ]

    def agent_hierarchy_map(self) -> Dict[str, Any]:
        """Return the complete static agent hierarchy (child -> parent and parent -> children)."""
        parents: Dict[str, str] = {}
        children: Dict[str, List[str]] = {}

        for a in self.agents.values():
            direct_children = list(a.subordinates) + list(a.children)
            if direct_children:
                children[a.name] = direct_children
            for child_name in direct_children:
                parents[child_name] = a.name

        # Pipeline pre/post stages belong to the main orchestration lifecycle
        root_name = self.root.name if hasattr(self, "root") and self.root else "OrchestratorAgent"
        for pre in self.pipeline.pre:
            if pre not in parents:
                parents[pre] = root_name
        for post in self.pipeline.post:
            if post not in parents:
                parents[post] = root_name

        return {
            "parents": parents,
            "children": children,
            "root": root_name,
            "pipeline_pre": list(self.pipeline.pre),
            "pipeline_post": list(self.pipeline.post),
        }

    def delegatable_names(self) -> set:
        """Names of every agent that some agent delegates to via AgentTool."""
        return {s for a in self.agents.values() for s in a.subordinates}

    def reported_output_agents(self) -> frozenset:
        """Enabled agents whose final answer is shown in the chat."""
        return frozenset(
            a.name for a in self.agents.values()
            if a.report_output and a.is_enabled()
        )

    def a2a_agents(self) -> List[AgentConfig]:
        return [a for a in self.agents.values() if a.a2a]

    def a2a_agent_by_key(self, key: str) -> AgentConfig:
        for a in self.agents.values():
            if a.a2a and a.a2a.key == key:
                return a
        known = ", ".join(sorted(x.a2a.key for x in self.a2a_agents()))
        raise KeyError(f"No agent with a2a key {key!r}. Known: {known}")


def _merge_raw(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Lay an overlay config's raw dict over a base's.

    ``defaults`` merges shallowly. ``agents`` merges per agent — a named
    agent's fields update the base agent's, so a profile can flip only
    ``enabled`` without re-declaring the agent, and an agent the base does not
    have is added whole. Every other top-level section (``pipeline``, and
    anything added later) is taken from the overlay when it declares one, so a
    new section can never be silently dropped on the way through an overlay.
    """
    merged: Dict[str, Any] = {
        **base,
        **{k: v for k, v in overlay.items() if k not in ("defaults", "agents")},
    }
    if "defaults" in overlay:
        merged["defaults"] = {**(base.get("defaults") or {}), **overlay["defaults"]}
    agents = dict(base.get("agents") or {})
    for name, cfg in (overlay.get("agents") or {}).items():
        if isinstance(cfg, dict) and isinstance(agents.get(name), dict):
            agents[name] = {**agents[name], **cfg}
        else:
            agents[name] = cfg
    merged["agents"] = agents
    return merged


def _load_raw(path: Path, _seen: frozenset = frozenset()) -> Dict[str, Any]:
    """Load a config's raw dict, resolving an optional ``extends`` overlay.

    A profile sets ``extends: <name-or-path>`` to inherit another config and
    override only what it names (see :func:`_merge_raw`) instead of copying the
    whole system — a copy starts drifting from the original the day it is made.
    ``extends`` chains are followed; a cycle is an error.
    """
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Config 'extends' cycle involving {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base_ref = raw.pop("extends", None)
    if base_ref is None:
        return raw
    base = _load_raw(resolve_config_path(str(base_ref)), _seen | {path})
    return _merge_raw(base, raw)


def load_config(path: Optional[Path] = None) -> SystemConfig:
    path = Path(path) if path else resolve_config_path()
    return SystemConfig.model_validate(_load_raw(path))


@lru_cache(maxsize=1)
def get_config() -> SystemConfig:
    """The process-wide system config ($COSCIENTIST_CONFIG or the default),
    loaded once per process."""
    return load_config()


__all__ = [
    "A2AConfig",
    "AgentConfig",
    "CallbacksConfig",
    "CONFIG_ENV_VAR",
    "DEFAULT_CONFIG_PATH",
    "DefaultsConfig",
    "PipelineConfig",
    "SkillConfig",
    "SystemConfig",
    "get_config",
    "load_config",
    "resolve_config_path",
]
