"""Pydantic schema + loader for ``system.yaml``.

The YAML declares every agent of the system in one place. Per agent:

  class:        llm | sequential | parallel | loop | custom:<registered name>
                (`loop` repeats its children until a child escalates; cap it
                with options.max_iterations)
  enabled:      bool, or "${settings.path}" resolved against app settings —
                a disabled agent is still BUILT (so it can be served standalone
                over A2A) but is not attached to / advertised by its parents
  model:        "main" | "coder" | "nir" | a literal litellm model string
  reasoning:    model "thinking" for this agent — false/"off" to switch it off,
                or "minimal"|"low"|"medium"|"high"; unset inherits defaults
  prompt:       name of a registered prompt template
  tools:        registered tool names
  subordinates: agents attached as AgentTool (and rendered into <<AGENTS>>/<<ROUTING>>)
  children:     composite children (sequential/parallel execution order)
  callbacks:    {before_model|after_model|before_tool|after_tool|before_agent|after_agent: [names]}
  hitl:         whether the agent uses human-in-the-loop (tools + prompt section
                for llm agents, review-loop handler for session agents)
  work_order:   before acting, the agent declares a Work Order (goal, assumptions,
                steps, tools, side effects) for the human to review, and
                a guard keeps it inside the approved contract (llm agents with
                hitl only; see CoScientist/hitl/work_order.py)
  work_order_step_review: every finished step of the Work Order goes before the
                human — what was sent, expected and found (needs work_order)
  critic:       an LLM critic reviews the agent's output once and it rewrites
                on request (session agents only; bool or "${settings.path}")
  report_output: the agent's final answer is a deliverable — show it in the chat
  internal:     plumbing agent (pipeline stage, composite wrapper) — hidden in the web UI
  output_key / output_schema / planner / options: passthrough constructor config
                (an ``options`` value may be "${settings.path}" too)
  a2a:          how the agent is exposed as an A2A service (key, port, skill, env)

A config may also start with ``extends: <name-or-path>``, inheriting another
config and overriding only the agents and fields it names — so a variant of the
system (a limited profile, a deployment with one agent off) is a short overlay
rather than a copy that drifts from the original. See :func:`_merge_raw`.

A profile that brings its own prompts, tools and callbacks (a case package
such as ``CoScientist/microfluidics``) lists them under ``plugins:`` — modules
imported when the config is loaded, which register those names in the
assembly registry. The core never imports a case package itself.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from CoScientist.config import get_settings

_log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACKAGE_DIR / "agents"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "system.yaml"

# Env var selecting an alternative system profile for the whole process.
# Accepts a bare profile name or a filesystem path to a YAML file. A name is
# looked up in CoScientist/agents/ first ("experiments" -> agents/
# experiments.yaml), then as a case package that carries its own profile
# ("microfluidics" -> CoScientist/microfluidics/microfluidics.yaml). Every entry point
# that builds the system through get_config() (CLI, web server, A2A serving)
# honours it, so one deployment can run a differently-shaped CoScientist
# without touching the default system.yaml.
CONFIG_ENV_VAR = "COSCIENTIST_CONFIG"

# Classes that only sequence `children` and carry no prompt/tools of their own.
COMPOSITE_CLASSES = ("sequential", "parallel", "loop")

# Name of the SequentialAgent the assembler wraps around pipeline.pre + root +
# pipeline.post. It is not declared in YAML, so it cannot carry `internal:`.
PIPELINE_ROOT_NAME = "ResearchPipeline"


def resolve_config_path(ref: Optional[str] = None) -> Path:
    """Resolve a config reference: explicit ref, $COSCIENTIST_CONFIG, or default."""
    ref = ref or os.environ.get(CONFIG_ENV_VAR)
    if not ref:
        return DEFAULT_CONFIG_PATH
    path = Path(ref)
    if path.suffix in (".yaml", ".yml"):
        return path
    local = CONFIG_DIR / f"{ref}.yaml"
    packaged = PACKAGE_DIR / ref / f"{ref}.yaml"
    if not local.is_file() and packaged.is_file():
        return packaged
    return local


def profile_paths() -> List[Path]:
    """Every named profile: the configs in agents/ and the case packages'."""
    return sorted(CONFIG_DIR.glob("*.yaml")) + sorted(
        p for p in PACKAGE_DIR.glob("*/*.yaml") if p.stem == p.parent.name
    )


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


def normalize_reasoning(value: Any) -> Optional[Union[bool, str]]:
    """A literal reasoning value in the ``reasoning:`` vocabulary, or None.

    None for anything the vocabulary does not accept (including "" and a
    "${settings.path}" reference), so a caller can treat it as "unset".
    """
    if value is None or isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in REASONING_OFF or normalized in REASONING_EFFORTS:
        return normalized
    return None


# Agents whose presence the start mode decides (agents.build_for_mode patches
# their `enabled`/`root`): an operator override of `enabled` must not fight it.
MODE_CONTROLLED_AGENTS = frozenset({"PlanningPipelineAgent", "InitAgent", "PlannerAgent"})


def agent_override(name: Optional[str]):
    """The operator's override for one agent (settings.agents.overrides), or None."""
    if not name:
        return None
    try:
        return get_settings().agents.overrides.get(name)
    except Exception:  # noqa: BLE001 — a settings object without the block
        return None


def agent_limit(name: Optional[str], default: int) -> int:
    """The operator's tool budget for one agent (override ``limit``), else ``default``."""
    override = agent_override(name)
    value = getattr(override, "limit", None)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else int(default)


def default_reasoning_override() -> Optional[Union[bool, str]]:
    """settings.agents.default_reasoning, normalized; None leaves the profile's."""
    try:
        raw = get_settings().agents.default_reasoning
    except Exception:  # noqa: BLE001
        return None
    value = normalize_reasoning(raw)
    if raw not in (None, "") and value is None:
        _log.warning("AGENTS__DEFAULT_REASONING=%r is not one of %s or %s; ignored",
                     raw, REASONING_OFF, REASONING_EFFORTS)
    return value


def _validate_reasoning(value: Any) -> Any:
    """A ``reasoning:`` declaration: unset, a bool, an effort/off keyword, or a
    "${settings.path}" reference resolved at assembly time.

    A reference is kept verbatim here: what it points at is read when the model
    is built, so one run can turn an agent's thinking down without the YAML's
    default moving. `resolved_reasoning` validates whatever comes back.
    """
    if value is None or isinstance(value, bool):
        return value
    if _is_setting_ref(value):
        return str(value).strip()
    normalized = str(value).strip().lower()
    if normalized in REASONING_OFF or normalized in REASONING_EFFORTS:
        return normalized
    raise ValueError(
        f"reasoning must be a bool, one of {REASONING_OFF} or {REASONING_EFFORTS}, "
        f"a '${{settings.path}}' reference, got {value!r}"
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
    # Short human name, e.g. «Техническое задание» — the stage label in the web
    # status indicator of a linear pipeline (falls back to the agent's name).
    title: str = ""
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
    # Dedicated review agents need a handler for their internal review loop,
    # but must not expose generic request_approval/request_selection tools to
    # the model as a second, unvalidated approval path.
    hitl_tools: bool = True
    # Declare a Work Order before acting; a guard enforces it (needs hitl).
    work_order: bool = False
    # Each finished Work Order step is reviewed by the human: sent / expected /
    # found, with the calls the system recorded for it (needs work_order).
    work_order_step_review: bool = False
    # An LLM critic reviews my proposed output once before it is accepted, and
    # I rewrite it if the critic asks (session-style custom agents only — the
    # review loop is theirs). Independent of the orchestrator's pre/post-action
    # critic callbacks. Accepts "${settings.path}" like `enabled`.
    critic: Union[bool, str] = False
    # My final answer is a deliverable in its own right (hypotheses, a research
    # summary): report it to the chat instead of leaving it buried in the
    # delegation's function_response. See logging/agent_output.py.
    report_output: bool = False
    # Plumbing, not a participant the user reasons about (a composite wrapper,
    # a tool-pipeline stage): the web UI hides it from the activity rail and
    # the agent tree.
    internal: bool = False
    # Presentation/control policy only.  A required agent remains governed by
    # the normal assembler semantics, but the session UI must not let an
    # operator remove it from the pipeline.  Keeping this declarative lets
    # domain profiles protect their own terminal or execution stages without
    # teaching the web layer their names.
    required_for_pipeline: bool = False
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
        if v in COMPOSITE_CLASSES or v == "llm" or v.startswith("custom:"):
            return v
        raise ValueError(
            f"class must be llm | {' | '.join(COMPOSITE_CLASSES)} | custom:<name>, "
            f"got {v!r}"
        )

    @model_validator(mode="after")
    def _shape(self) -> "AgentConfig":
        composite = self.cls in COMPOSITE_CLASSES
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
        if self.work_order and (self.cls != "llm" or not self.hitl):
            # The contract is declared through tools and reviewed through the
            # HITL channel: only a plain llm agent with hitl has both.
            raise ValueError("work_order needs class: llm and hitl: true")
        if self.work_order_step_review and not self.work_order:
            raise ValueError("work_order_step_review needs work_order: true")
        return self

    def enabled_overridable(self) -> bool:
        """Whether an ``enabled`` override (settings.agents.overrides) applies.

        Not the root (the run has no other entry point), not plumbing
        (`internal`: a composite's stage, invisible to the operator), not an
        agent the start mode attaches or removes on its own, and not one whose
        ``enabled`` is a "${settings.path}" reference: that setting is its
        switch, and the code that reads it at run time (the medical and FEDOT
        routes, the research-frame seed) would never see an override.
        """
        return not (self.root or self.internal or self.name in MODE_CONTROLLED_AGENTS
                    or _is_setting_ref(self.enabled))

    def declared_enabled(self) -> bool:
        """``enabled`` as system.yaml (and the settings it references) says."""
        return _resolve_setting_ref(self.enabled)

    def is_enabled(self) -> bool:
        override = agent_override(self.name)
        if override is not None and override.enabled is not None and self.enabled_overridable():
            return bool(override.enabled)
        return self.declared_enabled()

    def resolved_reasoning(self) -> Optional[Union[bool, str]]:
        """The operator's override, else ``reasoning`` with a "${settings.path}"
        reference read off settings.

        A reference that resolves to something `reasoning:` does not accept is
        treated as UNSET, with a warning: an agent then inherits
        `defaults.reasoning` and the run goes on. Raising here would take the
        whole system down over one mistyped environment variable, and this dial
        is an optimisation, not a correctness switch.
        """
        override = agent_override(self.name)
        if override is not None and override.reasoning not in (None, ""):
            value = normalize_reasoning(override.reasoning)
            if value is not None:
                return value
            _log.warning("reasoning override %r for %s is not one of %s or %s; ignored",
                         override.reasoning, self.name, REASONING_OFF, REASONING_EFFORTS)
        return self.declared_reasoning()

    def declared_reasoning(self) -> Optional[Union[bool, str]]:
        """``reasoning`` as system.yaml declares it, references resolved."""
        value = self.reasoning
        if not _is_setting_ref(value):
            return value
        try:
            resolved = _setting_value(value)
        except Exception as exc:  # noqa: BLE001
            _log.warning("reasoning %s for %s is unreadable (%s); "
                         "inheriting the default", value, self.name, exc)
            return None
        if resolved is None or isinstance(resolved, bool):
            return resolved
        normalized = str(resolved).strip().lower()
        if normalized in REASONING_OFF or normalized in REASONING_EFFORTS:
            return normalized
        _log.warning("reasoning %s for %s resolved to %r, which is not one of "
                     "%s or %s; inheriting the default", value, self.name,
                     resolved, REASONING_OFF, REASONING_EFFORTS)
        return None

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
    # The root runs its subordinates one after another, in the declared order
    # (e.g. the microfluidics modules), so "stage k of N" is a meaningful thing
    # to show the user — see SystemConfig.linear_stages(). Off for an
    # orchestrator that picks subordinates freely.
    linear: bool = False

    def stage_names(self) -> List[str]:
        return list(self.pre) + list(self.post)


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    # Tool names (as the model calls them) that serve the system rather than the
    # task: a Work Order allows them without declaring, and the web card never
    # shows them — even when the agent lists them anyway.
    internal_tools: List[str] = Field(default_factory=list)
    # Modules imported before the system is built; they register the profile's
    # own prompts / tools / callbacks / classes (see the module docstring).
    plugins: List[str] = Field(default_factory=list)
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
            # A disabled agent is never attached, so it is nobody's parent at
            # runtime (PlanningPipelineAgent would otherwise claim the root).
            if not a.is_enabled():
                continue
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

    def linear_stages(self) -> List[Dict[str, Any]]:
        """The run as a list of stages, when ``pipeline.linear`` says it is one.

        Stages, in order: the ``pre`` stages, then the root's enabled
        subordinates with every sequential composite unrolled into its children
        (a module is not a stage, its steps are), then the ``post`` stages.
        Anything else — an LLM agent, a loop — is one stage, and every agent in
        its subtree (children and subordinates) counts as that stage's work.
        Each stage is ``{"agent", "title", "members"}``; [] when not linear.
        """
        if not self.pipeline.linear:
            return []

        def unroll(name: str) -> List[str]:
            agent = self.agent(name)
            if not agent.is_enabled():
                return []
            if agent.cls == "sequential":
                return [s for child in agent.children for s in unroll(child)]
            return [name]

        def subtree(name: str, seen: set) -> List[str]:
            if name in seen:
                return []
            seen.add(name)
            out = [name]
            for dep in self.deps(name):
                out.extend(subtree(dep, seen))
            return out

        names = list(self.pipeline.pre)
        for sub in self.root.subordinates:
            names.extend(unroll(sub))
        names.extend(self.pipeline.post)

        claimed: set = set()
        stages = []
        for name in names:
            agent = self.agent(name)
            # An agent shared by several stages counts toward the first one.
            members = [m for m in subtree(name, set()) if m not in claimed]
            claimed.update(members)
            stages.append({
                "agent": name,
                "title": agent.title or name,
                "members": members,
            })
        return stages

    def delegatable_names(self) -> set:
        """Names of every agent that some agent delegates to via AgentTool."""
        return {s for a in self.agents.values() for s in a.subordinates}

    def reported_output_agents(self) -> frozenset:
        """Enabled agents whose final answer is shown in the chat."""
        return frozenset(
            a.name for a in self.agents.values()
            if a.report_output and a.is_enabled()
        )

    def internal_agent_names(self) -> frozenset:
        """Agents the web UI hides: those marked ``internal`` plus the
        synthesized pipeline wrapper, which has no config entry of its own."""
        return frozenset(
            {a.name for a in self.agents.values() if a.internal}
            | {PIPELINE_ROOT_NAME}
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


def load_plugins(modules: List[str]) -> None:
    """Import a profile's plugin modules (idempotent: Python caches modules)."""
    for module in modules:
        import_module(module)


def load_config(path: Optional[Path] = None) -> SystemConfig:
    path = Path(path) if path else resolve_config_path()
    config = SystemConfig.model_validate(_load_raw(path))
    load_plugins(config.plugins)
    return config


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
    "load_plugins",
    "profile_paths",
    "resolve_config_path",
]
