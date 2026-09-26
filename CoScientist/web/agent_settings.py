"""The Agents section of the web settings: what each agent is, and what the
operator changed about it.

system.yaml stays the declaration; ``settings.agents`` holds the operator's
overrides (on/off, reasoning, model) and the assembler reads them when a
session's agent tree is built. This module is the web side of that: the
catalog the settings modal lists (read off a fresh load of the profile, with
start-mode agents adjusted to the effective runtime state), and the read/write
of the overrides in the ``appSettings`` shape.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from CoScientist.assembly.schema import (
    MODE_CONTROLLED_AGENTS,
    REASONING_EFFORTS,
    AgentConfig,
    SystemConfig,
    _is_setting_ref,
    load_config,
    normalize_reasoning,
)
from CoScientist.config import get_settings
from CoScientist.config.settings import AgentOverride
from CoScientist.web.agent_configuration_policy import (
    ENABLED_SETTING_FIELDS,
    agent_control_policy,
    public_control_fields,
    validate_agent_enabled_change,
)

_log = logging.getLogger(__name__)

# What the modal offers for reasoning, in order. "" = as declared.
REASONING_CHOICES = ["off", *REASONING_EFFORTS]
MODEL_ALIASES = ("main", "coder", "nir")


def _has_model(cfg: AgentConfig) -> bool:
    """Whether the agent calls a model of its own (so reasoning/model apply)."""
    if cfg.cls == "llm":
        return True
    if not cfg.cls.startswith("custom:"):
        return False
    try:
        from google.adk.agents import LlmAgent

        from CoScientist.assembly.registry import REGISTRY

        return issubclass(REGISTRY.agent_class(cfg.cls.split(":", 1)[1]), LlmAgent)
    except Exception:  # noqa: BLE001 — an unregistered class: say "no model"
        return False


def _flow_order(system: SystemConfig) -> List[str]:
    """Agents in the order a run meets them: pre-stages, the delegation tree
    from the root (depth first), post-stages, then whatever is left over."""
    order: List[str] = []

    def visit(name: str) -> None:
        if name in order or name not in system.agents:
            return
        order.append(name)
        for dep in system.agents[name].subordinates + system.agents[name].children:
            visit(dep)

    for name in system.pipeline.pre:
        visit(name)
    visit(system.root.name)
    for name in system.pipeline.post:
        visit(name)
    for name in system.agents:
        visit(name)
    return order


def _model_names() -> Dict[str, Optional[str]]:
    llm = get_settings().llm
    return {
        "main": llm.main_model,
        "coder": llm.coder_model or llm.main_model,
        "nir": llm.nir_model or llm.main_model,
    }


# `enabled: "${...}"` references and the appSettings path of the field that
# edits each one. Such an agent is switched in the Agents section through that
# setting, never through an override: the runtime reads the same setting (the
# medical and FEDOT routes, the reranker fallback, the research-frame seed),
# so an override would switch the agent and leave the route behind.
# `nir_buildable` is the deployment (a normcontrol server) rather than a
# switch; the operator's switch for NirReportAgent is whether it is offered.
def _enabled_ref(cfg: AgentConfig) -> Optional[str]:
    return str(cfg.enabled).strip()[2:-1].strip() if _is_setting_ref(cfg.enabled) else None


def _lock_reason(cfg: AgentConfig, declared_enabled: bool) -> Optional[str]:
    if cfg.root:
        return "root"
    # A mode-controlled composite may also be marked internal.  The mode is
    # nevertheless the reason its switch is read-only, and lets the browser
    # update its indicator immediately when the start mode changes.
    if cfg.name in MODE_CONTROLLED_AGENTS:
        return "startMode"
    if cfg.internal:
        return "internal"
    ref = _enabled_ref(cfg)
    if ref is not None:
        if ref not in ENABLED_SETTING_FIELDS:
            return "setting"
        if ref == "nir_buildable" and not declared_enabled:
            return "unavailable"
        return None
    if not cfg.enabled_overridable():
        return "startMode"
    return None


def _mode_controlled_enabled(
    cfg: AgentConfig, system: SystemConfig, declared_enabled: bool,
) -> bool:
    """Display the state produced by ``build_for_mode`` for locked agents.

    Their YAML value is the baseline, while the selected start mode patches
    the live tree.  Showing only the baseline made PlannerAgent look disabled
    in ``planner`` mode even though it was the first stage being executed.
    """
    if cfg.name not in MODE_CONTROLLED_AGENTS:
        return declared_enabled
    mode = get_settings().web.start_mode
    if cfg.name == "PlannerAgent":
        return mode not in ("orchestrator_planner", "orchestrator_plan")
    if cfg.name in ("PlanningPipelineAgent", "InitAgent"):
        selected = (
            "PlanningPipelineAgent"
            if "PlanningPipelineAgent" in system.agents
            else "InitAgent"
        )
        return cfg.name == selected and mode in ("init", "planner")
    return declared_enabled


def _tool_limit(cfg: AgentConfig) -> Optional[Dict[str, Any]]:
    """The budget of the agent's limiter callback, for a number field in its row."""
    from CoScientist.assembly.registry import REGISTRY

    limit = REGISTRY.tool_limit(cfg.callbacks.before_tool)
    if limit is None:
        return None
    return {
        "kind": limit.kind,
        "default": limit.default(),
        "setting": limit.setting,
        "min": limit.minimum,
        "max": limit.maximum,
    }


def agents_catalog() -> Dict[str, Any]:
    """Every agent of the active profile, with its declared values.

    Loaded afresh (not the cached get_config()) and read through the
    ``declared_*`` accessors, so a toggle shows what the YAML says and the
    override is shown next to it, never folded into it.
    """
    from CoScientist.web.agent_tree import agent_presentation

    system = load_config()
    controls = agent_control_policy()
    parents: Dict[str, List[str]] = {}
    for cfg in system.agents.values():
        for dep in cfg.subordinates + cfg.children:
            parents.setdefault(dep, []).append(cfg.name)

    agents = []
    for name in _flow_order(system):
        cfg = system.agents[name]
        has_model = _has_model(cfg)
        try:
            declared_enabled = cfg.declared_enabled()
        except Exception:  # noqa: BLE001 — a reference to a missing setting
            declared_enabled = False
        stage = ("pre" if name in system.pipeline.pre
                 else "post" if name in system.pipeline.post else None)
        control = controls.get(name, {
            "availableInProfile": False,
            "effectiveEnabled": False,
            "canEnable": False,
            "canDisable": False,
            "controlReason": "Агент недоступен в текущем профиле.",
            "controlCode": "unavailable",
            "requiredForPipeline": False,
        })
        old_lock = _lock_reason(cfg, declared_enabled)
        # Keep the historical field for existing clients while the explicit
        # canEnable/canDisable pair describes the asymmetric restore case.
        lock = old_lock
        if not control["availableInProfile"]:
            lock = "unavailable"
        elif control["controlCode"] == "parentDisabled":
            lock = "parentDisabled"
        elif control["requiredForPipeline"] and not control["canEnable"]:
            lock = "required"
        agents.append({
            "name": name,
            **agent_presentation(name, cfg.title, cfg.description),
            "class": cfg.cls,
            "description": cfg.description or "",
            "root": bool(cfg.root),
            "internal": bool(cfg.internal),
            "stage": stage,
            "parents": parents.get(name, []),
            "subordinates": list(cfg.subordinates) + list(cfg.children),
            "enabled": _mode_controlled_enabled(cfg, system, declared_enabled),
            "effectiveEnabled": control["effectiveEnabled"],
            **public_control_fields(control),
            # The setting that decides `enabled` in the YAML, if any, and the
            # appSettings field the switch then edits instead of an override.
            "enabledRef": _enabled_ref(cfg),
            "enabledSetting": ENABLED_SETTING_FIELDS.get(_enabled_ref(cfg) or ""),
            "lock": lock,
            "hasModel": has_model,
            "model": (cfg.model or system.defaults.model) if has_model else None,
            "reasoning": _reasoning_label(cfg.declared_reasoning()) if has_model else None,
            "reasoningRef": str(cfg.reasoning)[2:-1] if _is_setting_ref(cfg.reasoning) else None,
            # Search calls or calls per tool, when a limiter callback caps them.
            "toolLimit": _tool_limit(cfg),
        })
    return {
        "agents": agents,
        "defaults": {
            "model": system.defaults.model,
            "reasoning": _reasoning_label(system.defaults.reasoning),
        },
        "models": _model_names(),
        "reasoningChoices": REASONING_CHOICES,
    }


def agent_enabled_patch(name: str, enabled: bool) -> Dict[str, Any]:
    """One switch, merged with the current session; never edit global settings."""
    system = load_config()
    if name not in system.agents:
        raise KeyError(name)
    validate_agent_enabled_change(name, enabled)
    cfg = system.agents[name]
    overrides = current_agent_settings()["overrides"]
    override = overrides.setdefault(name, {})
    setting = ENABLED_SETTING_FIELDS.get(_enabled_ref(cfg) or "")
    payload: Dict[str, Any] = {"agents": {"overrides": overrides}}
    if setting:
        section, field = setting.split(".", 1)
        payload[section] = {field: enabled}
        override.pop("enabled", None)
    else:
        override["enabled"] = enabled
    if not override:
        overrides.pop(name)
    return payload


def _reasoning_label(value: Any) -> Optional[str]:
    """A reasoning value as the modal shows it: "off", an effort, or None."""
    if value is False:
        return "off"
    if value is True:
        return "medium"
    normalized = normalize_reasoning(value)
    if normalized in ("none", "disabled"):
        return "off"
    return normalized


def current_agent_settings() -> Dict[str, Any]:
    """``settings.agents`` in the appSettings shape (camelCase, unset fields dropped)."""
    block = get_settings().agents
    return {
        "defaultReasoning": block.default_reasoning or "",
        "overrides": {
            name: override.model_dump(exclude_none=True)
            for name, override in sorted(block.overrides.items())
            if override.model_dump(exclude_none=True)
        },
    }


def apply_agent_settings(section: Dict[str, Any]) -> None:
    """Write the modal's Agents section into ``settings.agents``.

    Values the ``reasoning:`` vocabulary does not accept are dropped with a
    warning rather than stored, so a bad import cannot reach the assembler.
    Takes effect when the next session's tree is built.
    """
    block = get_settings().agents
    if "defaultReasoning" in section:
        raw = section["defaultReasoning"]
        value = normalize_reasoning(raw) if raw not in (None, "") else None
        if raw not in (None, "") and value is None:
            _log.warning("agents.defaultReasoning=%r is not a reasoning level; ignored", raw)
        block.default_reasoning = value if isinstance(value, str) else None

    if "overrides" in section:
        try:
            system: Optional[SystemConfig] = load_config()
        except Exception:  # noqa: BLE001 — keep what was sent; is_enabled decides
            system = None
        overrides: Dict[str, AgentOverride] = {}
        for name, raw in (section.get("overrides") or {}).items():
            if not isinstance(raw, dict):
                continue
            enabled = raw.get("enabled")
            # An agent the switch cannot override keeps no dead `enabled`
            # (an old save or an imported file): it would read as a change.
            cfg = system.agents.get(str(name)) if system is not None else None
            if cfg is not None and not cfg.enabled_overridable():
                enabled = None
            reasoning = raw.get("reasoning")
            if reasoning not in (None, ""):
                if normalize_reasoning(reasoning) is None:
                    _log.warning("reasoning %r for %s is not a reasoning level; ignored", reasoning, name)
                    reasoning = None
                else:
                    reasoning = normalize_reasoning(reasoning)
            else:
                reasoning = None
            model = str(raw.get("model") or "").strip() or None
            override = AgentOverride(
                enabled=bool(enabled) if isinstance(enabled, bool) else None,
                reasoning=reasoning,
                model=model,
                limit=_limit_value(name, raw.get("limit"), cfg),
            )
            if override.model_dump(exclude_none=True):
                overrides[str(name)] = override
        block.overrides = overrides


def _limit_value(name: Any, raw: Any, cfg: Optional[AgentConfig]) -> Optional[int]:
    """A tool budget within its limiter's bounds, or None (dropped with a
    warning) for a non-number or an agent without a limiter."""
    if raw in (None, "") or isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        _log.warning("limit %r for %s is not a number; ignored", raw, name)
        return None
    if cfg is None:
        return value if value >= 0 else None
    from CoScientist.assembly.registry import REGISTRY

    limit = REGISTRY.tool_limit(cfg.callbacks.before_tool)
    if limit is None:
        _log.warning("%s has no tool limiter; limit %r ignored", name, raw)
        return None
    return max(limit.minimum, min(limit.maximum, value))
