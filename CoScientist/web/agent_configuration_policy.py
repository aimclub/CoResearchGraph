"""One server-side policy for changing a session's agent composition.

The assembler remains the authority for execution.  This module only answers
which declared agents belong to the mode-adjusted topology and which of those
the UI may disconnect.  Structural reachability deliberately ignores a
session's enabled overrides: an optional agent that was switched off must stay
available so it can be connected again.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from CoScientist.assembly.schema import (
    MODE_CONTROLLED_AGENTS,
    AgentConfig,
    SystemConfig,
    _is_setting_ref,
    load_config,
)
from CoScientist.config import get_settings, settings_scope
from CoScientist.config.settings import Settings


ENABLED_SETTING_FIELDS: Dict[str, str] = {
    "context_init.enabled": "general.contextInitEnabled",
    "web.medical_agent_enabled": "medicalAgent.enabled",
    "web.fedot_fallback_enabled": "taskExecutorAgent.fedotFallback",
    "experiments.route_fedot": "experimentModule.routeFedot",
    "experiments.route_alembic": "experimentModule.routeAlembic",
    "nir_buildable": "nirReport.enabled",
}


def _enabled_ref(cfg: AgentConfig) -> Optional[str]:
    return str(cfg.enabled).strip()[2:-1].strip() if _is_setting_ref(cfg.enabled) else None


def _effective_enabled(cfg: AgentConfig) -> bool:
    enabled = cfg.is_enabled()
    if cfg.name == "NirReportAgent":
        enabled = enabled and get_settings().nir.enabled
    return bool(enabled)


def _switch_supported(cfg: AgentConfig) -> bool:
    """Whether a session setting can express this agent's enabled state."""
    if cfg.root or cfg.internal or cfg.name in MODE_CONTROLLED_AGENTS:
        return False
    ref = _enabled_ref(cfg)
    if ref is not None:
        if ref not in ENABLED_SETTING_FIELDS:
            return False
        # ``nir_buildable`` is a deployment capability.  The associated UI
        # setting may gate use of the report, but cannot build a missing agent.
        if ref == "nir_buildable" and not cfg.declared_enabled():
            return False
        return True
    return cfg.enabled_overridable()


def _relations(config: SystemConfig, name: str) -> Iterable[str]:
    agent = config.agent(name)
    # The mode transformation edits these lists (notably removing Planner from
    # direct-orchestrator planning), which is exactly the topology the next run
    # will receive.  Enabled flags are intentionally ignored here.
    yield from agent.subordinates
    yield from agent.children


def agent_control_policy(config: Optional[SystemConfig] = None) -> Dict[str, Dict[str, Any]]:
    """Return composition controls for every declaration in the active profile."""
    if config is None:
        from CoScientist.agents import config_for_mode

        config = config_for_mode()

    seeds = [*config.pipeline.pre, config.root.name, *config.pipeline.post]
    available: set[str] = set()
    parents: Dict[str, list[str]] = {}
    queue = list(seeds)
    while queue:
        name = queue.pop(0)
        if name in available or name not in config.agents:
            continue
        available.add(name)
        for child in _relations(config, name):
            parents.setdefault(child, []).append(name)
            queue.append(child)

    required = {
        config.root.name,
        *config.pipeline.pre,
        *config.pipeline.post,
        *(name for name in available if config.agent(name).required_for_pipeline),
        *(name for name in available if name in MODE_CONTROLLED_AGENTS),
    }
    # A sequential composite has no meaningful partial configuration: every
    # declared child is an ordered step and therefore protected as well.
    for name in available:
        agent = config.agent(name)
        if agent.cls == "sequential":
            required.update(child for child in agent.children if child in available)

    effective = {
        name: _effective_enabled(config.agent(name))
        for name in config.agents
    }

    # Fixed-point reachability is deliberately iterative.  Validated YAML is a
    # DAG, but workflow projection treats a repeated ancestor defensively and
    # tests/custom mode transforms may contain a recursive subordinate link.
    # Recursive memoisation cannot break that cycle before a value is cached.
    active_routes = {name for name in seeds if name in available}
    changed = True
    while changed:
        changed = False
        for name in available - active_routes:
            if any(
                parent in active_routes and effective[parent]
                for parent in parents.get(name, ())
            ):
                active_routes.add(name)
                changed = True

    result: Dict[str, Dict[str, Any]] = {}
    for name, cfg in config.agents.items():
        in_profile = name in available
        is_required = in_profile and name in required
        is_enabled = effective[name]
        parent_ready = in_profile and name in active_routes
        supported = _switch_supported(cfg)
        can_enable = bool(in_profile and not is_enabled and parent_ready and supported)
        can_disable = bool(in_profile and is_enabled and supported and not is_required)

        reason: Optional[str] = None
        code: Optional[str] = None
        if not in_profile:
            code = "unavailable"
            reason = "Агент не входит в структуру текущего профиля и режима запуска."
        elif not parent_ready:
            code = "parentDisabled"
            reason = "Недоступен: сначала включите родительскую ветку."
        elif is_required:
            code = "required"
            reason = (
                "Обязательный участник основного процесса. Его можно включить обратно, "
                "но нельзя отключить."
                if not is_enabled and can_enable
                else "Обязательный участник основного процесса."
            )
        elif cfg.root:
            code = "root"
            reason = "Корневой координатор обязателен для запуска системы."
        elif cfg.name in MODE_CONTROLLED_AGENTS:
            code = "startMode"
            reason = "Подключение этого агента определяется режимом запуска."
        elif cfg.internal:
            code = "internal"
            reason = "Системный агент управляется основным процессом."
        else:
            ref = _enabled_ref(cfg)
            if ref == "nir_buildable" and not cfg.declared_enabled():
                code = "unavailable"
                reason = "Агент недоступен: необходимый сервис не настроен."
            elif ref is not None and ref not in ENABLED_SETTING_FIELDS:
                code = "setting"
                reason = "Подключение этого агента определяется системной настройкой."

        result[name] = {
            "availableInProfile": in_profile,
            "effectiveEnabled": is_enabled,
            "canEnable": can_enable,
            "canDisable": can_disable,
            "controlReason": reason,
            "controlCode": code,
            "requiredForPipeline": is_required,
            # Private validation facts; callers exposing the policy use
            # ``public_control_fields`` below.
            "_switchSupported": supported,
            "_parentReady": parent_ready,
        }
    return result


def public_control_fields(control: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in control.items() if not key.startswith("_")}


def validate_agent_enabled_change(name: str, enabled: bool) -> None:
    raw = load_config()
    if name not in raw.agents:
        raise KeyError(name)
    control = agent_control_policy().get(name)
    if control is None or not control["availableInProfile"]:
        raise ValueError(control["controlReason"] if control else "Агент недоступен в текущем профиле.")
    current = bool(control["effectiveEnabled"])
    if enabled == current:
        return
    allowed = control["canEnable"] if enabled else control["canDisable"]
    if not allowed:
        raise ValueError(control["controlReason"] or "Состав этого этапа нельзя изменить.")


def validate_raw_agent_configuration_request(before: Settings, requested: Dict[str, Any]) -> None:
    """Validate explicitly submitted ``enabled`` overrides before normalising.

    ``apply_agent_settings`` intentionally drops dead overrides for roots,
    mode-controlled, internal and setting-backed agents.  Without this pass an
    explicit forbidden flag could disappear during normalisation and the rest
    of a mixed settings form would still be saved.  Values equal to the current
    state are accepted so old full-form payloads remain loadable.
    """
    agents = requested.get("agents")
    overrides = agents.get("overrides") if isinstance(agents, dict) else None
    if not isinstance(overrides, dict):
        return

    mode_snapshot = before.model_copy(deep=True)
    general = requested.get("general")
    mode_changed = False
    if isinstance(general, dict) and "startMode" in general:
        requested_mode = general["startMode"]
        if requested_mode != mode_snapshot.web.start_mode:
            mode_snapshot.web.start_mode = requested_mode
            mode_changed = True

    with settings_scope(before):
        current_policy = agent_control_policy()
    with settings_scope(mode_snapshot):
        mode_policy = agent_control_policy()
        raw_config = load_config()

    for raw_name, value in overrides.items():
        if not isinstance(value, dict) or not isinstance(value.get("enabled"), bool):
            continue
        name = str(raw_name)
        desired = value["enabled"]
        current = current_policy.get(name)
        after_mode = mode_policy.get(name)
        if current is None or after_mode is None or name not in raw_config.agents:
            raise ValueError(f"Агент {name} недоступен в текущем профиле.")

        # An unchanged legacy flag is data to preserve, not a new action.  For
        # mode-controlled agents also accept either side of a simultaneous
        # start-mode change; the mode itself remains the only effective switch.
        if desired == current["effectiveEnabled"]:
            continue
        if (
            mode_changed
            and name in MODE_CONTROLLED_AGENTS
            and desired == after_mode["effectiveEnabled"]
        ):
            continue

        cfg = raw_config.agent(name)
        if _enabled_ref(cfg) is not None:
            raise ValueError(
                "Подключение этого агента изменяется отдельной настройкой, "
                "а не полем agents.overrides.enabled."
            )
        with settings_scope(mode_snapshot):
            validate_agent_enabled_change(name, desired)


def _declared_switch_values(snapshot: Settings) -> Dict[str, bool]:
    """Raw switches, excluding changes produced solely by a start-mode patch."""
    with settings_scope(snapshot):
        config = load_config()
        values = {name: _effective_enabled(cfg) for name, cfg in config.agents.items()}
    return values


def validate_agent_configuration_transition(before: Settings, after: Settings) -> None:
    """Reject forbidden composition changes without rewriting legacy state.

    Only actual switch transitions are checked.  Thus an old saved override
    that already disables a required agent remains untouched, while changing
    it back to true is allowed whenever its parent branch exists.
    """
    old_values = _declared_switch_values(before)
    new_values = _declared_switch_values(after)
    with settings_scope(before):
        old_policy = agent_control_policy()
    with settings_scope(after):
        new_policy = agent_control_policy()

    for name in sorted(old_values.keys() | new_values.keys()):
        old = old_values.get(name)
        new = new_values.get(name)
        if old is None or new is None or old == new:
            continue
        if new:
            control = new_policy.get(name)
            if not control or not control["availableInProfile"] or not control["_switchSupported"] or not control["_parentReady"]:
                reason = control.get("controlReason") if control else None
                raise ValueError(reason or f"Агент {name} недоступен в текущем профиле.")
        else:
            control = old_policy.get(name)
            if not control or not control["canDisable"]:
                reason = control.get("controlReason") if control else None
                raise ValueError(reason or f"Агент {name} обязателен для основного процесса.")
