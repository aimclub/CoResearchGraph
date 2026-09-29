"""Central runtime policy for optional execution capabilities.

Agent-tree assembly is the first line of defence, but tools can also be
invoked directly (debug endpoints, A2A, or a tree built before settings were
changed).  These checks therefore live below routing and are intentionally
cheap enough to evaluate for every tool call.
"""

from __future__ import annotations

from typing import Any


def fedot_mas_enabled(tool_context: Any = None) -> bool:
    """Whether a direct FEDOT.MAS execution is allowed right now.

    An experiment-scoped call obeys the experiment route.  Standalone/A2A and
    diagnostic calls may run when either explicit FEDOT switch is enabled;
    both are off in the default configuration.
    """
    from CoScientist.config import get_settings

    settings = get_settings()
    state = getattr(tool_context, "state", None) or {}
    experiment_scoped = any(
        state.get(key)
        for key in (
            "experiment_runtime",
            "experiment_context",
            "experiment_active_envelope",
        )
    )
    if experiment_scoped:
        return bool(settings.experiments.route_fedot)
    return bool(
        settings.web.fedot_fallback_enabled
        or settings.experiments.route_fedot
    )


def alembic_enabled() -> bool:
    """Whether CoScientist may start a new Alembic MCP build right now."""
    from CoScientist.config import get_settings

    return bool(get_settings().experiments.route_alembic)


def capability_disabled(name: str, setting: str) -> dict[str, str]:
    """Stable tool result for a disabled optional capability."""
    return {
        "status": "error",
        "error_code": "capability_disabled",
        "error": f"{name} is disabled. Enable {setting} before starting it.",
        "required_setting": setting,
    }
