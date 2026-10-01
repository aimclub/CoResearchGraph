"""Canonical non-human outcomes for all HITL transports."""

from __future__ import annotations

from typing import Any

from CoScientist.hitl.models import (
    HITLAction,
    HITLDecisionSource,
    HITLRequest,
    HITLResponse,
)


def _automatic_form_values(form: dict[str, Any] | None) -> dict[str, Any] | None:
    """Materialise every form field using its value/default or an empty string."""
    if not form:
        return None
    values: dict[str, Any] = {}
    for block in form.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        title = str(block.get("title") or "")
        answers: dict[str, Any] = {}
        for field in block.get("fields") or []:
            if not isinstance(field, dict) or not field.get("name"):
                continue
            value = field.get("value", field.get("default", ""))
            answers[str(field["name"])] = "" if value is None else value
        if answers:
            values[title] = answers
    return values or None


def resolve_auto(request: HITLRequest) -> HITLResponse:
    """Approve immediately with deterministic values for the request shape."""
    if request.requires_human:
        return resolve_timeout(reason="human_decision_required")
    selected = None
    action = request.action_type
    instructions = None
    free_input = None
    form_values = _automatic_form_values(request.form)

    if action == HITLAction.SELECT:
        if request.default_option in request.options:
            selected = request.default_option
        elif request.options:
            selected = request.options[0]
    elif action == HITLAction.PROVIDE_INPUT:
        # Empty input is deliberate in auto mode.  Keep both historical fields
        # populated so older consumers and the normalized SessionAgent agree.
        instructions = ""
        free_input = ""

    return HITLResponse(
        action=action,
        approved=True,
        selected_option=selected,
        instructions=instructions,
        free_input=free_input,
        form_values=form_values,
        decision_source=HITLDecisionSource.MODE_AUTO,
        system_reason="hitl_mode_auto",
    )


def resolve_timeout(*, reason: str = "review_window_elapsed") -> HITLResponse:
    """Fail closed without attributing the absence to a human."""
    return HITLResponse(
        action=HITLAction.REJECT,
        approved=False,
        timed_out=True,
        decision_source=HITLDecisionSource.TIMEOUT,
        system_reason=reason,
    )


__all__ = ["resolve_auto", "resolve_timeout"]
