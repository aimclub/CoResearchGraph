"""Human-authorized escape hatch for a rejected route's verification plan.

The override is intentionally narrower than route qualification: it never
changes a route to ``eligible`` and it is never usable for production costing
or equipment execution.  Its only effect is to let Module C ask the external
system for a planning-only evidence/verification plan.
"""
from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from CoScientist.config import get_settings
from CoScientist.graph.session_scope import session_key
from CoScientist.hitl.models import HITLAction, HITLRequest
from CoScientist.microfluidics.models import (
    OPERATOR_ROUTE_OVERRIDE_KEY,
    OperatorRouteOverride,
    QualifiedRoutes,
    SynthesisRoutes,
)


async def operator_authorize_screening_override(
    route_ids: list[str],
    rationale: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Ask the operator to authorize a planning-only hand-off for rejected routes.

    Use only after code set ``qualified_routes.status=no_compliant_routes``.
    The selected routes must be real, fully described proposals.  Approval does
    not waive a hard constraint or permit equipment execution.
    """
    if not get_settings().web.hitl_enabled:
        return {
            "authorized": False,
            "reason": "operator override requires an enabled interactive HITL channel",
        }
    try:
        qualified = QualifiedRoutes.model_validate(tool_context.state.get("qualified_routes"))
        proposals = SynthesisRoutes.model_validate(tool_context.state.get("synthesis_routes"))
        selected = [route_id.strip() for route_id in route_ids]
        if qualified.status != "no_compliant_routes":
            raise ValueError("override is only needed when qualified_routes.status=no_compliant_routes")
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("select one or more distinct route_id values")
        proposal_by_id = {route.route_id: route for route in proposals.routes}
        missing = [route_id for route_id in selected if route_id not in proposal_by_id]
        unsafe = [
            route_id for route_id in selected
            if route_id in proposal_by_id
            and (proposal_by_id[route_id].stub or not proposal_by_id[route_id].steps)
        ]
        if missing or unsafe:
            raise ValueError(
                "selected routes must be real proposals with nonempty steps"
                + (f"; unknown={missing}" if missing else "")
                + (f"; unusable={unsafe}" if unsafe else "")
            )
        if not rationale.strip():
            raise ValueError("rationale is required")
    except (TypeError, ValueError) as exc:
        return {"authorized": False, "reason": str(exc)}

    selected_summary = [
        {
            "route_id": route_id,
            "product": proposal_by_id[route_id].product.name or proposal_by_id[route_id].product.smiles,
            "automatic_status": proposal_by_id[route_id].overall_status,
            "checks": [
                {"constraint_id": check.constraint_id, "status": check.status, "reason": check.reason}
                for check in proposal_by_id[route_id].tz_compliance
                if check.status in {"fail", "unknown"}
            ],
        }
        for route_id in selected
    ]
    user_id, session_id = session_key(tool_context)
    from CoScientist.agents.common import hitl_handler

    response = await hitl_handler.handle_request(HITLRequest(
        agent_name="RootOrchestrator",
        action_type=HITLAction.APPROVE,
        message=(
            "Маршруты не прошли автоматическую квалификацию. Разрешить только "
            "передачу во внешнюю систему для плана верификации? Это НЕ отменяет "
            "ограничения ТЗ и НЕ запускает CFD, оборудование или эксперимент."
        ),
        context={
            "output": {
                "selected_routes": selected_summary,
                "automatic_gaps": qualified.gaps,
                "requested_rationale": rationale.strip(),
                "effect": "planning_only verification hand-off; no equipment execution",
            },
            "_session": {"user_id": user_id, "session_id": session_id},
        },
        invoked_via="tool",
        trigger="operator_screening_override",
    ))
    if not response.approved:
        return {
            "authorized": False,
            "reason": response.instructions or response.free_input or "operator declined override",
        }

    override = OperatorRouteOverride(
        mode="screening_only",
        approved_by_human=True,
        route_ids=selected,
        rationale=rationale.strip(),
        operator_feedback=response.instructions or response.free_input or "",
    )
    tool_context.state[OPERATOR_ROUTE_OVERRIDE_KEY] = override.model_dump()
    return {
        "authorized": True,
        "mode": override.mode,
        "route_ids": override.route_ids,
        "note": "Only a planning-only verification hand-off is authorized.",
    }


__all__ = ["operator_authorize_screening_override"]
