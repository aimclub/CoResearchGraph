"""How the microfluidics agents' JSON answers are repaired.

Registered with the core ``sanitize_json_output`` callback
(CoScientist/agents/callbacks/json_output.py): the schema each agent's answer
must satisfy, the conservative answer used when the model's text cannot be
parsed, and the route-selection fix-ups.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from CoScientist.agents.callbacks.json_output import register_structured_answer
from CoScientist.microfluidics.models import (
    DesignCandidates,
    LiteratureAnalysis,
    LiteratureQueries,
    RouteSelection,
)

logger = logging.getLogger(__name__)


def _lift_route_selection_fields(payload: Any) -> Any:
    """Recover two root fields commonly indented into the last decision.

    Some models answer a JSON-schema request using YAML indentation.  In the
    usual failure shape the final ``selected_route_id`` and
    ``selection_reason`` wind up in the final member of ``decisions``.  YAML
    can parse that response, but the route-selection schema cannot.  The two
    fields are unambiguous root-only fields, so moving them back is lossless.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list):
        return payload
    for field in ("selected_route_id", "selection_reason"):
        if field in payload:
            continue
        for decision in reversed(payload["decisions"]):
            if isinstance(decision, dict) and field in decision:
                payload[field] = decision.pop(field)
                break
    return payload


def _normalize_route_selection(payload: Any) -> Any:
    """Make a RouteSelection's duplicated choice fields internally coherent.

    ``selected_route_id`` and ``recommendation`` describe the same decision,
    but are generated independently by the model.  ADK validates the output
    immediately after this callback, so an otherwise useful answer used to
    abort the whole workflow before the human reviewer could see it.

    A known selected id is authoritative.  In the inverse, unambiguous case
    (one ``оставить`` and no id), derive the id from that decision.  If a model
    gives an unknown id or several kept routes without an id, there is no
    defensible automatic choice: publish a safe "none selected" proposal for
    HITL rather than inventing a route or crashing Module A.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list):
        return payload

    decisions = payload["decisions"]
    if not all(isinstance(item, dict) for item in decisions):
        return payload

    changed = False
    for item in decisions:
        route_id = item.get("route_id")
        if isinstance(route_id, str):
            normalized = route_id.strip()
            if normalized != route_id:
                item["route_id"] = normalized
                changed = True

    selected = payload.get("selected_route_id", "")
    if not isinstance(selected, str):
        return payload
    normalized_selected = selected.strip()
    if normalized_selected != selected:
        payload["selected_route_id"] = normalized_selected
        changed = True
    selected = normalized_selected

    ids = [item.get("route_id") for item in decisions]
    kept = [item for item in decisions if item.get("recommendation") == "оставить"]
    repair_note = ""
    if selected and selected in ids:
        for item in decisions:
            recommendation = "оставить" if item.get("route_id") == selected else "отсеять"
            if item.get("recommendation") != recommendation:
                item["recommendation"] = recommendation
                changed = True
        if changed:
            repair_note = "Рекомендации синхронизированы с selected_route_id."
    elif not selected and len(kept) == 1:
        payload["selected_route_id"] = kept[0]["route_id"]
        changed = True
        repair_note = "selected_route_id восстановлен из единственной рекомендации «оставить»."
    elif selected or len(kept) > 1:
        # Do not silently pick the first candidate: this is a scientific and
        # operational decision reserved for the reviewer.
        payload["selected_route_id"] = ""
        for item in decisions:
            if item.get("recommendation") == "оставить":
                item["recommendation"] = "отсеять"
        changed = True
        repair_note = (
            "Неоднозначный выбор модели сброшен: оператору нужно выбрать "
            "маршрут вручную."
        )

    if changed:
        if repair_note:
            previous_reason = str(payload.get("selection_reason") or "").strip()
            payload["selection_reason"] = (
                f"{repair_note} {previous_reason}".strip()
            )
        logger.warning("[RouteSelectionAgent] normalized inconsistent route selection")
    return payload


def _normalize_route_selection_answer(payload: Any) -> Any:
    return _normalize_route_selection(_lift_route_selection_fields(payload))


def _literature_fallback(agent_name: str):
    def fallback(state: Mapping[str, Any]) -> dict:
        if agent_name == "EvidenceVerifierAgent":
            candidate = state.get("literature_analysis_draft") or state.get("literature_analysis")
            try:
                return LiteratureAnalysis.model_validate(candidate or {}).model_dump()
            except Exception:  # noqa: BLE001
                pass
        return {
            "target_molecule": {}, "source_records": [], "analogues": [],
            "synthesis_routes": [], "facts": [],
            "gaps": [f"{agent_name}: ответ модели не удалось разобрать."],
        }
    return fallback


def _route_selection_fallback(state: Mapping[str, Any]) -> dict:
    decisions = []
    analysis = state.get("literature_analysis") or {}
    routes = analysis.get("synthesis_routes") if isinstance(analysis, dict) else []
    for route in routes or []:
        if not isinstance(route, dict):
            continue
        product = route.get("product") or ""
        if isinstance(product, dict):
            product = product.get("name") or product.get("smiles") or ""
        decisions.append({
            "route_id": str(route.get("route_id") or "").strip(),
            "product": str(product), "recommendation": "отсеять",
            "reason": "Автоматическая рекомендация не сформирована; требуется решение оператора.",
        })
    try:
        return RouteSelection(decisions=decisions).model_dump()
    except Exception:  # noqa: BLE001
        return {"decisions": [], "selected_route_id": "", "selection_reason":
                "Ответ модели не разобран; требуется ручная проверка."}


def _register() -> None:
    register_structured_answer(
        "LiteratureSynthesisAgent",
        schema=LiteratureAnalysis,
        fallback=_literature_fallback("LiteratureSynthesisAgent"),
        # Older direct callers (no ADK callback context) got a selection shape.
        contextless={"selected_ids": [], "reason": "Ответ метаагента нельзя было разобрать.",
                     "warnings": ["Ответ метаагента нельзя было разобрать."]},
    )
    register_structured_answer(
        "EvidenceVerifierAgent",
        schema=LiteratureAnalysis,
        fallback=_literature_fallback("EvidenceVerifierAgent"),
    )
    register_structured_answer(
        "RouteSelectionAgent",
        schema=RouteSelection,
        fallback=_route_selection_fallback,
        normalize=_normalize_route_selection_answer,
    )
    register_structured_answer(
        "TZQueryGenAgent",
        schema=LiteratureQueries,
        fallback=lambda state: {"queries": []},
    )
    register_structured_answer(
        "MolDesignAgent",
        schema=DesignCandidates,
        fallback=lambda state: {"fixed_target": False, "candidates": [],
                                "gaps": ["Ответ агента дизайна не удалось разобрать."]},
    )


_register()
