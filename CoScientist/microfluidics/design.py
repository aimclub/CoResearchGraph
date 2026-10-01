"""Module B, stage 7: which molecule to make — decided, never designed.

``MoleculeSelectionAgent`` (MolDesignAgent in the microfluidics profile) runs
no model. The molecule comes, in order, from:

1. the ТЗ — when the customer named the substance (``target_molecule.fixed``),
   with the properties module A found for it in the literature;
2. the route selected in module A — its product, SMILES as the source gives it;
3. the routes handed on without a single selection — their products.

Nothing handed on means a gap for the operator: a molecule no route makes
would be costed and tested by nobody. It reads the ТЗ itself (not the stored
``target_molecule``), so a ТЗ edited after module A still decides.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from CoScientist.microfluidics.literature import extract_target_molecule
from CoScientist.microfluidics.models import (
    DesignCandidate,
    DesignCandidates,
    NamedValue,
    TargetMolecule,
)

logger = logging.getLogger(__name__)

CANDIDATES_KEY = "design_candidates"


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _matching_analogue(target: TargetMolecule, analysis: Any) -> Optional[Dict[str, Any]]:
    """The analogue from the literature that IS the target (by SMILES, then name)."""
    analogues = (analysis or {}).get("analogues") if isinstance(analysis, dict) else None
    for analogue in analogues or []:
        if not isinstance(analogue, dict):
            continue
        if target.smiles and analogue.get("smiles") and analogue["smiles"].strip() == target.smiles.strip():
            return analogue
    for analogue in analogues or []:
        if isinstance(analogue, dict) and target.name and _norm(analogue.get("name")) == _norm(target.name):
            return analogue
    return None


def fixed_target_candidates(structured_tz: Any, literature_analysis: Any) -> Optional[DesignCandidates]:
    """The candidates to hand on for a fixed molecule, or None when nothing is fixed."""
    target = extract_target_molecule(structured_tz)
    if not target.fixed:
        return None

    analogue = _matching_analogue(target, literature_analysis) or {}
    properties: List[NamedValue] = []
    for item in analogue.get("properties") or []:
        try:
            properties.append(NamedValue.model_validate(item))
        except Exception:  # noqa: BLE001 — a malformed property is skipped, not fatal
            continue

    name = target.name or analogue.get("name") or target.cas or target.smiles
    gaps: List[str] = []
    smiles = target.smiles or str(analogue.get("smiles") or "")
    if not smiles:
        gaps.append(
            "SMILES целевой молекулы не задан ни в ТЗ, ни в литературе — маршрут "
            "синтеза и стоимость придётся искать по названию"
        )
    if not properties:
        gaps.append("В литературе не найдено измеренных свойств целевой молекулы")

    candidate = DesignCandidate(
        name=name,
        smiles=smiles,
        compound_class=str(analogue.get("compound_class") or ""),
        properties=properties,
        tz_fit="Задана заказчиком в ТЗ — подбор кандидатов не выполнялся",
        risks="",
        source="ТЗ",
        stub=False,
    )
    return DesignCandidates(fixed_target=True, candidates=[candidate], gaps=gaps)


def _answer(result: DesignCandidates) -> types.Content:
    return types.Content(
        role="model", parts=[types.Part(text=result.model_dump_json(ensure_ascii=False))]
    )


def use_fixed_target_molecule(callback_context: CallbackContext) -> Optional[types.Content]:
    """Skip design when the ТЗ fixes the molecule; hand the molecule on instead."""
    state = callback_context.state
    try:
        result = fixed_target_candidates(state.get("structured_tz"), state.get("literature_analysis"))
    except Exception as exc:  # noqa: BLE001 — fall back to the next source
        logger.warning("fixed target molecule: could not build candidates: %s", exc)
        return None
    if result is None:
        return None

    state[CANDIDATES_KEY] = result.model_dump()
    logger.info("fixed target molecule %r handed on — design skipped", result.candidates[0].name)
    return _answer(result)


def _selected_route(state: Any) -> Optional[Dict[str, Any]]:
    """The route whose product Module B takes: the one ``route_selection``
    selected, found among the active routes (``synthesis_routes``) or the
    literature routes; without a selection, the only ACTIVE route. A literature
    route that was not handed on is never taken — the operator may have
    rejected it."""
    selection = state.get("route_selection")
    if isinstance(selection, str):
        try:
            selection = json.loads(selection)
        except ValueError:
            selection = None
    selected_id = str(selection.get("selected_route_id") or "").strip() if isinstance(selection, dict) else ""
    synthesis = state.get("synthesis_routes")
    analysis = state.get("literature_analysis")
    for doc, key in ((synthesis, "routes"), (analysis, "synthesis_routes")):
        routes = [r for r in (doc.get(key) if isinstance(doc, dict) else None) or [] if isinstance(r, dict)]
        if selected_id:
            match = next((r for r in routes if str(r.get("route_id") or "").strip() == selected_id), None)
            if match is not None:
                return match
        elif len(routes) == 1 and doc is synthesis:
            return routes[0]
    return None


def _route_product(route: Dict[str, Any]) -> tuple[str, str]:
    product = route.get("product") or {}
    if isinstance(product, dict):
        name = str(product.get("name") or product.get("smiles") or "").strip()
        smiles = str(product.get("smiles") or "").strip()
    else:
        name = str(product).strip()
        smiles = str(route.get("product_smiles") or "").strip()
    return name or str(route.get("route_id") or "").strip(), smiles


def _route_candidate(route: Dict[str, Any], derivation: str, tz_fit: str) -> tuple[Optional[DesignCandidate], str]:
    """The route's product as a candidate, or the gap explaining why not.
    The structure is taken as the source gives it — never invented."""
    from rdkit import Chem

    route_id = str(route.get("route_id") or "")
    name, smiles = _route_product(route)
    if not smiles:
        return None, (f"Маршрут {route_id} не содержит явного product SMILES: "
                      "структура не выдумывается и требует ручного уточнения.")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not molecule.GetNumAtoms():
        return None, f"Некорректный product SMILES маршрута {route_id}."
    return DesignCandidate(
        name=name,
        smiles=Chem.MolToSmiles(molecule, isomericSmiles=True),
        compound_class="продукт литературного маршрута",
        source="литература",
        derivation=derivation,
        route_ids=[route_id],
        tz_fit=tz_fit,
        risks="Целевые эксплуатационные свойства требуют экспериментальной проверки.",
    ), ""


def selected_route_candidates(state: Any) -> Optional[DesignCandidates]:
    """The selected route's product, or None when no single route was selected."""
    route = _selected_route(state)
    if route is None:
        return None
    candidate, gap = _route_candidate(
        route,
        derivation="Продукт выбранного литературного маршрута",
        tz_fit="Маршрут выбран в Module A; повторный подбор не выполнялся",
    )
    return DesignCandidates(candidates=[candidate]) if candidate else DesignCandidates(gaps=[gap])


def use_selected_route_product(callback_context: CallbackContext) -> Optional[types.Content]:
    """Hand the selected literature product to Module B verbatim.

    A selected route is already a design decision.  Do not run BRICS or ask an
    LLM to invent another molecule at this point.  If the source did not give
    an explicit structure, leave the stage fail-soft with an actionable gap.
    """
    state = callback_context.state
    try:
        result = selected_route_candidates(state)
    except Exception as exc:  # noqa: BLE001 - stage boundary must never crash the run
        logger.exception("selected route product hand-off failed: %s", exc)
        result = DesignCandidates(gaps=[
            f"Не удалось зафиксировать продукт выбранного маршрута ({type(exc).__name__})."
        ])
    if result is None:
        return None
    state[CANDIDATES_KEY] = result.model_dump()
    return _answer(result)


def active_route_candidates(state: Any) -> DesignCandidates:
    """No fixed molecule, no single selected route: the products of the routes
    handed on (several, when no route was selected in a headless run). No
    route handed on — a gap for the operator, never an invented molecule."""
    doc = state.get("synthesis_routes")
    routes = [r for r in (doc.get("routes") if isinstance(doc, dict) else None) or [] if isinstance(r, dict)]
    if not routes:
        return DesignCandidates(gaps=[
            "Молекула не определена: в ТЗ она не задана, а маршрут синтеза не "
            "передан дальше. Новые молекулы на этом этапе не придумываются — "
            "нужен выбор маршрута в модуле A."
        ])
    candidates, gaps = [], []
    for route in routes:
        candidate, gap = _route_candidate(
            route,
            derivation="Продукт маршрута, переданного без единственного выбора",
            tz_fit="Маршрут передан дальше вместе с другими кандидатами; выбор не сделан",
        )
        if candidate:
            candidates.append(candidate)
        else:
            gaps.append(gap)
    return DesignCandidates(candidates=candidates, gaps=gaps)


def choose_design_candidates(state: Any) -> DesignCandidates:
    """Stage 7, deterministic: the ТЗ's molecule, else the selected route's
    product, else the products of the routes handed on."""
    try:
        fixed = fixed_target_candidates(state.get("structured_tz"), state.get("literature_analysis"))
    except Exception as exc:  # noqa: BLE001 — fall back to the routes
        logger.warning("fixed target molecule: could not build candidates: %s", exc)
        fixed = None
    if fixed is not None:
        return fixed
    try:
        return selected_route_candidates(state) or active_route_candidates(state)
    except Exception as exc:  # noqa: BLE001 - stage boundary must never crash the run
        logger.exception("design candidates: route products failed: %s", exc)
        return DesignCandidates(gaps=[
            f"Не удалось определить молекулу по маршрутам ({type(exc).__name__})."
        ])


class MoleculeSelectionAgent(BaseAgent):
    """Module B, stage 7, without a model.

    The molecule is fixed by the ТЗ or by the route chosen in module A; this
    stage only records it: ``design_candidates``, one research-graph
    Hypothesis per candidate, and the DesignCandidates JSON as its answer (the
    shape ModuleB's AgentTool validates).
    """

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        from CoScientist.microfluidics.research_record import record_design_hypotheses

        result = choose_design_candidates(ctx.session.state)
        actions = EventActions()
        callback_context = CallbackContext(ctx, event_actions=actions)
        callback_context.state[CANDIDATES_KEY] = result.model_dump()
        try:
            record_design_hypotheses(callback_context)
        except Exception as exc:  # noqa: BLE001 - graph recording is best-effort
            logger.warning("design candidates: research graph not updated: %s", exc)
        logger.info(
            "design candidates: %s", [c.name for c in result.candidates] or result.gaps,
        )
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=_answer(result),
            actions=actions,
        )


__all__ = [
    "CANDIDATES_KEY", "MoleculeSelectionAgent", "choose_design_candidates",
    "fixed_target_candidates", "use_fixed_target_molecule", "use_selected_route_product",
]
