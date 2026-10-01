"""Keep the economics server's numbers in the state, as the server gave them.

``collect_economics_result`` (after_tool, EconomicsAgent) reads the answers of
the two costing tools and writes them to the session state, so the report and
the later modules take costs from the server instead of from the agent's
retelling:

``economics_ranking``
    ``{target_qty, target_unit, preferred_currency, rank_by, routes: {route_id:
    {status, rank, currency, cost_per_unit, cost_packs, starting_materials,
    missing, warnings}}}`` —
    one entry per route; a later ranking of the same route replaces it (a
    corrected route is recalculated under its id).
``economics_estimates``
    the plain reagent estimates: ``[{reagents, total_by_currency, missing}]``.
``economics_raw``
    the full answer of the last ranking, breakdown included.

The answer shapes are the ones recorded from the live server in
``tests/unit/microfluidics/fixtures/economics_mcp/``. A transport failure (``{"error": ...}``) or an
``isError`` answer changes nothing: there are no numbers to keep.
"""
from __future__ import annotations

import copy
import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

from CoScientist.microfluidics.models import (
    OPERATOR_ECONOMICS_OVERRIDE_KEY,
    OperatorEconomicsOverride,
    SynthesisRoutes,
)

logger = logging.getLogger(__name__)

# Project-owned fallback requested for the microfluidics case.  It deliberately
# wins over fuzzy supplier matches (the live service once returned aniline).
VANILLIN_PRICE_RUB_PER_G = Decimal("3.00")
VANILLIN_MOLAR_MASS_G_PER_MOL = Decimal("152.149")

RANKING_KEY = "economics_ranking"
ESTIMATES_KEY = "economics_estimates"
RAW_KEY = "economics_raw"
PRELIMINARY_RANKING_KEY = "economics_preliminary_ranking"

_MAX_WARNINGS = 5


def _decimal(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except (InvalidOperation, TypeError, ValueError):
        return None


def _is_vanillin(*, smiles: Any = "", name: Any = "") -> bool:
    from CoScientist.microfluidics.chemistry_identity import VANILLIN_SMILES, canonical_smiles

    if str(name or "").strip().casefold() in {"vanillin", "ванилин"}:
        return True
    actual = canonical_smiles(str(smiles or ""))
    expected = canonical_smiles(VANILLIN_SMILES)
    return bool(actual and expected and actual == expected)


def _quantity_g(qty: Any, unit: Any) -> Decimal | None:
    amount = _decimal(qty)
    if amount is None or amount < 0:
        return None
    normalized = str(unit or "").strip().casefold()
    factors = {
        "g": Decimal("1"),
        "kg": Decimal("1000"),
        "mol": VANILLIN_MOLAR_MASS_G_PER_MOL,
        "mmol": VANILLIN_MOLAR_MASS_G_PER_MOL / Decimal("1000"),
    }
    return amount * factors[normalized] if normalized in factors else None


def _money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01")))


def apply_vanillin_price_override(result: Dict[str, Any]) -> Dict[str, Any]:
    """Replace any vanillin match/miss with the local 3 RUB/g cost basis."""
    patched = copy.deepcopy(result)
    changed = False
    for route in patched.get("routes") or []:
        if not isinstance(route, dict):
            continue
        estimate = route.get("estimate") if isinstance(route.get("estimate"), dict) else {}
        route.setdefault("estimate", estimate)
        line_items = estimate.setdefault("line_items", [])
        matched_line = next((item for item in line_items if isinstance(item, dict) and _is_vanillin(
            smiles=item.get("query_smiles"), name=item.get("display_name_en")
        )), None)
        material = next((item for item in route.get("starting_materials") or [] if isinstance(item, dict)
                         and _is_vanillin(smiles=item.get("smiles"), name=item.get("name"))), None)
        missing = [item for item in route.get("missing") or [] if isinstance(item, dict)]
        missing_vanillin = next((item for item in missing if _is_vanillin(
            smiles=item.get("smiles"), name=item.get("name")
        )), None)
        source = matched_line or material or missing_vanillin
        if source is None:
            continue
        qty = source.get("qty")
        unit = source.get("unit")
        if matched_line is not None:
            qty = matched_line.get("qty", qty)
            unit = matched_line.get("unit", unit)
        if (qty is None or unit is None) and material is not None:
            qty, unit = material.get("qty"), material.get("unit")
        grams = _quantity_g(qty, unit)
        if grams is None:
            route.setdefault("warnings", []).append(
                "Цена ванилина задана как 3 RUB/g, но количество нельзя привести к граммам."
            )
            continue

        new_cost = grams * VANILLIN_PRICE_RUB_PER_G
        old_unit_cost = _decimal((matched_line or {}).get("cost_per_unit")) or Decimal("0")
        old_pack_cost = _decimal((matched_line or {}).get("cost_packs")) or Decimal("0")
        base_unit = (_decimal(route.get("cost_per_unit")) or Decimal("0")) - old_unit_cost
        base_packs = (_decimal(route.get("cost_packs")) or Decimal("0")) - old_pack_cost
        route["cost_per_unit"] = _money(max(Decimal("0"), base_unit) + new_cost)
        route["cost_packs"] = _money(max(Decimal("0"), base_packs) + new_cost)
        route["currency"] = "RUB"

        replacement = matched_line if matched_line is not None else {
            "query_smiles": (material or missing_vanillin or {}).get("smiles"),
            "display_name_en": "vanillin",
            "qty": str(qty),
            "unit": str(unit),
        }
        if matched_line is None:
            line_items.append(replacement)
        replacement.update({
            "chosen": {
                "supplier": "hardcoded",
                "name_raw": "Vanillin",
                "pack_qty": "1",
                "pack_unit": "g",
                "price": _money(VANILLIN_PRICE_RUB_PER_G),
                "price_currency": "RUB",
                "unit_price": _money(VANILLIN_PRICE_RUB_PER_G),
            },
            "packs_needed": None,
            "cost_per_unit": _money(new_cost),
            "cost_packs": _money(new_cost),
            "comment": "hardcoded project price: 3 RUB/g",
        })
        route["missing"] = [item for item in missing if item is not missing_vanillin]
        if not route["missing"] and route.get("status") in {"partial", "unpriceable"}:
            route["status"] = "ok"
        route.setdefault("warnings", []).append(
            "Ванилин рассчитан по фиксированной проектной цене 3 RUB/g, не по прайсу поставщика."
        )
        changed = True

    if changed:
        patched.setdefault("assumptions", []).append(
            "Цена ванилина захардкожена: 3 RUB/g; fuzzy-совпадения поставщиков игнорируются."
        )
        usable = [route for route in patched.get("routes") or [] if isinstance(route, dict)
                  and route.get("status") in {"ok", "partial"}]
        usable.sort(key=lambda route: _decimal(route.get("cost_per_unit")) or Decimal("Infinity"))
        for rank, route in enumerate(usable, 1):
            route["rank"] = rank
    return patched


def structured_result(response: Any) -> Optional[Dict[str, Any]]:
    """The structured answer of an MCP call, or None for an error / no data.

    ADK hands after_tool a dumped CallToolResult: ``structuredContent`` when the
    tool declares an output schema, else JSON inside the text ``content``.
    """
    if not isinstance(response, dict):
        return None
    if response.get("isError") or response.get("is_error"):
        return None
    if set(response) == {"error"}:
        return None
    structured = response.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    for block in response.get("content") or []:
        text = block.get("text") if isinstance(block, dict) else None
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _route_summary(route: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": route.get("status"),
        "rank": route.get("rank"),
        "currency": route.get("currency"),
        "cost_per_unit": route.get("cost_per_unit"),
        "cost_packs": route.get("cost_packs"),
        # What to buy for the target amount — the recipe in the report.
        "starting_materials": [
            {"smiles": m.get("smiles"), "qty": m.get("qty"), "unit": m.get("unit")}
            for m in route.get("starting_materials") or [] if isinstance(m, dict)
        ],
        "missing": [
            {"smiles": m.get("smiles"), "reason": m.get("reason")}
            for m in route.get("missing") or [] if isinstance(m, dict)
        ],
        "warnings": list(route.get("warnings") or [])[:_MAX_WARNINGS],
    }


def _collect_ranking(state: Any, args: Dict[str, Any], result: Dict[str, Any]) -> None:
    result = apply_vanillin_price_override(result)
    qualified = state.get("qualified_routes") or {}
    allowed = {
        str(route.get("route_id")) for route in qualified.get("routes") or []
        if isinstance(route, dict) and route.get("route_id")
    }
    preliminary = False
    if not allowed:
        try:
            override = OperatorEconomicsOverride.model_validate(
                state.get(OPERATOR_ECONOMICS_OVERRIDE_KEY)
            )
            proposal_ids = {
                route.route_id for route in SynthesisRoutes.model_validate(
                    state.get("synthesis_routes")
                ).routes
            }
            if set(override.route_ids).issubset(proposal_ids):
                allowed = set(override.route_ids)
                preliminary = True
        except (TypeError, ValueError):
            pass
    returned = {
        str(route.get("route_id")) for route in result.get("routes") or []
        if isinstance(route, dict) and route.get("route_id")
    }
    if not allowed or returned != allowed:
        raise ValueError(
            "economics result route_ids must match qualified_routes exactly: "
            f"allowed={sorted(allowed)}, returned={sorted(returned)}"
        )
    ranking_key = PRELIMINARY_RANKING_KEY if preliminary else RANKING_KEY
    previous = state.get(ranking_key) or {}
    routes = dict(previous.get("routes") or {})
    for route in result.get("routes") or []:
        if isinstance(route, dict) and route.get("route_id"):
            routes[str(route["route_id"])] = _route_summary(route)
    fallback_steps: dict[str, list[int]] = {}
    for submitted in args.get("routes") or []:
        if not isinstance(submitted, dict) or not submitted.get("route_id"):
            continue
        missing = [
            index for index, step in enumerate(submitted.get("steps") or [], 1)
            if isinstance(step, dict) and step.get("yield", step.get("yield_fraction")) is None
        ]
        if missing:
            fallback_steps[str(submitted["route_id"])] = missing
    # Written whole: AgentTool forwards only whole-key deltas to the parent.
    state[ranking_key] = {
        "target_qty": args.get("target_qty", previous.get("target_qty")),
        "target_unit": args.get("target_unit", previous.get("target_unit")),
        "preferred_currency": result.get("preferred_currency"),
        "rank_by": result.get("rank_by"),
        "preliminary": bool(fallback_steps),
        "default_yield": args.get("default_yield") if fallback_steps else None,
        "yield_fallback_steps": fallback_steps,
        "routes": routes,
    }
    state[RAW_KEY] = result


def _collect_estimate(state: Any, args: Dict[str, Any], result: Dict[str, Any]) -> None:
    estimates = list(state.get(ESTIMATES_KEY) or [])
    estimates.append({
        "reagents": args.get("reagents") or [],
        "total_by_currency": result.get("total_by_currency") or {},
        "missing": result.get("missing") or [],
    })
    state[ESTIMATES_KEY] = estimates


_COLLECTORS = {
    "rank_routes_by_cost": _collect_ranking,
    "estimate_synthesis_cost": _collect_estimate,
}


def collect_economics_result(
    tool: Any = None, args: Any = None, tool_context: Any = None, tool_response: Any = None,
    **kwargs: Any,
) -> None:
    """after_tool: file the costing answers under their state keys."""
    name = str(getattr(tool, "name", "") or "")
    collector = _COLLECTORS.get(name)
    if collector is None or tool_context is None:
        return None
    result = structured_result(tool_response)
    if result is None:
        return None
    try:
        collector(tool_context.state, args if isinstance(args, dict) else {}, result)
    except Exception as exc:  # noqa: BLE001 — bookkeeping must never break the call
        logger.warning("economics: could not record %s: %s", name, exc)
    return None


__all__ = [
    "ESTIMATES_KEY",
    "RANKING_KEY",
    "RAW_KEY",
    "PRELIMINARY_RANKING_KEY",
    "collect_economics_result",
    "apply_vanillin_price_override",
    "structured_result",
    "VANILLIN_PRICE_RUB_PER_G",
]
