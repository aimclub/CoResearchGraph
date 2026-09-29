"""Validate CoScientist's hand-off, without inventing an external result schema."""
from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any

from CoScientist.microfluidics.models import (
    OPERATOR_ROUTE_OVERRIDE_KEY,
    LiteratureAnalysis,
    OperatorRouteOverride,
    QualifiedRoutes,
    SynthesisRoute,
    SynthesisRoutes,
)

INPUT_KEYS = (
    "structured_tz", "literature_analysis", "synthesis_routes", "qualified_routes",
    "economics", "economics_ranking",
)
TARGET_UNITS = ("g", "kg", "mol", "mmol")
RANK_BY = ("per_unit", "packs")
ROUTE_STATUSES = ("ok", "partial", "invalid", "unpriceable")
# TZ rows that carry no requirement for the external system.
UNSET_TZ_STATUSES = ("не требуется", "не задано")
CLOSED_CHECKS = ("pass", "not_applicable")


class EconomicsRankingError(ValueError):
    """Everything else in the production hand-off is valid; only a usable
    ``economics_ranking`` for ``route_ids`` is missing — the one gap an
    operator can close without redoing an earlier stage."""

    def __init__(self, message: str, route_ids: list[str]) -> None:
        super().__init__(message)
        self.route_ids = list(route_ids)


def _object(value: Any, name: str) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError(f"{name}: expected a JSON object") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name}: nonempty object required")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> None:
    try:
        if isinstance(value, bool):
            raise ValueError(name)
        number = Decimal(str(value))
        if not number.is_finite() or number < 0 or (positive and number == 0):
            raise ValueError(name)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name}: finite {'positive' if positive else 'nonnegative'} number required") from exc


def _kept(mapping: dict, keys: tuple) -> dict:
    return {key: mapping[key] for key in keys if mapping.get(key) not in (None, "", [], {})}


def _substance(item: dict) -> dict:
    return _kept(item, ("name", "smiles", "amount"))


def _tz(tz: dict) -> dict:
    """The request and every answered TZ field, grouped by block.

    Block usage notes and unanswered rows are internal to CoScientist; a
    field name repeated in a later block is sent once.
    """
    seen, requirements = set(), {}
    for block in tz.get("blocks") or []:
        fields = {}
        for row in block.get("fields") or []:
            name, value = str(row.get("name", "")).strip(), str(row.get("value", "")).strip()
            unset = row.get("status") in UNSET_TZ_STATUSES or value.lower().startswith(UNSET_TZ_STATUSES)
            if not name or not value or unset or name in seen:
                continue
            seen.add(name)
            fields[name] = value
        if fields:
            requirements[str(block.get("title", "")).strip()] = fields
    return _kept({"original_request": tz.get("original_request"), "requirements": requirements},
                 ("original_request", "requirements"))


def _route(route: SynthesisRoute) -> dict:
    """What an experiment is planned from: substances, conditions, yield and
    flow notes. Literature evidence and selection rationale stay behind."""
    data = route.model_dump(mode="json")
    steps = []
    for step in data["steps"]:
        out = _kept({
            **step,
            **{key: [_substance(item) for item in step[key]] for key in ("reactants", "agents", "products")},
            "conditions": [_kept(item, ("name", "value", "conditions")) for item in step["conditions"]],
        }, ("operation", "reactants", "agents", "products", "conditions", "conditions_status",
            "conditions_missing_reason", "yield_fraction", "yield_status", "yield_missing_reason",
            "flow_notes"))
        if out.get("flow_notes") == data["flow_suitability"]:
            del out["flow_notes"]  # Same text as the route-level note.
        steps.append(out)
    open_checks = [_kept(check, ("constraint_id", "status", "reason"))
                   for check in data["tz_compliance"] if check["status"] not in CLOSED_CHECKS]
    return _kept({**data, "product": _substance(data["product"]), "steps": steps, "open_checks": open_checks},
                 ("route_id", "product", "variant_label", "overall_status", "steps", "flow_suitability",
                  "bottlenecks", "product_purity_percent", "product_purity_status", "open_checks"))


def prepare_inputs(state: Any, *, planning_only: bool = False) -> dict:
    """Prepare either a production hand-off or a non-executing screening plan.

    Production needs fully qualified routes and cost rankings.  Screening is
    intentionally narrower: it may carry real routes with open evidence/yield
    gaps so the external system can design the measurements that close them.

    Every input is validated in full, but only what an experiment is planned
    from is sent: the TZ, the selected routes and the cost ranking. The
    literature review, the free-text economics narrative and the duplicate
    route lists stay in CoScientist.
    """
    inputs = {key: state.get(key) for key in INPUT_KEYS}
    # economics_ranking is checked last (production only), after every other
    # input: its failure alone raises EconomicsRankingError.
    for key in ("structured_tz", "literature_analysis", "synthesis_routes", "qualified_routes"):
        inputs[key] = _object(inputs[key], key)
    LiteratureAnalysis.model_validate(inputs["literature_analysis"])
    proposals = SynthesisRoutes.model_validate(inputs["synthesis_routes"]).routes
    proposal_ids = [route.route_id for route in proposals]
    if not proposals or any(not route_id.strip() for route_id in proposal_ids) or len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("synthesis_routes: nonempty unique route_id values required")
    if any(route.stub or not route.steps for route in proposals):
        raise ValueError("synthesis_routes: real routes with nonempty steps required")
    qualified = QualifiedRoutes.model_validate(inputs["qualified_routes"])
    routes = qualified.routes
    if planning_only and not routes:
        routes = qualified.experimental_routes
    override = None
    if planning_only and not routes and qualified.status == "no_compliant_routes":
        # A human may ask the external system to design evidence-gathering for
        # a rejected proposal.  This does not alter qualification and cannot
        # enter the production branch below.
        override = OperatorRouteOverride.model_validate(
            state.get(OPERATOR_ROUTE_OVERRIDE_KEY)
        )
        proposal_by_id = {route.route_id: route for route in proposals}
        unknown = set(override.route_ids) - set(proposal_by_id)
        if unknown:
            raise ValueError(
                "operator_route_override.route_ids must be a subset of synthesis_routes: "
                f"{sorted(unknown)}"
            )
        routes = [proposal_by_id[route_id] for route_id in override.route_ids]
    ids = [route.route_id for route in routes]
    if not routes or not set(ids).issubset(set(proposal_ids)):
        mode = "eligible or experimental" if planning_only else "eligible"
        raise ValueError(f"qualified_routes: nonempty {mode} subset of synthesis_routes required")
    handoff = {"tz": _tz(inputs["structured_tz"]), "routes": [_route(route) for route in routes]}
    if not handoff["tz"]:
        raise ValueError("structured_tz: original_request or an answered field required")
    if planning_only:
        handoff = {"handoff_mode": "screening", **handoff}
        if override is not None:
            handoff[OPERATOR_ROUTE_OVERRIDE_KEY] = _kept(
                override.model_dump(), ("rationale", "operator_feedback"))
    else:
        try:
            handoff["economics_ranking"] = validate_economics_ranking(inputs["economics_ranking"], ids)
        except ValueError as exc:
            raise EconomicsRankingError(str(exc), ids) from exc
    # Snapshot with no references to mutable session data. Reject NaN in inputs.
    return json.loads(json.dumps(handoff, ensure_ascii=False, allow_nan=False))


def validate_economics_ranking(value: Any, route_ids: list[str]) -> dict:
    """The ranking contract of the production hand-off; returns the ranking.

    One row per qualified route; ``ok`` / ``partial`` rows carry a unique
    positive rank and finite nonnegative costs in the preferred currency;
    ``invalid`` / ``unpriceable`` rows are kept unranked, with their reasons.
    """
    ranking = _object(value, "economics_ranking")
    ranked = ranking.get("routes")
    if not isinstance(ranked, dict) or set(ranked) != set(route_ids):
        raise ValueError(
            "economics_ranking.routes must match qualified_routes route_id values exactly: "
            f"expected {sorted(route_ids)}, got "
            f"{sorted(ranked) if isinstance(ranked, dict) else type(ranked).__name__}"
        )
    _number(ranking.get("target_qty"), "economics_ranking.target_qty", positive=True)
    if ranking.get("target_unit") not in TARGET_UNITS:
        raise ValueError("economics_ranking.target_unit must be g, kg, mol or mmol")
    if ranking.get("rank_by") not in RANK_BY:
        raise ValueError("economics_ranking.rank_by must be per_unit or packs")
    currency = ranking.get("preferred_currency")
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("economics_ranking.preferred_currency required")
    usable_ranks = []
    for route_id, row in ranked.items():
        if not isinstance(row, dict) or row.get("stub"):
            raise ValueError(f"{route_id}: real economics result required")
        status = row.get("status")
        if status not in ROUTE_STATUSES:
            raise ValueError(f"{route_id}: unsupported economics status {status!r}")
        if status in {"invalid", "unpriceable"}:
            continue  # Preserve excluded routes and their reasons for the remote system.
        rank = row.get("rank")
        if type(rank) is not int or rank < 1:
            raise ValueError(f"{route_id}: positive integer rank required")
        if row.get("currency") != currency:
            raise ValueError(f"{route_id}: ranking currencies must match")
        for key in ("cost_per_unit", "cost_packs"):
            _number(row.get(key), f"{route_id}.{key}")
        usable_ranks.append(rank)
    if not usable_ranks or len(usable_ranks) != len(set(usable_ranks)):
        raise ValueError(
            "economics_ranking: at least one costed route (status ok/partial) and unique ranks required"
        )
    return ranking
