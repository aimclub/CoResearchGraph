"""Stage 4 tools on the retrosynthesis service (ASKCOS proxy, the ГПН block).

Three agent tools over ``CoScientist/chemical_utils/retrosynthesis.py``:

``retrosynthesis_routes``
    routes to a target molecule. The raw answer is large (46 routes for aspirin,
    ~130 KB, atom maps and templates included), so it is cut to what stage 4
    needs, and each route's steps are put in FORWARD order — the order a chemist
    runs them and the economics server chains them.
``predict_reaction_products``
    forward prediction: what a set of reactants gives.
``classify_reactions``
    reaction class names for reaction SMILES.

What the live service does (tests/fixtures/retrosynthesis/, recorded
17.09.2026): mode "fast" always answers with no routes, so only "balanced"
(~30 s for aspirin) and "deep" are offered; a target without routes comes back
as ``target: null, routes: []``; the service gives no conditions and no yields.

The client is synchronous (requests): calls run in a worker thread so a tree
search does not block the event loop the other agents share.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODES = ("balanced", "deep")
_TIMEOUTS = {"balanced": 240.0, "deep": 900.0}
MAX_ROUTES = 5
_UNROUTABLE_HOSTS = {"", "0.0.0.0"}


def service_configured() -> bool:
    """True when .env points at a reachable host (0.0.0.0 is a bind address)."""
    from CoScientist.config import get_settings

    hosts = get_settings().hosts_ports
    host = str(hosts.retrosynthesis_services_host or "").strip()
    return host not in _UNROUTABLE_HOSTS and bool(hosts.retrosynthesis_services_port)


# ── Shaping the answers ──────────────────────────────────────────────────────

def _molecules(items: Any) -> List[Dict[str, Any]]:
    return [
        {
            "smiles": m.get("smiles"),
            "purchasable": bool(m.get("terminal")),
            "stoichiometry": m.get("stoichiometry") or 1,
        }
        for m in items or [] if isinstance(m, dict)
    ]


def forward_order(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Steps ordered so each one's non-purchasable reactants are made earlier.

    A route from the tree search lists its steps from the target backwards; a
    step that needs an intermediate must come after the step that makes it.
    Anything that cannot be placed (a cycle, a reactant nobody makes) keeps the
    reverse of the service's order.
    """
    remaining = list(steps)
    made: set = set()
    ordered: List[Dict[str, Any]] = []
    while remaining:
        ready = [
            s for s in remaining
            if all(r["purchasable"] or r["smiles"] in made for r in s["reactants"])
        ]
        if not ready:
            ordered.extend(reversed(remaining))
            break
        for step in ready:
            ordered.append(step)
            made.update(p["smiles"] for p in step["products"])
            remaining.remove(step)
    return ordered


def compact_route(route: Dict[str, Any], target: str = "") -> Dict[str, Any]:
    steps = [
        {
            "reaction_smiles": s.get("reaction_smiles"),
            "plausibility": s.get("plausibility"),
            "template_examples": (s.get("template") or {}).get("num_examples"),
            "reactants": _molecules(s.get("reactants")),
            "products": _molecules(s.get("products")),
        }
        for s in route.get("steps") or [] if isinstance(s, dict)
    ]
    ordered = forward_order(steps)
    made = {p["smiles"] for s in ordered for p in s["products"]}
    starting = []
    for step in ordered:
        for r in step["reactants"]:
            if r["smiles"] not in made and r["smiles"] not in [m["smiles"] for m in starting]:
                starting.append({"smiles": r["smiles"], "purchasable": r["purchasable"]})
    for step in ordered:
        for p in step["products"]:
            p.pop("purchasable", None)
    source_route_id = str(route.get("id") or "route")
    target_key = hashlib.sha256(str(target or "unknown").encode("utf-8")).hexdigest()[:10].upper()
    safe_source_id = "".join(ch if ch.isalnum() else "-" for ch in source_route_id).strip("-") or "ROUTE"
    return {
        # ASKCOS restarts numbering at route_1 for every target.  A target-derived
        # prefix makes the ID globally unique before an LLM ever sees it.
        "route_id": f"GPN-{target_key}-{safe_source_id.upper()}",
        "source_route_id": source_route_id,
        "depth": route.get("depth"),
        "precursor_cost": route.get("precursor_cost"),
        "min_step_plausibility": route.get("min_step_plausibility"),
        "all_starting_materials_purchasable": all(m["purchasable"] for m in starting),
        "starting_materials": starting,
        "steps": ordered,
    }


def shape_routes(raw: Any, smiles: str, mode: str, max_routes: int) -> Dict[str, Any]:
    routes = (raw or {}).get("routes") if isinstance(raw, dict) else None
    if not routes:
        return {
            "status": "no_routes",
            "target": smiles,
            "mode": mode,
            "routes": [],
            "message": (
                "The service found no route to this target. For a salt or a "
                "multi-component SMILES try the neutral parent molecule; "
                "otherwise rely on the literature routes and name this gap."
            ),
        }
    return {
        "status": "ok",
        "target": raw.get("target") or smiles,
        "mode": mode,
        "routes_found": len(routes),
        "routes": [compact_route(r, target=smiles) for r in routes[:max_routes] if isinstance(r, dict)],
    }


def _error(exc: Exception) -> Dict[str, Any]:
    return {"status": "error", "message": f"{type(exc).__name__}: {str(exc)[:500]}"}


# ── Tools ────────────────────────────────────────────────────────────────────

async def retrosynthesis_routes(
    smiles: str, mode: str = "balanced", max_routes: int = MAX_ROUTES
) -> Dict[str, Any]:
    """Find synthesis routes to a target molecule (retrosynthesis tree search).

    Args:
        smiles: SMILES of the target — a single neutral molecule works best.
        mode: "balanced" (about half a minute) or "deep" (slower, wider search).
        max_routes: How many of the best routes to return (1-10).

    Returns:
        {"status": "ok" | "no_routes" | "error", "routes": [{route_id, depth,
        precursor_cost, min_step_plausibility, all_starting_materials_purchasable,
        starting_materials, steps (forward order: reaction_smiles, plausibility,
        template_examples, reactants [smiles, purchasable, stoichiometry],
        products)}]}. No conditions and no yields: the service has none.
    """
    from CoScientist.chemical_utils import retrosynthesis as client

    mode = mode if mode in MODES else "balanced"
    max_routes = max(1, min(int(max_routes or MAX_ROUTES), 10))
    try:
        raw = await asyncio.to_thread(
            client.retrosynthesis_result, smiles, mode=mode, max_routes=max_routes,
            timeout=_TIMEOUTS[mode],
        )
    except Exception as exc:  # noqa: BLE001 — the agent reads the error and moves on
        logger.warning("retrosynthesis_routes(%s): %s", smiles, exc)
        return _error(exc)
    return shape_routes(raw, smiles, mode, max_routes)


async def predict_reaction_products(
    reactants: List[str], reagents: str = "", solvent: str = "", top_n: int = 5
) -> Dict[str, Any]:
    """Predict what a set of reactants gives (forward prediction).

    Args:
        reactants: Reactant SMILES, e.g. ["CCCCCCCCCCCCO", "OS(=O)(=O)Cl"].
        reagents: Reagent SMILES, dot-separated; empty if none.
        solvent: Solvent SMILES; empty if none.
        top_n: How many predictions to return (1-10).

    Returns:
        {"status": "ok" | "error", "predictions": [{smiles, score}]} — scores are
        the model's; a confident prediction can still be chemically wrong.
    """
    from CoScientist.chemical_utils import retrosynthesis as client

    try:
        raw = await asyncio.to_thread(
            client.forward_predict_products, list(reactants), reagents=reagents, solvent=solvent,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("predict_reaction_products(%s): %s", reactants, exc)
        return _error(exc)
    top_n = max(1, min(int(top_n or 5), 10))
    return {
        "status": "ok",
        "inputs": raw.get("inputs"),
        "predictions": [
            {"smiles": p.get("smiles"), "score": p.get("score")}
            for p in (raw.get("predictions") or [])[:top_n] if isinstance(p, dict)
        ],
    }


async def classify_reactions(reaction_smiles: List[str]) -> Dict[str, Any]:
    """Name the reaction class of each reaction SMILES ("A.B>>C").

    Args:
        reaction_smiles: Reaction SMILES to classify.

    Returns:
        {"status": "ok" | "error", "classes": [{rank, reaction_name,
        reaction_classname, reaction_superclassname, certainty}]} — the top hits.
    """
    from CoScientist.chemical_utils import retrosynthesis as client

    try:
        raw = await asyncio.to_thread(client.classify_reaction_smiles, list(reaction_smiles), 3)
    except Exception as exc:  # noqa: BLE001
        logger.warning("classify_reactions(%s): %s", reaction_smiles, exc)
        return _error(exc)
    return {
        "status": "ok",
        "classes": [
            {
                "rank": h.get("rank"),
                "reaction_name": h.get("reaction_name"),
                "reaction_classname": h.get("reaction_classname"),
                "reaction_superclassname": h.get("reaction_superclassname"),
                "certainty": h.get("prediction_certainty"),
            }
            for h in (raw.get("result") or []) if isinstance(h, dict)
        ],
    }


TOOLS = (retrosynthesis_routes, predict_reaction_products, classify_reactions)


def tools() -> Optional[list]:
    """The three tools as FunctionTools, or None when the service is not configured."""
    if not service_configured():
        return None
    from google.adk.tools import FunctionTool

    return [FunctionTool(fn) for fn in TOOLS]


__all__ = [
    "MODES",
    "TOOLS",
    "classify_reactions",
    "compact_route",
    "forward_order",
    "predict_reaction_products",
    "retrosynthesis_routes",
    "service_configured",
    "shape_routes",
    "tools",
]
