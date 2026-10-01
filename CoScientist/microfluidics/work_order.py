"""Work Order rules for the microfluidics tools: the risk tier of each tool
and the hints that tell an agent what counts as an assumption for it.

Registered in the core Work Order tables (CoScientist/hitl/work_order_risk.py,
CoScientist/assembly/prompting.py).
"""
from CoScientist.assembly.prompting import register_work_order_hint
from CoScientist.hitl.work_order_risk import Tier, register_tool_tiers

register_tool_tiers({
    # microfluidics: economics server — price lists and name resolution, read only
    "search_reagents_by_name": Tier.READ,
    "get_price": Tier.READ,
    "search_by_structure": Tier.READ,
    "resolve_chemicals": Tier.READ,
    "estimate_synthesis_cost": Tier.READ,
    "rank_routes_by_cost": Tier.READ,
    # A2A messages can start/resume the external system's physical equipment.
    "optimization_start": Tier.SIDE_EFFECT,
    "optimization_get_status": Tier.READ,
    "optimization_provide_input": Tier.SIDE_EFFECT,
    "optimization_approve": Tier.SIDE_EFFECT,
    # The rig campaign block that runs before optimization — the same risk.
    "campaign_start": Tier.SIDE_EFFECT,
    "campaign_get_status": Tier.READ,
    "campaign_provide_input": Tier.SIDE_EFFECT,
    "campaign_approve": Tier.SIDE_EFFECT,
    # Legacy CFD tools remain registered for other profiles, not microfluidics.
    "cfd_list_reactors": Tier.READ,
    "cfd_get_experiment_result": Tier.READ,
    "cfd_list_artifacts": Tier.READ,
    "cfd_run_reactor_experiment": Tier.COMPUTE,
    "cfd_cancel_run": Tier.COMPUTE,
    # microfluidics: retrosynthesis service — the tree search takes shared compute.
    "molecular_design": Tier.COMPUTE,
    "retrosynthesis_routes": Tier.COMPUTE,
    "predict_reaction_products": Tier.READ,
    "classify_reactions": Tier.READ,
    # microfluidics: the chip simulation and the rig. The rig is a stub today,
    # but it stands for physical hardware: its commands are reviewed as such.
    "cfd_mcp_stub": Tier.COMPUTE,
    "rig_mcp_stub": Tier.SIDE_EFFECT,
})

_HINTS = (
    (("economics_mcp",),
     "For the economics server, each of these is a separate atomic assumption: "
     "the target amount and unit of product (g, kg, mol or mmol), the step "
     "yields you use (from the source, or default_yield), strategy (cheapest or "
     "single_supplier), similarity (soft or hard), preferred_currency, and how "
     "solvents and catalysts are counted (amount / overrides, or left out). In "
     "`inputs` list the route ids and every substance you send, by English name "
     'or SMILES (e.g. "Целевое количество продукта — 100 g").'),
    (("retrosynthesis",),
     "For the retrosynthesis service, the molecule forms you send (neutral "
     "parent instead of a salt), the search mode and how many routes you keep "
     "per candidate are assumptions; in `inputs` list every SMILES you send "
     '(e.g. "Для соли ищем маршрут к нейтральной кислоте").'),
    (("cfd_mcp",),
     "For the CFD service, each value you pass is an atomic assumption with its "
     "unit: the reactor id, inlet speed (m/s), concentrations (mol/m3), rate "
     "constant (m3/(mol*s)), temperature (K), turnovers. In `inputs` name the "
     "request_id of each run; a wait between polls is not a step "
     '(e.g. "Константа скорости 1e-3 м3/(моль·с) — из литературы").'),
    (("cfd_mcp_stub",),
     "For the CFD simulation, the channel geometry, flow rates and fluid "
     "properties you pass are assumptions — list them with units "
     '(e.g. "Суммарный расход 0.5 мл/мин").'),
    (("rig_mcp_stub", "microfluidics"),
     "Commands to the microfluidic rig change a physical setup: put each "
     "command (or one experiment point) in its own step, with the setpoints in "
     "`inputs` and the telemetry you expect in `expected_outcome`."),
)

for _keys, _hint in _HINTS:
    register_work_order_hint(_keys, _hint)
