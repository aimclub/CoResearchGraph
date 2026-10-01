"""Registers the microfluidics profile's tools, callbacks, agent classes and
output schemas in the shared assembly registry.

The core bindings (CoScientist/assembly/bindings.py) know nothing about this
case; the profile loads this module through ``plugins:`` in
CoScientist/microfluidics/microfluidics.yaml. As in the core, tool factories
import lazily so that registering does not open MCP sessions.
"""
from __future__ import annotations

from CoScientist.assembly.bindings import _cb
from CoScientist.assembly.registry import REGISTRY, ToolDoc, ToolEntry, ToolLimit

# ── Tools ────────────────────────────────────────────────────────────────────

def _microfluidics():
    from CoScientist.microfluidics.toolsets import microfluidics_toolset_instance
    return microfluidics_toolset_instance


def _microfluidic_economic():
    from CoScientist.microfluidics.toolsets import microfluidic_economic_toolset_instance
    return microfluidic_economic_toolset_instance


def _microfluidic_cfd():
    from CoScientist.microfluidics.toolsets import microfluidic_cfd_toolset_instance
    return microfluidic_cfd_toolset_instance


def _microfluidics_stub_unless(name: str, real_url_setting: str):
    """The stub, only while its real MCP server is not configured.

    The YAML lists both the real toolset and the stub; exactly one of them is
    attached, so the agent never sees a stub next to the service it replaces.
    """
    stub_factory = _microfluidics_stub(name)

    def factory():
        from CoScientist.microfluidics.settings import get_microfluidics_settings

        if getattr(get_microfluidics_settings(), real_url_setting):
            return None
        return stub_factory()

    return factory


def _retrosynthesis():
    from CoScientist.microfluidics.retrosynthesis import tools
    return tools()


def _microfluidics_stub(name: str):
    """Wrap one microfluidics STUB (stages 3–11) as an attachable function tool.

    The external services are not connected yet; only the function BODY in
    CoScientist/microfluidics/stubs.py changes when they are, so the names
    registered below — and the YAML that references them — stay put.
    """
    def factory():
        from google.adk.tools import FunctionTool
        from CoScientist.microfluidics import stubs

        return [FunctionTool(getattr(stubs, name))]

    return factory


def _tz_builder_tool(name: str):
    """Wrap one section tool of the microfluidics ТЗ (tz_builder.py)."""
    def factory():
        from google.adk.tools import FunctionTool
        from CoScientist.microfluidics import tz_builder

        return [FunctionTool(getattr(tz_builder, name))]

    return factory


REGISTRY.register_tool(ToolEntry(
    key="microfluidics",
    factory=_microfluidics,
    optional=True,  # built only when MCP__MICROFLUIDICS_URL is configured
    runtime_resolved=True,  # real MCP server — tool surface comes from it
    docs=(
        ToolDoc(
            name="<microfluidics MCP tools>",
            signature="(varies)",
            purpose=(
                "Tools exposed by the microfluidics MCP server (chip CFD / rig "
                "control) — call them directly."
            ),
        ),
    ),
))


# ── Microfluidics ТЗ (stage 1) ───────────────────────────────────────────────
# TZSpecAgent fills the ТЗ one section per call; both tools keep the ТЗ in
# state["structured_tz"] (see CoScientist/microfluidics/tz_builder.py).

REGISTRY.register_tool(ToolEntry(
    key="fill_tz_section",
    factory=_tz_builder_tool("fill_tz_section"),
    docs=(
        ToolDoc(
            name="fill_tz_section",
            signature="fill_tz_section(section, usage, fields)",
            purpose=(
                "Сохраняет ОДИН раздел ТЗ — следующий по порядку. Отвечает "
                "прогрессом (заполнено k из N) и называет раздел, который нужно "
                "заполнить следующим, с его рекомендуемыми полями."
            ),
            usage=(
                "fields — строки таблицы раздела: "
                '[{"name": ..., "value": ..., "status": ...}, ...]',
                'status "error" — раздел не сохранён; в errors указано, на каком '
                "шаге, в каком разделе и в каком поле (номер и имя) ошибка",
            ),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="fill_agent_fields",
    factory=_tz_builder_tool("fill_agent_fields"),
    docs=(
        ToolDoc(
            name="fill_agent_fields",
            signature="fill_agent_fields(section, fields)",
            purpose=(
                "Заполняет поля, которые оператор оставил пустыми при проверке "
                "ТЗ, — по одному разделу за вызов, в указанном порядке. Статус "
                "«заполнено агентом» ставится автоматически."
            ),
            usage=(
                'fields — [{"name": ..., "value": ...}, ...]: ровно поля, '
                "названные в запросе, с конкретными значениями",
                "вызывай только после сообщения, что оператор оставил поля тебе",
            ),
        ),
    ),
))

# Registered for when an agent is allowed to edit an assembled ТЗ; NOT attached
# to any agent yet — a rewrite after review refills the ТЗ from section 1.
REGISTRY.register_tool(ToolEntry(
    key="edit_tz_section",
    factory=_tz_builder_tool("edit_tz_section"),
    docs=(
        ToolDoc(
            name="edit_tz_section",
            signature="edit_tz_section(section, fields, usage='')",
            purpose=(
                "Переписывает ОДИН раздел уже собранного ТЗ: fields полностью "
                "заменяют прежнюю таблицу раздела, остальные разделы не меняются."
            ),
            usage=(
                "usage можно не передавать — останется прежний",
            ),
        ),
    ),
))

# ── Microfluidics stages 3–11 ────────────────────────────────────────────────
# Stubs for the services behind nodes 3, 5, 9 and 10 (see the design in
# docs/superpowers/specs/2026-07-14-microfluidics-graph-modules-design.md), plus
# finish_optimization — the REAL tool that ends the 7⇄8 optimization loop.

def _molecular_design():
    from CoScientist.microfluidics.molecular_design import molecular_design
    return [molecular_design]


REGISTRY.register_tool(ToolEntry(
    key="molecular_design",
    factory=_molecular_design,
    docs=(
        ToolDoc(
            name="molecular_design",
            signature="molecular_design(requirements)",
            purpose=(
                "Screens literature analogues with RDKit descriptors and explicit TZ "
                "constraints, optionally enumerating BRICS hypotheses. Reads TZ and "
                "literature from session state; saves molecular_design_result. "
                "Returns candidates, criteria_checks, rejected structures and gaps; "
                "unknown properties are not predicted."
            ),
            usage=(
                'requirements is a JSON string: criteria [{name, minimum and/or maximum, '
                'unit, conditions}], required_smarts, forbidden_smarts, generate (bool), '
                'max_candidates (1..30). All fields are optional; {} is valid.',
                'Descriptor names/units: MolWt (g/mol), TPSA (angstrom^2); MolLogP, '
                'HBD, HBA, RotatableBonds, FormalCharge use unit="". Other properties '
                'require matching literature name, unit and measurement conditions.',
                'Use only explicit TZ bounds. status=error/no_candidates means no '
                'usable candidates; do not invent a fallback result.',
            ),
        ),
    ),
))

# The retrosynthesis service (ASKCOS proxy — the ГПН block "ретро / forward /
# классиф."), recorded in tests/fixtures/retrosynthesis/. Plain HTTP, not MCP:
# the tools are async wrappers in microfluidics/retrosynthesis.py.
REGISTRY.register_tool(ToolEntry(
    key="retrosynthesis",
    factory=_retrosynthesis,
    optional=True,  # built only when HOSTS_PORTS__RETROSYNTHESIS_SERVICES_HOST/PORT are set
    docs=(
        ToolDoc(
            name="retrosynthesis_routes",
            signature="retrosynthesis_routes(smiles, mode=\"balanced\", max_routes=5)",
            purpose=(
                "Routes to a target molecule from the retrosynthesis tree search. "
                "Returns status (ok | no_routes | error) and routes[]: route_id, "
                "depth (steps), precursor_cost, min_step_plausibility, "
                "all_starting_materials_purchasable, starting_materials, and steps "
                "in FORWARD order — reaction_smiles, plausibility, "
                "template_examples, reactants (smiles, purchasable, "
                "stoichiometry), products."
            ),
            usage=(
                "mode: \"balanced\" (about 30 s) first; \"deep\" only once, if "
                "balanced found nothing — it is much slower.",
                "Send ONE neutral molecule: a salt or a dotted SMILES often finds "
                "no route — use the parent acid or base.",
                "No conditions and no yields: take them from the literature.",
            ),
        ),
        ToolDoc(
            name="predict_reaction_products",
            signature="predict_reaction_products(reactants, reagents=\"\", solvent=\"\", top_n=5)",
            purpose=(
                "Forward prediction: predictions[] {smiles, score} for a set of "
                "reactant SMILES."
            ),
            usage=(
                "A cross-check, not a verdict: a confident prediction can be wrong "
                "(dodecanol + chlorosulfonic acid came back as dodecanal). A "
                "mismatch with the expected product is a bottleneck to name.",
            ),
        ),
        ToolDoc(
            name="classify_reactions",
            signature="classify_reactions(reaction_smiles)",
            purpose=(
                "Reaction class per reaction SMILES: classes[] {rank, "
                "reaction_name, reaction_classname, reaction_superclassname, "
                "certainty}."
            ),
            usage=("Use it to name a step's operation from its reaction SMILES.",),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="economics_mcp_stub",
    factory=_microfluidics_stub_unless("economics_mcp_stub", "microfluidic_economic_url"),
    optional=True,  # dropped once the real MCP server is configured
    docs=(
        ToolDoc(
            name="economics_mcp_stub",
            signature="economics_mcp_stub(route)",
            purpose=(
                "(ЗАГЛУШКА) Costs a synthesis route: price per kg, reagent "
                "availability in Russia, and supply risks."
            ),
        ),
    ),
))

# The economics server's contract, as recorded in
# tests/unit/microfluidics/fixtures/economics_mcp/ (scripts/microfluidics/mcp_contract_dump.py). Real tool names,
# so a Work Order can check them; the answers arrive as structuredContent.
REGISTRY.register_tool(ToolEntry(
    key="economics_mcp",
    factory=_microfluidic_economic,
    optional=True,  # built only when MCP_MICROFLUIDIC_ECONOMIC is configured
    runtime_resolved=True,  # real MCP server — tool surface comes from it
    docs=(
        ToolDoc(
            name="resolve_chemicals",
            signature="resolve_chemicals(names, use_llm=false)",
            purpose=(
                "Names or SMILES -> structures. Up to 200 per call. Returns "
                "items[]: {input, canonical_smiles, inchikey, formula, molar_mass, "
                "method, confidence, error, hint}; error \"unresolved\" means no "
                "structure, and a route using that name cannot be costed."
            ),
            usage=(
                "Russian trivial names often do NOT resolve (\"додеканол-1\", "
                "\"хлорсульфоновая кислота\" fail); English names and SMILES do "
                "(\"1-dodecanol\", \"chlorosulfonic acid\"). Send English names or "
                "SMILES.",
                "A string that reads both as a name and as SMILES (CO, NO, Br) is "
                "refused: write it as a SMILES object or a full name.",
            ),
        ),
        ToolDoc(
            name="rank_routes_by_cost",
            signature=(
                "rank_routes_by_cost(routes, target_qty, target_unit, default_yield=1.0, "
                "strategy=\"cheapest\", similarity=\"soft\", preferred_currency=\"RUB\", "
                "rank_by=\"per_unit\", include_breakdown=true)"
            ),
            purpose=(
                "Costs synthesis routes for ONE target amount and ranks them "
                "cheapest first. Returns routes[]: {route_id, status (ok | partial | "
                "invalid | unpriceable), rank, currency, cost_per_unit, cost_packs, "
                "starting_materials[] (smiles, qty, unit, moles), intermediates, "
                "missing[] (smiles, reason), steps_smiles, resolved_inputs, warnings, "
                "estimate (per-reagent line_items with the chosen offer)}, plus "
                "assumptions."
            ),
            usage=(
                "Route: {route_id, steps, target_smiles?, target_name?, overrides?}. "
                "Step: {reactants: [...], agents: [...], products: [...], conditions, "
                "yield (0-1)} — substances as {\"smiles\": ...} or English names; "
                "\"@prev\" in reactants is the previous step's product (never in the "
                "first step). Or a reaction SMILES string.",
                "target_unit is g, kg, mol or mmol (not a volume). default_yield 1.0 "
                "means no losses — pass real yields.",
                "Only starting materials are bought; agents (solvents, catalysts) are "
                "NOT in the sum unless given an amount or an override.",
                "partial = a lower bound (missing items are not priced). invalid = a "
                "name did not resolve (see warnings) — fix the name, run again.",
                "cost_per_unit is the cost of the quantity used; cost_packs is the "
                "real bill for whole packs. Compare costs only within one currency.",
            ),
        ),
        ToolDoc(
            name="estimate_synthesis_cost",
            signature=(
                "estimate_synthesis_cost(reagents, strategy=\"cheapest\", "
                "similarity=\"hard\", preferred_currency=\"RUB\")"
            ),
            purpose=(
                "Costs a plain list of reagents {smiles | name, qty, unit, "
                "purity_min?}. Returns line_items[] (match_level, chosen {supplier, "
                "name_raw, pack_qty, pack_unit, price, price_currency, unit_price}, "
                "packs_needed, cost_packs, cost_per_unit), total_by_currency, "
                "resolved_inputs, missing[]."
            ),
            usage=(
                "Use it for a set of reagents without a route (a solvent, a "
                "catalyst, a recheck). With similarity=hard any miss is an error "
                "no_exact_match; soft allows the nearest structure.",
            ),
        ),
        ToolDoc(
            name="get_price",
            signature="get_price(name, pack_unit=null, purity_grade=null, limit=50)",
            purpose=(
                "All offers for a fuzzy-matched name, cheapest per unit first: "
                "result[] {name_raw, supplier, purity_grade, pack_qty, pack_unit, "
                "price, price_currency, unit_price, price_basis}."
            ),
            usage=(
                "Explains a missing or suspicious item: read name_raw — a match may "
                "be a solution (\"0,1Н\"), another grade or a neighbour by name "
                "(\"ацетон\" also matches \"ацетилацетон\").",
            ),
        ),
        ToolDoc(
            name="search_by_structure",
            signature="search_by_structure(smiles=null, name=null, mode=\"exact\", limit=50)",
            purpose=(
                "Offers by structure (give smiles OR name): per compound the cheapest "
                "offer, with match_level (точный, по_связности, подструктура)."
            ),
            usage=("mode=substructure finds molecules containing the fragment.",),
        ),
        ToolDoc(
            name="search_reagents_by_name",
            signature="search_reagents_by_name(query, limit=20)",
            purpose=(
                "Fuzzy search by Russian name, one row per reagent: result[] "
                "{name_norm, supplier, best_price, price_currency, unit_price, "
                "pack_qty, pack_unit, offer_count, similarity}."
            ),
            usage=("For finding how a reagent is listed in the price lists.",),
        ),
    ),
))

# The CFD service's contract, as recorded in tests/unit/microfluidics/fixtures/cfd_mcp/
# (scripts/microfluidics/mcp_contract_dump.py). A run is asynchronous: it blocks up to
# wait_seconds, then the result is fetched by the caller-chosen request_id.
REGISTRY.register_tool(ToolEntry(
    key="cfd_mcp",
    factory=_microfluidic_cfd,
    optional=True,  # built only when MCP_MICROFLUIDIC_CFD_3_TOOLS is configured
    runtime_resolved=True,  # real MCP server — tool surface comes from it
    docs=(
        ToolDoc(
            name="cfd_list_reactors",
            signature="cfd_list_reactors()",
            purpose=(
                "The reactor geometries: reactors[] {reactor (the id to run), title, "
                "description, available, zones {inlets, outlets, inlet_count, "
                "supports_two_reactant_feed}}."
            ),
            usage=(
                "An A + B reaction needs supports_two_reactant_feed=true; "
                "available=false means the mesh is not on the host.",
            ),
        ),
        ToolDoc(
            name="cfd_run_reactor_experiment",
            signature=(
                "cfd_run_reactor_experiment(reactor, inlet_speed_m_per_s, "
                "concentration_a_mol_per_m3, concentration_b_mol_per_m3, "
                "rate_constant_m3_per_mol_s, temperature_k=298.15, turnovers=4, "
                "wait_seconds=600, request_id)"
            ),
            purpose=(
                "Runs flow -> mixing -> reaction on one reactor for A + B -> C and "
                "returns status (succeeded | failed | cancelled | pending), design "
                "(the parameters used), derived.operating_point (flow rate, "
                "residence_time_s_estimate), results (pressure_drop_pa, "
                "residence_time_s, damkohler, conversion_by_reactant, outlet "
                "concentrations, mixing index), trustworthy, blocking, stages[] "
                "(exit_code per stage) and error {code, message}."
            ),
            usage=(
                "Always pass your own request_id (e.g. \"exp1-cfd-1\"): reusing it "
                "returns the same run instead of starting another; a FAILED run "
                "reused starts afresh.",
                "Units: concentrations in mol/m3 (1 mol/L = 1000 mol/m3), rate "
                "constant in m3/(mol*s), temperature in K, inlet speed in m/s "
                "(flow rate = speed x inlet area).",
                "status pending: the call stopped waiting, the run goes on — fetch it "
                "with cfd_get_experiment_result. The service runs ONE experiment at a "
                "time: error capacity_exceeded means wait and retry the same request_id.",
                "A result is valid only with blocking empty; a run under 3 residence "
                "times reports near-zero conversion that means \"still filling\" — "
                "raise turnovers. Aim for damkohler 1-10; do not tune the diffusivity.",
            ),
        ),
        ToolDoc(
            name="cfd_get_experiment_result",
            signature="cfd_get_experiment_result(request_id)",
            purpose=(
                "The run by request_id, same payload as the run call. pending — "
                "still running; succeeded, failed, cancelled — final."
            ),
            usage=(
                "Between polls of a pending run wait with sleep_tool — do not poll "
                "in a tight loop.",
            ),
        ),
        ToolDoc(
            name="cfd_list_artifacts",
            signature="cfd_list_artifacts(request_id, offset=0, limit=50)",
            purpose=(
                "Files the run produced: keys[] are paths on the CFD host, not "
                "links — cite their count, not the paths."
            ),
        ),
        ToolDoc(
            name="cfd_cancel_run",
            signature="cfd_cancel_run(request_id)",
            purpose="Stops a run in progress (only your own, only if it is no longer needed).",
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="cfd_mcp_stub",
    factory=_microfluidics_stub_unless("cfd_mcp_stub", "microfluidic_cfd_url"),
    optional=True,  # dropped once the real MCP server is configured
    docs=(
        ToolDoc(
            name="cfd_mcp_stub",
            signature="cfd_mcp_stub(geometry, flow)",
            purpose=(
                "(ЗАГЛУШКА) Simulates the flow in the chip (CFD): pressure "
                "drop, mixing efficiency, residence time."
            ),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="rig_mcp_stub",
    factory=_microfluidics_stub("rig_mcp_stub"),
    docs=(
        ToolDoc(
            name="rig_mcp_stub",
            signature="rig_mcp_stub(command)",
            purpose=(
                "(ЗАГЛУШКА) Sends a command to the microfluidic rig and reads "
                "back its status and telemetry."
            ),
        ),
    ),
))

def _optimization_a2a():
    from CoScientist.microfluidics.a2a_optimization.adapter import (
        optimization_start, optimization_get_status,
        optimization_provide_input, optimization_approve,
    )
    return [optimization_start, optimization_get_status, optimization_provide_input, optimization_approve]


def _campaign_a2a():
    from CoScientist.microfluidics.a2a_optimization.campaign import (
        campaign_start, campaign_get_status, campaign_provide_input, campaign_approve,
    )
    return [campaign_start, campaign_get_status, campaign_provide_input, campaign_approve]


def _hitl_before_campaign_start():
    from CoScientist.agents.common import hitl_handler
    from CoScientist.hitl.callbacks import make_hitl_before_tool_callback
    return make_hitl_before_tool_callback(
        hitl_handler, target_tools=("campaign_start",), require_hitl=True,
    )


def _operator_screening_override():
    from CoScientist.microfluidics.operator_override import operator_authorize_screening_override
    return [operator_authorize_screening_override]


REGISTRY.register_tool(ToolEntry(
    key="optimization_a2a",
    factory=_optimization_a2a,
    docs=(
        ToolDoc(
            name="optimization_start",
            signature="optimization_start(planning_only=False)",
            purpose="Delegate the whole CFD/equipment/optimization workflow with validated TZ, literature, routes and ranking. Reuses the existing task even after completion. planning_only=True forbids execution.",
        ),
        ToolDoc(
            name="optimization_get_status",
            signature="optimization_get_status()",
            purpose="Poll the same A2A task and preserve all raw responses/artifacts. input_required is a request for clarification or approval, not completion.",
        ),
        ToolDoc(
            name="optimization_provide_input",
            signature="optimization_provide_input(details)",
            purpose="Reply to waiting_input with known facts or user clarification, preserving task/context/experiment IDs. Never invent parameters.",
        ),
        ToolDoc(
            name="optimization_approve",
            signature="optimization_approve()",
            purpose="Approve the external plan at input_required/approval within the authorized work order. May start remote equipment. Unavailable in planning-only mode.",
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="campaign_a2a",
    factory=_campaign_a2a,
    docs=(
        ToolDoc(
            name="campaign_start",
            signature="campaign_start()",
            purpose="Start the rig campaign of the flow-synthesis condition-optimization block with the validated TZ, routes and cost ranking (the same hand-off as optimization_start). Reuses the existing task even after completion.",
        ),
        ToolDoc(
            name="campaign_get_status",
            signature="campaign_get_status()",
            purpose="Poll the same campaign task; its campaign_result is normalized into state optimization (current/history/has_blockers). input_required is a request for clarification or approval, not completion.",
        ),
        ToolDoc(
            name="campaign_provide_input",
            signature="campaign_provide_input(details)",
            purpose="Reply to waiting_input with known facts or user clarification in the same campaign task. Never invent parameters.",
        ),
        ToolDoc(
            name="campaign_approve",
            signature="campaign_approve()",
            purpose="Approve the campaign plan at input_required/approval after the operator agrees. May start the physical rig.",
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="operator_screening_override",
    factory=_operator_screening_override,
    docs=(
        ToolDoc(
            name="operator_authorize_screening_override",
            signature="operator_authorize_screening_override(route_ids, rationale)",
            purpose=(
                "Ask the human operator to authorize a planning-only verification "
                "handoff for real routes rejected by automatic qualification. "
                "It cannot make a route eligible or authorize equipment execution."
            ),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="finish_optimization",
    factory=_microfluidics_stub("finish_optimization"),
    docs=(
        ToolDoc(
            name="finish_optimization",
            signature="finish_optimization(reason)",
            purpose=(
                "Ends the experiment optimization loop and moves on to the "
                "report. Call it once the plan needs no further refining."
            ),
        ),
    ),
))


# ── Callbacks ────────────────────────────────────────────────────────────────

def _microfluidics_evidence(name: str):
    def factory(ctx):
        from CoScientist.microfluidics import evidence
        return getattr(evidence, name)
    return factory


def _save_tz_document():
    from CoScientist.microfluidics.tz_agent import save_tz_document
    return save_tz_document


def _microfluidics_literature(name: str):
    def factory(ctx):
        from CoScientist.microfluidics import literature
        return getattr(literature, name)
    return factory


def _use_fixed_target_molecule():
    from CoScientist.microfluidics.design import use_fixed_target_molecule
    return use_fixed_target_molecule


def _use_selected_route_product():
    from CoScientist.microfluidics.design import use_selected_route_product
    return use_selected_route_product


def _collect_cfd_result():
    from CoScientist.microfluidics.cfd import collect_cfd_result
    return collect_cfd_result


def _collect_economics_result():
    from CoScientist.microfluidics.economics import collect_economics_result
    return collect_economics_result


def _microfluidics_requirements():
    from CoScientist.microfluidics.requirements import compile_requirements_callback
    return compile_requirements_callback


def _microfluidics_research_record(name: str):
    def factory(ctx):
        from CoScientist.microfluidics import research_record
        return getattr(research_record, name)
    return factory


def _publish_literature_summary():
    from CoScientist.microfluidics.literature_report import publish_literature_summary
    return publish_literature_summary


def _microfluidics_route_compliance(name: str):
    from CoScientist.microfluidics import route_compliance
    return getattr(route_compliance, name)


def _normalize_literature_identities():
    from CoScientist.microfluidics.chemistry_identity import normalize_literature_identities
    return normalize_literature_identities


def _export_tz_and_queries():
    from CoScientist.microfluidics.export import export_tz_and_queries
    return export_tz_and_queries




_cb("begin_evidence_verification", "before_agent",
    factory=_microfluidics_evidence("begin_evidence_verification"))
_cb("capture_evidence_verification", "after_tool",
    factory=_microfluidics_evidence("capture_evidence_verification"))
_cb("authenticate_evidence_verification", "after_agent",
    factory=_microfluidics_evidence("authenticate_evidence_verification"))
_cb("normalize_literature_identities", "after_agent",
    factory=lambda ctx: _normalize_literature_identities())

# Render the approved ТЗ into the reference Markdown document (state + file).
_cb("save_tz_document", "after_agent", factory=lambda ctx: _save_tz_document())
# Save the ТЗ + literature queries as shareable Markdown & HTML for hand-off.
_cb("export_tz_and_queries", "after_agent", factory=lambda ctx: _export_tz_and_queries())

# Microfluidics module A: keep EVERY LIT-xx answer (search_results is
# overwritten per call), expose the ТЗ's target molecule as its own key, and
# let a fixed molecule from the ТЗ win in the literature analysis.
_cb("collect_literature_finding", "after_agent",
    factory=_microfluidics_literature("collect_literature_finding"))
_cb("inject_target_molecule", "before_agent",
    factory=_microfluidics_literature("inject_target_molecule"))
_cb("pin_target_molecule", "after_agent",
    factory=_microfluidics_literature("pin_target_molecule"))
_cb("assemble_selected_literature", "after_agent",
    factory=_microfluidics_literature("assemble_selected_literature"))
# Compile the approved human-readable TZ into the only requirements contract
# downstream chemistry stages may use.
_cb("compile_requirements", "before_agent", factory=lambda ctx: _microfluidics_requirements())
# Stage 4 proposals are assessed in code; stage 5 is skipped/guarded unless all
# hard constraints passed.
_cb("qualify_synthesis_routes", "after_agent",
    factory=lambda ctx: _microfluidics_route_compliance("qualify_synthesis_routes"))
_cb("gate_economics", "before_agent",
    factory=lambda ctx: _microfluidics_route_compliance("gate_economics"))
_cb("review_incomplete_economics_routes", "before_agent",
    factory=lambda ctx: _microfluidics_route_compliance("review_incomplete_economics_routes"))
_cb("review_preliminary_economics", "before_agent",
    factory=lambda ctx: _microfluidics_route_compliance("review_preliminary_economics"))
_cb("guard_economics_routes", "before_tool",
    factory=lambda ctx: _microfluidics_route_compliance("guard_economics_routes"))
# The rig campaign is sent only after the operator approves it in the web
# interface; with HITL switched off the call is blocked, not run unreviewed.
_cb("hitl_before_campaign_start", "before_tool",
    factory=lambda ctx: _hitl_before_campaign_start())
# Microfluidics module B: keep the economics server's costing answers as given.
_cb("collect_economics_result", "after_tool", factory=lambda ctx: _collect_economics_result())
# Stage 3: a molecule fixed in the ТЗ is handed on as the only candidate — no design.
_cb("use_fixed_target_molecule", "before_agent", factory=lambda ctx: _use_fixed_target_molecule())
_cb("use_selected_route_product", "before_agent",
    factory=lambda ctx: _use_selected_route_product())
# Stage 9: keep the CFD service's run results as given, under their request ids.
_cb("collect_cfd_result", "after_tool", factory=lambda ctx: _collect_cfd_result())
# Microfluidics → research graph (the scientific-process record). Each stage
# writes its STRUCTURED result into the typed research graph from code
# (microfluidics/research_record.py): ТЗ → ResearchQuestion + Constraints +
# Tools + literature VerificationMethods; literature → Evidence; candidates →
# Hypotheses; routes → VerificationMethods; economics / optimisation →
# Evidence; report → Conclusion + Report. Best-effort, never breaks a run.
# They MUST precede any after_agent callback that returns chat Content
# (export_tz_and_queries, publish_literature_summary): ADK stops the chain at
# the first Content.
for _name in ("record_research_question", "record_literature_evidence",
              "record_design_hypotheses", "record_synthesis_routes",
              "record_economics_evidence", "record_experiment_evidence",
              "record_conclusion"):
    _cb(_name, "after_agent", factory=_microfluidics_research_record(_name))
# The literature agent's own deliverable for the report: summary + tables
# rendered from literature_analysis into state["literature_markdown"] and
# posted to the chat.
_cb("publish_literature_summary", "after_agent",
    factory=lambda ctx: _publish_literature_summary())


# ── Tool-call budgets ────────────────────────────────────────────────────────

# Calls per tool; Settings → Agents can set another budget for each agent.
PER_TOOL_CALLS = 2
EVIDENCE_VERIFIER_CALLS = 10


def _per_tool_call_limiter(ctx):
    from CoScientist.agents.callbacks.tool_callbacks import PerToolCallLimiter
    from CoScientist.assembly.schema import agent_limit
    max_calls = agent_limit(ctx.config.name, PER_TOOL_CALLS)
    # PaperRetriever's full-text analysis needs a third pass; ResearchAgent is
    # forbidden explore_my_papers, so the override never widens its budget.
    return PerToolCallLimiter(
        max_calls=max_calls, per_tool={"explore_my_papers": max(3, max_calls)}
    ).limit_tool_calls


def _evidence_verifier_tool_limiter(ctx):
    from CoScientist.agents.callbacks.tool_callbacks import PerToolCallLimiter
    from CoScientist.assembly.schema import agent_limit
    return PerToolCallLimiter(
        max_calls=agent_limit(ctx.config.name, EVIDENCE_VERIFIER_CALLS)
    ).limit_tool_calls


# ResearchAgent budget: two calls per concrete tool (three for explore_my_papers)
# and per delegated agent branch, so parallel LIT-* tasks never share a counter.
_cb("PerToolCallLimiter", "before_tool", factory=_per_tool_call_limiter,
    limit=ToolLimit(kind="perTool", default=lambda: PER_TOOL_CALLS))
_cb("EvidenceVerifierToolLimiter", "before_tool", factory=_evidence_verifier_tool_limiter,
    limit=ToolLimit(kind="perTool", default=lambda: EVIDENCE_VERIFIER_CALLS))


# ── Agent classes / output schemas ───────────────────────────────────────────

def _register_classes() -> None:
    from CoScientist.microfluidics.a2a_optimization.session_agent import (
        CampaignSessionAgent,
        OptimizationSessionAgent,
    )
    from CoScientist.microfluidics.design import MoleculeSelectionAgent
    from CoScientist.microfluidics.route_selection import RouteSelectionSessionAgent
    from CoScientist.microfluidics.tz_agent import TZSessionAgent

    # ТЗ stage: the review loop shows the RENDERED ТЗ document.
    REGISTRY.register_agent_class("tz_session", TZSessionAgent)
    REGISTRY.register_agent_class("optimization_session", OptimizationSessionAgent)
    REGISTRY.register_agent_class("campaign_session", CampaignSessionAgent)
    REGISTRY.register_agent_class("route_selection_session", RouteSelectionSessionAgent)
    # Stage 7 without a model: the ТЗ's molecule or the chosen route's product
    # (microfluidics/design.py).
    REGISTRY.register_agent_class("molecule_selection", MoleculeSelectionAgent)


def _register_schemas() -> None:
    from CoScientist.microfluidics.models import (
        DesignCandidates,
        LiteratureAnalysis,
        LiteratureQueries,
        LiteratureSelection,
        RouteSelection,
        StructuredTZ,
        SynthesisRoutes,
    )

    # Structured ТЗ and the literature queries derived from it.
    REGISTRY.register_output_schema("structured_tz", StructuredTZ)
    REGISTRY.register_output_schema("tz_literature_queries", LiteratureQueries)
    REGISTRY.register_output_schema("literature_analysis", LiteratureAnalysis)
    REGISTRY.register_output_schema("literature_selection", LiteratureSelection)
    REGISTRY.register_output_schema("route_selection", RouteSelection)
    # Module B hand-off: candidates and routes in the shape the economics server costs.
    REGISTRY.register_output_schema("design_candidates", DesignCandidates)
    REGISTRY.register_output_schema("synthesis_routes", SynthesisRoutes)


_register_classes()
_register_schemas()
