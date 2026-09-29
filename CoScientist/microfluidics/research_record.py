"""Deterministic recorders of the microfluidics pipeline into the research graph.

The typed Research Context Graph (``CoScientist.graph.research``) is normally
written by LLM agents through ``research_commit`` under a per-agent ACL. The
microfluidics pipeline has no such agents: its stages return STRUCTURED state
(``structured_tz``, ``literature_analysis``, ``design_candidates``,
``synthesis_routes``, ``economics_ranking``, ``optimization_result``,
``final_report``), so the scientific process is recorded from that state by
code — one ``after_agent`` callback per stage, each a privileged
:meth:`ResearchGraphStore.record` attributed to the stage's agent.

What ends up in the graph (the "science in the record" view):

    ResearchQuestion (ТЗ) ← contextualizes ← Constraint × N (ТЗ blocks)
      ├─ defines_scope → EmpiricalBase (corpus of literature sources)
      ├─ motivates → Hypothesis × N (design candidates)
      │     └─ tested_by → VerificationMethod × N (synthesis routes)
      │           ├─ uses → Tool (retrosynthesis / chemquote / CFD / A2A rig)
      │           └─ produces → Evidence (costing, optimisation / experiment)
      ├─ relates_to ← Evidence (literature: analogues, routes, facts)
      │     ← produces ← VerificationMethod (literature search LIT-xx)
      └─ produces ← Conclusion (final report) ← derived_from ← Report

Node ids created by earlier stages are kept in ``state["research_record"]``
so later stages can attach to them. Every recorder is best-effort: a graph
failure is logged and never breaks the run.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.agents.callback_context import CallbackContext

logger = logging.getLogger(__name__)

RECORD_KEY = "research_record"
REPORTS_DIR = Path("reports")

def _tools() -> List[Dict[str, Any]]:
    """Tool nodes for the research star — only the services this run can use.

    ``tool_type`` is the key later stages attach to (uses edges). A service
    that is not configured (e.g. retrosynthesis, excluded from the runs) is
    not listed, so the graph shows what the pipeline really ran on.
    """
    try:
        from CoScientist.config import get_settings
        mcp = get_settings().mcp
        from CoScientist.microfluidics.retrosynthesis import service_configured
        retro = service_configured()
    except Exception:  # noqa: BLE001
        mcp, retro = None, False
    tools: List[Dict[str, Any]] = []
    if mcp is not None and (mcp.paper_analysis_url or mcp.papers_search_url):
        tools.append({"name": "PaperAnalysis RAG (paper_analysis / papers_search)",
                      "tool_type": "literature", "status": "available",
                      "location": mcp.paper_analysis_url or mcp.papers_search_url or ""})
    if retro:
        tools.append({"name": "Retrosynthesis Proxy API (ASKCOS)", "tool_type": "retrosynthesis",
                      "status": "available", "location": "HOSTS_PORTS__RETROSYNTHESIS_SERVICES_*"})
    from CoScientist.microfluidics.settings import get_microfluidics_settings

    economic_url = get_microfluidics_settings().microfluidic_economic_url
    if economic_url:
        tools.append({"name": "chemquote (реагенты, прайс-листы РФ)", "tool_type": "economics",
                      "status": "available", "location": economic_url})
    tools.append({"name": "Модуль оптимизации / CFD / установка (A2A)", "tool_type": "experiment",
                  "status": "available", "location": "OPTIMIZATION_A2A_URL"})
    return tools


# ── helpers ──────────────────────────────────────────────────────────────────

def _graph(callback_context: Any):
    """The session's research graph, or None when the feature is off."""
    try:
        from CoScientist.config import get_settings
        if not get_settings().research_graph.enabled:
            return None
        from CoScientist.graph.research.store import get_research_graph
        return get_research_graph(callback_context)
    except Exception as exc:  # noqa: BLE001 — recording must never break a run
        logger.warning("research_record: graph unavailable: %s", exc)
        return None


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _short(value: Any, n: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _record(state: Any) -> Dict[str, Any]:
    rec = state.get(RECORD_KEY)
    return dict(rec) if isinstance(rec, dict) else {}


def _save_record(state: Any, rec: Dict[str, Any]) -> None:
    # Written whole: AgentTool forwards only whole-key deltas to the parent.
    state[RECORD_KEY] = rec


def _ids_by_ref(result: Any) -> Dict[str, str]:
    committed = getattr(result, "committed", None) or {}
    return {n["ref"]: n["id"] for n in committed.get("nodes", []) if n.get("ref") and n.get("id")}


def _write(graph: Any, source: str, nodes: List[Dict[str, Any]],
           edges: Optional[List[Dict[str, Any]]] = None,
           status_updates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
    """Privileged commit; returns {ref: id} of created nodes ({} on failure)."""
    if not (nodes or edges or status_updates):
        return {}
    result = graph.record(source, nodes=nodes, edges=edges or [],
                          status_updates=status_updates or [])
    if not result.ok:
        logger.warning("research_record[%s]: commit rejected: %s",
                       source, "; ".join(result.errors))
        return {}
    for w in result.warnings:
        logger.info("research_record[%s]: %s", source, w)
    return _ids_by_ref(result)


def _set_status(graph: Any, source: str, node_id: Optional[str], status: str,
                reason: str = "") -> None:
    """A status change in its own transaction, so a refused transition never
    discards the nodes committed alongside it."""
    if not node_id:
        return
    upd: Dict[str, Any] = {"id": node_id, "status": status}
    if reason:
        upd["reason"] = reason
    result = graph.record(source, status_updates=[upd])
    if not result.ok:
        logger.info("research_record[%s]: status %s→%s refused: %s",
                    source, node_id, status, "; ".join(result.errors))


def _advance(graph: Any, source: str, node_id: Optional[str], *statuses: str) -> None:
    """Walk a node through several statuses in order (VerificationMethod only
    accepts planned → running → done, one step per transaction)."""
    for status in statuses:
        _set_status(graph, source, node_id, status)


def _evidence_ids(rec: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("literature_evidence", "economics_evidence", "experiment_evidence"):
        out.extend(v for v in (rec.get(key) or []) if isinstance(v, str))
    return out


def _tz_blocks(tz: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [b for b in (tz.get("blocks") or []) if isinstance(b, dict)]


def _set_fields(block: Dict[str, Any]) -> List[str]:
    from CoScientist.hitl.field_status import OPEN_STATUSES
    rows = []
    for f in block.get("fields") or []:
        if not isinstance(f, dict):
            continue
        if f.get("status") in OPEN_STATUSES:
            continue
        value = str(f.get("value") or "").strip()
        if value and value.lower() not in ("не задано", "не требуется"):
            rows.append(f"{f.get('name')}: {value}")
    return rows


def _tz_field(tz: Dict[str, Any], *needles: str) -> str:
    """First set value in a field whose name contains any needle (lower-case)."""
    for block in _tz_blocks(tz):
        for f in block.get("fields") or []:
            name = str(f.get("name") or "").lower()
            if any(n in name for n in needles):
                value = str(f.get("value") or "").strip()
                if value and value.lower() != "не задано":
                    return value
    return ""


def _question(tz: Dict[str, Any], fallback: str) -> str:
    product = _tz_field(tz, "целев", "продукт", "вещество")
    task = _tz_field(tz, "тип задачи", "задача")
    if product:
        head = f"Получить {product}" if not task else f"{task}: {product}"
        return _short(head, 400)
    return _short(fallback or "Кейс микрофлюидики", 400)


# ── stage 1–2: ТЗ + literature queries → ResearchQuestion star ───────────────

def record_research_question(callback_context: CallbackContext) -> None:
    """After TZQueryGenAgent: seed the research from the approved ТЗ.

    ResearchQuestion + one Constraint per filled ТЗ block + the service Tools +
    one planned VerificationMethod per literature query (LIT-xx)."""
    state = callback_context.state
    tz = _as_dict(state.get("structured_tz"))
    if not tz:
        return None
    graph = _graph(callback_context)
    if graph is None:
        return None
    try:
        constraints = []
        for block in _tz_blocks(tz):
            rows = _set_fields(block)
            if not rows:
                continue
            constraints.append({
                "subtype": "domain_standards",
                "content": f"{block.get('title')}: " + "; ".join(rows),
                "source": "human",
            })
        tools = _tools()
        result = graph.init_research(
            "TZSpecAgent",
            _question(tz, tz.get("original_request", "")),
            attrs={
                "domain": "микрофлюидика / проточный синтез",
                "research_form": "applied",
                "target_setting": _short(tz.get("original_request", ""), 600),
                "completion_criteria": "pragmatic",
            },
            constraints=constraints,
            tools=[dict(t) for t in tools],
            question_source="human",
        )
        if not result.get("ok"):
            logger.warning("research_record: init_research rejected: %s", result.get("errors"))
            return None
        ids = {n["ref"]: n["id"] for n in result["committed"]["nodes"] if n.get("ref")}
        rec: Dict[str, Any] = {
            "question": result.get("root_id") or ids.get("q"),
            "tools": {t["tool_type"]: ids.get(f"t{i}") for i, t in enumerate(tools)},
            "constraints": [ids[f"c{i}"] for i in range(len(constraints)) if f"c{i}" in ids],
            "lit_methods": {},
            "hypotheses": {},
            "routes": {},
            "literature_evidence": [],
            "economics_evidence": [],
            "experiment_evidence": [],
        }
        queries = _as_dict(state.get("tz_literature_queries")).get("queries") or []
        nodes, edges = [], []
        for i, q in enumerate(q for q in queries if isinstance(q, dict)):
            qid = str(q.get("id") or f"LIT-{i + 1:02d}").upper()
            nodes.append({
                "type": "VerificationMethod", "ref": f"lit{i}", "status": "planned",
                "attrs": {
                    "name": f"Литературный поиск {qid}",
                    "method_type": "literature_search",
                    "description": _short(q.get("task") or q.get("query_en") or "", 400),
                    "query_id": qid,
                    "extract": [str(x) for x in (q.get("extract") or [])][:12],
                },
            })
            if rec["tools"].get("literature"):
                edges.append({"type": "uses", "from": f"#lit{i}", "to": rec["tools"]["literature"]})
        ids = _write(graph, "TZQueryGenAgent", nodes, edges)
        for i, q in enumerate(q for q in queries if isinstance(q, dict)):
            qid = str(q.get("id") or f"LIT-{i + 1:02d}").upper()
            if f"lit{i}" in ids:
                rec["lit_methods"][qid] = ids[f"lit{i}"]
        _save_record(state, rec)
        logger.info("research_record: question %s, %d constraints, %d literature methods",
                    rec["question"], len(rec["constraints"]), len(rec["lit_methods"]))
    except Exception as exc:  # noqa: BLE001
        logger.warning("research_record: record_research_question failed: %s", exc)
    return None


# ── stage 2: literature analysis → Evidence (literature) ─────────────────────

def _sources_of(item: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for ev in item.get("evidence") or []:
        if isinstance(ev, dict):
            src = records.get(str(ev.get("source_id") or ""))
            if src:
                out.append(src.get("url") or src.get("title") or src["source_id"])
    out.extend(str(s) for s in (item.get("sources") or []))
    return list(dict.fromkeys(s for s in out if s))[:8]


def _verification(item: Dict[str, Any]) -> str:
    statuses = {str(ev.get("verification_status") or "unverified")
                for ev in (item.get("evidence") or []) if isinstance(ev, dict)}
    if not statuses:
        return "unverified"
    if statuses == {"verified"}:
        return "verified"
    return "conflicting" if "conflicting" in statuses else "partially verified"


def record_literature_evidence(callback_context: CallbackContext) -> None:
    """After EvidenceVerifierAgent: every analogue, route and fact of
    ``literature_analysis`` becomes a literature Evidence node; the sources
    become the EmpiricalBase; the LIT-xx methods are marked done."""
    state = callback_context.state
    analysis = _as_dict(state.get("literature_analysis"))
    if not analysis:
        return None
    graph = _graph(callback_context)
    if graph is None:
        return None
    rec = _record(state)
    q = rec.get("question")
    try:
        records = {str(r.get("source_id")): r for r in (analysis.get("source_records") or [])
                   if isinstance(r, dict) and r.get("source_id")}
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        if records:
            nodes.append({
                "type": "EmpiricalBase", "ref": "corpus", "attrs": {
                    "name": "Корпус литературных источников",
                    "base_type": "literature",
                    "volume": f"{len(records)} источников",
                    "sources": [(r.get("url") or r.get("title") or sid)
                                for sid, r in list(records.items())[:20]],
                    "verified_by": sorted({r.get("verified_by") for r in records.values()
                                           if r.get("verified_by")}),
                },
            })
            if q:
                edges.append({"type": "defines_scope", "from": q, "to": "#corpus"})

        n = 0

        def add(kind: str, title: str, attrs: Dict[str, Any], query_id: str = "") -> None:
            nonlocal n
            ref = f"e{n}"
            n += 1
            nodes.append({
                "type": "Evidence", "ref": ref, "status": "obtained",
                "attrs": {"subtype": "literature", "kind": kind, "content": _short(title, 400),
                          **attrs},
            })
            if q:
                edges.append({"type": "relates_to", "from": f"#{ref}", "to": q})
            vm = rec.get("lit_methods", {}).get(query_id.upper()) if query_id else None
            if vm:
                edges.append({"type": "produces", "from": vm, "to": f"#{ref}"})

        for a in analysis.get("analogues") or []:
            if not isinstance(a, dict):
                continue
            props = "; ".join(f"{p.get('name')}={p.get('value')}" for p in (a.get("properties") or [])
                              if isinstance(p, dict))
            add("analogue", f"Аналог: {a.get('name')} {a.get('smiles') or ''}".strip(), {
                "name": a.get("name"), "smiles": a.get("smiles") or "",
                "compound_class": a.get("compound_class") or "",
                "properties": _short(props, 400), "relevance": _short(a.get("relevance") or "", 300),
                "sources": _sources_of(a, records),
            })
        for r in analysis.get("synthesis_routes") or []:
            if not isinstance(r, dict):
                continue
            steps = [s for s in (r.get("steps") or []) if isinstance(s, dict)]
            ops = " → ".join(str(s.get("operation") or "") for s in steps)
            conds = "; ".join(
                f"{c.get('name')}={c.get('value')}"
                for s in steps for c in (s.get("conditions") or []) if isinstance(c, dict)
            )
            add("literature_route", f"Маршрут из литературы: {r.get('product')}: {ops}", {
                "product": r.get("product"), "steps": len(steps),
                "operations": _short(ops, 300), "conditions": _short(conds, 400),
                "yields": "; ".join(str(s.get("yield_value") or "") for s in steps if s.get("yield_value")),
                "flow_suitability": _short(r.get("flow_suitability") or "", 300),
                "verification": _verification(r), "sources": _sources_of(r, records),
            })
        for f in analysis.get("facts") or []:
            if not isinstance(f, dict):
                continue
            add("fact", f.get("statement") or "", {
                "query_id": f.get("query_id") or "", "verification": _verification(f),
                "sources": _sources_of(f, records),
            }, query_id=str(f.get("query_id") or ""))

        ids = _write(graph, "EvidenceVerifierAgent", nodes, edges)
        rec["literature_evidence"] = [v for k, v in ids.items() if k.startswith("e")]
        if "corpus" in ids:
            rec["corpus"] = ids["corpus"]
        rec["literature_gaps"] = [str(g) for g in (analysis.get("gaps") or [])][:20]
        for vm in (rec.get("lit_methods") or {}).values():
            _advance(graph, "EvidenceVerifierAgent", vm, "running", "done")
        _save_record(state, rec)
        logger.info("research_record: %d literature evidence nodes", len(rec["literature_evidence"]))
    except Exception as exc:  # noqa: BLE001
        logger.warning("research_record: record_literature_evidence failed: %s", exc)
    return None


# ── stage 3: design candidates → Hypotheses ──────────────────────────────────

def _candidate_key(name: str, smiles: str) -> str:
    return (smiles or name or "").strip().lower()


def record_design_hypotheses(callback_context: CallbackContext) -> None:
    """After MolDesignAgent: each candidate is a Hypothesis «кандидат X
    удовлетворяет ТЗ», motivated by the ResearchQuestion."""
    state = callback_context.state
    design = _as_dict(state.get("design_candidates"))
    candidates = [c for c in (design.get("candidates") or []) if isinstance(c, dict)]
    if not candidates:
        return None
    graph = _graph(callback_context)
    if graph is None:
        return None
    rec = _record(state)
    q = rec.get("question")
    try:
        nodes, edges = [], []
        for i, c in enumerate(candidates):
            name, smiles = str(c.get("name") or ""), str(c.get("smiles") or "")
            props = "; ".join(f"{p.get('name')}={p.get('value')}" for p in (c.get("properties") or [])
                              if isinstance(p, dict))
            nodes.append({
                "type": "Hypothesis", "ref": f"h{i}", "status": "formulated",
                "attrs": {
                    "statement": _short(
                        f"Кандидат {name}{' (' + smiles + ')' if smiles else ''} удовлетворяет ТЗ "
                        f"и может быть получен на проточной установке", 400),
                    "candidate": name, "smiles": smiles,
                    "compound_class": c.get("compound_class") or "",
                    "properties": _short(props, 400),
                    "tz_fit": _short(c.get("tz_fit") or "", 400),
                    "risks": _short(c.get("risks") or "", 300),
                    "origin": c.get("source") or "дизайн",
                    "stub": bool(c.get("stub")),
                },
            })
            if q:
                edges.append({"type": "motivates", "from": q, "to": f"#h{i}"})
        ids = _write(graph, "MolDesignAgent", nodes, edges)
        hyps = dict(rec.get("hypotheses") or {})
        for i, c in enumerate(candidates):
            if f"h{i}" in ids:
                hyps[_candidate_key(str(c.get("name") or ""), str(c.get("smiles") or ""))] = ids[f"h{i}"]
        rec["hypotheses"] = hyps
        rec["design_gaps"] = [str(g) for g in (design.get("gaps") or [])][:20]
        _save_record(state, rec)
        logger.info("research_record: %d design hypotheses", len(ids))
    except Exception as exc:  # noqa: BLE001
        logger.warning("research_record: record_design_hypotheses failed: %s", exc)
    return None


# ── stage 4: synthesis routes → VerificationMethods ──────────────────────────

def _hypothesis_for(rec: Dict[str, Any], product: Dict[str, Any]) -> Optional[str]:
    hyps = rec.get("hypotheses") or {}
    for key in (_candidate_key("", str(product.get("smiles") or "")),
                _candidate_key(str(product.get("name") or ""), "")):
        if key and key in hyps:
            return hyps[key]
    return next(iter(hyps.values()), None) if len(hyps) == 1 else None


def _route_summary(route: Dict[str, Any]) -> Dict[str, Any]:
    steps = [s for s in (route.get("steps") or []) if isinstance(s, dict)]
    ops = " → ".join(str(s.get("operation") or "") for s in steps)
    conds = "; ".join(
        f"{c.get('name')}={c.get('value')}" for s in steps
        for c in (s.get("conditions") or []) if isinstance(c, dict)
    )
    reagents = "; ".join(
        ", ".join(str(r.get("name") or r.get("smiles") or "") for r in (s.get("reactants") or [])
                  if isinstance(r, dict))
        for s in steps
    )
    yields = [s.get("yield_fraction") for s in steps]
    return {
        "steps": len(steps), "operations": _short(ops, 300), "reagents": _short(reagents, 300),
        "conditions": _short(conds, 400),
        "yields": ", ".join("—" if y is None else f"{float(y):.2f}" for y in yields),
        "flow_suitability": _short(route.get("flow_suitability") or "", 300),
        "bottlenecks": [str(b) for b in (route.get("bottlenecks") or [])][:10],
        "compliance": route.get("overall_status") or "unassessed",
        "compliance_checks": _short(
            "; ".join(f"{c.get('requirement_id') or c.get('name') or ''}: {c.get('status') or c.get('verdict') or ''}"
                      for c in (route.get("tz_compliance") or []) if isinstance(c, dict)), 400),
        "origin": route.get("source") or "ретросинтез",
        "stub": bool(route.get("stub")),
    }


def record_synthesis_routes(callback_context: CallbackContext) -> None:
    """After SynthRouteAgent (after qualify_synthesis_routes): each route is a
    planned VerificationMethod testing its candidate's Hypothesis; eligible
    routes move the hypothesis to under_verification."""
    state = callback_context.state
    routes_doc = _as_dict(state.get("synthesis_routes"))
    routes = [r for r in (routes_doc.get("routes") or []) if isinstance(r, dict)]
    if not routes:
        return None
    graph = _graph(callback_context)
    if graph is None:
        return None
    rec = _record(state)
    tool = (rec.get("tools") or {}).get("retrosynthesis")
    try:
        nodes, edges = [], []
        for i, r in enumerate(routes):
            product = r.get("product") if isinstance(r.get("product"), dict) else {}
            rid = str(r.get("route_id") or f"R{i + 1}")
            nodes.append({
                "type": "VerificationMethod", "ref": f"vm{i}", "status": "planned",
                "attrs": {
                    "name": f"Маршрут синтеза {rid}: {product.get('name') or product.get('smiles') or ''}",
                    "method_type": "synthesis_route",
                    "route_id": rid, "product": product.get("name") or "",
                    "product_smiles": product.get("smiles") or "",
                    **_route_summary(r),
                },
            })
            h = _hypothesis_for(rec, product)
            if h:
                edges.append({"type": "tested_by", "from": h, "to": f"#vm{i}"})
            if tool and (r.get("source") or "ретросинтез") != "литература":
                edges.append({"type": "uses", "from": f"#vm{i}", "to": tool})
        ids = _write(graph, "SynthRouteAgent", nodes, edges)
        route_ids = dict(rec.get("routes") or {})
        for i, r in enumerate(routes):
            if f"vm{i}" in ids:
                route_ids[str(r.get("route_id") or f"R{i + 1}")] = ids[f"vm{i}"]
        rec["routes"] = route_ids
        # Eligible routes: their hypotheses are now being verified.
        for r in routes:
            if r.get("overall_status") == "eligible":
                product = r.get("product") if isinstance(r.get("product"), dict) else {}
                _set_status(graph, "SynthRouteAgent", _hypothesis_for(rec, product), "under_verification")
        qualified = _as_dict(state.get("qualified_routes"))
        if qualified:
            rec["qualified_status"] = qualified.get("status")
        _save_record(state, rec)
        logger.info("research_record: %d route verification methods", len(ids))
    except Exception as exc:  # noqa: BLE001
        logger.warning("research_record: record_synthesis_routes failed: %s", exc)
    return None


# ── stage 5: economics → Evidence (computational) ────────────────────────────

def record_economics_evidence(callback_context: CallbackContext) -> None:
    """After EconomicsAgent: one computational Evidence per costed route (the
    chemquote ranking as the server returned it), produced by that route's
    VerificationMethod."""
    state = callback_context.state
    ranking = _as_dict(state.get("economics_ranking"))
    routes = ranking.get("routes") if isinstance(ranking.get("routes"), dict) else {}
    graph = _graph(callback_context)
    if graph is None:
        return None
    rec = _record(state)
    q = rec.get("question")
    tool = (rec.get("tools") or {}).get("economics")
    try:
        nodes, edges = [], []
        items = list(routes.items())
        if not items:
            summary = state.get("economics")
            if not summary:
                return None
            items = [("economics", {"summary": _short(summary, 600)})]
        for i, (rid, info) in enumerate(items):
            info = info if isinstance(info, dict) else {"value": info}
            cost = info.get("cost_per_unit")
            packs = info.get("cost_packs")
            cur = info.get("currency") or ranking.get("preferred_currency") or ""
            content = (
                f"Стоимость маршрута {rid}: себестоимость {cost} {cur}, чек по упаковкам {packs} {cur} "
                f"за {ranking.get('target_qty') or ''} {ranking.get('target_unit') or ''}; "
                f"статус {info.get('status') or ''}"
                if cost is not None or packs is not None
                else f"Экономика {rid}: {_short(info, 300)}"
            )
            nodes.append({
                "type": "Evidence", "ref": f"ec{i}", "status": "obtained",
                "attrs": {
                    "subtype": "computational", "measured_on": "chemquote (прайс-листы РФ)",
                    "kind": "economics", "content": _short(content, 400),
                    "route_id": rid, "rank": info.get("rank"), "status": info.get("status"),
                    "cost_per_unit": cost, "cost_packs": packs, "currency": cur,
                    "missing": _short(info.get("missing") or [], 300),
                    "warnings": _short(info.get("warnings") or [], 300),
                },
            })
            vm = (rec.get("routes") or {}).get(str(rid))
            if vm:
                edges.append({"type": "produces", "from": vm, "to": f"#ec{i}"})
                if tool:
                    edges.append({"type": "uses", "from": vm, "to": tool})
            elif q:
                edges.append({"type": "relates_to", "from": f"#ec{i}", "to": q})
        ids = _write(graph, "EconomicsAgent", nodes, edges)
        rec["economics_evidence"] = list(ids.values())
        _save_record(state, rec)
        logger.info("research_record: %d economics evidence nodes", len(ids))
    except Exception as exc:  # noqa: BLE001
        logger.warning("research_record: record_economics_evidence failed: %s", exc)
    return None


# ── stage 6–10: external optimisation / experiment → Evidence + GeneratedData ─

def record_experiment_evidence(callback_context: CallbackContext) -> None:
    """After ReactorAgent: the external optimisation module's result (and the
    CFD runs it reported) become experimental Evidence + GeneratedData, marking
    the route VerificationMethods done."""
    state = callback_context.state
    result = _as_dict(state.get("optimization_result"))
    summary = state.get("optimization_summary")
    runs = state.get("optimization_a2a_runs")
    cfd = state.get("cfd_runs")
    if not (result or summary or runs or cfd):
        return None
    graph = _graph(callback_context)
    if graph is None:
        return None
    rec = _record(state)
    q = rec.get("question")
    tool = (rec.get("tools") or {}).get("experiment")
    try:
        nodes, edges = [], []
        status = str(result.get("status") or result.get("state") or "")
        task_id = str(result.get("task_id") or result.get("id") or "")
        content = (
            f"Внешняя оптимизация (A2A{' ' + task_id if task_id else ''}, статус {status or '—'}): "
            f"{_short(result.get('result') or result.get('output') or summary or result, 350)}"
        )
        nodes.append({
            "type": "Evidence", "ref": "ex0", "status": "obtained",
            "attrs": {
                "subtype": "experimental", "measured_on": "модуль оптимизации / установка (A2A)",
                "kind": "optimization", "content": _short(content, 400),
                "task_id": task_id, "status": status,
                "summary": _short(summary or "", 600),
            },
        })
        if q:
            edges.append({"type": "relates_to", "from": "#ex0", "to": q})
        if result or runs:
            nodes.append({
                "type": "GeneratedData", "ref": "gd0", "attrs": {
                    "name": "Ответы модуля оптимизации / CFD",
                    "description": _short({"optimization_result": result, "runs": runs, "cfd_runs": cfd}, 800),
                },
            })
            edges.append({"type": "derived_from", "from": "#gd0", "to": "#ex0"})
        for vm in (rec.get("routes") or {}).values():
            edges.append({"type": "produces", "from": vm, "to": "#ex0"})
            if tool:
                edges.append({"type": "uses", "from": vm, "to": tool})
        ids = _write(graph, "ReactorAgent", nodes, edges)
        rec["experiment_evidence"] = [v for k, v in ids.items() if k.startswith("ex")]
        done = status.lower() in ("completed", "done", "success", "succeeded", "finished")
        for vm in (rec.get("routes") or {}).values():
            _advance(graph, "ReactorAgent", vm, *(("running", "done") if done else ("running",)))
        _save_record(state, rec)
        logger.info("research_record: experiment evidence recorded (status=%s)", status)
    except Exception as exc:  # noqa: BLE001
        logger.warning("research_record: record_experiment_evidence failed: %s", exc)
    return None


# ── stage 11: final report → Conclusion + Report ─────────────────────────────

def _save_report_file(report: str) -> Optional[Path]:
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / f"REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        path.write_text(report, encoding="utf-8")
        return path
    except OSError as exc:
        logger.warning("research_record: could not save the report: %s", exc)
        return None


def record_conclusion(callback_context: CallbackContext) -> None:
    """After ReportAgent: the report is saved to ``reports/`` and recorded as a
    draft Conclusion (based_on every Evidence of the run) plus a Report node;
    the ResearchQuestion is closed."""
    state = callback_context.state
    report = state.get("final_report")
    if not isinstance(report, str) or not report.strip():
        return None
    path = _save_report_file(report)
    if path:
        state["final_report_path"] = str(path.resolve())
    graph = _graph(callback_context)
    if graph is None:
        return None
    rec = _record(state)
    q = rec.get("question")
    try:
        nodes = [{
            "type": "Conclusion", "ref": "cl", "status": "draft",
            "attrs": {
                "statement": _short(report, 1200),
                "answers": q or "",
                "hypotheses": list((rec.get("hypotheses") or {}).values()),
                "routes": list((rec.get("routes") or {}).values()),
            },
        }, {
            "type": "Report", "ref": "rp", "attrs": {
                "name": "Итоговый отчёт заказчику (стадия 11)",
                "path": str(path.resolve()) if path else "",
                "length": len(report),
            },
        }]
        edges = [{"type": "derived_from", "from": "#rp", "to": "#cl"}]
        if q:
            edges.append({"type": "produces", "from": "#cl", "to": q})
        for eid in _evidence_ids(rec):
            edges.append({"type": "based_on", "from": "#cl", "to": eid})
        ids = _write(graph, "ReportAgent", nodes, edges)
        rec["conclusion"] = ids.get("cl")
        rec["report"] = ids.get("rp")
        _set_status(graph, "ReportAgent", q, "closed")
        _save_record(state, rec)
        logger.info("research_record: conclusion %s, report %s", ids.get("cl"), ids.get("rp"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("research_record: record_conclusion failed: %s", exc)
    # No chat Content here: ReportAgent runs as the root's AgentTool, and the
    # tool result is the LAST event's text — a note would replace the report.
    return None


__all__ = [
    "RECORD_KEY",
    "record_conclusion",
    "record_design_hypotheses",
    "record_economics_evidence",
    "record_experiment_evidence",
    "record_literature_evidence",
    "record_research_question",
    "record_synthesis_routes",
]
