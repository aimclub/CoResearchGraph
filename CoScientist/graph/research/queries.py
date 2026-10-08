"""Named trigger queries over the Research Context Graph (spec §3).

The orchestrator's decision logic works on statuses and edges, not on node
content — these functions are those decisions as code: pure, rule-based (no
LLM), each returning {"items": [...], "rendered": "..."} so the result can be
consumed programmatically or dropped into a prompt. ``trigger_report`` bundles
all of them into the digest the orchestrator sees before every turn.
"""
from __future__ import annotations

import re

from typing import Any, Dict, List, Optional

from CoScientist.graph.research.store import (
    ResearchGraphStore,
    priority_rank,
    research_graph,
)

_EVIDENCE_EDGES = ("supports", "refutes", "refines")


def _graph(store: Optional[ResearchGraphStore]):
    return (store or research_graph)


def _nodes_of(g, node_type: str) -> List[str]:
    return [n for n, d in g.nodes(data=True) if d.get("type") == node_type]


def _out(g, node_id: str, edge_type: str) -> List[str]:
    """Targets of node's outgoing edges of the given type."""
    return [v for _, v, k in g.out_edges(node_id, keys=True) if k == edge_type]


def _in(g, node_id: str, edge_type: str) -> List[str]:
    """Sources of node's incoming edges of the given type."""
    return [u for u, _, k in g.in_edges(node_id, keys=True) if k == edge_type]


def _label(g, node_id: str, n: int = 80) -> str:
    attrs = g.nodes[node_id].get("attrs") or {}
    for key in ("formulation", "content", "synthesis", "name", "title", "description"):
        if attrs.get(key):
            value = str(attrs[key])
            value = " ".join(value.split())
            return value if len(value) <= n else value[:n] + "…"
    return ""


def _status(g, node_id: str) -> str:
    return g.nodes[node_id].get("status", "")


def _hypothesis_tools(g, hyp_id: str) -> List[str]:
    """All Tools a hypothesis depends on: requires(H→T) plus uses(VM→T) of the
    VerificationMethods the hypothesis is tested_by."""
    tools = set(_out(g, hyp_id, "requires"))
    for vm in _out(g, hyp_id, "tested_by"):
        tools.update(_out(g, vm, "uses"))
    return sorted(tools)


# ── individual triggers ──────────────────────────────────────────────────────

def _id_order(node_id: str) -> tuple:
    """H2 before H10 (numeric suffix), unknown shapes fall back to the string."""
    digits = "".join(c for c in str(node_id) if c.isdigit())
    return (int(digits) if digits else 0, str(node_id))


def ready_hypotheses(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    """The ONE hypothesis to verify next.

    Every `formulated` hypothesis whose referenced Tools are available (vacuously
    ready when it references none) is a candidate, but only the most relevant one
    — the agent's pick (`attrs.selected`) / highest `attrs.priority`, ties broken
    by id order — is offered as actionable in `items`. The rest are reported as
    `queued` so the orchestrator sees the backlog without starting it: one branch
    at a time, otherwise a batch of hypotheses gets verified in parallel and the
    evidence of one contaminates the verdict of another.
    """
    g = _graph(store).full_graph()
    candidates = []
    for h in _nodes_of(g, "Hypothesis"):
        if _status(g, h) != "formulated":
            continue
        tools = _hypothesis_tools(g, h)
        if all(_status(g, t) == "available" for t in tools):
            candidates.append({"hypothesis": h, "label": _label(g, h),
                               "methods": _out(g, h, "tested_by"), "tools": tools})
    candidates.sort(key=lambda i: (priority_rank(g.nodes[i["hypothesis"]].get("attrs")),
                                   _id_order(i["hypothesis"])))
    items, queued = candidates[:1], candidates[1:]
    active = [h for h in _nodes_of(g, "Hypothesis")
              if _status(g, h) == "under_verification"]

    lines = [f"IN VERIFICATION: {h} \"{_label(g, h)}\" — this branch is active; "
             f"finish it (gather evidence → verdict) before starting another"
             for h in sorted(active, key=_id_order)]
    lines += [f"READY (verify this ONE next): {i['hypothesis']} \"{i['label']}\" — "
              f"{'tools available' if i['tools'] else 'no tools needed'} "
              f"→ research_set_focus({i['hypothesis']}) then delegate evidence "
              f"gathering for it" for i in items]
    if queued:
        lines.append(
            "QUEUED (do NOT verify in parallel): "
            + ", ".join(f"{i['hypothesis']} \"{i['label']}\"" for i in queued)
            + " → take the next one only after the active branch has a verdict")
    return {"items": items, "queued": queued, "in_verification": active,
            "rendered": "\n".join(lines)}


def postponed_hypotheses(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    """The backlog: alternatives set aside while one hypothesis is verified.

    Rendered only when nothing is ready or under verification — i.e. exactly when
    reviving one (postponed→formulated) is the sensible next move; otherwise the
    backlog is noise the orchestrator must not act on."""
    g = _graph(store).full_graph()
    items = [{"hypothesis": h, "label": _label(g, h)}
             for h in sorted(_nodes_of(g, "Hypothesis"), key=_id_order)
             if _status(g, h) == "postponed"]
    idle = not ready_hypotheses(store)["items"] and not any(
        _status(g, h) == "under_verification" for h in _nodes_of(g, "Hypothesis"))
    rendered = ""
    if items and idle:
        rendered = ("BACKLOG: " + ", ".join(f"{i['hypothesis']} \"{i['label']}\""
                                            for i in items)
                    + " → no hypothesis is active; revive the most relevant one "
                      "(postponed→formulated) and verify it next")
    return {"items": items, "rendered": rendered}


def blocked_hypotheses(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    """Formulated hypotheses blocked by a Tool that is not available."""
    g = _graph(store).full_graph()
    items = []
    for h in _nodes_of(g, "Hypothesis"):
        if _status(g, h) != "formulated":
            continue
        blocking = [{"id": t, "status": _status(g, t)}
                    for t in _hypothesis_tools(g, h)
                    if _status(g, t) != "available"]
        if blocking:
            items.append({"hypothesis": h, "label": _label(g, h),
                          "blocking_tools": blocking})
    lines = [f"BLOCKED: {i['hypothesis']} \"{i['label']}\" — waiting on "
             + ", ".join(f"{b['id']}({b['status']})" for b in i["blocking_tools"])
             for i in items]
    return {"items": items, "rendered": "\n".join(lines)}


def refuting_evidence(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    """Evidence with a `refutes` edge to a hypothesis that has not been
    refuted/postponed yet → the branch needs review."""
    g = _graph(store).full_graph()
    items = []
    for h in _nodes_of(g, "Hypothesis"):
        if _status(g, h) in ("refuted", "postponed"):
            continue
        for e in _in(g, h, "refutes"):
            items.append({"evidence": e, "hypothesis": h,
                          "hypothesis_status": _status(g, h),
                          "label": _label(g, h)})
    lines = [f"REFUTE SIGNAL: {i['evidence']} refutes {i['hypothesis']} "
             f"({i['hypothesis_status']}) → review the branch" for i in items]
    return {"items": items, "rendered": "\n".join(lines)}


def closable_hypotheses(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    """The spec's "path Hypothesis → Evidence → ConfirmationCriteria is closed"
    trigger: an under_verification hypothesis with evidence attached whose
    criteria (formulated_for) are all met → the validator can write a
    Conclusion. Sub-buckets: criteria exist but unmet (awaiting_criteria) and
    no criteria defined at all (missing_criteria)."""
    g = _graph(store).full_graph()
    closable, awaiting, missing = [], [], []
    for h in _nodes_of(g, "Hypothesis"):
        if _status(g, h) != "under_verification":
            continue
        evidence = sorted({e for et in _EVIDENCE_EDGES for e in _in(g, h, et)})
        if not evidence:
            continue
        criteria = _in(g, h, "formulated_for")
        entry = {"hypothesis": h, "label": _label(g, h), "evidence": evidence,
                 "criteria": sorted(criteria)}
        if not criteria:
            missing.append(entry)
        elif all(_status(g, c) == "met" for c in criteria):
            closable.append(entry)
        else:
            entry["unmet"] = [c for c in criteria if _status(g, c) != "met"]
            awaiting.append(entry)
    lines = [f"CLOSABLE: {i['hypothesis']} — evidence {','.join(i['evidence'])}; "
             f"{','.join(i['criteria'])} met → write a Conclusion (draft) with "
             f"based_on edges" for i in closable]
    lines += [f"AWAITING CRITERIA: {i['hypothesis']} — evidence collected but "
              f"{','.join(i['unmet'])} not met" for i in awaiting]
    lines += [f"MISSING CRITERIA: {i['hypothesis']} has evidence "
              f"{','.join(i['evidence'])} but no ConfirmationCriteria — define "
              f"them (HypothesesAgent) or judge sufficiency" for i in missing]
    return {"items": closable, "awaiting_criteria": awaiting,
            "missing_criteria": missing, "rendered": "\n".join(lines)}


def pending_conclusions(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    g = _graph(store).full_graph()
    items = [{"conclusion": c, "label": _label(g, c)}
             for c in _nodes_of(g, "Conclusion") if _status(g, c) == "draft"]
    lines = [f"PENDING CONCLUSION: {i['conclusion']} \"{i['label']}\" — "
             f"draft awaiting approval" for i in items]
    return {"items": items, "rendered": "\n".join(lines)}


def missing_tools(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    """Tools that are not available, with who needs them."""
    g = _graph(store).full_graph()
    items = []
    for t in _nodes_of(g, "Tool"):
        status = _status(g, t)
        if status == "available":
            continue
        items.append({"tool": t, "label": _label(g, t), "status": status,
                      "required_by": _in(g, t, "requires"),
                      "used_by": _in(g, t, "uses")})
    lines = [f"TOOL NOT READY: {i['tool']} \"{i['label']}\" ({i['status']})"
             + (f" — required by {','.join(i['required_by'])}" if i["required_by"] else "")
             for i in items]
    return {"items": items, "rendered": "\n".join(lines)}


def resources_low(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    """Exhausted resources, or numeric remainder below 10% of the limit —
    the spec's signal to wrap up / escalate."""
    g = _graph(store).full_graph()
    items = []
    for r in _nodes_of(g, "Resource"):
        attrs = g.nodes[r].get("attrs") or {}
        status = _status(g, r)
        remaining = _num(attrs.get("remaining", attrs.get("остаток")))
        limit = _num(attrs.get("limit", attrs.get("лимит")))
        low = status == "exhausted" or (remaining is not None and remaining <= 0) \
            or (remaining is not None and limit and remaining / limit < 0.1)
        if low:
            items.append({"resource": r, "label": _label(g, r), "status": status,
                          "remaining": remaining, "limit": limit})
    lines = [f"RESOURCES LOW: {i['resource']} \"{i['label']}\" "
             f"(remaining={i['remaining']}, limit={i['limit']}, "
             f"status={i['status']}) → consider wrapping up" for i in items]
    return {"items": items, "rendered": "\n".join(lines)}


def open_questions(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    """Open/decomposed questions, annotated with how developed each branch is
    (no hypotheses yet → needs branching; no evidence yet → needs exploration)."""
    g = _graph(store).full_graph()
    items = []
    for q in _nodes_of(g, "ResearchQuestion"):
        status = _status(g, q)
        if status == "closed":
            continue
        hyps = _out(g, q, "motivates")
        evidence = _in(g, q, "relates_to")
        items.append({"question": q, "label": _label(g, q), "status": status,
                      "hypotheses": len(hyps), "evidence": len(evidence)})
    lines = []
    for i in items:
        need = []
        if i["evidence"] == 0:
            need.append("no evidence yet (explore literature)")
        if i["hypotheses"] == 0:
            need.append("no hypotheses yet (branch)")
        tail = " — " + "; ".join(need) if need else ""
        lines.append(f"QUESTION: {i['question']} \"{i['label']}\" "
                     f"[{i['status']}, {i['hypotheses']} hypotheses, "
                     f"{i['evidence']} evidence]{tail}")
    return {"items": items, "rendered": "\n".join(lines)}


def unresolved_hypotheses(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    """Hypotheses that already have evidence attached but still carry NO verdict
    (status not confirmed/refuted/postponed). This is the #1 reason a finished
    graph looks 'all formulated'. The orchestrator does NOT judge these itself —
    it delegates each to the ValidatorAgent, which weighs the evidence against the
    criteria and writes the verdict + Conclusion."""
    g = _graph(store).full_graph()
    items = []
    for h in _nodes_of(g, "Hypothesis"):
        st = _status(g, h)
        if st in ("confirmed", "refuted", "postponed"):
            continue
        supporting = _in(g, h, "supports")
        refuting = _in(g, h, "refutes")
        related = _in(g, h, "relates_to")   # polarity not yet assigned
        if not (supporting or refuting or related):
            continue
        items.append({"hypothesis": h, "label": _label(g, h), "status": st,
                      "supporting": sorted(supporting), "refuting": sorted(refuting),
                      "related": sorted(related)})
    lines = [f"NEEDS VERDICT: {i['hypothesis']} (still {i['status']}) has evidence "
             f"[support: {','.join(i['supporting']) or '—'} | refute: "
             f"{','.join(i['refuting']) or '—'} | unclassified: "
             f"{','.join(i['related']) or '—'}] → the background validator will "
             f"assign polarity and a verdict" for i in items]
    return {"items": items, "rendered": "\n".join(lines)}


def verdict_without_evidence(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    """Branches being tested that have nothing attached, and criteria never measured.

    A run once benchmarked for an hour, wrote a table of p-values into its chat
    answer, and committed none of it. The validator reads the graph, found no
    measurement, and rejected a result that existed — so the answer and the
    record contradicted each other and neither could be trusted.

    The condition is computable before that happens: a hypothesis under
    verification with no evidence, or with criteria none of which has been
    marked met or not met, means the work is happening somewhere the record
    cannot see.
    """
    g = _graph(store).full_graph()
    stalled = []
    for h in sorted(_nodes_of(g, "Hypothesis"), key=_id_order):
        if _status(g, h) != "under_verification":
            continue
        evidence = set(_in(g, h, "supports")) | set(_in(g, h, "refutes")) \
            | set(_in(g, h, "relates_to"))
        criteria = [c for c in _in(g, h, "formulated_for")
                    if g.nodes[c].get("type") == "ConfirmationCriteria"]
        untouched = [c for c in criteria if _status(g, c) == "not_met"]
        if evidence and len(untouched) < len(criteria):
            continue
        stalled.append({"hypothesis": h, "label": _label(g, h),
                        "evidence": len(evidence),
                        "criteria": len(criteria), "untouched": len(untouched)})

    lines = []
    for item in stalled:
        missing = ("no evidence is attached" if not item["evidence"]
                   else f"none of its {item['criteria']} criteria has been measured")
        lines.append(
            f"WORK NOT IN THE RECORD: {item['hypothesis']} is under verification and "
            f"{missing}. Whatever was run has to be committed as Evidence, with the "
            f"criteria it meets marked met, BEFORE any verdict or final answer "
            f"cites it — an answer the graph cannot support will be rejected.")
    return {"items": stalled, "rendered": "\n".join(lines)}


def study_open(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    """Whether the study may be called finished — as a graph condition.

    A run used to end whenever the orchestrator judged it had answered enough.
    The backlog was rendered as advice only, so a first confirmed hypothesis
    ended the study with its alternatives never tested and never explicitly
    dropped: the reviewers' objection was that nothing recorded *why* they were
    not tested.

    The condition is now computable. The study is open while any hypothesis
    carries neither a verdict nor a recorded reason for being set aside, and it
    offers exactly two ways to settle each one: verify it, or record
    `attrs.not_tested_reason` stating why the verdict already obtained makes
    testing it unnecessary. Either is a write to the graph, so the decision is
    auditable instead of being a silence.

    `postponed_reason` deliberately does not count. The store writes that itself
    when it files the alternatives of one commit as a backlog, so accepting it
    here would mark every hypothesis settled the moment it was created.
    """
    g = _graph(store).full_graph()
    settled, unsettled = [], []
    for h in sorted(_nodes_of(g, "Hypothesis"), key=_id_order):
        status = _status(g, h)
        attrs = g.nodes[h].get("attrs") or {}
        reason = str(attrs.get("not_tested_reason") or "").strip()
        if status in ("confirmed", "refuted"):
            settled.append({"hypothesis": h, "status": status})
        elif status == "inconclusive":
            # Tested and not settled. It counts as work done — the study is not
            # idle — but the question it was asked is still open, so it is
            # reported separately rather than as a verdict.
            settled.append({"hypothesis": h, "status": "inconclusive",
                            "reason": str(attrs.get("inconclusive_reason") or "").strip()})
        elif reason:
            settled.append({"hypothesis": h, "status": "not tested", "reason": reason})
        else:
            unsettled.append({"hypothesis": h, "label": _label(g, h), "status": status})

    verdicts = [i for i in settled
                if i["status"] in ("confirmed", "refuted", "inconclusive")]
    active = [h for h in sorted(_nodes_of(g, "Hypothesis"), key=_id_order)
              if _status(g, h) == "under_verification"]

    # A branch still being tested means the study is open, whatever the answer
    # being drafted says. This is the case that produced a final report reading
    # "H1 SUPPORTED", with tables and p-values, while the graph still held the
    # literature-only evidence the branch started with: the work happened and
    # was never committed, so the validator judged what it could see and
    # disagreed with the answer the user was given.
    if active:
        return {
            "items": unsettled, "settled": settled, "open": True,
            "rendered": (
                "STUDY NOT FINISHED: "
                + ", ".join(f"{h} \"{_label(g, h)}\"" for h in active)
                + " is still under verification. Do NOT report a verdict in your "
                  "answer that the graph does not hold. Commit the evidence you "
                  "obtained (research_commit) and let the validator write the "
                  "verdict; if you have no evidence to commit, say so instead of "
                  "concluding."),
        }

    rendered = ""
    if unsettled and verdicts:
        rendered = (
            "STUDY NOT FINISHED: "
            + ", ".join(f"{i['hypothesis']} \"{i['label']}\" ({i['status']})"
                        for i in unsettled)
            + " still carry no verdict while "
            + ", ".join(i["hypothesis"] for i in verdicts)
            + " already has one. Do NOT report the study as complete. For each, "
              "either revive it (postponed→formulated) and verify it next, or "
              "commit attrs.not_tested_reason saying why the verdict already "
              "obtained makes testing it unnecessary."
        )
    return {"items": unsettled, "settled": settled,
            "open": bool(unsettled), "rendered": rendered}


def progress(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    g = _graph(store).full_graph()
    counts: Dict[str, Dict[str, int]] = {}
    for _, d in g.nodes(data=True):
        t, s = d.get("type", "?"), d.get("status", "?")
        counts.setdefault(t, {}).setdefault(s, 0)
        counts[t][s] += 1
    parts = []
    for t in sorted(counts):
        inner = ", ".join(f"{s}:{n}" for s, n in sorted(counts[t].items()))
        parts.append(f"{t}({inner})")
    return {"items": counts, "rendered": "PROGRESS: " + "; ".join(parts) if parts else ""}


def _num(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── the digest the orchestrator consumes ─────────────────────────────────────

# Evaluation metrics and methodological norms, with the spellings they actually
# appear in. Matching is deterministic and explainable on purpose: a research is
# told which WORD it recorded and never thresholded, so the finding can be checked
# by reading two nodes. Extend per domain; unknown vocabulary simply is not audited.
_METRIC_LEXICON: Dict[str, tuple] = {
    "validity": ("validity", "valid smiles", "% valid", "chemically valid"),
    "uniqueness": ("uniqueness", "unique molecules", "% unique", "duplicate rate"),
    "novelty": ("novelty", "novel molecules", "not in the training set"),
    "diversity": ("diversity", "internal diversity", "scaffold diversity"),
    "target hit rate": ("hit rate", "hit fraction", "target hit", "target-property hit"),
    "synthetic accessibility": ("synthetic accessibility", "sa score", "sa>", "sa >"),
    "drug-likeness": ("qed", "drug-likeness", "druglikeness"),
    "distribution match": ("fcd", "frechet", "kl divergence to the data"),
    "accuracy": ("accuracy", "top-1", "error rate"),
    # Bare "precision"/"recall" are dropped: they matched "precision medicine" in
    # a literature abstract. Only spellings that cannot mean anything else.
    "precision/recall/F1": ("f1", "f1-score", "f-score", "precision and recall",
                            "precision/recall"),
    "auc": ("auc", "roc", "average precision"),
    "regression error": ("rmse", "mae", "r2", "r^2"),
    "baseline comparison": ("baseline", "compared against", "vs. random"),
    "statistical significance": ("p-value", "p <", "significance", "z-score",
                                 "confidence interval", "effect size"),
    "ablation": ("ablation",),
    "cross-validation": ("cross-validation", "cross validation", "held-out", "hold-out"),
    "reproducibility": ("reproducib", "fixed seed", "random seed"),
}

# Nodes that may CARRY a norm (what the field expects) and nodes that IMPOSE it
# (what this research actually committed to check).
_NORM_SOURCES = ("Evidence", "Constraint")
_NORM_CONSTRAINT_SUBTYPES = ("methodological_norms", "domain_standards", "expert_knowledge")


def _text_of(g, node_id: str) -> str:
    attrs = (g.nodes[node_id].get("attrs") or {}) if node_id in g.nodes else {}
    return " ".join(str(v) for v in attrs.values() if v).lower()


def _spelling_re(s: str):
    # Plain substring matching is not safe here: short metric names live inside
    # identifiers — "f1" matches the PDB code 5F19, "auc" matches "glaucoma" —
    # and a false gap is worse than a missed one, because it sends the run off to
    # add a threshold nobody asked for. The boundary is only applied on the sides
    # that end in a word character: "sa>" is followed by the threshold value
    # ("sa>3"), and demanding a non-alphanumeric there would miss every use.
    head = r"(?<![a-z0-9])" if s[:1].isalnum() else ""
    tail = r"(?![a-z0-9])" if s[-1:].isalnum() else ""
    return re.compile(head + re.escape(s) + tail)


_METRIC_RE = {name: tuple(_spelling_re(s) for s in spellings)
              for name, spellings in _METRIC_LEXICON.items()}


def _metrics_in(text: str) -> List[str]:
    return sorted({name for name, patterns in _METRIC_RE.items()
                   if any(p.search(text) for p in patterns)})


def criteria_coverage(store: Optional[ResearchGraphStore] = None) -> Dict[str, Any]:
    """Confirmation criteria that omit a metric this research itself recorded.

    A run can pass every threshold it set and still be void, because the
    thresholds never covered what the field requires. Both halves of that
    comparison are nodes here — the norm arrives as literature Evidence or a
    methodological Constraint, the threshold lives in ConfirmationCriteria — so
    the omission is a query rather than a matter of noticing. Cheap enough to run
    before a costly VerificationMethod, which is the point: it fires while the
    experiment can still be changed.
    """
    g = _graph(store).full_graph()

    recorded: Dict[str, List[str]] = {}
    for ntype in _NORM_SOURCES:
        for nid in _nodes_of(g, ntype):
            attrs = g.nodes[nid].get("attrs") or {}
            if ntype == "Constraint":
                sub = str(attrs.get("subtype") or "").lower()
                if sub not in _NORM_CONSTRAINT_SUBTYPES:
                    continue
            for m in _metrics_in(_text_of(g, nid)):
                recorded.setdefault(m, []).append(nid)

    items = []
    for cc in _nodes_of(g, "ConfirmationCriteria"):
        thresholded = set(_metrics_in(_text_of(g, cc)))
        missing = sorted(set(recorded) - thresholded)
        hyps = _in(g, cc, "formulated_for") or _out(g, cc, "formulated_for")
        if missing:
            items.append({
                "criteria": cc,
                "hypotheses": hyps,
                "thresholded": sorted(thresholded),
                "missing": missing,
                "recorded_in": {m: recorded[m] for m in missing},
            })

    lines = [
        f"CRITERIA GAP: {i['criteria']} thresholds {', '.join(i['thresholded']) or 'nothing measurable'} "
        f"but this research recorded {', '.join(i['missing'])} "
        f"(in {', '.join(sorted({n for ns in i['recorded_in'].values() for n in ns}))}) "
        f"— add them before running the experiment, or state why they do not apply"
        for i in items
    ]
    return {"items": items, "rendered": "\n".join(lines),
            "recorded_metrics": sorted(recorded)}


TRIGGERS = {
    "criteria_coverage": criteria_coverage,
    "ready_hypotheses": ready_hypotheses,
    "blocked_hypotheses": blocked_hypotheses,
    "postponed_hypotheses": postponed_hypotheses,
    "refuting_evidence": refuting_evidence,
    "unresolved_hypotheses": unresolved_hypotheses,
    "closable_hypotheses": closable_hypotheses,
    "study_open": study_open,
    "verdict_without_evidence": verdict_without_evidence,
    "pending_conclusions": pending_conclusions,
    "missing_tools": missing_tools,
    "resources_low": resources_low,
    "open_questions": open_questions,
    "progress": progress,
}


def trigger_report(store: Optional[ResearchGraphStore] = None,
                   char_budget: int = 4000) -> Dict[str, Any]:
    """Run every named trigger and produce the compact digest injected into the
    orchestrator's prompt (and returned by the research_triggers tool)."""
    g = _graph(store)
    if g.is_empty():
        return {"triggers": {}, "rendered": "Research graph is EMPTY."}
    report: Dict[str, Any] = {}
    rendered_parts: List[str] = []
    for name, fn in TRIGGERS.items():
        result = fn(store)
        report[name] = result
        if result.get("rendered"):
            rendered_parts.append(result["rendered"])
    text = "\n".join(rendered_parts) or "No active triggers."
    if len(text) > char_budget:
        text = text[:char_budget] + "\n…[trigger digest truncated]"
    return {"triggers": report, "rendered": text}
