"""One deterministic reading of whether the ask was met.

Reports, triggers and the final answer all use this. An empty obligation list
after a failed parse is not ``fulfilled``. A finished report does not close
open questions or deliverables by itself.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional  # noqa: F401

from CoScientist.requirements.deliverables import deliverable_status
from CoScientist.requirements.hypotheses import hypothesis_met
from CoScientist.requirements.models import (
    Completion,
    NormalizedStatement,
    OutcomeRecord,
    RequirementPart,
)
from CoScientist.requirements.questions import question_met


def _index(outcomes: Iterable[OutcomeRecord | dict[str, Any]]) -> dict[str, OutcomeRecord]:
    found: dict[str, OutcomeRecord] = {}
    for item in outcomes:
        if isinstance(item, OutcomeRecord):
            found[item.part_id] = item
        elif isinstance(item, dict) and item.get("part_id"):
            found[str(item["part_id"])] = OutcomeRecord.model_validate(item)
    return found


def assess_requirement(
    part: RequirementPart,
    outcome: OutcomeRecord | None,
    volume: Any = None,
) -> tuple[str, str]:
    if part.kind == "question":
        ok, why = question_met(part, outcome)
        status = "met" if ok else "unmet"
    elif part.kind == "hypothesis":
        ok, why = hypothesis_met(part, outcome)
        status = "met" if ok else "unmet"
    else:
        status, why = deliverable_status(part, outcome, volume=volume)
    if status != "unmet" and part.obligation and not part.retired:
        pending = [c.text for c in part.criteria if c.obligation
                   and (not c.id or outcome is None or outcome.criteria_checks.get(c.id) is not True)]
        if pending:
            return "partial", "; ".join(filter(None, [why, "unverified criteria: " + "; ".join(pending)]))
    return status, why


def evaluate_fulfillment(
    statement: NormalizedStatement | None,
    outcomes: Iterable[OutcomeRecord | dict[str, Any]] = (),
) -> Completion:
    if statement is None or statement.uncertain or not statement.obligations():
        reason = (statement.uncertainty if statement else "") or "the request was not normalized"
        return Completion(
            fulfillment="unfulfilled",
            report_ready=bool(reason),
            remaining=[{"part_id": "", "reason": reason}],
            asserted=True,
        )
    found = _index(outcomes)
    return _completion_from_assessments(statement, {
        part.id: assess_requirement(part, found.get(part.id), statement.volume)
        for part in statement.obligations()
    })


def _completion_from_assessments(
    statement: NormalizedStatement,
    assessments: dict[str, tuple[str, str]],
) -> Completion:
    """Aggregate statuses already produced by the requirement assessors."""
    remaining: list[dict[str, Any]] = []
    met = 0
    partial = False
    for part in statement.obligations():
        status, why = assessments.get(part.id, ("unmet", "no closing result yet"))
        if status == "met":
            met += 1
        elif status == "partial":
            partial = True
            remaining.append({
                "part_id": part.id,
                "kind": part.kind,
                "formulation": part.formulation,
                "reason": why or "partial",
            })
        else:
            remaining.append({
                "part_id": part.id,
                "kind": part.kind,
                "formulation": part.formulation,
                "reason": why or "not done",
            })
    total = len(statement.obligations())
    if met == total and not partial:
        fulfillment = "fulfilled"
    elif met or partial:
        fulfillment = "partial"
    else:
        fulfillment = "unfulfilled"
    return Completion(
        fulfillment=fulfillment,  # type: ignore[arg-type]
        report_ready=True,
        remaining=remaining,
    )


def _legacy_completion(nodes: list[dict[str, Any]]) -> Completion:
    """Old graphs stay readable. Emptiness is not success and not a rewrite."""
    return Completion(
        fulfillment="unfulfilled",
        report_ready=True,
        remaining=[],
        asserted=False,
        legacy=True,
    ) if not any(
        (n.get("attrs") or {}).get("obligation") is True for n in nodes
    ) else Completion(fulfillment="unfulfilled", report_ready=True, remaining=[], asserted=False)


def evaluate_nodes(
    nodes: Iterable[dict[str, Any]],
    edges: Iterable[dict[str, Any]] = (),
    *,
    outcomes: Iterable[OutcomeRecord | dict[str, Any]] = (),
    statement: Optional[NormalizedStatement] = None,
) -> Completion:
    """Project completion from graph nodes, optionally joined with a statement."""
    node_list = [n for n in nodes if isinstance(n, dict)]
    persisted_assessment = None
    if statement is None:
        for node in node_list:
            saved = (node.get("attrs") or {}).get("normalized_statement")
            if isinstance(saved, dict):
                statement = NormalizedStatement.model_validate(saved)
                persisted_assessment = (node.get("attrs") or {}).get("requirement_assessment")
                break
    if statement is not None:
        external = list(outcomes)
        if external or statement.uncertain:
            return evaluate_fulfillment(statement, external)
        if isinstance(persisted_assessment, list):
            # Runtime mirrors its complete projection here on every result and
            # repair. The graph/report aggregates that same assessment rather
            # than guessing completion from node statuses or artifact links.
            return _completion_from_assessments(statement, {
                str(row.get("id")): (
                    {"met": "met", "partial": "partial"}.get(row.get("status"), "unmet"),
                    str(row.get("debt") or ""),
                )
                for row in persisted_assessment if isinstance(row, dict)
            })
        outcomes = external
    marked = []
    for node in node_list:
        attrs = node.get("attrs") or {}
        if attrs.get("container") is True:
            continue
        if attrs.get("obligation") is not True:
            continue
        if str(node.get("type") or "") not in {"ResearchQuestion", "Hypothesis", "Deliverable"}:
            continue
        marked.append(node)
    if not marked and statement is not None:
        return evaluate_fulfillment(statement, outcomes)
    if not marked:
        return _legacy_completion(node_list)
    # Status on the node is the recorded outcome when no external outcome list is passed.
    synthetic: list[OutcomeRecord] = []
    edge_list = list(edges)
    for node in marked:
        attrs = node.get("attrs") or {}
        kind = {"ResearchQuestion": "question", "Hypothesis": "hypothesis",
                "Deliverable": "deliverable"}[str(node["type"])]
        part_id = str(attrs.get("stable_id") or node.get("id"))
        status = str(node.get("status") or "")
        if kind == "hypothesis":
            verdict = status if status in {"confirmed", "refuted", "inconclusive"} else ""
            synthetic.append(OutcomeRecord(
                part_id=part_id,
                executed=bool(verdict) or status == "formulated",
                grounded=bool(attrs.get("grounds") or verdict in {"confirmed", "refuted"}),
                verdict=verdict,
                answer=str(attrs.get("formulation") or ""),
                infrastructure_error=bool(attrs.get("infrastructure_error")),
            ))
        elif kind == "question":
            synthetic.append(OutcomeRecord(
                part_id=part_id,
                executed=status == "closed" or bool(attrs.get("answer")),
                grounded=bool(attrs.get("answer_grounded") or attrs.get("answer")),
                answer_grounded=attrs.get("answer_grounded"),
                answer=str(attrs.get("answer") or ""),
                infrastructure_error=bool(attrs.get("infrastructure_error")),
            ))
        else:
            linked = [
                e for e in edge_list
                if e.get("type") == "satisfies" and e.get("to") == node.get("id")
            ]
            artifacts: list[dict[str, Any]] = []
            if status in {"delivered", "partial"} or linked:
                if attrs.get("location"):
                    artifacts = [{"path": attrs.get("location"), "lab_trace": attrs.get("lab_trace")}]
                elif status == "delivered":
                    artifacts = [{"inline": True, "lab_trace": attrs.get("lab_trace")}]
            synthetic.append(OutcomeRecord(
                part_id=part_id,
                executed=status in {"delivered", "partial"},
                artifacts=artifacts,
                produced=attrs.get("produced_volume"),
                property_verified=attrs.get("property_verified"),
                limitation=str(attrs.get("limitation") or ""),
            ))
    # Build a transient statement from node attrs so the same assessors run.
    from CoScientist.requirements.models import Provenance
    parts = []
    for node in marked:
        attrs = node.get("attrs") or {}
        kind = {"ResearchQuestion": "question", "Hypothesis": "hypothesis",
                "Deliverable": "deliverable"}[str(node["type"])]
        parts.append(RequirementPart(
            id=str(attrs.get("stable_id") or node.get("id")),
            kind=kind,  # type: ignore[arg-type]
            formulation=str(attrs.get("formulation") or ""),
            role="user_obligation",
            obligation=True,
            provenance=Provenance(
                source="user_request",
                quote=str(attrs.get("provenance_quote") or ""),
                accepted=True,
            ),
            hypothesis_mode=attrs.get("hypothesis_mode"),
            physical_sample=bool(attrs.get("physical_sample")),
            protocol_only=bool(attrs.get("protocol_only")),
        ))
    transient = NormalizedStatement(
        source_request="",
        parts=parts,
        root_mode="container",
        uncertain=False,
    )
    external = list(outcomes)
    return evaluate_fulfillment(statement or transient, external or synthetic)


__all__ = ["evaluate_fulfillment", "evaluate_nodes"]
