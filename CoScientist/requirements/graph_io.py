"""Write a normalized statement into the research graph without duplicating parts.

The graph is the store. Re-running this with the same quotes updates the same
nodes and does not clear evidence already attached to them.
"""
from __future__ import annotations

from typing import Any, Optional

from CoScientist.requirements.models import NormalizedStatement, RequirementPart

_TYPE = {
    "question": "ResearchQuestion",
    "hypothesis": "Hypothesis",
    "deliverable": "Deliverable",
}


def _attrs(part: RequirementPart, *, container: bool = False) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "formulation": part.formulation,
        "stable_id": part.id,
        "obligation": part.obligation,
        "role": part.role,
        "provenance_source": part.provenance.source,
        "provenance_quote": part.provenance.quote,
        "statement_version": part.statement_version,
        "ready": part.ready,
    }
    if part.provenance.message_id:
        attrs["message_id"] = part.provenance.message_id
    if part.hypothesis_mode:
        attrs["hypothesis_mode"] = part.hypothesis_mode
    if part.physical_sample:
        attrs["physical_sample"] = True
    if part.protocol_only:
        attrs["protocol_only"] = True
    if part.previous_formulation:
        attrs["previous_formulation"] = part.previous_formulation
    if container:
        attrs["container"] = True
        attrs["obligation"] = False
    return attrs


def _by_stable_id(store: Any) -> dict[str, dict[str, Any]]:
    full = store.full() if hasattr(store, "full") else {}
    found: dict[str, dict[str, Any]] = {}
    for node in full.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        stable = str((node.get("attrs") or {}).get("stable_id") or "")
        if stable:
            found[stable] = node
    return found


def commit_statement(store: Any, statement: NormalizedStatement) -> dict[str, Any]:
    """Create or update parts. Retired parts are marked, not deleted.

    When ``store`` is missing (research graph switched off) the statement is
    returned and nothing is invented, including no placeholder hypothesis.
    """
    if store is None:
        return {
            "ok": True,
            "persisted": False,
            "node_ids": {},
            "statement": statement.model_dump(),
        }
    existing = _by_stable_id(store)
    root_id = store.root_id() if hasattr(store, "root_id") else None
    if not root_id or (hasattr(store, "is_empty") and store.is_empty()):
        root_formulation = statement.source_request
        question_parts = [p for p in statement.obligations() if p.kind == "question"]
        if statement.root_mode == "question" and question_parts:
            root_formulation = question_parts[0].formulation
        attrs = {
            "original_request": statement.source_request,
            "container": statement.root_mode == "container",
            "obligation": statement.root_mode == "question",
            "normalized_statement": statement.model_dump(),
            "requirement_assessment": None,
            "stable_id": question_parts[0].id if statement.root_mode == "question" and question_parts else "",
            "requires_hypothesis": any(
                p.kind == "hypothesis" and p.obligation and p.hypothesis_mode == "check"
                for p in statement.parts
            ),
        }
        if statement.root_mode == "question" and question_parts:
            attrs["provenance_quote"] = question_parts[0].provenance.quote
            attrs["provenance_source"] = question_parts[0].provenance.source
        result = store.init_research(
            source="ContextInitAgent",
            question=root_formulation,
            attrs=attrs,
            question_source="human" if statement.root_mode == "question" else "ContextInitAgent",
        )
        if not result.get("ok"):
            return result
        existing = _by_stable_id(store)
        root_id = result.get("root_id") or store.root_id()

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    # Keep the verbatim request on the root even when the root is only a container.
    full = store.full() if hasattr(store, "full") else {}
    root_attrs = next(((node.get("attrs") or {}) for node in full.get("nodes") or []
                       if node.get("id") == root_id), {})
    assessment = root_attrs.get("requirement_assessment") if root_attrs.get("normalized_statement") == statement.model_dump() else None
    nodes.append({
        "id": root_id,
        "attrs": {
            "original_request": statement.source_request,
            "container": statement.root_mode == "container",
            "obligation": statement.root_mode == "question",
            "normalized_statement": statement.model_dump(),
            "requirement_assessment": assessment,
            "requires_hypothesis": any(
                p.kind == "hypothesis" and p.obligation and p.hypothesis_mode == "check"
                for p in statement.parts
            ),
        },
    })
    for part in statement.parts:
        if statement.root_mode == "question" and part.kind == "question" and part.obligation:
            prior = existing.get(part.id)
            if prior and prior.get("id") == root_id:
                nodes.append({"id": root_id, "attrs": _attrs(part)})
            elif not prior:
                nodes.append({"id": root_id, "attrs": _attrs(part)})
            else:
                nodes.append({"id": prior["id"], "attrs": _attrs(part)})
            continue
        prior = existing.get(part.id)
        if prior:
            nodes.append({"id": prior["id"], "attrs": _attrs(part)})
            continue
        ref = part.id
        nodes.append({
            "type": _TYPE[part.kind],
            "ref": ref,
            "status": _initial_status(part),
            "attrs": _attrs(part),
        })
        if part.kind == "deliverable":
            edges.append({"type": "asks_for", "from": root_id, "to": f"#{ref}"})
        elif part.kind == "hypothesis":
            edges.append({"type": "motivates", "from": root_id, "to": f"#{ref}"})
        elif part.kind == "question":
            edges.append({"type": "contains", "from": root_id, "to": f"#{ref}"})
    result = store.commit(
        source="ContextInitAgent",
        nodes=nodes,
        edges=edges,
        enforce_permissions=False,
    )
    dumped = result.model_dump() if hasattr(result, "model_dump") else result
    saved = _by_stable_id(store)
    node_ids = {
        part.id: saved[part.id]["id"]
        for part in statement.parts
        if part.id in saved
    }
    return {
        "ok": bool(dumped.get("ok")),
        "persisted": True,
        "node_ids": node_ids,
        "result": dumped,
    }


def _initial_status(part: RequirementPart) -> str:
    if part.kind == "hypothesis":
        return "formulated"
    if part.kind == "deliverable":
        return "specified"
    return "open"


__all__ = ["commit_statement"]
