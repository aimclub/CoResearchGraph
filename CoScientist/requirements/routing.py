"""Validate provenance and structure of the model's semantic statement draft.

Kinds, hypothesis modes, result properties, counts and conditions come from
explicit draft fields. Code never classifies natural language using word lists.
A real quote proves provenance, not that the model understood the request.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Optional

from CoScientist.requirements.models import (
    Criterion,
    NormalizedStatement,
    Provenance,
    RequirementPart,
    RunCondition,
    Volume,
)

_USER_SOURCES = {"user_request", "clarification", "frame_field"}
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _fold(text: str) -> str:
    return _norm(text).casefold()


def quote_in_text(quote: str, text: str) -> bool:
    """The quote must be a real fragment of the trusted text, not a non-empty token."""
    q = _fold(quote)
    if not q:
        return False
    return q in _fold(text)


def _stable_id(kind: str, quote: str, formulation: str = "") -> str:
    """Same quote and formulation stay the same part.

    Two formulations that share one request (two hypotheses the user asked
    to propose) stay distinct. A later paraphrase is joined by
    ``merge_statements`` on the quote, which keeps the earlier id.
    """
    prefix = {"question": "Q", "hypothesis": "H", "deliverable": "DL"}[kind]
    digest = hashlib.sha256(
        f"{kind}|{_fold(quote)}|{_fold(formulation)}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{prefix}-{digest}"


def _as_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return dict(item) if isinstance(item, dict) else {}


def _trusted_text(source: str, quote_against: str, frame_fields: Optional[dict[str, Any]]) -> str:
    if source == "frame_field" and frame_fields:
        return "\n".join(
            str(row.get("value") or "")
            for row in frame_fields.values()
            if isinstance(row, dict) and row.get("accepted")
        )
    return quote_against


def _threshold_grounded(threshold: Optional[str], quote: str) -> bool:
    if threshold is None or str(threshold).strip() == "":
        return True
    token = str(threshold).replace("−", "-").strip()
    hay = quote.replace("−", "-")
    if token and token in hay:
        return True
    # A numeric bar must appear as a number in the quote. "bind" does not justify -7.
    nums = _NUMBER.findall(token)
    if not nums:
        return token.casefold() in hay.casefold()
    hay_nums = set(_NUMBER.findall(hay.replace("−", "-")))
    return all(n in hay_nums for n in nums)


def _volume(raw: Any, trusted: str) -> tuple[Volume, list[dict[str, Any]]]:
    data = _as_dict(raw)
    volume = Volume()
    rejected: list[dict[str, Any]] = []
    if raw is not None and not isinstance(raw, dict) and not hasattr(raw, "model_dump"):
        rejected.append({"kind": "volume", "reason": "volume must be an object"})
    for name in ("requested", "run_cap", "input_count"):
        value = data.get(name)
        if value is None:
            continue
        quote = _norm(str(data.get(f"{name}_quote") or ""))
        if (type(value) is not int or value < 0 or not quote_in_text(quote, trusted)
                or str(value) not in _NUMBER.findall(quote)):
            rejected.append({"kind": "volume", "field": name,
                             "reason": "count lacks a source quote containing its value"})
            continue
        setattr(volume, name, value)
        setattr(volume, f"{name}_quote", quote)
    if volume.run_cap is not None:
        volume.limit_scope = _norm(str(data.get("limit_scope") or ""))
    # The model cannot supply the result of an execution in the request draft.
    return volume, rejected


def _conditions(raw: Any, trusted: str) -> tuple[list[RunCondition], list[dict[str, Any]]]:
    conditions: list[RunCondition] = []
    rejected: list[dict[str, Any]] = []
    if raw is not None and not isinstance(raw, list):
        rejected.append({"kind": "condition", "reason": "conditions must be a list"})
    for item in raw if isinstance(raw, list) else []:
        data = _as_dict(item)
        text = _norm(str(data.get("text") or ""))
        quote = _norm(str(data.get("quote") or ""))
        if text and quote_in_text(quote, trusted):
            conditions.append(RunCondition(text=text, quote=quote))
        else:
            rejected.append({"kind": "condition", "reason": "condition lacks a source quote"})
    return conditions, rejected


def _criteria(raw: Any, *, quote: str, trusted: str) -> tuple[list[Criterion], list[dict[str, Any]]]:
    rows: list[Criterion] = []
    rejected: list[dict[str, Any]] = []
    items = raw if isinstance(raw, list) else []
    for item in items:
        data = _as_dict(item)
        text = _norm(str(data.get("text") or data.get("formulation") or ""))
        if not text:
            continue
        criterion_id = _norm(str(data.get("id") or "")) or "C-" + hashlib.sha256(
            f"{data.get('quote') or quote}|{text}".encode("utf-8")
        ).hexdigest()[:10]
        origin = str(data.get("origin") or "agent_method")
        if origin not in {"user", "agent_method", "technical"}:
            origin = "agent_method"
        cquote = _norm(str(data.get("quote") or ""))
        threshold = data.get("threshold")
        threshold_s = None if threshold is None else str(threshold)
        wants_user = origin == "user"
        basis = cquote or quote
        if wants_user:
            if not quote_in_text(basis, trusted):
                rejected.append({
                    "kind": "criterion",
                    "text": text,
                    "reason": "user criterion lacks a quote that contains its threshold",
                })
                rows.append(Criterion(
                    id=criterion_id,
                    text=text, origin="agent_method", quote=cquote,
                    threshold=threshold_s, obligation=False,
                ))
                continue
            if not _threshold_grounded(threshold_s, basis):
                # The quote is a real obligation even if the model's separate
                # threshold is unsupported (including a reworded quantity).
                # Keep the user's exact criterion, not a guessed conversion or
                # the model's potentially invented numerical formulation.
                rows.append(Criterion(
                    id=criterion_id, text=basis, origin="user", quote=basis,
                    threshold=None, obligation=True,
                ))
                continue
            rows.append(Criterion(
                id=criterion_id,
                text=text, origin="user", quote=basis,
                threshold=threshold_s, obligation=True,
            ))
        else:
            rows.append(Criterion(
                id=criterion_id,
                text=text,
                origin="technical" if origin == "technical" or data.get("technical") else "agent_method",
                quote=cquote,
                threshold=threshold_s,
                obligation=False,
                technical=bool(data.get("technical") or origin == "technical"),
            ))
    seen_ids: set[str] = set()
    for row in rows:
        if row.id in seen_ids:
            rejected.append({"kind": "criterion_id", "reason": "duplicate criterion id", "id": row.id})
        seen_ids.add(row.id)
    return rows, rejected


def _root_mode(parts: list[RequirementPart], *, uncertain: bool) -> str:
    if uncertain or not parts:
        return "uncertain"
    obligations = [p for p in parts if p.obligation]
    questions = [p for p in obligations if p.kind == "question"]
    others = [p for p in obligations if p.kind != "question"]
    if len(questions) == 1 and not others:
        return "question"
    if obligations or parts:
        return "container"
    return "uncertain"


def normalize_statement(
    source_request: str,
    draft: Any = None,
    *,
    frame_fields: Optional[dict[str, Any]] = None,
    version: str = "1",
) -> NormalizedStatement:
    """Validate a draft against the verbatim request.

    ``draft`` is the model output (or None). Missing parts, a missing quote,
    or an explicit uncertainty flag do not become a fulfilled empty obligation
    set — the statement is marked uncertain.
    """
    source = str(source_request or "")
    payload = _as_dict(draft) if not isinstance(draft, list) else {"parts": draft}
    if draft is None:
        payload = {}
    rejected: list[dict[str, Any]] = []
    uncertain = bool(payload.get("uncertain") or payload.get("uncertainty"))
    uncertainty = _norm(str(payload.get("uncertainty") or ""))
    if "clarification" in payload:
        clarification = _as_dict(payload.get("clarification"))
        uncertain = payload.get("clarification") is not None
        uncertainty = _norm(str(clarification.get("question") or ""))
        if uncertain and (not uncertainty or not quote_in_text(str(clarification.get("quote") or ""), source)):
            rejected.append({"kind": "clarification", "reason": "clarification needs a question and a verbatim ambiguous request fragment"})
    raw_parts = payload.get("parts") if isinstance(payload, dict) else None
    if payload.get("clarification") is not None and raw_parts:
        rejected.append({"kind": "clarification", "reason": "A draft cannot both declare its requested parts and require clarification of the request. Put open implementation choices in planning_notes; if the requested scope is unidentifiable, ask a sourced clarification with parts=[]."})
    if not isinstance(raw_parts, list):
        raw_parts = []
        if draft is not None and not payload.get("uncertainty"):
            uncertain = True
            uncertainty = uncertainty or "statement draft has no parts"

    parts: list[RequirementPart] = []
    seen: set[str] = set()
    for item in raw_parts:
        data = _as_dict(item)
        kind = str(data.get("kind") or "").strip().lower()
        if kind not in {"question", "hypothesis", "deliverable"}:
            rejected.append({"kind": kind or "?", "reason": "unknown kind"})
            continue
        formulation = _norm(str(data.get("formulation") or data.get("text") or ""))
        if not formulation:
            rejected.append({"kind": kind, "reason": "empty formulation"})
            continue
        source_kind = str(data.get("source") or "")
        if isinstance(data.get("provenance"), dict):
            prov_in = data["provenance"]
            source_kind = source_kind or str(prov_in.get("source") or "")
            quote = _norm(str(data.get("quote") or prov_in.get("quote") or ""))
            message_id = str(data.get("message_id") or prov_in.get("message_id") or "")
            field_id = str(data.get("field_id") or prov_in.get("field_id") or "")
            part_version = str(data.get("version") or prov_in.get("version") or version)
        else:
            quote = _norm(str(data.get("quote") or ""))
            message_id = str(data.get("message_id") or "")
            field_id = str(data.get("field_id") or "")
            part_version = str(data.get("version") or version)
        role = str(data.get("role") or "user_obligation")
        if source_kind not in _USER_SOURCES | {"agent_proposal"}:
            source_kind = ""
        trusted = _trusted_text(source_kind or "user_request", source, frame_fields)
        if source_kind == "frame_field":
            field = (frame_fields or {}).get(field_id) if field_id else None
            accepted_field = isinstance(field, dict) and bool(field.get("accepted"))
            field_text = str(field.get("value") or "") if isinstance(field, dict) else ""
            quote_ok = accepted_field and quote_in_text(quote, field_text)
        else:
            quote_ok = source_kind in _USER_SOURCES and quote_in_text(quote, trusted)

        # The model does not get to set accepted=true. Acceptance is the quote check.
        if source_kind == "agent_proposal" or not quote_ok:
            if role == "user_obligation" or data.get("obligation") is True:
                rejected.append({
                    "kind": kind,
                    "formulation": formulation,
                    "reason": "user obligation without a quote in the trusted source",
                })
            # Agent proposals of a hypothesis are not created here. A working
            # hypothesis is added later by add_working_hypothesis, not by a
            # draft that invents one "for completeness".
            if kind == "hypothesis" or not quote_ok:
                continue
            role = "supporting"

        quote_body = quote or formulation
        mode = data.get("hypothesis_mode")
        if kind == "hypothesis" and mode not in {"check", "formulate"}:
            rejected.append({"kind": kind, "formulation": formulation,
                             "reason": "hypothesis_mode must be explicit: check or formulate"})
            continue
        if kind != "hypothesis" and mode is not None:
            rejected.append({"kind": kind, "reason": "hypothesis_mode belongs only to a hypothesis"})
            continue
        physical = data.get("physical_sample", False)
        protocol = data.get("protocol_only", False)
        if (type(physical) is not bool or type(protocol) is not bool
                or (physical and protocol) or (kind != "deliverable" and (physical or protocol))):
            rejected.append({"kind": kind, "reason": "invalid deliverable properties"})
            continue
        if role not in {"user_obligation", "supporting"}:
            rejected.append({"kind": kind, "reason": "unknown role"})
            continue

        criteria, crit_rejected = _criteria(data.get("criteria"), quote=quote, trusted=trusted)
        rejected.extend(crit_rejected)
        part_id = _norm(str(data.get("id") or "")) or _stable_id(kind, quote_body, formulation)
        if part_id in seen:
            rejected.append({"kind": kind, "id": part_id, "reason": "duplicate id"})
            continue
        seen.add(part_id)
        obligation = role == "user_obligation" and quote_ok
        parts.append(RequirementPart(
            id=part_id,
            kind=kind,  # type: ignore[arg-type]
            formulation=formulation,
            role="user_obligation" if obligation else "supporting",
            provenance=Provenance(
                source=source_kind if source_kind in _USER_SOURCES else "agent_proposal",  # type: ignore[arg-type]
                quote=quote,
                message_id=message_id,
                field_id=field_id,
                version=part_version,
                accepted=bool(quote_ok and source_kind in _USER_SOURCES),
            ),
            criteria=criteria,
            obligation=obligation,
            statement_version=version,
            hypothesis_mode=mode,  # type: ignore[arg-type]
            physical_sample=physical and kind == "deliverable",
            protocol_only=protocol and kind == "deliverable",
            ready=bool(formulation) and not bool(data.get("draft")),
        ))

    if not parts and not uncertainty:
        uncertain = True
        uncertainty = "no obligation survived normalization"
    # A draft that explicitly parsed to zero because the model set uncertainty
    # stays uncertain. A draft that was rejected down to zero is also uncertain
    # rather than an empty success.
    if payload.get("parts") == [] and not payload.get("uncertainty"):
        uncertain = True
        uncertainty = uncertainty or "empty parts list"

    volume, volume_rejected = _volume(payload.get("volume"), source)
    conditions, condition_rejected = _conditions(payload.get("conditions"), source)
    rejected.extend(volume_rejected)
    rejected.extend(condition_rejected)
    # A surviving row does not make a partially rejected catalog complete.
    structural_rejections = [r for r in rejected if r.get("kind") != "criterion"]
    uncertain = uncertain or bool(structural_rejections)
    if uncertain and not uncertainty:
        uncertainty = "statement draft is incomplete or invalid"
    statement = NormalizedStatement(
        source_request=source,
        version=version,
        parts=parts,
        conditions=conditions,
        volume=volume,
        uncertain=uncertain,
        uncertainty=uncertainty if uncertain else "",
        planning_notes=[_norm(note) for note in payload.get("planning_notes") or [] if isinstance(note, str) and _norm(note)],
        rejected=rejected,
    )
    statement.root_mode = _root_mode(parts, uncertain=statement.uncertain)  # type: ignore[assignment]
    return statement


def modules_for(statement: NormalizedStatement) -> list[str]:
    """Only kinds that are actually present. An empty module is not invoked."""
    present = []
    for kind in ("question", "hypothesis", "deliverable"):
        if statement.of_kind(kind):  # type: ignore[arg-type]
            present.append(kind)
    return present


def merge_statements(
    previous: NormalizedStatement,
    new: NormalizedStatement,
    *,
    retire_ids: Iterable[str] = (),
) -> NormalizedStatement:
    """Re-formulation updates the same part. Silence does not delete it.

    Match stable ids first; use the source quote only when unambiguous.
    """
    retired = {str(i) for i in retire_ids}
    remaining = {part.id: part for part in previous.parts
                 if not part.retired and part.id not in retired}
    incoming_ids = {part.id for part in new.parts}
    merged: list[RequirementPart] = []
    seen: set[str] = set()
    for part in new.parts:
        prior = remaining.pop(part.id, None)
        if prior is None:
            candidates = [old for old in remaining.values()
                          if old.id not in incoming_ids and old.kind == part.kind
                          and _fold(old.provenance.quote) == _fold(part.provenance.quote)]
            exact = [old for old in candidates if old.formulation == part.formulation]
            matches = exact or candidates
            # A shared quote alone cannot identify which requirement changed.
            if len(matches) == 1:
                prior = remaining.pop(matches[0].id)
        if prior is not None and prior.formulation != part.formulation:
            part = part.model_copy(update={
                "id": prior.id,
                "previous_formulation": prior.formulation,
                "previous_requested": previous.volume.requested,
            })
        elif prior is not None:
            part = part.model_copy(update={"id": prior.id})
        elif retired:
            replaced = [
                old for old in previous.parts
                if old.id in retired and old.kind == part.kind
            ]
            if len(replaced) == 1:
                part = part.model_copy(update={
                    "previous_formulation": replaced[0].formulation,
                    "previous_requested": previous.volume.requested,
                })
        if part.id in seen:
            continue
        seen.add(part.id)
        merged.append(part)
    for prior in remaining.values():
        if prior.id in retired or prior.id in seen:
            continue
        seen.add(prior.id)
        merged.append(prior)
    for part in previous.parts:
        if (part.retired or part.id in retired) and part.id not in seen:
            merged.append(part.model_copy(update={"retired": True, "obligation": False}))
    volume = new.volume.model_copy()
    for name in ("requested", "run_cap", "input_count"):
        if getattr(volume, name) is None:
            setattr(volume, name, getattr(previous.volume, name))
            setattr(volume, f"{name}_quote", getattr(previous.volume, f"{name}_quote"))
    if new.volume.run_cap is None:
        volume.limit_scope = previous.volume.limit_scope
    return new.model_copy(update={"parts": merged, "volume": volume})


def retire_parts(statement: NormalizedStatement, part_ids: Iterable[str]) -> NormalizedStatement:
    """Explicit retirement. Omitting a part from a later draft is not retirement."""
    ids = {str(i) for i in part_ids}
    parts = []
    for part in statement.parts:
        if part.id in ids:
            parts.append(part.model_copy(update={"retired": True, "obligation": False}))
        else:
            parts.append(part)
    return statement.model_copy(update={"parts": parts})


def add_working_hypothesis(
    statement: NormalizedStatement,
    formulation: str,
    *,
    because: str,
) -> NormalizedStatement:
    """A supporting claim the research introduces. It is not a new user obligation."""
    text = _norm(formulation)
    if not text or not _norm(because):
        return statement
    part_id = _stable_id("hypothesis", f"agent:{text}", text)
    if any(p.id == part_id for p in statement.parts):
        return statement
    part = RequirementPart(
        id=part_id,
        kind="hypothesis",
        formulation=text,
        role="supporting",
        provenance=Provenance(source="agent_proposal", quote="", accepted=False),
        obligation=False,
        hypothesis_mode="check",
        statement_version=statement.version,
        criteria=[Criterion(text=because, origin="agent_method", obligation=False)],
    )
    return statement.model_copy(update={"parts": [*statement.parts, part]})


__all__ = [
    "add_working_hypothesis",
    "merge_statements",
    "modules_for",
    "normalize_statement",
    "quote_in_text",
    "retire_parts",
]
