"""Authenticate evidence claims at the tool boundary, not from LLM assertions."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext

from CoScientist.microfluidics.models import LiteratureAnalysis

TRACE_KEY = "_evidence_verifier_trace"
DRAFT_KEY = "literature_analysis_draft"
logger = logging.getLogger(__name__)
_URL = re.compile(r"https?://[^\s\"'<>]+", re.I)
_PATENT = re.compile(r"\b(?:WO|EP|US|RU)\s*[-/]?\s*\d{5,}[A-Z]\d?\b", re.I)
_STANDARD = re.compile(r"\b(?:ASTM|ISO|EN|GOST|ГОСТ)\s+[A-ZА-Я0-9][A-ZА-Я0-9.:-]*", re.I)
_FULL_TEXT_TOOL = re.compile(r"(?:extract|explore|analy[sz]|read|full.?text)", re.I)
_NUMERIC_CLAIM = re.compile(r"\d")


def _serialized(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 - audit must fail closed, never break a run
        return str(value)


def _identifiers(text: str) -> set[str]:
    found = {item.rstrip(".,);]") for item in _URL.findall(text)}
    found.update(" ".join(item.split()).casefold() for item in _PATENT.findall(text))
    found.update(" ".join(item.split()).casefold() for item in _STANDARD.findall(text))
    return found


def _response_failed(response: Any) -> bool:
    if response is None:
        return True
    if isinstance(response, dict):
        if response.get("isError") is True or response.get("error"):
            return True
        content = response.get("content")
        if isinstance(content, list) and any(
            isinstance(item, dict) and item.get("isError") is True for item in content
        ):
            return True
    text = _serialized(response).strip().casefold()
    return len(text) < 20 or text.startswith(("error", "tool error"))


def begin_evidence_verification(callback_context: CallbackContext) -> None:
    callback_context.state[TRACE_KEY] = []
    return None


def capture_evidence_verification(
    tool: Any = None,
    args: Any = None,
    tool_context: Any = None,
    tool_response: Any = None,
    **kwargs: Any,
) -> None:
    """Record immutable hashes and identifiers from actual verifier tool results."""
    if tool_context is None:
        return None
    response = tool_response if tool_response is not None else kwargs.get("response")
    response_text = _serialized(response)
    args_text = _serialized(args)
    name = str(getattr(tool, "name", "") or "")
    trace = list(tool_context.state.get(TRACE_KEY) or [])
    trace.append({
        "tool": name,
        "full_text_capable": bool(_FULL_TEXT_TOOL.search(name)),
        "successful": not _response_failed(response),
        "requested_identifiers": sorted(_identifiers(args_text)),
        "returned_identifiers": sorted(_identifiers(response_text)),
        "content_hash": "sha256:" + hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
    })
    tool_context.state[TRACE_KEY] = trace
    return None


def authenticate_analysis(analysis: Any, trace: Any) -> LiteratureAnalysis:
    """Authenticate only high-impact numeric route claims at the tool boundary.

    The verifier is deliberately not a second literature-discovery agent:
    qualitative facts and broad route provenance stay ``unverified`` even if a
    source was opened. Numeric operation conditions and stage yields can be
    promoted only when the tool trace proves that the exact source was read.
    """
    model = analysis if isinstance(analysis, LiteratureAnalysis) else LiteratureAnalysis.model_validate(analysis or {})
    entries = [
        entry for entry in (trace or [])
        if isinstance(entry, dict)
        and entry.get("full_text_capable")
        and entry.get("successful", True)
    ]
    authenticated: dict[str, dict] = {}
    for source in model.source_records:
        candidates = {source.url.rstrip(".,);]").casefold()} if source.url else set()
        if source.external_id:
            candidates.add(" ".join(source.external_id.split()).casefold())
        match = next((entry for entry in entries if candidates.intersection({
            str(item).casefold()
            for key in ("returned_identifiers", "requested_identifiers", "identifiers")
            for item in (entry.get(key) or [])
        })), None)
        if match is None:
            source.full_text_available = False
            source.content_hash = ""
            source.verified_by = ""
            source.verification_tool = ""
            continue
        source.full_text_available = True
        source.content_hash = str(match.get("content_hash") or "")
        source.verified_by = "evidence_verifier"
        source.verification_tool = str(match.get("tool") or "")
        authenticated[source.source_id] = match

    def mark_unverified(refs: list[Any]) -> None:
        for ref in refs:
            ref.verification_status = "unverified"

    def secure_numeric_refs(refs: list[Any]) -> None:
        for ref in refs:
            if ref.source_id not in authenticated or not ref.locator.strip():
                ref.verification_status = "unverified"

    for analogue in model.analogues:
        for prop in analogue.properties:
            mark_unverified(prop.evidence)
    for route in model.synthesis_routes:
        mark_unverified(route.evidence)
        for step in route.steps:
            if _NUMERIC_CLAIM.search(step.yield_value or ""):
                secure_numeric_refs(step.evidence)
            else:
                mark_unverified(step.evidence)
            for condition in step.conditions:
                if _NUMERIC_CLAIM.search(condition.value or ""):
                    secure_numeric_refs(condition.evidence)
                else:
                    mark_unverified(condition.evidence)
    for fact in model.facts:
        mark_unverified(fact.evidence)
    return model


def _lost_draft_content(verified: Any, draft: Any) -> list[str]:
    """What the verifier's answer dropped from the synthesis draft.

    The verifier only re-grades claims; it never removes routes or sources.
    An answer without them is a failed structured answer (a malformed call
    validates to an empty analysis), not a verdict.
    """
    try:
        before = LiteratureAnalysis.model_validate(draft or {})
        after = LiteratureAnalysis.model_validate(verified or {})
    except Exception:  # noqa: BLE001 - an unreadable draft has nothing to protect
        return []
    lost = []
    missing_routes = (
        {route.route_id for route in before.synthesis_routes}
        - {route.route_id for route in after.synthesis_routes}
    )
    if missing_routes:
        lost.append(f"routes {sorted(missing_routes)}")
    if before.source_records and not after.source_records:
        lost.append(f"{len(before.source_records)} source records")
    return lost


def authenticate_evidence_verification(callback_context: CallbackContext) -> None:
    state = callback_context.state
    analysis = state.get("literature_analysis")
    lost = _lost_draft_content(analysis, state.get(DRAFT_KEY))
    if lost:
        # Authenticate the draft instead: its claims stay unverified unless
        # the trace proves the exact source was read.
        logger.warning(
            "evidence verifier answer dropped %s from the draft — authenticating the draft",
            ", ".join(lost),
        )
        analysis = state.get(DRAFT_KEY)
    state["literature_analysis"] = authenticate_analysis(analysis, state.get(TRACE_KEY)).model_dump()
    return None


__all__ = [
    "TRACE_KEY",
    "authenticate_analysis",
    "authenticate_evidence_verification",
    "begin_evidence_verification",
    "capture_evidence_verification",
]
