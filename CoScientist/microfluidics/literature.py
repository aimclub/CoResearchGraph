"""Module A plumbing: keep every literature result and the target molecule.

Three callbacks close the gaps between the literature stage and the modules
that read it:

``collect_literature_finding`` (after_agent, ResearchAgent)
    ResearchAgent writes its answer to ``search_results`` on every call, so each
    LIT-xx task overwrote the one before and only the last survived. This
    callback stores every answer both in the legacy ``literature_findings``
    list and under a query-specific key (``literature_finding_LIT_01`` etc.).
    The distinct keys matter when several AgentTool calls run concurrently:
    their state deltas can then be merged without one whole-list write winning
    over another.

``inject_target_molecule`` (before_agent)
    Reads the customer's target molecule out of the structured ТЗ into
    ``target_molecule``, so the prompts downstream see it as a field instead of
    digging through 16 blocks.

``pin_target_molecule`` (after_agent, LiteratureSynthesisAgent)
    When the ТЗ fixes the molecule, the ТЗ wins over whatever the LLM wrote into
    ``literature_analysis.target_molecule``.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

from google.adk.agents.callback_context import CallbackContext

from CoScientist.hitl.field_status import OPEN_STATUSES
from CoScientist.microfluidics.models import (
    LiteratureAnalysis,
    LiteratureFact,
    LiteratureSelection,
    SourceRecord,
    StructuredTZ,
    SynthesisRoutes,
    TargetMolecule,
)
from CoScientist.microfluidics.chemistry_identity import known_smiles

logger = logging.getLogger(__name__)

FINDINGS_KEY = "literature_findings"
FINDING_KEY_PREFIX = "literature_finding_"
TARGET_KEY = "target_molecule"
ANALYSIS_KEY = "literature_analysis"
SELECTION_KEY = "literature_selection"
SELECTED_FINDINGS_KEY = "selected_literature_findings"

_QUERY_ID = re.compile(r"\bLIT-\d+\b", re.IGNORECASE)
_EMPTY = {"", "-", "—", "не задано", "не задан", "не задана", "нет данных"}
_YES = ("да", "yes", "true")
_URL = re.compile(r"https?://[^\s<>\]\[)]+", re.IGNORECASE)


# ── Target molecule ──────────────────────────────────────────────────────────

def _field(tz: StructuredTZ, block: str, *needles: str) -> str:
    """Value of the first set field of ``block`` whose name contains a needle.

    Field names are written by the LLM from a guide, not fixed, so they are
    matched by substring.
    """
    b = tz.block(block)
    if b is None:
        return ""
    for row in b.fields:
        name = row.name.lower()
        if any(n in name for n in needles):
            value = row.value.strip()
            if row.status not in OPEN_STATUSES and value.lower() not in _EMPTY:
                return value
    return ""


def extract_target_molecule(structured_tz: Any) -> TargetMolecule:
    """The customer's target molecule as the ТЗ states it."""
    try:
        tz = (
            structured_tz
            if isinstance(structured_tz, StructuredTZ)
            else StructuredTZ.model_validate(structured_tz or {})
        )
    except Exception as exc:  # noqa: BLE001 — a malformed ТЗ must not break the run
        logger.warning("target molecule: unreadable structured_tz: %s", exc)
        return TargetMolecule()

    name = _field(tz, "Целевой продукт", "целевое вещество", "целевая молекула")
    smiles = _field(tz, "Целевой продукт", "smiles")
    cas = _field(tz, "Целевой продукт", "cas")
    fixed_flag = _field(tz, "Тип задачи", "фиксированн").lower()
    fixed = fixed_flag.startswith(_YES) or (not fixed_flag and bool(smiles or cas))

    if not (name or smiles or cas):
        return TargetMolecule(fixed=False)
    # Known identities are code-owned: an LLM-generated positional isomer must
    # never override an unambiguous customer name such as "ванилин".
    smiles = known_smiles(name) or smiles
    return TargetMolecule(fixed=fixed, name=name, smiles=smiles, cas=cas, source="ТЗ")


def inject_target_molecule(callback_context: CallbackContext) -> None:
    """Refresh ``target_molecule`` from the current ТЗ (idempotent)."""
    tz = callback_context.state.get("structured_tz")
    if not tz:
        return None
    callback_context.state[TARGET_KEY] = extract_target_molecule(tz).model_dump()
    return None


def pin_target_molecule(callback_context: CallbackContext) -> None:
    """Make the ТЗ's fixed molecule win inside ``literature_analysis``."""
    state = callback_context.state
    analysis = state.get(ANALYSIS_KEY)
    tz = state.get("structured_tz")
    if not isinstance(analysis, dict) or not tz:
        return None
    target = extract_target_molecule(tz)
    if not target.fixed:
        return None
    state[ANALYSIS_KEY] = {**analysis, "target_molecule": target.model_dump()}
    return None


# ── Literature findings ──────────────────────────────────────────────────────

def _request_text(callback_context: CallbackContext) -> str:
    content = getattr(callback_context, "user_content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(p, "text", None) or "" for p in parts).strip()


def collect_literature_finding(callback_context: CallbackContext) -> None:
    """Append this ResearchAgent answer to ``literature_findings``."""
    state = callback_context.state
    result = state.get("search_results")
    if not result:
        return None
    request = _request_text(callback_context)
    match = _QUERY_ID.search(request) or _QUERY_ID.search(str(result)[:500])
    finding = {
        "query_id": match.group(0).upper() if match else "",
        "request": request,
        "result": result,
    }

    # Parallel AgentTool calls start from the same parent-state snapshot.  A
    # shared list is consequently a last-writer-wins value, whereas distinct
    # keys survive ADK's merged state delta.  Keep the list for compatibility
    # with sequential/general profiles and make the per-query key canonical for
    # the parallel microfluidics literature stage.
    if finding["query_id"]:
        state[f"{FINDING_KEY_PREFIX}{finding['query_id'].replace('-', '_')}"] = finding

    findings = list(state.get(FINDINGS_KEY) or [])
    # The AgentTool copies the parent state in, so a call that wrote nothing
    # still sees the previous task's search_results — do not record it twice.
    if any(f.get("result") == result for f in findings):
        return None
    # A retry of the same LIT-xx replaces the earlier answer instead of piling up.
    if finding["query_id"]:
        findings = [f for f in findings if f.get("query_id") != finding["query_id"]]
    findings.append(finding)
    state[FINDINGS_KEY] = findings
    return None


def _state_snapshot(state: Any) -> dict[str, Any]:
    """Read ADK ``State`` and ordinary mappings through one safe interface."""
    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        try:
            value = to_dict()
            return value if isinstance(value, dict) else {}
        except Exception as exc:  # noqa: BLE001 - fallback paths must not abort a run
            logger.warning("literature state snapshot failed: %s", exc)
    if isinstance(state, Mapping):
        return dict(state)
    return {}


def _finding_index(state: Any) -> dict[str, dict[str, Any]]:
    """Return one canonical finding per LIT id from parallel-safe state."""
    indexed: dict[str, dict[str, Any]] = {}
    for key, value in _state_snapshot(state).items():
        if not key.startswith(FINDING_KEY_PREFIX) or not isinstance(value, dict):
            continue
        fallback = key.removeprefix(FINDING_KEY_PREFIX).replace("_", "-")
        query_id = str(value.get("query_id") or fallback).upper()
        if query_id:
            indexed[query_id] = value
    for value in state.get(FINDINGS_KEY) or []:
        if isinstance(value, dict):
            query_id = str(value.get("query_id") or "").upper()
            if query_id and query_id not in indexed:
                indexed[query_id] = value
    return indexed


def _result_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else str(value or "").strip()


def _sources_from_text(text: str) -> list[SourceRecord]:
    """Extract only resolvable identifiers; verification remains false."""
    identifiers = list(dict.fromkeys(
        item.rstrip(".,;)") for item in _URL.findall(text)
    ))[:8]
    records: list[SourceRecord] = []
    for index, identifier in enumerate(identifiers, 1):
        records.append(SourceRecord(
            source_id=f"src-{index:02d}",
            url=identifier,
            source_type="web",
        ))
    return records


def _assemble_selected_literature(callback_context: CallbackContext) -> None:
    """Build a valid minimal analysis from at most two raw research findings.

    The LLM selects existing LIT ids only.  Its malformed answer therefore
    degrades to an empty, explicitly documented analysis rather than taking
    Module A down during schema validation.
    """
    state = callback_context.state
    available = _finding_index(state)
    try:
        selection = LiteratureSelection.model_validate(state.get(SELECTION_KEY) or {})
    except Exception as exc:  # defensive: malformed model output must be fail-soft
        logger.warning("literature selection unreadable: %s", exc)
        selection = LiteratureSelection(warnings=["Выбор литературы не удалось разобрать."])

    selected_ids = [item for item in selection.selected_ids if item in available][:2]
    unknown_ids = [item for item in selection.selected_ids if item not in available]
    selected = [available[item] for item in selected_ids]
    state[SELECTED_FINDINGS_KEY] = selected

    source_records: list[SourceRecord] = []
    facts: list[LiteratureFact] = []
    seen_sources: set[str] = set()
    for finding in selected:
        text = _result_text(finding.get("result"))
        sources = _sources_from_text(text)
        for source in sources:
            identity = source.url
            if identity in seen_sources:
                continue
            seen_sources.add(identity)
            source.source_id = f"src-{len(source_records) + 1:02d}"
            source_records.append(source)
        facts.append(LiteratureFact(
            statement=text[:12000] or "Результат исследования не содержит текста.",
            query_id=str(finding.get("query_id") or ""),
            sources=[source.url for source in sources],
        ))

    gaps = list(selection.warnings)
    if unknown_ids:
        gaps.append("Выбор содержит неизвестные результаты: " + ", ".join(unknown_ids))
    if not selected:
        gaps.append("Подходящие результаты не выбраны; доступен только отчёт по исследованию.")
    else:
        gaps.append("Выбраны исходные результаты исследования; маршруты требуют отдельной ручной экстракции.")
    if selection.reason:
        gaps.append("Причина выбора: " + selection.reason)

    analysis = LiteratureAnalysis(
        target_molecule=extract_target_molecule(state.get("structured_tz")),
        source_records=source_records,
        facts=facts,
        gaps=list(dict.fromkeys(gaps)),
    )
    state[ANALYSIS_KEY] = analysis.model_dump()
    # Research findings are deliberately not hallucinated into chemical route
    # records.  Downstream code treats this explicit empty hand-off as a
    # no-route condition and skips route-dependent work safely.
    state["synthesis_routes"] = SynthesisRoutes(routes=[], gaps=analysis.gaps).model_dump()
    state[SELECTION_KEY] = LiteratureSelection(
        selected_ids=selected_ids, reason=selection.reason, warnings=selection.warnings,
    ).model_dump()
    return None


def assemble_selected_literature(callback_context: CallbackContext) -> None:
    """Never let a malformed result or an ADK state variant abort Module A."""
    try:
        return _assemble_selected_literature(callback_context)
    except Exception as exc:  # noqa: BLE001 - this callback is a stage boundary
        logger.exception("literature hand-off degraded to safe fallback: %s", exc)
        state = getattr(callback_context, "state", None)
        if state is None:
            return None
        try:
            target = extract_target_molecule(state.get("structured_tz"))
            message = (
                "Не удалось собрать выбранные результаты; исследование сохранено "
                f"для ручного просмотра ({type(exc).__name__})."
            )
            analysis = LiteratureAnalysis(target_molecule=target, gaps=[message])
            state[ANALYSIS_KEY] = analysis.model_dump()
            state[SELECTED_FINDINGS_KEY] = []
            state[SELECTION_KEY] = LiteratureSelection(
                warnings=[message],
            ).model_dump()
            state["synthesis_routes"] = SynthesisRoutes(
                routes=[], gaps=[message],
            ).model_dump()
        except Exception as fallback_exc:  # noqa: BLE001
            # Some future ADK State implementation may itself be unavailable;
            # logging is still preferable to replacing the original failure.
            logger.exception("could not persist literature fallback: %s", fallback_exc)
        return None


__all__ = [
    "ANALYSIS_KEY",
    "FINDINGS_KEY",
    "FINDING_KEY_PREFIX",
    "SELECTION_KEY",
    "SELECTED_FINDINGS_KEY",
    "TARGET_KEY",
    "assemble_selected_literature",
    "collect_literature_finding",
    "extract_target_molecule",
    "inject_target_molecule",
    "pin_target_molecule",
]
