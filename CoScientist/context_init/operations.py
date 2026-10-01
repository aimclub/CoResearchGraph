"""Authoritative experiment operations from the research frame / ask structure.

Slots are first-class: the planner may only choose a route, not invent, merge,
or drop them. Extraction uses the ask's own numbered/separated structure,
imperative sentence glue (unnumbered multi-target asks), and the frame's
``operations`` field — not domain keyword tables.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, List

from CoScientist.context_init.models import FrameOperation, ResearchFrame

MAX_OPERATIONS = 20
OPS_FORM_BLOCK = "Задачи исследования"

_NUMBERED_STEP_RE = re.compile(
    r"(?P<n>\d{1,2})[.)]\s+(?P<body>\S.+?)(?=\s+\d{1,2}[.)]\s+|\Z)",
    re.DOTALL,
)
_BULLET_RE = re.compile(
    r"(?:^|\n)\s*[-*•]\s+(\S.+?)(?=(?:\n\s*[-*•]\s+)|\Z)",
    re.DOTALL,
)
_IMPERATIVE_HEAD_RE = re.compile(
    r"(?i)^(generate|design|develop|suggest|discover|propose|dock|"
    r"calculate|predict|compute|curate|cluster)\b"
)


_NARRATIVE_RE = re.compile(
    r"(?i)"
    r"(?:^|\b)(?:write|draft|prepare)\s+(?:a\s+)?(?:final\s+)?(?:comprehensive\s+)?"
    r"(?:narrative\s+)?(?:report|write-?up)\b"
    r"|(?:synthesize|synthesis of)\s+(?:a\s+)?(?:comprehensive\s+)?(?:\w+\s+)?"
    r"(?:report|findings)"
    r"|(?:summarize|summary of)\s+findings"
    r"|comprehensive\s+(?:toxicological\s+)?report(?:\s+synthesis)?"
    r"|\bвыводы\b"
    r"|итоговый\s+отч[её]т"
    r"|(?:^|\b)отч[её]т\b"
)


_EVIDENCE_HEAD_RE = re.compile(r"(?i)^(literature|литератур)")
_EVIDENCE_BODY_RE = re.compile(
    r"(?i)\bliterature\b|литератур|pubmed|openalex|обзор публикац"
)
_EVIDENCE_COMPUTE_RE = re.compile(
    r"(?i)\b(cluster|model|dock|generat|predict|curate|dataset|"
    r"кластер|модел|данн)\b"
)


def is_narrative_report_operation(statement: str) -> bool:
    """True for a report/synthesis-only slot (ResultAggregator, not a plan task)."""
    text = re.sub(r"\s+", " ", statement or "").strip()
    if len(text) < 4:
        return True
    if re.search(r"(?i)\bliterature\b|литератур", text):
        return False
    # Title/first clause only — body text may mention "выводы" without being a report slot.
    head = text.split(".")[0].strip()
    if re.match(r"(?i)^(report|write-?up|отч[её]т|выводы)\b", head):
        return True
    return bool(_NARRATIVE_RE.search(head))


def is_evidence_operation(statement: str) -> bool:
    """True for a literature/review slot (research route, not compute MCP / Alembic)."""
    text = re.sub(r"\s+", " ", statement or "").strip()
    if not text or is_narrative_report_operation(text):
        return True
    head = text.split(".")[0].strip()
    if _EVIDENCE_HEAD_RE.match(head):
        return True
    return bool(_EVIDENCE_BODY_RE.search(text) and not _EVIDENCE_COMPUTE_RE.search(text))


def _clean_statement(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip(" ;.")


def _rows_from_statements(statements: Iterable[str]) -> List[FrameOperation]:
    out: List[FrameOperation] = []
    seen: set[str] = set()
    for raw in statements:
        statement = _clean_statement(str(raw or ""))
        if len(statement) < 8 or is_narrative_report_operation(statement):
            continue
        key = statement.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(FrameOperation(
            operation_id=f"OP-{len(out) + 1}",
            statement=statement[:500],
        ))
        if len(out) >= MAX_OPERATIONS:
            break
    return out


def parse_numbered_operations(text: str) -> List[FrameOperation]:
    """Longest consecutive 1..k run in the ask (k≥2). Mid-paragraph ``1.`` counts."""
    if not (text or "").strip():
        return []
    matches: list[tuple[int, int, str]] = []
    for match in _NUMBERED_STEP_RE.finditer(text):
        body = _clean_statement(match.group("body"))
        if len(body) < 8:
            continue
        matches.append((match.start(), int(match.group("n")), body))
    best: list[str] = []
    for i, (_pos, number, body) in enumerate(matches):
        if number != 1:
            continue
        run = [body]
        expect = 2
        j = i + 1
        while j < len(matches) and matches[j][1] == expect:
            run.append(matches[j][2])
            expect += 1
            j += 1
        if len(run) > len(best):
            best = run
    return _rows_from_statements(best) if len(best) >= 2 else []


def parse_bullet_operations(text: str) -> List[FrameOperation]:
    if not (text or "").strip():
        return []
    bodies = [_clean_statement(m.group(1)) for m in _BULLET_RE.finditer(text)]
    rows = _rows_from_statements(bodies)
    return rows if len(rows) >= 2 else []


def parse_glued_imperative_operations(text: str) -> List[FrameOperation]:
    """Split unnumbered glue on sentence boundaries when each clause is an imperative.

    User-facing asks stay unnumbered; OP-1…N are internal slots only.
    """
    if not (text or "").strip():
        return []
    parts = [
        _clean_statement(part)
        for part in re.split(r"(?<=[.!?])\s+", text)
        if _clean_statement(part)
    ]
    if len(parts) < 2:
        return []
    if not all(_IMPERATIVE_HEAD_RE.match(part) for part in parts):
        return []
    # Keep repeats: 3L+1S glue reuses the same generate lines on purpose.
    out: List[FrameOperation] = []
    for part in parts:
        if len(part) < 8 or is_narrative_report_operation(part):
            continue
        out.append(FrameOperation(
            operation_id=f"OP-{len(out) + 1}",
            statement=part[:500],
        ))
        if len(out) >= MAX_OPERATIONS:
            break
    return out if len(out) >= 2 else []


def parse_decomposition_operations(value: str) -> List[FrameOperation]:
    """JSON list, newline list, or numbered text in the frame decomposition field."""
    text = str(value or "").strip()
    if not text or text.lower() in {"не задано", "n/a", "none"}:
        return []
    parsed: Any = None
    if text[0] in "[{":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, list):
        statements: list[str] = []
        for item in parsed:
            if isinstance(item, str):
                statements.append(item)
            elif isinstance(item, dict):
                statements.append(str(
                    item.get("statement") or item.get("content") or item.get("text") or ""
                ))
        rows = _rows_from_statements(statements)
        return rows if len(rows) >= 2 else rows
    numbered = parse_numbered_operations(text)
    if numbered:
        return numbered
    bullets = parse_bullet_operations(text)
    if bullets:
        return bullets
    lines = [_clean_statement(line.lstrip("-*• ").strip()) for line in text.splitlines()]
    rows = _rows_from_statements(line for line in lines if line)
    return rows if len(rows) >= 2 else []


def normalize_operation_rows(raw: Any) -> List[dict[str, str]]:
    """Coerce session/context payloads into ``{operation_id, statement}`` rows."""
    if not raw:
        return []
    statements: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, FrameOperation):
                statements.append(item.statement)
            elif isinstance(item, str):
                statements.append(item)
            elif isinstance(item, dict):
                statements.append(str(
                    item.get("statement") or item.get("content") or item.get("text") or ""
                ))
    elif isinstance(raw, str):
        return [row.model_dump() for row in parse_decomposition_operations(raw)]
    return [row.model_dump() for row in _rows_from_statements(statements)]


def _decomposition_value(frame: ResearchFrame) -> str:
    block = frame.block("Вопрос исследования")
    if block is None:
        return ""
    for field in block.fields:
        if field.name == "decomposition" and field.is_set():
            return field.value
    return ""


def extract_frame_operations(frame: ResearchFrame) -> List[FrameOperation]:
    """Ask structure first (numbered/bullets/glue), then LLM/HITL ops, then decomposition."""
    numbered = parse_numbered_operations(frame.original_request)
    if len(numbered) >= 2:
        return numbered
    bullets = parse_bullet_operations(frame.original_request)
    if len(bullets) >= 2:
        return bullets
    glued = parse_glued_imperative_operations(frame.original_request)
    if len(glued) >= 2:
        return glued
    existing = _rows_from_statements(op.statement for op in (frame.operations or []))
    if existing:
        return existing
    return parse_decomposition_operations(_decomposition_value(frame))


def fill_operations_if_missing(frame: ResearchFrame) -> ResearchFrame:
    """Prefer a longer numbered/bulleted/glued ask list over a collapsed LLM draft."""
    structured = parse_numbered_operations(frame.original_request)
    if len(structured) < 2:
        structured = parse_bullet_operations(frame.original_request)
    if len(structured) < 2:
        structured = parse_glued_imperative_operations(frame.original_request)
        if len(structured) >= 2:
            frame.operations = structured
            return frame
    current = _rows_from_statements(op.statement for op in (frame.operations or []))
    if len(structured) >= 2 and len(current) < len(structured):
        frame.operations = structured
        return frame
    if current:
        frame.operations = current
        return frame
    decomp = parse_decomposition_operations(_decomposition_value(frame))
    if decomp:
        frame.operations = decomp
    return frame


def operations_as_dicts(frame: ResearchFrame) -> List[dict[str, str]]:
    filled = fill_operations_if_missing(frame.model_copy(deep=True))
    return [op.model_dump() for op in filled.operations]


__all__ = [
    "MAX_OPERATIONS",
    "OPS_FORM_BLOCK",
    "extract_frame_operations",
    "fill_operations_if_missing",
    "is_evidence_operation",
    "is_narrative_report_operation",
    "normalize_operation_rows",
    "operations_as_dicts",
    "parse_decomposition_operations",
    "parse_glued_imperative_operations",
    "parse_numbered_operations",
]
