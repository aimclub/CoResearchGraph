"""Compile the free-form TZ tables into an atomic, provenance-preserving contract.

The structured TZ is deliberately human-readable and its field names are not a
stable API.  Downstream chemistry code therefore must not scrape it independently
or trust a module's prose paraphrase.  This module performs one conservative
compilation pass and records anything it cannot make machine-checkable as an open
question instead of inventing a threshold or a policy.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from google.adk.agents.callback_context import CallbackContext

from CoScientist.hitl.field_status import OPEN_STATUSES
from CoScientist.microfluidics.models import (
    RequirementConstraint,
    RequirementSource,
    RequirementsSpec,
    StructuredTZ,
    TZFieldRow,
)

REQUIREMENTS_KEY = "requirements_spec"

_CONFIRMED_STATUSES = {"задано заказчиком", "уточнено оператором"}
_RANGE_C = re.compile(
    r"(?P<minimum>[+-]?\d+(?:[.,]\d+)?)\s*(?:[–—-]|\.\.)\s*"
    r"(?P<maximum>[+-]?\d+(?:[.,]\d+)?)\s*°?\s*[CcСс]"
)
_NUMBER_PERCENT = re.compile(r"(?P<number>\d+(?:[.,]\d+)?)\s*%")
_NUMBER_MASS = re.compile(r"(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>мг|г|кг|mg|g|kg)\b", re.I)
_EXPLICIT_SOLVENT_LIST = re.compile(
    r"(?:запрещ(?:енные|ены)?|исключить|forbidden|exclude(?:d)?)"
    r"[^:;]{0,40}(?:растворител\w*|solvents?)\s*:\s*(?P<items>[^.;]+)",
    re.I,
)


def _explicit_solvent_names(value: str) -> list[str]:
    """Parse only a clearly labelled deny-list; never guess what «harmful» means."""
    match = _EXPLICIT_SOLVENT_LIST.search(value)
    if not match:
        return []
    items = re.split(r"\s*(?:,|;|/|\n|\band\b|\bи\b)\s*", match.group("items"), flags=re.I)
    return [item.strip(" -\t") for item in items if item.strip(" -\t")]


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").split())


def _rows(tz: StructuredTZ) -> Iterable[tuple[str, TZFieldRow]]:
    for block in tz.blocks:
        for row in block.fields:
            if row.status not in OPEN_STATUSES and _norm(row.value) not in {"", "не задано"}:
                yield block.title, row


def _source(block: str, row: TZFieldRow) -> RequirementSource:
    return RequirementSource(
        block=block,
        field=row.name,
        field_status=str(row.status),
        value=row.value,
    )


def _id(kind: str, block: str, field: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{block}\0{field}".encode("utf-8")).hexdigest()[:10]
    return f"REQ-{kind.upper().replace('_', '-')}-{digest}"


def _confirmed(row: TZFieldRow) -> bool:
    return row.status in _CONFIRMED_STATUSES


def _pick(
    rows: list[tuple[str, TZFieldRow]],
    *,
    names: tuple[str, ...] = (),
    values: tuple[str, ...] = (),
) -> tuple[str, TZFieldRow] | None:
    matches = []
    for block, row in rows:
        name, value = _norm(row.name), _norm(row.value)
        if names and not any(token in name for token in names):
            continue
        if values and not any(token in value for token in values):
            continue
        matches.append((block, row))
    if not matches:
        return None
    matches.sort(key=lambda pair: (not _confirmed(pair[1]), pair[0], pair[1].name))
    return matches[0]


_ALCOHOL = re.compile(r"(?:спирт\w*|alcohol|ethanol|methanol|этанол|метанол)", re.I)


def compile_requirements(structured_tz: Any) -> RequirementsSpec:
    """Compile supported TZ clauses and preserve unsupported ambiguity explicitly."""
    tz = structured_tz if isinstance(structured_tz, StructuredTZ) else StructuredTZ.model_validate(structured_tz or {})
    rows = list(_rows(tz))
    constraints: list[RequirementConstraint] = []
    questions: list[str] = []
    consumed: set[tuple[str, str, str]] = set()

    def add(
        kind: str,
        scope: str,
        hardness: str,
        operator: str,
        value: Any,
        block: str,
        row: TZFieldRow,
        *,
        machine: bool,
        confirmed: bool | None = None,
        unit: str = "",
    ) -> None:
        is_confirmed = _confirmed(row) if confirmed is None else confirmed
        constraints.append(RequirementConstraint(
            constraint_id=_id(kind, block, row.name),
            scope=scope,
            kind=kind,
            hardness=hardness,
            operator=operator,
            value=value,
            unit=unit,
            resolution="confirmed" if is_confirmed else "needs_confirmation",
            machine_evaluable=machine,
            source=_source(block, row),
        ))
        consumed.add((block, row.name, row.value))

    # Required process medium.
    item = _pick(
        rows,
        names=("растворител", "сред", "ограничения заказчика", "схема проверки"),
        values=("вод", "water", "aqueous"),
    )
    if item:
        block, row = item
        # «в спирте или воде» is not a water-only requirement: keep the
        # alternative the customer allowed, or the route in ethanol would be
        # held against a constraint the ТЗ never set.
        allows_alcohol = bool(_ALCOHOL.search(f"{row.name} {row.value}"))
        add(
            "aqueous_medium", "step", "hard", "requires",
            {"medium": "water_or_alcohol" if allows_alcohol else "water"},
            block, row, machine=True,
        )

    # Metal-containing catalysts are a category that can be checked from a
    # normalized catalyst identity.  Unknown catalyst identity remains unknown.
    item = _pick(
        rows,
        names=("запрещ", "исключ", "катализ", "ограничения заказчика", "списки"),
        values=("металл", "переходн", "metal"),
    )
    if item:
        block, row = item
        add(
            "forbidden_catalyst_category", "step", "hard", "forbids",
            {"category": "metal-containing"}, block, row, machine=True,
        )

    # A phrase such as "harmful solvents" is not an executable deny-list.  A
    # clearly labelled explicit list is executable; otherwise keep the clause
    # hard and unresolved until a named/versioned policy or list is supplied.
    item = next((
        (block, row) for block, row in rows
        if any(token in _norm(row.name) for token in (
            "запрещ", "исключ", "списки", "растворител", "ограничения заказчика",
        ))
        and (
            any(token in _norm(row.value) for token in ("вредн", "токсич"))
            or bool(_explicit_solvent_names(row.value))
        )
    ), None)
    if item:
        block, row = item
        names = _explicit_solvent_names(row.value)
        add(
            "solvent_hazard_policy", "step", "hard", "forbids",
            {"raw_policy": row.value, "policy_id": "", "forbidden_names": names},
            block, row, machine=bool(names), confirmed=_confirmed(row) and bool(names),
        )
        if not names:
            questions.append(
                "Укажите перечень запрещённых растворителей или ID/версию корпоративной hazard-политики."
            )

    # Never turn "near room temperature" into a numeric range.  An explicit
    # range proposed by the agent is retained, but remains unconfirmed.
    temperature_rows = [
        (block, row) for block, row in rows
        if "температур" in _norm(row.name) or "температур" in _norm(row.value)
    ]
    explicit = []
    for block, row in temperature_rows:
        match = _RANGE_C.search(row.value)
        if match:
            explicit.append((block, row, float(match.group("minimum").replace(",", ".")), float(match.group("maximum").replace(",", "."))))
    if explicit:
        explicit.sort(key=lambda item: (not _confirmed(item[1]), item[0], item[1].name))
        block, row, minimum, maximum = explicit[0]
        add(
            "temperature_range", "step", "hard", "between",
            {"minimum": minimum, "maximum": maximum}, block, row,
            machine=True, unit="degC",
        )
        if not _confirmed(row):
            questions.append(
                f"Подтвердите предложенный температурный диапазон {minimum:g}–{maximum:g} °C; "
                "в исходном требовании могло быть только качественное «близко к комнатной»."
            )
    else:
        item = next(((block, row) for block, row in temperature_rows if "комнат" in _norm(row.value)), None)
        if item:
            block, row = item
            add(
                "temperature_range", "step", "hard", "qualitative",
                {"description": row.value}, block, row, machine=False, confirmed=False,
            )
            questions.append("Задайте числовой допустимый диапазон температуры процесса.")

    # Feedstock origin is soft when the request says "preferably".  Do not let a
    # generated field titled "allowed feedstocks" strengthen the original request.
    item = _pick(rows, names=("желательные вещества", "исходные вещества", "сыр"), values=("лигнин", "древес"))
    if item:
        block, row = item
        original = _norm(tz.original_request)
        explicitly_exclusive = any(token in original for token in ("только из", "обязательно из", "исключительно из"))
        add(
            "feedstock_origin", "feedstock", "hard" if explicitly_exclusive else "soft",
            "requires" if explicitly_exclusive else "prefers",
            {"origin": row.value}, block, row, machine=False,
            confirmed=_confirmed(row),
        )

    item = _pick(rows, names=("обязательные структурные",), values=("фенол",))
    if item:
        block, row = item
        add("required_structure", "molecule", "hard", "contains", {"description": row.value}, block, row, machine=False)

    item = _pick(rows, names=("минимальная чистота образца",))
    if item and (match := _NUMBER_PERCENT.search(item[1].value)):
        block, row = item
        add("minimum_product_purity", "deliverable", "hard", "gte", float(match.group("number").replace(",", ".")), block, row, machine=False, unit="%")

    item = _pick(rows, names=("минимальная масса образца",))
    if item and (match := _NUMBER_MASS.search(item[1].value)):
        block, row = item
        add("minimum_sample_mass", "deliverable", "hard", "gte", float(match.group("number").replace(",", ".")), block, row, machine=False, unit=match.group("unit"))

    item = _pick(rows, names=("требования к промывке",), values=("миним",))
    if item:
        block, row = item
        add("minimal_purification", "route", "soft", "prefers", {"description": row.value}, block, row, machine=False)

    # Preserve unfamiliar restrictive clauses instead of silently dropping
    # them.  They remain non-machine-evaluable and therefore fail closed when
    # hard; adding a new domain rule later does not require changing the TZ.
    constraint_blocks = {
        "Ограничения по сырью": "feedstock",
        "Ограничения по технологии": "step",
        "Безопасность и регуляторика": "step",
    }
    hard_cues = (
        "запрещ", "исключ", "не допуск", "обязат", "только",
        "миним", "максим", "разрешен", "допустим", "огранич", "без ",
    )
    soft_cues = ("предпочт", "желател", "приоритет")
    for block, row in rows:
        key = (block, row.name, row.value)
        if block not in constraint_blocks or key in consumed:
            continue
        text = _norm(f"{row.name} {row.value}")
        if not any(cue in text for cue in (*hard_cues, *soft_cues)):
            continue
        is_soft = any(cue in text for cue in soft_cues) and not any(
            cue in text for cue in hard_cues
        )
        add(
            "unparsed_constraint", constraint_blocks[block],
            "soft" if is_soft else "hard",
            "prefers" if is_soft else "requires_review",
            {"description": row.value}, block, row,
            machine=False, confirmed=_confirmed(row),
        )
        if not is_soft:
            questions.append(
                f"Уточните машинное правило для ограничения «{row.name}: {row.value}»."
            )

    return RequirementsSpec(
        constraints=constraints,
        open_questions=list(dict.fromkeys(questions)),
    )


def compile_requirements_callback(callback_context: CallbackContext) -> None:
    """Before-agent callback: refresh the contract from the current approved TZ."""
    tz = callback_context.state.get("structured_tz")
    if not tz:
        return None
    callback_context.state[REQUIREMENTS_KEY] = compile_requirements(tz).model_dump()
    return None


__all__ = [
    "REQUIREMENTS_KEY",
    "compile_requirements",
    "compile_requirements_callback",
]
