"""Deterministic, fail-closed qualification of synthesis-route proposals."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from CoScientist.microfluidics.models import (
    ComplianceCheck,
    EvidenceRef,
    OPERATOR_ECONOMICS_OVERRIDE_KEY,
    OperatorEconomicsOverride,
    QualifiedRoutes,
    RequirementConstraint,
    RequirementsSpec,
    RouteDecision,
    SourceRecord,
    SynthesisRoute,
    SynthesisRoutes,
)
from CoScientist.microfluidics.requirements import REQUIREMENTS_KEY, compile_requirements

logger = logging.getLogger(__name__)

QUALIFIED_ROUTES_KEY = "qualified_routes"

_NUMBER_C = re.compile(r"(?<![\d.])([+-]?\d+(?:[.,]\d+)?)\s*°?\s*[CcСс]")
_RANGE_C = re.compile(
    r"([+-]?\d+(?:[.,]\d+)?)\s*(?:[–—-]|\.\.)\s*"
    r"([+-]?\d+(?:[.,]\d+)?)\s*°?\s*[CcСс]"
)
# A temperature-labelled condition ("температура: 50–70"): the number carries
# no °C, so read it from the value once the FIELD is known to be temperature.
_TEMP_LABEL = re.compile(r"температур|temperature|\btemp\b|t\s*[,=]|t\s*°", re.I)
_RANGE_BARE = re.compile(
    r"([+-]?\d+(?:[.,]\d+)?)\s*(?:[–—-]|\.\.|to|до)\s*([+-]?\d+(?:[.,]\d+)?)"
)
_NUMBER_BARE = re.compile(r"[+-]?\d+(?:[.,]\d+)?")
_WATER = re.compile(r"\b(?:water|aqueous|вод(?:а|ы|е|у|ой)|водн\w*)\b", re.I)
_ALCOHOL_MEDIUM = re.compile(
    r"(?:спирт\w*|этанол\w*|метанол\w*|изопропанол\w*|alcohol|ethanol|methanol|isopropanol|EtOH|MeOH|iPrOH)",
    re.I,
)
_SOLVENT_DECLARATION = re.compile(
    r"\b(?:solvent|medium|aqueous|water|растворител\w*|среда|вода|водн\w*)\b", re.I
)
_NO_CATALYST = re.compile(
    r"(?:без|отсутств\w*)[^.;]{0,50}(?:металл\w*\s+)?катализ|"
    r"катализ\w*[^.;]{0,30}(?:не\s+(?:требуетс\w*|нужн\w*|использ\w*|примен\w*)|отсутств\w*)|"
    r"(?:no|without)\s+(?:metal(?:-containing)?\s+)?catalyst|"
    r"catalyst[- ]free",
    re.I,
)
# A catalyst FIELD whose value says "none / not needed" — an explicit absence.
_NO_CATALYST_VALUE = re.compile(
    r"(?:не\s+(?:требуетс\w*|нужн\w*|использ\w*|примен\w*)|"
    r"отсутств\w*|^\s*нет\s*$|^\s*none\s*$|"
    r"not\s+(?:required|needed|used)|catalyst[- ]free)",
    re.I,
)
_METAL_SYMBOL = re.compile(
    r"(?<![A-Za-z])(?:Pd|Pt|Ni|Co|Fe|Cu|Zn|Rh|Ru|Ir|Ag|Au|Cr|Mn|Mo|W|V|Ti|Zr|Hf)(?![a-z])"
)
_METAL_NAME = re.compile(
    r"\b(?:palladium|platinum|nickel|cobalt|iron|copper|zinc|rhodium|ruthenium|"
    r"iridium|silver|gold|chromium|manganese|molybdenum|tungsten|vanadium|"
    r"titanium|zirconium|hafnium|паллад\w*|платин\w*|никел\w*|"
    r"кобальт\w*|желез\w*|мед\w*|цинк\w*|"
    r"роди\w*|рутени\w*|ириди\w*|серебр\w*|золот\w*|хром\w*|марган\w*)\b",
    re.I,
)
_CATALYST_LABEL = re.compile(r"(?:катализ|catal)", re.I)
_NO_METAL_VALUE = re.compile(
    r"(?:без\s+(?:переходн\w*\s+)?металл\w*|"
    r"отсутств\w*|(?:no|without)\s+(?:transition\s+)?metals?)",
    re.I,
)


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").split())


def _metal_named(value: str) -> bool:
    return bool(_METAL_SYMBOL.search(value) or _METAL_NAME.search(value))


def _temperature_values(step: Any) -> list[float]:
    """Temperatures declared for a step, °C.

    A condition whose NAME marks it as temperature ("температура", "temperature",
    "T") has its numbers read even without a °C unit — the model often writes
    ``температура: 50–70``. Any other condition contributes only numbers that
    explicitly carry °C, so a bare "время: 3–6" never leaks in as a temperature.
    """
    values: list[float] = []
    for condition in step.conditions:
        name = str(condition.name or "")
        value = str(condition.value or "")
        is_temp_field = bool(_TEMP_LABEL.search(name))
        if is_temp_field:
            ranges = list(_RANGE_BARE.finditer(value))
            if ranges:
                for match in ranges:
                    values.extend(float(v.replace(",", ".")) for v in match.groups())
                continue
            nums = _NUMBER_BARE.findall(value)
            if nums:
                values.extend(float(n.replace(",", ".")) for n in nums)
                continue
        text = f"{name} {value}"
        ranges = list(_RANGE_C.finditer(text))
        if ranges:
            for match in ranges:
                values.extend(float(v.replace(",", ".")) for v in match.groups())
            continue
        values.extend(float(match.group(1).replace(",", ".")) for match in _NUMBER_C.finditer(text))
    return values


def _source_map(records: Iterable[SourceRecord]) -> dict[str, SourceRecord]:
    return {record.source_id: record for record in records}


def _verified_evidence(refs: Iterable[EvidenceRef], records: dict[str, SourceRecord]) -> list[str]:
    verified = []
    for ref in refs:
        record = records.get(ref.source_id)
        if (
            ref.verification_status == "verified"
            and bool(ref.locator.strip())
            and record is not None
            and record.full_text_available
            and bool(record.content_hash.strip())
            and record.verified_by == "evidence_verifier"
            and bool(record.url.strip() or record.external_id.strip())
        ):
            verified.append(ref.source_id)
    return sorted(set(verified))


def _route_evidence(route: SynthesisRoute, records: dict[str, SourceRecord]) -> list[str]:
    refs = list(route.evidence)
    for step in route.steps:
        refs.extend(step.evidence)
        for condition in step.conditions:
            refs.extend(condition.evidence)
    return _verified_evidence(refs, records)


def _step_evidence(route: SynthesisRoute, records: dict[str, SourceRecord]) -> list[list[str]]:
    result: list[list[str]] = []
    for step in route.steps:
        # Route-level evidence can establish provenance of the route, but must
        # not silently certify every individual operation.  Numeric conditions
        # and yields need a claim-level pointer on that step (or its condition).
        refs = list(step.evidence)
        for condition in step.conditions:
            refs.extend(condition.evidence)
        result.append(_verified_evidence(refs, records))
    return result


def _check_product_purity(
    constraint: RequirementConstraint,
    route: SynthesisRoute,
    records: dict[str, SourceRecord],
) -> ComplianceCheck:
    if constraint.resolution != "confirmed" or not constraint.machine_evaluable:
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="unknown",
            reason="Порог чистоты не подтверждён как машинно проверяемое требование.",
        )
    minimum = float(constraint.value)
    if (
        route.product_purity_percent is None
        or route.product_purity_status != "reported"
        or not _verified_evidence(route.product_purity_evidence, records)
    ):
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="unknown",
            reason="Для выделенного продукта нет верифицированного измерения чистоты.",
        )
    if route.product_purity_percent < minimum:
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="fail",
            reason=f"Чистота продукта {route.product_purity_percent:g} % ниже требуемых {minimum:g} %.",
        )
    return ComplianceCheck(
        constraint_id=constraint.constraint_id,
        status="pass",
        reason="Измеренная чистота продукта достигает минимального порога.",
    )


def _check_temperature(
    constraint: RequirementConstraint,
    route: SynthesisRoute,
    evidence_ids: list[str],
) -> ComplianceCheck:
    """A step's temperature is a RANGE; it complies when the range overlaps
    the ТЗ window (the operating point is then the overlap), and fails only
    when the whole range lies outside it. «50–80 °C» against «25–70 °C» is
    therefore compliant at 50–70 °C, not rejected for the 80."""
    if constraint.resolution != "confirmed" or not constraint.machine_evaluable:
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="needs_review",
            reason="Температурное требование не задано как числовой машинный порог — нужна проверка оператором.",
        )
    per_step = [_temperature_values(step) for step in route.steps]
    if any(not values for values in per_step):
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="unknown",
            reason="В маршруте нет проверяемой температуры для каждой стадии.",
        )
    minimum = float(constraint.value["minimum"])
    maximum = float(constraint.value["maximum"])
    windows: list[str] = []
    for index, values in enumerate(per_step, 1):
        low, high = min(values), max(values)
        if high < minimum or low > maximum:
            return ComplianceCheck(
                constraint_id=constraint.constraint_id,
                status="fail",
                reason=(
                    f"Стадия {index}: заявленная температура {low:g}–{high:g} °C целиком вне "
                    f"допустимого диапазона {minimum:g}–{maximum:g} °C."
                ),
                evidence_ids=evidence_ids,
            )
        lo, hi = max(low, minimum), min(high, maximum)
        if (low, high) != (lo, hi):
            windows.append(
                f"стадия {index}: в источнике {low:g}–{high:g} °C, работать при {lo:g}–{hi:g} °C"
            )
    note = ("; ".join(windows) + ". ") if windows else ""
    if not evidence_ids:
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="unverified",
            reason=(
                f"Температура совместима с диапазоном {minimum:g}–{maximum:g} °C. {note}"
                "Источник условия не верифицирован по полному тексту — подтвердить опытом."
            ).strip(),
        )
    return ComplianceCheck(
        constraint_id=constraint.constraint_id,
        status="pass",
        reason=(f"Все заявленные температуры совместимы с подтверждённым диапазоном. {note}").strip(),
        evidence_ids=evidence_ids,
    )


def _condition_text(step: Any) -> str:
    chunks = [f"{condition.name}: {condition.value}" for condition in step.conditions]
    chunks.extend(f"{agent.name} {agent.smiles}" for agent in step.agents)
    return "\n".join(chunks)


def _check_aqueous(
    constraint: RequirementConstraint,
    route: SynthesisRoute,
    evidence_ids: list[str],
) -> ComplianceCheck:
    if constraint.resolution != "confirmed" or not constraint.machine_evaluable:
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="needs_review",
            reason="Требование среды не задано как машинное правило — нужна проверка оператором.",
        )
    medium = str((constraint.value or {}).get("medium") or "water")
    allows_alcohol = medium == "water_or_alcohol"
    label = "водная или спиртовая среда" if allows_alcohol else "водная среда"

    def _matches(text: str) -> bool:
        return bool(_WATER.search(text) or (allows_alcohol and _ALCOHOL_MEDIUM.search(text)))

    texts = [_condition_text(step) for step in route.steps]
    if any(not text.strip() or not _matches(text) for text in texts):
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="unknown",
            reason=f"{label.capitalize()} не указана явно для всех стадий маршрута.",
        )
    if not evidence_ids:
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="unverified",
            reason=f"{label.capitalize()} указана, но источник условия не верифицирован по полному тексту.",
        )
    return ComplianceCheck(
        constraint_id=constraint.constraint_id,
        status="pass",
        reason=f"{label.capitalize()} подтверждена источником.",
        evidence_ids=evidence_ids,
    )


def _contains_transition_metal(smiles: str) -> bool:
    if not smiles.strip():
        return False
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
    except ImportError:
        return _metal_named(smiles)
    if mol is None:
        return _metal_named(smiles)
    # d-block elements, including groups 3–12 in periods 4–7.
    return any(
        atom.GetAtomicNum() in set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113))
        for atom in mol.GetAtoms()
    )


def _check_metal_catalyst(
    constraint: RequirementConstraint,
    route: SynthesisRoute,
    evidence_ids: list[str],
) -> ComplianceCheck:
    if constraint.resolution != "confirmed" or not constraint.machine_evaluable:
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="needs_review",
            reason="Запрет металлсодержащих катализаторов не задан как машинное правило — нужна проверка оператором.",
        )
    catalyst_texts: list[str] = []
    per_step_texts: list[str] = []
    for step in route.steps:
        step_texts = [f"{agent.name} {agent.smiles}" for agent in step.agents]
        step_texts.extend(
            f"{condition.name} {condition.value}"
            for condition in step.conditions if "катализ" in _norm(condition.name) or "catal" in _norm(condition.name)
        )
        catalyst_texts.extend(step_texts)
        per_step_texts.append("\n".join(step_texts))
    text = "\n".join(catalyst_texts)
    for step in route.steps:
        for agent in step.agents:
            if _metal_named(agent.name) or _contains_transition_metal(agent.smiles):
                return ComplianceCheck(
                    constraint_id=constraint.constraint_id,
                    status="fail",
                    reason=f"В agents указан металлсодержащий компонент: {agent.name or agent.smiles}.",
                    evidence_ids=evidence_ids,
                )
    if _metal_named(text):
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="fail",
            reason="В условиях указан металлсодержащий катализатор.",
            evidence_ids=evidence_ids,
        )
    explicit_absence = []
    for step in route.steps:
        explicit_absence.append(
            any(
                _CATALYST_LABEL.search(condition.name)
                and (_NO_METAL_VALUE.search(condition.value)
                     or _NO_CATALYST_VALUE.search(condition.value))
                for condition in step.conditions
            )
            # The same statement written on the catalyst itself:
            # «полиэлектролитный коацерват (катализатор, без металлов)».
            or any(
                _CATALYST_LABEL.search(agent.name) and _NO_METAL_VALUE.search(agent.name)
                for agent in step.agents
            )
            or bool(_NO_CATALYST.search(_condition_text(step)))
        )
    if not all(explicit_absence):
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="unknown",
            reason="Отсутствие металлсодержащего катализатора не подтверждено явно.",
        )
    if not evidence_ids:
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="unverified",
            reason="Заявлено отсутствие металлокатализатора, но источник условия не верифицирован.",
        )
    return ComplianceCheck(
        constraint_id=constraint.constraint_id,
        status="pass",
        reason="Источник явно подтверждает отсутствие металлсодержащего катализатора.",
        evidence_ids=evidence_ids,
    )


def _check_solvent_policy(
    constraint: RequirementConstraint,
    route: SynthesisRoute,
    evidence_ids: list[str],
) -> ComplianceCheck:
    names = [str(item).strip() for item in (constraint.value or {}).get("forbidden_names", []) if str(item).strip()]
    if constraint.resolution != "confirmed" or not constraint.machine_evaluable or not names:
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="needs_review",
            reason="Запрет вредных растворителей не задан как проверяемый перечень — нужна проверка оператором.",
        )
    normalized_names = {_norm(name) for name in names}
    for step in route.steps:
        text = _norm(_condition_text(step))
        if not _SOLVENT_DECLARATION.search(text):
            return ComplianceCheck(
                constraint_id=constraint.constraint_id,
                status="unknown",
                reason="Растворитель/среда указаны не для каждой стадии.",
            )
        for name in normalized_names:
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text):
                return ComplianceCheck(
                    constraint_id=constraint.constraint_id,
                    status="fail",
                    reason=f"В маршруте используется запрещённый растворитель: {name}.",
                    evidence_ids=evidence_ids,
                )
    if not evidence_ids:
        return ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="unverified",
            reason="Состав среды не привязан к верифицированному месту источника.",
        )
    return ComplianceCheck(
        constraint_id=constraint.constraint_id,
        status="pass",
        reason="В описании среды нет растворителей из подтверждённого запретного списка.",
        evidence_ids=evidence_ids,
    )


def _completeness_checks(
    route: SynthesisRoute,
    records: dict[str, SourceRecord],
) -> list[ComplianceCheck]:
    """Conditions must EXIST for every step (missing data blocks the route);
    an unverified source is a caveat. Yields are advisory: the economics
    server costs a route without them (default_yield, flagged in its
    assumptions), so a missing yield must not stop the hand-off."""
    per_step_evidence = _step_evidence(route, records)
    conditions_present = all(
        step.conditions and step.conditions_status != "missing" for step in route.steps
    )
    conditions_verified = conditions_present and all(
        step.conditions_status == "reported" and bool(per_step_evidence[index])
        for index, step in enumerate(route.steps)
    )
    yields_present = all(step.yield_fraction is not None for step in route.steps)
    yields_verified = yields_present and all(
        step.yield_status == "reported" and bool(per_step_evidence[index])
        for index, step in enumerate(route.steps)
    )
    evidence = sorted({item for ids in per_step_evidence for item in ids})
    if conditions_verified:
        conditions = ("pass", "Условия каждой стадии заполнены и подтверждены источником.")
    elif conditions_present:
        conditions = ("unverified", "Условия каждой стадии заполнены, но не подтверждены по полному тексту источника.")
    else:
        conditions = ("unknown", "Не для каждой стадии есть условия проведения.")
    if yields_verified:
        yields = ("pass", "Выход каждой стадии заполнен и подтверждён источником.")
    elif yields_present:
        yields = ("unverified", "Выход каждой стадии заполнен, но не подтверждён по полному тексту источника.")
    else:
        yields = ("unknown", "Не для каждой стадии есть числовой выход; экономика считается с оговоркой (default_yield).")
    return [
        ComplianceCheck(constraint_id="SYS-CONDITIONS-COMPLETE", status=conditions[0],
                        reason=conditions[1], evidence_ids=evidence),
        ComplianceCheck(constraint_id="SYS-YIELDS-COMPLETE", status=yields[0],
                        reason=yields[1], evidence_ids=evidence),
    ]


def _costing_shape_check(route: SynthesisRoute) -> ComplianceCheck:
    """Check the minimum reaction graph required by the costing MCP.

    The economics service processes every submitted route as one request.  A
    step without either side of the reaction makes that entire request fail,
    rather than returning an ``invalid`` row for just that route.  Therefore a
    route with an incomplete reaction graph is suitable for evidence-gathering
    only and must not enter ``qualified_routes.routes``.
    """
    incomplete: list[str] = []
    for index, step in enumerate(route.steps, 1):
        missing = []
        if not step.reactants:
            missing.append("reactants")
        if not step.products:
            missing.append("products")
        if missing:
            incomplete.append(f"стадия {index}: {', '.join(missing)}")
    if incomplete:
        return ComplianceCheck(
            constraint_id="SYS-ECONOMICS-ROUTE-SHAPE",
            status="unknown",
            reason=(
                "Маршрут нельзя передать в rank_routes_by_cost: "
                + "; ".join(incomplete) + "."
            ),
        )
    return ComplianceCheck(
        constraint_id="SYS-ECONOMICS-ROUTE-SHAPE",
        status="pass",
        reason="Каждая стадия содержит reactants и products для расчёта экономики.",
    )


def evaluate_route(
    route: SynthesisRoute,
    spec: RequirementsSpec,
    source_records: Iterable[SourceRecord] = (),
) -> SynthesisRoute:
    """Return a copy with deterministic per-constraint decisions and status."""
    records = _source_map(source_records)
    evidence_ids = _route_evidence(route, records)
    checks: list[ComplianceCheck] = []
    handlers = {
        "temperature_range": _check_temperature,
        "aqueous_medium": _check_aqueous,
        "forbidden_catalyst_category": _check_metal_catalyst,
        "solvent_hazard_policy": _check_solvent_policy,
    }
    for constraint in spec.constraints:
        if constraint.scope not in {"step", "route", "feedstock", "product", "deliverable"}:
            checks.append(ComplianceCheck(
                constraint_id=constraint.constraint_id,
                status="not_applicable",
                reason=f"Ограничение области {constraint.scope} проверяется на другой стадии.",
            ))
            continue
        handler = handlers.get(constraint.kind)
        if handler is not None:
            checks.append(handler(constraint, route, evidence_ids))
            continue
        if constraint.kind == "minimum_product_purity":
            checks.append(_check_product_purity(constraint, route, records))
            continue
        # No machine rule for this ТЗ clause: code cannot judge it, so it
        # never blocks — it rides along as a review flag for the operator.
        checks.append(ComplianceCheck(
            constraint_id=constraint.constraint_id,
            status="needs_review",
            reason="Для этого ограничения нет машинного правила — требует проверки оператором.",
        ))

    checks.extend(_completeness_checks(route, records))
    checks.append(_costing_shape_check(route))

    hard = {constraint.constraint_id for constraint in spec.constraints if constraint.hardness == "hard"}
    if any(check.constraint_id in hard and check.status == "fail" for check in checks):
        status = "rejected"
    elif any(
        check.constraint_id in {
            "SYS-CONDITIONS-COMPLETE",
            "SYS-YIELDS-COMPLETE",
            "SYS-ECONOMICS-ROUTE-SHAPE",
        }
        and check.status in {"unknown", "unverified"}
        for check in checks
    ):
        status = "experimental"
    elif any(
        check.status == "unknown"
        and check.constraint_id in hard
        for check in checks
    ):
        # Missing provenance or measurements are not chemistry violations.  Keep
        # the route available for a planning-only verification hand-off, while
        # preventing it from being treated as production-ready economics.
        status = "experimental"
    else:
        status = "eligible"
    return route.model_copy(update={"tz_compliance": checks, "overall_status": status})


def qualify_routes(
    routes: Any,
    spec: Any,
    source_records: Iterable[SourceRecord | dict] = (),
) -> QualifiedRoutes:
    if isinstance(routes, SynthesisRoutes):
        proposals = routes
    elif isinstance(routes, list):
        proposals = SynthesisRoutes(routes=[
            r if isinstance(r, SynthesisRoute) else SynthesisRoute.model_validate(r) for r in routes
        ])
    else:
        proposals = SynthesisRoutes.model_validate(routes)
    requirements = spec if isinstance(spec, RequirementsSpec) else RequirementsSpec.model_validate(spec)
    records = [record if isinstance(record, SourceRecord) else SourceRecord.model_validate(record) for record in source_records]
    assessed = [evaluate_route(route, requirements, records) for route in proposals.routes]
    eligible = [route for route in assessed if route.overall_status == "eligible"]
    experimental = [route for route in assessed if route.overall_status == "experimental"]

    def decisions(status: str) -> list[RouteDecision]:
        result = []
        for route in assessed:
            if route.overall_status != status:
                continue
            reasons = [
                check.reason for check in route.tz_compliance
                if check.status in ({"fail"} if status == "rejected" else {"unknown"})
            ]
            result.append(RouteDecision(
                route_id=route.route_id,
                product=route.product.name or route.product.smiles,
                overall_status=status,
                reasons=reasons,
            ))
        return result

    gaps = list(proposals.gaps)
    if not eligible and not experimental:
        gaps.append("Нет маршрута, для которого все жёсткие ограничения ТЗ подтверждены.")
    return QualifiedRoutes(
        status="ok" if eligible else ("screening_only" if experimental else "no_compliant_routes"),
        routes=eligible,
        experimental_routes=experimental,
        rejected=decisions("rejected"),
        blocked=decisions("blocked"),
        gaps=list(dict.fromkeys(gaps)),
    )


def qualify_synthesis_routes(callback_context: CallbackContext) -> None:
    """After stage 4: persist assessed proposals and the fail-closed hand-off."""
    state = callback_context.state
    raw_routes = state.get("synthesis_routes")
    if not raw_routes:
        return None
    try:
        spec = state.get(REQUIREMENTS_KEY) or compile_requirements(state.get("structured_tz"))
        analysis = state.get("literature_analysis") or {}
        result = qualify_routes(raw_routes, spec, analysis.get("source_records") or [])
        # Preserve all proposals with their decisions for the report/audit.
        proposals = SynthesisRoutes.model_validate(raw_routes)
        decisions_by_id = {
            route.route_id: route
            for route in [*result.routes]
        }
        # Rejected/blocked are summarized in QualifiedRoutes, so evaluate only
        # those proposals once here to preserve their full audit rows.
        requirements = spec if isinstance(spec, RequirementsSpec) else RequirementsSpec.model_validate(spec)
        records = [SourceRecord.model_validate(item) for item in analysis.get("source_records") or []]
        for route in proposals.routes:
            if route.route_id not in decisions_by_id:
                decisions_by_id[route.route_id] = evaluate_route(route, requirements, records)
        state["synthesis_routes"] = SynthesisRoutes(
            routes=[decisions_by_id[route.route_id] for route in proposals.routes],
            gaps=proposals.gaps,
        ).model_dump()
        state[QUALIFIED_ROUTES_KEY] = result.model_dump()
    except Exception as exc:  # noqa: BLE001 - convert qualification failures into a closed gate
        logger.warning("route qualification failed closed: %s", exc)
        state[QUALIFIED_ROUTES_KEY] = QualifiedRoutes(
            status="no_compliant_routes",
            routes=[],
            gaps=[f"Маршруты не прошли кодовую квалификацию: {type(exc).__name__}: {exc}"],
        ).model_dump()
    return None


def gate_economics(callback_context: CallbackContext) -> types.Content | None:
    """Skip stage 5 when no route passed every hard constraint."""
    state = callback_context.state
    qualified = state.get(QUALIFIED_ROUTES_KEY)
    if qualified is None and state.get("synthesis_routes"):
        # Defensive path for resumed/legacy sessions whose stage-4 callback did not run.
        qualify_synthesis_routes(callback_context)
        qualified = state.get(QUALIFIED_ROUTES_KEY)
    try:
        result = QualifiedRoutes.model_validate(qualified or {
            "status": "no_compliant_routes", "routes": [],
            "gaps": ["Квалифицированные маршруты отсутствуют."],
        })
    except Exception as exc:  # noqa: BLE001
        result = QualifiedRoutes(
            status="no_compliant_routes", routes=[],
            gaps=[f"Некорректный qualified_routes: {type(exc).__name__}: {exc}"],
        )
    if result.routes or _preliminary_economics_routes(state, result):
        return None
    payload = {
        "status": "not_run",
        # This describes only the economics stage.  Do not reuse it as a
        # routing decision: screening_only remains a valid Module-C hand-off.
        "economics_skipped": "missing_eligible_routes",
        "message": "Экономика не рассчитана: нет маршрута для производственного калькулирования.",
        "gaps": result.gaps,
    }
    state["economics"] = payload
    return types.Content(
        role="model",
        parts=[types.Part(text=json.dumps(payload, ensure_ascii=False))],
    )


def _route_repair_form(proposals: SynthesisRoutes) -> tuple[dict[str, Any] | None, dict[str, tuple[int, int]]]:
    """Build a HITL form for reaction steps that the economics MCP cannot consume."""
    blocks: list[dict[str, Any]] = []
    locations: dict[str, tuple[int, int]] = {}
    for route_index, route in enumerate(proposals.routes):
        for step_index, step in enumerate(route.steps):
            if step.reactants and step.products:
                continue
            title = f"{route.route_id} · стадия {step_index + 1}"
            locations[title] = (route_index, step_index)
            blocks.append({
                "title": title,
                "usage": (
                    "Исправьте JSON-массивы веществ. Для вещества допустимы name, smiles и amount; "
                    "продукт предыдущей стадии задаётся как {\"name\": \"@prev\"}."
                ),
                "fields": [
                    {
                        "name": "reactants",
                        "value": json.dumps(
                            [item.model_dump(exclude_defaults=True) for item in step.reactants],
                            ensure_ascii=False,
                        ),
                        "status": "missing" if not step.reactants else "current",
                        "open": not step.reactants,
                        "label": "Reactants (JSON-массив)",
                        "placeholder": '[{"name": "starting material", "smiles": "..."}]',
                    },
                    {
                        "name": "products",
                        "value": json.dumps(
                            [item.model_dump(exclude_defaults=True) for item in step.products],
                            ensure_ascii=False,
                        ),
                        "status": "missing" if not step.products else "current",
                        "open": not step.products,
                        "label": "Products (JSON-массив)",
                        "placeholder": '[{"name": "stage product", "smiles": "..."}]',
                    },
                    {
                        "name": "yield_fraction",
                        "value": "" if step.yield_fraction is None else str(step.yield_fraction),
                        "status": step.yield_status,
                        "open": step.yield_fraction is None,
                        "label": "Выход стадии, доля 0–1",
                        "placeholder": "например, 0.75",
                    },
                ],
            })
    if not blocks:
        return None, locations
    return {
        "kind": "economics_route_repair",
        "title": "Исправление маршрутов перед расчётом экономики",
        "intro": (
            "Некоторые стадии не содержат reactants или products. Проверьте структуру; "
            "после сохранения маршруты будут заново провалидированы и квалифицированы."
        ),
        "blocks": blocks,
    }, locations


def _apply_route_repair_values(
    proposals: SynthesisRoutes,
    form_values: Any,
    locations: dict[str, tuple[int, int]],
) -> SynthesisRoutes:
    """Apply and fully validate operator edits; never partially mutate session state."""
    if not isinstance(form_values, dict):
        return proposals
    document = proposals.model_dump()
    for title, location in locations.items():
        submitted = form_values.get(title)
        if not isinstance(submitted, dict):
            continue
        route_index, step_index = location
        step = document["routes"][route_index]["steps"][step_index]
        for field in ("reactants", "products"):
            if field not in submitted:
                continue
            try:
                value = json.loads(str(submitted[field]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{title}.{field}: требуется JSON-массив") from exc
            if not isinstance(value, list) or not value:
                raise ValueError(f"{title}.{field}: требуется непустой JSON-массив")
            step[field] = value
        if "yield_fraction" in submitted:
            raw_yield = str(submitted["yield_fraction"]).strip().replace(",", ".")
            try:
                value = float(raw_yield)
            except ValueError as exc:
                raise ValueError(f"{title}.yield_fraction: требуется число от 0 до 1") from exc
            if not 0 < value <= 1:
                raise ValueError(f"{title}.yield_fraction: требуется число от 0 до 1")
            step["yield_fraction"] = value
            step["yield_status"] = "reported"
            step["yield_missing_reason"] = ""
    return SynthesisRoutes.model_validate(document)


async def review_incomplete_economics_routes(
    callback_context: CallbackContext,
) -> types.Content | None:
    """Let the operator repair malformed reaction steps before economics runs."""
    from CoScientist.config import get_settings

    if not get_settings().web.hitl_enabled:
        return None
    try:
        proposals = SynthesisRoutes.model_validate(callback_context.state.get("synthesis_routes"))
    except (TypeError, ValueError):
        return None
    form, locations = _route_repair_form(proposals)
    if form is None:
        return None

    from CoScientist.agents.common import hitl_handler
    from CoScientist.graph.session_scope import session_key
    from CoScientist.hitl.models import HITLAction, HITLRequest

    user_id, session_id = session_key(callback_context)
    response = await hitl_handler.handle_request(HITLRequest(
        agent_name="EconomicsAgent",
        action_type=HITLAction.EDIT,
        message="Исправьте неполные стадии маршрутов перед расчётом экономики.",
        context={
            "output": {"affected_steps": list(locations)},
            "_session": {"user_id": user_id, "session_id": session_id},
        },
        form=form,
        invoked_via="callback",
        trigger="economics_route_repair",
    ))
    if not response.form_values:
        return None
    try:
        repaired = _apply_route_repair_values(proposals, response.form_values, locations)
    except ValueError as exc:
        callback_context.state["route_repair_error"] = str(exc)
        return types.Content(
            role="model",
            parts=[types.Part(text=f"Исправления маршрутов отклонены: {exc}")],
        )

    callback_context.state["synthesis_routes"] = repaired.model_dump()
    # ADK's State implements item assignment but is not guaranteed to expose
    # every MutableMapping method (notably ``pop``) in every runtime version.
    callback_context.state["route_repair_error"] = None
    qualify_synthesis_routes(callback_context)
    return None


def _preliminary_economics_routes(state: Any, qualified: QualifiedRoutes) -> list[SynthesisRoute]:
    """Selected real routes, only after the economics-specific HITL approval."""
    if qualified.routes:
        return qualified.routes
    try:
        override = OperatorEconomicsOverride.model_validate(
            state.get(OPERATOR_ECONOMICS_OVERRIDE_KEY)
        )
        proposals = SynthesisRoutes.model_validate(state.get("synthesis_routes"))
        by_id = {route.route_id: route for route in proposals.routes}
        if set(override.route_ids) - set(by_id):
            return []
        selected = [by_id[route_id] for route_id in override.route_ids]
        return [] if any(route.stub or not route.steps for route in selected) else selected
    except (TypeError, ValueError):
        return []


async def review_preliminary_economics(callback_context: CallbackContext) -> None:
    """Offer a human-approved preliminary pricing pass when code skipped it.

    The resulting state never promotes a route or feeds production economics;
    it merely allows supplier-price discovery under a clearly labelled mode.
    """
    state = callback_context.state
    if state.get(OPERATOR_ECONOMICS_OVERRIDE_KEY):
        return None
    try:
        qualified = QualifiedRoutes.model_validate(state.get(QUALIFIED_ROUTES_KEY))
        proposals = SynthesisRoutes.model_validate(state.get("synthesis_routes"))
    except (TypeError, ValueError):
        return None
    if qualified.routes or not proposals.routes:
        return None
    from CoScientist.config import get_settings
    if not get_settings().web.hitl_enabled:
        return None
    from CoScientist.agents.common import hitl_handler
    from CoScientist.graph.session_scope import session_key
    from CoScientist.hitl.models import HITLAction, HITLRequest

    user_id, session_id = session_key(callback_context)
    response = await hitl_handler.handle_request(HITLRequest(
        agent_name="EconomicsAgent",
        action_type=HITLAction.APPROVE,
        message=(
            "Автоматическая квалификация не дала маршрутов для production-экономики. "
            "Запустить предварительный поиск цен и доступности для предложенных "
            "маршрутов? Результат не станет production-ранжированием и не отменяет "
            "незакрытые ограничения или выходы."
        ),
        context={
            "output": {
                "routes": [
                    {"route_id": route.route_id, "status": route.overall_status,
                     "product": route.product.name or route.product.smiles}
                    for route in proposals.routes if not route.stub and route.steps
                ],
                "gaps": qualified.gaps,
            },
            "_session": {"user_id": user_id, "session_id": session_id},
        },
        invoked_via="callback",
        trigger="preliminary_economics_override",
    ))
    if response.approved:
        state[OPERATOR_ECONOMICS_OVERRIDE_KEY] = OperatorEconomicsOverride(
            mode="preliminary_only", approved_by_human=True,
            route_ids=[route.route_id for route in proposals.routes if not route.stub and route.steps],
        ).model_dump()
    return None


def guard_economics_routes(
    tool: Any = None,
    args: Any = None,
    tool_context: Any = None,
    **_: Any,
) -> dict | None:
    """Before-tool guard: costing payload must match qualified route IDs exactly."""
    name = str(getattr(tool, "name", "") or "")
    if name != "rank_routes_by_cost" or tool_context is None:
        return None
    qualified = QualifiedRoutes.model_validate(tool_context.state.get(QUALIFIED_ROUTES_KEY) or {
        "status": "no_compliant_routes", "routes": [],
    })
    allowed_routes = _preliminary_economics_routes(tool_context.state, qualified)
    allowed = {route.route_id for route in allowed_routes}
    sent = {
        str(route.get("route_id")) for route in (args or {}).get("routes") or []
        if isinstance(route, dict) and route.get("route_id")
    }
    errors: list[str] = []
    if not allowed or sent != allowed:
        errors.append("route_id set differs from qualified_routes")
    qualified_by_id = {route.route_id: route for route in allowed_routes}
    for submitted in (args or {}).get("routes") or []:
        if not isinstance(submitted, dict) or submitted.get("route_id") not in qualified_by_id:
            continue
        route_id = str(submitted["route_id"])
        expected_steps = qualified_by_id[route_id].steps
        actual_steps = submitted.get("steps")
        if not isinstance(actual_steps, list) or len(actual_steps) != len(expected_steps):
            errors.append(f"{route_id}: step count differs from qualified route")
            continue
        for index, (actual, expected) in enumerate(zip(actual_steps, expected_steps), 1):
            if not isinstance(actual, dict):
                errors.append(f"{route_id}: step {index} must remain structured")
                continue
            actual_yield = actual.get("yield", actual.get("yield_fraction"))
            if actual_yield != expected.yield_fraction:
                errors.append(f"{route_id}: step {index} yield differs from verified value")
            missing = [field for field in ("reactants", "products") if not actual.get(field)]
            if missing:
                errors.append(
                    f"{route_id}: step {index} requires nonempty {', '.join(missing)} "
                    "for rank_routes_by_cost"
                )
    if not errors:
        return None
    return {
        "error": "rank_routes_by_cost blocked by TZ compliance gate",
        "reasons": errors,
        "eligible_route_ids": sorted(allowed),
        "submitted_route_ids": sorted(sent),
    }


__all__ = [
    "QUALIFIED_ROUTES_KEY",
    "evaluate_route",
    "gate_economics",
    "guard_economics_routes",
    "review_incomplete_economics_routes",
    "review_preliminary_economics",
    "qualify_routes",
    "qualify_synthesis_routes",
]
