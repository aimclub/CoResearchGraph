"""Domain models for the microfluidics case.

The ТЗ follows the reference document
``Пример_уточненного_структурированного_ТЗ_для_агентов.md``: a document made
of BLOCKS, where every block is a table of concrete fields and every field
carries a provenance status. The TZAgent pipeline outputs:

  TZSpecAgent      -> StructuredTZ        (state key ``structured_tz``)
  TZQueryGenAgent  -> LiteratureQueries   (state key ``tz_literature_queries``)
  LiteratureSynthesisAgent -> LiteratureAnalysis (state key ``literature_analysis``)
  MolDesignAgent   -> DesignCandidates    (state key ``design_candidates``)
  RouteSelectionAgent -> SynthesisRoutes (state key ``synthesis_routes``)

``CoScientist.microfluidics.render.render_tz_document`` turns a validated
StructuredTZ into the human-readable Markdown document of the reference
format (interpretation rules, per-block tables, open fields, questionnaire
coverage check).

Statuses are plain string Literals (not Enum) so the dicts ADK stores in
session state — and injects into downstream prompts — render as readable
JSON-like text instead of Enum reprs.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# Field-status vocabulary is shared with the research frame intake — one name
# for one thing. Re-exported here so existing importers keep working.
from CoScientist.hitl.field_status import FieldStatus, OPEN_STATUSES

# Canonical block set of the reference ТЗ document, in document order.
CANONICAL_BLOCKS: tuple[str, ...] = (
    "Тип задачи",
    "Целевой продукт",
    "Область применения",
    "Требуемые свойства",
    "Критерии качества",
    "Масштаб результата",
    "Ограничения по сырью",
    "Ограничения по поставкам",
    "Ограничения по себестоимости",
    "Ограничения по технологии",
    "Доступное оборудование",
    "Аналитические методы",
    "Известные данные заказчика",
    "Безопасность и регуляторика",
    "Приоритеты отбора",
    "Форма результата",
)


class TZFieldRow(BaseModel):
    """One row of a block table: |Поле|Значение|Статус|."""

    name: str = Field(description="Название поля, напр. «Минимальная чистота образца»")
    value: str = Field(default="Не задано", description="Значение поля")
    status: FieldStatus = Field(default="не задано")

    def is_set(self) -> bool:
        return self.status not in OPEN_STATUSES


class TZBlock(BaseModel):
    """One block of the ТЗ document: a titled table of concrete fields."""

    title: str = Field(description="Название блока, напр. «Критерии качества»")
    usage: str = Field(
        default="",
        description="Как блок используется дальше по пайплайну (одна фраза)",
    )
    fields: List[TZFieldRow] = Field(default_factory=list)


class StructuredTZ(BaseModel):
    """Структурированное техническое задание (кейс микрофлюидики).

    Shape mirrors the reference document: the verbatim customer request plus
    the canonical blocks, each a table of measurable fields with statuses.
    """

    original_request: str = Field(
        default="", description="Исходный свободный запрос заказчика дословно"
    )
    blocks: List[TZBlock] = Field(default_factory=list)

    def block(self, title: str) -> TZBlock | None:
        for b in self.blocks:
            if b.title.strip().lower() == title.strip().lower():
                return b
        return None


class LiteratureQuery(BaseModel):
    """Одна конкретная поисковая задача для агента анализа литературы."""

    id: str = Field(description="Идентификатор задачи, например LIT-01")
    task: str = Field(description="Формулировка задачи на русском")
    extract: List[str] = Field(
        default_factory=list, description="Какие данные нужно извлечь из источников"
    )
    assignee: Optional[str] = Field(
        default=None,
        description="Исполнитель задачи: PaperRetriever для первой задачи поиска маршрутов по статьям, ResearchAgent для остальных задач",
    )


class LiteratureQueries(BaseModel):
    """Пакет поисковых задач, выведенных из структурированного ТЗ."""

    queries: List[LiteratureQuery] = Field(default_factory=list)


class TargetMolecule(BaseModel):
    """Целевая молекула заказчика — то, что передаётся в модуль дизайна.

    ``fixed`` = заказчик задал конкретное вещество, подбирать кандидатов не
    нужно. Когда ТЗ его задаёт, значения берутся из ТЗ, а не из литературы.
    """

    fixed: bool = Field(
        default=False, description="Задача с фиксированной молекулой (задана заказчиком)"
    )
    name: str = Field(default="", description="Название вещества")
    smiles: str = Field(default="", description="SMILES, если известен")
    cas: str = Field(default="", description="CAS, если известен")
    source: str = Field(
        default="не задано",
        description="Откуда значения: «ТЗ», «литература» или «не задано»",
    )


class NamedValue(BaseModel):
    """Свойство или условие: название, значение с единицами, условия измерения."""

    name: str = Field(description="Напр. «ККМ» или «Температура»")
    value: str = Field(description="Значение с единицами, напр. «1.2 ммоль/л»")
    conditions: str = Field(default="", description="Условия измерения, если указаны")
    evidence: List["EvidenceRef"] = Field(
        default_factory=list,
        description="Точные ссылки на источник значения; пусто означает, что значение не верифицировано",
    )


class EvidenceRef(BaseModel):
    """A claim-level pointer into a real source, not a literature-task label."""

    source_id: str = Field(min_length=1, description="ID из LiteratureAnalysis.source_records")
    locator: str = Field(
        default="",
        description="Страница, раздел, таблица, номер абзаца патента или устойчивый фрагмент текста",
    )
    quote: str = Field(
        default="",
        description="Короткий подтверждающий фрагмент; не заменяет locator",
    )
    verification_status: Literal["unverified", "verified", "conflicting"] = "unverified"


class SourceRecord(BaseModel):
    """Resolvable bibliographic source used by one or more extracted claims."""

    source_id: str = Field(min_length=1)
    title: str = ""
    url: str = ""
    external_id: str = Field(default="", description="Patent/standard identifier")
    source_type: Literal["paper", "patent", "standard", "web", "other"] = "other"
    full_text_available: bool = False
    content_hash: str = Field(
        default="", description="Hash of the exact full text/version inspected by a verifier"
    )
    verified_by: Literal["", "evidence_verifier"] = ""
    verification_tool: str = ""


class Analogue(BaseModel):
    """Аналог целевого продукта, найденный в литературе."""

    name: str
    smiles: str = Field(default="", description="SMILES, если удалось установить")
    compound_class: str = Field(default="", description="Химический класс")
    properties: List[NamedValue] = Field(default_factory=list)
    relevance: str = Field(default="", description="Чем аналог полезен для ТЗ")
    sources: List[str] = Field(default_factory=list, description="Ссылки")


class RouteStep(BaseModel):
    """Одна операция маршрута синтеза.

    ``products`` и ``yield_value`` нужны для экономической оценки: сервер
    стоимости собирает маршрут по продуктам стадий и выводит расход реагентов
    обратным ходом от выхода.
    """

    operation: str
    reactants: List[str] = Field(
        default_factory=list,
        description="Исходные вещества стадии; предпочтительнее legacy-поля reagents",
    )
    agents: List[str] = Field(
        default_factory=list,
        description="Растворители, катализаторы и технологические среды стадии",
    )
    reagents: List[str] = Field(default_factory=list)
    products: List[str] = Field(
        default_factory=list, description="Что получается на стадии (название / SMILES)"
    )
    yield_value: str = Field(
        default="", description="Выход стадии как в источнике, напр. «75 %»; пусто — не указан"
    )
    conditions: List[NamedValue] = Field(default_factory=list)
    evidence: List[EvidenceRef] = Field(default_factory=list)


class LiteratureRoute(BaseModel):
    """Маршрут синтеза, описанный в литературе."""

    route_id: str = Field(
        default="",
        description="Устойчивый ID литературного маршрута, например LIT-ROUTE-01",
    )
    product: str = Field(description="Какое вещество получают (название / SMILES)")
    product_smiles: str = Field(
        default="",
        description="SMILES целевого продукта маршрута, если структура однозначно установлена",
    )
    variant_label: str = Field(
        default="",
        description="Идентификатор или краткое имя варианта процедуры/строки таблицы",
    )
    comparison_notes: str = Field(
        default="",
        description="С чем сравнивался вариант и почему он выбран или отклонён",
    )
    steps: List[RouteStep] = Field(default_factory=list)
    flow_suitability: str = Field(
        default="", description="Пригодность для проточного / микрофлюидного реактора"
    )
    sources: List[str] = Field(default_factory=list)
    evidence: List[EvidenceRef] = Field(default_factory=list)


class LiteratureFact(BaseModel):
    """Факт из литературы, привязанный к поисковой задаче."""

    statement: str
    query_id: str = Field(default="", description="LIT-xx, по которой найден факт")
    sources: List[str] = Field(default_factory=list)
    evidence: List[EvidenceRef] = Field(default_factory=list)


class LiteratureAnalysis(BaseModel):
    """Итог модуля A (ТЗ + литература) — вход модуля дизайна и отчёта."""

    target_molecule: TargetMolecule = Field(default_factory=TargetMolecule)
    source_records: List[SourceRecord] = Field(
        default_factory=list,
        description="Реальные URL/патенты, на которые ссылаются EvidenceRef",
    )
    analogues: List[Analogue] = Field(default_factory=list)
    synthesis_routes: List[LiteratureRoute] = Field(default_factory=list)
    facts: List[LiteratureFact] = Field(default_factory=list)
    gaps: List[str] = Field(
        default_factory=list, description="Что не удалось найти в литературе"
    )

    @model_validator(mode="after")
    def unique_source_ids(self):
        ids = [source.source_id for source in self.source_records]
        if len(ids) != len(set(ids)):
            raise ValueError("literature_analysis source_id values must be unique")
        return self


class LiteratureSelection(BaseModel):
    """Small, fail-safe decision made after the literature research phase."""

    selected_ids: List[str] = Field(default_factory=list)
    reason: str = ""
    warnings: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def accept_imperfect_llm_payload(cls, value: Any):
        """Coerce nulls and malformed values into a safe empty selection."""
        raw = value if isinstance(value, dict) else {}
        candidate_ids = raw.get("selected_ids", raw.get("selected_id", []))
        if isinstance(candidate_ids, str):
            candidate_ids = [candidate_ids]
        if not isinstance(candidate_ids, list):
            candidate_ids = []
        ids: list[str] = []
        for item in candidate_ids:
            text = str(item or "").strip().upper()
            if text and text not in ids:
                ids.append(text)
        warnings = raw.get("warnings") or []
        if isinstance(warnings, str):
            warnings = [warnings]
        if not isinstance(warnings, list):
            warnings = []
        return {
            "selected_ids": ids,
            "reason": str(raw.get("reason") or ""),
            "warnings": [str(item) for item in warnings if str(item or "").strip()],
        }


class RouteSelectionItem(BaseModel):
    """Короткое, показываемое оператору решение по одному маршруту Module A."""

    route_id: str = Field(min_length=1)
    product: str = ""
    recommendation: Literal["оставить", "отсеять"]
    reason: str = Field(min_length=1, description="Краткая причина в 1–2 предложениях")
    hard_violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class RouteSelection(BaseModel):
    """Предложение агента; окончательным оно становится только после HITL review."""

    decisions: List[RouteSelectionItem] = Field(default_factory=list)
    selected_route_id: str = Field(
        default="",
        description="Ровно один рекомендуемый route_id; пусто означает отклонить все",
    )
    selection_reason: str = Field(default="")

    @model_validator(mode="after")
    def selected_route_is_known_and_unique(self):
        ids = [item.route_id.strip() for item in self.decisions]
        if any(not route_id for route_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("route selection requires unique nonempty route_id values")
        if self.selected_route_id and self.selected_route_id not in ids:
            raise ValueError("selected_route_id must occur in decisions")
        kept = [item.route_id for item in self.decisions if item.recommendation == "оставить"]
        if self.selected_route_id and kept != [self.selected_route_id]:
            raise ValueError("exactly selected_route_id must be marked оставить")
        if not self.selected_route_id and kept:
            raise ValueError("no route may be marked оставить when selected_route_id is empty")
        return self


# ── Module B hand-off: design and synthesis routes ───────────────────────────
# The shape the design system (ГПН) is expected to return, and the one the
# economics server costs: a step names its reactants, agents and products and
# carries a yield as a fraction — the fields rank_routes_by_cost chains a route by.

class Substance(BaseModel):
    """Вещество: название и, если известна, структура."""

    name: str = Field(default="", description="Название; английское, если известно")
    smiles: str = Field(default="", description="SMILES, если известен")
    amount: str = Field(
        default="",
        description="Только для растворителей и катализаторов: сколько закупать "
                    "на всю наработку, напр. «500 ml»; пусто — не учитывать",
    )


class DesignCandidate(BaseModel):
    """Кандидат на синтез (или сама целевая молекула заказчика)."""

    name: str
    smiles: str = Field(default="", description="SMILES; пусто — структура не установлена")
    compound_class: str = Field(default="", description="Химический класс")
    properties: List[NamedValue] = Field(default_factory=list)
    tz_fit: str = Field(default="", description="Какие требования ТЗ закрывает, какие нет")
    risks: str = Field(default="")
    source: str = Field(
        default="дизайн", description="Откуда кандидат: «ТЗ», «дизайн» или «литература»"
    )
    sources: List[str] = Field(default_factory=list, description="Источники литературных свойств")
    derivation: str = Field(default="", description="Происхождение структуры; не маршрут синтеза")
    route_ids: List[str] = Field(
        default_factory=list,
        description="Литературные маршруты, непосредственно ведущие к кандидату",
    )
    stub: bool = Field(default=False, description="Данные получены от заглушки")


class DesignCandidates(BaseModel):
    """Выход стадии 3 — кого синтезировать."""

    fixed_target: bool = Field(
        default=False, description="Заказчик задал молекулу — подбора не было"
    )
    candidates: List[DesignCandidate] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list, description="Каких данных не хватает")


class ProcessStep(BaseModel):
    """Стадия маршрута в форме, которую принимает сервер стоимости."""

    operation: str
    reactants: List[Substance] = Field(
        default_factory=list,
        description="Исходные вещества стадии; продукт предыдущей стадии — name «@prev»",
    )
    agents: List[Substance] = Field(
        default_factory=list, description="Растворители, катализаторы, среды"
    )
    products: List[Substance] = Field(default_factory=list)
    conditions: List[NamedValue] = Field(default_factory=list)
    conditions_status: Literal["reported", "missing", "unverified"] = "missing"
    conditions_missing_reason: str = ""
    yield_fraction: Optional[float] = Field(
        default=None, gt=0, le=1, description="Выход стадии, доля 0–1; нет данных — null"
    )
    yield_status: Literal["reported", "missing", "unverified"] = "missing"
    yield_missing_reason: str = ""
    evidence: List[EvidenceRef] = Field(default_factory=list)
    flow_notes: str = Field(
        default="", description="Как стадия переносится на проточный реактор"
    )

    @model_validator(mode="after")
    def explain_missing_operating_data(self):
        if self.conditions:
            if self.conditions_status == "missing":
                self.conditions_status = "reported"
        elif self.conditions_status == "reported":
            raise ValueError("conditions_status=reported requires nonempty conditions")
        elif not self.conditions_missing_reason.strip():
            raise ValueError("empty conditions require conditions_missing_reason")

        if self.yield_fraction is not None:
            if self.yield_status == "missing":
                self.yield_status = "reported"
        elif self.yield_status == "reported":
            raise ValueError("yield_status=reported requires yield_fraction")
        elif not self.yield_missing_reason.strip():
            raise ValueError("yield_fraction=null requires yield_missing_reason")
        return self


class RequirementSource(BaseModel):
    block: str
    field: str
    field_status: str
    value: str


class RequirementConstraint(BaseModel):
    """One atomic, scoped and provenance-preserving requirement compiled from the TZ."""

    constraint_id: str = Field(min_length=1)
    scope: Literal["molecule", "feedstock", "step", "route", "product", "deliverable"]
    kind: str = Field(min_length=1)
    hardness: Literal["hard", "soft"]
    operator: str = Field(min_length=1)
    value: Any = None
    unit: str = ""
    resolution: Literal["confirmed", "needs_confirmation"] = "confirmed"
    machine_evaluable: bool = False
    source: RequirementSource


class RequirementsSpec(BaseModel):
    schema_version: str = "1.0"
    constraints: List[RequirementConstraint] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_constraint_ids(self):
        ids = [item.constraint_id for item in self.constraints]
        if len(ids) != len(set(ids)):
            raise ValueError("requirements_spec.constraint_id values must be unique")
        return self


class ComplianceCheck(BaseModel):
    constraint_id: str
    # Beyond pass/fail: two "cannot conclude" outcomes that must be told
    # apart, because one blocks the route and the other does not.
    #   unknown       — a MACHINE-checkable hard constraint whose data the route
    #                   is missing (e.g. no temperature). Blocks: we can't tell.
    #   needs_review  — a constraint with NO machine rule at all (an unparsed ТЗ
    #                   clause). Code cannot judge it, so it never blocks; it
    #                   travels as a review flag for the operator.
    #   unverified    — the constraint is satisfied as stated, but the source
    #                   could not be checked against a full text. A caveat, not
    #                   a block.
    status: Literal["pass", "fail", "unknown", "needs_review", "unverified", "not_applicable"]
    reason: str
    evidence_ids: List[str] = Field(default_factory=list)
    evaluated_by: Literal["code", "human", "agent"] = "code"


class SynthesisRoute(BaseModel):
    """Маршрут синтеза одного продукта."""

    route_id: str = Field(description="GPN-1, GPN-2… — ретросинтез; LIT-1… — из литературы")
    source_route_id: str = Field(
        default="", description="Идентификатор маршрута во внешнем сервисе/источнике"
    )
    product: Substance
    source: str = Field(default="ретросинтез", description="«ретросинтез» или «литература»")
    variant_label: str = Field(
        default="",
        description="Идентификатор варианта процедуры/строки таблицы, если источник их различает",
    )
    selection_rationale: str = Field(
        default="",
        description="Сопоставление с другими вариантами маршрута; не заменяет числовые данные",
    )
    steps: List[ProcessStep] = Field(
        min_length=1, description="Все операции маршрута по порядку; обязательное непустое поле"
    )
    flow_suitability: str = Field(default="")
    bottlenecks: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list, description="Ссылки")
    evidence: List[EvidenceRef] = Field(default_factory=list)
    product_purity_percent: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="Измеренная чистота выделенного продукта, %, если сообщена",
    )
    product_purity_status: Literal["reported", "missing", "unverified"] = "missing"
    product_purity_evidence: List[EvidenceRef] = Field(default_factory=list)
    tz_compliance: List[ComplianceCheck] = Field(default_factory=list)
    overall_status: Literal["unassessed", "eligible", "experimental", "rejected", "blocked"] = "unassessed"
    stub: bool = Field(default=False, description="Маршрут получен от заглушки")


class SynthesisRoutes(BaseModel):
    """Выход стадии 4 — как синтезировать."""

    routes: List[SynthesisRoute] = Field(
        description="Полные маршруты с операциями. Обязательное поле; [] только если маршруты не найдены."
    )
    gaps: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_route_ids(self):
        ids = [route.route_id.strip() for route in self.routes]
        if any(not route_id for route_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("routes require nonempty globally unique route_id values")
        return self


class RouteDecision(BaseModel):
    route_id: str
    product: str = ""
    overall_status: Literal["experimental", "rejected", "blocked"]
    reasons: List[str] = Field(default_factory=list)


class QualifiedRoutes(BaseModel):
    """Route qualification split between production costing and experimental screening."""

    status: Literal["ok", "screening_only", "no_compliant_routes"]
    routes: List[SynthesisRoute] = Field(default_factory=list)
    experimental_routes: List[SynthesisRoute] = Field(default_factory=list)
    rejected: List[RouteDecision] = Field(default_factory=list)
    blocked: List[RouteDecision] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_matches_routes(self):
        if self.status == "ok" and not self.routes:
            raise ValueError("qualified_routes status=ok requires eligible routes")
        if self.status == "screening_only" and (self.routes or not self.experimental_routes):
            raise ValueError("screening_only requires experimental routes and no eligible routes")
        if self.status == "no_compliant_routes" and (self.routes or self.experimental_routes):
            raise ValueError("no_compliant_routes cannot contain hand-off routes")
        if any(route.overall_status != "eligible" for route in self.routes):
            raise ValueError("qualified_routes.routes may contain only eligible routes")
        if any(route.overall_status != "experimental" for route in self.experimental_routes):
            raise ValueError("qualified_routes.experimental_routes may contain only experimental routes")
        ids = [route.route_id for route in [*self.routes, *self.experimental_routes]]
        if len(ids) != len(set(ids)):
            raise ValueError("qualified_routes route_id values must be unique")
        return self


# This is deliberately not a qualification result.  It records a human's
# exception to the automatic gate and can only be used to request a
# non-executing verification plan from the external system.
OPERATOR_ROUTE_OVERRIDE_KEY = "operator_route_override"
OPERATOR_ECONOMICS_OVERRIDE_KEY = "operator_economics_override"


class OperatorRouteOverride(BaseModel):
    """Explicit operator authorization to plan verification of rejected routes."""

    mode: Literal["screening_only"]
    approved_by_human: Literal[True]
    route_ids: List[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    operator_feedback: str = ""

    @model_validator(mode="after")
    def route_ids_are_unique(self):
        ids = [route_id.strip() for route_id in self.route_ids]
        if any(not route_id for route_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("operator override requires nonempty unique route_id values")
        self.route_ids = ids
        return self


class OperatorEconomicsOverride(BaseModel):
    """Human approval to price non-eligible routes as preliminary evidence."""

    mode: Literal["preliminary_only"]
    approved_by_human: Literal[True]
    route_ids: List[str] = Field(min_length=1)

    @model_validator(mode="after")
    def route_ids_are_unique(self):
        ids = [route_id.strip() for route_id in self.route_ids]
        if any(not route_id for route_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("economics override requires nonempty unique route_id values")
        self.route_ids = ids
        return self


__all__ = [
    "Analogue",
    "CANONICAL_BLOCKS",
    "DesignCandidate",
    "DesignCandidates",
    "FieldStatus",
    "LiteratureAnalysis",
    "LiteratureFact",
    "LiteratureQueries",
    "LiteratureQuery",
    "LiteratureRoute",
    "NamedValue",
    "EvidenceRef",
    "SourceRecord",
    "OPEN_STATUSES",
    "OPERATOR_ROUTE_OVERRIDE_KEY",
    "OPERATOR_ECONOMICS_OVERRIDE_KEY",
    "OperatorEconomicsOverride",
    "OperatorRouteOverride",
    "ProcessStep",
    "RequirementConstraint",
    "RequirementSource",
    "RequirementsSpec",
    "ComplianceCheck",
    "QualifiedRoutes",
    "RouteDecision",
    "RouteStep",
    "StructuredTZ",
    "Substance",
    "SynthesisRoute",
    "SynthesisRoutes",
    "TargetMolecule",
    "TZBlock",
    "TZFieldRow",
]
