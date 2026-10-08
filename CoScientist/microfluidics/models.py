"""Domain models for the microfluidics case.

The ТЗ follows the reference document
``Пример_уточненного_структурированного_ТЗ_для_агентов.md``: a document made
of BLOCKS, where every block is a table of concrete fields and every field
carries a provenance status. The TZAgent pipeline outputs:

  TZSpecAgent      -> StructuredTZ        (state key ``structured_tz``)
  TZQueryGenAgent  -> LiteratureQueries   (state key ``tz_literature_queries``)

``CoScientist.microfluidics.render.render_tz_document`` turns a validated
StructuredTZ into the human-readable Markdown document of the reference
format (interpretation rules, per-block tables, open fields, questionnaire
coverage check).

Statuses are plain string Literals (not Enum) so the dicts ADK stores in
session state — and injects into downstream prompts — render as readable
JSON-like text instead of Enum reprs.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

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
    query_en: str = Field(description="Поисковый запрос на английском")
    extract: List[str] = Field(
        default_factory=list, description="Какие данные нужно извлечь из источников"
    )


class LiteratureQueries(BaseModel):
    """Пакет поисковых задач, выведенных из структурированного ТЗ."""

    queries: List[LiteratureQuery] = Field(default_factory=list)


__all__ = [
    "CANONICAL_BLOCKS",
    "FieldStatus",
    "LiteratureQueries",
    "LiteratureQuery",
    "OPEN_STATUSES",
    "StructuredTZ",
    "TZBlock",
    "TZFieldRow",
]
