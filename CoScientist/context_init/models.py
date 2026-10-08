"""The Research Frame — the framing entities of the meta-model as a filled form.

One frame = one run. The frame mirrors the microfluidics ТЗ shape (blocks of
fields, each field carrying a provenance status) but its blocks map onto the
Research Context Graph seeding (``context_init.commit``): question attributes,
the constraint context star, budgets, tools, empirical base, confirmation
criteria and the cost model.

``FRAME_SPEC`` is the single source of truth for the block set, in document
order. Each block declares HOW it seeds the graph (``kind`` + optional
``subtype``) and its field names. ``ResearchFrame.blank`` builds an all-open
frame from it, and ``ResearchFrame.normalized`` re-imposes the canonical
structure on an LLM-produced frame so the model can drift on values but never on
structure.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from CoScientist.hitl.field_status import FieldStatus, OPEN_STATUSES


# ── the canonical frame ───────────────────────────────────────────────────────
# (title, kind, subtype, usage, field_names). ``kind`` drives graph seeding:
#   question              -> attrs merged onto the root ResearchQuestion
#   constraint            -> one Constraint node (attrs.subtype = subtype)
#   resources             -> one Resource node per field
#   tools                 -> one Tool node per field
#   empirical_base        -> one EmpiricalBase node per field
#   confirmation_criteria -> one ConfirmationCriteria node
#   cost_model            -> one CostModel node (applies_to the question)
FrameSpecEntry = Tuple[str, str, Optional[str], str, Tuple[str, ...]]

FRAME_SPEC: Tuple[FrameSpecEntry, ...] = (
    ("Вопрос исследования", "question", None,
     "атрибуты корневого исследовательского вопроса",
     ("formulation", "domain", "specificity", "gap", "decomposition",
      "target_setting", "research_form", "trl")),
    ("Режим и завершение", "question", None,
     "режим применения ИИ и критерий остановки исследования",
     ("ai_application_model", "completion_criteria")),
    ("Профиль исследования", "constraint", "profile",
     "модальность и постановка — определяет активируемые модули",
     ("modality", "target_setting", "form_trl")),
    ("Методологические нормы", "constraint", "methodological_norms",
     "принятые способы проведения исследований этого типа",
     ("norms",)),
    ("Теоретические рамки", "constraint", "theoretical_framework",
     "формальные модели и теории домена",
     ("frameworks",)),
    ("Доменные стандарты", "constraint", "domain_standards",
     "ГОСТ/ISO/ICH/CLSI и требования журналов",
     ("standards",)),
    ("Этика и регуляторика", "constraint", "ethics",
     "жёсткие ограничения, которые нельзя нарушать",
     ("constraints",)),
    ("Экспертное знание", "constraint", "expert_knowledge",
     "неформализованное знание от человека в контуре",
     ("notes",)),
    ("Роли участников", "constraint", "roles",
     "состав лиц, чья деятельность автоматизируется",
     ("roles",)),
    ("Ресурсы и бюджеты", "resources", None,
     "конечные бюджеты — ограничивают критерии завершения",
     ("gpu_hours", "tokens", "money", "time", "expert_hours")),
    ("Инструменты", "tools", None,
     "известные инструменты и их статус",
     ("computational", "laboratory", "analytical", "informational")),
    ("Эмпирическая база", "empirical_base", None,
     "доступные наблюдения, данные и корпус",
     ("datasets", "corpora", "trusted_kb")),
    ("Условия подтверждения", "confirmation_criteria", None,
     "формальные условия достаточности свидетельств",
     ("threshold", "confirmations_needed", "reproducibility")),
    ("Модель стоимости", "cost_model", None,
     "правило стоимости шага и правило остановки по стоимости",
     ("cost_rule", "stop_rule")),
)

# Canonical block titles in document order.
CANONICAL_FRAME_BLOCKS: Tuple[str, ...] = tuple(e[0] for e in FRAME_SPEC)

_SPEC_BY_TITLE = {e[0]: e for e in FRAME_SPEC}


# ── UI labels and translations (not LLM-facing) ───────────────────────────────
# ``FRAME_SPEC`` titles, usages and field names stay canonical — the LLM schema
# and the form round-trip key on them. The maps below only ADD human-readable
# labels and translations for the web form; nothing here feeds back into the
# frame structure. Keyed by the canonical Russian block title / field name.
BLOCK_I18N: Dict[str, Dict[str, Dict[str, str]]] = {
    "Вопрос исследования": {
        "title": {"en": "Research question", "ru": "Вопрос исследования"},
        "usage": {"en": "Attributes of the root research question.",
                  "ru": "атрибуты корневого исследовательского вопроса"},
    },
    "Режим и завершение": {
        "title": {"en": "Mode and completion", "ru": "Режим и завершение"},
        "usage": {"en": "The AI application mode and the research stop criterion.",
                  "ru": "режим применения ИИ и критерий остановки исследования"},
    },
    "Профиль исследования": {
        "title": {"en": "Research profile", "ru": "Профиль исследования"},
        "usage": {"en": "Modality and setting. This block defines which modules activate.",
                  "ru": "модальность и постановка — определяет активируемые модули"},
    },
    "Методологические нормы": {
        "title": {"en": "Methodological norms", "ru": "Методологические нормы"},
        "usage": {"en": "The accepted methods for this type of research.",
                  "ru": "принятые способы проведения исследований этого типа"},
    },
    "Теоретические рамки": {
        "title": {"en": "Theoretical framework", "ru": "Теоретические рамки"},
        "usage": {"en": "The formal models and theories of the domain.",
                  "ru": "формальные модели и теории домена"},
    },
    "Доменные стандарты": {
        "title": {"en": "Domain standards", "ru": "Доменные стандарты"},
        "usage": {"en": "GOST, ISO, ICH, CLSI, and journal requirements.",
                  "ru": "ГОСТ/ISO/ICH/CLSI и требования журналов"},
    },
    "Этика и регуляторика": {
        "title": {"en": "Ethics and regulation", "ru": "Этика и регуляторика"},
        "usage": {"en": "Hard limits that the research must not violate.",
                  "ru": "жёсткие ограничения, которые нельзя нарушать"},
    },
    "Экспертное знание": {
        "title": {"en": "Expert knowledge", "ru": "Экспертное знание"},
        "usage": {"en": "Informal knowledge from the human in the loop.",
                  "ru": "неформализованное знание от человека в контуре"},
    },
    "Роли участников": {
        "title": {"en": "Participant roles", "ru": "Роли участников"},
        "usage": {"en": "The people whose work the system automates.",
                  "ru": "состав лиц, чья деятельность автоматизируется"},
    },
    "Ресурсы и бюджеты": {
        "title": {"en": "Resources and budgets", "ru": "Ресурсы и бюджеты"},
        "usage": {"en": "Finite budgets. They limit the completion criteria.",
                  "ru": "конечные бюджеты — ограничивают критерии завершения"},
    },
    "Инструменты": {
        "title": {"en": "Tools", "ru": "Инструменты"},
        "usage": {"en": "The known tools and their status.",
                  "ru": "известные инструменты и их статус"},
    },
    "Эмпирическая база": {
        "title": {"en": "Empirical base", "ru": "Эмпирическая база"},
        "usage": {"en": "The available observations, data, and corpora.",
                  "ru": "доступные наблюдения, данные и корпус"},
    },
    "Условия подтверждения": {
        "title": {"en": "Confirmation criteria", "ru": "Условия подтверждения"},
        "usage": {"en": "The formal conditions for sufficient evidence.",
                  "ru": "формальные условия достаточности свидетельств"},
    },
    "Модель стоимости": {
        "title": {"en": "Cost model", "ru": "Модель стоимости"},
        "usage": {"en": "The step cost rule and the cost stop rule.",
                  "ru": "правило стоимости шага и правило остановки по стоимости"},
    },
}

FIELD_I18N: Dict[str, Dict[str, Dict[str, str]]] = {
    "formulation": {
        "label": {"en": "Formulation", "ru": "Формулировка"},
        "placeholder": {
            "en": "Enter the full text of the research question, or leave it empty so the agent fills in a working value.",
            "ru": "Введите полный текст исследовательского вопроса или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "domain": {
        "label": {"en": "Domain", "ru": "Домен"},
        "placeholder": {
            "en": "Enter the subject area of the research, or leave it empty so the agent fills in a working value.",
            "ru": "Введите предметную область исследования или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "specificity": {
        "label": {"en": "Specificity", "ru": "Специфичность"},
        "placeholder": {
            "en": "Enter how narrow the question is, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите, насколько узок вопрос, или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "gap": {
        "label": {"en": "Knowledge gap", "ru": "Пробел в знаниях"},
        "placeholder": {
            "en": "Enter the knowledge gap that the research addresses, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите пробел в знаниях, который закрывает исследование, или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "decomposition": {
        "label": {"en": "Decomposition", "ru": "Декомпозиция"},
        "placeholder": {
            "en": "Enter the split of the question into sub-questions, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите разбиение вопроса на подвопросы или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "target_setting": {
        "label": {"en": "Target setting", "ru": "Целевая установка"},
        "placeholder": {
            "en": "Enter the result that the research must produce, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите результат, который должно дать исследование, или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "research_form": {
        "label": {"en": "Research form", "ru": "Форма исследования"},
        "placeholder": {
            "en": "Enter the study type, such as review or experiment, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите форму исследования (обзор, эксперимент, моделирование) или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "trl": {
        "label": {"en": "Technology readiness level (TRL)",
                  "ru": "Уровень технологической готовности (TRL)"},
        "placeholder": {
            "en": "Enter the technology readiness level from 1 to 9, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите уровень технологической готовности от 1 до 9 или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "ai_application_model": {
        "label": {"en": "AI application model", "ru": "Модель применения ИИ"},
        "placeholder": {
            "en": "Enter how the AI takes part, for example assistant or autonomous agent, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите, как участвует ИИ (ассистент, соавтор, автономный агент), или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "completion_criteria": {
        "label": {"en": "Completion criteria", "ru": "Критерии завершения"},
        "placeholder": {
            "en": "Enter the conditions that stop the research, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите условия остановки исследования или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "modality": {
        "label": {"en": "Modality", "ru": "Модальность"},
        "placeholder": {
            "en": "Enter the modality (theoretical, experimental, computational), or leave it empty so the agent fills in a working value.",
            "ru": "Укажите модальность (теоретическая, экспериментальная, вычислительная) или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "form_trl": {
        "label": {"en": "Form and TRL", "ru": "Форма и TRL"},
        "placeholder": {
            "en": "Enter the study form and readiness level, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите форму исследования и уровень готовности или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "norms": {
        "label": {"en": "Norms", "ru": "Нормы"},
        "placeholder": {
            "en": "Enter the accepted methods for this type of research, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите принятые методы для этого типа исследований или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "frameworks": {
        "label": {"en": "Models and theories", "ru": "Модели и теории"},
        "placeholder": {
            "en": "Enter the formal models and theories of the domain, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите формальные модели и теории домена или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "standards": {
        "label": {"en": "Standards", "ru": "Стандарты"},
        "placeholder": {
            "en": "Enter the applicable standards, such as GOST, ISO, ICH, or CLSI, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите применимые стандарты (ГОСТ, ISO, ICH, CLSI) или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "constraints": {
        "label": {"en": "Constraints", "ru": "Ограничения"},
        "placeholder": {
            "en": "Enter the hard limits that the research must not violate, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите жёсткие ограничения, которые нельзя нарушать, или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "notes": {
        "label": {"en": "Notes", "ru": "Заметки"},
        "placeholder": {
            "en": "Enter informal knowledge from the human in the loop, or leave it empty so the agent fills in a working value.",
            "ru": "Введите неформализованное знание от эксперта или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "roles": {
        "label": {"en": "Roles", "ru": "Роли"},
        "placeholder": {
            "en": "Enter the people whose work the system automates, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите лиц, чья деятельность автоматизируется, или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "gpu_hours": {
        "label": {"en": "GPU hours", "ru": "Часы GPU"},
        "placeholder": {
            "en": "Enter the GPU time budget, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите бюджет времени GPU или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "tokens": {
        "label": {"en": "Tokens", "ru": "Токены"},
        "placeholder": {
            "en": "Enter the token budget for the language model calls, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите бюджет токенов для вызовов языковой модели или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "money": {
        "label": {"en": "Money", "ru": "Деньги"},
        "placeholder": {
            "en": "Enter the money budget, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите денежный бюджет или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "time": {
        "label": {"en": "Time", "ru": "Время"},
        "placeholder": {
            "en": "Enter the time budget, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите бюджет времени или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "expert_hours": {
        "label": {"en": "Expert hours", "ru": "Часы экспертов"},
        "placeholder": {
            "en": "Enter the expert time budget, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите бюджет времени экспертов или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "computational": {
        "label": {"en": "Computational tools", "ru": "Вычислительные инструменты"},
        "placeholder": {
            "en": "Enter the computational tools and their status, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите вычислительные инструменты и их статус или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "laboratory": {
        "label": {"en": "Laboratory tools", "ru": "Лабораторные инструменты"},
        "placeholder": {
            "en": "Enter the laboratory tools and their status, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите лабораторные инструменты и их статус или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "analytical": {
        "label": {"en": "Analytical tools", "ru": "Аналитические инструменты"},
        "placeholder": {
            "en": "Enter the analytical tools and their status, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите аналитические инструменты и их статус или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "informational": {
        "label": {"en": "Informational tools", "ru": "Информационные инструменты"},
        "placeholder": {
            "en": "Enter the informational tools and their status, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите информационные инструменты и их статус или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "datasets": {
        "label": {"en": "Datasets", "ru": "Наборы данных"},
        "placeholder": {
            "en": "Enter the available datasets, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите доступные наборы данных или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "corpora": {
        "label": {"en": "Corpora", "ru": "Корпуса"},
        "placeholder": {
            "en": "Enter the available text corpora, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите доступные корпуса текстов или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "trusted_kb": {
        "label": {"en": "Trusted knowledge bases", "ru": "Доверенные базы знаний"},
        "placeholder": {
            "en": "Enter the trusted knowledge bases, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите доверенные базы знаний или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "threshold": {
        "label": {"en": "Threshold", "ru": "Порог"},
        "placeholder": {
            "en": "Enter the evidence threshold, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите порог достаточности свидетельств или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "confirmations_needed": {
        "label": {"en": "Confirmations needed", "ru": "Число подтверждений"},
        "placeholder": {
            "en": "Enter how many independent confirmations a result needs, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите, сколько независимых подтверждений нужно результату, или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "reproducibility": {
        "label": {"en": "Reproducibility", "ru": "Воспроизводимость"},
        "placeholder": {
            "en": "Enter the reproducibility requirement, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите требование к воспроизводимости или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "cost_rule": {
        "label": {"en": "Cost rule", "ru": "Правило стоимости"},
        "placeholder": {
            "en": "Enter the rule that computes the cost of one step, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите правило стоимости одного шага или оставьте поле пустым — агент заполнит рабочее значение."},
    },
    "stop_rule": {
        "label": {"en": "Stop rule", "ru": "Правило остановки"},
        "placeholder": {
            "en": "Enter the rule that stops the research because of cost, or leave it empty so the agent fills in a working value.",
            "ru": "Укажите правило остановки по стоимости или оставьте поле пустым — агент заполнит рабочее значение."},
    },
}

# Every canonical block and field must have a UI label; fail fast at import.
assert all(e[0] in BLOCK_I18N for e in FRAME_SPEC)
assert all(n in FIELD_I18N for _t, _k, _s, _u, names in FRAME_SPEC for n in names)


class FrameField(BaseModel):
    """One field of a block: |Поле|Значение|Статус|."""

    name: str = Field(description="Имя поля, напр. «formulation»")
    value: str = Field(default="Не задано", description="Значение поля")
    status: FieldStatus = Field(default="не задано")

    def is_set(self) -> bool:
        return self.status not in OPEN_STATUSES


class FrameBlock(BaseModel):
    """One block of the frame: a titled group of fields with a graph mapping."""

    title: str = Field(description="Название блока, напр. «Профиль исследования»")
    kind: str = Field(default="question", description="как блок ложится в граф")
    subtype: Optional[str] = Field(
        default=None, description="подтип Constraint, если kind == constraint")
    usage: str = Field(default="", description="как блок используется дальше")
    fields: List[FrameField] = Field(default_factory=list)

    def set_fields(self) -> List[FrameField]:
        return [f for f in self.fields if f.is_set()]


class ResearchFrame(BaseModel):
    """Структурированная рамка исследования (framing-сущности мета-модели).

    Blocks in document order, each a group of fields with a provenance status.
    The blocks seed the Research Context Graph (see ``context_init.commit``).
    """

    original_request: str = Field(
        default="", description="Исходный запрос пользователя дословно")
    blocks: List[FrameBlock] = Field(default_factory=list)

    def block(self, title: str) -> Optional[FrameBlock]:
        for b in self.blocks:
            if b.title.strip().lower() == title.strip().lower():
                return b
        return None

    def open_fields(self) -> List[Tuple[str, str]]:
        """(block_title, field_name) for every field still needing a value."""
        out: List[Tuple[str, str]] = []
        for b in self.blocks:
            for f in b.fields:
                if not f.is_set():
                    out.append((b.title, f.name))
        return out

    @classmethod
    def blank(cls, original_request: str = "") -> "ResearchFrame":
        """An all-open frame with every canonical block and field."""
        blocks = [
            FrameBlock(
                title=title, kind=kind, subtype=subtype, usage=usage,
                fields=[FrameField(name=n) for n in field_names],
            )
            for title, kind, subtype, usage, field_names in FRAME_SPEC
        ]
        return cls(original_request=original_request, blocks=blocks)

    def normalized(self) -> "ResearchFrame":
        """Re-impose the canonical structure, keeping values/status the model set.

        The LLM may reorder blocks, rename kinds, or drop fields. This maps its
        output back onto ``FRAME_SPEC`` by title/field name so the graph mapping
        can never break, while preserving the values and statuses it produced.
        """
        blocks: List[FrameBlock] = []
        for title, kind, subtype, usage, field_names in FRAME_SPEC:
            src = self.block(title)
            src_fields = {f.name.strip().lower(): f for f in (src.fields if src else [])}
            fields = []
            for n in field_names:
                got = src_fields.get(n.strip().lower())
                if got is not None:
                    fields.append(FrameField(name=n, value=got.value, status=got.status))
                else:
                    fields.append(FrameField(name=n))
            blocks.append(FrameBlock(
                title=title, kind=kind, subtype=subtype, usage=usage, fields=fields))
        return ResearchFrame(original_request=self.original_request, blocks=blocks)


__all__ = [
    "BLOCK_I18N",
    "CANONICAL_FRAME_BLOCKS",
    "FIELD_I18N",
    "FRAME_SPEC",
    "FrameBlock",
    "FrameField",
    "ResearchFrame",
]
