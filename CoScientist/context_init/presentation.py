"""Human-facing structure of the research technical specification.

The canonical frame in :mod:`CoScientist.context_init.models` is a machine
contract.  Its field names are intentionally stable and therefore appear in
English.  This module says how the same values are read by a person: which
section they belong to, what the label means, and what short explanation is
shown next to it.

Nothing here changes the frame, seeds the graph, or invokes a model.  Both the
graph sidebar and future document views can consume this registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


Localized = Dict[str, str]


def _text(ru: str, en: str) -> Localized:
    return {"ru": ru, "en": en}


@dataclass(frozen=True)
class FieldPresentation:
    block: str
    name: str
    label: Localized
    help: Localized

    @property
    def id(self) -> str:
        return f"{self.block}:{self.name}"


@dataclass(frozen=True)
class SectionPresentation:
    id: str
    title: Localized
    fields: Tuple[FieldPresentation, ...]
    open_by_default: bool = False
    documents: bool = False


def _field(block: str, name: str, ru: str, en: str,
           help_ru: str, help_en: str) -> FieldPresentation:
    return FieldPresentation(
        block=block,
        name=name,
        label=_text(ru, en),
        help=_text(help_ru, help_en),
    )


SECTIONS: Tuple[SectionPresentation, ...] = (
    SectionPresentation(
        id="general",
        title=_text("1. Общие сведения и основание", "1. General information and basis"),
        open_by_default=True,
        fields=(
            _field("Основание и приёмка", "topic_name", "Название исследования",
                   "Research title", "Наименование работы.", "The name of the work."),
            _field("Вопрос исследования", "domain", "Предметная область", "Subject area",
                   "Область знаний, к которой относится задача.",
                   "The field of knowledge the task belongs to."),
            _field("Основание и приёмка", "customer", "Заказчик", "Customer",
                   "Для кого выполняется работа.", "Who commissioned the work."),
            _field("Основание и приёмка", "basis_document", "Основание для работы",
                   "Basis for the work", "Запрос, договор или иной исходный документ.",
                   "The request, contract, or other authorising document."),
            _field("Роли участников", "roles", "Участники и ответственность",
                   "Participants and responsibilities", "Кто участвует в работе и за что отвечает.",
                   "Who participates in the work and what they are responsible for."),
        ),
    ),
    SectionPresentation(
        id="goal",
        title=_text("2. Цель и границы исследования", "2. Goal and research boundaries"),
        open_by_default=True,
        fields=(
            _field("Вопрос исследования", "formulation", "Исследовательский вопрос",
                   "Research question", "Вопрос, на который должно ответить исследование.",
                   "The question the research must answer."),
            _field("Вопрос исследования", "gap", "Проблема, которую решаем",
                   "Problem being addressed", "Какого знания или решения сейчас не хватает.",
                   "The knowledge or solution that is currently missing."),
            _field("Вопрос исследования", "target_setting", "Ожидаемый результат",
                   "Expected result", "Какой ответ или результат требуется получить.",
                   "The answer or result that must be produced."),
            _field("Вопрос исследования", "specificity", "Границы исследования",
                   "Research boundaries", "Насколько узко определён исследовательский вопрос.",
                   "How narrowly the research question is scoped."),
            _field("Вопрос исследования", "trl", "Уровень готовности результата (TRL)",
                   "Result readiness level (TRL)",
                   "Технологическая зрелость результата, если показатель применим.",
                   "The technology readiness of the result, when applicable."),
        ),
    ),
    SectionPresentation(
        id="tasks",
        title=_text("3. Задачи и ожидаемые результаты", "3. Tasks and expected deliverables"),
        fields=(
            _field("$operations", "operations", "Обязательные задачи", "Required tasks",
                   "Задачи, которые должны быть выполнены в рамках исследования.",
                   "Tasks that must be completed during the research."),
            _field("Вопрос исследования", "decomposition", "Подзадачи исследования",
                   "Research subtasks", "Содержательное разбиение вопроса на части.",
                   "A substantive decomposition of the question."),
            _field("Основание и приёмка", "deliverables", "Результаты и материалы к передаче",
                   "Deliverables", "Отчёт, код, данные, статья и другие результаты работы.",
                   "The report, code, data, article, and other outputs to be handed over."),
        ),
    ),
    SectionPresentation(
        id="method",
        title=_text("4. Методика и ограничения", "4. Method and constraints"),
        fields=(
            _field("Профиль исследования", "modality", "Подход к исследованию",
                   "Research approach", "Теоретический, вычислительный, экспериментальный или смешанный.",
                   "Theoretical, computational, experimental, or mixed."),
            _field("Вопрос исследования", "research_form", "Формат исследования",
                   "Research format", "Обзор, эксперимент, моделирование или иной формат работы.",
                   "Review, experiment, modelling, or another form of research."),
            _field("Профиль исследования", "target_setting", "Тип постановки задачи",
                   "Problem setting", "Как задача поставлена с точки зрения выбранного подхода.",
                   "How the problem is formulated for the selected approach."),
            _field("Профиль исследования", "form_trl", "Форма исследования и готовность",
                   "Research form and readiness", "Составная характеристика формы работы и зрелости результата.",
                   "A combined description of the work form and result readiness."),
            _field("Режим и завершение", "ai_application_model", "Роль ИИ в работе",
                   "Role of AI", "Как ИИ участвует в исследовании.",
                   "How AI participates in the research."),
            _field("Методологические нормы", "norms", "Правила проведения исследования",
                   "Research conduct rules", "Требования к методике, корректности и качеству работы.",
                   "Requirements for the method, correctness, and quality of the work."),
            _field("Теоретические рамки", "frameworks", "Теоретические модели и подходы",
                   "Theoretical models and approaches", "Теории и модели предметной области; это не обязательно библиотеки ПО.",
                   "Domain theories and models; these are not necessarily software libraries."),
            _field("Доменные стандарты", "standards", "Применяемые стандарты",
                   "Applicable standards", "ГОСТ, ISO и другие явно заданные нормы.",
                   "GOST, ISO, and other explicitly applicable standards."),
            _field("Этика и регуляторика", "constraints", "Этические и регуляторные ограничения",
                   "Ethical and regulatory constraints", "Обязательные ограничения и недопустимые действия.",
                   "Mandatory constraints and prohibited actions."),
            _field("Экспертное знание", "notes", "Примечания эксперта", "Expert notes",
                   "Сведения и рекомендации, предоставленные человеком.",
                   "Information and recommendations supplied by a human expert."),
        ),
    ),
    SectionPresentation(
        id="inputs",
        title=_text("5. Исходные данные и инструменты", "5. Input data and tools"),
        fields=(
            _field("Эмпирическая база", "datasets", "Наборы данных", "Datasets",
                   "Исходные структурированные данные исследования.", "Structured input data for the research."),
            _field("Эмпирическая база", "corpora", "Коллекции текстов и публикаций",
                   "Text and publication collections", "Материалы для изучения и анализа.",
                   "Materials used for review and analysis."),
            _field("Эмпирическая база", "trusted_kb", "Проверенные базы знаний",
                   "Trusted knowledge bases", "Источники, заявленные доверенными в задании.",
                   "Sources designated as trusted in the specification."),
            _field("Инструменты", "computational", "Средства расчёта и моделирования",
                   "Computation and modelling tools", "Вычислительные инструменты, доступные для работы.",
                   "Computational tools available for the work."),
            _field("Инструменты", "laboratory", "Лабораторные средства", "Laboratory tools",
                   "Оборудование и лабораторные инструменты.", "Laboratory equipment and tools."),
            _field("Инструменты", "analytical", "Средства анализа", "Analysis tools",
                   "Инструменты для анализа результатов.", "Tools used to analyse results."),
            _field("Инструменты", "informational", "Средства поиска информации",
                   "Information retrieval tools", "Инструменты для поиска и получения информации.",
                   "Tools used to find and retrieve information."),
        ),
    ),
    SectionPresentation(
        id="plan",
        title=_text("6. План, сроки и ресурсы", "6. Plan, schedule, and resources"),
        fields=(
            _field("Основание и приёмка", "stages", "Этапы и сроки", "Stages and deadlines",
                   "Планируемые этапы работы и сроки их выполнения.",
                   "The planned work stages and their deadlines."),
            _field("Ресурсы и бюджеты", "gpu_hours", "Вычислительный бюджет, GPU-часы",
                   "Compute budget, GPU hours", "Лимит вычислительных ресурсов.",
                   "The compute resource limit."),
            _field("Ресурсы и бюджеты", "tokens", "Лимит токенов ИИ", "AI token limit",
                   "Бюджет обращений к языковым моделям.", "The language-model token budget."),
            _field("Ресурсы и бюджеты", "money", "Денежный бюджет", "Financial budget",
                   "Допустимые денежные затраты в указанной валюте.",
                   "Permitted financial expenditure in the stated currency."),
            _field("Ресурсы и бюджеты", "time", "Лимит времени", "Time limit",
                   "Ограничение по календарному времени или длительности выполнения.",
                   "The calendar or execution-time limit."),
            _field("Ресурсы и бюджеты", "expert_hours", "Время эксперта, часы",
                   "Expert time, hours", "Бюджет участия человека-эксперта.",
                   "The available human-expert time."),
            _field("Модель стоимости", "cost_rule", "Как рассчитываются затраты",
                   "How costs are calculated", "Правило оценки стоимости одного шага.",
                   "The rule used to estimate the cost of a step."),
            _field("Модель стоимости", "stop_rule", "Когда остановиться из-за затрат",
                   "When cost stops the work", "Условие прекращения работы из-за стоимости.",
                   "The condition that stops the work because of cost."),
            _field("Модель стоимости", "expected_effect", "Ожидаемая польза",
                   "Expected benefit", "Что даст работа по сравнению с исходным положением.",
                   "What the work is expected to improve over the starting point."),
        ),
    ),
    SectionPresentation(
        id="acceptance",
        title=_text("7. Проверка результата и приёмка", "7. Result verification and acceptance"),
        fields=(
            _field("Условия подтверждения", "threshold", "Условие подтверждения результата",
                   "Result confirmation condition", "Порог или качественное условие достаточности свидетельств.",
                   "The threshold or qualitative condition for sufficient evidence."),
            _field("Условия подтверждения", "confirmations_needed",
                   "Требуемое число независимых подтверждений", "Required independent confirmations",
                   "Сколько независимых подтверждений требуется.",
                   "How many independent confirmations are required."),
            _field("Условия подтверждения", "reproducibility", "Требования к воспроизводимости",
                   "Reproducibility requirements", "Как результат должен воспроизводиться.",
                   "How the result must be reproduced."),
            _field("Режим и завершение", "completion_criteria", "Условия завершения исследования",
                   "Research completion conditions", "При каких условиях процесс исследования заканчивается.",
                   "The conditions under which the research process ends."),
            _field("Основание и приёмка", "acceptance", "Порядок приёмки", "Acceptance procedure",
                   "Кто принимает работу и как проверяется результат.",
                   "Who accepts the work and how the result is checked."),
        ),
    ),
    SectionPresentation(
        id="documents",
        title=_text("8. Документы и источники", "8. Documents and sources"),
        documents=True,
        fields=(
            _field("$meta", "original_request", "Исходный запрос", "Original request",
                   "Запрос пользователя, на основании которого сформировано задание.",
                   "The user request from which the specification was formed."),
        ),
    ),
)


FIELD_BY_ID = {field.id: field for section in SECTIONS for field in section.fields}


STATUS_I18N: Dict[str, Localized] = {
    "задано заказчиком": _text("Задано заказчиком", "Provided by the customer"),
    "уточнено оператором": _text("Уточнено оператором", "Confirmed by the operator"),
    "автоподбор": _text("Подобрано автоматически", "Selected automatically"),
    "предложено агентом": _text("Предложено агентом", "Proposed by an agent"),
    "заполнено агентом": _text("Заполнено агентом", "Filled by an agent"),
    "рассчитывается агентом": _text("Рассчитывается агентом", "To be calculated by an agent"),
    "свободный комментарий": _text("Свободный комментарий", "Free-form comment"),
    "не требуется": _text("Не требуется", "Not required"),
    "не задано": _text("Не задано", "Not specified"),
    "восстановлено из графа": _text("Восстановлено из графа", "Recovered from the graph"),
    "происхождение неизвестно": _text("Происхождение неизвестно", "Source unknown"),
}


__all__ = ["FIELD_BY_ID", "SECTIONS", "STATUS_I18N", "FieldPresentation",
           "SectionPresentation"]
