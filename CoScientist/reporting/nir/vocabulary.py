"""Russian words for the graph's vocabulary, in the register a GOST report uses.

The repository already holds five Russian vocabularies and they disagree:
``Evidence`` renders as «Свидетельство» in ``store._KIND_WORDS``, «Данные» in the
web i18n table, «полученное свидетельство» in this package's appendix and
«свидетельство» in the slide renderer; statuses differ in grammatical gender
between ``store._STATUS_WORDS`` and ``slide_render._L``. Mixing them produces a
document that names the same thing three ways, so this module is the single
table the report speaks from.

It reuses rather than reinvents. ``store._STATUS_WORDS`` is the widest status
map anywhere — it is the only one carrying ``inconclusive``, ``skipped`` and the
task-tracker states — and ``store._KIND_WORDS`` is the canonical node vocabulary.
Both are display-only by their own comment, so importing them is safe. Edge types
exist in Russian only in ``web/static/js/i18n.js``; that table is ported below
because Python has no copy of it.

Where the GOST register needs a different word, the override says why. A
research card may read «проверена — без ответа»; a normative report says «не
разрешена имеющимися данными», because the reader is a commission and not the
scientist who ran the study.

**Deliberately unlike ``agents/callbacks/report_language.py``.** That module's
``_RU_BLOCK`` tells the aggregator to keep node ids, file paths and tool
identifiers *untranslated* — correct for a short Markdown report a researcher
reads beside the graph. A NIR report is read by people who have never seen the
graph, so here every internal designation is dissolved instead. The divergence
is intentional; see ``register_check`` for what enforces it.
"""
from __future__ import annotations

from typing import Dict

from CoScientist.graph.research.store import _KIND_WORDS, _STATUS_WORDS

# ── Node types ──────────────────────────────────────────────────────────────

#: Overrides on top of ``_KIND_WORDS``, for the report's register.
_KIND_OVERRIDES: Dict[str, str] = {
    # "Свидетельство" is the spec's word and reads as testimony in a document
    # meant for a commission. What the study actually recorded is an
    # observation, and that is the word GOST prose uses.
    "Evidence": "Наблюдение",
    # "Бюджет" is the viewer's shorthand; a report names the thing itself.
    "Resource": "Ресурс",
    # The plan track, named the way a work plan names its rows.
    "PlanStep": "Задача",
    "ExperimentTask": "Эксперимент",
    "Conclusion": "Заключение",
}

NODE_WORDS: Dict[str, str] = {**_KIND_WORDS, **_KIND_OVERRIDES}

#: What each node type is *for*, for the reference appendix that explains the
#: record to someone who will never open it.
NODE_PURPOSE: Dict[str, str] = {
    "ResearchQuestion": "вопрос, на который отвечает работа",
    "Hypothesis": "проверяемое предположение",
    "Evidence": "зафиксированное наблюдение",
    "Conclusion": "вывод по итогам проверки",
    "VerificationMethod": "способ проверки предположения",
    "ConfirmationCriteria": "условие, при котором предположение считается подтверждённым",
    "PlanStep": "задача плана работ",
    "ExperimentTask": "отдельный эксперимент",
    "Tool": "использованное программное средство",
    "Resource": "израсходованный ресурс",
    "EmpiricalBase": "источник исходных данных",
    "Constraint": "ограничение, принятое в работе",
    "CodeArtifact": "разработанный программный код",
    "GeneratedData": "полученный набор данных",
    "Report": "отчётный материал",
    "Publication": "публикация по результатам",
    "Spec": "техническое задание",
    "CostModel": "модель стоимости",
    "EfficiencyMetric": "показатель эффективности",
    "EfficiencyJustification": "обоснование эффективности",
    "Framing": "постановка задачи",
    "Outcome": "итог работы",
}

# ── Statuses ────────────────────────────────────────────────────────────────

#: Overrides on top of ``_STATUS_WORDS``. The viewer writes for the scientist
#: who ran the study; a report writes for a reader who needs the verdict stated
#: in full.
_STATUS_OVERRIDES: Dict[str, str] = {
    "inconclusive": "не разрешена имеющимися данными",
    "formulated": "сформулирована, проверка не завершена",
    "postponed": "не проверялась",
    "under_verification": "на момент составления отчёта проверялась",
}

STATUS_WORDS: Dict[str, str] = {**_STATUS_WORDS, **_STATUS_OVERRIDES}

# ── Evidence kinds ──────────────────────────────────────────────────────────

EVIDENCE_KINDS: Dict[str, str] = {
    "literature": "литературное",
    "experimental": "экспериментальное",
    "computational": "расчётное",
    "expert": "экспертное",
    "meta": "обобщающее",
}

#: ``method_type``, ``tool_type`` and ``base_type`` share a vocabulary and are
#: free text, so an agent writes whichever word it likes. Adjectival forms, to
#: sit after the noun they qualify: «метод проверки 1 (расчётный)».
KIND_WORDS: Dict[str, str] = {
    "computational": "расчётный",
    "experimental": "экспериментальный",
    "laboratory": "лабораторный",
    "analytical": "аналитический",
    "informational": "информационный",
    "literature": "литературный",
    "expert": "экспертный",
    "meta": "обобщающий",
    "dataset": "набор данных",
    "corpus": "корпус текстов",
    "knowledge_base": "база знаний",
}

# ── Edge types ──────────────────────────────────────────────────────────────

#: Ported from ``graph.edge.*`` in ``web/static/js/i18n.js`` — the only EN→RU
#: edge table in the project, and it lives in JavaScript. Kept as nouns rather
#: than the viewer's verbs, because the appendix names a kind of link, not an
#: action ("проверяется" is a caption under an arrow; a table row needs
#: "проверка предположения").
EDGE_WORDS: Dict[str, str] = {
    "motivates": "постановка вопроса",
    "tested_by": "проверка предположения",
    "requires": "необходимое условие",
    "uses": "использование средства",
    "consumes": "расход ресурса",
    "produces": "получение результата",
    "supports": "подтверждение",
    "refutes": "опровержение",
    "refines": "уточнение",
    "supersedes": "замещение",
    "conditional_successor": "условный переход после опровержения",
    "based_on": "основание вывода",
    "determines_sufficiency": "достаточность данных",
    "formulated_for": "условие подтверждения",
    "regulates": "регулирование",
    "constrains": "ограничение",
    "derived_from": "происхождение данных",
    "contextualizes": "контекст работы",
    "defines_scope": "определение области",
    "relates_to": "смысловая связь",
    "applies_to": "область применения",
    "elaborates": "детализация",
    "realises": "реализация плана",
}

# ── Ordinals ────────────────────────────────────────────────────────────────

#: Feminine, to agree with «гипотеза» and «задача». Past ten the report says
#: «гипотеза 11», which is plain and correct — spelling out «одиннадцатая» in a
#: heading reads worse than the numeral.
_ORDINALS_F = (
    "первая", "вторая", "третья", "четвёртая", "пятая",
    "шестая", "седьмая", "восьмая", "девятая", "десятая",
)


#: Genitive, for "проверка первой гипотезы". Russian headings decline, and
#: "Проверка первая гипотезы" is the kind of mistake a reader stops on.
_ORDINALS_F_GEN = (
    "первой", "второй", "третьей", "четвёртой", "пятой",
    "шестой", "седьмой", "восьмой", "девятой", "десятой",
)


def ordinal_f(index: int) -> str:
    """Feminine nominative ordinal for a 1-based index, or the numeral past ten."""
    if 1 <= index <= len(_ORDINALS_F):
        return _ORDINALS_F[index - 1]
    return str(index)


def ordinal_f_gen(index: int) -> str:
    """Feminine genitive ordinal — «проверка первой гипотезы»."""
    if 1 <= index <= len(_ORDINALS_F_GEN):
        return _ORDINALS_F_GEN[index - 1]
    return str(index)


def node_word(node_type: str) -> str:
    return NODE_WORDS.get(str(node_type), str(node_type))


def status_word(status: str) -> str:
    """The verdict in words. Unknown statuses come back untouched.

    Returning the raw code for an unknown status is deliberate: it surfaces in
    the register check as a leak, which is how a status added to the schema and
    forgotten here gets noticed.
    """
    return STATUS_WORDS.get(str(status), str(status))


def evidence_kind(subtype: str) -> str:
    return EVIDENCE_KINDS.get(str(subtype), "")


def kind_word(kind: str) -> str:
    """Adjective for a method/tool/base type. Unknown kinds vanish.

    Dropping an unrecognised kind is deliberate: printing it would put the
    schema's English into the document, and the word it qualifies reads fine
    without it.
    """
    return KIND_WORDS.get(str(kind or "").strip().lower(), "")


def edge_word(edge_type: str) -> str:
    return EDGE_WORDS.get(str(edge_type), str(edge_type))


def purpose(node_type: str) -> str:
    return NODE_PURPOSE.get(str(node_type), "—")


__all__ = [
    "NODE_WORDS",
    "NODE_PURPOSE",
    "STATUS_WORDS",
    "EVIDENCE_KINDS",
    "KIND_WORDS",
    "EDGE_WORDS",
    "ordinal_f",
    "ordinal_f_gen",
    "node_word",
    "status_word",
    "evidence_kind",
    "kind_word",
    "edge_word",
    "purpose",
]
