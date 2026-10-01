"""Structural checks against AI-writing tells, adapted for Russian prose.

Informed by conorbronsdon/avoid-ai-writing, which is a SKILL.md for coding
agents rather than a library this runtime can call, and whose word list is
English ("delve", "leverage", "robust", "Moreover"). What ports is the shape of
the problem, not the vocabulary, so the rules below were rewritten for a
Russian scientific register.

Two rules from that skill are deliberately NOT ported:

* **Em-dash frequency.** Russian uses "—" as a copula where English uses a verb
  ("Цель работы — проверить..."), so the English threshold would fire on
  correct writing several times per page.
* **Bare noun-phrase bullet lists.** A GOST report carries its enumerations as
  tables, which the builder constructs; flagging noun phrases would hit the
  table cells, where they are right.

The skill's own caveat applies and is worth repeating: these are writing-quality
signals, not proof of authorship. Everything here is a warning for the author to
weigh, never a hard failure — a real finding and a false positive look the same
from inside a regex.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

# ── Phrase families ─────────────────────────────────────────────────────────

#: P0 — a report is not a conversation. Any of these means the model slipped
#: into chat register, and a reader of a normative document will notice.
_CHATBOT = (
    "надеюсь, это поможет",
    "надеюсь, что это поможет",
    "если у вас остались вопросы",
    "давайте рассмотрим",
    "давайте разберём",
    "давайте разберем",
    "как мы видим",
    "как видите",
    "стоит отметить, что важно",
    "в этой статье",
    "в данной статье мы",
)

#: P1 — openings that announce significance instead of stating a fact.
_FORMULAIC_OPENERS = (
    "в современном мире",
    "в быстро меняющемся мире",
    "в эпоху цифровых технологий",
    "в последние годы наблюдается стремительный",
    "трудно переоценить",
    "сложно переоценить",
    "как никогда актуально",
    "не секрет, что",
    "играет важную роль в современном",
)

#: P2 — endings that fill space where a conclusion belongs.
_GENERIC_CONCLUSIONS = (
    "время покажет",
    "покажет время",
    "будущее выглядит многообещающим",
    "открывает новые горизонты",
    "не стоит останавливаться на достигнутом",
    "подводя итог, можно сказать, что",
    "таким образом, мы видим, что",
)

#: P1 — hedges are legitimate in science; a *stack* of them in one clause is the
#: tell. Counted per sentence, flagged from three.
_HEDGES = (
    "возможно",
    "вероятно",
    "по-видимому",
    "потенциально",
    "предположительно",
    "как правило",
    "в некоторой степени",
    "в определённой мере",
    "в определенной мере",
    "может быть",
    "могло бы",
    "мог бы",
    "скорее всего",
    "в перспективе",
)

#: Inflation: adjectives that assert importance the evidence has to earn.
_SIGNIFICANCE_INFLATION = (
    "революционный",
    "прорывной",
    "беспрецедентный",
    "уникальная возможность",
    "критически важнейший",
    "всеобъемлющий анализ",
)

_HEDGE_STACK_THRESHOLD = 3
#: Below this, paragraph lengths are suspiciously uniform. Human scientific
#: prose varies: a one-line qualification next to a twelve-line argument.
_LENGTH_CV_THRESHOLD = 0.22
_MIN_PARAGRAPHS_FOR_SHAPE = 5
#: How many paragraphs may share an opening word before it reads as a template.
_REPEATED_OPENER_THRESHOLD = 3

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z-]*")


@dataclass
class SlopWarning:
    """One finding, addressed to whoever writes the next revision."""

    code: str
    where: str
    message: str

    def render(self) -> str:
        return f"[{self.code}] {self.where}: {self.message}"


def _contains(haystack: str, needles: Iterable[str]) -> List[str]:
    low = haystack.lower()
    return [n for n in needles if n in low]


def _check_phrases(text: str, where: str) -> List[SlopWarning]:
    found: List[SlopWarning] = []
    for hit in _contains(text, _CHATBOT):
        found.append(SlopWarning(
            "chat-register", where,
            f"обращение к читателю «{hit}» — отчёт о НИР не диалог, убрать",
        ))
    for hit in _contains(text, _FORMULAIC_OPENERS):
        found.append(SlopWarning(
            "formulaic-opener", where,
            f"шаблонный зачин «{hit}» — начать с факта, а не с заявления о важности",
        ))
    for hit in _contains(text, _GENERIC_CONCLUSIONS):
        found.append(SlopWarning(
            "generic-conclusion", where,
            f"дежурная концовка «{hit}» — заменить на конкретный вывод или убрать",
        ))
    for hit in _contains(text, _SIGNIFICANCE_INFLATION):
        found.append(SlopWarning(
            "significance-inflation", where,
            f"оценочное «{hit}» — показать значимость числом из Evidence, а не эпитетом",
        ))
    return found


def _check_hedge_stacks(text: str, where: str) -> List[SlopWarning]:
    found: List[SlopWarning] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        hits = _contains(sentence, _HEDGES)
        if len(hits) >= _HEDGE_STACK_THRESHOLD:
            found.append(SlopWarning(
                "hedge-stack", where,
                f"нагромождение оговорок ({', '.join(hits)}) в одном предложении — "
                "оставить одну и назвать источник неопределённости",
            ))
    return found


def _check_shape(paragraphs: Sequence[str], where: str) -> List[SlopWarning]:
    """Uniform length and repeated openings: the two tells that survive translation."""
    found: List[SlopWarning] = []
    lengths = [len(p) for p in paragraphs if p.strip()]
    if len(lengths) >= _MIN_PARAGRAPHS_FOR_SHAPE:
        mean = statistics.fmean(lengths)
        if mean > 0:
            cv = statistics.pstdev(lengths) / mean
            if cv < _LENGTH_CV_THRESHOLD:
                found.append(SlopWarning(
                    "uniform-paragraphs", where,
                    f"все {len(lengths)} абзацев почти одной длины "
                    f"(разброс {cv:.0%}) — признак заполнения шаблона, а не изложения",
                ))

    openers: Dict[str, int] = {}
    for paragraph in paragraphs:
        words = _WORD_RE.findall(paragraph)
        if words:
            key = words[0].lower()
            openers[key] = openers.get(key, 0) + 1
    for word, count in sorted(openers.items(), key=lambda kv: -kv[1]):
        if count >= _REPEATED_OPENER_THRESHOLD:
            found.append(SlopWarning(
                "repeated-opener", where,
                f"{count} абзацев начинаются со слова «{word}» — разнообразить начала",
            ))
    return found


def check_paragraphs(paragraphs: Sequence[str], where: str) -> List[SlopWarning]:
    """Every check, over one section's prose."""
    found: List[SlopWarning] = []
    joined = "\n".join(paragraphs)
    found += _check_phrases(joined, where)
    found += _check_hedge_stacks(joined, where)
    found += _check_shape(paragraphs, where)
    return found


def check_document(sections: Dict[str, Sequence[str]]) -> List[SlopWarning]:
    """Check a whole draft: ``{section label: paragraphs}``.

    The label is what the author sees in the warning, so pass something they can
    navigate by — a section id or a structural name like "введение".
    """
    found: List[SlopWarning] = []
    for label, paragraphs in sections.items():
        if paragraphs:
            found += check_paragraphs(list(paragraphs), label)
    return found


def render(warnings: Sequence[SlopWarning]) -> List[str]:
    """Flatten to strings for a tool result."""
    return [w.render() for w in warnings]


__all__ = ["SlopWarning", "check_paragraphs", "check_document", "render"]
