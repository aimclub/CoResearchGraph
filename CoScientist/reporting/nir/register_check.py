"""Catch the study's internal notation leaking into the report's prose.

A separate module from ``slop_check`` on purpose. That one asks whether the text
reads like a machine wrote it; this one asks whether the text quotes the
machinery. Folding them together would cost both names their meaning.

The first run of NirReportAgent produced 173 paragraphs carrying roughly 160
leaks — node ids, plan ids, function names, file names, English statuses, and
one fragment of a filesystem path (``code_a``). None of it was the model going
wrong: the digest it was shown opened every hypothesis with ``H1.`` and the
prompt told it to copy verbatim. The digest and the prompt are fixed elsewhere;
this is the check that says whether the fix held.

**How identifier matching is narrowed, and where it still collides.** ``H1`` is
a proton in NMR and ``T1`` is a relaxation time, so a pattern alone would flag
correct chemistry in any study. An identifier therefore counts as a leak only
when that exact string is an id in *this run's* graph: a study with no node
called ``H1`` may write ``H1`` freely.

That removes most collisions but not all, and the remaining case is real — a
graph whose first hypothesis is ``H1`` and whose subject is NMR will flag the
nucleus too. It is one reason these findings warn rather than refuse. Read a
``graph-id`` warning as "check this one", not as "this is wrong".

Findings are advisory. They reach the author through ``nir_report_draft`` so a
revision can clear them, and ``nir_report_submit`` builds the document either
way — a report that exists with a stray identifier beats no report at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

from CoScientist.reporting.nir import vocabulary

#: Plan and experiment rows, which the tracker numbers rather than slugs.
_PLAN_ID_RE = re.compile(r"\b(?:TASK|EXP|EXRUN|PLAN|OP)-\d+\b")

#: A function or variable name: lowercase words joined by underscores. Product
#: names never match — they are CamelCase (RDKit, PubChem, CatBoost) or single
#: tokens (ECFP4, LD50, SMILES). A chemical designation containing an
#: underscore would be flagged wrongly; none has appeared in the runs so far,
#: and the finding is a warning rather than a refusal.
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

#: A filename with an extension the run actually produces.
_FILE_RE = re.compile(
    r"\b[\w.-]+\.(?:png|jpe?g|svg|csv|tsv|json|ya?ml|py|docx|pdf|zip|tar\.gz|pkl|h5)\b",
    re.IGNORECASE,
)

#: Agent class names, which say "a machine did this" where the report should
#: say what was done.
_AGENT_RE = re.compile(r"\b[A-Z][A-Za-z]*Agent\b")

#: Schema words that must have been translated. Built from the vocabulary so a
#: status added to the schema and forgotten in the tables shows up here.
_ENGLISH_TOKENS: Set[str] = {
    token.lower()
    for token in list(vocabulary.STATUS_WORDS) + list(vocabulary.NODE_WORDS)
    + list(vocabulary.EDGE_WORDS) + list(vocabulary.EVIDENCE_KINDS)
}
#: …minus the ones that are ordinary English words a report may legitimately
#: use inside a quoted title or a product name.
#: …plus the method vocabulary, whose codes are ordinary English words that a
#: Russian report may legitimately carry inside a quoted title, a metric name
#: or a tool's own documentation.
_ENGLISH_TOKENS -= {"report", "tool", "spec", "publication", "resource", "meta",
                    "used", "proposed"}

_ENGLISH_RE = re.compile(
    r"\b(?:" + "|".join(sorted(re.escape(t) for t in _ENGLISH_TOKENS)) + r")\b",
    re.IGNORECASE,
)


@dataclass
class RegisterWarning:
    """One leak, quoted so the author can find it."""

    code: str
    where: str
    quote: str
    advice: str

    def render(self) -> str:
        return f"[{self.code}] {self.where}: «{self.quote}» — {self.advice}"


def graph_identifiers(nodes: Iterable[Dict]) -> Set[str]:
    """Every id this study actually used.

    The set is what makes the identifier check exact: matching against it
    cannot flag a token the graph never contained.
    """
    found: Set[str] = set()
    for node in nodes or []:
        node_id = str((node or {}).get("id") or "").strip()
        if node_id:
            found.add(node_id)
    return found


def _unique(warnings: List[RegisterWarning], limit: int) -> List[RegisterWarning]:
    """One warning per distinct quote, capped.

    A leaked identifier usually repeats a dozen times; reporting each occurrence
    would bury the other categories and tell the author nothing new.
    """
    seen: Set[tuple] = set()
    out: List[RegisterWarning] = []
    for warning in warnings:
        key = (warning.code, warning.quote.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(warning)
        if len(out) >= limit:
            break
    return out


def check_text(text: str, where: str, ids: Set[str]) -> List[RegisterWarning]:
    """Every leak in one block of prose."""
    found: List[RegisterWarning] = []

    for token in re.findall(r"\b[A-Za-z]{1,3}\d+\b", text):
        if token in ids:
            found.append(RegisterWarning(
                "graph-id", where, token,
                "внутреннее обозначение узла графа; назвать сущность словами "
                "(«первая гипотеза», «условие подтверждения»)",
            ))

    for token in _PLAN_ID_RE.findall(text):
        found.append(RegisterWarning(
            "plan-id", where, token,
            "обозначение из трекера задач; писать «Задача 1», «Эксперимент 2»",
        ))

    for token in _FILE_RE.findall(text):
        found.append(RegisterWarning(
            "file-name", where, token,
            "имя файла из рабочего каталога; описать, что в нём, а не как он называется",
        ))

    # Filenames first: `admet_predictions.json` would otherwise also match the
    # snake_case pattern and be reported twice under different codes.
    without_files = _FILE_RE.sub(" ", text)
    for token in _SNAKE_RE.findall(without_files):
        found.append(RegisterWarning(
            "code-identifier", where, token,
            "имя функции или переменной из кода; назвать метод и продукт "
            "(«кластеризация по алгоритму Бьютины», «модель CatBoost»)",
        ))

    for token in _AGENT_RE.findall(text):
        found.append(RegisterWarning(
            "agent-name", where, token,
            "имя программного агента; писать, что было сделано, а не кем из машин",
        ))

    for token in _ENGLISH_RE.findall(text):
        found.append(RegisterWarning(
            "untranslated", where, token,
            "служебное слово схемы осталось по-английски; перевести на русский",
        ))

    return found


def check_document(
    sections: Dict[str, Sequence[str]],
    nodes: Iterable[Dict],
    limit: int = 25,
) -> List[RegisterWarning]:
    """Check a whole draft: ``{section label: paragraphs}`` against the graph."""
    ids = graph_identifiers(nodes)
    found: List[RegisterWarning] = []
    for label, paragraphs in sections.items():
        for paragraph in paragraphs or []:
            found += check_text(str(paragraph), label, ids)
    return _unique(found, limit)


def render(warnings: Sequence[RegisterWarning]) -> List[str]:
    return [w.render() for w in warnings]


__all__ = [
    "RegisterWarning",
    "graph_identifiers",
    "check_text",
    "check_document",
    "render",
]
