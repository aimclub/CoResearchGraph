"""Persistent, schema-validated store for the Research Context Graph.

One active research per user/session scope (one graph = one root question), held
in a NetworkX MultiDiGraph (parallel typed edges like E1-supports→H1 plus
E1-relates_to→H1 must coexist) and snapshotted atomically to JSON after every
write — the blackboard survives restarts, browser refresh and Web Stop. An
explicit reset or re-initialization archives the previous active graph first;
refuted branches remain available as negative results in that archive.

Writes go through ``commit`` — the transactional API from spec §5.3: ALL nodes,
edges and status changes of one agent step are validated together against the
schema (types, per-agent permissions, status transitions, edge endpoint pairs)
and either applied atomically or rejected with instructive per-item errors.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from uuid import uuid4

import networkx as nx

from CoScientist.graph.research import schema
from CoScientist.graph.research.models import CommitResult, ResearchEdge, ResearchNode
from CoScientist.graph.session_scope import (
    DEFAULT_SESSION_KEY,
    SessionKey,
    session_key,
    storage_dir,
)

logger = logging.getLogger(__name__)

_REF_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_ATTR_CHAR_CAP = 2000
#: The write-up is the one attribute meant to be read at length, so it is not
#: held to the cap every other attribute is. Still bounded: `to_view` ships
#: every node on the page's poll, and an unbounded document there is bandwidth
#: spent once a second. The file itself is attached beside the node, so the cap
#: costs a reader nothing.
_REPORT_CHAR_CAP = 120_000
_COMMIT_HINT = ("Fix the listed items and call research_commit again. "
                "NOTHING from this call was saved.")

# Attrs consulted (in order) when a short human-readable label is needed.
# ``display`` comes first so a node can carry a short form for the viewer while
# keeping its full text in the attribute the schema names. Nothing sets it
# automatically; it is for a record meant to be read on a screen.
_LABEL_ATTRS = ("display", "formulation", "content", "synthesis", "name", "title",
                "description", "rule", "threshold", "path")

# Priority words accepted in attrs.priority, most important first.
_PRIORITY_WORDS = {"critical": 0, "highest": 0, "high": 1, "primary": 0,
                   "medium": 2, "normal": 2, "moderate": 2, "low": 3, "lowest": 4}


def _is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "да", "selected",
                                         "primary")
    return bool(value)


def priority_rank(attrs: Dict[str, Any]) -> Tuple[float, float]:
    """Sort key for hypothesis selection: lower = more important.

    An explicit ``attrs.selected`` (the agent's own pick) always wins; otherwise
    ``attrs.priority`` is read both as a word (high/medium/low) and as a number
    (1 = most important, the spec's 1..5 scale). Unspecified sorts last but keeps
    the commit order among equals.
    """
    attrs = attrs or {}
    selected = 0 if _is_truthy(attrs.get("selected") or attrs.get("primary")) else 1
    raw = str(attrs.get("priority", "")).strip().lower()
    if raw in _PRIORITY_WORDS:
        return (selected, float(_PRIORITY_WORDS[raw]))
    try:
        return (selected, float(raw))
    except ValueError:
        return (selected, 99.0)


def _default_dir() -> str:
    """Snapshot directory from settings, tolerating a missing settings group
    (the store must stay importable/usable standalone, e.g. in unit tests)."""
    try:
        from CoScientist.config import get_settings
        return get_settings().research_graph.dir
    except Exception:  # noqa: BLE001
        return os.getenv("RESEARCH_GRAPH_DIR", "./graph_runs")


def _default_file() -> str:
    try:
        from CoScientist.config import get_settings
        return get_settings().research_graph.active_file
    except Exception:  # noqa: BLE001
        return "research_active.json"


def _short(value: Any, n: int = 200) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else str(value)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n] + "…"



# ── the record, said in words ────────────────────────────────────────────────
# The viewer is read by scientists who do not read the code. Everything below
# turns the store's internal vocabulary into the words they already use: node
# types become the thing they stand for, statuses become what happened, and
# attribute keys become field names rather than identifiers.

#: The reader of this graph is a scientist, and the one we build it for reads
#: Russian. The statuses below are translated for the same reason: a card
#: saying "Гипотеза · being tested" is harder to read than either language on
#: its own. Both tables are display-only — nothing matches on these strings.
_KIND_WORDS = {
    "ResearchQuestion": "Вопрос", "Hypothesis": "Гипотеза",
    "VerificationMethod": "Метод проверки",
    "ConfirmationCriteria": "Критерий подтверждения",
    "Evidence": "Свидетельство", "Conclusion": "Вывод", "Constraint": "Ограничение",
    "Tool": "Инструмент", "Resource": "Бюджет", "EmpiricalBase": "Источник данных",
    "CodeArtifact": "Код", "GeneratedData": "Полученные данные", "Report": "Отчёт",
    "Publication": "Публикация", "Spec": "Спецификация",
    "CostModel": "Стоимость", "EfficiencyMetric": "Эффективность",
    "EfficiencyJustification": "Обоснование эффективности",
    # The plan track, beside the record rather than part of it.
    "PlanStep": "Шаг плана", "ExperimentTask": "Задача эксперимента",
    # Derived cards, projected rather than written.
    "Framing": "Техническое задание", "Outcome": "Итог",
}

_STATUS_WORDS = {
    "skipped": "пропущена",
    "open": "открыт", "decomposed": "разбит на части", "closed": "закрыт",
    "formulated": "предложена", "under_verification": "проверяется",
    "confirmed": "подтверждена", "refuted": "опровергнута",
    "inconclusive": "проверена — без ответа", "postponed": "отложена",
    # Evidence exists only after something has been observed or recorded. Its
    # scientific assessment is a separate concern: the viewer always says the
    # evidence was received, while validated/rejected remain in history and in
    # the assessment field of the detail panel.
    "obtained": "получено", "validated": "проверено", "rejected": "отклонено",
    "planned": "запланирован", "running": "выполняется", "done": "выполнено",
    "failed": "не удался", "not_met": "ещё не выполнен", "met": "выполнен",
    # A method's own words — see NODE_TYPES["VerificationMethod"] for why it
    # has a vocabulary of its own rather than a task's.
    "proposed": "предложен", "used": "использован",
    "not_used": "не использован",
    "available": "доступен", "exhausted": "исчерпан",
    "needs_adaptation": "нужна доработка", "being_created": "создаётся",
    "creation_failed": "создать не удалось",
    "draft": "черновик", "approved": "утверждён", "created": "записан",
    "active": "действует", "derived": "сводка",
    # A plan step's own states, from the task tracker. `in_progress` says the
    # same thing an experiment task's `running` says, and is worded the same
    # way: the reader is looking for what is happening NOW, and two words for
    # one fact made them hunt for a difference that is not there.
    "todo": "не начат", "in_progress": "выполняется", "blocked": "заблокирован",
}

_FIELD_WORDS = {
    "formulation": "Statement", "rationale": "Why", "priority": "Priority",
    "content": "Finding", "subtype": "Kind", "reliability": "Confidence",
    "source_ref": "Source", "synthesis": "Conclusion",
    "validity_bounds": "Limits of validity", "new_question": "Opens next",
    "procedure": "Procedure", "limits": "Limits", "threshold": "Threshold",
    "metric": "Metric", "value": "Value", "name": "Name", "location": "Where",
    "tool_type": "Type", "base_type": "Type", "volume": "Size",
    "resource_type": "Resource", "remaining": "Remaining", "limit": "Total",
    "domain": "Field", "gap": "Knowledge gap", "not_tested_reason": "Why untested",
    "description": "Description", "method_type": "Type", "path": "File",
    # Why something did not work out. `postponed_reason` is written by the store
    # itself when it moves a surplus hypothesis to the backlog, and used to be
    # hidden — so a card said "отложена" and never said what it was waiting for.
    "failure_reason": "Why it failed", "postponed_reason": "Why postponed",
    "inconclusive_reason": "What stayed unsettled",
    "plan_task_id": "Plan step", "assignee": "Assigned to",
}

#: Never shown: bookkeeping the reader has no use for.
_HIDDEN_FIELDS = {"_provenance", "selected", "display",
                  "contributors", "contributors_more",
                  "report_artifact_id", "report_stamp", "report_lang"}

#: Never shipped to an AGENT either. `get_context_slice` renders each node as
#: 240 characters of its attrs dict, which is the whole of what a worker is
#: told about its neighbours; a participation list would eat that window and
#: hide the node's actual content behind the names of who touched it.
_AGENT_HIDDEN = {"contributors", "contributors_more", "_provenance",
                 "report_artifact_id", "report_stamp", "report_lang"}


#: One recorded act of participation. `assignee` is the only basis that is an
#: INTENTION — the plan named someone — and every other is something the system
#: watched happen. The panel is required to draw that difference, which is the
#: whole reason a basis is stored instead of a bare list of names.
CONTRIB_BASES = ("commit", "status", "delegation", "provenance",
                 "work_order", "route", "assignee")
#: Observed, as opposed to merely planned.
CONTRIB_OBSERVED = frozenset(CONTRIB_BASES) - {"assignee"}
_MAX_CONTRIBUTORS = 40

#: Sources that write on someone's behalf rather than taking part. A mirror
#: writes every node of its kind in the graph, so crediting it would say the
#: same non-agent participated in everything — which is no information at all.
#: `ExperimentModule` and `ValidatorAgent` are NOT here: they are real actors.
_MACHINE_SOURCES = frozenset({
    "plan-mirror", "experiment-plan-mirror", "graph-maintainer",
    "report-writer", "node-report", "paper-linker",
})


def _strip_reserved(attrs: Dict[str, Any], where: str, warnings: List[str],
                    allow: bool = False) -> Dict[str, Any]:
    """Drop the attributes the graph writes about a node, not the ones it claims.

    Dropped rather than refused: the rest of the commit is worth more than the
    key, and an agent that echoes back a `contributors` list it saw in its
    context slice should not lose its evidence over it. `schema.RESERVED_ATTRS`
    says why each one is reserved.
    """
    if allow:
        return attrs
    taken = [k for k in attrs if k in schema.RESERVED_ATTRS]
    if not taken:
        return attrs
    warnings.append(
        f"{where}: {', '.join(sorted(taken))} "
        f"{'is' if len(taken) == 1 else 'are'} written by the graph itself and "
        f"cannot be set from a commit — the rest of this entry was applied.")
    return {k: v for k, v in attrs.items() if k not in schema.RESERVED_ATTRS}


#: How many instruments a method's card names. The one authority: the card
#: renders the whole string it is sent, so a second, smaller cap in the page
#: meant the tail was polled to the browser every 1.5 s and drawn nowhere. The
#: Tools a method `uses` are folded onto it as chips as well, and the panel
#: lists those in full — so nothing here is the only place a name appears.
_MAX_INSTRUMENTS = 8

#: What a method's status used to be called, and what it says now. A study
#: written before the vocabulary changed does not hold unknown statuses — it
#: holds statuses worded as a task's, and this is the reading of them.
#: Re-spoken on LOAD rather than by a migration script: a graph arrives from
#: three directions (the live session file, an imported bundle, an archived
#: study) and the loader is the one place all three pass through. Without it
#: every stored method freezes: `validate_transition` reads `from` off the
#: node, and no pair starting at `planned` exists any more.
_LEGACY_METHOD_STATUS = {"planned": "proposed", "running": "proposed",
                         "done": "used", "failed": "not_used"}


def _respeak_method_status(node: Dict[str, Any]) -> None:
    """Re-word one stored method, and its trail, in today's vocabulary."""
    if not isinstance(node, dict) or node.get("type") != "VerificationMethod":
        return
    node["status"] = _LEGACY_METHOD_STATUS.get(node.get("status"),
                                               node.get("status"))
    history = node.get("status_history")
    if not isinstance(history, list):
        return
    trail: List[Dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            trail.append(entry)
            continue
        moved = dict(entry)
        for side in ("from", "to"):
            if moved.get(side) in _LEGACY_METHOD_STATUS:
                moved[side] = _LEGACY_METHOD_STATUS[moved[side]]
        trail.append(moved)
    # Every entry kept, including the ones that now read as a move from a
    # status to itself (`planned → running` becomes `proposed → proposed`).
    # Dropping those was tidier to look at and wrong: the entry carries its own
    # SOURCE and its own REASON, `_save` writes the shortened trail straight
    # back over the file, and there is no second copy — so across the studies
    # on disk it would have deleted 46 transitions, 41 of them with a written
    # reason, and four that were the only record that a particular agent had
    # touched the method at all. A row that says an agent noted something
    # without moving the state is still the record of what it did.
    node["status_history"] = trail


def _named_instruments(value: Any) -> List[str]:
    """Instrument names out of whatever an author wrote them as.

    The same list reaches the graph as a list, as a comma string and as a
    semicolon string depending on who wrote it, and a method that names its
    instruments in the shape the other writer uses is not a method that names
    none.
    """
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (list, tuple, set)):
        parts: List[str] = []
        for item in value:
            parts.extend(_named_instruments(item))
        return parts
    if isinstance(value, dict):
        return _mcp_instruments([value])
    return [p.strip() for p in re.split(r"[;,\n]", str(value)) if p.strip()]


def _mcp_instruments(servers: Any) -> List[str]:
    """`[{"name": "tox", "tools": ["predict_ld50"]}]` → `["tox:predict_ld50"]`.

    This is where the approved experiment plan already records which tools a
    method will call, and until now the card could not read it: the reader saw
    a method with no instrument beside a run that had called two.
    """
    out: List[str] = []
    if not isinstance(servers, (list, tuple)):
        return out
    for server in servers:
        if not isinstance(server, dict):
            out.extend(_named_instruments(server))
            continue
        name = str(server.get("name") or server.get("server") or "").strip()
        tools = [str(t).strip() for t in (server.get("tools") or [])
                 if str(t).strip()]
        if name and tools:
            out.extend(f"{name}:{tool}" for tool in tools)
        elif name or tools:
            out.extend([name] if name else tools)
    return out


def _one_name_each(names: List[str]) -> List[str]:
    """One entry per instrument, whichever way each source named it.

    The same tool arrives twice with two spellings — `uses` gives the Tool
    node's bare name (`predict_ld50`) and the approved plan gives it qualified
    (`heracleum-tox:predict_ld50`) — and a card listing both says the method
    ran two instruments. The qualified form wins: it says which server, which
    is the part a reader cannot reconstruct. Two servers offering a tool of the
    same name stay two entries, because neither of them is bare.
    """
    out: List[str] = []
    qualified = {n.split(":")[-1].strip().lower() for n in names
                 if n and ":" in n}
    taken: set = set()
    for name in names:
        clean = (name or "").strip()
        key = clean.lower()
        if not clean or key in taken:
            continue
        # A bare name the qualified list already accounts for is the same tool.
        if ":" not in clean and key in qualified:
            continue
        taken.add(key)
        out.append(clean)
    return out


def _headline(kind: str, attrs: Dict[str, Any]) -> str:
    """One line saying what this node is, in the reader's own words.

    A budget used to be rendered as its raw record — `{"resource_type":
    "GPU-hours", "remaining": 50, "limit": 50}` — which is the storage format
    and not a sentence. Each type gets the phrasing that suits it, and only
    something genuinely unnameable falls back to the record.
    """
    def text(*keys: str) -> str:
        for key in keys:
            value = attrs.get(key)
            if value not in (None, "", [], {}):
                return str(value).strip()
        return ""

    if kind == "Report":
        # The card is a title, the panel is the document. Without this the whole
        # write-up becomes the label and the card is unreadable.
        title = text("title", "name")
        if title:
            return _short(title, 120)
        for line in str(attrs.get("content") or "").splitlines():
            if line.strip().startswith("#"):
                return _short(line.lstrip("# ").strip(), 120)
            if line.strip():
                return _short(line.strip(), 120)
        return "Результаты исследования"
    if kind == "Resource":
        left, total = attrs.get("remaining"), attrs.get("limit")
        unit = text("resource_type")
        if unit and left is not None and total is not None:
            return f"{unit}: {left} of {total} left"
        if unit:
            return unit
    elif kind == "EmpiricalBase":
        size = text("volume")
        base = text("name", "description", "base_type")
        where = text("source_ref")
        # A node whose only content is `base_type` used to render as the bare
        # word "dataset", which tells a reader nothing about which dataset.
        detail = size or where
        if base:
            return f"{base} — {detail}" if detail else base
    elif kind == "ConfirmationCriteria":
        # Agents name this field whatever the prompt made natural, so the list
        # is wide on purpose; anything it misses still reaches the reader
        # through the fallback below.
        stated = text("threshold", "thresholds", "criteria", "content",
                      "confirmation_criteria", "confirm_refute_rule",
                      "success_metric", "rule", "description")
        extra = text("confirmations_needed", "reproducibility")
        if stated and extra:
            return f"{stated}; {extra}"
        if stated or extra:
            return stated or extra
    elif kind == "Evidence":
        found = text("content", "description", "finding", "summary")
        if found:
            return found
        metric, value = text("metric"), text("value")
        if metric and value:
            return f"{metric}: {value}"
    elif kind == "Tool":
        named = text("name", "description")
        if named:
            return named
    elif kind == "VerificationMethod":
        # `name` before `procedure`, and both before `method_type`: the
        # experiment module writes a title and no description, so with
        # `method_type` standing second every method it published drew as the
        # single word "computational" — twenty-three cards on one canvas, all
        # with the same headline, none of them saying what it was a method OF.
        described = text("description", "name", "label", "experiment_question",
                         "procedure", "method_type")
        if described:
            return described
    elif kind == "Conclusion":
        drawn = text("synthesis", "content", "description")
        if drawn:
            return drawn

    said = text("formulation", "content", "synthesis", "name", "title",
                "description", "rule", "threshold", "path")
    if said:
        return said
    # Last resort, and better than a stock word: a node whose content sits
    # under a key nobody anticipated still says what it says. The type name is
    # already on the card, so a headline repeating it says nothing at all —
    # which is how "Acceptance criteria" came to stand in for the criteria.
    readable = [f"{_FIELD_WORDS.get(k, k)}: {v}" for k, v in attrs.items()
                if k not in _HIDDEN_FIELDS and v not in (None, "", [], {})]
    return "; ".join(readable)


#: Keys a headline already speaks for, per type; repeating them underneath is
#: the same sentence twice.
_CONSUMED_BY_HEADLINE = {
    # The body is the card's whole point and the panel already shows it; listing
    # it again under details would print the report twice.
    "Report": {"content"},
    "Resource": {"resource_type", "remaining", "limit"},
    "EmpiricalBase": {"base_type", "volume", "name", "description"},
    "ConfirmationCriteria": {"threshold", "thresholds", "criteria", "content",
                             "confirmation_criteria", "confirm_refute_rule",
                             "success_metric", "rule", "description"},
    "Tool": {"name", "description"},
    # Whichever of the two supplied the headline, the other is the same
    # sentence: an agent writes `description` and the experiment module writes
    # `name`, never both.
    "VerificationMethod": {"description", "name", "label"},
    "Conclusion": {"synthesis", "content", "description"},
    "Evidence": {"content", "description", "finding", "summary"},
}


def _fields(attrs: Dict[str, Any], headline: str,
            kind: str = "") -> Dict[str, str]:
    """The node's attributes under names a reader recognises.

    Keys are the attribute CODES, never display words: the page localises them
    through `graph.field.*`, the same way it already localises gap codes. Baking
    the English word in here meant a Russian panel captioned "Procedure" and an
    English one captioned nothing else — the server cannot know the reader.
    Whatever the headline already says is dropped rather than repeated underneath.
    """
    out: Dict[str, str] = {}
    spoken = _CONSUMED_BY_HEADLINE.get(kind, set())
    for key, value in attrs.items():
        if key in _HIDDEN_FIELDS or key in spoken or value in (None, "", [], {}):
            continue
        rendered = _short(value, 600) if not isinstance(value, str) else value
        if rendered.strip() and rendered.strip() == headline.strip():
            continue
        out[key] = rendered
    return out


# ── projection: from the record to the drawing ───────────────────────────────
# The store keeps the whole engineering record — datasets, code, tools, budget,
# cost models — and the canvas is read as a scientific narrative, in which those
# outnumber the findings. They are NOT dropped: each is folded into the card it
# belongs to, as a chip on the method that used it or an attachment on the
# evidence it produced, and the panel still shows it in full.
#
# The viewer used to do this filtering on the client, by name, and threw away
# every edge that touched a hidden node. In a real session the entire structure
# of the graph hung off GeneratedData, so that was 100% of the edges: the canvas
# drew eleven cards and not one line between them. Folding belongs here, where
# the structure is known and an edge can be re-pointed at the card that absorbed
# its endpoint instead of being discarded.

#: Drawn as cards of their own: the scientific record, plus the PLAN beside it.
#: A PlanStep is not part of the record — it is the intention the record was
#: made against — so the viewer draws it in its own column rather than in the
#: band flow, keyed on `track: "plan"`. It still has to be projected, or the
#: reader cannot see what was intended and what came of it.
_STORY_TYPES = ("ResearchQuestion", "Hypothesis", "VerificationMethod",
                "Evidence", "Conclusion", "Framing", "Outcome", "PlanStep",
                "ExperimentTask", "Report")
#: Products of one finding — they belong to whatever they were derived from.
#: `Report` is NOT among them: folded, the write-up would have become the label
#: of an attachment chip on the Outcome card, which is where an 18 KB document
#: goes to be unreadable. It gets a card, and the panel renders its markdown.
_ARTIFACT_FOLD_TYPES = ("CodeArtifact", "GeneratedData", "Spec",
                        "EfficiencyJustification", "Publication")
#: The framing a study starts from: the context star.
_FRAME_FOLD_TYPES = ("Constraint", "Resource", "EmpiricalBase", "CostModel",
                     "EfficiencyMetric")

#: The two derived cards. Literal ids, because `_next_id` can only ever mint
#: `<PREFIX><digits>` — so these can never collide with a stored node.
#:
#: They exist in the PROJECTION only. Keep them out of `overview()`, `_ids_hint`
#: and `get_context_slice`: those reach an agent's prompt, and an agent that saw
#: one would reference a node the store does not have and lose its whole commit.
FRAME_ID = "FRAME"
OUTCOME_ID = "OUTCOME"
_DERIVED_IDS = (FRAME_ID, OUTCOME_ID)

#: One column per epistemic layer: the story reads left to right.
_LEVEL = {"Framing": 0, "ResearchQuestion": 0, "Hypothesis": 1,
          "VerificationMethod": 2, "Evidence": 3, "Conclusion": 4, "Outcome": 5,
          # Off the argument's ladder entirely: a step is an intention, so it
          # gets the level of the work it asks for and is drawn beside the band.
          "PlanStep": 2, "ExperimentTask": 2}

#: Which lane of the plan track a node belongs in, coarsest first. Two
#: intentions at different grains must not share a column: stacked together a
#: reader cannot tell the step the study planned from the task the experiment
#: module designed under it, which is the distinction the two types exist for.
_PLAN_LANE = {"PlanStep": 0, "ExperimentTask": 1}

#: The stages a study moves through, in the order a reader meets them. The
#: viewer draws one band per stage, top to bottom, and puts a card in the band
#: of the stage that produced it.
#:
#: A stage is NOT the epistemic level. The same Evidence type belongs to the
#: reading stage when it came out of a paper and to the experiment stage when a
#: run produced it; the search that gathered the papers belongs beside its own
#: findings rather than beside the experiments.
#:
#: The execution trace has its own unrelated band called "experiment"
#: (graph/projection.py) with a different vocabulary — the two never meet.
STAGES = ("framing", "literature", "hypotheses", "experiment", "report")

#: Every drawn type must stay in this table: `_stages_of` skips a type that is
#: missing from it (see below), and `_project_nodes` falls back to it, so
#: deleting an entry does not move a card — it silently drops the card into the
#: default band and makes the per-type rules underneath unreachable.
_STAGE_BY_TYPE = {
    "Framing": "framing", "ResearchQuestion": "framing",
    "PlanStep": "experiment",
    # The claim and the bar written for it; what tests the claim is the
    # experiment beside it, not the claim itself.
    "Hypothesis": "hypotheses", "ConfirmationCriteria": "hypotheses",
    "VerificationMethod": "experiment", "Evidence": "experiment",
    "Conclusion": "report", "Outcome": "report", "Report": "report",
}
#: Subtypes and method types that mean "read", not "run". Whole tokens, never
#: substrings: the schema allows Evidence subtypes literature / experimental /
#: computational / expert / meta, and a substring test would read the
#: "metabolomics" of a chemistry study as a meta-analysis.
_READING_SUBTYPES = {"literature", "meta", "meta_analysis", "meta-analysis",
                     "review", "literature_review", "literature review",
                     "publication", "bibliography", "desk_research"}
#: The last resort for a method nobody typed and that has produced nothing yet.
#: `method_type` is free text — the schema documents computational / laboratory
#: / analytical / statistical / expert and validates none of them — and the plan
#: mirror writes its step as prose, so "collect the literature" exists nowhere
#: but the procedure. Word-ish boundaries, and only stems that cannot mean
#: anything else: "обзор" alone is what a dataset overview is called too.
_READING_TEXT = re.compile(
    r"литератур|публикац|библиограф"
    r"|\bliterature\b|\bpublications?\b|\bbibliograph\w*"
    r"|\bpapers\b|\blit[_\s-]?review\b",
    re.IGNORECASE)
#: The agents whose whole job is reading. A method one of them opened is a
#: review even before it has produced anything to judge it by. The dataset
#: collector belongs here too: assembling a table out of ChEMBL, PubChem or
#: HuggingFace is desk gathering, not a measurement.
_READING_AGENTS = {"ResearchAgent", "MedicalAgent", "DatasetCollectorAgent"}
#: A GATHERING verb. The literature pattern names the object — papers,
#: publications, bibliography — and the object alone does not say what a step
#: does with it: "Кластеризация литературных SMILES" clusters molecules that
#: happen to have come out of the literature, and it is a run. So a step counts
#: as reading only when it says it is GATHERING something bibliographic.
#: Deliberately excludes download verbs: pulling a prepared dataset off S3 is
#: not a literature search.
_GATHERING_TEXT = re.compile(
    r"\bсбор\w*|собра\w*|собер\w*|поиск\w*|\bнайти\b|найд\w*"
    r"|извлеч\w*|подобра\w*|\bобзор\w*"
    r"|\bcollect\w*|\bgather\w*|\bsearch\w*|\bretriev\w*"
    r"|\breview\b|\bsurvey\b",
    re.IGNORECASE)
#: The rest of the roster, by what a step assigned to it is a step OF. Only the
#: three remaining non-experiment roles are named: everything else in the
#: roster — every executor, coder, tool pipeline and FEDOT agent, and whatever
#: the roster grows next — RUNS something, and running something is the
#: experiment band. So a new agent lands in the right band without being added
#: here, which is the direction the roster actually grows; the cost is that a
#: new READING agent would have to be added, and that is rarer.
_FRAMING_AGENTS = {"ContextInitAgent"}
_HYPOTHESIS_AGENTS = {"HypothesesAgent"}
_REPORTING_AGENTS = {"ResultAggregatorAgent"}

#: The bar itself, as opposed to the prose around it. Once a measurement is
#: aimed at a criterion these stop being editable: a threshold that follows the
#: result is a result with extra steps.
#: What makes two nodes of one type the SAME node to whoever reads the graph:
#: the free-text field the card is built from, tried in order. This is not
#: `_headline` — that one falls back to whatever key the agent invented, which
#: is right for DISPLAY and wrong for identity. Enum-ish fields are absent on
#: purpose: two methods both declaring `method_type: computational` are not one
#: method, and «dataset» is not the name of a dataset.
#:
#: A type that is missing here, or whose fields the draft leaves empty, has no
#: identity the store can judge and is simply created.
_IDENTITY_ATTRS: Dict[str, Tuple[str, ...]] = {
    "ResearchQuestion": ("formulation",),
    "Hypothesis": ("formulation",),
    "PlanStep": ("title",),
    # The plan's own id for the task. An id, not a sentence — two drafts under
    # one EXP id are one task however the title was reworded that turn. No
    # fallback to the title: two tasks may legitimately be titled the same.
    "ExperimentTask": ("experiment_task_id",),
}

#: Types deliberately NOT above, and why — this list is the rule, not an
#: oversight, and a type joins it only with a reason of the same kind.
#:
#: A sentence is an identity only where a node IS its sentence. It is not one
#: where the node belongs to something else:
#:   * `ConfirmationCriteria` — a bar («p < 0.05») says nothing about WHICH
#:     hypothesis it is the bar for. Keyed on the threshold alone, a criterion
#:     written for H2 was swallowed by H1's and both `formulated_for` edges
#:     landed on one node, so meeting the bar for one claim met it for the
#:     other.
#:   * `Conclusion` — the judge's verdict sentence repeats verbatim
#:     («Данных недостаточно для однозначного вывода.»), and the second
#:     hypothesis judged would have overwritten the first one's card.
#:   * `Evidence` — two measurements can read alike and still be two
#:     measurements, each produced by a different method.
#: nor where the text is a name that is only unique in a context the store
#: cannot see:
#:   * `Tool` — «search» is an ordinary MCP tool name, and two servers offering
#:     one collapsed into a single card whose `location` was the wrong server.
#:   * `EmpiricalBase`, `Constraint` — two rows of a confirmed research frame
#:     may carry the same text under different headings, and the frame means
#:     both.
#:   * `VerificationMethod` — two methods can share a one-line description and
#:     differ entirely in `procedure`.
#: The remaining types (CodeArtifact, GeneratedData, Report, …) are left out
#: for want of a measured case: a duplicate there has never been reported, and
#: a rule that has not been needed is a rule that has not been tested.


def _name_every_alias(echo: Dict[str, Any], aliases: Any) -> None:
    """Say which refs ended up on this node, when more than one did.

    Two drafts of one node — the same sentence written twice, or a plan that
    names one tool from two servers — are recorded once, and the echo used to
    name only the ref written on the draft that survived. A caller rebuilding
    a ref->id map from the answer then lost the other ref and, with it,
    whatever it was bookkeeping under it. `ref` stays as it was so nothing
    reading it has to change.
    """
    named = sorted(a for a in (aliases or ()) if a)
    if len(named) > 1:
        echo["refs"] = named


def _identity_text(ntype: str, attrs: Dict[str, Any]) -> str:
    """The words that make this node itself, or "" when it has none.

    Compared as the store will HOLD it, not as the caller wrote it: a long
    attribute is capped by `_truncate_attrs` on the way in, so a raw draft and
    its own stored copy are different strings. Keyed on the raw text, a
    formulation over the cap could never match its own twin and every resend
    made another node — precisely the case this is here to stop.
    """
    for key in _IDENTITY_ATTRS.get(ntype, ()):
        raw = str(attrs.get(key) or "")
        if not raw.strip():
            continue
        if len(raw) > _ATTR_CHAR_CAP:
            raw = raw[:_ATTR_CHAR_CAP] + "…[truncated]"
        return " ".join(raw.split()).casefold()
    return ""


_BAR_ATTRS = frozenset({"threshold", "confirmations_needed", "reproducibility"})
#: Edges by which a piece of Evidence is attached to a hypothesis. The neutral
#: `relates_to` counts: the store writes it itself when a worker records a
#: finding under a focus, so by the time one exists the branch has stopped being
#: drafted and started being gathered for — which is the point past which its
#: bar may no longer follow the numbers.
_POLARITY_EDGES = ("supports", "refutes", "refines", "relates_to")


#: A step whose product is the write-up rather than a measurement. Deliberately
#: narrow: "report" appears inside plenty of experimental steps ("report the
#: medians"), so the pattern asks for the deliverable, not the verb.
_REPORTING_TEXT = re.compile(
    r"\bитогов\w*\s+отч|\bфинальн\w*\s+отч"
    r"|\bсобрать\s+отч|\bоформ\w*\s+отч"
    r"|\bfinal\s+report\b|\bwrite\s+(?:up|the\s+report)\b"
    r"|\bpublication\b",
    re.IGNORECASE)

#: Attributes that answer "why did it end like that", best first.
_REASON_ATTRS = ("failure_reason", "inconclusive_reason", "not_tested_reason",
                 "postponed_reason")
#: Outcomes a reader will ask "why" about, and deserves an answer to.
_UNRESOLVED_STATUSES = {"failed", "refuted", "inconclusive", "rejected",
                        "creation_failed", "postponed", "not_used"}
#: Verdict on the superseded hypothesis -> how the next one came about.
_ORIGIN_BY_VERDICT = {"refuted": "modified", "confirmed": "refined",
                      "inconclusive": "retried"}


def _id_order(node_id: str) -> Tuple[str, int]:
    m = re.match(r"^([A-Z]+)(\d+)$", str(node_id))
    return (m.group(1), int(m.group(2))) if m else (str(node_id), 0)


def _why(attrs: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    """Why this node ended where it did, in one line.

    The answer was recorded all along — the store writes a `reason` into
    status_history on every transition — and never left the store, because
    to_view projected the current status and dropped the history. So a failed
    method drew as "не удался" and nothing else, which is the one question a
    reader of a failed step actually has.
    """
    last = (history or [])[-1] if history else {}
    for candidate in (last.get("reason"), *(attrs.get(k) for k in _REASON_ATTRS)):
        if str(candidate or "").strip():
            return _short(candidate, 600)
    return ""


def _href(kind: str, attrs: Dict[str, Any], scope: Optional[Tuple[str, str]] = None) -> str:
    """A link the chip can actually open, or "" when there is none.

    Returning "" is the point. This used to hand back whatever string the attr
    held, so a ``GeneratedData`` node whose ``path`` was
    ``D:\\projects26\\...\\clusters.json`` rendered as an anchor that navigates
    nowhere — in one real session, twelve of twelve attachments looked like
    that. An artifact is openable when it was mirrored (``artifact_id``) or
    lives in S3; a bare local path is a fact about this machine, and the panel
    shows it as a field instead.
    """
    from CoScientist.utils.report_links import resolve_ref

    # `doi` and `pmcid` sit before `source_ref`: they are the resolved citation
    # the graph wrote, and `source_ref` is the prose the agent wrote — which may
    # name three sources and open none of them.
    keys = ("session_artifact_id", "location", "path", "uri") if kind == "Tool" else (
        "session_artifact_id", "path", "uri", "doi", "pmcid", "source_ref",
        "location")
    for key in keys:
        value = attrs.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        # `session_artifact_id` holds a bare id — and only ours. The experiment
        # runtime's `artifact_id` is a different namespace entirely and is not
        # consulted here; treating it as ours pointed a chip at a file that was
        # never stored under that name.
        candidate = (
            f"cos-artifact:{value.strip()}"
            if key == "session_artifact_id" else value.strip()
        )
        resolved = resolve_ref(candidate, scope)
        if resolved:
            return resolved
    return ""


def _adjacency(raw_edges: List[Dict[str, Any]]):
    out: Dict[str, List[Tuple[str, str]]] = {}
    inc: Dict[str, List[Tuple[str, str]]] = {}
    for e in raw_edges:
        out.setdefault(e["src"], []).append((e["type"], e["dst"]))
        inc.setdefault(e["dst"], []).append((e["type"], e["src"]))
    return out, inc


def _plan_step_view_status(nid: str, status: str,
                           raw_nodes: Dict[str, Dict[str, Any]],
                           raw_edges: List[Dict[str, Any]]) -> str:
    """Repair a stale running step from the terminal tasks underneath it.

    Result mirroring now settles the stored parent immediately, but snapshots
    written before that fix can still be opened.  Projection is the compatibility
    boundary: old sessions stop pulsing without rewriting their research record.
    """
    if status != "in_progress":
        return status
    children = [raw_nodes.get(e.get("src"), {}) for e in raw_edges
                if e.get("type") == "elaborates" and e.get("dst") == nid]
    child_states = [str(child.get("status") or "") for child in children
                    if child.get("type") == "ExperimentTask"]
    states = [state for state in child_states if state != "skipped"]
    if child_states and not states and all(
            state == "skipped" for state in child_states):
        return "blocked"
    if states and all(state == "done" for state in states):
        return "done"
    if states and not any(state == "running" for state in states) \
            and any(state == "failed" for state in states):
        return "blocked"
    return status


def _is_reading(value: Any) -> bool:
    """Whether a declared subtype / method_type names reading rather than running."""
    return str(value or "").strip().lower().replace(" ", "_") in {
        token.replace(" ", "_") for token in _READING_SUBTYPES}


def _reads_like_review(attrs: Dict[str, Any]) -> bool:
    """Whether what a method SAYS it does is gather sources.

    Only reached when nothing stronger is on record: no evidence produced and no
    method_type written. The plan mirror's steps are prose, so this is the one
    place the intent of "collect the literature" survives.
    """
    text = " ".join(str(attrs.get(key) or "") for key in
                    ("method_type", "procedure", "description", "name", "subtype"))
    return bool(_READING_TEXT.search(text))


def _step_stage(attrs: Dict[str, Any]) -> str:
    """Which stage a plan step ASKS FOR, so the plan column reads alongside the
    research it planned.

    An assignee with ONE job decides it: the planner picks the roster entry, it
    is the one part of a step that is not prose, and `create_plan` refuses a
    task without one.

    An assignee whose job is "whatever this turns out to need" decides nothing.
    TaskExecutorAgent describes itself as routing each task to the path that can
    deliver it, and the planner hands it literature collection as readily as a
    QSAR run — four of the five steps of one real study were its, across three
    different stages. So for a generic executor, and for a step nobody assigned,
    the WORDING decides.

    And only the TITLE. What a step IS is in its title; what it TOUCHES is in
    its description. Reading descriptions put a Tanimoto clustering run in the
    literature band, because its description said "merge with the literature
    SMILES from TASK-2", and a QSAR LD50 prediction in the report band, because
    its description said the table goes on to the final report. Both then drew a
    step under the wrong stage heading, which is worse than drawing it plainly:
    the column's only job is to say which stage the plan asked for.
    """
    assignee = str(attrs.get("assignee") or "").strip()
    if assignee in _HYPOTHESIS_AGENTS:
        return "hypotheses"
    if assignee in _READING_AGENTS:
        return "literature"
    if assignee in _REPORTING_AGENTS:
        return "report"
    if assignee in _FRAMING_AGENTS:
        return "framing"
    title = str(attrs.get("title") or "")
    # Reporting first: "собрать итоговый отчёт по литературе" is a write-up.
    if _REPORTING_TEXT.search(title):
        return "report"
    if _READING_TEXT.search(title) and _GATHERING_TEXT.search(title):
        return "literature"
    return "experiment"


def _stages_of(raw_nodes: Dict[str, Dict[str, Any]],
               raw_edges: List[Dict[str, Any]]) -> Dict[str, str]:
    """Which stage band each drawn node belongs in.

    Evidence is placed by its own subtype, which the schema requires. A method
    is placed by what it actually produced: a search that returned nothing but
    papers belongs with the literature whatever its free-text `method_type`
    says, and one that has produced nothing yet is read off its declared type or
    the agent that opened it. Computed for the whole graph at once because a
    method's stage depends on its evidence.
    """
    out, _ = _adjacency(raw_edges)
    stage: Dict[str, str] = {}
    for nid, data in raw_nodes.items():
        kind = data.get("type", "")
        if kind not in _STAGE_BY_TYPE:
            continue
        attrs = data.get("attrs") or {}
        if kind == "Evidence":
            stage[nid] = ("literature" if _is_reading(attrs.get("subtype"))
                          else "experiment")
        elif kind == "VerificationMethod":
            produced = [dst for etype, dst in out.get(nid, [])
                        if etype == "produces"
                        and (raw_nodes.get(dst) or {}).get("type") == "Evidence"]
            if produced:
                # All of it, not any of it: a method that also ran something
                # belongs with the experiments it ran.
                stage[nid] = "literature" if all(
                    _is_reading(((raw_nodes[dst].get("attrs") or {}).get("subtype")))
                    for dst in produced) else "experiment"
            elif (_is_reading(attrs.get("method_type") or attrs.get("subtype"))
                  or data.get("source") in _READING_AGENTS
                  or _reads_like_review(attrs)):
                stage[nid] = "literature"
            else:
                stage[nid] = "experiment"
        elif kind == "PlanStep":
            stage[nid] = _step_stage(attrs)
        elif kind == "ExperimentTask":
            # Always the experiment band: the module plans experiments. Its
            # research/medical routes gather for one, and the card says which
            # route it took, so the band stays the truthful one.
            stage[nid] = "experiment"
        else:
            stage[nid] = _STAGE_BY_TYPE[kind]
    return stage


def _fold_plan(raw_nodes: Dict[str, Dict[str, Any]],
               raw_edges: List[Dict[str, Any]]):
    """Decide which nodes are not cards, and which card carries each of them.

    Returns ``(host, carried)``. ``host`` maps a folded id to the ONE card that
    absorbs its edges; ``carried`` maps a card id to ``[(role, folded_id)]``.

    A node may be SHOWN on several cards — a tool used by three methods is a
    chip on all three — but it is absorbed by exactly one. Re-pointing its edges
    at every host instead would multiply them and turn the picture into a mesh.
    """
    out, inc = _adjacency(raw_edges)
    types = {n: d.get("type", "") for n, d in raw_nodes.items()}

    def methods_of(hypothesis: str) -> List[str]:
        return [v for t, v in out.get(hypothesis, [])
                if t == "tested_by" and types.get(v) == "VerificationMethod"]

    host: Dict[str, str] = {}
    carried: Dict[str, List[Tuple[str, str]]] = {}
    hosts_of: Dict[str, set] = {}

    def place(nid: str, hosts: List[str], role: str, fallback: str) -> None:
        hosts = sorted({h for h in hosts if types.get(h) in _STORY_TYPES},
                       key=_id_order)
        host[nid] = hosts[0] if hosts else fallback
        hosts_of[nid] = set(hosts or [fallback])
        for h in (hosts or [fallback]):
            carried.setdefault(h, []).append((role, nid))

    for nid in sorted(raw_nodes, key=_id_order):
        kind = types.get(nid, "")
        if kind in _ARTIFACT_FOLD_TYPES:
            # Whatever it was derived from carries it; a product of the study as
            # a whole (a report nobody linked) belongs to the outcome.
            #
            # Except the one artifact that exists before there is an outcome:
            # the техническое задание, written from the confirmed frame. It is
            # about the setting, so it belongs on the setting's card — and it
            # is the only way a reader can open the document at all, since the
            # panel makes a link out of an attachment and inert text out of an
            # attribute.
            hosts = [v for t, v in out.get(nid, []) if t == "derived_from"]
            fallback = OUTCOME_ID
            if kind == "Spec" and all(types.get(v) == "ResearchQuestion"
                                      for v in hosts):
                hosts, fallback = [], FRAME_ID
            place(nid, hosts, "attachment", fallback)
        elif kind == "Tool":
            hosts = [u for t, u in inc.get(nid, []) if t == "uses"]
            for t, u in inc.get(nid, []):
                if t == "requires" and types.get(u) == "Hypothesis":
                    hosts += methods_of(u)
            place(nid, hosts, "chip", FRAME_ID)
        elif kind == "ConfirmationCriteria":
            hosts: List[str] = []
            for t, v in out.get(nid, []):
                if t == "formulated_for" and types.get(v) == "Hypothesis":
                    hosts += methods_of(v)
            place(nid, hosts, "criterion", FRAME_ID)
        elif kind in _FRAME_FOLD_TYPES:
            place(nid, [], "attachment", FRAME_ID)
    return host, carried, hosts_of


def _readable_body(content: Any, scope: Optional[Tuple[str, str]] = None) -> str:
    """A Report node's markdown, with its artifact references made openable.

    Capped the way it always was — the card rides along on a poll — and the cap
    is applied AFTER resolution so a reference is never cut in half.
    """
    text = str(content or "")
    if not text:
        return ""
    try:
        from CoScientist.utils.report_links import resolve_artifact_refs

        text = resolve_artifact_refs(text, scope)
    except Exception:  # noqa: BLE001 — the document outranks one link
        pass
    return text[:_REPORT_CHAR_CAP]


#: Node kinds whose headline is supposed to NAME A FILE, and may therefore be
#: replaced by the name of the file actually stored. A ``Tool`` also carries a
#: ``session_artifact_id`` and is deliberately absent: its headline is the
#: tool's name, and swapping that for a file name would say less, not more.
_NAMED_BY_THEIR_FILE = frozenset({"CodeArtifact", "GeneratedData"})


def _stored_name(kind: str, attrs: Dict[str, Any],
                 scope: Optional[Tuple[str, str]] = None) -> str:
    """The real file name behind this node's artifact, or "" when there is none.

    Only ``session_artifact_id`` is consulted: it is the one attr that names a
    file this session actually holds, so it is the one whose name we can state
    as a fact rather than as whatever the plan hoped the file would be called.
    """
    if kind not in _NAMED_BY_THEIR_FILE:
        return ""
    aid = attrs.get("session_artifact_id")
    if not isinstance(aid, str) or not aid.strip() or not scope:
        return ""
    try:
        from CoScientist.utils.report_links import artifact_citation

        citation = artifact_citation(scope, aid.strip())
        return citation["name"] if citation else ""
    except Exception:  # noqa: BLE001 — a label is not worth a failed render
        return ""


def _folded_view(nid: str, data: Dict[str, Any],
                 scope: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
    """A folded node as it appears on the card that carries it."""
    attrs = data.get("attrs") or {}
    kind = data.get("type", "?")
    headline = _headline(kind, attrs)
    status = data.get("status", "")
    href = _href(kind, attrs, scope)
    # Label and target from ONE record, whenever there is a record to read.
    # They used to be independent lookups — `description` for the chip's text,
    # `session_artifact_id` for its href — and a live session shows what that
    # costs: a chip reading `metabolite_smiles.json` that downloads a PDF. The
    # description stays visible as a field, so nothing is hidden, but the name
    # next to a link is now the name of the file behind it.
    stored_name = _stored_name(kind, attrs, scope)
    # A preliminary specification used to inherit its whole markdown body as
    # the attachment label.  The result was a paragraph-long hyperlink beside
    # the framing card.  This display rule also fixes already stored studies:
    # no snapshot migration is required.
    label = ("Предварительный текст ТЗ" if kind == "Spec"
             else stored_name or headline)
    view_status = "obtained" if kind == "Evidence" else status
    return {
        "id": nid,
        "kind": kind.lower(),
        "type_word": _KIND_WORDS.get(kind, kind),
        "label": label,
        "href": href,
        # `label`, not `headline`: what the headline no longer says has to be
        # visible somewhere, so a description the plan wrote and the file did
        # not match reappears as a field instead of vanishing.
        "fields": _fields(attrs, label, kind),
        "status": view_status,
        "status_word": _STATUS_WORDS.get(view_status, view_status),
        "source": data.get("source", ""),
    }


def _own_artifact(nid: str, attrs: Dict[str, Any],
                  scope: Optional[Tuple[str, str]] = None) -> Optional[Dict[str, Any]]:
    """The node's OWN file, when this session holds it.

    `attachments` were only ever built from the nodes a card folds in, so an
    Evidence citing a paper the run downloaded — or a Report whose markdown is
    stored — offered no way to open it. `_href` already knew how; nothing was
    asking it about the node itself.

    Returns None when the bytes are not ours, on the rule `_href` follows: a
    link that opens nothing is worse than no link.
    """
    aid = attrs.get("session_artifact_id")
    if not isinstance(aid, str) or not aid.strip() or not scope:
        return None
    try:
        from CoScientist.utils.report_links import artifact_citation

        citation = artifact_citation(scope, aid.strip())
    except Exception:  # noqa: BLE001 — an attachment is not worth a failed render
        return None
    if not citation or not citation.get("href"):
        return None
    return {
        # Suffixed on purpose: the panel indexes attachments by id, and a folded
        # child is keyed by its own node id. A bare `nid` would collide with the
        # card that carries it.
        "id": f"{nid}#file",
        "kind": str(citation.get("kind") or "file"),
        "type_word": "",
        # The name of the file behind the link, for the reason `_folded_view`
        # gives: a row that says one thing and downloads another is a trap.
        "label": citation.get("name") or aid.strip(),
        "href": citation["href"],
        "fields": {},
        "status": "",
        "status_word": "",
        "source": "",
    }


def _node_report_body(attrs: Dict[str, Any],
                      scope: Optional[Tuple[str, str]] = None) -> str:
    """The node's write-up, read back from the store and made openable.

    References are resolved HERE rather than when the report was written, for
    the reason `_readable_body` gives about a Report: the document keeps the
    session-free form, so an imported bundle resolves its files under whatever
    scope is reading.
    """
    aid = attrs.get("report_artifact_id")
    if not isinstance(aid, str) or not aid.strip() or not scope:
        return ""
    try:
        from CoScientist.reporting import session_files

        path = session_files.resolve_path(scope, aid.strip())
        if path is None:
            return ""
        return _readable_body(path.read_text(encoding="utf-8"), scope)
    except Exception:  # noqa: BLE001 — a missing write-up is not a failed render
        return ""


def _mark_stale_reports(view: Dict[str, Any]) -> None:
    """Say, per node, whether its write-up still describes the node.

    The panel cannot work this out for itself and was never able to: the only
    stamp it ever held was the one the write returned, which is the same value
    the node carries, so its comparison was of a value with itself and could
    never come out true. A node that had moved went on presenting an account of
    what it used to be, with nothing saying so.

    Computed here, where the current facts are, and only for nodes that actually
    hold a write-up — a handful per study, against a 1.5-second poll.
    """
    reported = [n for n in (view.get("nodes") or [])
                if n.get("report") and n.get("report_stamp")]
    if not reported:
        return
    try:
        from CoScientist.reporting import node_report
    except Exception:  # noqa: BLE001 — a missing verdict is not a failed render
        return
    for node in reported:
        lang = node.get("report_lang") or "ru"
        try:
            facts = node_report.build_facts(view, node["id"], lang=lang)
            node["report_stale"] = bool(
                facts is not None
                and node_report.stamp(facts, lang) != node["report_stamp"])
        except Exception:  # noqa: BLE001
            node["report_stale"] = False


def _contributors_view(attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The participation record as the panel reads it.

    Observed rows first, then the merely planned, each group oldest first — the
    order the work happened in. One row per agent per basis; the same agent
    committing twice is one contributor, not two.
    """
    rows = [r for r in (attrs.get("contributors") or []) if isinstance(r, dict)]
    if not rows:
        return []
    ordered = sorted(
        rows, key=lambda r: (r.get("basis") not in CONTRIB_OBSERVED,
                             float(r.get("at") or 0)))
    return [{"agent": str(r.get("agent") or ""),
             "basis": str(r.get("basis") or ""),
             "observed": r.get("basis") in CONTRIB_OBSERVED,
             "exec_id": str(r.get("exec_id") or "")}
            for r in ordered if r.get("agent")]


def _history_view(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The status trail, in words, with the reason for every move."""
    return [{
        "from": h.get("from"),
        "to": h.get("to"),
        "from_word": _STATUS_WORDS.get(h.get("from") or "", h.get("from") or ""),
        "to_word": _STATUS_WORDS.get(h.get("to") or "", h.get("to") or ""),
        "source": h.get("source", ""),
        "at": h.get("at"),
        # Capped: a long study keeps a dozen transitions, and the whole history
        # rides along on a poll every 1.5 seconds.
        "reason": _short(h.get("reason") or "", 400),
    } for h in (data.get("status_history") or [])]


def _reroute(raw_edges: List[Dict[str, Any]], host: Dict[str, str],
             hosts_of: Dict[str, set], drawn: set) -> List[Dict[str, Any]]:
    """Re-point every edge at the cards that absorbed its endpoints.

    Folding is a graph homomorphism — each node maps to itself or to a card it
    is attached to — so a path A→X→B survives as A→host(X)→B and the drawing
    cannot fall apart into islands the record does not have.

    Two kinds of edge are dropped rather than re-pointed. One collapses to a
    self-loop. The other is the edge that BOUND the folded node to a card it is
    shown on: one dataset feeding two findings would otherwise be re-pointed as
    "finding A derived_from finding B" — an ancestry between siblings that the
    record never claimed. The card already shows that dataset as an attachment,
    on both of them.
    """
    direct: List[Dict[str, Any]] = []
    inferred: List[Dict[str, Any]] = []
    seen = set()
    for e in raw_edges:
        src, dst = e["src"], e["dst"]
        u, v = host.get(src, src), host.get(dst, dst)
        if u == v or u not in drawn or v not in drawn:
            continue
        if dst in hosts_of.get(src, ()) or src in hosts_of.get(dst, ()):
            continue
        key = (u, v, e["type"])
        if key in seen:
            continue
        seen.add(key)
        edge = {"src": u, "dst": v, "type": e["type"]}
        if e.get("attrs"):
            # A `supersedes` edge says WHICH verdict moved the study on; without
            # its attrs the arrow between two hypotheses is unlabelled.
            edge["attrs"] = e["attrs"]
        if (u, v) == (src, dst):
            direct.append(edge)
        else:
            edge["rerouted"] = True
            edge["via"] = src if u != src else dst
            inferred.append(edge)

    # An inferred edge earns its place only by connecting something that is not
    # connected already. The derived cards are excluded outright: they are tied
    # on by their own synthetic edge, and every context node folded into them
    # would otherwise draw its own line back to the question — the frame alone
    # would fan out to a dozen arrows saying what one already says.
    linked = {(e["src"], e["dst"]) for e in direct}
    linked |= {(v, u) for u, v in linked}
    out = list(direct)
    for edge in inferred:
        pair = (edge["src"], edge["dst"])
        if edge["src"] in _DERIVED_IDS or edge["dst"] in _DERIVED_IDS:
            continue
        if pair in linked:
            continue
        linked |= {pair, (pair[1], pair[0])}
        out.append(edge)
    return out


def _origin_of(nid: str, raw_edges: List[Dict[str, Any]]) -> Dict[str, str]:
    """Where a hypothesis came from: the question, or the one it replaced."""
    for e in raw_edges:
        if e["type"] == "supersedes" and e["dst"] == nid:
            verdict = str((e.get("attrs") or {}).get("verdict") or "")
            return {"code": _ORIGIN_BY_VERDICT.get(verdict, "other"),
                    "reason": _short((e.get("attrs") or {}).get("reason") or "", 160),
                    "from": e["src"]}
    return {"code": "question", "reason": "", "from": ""}


def _supersede_chain(raw_edges: List[Dict[str, Any]]) -> List[List[str]]:
    """The iteration chains: H1 → H2 → H3, in the order the study walked them."""
    nxt = {e["src"]: e["dst"] for e in raw_edges if e["type"] == "supersedes"}
    if not nxt:
        return []
    targets = set(nxt.values())
    chains = []
    for start in sorted(set(nxt) - targets, key=_id_order):
        chain, cur = [start], start
        while cur in nxt and nxt[cur] not in chain:  # a cycle must not hang us
            cur = nxt[cur]
            chain.append(cur)
        chains.append(chain)
    return chains


def _virtual_nodes(raw_nodes: Dict[str, Dict[str, Any]],
                   raw_edges: List[Dict[str, Any]],
                   carried: Dict[str, List[Tuple[str, str]]],
                   root: Optional[str], research_id: str,
                   scope: Optional[Tuple[str, str]] = None) -> List[Dict[str, Any]]:
    """The two cards nobody authors: the framing, and what it all added up to.

    Both are derived. Materializing them as stored nodes would mean a second
    source of truth that goes stale the moment a new Constraint lands, and
    would need an agent allowed to write them — which, with the context-init
    pre-stage off, there is not. Projected here they are always in step with
    the record, and to the page they are ordinary cards.
    """
    if not raw_nodes:
        return []
    out: List[Dict[str, Any]] = []
    human = {"human", "user", "operator"}

    members = [(_folded_view(f, raw_nodes[f], scope), role)
               for role, f in carried.get(FRAME_ID, []) if f in raw_nodes]
    root_attrs = dict((raw_nodes.get(root) or {}).get("attrs") or {}) if root else {}
    root_attrs.pop("formulation", None)
    fields = _fields(root_attrs, "", "ResearchQuestion")
    grouped: Dict[str, List[str]] = {}
    for view, _role in members:
        grouped.setdefault(view["kind"], []).append(view["label"])
    # `type:<kind>` rather than the display word, so the whole dict stays one
    # vocabulary the page can translate. Keyed on the Russian type_word, half of
    # this card's fields were untranslatable by construction.
    for kind, labels in grouped.items():
        fields["type:" + kind] = "; ".join(l for l in labels if l)[:600]
    seeded_by = {(raw_nodes.get(root) or {}).get("source", "")} | {
        v["source"] for v, _ in members}
    # An empty framing card is worse than none: it promises the reader a setup
    # and then has nothing in it. When nothing was ever recorded the page says
    # so instead, through the `no_frame` gap.
    if members or fields:
        out.append({
            "id": FRAME_ID, "run_id": research_id, "kind": "framing", "virtual": True,
            "label": _KIND_WORDS.get("Framing", "Техническое задание"),
            "type_word": _KIND_WORDS.get("Framing", "Техническое задание"),
            "index": None, "status": "derived",
            "status_word": _STATUS_WORDS.get("derived", ""),
            "executor_agent": "", "input": fields, "output": "",
            "provenance": [], "t_start": None, "t_end": None, "status_history": [],
            "why": "", "why_missing": False, "chips": [],
            "attachments": [v for v, _ in members],
            "level": _LEVEL["Framing"], "stage": "framing",
            "shown": True, "display_level": "primary",
            # Whether a human signed off on this setup: "agreed with the
            # operator" reads very differently from "the model assumed it".
            "hitl": bool(seeded_by & human),
        })

    conclusions = [n for n, d in raw_nodes.items()
                   if d.get("type") == "Conclusion"]
    hypotheses = {n: d for n, d in raw_nodes.items()
                  if d.get("type") == "Hypothesis"}
    settled = {n for n, d in hypotheses.items()
               if d.get("status") in ("confirmed", "refuted", "inconclusive")}
    spare = [(_folded_view(f, raw_nodes[f], scope))
             for _role, f in carried.get(OUTCOME_ID, []) if f in raw_nodes]
    if not (conclusions or settled or spare):
        return out

    chains = _supersede_chain(raw_edges)
    walk = chains[0] if chains else sorted(hypotheses, key=_id_order)
    steps = []
    for hid in walk:
        d = hypotheses.get(hid)
        if not d:
            continue
        status = d.get("status", "")
        why = _why(d.get("attrs") or {}, _history_view(d))
        step = f"{_short(_headline('Hypothesis', d.get('attrs') or {}), 70)} — " \
               f"{_STATUS_WORDS.get(status, status)}"
        steps.append(step + (f" ({_short(why, 90)})" if why else ""))
    out.append({
        "id": OUTCOME_ID, "run_id": research_id, "kind": "outcome", "virtual": True,
        "label": " → ".join(steps),
        "type_word": _KIND_WORDS.get("Outcome", "Итог"),
        "index": None, "status": "derived",
        "status_word": _STATUS_WORDS.get("derived", ""),
        "executor_agent": "", "input": {}, "output": " → ".join(steps),
        "provenance": [], "t_start": None, "t_end": None, "status_history": [],
        "why": "", "why_missing": False, "chips": [], "attachments": spare,
        "level": _LEVEL["Outcome"], "stage": "report",
        "shown": True, "display_level": "primary",
        "empty": not steps,
    })
    return out


def _derived_edges(nodes: List[Dict[str, Any]],
                   root: Optional[str]) -> List[Dict[str, Any]]:
    """Attach the two derived cards, so neither of them floats."""
    ids = {n["id"] for n in nodes}
    out: List[Dict[str, Any]] = []
    if FRAME_ID in ids and root in ids:
        out.append({"src": FRAME_ID, "dst": root, "type": "frames",
                    "synthetic": True})
    if OUTCOME_ID in ids:
        ends = [n["id"] for n in nodes if n["kind"] == "conclusion"]
        if not ends and root in ids:
            ends = [root]
        for end in ends:
            out.append({"src": end, "dst": OUTCOME_ID, "type": "concludes",
                        "synthetic": True})
    return out


def _context_edges(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
                   root: Optional[str]) -> List[Dict[str, Any]]:
    """Tie whatever the agents left unattached to the question, faintly.

    Computed over the PROJECTION, not the stored graph: a node can be connected
    in the record and orphaned in the drawing (its only link ran through a node
    that got folded), and the old check — run against the store — could not see
    that. A dotted line to the root is honest about being inferred, and it keeps
    the promise the page makes: one picture, not a field of islands.
    """
    if not root or len(nodes) < 2:
        return []
    # The PLAN track is left out on both sides. A step nobody has carried out
    # yet is not an orphan of the record — it is an intention waiting, drawn in
    # its own column — and tying it to the question with an inferred line said
    # the opposite: that the question had spawned it as work. Worse, one
    # unrealised step would be picked as its component's representative and the
    # real orphans behind it would stay unlinked.
    record = [n for n in nodes if n.get("track") != "plan"]
    plan_ids = {n["id"] for n in nodes if n.get("track") == "plan"}
    if len(record) < 2:
        return []
    und = nx.Graph()
    und.add_nodes_from(n["id"] for n in record)
    und.add_edges_from((e["src"], e["dst"]) for e in edges
                       if e["src"] not in plan_ids and e["dst"] not in plan_ids)
    out = []
    for comp in nx.connected_components(und):
        if root in comp:
            continue
        rep = sorted(comp, key=_id_order)[0]
        out.append({"src": root, "dst": rep, "type": "context",
                    "synthetic": True})
    return out


def _counters(raw_nodes: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """How the branches stand — the tally the page prints above the canvas."""
    counts = {"confirmed": 0, "refuted": 0, "under_verification": 0,
              "formulated": 0, "inconclusive": 0, "postponed": 0}
    for d in raw_nodes.values():
        if d.get("type") == "Hypothesis":
            status = d.get("status", "")
            if status in counts:
                counts[status] += 1
    return counts


def _headline_meta(raw_nodes: Dict[str, Dict[str, Any]],
                   raw_edges: List[Dict[str, Any]],
                   root: Optional[str]) -> Dict[str, Any]:
    chains = _supersede_chain(raw_edges)
    hypotheses = [n for n, d in raw_nodes.items() if d.get("type") == "Hypothesis"]
    root_attrs = (raw_nodes.get(root) or {}).get("attrs") or {} if root else {}
    return {
        "question": _headline("ResearchQuestion", root_attrs) if root else "",
        "root_id": root or "",
        "branches": len(chains) or (1 if hypotheses else 0),
        "iterations": len(hypotheses),
    }


def _gaps(raw_nodes: Dict[str, Dict[str, Any]], raw_edges: List[Dict[str, Any]],
          root: Optional[str], rejected: int) -> List[Dict[str, Any]]:
    """What the record is missing, as codes the page can say out loud.

    An empty canvas has meant three different things — nobody wrote anything,
    the write was refused, the server failed — and looked identical in all
    three. Naming the hole is the difference between a reader who waits and a
    reader who knows.
    """
    out: List[Dict[str, Any]] = []

    def gap(code: str, ids: List[str]) -> None:
        out.append({"code": code, "count": len(ids), "ids": sorted(ids, key=_id_order)})

    by_type: Dict[str, List[str]] = {}
    for n, d in raw_nodes.items():
        by_type.setdefault(d.get("type", ""), []).append(n)
    if not root:
        gap("no_root", [])
    if not any(by_type.get(t) for t in _FRAME_FOLD_TYPES):
        gap("no_frame", [])
    if not by_type.get("Hypothesis"):
        gap("no_hypotheses", [])
    else:
        tested = {e["src"] for e in raw_edges if e["type"] == "tested_by"}
        untested = [h for h in by_type["Hypothesis"] if h not in tested]
        if untested:
            gap("no_methods", untested)
    if by_type.get("VerificationMethod") and not by_type.get("Evidence"):
        gap("no_evidence", by_type["VerificationMethod"])
    silent = [n for n, d in raw_nodes.items()
              if d.get("status") in _UNRESOLVED_STATUSES
              and not _why(d.get("attrs") or {}, _history_view(d))]
    if silent:
        gap("unreasoned_failures", silent)
    if rejected:
        out.append({"code": "rejected_commits", "count": rejected, "ids": []})
    return out


class ResearchGraphStore:
    def __init__(self, directory: Optional[str] = None,
                 active_file: Optional[str] = None,
                 scope: Optional[Tuple[str, str]] = None) -> None:
        #: Which session this blackboard belongs to, or None for the CLI and
        #: unit-test constructions that have no ADK context. Only the view uses
        #: it, to build a link to a mirrored artifact — and a link needs the
        #: scope that is *reading*, so an imported study resolves under its new
        #: session id without anything being rewritten.
        self._scope = scope
        self._dir = Path(directory or _default_dir())
        self._path = self._dir / (active_file or _default_file())
        self._lock = threading.RLock()
        self._g = nx.MultiDiGraph()
        self._research_id = "research"
        self._created_at: float = time.time()
        self._root_id: Optional[str] = None
        # The confirmed, per-field research frame that produced this graph.
        # Stored with the graph (and therefore with its archive), so opening an
        # older study cannot accidentally show the active study's specification.
        self._framing_snapshot: Optional[Dict[str, Any]] = None
        #: Commits refused since the last successful one. Deliberately NOT
        #: persisted: it describes what is happening to this run, not what the
        #: study is, and a restart should not accuse the graph of old failures.
        self._rejected_commits = 0
        self._load()

    # ── public API ────────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        with self._lock:
            return self._g.number_of_nodes() == 0

    def root_id(self) -> Optional[str]:
        with self._lock:
            if self._root_id and self._g.has_node(self._root_id):
                return self._root_id
            # fallback: earliest ResearchQuestion
            questions = [(d.get("created_at", 0), n) for n, d in self._g.nodes(data=True)
                         if d.get("type") == "ResearchQuestion"]
            return min(questions)[1] if questions else None

    def ensure_root(self, question: str,
                    source: str = "human") -> Dict[str, Any]:
        """Guarantee a root ResearchQuestion WITHOUT destroying anything.

        `init_research` wipes the graph and starts a new generation, so calling
        it at the start of every message would archive the study the second
        message meant to continue. This fills the one hole that makes a graph
        unreadable: a run whose question was never written. That is not an edge
        case — with the context-init pre-stage off (RESEARCH_FRAME=0) nothing
        seeds the frame, and the only other path is an orchestrator that has to
        *remember* to call `research_init`. A study whose evidence hangs off no
        question cannot be read, and nothing in the viewer can repair that.

        Deliberately does NOT touch `_research_id`: the background validator
        keys its dedup and its discard-stale-verdict check on that id
        (validator.py), so rotating it mid-run would make every in-flight
        judgment throw itself away.
        """
        text = " ".join(str(question or "").split())
        if not text:
            return {"ok": False, "created": False,
                    "errors": ["question must be a non-empty string"]}
        with self._lock:  # RLock: root_id() below re-enters safely
            existing = self.root_id()
            if existing:
                if self._root_id != existing:
                    # A root written by a plain commit leaves `_root_id` unset —
                    # it is assigned only in init_research and _load — and then
                    # _archive_data names the study `research_graph_*` instead of
                    # `research_Q1_*`. Adopt it so the archive keeps its name.
                    self._root_id = existing
                    self._save()
                return {"ok": True, "created": False, "root_id": existing}
            result = self._commit_locked(
                source,
                [{"type": "ResearchQuestion", "ref": "q",
                  "attrs": {"formulation": text}}],
                [], [],
                # The seeding caller is whoever started the run ("human" by
                # default), and no ACL entry lets them create a question. The
                # structural validation still applies — only the per-agent table
                # is skipped, exactly as init_research does.
                enforce_permissions=False,
            )
            if not result.ok:
                return {"ok": False, "created": False, "errors": result.errors}
            self._root_id = next(n["id"] for n in result.committed["nodes"]
                                 if n.get("ref") == "q")
            self._save()  # _commit_locked never saves; commit() does it for us
            return {"ok": True, "created": True, "root_id": self._root_id}

    def init_research(self, source: str, question: str,
                      attrs: Optional[Dict[str, Any]] = None,
                      constraints: Optional[List[Dict[str, Any]]] = None,
                      tools: Optional[List[Dict[str, Any]]] = None,
                      resources: Optional[List[Dict[str, Any]]] = None,
                      empirical_bases: Optional[List[Dict[str, Any]]] = None,
                      confirmation_criteria: Optional[List[Dict[str, Any]]] = None,
                      cost_models: Optional[List[Dict[str, Any]]] = None,
                      question_source: Optional[str] = None) -> Dict[str, Any]:
        """Start a NEW research: root ResearchQuestion + the context star
        (Constraints —contextualizes→ Q, Q —defines_scope→ EmpiricalBases,
        CostModels —applies_to→ Q, standalone Tool/Resource/ConfirmationCriteria
        nodes). The previous graph, if any, is archived to a timestamped file —
        only after the new one validates.

        Each seed item may carry its own ``source`` (e.g. "human" for a field the
        operator set, "ContextInitAgent" for one the agent drafted) so the graph
        records the automation-vs-human split; items without it fall back to the
        top-level ``source``. ``question_source`` sets the root question's source.
        """
        if not (question or "").strip():
            return CommitResult(ok=False, errors=["question must be a non-empty string"],
                                hint=_COMMIT_HINT).model_dump()

        root: Dict[str, Any] = {
            "type": "ResearchQuestion", "ref": "q",
            "attrs": {"formulation": question.strip(), **(attrs or {})},
        }
        if question_source:
            root["source"] = question_source
        nodes: List[Dict[str, Any]] = [root]
        edges: List[Dict[str, Any]] = []

        def _star(items, node_type, ref_prefix):
            for i, item in enumerate(dict(it) for it in (items or []) if isinstance(it, dict)):
                status = item.pop("status", None)
                node_source = item.pop("source", None)
                a = item.pop("attrs", None) or item  # accept flat or {"attrs": …}
                draft = {"type": node_type, "ref": f"{ref_prefix}{i}", "attrs": a}
                if status:
                    draft["status"] = status
                if node_source:
                    draft["source"] = node_source
                nodes.append(draft)

        _star(constraints, "Constraint", "c")
        _star(tools, "Tool", "t")
        _star(resources, "Resource", "r")
        _star(empirical_bases, "EmpiricalBase", "eb")
        _star(confirmation_criteria, "ConfirmationCriteria", "cc")
        _star(cost_models, "CostModel", "cm")
        for draft in nodes:
            ref = draft["ref"]
            if draft["type"] == "Constraint":
                edges.append({"type": "contextualizes", "from": f"#{ref}", "to": "#q"})
            elif draft["type"] == "EmpiricalBase":
                edges.append({"type": "defines_scope", "from": "#q", "to": f"#{ref}"})
            elif draft["type"] == "CostModel":
                edges.append({"type": "applies_to", "from": f"#{ref}", "to": "#q"})

        with self._lock:
            old_graph, old_meta = self._g, (
                self._research_id, self._created_at, self._root_id,
                self._framing_snapshot,
            )
            old_data = self._serialize() if old_graph.number_of_nodes() else None
            self._g = nx.MultiDiGraph()
            self._research_id = (
                "research-"
                + datetime.now().strftime("%Y%m%d-%H%M%S")
                + "-"
                + uuid4().hex[:8]
            )
            self._created_at = time.time()
            self._root_id = None
            self._framing_snapshot = None
            # Privileged seeding: the context star (Question/Tool/Resource/
            # EmpiricalBase/Constraint) is created here regardless of the caller's
            # general create-set (schema.INIT_SEED_TYPES) — structural validation
            # still applies, only the per-agent ACL is skipped.
            result = self._commit_locked(source, nodes, edges, [],
                                         enforce_permissions=False)
            if not result.ok:
                # restore the previous research untouched
                self._g = old_graph
                (self._research_id, self._created_at, self._root_id,
                 self._framing_snapshot) = old_meta
                return result.model_dump()
            self._root_id = next(n["id"] for n in result.committed["nodes"] if n.get("ref") == "q")
            archived = self._archive_data(old_data) if old_data else None
            self._save()
        out = result.model_dump()
        out["root_id"] = self._root_id
        if archived:
            out["archived"] = archived
            out["message"] += f" (previous research archived to {archived})"
        return out

    def commit(self, source: str,
               nodes: Optional[List[Dict[str, Any]]] = None,
               edges: Optional[List[Dict[str, Any]]] = None,
               status_updates: Optional[List[Dict[str, Any]]] = None,
               autolink_focus: Optional[str] = None,
               partial_edges: bool = False,
               enforce_permissions: bool = True,
               allow_reserved: bool = False,
               exec_id: str = "") -> CommitResult:
        """Transactional write: validate EVERYTHING, then apply all-or-nothing.

        `autolink_focus` (a Hypothesis id): any Evidence created in this commit
        that isn't already linked to a hypothesis is auto-linked to it with a
        `relates_to` edge — so evidence a worker records while focused on a
        hypothesis is never orphaned, and the background validator can pick it up
        and decide its polarity.

        `partial_edges` downgrades a refused EDGE from an error to a warning: the
        valid nodes land and the bad link is dropped. It is OFF for agents on
        purpose — an agent told "nothing was saved" fixes its payload and retries,
        while one whose edge vanished quietly never learns that its finding is
        attached to nothing. It is ON for the deterministic writers, where losing
        a dozen good methods to one stale id is the worse failure.

        `enforce_permissions=False` is the privileged code-path (same contract as
        ``init_research``): structural validation stays, only the per-agent ACL
        is skipped. Reserved for deterministic module hooks — never LLM tools.

        Both keywords have live callers and dropping either breaks the other
        side at call time, silently: the experiment module's bridge swallows
        every exception by contract, so a missing `enforce_permissions` would
        show up only as an audit line and an empty graph."""
        with self._lock:
            result = self._commit_locked(source, list(nodes or []),
                                         list(edges or []), list(status_updates or []),
                                         enforce_permissions=enforce_permissions,
                                         autolink_focus=autolink_focus,
                                         partial_edges=partial_edges,
                                         allow_reserved=allow_reserved)
            if result.ok:
                self._rejected_commits = 0
                self._note_authorship(source, result, exec_id)
                self._save()
            else:
                # A refused commit writes NOTHING, and used to say so only to the
                # agent that made it: no log line, and the execution view marked
                # the call a success. Whole steps disappeared from the record
                # with nobody, anywhere, being told.
                self._rejected_commits += 1
                logger.warning("research commit refused (source=%s): %s",
                               source, "; ".join(result.errors[:3]))
            return result

    def record(self, source: str,
               nodes: Optional[List[Dict[str, Any]]] = None,
               edges: Optional[List[Dict[str, Any]]] = None,
               status_updates: Optional[List[Dict[str, Any]]] = None) -> CommitResult:
        """Privileged write for DETERMINISTIC stage recorders (code, not an LLM).

        Same transaction as :meth:`commit`, but the per-agent ACL is skipped —
        exactly like the context star in :meth:`init_research`. Structural
        validation (node types, subtypes, edge endpoints, status vocabularies)
        still applies. ``source`` is recorded on every node as the writer, so
        the graph still shows which stage produced what.
        """
        with self._lock:
            result = self._commit_locked(source, list(nodes or []),
                                         list(edges or []), list(status_updates or []),
                                         enforce_permissions=False)
            if result.ok:
                self._save()
            return result

    def _note_authorship(self, source: str, result: "CommitResult",
                         exec_id: str = "") -> None:
        """Who wrote this commit, recorded from the commit itself.

        The two bases nothing else can supply: `commit` for a node created here,
        `status` for one moved here. Called inside the commit's own lock and
        before its `_save`, so the rows ride the same snapshot write.

        A node created with a per-node `source` (an operator-set frame field
        arrives as "human") is credited to that source, not to the caller.

        `exec_id`, when the caller knows it, is the activation of the run that
        made this commit — so the record points at ONE run of the agent rather
        than at its name. An agent that worked on three nodes in a study is
        three runs, and a reader following a row wants the one that produced
        the node in front of them.
        """
        if source in _MACHINE_SOURCES:
            # A mirror is the pen, not the hand: `plan-mirror` writes every
            # PlanStep in the graph, and crediting it would make the record say
            # that the same non-agent took part in everything.
            return
        try:
            committed = result.committed or {}
            rows: List[Dict[str, Any]] = []
            for e in committed.get("nodes") or []:
                # An edit, or a draft that turned out to be a node already in
                # the graph — neither is authorship of it.
                if e.get("updated_attrs") or e.get("reused") or not e.get("id"):
                    continue
                nid = e["id"]
                wrote = source
                if self._g.has_node(nid):
                    wrote = self._g.nodes[nid].get("source") or source
                if wrote not in _MACHINE_SOURCES:
                    # Only the caller's own activation. A per-node source such
                    # as "human" was not that run, and saying it was would
                    # point the reader at the wrong place in the log.
                    rows.append({"node_id": nid, "agent": wrote,
                                 "basis": "commit",
                                 "exec_id": exec_id if wrote == source else ""})
            for e in committed.get("status_updates") or []:
                # `_auto_maintain` marks its own moves; the graph maintainer is
                # not a participant.
                if e.get("auto") or not e.get("id"):
                    continue
                rows.append({"node_id": e["id"], "agent": source,
                             "basis": "status", "exec_id": exec_id})
            if rows:
                self.add_contributors(rows, source=source, save=False)
        except Exception:  # noqa: BLE001 — bookkeeping never fails a commit
            logger.debug("contributors: could not record authorship", exc_info=True)

    def add_contributors(self, entries: List[Dict[str, Any]], *,
                         source: str = "", save: bool = True) -> Dict[str, int]:
        """Append who took part in a node, and on what basis we say so.

        Deliberately NOT a commit. The attrs merge in `_commit_locked` is a
        shallow `{**stored, **incoming}`, so a list routed through it keeps only
        the last writer's rows — which is precisely the confusion between nodes
        this record exists to prevent. It also validates nothing, moves no
        status and writes no edge: participation is an observation about a node,
        never a change to it.

        `updated_at` is left alone on purpose. `to_view` maps it to the card's
        end time and the study list sorts on it, so bookkeeping would make an
        idle study look freshly worked on.

        Each entry is `{node_id, agent, basis, exec_id?}`. Unknown ids and
        unknown bases are skipped rather than raised: this is called from
        callbacks and tool paths where nothing may break the run.
        """
        wanted = []
        for e in entries or []:
            nid = str((e or {}).get("node_id") or "").strip()
            agent = str((e or {}).get("agent") or "").strip()
            basis = str((e or {}).get("basis") or "").strip()
            if not nid or not agent or basis not in CONTRIB_BASES:
                continue
            wanted.append((nid, agent, basis,
                           str(e.get("exec_id") or "").strip()))
        if not wanted:
            return {"nodes": 0, "appended": 0}

        touched, appended = set(), 0
        with self._lock:
            for nid, agent, basis, exec_id in wanted:
                if not self._g.has_node(nid):
                    continue
                node = self._g.nodes[nid]
                attrs = dict(node.get("attrs") or {})
                rows = list(attrs.get("contributors") or [])
                # The same act recorded twice is one act: a resolver may run
                # again on the next commit, and an agent that writes forty
                # times did not participate forty times.
                seen = {(r.get("agent"), r.get("basis"), r.get("exec_id") or "")
                        for r in rows if isinstance(r, dict)}
                if (agent, basis, exec_id) in seen:
                    continue
                if len(rows) >= _MAX_CONTRIBUTORS:
                    # The first ones are the meaningful ones; the rest become a
                    # count, so a chatty run cannot grow the graph without bound.
                    attrs["contributors_more"] = int(
                        attrs.get("contributors_more") or 0) + 1
                else:
                    row = {"agent": agent, "basis": basis, "at": time.time()}
                    if exec_id:
                        row["exec_id"] = exec_id
                    rows.append(row)
                    attrs["contributors"] = rows
                node["attrs"] = attrs
                touched.add(nid)
                appended += 1
            if touched and save:
                # One snapshot for the batch: `_save` rewrites the whole graph.
                # `save=False` is for a caller inside a commit, which is about
                # to write that snapshot anyway.
                self._save()
        if touched:
            logger.debug("contributors: +%d row(s) on %s (source=%s)",
                         appended, ", ".join(sorted(touched)), source or "?")
        return {"nodes": len(touched), "appended": appended}

    # ── reads ─────────────────────────────────────────────────────────────────

    def get_context_slice(self, node_id: str, depth: int = 1,
                          char_budget: Optional[int] = None) -> Dict[str, Any]:
        """The subgraph a worker agent gets instead of the whole graph:
        the focus node + everything within `depth` edges (either direction)."""
        depth = max(1, min(int(depth or 1), self._max_depth()))
        with self._lock:
            if not self._g.has_node(node_id):
                return {"error": f"no node '{node_id}'. {self._ids_hint()}"}
            keep = {node_id}
            frontier = {node_id}
            for _ in range(depth):
                nxt = set()
                for n in frontier:
                    nxt.update(self._g.successors(n))
                    nxt.update(self._g.predecessors(n))
                frontier = nxt - keep
                keep |= nxt
            nodes = [self._node_public(n) for n in keep]
            edges = [{"type": k, "from": u, "to": v, **({"attrs": d["attrs"]} if d.get("attrs") else {})}
                     for u, v, k, d in self._g.edges(keys=True, data=True)
                     if u in keep and v in keep]
        nodes.sort(key=self._sort_key)
        focus = next(n for n in nodes if n["id"] == node_id)
        rendered = self._render_slice(focus, nodes, edges, char_budget or self._char_budget())
        return {"focus": node_id, "depth": depth, "nodes": nodes,
                "edges": edges, "rendered": rendered}

    def get_provenance(self, node_id: str) -> Dict[str, Any]:
        """Trace a node back to the root question (spec §5.5): the chain of
        typed nodes/edges plus who produced each step (automation vs human)."""
        with self._lock:
            if not self._g.has_node(node_id):
                return {"error": f"no node '{node_id}'. {self._ids_hint()}"}
            root = self.root_id()
            if root is None:
                return {"error": "the graph has no ResearchQuestion root yet."}
            if root == node_id:
                return {"root": root, "chain": [self._node_public(root)],
                        "rendered": f"{node_id} is the root question."}
            undirected = self._g.to_undirected(as_view=True)
            try:
                path = nx.shortest_path(undirected, source=node_id, target=root)
            except nx.NetworkXNoPath:
                return {"error": f"no path from '{node_id}' to the root '{root}'."}
            chain, lines = [], []
            for idx, n in enumerate(path):
                node = self._node_public(n)
                chain.append(node)
                lines.append(f"{n} [{node['type']}|{node['status']}] by {node['source']}")
                if idx + 1 < len(path):
                    lines.append(f"  {self._edge_between(n, path[idx + 1])}")
            return {"root": root, "chain": chain, "rendered": "\n".join(lines)}

    def overview(self) -> Dict[str, Any]:
        """Compact whole-graph index (ids/types/statuses/labels — no attrs):
        what the orchestrator consults; workers use it to find node ids."""
        with self._lock:
            counts: Dict[str, Dict[str, int]] = {}
            index = []
            for n, d in self._g.nodes(data=True):
                t, s = d.get("type", "?"), d.get("status", "?")
                counts.setdefault(t, {}).setdefault(s, 0)
                counts[t][s] += 1
                index.append({"id": n, "type": t, "status": s,
                              "label": self._label(d), "source": d.get("source", "")})
            root = self.root_id()
        index.sort(key=self._sort_key)
        lines = []
        if root:
            root_entry = next((e for e in index if e["id"] == root), None)
            if root_entry:
                lines.append(f"Root question {root}: {root_entry['label']}")
        for e in index:
            lines.append(f"- {e['id']} [{e['type']}|{e['status']}] {e['label']}")
        return {"root": root, "counts": counts, "nodes": index,
                "rendered": "\n".join(lines)}

    def full(self) -> Dict[str, Any]:
        with self._lock:
            return self._serialize()

    def set_framing_snapshot(self, frame: Dict[str, Any]) -> None:
        """Attach the confirmed research frame to the active study.

        This is presentation metadata, not a scientific graph node.  Keeping it
        in the graph snapshot gives every archived study its own per-field
        values and provenance while leaving agent context and graph semantics
        unchanged.
        """
        canonical = json.dumps(frame, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            self._framing_snapshot = {
                "schema_version": 1,
                "research_id": self._research_id,
                "root_id": self.root_id(),
                "saved_at": time.time(),
                "revision": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
                "frame": frame,
            }
            self._save()

    def _study_data(self, study_id: Optional[str]) -> Dict[str, Any]:
        chosen = study_id or "active"
        if chosen == "active":
            with self._lock:
                return self._serialize()
        path = self._dir / chosen
        if path.name != chosen or not path.is_file():
            raise KeyError(f"no archived study '{chosen}' in this session")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise KeyError(f"archived study '{chosen}' is unreadable") from exc
        if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
            raise KeyError(f"archived study '{chosen}' is unreadable")
        return data

    def framing_view_of(self, study_id: Optional[str] = None) -> Dict[str, Any]:
        """The structured technical specification for one selected study."""
        chosen = study_id or "active"
        data = self._study_data(chosen)
        snapshot = data.get("framing_snapshot")

        # Compatibility for active studies created before the frame became part
        # of the graph snapshot.  The sidecar has a root id; use it only when it
        # demonstrably belongs to the graph currently being read.  An archived
        # study must never borrow the active study's sidecar.
        if snapshot is None and chosen == "active":
            sidecar = self._dir / "research_frame.json"
            try:
                candidate = json.loads(sidecar.read_text(encoding="utf-8"))
                if not isinstance(candidate, dict):
                    candidate = {}
                candidate_frame = candidate.get("frame") or {}
                if not isinstance(candidate_frame, dict):
                    candidate_frame = {}
                candidate_question = ""
                for block in candidate_frame.get("blocks") or []:
                    if block.get("title") != "Вопрос исследования":
                        continue
                    for field in block.get("fields") or []:
                        if field.get("name") == "formulation":
                            candidate_question = str(field.get("value") or "").strip()
                            break
                graph_question = self._study_label(data).strip()
                has_archives = any(
                    path.name not in {self._path.name, "research_frame.json"}
                    for path in self._dir.glob("research_*.json")
                )
                exact_identity = (
                    candidate.get("research_id")
                    and candidate.get("research_id") == data.get("research_id")
                )
                safe_legacy_identity = (
                    not candidate.get("research_id")
                    and not has_archives
                    and candidate.get("root_id") == data.get("root_id")
                    and candidate_question == graph_question
                )
                if isinstance(candidate, dict) and (exact_identity or safe_legacy_identity):
                    snapshot = candidate
            except (OSError, ValueError):
                pass

        raw_nodes = {
            str(node.get("id")): node for node in (data.get("nodes") or [])
            if isinstance(node, dict) and node.get("id")
        }
        raw_edges = [{
            "src": edge.get("from"), "dst": edge.get("to"),
            "type": edge.get("type"), "attrs": dict(edge.get("attrs") or {}),
        } for edge in (data.get("edges") or []) if isinstance(edge, dict)]
        _host, carried, _hosts_of = _fold_plan(raw_nodes, raw_edges)
        frame_documents = {
            nid for role, nid in carried.get(FRAME_ID, [])
            if role == "attachment" and raw_nodes.get(nid, {}).get("type") == "Spec"
        }

        documents: List[Dict[str, Any]] = []
        for node in data.get("nodes") or []:
            if (not isinstance(node, dict) or node.get("type") != "Spec"
                    or str(node.get("id") or "") not in frame_documents):
                continue
            attrs = dict(node.get("attrs") or {})
            nid = str(node.get("id") or "")
            view = _folded_view(nid, node, self._scope)
            name = str(attrs.get("name") or "").strip()
            suffix = Path(name).suffix.lstrip(".").upper() if name else ""
            documents.append({
                "id": nid,
                "title": {"ru": "Предварительный текст ТЗ",
                          "en": "Preliminary specification"},
                "href": str(view.get("href") or ""),
                "format": suffix,
                # A legacy Spec can contain text but no downloadable file.
                "content": str(attrs.get("content") or "") if not view.get("href") else "",
                "source": str(node.get("source") or ""),
            })

        from CoScientist.graph.research.framing_view import build_framing_details

        return build_framing_details(
            data,
            snapshot=snapshot,
            documents=documents,
            study_id=chosen,
        )

    def restore(self, data: Dict[str, Any], archive: bool = True) -> Optional[str]:
        """Replace the active blackboard with a checkpointed snapshot.

        The displaced graph is archived by default, so rollback never destroys
        the later research branch even though it is removed from active state.
        """
        if not isinstance(data, dict):
            raise ValueError("research graph snapshot must be an object")
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("research graph nodes and edges must be arrays")
        graph = nx.MultiDiGraph()
        for node in nodes:
            if not isinstance(node, dict) or not node.get("id"):
                raise ValueError("every research graph node must have an id")
            graph.add_node(str(node["id"]), **dict(node))
        for edge in edges:
            if not isinstance(edge, dict) or not edge.get("from") or not edge.get("to"):
                raise ValueError("every research graph edge must have from and to")
            value = dict(edge)
            source = str(value.pop("from"))
            target = str(value.pop("to"))
            edge_type = value.get("type")
            graph.add_edge(source, target, key=edge_type, **value)
        with self._lock:
            archived = None
            if archive and self._g.number_of_nodes():
                archived = self._archive_data(self._serialize())
            self._g = graph
            self._research_id = data.get("research_id", "research")
            self._created_at = data.get("created_at", time.time())
            self._root_id = data.get("root_id")
            snapshot = data.get("framing_snapshot")
            self._framing_snapshot = snapshot if isinstance(snapshot, dict) else None
            self._save()
        return archived

    def full_graph(self) -> nx.MultiDiGraph:
        """Snapshot copy for the trigger queries. NetworkX ``copy()`` copies the
        structure and the attribute dicts one level deep — safe against this
        store's write patterns (we mutate/replace node data in place) and cheap
        at blackboard scale."""
        with self._lock:
            return self._g.copy()

    def to_view(self) -> Dict[str, Any]:
        """Project onto the shape web/templates/graph.html already renders
        (the execution-graph node/edge dicts).

        The tool calls behind a node travel on the node itself, in
        ``provenance``, and the panel lists them with a link into the execution
        log. They are deliberately NOT nodes of their own: this graph is the
        research record a scientist reads, and one Evidence can be the product
        of a dozen calls — drawn as nodes they outnumber the findings and bury
        the thing the reader came for.
        """
        with self._lock:
            raw_nodes = {n: dict(d) for n, d in self._g.nodes(data=True)}
            raw_edges = [{"src": u, "dst": v, "type": k,
                          "attrs": dict(d.get("attrs") or {})}
                         for u, v, k, d in self._g.edges(keys=True, data=True)]
            root = self.root_id()
            research_id = self._research_id
            rejected = self._rejected_commits
            framing_revision = str((self._framing_snapshot or {}).get("revision") or "legacy")

        host, carried, hosts_of = _fold_plan(raw_nodes, raw_edges)
        nodes = self._project_nodes(raw_nodes, raw_edges, carried, research_id)
        drawn = {n["id"] for n in nodes}
        nodes += _virtual_nodes(raw_nodes, raw_edges, carried, root, research_id,
                                self._scope)
        drawn |= {FRAME_ID, OUTCOME_ID} & {n["id"] for n in nodes}

        edges = _reroute(raw_edges, host, hosts_of, drawn)
        edges += _derived_edges(nodes, root)
        edges += _context_edges(nodes, edges, root)

        stamps = [d.get("updated_at") or d.get("created_at")
                  for d in raw_nodes.values()]
        stamps = [t for t in stamps if isinstance(t, (int, float))]
        latest_stamp = max(stamps) if stamps else None
        for node in nodes:
            if node.get("id") == FRAME_ID:
                # The browser uses this to cache the large, separately fetched
                # specification.  A new document or graph update invalidates it.
                node["framing_revision"] = (
                    f"{framing_revision}:{latest_stamp or 0}:{len(raw_nodes)}"
                )
        view = {
            "run_id": research_id,
            "nodes": nodes,
            "edges": edges,
            # When this study was last written to. A research graph outlives a
            # single prompt, so a session can show one that has not moved for a
            # day — which reads as "the new run produced nothing" only if the
            # reader can see the date.
            "updated_at": latest_stamp,
            "node_count": len(raw_nodes),
            "counters": _counters(raw_nodes),
            "headline": _headline_meta(raw_nodes, raw_edges, root),
            # Codes, never sentences: the page renders them through its own
            # dictionary, so an English reader does not get a Russian banner.
            "gaps": _gaps(raw_nodes, raw_edges, root, rejected),
        }
        _mark_stale_reports(view)
        return view

    def _project_nodes(self, raw_nodes: Dict[str, Dict[str, Any]],
                       raw_edges: List[Dict[str, Any]],
                       carried: Dict[str, List[Tuple[str, str]]],
                       research_id: str) -> List[Dict[str, Any]]:
        """One card per node of the scientific record, with what it carries."""
        drawn_ids = [n for n, d in raw_nodes.items()
                     if d.get("type") in _STORY_TYPES]
        ordinal: Dict[str, int] = {}
        seen_kind: Dict[str, int] = {}
        for nid in sorted(drawn_ids, key=_id_order):
            kind = raw_nodes[nid].get("type", "")
            seen_kind[kind] = seen_kind.get(kind, 0) + 1
            ordinal[nid] = seen_kind[kind]

        stages = _stages_of(raw_nodes, raw_edges)
        # Which CLAIM each method tests. Only a hypothesis is worth naming here:
        # when the parent is the root question — which is every method until a
        # hypothesis exists — the question is already the page banner, and
        # repeating it turns four method cards into four copies of one sentence.
        tests_for: Dict[str, str] = {}
        for e in raw_edges:
            if e.get("type") != "tested_by":
                continue
            parent = raw_nodes.get(e.get("src"))
            if parent is not None and parent.get("type") == "Hypothesis":
                tests_for[e.get("dst")] = _short(
                    _headline("Hypothesis", parent.get("attrs") or {}), 90)
        # What each plan step turned into. A step with nothing pointing at it is
        # an intention nobody carried out, and the reader should see that.
        realised_by: Dict[str, List[str]] = {}
        # And which detailed tasks were designed under it. A general step with
        # no task under it is a step the experiment module never planned for,
        # which is a different thing from a step nobody carried out.
        elaborated_by: Dict[str, List[str]] = {}
        elaborates: Dict[str, str] = {}
        for e in raw_edges:
            if e.get("type") == "realises":
                realised_by.setdefault(e.get("dst"), []).append(e.get("src"))
            elif e.get("type") == "elaborates":
                elaborated_by.setdefault(e.get("dst"), []).append(e.get("src"))
                elaborates[e.get("src")] = e.get("dst")

        # WITH WHAT a method is run — the part of "method" that was missing
        # altogether. A method is a method OF a claim, FOR settling it, BY some
        # instrument; the card used to carry only the first two. Five sources,
        # best first: the instruments the method itself names, the Tools it
        # declares it `uses`, the MCP servers and tools the approved experiment
        # plan pinned to it, the tools the PLAN already named for the step (the
        # planner can read the tool list, so this is the one part of the method
        # it can answer in advance), and the calls actually recorded against it.
        step_tools: Dict[str, str] = {}
        for sid, sd in raw_nodes.items():
            if sd.get("type") != "PlanStep":
                continue
            sa = sd.get("attrs") or {}
            task_id = str(sa.get("plan_task_id") or "").strip()
            if task_id:
                step_tools[task_id] = str(sa.get("tools") or "")
        declared: Dict[str, List[str]] = {}
        for e in raw_edges:
            if e.get("type") != "uses":
                continue
            tool = raw_nodes.get(e.get("dst"))
            if tool is None or tool.get("type") != "Tool":
                continue
            name = str((tool.get("attrs") or {}).get("name") or "").strip()
            if name:
                declared.setdefault(e.get("src"), []).append(name)
        instruments: Dict[str, str] = {}
        for mid, md in raw_nodes.items():
            if md.get("type") != "VerificationMethod":
                continue
            ma = md.get("attrs") or {}
            names = _named_instruments(ma.get("instruments"))
            names += declared.get(mid, [])
            names += _mcp_instruments(ma.get("mcp_servers"))
            names += _named_instruments(ma.get("tools"))
            names += _named_instruments(
                step_tools.get(str(ma.get("plan_task_id") or ""), ""))
            # The agent the plan put on it is an instrument too: "by whom" is
            # as much an answer to "by what means" as a library name is.
            names += _named_instruments(ma.get("assignee"))
            names += [str(c.get("tool") or "").strip()
                      for c in (ma.get("_provenance") or [])
                      if isinstance(c, dict) and str(c.get("tool") or "").strip()]
            ordered = _one_name_each(names)
            if ordered:
                # Capped: the card is a headline, and a method that called
                # twenty tools is better read in the panel.
                instruments[mid] = ", ".join(ordered[:_MAX_INSTRUMENTS])

        nodes: List[Dict[str, Any]] = []
        for nid in sorted(drawn_ids, key=_id_order):
            d = raw_nodes[nid]
            attrs = d.get("attrs") or {}
            kind = d.get("type", "?")
            status = d.get("status", "")
            headline = _headline(kind, attrs)
            history = _history_view(d)
            why = _why(attrs, history)
            chips, attachments, criterion = [], [], ""
            for role, folded in carried.get(nid, []):
                view = _folded_view(folded, raw_nodes[folded], self._scope)
                if role == "chip":
                    chips.append(view)
                elif role == "criterion":
                    criterion = criterion or view["label"]
                    attachments.append(view)
                else:
                    attachments.append(view)
            own = _own_artifact(nid, attrs, self._scope)
            if own:
                # First: it is this card's own file, not something it carries.
                attachments.insert(0, own)
            # Whether an Evidence record has been accepted or rejected is an
            # assessment of its reliability, not whether the observation was
            # obtained.  Keep the internal lifecycle untouched, but expose the
            # one user-facing state the card can actually have.
            if kind == "Evidence":
                view_status = "obtained"
            elif kind == "PlanStep":
                view_status = _plan_step_view_status(
                    nid, status, raw_nodes, raw_edges,
                )
                if view_status != status and not why:
                    why = ("все задачи эксперимента завершены"
                           if view_status == "done"
                           else "задачи эксперимента завершены без результата")
            else:
                view_status = status
            input_fields = _fields(attrs, headline, kind)
            if kind == "Evidence" and status in {"validated", "rejected"}:
                input_fields["evidence_assessment"] = _STATUS_WORDS[status]
            node = {
                "id": nid,
                "run_id": research_id,
                "kind": kind.lower(),
                # The id used to open every label. It means nothing to a reader
                # and cost a third of the line, so it moves to the panel and the
                # label says what the node is instead.
                "label": headline,
                "type_word": _KIND_WORDS.get(kind, kind),
                "index": ordinal.get(nid),
                "status": view_status,
                "status_word": _STATUS_WORDS.get(view_status, view_status),
                "executor_agent": d.get("source", ""),
                "input": input_fields,
                # For a write-up the card is the title and the panel is the
                # document; `reportBlock` renders this as markdown. Stored
                # references become URLs here rather than when the node was
                # written: the node keeps the session-free form, so an imported
                # bundle resolves its figures under whatever scope is reading.
                "output": (_readable_body(attrs.get("content"), self._scope)
                           if kind == "Report" else headline),
                "provenance": attrs.get("_provenance") or [],
                # A node's own write-up, inlined the way a Report's body is:
                # the browser has no way to fetch an artifact's TEXT. Read from
                # the store by id rather than kept on the node — `_truncate_attrs`
                # caps an ordinary attribute at 2 000 characters, and every card
                # rides along on a 1.5-second poll.
                "report": _node_report_body(attrs, self._scope),
                "report_stamp": str(attrs.get("report_stamp") or ""),
                "report_lang": str(attrs.get("report_lang") or ""),
                # Who took part, and on what basis we say so. Observed first,
                # planned last: a plan's assignee is an intention, and drawing
                # it as an executor is the confusion this record exists to end.
                "contributors": _contributors_view(attrs),
                "t_start": d.get("created_at"),
                "t_end": d.get("updated_at"),
                "status_history": history,
                "why": why,
                # A step that ended badly and cannot say why is a hole in the
                # record, not a rendering detail — the page says so out loud.
                "why_missing": bool(status in _UNRESOLVED_STATUSES and not why),
                "chips": chips,
                "attachments": attachments,
                "level": _LEVEL.get(kind, 3),
                # Which stage band the viewer draws it in. The level says where
                # it sits in the argument; the stage says which part of the work
                # produced it, and for Evidence and methods those differ.
                "stage": stages.get(nid, _STAGE_BY_TYPE.get(kind, "experiment")),
                "shown": True,
                "display_level": "primary",
            }
            if criterion:
                node["criterion"] = criterion
            if kind == "Hypothesis":
                node["origin"] = _origin_of(nid, raw_edges)
            if kind in _PLAN_LANE:
                # The plan is a parallel track, not a card in the band: the
                # viewer draws it in a column beside the stages, at the height
                # of the stage the step serves. Mixing the two is what made the
                # graph unreadable — a reader could not tell the intention from
                # the record.
                node["track"] = "plan"
                node["lane"] = _PLAN_LANE[kind]
                if realised_by.get(nid):
                    node["realised_by"] = sorted(realised_by[nid], key=_id_order)
                if elaborated_by.get(nid):
                    node["elaborated_by"] = sorted(elaborated_by[nid],
                                                   key=_id_order)
                # The general step this detailed task serves, by the words the
                # outer plan used — a bare "PS3" tells a reader nothing.
                step = elaborates.get(nid)
                if step and step in raw_nodes:
                    node["elaborates"] = _short(
                        _headline("PlanStep", raw_nodes[step].get("attrs") or {}),
                        90)
            if kind == "VerificationMethod":
                if tests_for.get(nid):
                    node["tests"] = tests_for[nid]
                # WITH WHAT it was run. A method is a method OF something, FOR
                # something, BY something — and the last of those was missing
                # entirely: the card said what the step was called and never
                # which instrument answered it. Collected from the Tools the
                # method `uses`, the plan step's declared tools, and the tool
                # calls actually recorded against it.
                if instruments.get(nid):
                    node["instruments"] = instruments[nid]
                # The plan step, always, on any method the mirror derived. Two
                # steps that read alike ("… органные эндпоинты ДЛЯ токсичного
                # кластера" against "… органные эндпоинты токсичного кластера")
                # produced two cards a reader could not tell apart; matching
                # their headlines would not have caught it, and loosening the
                # match to near-identical risks merging methods that differ.
                # The step number is short, authoritative and traceable.
                if attrs.get("plan_task_id"):
                    node["plan_step"] = str(attrs["plan_task_id"])
            nodes.append(node)
        return nodes

    def view_of(self, study_id: Optional[str] = None) -> Dict[str, Any]:
        """One study's projection, plus the list of the session's other studies.

        Mirrors the execution log, where a session lists its requests and draws
        one. `study_id` is "active" or an archive filename; the live study is
        used when nothing is asked for.
        """
        catalogue = self.studies()
        chosen = study_id or "active"
        if chosen == "active":
            view = self.to_view()
        else:
            view = self.archived_view(chosen)
        view["studies"] = catalogue
        view["study_id"] = chosen
        return view

    # ── the studies this session holds ───────────────────────────────────────

    def studies(self) -> List[Dict[str, Any]]:
        """Every study in this session, newest first, the live one first of all.

        `research_init` archives the study in progress and starts a new one, so
        a session accumulates them. Only the live one was ever reachable, which
        made a finished study look deleted and a stale one look like the current
        run's output.
        """
        out = [{"study_id": "active", "label": self._study_label(self._serialize()),
                "updated_at": self._latest_stamp(), "live": True,
                "node_count": self._g.number_of_nodes()}]
        for path in sorted(self._dir.glob("research_*.json"), reverse=True):
            if path.name in {self._path.name, "research_frame.json"}:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
                continue
            nodes = data.get("nodes") or []
            stamps = [n.get("updated_at") or n.get("created_at") for n in nodes]
            stamps = [t for t in stamps if isinstance(t, (int, float))]
            out.append({"study_id": path.name, "label": self._study_label(data),
                        "updated_at": max(stamps) if stamps else None,
                        "live": False, "node_count": len(nodes)})
        return out

    @staticmethod
    def _study_label(data: Dict[str, Any]) -> str:
        """A study is known by the question it asks."""
        for node in data.get("nodes") or []:
            if node.get("type") == "ResearchQuestion":
                said = (node.get("attrs") or {}).get("formulation")
                if said:
                    return str(said)
        return "untitled study"

    def _latest_stamp(self) -> Optional[float]:
        stamps = [d.get("updated_at") or d.get("created_at")
                  for _, d in self._g.nodes(data=True)]
        stamps = [t for t in stamps if isinstance(t, (int, float))]
        return max(stamps) if stamps else None

    def archived_view(self, study_id: str) -> Dict[str, Any]:
        """The same projection, over a study that has already been archived."""
        path = self._dir / study_id
        if path.name != study_id or not path.is_file():
            raise KeyError(f"no archived study '{study_id}' in this session")
        frozen = ResearchGraphStore(directory=str(self._dir), active_file=study_id)
        return frozen.to_view()

    def reset(self, archive: bool = True) -> Optional[str]:
        """Start over with an empty graph. The old graph is archived (never
        silently destroyed) unless archive=False. Returns the archive path."""
        with self._lock:
            archived = None
            if archive and self._g.number_of_nodes():
                archived = self._archive_data(self._serialize())
            self._g = nx.MultiDiGraph()
            self._research_id = "research"
            self._created_at = time.time()
            self._root_id = None
            self._framing_snapshot = None
            self._save()
            return archived

    # ── commit internals ──────────────────────────────────────────────────────

    def _commit_locked(self, source: str, node_drafts: List[Any],
                       edge_drafts: List[Any], status_drafts: List[Any],
                       enforce_permissions: bool = True,
                       autolink_focus: Optional[str] = None,
                       partial_edges: bool = False,
                       allow_reserved: bool = False) -> CommitResult:
        if not (node_drafts or edge_drafts or status_drafts):
            return CommitResult(ok=False, errors=[
                "empty commit — provide nodes, edges and/or status_updates"],
                hint=_COMMIT_HINT)
        errors: List[str] = []
        warnings: List[str] = []
        creates: List[Dict[str, Any]] = []     # {ref, type, status, attrs}
        merges: List[Dict[str, Any]] = []      # {id, attrs}
        refs: Dict[str, int] = {}              # ref -> index into creates
        # (type, identity text) -> (index into creates, index into the drafts),
        # so a sentence repeated inside ONE commit lands on one node too.
        staged: Dict[Tuple[str, str], Tuple[int, int]] = {}
        # ref -> an EXISTING node id, for a draft the store read as a
        # change rather than a creation. The edges drawn from that ref
        # still have to land somewhere, and it is not a new node.
        pinned: Dict[str, str] = {}

        # -- nodes: creations and attrs-merges -------------------------------
        for i, d in enumerate(node_drafts):
            if not isinstance(d, dict):
                errors.append(f"nodes[{i}]: must be an object")
                continue
            # An id is an IDENTITY: naming one means THIS node, whatever else
            # the draft carries. That used to hold only when `type` was ABSENT,
            # so {"id": "H3", "type": "Hypothesis", "attrs": {…}} — the form a
            # model writes when it is being helpful — fell through to the
            # create branch, where the id is never read again. The store
            # answered `ok: true` with a brand-new node and the agent, seeing
            # its change had not landed, sent it again. On
            # session_d9765b9e6de44530a3540da3aa0cd4c0 that silence turned one
            # hypothesis into ten.
            if d.get("id"):
                known = self._canon_node(str(d["id"]))
                if known is not None:
                    want = schema.normalize_node_type(d.get("type") or "")
                    stored = self._g.nodes[known].get("type")
                    if want and want != stored:
                        errors.append(
                            f"nodes[{i}]: '{known}' is a {stored}, not a {want}. "
                            f"To change it, write "
                            f'{{"id": "{known}", "attrs": {{…}}}}; to record a '
                            f"new {want}, leave the id out.")
                        continue
                    if d.get("status"):
                        warnings.append(
                            f"nodes[{i}]: a status on an update of '{known}' is "
                            f"ignored — move a node through `status_updates`.")
                    errors.extend(self._stage_merge(
                        source, i, d, merges, enforce_permissions,
                        warnings, allow_reserved))
                    continue
                if not d.get("type"):
                    # No such node and nothing to create from: `_stage_merge`
                    # owns that message, and it lists the ids that do exist.
                    errors.extend(self._stage_merge(
                        source, i, d, merges, enforce_permissions,
                        warnings, allow_reserved))
                    continue
                # An id that names nothing, next to a type, is a model
                # numbering its own draft. Create it — but say the id was not
                # honoured, so the next call does not point at it.
                warnings.append(
                    f"nodes[{i}]: there is no node '{str(d['id']).strip()}', so "
                    f"this was recorded as a NEW node; ids are minted by the "
                    f"store, never chosen by the caller.")
            ntype = schema.normalize_node_type(d.get("type", ""))
            spec = schema.NODE_TYPES.get(ntype)
            status = schema.normalize_token(
                d.get("status") or (spec.statuses[0] if spec else ""))
            attrs = d.get("attrs")
            if attrs is None:
                attrs = {}
            if not isinstance(attrs, dict):
                errors.append(f"nodes[{i}]: attrs must be an object")
                attrs = {}
            attrs = _strip_reserved(attrs, f"nodes[{i}]", warnings,
                                    allow_reserved)
            if "subtype" in attrs:
                attrs["subtype"] = schema.normalize_token(str(attrs["subtype"]))
            errors.extend(f"nodes[{i}]: {e}" for e in
                          schema.validate_node_draft(source, ntype, status, attrs,
                                                     enforce_permissions=enforce_permissions))
            # The same sentence, said twice, is one node — a reader sees one
            # card per node, so a second copy only makes it impossible to tell
            # which of them the edges belong to. Whether the first copy is
            # already in the graph or earlier in this very list, the answer is
            # to reuse it, not to refuse: a refusal costs the agent the rest of
            # an otherwise sound commit, and re-sending what is already
            # recorded has to be a no-op, not a loss.
            said = _identity_text(ntype, attrs)
            twin = self._twin_of(ntype, attrs)
            earlier = staged.get((ntype, said)) if said and twin is None else None
            at = earlier[0] if earlier else None

            ref = d.get("ref")
            if ref is not None:
                ref = str(ref)
                if not _REF_RE.match(ref):
                    errors.append(f"nodes[{i}]: invalid ref '{ref}' "
                                  "(1-32 chars of A-Za-z0-9_-)")
                    ref = None
                elif any(r.lower() == ref.lower() for r in refs):
                    errors.append(f"nodes[{i}]: duplicate ref '{ref}' in this commit "
                                  "(refs are matched case-insensitively)")
                    ref = None
                elif (named := self._canon_node(ref)) is not None:
                    # A ref is a LOCAL alias for a node created in THIS call.
                    # Putting an existing node's id there reads, to a model, as
                    # "the node I mean" — and the store read it as "a new node,
                    # call it that". Both the live run and its repair loop did
                    # this; nothing in the answer said otherwise.
                    #
                    # One shape of it is not ambiguous at all: a draft of a
                    # type whose identity the store knows, carrying none of
                    # that identity — a Hypothesis with no formulation, a
                    # PlanStep with no title — is not a new node under any
                    # reading. That is the live payload
                    # `{"type": "Hypothesis", "ref": "h1", "attrs":
                    # {"selected": "true"}}`: the agent marking H1 as the one
                    # it picked. Refusing it would cost the criterion and the
                    # method it was committed with.
                    if (ntype in _IDENTITY_ATTRS and not said
                            and self._g.nodes[named].get("type") == ntype):
                        warnings.append(
                            f"nodes[{i}]: ref '{ref}' names the existing "
                            f"{ntype} {named} and the draft says nothing that "
                            f"would make it a new one, so it was read as a "
                            f"change to {named}. Write "
                            f'{{"id": "{named}", "attrs": {{…}}}} to say so.')
                        errors.extend(self._stage_merge(
                            source, i, {"id": named, "attrs": attrs}, merges,
                            enforce_permissions, warnings, allow_reserved))
                        pinned[ref] = named
                        continue
                    errors.append(
                        f"nodes[{i}]: ref '{ref}' is the id of an existing node "
                        f"({named}). A ref only names a node created in THIS "
                        f"call. To change {named}, write "
                        f'{{"id": "{named}", "attrs": {{…}}}}. To record '
                        f"something new, pick a ref that is not an existing id.")
                    ref = None
                else:
                    # Both refs of a doubled draft point at the one node, so
                    # the edges from either of them land on it.
                    refs[ref] = len(creates) if at is None else at

            if earlier is not None:
                warnings.append(
                    f"nodes[{i}]: the same {ntype} is already nodes[{earlier[1]}] "
                    f"in this commit — recorded once.")
                first = creates[earlier[0]]
                # The LATER draft wins, the same way a live twin takes the new
                # attributes: a model that says a thing twice in one breath is
                # correcting itself, and the correction came second.
                first["attrs"] = {**first["attrs"], **attrs}
                continue
            if twin is not None:
                warnings.append(
                    f"nodes[{i}]: {twin} already says this, so it was reused "
                    f"instead of recording a second copy. To change {twin}, "
                    f'write {{"id": "{twin}", "attrs": {{…}}}}.')
                errors.extend(self._stage_merge(
                    source, i, {"id": twin, "attrs": attrs}, merges,
                    enforce_permissions, warnings, allow_reserved))
            elif said:
                staged[(ntype, said)] = (len(creates), i)
            creates.append({"ref": ref, "type": ntype, "status": status,
                            "attrs": attrs, "source": d.get("source"),
                            "twin": twin})

        # The hypothesis ceiling is applied further down, once the commit's
        # status_updates are staged too: a commit that closes a branch AND
        # proposes a new hypothesis is one move, and judging its halves apart
        # made the new one wait for a slot the same commit had just freed.

        # -- edges: resolve endpoints against existing nodes + this commit ---
        staged_edges: List[Dict[str, Any]] = []
        for j, d in enumerate(edge_drafts):
            if not isinstance(d, dict):
                errors.append(f"edges[{j}]: must be an object")
                continue
            etype = schema.normalize_token(d.get("type", ""))
            src, e1 = self._resolve_endpoint(d.get("from"), j, "from", refs,
                                             warnings, pinned)
            dst, e2 = self._resolve_endpoint(d.get("to"), j, "to", refs,
                                             warnings, pinned)
            edge_errs = [e for e in (e1, e2) if e]
            if src is None or dst is None:
                (warnings if partial_edges else errors).extend(edge_errs)
                continue
            ftype = creates[refs[src[1]]]["type"] if src[0] == "ref" else self._g.nodes[src[1]]["type"]
            ttype = creates[refs[dst[1]]]["type"] if dst[0] == "ref" else self._g.nodes[dst[1]]["type"]
            edge_errs += [f"edges[{j}]: {e}" for e in schema.validate_edge(
                source, etype, ftype, ttype, enforce_permissions=enforce_permissions)]
            attrs = d.get("attrs") or {}
            if etype == "conditional_successor":
                required = schema.normalize_token(str(attrs.get("required_status") or "refuted"))
                if required != "refuted":
                    edge_errs.append(
                        f"edges[{j}]: conditional_successor supports only "
                        "attrs.required_status='refuted'.")
                attrs = {**attrs, "required_status": required}
            # In partial mode a rejected LINK costs the link, not the commit.
            (warnings if partial_edges else errors).extend(edge_errs)
            if not edge_errs:
                staged_edges.append({"type": etype, "from": src, "to": dst,
                                     "attrs": attrs})

        def _edge_node(endpoint: Tuple[str, str]) -> str:
            if endpoint[0] == "id":
                return endpoint[1]
            created = creates[refs[endpoint[1]]]
            return str(created.get("twin") or f"@{endpoint[1].lower()}")

        conditional = nx.DiGraph()
        conditional.add_edges_from(
            (u, v) for u, v, key in self._g.edges(keys=True)
            if key == "conditional_successor"
        )
        for e in staged_edges:
            if e["type"] != "conditional_successor":
                continue
            u, v = _edge_node(e["from"]), _edge_node(e["to"])
            if u == v:
                errors.append("conditional_successor cannot link a hypothesis to itself.")
            conditional.add_edge(u, v)
        if conditional.number_of_edges() and not nx.is_directed_acyclic_graph(conditional):
            errors.append("conditional_successor edges must form an acyclic chain.")

        # -- status updates: existing nodes only ------------------------------
        # Criteria this commit marks met, whatever order they arrive in. The
        # validator writes the verdict and the criteria it rests on together,
        # and it lists the verdict first, so scanning only what is already
        # staged would judge the verdict against a bar this very commit is
        # raising.
        # Canonicalized, because `_unmet_criteria` returns stored ids while a
        # model writes what it likes: keyed on the raw string, a commit marking
        # `cc1` met was judged against `CC1` and refused for a bar it had just
        # cleared.
        #
        # A criterion may be marked met only WITH the measurement that meets it.
        # One LLM response supplies both the bar and the claim that the bar was
        # cleared, so the one thing that can be asked of it is that it say on
        # what. A non-empty reason, not a parsed id: the validator builds its
        # reason out of the evidence ids it used, and refusing prose would turn
        # a sound verdict into `inconclusive` over a formatting miss.
        met_here: Set[str] = set()
        for k, d in enumerate(status_drafts):
            if not isinstance(d, dict):
                continue
            if schema.normalize_token(d.get("status") or "") != "met":
                continue
            canon = self._canon_node(d.get("id") or "")
            if canon is None:
                continue
            if not str(d.get("reason") or "").strip():
                errors.append(
                    f"status_updates[{k}]: {canon} cannot be marked met without a "
                    f"`reason` naming the measurement that meets it — the bar and "
                    f"the claim that it was cleared are otherwise the same "
                    f"unargued sentence.")
                continue
            met_here.add(canon)

        def _has_supporting_evidence(hid: str) -> bool:
            """Whether an Evidence node positively supports this hypothesis.

            Counts what is already in the graph and what this commit stages —
            the validator writes the polarity edge and the verdict together.
            Only `supports`: `relates_to` means related, `refines` means
            modified, and neither is grounds for calling a claim confirmed.
            """
            for esrc, _edst, key in self._g.in_edges(hid, keys=True):
                if (key == "supports"
                        and self._g.nodes[esrc].get("type") == "Evidence"):
                    return True
            for e in staged_edges:
                if e["type"] != "supports":
                    continue
                dst = e["to"]
                if dst[0] != "id" or dst[1] != hid:
                    continue
                esrc = e["from"]
                ftype = (creates[refs[esrc[1]]]["type"] if esrc[0] == "ref"
                         else self._g.nodes[esrc[1]].get("type"))
                if ftype == "Evidence":
                    return True
            return False
        staged_status: List[Dict[str, Any]] = []
        for k, d in enumerate(status_drafts):
            if not isinstance(d, dict):
                errors.append(f"status_updates[{k}]: must be an object")
                continue
            new = schema.normalize_token(d.get("status") or "")
            nid = self._canon_node(d.get("id") or "")
            if nid is None:
                errors.append(f"status_updates[{k}]: no node "
                              f"'{str(d.get('id') or '').strip()}'. {self._ids_hint()}")
                continue
            node = self._g.nodes[nid]
            cur, ntype = node.get("status"), node.get("type")
            # Branch lock (spec §5.4): an under_verification hypothesis is busy —
            # a second attempt to start verification is a CONFLICT, not a no-op.
            if (ntype, cur, new) == ("Hypothesis", "under_verification",
                                     "under_verification"):
                owner, since = self._lock_owner(nid)
                errors.append(
                    f"status_updates[{k}]: {nid} is already under verification "
                    f"(locked by {owner} since {since}). Pick a different "
                    f"hypothesis or wait for the branch to close.")
                continue
            if cur == new:
                warnings.append(f"status_updates[{k}]: {nid} already has "
                                f"status '{new}' — skipped.")
                continue
            tr_errs = schema.validate_transition(source, ntype, cur, new,
                                                enforce_permissions=enforce_permissions)
            errors.extend(f"status_updates[{k}]: {e}" for e in tr_errs)
            confirming = (ntype, new) == ("Hypothesis", "confirmed")
            # A verdict with nothing behind it. The criteria gate never caught
            # this: a hypothesis with no criteria at all passed it vacuously,
            # which is exactly the study that was sloppiest.
            if confirming and not _has_supporting_evidence(nid):
                errors.append(
                    f"status_updates[{k}]: {nid} cannot be confirmed with no "
                    f"Evidence supporting it. Commit the finding and a "
                    f"`supports` edge from it to {nid}, or record the verdict as "
                    f"refuted or inconclusive.")
                continue
            unmet = ([c for c in self._unmet_criteria(nid) if c not in met_here]
                     if confirming else [])
            if unmet:
                # A hypothesis is confirmed against the bar written for it. A run
                # once reported "all criteria satisfied" while both of its
                # criteria still stood at not_met, and nothing contradicted it:
                # the claim and the bar were separate objects that never had to
                # agree. Now they do, and the refusal names what is outstanding.
                errors.append(
                    f"status_updates[{k}]: {nid} cannot be confirmed while its "
                    f"acceptance criteria are unmet ({', '.join(unmet)}). Either "
                    f"mark each criterion met with the measurement that meets it, "
                    f"or record the verdict as refuted or inconclusive.")
                continue
            if not tr_errs:
                staged_status.append({"id": nid, "type": ntype, "from": cur,
                                      "to": new, "reason": d.get("reason"),
                                      "index": k})

        final_status = {s["id"]: s["to"] for s in staged_status}
        for s in staged_status:
            if s["type"] != "Hypothesis" or s["to"] not in self._BUSY:
                continue
            unmet = []
            for pred, _, key, data in self._g.in_edges(s["id"], keys=True, data=True):
                if key != "conditional_successor":
                    continue
                required = str((data.get("attrs") or {}).get("required_status") or "refuted")
                actual = final_status.get(pred, self._g.nodes[pred].get("status"))
                if actual != required:
                    unmet.append(f"{pred}={actual} (needs {required})")
            if unmet:
                errors.append(
                    f"status_updates[{s['index']}]: {s['id']} is a conditional successor "
                    f"and cannot become active until {', '.join(unmet)}.")

        # -- verification slots: the whole commit at once ---------------------
        # After the loop, not inside it. Inside, each update could only see the
        # ones staged BEFORE it, so the same swap — close one branch, open
        # another — passed or was refused depending on the order it was written
        # in. `graph_bridge._sync_uncovered_hypotheses` builds its list in node
        # id order and does not get to choose that order.
        taken = self._charge_slots(staged_status, errors)
        self._normalize_hypothesis_selection(creates, warnings, taken)

        if errors:
            return CommitResult(ok=False, errors=errors, warnings=warnings,
                                hint=_COMMIT_HINT)

        # -- apply (all validation passed) ------------------------------------
        now = time.time()
        committed: Dict[str, List[Dict[str, Any]]] = {"nodes": [], "edges": [],
                                                      "status_updates": []}
        ref_ids: Dict[str, str] = {}
        # Every alias that resolved to this draft, not just the one written on
        # it: two drafts saying the same thing are recorded once, and BOTH
        # their refs have to reach that node or the edges from the second would
        # be drawn to nothing.
        aliases: Dict[int, List[str]] = {}
        for alias, idx in refs.items():
            aliases.setdefault(idx, []).append(alias)
        for k, c in enumerate(creates):
            if c.get("twin"):
                # Already in the graph; its attrs were staged as a merge above.
                # The refs still have to resolve, or every edge the agent drew
                # to this node would be lost with it — and the echo still has
                # to name it, so the caller learns which node its draft turned
                # out to be instead of assuming a new one.
                echo = {"id": c["twin"], "type": c["type"], "reused": True}
                for alias in aliases.get(k, ()):
                    ref_ids[alias] = c["twin"]
                if c["ref"]:
                    echo["ref"] = c["ref"]
                _name_every_alias(echo, aliases.get(k, ()))
                committed["nodes"].append(echo)
                continue
            nid = self._next_id(c["type"])
            attrs = self._truncate_attrs(c["attrs"], warnings, c["type"])
            # A per-node source (e.g. "human" for an operator-set frame field)
            # overrides the commit's default source; edges/status keep the default.
            node_source = c.get("source") or source
            node = ResearchNode(
                id=nid, type=c["type"], attrs=attrs, status=c["status"],
                source=node_source, created_at=now, updated_at=now,
                status_history=[{"from": None, "to": c["status"],
                                 "source": node_source, "at": now}],
            )
            self._g.add_node(nid, **node.model_dump())
            for alias in aliases.get(k, ()):
                ref_ids[alias] = nid
            echo = {"id": nid, "type": c["type"], "status": c["status"]}
            if c["ref"]:
                echo["ref"] = c["ref"]
            _name_every_alias(echo, aliases.get(k, ()))
            committed["nodes"].append(echo)
        reused = {e["id"] for e in committed["nodes"] if e.get("reused")}
        for m in merges:
            node = self._g.nodes[m["id"]]
            node["attrs"] = {**(node.get("attrs") or {}),
                             **self._truncate_attrs(m["attrs"], warnings,
                                                   node.get("type", ""))}
            node["updated_at"] = now
            if m["id"] in reused:
                # Already named in the echo as the node a draft turned out to
                # be; saying it twice would read as two nodes.
                continue
            committed["nodes"].append({"id": m["id"], "type": node.get("type"),
                                       "updated_attrs": sorted(m["attrs"])})
        seen_edges = set()
        for e in staged_edges:
            u = ref_ids[e["from"][1]] if e["from"][0] == "ref" else e["from"][1]
            v = ref_ids[e["to"][1]] if e["to"][0] == "ref" else e["to"][1]
            if (u, v, e["type"]) in seen_edges or self._g.has_edge(u, v, key=e["type"]):
                warnings.append(f"edge {u} -{e['type']}→ {v} already exists — skipped.")
                continue
            seen_edges.add((u, v, e["type"]))
            edge = ResearchEdge(id=f"{e['type']}:{u}->{v}", type=e["type"],
                                **{"from": u, "to": v},
                                attrs=e["attrs"], source=source, created_at=now)
            data = edge.model_dump(by_alias=True)
            data.pop("from"), data.pop("to")
            self._g.add_edge(u, v, key=e["type"], **data)
            committed["edges"].append({"type": e["type"], "from": u, "to": v})
        for s in staged_status:
            node = self._g.nodes[s["id"]]
            node["status"] = s["to"]
            node["updated_at"] = now
            history = list(node.get("status_history") or [])
            history.append({"from": s["from"], "to": s["to"], "source": source,
                            "at": now, "reason": s.get("reason")})
            node["status_history"] = history
            committed["status_updates"].append({"id": s["id"], "from": s["from"],
                                                "to": s["to"]})
        self._activate_conditional_successors(committed, now)
        # Focus auto-link (Option A): Evidence created here that isn't linked to
        # any hypothesis gets a `relates_to` edge to the focused hypothesis, so a
        # worker's finding is never orphaned. The background validator then decides
        # its polarity (supports/refutes). Deterministic; source graph-maintainer.
        self._autolink_focus(committed, autolink_focus, source, now)
        # Deterministic graph maintainer (no LLM): a `formulated` Hypothesis that
        # just received evidence (incl. an auto-linked relates_to) moves to
        # `under_verification` — work has started. Keeps the graph visibly live
        # (blue → amber) for free; the VERDICT is a judgment left to the validator.
        self._auto_maintain(committed, now)
        # A verdict is one value: collapse any pair that now carries more than
        # one, and say so when it was this commit's edge that lost.
        self._collapse_polarity(committed, warnings)
        message = (f"committed {len(committed['nodes'])} node(s), "
                   f"{len(committed['edges'])} edge(s), "
                   f"{len(committed['status_updates'])} status change(s)")
        return CommitResult(
            ok=True, message=message, committed=committed, warnings=warnings,
            graph_stats={"nodes": self._g.number_of_nodes(),
                         "edges": self._g.number_of_edges()},
        )

    # Edge types that mean "this evidence bears on this hypothesis". relates_to is
    # included so focus-auto-linked (polarity-unknown) evidence also advances the
    # hypothesis to under_verification and reaches the validator.
    _EVIDENCE_EDGES = ("supports", "refutes", "refines", "relates_to")
    #: The three that are a JUDGEMENT about the evidence, as opposed to
    #: `relates_to`, which only says it was recorded under this hypothesis.
    _VERDICT_EDGES = ("supports", "refutes", "refines")
    #: Who is entitled to the verdict. The background validator judges evidence
    #: it did not produce, which is the whole reason it exists; every other
    #: source is asserting a polarity for its own finding. So its verdict
    #: outranks theirs rather than merely arriving later — otherwise a worker
    #: recording the next piece of evidence overwrites the judge's answer.
    _VERDICT_RANK = {"ValidatorAgent": 2, "human": 2}

    def _verdict_rank(self, src: str, dst: str, key: str,
                      fresh: Iterable[str] = ()) -> Tuple[int, float, int]:
        """How strong a claim this edge is on the pair's one polarity slot.

        Rank first, so the judge's verdict beats a self-assertion however the
        clock falls. Then the timestamp. Then whether THIS commit wrote it,
        which is what breaks a tie the timestamp cannot: `created_at` comes
        from a wall clock with about 15ms of resolution on Windows, so a judge
        that re-judges twice in the same tick produced two verdicts of equal
        rank and equal time, and the winner fell out of the order the edge
        types happen to be declared in. An edge written in the commit being
        applied is the newest by construction.
        """
        data = self._g.edges[src, dst, key]
        source = str(data.get("source") or "")
        return (self._VERDICT_RANK.get(source, 1),
                float(data.get("created_at") or 0.0),
                1 if key in fresh else 0)

    def _collapse_polarity(self, committed: Dict[str, List[Dict[str, Any]]],
                           warnings: List[str]) -> None:
        """Leave one edge between each Evidence and Hypothesis this commit touched.

        Only pairs this commit touched are examined: a study is not re-swept on
        every write, and a pair nobody wrote to cannot have grown a second edge.
        """
        pairs = {(e["from"], e["to"]) for e in committed.get("edges", [])
                 if e.get("type") in self._EVIDENCE_EDGES}
        for src, dst in sorted(pairs):
            if not (self._g.has_node(src) and self._g.has_node(dst)):
                continue
            if self._g.nodes[src].get("type") != "Evidence" \
                    or self._g.nodes[dst].get("type") != "Hypothesis":
                continue
            present = [k for k in self._VERDICT_EDGES
                       if self._g.has_edge(src, dst, key=k)]
            if not present:
                continue
            # What this commit wrote for this pair, for the tie-break.
            fresh = {e.get("type") for e in committed.get("edges", [])
                     if e.get("from") == src and e.get("to") == dst}
            # The neutral link has been answered, so it goes.
            if self._g.has_edge(src, dst, key="relates_to"):
                self._g.remove_edge(src, dst, key="relates_to")
                committed["edges"] = [e for e in committed["edges"]
                                      if not (e.get("type") == "relates_to"
                                              and e.get("from") == src
                                              and e.get("to") == dst)]
            if len(present) == 1:
                continue
            keep = max(present,
                       key=lambda k: self._verdict_rank(src, dst, k, fresh))
            superseded = []
            for key in present:
                if key == keep:
                    continue
                data = self._g.edges[src, dst, key]
                superseded.append({"type": key,
                                   "source": str(data.get("source") or ""),
                                   "at": data.get("created_at")})
                self._g.remove_edge(src, dst, key=key)
                # If the loser is what this commit just wrote, the writer is
                # told: an agent whose verdict vanished without a word would
                # go on believing the graph agrees with it.
                if any(e.get("type") == key and e.get("from") == src
                       and e.get("to") == dst for e in committed["edges"]):
                    committed["edges"] = [
                        e for e in committed["edges"]
                        if not (e.get("type") == key and e.get("from") == src
                                and e.get("to") == dst)]
                    holder = str(self._g.edges[src, dst, keep].get("source")
                                 or "another source")
                    warnings.append(
                        f"{src} -{key}-> {dst} was dropped: that pair already "
                        f"carries `{keep}` from {holder}, and a piece of "
                        f"evidence has ONE polarity. Re-judging is the "
                        f"validator's call.")
            kept_data = self._g.edges[src, dst, keep]
            history = list(kept_data.get("superseded") or [])
            history.extend(superseded)
            kept_data["superseded"] = history

    def _focus_hypothesis(self, focus: str) -> Optional[str]:
        """Resolve a focus node to the Hypothesis it belongs to, so evidence
        gathered while focused on a method/tool/criteria of a hypothesis still
        auto-links to that hypothesis (the orchestrator often focuses on the VM
        or Tool it is verifying, not the hypothesis itself)."""
        t = self._g.nodes[focus].get("type")
        if t == "Hypothesis":
            return focus
        if t == "VerificationMethod":          # H -tested_by-> VM
            for u, _, k in self._g.in_edges(focus, keys=True):
                if k == "tested_by" and self._g.nodes[u].get("type") == "Hypothesis":
                    return u
        elif t == "Tool":                      # H -requires-> Tool
            for u, _, k in self._g.in_edges(focus, keys=True):
                if k == "requires" and self._g.nodes[u].get("type") == "Hypothesis":
                    return u
        elif t == "ConfirmationCriteria":      # CC -formulated_for-> H
            for _, v, k in self._g.out_edges(focus, keys=True):
                if k == "formulated_for" and self._g.nodes[v].get("type") == "Hypothesis":
                    return v
        return None

    def _autolink_focus(self, committed: Dict[str, List[Dict[str, Any]]],
                        focus: Optional[str], source: str, now: float) -> None:
        if not focus or not self._g.has_node(focus):
            return
        focus = self._focus_hypothesis(focus)
        if not focus:
            return
        # evidence ids already linked to SOME hypothesis in this commit
        linked = {e["from"] for e in committed["edges"]
                  if e["type"] in self._EVIDENCE_EDGES
                  and self._g.nodes.get(e["to"], {}).get("type") == "Hypothesis"}
        for n in committed["nodes"]:
            eid = n.get("id")
            if n.get("type") != "Evidence" or eid in linked:
                continue
            if self._g.has_edge(eid, focus, key="relates_to"):
                continue
            edge = ResearchEdge(id=f"relates_to:{eid}->{focus}", type="relates_to",
                                **{"from": eid, "to": focus}, attrs={},
                                source="graph-maintainer", created_at=now)
            data = edge.model_dump(by_alias=True)
            data.pop("from"), data.pop("to")
            self._g.add_edge(eid, focus, key="relates_to", **data)
            committed["edges"].append({"type": "relates_to", "from": eid,
                                       "to": focus, "auto": True})

    def _auto_maintain(self, committed: Dict[str, List[Dict[str, Any]]],
                       now: float) -> None:
        """Store invariant: a `formulated` Hypothesis that just received evidence
        moves to `under_verification` (work has started). Mechanical only — never
        decides a verdict. Runs inside the same locked commit."""
        targets = {e["to"] for e in committed.get("edges", [])
                   if e["type"] in self._EVIDENCE_EDGES}
        for hid in targets:
            if not self._g.has_node(hid):
                continue
            node = self._g.nodes[hid]
            if node.get("type") == "Hypothesis" and node.get("status") == "formulated":
                node["status"] = "under_verification"
                node["updated_at"] = now
                history = list(node.get("status_history") or [])
                history.append({"from": "formulated", "to": "under_verification",
                                "source": "graph-maintainer",
                                "at": now, "reason": "auto: к гипотезе приложено свидетельство"})
                node["status_history"] = history
                committed["status_updates"].append(
                    {"id": hid, "from": "formulated", "to": "under_verification",
                     "auto": True})

    def hypothesis_eligible(self, hypothesis_id: str) -> bool:
        """Whether every conditional predecessor has reached its required verdict."""
        hid = self._canon_node(hypothesis_id)
        if hid is None or self._g.nodes[hid].get("type") != "Hypothesis":
            return False
        for pred, _, key, data in self._g.in_edges(hid, keys=True, data=True):
            if key != "conditional_successor":
                continue
            required = str((data.get("attrs") or {}).get("required_status") or "refuted")
            if self._g.nodes[pred].get("status") != required:
                return False
        return True

    def _activate_conditional_successors(
        self, committed: Dict[str, List[Dict[str, Any]]], now: float,
    ) -> None:
        """Open the next chain link after a real refuted verdict, slot permitting."""
        refuted = [
            row["id"] for row in committed.get("status_updates", [])
            if row.get("to") == "refuted"
        ]
        if not refuted:
            return
        active = len(self._active_hypotheses())
        limit = self.max_active_hypotheses()
        for pred in sorted(refuted):
            successors = sorted(
                (v for _, v, key in self._g.out_edges(pred, keys=True)
                 if key == "conditional_successor"),
                key=str,
            )
            for hid in successors:
                if active >= limit:
                    return
                node = self._g.nodes[hid]
                if node.get("type") != "Hypothesis" or node.get("status") != "postponed":
                    continue
                if not self.hypothesis_eligible(hid):
                    continue
                node["status"] = "formulated"
                node["updated_at"] = now
                history = list(node.get("status_history") or [])
                history.append({
                    "from": "postponed", "to": "formulated",
                    "source": "graph-maintainer", "at": now,
                    "reason": f"conditional predecessor {pred} was refuted",
                })
                node["status_history"] = history
                committed["status_updates"].append({
                    "id": hid, "from": "postponed", "to": "formulated", "auto": True,
                })
                if not self._g.has_edge(pred, hid, key="supersedes"):
                    edge = ResearchEdge(
                        id=f"supersedes:{pred}->{hid}", type="supersedes",
                        **{"from": pred, "to": hid},
                        attrs={
                            "verdict": "refuted",
                            "reason": "conditional successor activated",
                        },
                        source="graph-maintainer", created_at=now,
                    )
                    data = edge.model_dump(by_alias=True)
                    data.pop("from"), data.pop("to")
                    self._g.add_edge(pred, hid, key="supersedes", **data)
                    committed["edges"].append({
                        "type": "supersedes", "from": pred, "to": hid, "auto": True,
                    })
                active += 1


    def _unmet_criteria(self, hypothesis_id: str) -> List[str]:
        """Criteria written for this hypothesis that are still not met."""
        outstanding = []
        for src, dst, key in self._g.in_edges(hypothesis_id, keys=True):
            if key != "formulated_for":
                continue
            node = self._g.nodes[src]
            if node.get("type") == "ConfirmationCriteria" \
                    and node.get("status") != "met":
                outstanding.append(src)
        return sorted(outstanding)

    def max_active_hypotheses(self) -> int:
        """How many hypotheses the run may verify at once — the one ceiling.

        `settings.web.max_active_hypotheses` (default 1) is the single place the
        number is set: the generator's prompt asks for up to this many, this
        store admits up to this many, and `queries.ready_hypotheses` offers up
        to this many for verification. Three readers, one number — when they
        disagreed, the graph filled with hypotheses nothing would ever test.
        """
        from CoScientist.config import get_settings

        return max(1, min(5, get_settings().web.max_active_hypotheses))

    #: A hypothesis in one of these states holds a verification slot: it is
    #: either offered for verification or being verified. Every other state is
    #: either a verdict or a shelf.
    _BUSY = ("formulated", "under_verification")

    def _active_hypotheses(self) -> List[str]:
        """Hypotheses already occupying a verification slot in the graph."""
        return [n for n, d in self._g.nodes(data=True)
                if d.get("type") == "Hypothesis"
                and d.get("status") in self._BUSY]

    def _charge_slots(self, staged_status: List[Dict[str, Any]],
                      errors: List[str]) -> int:
        """Refuse the status updates that would put the run over its ceiling.

        Returns how many slots are taken once the surviving updates apply —
        the occupancy the newly created hypotheses then have to fit into.

        Entering the busy set is what costs a slot, whichever door it comes
        through. Charging only `postponed → formulated` left the other one
        open: `inconclusive → under_verification` is an ordinary move (the
        validator writes `inconclusive` on every verdict it refuses to
        confirm, and reopening such a branch is the orchestrator's documented
        scheduling step), and it walked straight past the ceiling.
        """
        max_active = self.max_active_hypotheses()
        taken = len(self._active_hypotheses())
        # One node, one outcome. `from` is read from the GRAPH for every update,
        # so two verdicts naming the same hypothesis both saw it as busy and
        # both were counted as freeing a slot: `taken` went negative and bought
        # room that does not exist — two hypotheses went active under a ceiling
        # of one, with no error and no warning. The apply loop runs the updates
        # in order and the last one lands, so the last one is what may be
        # counted; a dict keeps that value at the position the node first
        # appeared, which is the order the refusals below are handed out in.
        final: Dict[str, Dict[str, Any]] = {}
        for s in staged_status:
            if s["type"] == "Hypothesis":
                final[s["id"]] = s
        entering: List[Dict[str, Any]] = []
        for s in final.values():
            was, now = s["from"] in self._BUSY, s["to"] in self._BUSY
            if was and not now:
                taken -= 1        # this branch closes and frees its slot
            elif now and not was:
                entering.append(s)
        room = max(0, max_active - taken)
        for s in entering[room:]:
            errors.append(
                f"status_updates[{s['index']}]: {s['id']} cannot be made "
                f"active — this commit would leave the run holding "
                f"{taken + room} of {max_active} hypotheses under "
                f"verification. Close a branch "
                f"(confirmed/refuted/inconclusive) in the same commit, or "
                f"raise the limit in the settings.")
        return taken + min(len(entering), room)

    def _normalize_hypothesis_selection(self, creates: List[Dict[str, Any]],
                                        warnings: List[str],
                                        taken: int = 0) -> None:
        """Store invariant: the RUN never holds more than N active hypotheses.

        N = ``settings.web.max_active_hypotheses`` (default 1). `taken` is what
        `_charge_slots` worked out — the occupancy once this commit's own
        status updates apply, so a commit that closes a branch leaves room for
        the hypothesis it proposes in the same breath.

        The slots already taken in the graph count. They did not use to: the
        ceiling looked at one commit's drafts only, so an agent told to «commit
        at most three per call, make more calls if you need more» produced one
        active hypothesis per call and the run ended up verifying several at
        once while the setting said one.

        Surplus is created ``postponed``, never dropped — the agent's work is
        its own, and the operator can see what was proposed. But postponed is a
        dead end by the schema (there is no ``postponed → under_verification``),
        so a hypothesis filed here will not be tested until something revives
        it. That is why this is a guard rail and not a plan: the generator is
        asked for at most N in the first place, and reaching this code means
        something went past that.
        """
        max_active = self.max_active_hypotheses()

        active = [c for c in creates
                  if c["type"] == "Hypothesis" and c["status"] == "formulated"
                  and not c.get("twin")]
        room = max(0, max_active - taken)
        if len(active) <= room:
            return
        # Sort by priority_rank (lower = higher priority) and keep the top ones
        # that still fit.
        ranked = sorted(active, key=lambda c: priority_rank(c["attrs"]))
        primary_set = set(id(c) for c in ranked[:room])
        for c in active:
            if id(c) in primary_set:
                continue
            c["status"] = "postponed"
            c["attrs"].setdefault(
                "postponed_reason",
                "сверх предела одновременно проверяемых гипотез — отложена, "
                "пока проверяются выбранные")
        proposed = (f"{len(active)} hypotheses were proposed as active at once"
                    if len(active) > 1 else "this hypothesis was proposed as active")
        held = (f", and {taken} of the {max_active} slots "
                f"{'is' if taken == 1 else 'are'} already taken in the graph"
                if taken else "")
        if room:
            kept = ", ".join(f'"{self._label(c, 60) or c.get("ref") or "?"}"'
                             for c in ranked[:room])
            outcome = (f"so {kept} stay 'formulated' and the remaining "
                       f"{len(active) - room} were created as 'postponed'")
        else:
            outcome = (f"so all {len(active)} were created as 'postponed' — the "
                       f"run has no free slot right now")
        warnings.append(
            f"{proposed}; only {max_active} may be verified at a time{held}, "
            f"{outcome}. A postponed hypothesis is NOT verified: the graph gives "
            f"it no route to a verdict until someone revives it "
            f"(postponed→formulated), and the study stays open while it sits. "
            f"Propose at most {max_active} so nothing is stranded, and mark your "
            f"pick with attrs.selected=true or a higher attrs.priority.")

    def _stage_merge(self, source: str, i: int, draft: Dict[str, Any],
                     merges: List[Dict[str, Any]],
                     enforce_permissions: bool = True,
                     warnings: Optional[List[str]] = None,
                     allow_reserved: bool = False) -> List[str]:
        """Validate an attrs-merge entry ({"id": …, "attrs": {…}}) — how e.g.
        the researcher enriches an existing EmpiricalBase (spec §2)."""
        nid = self._canon_node(draft["id"])
        if nid is None:
            return [f"nodes[{i}]: no node '{str(draft['id']).strip()}' to update. "
                    f"{self._ids_hint()}"]
        ntype = self._g.nodes[nid].get("type")
        perm = schema.AGENT_PERMISSIONS.get(source)
        attrs = draft.get("attrs")
        if not isinstance(attrs, dict) or not attrs:
            return [f"nodes[{i}]: an attrs object with the fields to merge is "
                    f"required to update '{nid}'."]
        attrs = _strip_reserved(attrs, f"nodes[{i}]", warnings, allow_reserved)
        if not attrs:
            return []
        draft = dict(draft, attrs=attrs)
        # A role that does not own the node may still owe it one field — the
        # reason a branch was left untested, what the evidence failed to
        # settle. Those are granted one (type, attribute) at a time, and only
        # a merge confined to them gets through this way.
        granted = {a for t, a in (perm.update_fields if perm else ()) if t == ntype}
        owns_type = perm is not None and (ntype in perm.update_attrs
                                          or ntype in perm.create)
        if enforce_permissions and not owns_type and not (granted and set(attrs) <= granted):
            allowed = ", ".join(sorted(perm.update_attrs | perm.create)) if perm else "none"
            if granted:
                allowed += f"; on {ntype} only attrs.{', attrs.'.join(sorted(granted))}"
            return [f"nodes[{i}]: agent '{source}' may not update attrs of "
                    f"{ntype} nodes (yours: {allowed})."]
        if ntype == "ConfirmationCriteria":
            # Compared against what the node already holds, because WRITING a
            # bar and MOVING one are different acts. A criterion is legal with
            # no attrs at all, so the generator may commit `{metric: "docking
            # score"}` and fill in the threshold a turn later; keyed on the
            # attribute name alone, that first write was refused as a move and
            # the hypothesis was left with a bar-less criterion that
            # `_unmet_criteria` still counted and the judge was asked to weigh
            # evidence against. An unchanged re-write is not a move either.
            stored = self._g.nodes[nid].get("attrs") or {}
            moved = sorted(k for k in set(attrs) & _BAR_ATTRS
                           if str(stored.get(k, "")).strip()
                           and str(attrs[k]).strip() != str(stored[k]).strip())
            if moved and self._criterion_is_under_measurement(nid):
                return [f"nodes[{i}]: the bar on {nid} is frozen — "
                        f"{', '.join(moved)} cannot be changed once evidence is "
                        f"being weighed against it. A bar that moves after the "
                        f"measurement is not a bar. Write a NEW "
                        f"ConfirmationCriteria for the revised standard and say "
                        f"in its text that it replaces {nid}."]
        if "subtype" in attrs:
            attrs = {**attrs, "subtype": schema.normalize_token(str(attrs["subtype"]))}
        merges.append({"id": nid, "attrs": attrs})
        return []

    def _criterion_is_under_measurement(self, ccid: str) -> bool:
        """Whether evidence has started arriving against this criterion's claim.

        Measured from the hypothesis the criterion was formulated for: once any
        Evidence points at that hypothesis, the bar has been aimed at and the
        run can no longer discover that it meant a different number all along.
        A criterion already marked met is frozen outright.
        """
        if self._g.nodes[ccid].get("status") == "met":
            return True
        for _src, hid, key in self._g.out_edges(ccid, keys=True):
            if key != "formulated_for":
                continue
            for esrc, _d, ekey in self._g.in_edges(hid, keys=True):
                if ekey in _POLARITY_EDGES and                         self._g.nodes[esrc].get("type") == "Evidence":
                    return True
        return False

    def _resolve_endpoint(self, value: Any, j: int, side: str,
                          refs: Dict[str, int],
                          warnings: List[str],
                          pinned: Optional[Dict[str, str]] = None,
                          ) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
        v = str(value or "").strip()
        if not v:
            return None, f"edges[{j}]: missing '{side}'"

        # Ref/id matching is case-INSENSITIVE and returns the CANONICAL stored
        # key — LLMs routinely define ref "E4" then cite it as "#e4", and node
        # ids are all uppercase, so forgiving the casing removes a whole class of
        # avoidable retries. The returned key stays the stored one so downstream
        # lookups (refs[...] / ref_ids[...] / graph node) are unchanged.
        def _match_ref(name: str) -> Optional[str]:
            if name in refs:
                return name
            low = name.lower()
            return next((r for r in refs if r.lower() == low), None)

        def _match_node(name: str) -> Optional[str]:
            return self._canon_node(name)

        if v.startswith("#"):
            r = _match_ref(v[1:])
            if r is not None:
                return ("ref", r), None
            # A ref the node loop resolved to a node already in the graph: the
            # draft was a change, not a creation, and an edge drawn from it
            # belongs on that node.
            for alias, nid in (pinned or {}).items():
                if alias.lower() == v[1:].lower():
                    return ("id", nid), None
            return None, (f"edges[{j}]: no ref '{v[1:]}' among the nodes created in "
                          f"this call (refs: {', '.join(sorted(refs)) or 'none'}).")
        node = _match_node(v)
        if node is not None:
            return ("id", node), None
        r = _match_ref(v)
        if r is not None:
            warnings.append(f"edges[{j}]: '{v}' treated as ref '#{r}' "
                            "(prefer the # prefix).")
            return ("ref", r), None
        return None, (f"edges[{j}]: no node '{v}'. {self._ids_hint()} To "
                      f"reference a node created in THIS call, use '#<ref>'.")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _canon_node(self, name: str) -> Optional[str]:
        """The stored node id matching `name` case-insensitively, or None.
        Node ids are uppercase-prefix + digits, so an uppercased probe is an
        unambiguous canonicalization (lets an LLM write 'h1' for 'H1')."""
        name = str(name or "").strip()
        if self._g.has_node(name):
            return name
        up = name.upper()
        return up if self._g.has_node(up) else None

    def _twin_of(self, ntype: str, attrs: Dict[str, Any]) -> Optional[str]:
        """The live node this draft would duplicate, or None.

        Identity is the text, not the id: an agent re-sending a hypothesis it
        already recorded has no id to give — the answer it got named a `ref`,
        not the node — so the only thing the two drafts share is what they say.
        """
        said = _identity_text(ntype, attrs)
        if not said:
            return None
        for nid, data in self._g.nodes(data=True):
            if data.get("type") != ntype:
                continue
            if _identity_text(ntype, data.get("attrs") or {}) == said:
                return str(nid)
        return None

    def _next_id(self, node_type: str) -> str:
        prefix = schema.NODE_TYPES[node_type].prefix
        pat = re.compile(rf"^{prefix}(\d+)$")
        highest = 0
        for n in self._g.nodes:
            m = pat.match(str(n))
            if m:
                highest = max(highest, int(m.group(1)))
        return f"{prefix}{highest + 1}"

    def _lock_owner(self, node_id: str) -> Tuple[str, str]:
        history = self._g.nodes[node_id].get("status_history") or []
        for entry in reversed(history):
            if entry.get("to") == "under_verification":
                at = entry.get("at")
                when = datetime.fromtimestamp(at).strftime("%Y-%m-%d %H:%M") if at else "?"
                return entry.get("source", "?"), when
        return "?", "?"

    def _ids_hint(self) -> str:
        ids = sorted(self._g.nodes, key=self._sort_key_id)
        if not ids:
            return "The graph is empty."
        if len(ids) <= 40:
            return f"Existing ids: {', '.join(ids)}."
        return (f"The graph has {len(ids)} nodes — call research_overview() "
                "to list them.")

    def _truncate_attrs(self, attrs: Dict[str, Any],
                        warnings: List[str],
                        kind: str = "") -> Dict[str, Any]:
        out = {}
        for k, v in (attrs or {}).items():
            # The write-up is the one attribute meant to be long. Held to the
            # 2 000-character cap, an 18 KB report reached the page as its first
            # two paragraphs and an ellipsis.
            cap = (_REPORT_CHAR_CAP if kind == "Report" and k == "content"
                   else _ATTR_CHAR_CAP)
            if isinstance(v, str) and len(v) > cap:
                out[k] = v[:cap] + "…[truncated]"
                warnings.append(f"attr '{k}' exceeded {cap} chars "
                                "and was truncated.")
            else:
                out[k] = v
        return out

    def _label(self, data: Dict[str, Any], n: int = 100) -> str:
        attrs = data.get("attrs") or {}
        for key in _LABEL_ATTRS:
            if attrs.get(key):
                return _short(attrs[key], n)
        # The last resort prints the record itself, so it must not print the
        # bookkeeping: a node whose only attrs were a participation list would
        # be labelled with the names of everyone who touched it.
        said = {k: v for k, v in attrs.items() if k not in _HIDDEN_FIELDS}
        return _short(json.dumps(said, ensure_ascii=False, default=str), n) if said else ""

    def _node_public(self, node_id: str) -> Dict[str, Any]:
        d = self._g.nodes[node_id]
        return {
            "id": node_id,
            "type": d.get("type"),
            "status": d.get("status"),
            "source": d.get("source"),
            "attrs": {k: _short(v, 400) if isinstance(v, str) else v
                      for k, v in (d.get("attrs") or {}).items()
                      if k not in _AGENT_HIDDEN},
        }

    @staticmethod
    def _sort_key(entry: Dict[str, Any]):
        return ResearchGraphStore._sort_key_id(entry["id"])

    @staticmethod
    def _sort_key_id(node_id: str):
        m = re.match(r"^([A-Z]+)(\d+)$", str(node_id))
        return (m.group(1), int(m.group(2))) if m else (str(node_id), 0)

    def _edge_between(self, a: str, b: str) -> str:
        for u, v in ((a, b), (b, a)):
            data = self._g.get_edge_data(u, v)
            if data:
                return f"{u} -{next(iter(data))}→ {v}"
        return f"{a} ~ {b}"

    def _render_slice(self, focus: Dict[str, Any], nodes: List[Dict[str, Any]],
                      edges: List[Dict[str, Any]], char_budget: int) -> str:
        lines = [f"CONTEXT SLICE for {focus['id']} "
                 f"({focus['type']}, status={focus['status']})", "Nodes:"]
        for n in nodes:
            attrs = json.dumps(n["attrs"], ensure_ascii=False, default=str)
            lines.append(f"- {n['id']} [{n['type']}|{n['status']}] {_short(attrs, 240)}")
        lines.append("Edges:")
        for e in edges:
            lines.append(f"- {e['from']} -{e['type']}→ {e['to']}")
        text = "\n".join(lines)
        if len(text) > char_budget:
            text = text[: char_budget] + ("\n…[slice truncated — call "
                                          "research_context_slice on specific nodes]")
        return text

    def _max_depth(self) -> int:
        try:
            from CoScientist.config import get_settings
            return get_settings().research_graph.slice_depth_max
        except Exception:  # noqa: BLE001
            return 2

    def _char_budget(self) -> int:
        try:
            from CoScientist.config import get_settings
            return get_settings().research_graph.slice_char_budget
        except Exception:  # noqa: BLE001
            return 4000

    # ── persistence ───────────────────────────────────────────────────────────

    def _serialize(self) -> Dict[str, Any]:
        return {
            "research_id": self._research_id,
            "created_at": self._created_at,
            "root_id": self._root_id,
            "framing_snapshot": self._framing_snapshot,
            "nodes": [dict(self._g.nodes[n]) for n in self._g.nodes],
            "edges": [{"type": k, "from": u, "to": v, **d}
                      for u, v, k, d in self._g.edges(keys=True, data=True)],
        }

    def _save(self) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._serialize(), f, ensure_ascii=False, default=str)
            os.replace(tmp, self._path)
        except Exception as exc:  # noqa: BLE001 — persistence must never break writes
            logger.warning("Research graph snapshot failed: %s", exc)

    def _archive_data(self, data: Dict[str, Any]) -> Optional[str]:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            root = data.get("root_id") or "graph"
            path = self._dir / (
                f"research_{root}_{stamp}_{uuid4().hex[:8]}.json"
            )
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
            return str(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Research graph archive failed: %s", exc)
            return None

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            g = nx.MultiDiGraph()
            for node in data.get("nodes", []):
                _respeak_method_status(node)
                g.add_node(node["id"], **node)
            for edge in data.get("edges", []):
                e = dict(edge)
                u, v, k = e.pop("from"), e.pop("to"), e.get("type")
                g.add_edge(u, v, key=k, **e)
            self._g = g
            self._research_id = data.get("research_id", "research")
            self._created_at = data.get("created_at", time.time())
            self._root_id = data.get("root_id")
            snapshot = data.get("framing_snapshot")
            self._framing_snapshot = snapshot if isinstance(snapshot, dict) else None
        except Exception as exc:  # noqa: BLE001 — a corrupt file must not kill startup
            logger.warning("Research graph load failed (%s); starting empty. "
                           "Keeping the file untouched at %s", exc, self._path)
            self._g = nx.MultiDiGraph()


# Legacy/default graph for standalone utilities and unit tests.
research_graph = ResearchGraphStore()
_research_graphs: Dict[SessionKey, ResearchGraphStore] = {
    DEFAULT_SESSION_KEY: research_graph,
}
_registry_lock = threading.RLock()


def get_research_graph(
    context: Any = None,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> ResearchGraphStore:
    """Return the typed research blackboard for one ADK user/session."""
    key = session_key(context, user_id=user_id, session_id=session_id)
    with _registry_lock:
        graph = _research_graphs.get(key)
        if graph is None:
            directory = storage_dir(_default_dir(), key)
            graph = ResearchGraphStore(
                directory=str(directory),
                active_file=_default_file(),
                scope=key,
            )
            _research_graphs[key] = graph
        return graph
