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

import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
}

_STATUS_WORDS = {
    "open": "открыт", "decomposed": "разбит на части", "closed": "закрыт",
    "formulated": "предложена", "under_verification": "проверяется",
    "confirmed": "подтверждена", "refuted": "опровергнута",
    "inconclusive": "проверена — без ответа", "postponed": "отложена",
    "obtained": "получено", "validated": "проверено", "rejected": "отклонено",
    "planned": "запланирован", "running": "выполняется", "done": "выполнен",
    "failed": "не удался", "not_met": "ещё не выполнен", "met": "выполнен",
    "available": "доступен", "exhausted": "исчерпан",
    "needs_adaptation": "нужна доработка", "being_created": "создаётся",
    "creation_failed": "создать не удалось",
    "draft": "черновик", "approved": "утверждён", "created": "записан",
    "active": "действует",
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
}

#: Never shown: bookkeeping the reader has no use for.
_HIDDEN_FIELDS = {"_provenance", "selected", "display", "postponed_reason"}


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
        described = text("description", "procedure", "method_type")
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
    "Resource": {"resource_type", "remaining", "limit"},
    "EmpiricalBase": {"base_type", "volume", "name", "description"},
    "ConfirmationCriteria": {"threshold", "thresholds", "criteria", "content",
                             "confirmation_criteria", "confirm_refute_rule",
                             "success_metric", "rule", "description"},
    "Tool": {"name", "description"},
    "Conclusion": {"synthesis", "content", "description"},
    "Evidence": {"content", "description", "finding", "summary"},
}


def _fields(attrs: Dict[str, Any], headline: str,
            kind: str = "") -> Dict[str, str]:
    """The node's attributes under names a reader recognises.

    The panel used to list the record verbatim, keys and all, so a scientist
    read `base_type` and `source_ref`. Whatever the headline already says is
    dropped rather than repeated underneath it.
    """
    out: Dict[str, str] = {}
    spoken = _CONSUMED_BY_HEADLINE.get(kind, set())
    for key, value in attrs.items():
        if key in _HIDDEN_FIELDS or key in spoken or value in (None, "", [], {}):
            continue
        rendered = _short(value, 600) if not isinstance(value, str) else value
        if rendered.strip() and rendered.strip() == headline.strip():
            continue
        out[_FIELD_WORDS.get(key, key.replace("_", " ").capitalize())] = rendered
    return out


class ResearchGraphStore:
    def __init__(self, directory: Optional[str] = None,
                 active_file: Optional[str] = None) -> None:
        self._dir = Path(directory or _default_dir())
        self._path = self._dir / (active_file or _default_file())
        self._lock = threading.RLock()
        self._g = nx.MultiDiGraph()
        self._research_id = "research"
        self._created_at: float = time.time()
        self._root_id: Optional[str] = None
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
            old_graph, old_meta = self._g, (self._research_id, self._created_at, self._root_id)
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
            # Privileged seeding: the context star (Question/Tool/Resource/
            # EmpiricalBase/Constraint) is created here regardless of the caller's
            # general create-set (schema.INIT_SEED_TYPES) — structural validation
            # still applies, only the per-agent ACL is skipped.
            result = self._commit_locked(source, nodes, edges, [],
                                         enforce_permissions=False)
            if not result.ok:
                # restore the previous research untouched
                self._g = old_graph
                self._research_id, self._created_at, self._root_id = old_meta
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
               autolink_focus: Optional[str] = None) -> CommitResult:
        """Transactional write: validate EVERYTHING, then apply all-or-nothing.

        `autolink_focus` (a Hypothesis id): any Evidence created in this commit
        that isn't already linked to a hypothesis is auto-linked to it with a
        `relates_to` edge — so evidence a worker records while focused on a
        hypothesis is never orphaned, and the background validator can pick it up
        and decide its polarity."""
        with self._lock:
            result = self._commit_locked(source, list(nodes or []),
                                         list(edges or []), list(status_updates or []),
                                         autolink_focus=autolink_focus)
            if result.ok:
                self._save()
            return result

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
            nodes = []
            for n, d in self._g.nodes(data=True):
                attrs = d.get("attrs") or {}
                provenance = attrs.get("_provenance") or []
                node_type = d.get("type", "?")
                status = d.get("status", "")
                headline = _headline(node_type, attrs)
                nodes.append({
                    "id": n,
                    "run_id": self._research_id,
                    "kind": node_type.lower(),
                    # The id used to open every label. It means nothing to a
                    # reader and cost a third of the line, so it moves to the
                    # panel and the label says what the node is instead.
                    "label": headline,
                    "type_word": _KIND_WORDS.get(node_type, node_type),
                    "status": status,
                    "status_word": _STATUS_WORDS.get(status, status),
                    "executor_agent": d.get("source", ""),
                    "input": _fields(attrs, headline, node_type),
                    "output": headline,
                    "provenance": provenance,
                    "t_start": d.get("created_at"),
                    "t_end": d.get("updated_at"),
                })
            edges = [{"src": u, "dst": v, "type": k}
                     for u, v, k in self._g.edges(keys=True)]
            # Cosmetic: tie orphan components (context nodes declared but not yet
            # referenced — a Tool no hypothesis requires, an unconsumed Resource,
            # an unlinked EmpiricalBase) to the root question with a faint
            # "context" edge, so nothing floats in the viewer (the spec's "star").
            root = self.root_id()
            if root and self._g.number_of_nodes() > 1:
                und = self._g.to_undirected(as_view=True)
                for comp in nx.connected_components(und):
                    if root in comp:
                        continue
                    rep = sorted(comp, key=self._sort_key_id)[0]
                    edges.append({"src": root, "dst": rep, "type": "context",
                                  "synthetic": True})
            # When this study was last written to. A research graph outlives a
            # single prompt, so a session can show one that has not moved for a
            # day — which reads as "the new run produced nothing" only if the
            # reader can see the date. Without it the stale graph is
            # indistinguishable from a fresh one.
            stamps = [d.get("updated_at") or d.get("created_at")
                      for _, d in self._g.nodes(data=True)]
            stamps = [t for t in stamps if isinstance(t, (int, float))]
            return {"run_id": self._research_id, "nodes": nodes, "edges": edges,
                    "updated_at": max(stamps) if stamps else None,
                    "node_count": self._g.number_of_nodes()}

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
            if path.name == self._path.name:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
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
            self._save()
            return archived

    # ── commit internals ──────────────────────────────────────────────────────

    def _commit_locked(self, source: str, node_drafts: List[Any],
                       edge_drafts: List[Any], status_drafts: List[Any],
                       enforce_permissions: bool = True,
                       autolink_focus: Optional[str] = None) -> CommitResult:
        if not (node_drafts or edge_drafts or status_drafts):
            return CommitResult(ok=False, errors=[
                "empty commit — provide nodes, edges and/or status_updates"],
                hint=_COMMIT_HINT)
        errors: List[str] = []
        warnings: List[str] = []
        creates: List[Dict[str, Any]] = []     # {ref, type, status, attrs}
        merges: List[Dict[str, Any]] = []      # {id, attrs}
        refs: Dict[str, int] = {}              # ref -> index into creates

        # -- nodes: creations and attrs-merges -------------------------------
        for i, d in enumerate(node_drafts):
            if not isinstance(d, dict):
                errors.append(f"nodes[{i}]: must be an object")
                continue
            if d.get("id") and not d.get("type"):
                errors.extend(self._stage_merge(source, i, d, merges))
                continue
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
            if "subtype" in attrs:
                attrs["subtype"] = schema.normalize_token(str(attrs["subtype"]))
            errors.extend(f"nodes[{i}]: {e}" for e in
                          schema.validate_node_draft(source, ntype, status, attrs,
                                                     enforce_permissions=enforce_permissions))
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
                else:
                    refs[ref] = len(creates)
            creates.append({"ref": ref, "type": ntype, "status": status,
                            "attrs": attrs, "source": d.get("source")})

        # -- one active hypothesis per commit --------------------------------
        self._normalize_hypothesis_selection(creates, warnings)

        # -- edges: resolve endpoints against existing nodes + this commit ---
        staged_edges: List[Dict[str, Any]] = []
        for j, d in enumerate(edge_drafts):
            if not isinstance(d, dict):
                errors.append(f"edges[{j}]: must be an object")
                continue
            etype = schema.normalize_token(d.get("type", ""))
            src, e1 = self._resolve_endpoint(d.get("from"), j, "from", refs, warnings)
            dst, e2 = self._resolve_endpoint(d.get("to"), j, "to", refs, warnings)
            errors.extend(e for e in (e1, e2) if e)
            if src is None or dst is None:
                continue
            ftype = creates[refs[src[1]]]["type"] if src[0] == "ref" else self._g.nodes[src[1]]["type"]
            ttype = creates[refs[dst[1]]]["type"] if dst[0] == "ref" else self._g.nodes[dst[1]]["type"]
            edge_errs = schema.validate_edge(source, etype, ftype, ttype,
                                             enforce_permissions=enforce_permissions)
            errors.extend(f"edges[{j}]: {e}" for e in edge_errs)
            if not edge_errs:
                staged_edges.append({"type": etype, "from": src, "to": dst,
                                     "attrs": d.get("attrs") or {}})

        # -- status updates: existing nodes only ------------------------------
        # Criteria this commit marks met, whatever order they arrive in. The
        # validator writes the verdict and the criteria it rests on together,
        # and it lists the verdict first, so scanning only what is already
        # staged would judge the verdict against a bar this very commit is
        # raising.
        met_here = {
            str(d.get("id"))
            for d in status_drafts
            if isinstance(d, dict)
            and schema.normalize_token(d.get("status") or "") == "met"
        }
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
            unmet = ([c for c in self._unmet_criteria(nid) if c not in met_here]
                     if (ntype, new) == ("Hypothesis", "confirmed") else [])
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
                                      "to": new, "reason": d.get("reason")})

        if errors:
            return CommitResult(ok=False, errors=errors, warnings=warnings,
                                hint=_COMMIT_HINT)

        # -- apply (all validation passed) ------------------------------------
        now = time.time()
        committed: Dict[str, List[Dict[str, Any]]] = {"nodes": [], "edges": [],
                                                      "status_updates": []}
        ref_ids: Dict[str, str] = {}
        for c in creates:
            nid = self._next_id(c["type"])
            attrs = self._truncate_attrs(c["attrs"], warnings)
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
            if c["ref"]:
                ref_ids[c["ref"]] = nid
            echo = {"id": nid, "type": c["type"], "status": c["status"]}
            if c["ref"]:
                echo["ref"] = c["ref"]
            committed["nodes"].append(echo)
        for m in merges:
            node = self._g.nodes[m["id"]]
            node["attrs"] = {**(node.get("attrs") or {}),
                             **self._truncate_attrs(m["attrs"], warnings)}
            node["updated_at"] = now
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
                                "at": now, "reason": "auto: evidence attached"})
                node["status_history"] = history
                committed["status_updates"].append(
                    {"id": hid, "from": "formulated", "to": "under_verification",
                     "auto": True})


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

    def _normalize_hypothesis_selection(self, creates: List[Dict[str, Any]],
                                        warnings: List[str]) -> None:
        """Store invariant: at most N hypotheses enter the run as active per commit.

        N = ``settings.web.max_active_hypotheses`` (default 1).

        A generator agent naturally proposes several hypotheses at once; if they
        all land as ``formulated``, every one of them shows up as READY and the
        orchestrator starts verifying them — which may not be desired. So
        exactly N (the agent's own picks: ``attrs.selected``, else the highest
        ``attrs.priority``, else the first N) stay ``formulated`` and the rest
        are created as ``postponed``: they remain in the graph as the ranked
        backlog, invisible to the READY trigger, and the orchestrator can revive
        one (postponed→formulated) once an active branch has a verdict.

        Deterministic and mechanical — it never drops or rewrites a hypothesis,
        only decides which ones are offered for verification next.
        """
        from CoScientist.config import get_settings
        max_active = max(1, min(5, get_settings().web.max_active_hypotheses))

        active = [c for c in creates
                  if c["type"] == "Hypothesis" and c["status"] == "formulated"]
        if len(active) <= max_active:
            return
        # Sort by priority_rank (lower = higher priority) and keep top N.
        ranked = sorted(active, key=lambda c: priority_rank(c["attrs"]))
        primary_set = set(id(c) for c in ranked[:max_active])
        for c in active:
            if id(c) in primary_set:
                continue
            c["status"] = "postponed"
            c["attrs"].setdefault(
                "postponed_reason",
                "alternative hypothesis — kept as backlog while the selected "
                "ones are verified")
        kept_labels = ", ".join(
            f'"{self._label(c, 60) or c.get("ref") or "?"}"'
            for c in ranked[:max_active])
        warnings.append(
            f"{len(active)} hypotheses were proposed as active at once; only "
            f"{max_active} may be verified at a time, so {kept_labels} "
            f"stay 'formulated' and the other "
            f"{len(active) - max_active} were created as 'postponed' (backlog). "
            f"To choose which ones are verified, mark them with "
            f"attrs.selected=true or a higher attrs.priority; the orchestrator "
            f"can revive a postponed one later.")

    def _stage_merge(self, source: str, i: int, draft: Dict[str, Any],
                     merges: List[Dict[str, Any]]) -> List[str]:
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
        # A role that does not own the node may still owe it one field — the
        # reason a branch was left untested, what the evidence failed to
        # settle. Those are granted one (type, attribute) at a time, and only
        # a merge confined to them gets through this way.
        granted = {a for t, a in (perm.update_fields if perm else ()) if t == ntype}
        owns_type = perm is not None and (ntype in perm.update_attrs
                                          or ntype in perm.create)
        if not owns_type and not (granted and set(attrs) <= granted):
            allowed = ", ".join(sorted(perm.update_attrs | perm.create)) if perm else "none"
            if granted:
                allowed += f"; on {ntype} only attrs.{', attrs.'.join(sorted(granted))}"
            return [f"nodes[{i}]: agent '{source}' may not update attrs of "
                    f"{ntype} nodes (yours: {allowed})."]
        if "subtype" in attrs:
            attrs = {**attrs, "subtype": schema.normalize_token(str(attrs["subtype"]))}
        merges.append({"id": nid, "attrs": attrs})
        return []

    def _resolve_endpoint(self, value: Any, j: int, side: str,
                          refs: Dict[str, int],
                          warnings: List[str]) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
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
                        warnings: List[str]) -> Dict[str, Any]:
        out = {}
        for k, v in (attrs or {}).items():
            if isinstance(v, str) and len(v) > _ATTR_CHAR_CAP:
                out[k] = v[:_ATTR_CHAR_CAP] + "…[truncated]"
                warnings.append(f"attr '{k}' exceeded {_ATTR_CHAR_CAP} chars "
                                "and was truncated.")
            else:
                out[k] = v
        return out

    def _label(self, data: Dict[str, Any], n: int = 100) -> str:
        attrs = data.get("attrs") or {}
        for key in _LABEL_ATTRS:
            if attrs.get(key):
                return _short(attrs[key], n)
        return _short(json.dumps(attrs, ensure_ascii=False, default=str), n) if attrs else ""

    def _node_public(self, node_id: str) -> Dict[str, Any]:
        d = self._g.nodes[node_id]
        return {
            "id": node_id,
            "type": d.get("type"),
            "status": d.get("status"),
            "source": d.get("source"),
            "attrs": {k: _short(v, 400) if isinstance(v, str) else v
                      for k, v in (d.get("attrs") or {}).items()},
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
                g.add_node(node["id"], **node)
            for edge in data.get("edges", []):
                e = dict(edge)
                u, v, k = e.pop("from"), e.pop("to"), e.get("type")
                g.add_edge(u, v, key=k, **e)
            self._g = g
            self._research_id = data.get("research_id", "research")
            self._created_at = data.get("created_at", time.time())
            self._root_id = data.get("root_id")
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
            )
            _research_graphs[key] = graph
        return graph
