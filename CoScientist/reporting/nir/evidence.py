"""Gather everything a NIR report can be grounded in, from six sources.

The research graph is the backbone but not the whole record. Measured on the
richest run on disk (``session_b8b88d83d09b463e90f61fd283241a6d``) the graph
holds 38 nodes and 63 edges, while the execution graph holds ~127 000
characters of agents' own final reports — the TaskExecutor's tables, the
Coder's findings, the Experiment agent's numbers — none of which was ever
committed as an Evidence node. A report built from the graph alone would omit
most of what the run actually established.

So this module reads, in descending order of trustworthiness:

1. the research graph      — typed, validated, attributed, with status history
2. the execution graph     — agents' full final texts (``full()``, not
                             ``history()``, which truncates to 200 chars)
3. ``state['research_frame']`` — the operator-confirmed framing
4. ADK state               — the plan, work orders, sandbox metrics
5. the report folder       — figures, tables, files, MANIFEST
6. the aggregator's own Markdown, when it has already been written

Every source is optional and every read is guarded. A run whose execution graph
was never persisted still produces a report from the graph; a run with no graph
at all still produces one from the frame and the agent texts. What is missing is
recorded in :attr:`NirEvidence.gaps` rather than passed off as absent.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Node types that carry a bibliographic or dataset reference worth citing.
_LITERATURE_SUBTYPE = "literature"

#: Agent names whose final text is bookkeeping about tool selection rather than
#: a finding. Including them would bury the substance under retrieval chatter.
_NON_SUBSTANTIVE_AGENTS = frozenset({
    "ToolRetrieverAgent",
    "ToolReranker",
    "FullSetToolReranker",
    "ToolWebSearcherAgent",
    "ToolPipelineAgent",
    "ResultAggregatorAgent",
    "NirReportAgent",
})


@dataclass
class NirEvidence:
    """Everything the builder may draw on, already normalised."""

    user_id: str = ""
    session_id: str = ""

    # 1 — research graph
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)

    # 2 — execution graph
    agent_reports: List[Dict[str, str]] = field(default_factory=list)

    # 3/4 — frame and state
    original_request: str = ""
    frame_blocks: List[Dict[str, Any]] = field(default_factory=list)
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # 5 — report folder
    report_dir: Optional[Path] = None
    figures: List[Path] = field(default_factory=list)
    tables: List[Path] = field(default_factory=list)
    files: List[Path] = field(default_factory=list)

    # 6 — the short report, when it exists already
    final_markdown: str = ""

    #: Sources that were empty or unreadable, in words the author can act on.
    gaps: List[str] = field(default_factory=list)

    # ── typed views over the graph ──────────────────────────────────────────

    def by_type(self, *types: str) -> List[Dict[str, Any]]:
        wanted = set(types)
        return [n for n in self.nodes if n.get("type") in wanted]

    def node(self, node_id: str) -> Optional[Dict[str, Any]]:
        for n in self.nodes:
            if n.get("id") == node_id:
                return n
        return None

    def attrs(self, node_id: str) -> Dict[str, Any]:
        node = self.node(node_id)
        return dict(node.get("attrs") or {}) if node else {}

    def edges_from(self, node_id: str, *edge_types: str) -> List[Dict[str, Any]]:
        wanted = set(edge_types)
        return [
            e for e in self.edges
            if _edge_from(e) == node_id and (not wanted or e.get("type") in wanted)
        ]

    def edges_to(self, node_id: str, *edge_types: str) -> List[Dict[str, Any]]:
        wanted = set(edge_types)
        return [
            e for e in self.edges
            if _edge_to(e) == node_id and (not wanted or e.get("type") in wanted)
        ]

    @property
    def question(self) -> Optional[Dict[str, Any]]:
        questions = self.by_type("ResearchQuestion")
        return questions[0] if questions else None

    @property
    def hypotheses(self) -> List[Dict[str, Any]]:
        return self.by_type("Hypothesis")

    @property
    def conclusions(self) -> List[Dict[str, Any]]:
        return self.by_type("Conclusion")

    def evidence_for(self, hypothesis_id: str) -> List[Dict[str, Any]]:
        """Evidence nodes that bear on one hypothesis, whichever way they cut.

        ``supports`` and ``refutes`` both matter, and so does ``refines``: a
        report that showed only confirming evidence would misrepresent the run.
        """
        linked = {
            _edge_from(e)
            for e in self.edges_to(hypothesis_id, "supports", "refutes", "refines", "relates_to")
        }
        return [n for n in self.by_type("Evidence") if n.get("id") in linked]

    @property
    def literature(self) -> List[Dict[str, Any]]:
        return [
            n for n in self.by_type("Evidence")
            if (n.get("attrs") or {}).get("subtype") == _LITERATURE_SUBTYPE
        ]

    def type_census(self) -> Dict[str, int]:
        census: Dict[str, int] = {}
        for n in self.nodes:
            key = str(n.get("type") or "?")
            census[key] = census.get(key, 0) + 1
        return dict(sorted(census.items(), key=lambda kv: (-kv[1], kv[0])))

    def edge_census(self) -> Dict[str, int]:
        census: Dict[str, int] = {}
        for e in self.edges:
            key = str(e.get("type") or "?")
            census[key] = census.get(key, 0) + 1
        return dict(sorted(census.items(), key=lambda kv: (-kv[1], kv[0])))


def _edge_from(edge: Dict[str, Any]) -> Optional[str]:
    """Edges serialise as ``from``/``to``; the model class names them ``from_id``."""
    return edge.get("from") or edge.get("from_id")


def _edge_to(edge: Dict[str, Any]) -> Optional[str]:
    return edge.get("to") or edge.get("to_id")


# ── source readers ──────────────────────────────────────────────────────────


def _state_to_dict(state: Any) -> Dict[str, Any]:
    """ADK ``state`` is a State wrapper, not a dict; ``dict(state)`` misreads it.

    Same conversion ``tools/result_formatter_tool.py`` performs, for the same
    reason.
    """
    if state is None:
        return {}
    if isinstance(state, dict):
        return dict(state)
    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # noqa: BLE001
            pass
    try:
        return {k: state[k] for k in state.keys()}
    except Exception:  # noqa: BLE001
        return dict(getattr(state, "_value", {}) or {})


def _read_research_graph(tool_context: Any, into: NirEvidence) -> None:
    try:
        from CoScientist.graph.research.agent_tools import get_research_graph

        full = get_research_graph(tool_context).full() or {}
        into.nodes = list(full.get("nodes") or [])
        into.edges = list(full.get("edges") or [])
    except Exception as exc:  # noqa: BLE001
        logger.info("nir evidence: research graph unavailable (%s)", exc)
    if not into.nodes:
        into.gaps.append("научный граф пуст — разделы придётся строить по текстам агентов")


def _read_execution_graph(tool_context: Any, into: NirEvidence) -> None:
    """Agents' own final reports, which the research graph never receives.

    ``full()`` and not ``history()``: the latter is built for a prompt and cuts
    every output to 200 characters, which is where the substance is.
    """
    try:
        from CoScientist.graph.memory import get_knowledge_graph

        full = get_knowledge_graph(tool_context).full() or {}
    except Exception as exc:  # noqa: BLE001
        logger.info("nir evidence: execution graph unavailable (%s)", exc)
        into.gaps.append("граф исполнения недоступен — отчёты агентов не прочитаны")
        return

    reports: List[Dict[str, str]] = []
    for node in full.get("nodes") or []:
        if node.get("kind") != "agent":
            continue
        agent = str(node.get("executor_agent") or "")
        text = node.get("output")
        if not isinstance(text, str) or not text.strip():
            continue
        if agent in _NON_SUBSTANTIVE_AGENTS:
            continue
        reports.append({
            "agent": agent,
            "text": text.strip(),
            "status": str(node.get("status") or ""),
            "t_start": node.get("t_start") or 0,
        })
    reports.sort(key=lambda r: r.get("t_start") or 0)
    into.agent_reports = [
        {"agent": r["agent"], "text": r["text"], "status": r["status"]} for r in reports
    ]


def _read_state(state: Dict[str, Any], into: NirEvidence) -> None:
    frame = state.get("research_frame")
    if isinstance(frame, str):
        try:
            frame = json.loads(frame)
        except (ValueError, TypeError):
            frame = None
    if isinstance(frame, dict):
        inner = frame.get("frame") if isinstance(frame.get("frame"), dict) else frame
        into.original_request = str(inner.get("original_request") or "")
        blocks = inner.get("blocks")
        if isinstance(blocks, list):
            into.frame_blocks = [b for b in blocks if isinstance(b, dict)]
    if not into.original_request:
        into.gaps.append("исходная постановка задачи не найдена в состоянии сессии")

    tasks = state.get("_master_active_tasks") or state.get("active_tasks")
    if isinstance(tasks, str):
        try:
            tasks = json.loads(tasks)
        except (ValueError, TypeError):
            tasks = None
    if isinstance(tasks, list):
        into.tasks = [t for t in tasks if isinstance(t, dict)]

    metrics = state.get("sandbox_metrics")
    if isinstance(metrics, dict):
        into.metrics = metrics

    final = state.get("final_report")
    if isinstance(final, str):
        into.final_markdown = final


def _read_report_dir(report_dir: Optional[Path], into: NirEvidence) -> None:
    if report_dir is None:
        return
    into.report_dir = report_dir
    for attribute, subdir in (("figures", "figures"), ("tables", "tables"), ("files", "files")):
        directory = report_dir / subdir
        if not directory.is_dir():
            continue
        found = sorted(p for p in directory.iterdir() if p.is_file())
        setattr(into, attribute, found)
    if not into.figures:
        into.gaps.append("иллюстраций в папке отчёта нет — отчёт будет без рисунков")
    markdown = report_dir / "report.md"
    if not into.final_markdown and markdown.is_file():
        try:
            into.final_markdown = markdown.read_text(encoding="utf-8")
        except OSError:
            pass


def collect_nir_evidence(
    tool_context: Any,
    report_dir: Optional[Path] = None,
    state: Optional[Dict[str, Any]] = None,
) -> NirEvidence:
    """Read all six sources. Never raises; what failed is listed in ``gaps``."""
    from CoScientist.graph.session_scope import session_key

    try:
        user_id, session_id = session_key(tool_context)
    except Exception:  # noqa: BLE001
        user_id, session_id = "", ""

    resolved_state = state if state is not None else _state_to_dict(
        getattr(tool_context, "state", None)
    )

    into = NirEvidence(user_id=user_id, session_id=session_id)
    _read_research_graph(tool_context, into)
    _read_execution_graph(tool_context, into)
    _read_state(resolved_state, into)
    _read_report_dir(report_dir, into)

    logger.info(
        "nir evidence: %d graph nodes, %d edges, %d agent reports, %d figures, %d gaps",
        len(into.nodes), len(into.edges), len(into.agent_reports),
        len(into.figures), len(into.gaps),
    )
    return into


def agent_digest(reports: Sequence[Dict[str, str]], limit: int = 4000) -> List[Dict[str, str]]:
    """Agent texts trimmed for a prompt, keeping the head of each.

    The head is where an agent states what it concluded; the tail is usually
    bookkeeping. Trimming is announced in the text so the author knows there is
    more rather than assuming the agent stopped there.
    """
    digested: List[Dict[str, str]] = []
    for report in reports:
        text = report.get("text") or ""
        if len(text) > limit:
            text = text[:limit] + f"\n[...обрезано, полный текст {len(report['text'])} символов]"
        digested.append({"agent": report.get("agent", ""), "text": text})
    return digested


__all__ = ["NirEvidence", "collect_nir_evidence", "agent_digest"]
