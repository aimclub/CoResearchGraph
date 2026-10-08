"""Session-scoped in-process execution graph shared by agents in one session.

Reuses the Dynamic Execution Graph data layer (GraphStore + Node/Edge) but, for
the single-process web/cli runs, exposes it directly (no HTTP service) so:

* it holds a single root node that the session's turns hang from. The agent
  roster is NOT part of it — that is configuration, read by ``agents_info`` and
  handed to the orchestrator and planner up front (see ``inject_graph_root``),
  so the graph shows what a session did rather than what the system is;
* it **grows** as agents act (``GraphMemoryPlugin``) — an agent node exists
  because that agent acted;
* it is **readable by ANY agent at ANY time** via the graph toolset
  (``CoScientist.graph.agent_tools``), so an agent can check the history and the
  info about all other agents.

Each ``(user_id, session_id)`` receives its own graph. The graph accumulates
across all prompts in that session and can be snapshotted and visualised without
leaking activity from another user's work.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

from CoScientist.graph.models import Edge, Node, StatusUpdate
from CoScientist.graph.session_scope import (
    DEFAULT_SESSION_KEY,
    SessionKey,
    session_key,
    storage_dir,
)
from CoScientist.graph.store import GraphStore

SESSION_RUN_ID = os.getenv("KG_RUN_ID", "session")
ROOT_ID = "system:root"

# Node kinds that represent things that actually happened during a run (as
# opposed to the static roster), in the order we show them as "history".
_HISTORY_KINDS = ("goal", "agent_call", "tool_call", "result", "decision", "reflection")


def _now() -> float:
    return time.time()


def _roster_from_config() -> List[Dict[str, Any]]:
    """Describe every agent in system.yaml for the graph root."""
    from CoScientist.assembly.schema import get_config

    cfg = get_config()
    roster: List[Dict[str, Any]] = []
    for name, a in cfg.agents.items():
        roster.append(
            {
                "name": name,
                "description": (a.description or "").strip(),
                "role": (getattr(a, "planning", "") or a.description or "").strip(),
                "routing": (a.routing or "").strip(),
                "tools": list(a.tools),
                "subordinates": list(a.subordinates),
                "is_root": bool(getattr(a, "root", False)),
            }
        )
    return roster


class KnowledgeGraph:
    """Thin in-process facade over a single-run GraphStore."""

    def __init__(self, run_id: str = SESSION_RUN_ID, snapshot_dir: Optional[str] = None) -> None:
        self.run_id = run_id
        self._store = GraphStore(
            snapshot_dir=snapshot_dir or os.getenv("GRAPH_SNAPSHOT_DIR", "./graph_runs")
        )
        self._seeded = False
        self._lock = threading.Lock()

    # ── seeding ───────────────────────────────────────────────────────────────
    def ensure_seeded(self) -> None:
        """Create the root the session's turns hang from (idempotent).

        The roster used to be seeded here as well — one node per configured
        agent, joined to the root by `has_member`. It made every graph open on
        the same picture of the system rather than on what the session did, and
        the viewer filtered most of it back out again. The roster is
        configuration, so `agents_info` now reads it from the config and the
        graph carries only what happened: an agent appears when it acts.
        """
        if self._seeded:
            return
        with self._lock:
            if self._seeded:
                return
            self._store.add_node(
                Node(
                    id=ROOT_ID,
                    run_id=self.run_id,
                    kind="system",
                    label="CoScientist multi-agent system",
                    status="success",
                    input={"agents": [a["name"] for a in self.agents_info()]},
                    t_start=_now(),
                )
            )
            self._seeded = True

    def reset_session(self) -> None:
        """Explicitly clear one session's trace and re-seed its root/roster.

        Normal prompts, browser refresh and Web Stop append/preserve the trace.
        This maintenance reset never touches global cross-run knowledge.
        """
        with self._lock:
            self._store.clear(self.run_id)
            self._seeded = False
        self.ensure_seeded()

    # ── growth (used by the plugin) ───────────────────────────────────────────
    def add_node(self, **kwargs: Any) -> None:
        self.ensure_seeded()
        self._store.add_node(Node(run_id=self.run_id, **kwargs))

    def add_edge(self, src: str, dst: str, type: str = "caused_by") -> None:
        self._store.add_edge(Edge(run_id=self.run_id, src=src, dst=dst, type=type))

    def set_status(self, node_id: str, **kwargs: Any) -> None:
        self._store.set_status(node_id, StatusUpdate(run_id=self.run_id, **kwargs))

    # ── reads (used by the agent tools + root injection) ──────────────────────
    def full(self) -> dict:
        self.ensure_seeded()
        return self._store.full(self.run_id)

    def tool_vs_coder(self) -> Dict[str, Any]:
        """Whether this run used a tool from the catalogue or wrote code from
        scratch. See :func:`CoScientist.graph.projection.tool_vs_coder`."""
        from CoScientist.graph.projection import tool_vs_coder

        return tool_vs_coder(self.full())

    def agents_info(self) -> List[Dict[str, Any]]:
        """Structured info about every agent in the system (the roster).

        Read from configuration, not from the graph: the roster is the same for
        every session, and keeping it out of the trace leaves the graph showing
        only what a session actually did.
        """
        try:
            return _roster_from_config()
        except Exception:  # noqa: BLE001 — the graph must never break the app
            return []

    def history(self, limit: int = 40) -> List[Dict[str, Any]]:
        """Chronological list of what has happened so far (compact)."""
        events = [n for n in self.full()["nodes"] if n.get("kind") in _HISTORY_KINDS]
        events.sort(key=lambda n: n.get("t_start") or 0.0)
        out = []
        for n in events[-limit:]:
            out.append(
                {
                    "id": n.get("id"),
                    "kind": n.get("kind"),
                    "agent": n.get("executor_agent"),
                    "label": n.get("label"),
                    "status": n.get("status"),
                    "output": _short(n.get("output"), 200),
                }
            )
        return out

    def root_summary(self, history_limit: int = 12) -> str:
        """Text the orchestrator/planner get up front: the agent roster plus a
        short trace of what already happened in this session."""
        agents = self.agents_info()
        lines = ["KNOWLEDGE GRAPH — system root.", "", "Agents in the system:"]
        for a in agents:
            subs = ", ".join(a.get("subordinates") or []) or "—"
            tools = ", ".join(a.get("tools") or []) or "—"
            root_tag = " (root)" if a.get("is_root") else ""
            lines.append(
                f"- {a.get('name')}{root_tag}: {_short(a.get('description'), 160)} "
                f"| tools: {tools} | delegates to: {subs}"
            )
        hist = self.history(limit=history_limit)
        if hist:
            lines += ["", "Recent activity in this session:"]
            for h in hist:
                tail = f" → {h['output']}" if h.get("output") else ""
                lines.append(f"- [{h.get('agent') or h.get('kind')}] {_short(h.get('label'), 120)}{tail}")
        lines += [
            "",
            "You can inspect the full graph at any time with the graph tools "
            "(read_research_graph / get_graph_history / get_agents_info).",
        ]
        return "\n".join(lines)


def _short(value: Any, n: int = 200) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else str(value)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


# Legacy/default graph used by standalone helpers and tests without an ADK
# context. Web and agent callbacks resolve through ``get_knowledge_graph``.
knowledge_graph = KnowledgeGraph()
_knowledge_graphs: Dict[SessionKey, KnowledgeGraph] = {
    DEFAULT_SESSION_KEY: knowledge_graph,
}
_registry_lock = threading.RLock()


def get_knowledge_graph(
    context: Any = None,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> KnowledgeGraph:
    """Return the execution graph belonging to one ADK user/session."""
    key = session_key(context, user_id=user_id, session_id=session_id)
    with _registry_lock:
        graph = _knowledge_graphs.get(key)
        if graph is None:
            directory = storage_dir(
                os.getenv("GRAPH_SNAPSHOT_DIR", "./graph_runs"),
                key,
            )
            graph = KnowledgeGraph(run_id="execution", snapshot_dir=str(directory))
            _knowledge_graphs[key] = graph
        return graph


def reset_knowledge_graph(
    context: Any = None,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Clear only the selected session's execution trace."""
    get_knowledge_graph(
        context,
        user_id=user_id,
        session_id=session_id,
    ).reset_session()
