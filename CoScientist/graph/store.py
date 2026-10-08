"""In-memory NetworkX store for execution graphs, one DiGraph per run_id.

MVP backend. Snapshots each run to JSON for persistence/replay (Fact 1). The
public methods are storage-engine agnostic so a Neo4j backend can replace this
without touching emitters/projection.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional

import networkx as nx

from CoScientist.graph.models import Edge, Node, StatusUpdate

logger = logging.getLogger(__name__)


class GraphStore:
    def __init__(self, snapshot_dir: Optional[str] = None) -> None:
        self._graphs: Dict[str, nx.DiGraph] = {}
        self._loaded_run_ids: set[str] = set()
        # A snapshot that could not be read must not be silently replaced by the
        # first write performed through a newly-created store. ``clear`` is the
        # explicit escape hatch for intentionally discarding such a snapshot.
        self._snapshot_load_failures: set[str] = set()
        self._lock = threading.Lock()
        self._snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        if self._snapshot_dir:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _g(self, run_id: str) -> nx.DiGraph:
        if run_id not in self._loaded_run_ids:
            self._load_snapshot_unlocked(run_id)
            self._loaded_run_ids.add(run_id)
        g = self._graphs.get(run_id)
        if g is None:
            g = nx.DiGraph()
            self._graphs[run_id] = g
        return g

    # ── writes ──────────────────────────────────────────────────────────────
    def add_node(self, node: Node) -> None:
        with self._lock:
            g = self._g(node.run_id)
            data = node.model_dump()
            if g.has_node(node.id):
                # merge: keep existing fields unless the new payload sets them
                existing = g.nodes[node.id]
                for k, v in data.items():
                    if v is not None or k not in existing:
                        existing[k] = v
            else:
                g.add_node(node.id, **data)
            self._snapshot(node.run_id)

    def add_edge(self, edge: Edge) -> None:
        with self._lock:
            g = self._g(edge.run_id)
            # ensure endpoints exist as placeholders so an edge never dangles
            for nid in (edge.src, edge.dst):
                if not g.has_node(nid):
                    g.add_node(nid, id=nid, run_id=edge.run_id, kind="tool_call", status="running")
            g.add_edge(edge.src, edge.dst, type=edge.type)
            self._snapshot(edge.run_id)

    def set_status(self, node_id: str, upd: StatusUpdate) -> bool:
        with self._lock:
            g = self._g(upd.run_id)
            if not g.has_node(node_id):
                return False
            n = g.nodes[node_id]
            if upd.status is not None:
                n["status"] = upd.status
            if upd.output is not None:
                n["output"] = upd.output
            if upd.output_files is not None:
                n["output_files"] = upd.output_files
            if upd.verdict is not None:
                n["verdict"] = upd.verdict
            if upd.t_end is not None:
                n["t_end"] = upd.t_end
            self._snapshot(upd.run_id)
            return True

    # ── reads ───────────────────────────────────────────────────────────────
    def _full_unlocked(self, run_id: str) -> dict:
        g = self._graphs.get(run_id)
        if g is None:
            return {"run_id": run_id, "nodes": [], "edges": []}
        nodes = [dict(g.nodes[n]) for n in g.nodes]
        edges = [{"src": u, "dst": v, "type": d.get("type")} for u, v, d in g.edges(data=True)]
        return {"run_id": run_id, "nodes": nodes, "edges": edges}

    def full(self, run_id: str) -> dict:
        with self._lock:
            self._g(run_id)
            return self._full_unlocked(run_id)

    def run_ids(self) -> list:
        with self._lock:
            return list(self._graphs.keys())

    def clear(self, run_id: str) -> None:
        """Explicitly drop one run in memory and replace its snapshot."""
        with self._lock:
            # Do not let the following snapshot operation lazy-load the graph
            # that is being deliberately cleared.
            self._loaded_run_ids.add(run_id)
            self._graphs.pop(run_id, None)
            self._snapshot_load_failures.discard(run_id)
            self._snapshot(run_id)  # writes an empty graph for this run

    # ── persistence ─────────────────────────────────────────────────────────
    def _load_snapshot_unlocked(self, run_id: str) -> None:
        """Restore one run on first access.

        Callers hold ``self._lock``. The graph is installed only after the
        whole snapshot has been validated, so a malformed file cannot result in
        a partially restored graph that is then written over the original.
        """
        if not self._snapshot_dir:
            return
        path = self._snapshot_dir / f"{run_id}.json"
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError("snapshot root must be an object")
            nodes = payload.get("nodes", [])
            edges = payload.get("edges", [])
            if not isinstance(nodes, list) or not isinstance(edges, list):
                raise ValueError("snapshot nodes and edges must be arrays")

            graph = nx.DiGraph()
            for raw_node in nodes:
                if not isinstance(raw_node, dict) or not raw_node.get("id"):
                    raise ValueError("every snapshot node must have an id")
                node = dict(raw_node)
                graph.add_node(str(node["id"]), **node)
            for raw_edge in edges:
                if not isinstance(raw_edge, dict) or not raw_edge.get("src") \
                        or not raw_edge.get("dst"):
                    raise ValueError("every snapshot edge must have src and dst")
                edge = dict(raw_edge)
                src = str(edge.pop("src"))
                dst = str(edge.pop("dst"))
                graph.add_edge(src, dst, **edge)
            self._graphs[run_id] = graph
            logger.info("Loaded graph snapshot for %s from %s", run_id, path)
        except Exception as exc:  # noqa: BLE001
            self._snapshot_load_failures.add(run_id)
            logger.warning(
                "Graph snapshot load failed for %s; refusing to overwrite %s: %s",
                run_id,
                path,
                exc,
            )

    def _snapshot(self, run_id: str) -> None:
        if not self._snapshot_dir:
            return
        if run_id in self._snapshot_load_failures:
            logger.warning(
                "Graph snapshot skipped for %s because its existing snapshot "
                "could not be loaded",
                run_id,
            )
            return
        try:
            path = self._snapshot_dir / f"{run_id}.json"
            tmp = path.with_suffix(".json.tmp")
            # NOTE: caller already holds self._lock — use the unlocked read to
            # avoid re-acquiring the non-reentrant lock (would deadlock).
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._full_unlocked(run_id), f, ensure_ascii=False, default=str)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 — persistence must never break writes
            logger.warning("Graph snapshot failed for %s: %s", run_id, exc)
