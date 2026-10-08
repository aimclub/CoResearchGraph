"""Central Graph Memory Service — FastAPI over a per-run NetworkX store.

Every A2A server's emitter POSTs nodes/edges/status here; the orchestrator GETs a
summary. Run it with:  python -m CoScientist.graph.service
"""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException

from CoScientist.graph.config import GRAPH_PORT, GRAPH_SNAPSHOT_DIR
from CoScientist.graph.models import Edge, Node, StatusUpdate
from CoScientist.graph.projection import local_view, orchestrator_summary
from CoScientist.graph.store import GraphStore

app = FastAPI(title="CoScientist Execution Graph")
_store = GraphStore(snapshot_dir=GRAPH_SNAPSHOT_DIR)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "runs": _store.run_ids()}


@app.post("/node")
def add_node(node: Node) -> dict:
    _store.add_node(node)
    return {"ok": True, "id": node.id}


@app.post("/edge")
def add_edge(edge: Edge) -> dict:
    _store.add_edge(edge)
    return {"ok": True}


@app.post("/node/{node_id}/status")
def set_status(node_id: str, upd: StatusUpdate) -> dict:
    if not _store.set_status(node_id, upd):
        raise HTTPException(status_code=404, detail=f"node {node_id} not found")
    return {"ok": True}


@app.get("/summary")
def summary(run_id: str, role: str = "orchestrator", node_id: str = "") -> dict:
    full = _store.full(run_id)
    if role == "orchestrator":
        text = orchestrator_summary(full)
    else:
        text = local_view(full, node_id) if node_id else ""
    return {"run_id": run_id, "role": role, "summary": text}


@app.get("/graph")
def graph(run_id: str) -> dict:
    return _store.full(run_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=GRAPH_PORT, log_level="warning")
