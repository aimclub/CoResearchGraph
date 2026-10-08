"""Thin async client used by emitters / the orchestrator to talk to the graph
service. Best-effort: if the service is down or GRAPH_ENABLED is off, every call
is a no-op so the agents never break because of the graph.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import httpx

from CoScientist.graph.config import GRAPH_ENABLED, GRAPH_URL

logger = logging.getLogger(__name__)

_TIMEOUT = 3.0


async def _post(path: str, json: dict) -> Optional[dict]:
    if not GRAPH_ENABLED:
        return None
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.post(f"{GRAPH_URL}{path}", json=json)
            r.raise_for_status()
            return r.json()
    except Exception as exc:  # noqa: BLE001 — graph must never break the agent
        logger.debug("graph POST %s failed: %s", path, exc)
        return None


async def _get(path: str, params: dict) -> Optional[dict]:
    if not GRAPH_ENABLED:
        return None
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.get(f"{GRAPH_URL}{path}", params=params)
            r.raise_for_status()
            return r.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("graph GET %s failed: %s", path, exc)
        return None


async def emit_node(
    *,
    run_id: str,
    node_id: str,
    kind: str,
    label: str = "",
    executor_agent: Optional[str] = None,
    status: str = "running",
    parent_ids: Optional[List[str]] = None,
    input: Any = None,
    input_files: Optional[List[str]] = None,
    t_start: Optional[float] = None,
) -> None:
    await _post("/node", {
        "id": node_id,
        "run_id": run_id,
        "kind": kind,
        "label": label,
        "executor_agent": executor_agent,
        "status": status,
        "parent_ids": parent_ids or [],
        "input": input,
        "input_files": input_files or [],
        "t_start": t_start,
    })


async def emit_edge(*, run_id: str, src: str, dst: str, type: str = "caused_by") -> None:
    await _post("/edge", {"run_id": run_id, "src": src, "dst": dst, "type": type})


async def set_status(
    *,
    run_id: str,
    node_id: str,
    status: Optional[str] = None,
    output: Optional[str] = None,
    output_files: Optional[List[str]] = None,
    verdict: Optional[str] = None,
    t_end: Optional[float] = None,
) -> None:
    await _post(f"/node/{node_id}/status", {
        "run_id": run_id,
        "status": status,
        "output": output,
        "output_files": output_files,
        "verdict": verdict,
        "t_end": t_end,
    })


async def get_summary(*, run_id: str, role: str = "orchestrator", node_id: str = "") -> str:
    res = await _get("/summary", {"run_id": run_id, "role": role, "node_id": node_id})
    return (res or {}).get("summary", "") or ""
