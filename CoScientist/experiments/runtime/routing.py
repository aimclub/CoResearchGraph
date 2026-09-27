"""Inventory cover and MCP URL fill for Experiment Module start/fallback."""
from __future__ import annotations

from typing import Any, Mapping

from CoScientist.experiments.critique.coverage import task_coverage_blob as coverage_blob
from CoScientist.experiments.capabilities.contracts import dataset_mismatch
from CoScientist.experiments.runtime.shared import session_inventory_rows
from CoScientist.experiments.schemas import ExecutionRoute, ExperimentTask


def session_inventory_nonempty(state: Mapping[str, Any]) -> bool:
    from CoScientist.experiments.capabilities.inventory import inventory_nonempty

    return inventory_nonempty(session_inventory_rows(state, scoped=True))


def _ops_index_from_state(state: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    context = state.get("experiment_context") if hasattr(state, "get") else None
    ops = context.get("operations") if isinstance(context, dict) else None
    if not isinstance(ops, list):
        return {}
    out: dict[str, dict[str, str]] = {}
    for op in ops:
        if not isinstance(op, dict):
            continue
        oid = str(op.get("operation_id") or "").strip().upper()
        if oid:
            out[oid] = op
    return out


def task_coverage_blob(state: Mapping[str, Any], task: ExperimentTask) -> str:
    """Frame slot first; leftover inventory names in task prose cannot fake cover."""
    return coverage_blob(task, _ops_index_from_state(state))


def inventory_covers_task(state: Mapping[str, Any], task: ExperimentTask) -> bool:
    """This-run retrieve covers the task: named, primary family, or bound tool."""
    from CoScientist.experiments.capabilities.inventory import (
        index_inventory_tools,
        match_inventory_tool,
        match_named_inventory_tool,
    )

    by_tool = index_inventory_tools([
        row for row in session_inventory_rows(state, scoped=False) if not dataset_mismatch(task, row)
    ])
    if not by_tool:
        return False
    blob = task_coverage_blob(state, task)
    if match_named_inventory_tool(blob, by_tool) is not None:
        return True
    if match_inventory_tool(blob, by_tool) is not None:
        return True
    for server in task.mcp_servers:
        for tool in server.tools:
            name = str(getattr(tool, "name", "") or "").strip()
            if name and name in by_tool:
                return True
    return False


def mcp_routes_tried(task_runtime: Mapping[str, Any]) -> bool:
    attempts = task_runtime.get("attempts") or {}
    used = {
        str((attempts.get(aid) or {}).get("route") or "")
        for aid in task_runtime.get("attempt_order") or []
    }
    return bool(used & {
        ExecutionRoute.FEDOT_MAS.value,
        ExecutionRoute.REACT_TOOLS.value,
    })


def match_session_inventory_tool(
    state: Mapping[str, Any], task: ExperimentTask, blob: str,
) -> dict[str, Any] | None:
    """Named, family-covered, or already-bound tool from session inventory."""
    from CoScientist.experiments.capabilities.inventory import (
        index_inventory_tools,
        match_named_inventory_tool,
    )

    by_tool = index_inventory_tools([
        row for row in session_inventory_rows(state, scoped=True) if not dataset_mismatch(task, row)
    ])
    matched = match_named_inventory_tool(blob, by_tool)
    if matched is not None:
        return matched
    for server in task.mcp_servers:
        for tool in server.tools:
            name = str(getattr(tool, "name", "") or "").strip()
            if name and name in by_tool:
                return by_tool[name]
    return None

