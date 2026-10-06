"""Execution identity and infrastructure errors.

Scientific routing does not read tool names. A service outage is an execution
fact: it stops calls to the observed failed capability and is not a verdict.
"""
from __future__ import annotations

from typing import Any, Mapping

INFRASTRUCTURE_CODES = frozenset({
    "service_unavailable",
    "upstream_unavailable",
    "upstream_service_unavailable",
    "infrastructure_failure",
    "connection_refused",
    "connection_error",
    "dns_error",
    "http_502",
    "http_503",
})

STOPS_KEY = "experiment_service_stops"


def is_infrastructure_failure(result: Mapping[str, Any]) -> bool:
    code = str(result.get("error_code") or "").strip().lower()
    return code in INFRASTRUCTURE_CODES


def note_service_stop(state: dict[str, Any], task: Mapping[str, Any], observation: Mapping[str, Any] | None = None) -> None:
    """An observed tool failure does not establish that every tool is down."""
    observation = observation or {}
    if observation.get("is_error") is not True:
        return
    tool = str(observation.get("tool") or "")
    servers = [s for s in task.get("mcp_servers") or []
               if any(t.get("name") == tool for t in s.get("tools") or [])]
    if len(servers) != 1:
        return  # No unambiguous observed capability: do not blacklist a server.
    sid = str(servers[0].get("server_id") or servers[0].get("name") or "")
    if not sid:
        return
    stops = state.setdefault(STOPS_KEY, {})
    task_id = str(task.get("id") or "")
    stop = stops.setdefault(sid, {"tools": {}})
    stop.setdefault("tools", {})[tool] = task_id
    if observation.get("failure_scope") == "server":
        stop.update(task_id=task_id, reason="service_unavailable", scope="server")


def service_block_reason(state: Mapping[str, Any], task: Any) -> str:
    """Block only explicitly affected capabilities, preserving legacy snapshots."""
    stops = state.get(STOPS_KEY) or {}
    if not isinstance(stops, Mapping):
        return ""
    task_id = str(getattr(task, "id", "") or "")
    blocked = []
    for server in getattr(task, "mcp_servers", []) or []:
        sid = str(getattr(server, "server_id", "") or getattr(server, "name", "") or "")
        stop = stops.get(sid) or {}
        if not stop:
            continue
        if stop.get("scope") == "server" or "tools" not in stop:
            if stop.get("task_id") != task_id:
                blocked.append(sid)
            continue
        for tool in getattr(server, "tools", []) or []:
            if not getattr(tool, "required_for_task", True):
                continue
            owner = (stop.get("tools") or {}).get(tool.name)
            if owner and owner != task_id:
                blocked.append(sid + "/" + tool.name)
    return "Service unavailable (" + ", ".join(blocked) + ")." if blocked else ""


def step_contract(design: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    """Identity of one step inside an operation.

    The question text, the task name and the route are not part of it, so a
    rename or a route change does not reset the attempt budget. Two steps of
    one operation differ by ``step_key`` or by their inputs, outputs and criteria.
    """
    return {
        "step_key": str(design.get("step_key") or "").strip(),
        "targets": sorted(str(x) for x in (design.get("target_refs") or []) if str(x).strip()),
        "inputs": [
            {
                "kind": item.get("kind"),
                "source_task_id": item.get("source_task_id"),
                "data_id": item.get("data_id"),
            }
            for item in (task.get("input_data") or [])
            if isinstance(item, Mapping)
        ],
        "outputs": [
            {"role": item.get("role"), "media_type": item.get("media_type")}
            for item in (task.get("expected_artifacts") or [])
            if isinstance(item, Mapping)
        ],
        "criteria": [
            {
                "kind": item.get("kind"),
                "metric": item.get("metric"),
                "operator": item.get("operator"),
                "target": item.get("target"),
            }
            for item in (task.get("success_criteria") or [])
            if isinstance(item, Mapping)
        ],
    }


__all__ = [
    "INFRASTRUCTURE_CODES",
    "STOPS_KEY",
    "is_infrastructure_failure",
    "note_service_stop",
    "service_block_reason",
    "step_contract",
]
