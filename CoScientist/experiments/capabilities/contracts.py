"""Data compatibility based on declarations, never similarity of tool names."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def capability_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("data_contract")
    if not isinstance(raw, Mapping):
        schema = row.get("input_schema") or {}
        raw = schema.get("x-data-contract") if isinstance(schema, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    result = {}
    if raw.get("input_mode") in {"fixed_dataset", "caller_data", "unknown"}:
        result["input_mode"] = raw["input_mode"]
    if isinstance(raw.get("dataset_scope"), str) and raw["dataset_scope"].strip():
        result["dataset_scope"] = raw["dataset_scope"].strip()[:300]
    return result


def contract_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep structured metadata through inventory projections."""
    out = {}
    if contract := capability_contract(row):
        out["data_contract"] = contract
    schema = row.get("output_schema") or row.get("outputSchema")
    if isinstance(schema, Mapping):
        out["output_schema"] = copy.deepcopy(dict(schema))
    return out


def task_dataset_scopes(task: Any) -> set[str]:
    row = task.model_dump(mode="json") if hasattr(task, "model_dump") else task
    dataset = (row.get("design") or {}).get("dataset") or {}
    values = [dataset.get("dataset_scope")]
    values.extend(ref.get("dataset_scope") for ref in row.get("input_data") or []
                  if isinstance(ref, Mapping) and ref.get("required", True))
    return {str(value).strip().casefold() for value in values if value and str(value).strip()}


def dataset_mismatch(task: Any, capability: Mapping[str, Any]) -> dict[str, Any] | None:
    contract = capability_contract(capability)
    required_scopes = task_dataset_scopes(task)
    actual = str(contract.get("dataset_scope") or "").strip().casefold()
    # Training on a different corpus does not forbid a tool accepting caller
    # data. Only an explicit fixed-dataset contract proves incompatibility.
    if (contract.get("input_mode") == "fixed_dataset" and actual and required_scopes
            and required_scopes != {actual}):
        return {
            "code": "tool_dataset_scope_mismatch", "tool": capability.get("tool") or capability.get("name"),
            "server_id": capability.get("server_id"), "required_scopes": sorted(required_scopes),
            "tool_dataset_scope": actual,
        }
    return None


def bound_dataset_mismatches(task: Any, inventory: Any) -> list[dict[str, Any]]:
    """Registry declarations win over planner-written MCP metadata."""
    by_pair = {(str(row.get("server_id") or ""), str(row.get("tool") or row.get("name") or "")): row
               for row in inventory if isinstance(row, Mapping)}
    dumped = task.model_dump(mode="json") if hasattr(task, "model_dump") else task
    found = []
    for server in dumped.get("mcp_servers") or []:
        for tool in server.get("tools") or []:
            name = tool.get("name") if isinstance(tool, Mapping) else str(tool)
            row = by_pair.get((str(server.get("server_id") or ""), name))
            if row is not None and (mismatch := dataset_mismatch(dumped, row)):
                found.append(mismatch)
    return found
