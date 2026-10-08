"""Pydantic models for the Research Context Graph (spec §1.1–1.2).

Distinct names from CoScientist.graph.models (the execution graph) on purpose.
Agent-facing commit INPUT stays plain dicts (validated by schema.py + the
store); these models are the canonical persisted shape.
"""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class ResearchNode(BaseModel):
    id: str
    type: str
    attrs: Dict[str, Any] = Field(default_factory=dict)
    status: str
    source: str                      # which agent/human created the node
    created_at: float
    updated_at: float
    # Every status change is appended here — the spec's layer-5 "transitions
    # must be logged" requirement: [{from, to, source, at, reason}].
    status_history: List[Dict[str, Any]] = Field(default_factory=list)


class ResearchEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str                          # "<type>:<from>-><to>"
    type: str
    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")
    attrs: Dict[str, Any] = Field(default_factory=dict)
    source: str
    created_at: float


class CommitResult(BaseModel):
    ok: bool
    message: str = ""
    # Echo of what was written: {"nodes": [{ref?, id, type, status}], "edges":
    # [...], "status_updates": [{id, from, to}]}.
    committed: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    graph_stats: Dict[str, int] = Field(default_factory=dict)
    hint: str = ""
