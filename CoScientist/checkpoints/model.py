"""Pydantic models for checkpoint manifests.

A checkpoint is a single zip bundle: ``manifest.json`` at the root plus
content-addressed blobs under ``blobs/``. The manifest is the durable contract
(forward-compatible with the Synapse snapshot schema, see SynapseNmas.md §6);
the zip layout is a local storage detail.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class SessionRef(BaseModel):
    """Where the ADK session lived and which blobs hold its export."""

    app_name: str
    user_id: str
    session_id: str
    event_count: int = 0


class HitlPending(BaseModel):
    """A human review that was about to be shown when the snapshot was taken.

    Restore re-presents the review instead of resuming a mid-await coroutine.
    """

    agent: str
    kind: str = "approval"
    payload: Dict[str, Any] = Field(default_factory=dict)


class CheckpointManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    checkpoint_id: str
    label: str
    run_id: str
    parent_checkpoint_id: Optional[str] = None
    created_at: str
    reason: str = "module_boundary"
    resume_from: Dict[str, Any] = Field(default_factory=dict)
    session: SessionRef
    # logical blob name -> path inside the bundle (e.g. "blobs/sha256-ab12…").
    # Logical names: session_events, session_state, task_tracker,
    # research_graph, execution_graph, knowledge_memory.
    blobs: Dict[str, str] = Field(default_factory=dict)
    hitl_pending: Optional[HitlPending] = None
    external: Dict[str, Any] = Field(default_factory=dict)
    validator_pending: bool = False
    pins: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    snapshot_ref: Optional[str] = None   # Synapse v1: platform-held reference to the bundle
