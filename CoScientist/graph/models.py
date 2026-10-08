"""Node/edge schema for the Dynamic Execution Graph (see docs/execution_graph.md).

Raw agent-call granularity for the MVP; the optional `semantic` slot is filled
later by the step-abstraction layer.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

NodeKind = Literal[
    "system",      # root: the whole MAS (parent of the agent roster)
    "agent",       # a roster entry: one agent and its capabilities
    "goal",        # a user query / top-level objective
    "agent_call",  # a delegation to a sub-agent
    "tool_call",   # an ordinary tool invocation
    "result",      # a produced answer / artifact
    "decision",
    "reflection",
    "entity",      # a knowledge-graph entity (its DOMAIN type lives in semantic.type)
]
NodeStatus = Literal["running", "success", "failed", "interrupted", "pruned"]
# Edge types are open: control-flow uses the names below; the knowledge layer
# adds domain relations (has_property, about, supports, generated_by, …). Kept as
# a free string so new relation types never need a code change.
EdgeType = str


class Semantic(BaseModel):
    type: Optional[str] = None
    goal: Optional[str] = None
    entity: Optional[str] = None


class Node(BaseModel):
    id: str
    run_id: str
    kind: NodeKind
    label: str = ""
    executor_agent: Optional[str] = None
    status: NodeStatus = "running"
    parent_ids: List[str] = Field(default_factory=list)
    input: Optional[Any] = None
    output: Optional[str] = None
    # Files the call read and wrote, as s3://bucket/key. The durable reference,
    # never a presigned URL: a URL in an old snapshot is a dead link, while the
    # key still resolves. Consumers mint a URL with the vault get_download_link.
    input_files: List[str] = Field(default_factory=list)
    output_files: List[str] = Field(default_factory=list)
    verdict: Optional[str] = None  # critic verdict — the reward signal (Fact 1)
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    # The user prompt this node belongs to: ADK's invocation id for the turn.
    # Agent nodes are one per agent for the whole session and carry none; a
    # turn's shape is its goal, the calls under it, and its result, all of which
    # do. Without it the only way to tell turns apart was to parse the goal id.
    turn_id: Optional[str] = None
    semantic: Optional[Semantic] = None


class Edge(BaseModel):
    run_id: str
    src: str
    dst: str
    type: EdgeType = "caused_by"


class StatusUpdate(BaseModel):
    run_id: str
    status: Optional[NodeStatus] = None
    output: Optional[str] = None
    # None leaves the node untouched, so a caller with nothing to report never
    # wipes the references another writer put there.
    output_files: Optional[List[str]] = None
    verdict: Optional[str] = None
    t_end: Optional[float] = None
