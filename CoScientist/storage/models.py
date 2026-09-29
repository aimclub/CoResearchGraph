from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class RetrievalFinalResult(BaseModel):
    """Result from a retrieval query."""
    servers_id: List[str] = Field(default_factory=list, description="List of unique identifier of selected MCP servers")
    queries: List[str] = Field(default_factory=list, description="Queries used to retrieve MCP servers")
    task: str = Field(..., description="Original task")

class RetrievalToolResult(BaseModel):
    tool: str
    server_id: str
    description: str
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema of the tool's accepted arguments (from the registry).",
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Declared result schema; absence means unknown, not an empty result.",
    )
    data_contract: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Registry-declared input mode and dataset scope, when available.",
    )
    score: float
    url: Optional[str] = Field(
        default=None,
        description="HTTP MCP endpoint for this server, when the registry has one.",
    )

class ToolScore(BaseModel):
    index: int
    score: float

class ToolRanking(BaseModel):
    tools: List[ToolScore]


class MCPScore(BaseModel):
    index: int
    # Nominally a binary deploy flag, but the prompt sits next to a reranker that
    # asks for 0.0-1.0 relevance and models do drift between the two. Accept
    # either; the callback that consumes it takes truthiness anyway.
    score: Union[bool, float]

class MCPRanking(BaseModel):
    mcp_scores: List[MCPScore]
    reasoning: str
