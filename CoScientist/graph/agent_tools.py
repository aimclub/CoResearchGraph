"""Read-only knowledge-graph tools available to EVERY agent.

Any agent can, at any point, inspect the shared graph: the history of what has
happened in the session and structured info about all the other agents. Wired
into agents via the ``graph`` tool key (see CoScientist/assembly/bindings.py).

Read-only on purpose — the graph grows automatically from agent activity
(GraphMemoryPlugin), so agents never have to maintain it by hand.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext

from CoScientist.graph.memory import get_knowledge_graph


def read_research_graph(tool_context: ToolContext) -> Dict[str, Any]:
    """Read the shared research/knowledge graph of the whole session.

    Returns the system root, every agent (the roster) and every step taken so
    far (goals, delegations, tool calls, results) with their status. Use it to
    understand what has already happened and avoid repeating work.
    """
    full = get_knowledge_graph(tool_context).full()
    nodes = [
        {
            "id": n.get("id"),
            "kind": n.get("kind"),
            "label": _short(n.get("label"), 160),
            "agent": n.get("executor_agent"),
            "status": n.get("status"),
            "output": _short(n.get("output"), 200),
        }
        for n in full.get("nodes", [])
    ]
    return {"nodes": nodes, "edges": full.get("edges", [])}


def get_graph_history(tool_context: ToolContext, limit: int = 30) -> Dict[str, Any]:
    """Get the chronological history of steps taken in this session.

    Args:
        limit: max number of most-recent events to return.
    Returns:
        A dict with an ``events`` list (goals, delegations, tool calls, results).
    """
    return {"events": get_knowledge_graph(tool_context).history(limit=limit)}


def get_agents_info(tool_context: ToolContext) -> Dict[str, Any]:
    """Get structured info about all agents in the system (the graph root):
    name, description, role, tools and which agents each one can delegate to."""
    return {"agents": get_knowledge_graph(tool_context).agents_info()}




class GraphReaderToolset(BaseToolset):
    """Exposes the read-only graph tools to an agent."""

    def __init__(self, prefix: Optional[str] = None) -> None:
        super().__init__(tool_name_prefix=prefix)

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> List[BaseTool]:
        return get_graph_tools()

    async def close(self) -> None:
        pass


def get_graph_tools() -> List[BaseTool]:
    return [
        FunctionTool(read_research_graph),
        FunctionTool(get_graph_history),
        FunctionTool(get_agents_info),
    ]


def _short(value: Any, n: int = 200) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else str(value)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


# Shared toolset instance (mirrors task_tracker_instance).
graph_reader_instance = GraphReaderToolset()
