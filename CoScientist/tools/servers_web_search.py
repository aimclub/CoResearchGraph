from typing import Any, Dict

from CoScientist.tools.tools_web_search.engine import MCPSearchTool
from CoScientist.tools.tools_web_search.models import MCPSearchResult

from google.adk.tools import ToolContext


_tool = MCPSearchTool()  # share across calls


async def search_mcp_servers(query: str,
                            tool_context: ToolContext = None) -> Dict[str, Any]:
    """
    Search public MCP server registries for servers matching the query.

    Use this tool to discover MCP servers for APIs, integrations, tools,
    databases, browser automation, research workflows, and other capabilities.

    Args:
        query: Natural language query describing the desired MCP server
            functionality or integration.
            Examples:
            - "github"
            - "youtube transcription"
            - "postgres database"
            - "browser automation"

    Returns:
        Compact LLM-friendly text containing up to 15 matching MCP servers,
        including descriptions, metadata, registry pages, and repository links.
    """
    async with _tool:
        result: MCPSearchResult = await _tool.search(query)

    if tool_context is None:
        # No session to accumulate into — still return the search result.
        return {
            "status": "success",
            "result": result.to_agent_text(limit=15),
            "accumulated_count": len(result.servers),
            "message": f"Retrieved {result.total_found} mcp servers (no session accumulation).",
        }

    # Read/write the SAME key so accumulation persists across calls.
    accumulated = tool_context.state.get('accumulated_web_mcps', [])
    existing_mcps = {t['name'] for t in accumulated}
    last_idx = len(accumulated) + 1

    for search_result in result.servers:
        if search_result.name not in existing_mcps:
            search_result_payload = search_result.model_dump()
            search_result_payload.update({'index': last_idx})
            accumulated.append(search_result_payload)
            last_idx += 1

    tool_context.state['accumulated_web_mcps'] = accumulated
    tool_context.state['retrieval_queries_mcp'] = tool_context.state.get('retrieval_queries_mcp', []) + [query]

    return {
        "status": "success",
        "result": result.to_agent_text(limit=15),
        "accumulated_count": len(accumulated),
        "message": f"Retrieved {result.total_found} mcp servers. Total accumulated: {len(accumulated)}."
    }
