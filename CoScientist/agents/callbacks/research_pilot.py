"""Narrow, schema-aware repairs for the scientific pilot's literature tools."""

import re

from google.adk.tools.mcp_tool import McpTool


_ROSTER_KEY = "_pilot_research_tool_roster"
_CHANNEL_SUFFIX = re.compile(
    r"([A-Za-z_][A-Za-z_0-9]*)<\|channel\|>(?:commentary|analysis|final)\Z"
)


def capture_research_tool_roster(callback_context, llm_request):
    """Remember the declarations offered in this exact ResearchAgent model turn."""
    names = {
        declaration.name
        for tool in llm_request.config.tools or []
        for declaration in tool.function_declarations or []
        if declaration.name
    }
    callback_context.state[_ROSTER_KEY] = {
        "invocation_id": callback_context._invocation_context.invocation_id,
        "names": sorted(names),
    }
    return None


def normalize_research_function_name(callback_context, llm_response):
    """Remove a leaked channel suffix only when the exact base tool was offered."""
    if llm_response.partial:
        return None
    roster = callback_context.state.get(_ROSTER_KEY) or {}
    if roster.get("invocation_id") != callback_context._invocation_context.invocation_id:
        return None
    offered = set(roster.get("names") or [])
    for part in getattr(llm_response.content, "parts", None) or []:
        call = getattr(part, "function_call", None)
        name = getattr(call, "name", None)
        match = _CHANNEL_SUFFIX.fullmatch(name) if isinstance(name, str) else None
        if match and match.group(1) in offered:
            call.name = match.group(1)
    return None


def normalize_paper_download_args(tool, args, tool_context):
    """Omit unsupported PDF filter and optional nulls before MCP validation."""
    if (not isinstance(tool, McpTool)
            or tool.name != "download_papers_from_search"
            or not isinstance(args, dict)):
        return None
    schema = tool.raw_mcp_tool.inputSchema or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    if properties and "has_pdf" not in properties:
        args.pop("has_pdf", None)
    for key, value in list(args.items()):
        if value is None and key not in required:
            args.pop(key)
    return None
