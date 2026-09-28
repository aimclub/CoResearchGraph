"""Narrow, schema-aware repairs for the scientific pilot's literature tools."""

from google.adk.tools.mcp_tool import McpTool


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
