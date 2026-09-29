"""Tools for websearch / literature research (MCP toolsets)."""
import os
from typing import Any, Callable, Optional

from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

from CoScientist.config import get_settings
from CoScientist.tools.mcp_resilience import ResilientMcpToolset
from CoScientist.utils.selective_proxy import create_mcp_proxy_httpx_factory

settings = get_settings()
PAPER_ANALYSIS_URL = settings.mcp.paper_analysis_url
PAPERS_SEARCH_URL = settings.mcp.papers_search_url
VAULT_URL = settings.mcp.vault_url


# ADK applies ``StreamableHTTPConnectionParams.timeout`` to *both* tool
# discovery and calls.  Its 5-second default is too short for remote MCPs on
# the first request, causing an entire toolset to disappear from an agent.
_MCP_REQUEST_TIMEOUT = float(os.getenv("MCP__REQUEST_TIMEOUT", "60"))
_PAPER_ANALYSIS_TIMEOUT = float(os.getenv("MCP__PAPER_ANALYSIS_TIMEOUT", str(60 * 15)))

def _http_mcp_toolset(
    url: Optional[str],
    sse_read_timeout: float = 60 * 5.0,
    timeout: Optional[float] = None,
    headers: Optional[dict] = None,
    tool_filter: Optional[list] = None,
    httpx_client_factory: Optional[Callable] = None,
    retry_calls: bool = False,
) -> Optional[McpToolset]:
    """Build an HTTP MCP toolset, or None when the URL is not configured.

    Returning None (instead of crashing at import on a missing URL) lets the app
    start without these optional services; the ResearchAgent simply runs without
    the corresponding toolset. Set the URLs in .env to enable them.

    ``tool_filter`` names the tools to keep. A server may expose more than an
    agent should see (see the vault below).

    Every toolset lists its tools resiliently (see tools/mcp_resilience.py).
    ``retry_calls`` also repeats a call whose connection was lost — set it only
    for a server whose tools are read-only, since the lost call may have run.
    """
    if not url:
        return None
    conn_kwargs: dict[str, Any] = {
        "url": url,
        "timeout": timeout if timeout is not None else _MCP_REQUEST_TIMEOUT,
        "sse_read_timeout": sse_read_timeout,
        "headers": headers or {},
    }
    if httpx_client_factory is not None:
        conn_kwargs["httpx_client_factory"] = httpx_client_factory
    return ResilientMcpToolset(
        connection_params=StreamableHTTPConnectionParams(**conn_kwargs),
        tool_filter=tool_filter,
        retry_calls=retry_calls,
    )


# ---------------------------------------------------------------------------
# Tavily websearch — optionally proxied via SERVICES__PROXY_URL
# ---------------------------------------------------------------------------
_tavily_conn_kwargs: dict[str, Any] = {
    "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={settings.services.tavily_api_key}",
}

if settings.services.proxy_url:
    _tavily_conn_kwargs["httpx_client_factory"] = create_mcp_proxy_httpx_factory(
        settings.services.proxy_url,
        enabled_fn=get_settings().web.use_proxy,
    )

# Search only — a call lost with its connection is safe to repeat.
websearch_toolset_instance = ResilientMcpToolset(
    connection_params=StreamableHTTPConnectionParams(**_tavily_conn_kwargs),
    retry_calls=True,
)

# Optional paper-analysis / paper-search MCP servers — only built when configured
# (MCP__PAPER_ANALYSIS_URL / MCP__PAPERS_SEARCH_URL in .env).
paper_analysis_toolset_instance = _http_mcp_toolset(
    PAPER_ANALYSIS_URL,
    sse_read_timeout=_PAPER_ANALYSIS_TIMEOUT,
    timeout=_PAPER_ANALYSIS_TIMEOUT,
)

# Per-user OpenAlex credentials forwarded as HTTP headers so the shared remote
# container uses each caller's own rate-limit quota instead of the server's env.
_openalex_headers = {
    k: v
    for k, v in {
        "x-openalex-email": settings.services.openalex_email,
        "x-openalex-api-key": settings.services.openalex_api_key,
    }.items()
    if v
}
papers_search_toolset_instance = _http_mcp_toolset(
    PAPERS_SEARCH_URL, headers=_openalex_headers, retry_calls=True,
)


# ---------------------------------------------------------------------------
# The file vault — worker surface only
# ---------------------------------------------------------------------------
# The server also exposes promote_artifact, cleanup_session,
# update_artifact_metadata, get_session_manifest and list_artifacts. Those are
# framework tools: they promote, delete, or read across a whole session. An
# agent gets the two that move one file, and nothing else. Framework code calls
# the rest through tools/vault_client.py, which does not go through an agent.
#
# get_upload_link and get_download_link both declare user_id / session_id, so
# SessionScopePlugin fills them at the tool boundary and the model never
# supplies a scope.
VAULT_WORKER_TOOLS = ["get_upload_link", "get_download_link"]

vault_toolset_instance = _http_mcp_toolset(VAULT_URL, tool_filter=VAULT_WORKER_TOOLS)
