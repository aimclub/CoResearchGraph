"""Dynamically-provisioned MCP toolset for a plain ReAct agent.

This exposes, AS NATIVE ADK TOOLS, the same MCP servers that FEDOT.MAS receives
dynamically — the ones the tool-prep pipeline (ToolPreparer → retrieval/rerank/
deploy) selected for the current task and left in session state
(``filtered_tools`` + ``deployed_mcps``). ``get_tools`` is re-evaluated by ADK
each turn with the live state, so a ReAct LlmAgent simply sees and calls those
tools itself (think → call → observe → repeat) instead of delegating to
FEDOT.MAS.

Read-side mirror of ``FedotMASToolset.fedot_tool`` server resolution, so the two
stay equivalent. Best-effort: an unreachable server is skipped, never fatal.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from google.adk.tools import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

logger = logging.getLogger(__name__)


def _disable_adk_mcp_mtls_probe() -> None:
    """Skip ADK's Google-mTLS client-certificate probe on every MCP connect.

    ADK's ``MCPSessionManager`` runs ``google.auth.default()`` +
    ``configure_mtls_channel()`` before each connection unless
    ``GOOGLE_API_USE_CLIENT_CERTIFICATE`` is ``false`` (it defaults to ``true``).
    Plain MCP servers never use Google client certificates, so the probe can
    only fail — after ~12s, on every single connect, which makes ``get_tools``
    roughly 75x slower. Worse, it runs while the manager holds its per-loop
    session lock, so a multi-agent run spends that window serialised behind it.

    ``setdefault`` so a deployment that genuinely uses GCP mTLS can opt back in.
    The value is read at call time, so setting it here takes effect for every
    entry point that ends up building an MCP toolset.
    """
    os.environ.setdefault("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")


_disable_adk_mcp_mtls_probe()

_SSE_READ_TIMEOUT = 60 * 5.0


class DynamicMCPToolset(BaseToolset):
    """Surfaces the task's dynamically-selected MCP tools to a ReAct agent."""

    def __init__(self, prefix: Optional[str] = None) -> None:
        super().__init__(tool_name_prefix=prefix)
        # url -> McpToolset, cached so we don't reconnect every turn.
        self._by_url: Dict[str, McpToolset] = {}

    async def _server_urls(self, state: dict) -> Dict[str, str]:
        """{url: name} for every MCP server selected for this task."""
        urls: Dict[str, str] = {}

        # Web MCPs deployed by WebToolsDeployerAgent (dicts already carry a url).
        for s in (state.get("deployed_mcps") or []):
            if s.get("url"):
                urls[s["url"]] = s.get("name", s["url"])

        # Retrieved/filtered local tools reference a server_id -> resolve to a url.
        server_ids = {t["server_id"] for t in (state.get("filtered_tools") or []) if t.get("server_id")}
        if server_ids:
            try:
                from rag_tools.storage import PostgresClient
                from rag_tools.config.settings import get_settings as _rag_settings
                pg = PostgresClient(_rag_settings().postgres)
                await pg.initialize()
                try:
                    for sid in server_ids:
                        srv = await pg.get_server(sid)
                        if srv is not None and getattr(srv, "protocol", None) == "http" and getattr(srv, "url", None):
                            urls[srv.url] = getattr(srv, "name", srv.url)
                finally:
                    await pg.close()
            except Exception as exc:  # noqa: BLE001 — never break the agent over tool discovery
                logger.warning("DynamicMCPToolset: server resolution failed: %s", exc)

        return urls

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> List[BaseTool]:
        state = dict(getattr(readonly_context, "state", {}) or {})
        try:
            urls = await self._server_urls(state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DynamicMCPToolset: %s", exc)
            return []

        tools: List[BaseTool] = []
        for url in urls:
            ts = self._by_url.get(url)
            if ts is None:
                ts = McpToolset(
                    connection_params=StreamableHTTPConnectionParams(url=url, sse_read_timeout=_SSE_READ_TIMEOUT)
                )
                self._by_url[url] = ts
            try:
                tools.extend(await ts.get_tools(readonly_context))
            except Exception as exc:  # noqa: BLE001 — skip a dead server, keep the rest
                logger.warning("DynamicMCPToolset: %s unreachable: %s", url, exc)
        return tools

    async def close(self) -> None:
        for ts in self._by_url.values():
            try:
                await ts.close()
            except Exception:  # noqa: BLE001
                pass


dynamic_mcp_toolset_instance = DynamicMCPToolset()
