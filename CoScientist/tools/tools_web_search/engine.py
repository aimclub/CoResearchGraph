import re
import httpx

import asyncio

from CoScientist.tools.tools_web_search.models import MCPSearchResult, FoundMCPServer, SourceError
from CoScientist.tools.tools_web_search.adapters import _McpServersCom, _McpServersOrg


class MCPSearchTool:
    """
    Async search across multiple MCP server registries.

    Typical usage from an agent::

        tool = MCPSearchTool()
        result = await tool.search("youtube")
        return result.to_agent_text()
    """

    DEFAULT_TIMEOUT = 15.0
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; mcp-search/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    }

    def __init__(
        self,
        *,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
        enable_com: bool = True,
        enable_org: bool = True,
    ):
        self._timeout = timeout or self.DEFAULT_TIMEOUT
        self._external_client = client
        self._client: httpx.AsyncClient | None = client
        self._enable_com = enable_com
        self._enable_org = enable_org

    async def __aenter__(self) -> "MCPSearchTool":
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers=self.DEFAULT_HEADERS,
                follow_redirects=True,
            )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._external_client is None and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def search(self, query: str) -> MCPSearchResult:
        """Run the query against every enabled source concurrently and merge results."""
        if not query or not query.strip():
            return MCPSearchResult(query=query, total_found=0, sources_used=[])

        manage_client = self._client is None
        if manage_client:
            await self.__aenter__()
        try:
            sources: list[_Source] = []
            if self._enable_com:
                sources.append(_McpServersCom(self._client))  # type: ignore[arg-type]
            if self._enable_org:
                sources.append(_McpServersOrg(self._client))  # type: ignore[arg-type]

            results = await asyncio.gather(
                *(src.search(query) for src in sources),
                return_exceptions=True,
            )

            servers: list[FoundMCPServer] = []
            sources_used: list[str] = []
            errors: list[SourceError] = []
            for src, res in zip(sources, results):
                if isinstance(res, Exception):
                    errors.append(SourceError(source=src.name, error=f"{type(res).__name__}: {res}"))
                else:
                    servers.extend(res)
                    sources_used.append(src.name)

            servers = self._dedupe(servers)
            return MCPSearchResult(
                query=query,
                servers=servers,
                total_found=len(servers),
                sources_used=sources_used,
                errors=errors,
            )
        finally:
            if manage_client:
                await self.__aexit__(None, None, None)

    @staticmethod
    def _dedupe(servers: list[FoundMCPServer]) -> list[FoundMCPServer]:
        """Drop duplicates that point at the same repo across registries."""
        seen: set[str] = set()
        out: list[FoundMCPServer] = []
        for s in servers:
            key = str(s.repo_url).lower() if s.repo_url else f"{s.source}:{s.slug}"
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out