"""A per-call MCP client for the paper database statistics, for the web UI.

The statistics page shows what ``get_papers_database_statistics`` on the
paper-analysis MCP server returns. Agents reach that server through an ADK
``McpToolset`` (see ``research_tools.py``); the web backend is plain framework
code, so - like ``vault_client.py`` - it opens one MCP session per call with the
raw MCP SDK and closes it again.

The tool answers from memory (the server keeps the statistics current in a
background thread), so a call returns quickly and a short deadline is enough.

Nothing here raises. A missing URL, an unreachable server, or a tool error all
come back as a status the page can show.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from CoScientist.config import get_settings

logger = logging.getLogger(__name__)

TOOL_NAME = "get_papers_database_statistics"
# One call moves a few KB of text from memory; anything slower is a fault.
_TIMEOUT = 20.0
# A finished report (chroma_stats.format_paper_statistics on the server) is
# Markdown with section headings; the two English markers match servers that
# still run the earlier plain-text report. Anything else is the server saying
# it is still computing or failed.
_FINISHED_REPORT = re.compile(r"^(## |Unique papers: |No papers found\.)", re.MULTILINE)


def paper_analysis_url() -> Optional[str]:
    """Read at call time, so a test can point this at a local server."""
    return get_settings().mcp.paper_analysis_url


def _describe(exc: BaseException) -> str:
    """One readable line. The MCP client wraps failures in exception groups."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    if isinstance(exc, TimeoutError):
        return f"нет ответа за {_TIMEOUT:.0f} с"
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


async def fetch_paper_statistics() -> Dict[str, Any]:
    """Call the tool once.

    Returns ``{"status": ..., "report": ...}`` or ``{"status": ..., "error": ...}``:

    - ``ready``: a full report.
    - ``pending``: the server has no report yet - its first scan is running, or
      its last attempt failed - and ``report`` carries its note.
    - ``not_configured``: ``MCP__PAPER_ANALYSIS_URL`` is not set.
    - ``unavailable``: the call itself failed, or the tool returned an error.
    """
    url = paper_analysis_url()
    if not url:
        return {"status": "not_configured", "error": "Не задан MCP__PAPER_ANALYSIS_URL."}

    try:
        # Imported here: a process that never shows the page does not load the
        # MCP stack, and an environment with an older ``mcp`` (this needs the
        # locked version) gets an error on the page instead of a bare HTTP 500.
        import anyio
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        # trust_env=False: the MCP server is an internal service, so go straight
        # to it. By default httpx would route the call through HTTP(S)_PROXY or,
        # on macOS and Windows, the system proxy - the detour that slowed the
        # Chroma traffic. Otherwise this is what mcp's create_mcp_http_client builds.
        with anyio.fail_after(_TIMEOUT):
            async with httpx.AsyncClient(timeout=httpx.Timeout(_TIMEOUT), trust_env=False) as http:
                async with streamable_http_client(url, http_client=http) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(TOOL_NAME, {})
    except Exception as exc:  # noqa: BLE001 - the page must render whatever happens
        logger.warning("paper statistics: %s at %s failed (%s)", TOOL_NAME, url, _describe(exc))
        return {"status": "unavailable", "error": f"вызов {TOOL_NAME} ({url}) не удался: {_describe(exc)}"}

    text = "\n".join(
        item.text for item in (result.content or []) if isinstance(getattr(item, "text", None), str)
    ).strip()
    if result.isError:
        return {"status": "unavailable", "error": f"{TOOL_NAME} ({url}) вернул ошибку: {text or 'без подробностей'}"}
    if not text:
        return {"status": "unavailable", "error": f"{TOOL_NAME} ({url}) не вернул текста"}
    status = "ready" if _FINISHED_REPORT.search(text) else "pending"
    return {"status": status, "report": text}
