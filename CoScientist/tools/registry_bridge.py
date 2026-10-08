"""Bridge: a freshly served MCP server → the tool registry + this run's state.

An Alembic build ends with a served FastMCP endpoint (``alembic_tools``'
``build_mcp_server`` reports its ``mcp_url``), but nothing puts that server
where CoScientist can find it. The catalogue therefore never grows: the next
run has no idea the tool exists and re-does the work.

This module closes both halves of that gap:

  (a) :func:`register_mcp_server` ingests the server **and its tools** into the
      rag_tools registry, so ``Retrieve_tools`` finds it in later runs and on
      other machines — the durable catalogue;
  (b) :func:`resolve_into_state` puts the served url into
      ``state['deployed_mcps']`` — the shape
      :class:`~CoScientist.tools.dynamic_tools.DynamicMCPToolset` reads — so the
      executor can call the tool immediately, without waiting for a retrieval
      round.

The two are called at different moments, which is why they are separate. (a)
happens when the build finishes, whether or not anybody is asking, because a
tool missing from the catalogue is a tool the next run builds again. (b) needs
a live session, so it happens when a build is polled from one.

``register_mcp_server`` connects to the live server to enumerate its tools, so
the server must be serving at ``mcp_url`` when it is called.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def _default_manager():
    """A rag_tools manager wired from env settings.

    It uses the same embedder/reranker stack as ``Retrieve_tools``
    (``CoScientist/tools/retrieval_tools.py``) — what we index has to be what
    retrieval later scores.
    """
    import rag_tools
    from rag_tools.config.settings import get_settings
    from rag_tools.retrieval import APIReranker, BM25Reranker, HybridReranker

    from CoScientist.tools.embedder_shim import SafeAPIEmbedder

    settings = get_settings()
    reranker = HybridReranker(
        [APIReranker(settings.api_reranker), BM25Reranker(settings.bm_reranker)],
        settings.hybrid_reranker,
    )
    return await rag_tools.create_manager(
        settings, SafeAPIEmbedder(settings.api_embedding), reranker
    )


async def register_mcp_server(
    mcp_url: str,
    name: str,
    description: str = "",
    *,
    headers: Optional[Dict[str, str]] = None,
    manager=None,
    sync_tools: bool = True,
):
    """Ingest a served MCP server and its tools into the rag_tools registry.

    Args:
        mcp_url: the streamable-http url the server is served at.
        name: registry name for the server (also part of its ``server_id``).
        description: optional server description.
        headers: optional HTTP headers (e.g. auth) for reaching the server.
        manager: an existing manager to reuse across a batch; when ``None`` one
            is built from env settings and closed before returning.
        sync_tools: connect to the live server and index its tools — the point
            of registering, and the default.

    Returns:
        The created ``MCPServer``. A failed tool sync leaves it with
        ``status == ERROR``, logged rather than raised, so a caller registering
        a batch can record the failure and carry on.
    """
    from rag_tools.storage.models import MCPProtocol, ToolStatus

    if not mcp_url:
        raise ValueError("register_mcp_server: mcp_url is required")

    own_manager = manager is None
    if own_manager:
        manager = await _default_manager()
    try:
        server = await manager.add_server(
            name=name,
            protocol=MCPProtocol.HTTP,
            url=mcp_url,
            description=description or "",
            headers=headers,
            sync_tools=sync_tools,
        )
    finally:
        if own_manager:
            await manager.close()

    if getattr(server, "status", None) == ToolStatus.ERROR:
        logger.warning(
            "register_mcp_server: %s is registered but its tool sync failed — "
            "check that it is serving at %s",
            name,
            mcp_url,
        )
    return server


async def unregister_mcp_server(server_id: str, *, manager=None) -> None:
    """Remove a server and its indexed tools from the rag_tools registry.

    Raises when the registry cannot be reached, so the caller can keep the
    record of the registration and try again.
    """
    if not server_id:
        raise ValueError("unregister_mcp_server: server_id is required")
    own_manager = manager is None
    if own_manager:
        manager = await _default_manager()
    try:
        await manager.remove_server(server_id)
    finally:
        if own_manager:
            await manager.close()


def resolve_into_state(
    state: Dict[str, Any],
    server_or_url: Any,
    name: Optional[str] = None,
) -> Dict[str, str]:
    """Add a ``deployed_mcps`` entry so the executor can call the tool this run.

    Accepts either an ``MCPServer`` (its ``url``/``name`` are used) or a bare
    url. Idempotent by url, so it composes with the retrieval pipeline, which
    may add the same entry.
    """
    url = getattr(server_or_url, "url", None) or server_or_url
    if not url:
        raise ValueError("resolve_into_state: no url to resolve")

    entry = {"url": url, "name": name or getattr(server_or_url, "name", None) or url}
    deployed: List[Dict[str, Any]] = state.setdefault("deployed_mcps", [])
    if not any(d.get("url") == url for d in deployed):
        deployed.append(entry)
    return entry
