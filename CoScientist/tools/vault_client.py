"""A per-call MCP client for the file vault, for framework code only.

Worker agents reach the vault through an ADK ``McpToolset`` (see
``research_tools.py``). This module is the other consumer: plain framework code
that has an ``s3://bucket/key`` and needs a file, or has an ``ephemeral/`` key
and needs it promoted. Report packaging and the workspace sync both run outside
any agent, so neither can use a toolset.

Every call opens its own session and closes it. The ADK session manager caches a
session and binds its exit stack to the loop that created it, so a shared
instance driven from a fresh loop fails on the second call. The raw MCP SDK has
no such state.

No function here raises. A missing URL, an unreachable server, or a vault error
all return ``None`` and log a warning. A run that produced results must still
produce a report.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from CoScientist.config import get_settings

logger = logging.getLogger(__name__)

# One vault call moves no payload, only a key and a URL. The object itself
# travels straight to S3, so the server answers fast or it is down.
_TIMEOUT = 30.0


def vault_url() -> Optional[str]:
    """Read at call time, so a test can point this at a local server."""
    return get_settings().mcp.vault_url


def _payload(result: Any) -> Optional[Dict[str, Any]]:
    """Unwrap the vault reply.

    Every vault tool returns a JSON string, and MCP wraps it again in
    ``content[].text``. The same double envelope ``utils/s3_refs.py`` walks.
    """
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if not isinstance(text, str):
            continue
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


async def call_vault(tool_name: str, **args: Any) -> Optional[Dict[str, Any]]:
    """Call one vault tool. Return its payload, or None."""
    url = vault_url()
    if not url:
        logger.debug("vault: MCP__VAULT_URL is not set, skipping %s", tool_name)
        return None

    # Imported here so that loading this module does not pull the MCP stack into
    # a process that never calls the vault.
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    try:
        async with streamablehttp_client(url, timeout=_TIMEOUT) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, args)
    except Exception as exc:  # noqa: BLE001 - the vault must never sink a run
        logger.warning("vault: %s failed (%s)", tool_name, exc)
        return None

    payload = _payload(result)
    if payload is None:
        logger.warning("vault: %s returned no readable payload", tool_name)
        return None
    # The vault reports a refusal in the body, not as a transport error.
    if payload.get("error"):
        logger.warning("vault: %s refused (%s)", tool_name, payload["error"])
        return None
    return payload


def call_vault_sync(tool_name: str, **args: Any) -> Optional[Dict[str, Any]]:
    """The same call from synchronous code.

    ``finalize_report`` is the caller. Both of its call sites reach it through
    ``asyncio.to_thread``, so the worker thread has no running loop and
    ``asyncio.run`` is safe. Refuse when a loop IS running: calling this from
    the event-loop thread would deadlock it, and ``call_vault`` is right there.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(call_vault(tool_name, **args))
    raise RuntimeError(
        f"call_vault_sync({tool_name!r}) ran on the event loop. Await call_vault instead."
    )


__all__ = ["call_vault", "call_vault_sync", "vault_url"]
