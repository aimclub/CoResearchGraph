"""Process-wide start-rate limiter for calls to MCP tools.

Research agents may run concurrently, but the MCP servers must not receive a
burst of six requests at once.  The limiter spaces *starts* of MCP tool calls
by five seconds.  It deliberately does not hold the lock while a call is in
flight: calls that have already started are allowed to run concurrently.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Awaitable, Callable

from google.adk.tools.mcp_tool.mcp_tool import McpTool

logger = logging.getLogger(__name__)

MCP_TOOL_START_INTERVAL_SECONDS = 5.0


class MCPToolStartLimiter:
    """Allow MCP tool calls to start at most once per configured interval."""

    def __init__(self, interval_seconds: float = MCP_TOOL_START_INTERVAL_SECONDS) -> None:
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be non-negative")
        self.interval_seconds = interval_seconds
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._next_start_at = 0.0

    def _lock_for_current_loop(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        # The application uses one event loop. Recreate the lock when a test or
        # embedding host starts a fresh loop after the previous one was closed.
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
            self._loop = loop
            self._next_start_at = 0.0
        return self._lock

    async def run(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        """Wait for the next slot, then start ``operation`` immediately.

        Only slot allocation is protected by the lock.  The awaited operation
        runs outside it, so several MCP calls can overlap after their starts
        have been spaced out.
        """
        lock = self._lock_for_current_loop()
        async with lock:
            now = time.monotonic()
            delay = self._next_start_at - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_start_at = time.monotonic() + self.interval_seconds

        return await operation()


mcp_tool_start_limiter = MCPToolStartLimiter()


def install_mcp_tool_start_limiter() -> None:
    """Patch ADK's MCP tool boundary once for all MCP toolsets in the app."""
    if getattr(McpTool, "_coscientist_start_limiter_installed", False):
        return

    original = McpTool._run_async_impl

    @functools.wraps(original)
    async def limited_run_async_impl(self: McpTool, *args: Any, **kwargs: Any) -> Any:
        return await mcp_tool_start_limiter.run(
            lambda: original(self, *args, **kwargs)
        )

    McpTool._run_async_impl = limited_run_async_impl
    McpTool._coscientist_start_limiter_installed = True
    logger.info(
        "MCP tool start limiter enabled: %.1f seconds between call starts",
        MCP_TOOL_START_INTERVAL_SECONDS,
    )


install_mcp_tool_start_limiter()
