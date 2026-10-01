"""An MCP toolset that survives a remote server dropping its transport.

Observed failure (EconomicsAgent, 2026-09-23): the streamable-HTTP transport
to the economics server died after the session was up ("GET stream
disconnected" → "Error on session runner task: unhandled errors in a
TaskGroup"). ADK then behaves badly in two places:

* Tool listing (``McpToolset._execute_with_session``) awaits ``list_tools``
  without watching the session task. When the transport dies, the session's
  receive loop is cancelled before it can fail the pending request, so the
  listing waits out the full read timeout — 600 s for economics. The agent
  stood still for ten minutes and looked crashed.
* A tool call that loses its transport is surfaced as an error and never
  retried (ADK cannot know whether the call had side effects), and listing is
  retried only once, immediately.

``ResilientMcpToolset`` races listing against the session task (so a dead
transport fails at once), retries listing with backoff, and — only for a
toolset declared read-only (``retry_calls=True``) — retries a tool call whose
transport was lost. Tool-level errors (``McpError``) and timeouts are not
retried: the server answered, and a slow call repeated is just slower.

One ``McpError`` is the exception: "Session terminated". It is what the MCP
client reports when the server answers 404 to our ``mcp-session-id`` — the
server was restarted and forgot the session (observed 2026-09-25: the
economics container was brought back up after a HITL pause, and every call
kept failing). The transport itself is healthy, so ADK keeps handing out the
pooled dead session forever. On that error every tool of the toolset drops
the pooled sessions and repeats the call once on a fresh one; this is safe for
side-effecting tools too, because a 404 means the server never ran the call.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, List, Optional

from google.adk.dependencies._mcp import McpError
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_tool import MCPTool, McpTool
from google.adk.tools.mcp_tool.session_context import SessionContext

logger = logging.getLogger(__name__)

MCP_RETRY_ATTEMPTS = max(1, int(os.getenv("MCP__RETRY_ATTEMPTS", "3")))
MCP_RETRY_BACKOFF = float(os.getenv("MCP__RETRY_BACKOFF", "2"))


def _is_cancelling() -> bool:
    task = asyncio.current_task()
    cancelling = getattr(task, "cancelling", None) if task else None
    return bool(cancelling and cancelling() > 0)


async def _backoff(attempt: int) -> None:
    await asyncio.sleep(MCP_RETRY_BACKOFF * (2 ** (attempt - 1)))


def _is_session_terminated(exc: BaseException) -> bool:
    """True if ``exc`` (or its cause chain) is the server forgetting our session."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, McpError) and "Session terminated" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


async def _drop_pooled_sessions(manager: Any) -> None:
    """Close every pooled session of ``manager`` so the next call reconnects.

    All of them, not just the failed one: they all point at the same server,
    and a restarted server has forgotten every one.
    """
    async with manager._session_lock:  # pylint: disable=protected-access
        for key, (_, exit_stack, loop) in list(manager._sessions.items()):  # pylint: disable=protected-access
            await manager._cleanup_session(key, exit_stack, loop)  # pylint: disable=protected-access


class _ReconnectingMcpTool(McpTool):
    """An McpTool that reconnects once when the server has forgotten its session."""

    async def _run_async_impl(self, *, args, tool_context, credential):
        try:
            return await super()._run_async_impl(
                args=args, tool_context=tool_context, credential=credential
            )
        except McpError as exc:
            if not _is_session_terminated(exc) or _is_cancelling():
                raise
            logger.warning(
                "MCP tool %s: server forgot the session (restarted?), reconnecting",
                self.name,
            )
            await _drop_pooled_sessions(self._mcp_session_manager)
            return await super()._run_async_impl(
                args=args, tool_context=tool_context, credential=credential
            )


class _RetryingMcpTool(_ReconnectingMcpTool):
    """An McpTool that repeats a call whose transport was lost.

    Only ever set on tools of a read-only toolset: ``ConnectionError`` here
    means the session died mid-call, so the server may or may not have run it.
    """

    async def _run_async_impl(self, *, args, tool_context, credential):
        for attempt in range(1, MCP_RETRY_ATTEMPTS + 1):
            try:
                return await super()._run_async_impl(
                    args=args, tool_context=tool_context, credential=credential
                )
            except ConnectionError as exc:
                if attempt == MCP_RETRY_ATTEMPTS or _is_cancelling():
                    raise
                logger.warning(
                    "MCP tool %s lost its connection (attempt %d/%d), retrying: %s",
                    self.name, attempt, MCP_RETRY_ATTEMPTS, exc,
                )
                await _backoff(attempt)


class ResilientMcpToolset(McpToolset):
    """McpToolset whose listing fails fast and retries; see module docstring."""

    def __init__(self, *args: Any, retry_calls: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._retry_calls = retry_calls

    async def _execute_with_session(
        self,
        coroutine_func,
        error_message: str,
        readonly_context=None,
        headers: Optional[dict] = None,
    ):
        if headers is None:
            headers = await self._build_headers(readonly_context)
        session_headers = headers or None

        async def guarded(session):
            # Race the request against the session task, as ADK does for tool
            # calls: a dead transport fails now instead of at the read timeout.
            ctx = self._mcp_session_manager._get_session_context(  # pylint: disable=protected-access
                headers=session_headers
            )
            coro = coroutine_func(session)
            if isinstance(ctx, SessionContext):
                return await ctx._run_guarded(coro)  # pylint: disable=protected-access
            return await coro

        return await super()._execute_with_session(
            guarded, error_message, readonly_context, headers=headers
        )

    async def get_tools(self, readonly_context=None) -> List[BaseTool]:
        for attempt in range(1, MCP_RETRY_ATTEMPTS + 1):
            try:
                tools = await super().get_tools(readonly_context)
                break
            except Exception as exc:  # noqa: BLE001 — retried, then re-raised
                if attempt == MCP_RETRY_ATTEMPTS or _is_cancelling():
                    raise
                logger.warning(
                    "MCP tool listing failed (attempt %d/%d), retrying: %s",
                    attempt, MCP_RETRY_ATTEMPTS, exc,
                )
                if _is_session_terminated(exc):
                    await _drop_pooled_sessions(self._mcp_session_manager)
                await _backoff(attempt)
        tool_cls = _RetryingMcpTool if self._retry_calls else _ReconnectingMcpTool
        for tool in tools:
            if type(tool) in (McpTool, MCPTool):
                # Built by ADK's get_tools; the subclass adds behavior only.
                tool.__class__ = tool_cls
        return tools
