"""Tests for tools/mcp_resilience.py — an MCP server that drops its transport.

Reproduces the EconomicsAgent stall: the session task dies after the session
is up, and ``list_tools`` is never answered. Stock ADK waits out the read
timeout (600 s for economics); the resilient toolset must fail at once and
retry, and a read-only toolset must repeat a call whose transport was lost.
"""
import asyncio
import time

import pytest
from google.adk.dependencies._mcp import McpError
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_tool import McpTool
from google.adk.tools.mcp_tool.session_context import SessionContext

from CoScientist.tools import mcp_resilience
from CoScientist.tools.mcp_resilience import (
    ResilientMcpToolset,
    _ReconnectingMcpTool,
    _RetryingMcpTool,
)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(mcp_resilience, "MCP_RETRY_BACKOFF", 0.0)


def _toolset(**kwargs):
    return ResilientMcpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url="http://test.invalid/mcp", timeout=600, sse_read_timeout=600,
        ),
        **kwargs,
    )


class _HangingSession:
    async def list_tools(self):
        await asyncio.Event().wait()  # the answer never comes


class _DeadTransportManager:
    """A session manager whose session task has already died."""

    def __init__(self, ctx):
        self._ctx = ctx

    async def create_session(self, headers=None):
        return _HangingSession()

    def _get_session_context(self, headers=None):
        return self._ctx

    def _begin_session_use(self, headers):
        pass

    def _end_session_use(self, headers):
        pass


async def _dead_session_context():
    ctx = SessionContext(client=None, timeout=600, sse_read_timeout=600)

    async def crash():
        raise RuntimeError("unhandled errors in a TaskGroup (1 sub-exception)")

    ctx._task = asyncio.create_task(crash())
    await asyncio.sleep(0)
    return ctx


def test_listing_on_a_dead_transport_fails_at_once():
    async def run():
        toolset = _toolset()
        toolset._mcp_session_manager = _DeadTransportManager(
            await _dead_session_context()
        )
        started = time.monotonic()
        with pytest.raises(ConnectionError, match="TaskGroup"):
            await toolset._execute_with_session(
                lambda session: session.list_tools(), "list", headers={},
            )
        return time.monotonic() - started

    assert asyncio.run(run()) < 5


def test_listing_is_retried(monkeypatch):
    calls = []

    async def flaky(self, readonly_context=None):
        calls.append(1)
        if len(calls) < mcp_resilience.MCP_RETRY_ATTEMPTS:
            raise ConnectionError("transport lost")
        return []

    monkeypatch.setattr(McpToolset, "get_tools", flaky)
    assert asyncio.run(_toolset().get_tools()) == []
    assert len(calls) == mcp_resilience.MCP_RETRY_ATTEMPTS


def test_listing_gives_up_after_the_last_attempt(monkeypatch):
    async def dead(self, readonly_context=None):
        raise ConnectionError("transport lost")

    monkeypatch.setattr(McpToolset, "get_tools", dead)
    with pytest.raises(ConnectionError):
        asyncio.run(_toolset().get_tools())


def _tool(cls):
    tool = McpTool.__new__(McpTool)
    tool.__class__ = cls
    tool.name = "get_price"
    return tool


def test_a_read_only_call_is_repeated_after_a_lost_connection(monkeypatch):
    calls = []

    async def flaky(self, *, args, tool_context, credential):
        calls.append(args)
        if len(calls) == 1:
            raise ConnectionError("MCP session connection lost")
        return {"ok": True}

    monkeypatch.setattr(McpTool, "_run_async_impl", flaky)
    result = asyncio.run(_tool(_RetryingMcpTool)._run_async_impl(
        args={"q": 1}, tool_context=None, credential=None,
    ))
    assert result == {"ok": True}
    assert calls == [{"q": 1}, {"q": 1}]


def test_a_tool_error_is_not_repeated(monkeypatch):
    calls = []

    async def failing(self, *, args, tool_context, credential):
        calls.append(args)
        raise ValueError("the server answered with an error")

    monkeypatch.setattr(McpTool, "_run_async_impl", failing)
    with pytest.raises(ValueError):
        asyncio.run(_tool(_RetryingMcpTool)._run_async_impl(
            args={}, tool_context=None, credential=None,
        ))
    assert len(calls) == 1


@pytest.mark.parametrize("retry_calls, expected", [(True, _RetryingMcpTool), (False, _ReconnectingMcpTool)])
def test_only_a_read_only_toolset_gets_retrying_tools(monkeypatch, retry_calls, expected):
    async def listing(self, readonly_context=None):
        return [_tool(McpTool)]

    monkeypatch.setattr(McpToolset, "get_tools", listing)
    tools = asyncio.run(_toolset(retry_calls=retry_calls).get_tools())
    assert type(tools[0]) is expected


def _session_terminated():
    from mcp.types import ErrorData
    return McpError(ErrorData(code=32600, message="Session terminated"))


class _PoolManager:
    """Just enough of MCPSessionManager for _drop_pooled_sessions."""

    def __init__(self):
        self._session_lock = asyncio.Lock()
        self._sessions = {"k": (object(), None, None)}
        self.cleaned = []

    async def _cleanup_session(self, key, exit_stack, loop):
        self.cleaned.append(key)
        self._sessions.pop(key)


def test_a_forgotten_session_is_dropped_and_the_call_repeated_once(monkeypatch):
    calls = []

    async def restarted_server(self, *, args, tool_context, credential):
        calls.append(args)
        if len(calls) == 1:
            raise _session_terminated()
        return {"ok": True}

    monkeypatch.setattr(McpTool, "_run_async_impl", restarted_server)
    tool = _tool(_ReconnectingMcpTool)
    tool._mcp_session_manager = _PoolManager()
    result = asyncio.run(tool._run_async_impl(args={"q": 1}, tool_context=None, credential=None))
    assert result == {"ok": True}
    assert calls == [{"q": 1}, {"q": 1}]
    assert tool._mcp_session_manager.cleaned == ["k"]


def test_other_mcp_errors_are_not_repeated(monkeypatch):
    from mcp.types import ErrorData
    calls = []

    async def failing(self, *, args, tool_context, credential):
        calls.append(args)
        raise McpError(ErrorData(code=-32602, message="bad arguments"))

    monkeypatch.setattr(McpTool, "_run_async_impl", failing)
    tool = _tool(_ReconnectingMcpTool)
    tool._mcp_session_manager = _PoolManager()
    with pytest.raises(McpError):
        asyncio.run(tool._run_async_impl(args={}, tool_context=None, credential=None))
    assert len(calls) == 1
    assert tool._mcp_session_manager.cleaned == []


def test_listing_on_a_forgotten_session_reconnects(monkeypatch):
    calls = []

    async def restarted_server(self, readonly_context=None):
        calls.append(1)
        if len(calls) == 1:
            # ADK wraps the listing failure the way _execute_with_session does.
            raise ConnectionError("Failed to get tools") from _session_terminated()
        return []

    monkeypatch.setattr(McpToolset, "get_tools", restarted_server)
    toolset = _toolset()
    manager = _PoolManager()
    toolset._mcp_session_manager = manager
    assert asyncio.run(toolset.get_tools()) == []
    assert manager.cleaned == ["k"]
