"""Tests for the MCP SSE error-propagation backport (tools/mcp_patches.py).

Simulates the production failure: the hosted Tavily MCP server emits an SSE
frame whose JSON-RPC payload is truncated mid-string. Stock mcp<=1.27 strands
the pending request (hang until read timeout); the backport must answer it
with a JSON-RPC error carrying the original request id.
"""
import anyio
import httpx
import pytest
from httpx_sse import ServerSentEvent

from CoScientist.tools import mcp_patches
from mcp.client.streamable_http import RequestContext, StreamableHTTPTransport
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCError, JSONRPCMessage, JSONRPCRequest, JSONRPCResponse

TRUNCATED = '{"jsonrpc":"2.0","id":8,"result":{"content":[{"type":"text","text":"...T and NEXLETOL is used '
VALID = '{"jsonrpc":"2.0","id":8,"result":{"ok":true}}'


def _transport():
    return StreamableHTTPTransport(url="http://test.invalid/mcp")


def _ctx(read_stream_writer, request_id=8):
    request = JSONRPCMessage(
        JSONRPCRequest(jsonrpc="2.0", id=request_id, method="tools/call")
    )
    return RequestContext(
        client=httpx.AsyncClient(),
        session_id="s1",
        session_message=SessionMessage(request),
        metadata=None,
        read_stream_writer=read_stream_writer,
    )


def test_patch_is_applied_on_broken_mcp():
    assert mcp_patches._applied == mcp_patches._mcp_is_broken()


def test_truncated_sse_event_raises_instead_of_stranding():
    transport = _transport()

    async def run():
        send, _recv = anyio.create_memory_object_stream(10)
        sse = ServerSentEvent(event="message", data=TRUNCATED)
        with pytest.raises(mcp_patches._InvalidSSEPayload):
            await transport._handle_sse_event(sse, send)

    anyio.run(run)


def test_valid_sse_event_passes_through():
    transport = _transport()

    async def run():
        send, recv = anyio.create_memory_object_stream(10)
        sse = ServerSentEvent(event="message", data=VALID)
        is_complete = await transport._handle_sse_event(sse, send)
        assert is_complete is True  # a response completes the request
        delivered = recv.receive_nowait()
        assert isinstance(delivered.message.root, JSONRPCResponse)

    anyio.run(run)


def test_sse_response_with_truncated_frame_fails_request_fast():
    """End to end through _handle_sse_response: the pending request must
    receive a JSONRPCError with the ORIGINAL request id — not hang."""
    transport = _transport()

    class FakeResponse(httpx.Response):
        def __init__(self):
            super().__init__(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"event: message\ndata: {TRUNCATED}\n\n".encode(),
            )

        async def aclose(self):  # content already consumed; nothing to close
            pass

        async def aiter_bytes(self, chunk_size=None):
            yield self.content

    async def run():
        send, recv = anyio.create_memory_object_stream(10)
        ctx = _ctx(send, request_id=8)
        await transport._handle_sse_response(FakeResponse(), ctx)
        delivered = recv.receive_nowait()
        err = delivered.message.root
        assert isinstance(err, JSONRPCError)
        assert err.id == 8
        assert "invalid JSON-RPC" in err.error.message

    anyio.run(run)
