"""Tests for CoScientist.utils.selective_proxy — LiteLLMProxy resilience.

Validates that:
1. _UnclosableAsyncClient ignores aclose() but responds to force_close().
2. LiteLLMProxy.enable() creates per-handler clients (not a shared one).
3. LiteLLMProxy.disable() restores the original create_client.
"""
import asyncio

import httpx
import pytest

from CoScientist.utils.selective_proxy import (
    LiteLLMProxy,
    _UnclosableAsyncClient,
)


@pytest.fixture()
def proxy():
    """Create a LiteLLMProxy pointed at a dummy URL and ensure cleanup."""
    p = LiteLLMProxy("http://127.0.0.1:9999")
    yield p
    p.disable()


# ── _UnclosableAsyncClient ──────────────────────────────────────────────


def test_unclosable_aclose_is_noop():
    """aclose() must NOT actually close the transport."""
    async def _run():
        client = _UnclosableAsyncClient()
        await client.aclose()  # should be a no-op
        assert not client.is_closed
        # Clean up for real
        await client.force_close()

    asyncio.run(_run())


def test_unclosable_force_close_works():
    """force_close() must actually close the client."""
    async def _run():
        client = _UnclosableAsyncClient()
        await client.force_close()
        assert client.is_closed

    asyncio.run(_run())


# ── LiteLLMProxy ────────────────────────────────────────────────────────


def test_enable_sets_aclient_session(proxy):
    """After enable(), litellm.aclient_session must be an _UnclosableAsyncClient."""
    import litellm

    proxy.enable()
    assert isinstance(litellm.aclient_session, _UnclosableAsyncClient)


def test_disable_clears_aclient_session(proxy):
    """After disable(), litellm.aclient_session must be None."""
    import litellm

    proxy.enable()
    proxy.disable()
    assert litellm.aclient_session is None


def test_create_client_returns_distinct_clients(proxy):
    """Each call to the overridden create_client must return a NEW client."""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    proxy.enable()

    handler = AsyncHTTPHandler.__new__(AsyncHTTPHandler)
    client_a = AsyncHTTPHandler.create_client(handler)
    client_b = AsyncHTTPHandler.create_client(handler)

    assert isinstance(client_a, httpx.AsyncClient)
    assert isinstance(client_b, httpx.AsyncClient)
    assert client_a is not client_b, (
        "create_client must return a fresh client per call, not a shared instance"
    )


def test_disable_restores_original_create_client(proxy):
    """After disable(), create_client must be the original method."""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    original = AsyncHTTPHandler.create_client
    proxy.enable()
    assert AsyncHTTPHandler.create_client is not original
    proxy.disable()
    assert AsyncHTTPHandler.create_client is original


def test_enable_is_idempotent(proxy):
    """Calling enable() twice must not create a second session client."""
    proxy.enable()
    session_1 = proxy.proxy_client
    proxy.enable()
    session_2 = proxy.proxy_client
    assert session_1 is session_2


def test_closing_one_handler_client_does_not_affect_another(proxy):
    """Closing one handler's client must not break another handler's client.

    This is the exact scenario that caused the original crash.
    """
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    proxy.enable()

    handler = AsyncHTTPHandler.__new__(AsyncHTTPHandler)
    client_a = AsyncHTTPHandler.create_client(handler)
    client_b = AsyncHTTPHandler.create_client(handler)

    async def _run():
        # Simulate litellm closing handler A's client
        await client_a.aclose()
        assert client_a.is_closed
        # Handler B's client must still be alive
        assert not client_b.is_closed

    asyncio.run(_run())


def test_toggling_evicts_litellm_cached_http_clients(proxy):
    """Flipping the proxy must take effect on the very next call.

    litellm caches an ``AsyncHTTPHandler`` per (params, provider) for an hour,
    and each handler owns the ``httpx.AsyncClient`` ``create_client`` built for
    it — proxy setting baked in. Restoring the factory alone therefore changes
    nothing for traffic that reuses a cached handler: with a proxy whose VPN is
    down, every LLM call still stalls until it times out, long after the user
    turned the proxy off.
    """
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

    async def _run():
        proxy.enable()
        params = {"ssl_verify": None}
        proxied = get_async_httpx_client(llm_provider="openrouter", params=params)
        # Same key -> the cache hands back the very same handler.
        assert get_async_httpx_client(llm_provider="openrouter", params=params) is proxied

        proxy.disable()

        direct = get_async_httpx_client(llm_provider="openrouter", params=params)
        assert direct is not proxied

    asyncio.run(_run())
