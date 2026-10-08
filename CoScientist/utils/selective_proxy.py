from __future__ import annotations

import logging
from typing import Any, Callable

import httpx
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

_logger = logging.getLogger(__name__)


class _UnclosableAsyncClient(httpx.AsyncClient):

    async def aclose(self) -> None:  # noqa: D102 — intentional no-op
        pass

    async def force_close(self) -> None:
        """Actually close the underlying transport & connection pool."""
        await super().aclose()


class LiteLLMProxy:
    """Inject a forward-proxy into litellm's networking layer.

    Only litellm HTTP traffic is affected. Other ``httpx.AsyncClient``
    instances in the process are **not** touched.

    Implementation notes
    --------------------
    litellm uses two paths to obtain an ``httpx.AsyncClient``:

    1. **``litellm.aclient_session``** — a process-global client for OpenAI-
       compatible providers.  We set this to an :class:`_UnclosableAsyncClient`
       so that litellm cannot accidentally close our shared session.

    2. **``AsyncHTTPHandler.create_client()``** — called once per handler
       (cached by ``LLMClientCache`` with a 1-hour TTL).  We override this to
       create a **fresh** proxied ``httpx.AsyncClient`` for every handler.
       Each handler owns its client independently, so when litellm evicts or
       closes a handler, only *that* handler's client is affected.

    Previous implementation shared a **single** ``httpx.AsyncClient`` across
    all handlers.  When any handler closed "its" client, the shared instance
    was destroyed and all subsequent requests failed with
    ``RuntimeError: Cannot send a request, as the client has been closed.``
    """

    def __init__(self, proxy_url: str) -> None:
        self._proxy_url = proxy_url
        self._session_client: _UnclosableAsyncClient | None = None
        self._orig_create_client = AsyncHTTPHandler.create_client
        self._is_enabled = False

    @property
    def is_enabled(self) -> bool:
        """Whether litellm traffic is currently routed through the proxy."""
        return self._is_enabled

    @property
    def proxy_client(self) -> httpx.AsyncClient | None:
        """The proxied session ``httpx.AsyncClient``, or *None* before first :meth:`enable`."""
        return self._session_client

    @staticmethod
    def _flush_litellm_client_cache() -> None:
        """Drop litellm's cached HTTP handlers so a toggle takes effect at once.

        litellm caches one ``AsyncHTTPHandler`` per (params, provider) for an
        hour in ``in_memory_llm_clients_cache``. Each handler OWNS the
        ``httpx.AsyncClient`` that ``create_client`` built for it, with the
        proxy setting baked in — so restoring the factory in :meth:`disable` is
        not enough: every already-cached handler keeps sending through the old
        proxy until its TTL expires. With a proxy whose VPN is down that reads
        as the app hanging on every LLM call long after the user turned the
        proxy off (the handler answers nothing until the request times out).

        Eviction deliberately does not close the clients: an in-flight request
        may still hold one (litellm's own cache documents the same rule).
        """
        cache = getattr(litellm, "in_memory_llm_clients_cache", None)
        if cache is None:
            return
        try:
            cache.flush_cache()
        except Exception as err:  # noqa: BLE001 — never block the toggle
            _logger.warning("Could not flush litellm client cache: %s", err)

    def enable(self) -> None:
        """Start routing litellm HTTP traffic through the proxy. Idempotent."""
        if self._is_enabled:
            return

        if self._session_client is None:
            self._session_client = _UnclosableAsyncClient(
                proxy=self._proxy_url,
                follow_redirects=True,
                timeout=httpx.Timeout(60.0, connect=15.0),
            )

        # Path 1: OpenAI-compatible providers (public litellm API).
        # Wrapped in _UnclosableAsyncClient so litellm cannot close it.
        litellm.aclient_session = self._session_client

        # Path 2: Non-OpenAI providers (OpenRouter, etc.) — override client factory.
        self._install_create_client_override()
        # Handlers cached from before the toggle still hold a direct client.
        self._flush_litellm_client_cache()

        self._is_enabled = True
        _logger.info("LiteLLM proxy enabled → %s", self._proxy_url)

    def disable(self) -> None:
        """Stop proxying; restore litellm's original networking. Idempotent."""
        if not self._is_enabled:
            return

        litellm.aclient_session = None
        AsyncHTTPHandler.create_client = self._orig_create_client
        # Restoring the factory only affects handlers built from now on.
        self._flush_litellm_client_cache()

        self._is_enabled = False
        _logger.info("LiteLLM proxy disabled")

    def _install_create_client_override(self) -> None:
        """Replace ``AsyncHTTPHandler.create_client`` to create a **fresh**
        proxied client for each handler.

        Each handler gets its own ``httpx.AsyncClient`` with proxy settings,
        so closing one handler does not affect others.
        """
        proxy_url = self._proxy_url
        orig = self._orig_create_client

        def _proxied_create_client(handler_self, *args, **kwargs):
            return httpx.AsyncClient(
                proxy=proxy_url,
                follow_redirects=True,
                timeout=httpx.Timeout(60.0, connect=15.0),
            )

        AsyncHTTPHandler.create_client = _proxied_create_client


def create_mcp_proxy_httpx_factory(
    proxy_url: str,
    enabled_fn: Callable[[], bool] | bool = True,
    connect_timeout: float = 5.0,
    read_timeout: float = 300.0,
) -> Callable[..., httpx.AsyncClient]:
    """Return an httpx client factory for ADK MCP Toolsets (e.g., Tavily MCP).

    Conforms to the ``httpx_client_factory`` interface expected by ADK's
    ``StreamableHTTPConnectionParams``.

    Parameters
    ----------
    proxy_url:
        The forward-proxy URL.
    enabled_fn:
        Either a boolean or a zero-arg callable returning bool (for dynamic runtime toggles).
    """

    def factory(
        headers: dict[str, Any] | None = None,
        auth: httpx.Auth | None = None,
        timeout: httpx.Timeout | float | None = None,
    ) -> httpx.AsyncClient:
        if timeout is None:
            tot_timeout = httpx.Timeout(30.0, connect=connect_timeout, read=read_timeout)
        elif isinstance(timeout, (int, float)):
            tot_timeout = httpx.Timeout(timeout, connect=connect_timeout)
        else:
            tot_timeout = timeout

        is_active = enabled_fn() if callable(enabled_fn) else bool(enabled_fn)
        active_proxy = proxy_url if is_active else None

        return httpx.AsyncClient(
            headers=headers,
            auth=auth,
            timeout=tot_timeout,
            proxy=active_proxy,
            follow_redirects=True,
        )

    return factory


class DomainSelectiveAsyncTransport(httpx.AsyncBaseTransport):
    """An ``httpx.AsyncBaseTransport`` that proxies ONLY specified domain names.

    Requests targeting domains in *proxied_domains* go through *proxy_url*.
    All other requests connect directly.

    Parameters
    ----------
    proxy_url:
        The HTTP/HTTPS forward-proxy URL.
    proxied_domains:
        List/set of domain names to proxy, e.g. ``["openrouter.ai", "tavily.com"]``.
    enabled_fn:
        Optional boolean or callable returning bool to dynamically enable/disable proxying.
    """

    def __init__(
        self,
        proxy_url: str,
        proxied_domains: list[str] | set[str],
        enabled_fn: Callable[[], bool] | bool = True,
        **kwargs: Any,
    ) -> None:
        self._proxy_url = proxy_url
        self._proxied_domains = {d.lower().strip() for d in proxied_domains}
        self._enabled_fn = enabled_fn
        self._direct_transport = httpx.AsyncHTTPTransport(**kwargs)
        self._proxy_transport = httpx.AsyncHTTPTransport(proxy=proxy_url, **kwargs)

    def _should_proxy(self, url: httpx.URL) -> bool:
        is_active = self._enabled_fn() if callable(self._enabled_fn) else bool(self._enabled_fn)
        if not is_active:
            return False
        host = url.host.lower()
        return any(host == d or host.endswith("." + d) for d in self._proxied_domains)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._should_proxy(request.url):
            return await self._proxy_transport.handle_async_request(request)
        return await self._direct_transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._proxy_transport.aclose()
        await self._direct_transport.aclose()
