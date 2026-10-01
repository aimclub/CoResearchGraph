"""A slow model is not a broken proxy.

Observed 2026-09-17, 15:04, on a real web run: the orchestrator had already
answered twice through OpenRouter (25 324 tokens billed on the run), went quiet
for the length of LLM__REQUEST_TIMEOUT, and the UI then told the operator

    **Error connecting to proxy server**
    ... ensure the proxy container is running, the corporate VPN is enabled ...

The proxy was up the whole time, and the log recorded nothing about the failure
at all, so the message sent the operator after a VPN that was not the problem.

``is_proxy_error`` matched on the substring "timeout", which every read timeout
carries. That had a second effect nobody would guess from the message:
``_is_transient`` short-circuits on ``is_proxy_error``, so a timeout — listed in
``_RETRYABLE_TYPES`` precisely so it gets retried — was never retried once
USE_PROXY was on.
"""
from __future__ import annotations

import pytest

from CoScientist.agents.common import _is_transient, is_proxy_error
from CoScientist.config import get_settings


@pytest.fixture(autouse=True)
def _proxy_on(monkeypatch):
    """The whole question only arises with the proxy enabled."""
    settings = get_settings()
    monkeypatch.setattr(settings.web, "use_proxy", True)
    monkeypatch.setattr(settings.services, "proxy_url", "http://10.0.0.1:7890")


class _Timeout(Exception):
    pass


class _APIConnectionError(Exception):
    pass


class _ProxyError(Exception):
    pass


# ── what a dead proxy actually looks like ────────────────────────────────────

@pytest.mark.parametrize("exc", [
    _ProxyError("ProxyError: Cannot connect to proxy"),
    _Timeout("All connection attempts failed"),
    _Timeout("ConnectTimeout: timed out while connecting to 10.0.0.1:7890"),
    _Timeout("[Errno 111] Connection refused"),
    _APIConnectionError("APIConnectionError: Connection error."),
])
def test_a_connect_phase_failure_is_a_proxy_error(exc):
    assert is_proxy_error(exc) is True


# ── what is not ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("exc", [
    _Timeout("litellm.Timeout: Connection timed out after 180.0 seconds"),
    _Timeout("Request timed out."),
    _Timeout("APITimeoutError: Request timeout"),
])
def test_a_read_timeout_is_not_a_proxy_error(exc):
    """The request reached the provider and the answer was slow. Saying "check
    the VPN" here is advice about the wrong machine."""
    assert is_proxy_error(exc) is False


def test_a_read_timeout_is_still_retried_with_the_proxy_on():
    """The retry _RETRYABLE_TYPES grants a timeout must survive USE_PROXY."""
    assert _is_transient(_Timeout("Connection timed out after 180.0 seconds")) is True


def test_a_provider_refusal_is_not_a_proxy_error():
    exc = Exception(
        "litellm.APIError: APIError: OpenAIException - "
        "Country, region, or territory not supported"
    )

    assert is_proxy_error(exc) is False


def test_nothing_is_a_proxy_error_when_the_proxy_is_off(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.web, "use_proxy", False)

    assert is_proxy_error(_ProxyError("Cannot connect to proxy")) is False
