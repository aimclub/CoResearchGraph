"""Shared agent initialisation helpers.

Every per-agent module imports from here so settings are resolved once and the
LLM/tooling setup is consistent across agents.
"""
import asyncio
import logging
import os
import re
from typing import Any, AsyncGenerator, Optional

import litellm
from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from CoScientist.config import get_settings
from CoScientist.execution_control import before_model_attempt, before_tool_action
from CoScientist.hitl.handler import ConsoleHITLHandler, DelegatingHITLHandler
from CoScientist.utils.selective_proxy import LiteLLMProxy

settings = get_settings()

_logger = logging.getLogger(__name__)

# Transient upstream failures (provider hiccups, rate limits, 5xx) that are worth
# retrying. OpenRouter wraps a flaky underlying provider as a BadRequestError with
# "Provider returned error", which litellm's own num_retries does NOT retry — so
# we retry around the whole model call ourselves.
_RETRYABLE_SUBSTRINGS = (
    "provider returned error",
    "rate limit",
    "ratelimit",
    "overloaded",
    "service unavailable",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "client has been closed",
    "502",
    "503",
    "504",
    "529",
)
_RETRYABLE_TYPES = (
    "RateLimitError",
    "Timeout",
    "TimeoutError",
    "APIConnectionError",
    "ServiceUnavailableError",
    "InternalServerError",
    "APIError",
)
_LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# Seconds a single model call may take, and — separately — the longest silence
# tolerated between streamed chunks. A provider that goes quiet raises nothing
# on its own, so without this the agent waits forever.
REQUEST_TIMEOUT = settings.llm.request_timeout
# Provider-side throttles clear on their own, but on the provider's clock, not
# ours: OpenRouter's 402 "in_flight_budget_exhausted" (a cap on concurrent spend
# — NOT an empty balance) ships a Retry-After of a minute or two. The generic
# backoff below burns its whole budget in ~7s, so these get their own, patient
# allowance and honour the Retry-After the provider actually sent.
_THROTTLE_MARKERS = (
    "in_flight_budget_exhausted",
    "in-flight requests",
    "openrouter_in_flight_budget",
    "rate limit",
    "ratelimit",
    "too many requests",
    "429",
)
_THROTTLE_MAX_RETRIES = int(os.getenv("LLM_THROTTLE_MAX_RETRIES", "6"))
_THROTTLE_MAX_WAIT_S = float(os.getenv("LLM_THROTTLE_MAX_WAIT", "180"))
_THROTTLE_DEFAULT_WAIT_S = float(os.getenv("LLM_THROTTLE_DEFAULT_WAIT", "60"))

# 402 answers two very different questions, and OpenRouter uses it for both: an
# exhausted account balance, and the in-flight spend cap above. Only the second
# clears on its own, and _THROTTLE_MARKERS already catches it — which is why the
# throttle check must run BEFORE this one. What is left is a billing fault: no
# amount of waiting fixes it, and letting it through to the transient path
# retries it three times per call in every agent, multiplying dead requests
# against a dead key.
#
# Providers word the body differently ("Insufficient credits", "add credits",
# "quota exceeded", a bare HTML error page), so this keys on the STATUS CODE:
# the exception attribute where litellm kept it, else the code as it is rendered
# into the message. Deliberately not a bare "402" substring — that would also
# fire on a request id, a token count or a timestamp that happens to contain it.
_PAYMENT_REQUIRED_RE = re.compile(
    r"\"?(?:status_?)?code\"?\s*[:=]\s*\"?402\b"  # "code":402 / status_code=402
    r"|\berror\s+code:\s*402\b"                     # Error code: 402 - {...}
    r"|\bhttp/?[\d.]*\s+402\b"                      # HTTP 402 / HTTP/1.1 402
    r"|\b402\s+payment[\s_-]?required\b",
    re.I,
)

# A provider fault should end the agent's turn, not the whole invocation: the
# orchestrator can still route around one dead stage, and main.py can still
# deliver a partial report. Set LLM_FAIL_SOFT=0 to get the old hard crash back.
_FAIL_SOFT = os.getenv("LLM_FAIL_SOFT", "1").strip().lower() not in ("0", "false", "no")
_PROVIDER_ERROR_MODULES = ("litellm", "openai", "httpx", "httpcore")

# Matches both the raw header ("Retry-After: 120") and the copy OpenRouter
# embeds in its JSON body ('"Retry-After":"120"').
_RETRY_AFTER_RE = re.compile(r'retry[-_ ]?after["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)', re.I)


def _short_err(err: Exception) -> str:
    """One-line rendering of *err* — provider bodies are multi-KB JSON blobs."""
    first = str(err).strip().splitlines()[0] if str(err).strip() else ""
    return f"{type(err).__name__}: {first[:300]}"


def _is_throttle(err: Exception) -> bool:
    """True for a provider throttle that resolves by simply waiting longer."""
    return any(m in str(err).lower() for m in _THROTTLE_MARKERS)


def _is_payment_required(err: Exception) -> bool:
    """True for a 402 that means the account cannot pay, not a passing throttle.

    Call only after :func:`_is_throttle` — the in-flight-budget 402 is a 402 too,
    and it is the one that resolves by waiting.
    """
    for source in (err, getattr(err, "response", None)):
        if source is None:
            continue
        try:
            if int(getattr(source, "status_code", None)) == 402:
                return True
        except (TypeError, ValueError):
            pass
    return bool(_PAYMENT_REQUIRED_RE.search(str(err)))


def _is_provider_error(err: Exception) -> bool:
    """True when *err* came out of the LLM/HTTP stack rather than our own code."""
    module = (type(err).__module__ or "").split(".", 1)[0]
    return module in _PROVIDER_ERROR_MODULES


def _retry_after_s(err: Exception) -> Optional[float]:
    """Seconds the provider asked us to wait, from response headers or body."""
    headers = getattr(err, "headers", None)
    response = getattr(err, "response", None)
    if not headers and response is not None:
        headers = getattr(response, "headers", None)
    if headers:
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
        except AttributeError:
            raw = None
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    match = _RETRY_AFTER_RE.search(str(err))
    return float(match.group(1)) if match else None


#: Connect-phase failures: the request never reached a provider, so with a
#: forward proxy configured the proxy is the thing to check.
_PROXY_ERROR_SUBSTRINGS = (
    "proxy",
    "proxyerror",
    "all connection attempts failed",
    "connection refused",
    "cannot connect to host",
    "failed to connect",
    "connecterror",
    "connecttimeout",
)
#: Ambiguous with a proxy in the path — the connection failed somewhere between
#: here and the provider — so these only count when a proxy is configured.
_PROXY_ERROR_TYPES = ("ProxyError", "ConnectError", "ConnectTimeout", "APIConnectionError")


def is_proxy_error(err: Exception) -> bool:
    """True when the failure looks like the forward proxy is unreachable.

    Deliberately narrow, and a read timeout is deliberately NOT one. "timeout"
    and "timed out" used to match here, so a model that simply took longer than
    LLM__REQUEST_TIMEOUT to answer was reported to the operator as a dead proxy
    — while the proxy was fine and the provider had just billed the run. Worse,
    ``_is_transient`` short-circuits on this function, so classifying a timeout
    as a proxy failure also cancelled the retry `_RETRYABLE_TYPES` grants it:
    one slow answer ended the turn instead of being tried again.
    """
    if not settings.web.use_proxy:
        return False
    msg = str(err).lower()
    if any(k in msg for k in _PROXY_ERROR_SUBSTRINGS):
        return True
    return bool(settings.services.proxy_url) and (
        type(err).__name__ in _PROXY_ERROR_TYPES or "connectionerror" in msg
    )


def _is_transient(err: Exception) -> bool:
    if is_proxy_error(err):
        return False
    if type(err).__name__ in _RETRYABLE_TYPES:
        return True
    msg = str(err).lower()
    return any(s in msg for s in _RETRYABLE_SUBSTRINGS)


_proxy_verified: bool = False
_proxy_verify_lock: Optional[asyncio.Lock] = None


def _get_proxy_verify_lock() -> asyncio.Lock:
    global _proxy_verify_lock
    if _proxy_verify_lock is None:
        _proxy_verify_lock = asyncio.Lock()
    return _proxy_verify_lock


def reset_proxy_verification() -> None:
    """Reset the proxy verification state so the next check performs a fresh probe."""
    global _proxy_verified
    _proxy_verified = False


async def verify_proxy_reachable(force: bool = False) -> None:
    """Fast HTTP probe to confirm the corporate proxy can reach upstream endpoints.

    When a corporate VPN is not connected, the local proxy container accepts
    TCP connections on localhost, but hangs indefinitely when trying to reach
    upstream endpoints. litellm's default request_timeout is 6000s (100 min),
    causing an infinite freeze in the UI. We probe an upstream endpoint THROUGH
    the proxy during system initialization to fail fast when VPN is off.
    Once verified, subsequent calls are no-ops unless force=True.
    """
    global _proxy_verified
    if _proxy_verified and not force:
        return

    if not settings.web.use_proxy or not settings.services.proxy_url:
        _proxy_verified = True
        return

    lock = _get_proxy_verify_lock()
    async with lock:
        if _proxy_verified and not force:
            return

        import httpx

        probe_url = "https://openrouter.ai/api/v1/models"
        try:
            async with httpx.AsyncClient(
                proxy=settings.services.proxy_url,
                timeout=httpx.Timeout(15.0, connect=10.0, read=15.0)
            ) as client:
                async with client.stream("GET", probe_url) as response:
                    pass
            _proxy_verified = True
            _logger.info("Proxy pre-flight check succeeded: proxy is reachable.")
        except Exception as err:
            _logger.warning("Proxy pre-flight probe failed: %s", err)
            raise ConnectionError(
                f"Error connecting to proxy server."
                f"Please ensure the proxy container is running, corporate VPN is enabled (other - disabled), "
                f"and proxy is accessible: {err}"
            ) from err


class RetryingLiteLlm(LiteLlm):
    """LiteLlm that retries the whole call on transient upstream errors.

    Only retries when nothing has been yielded yet (so a partial stream is never
    duplicated) and only for transient errors; everything else propagates.

    ``deadline_s`` (opt-in, per agent via ``llm_timeout:`` in system.yaml) caps
    the wait for the FIRST response. litellm's own timeout is httpx's, i.e. a
    per-read limit: a provider that dribbles bytes — or stalls after accepting
    the request — never trips it, and the call hangs for good. Agents that leave
    it unset keep exactly the previous behaviour.
    """

    # Pydantic private attribute (same pattern as LiteLlm._additional_args).
    # Consumed in __init__ before super(), so it never leaks into the kwargs
    # LiteLlm forwards to litellm's completion API.
    _deadline_s: Optional[float] = None

    def __init__(self, model: str, *, deadline_s: Optional[float] = None, **kwargs):
        super().__init__(model=model, **kwargs)
        self._deadline_s = deadline_s

    @property
    def capabilities(self) -> LlmCapabilities:
        """No native output schema next to tools.

        LiteLlm declares it, so ADK would send the JSON schema as the response
        format together with the tools. Providers behind OpenRouter then answer
        with the JSON straight away and never call a tool (seen with DeepSeek
        on the microfluidics design stages). Declaring it unsupported makes ADK
        add its set_model_response tool instead: the model calls its tools
        first and gives the structured answer through that tool at the end.
        Agents with an output schema and no tools are not affected.
        """
        return LlmCapabilities(
            **super().capabilities.model_dump() | {"output_schema_and_tools": False}
        )

    @staticmethod
    async def _verify_proxy_reachable(force: bool = False) -> None:
        await verify_proxy_reachable(force=force)

    async def _stream(self, llm_request: LlmRequest, stream: bool):
        """Yield the upstream response, bounding the wait between chunks.

        `deadline_s` below covers time-to-first-response and is disarmed the
        moment the provider answers, and litellm's own `timeout` covers getting
        the request away. Neither bounds the silence *after* the stream opens: a
        provider that sends one chunk and then stops leaves `async for` waiting
        on a chunk that never comes — socket established, nothing raised, the
        agent parked for good.

        Timing each chunk separately closes that gap without re-arming a whole
        stream deadline: the clock measures only the wait for the next chunk, so
        a legitimately long generation is safe and — unlike a deadline left
        armed across the `yield` — the consumer's own work is never timed.
        """
        source = super().generate_content_async(llm_request, stream=stream)
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        source.__anext__(), timeout=REQUEST_TIMEOUT
                    )
                except StopAsyncIteration:
                    return
                except asyncio.TimeoutError:
                    # Bare TimeoutError carries no message; say what happened so
                    # `_is_transient` recognises it and the log names the cause.
                    raise TimeoutError(
                        f"model sent nothing for {REQUEST_TIMEOUT}s — request timed out"
                    ) from None
                yield chunk
        finally:
            # Drop the stalled HTTP response rather than leaking the connection.
            await source.aclose()

    def _apply_dynamic_openrouter_provider(self, effective_model: str) -> None:
        """Dynamically sync provider routing from settings.web without breaking custom kwargs."""
        if not effective_model or not (effective_model.startswith("openrouter/") or effective_model.startswith("~")):
            return
        extra = _openrouter_provider_kwargs(effective_model)
        if extra and "extra_body" in extra:
            if not isinstance(getattr(self, "_additional_args", None), dict):
                self._additional_args = {}
            extra_body = self._additional_args.setdefault("extra_body", {})
            if isinstance(extra_body, dict):
                extra_body["provider"] = extra["extra_body"]["provider"]
        else:
            if isinstance(getattr(self, "_additional_args", None), dict):
                extra_body = self._additional_args.get("extra_body")
                if isinstance(extra_body, dict):
                    extra_body.pop("provider", None)
                    if not extra_body:
                        self._additional_args.pop("extra_body", None)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        effective_model = getattr(llm_request, "model", None) or getattr(self, "model", "") or ""
        # The proxy probe is network I/O too.  It does not consume model
        # budget, but a paused/stopped run must not start it.
        await before_tool_action(
            "model_provider_preflight", metadata={"model": effective_model}
        )
        if not _proxy_verified:
            await verify_proxy_reachable()
        self._apply_dynamic_openrouter_provider(effective_model)
        attempt = throttle_attempt = 0
        while True:
            yielded = False
            try:
                # Reserve immediately before each real provider attempt.  This
                # is inside our retry loop, so failed attempts count too.
                await before_model_attempt(
                    "adk_litellm",
                    metadata={"model": effective_model, "stream": bool(stream)},
                )
                if self._deadline_s is None:
                    async for resp in self._stream(llm_request, stream=stream):
                        yielded = True
                        yield resp
                else:
                    async with asyncio.timeout(self._deadline_s) as deadline:
                        async for resp in self._stream(
                            llm_request, stream=stream
                        ):
                            # The provider answered, so stop the clock. The budget
                            # covers time-to-first-response only: leaving it armed
                            # would also time a long stream — and, because we
                            # suspend right here, whatever the CONSUMER does with
                            # the response downstream.
                            deadline.reschedule(None)
                            yielded = True
                            yield resp
                return
            except Exception as err:  # noqa: BLE001 — classify then re-raise
                # A partial stream can never be replayed, and a proxy/VPN fault
                # is a local misconfiguration the user must fix — both stay hard
                # failures.
                if yielded or is_proxy_error(err):
                    raise

                if _is_throttle(err):
                    # The provider's own clock: honour Retry-After (capped, so a
                    # bogus header can't park the run for an hour).
                    throttle_attempt += 1
                    kind, attempt_n, max_r = "throttle", throttle_attempt, _THROTTLE_MAX_RETRIES
                    delay = min(
                        _retry_after_s(err) or _THROTTLE_DEFAULT_WAIT_S,
                        _THROTTLE_MAX_WAIT_S,
                    )
                elif _is_payment_required(err):
                    # After the throttle branch on purpose (see above): this is
                    # the 402 that waiting cannot clear, so it gets no retries.
                    kind, attempt_n, max_r, delay = "billing", 1, 0, 0.0
                elif _is_transient(err):
                    attempt += 1
                    kind, attempt_n, max_r = "transient", attempt, settings.web.max_retries
                    delay = min(1.5 ** attempt, 8.0)
                else:
                    kind, attempt_n, max_r, delay = "fatal", 1, 0, 0.0

                if attempt_n <= max_r:
                    _logger.warning(
                        "%s LLM error (attempt %d/%d), retrying in %.0fs: %s",
                        kind.capitalize(), attempt_n, max_r, delay, _short_err(err),
                    )
                    await asyncio.sleep(delay)
                    continue

                if kind == "billing":
                    # The one fault that is NOT one dead stage to route around:
                    # every agent after this one hits the same wall. Failing soft
                    # would walk the whole pipeline through it and hand back a run
                    # of blank answers — which is precisely how a spent account
                    # reads from the outside ("the agents stopped being called").
                    # This ends the run, and LLM_FAIL_SOFT does not cover it.
                    _logger.error(
                        "Provider billing fault (HTTP 402) — ending the RUN rather "
                        "than this agent turn, since every later stage would fail "
                        "the same way: %s",
                        _short_err(err),
                    )
                    raise

                # Out of retries. For everything else, ending the turn beats
                # ending the run: the orchestrator can route around one dead
                # stage and main.py can still deliver a partial report. Anything
                # we retried is by definition an upstream fault; a "fatal" one
                # qualifies only if it came out of the LLM/HTTP stack (e.g. a
                # context-window BadRequestError). Our own bugs still propagate.
                if not (_FAIL_SOFT and (kind != "fatal" or _is_provider_error(err))):
                    raise
                _logger.error(
                    "LLM call failed permanently (%s, %d attempt(s)); ending this "
                    "agent turn instead of the run: %s",
                    kind, attempt_n, _short_err(err),
                )
                message = (
                    f"The model provider could not be reached: {_short_err(err)}. "
                    "This step produced no result."
                )
                # The text has to travel in `content`, not only in
                # `error_message`: nothing in ADK or in this codebase ever reads
                # `error_message` back, and AgentTool hands the PARENT the last
                # content-bearing event. With empty content a dead stage is
                # indistinguishable from one that ran fine and had nothing to
                # say — the orchestrator moves on and the operator sees a blank
                # answer with no reason for it.
                #
                # The exception is a schema-bound request: ADK validates the
                # final text against `output_schema` (LlmAgent's output-key save,
                # and AgentTool again on the way out), so prose there would raise
                # and kill the run — exactly what fail-soft exists to prevent.
                # There the content stays empty and the ERROR log is the record.
                # Defensive on both hops: this is the error path, and a
                # crash here would take down the run fail-soft is saving.
                _cfg = getattr(llm_request, "config", None)
                schema_bound = bool(getattr(_cfg, "response_schema", None))
                yield LlmResponse(
                    error_code=type(err).__name__,
                    error_message=message,
                    content=None if schema_bound else types.Content(
                        role="model", parts=[types.Part(text=message)]
                    ),
                )
                return

MODEL = settings.llm.main_model
litellm.api_key = settings.llm.openai_api_key
litellm.request_timeout = 600.0
# Silence litellm's "Provider List: https://docs.litellm.ai/docs/providers" spam.
# It fires when litellm can't map a model prefix (e.g. "qwen/...") to a known
# provider during cost/token bookkeeping — harmless, but it floods the console.
litellm.suppress_debug_info = True

_litellm_proxy: LiteLLMProxy | None = None
if settings.services.proxy_url:
    _litellm_proxy = LiteLLMProxy(settings.services.proxy_url)
    if settings.web.use_proxy:
        _litellm_proxy.enable()


def sync_proxy_session() -> None:
    """Synchronize litellm proxy with the runtime ``use_proxy`` toggle."""
    if _litellm_proxy is not None:
        if settings.web.use_proxy:
            _litellm_proxy.enable()
        else:
            _litellm_proxy.disable()
            reset_proxy_verification()

hitl_handler = DelegatingHITLHandler(ConsoleHITLHandler())

# The CoderAgent runs on a dedicated (stronger) model — its multi-step tool-use
# benefits from more capability. Falls back to the main model when unset.
#
# Routing mirrors the other agents exactly: the provider prefix in the model
# string (e.g. "openrouter/qwen/...") selects the provider/base-URL, and the
# global `litellm.api_key` (set above) carries the key. We deliberately do NOT
# pass `api_base` here — doing so makes litellm strip the provider prefix, fail
# to re-infer the provider, and spam "Provider List: ..." warnings.
CODER_MODEL = settings.llm.coder_model or settings.llm.main_model


# ── Reasoning ("thinking") control ───────────────────────────────────────────
# A hybrid model that reasons before every answer spends seconds (and tokens)
# on it. Agents that only route, reformat or fill a schema do not need that, so
# `reasoning:` in system.yaml turns it off — or down — per agent.
#
# The two provider dialects: OpenRouter takes a top-level `reasoning` object
# ({"enabled": false} / {"effort": ...}); everyone else takes OpenAI's
# `reasoning_effort`, which we send with drop_params so a model that has no
# such parameter is not an error. A model that ALWAYS reasons (deepseek-r1,
# the o-series) ignores the request — only hybrid models can be silenced.
REASONING_EFFORTS = ("minimal", "low", "medium", "high")
REASONING_OFF = ("off", "none", "disabled")


def _reasoning_kwargs(model: str, spec: Optional[Any]) -> dict:
    """litellm kwargs implementing a ``reasoning:`` declaration for *model*.

    ``None`` (unset) yields no kwargs at all, so the provider default stands
    and agents that say nothing keep exactly the previous behaviour.
    """
    if spec is None:
        return {}
    openrouter = model.startswith("openrouter/")
    if spec is True:
        return {"reasoning": {"enabled": True}} if openrouter else {}
    value = "off" if spec is False else str(spec).strip().lower()
    if value in REASONING_OFF:
        if openrouter:
            return {"reasoning": {"enabled": False}}
        return {"reasoning_effort": "none", "drop_params": True}
    if value in REASONING_EFFORTS:
        if openrouter:
            return {"reasoning": {"effort": value}}
        return {"reasoning_effort": value, "drop_params": True}
    raise ValueError(
        f"reasoning must be a bool, one of {REASONING_OFF} or {REASONING_EFFORTS}, "
        f"got {spec!r}"
    )


def _openrouter_provider_kwargs(model: str) -> dict:
    """litellm kwargs implementing OpenRouter provider routing for *model*.

    Returns an empty dict if the model is not routed through OpenRouter,
    or if default routing is active with no specific provider ordering.
    """
    if not (model.startswith("openrouter/") or model.startswith("~")):
        return {}
    web = settings.web
    sort_val = (getattr(web, "openrouter_provider_sort", None) or "default").strip().lower()
    order_val = getattr(web, "openrouter_provider_order", None)

    provider_cfg = {}
    if sort_val in ("price", "throughput", "latency"):
        provider_cfg["sort"] = sort_val
    if order_val:
        if isinstance(order_val, str):
            providers = [p.strip() for p in order_val.split(",") if p.strip()]
        elif isinstance(order_val, (list, tuple)):
            providers = [str(p).strip() for p in order_val if str(p).strip()]
        else:
            providers = []
        if providers:
            provider_cfg["order"] = providers

    if not provider_cfg:
        return {}
    return {"extra_body": {"provider": provider_cfg}}


def _combine_llm_kwargs(*kwarg_dicts: dict) -> dict:
    """Merge multiple kwargs dicts for LiteLlm, combining extra_body cleanly."""
    merged = {}
    for d in kwarg_dicts:
        for k, v in d.items():
            if k == "extra_body" and isinstance(v, dict) and isinstance(merged.get("extra_body"), dict):
                merged["extra_body"] = {**merged["extra_body"], **v}
            else:
                merged[k] = v
    return merged


def make_llm(
    model: str = MODEL,
    *,
    deadline_s: Optional[float] = None,
    reasoning: Optional[Any] = None,
) -> LiteLlm:
    """Return a (retry-wrapped) LiteLlm for the main model (or an override)."""
    kwargs = _combine_llm_kwargs(
        _reasoning_kwargs(model, reasoning),
        _openrouter_provider_kwargs(model),
    )
    return RetryingLiteLlm(
        model=model, deadline_s=deadline_s, timeout=REQUEST_TIMEOUT,
        num_retries=0,
        **kwargs
    )


def make_coder_llm(
    *, deadline_s: Optional[float] = None, reasoning: Optional[Any] = None
) -> LiteLlm:
    """Return a (retry-wrapped) LiteLlm for the dedicated coder model."""
    kwargs = _combine_llm_kwargs(
        _reasoning_kwargs(CODER_MODEL, reasoning),
        _openrouter_provider_kwargs(CODER_MODEL),
    )
    return RetryingLiteLlm(
        model=CODER_MODEL,
        deadline_s=deadline_s,
        timeout=REQUEST_TIMEOUT,
        num_retries=0,
        **kwargs
    )
