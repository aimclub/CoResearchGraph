from __future__ import annotations

import hashlib
import hmac
import logging
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Iterator, Mapping, Optional

from opentelemetry import baggage, context, trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import SpanContext
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)
_RUN_ID_KEY = "run_id"
_RUN_PROOF_HEADER = "x-coscientist-run-proof"
_TRUSTED_RUN: ContextVar[Optional[str]] = ContextVar("coscientist_trusted_run", default=None)
_PROPAGATOR = CompositePropagator(
    [TraceContextTextMapPropagator(), W3CBaggagePropagator()]
)


def _headers(carrier: Optional[Mapping[str, str]]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in (carrier or {}).items()}


def _extracted(carrier: Optional[Mapping[str, str]]):
    return _PROPAGATOR.extract(_headers(carrier), context=context.Context())


def _span_context(otel_context) -> SpanContext:
    return trace.get_current_span(otel_context).get_span_context()


def _registered_context(context_id: Optional[str]):
    if not context_id:
        return None
    from CoScientist.checkpoints.synapse import registered_run_context

    return registered_run_context(context_id)


def _proof_secret() -> Optional[bytes]:
    from CoScientist.config import get_settings

    settings = get_settings()
    token = settings.checkpoints.api_token
    if not settings.synapse.enabled or not settings.checkpoints.enabled or token is None:
        return None
    return token.get_secret_value().encode("utf-8")


def _run_proof(headers: Mapping[str, str]) -> Optional[str]:
    secret = _proof_secret()
    traceparent = headers.get("traceparent")
    baggage_header = headers.get("baggage")
    if not secret or not traceparent or not baggage_header:
        return None
    payload = f"coscientist-a2a-run-v1\n{traceparent}\n{baggage_header}".encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def _has_valid_proof(headers: Mapping[str, str]) -> bool:
    proof = headers.get(_RUN_PROOF_HEADER)
    expected = _run_proof(headers)
    return bool(proof and expected and hmac.compare_digest(proof, expected))


def _resolve(context_id: Optional[str], headers: Optional[Mapping[str, str]]):
    registered = _registered_context(context_id)
    incoming = _extracted(headers)
    incoming_span = _span_context(incoming)

    if registered is not None:
        registered_parent = _extracted(
            {"traceparent": registered.get("traceparent") or ""}
        )
        registered_span = _span_context(registered_parent)
        if not registered_span.is_valid:
            return context.Context(), None, False
        parent = (
            incoming
            if incoming_span.is_valid
            and incoming_span.trace_id == registered_span.trace_id
            else registered_parent
        )
        return parent, registered["run_id"], True

    run_id = baggage.get_baggage(_RUN_ID_KEY, incoming)
    if not incoming_span.is_valid or not isinstance(run_id, str) or not run_id:
        return None
    return incoming, run_id, _has_valid_proof(_headers(headers))


@contextmanager
def scope_for_request(
    context_id: Optional[str], headers: Optional[Mapping[str, str]]
) -> Iterator[None]:
    try:
        resolved = _resolve(context_id, headers)
    except Exception as exc:  # noqa: BLE001 - tracing cannot break science
        logger.warning("synapse: trace context resolution failed: %s", exc)
        resolved = None
    if resolved is None:
        trust_token = _TRUSTED_RUN.set(None)
        try:
            yield
        finally:
            _TRUSTED_RUN.reset(trust_token)
        return

    parent, run_id, trusted = resolved
    scoped = (
        baggage.set_baggage(_RUN_ID_KEY, run_id, context=parent)
        if run_id is not None
        else parent
    )
    token = context.attach(scoped)
    trust_token = _TRUSTED_RUN.set(run_id if trusted else None)
    try:
        yield
    finally:
        _TRUSTED_RUN.reset(trust_token)
        context.detach(token)


def active_run_id() -> Optional[str]:
    run_id = baggage.get_baggage(_RUN_ID_KEY)
    span_context = trace.get_current_span().get_span_context()
    return run_id if isinstance(run_id, str) and span_context.is_valid else None


def trusted_active_run_id() -> Optional[str]:
    """Only registered or signed scopes may supply a checkpoint's run identity."""
    run_id = active_run_id()
    return run_id if run_id is not None and run_id == _TRUSTED_RUN.get() else None


def outbound_carrier() -> dict[str, str]:
    if active_run_id() is None:
        return {}
    carrier: dict[str, str] = {}
    _PROPAGATOR.inject(carrier)
    if trusted_active_run_id() is not None:
        proof = _run_proof(carrier)
        if proof is not None:
            carrier[_RUN_PROOF_HEADER] = proof
    return carrier


class RunIdSpanProcessor(SpanProcessor):
    """Stamp native SDK spans at creation without exporting or blocking."""

    def on_start(self, span: Span, parent_context=None) -> None:
        run_id = baggage.get_baggage(_RUN_ID_KEY, parent_context)
        parent = _span_context(parent_context)
        if (
            isinstance(run_id, str)
            and run_id
            and parent.is_valid
            and parent.trace_id == span.get_span_context().trace_id
        ):
            span.set_attribute(_RUN_ID_KEY, run_id)

    def on_end(self, span: ReadableSpan) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
