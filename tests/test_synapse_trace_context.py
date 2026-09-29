"""Unit contracts for canonical Synapse trace scopes."""

from __future__ import annotations

import asyncio

from opentelemetry import baggage, context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

TRACE_1 = "0af7651916cd43dd8448eb211c80319c"
TRACE_2 = "1af7651916cd43dd8448eb211c80319c"
SPAN_1 = "b7ad6b7169203331"
SPAN_2 = "c7ad6b7169203331"


def _traceparent(trace_id: str, span_id: str) -> str:
    return f"00-{trace_id}-{span_id}-01"


def _provider(trace_context):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(trace_context.RunIdSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_registered_context_wins_over_foreign_headers():
    from CoScientist.checkpoints import synapse, trace_context

    provider, exporter = _provider(trace_context)
    tracer = provider.get_tracer("tests.registered")
    synapse.clear_runs()
    synapse.register_run("ctx-1", "run-1", _traceparent(TRACE_1, SPAN_1))

    headers = {
        "traceparent": _traceparent(TRACE_2, SPAN_2),
        "baggage": "run_id=foreign-run",
    }
    with trace_context.scope_for_request("ctx-1", headers):
        assert trace_context.active_run_id() == "run-1"
        with tracer.start_as_current_span("registered"):
            pass

    span = exporter.get_finished_spans()[0]
    assert format(span.context.trace_id, "032x") == TRACE_1
    assert format(span.parent.span_id, "016x") == SPAN_1
    assert span.attributes["run_id"] == "run-1"
    assert trace_context.active_run_id() is None
    provider.shutdown()


def test_registered_context_without_valid_root_rejects_foreign_binding():
    from CoScientist.checkpoints import synapse, trace_context

    synapse.clear_runs()
    synapse.register_run("ctx-null", "run-null", None)
    headers = {
        "traceparent": _traceparent(TRACE_2, SPAN_2),
        "baggage": "run_id=foreign-run",
    }

    ambient = TraceContextTextMapPropagator().extract(
        {"traceparent": _traceparent(TRACE_2, SPAN_2)},
        context=context.Context(),
    )
    ambient = baggage.set_baggage("run_id", "ambient-foreign", context=ambient)
    token = context.attach(ambient)
    try:
        with trace_context.scope_for_request("ctx-null", headers):
            assert trace_context.active_run_id() is None
    finally:
        context.detach(token)

    assert trace_context.active_run_id() is None


def test_unregistered_valid_distributed_context_is_accepted():
    from CoScientist.checkpoints import synapse, trace_context

    provider, exporter = _provider(trace_context)
    tracer = provider.get_tracer("tests.remote")
    synapse.clear_runs()
    headers = {
        "traceparent": _traceparent(TRACE_2, SPAN_2),
        "baggage": "run_id=remote-run",
    }

    with trace_context.scope_for_request("new-remote-context", headers):
        assert trace_context.active_run_id() == "remote-run"
        with tracer.start_as_current_span("remote"):
            pass

    span = exporter.get_finished_spans()[0]
    assert format(span.context.trace_id, "032x") == TRACE_2
    assert format(span.parent.span_id, "016x") == SPAN_2
    assert span.attributes["run_id"] == "remote-run"
    provider.shutdown()


def test_unregistered_trace_headers_do_not_authorize_checkpoint(monkeypatch):
    from CoScientist.checkpoints import synapse, trace_context

    synapse.clear_runs()
    monkeypatch.setattr(trace_context, "_proof_secret", lambda: b"demo-secret")
    headers = {
        "traceparent": _traceparent(TRACE_2, SPAN_2),
        "baggage": "run_id=known-run",
    }
    with trace_context.scope_for_request("unregistered", headers):
        assert trace_context.active_run_id() == "known-run"
        assert trace_context.trusted_active_run_id() is None


def test_registered_parent_signs_child_run_context(monkeypatch):
    from CoScientist.checkpoints import synapse, trace_context

    synapse.clear_runs()
    synapse.register_run("root-context", "run-registered", _traceparent(TRACE_1, SPAN_1))
    monkeypatch.setattr(trace_context, "_proof_secret", lambda: b"demo-secret")
    with trace_context.scope_for_request("root-context", {}):
        headers = trace_context.outbound_carrier()
    assert "x-coscientist-run-proof" in headers
    with trace_context.scope_for_request("child-context", headers):
        assert trace_context.trusted_active_run_id() == "run-registered"
    headers["baggage"] = "run_id=other-run"
    with trace_context.scope_for_request("forged-child", headers):
        assert trace_context.trusted_active_run_id() is None


def test_two_concurrent_scopes_do_not_mix_run_identity():
    from CoScientist.checkpoints import synapse, trace_context

    provider, exporter = _provider(trace_context)
    tracer = provider.get_tracer("tests.concurrent")
    synapse.clear_runs()
    synapse.register_run("ctx-1", "run-1", _traceparent(TRACE_1, SPAN_1))
    synapse.register_run("ctx-2", "run-2", _traceparent(TRACE_2, SPAN_2))
    barrier = asyncio.Barrier(2)

    async def worker(context_id: str, run_id: str):
        with trace_context.scope_for_request(context_id, {}):
            await barrier.wait()
            assert trace_context.active_run_id() == run_id
            with tracer.start_as_current_span(f"concurrent {run_id}"):
                await barrier.wait()

    async def exercise():
        await asyncio.gather(worker("ctx-1", "run-1"), worker("ctx-2", "run-2"))

    asyncio.run(exercise())

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert format(spans["concurrent run-1"].context.trace_id, "032x") == TRACE_1
    assert spans["concurrent run-1"].attributes["run_id"] == "run-1"
    assert format(spans["concurrent run-2"].context.trace_id, "032x") == TRACE_2
    assert spans["concurrent run-2"].attributes["run_id"] == "run-2"
    assert trace_context.active_run_id() is None
    provider.shutdown()


def test_invalid_or_incomplete_unregistered_headers_are_noop():
    from CoScientist.checkpoints import synapse, trace_context

    synapse.clear_runs()
    for headers in (
        {},
        {"traceparent": "invalid", "baggage": "run_id=bad"},
        {"traceparent": _traceparent(TRACE_1, SPAN_1)},
        {"baggage": "run_id=missing-parent"},
    ):
        with trace_context.scope_for_request(None, headers):
            assert trace_context.active_run_id() is None
        assert trace_context.active_run_id() is None
