"""Outbound A2A tracing must use SDK hooks without mutating science state."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from a2a.client.middleware import ClientCallContext
from google.adk.a2a.agent.config import ParametersConfig
from opentelemetry import baggage, trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.textmap import Getter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

TRACE_1 = "2af7651916cd43dd8448eb211c80319c"
TRACE_2 = "3af7651916cd43dd8448eb211c80319c"
SPAN_1 = "d7ad6b7169203331"
SPAN_2 = "e7ad6b7169203331"


class _DictGetter(Getter):
    def get(self, carrier, key):
        value = carrier.get(key)
        return [value] if value is not None else None

    def keys(self, carrier):
        return list(carrier)


def _traceparent(trace_id: str, span_id: str) -> str:
    return f"00-{trace_id}-{span_id}-01"


def _params(state: dict) -> ParametersConfig:
    return ParametersConfig(client_call_context=ClientCallContext(state=state))


def _headers(params: ParametersConfig) -> dict:
    return params.client_call_context.state["http_kwargs"]["headers"]


def test_outbound_interceptor_copy_on_write_preserves_science_state():
    from CoScientist.a2a.synapse_tracing import make_remote_agent_config
    from CoScientist.checkpoints import synapse, trace_context

    original_headers = {"Authorization": "Bearer existing", "X-Custom": "value"}
    original_http_kwargs = {"headers": original_headers, "timeout": 17}
    science_state = {
        "hypothesis": {"target": "GSK3B"},
        "http_kwargs": original_http_kwargs,
    }
    params = _params(science_state)
    hook = make_remote_agent_config().request_interceptors[0].before_request
    synapse.clear_runs()
    synapse.register_run("ctx-1", "run-1", _traceparent(TRACE_1, SPAN_1))

    async def invoke():
        with trace_context.scope_for_request("ctx-1", {}):
            return await hook(
                SimpleNamespace(session=SimpleNamespace(state=science_state)),
                SimpleNamespace(),
                params,
            )

    _request, outgoing = asyncio.run(invoke())
    headers = _headers(outgoing)

    assert science_state == {
        "hypothesis": {"target": "GSK3B"},
        "http_kwargs": original_http_kwargs,
    }
    assert science_state["http_kwargs"] is original_http_kwargs
    assert science_state["http_kwargs"]["headers"] is original_headers
    assert outgoing.client_call_context is not params.client_call_context
    assert outgoing.client_call_context.state is not science_state
    assert outgoing.client_call_context.state["http_kwargs"] is not original_http_kwargs
    assert headers is not original_headers
    assert headers["Authorization"] == "Bearer existing"
    assert headers["X-Custom"] == "value"
    assert headers["traceparent"] == _traceparent(TRACE_1, SPAN_1)
    assert "run_id=run-1" in headers["baggage"]


def test_concurrent_outbound_calls_get_isolated_headers():
    from CoScientist.a2a.synapse_tracing import make_remote_agent_config
    from CoScientist.checkpoints import synapse, trace_context

    provider = TracerProvider()
    tracer = provider.get_tracer("tests.outbound")
    hook = make_remote_agent_config().request_interceptors[0].before_request
    shared_state = {"http_kwargs": {"headers": {"Authorization": "Bearer shared"}}}
    synapse.clear_runs()
    synapse.register_run("ctx-1", "run-1", _traceparent(TRACE_1, SPAN_1))
    synapse.register_run("ctx-2", "run-2", _traceparent(TRACE_2, SPAN_2))
    barrier = asyncio.Barrier(2)

    async def invoke(context_id: str):
        with trace_context.scope_for_request(context_id, {}):
            with tracer.start_as_current_span(f"caller {context_id}"):
                await barrier.wait()
                _request, outgoing = await hook(
                    SimpleNamespace(session=SimpleNamespace(state=shared_state)),
                    SimpleNamespace(),
                    _params(shared_state),
                )
                return _headers(outgoing)

    async def exercise():
        return await asyncio.gather(invoke("ctx-1"), invoke("ctx-2"))

    first, second = asyncio.run(exercise())
    assert first is not second
    assert "run_id=run-1" in first["baggage"]
    assert "run_id=run-2" in second["baggage"]
    assert shared_state == {
        "http_kwargs": {"headers": {"Authorization": "Bearer shared"}}
    }

    propagator = CompositePropagator(
        [TraceContextTextMapPropagator(), W3CBaggagePropagator()]
    )
    first_context = propagator.extract(first, getter=_DictGetter())
    second_context = propagator.extract(second, getter=_DictGetter())
    assert trace.get_current_span(first_context).get_span_context().trace_id == int(
        TRACE_1, 16
    )
    assert trace.get_current_span(second_context).get_span_context().trace_id == int(
        TRACE_2, 16
    )
    assert baggage.get_baggage("run_id", first_context) == "run-1"
    assert baggage.get_baggage("run_id", second_context) == "run-2"
    provider.shutdown()


def test_invalid_science_http_state_does_not_break_remote_request(caplog):
    from CoScientist.a2a.synapse_tracing import make_remote_agent_config
    from CoScientist.checkpoints import synapse, trace_context

    science_state = {"http_kwargs": "scientific-value"}
    params = _params(science_state)
    hook = make_remote_agent_config().request_interceptors[0].before_request
    request = SimpleNamespace()
    synapse.clear_runs()
    synapse.register_run("ctx-1", "run-1", _traceparent(TRACE_1, SPAN_1))

    async def invoke():
        with trace_context.scope_for_request("ctx-1", {}):
            return await hook(
                SimpleNamespace(session=SimpleNamespace(state=science_state)),
                request,
                params,
            )

    outgoing_request, outgoing_params = asyncio.run(invoke())
    assert outgoing_request is request
    assert outgoing_params is params
    assert science_state == {"http_kwargs": "scientific-value"}
    assert "outbound trace headers skipped" in caplog.text
