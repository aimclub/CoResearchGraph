"""Fresh-process probe for the real A2A HTTP -> ADK native tracing path."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
ROOT_SPAN_ID = "b7ad6b7169203331"
CONTEXT_ID = "ctx-native"
RUN_ID = "run-native"


class _CollectorHandler(BaseHTTPRequestHandler):
    bodies: list[bytes] = []
    received = threading.Event()

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("content-length", "0"))
        self.bodies.append(self.rfile.read(length))
        self.received.set()
        self.send_response(200)
        self.send_header("content-type", "application/x-protobuf")
        self.send_header("content-length", "0")
        self.end_headers()

    def log_message(self, *_args):
        return


def _attribute_value(value):
    field = value.WhichOneof("value")
    if field == "array_value":
        return [_attribute_value(item) for item in value.array_value.values]
    if field == "kvlist_value":
        return {
            item.key: _attribute_value(item.value) for item in value.kvlist_value.values
        }
    return getattr(value, field) if field else None


def _decode_spans(bodies: list[bytes]) -> list[dict]:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    spans = []
    for body in bodies:
        request = ExportTraceServiceRequest.FromString(body)
        for resource_spans in request.resource_spans:
            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    spans.append(
                        {
                            "name": span.name,
                            "trace_id": span.trace_id.hex(),
                            "span_id": span.span_id.hex(),
                            "parent_span_id": span.parent_span_id.hex(),
                            "attributes": {
                                item.key: _attribute_value(item.value)
                                for item in span.attributes
                            },
                        }
                    )
    return spans


def _card(name: str, url: str):
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    return AgentCard(
        name=name,
        description="native trace probe",
        url=url,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[AgentSkill(id="probe", name="probe", description="probe", tags=[])],
    )


def _message_payload(context_id: str, *, streaming: bool = False) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": f"request-{context_id}",
        "method": "message/stream" if streaming else "message/send",
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "messageId": f"message-{context_id}",
                "contextId": context_id,
                "parts": [{"kind": "text", "text": "trace me"}],
            }
        },
    }


async def _exercise_app(endpoint: str) -> dict:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.events.event import Event
    from google.genai import types
    from opentelemetry import trace

    from CoScientist.a2a.server import make_a2a_app
    from CoScientist.checkpoints import synapse
    from CoScientist.config import settings

    class TinyAgent(BaseAgent):
        async def _run_async_impl(self, ctx):
            tracer = trace.get_tracer("coscientist.tests.native")
            with tracer.start_as_current_span(f"probe {ctx.session.id}"):
                await asyncio.sleep(0)
                yield Event(
                    author=self.name,
                    invocation_id=ctx.invocation_id,
                    content=types.Content(
                        role="model", parts=[types.Part(text="science-complete")]
                    ),
                )

    settings.checkpoints.enabled = False
    settings.synapse.enabled = True
    settings.synapse.otlp_endpoint = endpoint
    synapse.clear_runs()
    synapse._TRACER = None
    synapse._OTEL_READY = False
    synapse.register_run(
        CONTEXT_ID,
        RUN_ID,
        f"00-{TRACE_ID}-{ROOT_SPAN_ID}-01",
    )

    app = make_a2a_app(
        TinyAgent(name="TinyAgent"),
        _card("TinyAgent", "http://probe/"),
        "trace-probe",
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://probe"
    ) as client:
        response = await client.post("/", json=_message_payload(CONTEXT_ID))
    provider = trace.get_tracer_provider()
    provider.force_flush(timeout_millis=2000)
    return {"status": response.status_code, "body": response.json()}


def main() -> None:
    os.environ["A2A_DISABLE_OPIK"] = "1"
    os.environ["LOG_AGENT_EVENTS"] = "0"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CollectorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/traces"
        result = asyncio.run(_exercise_app(endpoint))
        _CollectorHandler.received.wait(timeout=2.0)
        result["spans"] = _decode_spans(_CollectorHandler.bodies)
        print("TRACE_PROBE=" + json.dumps(result, sort_keys=True))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
