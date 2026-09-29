"""Fresh-thread HTTP proof for RemoteA2aAgent W3C Run propagation."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _synapse_native_trace_probe import (  # noqa: E402
    _card,
    _CollectorHandler,
    _decode_spans,
)

TRACE_ID = "5af7651916cd43dd8448eb211c80319c"
ROOT_SPAN_ID = "17ad6b7169203331"
ROOT_CONTEXT_ID = "ctx-caller"
RUN_ID = "run-distributed"


def _free_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return candidate.getsockname()[1]


async def _run_remote_agent(card):
    from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from CoScientist.a2a.synapse_tracing import make_remote_agent_config
    from CoScientist.checkpoints import synapse
    from CoScientist.checkpoints.trace_context import scope_for_request

    science_state = {"hypothesis": {"target": "GSK3B"}}
    sessions = InMemorySessionService()
    await sessions.create_session(
        app_name="caller-app",
        user_id="caller-user",
        session_id=ROOT_CONTEXT_ID,
        state=science_state,
    )
    remote = RemoteA2aAgent(
        name="RemoteTiny",
        agent_card=card,
        config=make_remote_agent_config(),
    )
    runner = Runner(agent=remote, app_name="caller-app", session_service=sessions)
    events = []
    with scope_for_request(ROOT_CONTEXT_ID, {}):
        async for event in runner.run_async(
            user_id="caller-user",
            session_id=ROOT_CONTEXT_ID,
            new_message=types.Content(
                role="user", parts=[types.Part(text="distributed trace")]
            ),
        ):
            events.append(event)
    session = await sessions.get_session(
        app_name="caller-app", user_id="caller-user", session_id=ROOT_CONTEXT_ID
    )
    response_contexts = [
        event.custom_metadata.get("a2a:context_id")
        for event in events
        if event.custom_metadata and event.custom_metadata.get("a2a:context_id")
    ]
    return {
        "event_count": len(events),
        "science_state": dict(session.state),
        "remote_contexts": response_contexts,
        "remote_registered": [synapse.run_id_for(value) for value in response_contexts],
    }


def main() -> None:
    os.environ["A2A_DISABLE_OPIK"] = "1"
    os.environ["LOG_AGENT_EVENTS"] = "0"
    import uvicorn
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.events.event import Event
    from google.genai import types
    from opentelemetry import trace

    from CoScientist.a2a.server import make_a2a_app
    from CoScientist.checkpoints import synapse
    from CoScientist.config import settings

    class RemoteTiny(BaseAgent):
        async def _run_async_impl(self, ctx):
            with trace.get_tracer("tests.remote.socket").start_as_current_span(
                f"remote-probe {ctx.session.id}"
            ):
                yield Event(
                    author=self.name,
                    invocation_id=ctx.invocation_id,
                    content=types.Content(
                        role="model", parts=[types.Part(text="remote-complete")]
                    ),
                )

    _CollectorHandler.bodies = []
    _CollectorHandler.received.clear()
    collector = __import__("http.server").server.ThreadingHTTPServer(
        ("127.0.0.1", 0), _CollectorHandler
    )
    collector_thread = threading.Thread(target=collector.serve_forever, daemon=True)
    collector_thread.start()
    server = None
    server_thread = None
    try:
        endpoint = f"http://127.0.0.1:{collector.server_port}/v1/traces"
        settings.checkpoints.enabled = False
        settings.synapse.enabled = True
        settings.synapse.otlp_endpoint = endpoint
        synapse.clear_runs()
        synapse._TRACER = None
        synapse._OTEL_READY = False
        synapse.register_run(
            ROOT_CONTEXT_ID,
            RUN_ID,
            f"00-{TRACE_ID}-{ROOT_SPAN_ID}-01",
        )

        port = _free_port()
        card = _card("RemoteTiny", f"http://127.0.0.1:{port}/")
        app = make_a2a_app(RemoteTiny(name="RemoteTiny"), card, "remote-app")
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        )
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("remote A2A server did not start")

        result = asyncio.run(_run_remote_agent(card))
        trace.get_tracer_provider().force_flush(timeout_millis=2000)
        _CollectorHandler.received.wait(timeout=2)
        result["spans"] = _decode_spans(_CollectorHandler.bodies)
        print("REMOTE_TRACE_PROBE=" + json.dumps(result, sort_keys=True))
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=3)
        collector.shutdown()
        collector.server_close()
        collector_thread.join(timeout=2)


if __name__ == "__main__":
    main()
