"""Real HTTP/ADK checkpoint regression: admin access, stalled callbacks, restore.

Two A2A servers share one event loop. Scripted agents make zero LLM calls.
Run: LLM__MAIN_MODEL=test-model uv run --frozen python tests/e2e_checkpoint_control.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import httpx
from _synapse_native_trace_probe import _card, _decode_spans, _message_payload

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.getenv("E2E_CONTROL_PORT", "8148"))
BASE = f"http://127.0.0.1:{PORT}"
TOKEN = "test-only-platform-admin-credential"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
TRACE1, TRACE2 = "1" * 32, "2" * 32
SPAN = "a" * 16


class Receiver(BaseHTTPRequestHandler):
    entered = threading.Event()
    points: ClassVar[list] = []
    spans: ClassVar[list] = []

    def do_POST(self):
        data = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/v1/traces":
            self.spans.append(data)
        elif not self.entered.is_set():
            self.entered.set()
            # Real HTTP read timeout: the first callback is not acknowledged.
            time.sleep(5.5)
        else:
            self.points.append(json.loads(data))
        try:
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_):
        pass


async def serve():
    def local_network_only(event, args):
        if event not in {"socket.connect", "socket.getaddrinfo"}:
            return
        address = args[1] if event == "socket.connect" else args[0]
        host = address[0] if isinstance(address, tuple) else address
        if host is not None and host not in {"127.0.0.1", "localhost", "::1"}:
            sys.stderr.write(
                f"UNEXPECTED_NETWORK: blocked host={host!r} event={event}\n"
            )
            raise AssertionError("External connections are forbidden in this probe")

    sys.addaudithook(local_network_only)
    import uvicorn
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.events.event import Event
    from google.adk.events.event_actions import EventActions
    from google.genai import types

    from CoScientist.a2a.server import make_a2a_app

    class ScriptedAgent(BaseAgent):
        async def _run_async_impl(self, ctx):
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=types.Content(
                    role="model", parts=[types.Part(text="scripted result")]
                ),
                actions=EventActions(state_delta={"search_results": "local fixture"}),
            )

    servers = []
    for index in range(2):
        name = f"Scripted{index}"
        app = make_a2a_app(
            ScriptedAgent(name=name),
            _card(name, f"http://127.0.0.1:{PORT + index}/"),
            f"app{index}",
        )
        servers.append(
            uvicorn.Server(
                uvicorn.Config(
                    app,
                    host="127.0.0.1",
                    port=PORT + index,
                    log_level="warning",
                )
            )
        )
    import logging

    logging.getLogger("CoScientist.checkpoints.notifications").addHandler(
        logging.StreamHandler()
    )
    await asyncio.gather(*(server.serve() for server in servers))


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition timed out")


def main():
    receiver = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    threading.Thread(target=receiver.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{receiver.server_port}"
    result = {}
    with tempfile.TemporaryDirectory(prefix="control-live-") as temp:
        env = {
            **os.environ,
            "CHECKPOINTS__ENABLED": "1",
            "CHECKPOINTS__DIR": f"{temp}/snapshots",
            "CHECKPOINTS__API_TOKEN": TOKEN,
            "SYNAPSE__ENABLED": "1",
            "SYNAPSE__CALLBACK_URL": endpoint,
            "SYNAPSE__BUNDLE_BASE_URL": BASE,
            "SYNAPSE__OTLP_ENDPOINT": f"{endpoint}/v1/traces",
            "OTEL_BSP_SCHEDULE_DELAY": "100",
            "A2A_DISABLE_OPIK": "1",
            "LOG_AGENT_EVENTS": "0",
            "LLM__MAIN_MODEL": "test-model",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "RESEARCH_GRAPH__DIR": f"{temp}/graphs",
        }
        log_path = Path(temp) / "server.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, __file__, "--serve"],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        try:
            with httpx.Client(timeout=10) as client:

                def ready():
                    assert process.poll() is None, log_path.read_text()
                    try:
                        return all(
                            client.get(
                                f"http://127.0.0.1:{PORT+i}/.well-known/agent-card.json"
                            ).status_code
                            == 200
                            for i in range(2)
                        )
                    except httpx.TransportError:
                        return False

                wait_for(ready, 90)

                def control(method, path, **kwargs):
                    response = client.request(
                        method, BASE + path, headers=HEADERS, **kwargs
                    )
                    response.raise_for_status()
                    return response

                def register(ctx, run, trace):
                    control(
                        "POST",
                        "/api/checkpoints/runs",
                        json={
                            "context_id": ctx,
                            "run_id": run,
                            "traceparent": f"00-{trace}-{SPAN}-01",
                        },
                    )

                def send(ctx, port=PORT):
                    started = time.monotonic()
                    response = client.post(
                        f"http://127.0.0.1:{port}/", json=_message_payload(ctx)
                    )
                    response.raise_for_status()
                    body = response.json()
                    assert "error" not in body, body
                    assert body["result"]["status"]["state"] == "completed", body
                    return time.monotonic() - started

                register("original", "run-original", TRACE1)
                result["first_a2a_seconds"] = send("original")
                assert Receiver.entered.wait(1)
                result["other_a2a_while_callback_stalled_seconds"] = send(
                    "other", PORT + 1
                )
                assert result["first_a2a_seconds"] < 2, result
                assert result["other_a2a_while_callback_stalled_seconds"] < 2, result
                assert not Receiver.points, "callback was not actually stalled"
                listing = control("GET", "/api/checkpoints?run_id=run-original").json()[
                    "checkpoints"
                ]
                point = next(
                    p for p in listing if p["label"] == "T1_after_literature_review"
                )
                point_id = point["checkpoint_id"]
                prefix = f"/api/checkpoints/{point_id}"
                operations = [
                    ("GET", "/api/checkpoints", None),
                    ("GET", prefix, None),
                    ("GET", prefix + "/bundle", None),
                    ("POST", prefix + "/restore", {}),
                    (
                        "POST",
                        "/api/checkpoints/runs",
                        {"context_id": "original", "run_id": "foreign"},
                    ),
                ]
                denials = []
                for method, path, body in operations:
                    for headers in ({}, {"Authorization": "Bearer wrong"}):
                        response = client.request(
                            method, BASE + path, json=body, headers=headers
                        )
                        denials.append(response.status_code)
                        assert response.status_code == 401
                result["unauthorized_requests_rejected"] = len(denials)
                original_bundle = control("GET", prefix + "/bundle").content
                assert original_bundle.startswith(b"PK")
                manifest = control("GET", prefix).json()
                assert manifest["run_id"] == "run-original"
                restored = control("POST", prefix + "/restore", json={}).json()[
                    "context_id"
                ]
                assert restored != "original"
                register(restored, "run-restored", TRACE2)
                result["restored_a2a_seconds"] = send(restored)
                assert control("GET", prefix + "/bundle").content == original_bundle
                result["original_bundle_unchanged"] = True
                wait_for(
                    lambda: any(p["run_id"] == "run-restored" for p in Receiver.points)
                )
                assert any(p["run_id"] == "run-original" for p in Receiver.points)
                result["callbacks_after_timeout"] = len(Receiver.points)
                spans = _decode_spans(Receiver.spans)
                for run, trace in (("run-original", TRACE1), ("run-restored", TRACE2)):
                    native = [
                        s
                        for s in spans
                        if s["name"] == "invocation"
                        and s["attributes"].get("run_id") == run
                    ]
                    assert native and {s["trace_id"] for s in native} == {trace}, native
                result["original_and_restored_traces_correlated"] = True
                logs = log_path.read_text()
                assert "ReadTimeout" in logs and "delivery failed" in logs
                assert "UNEXPECTED_NETWORK" not in logs, "\n".join(
                    line for line in logs.splitlines() if "UNEXPECTED_NETWORK" in line
                )
                result["callback_timeout_logged"] = True
                result["llm_calls"] = 0
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if process.returncode not in (0, -15):
                print(log_path.read_text()[-6000:], file=sys.stderr)
            receiver.shutdown()
            receiver.server_close()
    print("CONTROL_LIVE_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    if "--serve" in sys.argv:
        asyncio.run(serve())
    else:
        main()
