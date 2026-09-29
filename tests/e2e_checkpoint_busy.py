"""Real HTTP/ADK regression for long-running invocations and process-wide restore.

Run: LLM__MAIN_MODEL=test-model uv run --frozen python tests/e2e_checkpoint_busy.py
Scripted agents make zero LLM calls; only the old busy gate's clock is advanced.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import httpx
from _synapse_native_trace_probe import _card, _message_payload

PORT = int(os.getenv("E2E_BUSY_PORT", "8158"))
TOKEN = "test-only-platform-admin-credential"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def local_network_only(event, args):
    if event not in {"socket.connect", "socket.getaddrinfo"}:
        return
    address = args[1] if event == "socket.connect" else args[0]
    host = address[0] if isinstance(address, tuple) else address
    if host is not None and host not in {"127.0.0.1", "localhost", "::1"}:
        raise AssertionError(f"External connections forbidden in this probe: {host!r}")


async def exercise():
    import uvicorn
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.events.event import Event
    from google.adk.events.event_actions import EventActions
    from google.genai import types

    from CoScientist.a2a.server import make_a2a_app
    from CoScientist.checkpoints import plugin

    entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    # Leave the event loop's clock alone. Reproduce a two-hour invocation age
    # for the old implementation while a real agent is still awaiting work.
    clock = SimpleNamespace(monotonic=lambda: 100.0)
    plugin.time = clock

    class ScriptedAgent(BaseAgent):
        async def _run_async_impl(self, ctx):
            try:
                if self.name == "LongRunning":
                    entered.set()
                    await release.wait()
                yield Event(
                    author=self.name,
                    invocation_id=ctx.invocation_id,
                    content=types.Content(
                        role="model", parts=[types.Part(text="scripted result")]
                    ),
                    actions=EventActions(state_delta={"search_results": "fixture"}),
                )
            finally:
                if self.name == "LongRunning":
                    closed.set()

    servers = []
    urls = [f"http://127.0.0.1:{PORT+i}" for i in range(2)]
    for index, name in enumerate(("CheckpointSource", "LongRunning")):
        app = make_a2a_app(
            ScriptedAgent(name=name), _card(name, urls[index] + "/"), f"app{index}"
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
    serving = [asyncio.create_task(server.serve()) for server in servers]
    running = None
    result = {}
    try:
        async with asyncio.timeout(90):
            while not all(server.started for server in servers):
                for task in serving:
                    if task.done():
                        await task
                        raise AssertionError("A2A server stopped before readiness")
                await asyncio.sleep(0.05)
        async with httpx.AsyncClient(timeout=15) as client:

            async def control(method, path, **kwargs):
                return await client.request(
                    method,
                    urls[0] + path,
                    headers=HEADERS,
                    **kwargs,
                )

            async def send(ctx, index=0):
                response = await client.post(
                    urls[index] + "/", json=_message_payload(ctx)
                )
                response.raise_for_status()
                body = response.json()
                assert "error" not in body, body
                assert body["result"]["status"]["state"] == "completed", body

            response = await control(
                "POST",
                "/api/checkpoints/runs",
                json={
                    "context_id": "original",
                    "run_id": "run-original",
                },
            )
            response.raise_for_status()
            await send("original")
            response = await control("GET", "/api/checkpoints?run_id=run-original")
            response.raise_for_status()
            point = next(
                p
                for p in response.json()["checkpoints"]
                if p["label"] == "T1_after_literature_review"
            )
            prefix = f"/api/checkpoints/{point['checkpoint_id']}"
            response = await control("GET", prefix + "/bundle")
            response.raise_for_status()
            original_bundle = response.content
            assert original_bundle.startswith(b"PK")

            running = asyncio.create_task(send("long-running", index=1))
            await asyncio.wait_for(entered.wait(), 5)
            clock.monotonic = lambda: 7301.0
            assert not running.done() and not closed.is_set()
            response = await control("POST", prefix + "/restore", json={})
            result["restore_while_other_port_agent_live"] = response.status_code
            assert response.status_code == 409, response.text
            assert not running.done() and not closed.is_set()
            result["simulated_invocation_age_seconds"] = 7201

            release.set()
            await running
            assert closed.is_set()
            response = await control("POST", prefix + "/restore", json={})
            result["restore_after_actual_completion"] = response.status_code
            assert response.status_code == 200, response.text
            restored = response.json()["context_id"]
            assert restored != "original"
            await send(restored)
            result["restored_context_completed_a2a"] = True
            response = await control("GET", prefix + "/bundle")
            response.raise_for_status()
            assert response.content == original_bundle
            result["original_bundle_unchanged"] = True
            result["llm_calls"] = 0
    finally:
        release.set()
        if running is not None:
            await asyncio.gather(running, return_exceptions=True)
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*serving)
    print("BUSY_LIVE_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="busy-live-") as temp:
        os.environ.update(
            {
                "CHECKPOINTS__ENABLED": "1",
                "CHECKPOINTS__DIR": f"{temp}/snapshots",
                "CHECKPOINTS__API_TOKEN": TOKEN,
                "SYNAPSE__ENABLED": "0",
                "A2A_DISABLE_OPIK": "1",
                "LOG_AGENT_EVENTS": "0",
                "LLM__MAIN_MODEL": "test-model",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "RESEARCH_GRAPH__DIR": f"{temp}/graphs",
            }
        )
        sys.addaudithook(local_network_only)
        asyncio.run(exercise())
