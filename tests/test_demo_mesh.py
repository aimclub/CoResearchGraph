import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from google.adk.agents import BaseAgent
from google.adk.events import Event

from CoScientist.a2a.demo_mesh import ScriptedOrchestratorAgent


REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_tcp_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return [item.getsockname()[1] for item in sockets]
    finally:
        for item in sockets:
            item.close()


def _wait_json(url: str, process: subprocess.Popen) -> dict:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"demo mesh exited: {process.stdout.read()}")
        try:
            response = httpx.get(url, timeout=1)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            time.sleep(0.1)
    output = _terminate(process)
    raise AssertionError(f"demo mesh did not serve {url} within 45 seconds: {output}")


def _terminate(process: subprocess.Popen) -> str:
    if process.poll() is not None:
        return process.communicate()[0]
    process.terminate()
    try:
        output = process.communicate(timeout=5)[0]
    except subprocess.TimeoutExpired:
        process.kill()
        output = process.communicate(timeout=5)[0]
    return output


def _message(text: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": "demo-request",
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "messageId": "demo-message",
                "contextId": "demo-context",
                "parts": [{"kind": "text", "text": text}],
            }
        },
    }


def test_demo_mesh_delegates_over_http_a2a():
    orchestrator_port, hypotheses_port = _free_tcp_ports(2)
    env = {
        **os.environ,
        "A2A_HOST": "127.0.0.1",
        "DEMO_ORCHESTRATOR_PORT": str(orchestrator_port),
        "DEMO_HYPOTHESES_PORT": str(hypotheses_port),
        "CHECKPOINTS__ENABLED": "0",
        "SYNAPSE__ENABLED": "0",
        "A2A_DISABLE_OPIK": "1",
        "LOG_AGENT_EVENTS": "0",
        "LLM__MAIN_MODEL": "openai/gpt-4o-mini",
        "OPENAI_API_KEY": "",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "CoScientist.a2a.demo_mesh"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_json(f"http://127.0.0.1:{orchestrator_port}/.well-known/agent-card.json", process)
        child_card = _wait_json(
            f"http://127.0.0.1:{hypotheses_port}/.well-known/agent-card.json", process
        )
        response = httpx.post(
            f"http://127.0.0.1:{orchestrator_port}/",
            json=_message("test integration"),
            timeout=30,
        )
        response.raise_for_status()
        payload = json.dumps(response.json())
        assert child_card["url"] == f"http://127.0.0.1:{hypotheses_port}/"
        assert "COSCIENTIST_CHILD_OK" in payload
        assert "COSCIENTIST_ORCHESTRATOR_OK" in payload
        assert "test integration" in payload
    finally:
        _terminate(process)


def test_orchestrator_rejects_remote_child_error():
    class FailedChild(BaseAgent):
        async def run_async(self, ctx):
            yield Event(
                author="DemoHypotheses",
                invocation_id=ctx.invocation_id,
                error_message="remote service unavailable",
            )

    agent = ScriptedOrchestratorAgent(
        name="DemoOrchestrator", sub_agents=[FailedChild(name="DemoHypotheses")]
    )
    async def run():
        async for _ in agent._run_async_impl(SimpleNamespace(invocation_id="demo")):
            pass

    with pytest.raises(RuntimeError, match="DemoHypotheses A2A call failed"):
        asyncio.run(run())
