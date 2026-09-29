"""A real HTTP delegation shares the research graph without trusting public A2A input."""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import threading
import time
from types import SimpleNamespace
from typing import ClassVar
from uuid import uuid4

import httpx
import pytest
import uvicorn
from a2a.types import AgentCapabilities, AgentCard
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from CoScientist.a2a.server import make_a2a_app
from CoScientist.graph.research.agent_tools import ResearchGraphToolset
from CoScientist.graph.research.store import get_research_graph
from CoScientist.graph.session_scope import (
    GRAPH_SCOPE_SESSION_KEY,
    GRAPH_SCOPE_USER_KEY,
    session_key,
)


def _card(name: str, port: int) -> AgentCard:
    return AgentCard(
        name=name,
        description=name,
        url=f"http://127.0.0.1:{port}/",
        version="1",
        capabilities=AgentCapabilities(streaming=True),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[],
    )


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Hypotheses(BaseAgent):
    calls: ClassVar[list[tuple[str, str]]] = []

    async def _run_async_impl(self, ctx):
        self.calls.append(session_key(ctx))
        result = get_research_graph(ctx).commit(
            source="HypothesesAgent",
            nodes=[{"type": "Hypothesis", "attrs": {"formulation": "X binds Y"}}],
        )
        assert result.ok, result.errors
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(role="model", parts=[types.Part(text="H1 created")]),
        )


class _Executor(BaseAgent):
    calls: ClassVar[list[tuple[str, str]]] = []

    async def _run_async_impl(self, ctx):
        self.calls.append(session_key(ctx))
        assert "error" not in get_research_graph(ctx).get_context_slice("H1")
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(role="model", parts=[types.Part(text="H1 executed")]),
        )


async def _delegate(name: str, card: AgentCard, sessions, root_id: str):
    from CoScientist.a2a.graph_scope import make_graph_scope_config

    remote = RemoteA2aAgent(
        name=name,
        agent_card=card,
        config=make_graph_scope_config(name),
    )
    runner = Runner(agent=remote, app_name="root-graph-test", session_service=sessions)
    return [
        event
        async for event in runner.run_async(
            user_id="root-user",
            session_id=root_id,
            new_message=types.Content(role="user", parts=[types.Part(text=name)]),
        )
    ]


@pytest.mark.asyncio
async def test_remote_hypothesis_is_visible_to_root_and_task_executor(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RESEARCH_GRAPH_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("COSCIENTIST_A2A_GRAPH_SCOPE_SECRET", "test-shared-secret")
    monkeypatch.setenv("A2A_DISABLE_OPIK", "1")
    from CoScientist.config import settings

    monkeypatch.setattr(settings.research_graph, "dir", str(tmp_path / "graphs"))
    monkeypatch.setattr(settings.checkpoints, "enabled", False)
    monkeypatch.setattr(settings.synapse, "enabled", False)
    ports = [_port(), _port()]
    cards = [_card("HypothesesAgent", ports[0]), _card("TaskExecutorAgent", ports[1])]
    servers = [
        uvicorn.Server(
            uvicorn.Config(
                make_a2a_app(agent, card, card.name),
                host="127.0.0.1",
                port=port,
                log_level="error",
            )
        )
        for agent, card, port in zip(
            [_Hypotheses(name="HypothesesAgent"), _Executor(name="TaskExecutorAgent")],
            cards,
            ports,
        )
    ]
    threads = [threading.Thread(target=server.run, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    try:
        deadline = time.monotonic() + 5
        while (
            not all(server.started for server in servers)
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.01)
        assert all(server.started for server in servers)

        sessions = InMemorySessionService()
        root_id = f"root-{uuid4().hex}"
        root = await sessions.create_session(
            app_name="root-graph-test",
            user_id="root-user",
            session_id=root_id,
        )
        root_context = SimpleNamespace(session=root, state=root.state)
        graph = get_research_graph(root_context)
        assert graph.init_research(
            source="OrchestratorAgent", question="Does X bind Y?"
        )["ok"]
        assert graph.get_context_slice("Q1")["nodes"][0]["id"] == "Q1"

        first = await _delegate("HypothesesAgent", cards[0], sessions, root_id)
        assert any(
            event.content and "H1 created" in str(event.content) for event in first
        )
        assert _Hypotheses.calls == [("root-user", root_id)], _Hypotheses.calls
        assert "error" not in graph.get_context_slice("H1")
        focus = ResearchGraphToolset(surface="orchestrator").research_set_focus(
            root_context, node_id="H1"
        )
        assert focus["ok"], focus

        second = await _delegate("TaskExecutorAgent", cards[1], sessions, root_id)
        assert any(
            event.content and "H1 executed" in str(event.content) for event in second
        )
        assert _Executor.calls == [session_key(root_context)]

        remote_context = next(
            event.custom_metadata["a2a:context_id"]
            for event in first
            if event.custom_metadata and event.custom_metadata.get("a2a:context_id")
        )
        forged_scope = base64.urlsafe_b64encode(
            json.dumps(["root-user", root_id]).encode()
        ).decode()
        # The public endpoint must reject a forged scope and an unsigned
        # continuation of a session already bound to the parent's graph.
        async with httpx.AsyncClient() as client:
            response = await client.post(
                cards[0].url,
                json={
                    "jsonrpc": "2.0",
                    "id": "external",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "kind": "message",
                            "role": "user",
                            "messageId": uuid4().hex,
                            "parts": [{"kind": "text", "text": "attempt hijack"}],
                        }
                    },
                },
                headers={
                    "x-coscientist-graph-scope": forged_scope,
                    "x-coscientist-graph-scope-proof": "0" * 64,
                },
            )
            assert response.status_code == 200
            assert "H1 created" not in response.text
            for headers in (
                {},
                {
                    "X-A2A-Extensions": "https://google.github.io/adk-docs/a2a/a2a-extension/"
                },
            ):
                response = await client.post(
                    cards[0].url,
                    json={
                        "jsonrpc": "2.0",
                        "id": "unsigned",
                        "method": "message/send",
                        "params": {
                            "message": {
                                "kind": "message",
                                "role": "user",
                                "messageId": uuid4().hex,
                                "contextId": remote_context,
                                "parts": [{"kind": "text", "text": "continue"}],
                            }
                        },
                    },
                    headers=headers,
                )
                assert response.status_code == 200
                assert "H1 created" not in response.text
            assert _Hypotheses.calls == [("root-user", root_id)]
    finally:
        for server in servers:
            server.should_exit = True
        for thread in threads:
            thread.join(timeout=3)


@pytest.mark.asyncio
async def test_public_a2a_session_can_continue_its_own_graph():
    from CoScientist.a2a.graph_scope import GraphScopeA2aAgentExecutor

    sessions = InMemorySessionService()
    user_id, context_id = "A2A_USER_public-context", "public-context"
    await sessions.create_session(
        app_name="public-agent",
        user_id=user_id,
        session_id=context_id,
        state={GRAPH_SCOPE_USER_KEY: user_id, GRAPH_SCOPE_SESSION_KEY: context_id},
    )
    runner = Runner(
        agent=_Hypotheses(name="HypothesesAgent"),
        app_name="public-agent",
        session_service=sessions,
    )
    executor = GraphScopeA2aAgentExecutor(runner=runner)
    request = SimpleNamespace(
        call_context=None, message=SimpleNamespace(message_id="public-message")
    )
    run_request = SimpleNamespace(
        user_id=user_id, session_id=context_id, state_delta=None
    )
    session = await executor._prepare_session(request, run_request, runner)
    assert session.id == context_id
    assert run_request.state_delta is None
