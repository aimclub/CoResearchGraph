from __future__ import annotations

import asyncio
import os
import signal

import uvicorn
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.utils.context_utils import Aclosing
from google.genai import types

from CoScientist.a2a.server import make_a2a_app
from CoScientist.a2a.synapse_tracing import make_remote_agent_config


CHILD_MARKER = "COSCIENTIST_CHILD_OK"
ORCHESTRATOR_MARKER = "COSCIENTIST_ORCHESTRATOR_OK"


def _content_text(content: types.Content | None) -> str:
    if content is None:
        return ""
    return " ".join(part.text for part in content.parts or [] if part.text)


def _card(name: str, url: str, skill_id: str) -> AgentCard:
    return AgentCard(
        name=name,
        description=f"Deterministic {name} agent for the Synapse integration demo",
        url=url,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[
            AgentSkill(
                id=skill_id,
                name=name,
                description=f"Return a deterministic result from {name}",
                tags=["demo", "integration"],
            )
        ],
    )


class ScriptedHypothesesAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        prompt = _content_text(ctx.user_content)
        child_text = f"{CHILD_MARKER}: hypothesis reviewed for {prompt}"
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(role="model", parts=[types.Part(text=child_text)]),
            actions=EventActions(state_delta={"search_results": child_text}),
        )


class ScriptedOrchestratorAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        child = list(self.sub_agents)[0]
        child_texts: list[str] = []
        async with Aclosing(child.run_async(ctx)) as events:
            async for event in events:
                if event.error_message:
                    raise RuntimeError("DemoHypotheses A2A call failed")
                text = _content_text(event.content)
                if text:
                    child_texts.append(text)
                yield event
        if not child_texts:
            raise RuntimeError("DemoHypotheses returned no result")
        aggregate = " | ".join(child_texts)
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"{ORCHESTRATOR_MARKER}: {aggregate}")],
            ),
        )


def _apps() -> list[tuple[object, int]]:
    host = os.getenv("A2A_HOST", "localhost")
    orchestrator_port = int(os.getenv("DEMO_ORCHESTRATOR_PORT", "8100"))
    hypotheses_port = int(os.getenv("DEMO_HYPOTHESES_PORT", "8102"))
    child_card = _card(
        "DemoHypotheses", f"http://{host}:{hypotheses_port}/", "demo-hypotheses"
    )
    remote_child = RemoteA2aAgent(
        name="DemoHypotheses",
        agent_card=child_card,
        config=make_remote_agent_config(),
    )
    orchestrator = ScriptedOrchestratorAgent(
        name="DemoOrchestrator", sub_agents=[remote_child]
    )
    orchestrator_card = _card(
        "DemoOrchestrator",
        f"http://{host}:{orchestrator_port}/",
        "demo-orchestrator",
    )
    return [
        (make_a2a_app(orchestrator, orchestrator_card, "demo-orchestrator"), orchestrator_port),
        (make_a2a_app(ScriptedHypothesesAgent(name="DemoHypotheses"), child_card, "demo-hypotheses"), hypotheses_port),
    ]


async def main() -> None:
    servers = [
        uvicorn.Server(
            uvicorn.Config(
                app,
                host="0.0.0.0",
                port=port,
                log_level="info",
                timeout_graceful_shutdown=8,
            )
        )
        for app, port in _apps()
    ]
    loop = asyncio.get_running_loop()

    def request_shutdown() -> None:
        for server in servers:
            server.should_exit = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_shutdown)
        except NotImplementedError:
            signal.signal(signum, lambda *_: request_shutdown())

    await asyncio.gather(*(server.serve() for server in servers))


if __name__ == "__main__":
    asyncio.run(main())
