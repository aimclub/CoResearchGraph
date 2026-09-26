"""Unknown model tool names must not end an orchestrator delegation turn."""

import asyncio

import pytest
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import PrivateAttr

from CoScientist.assembly import build_system


class ScriptedModel(BaseLlm):
    _first_name: str = PrivateAttr()
    _call_count: int = PrivateAttr(default=0)

    def __init__(self, first_name: str):
        super().__init__(model="scripted")
        self._first_name = first_name

    @property
    def call_count(self) -> int:
        return self._call_count

    async def generate_content_async(self, llm_request, stream=False):
        self._call_count += 1
        if self._call_count <= 2:
            name = self._first_name if self._call_count == 1 else "ResearchAgent"
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_function_call(
                            name=name,
                            args={"request": "Find a cited paper about surfactants"},
                        )
                    ],
                )
            )
        else:
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="ResearchAgent completed")]
                )
            )


@pytest.mark.parametrize("wrong_name", ["research_agent", "tavily_search"])
def test_orchestrator_retries_unknown_tool_name_and_delegates(wrong_name):
    """ADK's tool error lets the model retry with the advertised agent name."""
    called = []

    async def ResearchAgent(request: str) -> dict:
        called.append(request)
        return {"status": "ok"}

    model = ScriptedModel(wrong_name)
    orchestrator = build_system(remote_subagents=True).agent("OrchestratorAgent")
    agent = LlmAgent(
        name="TestOrchestrator",
        model=model,
        instruction="Use ResearchAgent for literature searches.",
        tools=[ResearchAgent],
        after_model_callback=orchestrator.after_model_callback,
    )

    async def run():
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="test_unknown_tool", user_id="user", session_id="session"
        )
        runner = Runner(
            agent=agent, app_name="test_unknown_tool", session_service=sessions
        )
        return [
            event
            async for event in runner.run_async(
                user_id="user",
                session_id="session",
                new_message=types.Content(
                    role="user", parts=[types.Part(text="Search the literature")]
                ),
            )
        ]

    events = asyncio.run(run())
    assert model.call_count == 3
    assert called == ["Find a cited paper about surfactants"]
    assert any(
        response.name == wrong_name
        for event in events
        for response in event.get_function_responses()
    )
    assert any(
        fc.name == "ResearchAgent"
        for event in events
        for fc in event.get_function_calls()
    )
