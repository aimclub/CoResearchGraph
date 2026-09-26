"""A scripted ADK run must leave all pilot tools available and observe delegation."""

import asyncio

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import PrivateAttr

from CoScientist.assembly import build_system
from CoScientist.assembly.schema import load_config, resolve_config_path


CALLS = (
    "retrieve_tools",
    "HypothesesAgent",
    "TaskExecutorAgent",
    "ResearchAgent",
)


class PilotModel(BaseLlm):
    _calls: int = PrivateAttr(default=0)
    _offered: list[tuple[str, ...]] = PrivateAttr(default_factory=list)
    _tool_configs: list = PrivateAttr(default_factory=list)

    def __init__(self):
        super().__init__(model="scripted-pilot")

    @property
    def offered(self) -> list[tuple[str, ...]]:
        return self._offered

    @property
    def tool_configs(self) -> list:
        return self._tool_configs

    async def generate_content_async(self, llm_request, stream=False):
        offered = tuple(
            declaration.name
            for tool in llm_request.config.tools or []
            for declaration in tool.function_declarations or []
        )
        self._offered.append(offered)
        self._tool_configs.append(llm_request.config.tool_config)
        if self._calls < len(CALLS):
            name = CALLS[self._calls]
            args = {"query": "surfactants"} if name == "retrieve_tools" else {
                "request": "Study surfactants"
            }
            content = types.Content(
                role="model",
                parts=[types.Part.from_function_call(name=name, args=args)],
            )
        else:
            content = types.Content(
                role="model", parts=[types.Part(text="Pilot delegation completed")]
            )
        self._calls += 1
        yield LlmResponse(content=content)


def test_synapse_pilot_observes_real_calls_and_responses_without_forcing_order():
    seen = []

    async def retrieve_tools(query: str) -> dict:
        seen.append("retrieve_tools")
        return {"status": "ok"}

    async def HypothesesAgent(request: str) -> dict:
        seen.append("HypothesesAgent")
        return {"status": "ok"}

    async def ResearchAgent(request: str) -> dict:
        seen.append("ResearchAgent")
        return {"status": "ok"}

    async def TaskExecutorAgent(request: str) -> dict:
        seen.append("TaskExecutorAgent")
        return {"status": "ok"}

    pilot = load_config(resolve_config_path("synapse_pilot"))
    orchestrator = build_system(pilot, remote_subagents=True).root
    model = PilotModel()
    agent = LlmAgent(
        name="TestPilotOrchestrator",
        model=model,
        instruction="Complete the pilot delegation sequence.",
        tools=[retrieve_tools, HypothesesAgent, ResearchAgent, TaskExecutorAgent],
        before_model_callback=orchestrator.before_model_callback,
        after_model_callback=orchestrator.after_model_callback,
        before_tool_callback=orchestrator.before_tool_callback,
        after_tool_callback=orchestrator.after_tool_callback,
    )

    async def run():
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="test_pilot", user_id="user", session_id="session"
        )
        runner = Runner(agent=agent, app_name="test_pilot", session_service=sessions)
        return [
            event
            async for event in runner.run_async(
                user_id="user",
                session_id="session",
                new_message=types.Content(
                    role="user", parts=[types.Part(text="Run the pilot")]
                ),
            )
        ]

    events = asyncio.run(run())
    observed_calls = [
        call.name for event in events for call in event.get_function_calls()
    ]
    observed_responses = [
        response.name
        for event in events
        for response in event.get_function_responses()
    ]
    assert observed_calls == list(CALLS)
    assert observed_responses == list(CALLS)
    assert seen == list(CALLS)
    assert all(set(CALLS).issubset(offered) for offered in model.offered)
    assert all(config is None for config in model.tool_configs)


class UnknownFirstModel(PilotModel):
    _unknown_sent: bool = PrivateAttr(default=False)

    async def generate_content_async(self, llm_request, stream=False):
        if not self._unknown_sent:
            self._unknown_sent = True
            self._calls = 1  # retrieve_tools is included in the same response.
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_function_call(
                            name="invented_subagent", args={}
                        ),
                        types.Part.from_function_call(
                            name="retrieve_tools", args={"query": "surfactants"}
                        )
                    ],
                )
            )
            return
        async for response in super().generate_content_async(llm_request, stream):
            yield response


def test_unknown_orchestrator_tool_returns_error_then_model_retries():
    async def run_profile(profile: str):
        seen = []

        async def retrieve_tools(query: str) -> dict:
            seen.append("retrieve_tools")
            return {"status": "ok"}

        async def HypothesesAgent(request: str) -> dict:
            seen.append("HypothesesAgent")
            return {"status": "ok"}

        async def ResearchAgent(request: str) -> dict:
            seen.append("ResearchAgent")
            return {"status": "ok"}

        async def TaskExecutorAgent(request: str) -> dict:
            seen.append("TaskExecutorAgent")
            return {"status": "ok"}

        config = load_config(resolve_config_path(profile))
        orchestrator = build_system(config, remote_subagents=True).root
        model = UnknownFirstModel()
        agent = LlmAgent(
            name="UnknownToolRetryProbe",
            model=model,
            instruction="Complete the pilot delegation sequence.",
            tools=[retrieve_tools, HypothesesAgent, ResearchAgent, TaskExecutorAgent],
            before_model_callback=orchestrator.before_model_callback,
            after_model_callback=orchestrator.after_model_callback,
            before_tool_callback=orchestrator.before_tool_callback,
            after_tool_callback=orchestrator.after_tool_callback,
        )
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="unknown_tool_retry", user_id="user", session_id="session"
        )
        runner = Runner(
            agent=agent, app_name="unknown_tool_retry", session_service=sessions
        )
        events = [
            event
            async for event in runner.run_async(
                user_id="user",
                session_id="session",
                new_message=types.Content(
                    role="user", parts=[types.Part(text="Run the pilot")]
                ),
            )
        ]
        return events, seen

    for profile in ("synapse_demo", "synapse_pilot"):
        events, seen = asyncio.run(run_profile(profile))
        calls = [call.name for event in events for call in event.get_function_calls()]
        responses = [
            response for event in events for response in event.get_function_responses()
        ]
        unknown = [r for r in responses if r.name == "invented_subagent"]
        assert calls == ["invented_subagent", *CALLS]
        assert len(unknown) == 1 and "error" in unknown[0].response
        assert all(name in str(unknown[0].response) for name in CALLS)
        assert [r.name for r in responses] == ["invented_subagent", *CALLS]
        assert seen == list(CALLS)
        assert events[-1].content.parts[0].text == "Pilot delegation completed"
