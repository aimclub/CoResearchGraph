"""A scripted ADK run must leave all pilot tools available and observe delegation."""

import asyncio
import json

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

SCIENCE_RESULT = json.dumps({
    "status": "computed",
    "scientific_mcp_calls": [
        {"tool": name, "args": {}, "result": {"answer": {"n_reconstructed": 225}}}
        for name in (
            "dataset_overview_heracleum_tox", "chemical_space_clustering",
            "predict_ld50", "predict_molecule_profile",
        )
    ],
})

FINAL_REPORT = (
    "## Научный отчёт по Heracleum\n\n"
    "| Этап | Наблюдение |\n|---|---|\n"
    "| Обзор | MCP подтвердил 225 реконструированных соединений |\n"
    "| Кластеризация | Результат получен вычислительным инструментом |\n"
    "| LD50 | Значения являются прогнозом модели |\n"
    "| Профиль | Получен отдельным вызовом MCP |\n\n"
    "Полные строки молекул и SMILES недоступны из агрегированного обзора. "
    "Экспериментальная проверка LD50 в доступных результатах не подтверждена. "
    "Тепловая карта и дендрограмма не приложены: инструменты не вернули "
    "файлы изображений. Таблица отражает только наблюдённые вычисления."
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
                role="model", parts=[types.Part(text=FINAL_REPORT)]
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
        return {"result": SCIENCE_RESULT}

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
    assert events[-1].content.parts[0].text == FINAL_REPORT


def test_pilot_a2a_artifact_contains_full_report_not_receipt():
    import httpx
    from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCapabilities, AgentCard
    from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor

    async def retrieve_tools(query: str) -> dict:
        return {"status": "ok"}

    async def HypothesesAgent(request: str) -> dict:
        return {"status": "ok"}

    async def TaskExecutorAgent(request: str) -> dict:
        return {"result": SCIENCE_RESULT}

    async def ResearchAgent(request: str) -> dict:
        return {"status": "ok"}

    async def run():
        pilot = build_system(
            load_config(resolve_config_path("synapse_pilot")), remote_subagents=True
        )
        agent = LlmAgent(
            name="PilotA2AReportProbe", model=PilotModel(),
            instruction="Complete the pilot report.",
            tools=[retrieve_tools, HypothesesAgent, TaskExecutorAgent, ResearchAgent],
            after_model_callback=pilot.root.after_model_callback,
        )
        runner = Runner(
            agent=agent, app_name="pilot_a2a_report",
            session_service=InMemorySessionService(),
        )
        card = AgentCard(
            name="PilotA2AReportProbe", description="Report test",
            url="http://pilot.test/", version="1",
            capabilities=AgentCapabilities(streaming=True),
            defaultInputModes=["text/plain"], defaultOutputModes=["text/plain"],
            skills=[],
        )
        handler = DefaultRequestHandler(
            agent_executor=A2aAgentExecutor(runner=runner),
            task_store=InMemoryTaskStore(),
        )
        app = A2AFastAPIApplication(agent_card=card, http_handler=handler).build()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://pilot.test"
        ) as client:
            response = await client.post("/", json={
                "jsonrpc": "2.0", "id": "pilot-report",
                "method": "message/send",
                "params": {"message": {
                    "kind": "message", "role": "user", "messageId": "pilot-message",
                    "parts": [{"kind": "text", "text": "Составь научный отчёт"}],
                }},
            })
        response.raise_for_status()
        return response.json()

    payload = asyncio.run(run())
    assert "error" not in payload
    result = payload["result"]
    artifact_texts = [
        part.get("text", "")
        for artifact in result.get("artifacts", [])
        for part in artifact.get("parts", [])
        if part.get("kind") == "text"
    ]
    assert any(FINAL_REPORT in text for text in artifact_texts)
    assert all("Observed scientific MCP results:" not in text for text in artifact_texts)


def test_pilot_final_response_runs_missing_profile_through_adk_tool_call():
    partial = json.dumps({
        "status": "computed",
        "scientific_mcp_calls": [
            {"tool": name, "args": {}, "result": {"answer": {"observed": True}}}
            for name in (
                "dataset_overview_heracleum_tox", "chemical_space_clustering",
                "predict_ld50",
            )
        ],
    })
    profile = json.dumps({
        "status": "computed",
        "scientific_mcp_calls": [{
            "tool": "predict_molecule_profile",
            "args": {"name_or_smiles": "xanthotoxin"},
            "result": {"answer": {"ld50_mgkg": 638.0}},
        }],
    })
    requests = []

    async def retrieve_tools(query: str) -> dict:
        return {"status": "ok"}

    async def HypothesesAgent(request: str) -> dict:
        return {"status": "ok"}

    async def ResearchAgent(request: str) -> dict:
        return {"status": "ok"}

    async def TaskExecutorAgent(request: str) -> dict:
        requests.append(request)
        return {"result": profile if "Target tool: predict_molecule_profile" in request else partial}

    pilot = build_system(load_config(resolve_config_path("synapse_pilot")), remote_subagents=True)
    model = PilotModel()
    agent = LlmAgent(
        name="PilotRecoveryProbe", model=model, instruction="Complete the pilot.",
        tools=[retrieve_tools, HypothesesAgent, ResearchAgent, TaskExecutorAgent],
        after_model_callback=pilot.root.after_model_callback,
    )

    async def run():
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="pilot_recovery", user_id="user", session_id="session",
            state={"accumulated_tools": [
                {"tool": name, "server_id": "heracleum-server"}
                for name in (
                    "dataset_overview_heracleum_tox", "chemical_space_clustering",
                    "predict_ld50", "predict_molecule_profile",
                )
            ]},
        )
        runner = Runner(agent=agent, app_name="pilot_recovery", session_service=sessions)
        return [event async for event in runner.run_async(
            user_id="user", session_id="session",
            new_message=types.Content(role="user", parts=[types.Part(text="Run pilot")]),
        )]

    events = asyncio.run(run())
    calls = [call.name for event in events for call in event.get_function_calls()]
    assert calls == [*CALLS, "TaskExecutorAgent"]
    assert len(requests) == 2
    assert "name_or_smiles=xanthotoxin" in requests[1]
    assert events[-1].content.parts[0].text == FINAL_REPORT


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
            return {"result": SCIENCE_RESULT}

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
        text = events[-1].content.parts[0].text
        assert text == FINAL_REPORT
