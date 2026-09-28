"""ExperimentAgent must not turn narrated MCP calls into a successful result."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool import McpTool
from google.genai import types
from mcp.types import Tool
from pydantic import PrivateAttr

from CoScientist.agents.callbacks.experiment_execution import (
    attest_executor_science,
    capture_scientific_pipeline_receipt,
    record_scientific_mcp_result,
    require_first_scientific_tool_call,
    require_scientific_execution,
    reset_executor_science_receipt,
    reset_scientific_execution,
)
from CoScientist.assembly import build_system
from CoScientist.assembly.schema import load_config, resolve_config_path


@pytest.fixture(scope="module")
def agent():
    return build_system(load_config(resolve_config_path("synapse_pilot"))).agent(
        "ExperimentAgent"
    )


def _context(invocation_id="current-run"):
    return SimpleNamespace(
        state={
            "filtered_tools": [{"server_id": "heracleum-tox"}],
            "executor_tool_match": {"matched": True},
        },
        _invocation_context=SimpleNamespace(
            invocation_id=invocation_id, session=SimpleNamespace(id="session", events=[])
        ),
    )


def _response(text):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _after_model(agent, context, response):
    for callback in agent.canonical_after_model_callbacks:
        replacement = callback(context, response)
        if replacement is not None:
            return replacement
    return response


def test_first_scientific_call_offers_only_explicit_target():
    context = _context()
    context.state["filtered_tools"] = [
        {"tool": "dataset_overview_heracleum_tox"},
        {"tool": "predict_molecule_profile"},
    ]
    context.state["explicit_tool_target"] = "predict_molecule_profile"
    request = SimpleNamespace(config=SimpleNamespace(
        tools=[types.Tool(function_declarations=[
            types.FunctionDeclaration(name="dataset_overview_heracleum_tox"),
            types.FunctionDeclaration(name="predict_molecule_profile"),
        ])],
        tool_config=None,
    ))
    require_first_scientific_tool_call(context, request)
    assert [declaration.name for declaration in request.config.tools[0].function_declarations] == [
        "predict_molecule_profile"
    ]


def test_first_scientific_call_rejects_missing_explicit_target():
    context = _context()
    context.state["filtered_tools"] = [{"tool": "dataset_overview_heracleum_tox"}]
    context.state["explicit_tool_target"] = "predict_molecule_profile"
    request = SimpleNamespace(config=SimpleNamespace(
        tools=[types.Tool(function_declarations=[
            types.FunctionDeclaration(name="dataset_overview_heracleum_tox"),
        ])],
        tool_config=None,
    ))
    with pytest.raises(RuntimeError, match="predict_molecule_profile"):
        require_first_scientific_tool_call(context, request)


def test_experiment_rejects_text_that_impersonates_mcp_results(agent):
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text="dataset_overview_heracleum_tox returned 12,654 rows")],
        )
    )
    with pytest.raises(RuntimeError, match="scientific MCP"):
        _after_model(agent, _context(), response)


def test_experiment_accepts_observed_successful_mcp_result(agent):
    context = _context()
    tool = Mock(spec=McpTool)
    tool.name = "dataset_overview_heracleum_tox"
    result = {
        "isError": False,
        "content": [
            {"type": "text", "text": '{"answer":{"n_reconstructed":225}}'}
        ],
    }

    for callback in agent.canonical_after_tool_callbacks:
        callback(tool, {}, context, result)

    final = _after_model(agent, context, _response("225 metabolites"))
    assert "225" in final.content.parts[0].text


def test_experiment_uses_tool_data_instead_of_invented_numbers_and_files(agent):
    context = _context()
    tool = Mock(spec=McpTool)
    tool.name = "dataset_overview_heracleum_tox"
    result = {
        "isError": False,
        "content": [
            {"type": "text", "text": '{"answer":{"n_reconstructed":225}}'}
        ],
    }
    for callback in agent.canonical_after_tool_callbacks:
        callback(tool, {}, context, result)

    final = _after_model(
        agent,
        context,
        _response("12,654 metabolites; wrote /workspace/fake.csv and /workspace/fake.png"),
    )
    text = final.content.parts[0].text
    assert "225" in text
    assert "12,654" not in text
    assert "fake.csv" not in text
    assert "fake.png" not in text


def test_experiment_does_not_reuse_a_prior_task_in_the_same_invocation(agent):
    context = _context()
    tool = Mock(spec=McpTool)
    tool.name = "dataset_overview_heracleum_tox"
    result = {"isError": False, "content": [{"type": "text", "text": "225"}]}
    for callback in agent.canonical_after_tool_callbacks:
        callback(tool, {}, context, result)

    for callback in agent.canonical_before_agent_callbacks:
        if callback.__name__ == "reset_scientific_execution":
            callback(context)

    with pytest.raises(RuntimeError, match="scientific MCP"):
        _after_model(agent, context, _response("Reused old result"))


def test_experiment_reports_only_artifacts_that_exist(agent, tmp_path):
    confirmed = tmp_path / "confirmed.csv"
    confirmed.write_text("compound,ld50\nA,225\n", encoding="utf-8")
    absent = tmp_path / "invented.png"
    context = _context()
    tool = Mock(spec=McpTool)
    tool.name = "predict_ld50"
    result = {
        "isError": False,
        "content": [{"type": "text", "text": json.dumps({
            "answer": {"count": 225, "csv": str(confirmed), "plot": str(absent)}
        })}],
    }
    for callback in agent.canonical_after_tool_callbacks:
        callback(tool, {}, context, result)

    final = _after_model(agent, context, _response("Both files exist"))
    recorded = json.loads(final.content.parts[0].text)["scientific_mcp_calls"][0]["result"]
    assert recorded["answer"]["csv"] == str(confirmed)
    assert "plot" not in recorded["answer"]


def test_experiment_checks_remote_artifact_before_reporting_it(agent, monkeypatch):
    checked = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"p"

    def get(url, **kwargs):
        checked.append((url, kwargs))
        return Response()

    monkeypatch.setattr("CoScientist.agents.callbacks.experiment_execution.requests.get", get)
    url = "https://artifacts.example/plot.png?token=test"
    context = _context()
    tool = Mock(spec=McpTool)
    tool.name = "predict_ld50"
    result = {"content": [{"type": "text", "text": json.dumps({
        "answer": {"ld50": 225, "plot": url}
    })}]}
    for callback in agent.canonical_after_tool_callbacks:
        callback(tool, {}, context, result)

    final = _after_model(agent, context, _response("Created plot"))
    recorded = json.loads(final.content.parts[0].text)["scientific_mcp_calls"][0]["result"]
    assert recorded["answer"]["plot"] == url
    assert checked == [(url, {"stream": True, "timeout": 5})]


@pytest.mark.parametrize(
    "tool_name,result",
    [
        ("dataset_overview_heracleum_tox", {"isError": True, "content": []}),
        ("dataset_overview_heracleum_tox", {"isError": False, "content": []}),
        ("dataset_overview_heracleum_tox", {"content": [{"type": "text", "text": ""}]}),
        ("dataset_overview_heracleum_tox", {
            "content": [{"type": "text", "text": '{"error":"dataset unavailable"}'}]
        }),
        ("dataset_overview_heracleum_tox", {
            "content": [{"type": "text", "text": '{"status":"success"}'}]
        }),
        ("dataset_overview_heracleum_tox", {
            "content": [{"type": "text", "text": '{"answer":{}}'}]
        }),
        ("update_task_status", {"status": "success"}),
    ],
)
def test_experiment_rejects_failed_empty_or_non_scientific_calls(
    agent, tool_name, result
):
    context = _context()
    tool = Mock(spec=McpTool if tool_name != "update_task_status" else object)
    tool.name = tool_name
    for callback in agent.canonical_after_tool_callbacks:
        callback(tool, {}, context, result)
    with pytest.raises(RuntimeError, match="scientific MCP"):
        _after_model(agent, context, _response("Done"))


def test_task_executor_passes_the_observed_pipeline_receipt_instead_of_model_prose():
    router = build_system(load_config(resolve_config_path("synapse_pilot"))).agent(
        "TaskExecutorAgent"
    )
    context = _context()
    pipeline = Mock(spec=AgentTool)
    pipeline.name = "ToolPipelineAgent"
    receipt = {
        "status": "computed",
        "scientific_mcp_calls": [
            {"tool": "dataset_overview_heracleum_tox", "result": {"n_reconstructed": 225}}
        ],
    }
    for callback in router.canonical_after_tool_callbacks:
        callback(pipeline, {}, context, json.dumps(receipt))

    final = _after_model(router, context, _response("Processed 12,654 metabolites"))
    assert "225" in final.content.parts[0].text
    assert "12,654" not in final.content.parts[0].text


def test_task_executor_rejects_a_model_authored_scientific_receipt():
    router = build_system(load_config(resolve_config_path("synapse_pilot"))).agent(
        "TaskExecutorAgent"
    )
    fake = json.dumps({"status": "computed", "scientific_mcp_calls": [
        {"tool": "predict_ld50", "result": {"ld50": 1.2}}
    ]})
    with pytest.raises(RuntimeError, match="observed ToolPipelineAgent"):
        _after_model(router, _context(), _response(fake))


class _ScriptedModel(BaseLlm):
    _responses: list = PrivateAttr()
    _index: int = PrivateAttr(default=0)
    _tool_configs: list = PrivateAttr(default_factory=list)
    _offered: list = PrivateAttr(default_factory=list)

    def __init__(self, responses):
        super().__init__(model="scripted-science")
        self._responses = responses

    async def generate_content_async(self, llm_request, stream=False):
        self._tool_configs.append(llm_request.config.tool_config)
        self._offered.append([
            declaration.name
            for tool in llm_request.config.tools or []
            for declaration in tool.function_declarations or []
        ])
        response = self._responses[self._index]
        self._index += 1
        yield response


class _ObservedMcpTool(McpTool):
    def __init__(self):
        super().__init__(
            mcp_tool=Tool(
                name="dataset_overview_heracleum_tox",
                inputSchema={"type": "object", "properties": {}},
            ),
            mcp_session_manager=Mock(),
        )
        self.calls = 0

    async def run_async(self, *, args, tool_context):
        self.calls += 1
        return {
            "isError": False,
            "content": [{"type": "text", "text": '{"answer":{"n_reconstructed":225}}'}],
        }


def test_adk_passes_real_mcp_receipt_through_pipeline_and_router():
    mcp_tool = _ObservedMcpTool()
    experiment_model = _ScriptedModel([
        LlmResponse(content=types.Content(role="model", parts=[
            types.Part.from_function_call(name="dataset_overview_heracleum_tox", args={})
        ])),
        _response("12,654 reconstructed metabolites, saved invented.png"),
    ])
    experiment = LlmAgent(
        name="ExperimentAgent",
        model=experiment_model,
        tools=[mcp_tool],
        before_agent_callback=reset_scientific_execution,
        before_model_callback=require_first_scientific_tool_call,
        after_tool_callback=record_scientific_mcp_result,
        after_model_callback=require_scientific_execution,
    )
    pipeline = SequentialAgent(name="ToolPipelineAgent", sub_agents=[experiment])
    router = LlmAgent(
        name="TaskExecutorAgent",
        model=_ScriptedModel([
            LlmResponse(content=types.Content(role="model", parts=[
                types.Part.from_function_call(
                    name="ToolPipelineAgent", args={"request": "Analyze Heracleum"}
                )
            ])),
            _response("Successfully analyzed 12,654 metabolites"),
        ]),
        tools=[AgentTool(agent=pipeline)],
        before_agent_callback=reset_executor_science_receipt,
        after_tool_callback=capture_scientific_pipeline_receipt,
        after_model_callback=attest_executor_science,
    )

    async def run():
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="science_test", user_id="user", session_id="session",
            state={
                "executor_tool_match": {"matched": True},
                "filtered_tools": [{"tool": "dataset_overview_heracleum_tox"}],
            },
        )
        runner = Runner(agent=router, app_name="science_test", session_service=sessions)
        events = [event async for event in runner.run_async(
            user_id="user", session_id="session",
            new_message=types.Content(role="user", parts=[types.Part(text="Analyze Heracleum")]),
        )]
        session = await sessions.get_session(
            app_name="science_test", user_id="user", session_id="session"
        )
        return events, session.state

    events, state = asyncio.run(run())
    assert mcp_tool.calls == 1
    assert (
        experiment_model._tool_configs[0].function_calling_config.mode
        == types.FunctionCallingConfigMode.ANY
    )
    assert experiment_model._offered[0] == ["dataset_overview_heracleum_tox"]
    assert experiment_model._tool_configs[1] is None
    assert state.get("_executor_science_receipt")
    receipt = json.loads(events[-1].content.parts[0].text)
    assert receipt["status"] == "computed"
    assert receipt["scientific_mcp_calls"][0]["result"] == {"answer": {"n_reconstructed": 225}}
    assert "12,654" not in events[-1].content.parts[0].text
