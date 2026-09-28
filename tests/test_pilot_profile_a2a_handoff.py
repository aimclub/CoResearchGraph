"""A2A executor must turn a model's JSON-shaped call into an observed MCP call."""

import asyncio
import json
from unittest.mock import Mock

import httpx
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
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
    reset_executor_science_receipt,
    reset_scientific_execution,
)
from CoScientist.assembly import build_system
from CoScientist.assembly.schema import load_config, resolve_config_path


SMILES = "Cc1cc2cc3c(C)cc(=O)oc3c(C)c2o1"


class _ScriptedModel(BaseLlm):
    _responses: list = PrivateAttr()
    _index: int = PrivateAttr(default=0)
    _offered: list = PrivateAttr(default_factory=list)
    _modes: list = PrivateAttr(default_factory=list)

    def __init__(self, responses):
        super().__init__(model="pilot-profile-a2a-probe")
        self._responses = responses

    async def generate_content_async(self, llm_request, stream=False):
        self._offered.append([
            declaration.name
            for tool in llm_request.config.tools or []
            for declaration in tool.function_declarations or []
        ])
        config = llm_request.config.tool_config
        self._modes.append(config.function_calling_config.mode if config else None)
        response = self._responses[self._index]
        self._index += 1
        yield response


def _text(value):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=value)]))


class _ObservedProfileTool(McpTool):
    def __init__(self):
        super().__init__(
            mcp_tool=Tool(
                name="predict_molecule_profile",
                inputSchema={
                    "type": "object",
                    "properties": {"name_or_smiles": {"type": "string"}},
                    "required": ["name_or_smiles"],
                },
            ),
            mcp_session_manager=Mock(),
        )
        self.calls = []

    async def run_async(self, *, args, tool_context):
        self.calls.append(args)
        return {
            "isError": False,
            "structuredContent": {
                "answer": {"smiles": SMILES, "ld50": {"ld50_mgkg": 635.5}}
            },
        }


def _run_a2a_profile_handoff(call_name="predict_molecule_profile"):
    pseudo_call = json.dumps([{
        "name": call_name,
        "parameters": {"name_or_smiles": SMILES},
    }])
    tool = _ObservedProfileTool()
    pilot = build_system(load_config(resolve_config_path("synapse_pilot")))
    configured_experiment = pilot.agent("ExperimentAgent")
    experiment_model = _ScriptedModel([_text(pseudo_call), _text("Profile complete")])
    experiment = LlmAgent(
        name="ExperimentAgent", model=experiment_model, tools=[tool],
        before_agent_callback=reset_scientific_execution,
        before_model_callback=configured_experiment.before_model_callback,
        after_tool_callback=configured_experiment.after_tool_callback,
        after_model_callback=configured_experiment.after_model_callback,
    )
    pipeline = SequentialAgent(name="ToolPipelineAgent", sub_agents=[experiment])

    def prepare_executor(callback_context):
        reset_executor_science_receipt(callback_context)
        callback_context.state.update({
            "executor_tool_match": {"matched": True},
            "filtered_tools": [{"tool": "predict_molecule_profile"}],
            "explicit_tool_target": "predict_molecule_profile",
        })
        return None

    router = LlmAgent(
        name="TaskExecutorAgent",
        model=_ScriptedModel([
            LlmResponse(content=types.Content(role="model", parts=[
                types.Part.from_function_call(
                    name="ToolPipelineAgent",
                    args={"request": (
                        "Target tool: predict_molecule_profile "
                        "(server_id=bfc62a287aaf7b5a). "
                        f"Call with name_or_smiles={SMILES}."
                    )},
                )
            ])),
            _text("Profile complete"),
        ]),
        tools=[AgentTool(agent=pipeline)],
        before_agent_callback=prepare_executor,
        after_tool_callback=capture_scientific_pipeline_receipt,
        after_model_callback=attest_executor_science,
    )

    async def run():
        runner = Runner(
            agent=router, app_name="pilot_profile_a2a",
            session_service=InMemorySessionService(),
        )
        card = AgentCard(
            name="TaskExecutorAgent", description="Profile handoff test",
            url="http://pilot-profile.test/", version="1",
            capabilities=AgentCapabilities(streaming=True),
            defaultInputModes=["text/plain"], defaultOutputModes=["text/plain"],
            skills=[],
        )
        app = A2AFastAPIApplication(
            agent_card=card,
            http_handler=DefaultRequestHandler(
                agent_executor=A2aAgentExecutor(runner=runner),
                task_store=InMemoryTaskStore(),
            ),
        ).build()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://pilot-profile.test"
        ) as client:
            response = await client.post("/", json={
                "jsonrpc": "2.0", "id": "profile-handoff", "method": "message/send",
                "params": {"message": {
                    "kind": "message", "role": "user", "messageId": "profile-message",
                    "parts": [{"kind": "text", "text": (
                        "Run predict_molecule_profile with "
                        f"name_or_smiles={SMILES}"
                    )}],
                }},
            })
        response.raise_for_status()
        return response.json()

    return asyncio.run(run()), tool.calls, experiment_model


def test_a2a_pipeline_executes_json_shaped_profile_call_as_real_mcp_tool():
    payload, calls, experiment_model = _run_a2a_profile_handoff()
    assert experiment_model._offered[0] == ["predict_molecule_profile"]
    assert experiment_model._modes[0] == types.FunctionCallingConfigMode.ANY
    assert calls == [{"name_or_smiles": SMILES}]
    assert "error" not in payload
    texts = [
        part.get("text", "")
        for artifact in payload["result"].get("artifacts", [])
        for part in artifact.get("parts", [])
    ]
    assert any('"status": "computed"' in text and SMILES in text for text in texts)


def test_a2a_pipeline_does_not_execute_unoffered_json_tool_name():
    payload, calls, _ = _run_a2a_profile_handoff("invented_profile_tool")
    assert calls == []
    assert payload.get("error") or payload.get("result", {}).get("status", {}).get("state") == "failed"
