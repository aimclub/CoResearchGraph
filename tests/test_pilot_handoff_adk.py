"""Exercise the pilot handoff guard with a real ADK AgentTool call."""

import asyncio

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
from pydantic import PrivateAttr

from CoScientist.assembly import build_system
from CoScientist.assembly.schema import load_config, resolve_config_path


class _ScriptedModel(BaseLlm):
    _responses: list = PrivateAttr()
    _index: int = PrivateAttr(default=0)

    def __init__(self, responses):
        super().__init__(model="pilot-handoff-probe")
        self._responses = responses

    @property
    def calls(self):
        return self._index

    async def generate_content_async(self, llm_request, stream=False):
        response = self._responses[self._index]
        self._index += 1
        yield response


def _response(*, request=None, text=None):
    part = (
        types.Part.from_function_call(
            name="TaskExecutorAgent", args={"request": request}
        ) if request else types.Part(text=text)
    )
    return LlmResponse(content=types.Content(role="model", parts=[part]))


def test_pilot_handoff_retries_missing_tool_before_agent_tool_runs():
    url = "heracleum-server"
    worker_model = _ScriptedModel([_response(text="worker called")])
    worker = LlmAgent(name="TaskExecutorAgent", model=worker_model,
                      instruction="Return a result.")
    pilot = build_system(load_config(resolve_config_path("synapse_pilot")),
                         remote_subagents=True)
    model = _ScriptedModel([
        _response(request=f"Run predict_ld50 (server_id={url})"),
        _response(request=(
            f"Run predict_ld50 (server_id={url}); available "
            f"chemical_space_clustering (server_id={url})"
        )),
        _response(text="done"),
    ])
    agent = LlmAgent(name="PilotHandoffProbe", model=model,
                     instruction="Delegate the computation.",
                     tools=[AgentTool(agent=worker)],
                     before_tool_callback=pilot.root.before_tool_callback)

    async def run():
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="pilot_handoff", user_id="user", session_id="session",
            state={"accumulated_tools": [
                {"tool": "predict_ld50", "server_id": url},
                {"tool": "chemical_space_clustering", "server_id": url},
            ]},
        )
        runner = Runner(agent=agent, app_name="pilot_handoff", session_service=sessions)
        return [event async for event in runner.run_async(
            user_id="user", session_id="session",
            new_message=types.Content(role="user", parts=[types.Part(text="Run pilot")]),
        )]

    events = asyncio.run(run())
    responses = [response.response for event in events
                 for response in event.get_function_responses()
                 if response.name == "TaskExecutorAgent"]
    assert len(responses) == 2
    assert "chemical_space_clustering" in responses[0]["error"]
    assert worker_model.calls == 1
    assert "worker called" in str(responses[1])
