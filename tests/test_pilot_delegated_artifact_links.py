"""A pilot report may cite artifact URLs from observed delegated MCP receipts."""

import asyncio
import json

import pytest
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
from pydantic import PrivateAttr

from CoScientist.agents.callbacks.link_registry import link_id_for, register_user_links
from CoScientist.assembly import build_system
from CoScientist.assembly.schema import load_config, resolve_config_path


FIGURES = {
    "chemical_space_clustering": "https://example.org/artifacts/fig1_cluster_map_2c8f7904.png",
    "predict_ld50": "https://example.org/artifacts/fig2_ld50_routes_29c2b12c.png",
    "predict_general_toxicity": "https://example.org/artifacts/fig3_tox_heatmap_e73ab873.png",
}


class _ScriptedModel(BaseLlm):
    _responses: list = PrivateAttr()
    _index: int = PrivateAttr(default=0)

    def __init__(self, responses):
        super().__init__(model="pilot-artifact-probe")
        self._responses = responses

    async def generate_content_async(self, llm_request, stream=False):
        response = self._responses[self._index]
        self._index += 1
        yield response


def _text(value):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=value)]))


def _call(name, **args):
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part.from_function_call(name=name, args=args)
    ]))


def _receipt():
    return json.dumps({
        "status": "computed",
        "scientific_mcp_calls": [
            {"tool": "dataset_overview_heracleum_tox", "args": {},
             "result": {"answer": {"n_reconstructed": 225}}},
            *(
                {"tool": tool, "args": {}, "result": {"metadata": {"figure": url}}}
                for tool, url in FIGURES.items()
            ),
            {"tool": "predict_molecule_profile", "args": {},
             "result": {"answer": {"ld50_mgkg": 638.0}}},
        ],
    })


def _report(extra=""):
    refs = [f"[[{link_id_for(url)}]]" for url in FIGURES.values()]
    return (
        "## Научный отчёт по Heracleum\n\n"
        "| Этап | Наблюдение |\n|---|---|\n"
        "| Обзор | MCP вернул 225 реконструированных соединений |\n"
        f"| Кластеризация | Рисунок {refs[0]} |\n"
        f"| LD50 | Рисунок {refs[1]} |\n"
        f"| Общая токсичность | Тепловая карта создана: {refs[2]} |\n\n"
        "Полные строки молекул и SMILES недоступны из агрегированного обзора. "
        "Значения LD50 являются прогнозом модели; независимое экспериментальное "
        "подтверждение в полученных результатах отсутствует. " + extra
    )


def _run(report, initial_state=None):
    async def retrieve_tools(query: str) -> dict:
        return {"status": "ok"}

    async def ResearchAgent(request: str) -> dict:
        return {"status": "ok"}

    worker = LlmAgent(
        name="TaskExecutorAgent", model=_ScriptedModel([_text(_receipt())]),
        instruction="Return the computed receipt.",
    )
    pilot = build_system(load_config(resolve_config_path("synapse_pilot")), remote_subagents=True)
    root = LlmAgent(
        name="PilotArtifactProbe",
        model=_ScriptedModel([
            _call("retrieve_tools", query="Heracleum"),
            _call("ResearchAgent", request="Find papers"),
            _call("TaskExecutorAgent", request="Run scientific MCP tools"),
            _text(report),
        ]),
        tools=[retrieve_tools, ResearchAgent, AgentTool(agent=worker)],
        instruction="Write a grounded scientific report.",
        # The remote executor's link registry does not cross the A2A boundary.
        after_model_callback=pilot.root.after_model_callback,
    )

    async def run():
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="pilot_artifact_probe", user_id="user", session_id="session",
            state=initial_state,
        )
        runner = Runner(
            agent=root, app_name="pilot_artifact_probe", session_service=sessions
        )
        events = [event async for event in runner.run_async(
            user_id="user", session_id="session",
            new_message=types.Content(role="user", parts=[types.Part(text="Analyze Heracleum")]),
        )]
        session = await sessions.get_session(
            app_name="pilot_artifact_probe", user_id="user", session_id="session"
        )
        return events, session.state

    return asyncio.run(run())


def test_observed_delegated_figure_links_survive_missing_parent_registry():
    events, state = _run(_report())
    assert events[-1].content.parts[0].text.count("https://example.org/artifacts/") == 3
    assert all(link_id_for(url) in state["user_links"] for url in FIGURES.values())


def test_existing_parent_registry_uses_the_full_link_id():
    state = {}
    first_url = next(iter(FIGURES.values()))
    register_user_links(state, first_url, with_mentions=False)
    events, _ = _run(_report(), initial_state=state)
    assert first_url in events[-1].content.parts[0].text


def test_delegated_receipt_does_not_authorize_an_unobserved_link():
    false_url = "https://example.org/artifacts/invented-heatmap.png"
    with pytest.raises(RuntimeError, match="unsupported artifact link"):
        _run(_report(f" Дополнительный рисунок [[{link_id_for(false_url)}]]."))
