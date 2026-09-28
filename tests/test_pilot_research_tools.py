"""Pilot ResearchAgent calls respect the live paper-search MCP schemas."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpTool
from google.genai import types
from mcp.types import Tool
from pydantic import PrivateAttr

from CoScientist.agents.callbacks import research_pilot
from CoScientist.assembly import build_system
from CoScientist.assembly.schema import load_config, resolve_config_path


def _paper_tool(name, properties):
    return McpTool(
        mcp_tool=Tool(name=name, inputSchema={
            "type": "object", "properties": {key: {"type": "string"} for key in properties},
        }),
        mcp_session_manager=Mock(),
    )


def test_pilot_download_normalizes_only_invalid_paper_arguments():
    agent = build_system(load_config(resolve_config_path("synapse_pilot"))).agent(
        "ResearchAgent"
    )
    downloader = _paper_tool(
        "download_papers_from_search", ("keywords", "open_access", "limit", "sort")
    )
    context = SimpleNamespace(state={})
    args = {
        "keywords": "Heracleum oral LD50", "open_access": True,
        "has_pdf": True, "limit": 10, "sort": None,
    }
    for callback in agent.canonical_before_tool_callbacks:
        assert callback(downloader, args, context) is None
    assert args == {
        "keywords": "Heracleum oral LD50", "open_access": True, "limit": 10
    }


def test_pilot_preserves_supported_has_pdf_filter_and_other_tools():
    agent = build_system(load_config(resolve_config_path("synapse_pilot"))).agent(
        "ResearchAgent"
    )
    search = _paper_tool("search_papers", ("keywords", "has_pdf"))
    downloader = _paper_tool("download_papers_from_search", ("keywords", "has_pdf"))
    context = SimpleNamespace(state={})
    for tool in (search, downloader):
        args = {"keywords": "Heracleum", "has_pdf": True}
        for callback in agent.canonical_before_tool_callbacks:
            assert callback(tool, args, context) is None
        assert args == {"keywords": "Heracleum", "has_pdf": True}


def test_pilot_research_keeps_existing_callbacks():
    profile = load_config(resolve_config_path("synapse_pilot"))
    callbacks = profile.agent("ResearchAgent").callbacks
    assert "guard_unknown_tools" in callbacks.after_model
    assert "log_research_tool_calls" in callbacks.after_tool
    assert "inject_uploaded_papers" in callbacks.before_model
    assert "capture_research_tool_roster" in callbacks.before_model
    assert callbacks.after_model.index("normalize_research_function_name") < (
        callbacks.after_model.index("guard_unknown_tools")
    )


def test_pilot_repairs_channel_suffix_only_for_offered_tool():
    context = SimpleNamespace(
        state={}, _invocation_context=SimpleNamespace(invocation_id="run-1")
    )
    request = SimpleNamespace(config=SimpleNamespace(tools=[
        types.Tool(function_declarations=[
            types.FunctionDeclaration(name="download_papers_from_search")
        ])
    ]))
    research_pilot.capture_research_tool_roster(context, request)
    malformed = LlmResponse(content=types.Content(role="model", parts=[
        types.Part.from_function_call(
            name="download_papers_from_search<|channel|>commentary",
            args={"keywords": "Heracleum"},
        )
    ]))
    assert research_pilot.normalize_research_function_name(context, malformed) is None
    assert malformed.content.parts[0].function_call.name == "download_papers_from_search"

    unavailable = LlmResponse(content=types.Content(role="model", parts=[
        types.Part.from_function_call(name="search_papers", args={"keywords": "Heracleum"})
    ]))
    assert research_pilot.normalize_research_function_name(context, unavailable) is None
    assert unavailable.content.parts[0].function_call.name == "search_papers"


class _MalformedNameModel(BaseLlm):
    _turn: int = PrivateAttr(default=0)

    def __init__(self):
        super().__init__(model="scripted-research")

    async def generate_content_async(self, llm_request, stream=False):
        if self._turn == 0:
            result = LlmResponse(content=types.Content(role="model", parts=[
                types.Part.from_function_call(
                    name="download_papers_from_search<|channel|>commentary",
                    args={"keywords": "Heracleum"},
                )
            ]))
        else:
            result = LlmResponse(content=types.Content(
                role="model", parts=[types.Part(text="Paper lookup completed")]
            ))
        self._turn += 1
        yield result


def test_pilot_channel_suffix_reaches_offered_tool_in_adk():
    seen = []

    async def download_papers_from_search(keywords: str) -> dict:
        seen.append(keywords)
        return {"status": "ok"}

    async def run():
        agent = LlmAgent(
            name="ResearchNameProbe", model=_MalformedNameModel(),
            tools=[download_papers_from_search],
            before_model_callback=research_pilot.capture_research_tool_roster,
            after_model_callback=research_pilot.normalize_research_function_name,
        )
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="research_name_probe", user_id="user", session_id="session"
        )
        runner = Runner(
            agent=agent, app_name="research_name_probe", session_service=sessions
        )
        return [event async for event in runner.run_async(
            user_id="user", session_id="session",
            new_message=types.Content(role="user", parts=[types.Part(text="Find papers")]),
        )]

    events = asyncio.run(run())
    assert seen == ["Heracleum"]
    assert events[-1].content.parts[0].text == "Paper lookup completed"
