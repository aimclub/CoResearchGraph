"""Pilot ResearchAgent calls respect the live paper-search MCP schemas."""

from types import SimpleNamespace
from unittest.mock import Mock

from google.adk.tools.mcp_tool import McpTool
from mcp.types import Tool

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
