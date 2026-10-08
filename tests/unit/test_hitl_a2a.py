"""HITL over A2A: the tools must not block the server, and must keep the names
the prompts/docs/unknown-tool guard expect."""
import asyncio
import importlib
import os

import pytest

from CoScientist.assembly.bindings import HITL_TOOL_DOCS
from CoScientist.hitl.handler import ConsoleHITLHandler
from CoScientist.hitl.models import HITLAction, HITLRequest


def _tools(a2a: bool, root: bool = True):
    prev = os.environ.get("COSCIENTIST_A2A_MODE")
    if a2a:
        os.environ["COSCIENTIST_A2A_MODE"] = "1"
    else:
        os.environ.pop("COSCIENTIST_A2A_MODE", None)
    try:
        import CoScientist.hitl.tool as tool_mod
        importlib.reload(tool_mod)
        return tool_mod.get_hitl_tools(a2a_root=root)
    finally:
        if prev is None:
            os.environ.pop("COSCIENTIST_A2A_MODE", None)
        else:
            os.environ["COSCIENTIST_A2A_MODE"] = prev
        import CoScientist.hitl.tool as tool_mod
        importlib.reload(tool_mod)


def test_a2a_mode_attaches_long_running_hitl_tools():
    """Over A2A the HITL tools must be long-running: ADK then reports the call to
    the client and the A2A task goes to `input-required` instead of the server
    blocking on a console it does not have."""
    tools = _tools(a2a=True)
    assert {t.name for t in tools} == {"request_approval", "request_selection"}
    assert all(getattr(t, "is_long_running", False) for t in tools)


def test_a2a_non_root_keeps_handler_tools():
    """A pause inside a sub-agent does not reach the caller: the parent wraps it
    in an AgentTool that simply returns the sub-run's final text, so a paused
    sub-agent yields an EMPTY result and the parent carries on — the human review
    silently vanishes. Non-root A2A agents therefore keep the handler path."""
    tools = _tools(a2a=True, root=False)
    assert {t.name for t in tools} == {"request_approval", "request_selection"}
    assert not any(getattr(t, "is_long_running", False) for t in tools)


def test_agent_tool_does_not_propagate_a_pause():
    """The ADK fact the rule above depends on: the AgentTool our assembler uses
    for sub-agents is neither long-running nor response-deferring, so a sub-agent
    pause cannot be surfaced to the caller as `input-required`."""
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.tools.agent_tool import AgentTool

    tool = AgentTool(agent=LlmAgent(name="Sub"))
    assert getattr(tool, "is_long_running", False) is False
    assert getattr(tool, "_defers_response", False) is False


def test_in_process_mode_keeps_blocking_tools():
    """The in-process/web path is unchanged (handler-driven, not long-running)."""
    tools = _tools(a2a=False)
    assert {t.name for t in tools} == {"request_approval", "request_selection"}
    assert not any(getattr(t, "is_long_running", False) for t in tools)


def test_tool_names_match_docs_so_the_guard_accepts_them():
    """Names must equal HITL_TOOL_DOCS, else guard_unknown_tools would block a
    legitimate HITL call (the guard builds its valid set from the docs)."""
    doc_names = {d.name for d in HITL_TOOL_DOCS}
    assert {t.name for t in _tools(a2a=True)} == doc_names
    assert {t.name for t in _tools(a2a=False)} == doc_names


def test_a2a_tools_return_nothing_so_the_run_actually_pauses():
    """ADK only skips the auto-FunctionResponse when a long-running tool returns a
    FALSY value (flows/llm_flows/functions.py:
    `if (tool.is_long_running or tool._defers_response) and not function_response`).
    A tool that returns a "pending" dict does NOT pause — the model would receive
    it and invent the human's answer. So these must return nothing, fast."""
    from CoScientist.hitl.a2a_tools import request_approval, request_selection

    assert asyncio.run(asyncio.wait_for(request_approval("A", "Approve?"), timeout=5)) is None
    assert asyncio.run(
        asyncio.wait_for(request_selection("A", "Pick", ["x", "y"]), timeout=5)) is None


def test_adk_pause_condition_still_holds():
    """Guard against an ADK upgrade silently changing the contract above."""
    import inspect

    from google.adk.flows.llm_flows import functions as adk_functions

    src = inspect.getsource(adk_functions)
    assert "is_long_running or tool._defers_response" in src
    assert "and not function_response" in src


def test_console_handler_does_not_hang_without_a_tty():
    """Safety net: any headless path (A2A/uvicorn/cron) must get an answer
    instead of blocking on stdin."""
    req = HITLRequest(agent_name="A", action_type=HITLAction.APPROVE, message="proceed?")
    resp = asyncio.run(asyncio.wait_for(ConsoleHITLHandler().handle_request(req), timeout=5))
    assert resp.action in (HITLAction.APPROVE, HITLAction.REJECT)
    assert resp.instructions and "headless" in resp.instructions.lower()


def test_console_headless_policy_can_reject(monkeypatch):
    monkeypatch.setenv("HITL_HEADLESS_POLICY", "reject")
    req = HITLRequest(agent_name="A", action_type=HITLAction.APPROVE, message="proceed?")
    resp = asyncio.run(asyncio.wait_for(ConsoleHITLHandler().handle_request(req), timeout=5))
    assert resp.approved is False
