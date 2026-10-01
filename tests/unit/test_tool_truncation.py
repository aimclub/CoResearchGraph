"""A plugin must never decide that an agent's own after_tool callbacks run.

ADK runs plugin `after_tool_callback`s first and the agent's own only `if
altered_function_response is None` (`flows/llm_flows/_tool_caller.py`, steps 4
and 5), and the plugin manager early-exits on the first plugin that answers with
anything at all. So a plugin that returns the value it just modified silently
cancels every agent callback for that call.

That is not hypothetical. The truncation plugin returned its bounded copy on
every result over 12 000 characters, and web-search results run about 20 000 —
so on a live session, of seven Tavily calls the agent chain ran for four, and
the three it skipped were the ones whose results carried the paper links that
the paper capture exists to collect. `register_tool_result_links` and
the research logger were lost the same way, on the same calls.
"""
from __future__ import annotations

import asyncio

import pytest

from CoScientist.agents.truncation_plugin import (
    ToolResultTruncationPlugin,
    _cap_tool_results,
    _truncate_in_place,
)


class _Tool:
    name = "tavily_search"


def _result(chars: int) -> dict:
    """The shape a Tavily call actually returns: one giant string, nested."""
    return {"content": [{"type": "text", "text": "x" * chars}]}


def _plugins():
    """Every first-party plugin that hooks after_tool, constructed cheaply."""
    from CoScientist.agents.truncation_plugin import ToolResultTruncationPlugin
    from CoScientist.graph.plugin import GraphMemoryPlugin
    from CoScientist.logging.event_logger import EventLoggerPlugin
    from CoScientist.logging.tool_activity import ToolActivityPlugin
    from CoScientist.tools.mcp_artifact_plugin import McpArtifactCapturePlugin

    return [ToolResultTruncationPlugin(), GraphMemoryPlugin(), EventLoggerPlugin(),
            ToolActivityPlugin(), McpArtifactCapturePlugin()]


# ── the lock ────────────────────────────────────────────────────────────────
def test_no_plugin_swallows_the_agents_after_tool_chain():
    """The regression lock the whole change rests on. Fails before the fix."""
    swallowed = []
    for plugin in _plugins():
        answer = asyncio.run(plugin.after_tool_callback(
            tool=_Tool(), tool_args={"query": "furanocoumarins"},
            tool_context=None, result=_result(20000)))
        if answer is not None:
            swallowed.append(type(plugin).__name__)
    assert swallowed == [], (
        f"{swallowed} answered after_tool, so ADK skipped the agent's own "
        f"callbacks for that call")


# ── the cut still happens, at the right boundary ────────────────────────────
class _FunctionResponse:
    """Enough of `types.FunctionResponse`: the one method the fix relies on."""

    def __init__(self, response):
        self.response = response

    def model_copy(self, *, update):
        return _FunctionResponse(update["response"])


class _Part:
    def __init__(self, function_response=None):
        self.function_response = function_response


class _Request:
    def __init__(self, parts):
        self.contents = [type("C", (), {"parts": parts})()]


def test_the_model_sees_a_bounded_copy():
    part = _Part(_FunctionResponse(_result(20000)))
    assert _cap_tool_results(_Request([part]), 12000) == 1
    assert len(part.function_response.response["content"][0]["text"]) < 13000


def test_the_session_event_keeps_the_full_result():
    """The aliasing test, and the reason the fix is a `model_copy`.

    A request's parts are shallow copies whose payloads are shared with the
    session events, so writing into the response here would rewrite the stored
    record of what the tool actually returned.
    """
    original = _FunctionResponse(_result(20000))
    payload = original.response
    part = _Part(original)

    _cap_tool_results(_Request([part]), 12000)

    assert part.function_response is not original, "the part was replaced"
    assert len(payload["content"][0]["text"]) == 20000, (
        "the object the session event holds is untouched")


def test_a_part_that_is_not_a_tool_result_is_left_alone():
    part = _Part(None)
    assert _cap_tool_results(_Request([part]), 12000) == 0


# ── the safety net ──────────────────────────────────────────────────────────
def test_the_net_only_bites_at_the_huge_cap():
    """An ordinary web-search result must reach the agent callbacks intact."""
    ordinary = _result(20000)
    plugin = ToolResultTruncationPlugin()
    assert asyncio.run(plugin.after_tool_callback(
        tool=_Tool(), tool_args={}, tool_context=None, result=ordinary)) is None
    assert len(ordinary["content"][0]["text"]) == 20000

    pathological = _result(300000)
    asyncio.run(plugin.after_tool_callback(
        tool=_Tool(), tool_args={}, tool_context=None, result=pathological))
    assert len(pathological["content"][0]["text"]) < 201000


# The huge string is built inside the test, not parametrized: pytest puts a
# parameter into the test id, the id reaches the subprocess as an environment
# variable, and Windows caps those at 32767 characters.
@pytest.mark.parametrize("shape", ["long string", "number", "none", "object"])
def test_a_result_the_net_cannot_cap_is_not_mangled(shape):
    """Only a container can be capped in place. Anything else passes through
    rather than being quietly replaced with something the turn did not expect."""
    result = {"long string": "x" * 300000, "number": 42,
              "none": None, "object": object()}[shape]
    plugin = ToolResultTruncationPlugin()
    assert asyncio.run(plugin.after_tool_callback(
        tool=_Tool(), tool_args={}, tool_context=None, result=result)) is None
    assert not _truncate_in_place(result, 100)


def test_before_model_never_answers_for_the_model():
    """A non-None answer there is an early end to the whole model call."""
    plugin = ToolResultTruncationPlugin()
    part = _Part(_FunctionResponse(_result(20000)))
    assert asyncio.run(plugin.before_model_callback(
        callback_context=None, llm_request=_Request([part]))) is None


def test_a_broken_request_does_not_fail_the_call():
    plugin = ToolResultTruncationPlugin()
    assert asyncio.run(plugin.before_model_callback(
        callback_context=None, llm_request=object())) is None
