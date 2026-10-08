"""Tests for the Coder↔Executor redirect mechanism (mechanism B).

When the Executor's tool-prep pipeline finds no tool that actually matches the
task, it must ABSTAIN and recommend CoderAgent — not run a nearest-but-wrong
tool (the "train a GAN for a 'train a transformer' task" failure).
"""
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()

from google.adk.models import LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402

from CoScientist.agents.callbacks import (  # noqa: E402
    after_tool_reranker_agent,
    make_unknown_tool_guard,
    redirect_when_no_tools,
)
from CoScientist.agents.callbacks.tool_callbacks import (  # noqa: E402
    NO_MATCHING_TOOL_TOKEN,
    TOOL_MATCH_STATE_KEY,
)


def _ctx(state):
    return SimpleNamespace(state=state)


def _rerank(state):
    after_tool_reranker_agent(_ctx(state))


def test_no_relevant_tool_abstains_and_redirects():
    state = {
        "reranked_tools": {"tools": [{"index": 0, "score": 0.05}, {"index": 1, "score": 0.02}]},
        "accumulated_tools": [
            {"tool_index": 0, "tool": "train_gan"},
            {"tool_index": 1, "tool": "dock"},
        ],
    }
    _rerank(state)
    assert state[TOOL_MATCH_STATE_KEY]["matched"] is False
    assert state["filtered_tools"] == []

    redirect = redirect_when_no_tools(_ctx(state))
    assert redirect is not None
    assert NO_MATCHING_TOOL_TOKEN in redirect.parts[0].text
    assert "CoderAgent" in redirect.parts[0].text


def test_real_match_proceeds_without_redirect():
    state = {
        "reranked_tools": {"tools": [{"index": 0, "score": 0.8}]},
        "accumulated_tools": [{"tool_index": 0, "tool": "compute_logp"}],
    }
    _rerank(state)
    assert state[TOOL_MATCH_STATE_KEY]["matched"] is True
    assert state["filtered_tools"]
    assert redirect_when_no_tools(_ctx(state)) is None


def test_marginal_scores_salvage_top2_and_proceed():
    """Scores between the abstain and keep bars salvage top-2 (cautious proceed),
    preserving the original pipeline behaviour for borderline matches."""
    state = {
        "reranked_tools": {"tools": [{"index": 0, "score": 0.25}, {"index": 1, "score": 0.22}]},
        "accumulated_tools": [{"tool_index": 0, "tool": "a"}, {"tool_index": 1, "tool": "b"}],
    }
    _rerank(state)
    assert state[TOOL_MATCH_STATE_KEY]["matched"] is True
    assert len(state["filtered_tools"]) == 2
    assert redirect_when_no_tools(_ctx(state)) is None


def test_redirect_skipped_when_web_mcp_deployed():
    """If a web MCP was deployed, the Executor has something to run — no abstain."""
    state = {
        "reranked_tools": {"tools": [{"index": 0, "score": 0.01}]},
        "accumulated_tools": [{"tool_index": 0, "tool": "x"}],
        "filtered_mcps": [{"server_id": "remote-1"}],
    }
    _rerank(state)
    assert state[TOOL_MATCH_STATE_KEY]["matched"] is False
    # filtered_mcps present -> guard does not redirect.
    assert redirect_when_no_tools(_ctx(state)) is None


# ── Hallucinated-tool guard (CoderAgent calling `find` directly) ─────────────

_CODER_TOOLS = [
    "execute_bash", "check_job", "read_file",
    "write_file", "list_directory", "install_package",
]


def _fc_response(name, args):
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
        )
    )


def test_unknown_tool_call_is_corrected_not_crashed():
    guard = make_unknown_tool_guard(_CODER_TOOLS)
    out = guard(SimpleNamespace(agent_name="CoderAgent"), _fc_response("find", {"path": "src"}))
    assert out is not None
    text = out.content.parts[0].text
    assert "do not exist" in text and "execute_bash" in text


def test_valid_tool_call_passes_through():
    guard = make_unknown_tool_guard(_CODER_TOOLS)
    out = guard(
        SimpleNamespace(agent_name="CoderAgent"),
        _fc_response("execute_bash", {"command": "find . | wc -l"}),
    )
    assert out is None


def test_plain_text_response_passes_through():
    guard = make_unknown_tool_guard(_CODER_TOOLS)
    resp = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="done")]))
    assert guard(SimpleNamespace(agent_name="CoderAgent"), resp) is None
