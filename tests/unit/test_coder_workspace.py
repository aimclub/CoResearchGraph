"""One sandbox workspace per ADK session.

Each CoderAgent delegation runs in its own ephemeral AgentTool sub-session, but
AgentTool copies the parent session state down and forwards deltas back — so a
workspace id pinned in session state must be reused across delegations and
across user messages, while staying isolated between sessions.
"""
import os
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()

from CoScientist.tools.coder_tools import (  # noqa: E402
    CoderToolset,
    seed_coder_workspace,
    _WORKSPACE_STATE_KEY,
)


def _sub_ctx(shared_state):
    """A CoderAgent sub-session sharing the (forwarded) top-level state dict."""
    return SimpleNamespace(
        state=shared_state,
        _invocation_context=SimpleNamespace(
            session=SimpleNamespace(id="sub-" + os.urandom(3).hex())
        ),
    )


def _orch_ctx(state, session_id):
    return SimpleNamespace(
        state=state,
        _invocation_context=SimpleNamespace(session=SimpleNamespace(id=session_id)),
    )


def setup_function():
    os.environ.pop("CODER_WORKSPACE_ID", None)


def test_single_workspace_across_delegations_and_messages():
    state = {}
    seed_coder_workspace(_orch_ctx(state, "adk-session-1"))
    assert state[_WORKSPACE_STATE_KEY] == "ws_adk-session-1"

    ws = [CoderToolset._workspace_id(_sub_ctx(state)) for _ in range(3)]
    assert len(set(ws)) == 1  # every delegation -> same sandbox

    # A later user message reuses the persisted session state.
    assert CoderToolset._workspace_id(_sub_ctx(state)) == ws[0]


def test_sessions_are_isolated():
    s1, s2 = {}, {}
    seed_coder_workspace(_orch_ctx(s1, "adk-session-1"))
    seed_coder_workspace(_orch_ctx(s2, "adk-session-2"))
    assert (
        CoderToolset._workspace_id(_sub_ctx(s1))
        != CoderToolset._workspace_id(_sub_ctx(s2))
    )


def test_self_seed_without_orchestrator_is_stable():
    """Standalone (no orchestrator seed): the toolset mints and pins its own id,
    stable across subsequent calls in the same session."""
    state = {}
    a = CoderToolset._workspace_id(_sub_ctx(state))
    b = CoderToolset._workspace_id(_sub_ctx(state))
    assert a == b == state[_WORKSPACE_STATE_KEY]


def test_explicit_pin_wins():
    os.environ["CODER_WORKSPACE_ID"] = "a2a_shared"
    try:
        assert CoderToolset._workspace_id(_sub_ctx({})) == "ws_a2a_shared"
    finally:
        os.environ.pop("CODER_WORKSPACE_ID", None)
