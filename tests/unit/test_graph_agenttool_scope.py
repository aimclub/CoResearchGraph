"""Graph scope survives ADK AgentTool child sessions.

AgentTool executes a delegated agent in a short-lived child session.  ADK
copies the parent state into that session, so graph resolution must use the
scope pinned in state instead of the child's generated ``session.id``.
"""

from types import SimpleNamespace
from uuid import uuid4

from CoScientist.graph.memory import get_knowledge_graph
from CoScientist.graph.research import store as research_store
from CoScientist.graph.session_scope import session_key


def _context(*, state: dict, user_id: str, session_id: str):
    session = SimpleNamespace(
        id=session_id,
        user_id=user_id,
        state=state,
    )
    return SimpleNamespace(
        state=state,
        session=session,
        _invocation_context=SimpleNamespace(session=session),
    )


def test_parent_scope_is_pinned_and_wins_over_agenttool_child_session_id():
    token = uuid4().hex
    user_id = f"user-{token}"
    parent_session_id = f"parent-{token}"
    parent_state = {}
    parent = _context(
        state=parent_state,
        user_id=user_id,
        session_id=parent_session_id,
    )

    assert session_key(parent) == (user_id, parent_session_id)
    assert parent_state, "the first parent resolution must pin its graph scope"

    child = _context(
        state=dict(parent_state),
        user_id=user_id,
        session_id=f"agent-tool-child-{uuid4().hex}",
    )

    assert session_key(child) == (user_id, parent_session_id)


