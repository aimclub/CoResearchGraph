"""HITL requests keep the public session scope inside AgentTool children."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from CoScientist.graph.session_scope import session_key
from CoScientist.hitl.callbacks import (
    make_hitl_after_callback,
    make_hitl_before_callback,
)
from CoScientist.hitl.models import HITLAction, HITLResponse
from CoScientist.hitl.tool import HITLToolset


class _RecordingHandler:
    def __init__(self):
        self.requests = []

    async def handle_request(self, request):
        self.requests.append(request)
        return HITLResponse(action=HITLAction.APPROVE, approved=True)


def _context(*, state: dict, user_id: str, session_id: str, **extra):
    session = SimpleNamespace(id=session_id, user_id=user_id, state=state)
    return SimpleNamespace(
        state=state,
        session=session,
        _invocation_context=SimpleNamespace(session=session, **extra),
    )


def _agenttool_child_context(**extra):
    token = uuid4().hex
    parent_user_id = f"parent-user-{token}"
    parent_session_id = f"parent-session-{token}"
    parent = _context(
        state={},
        user_id=parent_user_id,
        session_id=parent_session_id,
    )
    assert session_key(parent) == (parent_user_id, parent_session_id)

    child_session_id = f"agent-tool-child-{uuid4().hex}"
    child = _context(
        state=dict(parent.state),
        user_id=f"child-user-{uuid4().hex}",
        session_id=child_session_id,
        **extra,
    )
    return child, (parent_user_id, parent_session_id), child_session_id


def _request_scope(handler):
    assert len(handler.requests) == 1
    return handler.requests[0].context["_session"]


def test_hitl_tool_uses_parent_scope_in_agenttool_child_session():
    async def scenario():
        handler = _RecordingHandler()
        toolset = HITLToolset(handler)
        child, parent_key, child_session_id = _agenttool_child_context()

        await toolset.request_approval(
            agent_name="ChildAgent",
            message="Approve delegated work",
            tool_context=child,
        )

        assert _request_scope(handler) == {
            "user_id": parent_key[0],
            "session_id": parent_key[1],
        }
        assert _request_scope(handler)["session_id"] != child_session_id

    asyncio.run(scenario())


def test_hitl_selection_uses_parent_scope_in_agenttool_child_session():
    async def scenario():
        handler = _RecordingHandler()
        toolset = HITLToolset(handler)
        child, parent_key, child_session_id = _agenttool_child_context()

        await toolset.request_selection(
            agent_name="ChildAgent",
            message="Choose delegated work",
            options=["first", "second"],
            tool_context=child,
        )

        assert _request_scope(handler) == {
            "user_id": parent_key[0],
            "session_id": parent_key[1],
        }
        assert _request_scope(handler)["session_id"] != child_session_id

    asyncio.run(scenario())


def test_hitl_approval_resolves_scope_only_once(monkeypatch):
    async def scenario():
        handler = _RecordingHandler()
        toolset = HITLToolset(handler)
        child, parent_key, _ = _agenttool_child_context()
        calls = []

        def resolve_once(context):
            calls.append(context)
            return parent_key

        monkeypatch.setattr("CoScientist.hitl.tool.session_key", resolve_once)
        await toolset.request_approval(
            agent_name="ChildAgent",
            message="Approve delegated work",
            tool_context=child,
        )

        assert calls == [child]

    asyncio.run(scenario())


@pytest.mark.parametrize("callback_kind", ["before", "after"])
def test_hitl_callback_uses_parent_scope_in_agenttool_child_session(callback_kind, monkeypatch):
    async def scenario():
        from CoScientist.config import get_settings

        monkeypatch.setattr(get_settings().web, "hitl_enabled", True)
        handler = _RecordingHandler()
        agent = SimpleNamespace(output_key="delegated_output")
        child, parent_key, child_session_id = _agenttool_child_context(agent=agent)
        child.agent_name = "ChildAgent"

        if callback_kind == "before":
            callback = make_hitl_before_callback(handler)
        else:
            child.state["delegated_output"] = "Delegated result"
            callback = make_hitl_after_callback(handler, HITLAction.APPROVE)

        assert await callback(child) is None
        assert _request_scope(handler) == {
            "user_id": parent_key[0],
            "session_id": parent_key[1],
        }
        assert _request_scope(handler)["session_id"] != child_session_id

    asyncio.run(scenario())
