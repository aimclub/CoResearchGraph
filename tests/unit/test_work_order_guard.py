"""The Work Order guard: an approved contract binds the agent; going beyond it
is an amendment, not a silent deviation."""
import asyncio
from types import SimpleNamespace

import pytest

from CoScientist.config import get_settings
from CoScientist.hitl.models import HITLAction, HITLResponse
from CoScientist.hitl.work_order import load_order, order_key
from CoScientist.hitl.work_order_guard import make_reset_work_order, make_work_order_guard
from CoScientist.hitl.work_order_tools import WorkOrderToolset

AGENT = "DatasetCollectorAgent"
TOOLS = ["tavily_search", "execute_bash", "read_file", "list_directory", "install_package"]
INTERNAL = ["update_task_status", "sleep_tool"]


class _Handler:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.notices = []

    async def handle_request(self, request):
        return self.responses.pop(0) if self.responses else HITLResponse(
            action=HITLAction.APPROVE, approved=True
        )

    async def notify(self, payload):
        self.notices.append(payload)


class _State(dict):
    """Records every top-level write, the way ADK turns them into a state delta."""

    def __init__(self):
        super().__init__()
        self.writes = []

    def __setitem__(self, key, value):
        self.writes.append(key)
        super().__setitem__(key, value)


@pytest.fixture(autouse=True)
def hitl_on(monkeypatch):
    web = get_settings().web
    monkeypatch.setattr(web, "hitl_enabled", True)
    monkeypatch.setattr(web, "work_order_enabled", True)


@pytest.fixture
def env():
    state = _State()
    session = SimpleNamespace(id="s1", user_id="u1", state=state)
    ctx = SimpleNamespace(state=state, session=session, agent_name=AGENT,
                          _invocation_context=SimpleNamespace(session=session))
    handler = _Handler()
    return SimpleNamespace(
        ctx=ctx, handler=handler,
        toolset=WorkOrderToolset(AGENT, TOOLS, handler, internal_tools=INTERNAL),
        guard=make_work_order_guard(AGENT, handler, internal_tools=INTERNAL),
    )


def _call(env, tool_name, args=None):
    return asyncio.run(env.guard(
        tool=SimpleNamespace(name=tool_name), args=args or {}, tool_context=env.ctx,
    ))


def _declare(env, **overrides):
    args = dict(
        goal="Collect a dataset",
        done_criteria="CSV written",
        assumptions=["ChEMBL is the source"],
        steps=[{"title": "Download", "tools": ["execute_bash"]}],
        planned_tools=["execute_bash"],
        tool_context=env.ctx,
    )
    args.update(overrides)
    return asyncio.run(env.toolset.declare_work_order(**args))


def test_before_declaring_only_orientation_protocol_and_internal_tools_pass(env):
    assert _call(env, "list_directory") is None
    assert _call(env, "update_task_status") is None
    assert _call(env, "request_approval") is None

    blocked = _call(env, "tavily_search")
    assert blocked["blocked_by"] == "work_order_guard"
    assert blocked["reason"] == "no_work_order"
    assert "declare_work_order" in blocked["message"]


def test_declared_tools_pass(env):
    _declare(env)
    assert _call(env, "execute_bash", {"command": "python fetch.py"}) is None
    assert _call(env, "execute_bash", {"command": "wget https://x/y.gz"}) is None


def test_undeclared_tool_is_blocked_with_an_amendment_hint_and_recorded(env):
    _declare(env)
    blocked = _call(env, "tavily_search", {"query": "BTK"})
    assert blocked["reason"] == "undeclared_tool"
    assert "update_work_order" in blocked["message"]

    order = load_order(env.ctx.state, AGENT)
    assert order.deviations[0]["reason"] == "undeclared_tool"
    assert order.deviations[0]["tool"] == "tavily_search"
    assert env.handler.notices[-1]["kind"] == "deviation"


def test_declared_tool_runs_commands_without_side_effect_blocks(env):
    _declare(env)
    assert _call(env, "execute_bash", {"command": "git push origin main"}) is None


def test_after_an_approved_amendment_the_call_goes_through(env):
    _declare(env)
    assert _call(env, "tavily_search")["reason"] == "undeclared_tool"
    asyncio.run(env.toolset.update_work_order(
        reason="Need the target id", add_tools=["tavily_search"], tool_context=env.ctx,
    ))
    assert _call(env, "tavily_search") is None


def test_rejected_order_blocks_everything_but_the_protocol(env):
    env.handler.responses.append(HITLResponse(action=HITLAction.REJECT, approved=False))
    _declare(env)
    assert _call(env, "execute_bash", {"command": "ls"})["reason"] == "rejected"
    assert _call(env, "update_work_step") is None


def test_reset_starts_each_delegation_without_a_contract(env):
    _declare(env)
    _call(env, "execute_bash", {"command": "ls"})
    reset = make_reset_work_order(AGENT)
    reset(callback_context=env.ctx)

    assert env.ctx.state[order_key(AGENT)] is None
    assert _call(env, "execute_bash", {"command": "ls"})["reason"] == "no_work_order"


def test_state_is_written_through_top_level_keys(env):
    """AgentTool forwards only state deltas; an in-place nested mutation is lost."""
    _declare(env)
    env.ctx.state.writes.clear()
    _call(env, "tavily_search")
    assert order_key(AGENT) in env.ctx.state.writes


def test_internal_tools_pass_an_approved_order_without_being_declared(env):
    _declare(env)
    assert _call(env, "sleep_tool") is None
    assert _call(env, "update_task_status") is None
