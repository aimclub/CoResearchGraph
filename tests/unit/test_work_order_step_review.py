"""Step review: each finished Work Order step goes before the human with what was
sent, what was expected and what was found, plus the calls the system recorded."""
import asyncio
from types import SimpleNamespace

import pytest

from CoScientist.assembly.schema import AgentConfig
from CoScientist.hitl import work_order_tools
from CoScientist.hitl.models import HITLAction, HITLResponse
from CoScientist.hitl.work_order import load_order, report_warnings, save_order
from CoScientist.hitl.work_order_guard import (
    make_work_order_guard,
    make_work_step_journal,
    tool_result_excerpt,
)
from CoScientist.hitl.work_order_tools import WorkOrderToolset

AGENT = "EconomicsAgent"
TOOLS = ["resolve_chemicals", "rank_routes_by_cost"]


class FakeHandler:
    """Answers every blocking request with the next queued response."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.notices = []

    async def handle_request(self, request):
        self.requests.append(request)
        return self.responses.pop(0) if self.responses else _approve()

    async def notify(self, body):
        self.notices.append(body)


def _approve(notes=""):
    return HITLResponse(action=HITLAction.APPROVE, approved=True, instructions=notes or None)


def _edit(notes):
    return HITLResponse(action=HITLAction.EDIT, approved=False, instructions=notes)


def _reject(notes):
    return HITLResponse(action=HITLAction.REJECT, approved=False, instructions=notes)


def _ctx(state=None):
    return SimpleNamespace(state={} if state is None else state)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _active(monkeypatch):
    monkeypatch.setattr(work_order_tools, "work_order_active", lambda: True)


STEPS = [
    {"title": "Разрешить вещества", "tools": ["resolve_chemicals"],
     "inputs": "1-dodecanol, chlorosulfonic acid", "expected_outcome": "SMILES для всех"},
    {"title": "Рейтинг маршрутов", "tools": ["rank_routes_by_cost"],
     "inputs": "маршруты GPN-1, LIT-1; 100 g", "expected_outcome": "статус и цена в RUB"},
]


def _declare(toolset, ctx, steps=STEPS):
    return _run(toolset.declare_work_order(
        goal="Сравнить маршруты по стоимости", done_criteria="у каждого маршрута есть статус",
        assumptions=["Целевое количество — 100 g"], steps=steps,
        planned_tools=TOOLS, tool_context=ctx,
    ))


# ── Declaring ────────────────────────────────────────────────────────────────

def test_step_review_requires_inputs_for_tool_steps():
    toolset = WorkOrderToolset(AGENT, TOOLS, handler=FakeHandler(), step_review=True)
    steps = [{"title": "Рейтинг", "tools": ["rank_routes_by_cost"], "expected_outcome": "цены"}]
    result = _declare(toolset, _ctx(), steps)
    assert result["status"] == "error"
    assert "inputs" in result["message"]


def test_without_step_review_inputs_stay_optional():
    toolset = WorkOrderToolset(AGENT, TOOLS, handler=FakeHandler())
    steps = [{"title": "Рейтинг", "tools": ["rank_routes_by_cost"]}]
    assert _declare(toolset, _ctx(), steps)["status"] == "approved"


def test_inputs_and_mode_are_recorded_on_the_order():
    ctx = _ctx()
    toolset = WorkOrderToolset(AGENT, TOOLS, handler=FakeHandler(), step_review=True)
    assert _declare(toolset, ctx)["status"] == "approved"
    order = load_order(ctx.state, AGENT)
    assert order.step_review is True
    assert order.steps[0].inputs == "1-dodecanol, chlorosulfonic acid"


# ── Reviewing a step ─────────────────────────────────────────────────────────

def _started(handler):
    ctx = _ctx()
    toolset = WorkOrderToolset(AGENT, TOOLS, handler=handler, step_review=True)
    _declare(toolset, ctx)
    _run(toolset.update_work_step("S1", "in_progress", ctx))
    return toolset, ctx


def test_done_needs_a_result():
    toolset, ctx = _started(FakeHandler(_approve()))
    result = _run(toolset.update_work_step("S1", "done", ctx, note="ok"))
    assert result["status"] == "error" and "result" in result["message"]


def test_accepted_step():
    handler = FakeHandler(_approve(), _approve("проверь CAS"))
    toolset, ctx = _started(handler)
    result = _run(toolset.update_work_step("S1", "done", ctx, result="3 из 3 SMILES"))

    assert result["status"] == "accepted"
    assert result["operator_notes"] == "проверь CAS"
    request = handler.requests[-1]
    assert request.trigger == "work_step"
    assert request.context["step"]["result"] == "3 из 3 SMILES"
    step = load_order(ctx.state, AGENT).step("S1")
    assert step.status == "done" and step.review.status == "accepted"


def test_sent_back_step_is_in_progress_again_with_history():
    handler = FakeHandler(_approve(), _edit("используй английские имена"))
    toolset, ctx = _started(handler)
    order = load_order(ctx.state, AGENT)
    order.steps[0].calls.append({"tool": "resolve_chemicals", "args": "x", "result_excerpt": "y"})
    save_order(ctx.state, order)

    result = _run(toolset.update_work_step("S1", "done", ctx, result="1 из 3"))

    assert result["status"] == "revise"
    step = load_order(ctx.state, AGENT).step("S1")
    assert step.status == "in_progress"
    assert step.calls == []
    assert step.review.round == 2
    assert step.review.history[0]["result"] == "1 из 3"
    assert step.review.history[0]["calls"][0]["tool"] == "resolve_chemicals"


def test_stopped_step_rejects_the_order_and_the_guard_blocks():
    handler = FakeHandler(_approve(), _reject("данные неверны"))
    toolset, ctx = _started(handler)
    result = _run(toolset.update_work_step("S1", "done", ctx, result="мусор"))
    assert result["status"] == "rejected"
    assert load_order(ctx.state, AGENT).status == "rejected"

    guard = make_work_order_guard(AGENT, handler=handler)
    blocked = _run(guard(tool=SimpleNamespace(name="rank_routes_by_cost"), args={}, tool_context=ctx))
    assert blocked["status"] == "blocked"


def test_without_step_review_a_done_step_is_only_a_notice():
    handler = FakeHandler(_approve())
    ctx = _ctx()
    toolset = WorkOrderToolset(AGENT, TOOLS, handler=handler)
    _declare(toolset, ctx)
    result = _run(toolset.update_work_step("S1", "done", ctx, note="ok"))
    assert result == {"status": "ok", "step": "S1", "step_status": "done"}
    assert all(r.trigger != "work_step" for r in handler.requests)


def test_unreviewed_steps_are_flagged_in_the_report():
    toolset, ctx = _started(FakeHandler(_approve()))
    order = load_order(ctx.state, AGENT)
    order.steps[0].status = "done"
    assert {"code": "unreviewed_steps", "steps": ["S1"]} in report_warnings(order)


# ── Guard and journal ────────────────────────────────────────────────────────

def test_guard_blocks_a_call_with_no_step_in_progress():
    ctx = _ctx()
    toolset = WorkOrderToolset(AGENT, TOOLS, handler=FakeHandler(), step_review=True)
    _declare(toolset, ctx)
    guard = make_work_order_guard(AGENT, handler=FakeHandler())
    blocked = _run(guard(tool=SimpleNamespace(name="resolve_chemicals"), args={}, tool_context=ctx))
    assert blocked["reason"] == "no_active_step"

    _run(toolset.update_work_step("S1", "in_progress", ctx))
    assert _run(guard(tool=SimpleNamespace(name="resolve_chemicals"), args={}, tool_context=ctx)) is None


def test_journal_records_the_call_on_its_step():
    toolset, ctx = _started(FakeHandler(_approve()))
    journal = make_work_step_journal(AGENT)
    response = {
        "content": [{"type": "text", "text": "{...}"}],
        "structuredContent": {"items": [{"input": "1-dodecanol", "canonical_smiles": "CCCCCCCCCCCCO"}]},
        "isError": False,
    }
    journal(tool=SimpleNamespace(name="resolve_chemicals"),
            args={"names": ["1-dodecanol"]}, tool_context=ctx, tool_response=response)

    calls = load_order(ctx.state, AGENT).step("S1").calls
    assert len(calls) == 1
    assert calls[0]["tool"] == "resolve_chemicals"
    assert "1-dodecanol" in calls[0]["args"]
    assert "CCCCCCCCCCCCO" in calls[0]["result_excerpt"]
    assert calls[0]["is_error"] is False


def test_journal_skips_calls_the_guard_blocked():
    toolset, ctx = _started(FakeHandler(_approve()))
    journal = make_work_step_journal(AGENT)
    blocked = {"status": "blocked", "blocked_by": "work_order_guard", "reason": "undeclared_tool"}
    journal(tool=SimpleNamespace(name="resolve_chemicals"), args={}, tool_context=ctx,
            tool_response=blocked)
    assert load_order(ctx.state, AGENT).step("S1").calls == []


def test_excerpt_marks_mcp_errors():
    text, is_error = tool_result_excerpt({
        "content": [{"type": "text", "text": "Error executing tool resolve_chemicals: пустой список имён"}],
        "isError": True,
    })
    assert is_error and "пустой список" in text
    text, is_error = tool_result_excerpt({"error": "MCP tool execution failed: timeout"})
    assert is_error and "timeout" in text


def test_excerpt_is_capped():
    text, _ = tool_result_excerpt({"structuredContent": {"x": "a" * 10000}})
    assert len(text) < 2100


# ── Schema ───────────────────────────────────────────────────────────────────

def test_step_review_needs_a_work_order():
    with pytest.raises(ValueError, match="work_order_step_review"):
        AgentConfig.model_validate({"class": "llm", "hitl": True, "work_order_step_review": True})
    AgentConfig.model_validate(
        {"class": "llm", "hitl": True, "work_order": True, "work_order_step_review": True}
    )
