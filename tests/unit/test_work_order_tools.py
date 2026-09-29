"""Work Order tools: how a contract is put before the human, and what the agent
is told back."""
import asyncio
from types import SimpleNamespace

import pytest

from CoScientist.config import get_settings
from CoScientist.hitl.models import HITLAction, HITLResponse
from CoScientist.hitl.work_order import load_order
from CoScientist.hitl.work_order_tools import WorkOrderToolset

AGENT = "DatasetCollectorAgent"
TOOLS = ["tavily_search", "execute_bash", "read_file", "install_package", "research_commit"]


class _Handler:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.notices = []

    async def handle_request(self, request):
        self.requests.append(request)
        return self.responses.pop(0) if self.responses else HITLResponse(
            action=HITLAction.APPROVE, approved=True
        )

    async def notify(self, payload):
        self.notices.append(payload)


def _context(state=None):
    state = {} if state is None else state
    session = SimpleNamespace(id="s1", user_id="u1", state=state)
    return SimpleNamespace(state=state, session=session,
                           _invocation_context=SimpleNamespace(session=session))


@pytest.fixture
def hitl_on(monkeypatch):
    web = get_settings().web
    monkeypatch.setattr(web, "hitl_enabled", True)
    monkeypatch.setattr(web, "work_order_enabled", True)
    monkeypatch.setattr(web, "work_order_veto_seconds", 30)
    monkeypatch.setattr(web, "work_order_max_amendments", 3)
    return web


def _declare(toolset, ctx, **overrides):
    args = dict(
        goal="Collect BTK inhibitors with IC50 from ChEMBL",
        done_criteria="A CSV with >= 500 unique compounds",
        assumptions=["IC50 in nM", "Only human BTK (CHEMBL5251)"],
        steps=[
            {"title": "Download activities", "tools": ["execute_bash"],
             "expected_outcome": "~2k rows"},
            {"title": "Deduplicate", "tools": ["execute_bash"]},
        ],
        planned_tools=["execute_bash"],
        tool_context=ctx,
    )
    args.update(overrides)
    return asyncio.run(toolset.declare_work_order(**args))


def test_switched_off_approves_without_asking_but_still_records_the_order(monkeypatch):
    monkeypatch.setattr(get_settings().web, "hitl_enabled", False)
    handler = _Handler()
    ctx = _context()
    result = _declare(WorkOrderToolset(AGENT, TOOLS, handler), ctx)

    assert result["status"] == "approved"
    assert result["mode"] == "autonomous"
    assert handler.requests == [] and handler.notices == []
    assert load_order(ctx.state, AGENT).status == "approved"


def test_read_tier_declaration_still_asks_under_the_veto_window(hitl_on):
    handler = _Handler()
    ctx = _context()
    result = _declare(WorkOrderToolset(AGENT, TOOLS, handler), ctx,
                      steps=[{"title": "Search", "tools": ["tavily_search"]}],
                      planned_tools=["tavily_search"])

    assert result["status"] == "approved"
    assert handler.notices == []
    request = handler.requests[0]
    assert request.trigger == "work_order"
    assert request.timeout_seconds is None
    assert request.context["tier"] == "read"
    assert request.context["_session"] == {"user_id": "u1", "session_id": "s1"}


def test_read_tier_declaration_without_veto_window_defers_to_the_run_mode(hitl_on, monkeypatch):
    """No configured veto window means the WAIT is not this module's business:
    `None` hands it to the run's HITL mode. It used to pass an explicit -1 here
    and fall through to the global for side effects, which is how the tiers came
    out inverted — read and compute waited for the human forever while the
    riskiest tier was the only one a clock could sign off."""
    monkeypatch.setattr(hitl_on, "work_order_veto_seconds", -1)
    handler = _Handler()
    _declare(WorkOrderToolset(AGENT, TOOLS, handler), _context(),
             steps=[{"title": "Search", "tools": ["tavily_search"]}],
             planned_tools=["tavily_search"])

    assert handler.requests[0].timeout_seconds is None


def test_compute_tier_gets_the_veto_window(hitl_on):
    handler = _Handler()
    _declare(WorkOrderToolset(AGENT, TOOLS, handler), _context())

    request = handler.requests[0]
    assert request.trigger == "work_order"
    assert request.timeout_seconds is None
    assert request.context["tier"] == "compute"
    order = request.context["work_order"]
    assert [a["id"] for a in order["assumptions"]] == ["A1", "A2"]
    assert order["assumptions"][0] == {
        "id": "A1", "text": "IC50 in nM", "rejected": False,
    }
    assert [s["id"] for s in order["steps"]] == ["S1", "S2"]


def test_every_tier_waits_the_same_way_when_no_veto_window_is_set(hitl_on, monkeypatch):
    """The wait is one property of the run, not three properties of the tiers.
    With no configured veto window all three pass `None` and the HITL mode
    decides — which is what un-inverted them: `side_effect` was the only tier
    with a countdown, and that countdown approved."""
    monkeypatch.setattr(hitl_on, "work_order_veto_seconds", -1)
    windows = {}
    for tier, tools in (("read", ["tavily_search"]),
                        ("compute", ["execute_bash"]),
                        ("side_effect", ["execute_bash", "install_package"])):
        handler = _Handler()
        _declare(WorkOrderToolset(AGENT, TOOLS, handler), _context(),
                 steps=[{"title": "Step", "tools": tools}],
                 planned_tools=tools)
        request = handler.requests[0]
        windows[request.context["tier"]] = request.timeout_seconds

    assert set(windows) == {"read", "compute", "side_effect"}, windows
    assert set(windows.values()) == {None}, windows


def test_side_effect_tier_blocks_under_the_global_timeout(hitl_on):
    handler = _Handler()
    ctx = _context()
    _declare(WorkOrderToolset(AGENT, TOOLS, handler), ctx,
             planned_tools=["execute_bash", "install_package"])

    request = handler.requests[0]
    assert request.timeout_seconds is None
    assert request.context["tier"] == "side_effect"
    # install_package declares its own side effect.
    assert request.context["work_order"]["side_effects"][0]["kind"] == "package_install"


def test_approval_carries_rejected_assumptions_and_operator_notes(hitl_on):
    handler = _Handler(HITLResponse(
        action=HITLAction.APPROVE, approved=True,
        instructions="Use only pChEMBL values",
        form_values={"rejected_assumption_ids": ["A2"]},
    ))
    ctx = _context()
    result = _declare(WorkOrderToolset(AGENT, TOOLS, handler), ctx)

    assert result["status"] == "approved"
    assert result["rejected_assumptions"] == ["Only human BTK (CHEMBL5251)"]
    assert result["operator_notes"] == "Use only pChEMBL values"
    order = load_order(ctx.state, AGENT)
    assert [a.rejected for a in order.assumptions] == [False, True]


def test_revise_asks_for_a_new_declaration_and_records_nothing(hitl_on):
    handler = _Handler(HITLResponse(action=HITLAction.EDIT, approved=False,
                                    instructions="Add PubChem as a second source"))
    ctx = _context()
    result = _declare(WorkOrderToolset(AGENT, TOOLS, handler), ctx)

    assert result["status"] == "revise"
    assert result["feedback"] == "Add PubChem as a second source"
    assert load_order(ctx.state, AGENT) is None


def test_a_rejected_assumption_reaches_the_agent_when_notes_send_it_back(hitl_on):
    """Unticking an assumption and saying why is ONE act, not two.

    The card's single approval button becomes "revise" the moment a note is
    typed, and the untick used to be attached only to the approve action — so
    the ordinary move of refusing a premise and explaining it sent the agent
    back to the drawing board without telling it which premise had been refused.
    """
    handler = _Handler(HITLResponse(
        action=HITLAction.EDIT, approved=False,
        instructions="This assumption does not hold for mouse data",
        form_values={"rejected_assumption_ids": ["A2"]},
    ))
    ctx = _context()
    result = _declare(WorkOrderToolset(AGENT, TOOLS, handler), ctx)

    assert result["status"] == "revise"
    assert result["rejected_assumptions"] == ["Only human BTK (CHEMBL5251)"]
    assert "REJECTED" in result["message"]


def test_reject_records_a_rejected_order_that_cannot_be_redeclared(hitl_on):
    handler = _Handler(HITLResponse(action=HITLAction.REJECT, approved=False,
                                    instructions="Not needed"))
    ctx = _context()
    toolset = WorkOrderToolset(AGENT, TOOLS, handler)
    result = _declare(toolset, ctx)

    assert result["status"] == "rejected"
    assert load_order(ctx.state, AGENT).status == "rejected"
    assert _declare(toolset, ctx)["status"] == "rejected"
    assert len(handler.requests) == 1


def test_unknown_tools_are_refused(hitl_on):
    toolset = WorkOrderToolset(AGENT, TOOLS, _Handler())
    unknown = _declare(toolset, _context(), planned_tools=["execute_bash", "git_push"])
    assert unknown["status"] == "error" and "git_push" in unknown["message"]


def test_step_tools_join_the_planned_tools(hitl_on):
    ctx = _context()
    _declare(WorkOrderToolset(AGENT, TOOLS, _Handler()), ctx,
             steps=[{"title": "Read", "tools": ["read_file"]}],
             planned_tools=["execute_bash"])
    assert load_order(ctx.state, AGENT).planned_tools == ["execute_bash", "read_file"]


def test_an_approved_order_is_amended_not_redeclared(hitl_on):
    ctx = _context()
    toolset = WorkOrderToolset(AGENT, TOOLS, _Handler())
    _declare(toolset, ctx)
    again = _declare(toolset, ctx)
    assert again["status"] == "error" and "update_work_order" in again["message"]


def test_amendment_is_reviewed_by_the_tier_of_what_it_adds(hitl_on):
    handler = _Handler()
    ctx = _context()
    toolset = WorkOrderToolset(AGENT, TOOLS, handler)
    _declare(toolset, ctx)

    # Adding a read tool: a notice, no question.
    result = asyncio.run(toolset.update_work_order(
        reason="Need the ChEMBL target id", add_tools=["tavily_search"],
        tool_context=ctx,
    ))
    assert result == {"status": "approved", "revision": 2,
                      "message": "Amendment approved. Proceed within the updated work order."}
    assert handler.notices[-1]["kind"] == "amended"
    assert handler.notices[-1]["diff"]["added_tools"] == ["tavily_search"]

    # Adding a side effect: a blocking review showing the diff and the reason.
    asyncio.run(toolset.update_work_order(
        reason="rdkit is missing", add_tools=["install_package"], tool_context=ctx,
    ))
    request = handler.requests[-1]
    assert request.trigger == "work_order_amendment"
    assert request.timeout_seconds is None
    assert request.context["reason"] == "rdkit is missing"
    assert request.context["diff"]["added_side_effects"][0]["kind"] == "package_install"

    order = load_order(ctx.state, AGENT)
    assert order.revision == 3
    assert [a["revision"] for a in order.amendments] == [2, 3]


def test_amendments_past_the_limit_always_block(hitl_on, monkeypatch):
    monkeypatch.setattr(hitl_on, "work_order_max_amendments", 1)
    handler = _Handler()
    ctx = _context()
    toolset = WorkOrderToolset(AGENT, TOOLS, handler)
    _declare(toolset, ctx)

    asyncio.run(toolset.update_work_order(reason="r1", add_tools=["tavily_search"], tool_context=ctx))
    assert handler.requests[1:] == []  # read-tier delta: a notice
    asyncio.run(toolset.update_work_order(reason="r2", add_tools=["read_file"], tool_context=ctx))
    assert handler.requests[-1].trigger == "work_order_amendment"
    assert handler.requests[-1].timeout_seconds is None


def test_rejected_amendment_keeps_the_current_order(hitl_on):
    handler = _Handler(
        HITLResponse(action=HITLAction.APPROVE, approved=True),
        HITLResponse(action=HITLAction.REJECT, approved=False, instructions="No installs"),
    )
    ctx = _context()
    toolset = WorkOrderToolset(AGENT, TOOLS, handler)
    _declare(toolset, ctx)
    result = asyncio.run(toolset.update_work_order(
        reason="need rdkit", add_tools=["install_package"], tool_context=ctx,
    ))
    assert result["status"] == "rejected"
    order = load_order(ctx.state, AGENT)
    assert order.status == "approved" and order.revision == 1
    assert "install_package" not in order.planned_tools


def test_empty_amendment_is_refused(hitl_on):
    ctx = _context()
    toolset = WorkOrderToolset(AGENT, TOOLS, _Handler())
    _declare(toolset, ctx)
    result = asyncio.run(toolset.update_work_order(
        reason="nothing", add_tools=["execute_bash"], tool_context=ctx,
    ))
    assert result["status"] == "error"


def test_step_progress_is_saved_and_announced(hitl_on):
    handler = _Handler()
    ctx = _context()
    toolset = WorkOrderToolset(AGENT, TOOLS, handler)
    _declare(toolset, ctx)

    result = asyncio.run(toolset.update_work_step("S1", "done", tool_context=ctx, note="2143 rows"))
    assert result == {"status": "ok", "step": "S1", "step_status": "done"}
    assert load_order(ctx.state, AGENT).step("S1").note == "2143 rows"
    assert handler.notices[-1]["kind"] == "progress"
    assert handler.notices[-1]["step"]["status"] == "done"

    assert asyncio.run(toolset.update_work_step("S9", "done", tool_context=ctx))["status"] == "error"
    assert asyncio.run(toolset.update_work_step("S1", "finished", tool_context=ctx))["status"] == "error"


def test_tools_build_valid_function_declarations():
    names = []
    for tool in WorkOrderToolset(AGENT, TOOLS).tools():
        assert tool._get_declaration() is not None
        names.append(tool.name)
    # Four, since #363: declaring opens the contract, the report closes it.
    assert names == [
        "declare_work_order",
        "update_work_order",
        "update_work_step",
        "submit_work_report",
    ]


def test_assumptions_are_strings_and_the_older_dict_shape_is_tolerated(monkeypatch):
    monkeypatch.setattr(get_settings().web, "hitl_enabled", False)
    ctx = _context()
    toolset = WorkOrderToolset(AGENT, TOOLS, _Handler())

    result = _declare(toolset, ctx, assumptions=["  IC50 in nM ", {"text": "Only human BTK"}])
    assert result["status"] == "approved"
    order = load_order(ctx.state, AGENT)
    assert [(a.id, a.text) for a in order.assumptions] == [("A1", "IC50 in nM"), ("A2", "Only human BTK")]


def test_empty_assumption_is_an_error(monkeypatch):
    monkeypatch.setattr(get_settings().web, "hitl_enabled", False)
    toolset = WorkOrderToolset(AGENT, TOOLS, _Handler())

    for bad in ("  ", {"text": "  "}, {}):
        result = _declare(toolset, _context(), assumptions=[bad])
        assert result["status"] == "error"
        assert "non-empty string" in result["message"]


def test_internal_tools_may_be_named_but_stay_apart_from_the_contract(hitl_on):
    handler = _Handler()
    ctx = _context()
    toolset = WorkOrderToolset(AGENT, TOOLS, handler, internal_tools=["sleep_tool"])
    result = _declare(toolset, ctx,
                      steps=[{"title": "Wait for the job", "tools": ["execute_bash", "sleep_tool"]}],
                      planned_tools=["execute_bash", "sleep_tool"])

    assert result["status"] == "approved"
    card = handler.requests[0].context["work_order"]
    assert card["planned_tools"] == ["execute_bash"]
    assert card["steps"][0]["tools"] == ["execute_bash"]
    # Kept separately, so the web card can show them on request.
    assert card["internal_tools"] == ["sleep_tool"]
    assert card["steps"][0]["internal_tools"] == ["sleep_tool"]
    assert card["tier"] == "compute"
    assert "sleep_tool" not in handler.requests[0].context["output"]


def test_amending_only_internal_tools_needs_no_review(hitl_on):
    handler = _Handler()
    ctx = _context()
    toolset = WorkOrderToolset(AGENT, TOOLS, handler, internal_tools=["sleep_tool"])
    _declare(toolset, ctx)
    asked = len(handler.requests)

    result = asyncio.run(toolset.update_work_order(
        reason="the job is slow", add_tools=["sleep_tool"], tool_context=ctx,
    ))

    assert result["status"] == "approved"
    assert len(handler.requests) == asked
    order = load_order(ctx.state, AGENT)
    assert order.revision == 1
    assert order.internal_tools == ["sleep_tool"]
