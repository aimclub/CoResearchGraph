"""The Work Report: what the agent claims it did, set against what the system
recorded.

#363 added the whole protocol — the report, the warnings that compare it with
the journal, the after-agent fallback — and deleted the two Work Order test
suites in the same commit. Those are restored next door; this covers the half
that arrived with no tests at all.

The warnings are the point of the card. An agent grading its own homework is
worth little; an agent grading its own homework beside the record of what it
actually called, which steps it left open and where it went off contract is
worth a great deal.
"""
import asyncio
from types import SimpleNamespace

import pytest

from CoScientist.config import get_settings
from CoScientist.hitl.models import HITLAction, HITLResponse
from CoScientist.hitl.work_order import (
    Finding,
    WorkOrder,
    WorkReport,
    WorkStep,
    load_order,
    performed_side_effects,
    report_warnings,
)
from CoScientist.hitl.work_order_tools import WorkOrderToolset

AGENT = "DatasetCollectorAgent"
TOOLS = ["tavily_search", "execute_bash", "read_file", "install_package"]


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
    return web


def _order(**overrides) -> WorkOrder:
    data = dict(
        agent=AGENT,
        goal="Собрать ингибиторы BTK с IC50",
        done_criteria="CSV с >= 500 уникальными соединениями",
        steps=[WorkStep(id="S1", title="Скачать активности", status="done")],
        planned_tools=["execute_bash"],
        status="approved",
    )
    data.update(overrides)
    return WorkOrder(**data)


def _codes(order: WorkOrder) -> set:
    return {w["code"] for w in report_warnings(order)}


# ── what the human is asked to look at first ────────────────────────────────

def test_a_finding_with_nothing_behind_it_is_flagged():
    order = _order(report=WorkReport(
        summary="Собрано 812 соединений.",
        done_verdict="met",
        findings=[
            Finding(id="F1", text="IC50 распределён логнормально", evidence="figures/ic50.png"),
            Finding(id="F2", text="Селективность выше у производных пиразола"),
        ],
    ))
    warnings = {w["code"]: w for w in report_warnings(order)}

    assert "findings_without_evidence" in warnings
    assert warnings["findings_without_evidence"]["findings"] == ["F2"]


def test_steps_left_open_and_a_criterion_not_met_are_flagged():
    order = _order(
        steps=[
            WorkStep(id="S1", title="Скачать", status="done"),
            WorkStep(id="S2", title="Дедуплицировать", status="in_progress"),
            WorkStep(id="S3", title="Проверить", status="pending"),
        ],
        report=WorkReport(summary="Частично.", done_verdict="partial"),
    )
    warnings = {w["code"]: w for w in report_warnings(order)}

    assert warnings["open_steps"]["steps"] == ["S2", "S3"]
    assert warnings["done_not_met"]["verdict"] == "partial"


def test_going_off_contract_stays_on_the_card():
    order = _order(
        report=WorkReport(summary="Готово.", done_verdict="met"),
        deviations=[{"reason": "undeclared_tool", "tool": "install_package"}],
    )
    assert "deviations" in _codes(order)


def test_an_agent_that_never_reported_is_flagged_loudest():
    """The fallback builds a card from the order alone.

    Nothing in it is the agent's claim, so grading that claim would be
    meaningless: only the absence of a report is worth saying.
    """
    order = _order(report=WorkReport(fallback=True, done_verdict="not_met"))
    codes = _codes(order)

    assert "no_report" in codes
    assert "done_not_met" not in codes
    assert "findings_without_evidence" not in codes


def test_side_effects_are_counted_from_what_ran_not_from_what_was_promised():
    """A contract that promised nothing irreversible is not evidence that
    nothing irreversible happened. The journal is."""
    order = _order(tool_calls={"execute_bash": 5, "install_package": 2})

    assert performed_side_effects(order) == ["package_install"]
    assert performed_side_effects(_order(tool_calls={"execute_bash": 5})) == []


# ── the protocol ────────────────────────────────────────────────────────────

def _declare(toolset, ctx):
    return asyncio.run(toolset.declare_work_order(
        goal="Собрать ингибиторы BTK с IC50",
        done_criteria="CSV с >= 500 уникальными соединениями",
        assumptions=["IC50 в нМ"],
        steps=[{"title": "Скачать активности", "tools": ["execute_bash"]}],
        planned_tools=["execute_bash"],
        tool_context=ctx,
    ))


def _submit(toolset, ctx, **overrides):
    args = dict(
        summary="Собрано 812 соединений из ChEMBL.",
        findings=[{"text": "812 уникальных структур", "evidence": "data/btk.csv"}],
        done_verdict="met",
        done_evidence="812 >= 500",
        actual_outcome="812 строк, 812 уникальных InChIKey",
        artifacts=[{"kind": "file", "ref": "data/btk.csv"}],
        tool_context=ctx,
    )
    args.update(overrides)
    return asyncio.run(toolset.submit_work_report(**args))


def test_there_is_nothing_to_report_on_without_an_approved_order(hitl_on):
    ctx = _context()
    toolset = WorkOrderToolset(AGENT, TOOLS, _Handler())

    result = _submit(toolset, ctx)

    assert result["status"] == "error"
    assert "declare_work_order" in result["message"]


def test_an_accepted_report_closes_the_contract(hitl_on):
    ctx = _context()
    toolset = WorkOrderToolset(AGENT, TOOLS, _Handler())
    _declare(toolset, ctx)

    result = _submit(toolset, ctx)

    assert result["status"] == "accepted"
    order = load_order(ctx.state, AGENT)
    assert order.report.status == "accepted"
    assert order.report.round == 1


def test_a_doubted_finding_survives_an_accepted_report(hitl_on):
    """Accepting a result and doubting one of its findings are two acts.

    Only the rework path read these marks, so a finding ticked as wrong on a
    report the operator otherwise accepted was recorded nowhere and mentioned to
    nobody — and the agent carried it into the final answer as established.
    """
    ctx = _context()
    handler = _Handler(
        HITLResponse(action=HITLAction.APPROVE, approved=True),
        HITLResponse(action=HITLAction.APPROVE, approved=True,
                     form_values={"disputed_finding_ids": ["F2"]}),
    )
    toolset = WorkOrderToolset(AGENT, TOOLS, handler)
    _declare(toolset, ctx)

    result = _submit(toolset, ctx, findings=[
        {"text": "812 уникальных структур", "evidence": "data/btk.csv"},
        {"text": "Селективность выше у пиразолов"},
    ])

    assert result["status"] == "accepted"
    assert [f["id"] for f in result["disputed_findings"]] == ["F2"]
    assert "do not carry them" in result["message"]
    assert load_order(ctx.state, AGENT).report.disputed_finding_ids == ["F2"]


def test_a_report_sent_back_names_the_findings_the_human_disputes(hitl_on):
    ctx = _context()
    # The first answer approves the DECLARATION; the second judges the report.
    handler = _Handler(
        HITLResponse(action=HITLAction.APPROVE, approved=True),
        HITLResponse(
            action=HITLAction.EDIT,
            instructions="F2 не подтверждается данными — перепроверь.",
            form_values={"disputed_finding_ids": ["F2"]},
        ),
    )
    toolset = WorkOrderToolset(AGENT, TOOLS, handler)
    _declare(toolset, ctx)

    result = _submit(toolset, ctx, findings=[
        {"text": "812 уникальных структур", "evidence": "data/btk.csv"},
        {"text": "Селективность выше у пиразолов"},
    ])

    assert result["status"] == "revise"
    order = load_order(ctx.state, AGENT)
    assert order.report.status == "revise"
    assert order.report.disputed_finding_ids == ["F2"]
    assert "перепроверь" in order.report.operator_notes
    # The round is kept, so a second submission is visibly a second attempt.
    assert len(order.reports) == 1


def test_a_second_round_is_numbered_as_one(hitl_on):
    ctx = _context()
    handler = _Handler(
        HITLResponse(action=HITLAction.APPROVE, approved=True),
        HITLResponse(action=HITLAction.EDIT, instructions="ещё раз"),
    )
    toolset = WorkOrderToolset(AGENT, TOOLS, handler)
    _declare(toolset, ctx)
    _submit(toolset, ctx)

    _submit(toolset, ctx, summary="Перепроверено: 812 соединений.")

    order = load_order(ctx.state, AGENT)
    assert order.report.round == 2
    assert order.report.status == "accepted"


def test_a_rejected_report_is_the_end_of_it(hitl_on):
    ctx = _context()
    handler = _Handler(
        HITLResponse(action=HITLAction.APPROVE, approved=True),
        HITLResponse(action=HITLAction.REJECT, approved=False,
                     instructions="Результат не принят."),
    )
    toolset = WorkOrderToolset(AGENT, TOOLS, handler)
    _declare(toolset, ctx)

    first = _submit(toolset, ctx)
    second = _submit(toolset, ctx)

    assert first["status"] == "rejected"
    assert second["status"] == "error"
    assert "already rejected" in second["message"]


def test_the_card_carries_the_record_next_to_the_claim(hitl_on):
    """Everything the human needs to judge the claim travels with it: what was
    promised, what was called, where it went off contract, and what the system
    thinks is worth a second look."""
    ctx = _context()
    handler = _Handler()
    toolset = WorkOrderToolset(AGENT, TOOLS, handler)
    _declare(toolset, ctx)
    order = load_order(ctx.state, AGENT)
    order.tool_calls = {"execute_bash": 4, "install_package": 1}
    order.deviations = [{"reason": "undeclared_tool", "tool": "install_package"}]
    from CoScientist.hitl.work_order import save_order

    save_order(ctx.state, order)

    _submit(toolset, ctx, findings=[{"text": "без доказательства"}])

    request = handler.requests[-1]
    assert request.trigger == "work_report"
    context = request.context
    assert context["work_order"]["goal"].startswith("Собрать")
    assert context["work_report"]["summary"]
    assert context["journal"]["tool_calls"] == {"execute_bash": 4, "install_package": 1}
    assert context["journal"]["side_effects"] == ["package_install"]
    assert {w["code"] for w in context["warnings"]} >= {
        "findings_without_evidence", "deviations", "open_steps"}
    # The console and any other client read the same report — as Markdown now,
    # and in the session's language, because this text is what the chat writes
    # into the document a person opens.
    assert context["output"].startswith("# Отчёт о работе — DatasetCollectorAgent")
    assert "\n## Цель" in context["output"]


def test_with_work_orders_off_the_report_is_not_put_to_anyone(monkeypatch):
    web = get_settings().web
    monkeypatch.setattr(web, "hitl_enabled", True)
    monkeypatch.setattr(web, "work_order_enabled", False)
    ctx = _context()
    handler = _Handler()
    toolset = WorkOrderToolset(AGENT, TOOLS, handler)
    _declare(toolset, ctx)

    result = _submit(toolset, ctx)

    assert result["status"] == "accepted"
    assert handler.requests == []
