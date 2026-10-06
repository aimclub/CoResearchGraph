"""The plan is recorded in the call graph, and reaches the human structured.

Two records and one delivery, all of the same plan:

* the execution (call) graph gets a ``decision`` node hanging off the planner's
  activation, carrying the structured plan, opened when the plan is proposed and
  closed with whatever the human answered;
* the HITL request carries that same structure in ``context.experiment_plan``,
  next to the Markdown in ``context.output`` the console still uses.

Before this, a plan left no trace in the call graph at all and reached the
browser only as Markdown, so a plan sent back for revision was recoverable from
nothing but the log.
"""
from __future__ import annotations

import asyncio
import tempfile
import time
from types import SimpleNamespace

import pytest

from CoScientist.experiments import review as review_mod
from CoScientist.experiments.plan_view import plan_to_view
from CoScientist.experiments.review import (
    ExperimentReviewSessionAgent,
    plan_review_identity,
)
from CoScientist.experiments.runtime import execution_bridge
from CoScientist.graph.memory import KnowledgeGraph
from CoScientist.hitl.handler import AbstractHITLHandler
from CoScientist.hitl.models import HITLAction, HITLRequest, HITLResponse

from .helpers import _inventory, _plan, _task

PLANNER = "ExperimentPlannerAgent"


@pytest.fixture()
def call_graph(monkeypatch):
    """An execution graph with one planner activation already on it."""
    graph = KnowledgeGraph(run_id="execution", snapshot_dir=tempfile.mkdtemp())
    graph.ensure_seeded()
    graph.add_node(id="agent:planner@turn-1", kind="agent", turn_id="turn-1",
                   label=PLANNER, executor_agent=PLANNER, status="running",
                   t_start=time.time())
    monkeypatch.setattr(execution_bridge, "_enabled", lambda: True)
    monkeypatch.setattr(execution_bridge, "_graph", lambda _ctx: graph)
    return graph


def _decisions(graph: KnowledgeGraph) -> list[dict]:
    return [n for n in graph.full()["nodes"] if n["kind"] == "decision"]


def _view(plan=None):
    return plan_to_view(plan or _plan(_task("EXP-1")))


def test_a_proposed_plan_becomes_a_decision_under_its_planner(call_graph):
    view = _view()

    node_id = execution_bridge.record_plan_proposed(None, PLANNER, view)

    recorded = _decisions(call_graph)
    assert len(recorded) == 1
    node = recorded[0]
    assert node["id"] == node_id
    assert node["status"] == "running"          # the human has not answered yet
    assert node["turn_id"] == "turn-1"          # the request it belongs to
    assert node["executor_agent"] == PLANNER
    assert node["label"] == "plan rev 1 · 1 task · 1 min"
    assert node["input"]["kind"] == "experiment_plan"
    assert node["input"]["matrix"][0]["task_id"] == "EXP-1"
    assert {"src": "agent:planner@turn-1", "dst": node_id, "type": "caused_by"} \
        in call_graph.full()["edges"]


@pytest.mark.parametrize(
    "outcome,status",
    [("approved", "success"), ("revision_requested", "failed"),
     ("paused", "failed"), ("rejected", "failed")],
)
def test_the_human_answer_closes_the_record(call_graph, outcome, status):
    node_id = execution_bridge.record_plan_proposed(None, PLANNER, _view())

    execution_bridge.close_plan_record(None, node_id, outcome, reason="because")

    node = _decisions(call_graph)[0]
    assert node["status"] == status
    assert node["verdict"] == outcome
    assert node["output"] == "because"
    assert node["t_end"] is not None


def test_an_outcome_nobody_recognises_leaves_the_record_open(call_graph):
    """Better an unfinished record than one stamped with a status we invented."""
    node_id = execution_bridge.record_plan_proposed(None, PLANNER, _view())

    execution_bridge.close_plan_record(None, node_id, "shrugged")

    assert _decisions(call_graph)[0]["status"] == "running"


def test_each_revision_is_a_record_of_its_own(call_graph):
    """Three rounds read as three cards, in the order they happened."""
    first = _view()
    second = _view()
    second["revision"] = 2

    execution_bridge.record_plan_proposed(None, PLANNER, first)
    execution_bridge.record_plan_proposed(None, PLANNER, second)

    assert sorted(n["label"] for n in _decisions(call_graph)) == [
        "plan rev 1 · 1 task · 1 min", "plan rev 2 · 1 task · 1 min",
    ]


def test_nothing_is_recorded_when_the_planner_never_acted(call_graph):
    """A plan node with no activation to hang off would be drawn floating."""
    assert execution_bridge.record_plan_proposed(None, "SomeoneElse", _view()) is None
    assert _decisions(call_graph) == []


def test_the_graph_switch_is_obeyed(call_graph, monkeypatch):
    monkeypatch.setattr(execution_bridge, "_enabled", lambda: False)

    assert execution_bridge.record_plan_proposed(None, PLANNER, _view()) is None
    assert _decisions(call_graph) == []


def test_a_broken_graph_never_reaches_the_review(monkeypatch):
    """Best-effort by contract: a graph failure must not fail an approval."""
    monkeypatch.setattr(execution_bridge, "_enabled", lambda: True)
    monkeypatch.setattr(execution_bridge, "_graph",
                        lambda _ctx: (_ for _ in ()).throw(RuntimeError("no graph")))

    assert execution_bridge.record_plan_proposed(None, PLANNER, _view()) is None
    execution_bridge.close_plan_record(None, "plan:x", "approved")  # does not raise


# ── the review request the human receives ────────────────────────────────────

class _CapturingHandler(AbstractHITLHandler):
    def __init__(self, response: HITLResponse):
        self.requests: list[HITLRequest] = []
        self._response = response

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        self.requests.append(request)
        return self._response


def _review(monkeypatch, response: HITLResponse):
    plan = _plan(_task("EXP-1"))
    handler = _CapturingHandler(response)
    monkeypatch.delenv("COSCIENTIST_EXPERIMENT_HITL_AUTO_APPROVE", raising=False)
    monkeypatch.setattr(review_mod, "_publish_approved_plan_to_graph", lambda *a, **k: None)
    monkeypatch.setattr(review_mod, "record_plan_proposed", lambda *a, **k: "plan-node")
    closed: list[tuple] = []
    monkeypatch.setattr(review_mod, "close_plan_record",
                        lambda ctx, node, outcome, reason=None: closed.append((node, outcome)))
    state = {
        "experiment_context": {
            "experiment_run_id": plan.experiment_run_id,
            "source_request": plan.source_request,
            "available_mcp_capabilities": _inventory(),
        },
    }
    agent = ExperimentReviewSessionAgent(
        name="Reviewer", review_kind="plan", hitl_handler=handler)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-1")
    asyncio.run(agent._review_plan(ctx, plan.model_dump_json()))
    return handler, state, closed


def test_the_review_request_carries_the_structured_plan(monkeypatch):
    handler, state, _ = _review(
        monkeypatch, HITLResponse(action=HITLAction.APPROVE, approved=True))

    assert len(handler.requests) == 1
    context = handler.requests[0].context
    plan_view = context["experiment_plan"]
    assert plan_view["kind"] == "experiment_plan"
    assert plan_view["task_count"] == 1
    assert plan_view["matrix"][0]["task_id"] == "EXP-1"
    assert plan_view["critique"]["verdict"] == "approve"
    assert context["experiment_review_id"] == state["experiment_plan_review_id"]
    assert context["experiment_review_id"] == plan_review_identity(
        state["experiment_runtime"]["plan"]
    )[0]
    # The Markdown stays: the console and any other client still read it — and
    # it is now written in the session's language, which defaults to Russian.
    # (This copy becomes the document the chat panel opens, so an English frame
    # around Russian content was what a reader actually met.)
    assert context["output"].index("## Требования") < context["output"].index("## EXP-1")
    assert "Design matrix" not in context["output"]
    # And the view is on the state the module publishes to its caller.
    assert state["experiment_plan_view"]["plan_id"] == plan_view["plan_id"]


def test_standard_plan_review_id_replays_stably_and_changes_with_revision(monkeypatch):
    response = HITLResponse(action=HITLAction.APPROVE, approved=True)
    first, _, _ = _review(monkeypatch, response)
    replayed, _, _ = _review(monkeypatch, response)
    first_id = first.requests[0].context["experiment_review_id"]
    assert replayed.requests[0].context["experiment_review_id"] == first_id

    plan = _plan(_task("EXP-1"))
    revised = plan.model_copy(update={"revision": plan.revision + 1})
    assert plan_review_identity(revised)[0] != first_id


def test_the_record_is_closed_with_what_the_human_did(monkeypatch):
    for response, expected in (
        (HITLResponse(action=HITLAction.APPROVE, approved=True), "approved"),
        (HITLResponse(action=HITLAction.EDIT, approved=False), "revision_requested"),
        (HITLResponse(action=HITLAction.REJECT, approved=False), "rejected"),
        (HITLResponse(action=HITLAction.REJECT, approved=False, timed_out=True), "paused"),
    ):
        _, state, closed = _review(monkeypatch, response)
        assert closed == [("plan-node", expected)]
        assert state["experiment_plan_view"]["status"] == expected
