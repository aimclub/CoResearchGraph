"""Plan identity is runtime bookkeeping, and the revision budget counts failures.

Observed 2026-09-04 on a live Heracleum session: a human edited the plan through
HITL, the regenerated plan came back as ``PLAN-EXRUN-<uuid>`` where the runtime
held ``PLAN-<uuid>``, the deterministic critique blocked it as "plan_id changed
between revisions", the two-round budget ran out and the module returned
"Experiment plan review is paused for this session" — with zero MCP calls, while
every tool it needed was live and indexed.

Two things were wrong. The model was being asked for fields only the runtime can
know, and the budget was never reset on success, so it bounded a whole session
rather than a run of consecutive failures.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments import review as review_mod
from CoScientist.experiments.critique.validator import validate_and_critique_plan
from CoScientist.experiments.review import (
    ExperimentReviewSessionAgent,
    _stamp_context_invariants,
)
from CoScientist.hitl.models import (
    HITLAction,
    HITLDecisionSource,
    HITLResponse,
)

from .helpers import _inventory, _plan, _task


def _context(plan) -> dict:
    return {
        "experiment_run_id": plan.experiment_run_id,
        "source_request": plan.source_request,
        "available_mcp_capabilities": _inventory(),
    }


# --------------------------------------------------------------------------
# plan identity


def test_plan_identity_is_stamped_from_the_runtime():
    previous = _plan(_task("EXP-1"))
    payload = previous.model_dump(mode="json")
    payload["plan_id"] = "PLAN-EXRUN-invented-by-the-model"
    payload["revision"] = 1

    stamped = _stamp_context_invariants(payload, _context(previous), previous)

    assert stamped["plan_id"] == previous.plan_id
    assert stamped["revision"] == previous.revision + 1


def test_a_first_plan_keeps_the_identity_it_came_with():
    plan = _plan(_task("EXP-1"))
    payload = plan.model_dump(mode="json")

    stamped = _stamp_context_invariants(payload, _context(plan), None)

    assert stamped["plan_id"] == plan.plan_id
    assert stamped["revision"] == plan.revision


@pytest.mark.parametrize(
    ("plan_id", "revision"),
    [
        ("PLAN-EXRUN-acceptance", 2),   # prefix drift, revision fine
        ("PLAN-acceptance", 1),         # identity fine, revision not incremented
        ("PLAN-EXRUN-acceptance", 1),   # both wrong
    ],
)
def test_stamping_defuses_both_bookkeeping_blockers(plan_id, revision):
    """Neither validator check can fire on a model formatting slip any more."""
    previous = _plan(_task("EXP-1"))
    payload = previous.model_dump(mode="json")
    payload["plan_id"] = plan_id
    payload["revision"] = revision

    stamped = _stamp_context_invariants(payload, _context(previous), previous)
    _, critique = validate_and_critique_plan(
        stamped, settings=ExperimentsSettings(), available_tools=_inventory(),
        previous_plan=previous,
    )

    bookkeeping = [
        i for i in critique.issues
        if "plan_id changed" in i.message or "must increment revision" in i.message
    ]
    assert bookkeeping == [], [i.message for i in bookkeeping]


def test_the_validator_still_guards_identity_when_nothing_stamped_it():
    """The checks stay meaningful: they just stop punishing the model."""
    previous = _plan(_task("EXP-1"))
    payload = previous.model_dump(mode="json")
    payload["plan_id"] = "PLAN-EXRUN-acceptance"

    _, critique = validate_and_critique_plan(
        payload, settings=ExperimentsSettings(), available_tools=_inventory(),
        previous_plan=previous,
    )

    assert any("plan_id changed" in i.message for i in critique.issues)


# --------------------------------------------------------------------------
# budget


def test_a_plan_that_validates_resets_the_revision_budget(monkeypatch):
    """The budget bounds CONSECUTIVE failures, not the lifetime of a session."""
    plan = _plan(_task("EXP-1"))
    state = {
        "experiment_context": _context(plan),
        "experiment_plan_revision_count": 3,
        "experiment_inventory_blocker_hits": 1,
    }
    monkeypatch.setenv("HITL__MODE", "auto")
    # approve_plan resets the counters too; stub it so only the new reset can pass.
    monkeypatch.setattr(review_mod, "approve_plan", lambda _state: None)
    monkeypatch.setattr(review_mod, "_publish_approved_plan_to_graph", lambda *_a, **_k: None)

    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-1")
    response = asyncio.run(agent._review_plan(ctx, plan.model_dump_json()))

    assert response.approved, state.get("experiment_plan_validation_errors")
    assert state["experiment_plan_revision_count"] == 0
    assert state["experiment_inventory_blocker_hits"] == 0


def test_a_replan_starts_from_a_full_revision_budget():
    """A rejected result review opens a new planning round, not a spent one."""
    from CoScientist.experiments.runtime import mark_result_review

    from .helpers import _approved_state

    state = _approved_state(_plan(_task("EXP-1")))
    state["experiment_runtime"]["phase"] = "reporting"
    state["experiment_plan_revision_count"] = 3
    state["experiment_inventory_blocker_hits"] = 2

    out = mark_result_review(state, approved=False, feedback="metrics missing")

    assert out["phase"] == "replan_requested"
    assert state["experiment_plan_revision_count"] == 0
    assert state["experiment_inventory_blocker_hits"] == 0


def test_the_budget_leaves_room_for_a_human_round():
    """Two rounds could not absorb one HITL edit plus one planner slip."""
    assert ExperimentsSettings().max_plan_revisions >= 4


def test_exhausted_revisions_send_the_current_executable_plan_to_a_human(monkeypatch):
    task = _task("EXP-1", route="react_tools")
    plan = _plan(task)
    state = {
        "experiment_context": {
            **_context(plan),
            "operations": [
                {"operation_id": "OP-1", "statement": "Prepare input data"},
                {"operation_id": "OP-2", "statement": "Run the computation"},
            ],
        },
        "experiment_plan_revision_count": ExperimentsSettings().max_plan_revisions - 1,
    }
    asked = []

    async def approve(request):
        asked.append(request)
        return HITLResponse(
            action=HITLAction.APPROVE,
            approved=True,
            decision_source=HITLDecisionSource.HUMAN,
        )

    monkeypatch.setattr(review_mod, "record_plan_proposed", lambda *_a, **_k: "PR-1")
    monkeypatch.setattr(review_mod, "close_plan_record", lambda *_a, **_k: None)
    monkeypatch.setattr(review_mod, "_publish_approved_plan_to_graph", lambda *_a, **_k: None)
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=approve)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-1")

    response = asyncio.run(agent._review_plan(ctx, plan.model_dump_json()))

    assert response.approved and len(asked) == 1
    assert asked[0].requires_human is True
    assert asked[0].context["experiment_review_id"].endswith(
        state["experiment_plan_candidate"]["digest"]
    )
    assert asked[0].context["experiment_plan"]["review_exhausted"] is True
    assert state["experiment_runtime"]["phase"] == "execution"
    assert state["experiment_runtime"]["approved"] is True
    assert state["experiment_runtime"]["critique"]["verdict"] == "revise"
    assert state["experiment_runtime"]["human_critique_override"]["decision_source"] == "human"
    assert state["experiment_plan_candidate"]["status"] == "approved_with_issues"
    assert state["experiment_plan_fallback_pending"] is False


def test_exhausted_revisions_do_not_offer_human_override_for_a_dead_route(monkeypatch):
    plan = _plan(_task("EXP-1", route="fedot_mas"))
    state = {
        "experiment_context": _context(plan),
        "experiment_plan_revision_count": ExperimentsSettings().max_plan_revisions - 1,
    }
    asked = []
    monkeypatch.setattr(review_mod, "fedot_route_available", lambda *_a, **_k: False)

    async def should_not_ask(request):
        asked.append(request)
        raise AssertionError("an unavailable route must not be offered for approval")

    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=should_not_ask)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-1")

    response = asyncio.run(agent._review_plan(ctx, plan.model_dump_json()))

    assert not response.approved and response.stop_review_loop
    assert asked == []
    assert state["experiment_plan_review_paused"] is True
    assert state["experiment_plan_candidate"]["status"] == "blocked"
    assert state["experiment_plan_candidate"]["readiness"]["execution_blockers"]
    assert state.get("experiment_runtime") is None


def test_a_final_schema_failure_offers_the_last_schema_valid_candidate(monkeypatch):
    plan = _plan(_task("EXP-1", route="react_tools"))
    state = {
        "experiment_context": {
            **_context(plan),
            "operations": [
                {"operation_id": "OP-1", "statement": "Prepare input data"},
                {"operation_id": "OP-2", "statement": "Run the computation"},
            ],
        },
        "experiment_plan_revision_count": ExperimentsSettings().max_plan_revisions - 2,
    }
    asked = []

    async def approve(request):
        asked.append(request)
        return HITLResponse(
            action=HITLAction.APPROVE,
            approved=True,
            decision_source=HITLDecisionSource.HUMAN,
        )

    monkeypatch.setattr(review_mod, "record_plan_proposed", lambda *_a, **_k: "PR-1")
    monkeypatch.setattr(review_mod, "close_plan_record", lambda *_a, **_k: None)
    monkeypatch.setattr(review_mod, "_publish_approved_plan_to_graph", lambda *_a, **_k: None)
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=approve)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-1")

    first = asyncio.run(agent._review_plan(ctx, plan.model_dump_json()))
    assert first.action == HITLAction.EDIT
    assert state["experiment_plan_candidate"]["status"] == "needs_revision"

    broken = plan.model_dump(mode="json")
    broken["tasks"][0]["mcp_servers"][0]["url"] = None
    response = asyncio.run(agent._review_plan(ctx, broken))

    assert response.approved and len(asked) == 1
    shown = asked[0].context["experiment_plan"]
    assert shown["recovered_previous_candidate"] is True
    assert shown["superseded_validation_errors"]
    assert state["experiment_runtime"]["plan"] == plan.model_dump(mode="json")
    assert state["experiment_plan_validation_errors"] is None


def test_exhausted_plan_is_revalidated_after_the_human_wait(monkeypatch):
    task = _task("EXP-1", route="react_tools")
    plan = _plan(task)
    state = {
        "experiment_context": {
            **_context(plan),
            "operations": [
                {"operation_id": "OP-1", "statement": "Prepare input data"},
                {"operation_id": "OP-2", "statement": "Run the computation"},
            ],
        },
        "experiment_plan_revision_count": ExperimentsSettings().max_plan_revisions - 1,
    }

    async def approve_after_inventory_disappears(_request):
        state["experiment_context"]["available_mcp_capabilities"] = []
        return HITLResponse(
            action=HITLAction.APPROVE,
            approved=True,
            decision_source=HITLDecisionSource.HUMAN,
        )

    monkeypatch.setattr(review_mod, "record_plan_proposed", lambda *_a, **_k: "PR-1")
    monkeypatch.setattr(review_mod, "close_plan_record", lambda *_a, **_k: None)
    monkeypatch.setattr(review_mod, "_publish_approved_plan_to_graph", lambda *_a, **_k: None)
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=approve_after_inventory_disappears)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-1")

    response = asyncio.run(agent._review_plan(ctx, plan.model_dump_json()))

    assert not response.approved and response.stop_review_loop
    assert response.system_reason == "fallback_candidate_no_longer_executable"
    assert state["experiment_runtime"]["phase"] == "awaiting_review"
    assert state["experiment_plan_candidate"]["status"] == "blocked"
    assert state["experiment_module_outcome"]["status"] == "blocked"
    assert (
        state["experiment_module_outcome"]["reason"]
        == "fallback_candidate_no_longer_executable"
    )
