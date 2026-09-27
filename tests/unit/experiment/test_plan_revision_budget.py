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
import copy
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


def test_quality_feedback_does_not_open_or_reset_a_planning_round():
    """Assessment feedback is terminal until explicit task selection."""
    from CoScientist.experiments.runtime import mark_result_review

    from .helpers import _approved_state

    state = _approved_state(_plan(_task("EXP-1")))
    state["experiment_runtime"]["phase"] = "reporting"
    state["experiment_plan_revision_count"] = 3
    state["experiment_inventory_blocker_hits"] = 2

    out = mark_result_review(state, approved=False, feedback="metrics missing")

    assert out["phase"] == "completed"
    assert out["result_review_disposition"] == "changes_suggested"
    assert state["experiment_plan_revision_count"] == 3
    assert state["experiment_inventory_blocker_hits"] == 2


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


def _operations(count: int = 9) -> list[dict]:
    return [
        {"operation_id": f"OP-{index}", "statement": f"Required computation {index}"}
        for index in range(1, count + 1)
    ]


def _operation_plan(operation_ids: list[str]):
    tasks = []
    for index, operation_id in enumerate(operation_ids, start=1):
        task = _task(f"EXP-{index}", route="react_tools")
        task["design"]["operation_ref"] = operation_id
        tasks.append(task)
    return _plan(*tasks)


def _stub_plan_records(monkeypatch) -> None:
    monkeypatch.setattr(review_mod, "record_plan_proposed", lambda *_a, **_k: "PR-1")
    monkeypatch.setattr(review_mod, "close_plan_record", lambda *_a, **_k: None)
    monkeypatch.setattr(review_mod, "_publish_approved_plan_to_graph", lambda *_a, **_k: None)


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


def test_test1_blocked_nine_task_final_draft_offers_penultimate_partial_plan(monkeypatch):
    """9 required OPs/cap 8: blocked final draft cannot erase executable rev 3."""
    partial = _operation_plan(["OP-1", "OP-3", "OP-4", "OP-6", "OP-7", "OP-8", "OP-9"])
    over_limit = _operation_plan([f"OP-{index}" for index in range(1, 10)])
    state = {
        "experiment_context": {
            **_context(partial),
            "operations": _operations(),
        },
    }
    asked = []

    async def approve(request):
        asked.append(request)
        return HITLResponse(
            action=HITLAction.APPROVE,
            approved=True,
            decision_source=HITLDecisionSource.HUMAN,
        )

    _stub_plan_records(monkeypatch)
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=approve)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-test1")

    # Two malformed drafts plus the two schema-valid drafts are exactly four
    # failed revision rounds. The third is partial but executable; the fourth
    # covers all OPs by exceeding the hard task cap and is not executable.
    assert asyncio.run(agent._review_plan(ctx, "{" )).action == HITLAction.EDIT
    assert asyncio.run(agent._review_plan(ctx, "{" )).action == HITLAction.EDIT
    third = asyncio.run(agent._review_plan(ctx, partial.model_dump(mode="json")))
    assert third.action == HITLAction.EDIT
    response = asyncio.run(agent._review_plan(ctx, over_limit.model_dump(mode="json")))

    assert response.approved and len(asked) == 1
    assert state["experiment_plan_revision_count"] == 0  # approval resets the spent budget
    assert state["experiment_plan_last_schema_valid_candidate"]["digest"] == review_mod._plan_digest(over_limit)
    assert state["experiment_plan_last_schema_valid_candidate"]["readiness"]["executable"] is False
    assert state["experiment_plan_last_executable_candidate"]["digest"] == review_mod._plan_digest(partial)
    assert state["experiment_runtime"]["plan"] == partial.model_dump(mode="json")
    assert state["experiment_module_outcome"]["status"] == "running"
    assert state["experiment_module_outcome"]["accepted"] is True

    shown = asked[0].context["experiment_plan"]
    assert shown["task_count"] == 7
    assert shown["partial_candidate"] is True
    assert shown["uncovered_operations"] == ["OP-2", "OP-5"]
    assert shown["plan_digest"] == review_mod._plan_digest(partial)
    latest = shown["superseded_schema_valid_candidate"]
    assert latest["task_count"] == 9
    assert any(
        issue["category"] == "complexity" and issue["severity"] == "blocker"
        for issue in latest["critique"]["issues"]
    )


def test_legacy_candidate_is_migrated_before_schema_failure_fallback(monkeypatch):
    plan = _operation_plan(["OP-1"])
    state = {
        "experiment_context": {**_context(plan), "operations": _operations(2)},
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

    _stub_plan_records(monkeypatch)
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=approve)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-legacy")

    assert asyncio.run(agent._review_plan(ctx, plan.model_dump(mode="json"))).action == HITLAction.EDIT
    legacy = state["experiment_plan_candidate"]
    state.pop("experiment_plan_last_schema_valid_candidate")
    state.pop("experiment_plan_last_executable_candidate")
    state["experiment_plan_candidate"] = legacy

    response = asyncio.run(agent._review_plan(ctx, "{"))

    assert response.approved and len(asked) == 1
    assert state["experiment_runtime"]["plan"] == plan.model_dump(mode="json")
    assert state["experiment_plan_last_executable_candidate"]["digest"] == review_mod._plan_digest(plan)


def test_cross_run_candidate_cache_is_never_offered(monkeypatch):
    old_plan = _operation_plan(["OP-1"])
    state = {
        "experiment_context": {**_context(old_plan), "operations": _operations(2)},
        "experiment_plan_revision_count": ExperimentsSettings().max_plan_revisions - 2,
    }
    asked = []

    async def should_not_ask(request):
        asked.append(request)
        raise AssertionError("a candidate from another experiment run must not be offered")

    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=should_not_ask)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-stale")
    assert asyncio.run(agent._review_plan(ctx, old_plan.model_dump(mode="json"))).action == HITLAction.EDIT

    state["experiment_context"] = {
        **state["experiment_context"],
        "experiment_run_id": "EXRUN-new",
    }
    response = asyncio.run(agent._review_plan(ctx, "{"))

    assert not response.approved and response.stop_review_loop
    assert asked == []
    assert state["experiment_plan_last_executable_candidate"] is None
    assert state["experiment_plan_last_schema_valid_candidate"] is None
    assert state["experiment_module_outcome"]["reason"] == "max_plan_revisions"


def test_candidate_with_a_tampered_digest_is_not_a_recovery_checkpoint():
    plan = _operation_plan(["OP-1"])
    state = {
        "experiment_context": {**_context(plan), "operations": _operations(2)},
        "experiment_plan_revision_count": ExperimentsSettings().max_plan_revisions - 2,
    }
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-digest")
    assert asyncio.run(agent._review_plan(ctx, plan.model_dump(mode="json"))).action == HITLAction.EDIT

    state["experiment_plan_last_executable_candidate"]["digest"] = "tampered"
    state["experiment_plan_candidate"]["digest"] = "tampered"
    response = asyncio.run(agent._review_plan(ctx, "{"))

    assert not response.approved and response.stop_review_loop
    assert state["experiment_plan_last_executable_candidate"] is None
    assert state["experiment_module_outcome"]["reason"] == "max_plan_revisions"


def test_cached_candidate_is_revalidated_before_hitl_is_opened(monkeypatch):
    plan = _operation_plan(["OP-1"])
    state = {
        "experiment_context": {**_context(plan), "operations": _operations(2)},
        "experiment_plan_revision_count": ExperimentsSettings().max_plan_revisions - 2,
    }
    asked = []

    async def should_not_ask(request):
        asked.append(request)
        raise AssertionError("stale inventory must block before opening HITL")

    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=should_not_ask)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-cache")
    assert asyncio.run(agent._review_plan(ctx, plan.model_dump(mode="json"))).action == HITLAction.EDIT

    state["experiment_context"]["available_mcp_capabilities"] = []
    response = asyncio.run(agent._review_plan(ctx, "{"))

    assert not response.approved and response.stop_review_loop
    assert asked == []
    assert state["experiment_module_outcome"]["reason"] == "fallback_candidate_no_longer_executable"
    assert state["experiment_plan_candidate"]["status"] == "blocked"
    assert state.get("experiment_runtime") is None


@pytest.mark.parametrize(
    ("human_response", "expected_reason", "expected_status"),
    [
        (
            HITLResponse(
                action=HITLAction.REJECT,
                approved=False,
                decision_source=HITLDecisionSource.HUMAN,
            ),
            "fallback_rejected_by_operator",
            "rejected",
        ),
        (
            HITLResponse(
                action=HITLAction.REJECT,
                approved=False,
                decision_source=HITLDecisionSource.TIMEOUT,
                timed_out=True,
            ),
            "fallback_review_timeout",
            "timed_out",
        ),
    ],
)
def test_exhausted_fallback_decline_and_timeout_remain_blocked(
    monkeypatch, human_response, expected_reason, expected_status,
):
    plan = _operation_plan(["OP-1"])
    state = {
        "experiment_context": {**_context(plan), "operations": _operations(2)},
        "experiment_plan_revision_count": ExperimentsSettings().max_plan_revisions - 1,
    }

    async def answer(_request):
        return human_response

    _stub_plan_records(monkeypatch)
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=answer)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-terminal")

    response = asyncio.run(agent._review_plan(ctx, plan.model_dump(mode="json")))

    assert not response.approved and response.stop_review_loop
    assert state["experiment_module_outcome"]["status"] == "blocked"
    assert state["experiment_module_outcome"]["reason"] == expected_reason
    assert state["experiment_module_outcome"]["accepted"] is False
    assert state["experiment_plan_candidate"]["status"] == expected_status
    assert state["experiment_runtime"]["phase"] == "awaiting_review"


def test_explicit_recovery_reopens_same_digest_without_spending_a_revision(monkeypatch):
    plan = _operation_plan(["OP-1"])
    state = {
        "experiment_context": {**_context(plan), "operations": _operations(2)},
        "experiment_plan_revision_count": ExperimentsSettings().max_plan_revisions - 1,
    }
    requests = []
    responses = [
        HITLResponse(
            action=HITLAction.REJECT,
            approved=False,
            decision_source=HITLDecisionSource.TIMEOUT,
            timed_out=True,
        ),
        HITLResponse(
            action=HITLAction.APPROVE,
            approved=True,
            decision_source=HITLDecisionSource.HUMAN,
        ),
    ]

    async def answer(request):
        requests.append(request)
        return responses.pop(0)

    _stub_plan_records(monkeypatch)
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=answer)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-resume")

    first = asyncio.run(agent._review_plan(ctx, plan.model_dump(mode="json")))
    assert first.timed_out and state["experiment_plan_revision_count"] == 4
    state["experiment_plan_recovery_requested"] = {
        "source": "run_control",
        "reason": "max_plan_revisions",
    }

    second = asyncio.run(agent._review_plan(ctx, "this output must not be parsed"))

    assert second.approved and len(requests) == 2
    assert requests[0].context["experiment_review_id"] == requests[1].context["experiment_review_id"]
    assert requests[1].context["experiment_plan"]["recovery_resumed"] is True
    assert state["experiment_plan_recovery_requested"] is None
    assert state["experiment_runtime"]["phase"] == "execution"


def test_delayed_recovery_cannot_replace_an_approved_plan_or_results(monkeypatch):
    plan = _plan(_task("EXP-1", route="react_tools"))
    state = {"experiment_context": _context(plan)}
    monkeypatch.setenv("HITL__MODE", "auto")
    monkeypatch.setattr(review_mod, "_publish_approved_plan_to_graph", lambda *_a, **_k: None)
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-immutable")

    approved = asyncio.run(agent._review_plan(ctx, plan.model_dump(mode="json")))
    assert approved.approved
    state["experiment_runtime"]["results"] = [{"result_id": "RESULT-kept"}]
    snapshot = copy.deepcopy(state["experiment_runtime"])
    state["experiment_plan_recovery_requested"] = {
        "source": "run_control",
        "reason": "max_plan_revisions",
    }

    rejected = asyncio.run(agent._review_plan(ctx, "this must remain unread"))

    assert rejected.system_reason == "approved_plan_immutable"
    assert rejected.stop_review_loop
    assert state["experiment_runtime"] == snapshot
    assert state["experiment_plan_recovery_requested"] is None


def test_nine_task_fallback_revalidation_honors_exact_capacity_override(monkeypatch):
    from CoScientist.config import get_settings
    from CoScientist.experiments.plan_policy import operations_digest

    plan = _operation_plan([f"OP-{index}" for index in range(1, 10)])
    operations = _operations()
    context = {
        **_context(plan),
        "operations": operations,
        "plan_capacity_override": {
            "run_id": plan.experiment_run_id,
            "operations_digest": operations_digest(operations),
            "max_plan_tasks": 9,
            "decision_source": "human",
        },
    }
    monkeypatch.setattr(get_settings().experiments, "max_plan_tasks", 8)
    _, critique = validate_and_critique_plan(
        plan.model_dump(mode="json"),
        settings=ExperimentsSettings(max_plan_tasks=9),
        available_tools=_inventory(),
        operations=operations,
    )
    assert critique.verdict == "approve"
    candidate = review_mod._candidate_record(
        plan,
        critique,
        reason="max_plan_revisions",
        revision_count=ExperimentsSettings().max_plan_revisions,
    )
    candidate["status"] = "awaiting_human"
    state = {
        "experiment_context": context,
        "experiment_plan_candidate": candidate,
    }
    asked = []

    async def approve(request):
        asked.append(request)
        return HITLResponse(
            action=HITLAction.APPROVE,
            approved=True,
            decision_source=HITLDecisionSource.HUMAN,
        )

    _stub_plan_records(monkeypatch)
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=approve)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-capacity")

    response = asyncio.run(agent._review_exhausted_candidate(
        ctx=ctx,
        plan=plan,
        critique=critique,
        reason="max_plan_revisions",
        context=context,
        previous=None,
        route_agents=set(),
    ))

    assert response.approved and len(asked) == 1
    assert state["experiment_runtime"]["phase"] == "execution"
    assert state["experiment_runtime"]["plan"] == plan.model_dump(mode="json")


@pytest.mark.parametrize("stale_part", ["run_id", "operations_digest"])
def test_stale_capacity_override_cannot_bypass_eight_task_limit(monkeypatch, stale_part):
    from CoScientist.config import get_settings
    from CoScientist.experiments.plan_policy import operations_digest

    plan = _operation_plan([f"OP-{index}" for index in range(1, 10)])
    operations = _operations()
    grant = {
        "run_id": plan.experiment_run_id,
        "operations_digest": operations_digest(operations),
        "max_plan_tasks": 9,
        "decision_source": "human",
    }
    grant[stale_part] = (
        "EXRUN-prior"
        if stale_part == "run_id"
        else operations_digest(operations[:-1])
    )
    context = {
        **_context(plan),
        "operations": operations,
        "plan_capacity_override": grant,
    }
    monkeypatch.setattr(get_settings().experiments, "max_plan_tasks", 8)
    _, critique = validate_and_critique_plan(
        plan.model_dump(mode="json"),
        settings=ExperimentsSettings(max_plan_tasks=9),
        available_tools=_inventory(),
        operations=operations,
    )
    candidate = review_mod._candidate_record(
        plan,
        critique,
        reason="max_plan_revisions",
        revision_count=ExperimentsSettings().max_plan_revisions,
    )
    candidate["status"] = "awaiting_human"
    state = {
        "experiment_context": context,
        "experiment_plan_candidate": candidate,
    }
    asked = []

    async def should_not_ask(request):
        asked.append(request)
        raise AssertionError("a stale capacity grant must fail before HITL")

    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=should_not_ask)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-stale-capacity")

    response = asyncio.run(agent._review_exhausted_candidate(
        ctx=ctx,
        plan=plan,
        critique=critique,
        reason="max_plan_revisions",
        context=context,
        previous=None,
        route_agents=set(),
    ))

    assert not response.approved and response.stop_review_loop
    assert response.system_reason == "fallback_candidate_no_longer_executable"
    assert asked == []
    assert state["experiment_plan_candidate"]["status"] == "blocked"
    assert any(
        "Plan has 9 tasks; max is 8" in error.get("msg", "")
        for error in state["experiment_plan_validation_errors"]
    )


@pytest.mark.parametrize(
    ("first_response", "recovery_reason"),
    [
        (
            HITLResponse(
                action=HITLAction.REJECT,
                approved=False,
                decision_source=HITLDecisionSource.TIMEOUT,
                timed_out=True,
            ),
            "fallback_review_timeout",
        ),
        (
            HITLResponse(
                action=HITLAction.REJECT,
                approved=False,
                decision_source=HITLDecisionSource.HUMAN,
            ),
            "fallback_rejected_by_operator",
        ),
    ],
)
def test_explicit_recovery_control_path_does_not_call_the_planner(
    monkeypatch, first_response, recovery_reason,
):
    plan = _operation_plan(["OP-1"])
    state = {
        "experiment_context": {**_context(plan), "operations": _operations(2)},
        "experiment_plan_revision_count": ExperimentsSettings().max_plan_revisions - 1,
    }
    responses = [
        first_response,
        HITLResponse(
            action=HITLAction.APPROVE,
            approved=True,
            decision_source=HITLDecisionSource.HUMAN,
        ),
    ]

    async def answer(_request):
        return responses.pop(0)

    _stub_plan_records(monkeypatch)
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=answer)
    ctx = SimpleNamespace(
        session=SimpleNamespace(state=state),
        invocation_id="inv-control-resume",
        branch=None,
    )
    first = asyncio.run(agent._review_plan(ctx, plan.model_dump(mode="json")))
    assert not first.approved and first.stop_review_loop
    assert state["experiment_review_pause_reason"] == recovery_reason
    assert state.get("experiment_plan_recovery_requested") is None
    state["experiment_plan_recovery_requested"] = {
        "source": "run_control",
        "reason": recovery_reason,
    }
    produce_calls = []

    def forbidden_produce(_self, _ctx):
        produce_calls.append(True)

        async def events():
            raise AssertionError("recovery must not call the planner model")
            yield  # pragma: no cover

        return events()

    monkeypatch.setattr(ExperimentReviewSessionAgent, "_produce", forbidden_produce)

    async def run_control_path():
        return [event async for event in agent._run_async_impl(ctx)]

    events = asyncio.run(run_control_path())

    assert produce_calls == []
    assert len(events) == 1
    assert events[0].actions.state_delta["experiment_runtime"]["phase"] == "execution"
    assert state["experiment_runtime"]["phase"] == "execution"


def test_explicit_recovery_pauses_again_when_saved_candidate_is_now_invalid(monkeypatch):
    plan = _operation_plan(["OP-1"])
    state = {
        "experiment_context": {**_context(plan), "operations": _operations(2)},
        "experiment_plan_revision_count": ExperimentsSettings().max_plan_revisions - 1,
    }
    requests = []

    async def timeout_once(request):
        requests.append(request)
        if len(requests) > 1:
            raise AssertionError("invalidated recovery must stop before a second HITL")
        return HITLResponse(
            action=HITLAction.REJECT,
            approved=False,
            decision_source=HITLDecisionSource.TIMEOUT,
            timed_out=True,
        )

    _stub_plan_records(monkeypatch)
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=timeout_once)
    ctx = SimpleNamespace(
        session=SimpleNamespace(state=state),
        invocation_id="inv-invalid-resume",
        branch=None,
    )
    first = asyncio.run(agent._review_plan(ctx, plan.model_dump(mode="json")))
    assert first.timed_out
    assert state.get("experiment_plan_recovery_requested") is None
    state["experiment_context"]["available_mcp_capabilities"] = []
    state["experiment_plan_recovery_requested"] = {
        "source": "run_control",
        "reason": "fallback_review_timeout",
    }
    produce_calls = []

    def forbidden_produce(_self, _ctx):
        produce_calls.append(True)

        async def events():
            raise AssertionError("saved-plan recovery must not call the planner")
            yield  # pragma: no cover

        return events()

    monkeypatch.setattr(ExperimentReviewSessionAgent, "_produce", forbidden_produce)

    async def run_control_path():
        return [event async for event in agent._run_async_impl(ctx)]

    events = asyncio.run(run_control_path())

    assert produce_calls == []
    assert len(requests) == 1
    assert state["experiment_plan_recovery_requested"] is None
    assert state["experiment_plan_review_paused"] is True
    assert state["experiment_module_outcome"]["status"] == "blocked"
    assert (
        state["experiment_module_outcome"]["reason"]
        == "fallback_candidate_no_longer_executable"
    )
    assert events[0].actions.state_delta["experiment_plan_review_paused"] is True
    assert review_mod._is_plan_recovery_request({
        "source": "run_control",
        "reason": "fallback_candidate_no_longer_executable",
    }) is False
