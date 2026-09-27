"""Impossible plan sizes are resolved before spending planner/model calls."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.plan_policy import (
    CAPACITY_OVERRIDE_KEY, check_experiment_plan_capacity, effective_plan_settings, operations_digest,
)
from CoScientist.hitl.models import HITLAction, HITLDecisionSource, HITLResponse


def _context(monkeypatch, response, count=9):
    cfg = ExperimentsSettings(max_plan_tasks=8)
    monkeypatch.setattr("CoScientist.config.get_settings", lambda: SimpleNamespace(experiments=cfg))
    requests = []

    async def handle_request(request):
        requests.append(request)
        return response

    state = {"experiment_context": {
        "experiment_run_id": "EXRUN-policy",
        "operations": [{"operation_id": f"OP-{i}", "statement": f"Compute property {i}"}
                       for i in range(1, count + 1)],
    }, "experiment_plan_revision_count": 0}
    ctx = SimpleNamespace(state=state, _invocation_context=SimpleNamespace(
        agent=SimpleNamespace(hitl_handler=SimpleNamespace(handle_request=handle_request))))
    return cfg, ctx, requests


def test_human_can_resolve_capacity_once_for_exact_run_without_revision_or_global_change(monkeypatch):
    cfg, ctx, requests = _context(monkeypatch, HITLResponse(action=HITLAction.APPROVE, approved=True))
    assert asyncio.run(check_experiment_plan_capacity(ctx)) is None
    assert requests[0].requires_human
    assert len(requests) == 1
    assert effective_plan_settings(cfg, ctx.state["experiment_context"]).max_plan_tasks == 9
    assert cfg.max_plan_tasks == 8
    assert ctx.state["experiment_plan_revision_count"] == 0
    assert "experiment_runtime" not in ctx.state
    assert json.loads(ctx.state["experiment_planner_context"])["plan_limits"]["max_tasks"] == 9
    assert asyncio.run(check_experiment_plan_capacity(ctx)) is None
    assert len(requests) == 1


@pytest.mark.parametrize("response", [
    HITLResponse(action=HITLAction.REJECT),
    HITLResponse(action=HITLAction.APPROVE, approved=True, decision_source=HITLDecisionSource.MODE_AUTO),
    HITLResponse(action=HITLAction.REJECT, timed_out=True, decision_source=HITLDecisionSource.TIMEOUT),
])
def test_no_automatic_capacity_increase_or_planner_on_decline_timeout(monkeypatch, response):
    cfg, ctx, requests = _context(monkeypatch, response)
    assert asyncio.run(check_experiment_plan_capacity(ctx)) is not None
    assert ctx.state["experiment_module_outcome"]["reason"] == "plan_capacity_conflict"
    assert ctx.state["experiment_plan_review_paused"] is True
    assert CAPACITY_OVERRIDE_KEY not in ctx.state
    assert cfg.max_plan_tasks == 8


def test_test1_compute_only_projection_fits_existing_capacity_without_hitl(monkeypatch):
    _, ctx, requests = _context(monkeypatch, None, count=7)
    assert asyncio.run(check_experiment_plan_capacity(ctx)) is None
    assert requests == []


def test_capacity_approval_cannot_leak_to_another_run_or_changed_operations(monkeypatch):
    cfg, ctx, _ = _context(monkeypatch, HITLResponse(action=HITLAction.APPROVE, approved=True))
    asyncio.run(check_experiment_plan_capacity(ctx))
    original = ctx.state["experiment_context"]
    for field in ("experiment_run_id", "operations"):
        changed = copy.deepcopy(original)
        if field == "operations":
            changed[field][0]["statement"] = "Compute an unrelated operation"
        else:
            changed[field] = "EXRUN-other"
        assert effective_plan_settings(cfg, changed).max_plan_tasks == 8
    assert original["plan_capacity_override"]["operations_digest"] == operations_digest(original["operations"])


def test_schema_capacity_cannot_be_raised_or_hide_a_twenty_first_operation(monkeypatch):
    from CoScientist.experiments.context.builder import build_experiment_context

    # Use the real settings for context construction; the callback must see all
    # 21 committed operations, not a 20-row prompt projection.
    operations = [{"operation_id": f"OP-{i}", "statement": f"Compute property {i}"} for i in range(1, 22)]
    state = {"experiment_operations": operations, "experiment_source_request": "Compute all properties"}
    build_experiment_context(SimpleNamespace(state=state, user_content=None))
    assert len(state["experiment_context"]["operations"]) == 21
    response = asyncio.run(check_experiment_plan_capacity(SimpleNamespace(state=state)))
    assert response is not None
    assert len(state["experiment_module_outcome"]["required_operations"]) == 21
    assert state["experiment_module_outcome"]["can_raise_limit"] is False


def test_builder_preserves_same_run_capacity_and_candidate_on_explicit_resume(monkeypatch):
    from CoScientist.config import get_settings
    from CoScientist.experiments.context.builder import build_experiment_context

    monkeypatch.setattr(get_settings().experiments, "max_plan_tasks", 8)
    operations = [{"operation_id": f"OP-{i}", "statement": f"Compute property {i}"} for i in range(1, 10)]
    grant = {"run_id": "EXRUN-resume", "operations_digest": operations_digest(operations),
             "max_plan_tasks": 9, "decision_source": "human"}
    state = {
        "experiment_source_request": "Compute properties",
        "experiment_context": {"source_request": "Compute properties", "experiment_run_id": "EXRUN-resume"},
        "experiment_operations": operations,
        "experiment_plan_review_paused": True,
        "experiment_plan_recovery_requested": {"source": "run_control", "reason": "max_plan_revisions"},
        "experiment_plan_last_executable_candidate": {"sentinel": "saved"},
        CAPACITY_OVERRIDE_KEY: grant,
    }
    build_experiment_context(SimpleNamespace(state=state, user_content=None))
    assert state["experiment_context"]["experiment_run_id"] == "EXRUN-resume"
    assert json.loads(state["experiment_planner_context"])["plan_limits"]["max_tasks"] == 9
    assert state["experiment_plan_last_executable_candidate"] == {"sentinel": "saved"}
    assert state["experiment_plan_recovery_requested"]["source"] == "run_control"

    state["experiment_force_new_run"] = True
    build_experiment_context(SimpleNamespace(state=state, user_content=None))
    assert state["experiment_context"]["experiment_run_id"] != "EXRUN-resume"
    assert state[CAPACITY_OVERRIDE_KEY] is None
    assert state["experiment_plan_last_executable_candidate"] is None
    assert state["experiment_plan_recovery_requested"] is None
    assert json.loads(state["experiment_planner_context"])["plan_limits"]["max_tasks"] == 8
