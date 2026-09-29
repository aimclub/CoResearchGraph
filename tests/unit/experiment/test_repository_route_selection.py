from __future__ import annotations

import asyncio
from types import SimpleNamespace

from CoScientist.config import get_settings
from CoScientist.experiments import review as review_mod
from CoScientist.experiments.review import ExperimentReviewSessionAgent
from CoScientist.hitl.models import HITLAction, HITLResponse
from CoScientist.hitl.resolver import resolve_auto, resolve_timeout

from .helpers import _plan, _task


REPO = "https://github.com/whitead/synspace"


def _reuse_plan():
    task = _task("EXP-1", route="coder")
    task.update({
        "name": "Run existing repository entrypoint",
        "description": "Execute the repository's existing scoring entrypoint unchanged.",
        "repo_url": REPO,
        "code_assessment": {
            "requirement": "reuse",
            "evidence": "Inspection found an existing CLI with the required input and output contract.",
            "entrypoints": ["python -m synspace.score"],
        },
    })
    task["design"]["experiment_question"] = "Does the existing repository CLI produce the score?"
    task["design"]["analysis_artifacts"][0].update({
        "prepare_via": "coder",
        "path_or_tool": "scores.csv",
    })
    return _plan(task)


def _state(plan):
    return {
        "experiment_context": {
            "experiment_run_id": plan.experiment_run_id,
            "source_request": plan.source_request,
            "available_mcp_capabilities": [],
            "repo_candidates": [{"url": REPO, "repo_name": "synspace"}],
        }
    }


def _agent(monkeypatch, handler):
    monkeypatch.setattr(review_mod, "_alembic_preflight", lambda: {
        "available": True,
        "reason": "Docker daemon is reachable",
    })
    monkeypatch.setattr(review_mod, "_publish_approved_plan_to_graph", lambda *_a, **_k: None)
    monkeypatch.setattr(review_mod, "record_plan_proposed", lambda *_a, **_k: "PLANREC-test")
    monkeypatch.setattr(review_mod, "close_plan_record", lambda *_a, **_k: None)
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=handler)
    return agent


def test_human_can_select_alembic_for_proven_unchanged_repo(monkeypatch):
    plan = _reuse_plan()
    state = _state(plan)
    monkeypatch.setattr(get_settings().experiments, "route_alembic", True)
    requests = []

    async def _handler(request):
        requests.append(request)
        if request.action_type == HITLAction.SELECT:
            return HITLResponse(
                action=HITLAction.SELECT,
                approved=True,
                selected_option=review_mod._ROUTE_ALEMBIC_OPTION,
            )
        return HITLResponse(action=HITLAction.APPROVE, approved=True)

    agent = _agent(monkeypatch, _handler)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-route")
    response = asyncio.run(agent._review_plan(ctx, plan.model_dump_json()))

    assert response.approved
    assert [request.action_type for request in requests] == [HITLAction.SELECT, HITLAction.APPROVE]
    runtime_task = state["experiment_runtime"]["tasks"]["EXP-1"]
    assert runtime_task["planned_route"] == "alembic_build"
    assert runtime_task["task"]["repo_url"] == REPO
    assert runtime_task["task"]["post_build_route"] == "react_tools"
    assert runtime_task["route_history"][0]["reason"] == "repository_route_selected"
    assert runtime_task["route_history"][0]["decision_source"] == "human"


def test_auto_mode_selects_coder_as_the_safe_fast_default(monkeypatch):
    plan = _reuse_plan()
    state = _state(plan)
    monkeypatch.setattr(get_settings().experiments, "route_alembic", True)
    requests = []

    async def _handler(request):
        requests.append(request)
        return resolve_auto(request)

    agent = _agent(monkeypatch, _handler)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-auto")
    response = asyncio.run(agent._review_plan(ctx, plan.model_dump_json()))

    assert response.approved
    assert requests[0].action_type == HITLAction.SELECT
    assert requests[0].default_option == review_mod._ROUTE_CODER_OPTION
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["planned_route"] == "coder"


def test_unavailable_docker_uses_coder_without_offering_a_dead_route(monkeypatch):
    plan = _reuse_plan()
    state = _state(plan)
    monkeypatch.setattr(get_settings().experiments, "route_alembic", True)
    monkeypatch.setattr(review_mod, "_alembic_preflight", lambda: {
        "available": False,
        "reason": "Docker DNS b.dgx is unavailable",
    })
    requests = []

    async def _handler(request):
        requests.append(request)
        assert request.action_type != HITLAction.SELECT
        return HITLResponse(action=HITLAction.APPROVE, approved=True)

    agent = _agent(monkeypatch, _handler)
    # _agent installs the success probe; restore the failure after construction.
    monkeypatch.setattr(review_mod, "_alembic_preflight", lambda: {
        "available": False,
        "reason": "Docker DNS b.dgx is unavailable",
    })
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-down")
    response = asyncio.run(agent._review_plan(ctx, plan.model_dump_json()))

    assert response.approved
    assert len(requests) == 1
    runtime_task = state["experiment_runtime"]["tasks"]["EXP-1"]
    assert runtime_task["planned_route"] == "coder"
    assert any("b.dgx" in warning for warning in runtime_task["task"]["warnings"])
    selection = next(iter(state[review_mod.ROUTE_SELECTIONS_STATE_KEY].values()))
    assert "b.dgx" in selection["reason"]


def test_route_selection_timeout_pauses_fail_closed(monkeypatch):
    plan = _reuse_plan()
    state = _state(plan)
    monkeypatch.setattr(get_settings().experiments, "route_alembic", True)

    async def _handler(request):
        assert request.action_type == HITLAction.SELECT
        return resolve_timeout(reason="route_choice_elapsed")

    agent = _agent(monkeypatch, _handler)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-timeout")
    response = asyncio.run(agent._review_plan(ctx, plan.model_dump_json()))

    assert response.timed_out and response.stop_review_loop
    assert state[review_mod.PAUSE_REASON_STATE_KEY] == "repository_route_timeout"
    assert "experiment_runtime" not in state


def test_unchanged_plan_reuses_the_recorded_route_choice(monkeypatch):
    plan = _reuse_plan()
    state = _state(plan)
    calls = 0

    async def _handler(request):
        nonlocal calls
        calls += 1
        return HITLResponse(
            action=HITLAction.SELECT,
            approved=True,
            selected_option=review_mod._ROUTE_ALEMBIC_OPTION,
        )

    agent = _agent(monkeypatch, _handler)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-cache")
    kwargs = dict(
        ctx=ctx,
        route_alembic=True,
        user_id="user",
        session_id="session",
        timeout_seconds=600.0,
    )
    selected, response = asyncio.run(agent._select_repository_routes(plan=plan, **kwargs))
    selected_again, response_again = asyncio.run(agent._select_repository_routes(plan=plan, **kwargs))

    assert response is None and response_again is None
    assert calls == 1
    assert selected.tasks[0].route.value == "alembic_build"
    assert selected_again.tasks[0].route.value == "alembic_build"
