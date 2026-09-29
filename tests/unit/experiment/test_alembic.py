"""Alembic build route, await, auto-record, critique."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.critique import critique_plan
from CoScientist.experiments.runtime import (
    ExperimentRuntimeError,
    approve_plan,
    initialize_runtime,
    mark_route_returned,
    on_route_agent_returned,
    await_alembic_job_if_experiment,
    record_result,
    start_task,
)
from CoScientist.experiments.schemas import ExperimentTask

from .helpers import (
    _alembic_started_state,
    _alembic_task,
    _inventory,
    _plan,
    _task,
    _tool_context,
)

def test_alembic_task_requires_repo_and_post_build_route():
    bare = _alembic_task()
    del bare["repo_url"]
    with pytest.raises(ValidationError):
        ExperimentTask.model_validate(bare)
    missing_post = _alembic_task()
    missing_post["post_build_route"] = None
    with pytest.raises(ValidationError):
        ExperimentTask.model_validate(missing_post)
    with pytest.raises(ValidationError):
        ExperimentTask.model_validate(
            {**_task("EXP-1", route="coder"), "post_build_route": "fedot_mas"}
        )


def test_start_task_rejects_alembic_when_route_disabled():
    plan = _plan(_alembic_task())
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(route_alembic=False),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    assert critique.verdict == "revise"
    assert any("alembic_build" in i.message for i in critique.issues)

    # Bypass critique to assert runtime gate.
    state: dict = {}
    initialize_runtime(
        state,
        plan,
        critique={"verdict": "approve", "issues": [], "summary": "forced"},
    )
    approve_plan(state)
    with pytest.raises(ExperimentRuntimeError) as exc:
        start_task(state, "EXP-1", settings=ExperimentsSettings(route_alembic=False))
    assert exc.value.code == "route_disabled"


def test_alembic_success_reopens_task_on_post_build_route():
    plan = _plan(_alembic_task())
    state: dict = {}
    initialize_runtime(
        state,
        plan,
        critique={"verdict": "approve", "issues": [], "summary": "forced"},
    )
    approve_plan(state)
    settings = ExperimentsSettings(route_alembic=True)
    started = start_task(state, "EXP-1", settings=settings)
    assert started["route_agent"] == "McpBuilderAgent"
    mark_route_returned(state, "McpBuilderAgent")
    recorded = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "success",
            "summary": "Built MCP",
            "outputs": {
                "mcp_url": "http://127.0.0.1:9000/mcp",
                "mcp_endpoint": "http://127.0.0.1:9000/mcp",
                "tools": ["synspace_score"],
            },
            "criteria_checks": [
                {
                    "criterion_id": "EXP-1-C1",
                    "passed": True,
                    "details": "mcp_url present",
                }
            ],
        },
        settings=settings,
    )
    assert recorded["status"] == "success"
    assert recorded["post_build"]["post_build_route"] == "react_tools"
    runtime = state["experiment_runtime"]
    task_runtime = runtime["tasks"]["EXP-1"]
    assert task_runtime["status"] == "ready"
    assert task_runtime["current_route"] == "react_tools"
    assert task_runtime["task"]["mcp_servers"][0]["source"] == "alembic"
    assert task_runtime["task"]["mcp_servers"][0]["url"] == "http://127.0.0.1:9000/mcp"
    assert task_runtime["task"]["mcp_servers"][0]["tools"][0]["name"] == "synspace_score"
    assert state["deployed_mcps"][0]["url"] == "http://127.0.0.1:9000/mcp"

    second = start_task(state, "EXP-1", settings=settings)
    assert second["route_agent"] == "ExperimentAgent"
    assert second["route"] == "react_tools"


def test_alembic_success_defers_scientific_evidence_to_post_build(monkeypatch):
    """One task may list MCP + science artifacts; build attempt only owes MCP."""
    monkeypatch.setattr(
        "CoScientist.tools.alembic_tools.list_served_mcp_tools",
        lambda *a, **k: [{"name": "synspace_score", "description": "score"}],
    )
    task = _alembic_task()
    task["expected_artifacts"] = [
        {
            "name": "mcp_endpoint",
            "role": "mcp_server",
            "description": "Served Alembic MCP URL",
            "required": True,
        },
        {
            "name": "candidates.csv",
            "role": "data",
            "media_type": "text/csv",
            "description": "Scientific table",
            "required": True,
        },
    ]
    task["success_criteria"] = [
        {
            "criterion_id": "EXP-1-C1",
            "description": "MCP URL ready",
            "kind": "execution",
            "verification": "outputs.mcp_url is an http(s) URL.",
            "required": True,
        },
        {
            "criterion_id": "EXP-1-C2",
            "description": "candidates.csv exists",
            "kind": "artifact_exists",
            "verification": "Confirm output file presence",
            "required": True,
        },
    ]
    plan = _plan(task)
    state: dict = {}
    initialize_runtime(
        state,
        plan,
        critique={"verdict": "approve", "issues": [], "summary": "forced"},
    )
    approve_plan(state)
    settings = ExperimentsSettings(route_alembic=True)
    started = start_task(state, "EXP-1", settings=settings)
    mark_route_returned(state, "McpBuilderAgent")
    recorded = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "success",
            "summary": "Built MCP",
            "outputs": {
                "mcp_url": "http://127.0.0.1:9000/mcp",
                "mcp_endpoint": "http://127.0.0.1:9000/mcp",
            },
            "criteria_checks": [
                {
                    "criterion_id": "EXP-1-C1",
                    "passed": True,
                    "details": "mcp_url present",
                }
            ],
        },
        settings=settings,
    )
    assert recorded["status"] == "success"
    assert recorded["post_build"]["post_build_route"] == "react_tools"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "ready"


def test_await_alembic_job_waits_until_done(monkeypatch):
    from CoScientist.experiments.runtime import await_alembic_job_if_experiment

    state = _alembic_started_state()
    done = {
        "job_id": "dockstring-abc",
        "status": "done",
        "mcp_url": "http://127.0.0.1:9000/mcp",
        "tools": ["dock"],
    }

    def _wait(job_id, **kwargs):
        assert job_id == "dockstring-abc"
        return done

    monkeypatch.setattr(
        "CoScientist.tools.alembic_tools.wait_mcp_build", _wait,
    )
    running = {"job_id": "dockstring-abc", "status": "running"}
    out = await_alembic_job_if_experiment(
        SimpleNamespace(name="build_mcp_server"),
        {"repo_url": "https://github.com/dockstring/dockstring"},
        _tool_context(state),
        running,
    )
    assert out["status"] == "done"
    assert out["mcp_url"] == "http://127.0.0.1:9000/mcp"
    attempt = state["experiment_runtime"]["tasks"]["EXP-1"]["attempts"]
    att = next(iter(attempt.values()))
    assert att["alembic_job_id"] == "dockstring-abc"
    assert att["alembic_snapshot"]["status"] == "done"


def test_auto_record_alembic_success_with_mcp_url():
    from CoScientist.experiments.runtime.guards import (
        _auto_record_result_payload,
        on_route_agent_returned,
    )

    state = _alembic_started_state()
    snap = {
        "job_id": "dockstring-abc",
        "status": "done",
        "mcp_url": "http://127.0.0.1:9000/mcp",
        "tools": ["dock"],
    }
    att_id = state["experiment_runtime"]["active_attempt_id"]
    att = state["experiment_runtime"]["tasks"]["EXP-1"]["attempts"][att_id]
    att["alembic_snapshot"] = snap
    on_route_agent_returned(
        SimpleNamespace(name="McpBuilderAgent"), {}, _tool_context(state), "still running prose"
    )
    stored = state["experiment_last_route_response"]
    assert stored["mcp_url"] == "http://127.0.0.1:9000/mcp"
    payload = _auto_record_result_payload(
        state, state["experiment_runtime"]["tasks"]["EXP-1"], att,
    )
    assert payload["status"] == "success"
    assert payload["outputs"]["mcp_url"] == "http://127.0.0.1:9000/mcp"


def test_critique_approves_alembic_when_enabled_with_repo_and_post_build():
    plan = _plan(_alembic_task())
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(route_alembic=True),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
        repo_candidates=[{"url": "https://github.com/whitead/synspace", "repo_name": "synspace"}],
    )
    assert critique.verdict == "approve"


def test_schema_forbids_alembic_when_code_must_change():
    task = _alembic_task()
    task["code_assessment"] = {
        "requirement": "modify",
        "evidence": "The objective function must be replaced.",
        "entrypoints": ["synspace_score"],
    }
    with pytest.raises(ValidationError, match="requires code_assessment.requirement=reuse"):
        ExperimentTask.model_validate(task)


def test_coder_is_valid_when_repository_code_must_change():
    task = _task("EXP-1", route="coder")
    task["code_assessment"] = {
        "requirement": "modify",
        "evidence": "The objective function must be replaced.",
        "entrypoints": ["train.py"],
    }
    critique = critique_plan(
        _plan(task),
        settings=ExperimentsSettings(route_alembic=True),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    assert critique.verdict == "approve"


def test_alembic_preflight_reports_docker_dns_failure_without_starting_a_job():
    from CoScientist.tools.alembic_tools import alembic_preflight

    calls = []

    def _runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="dial tcp: lookup b.dgx: no such host",
        )

    result = alembic_preflight(runner=_runner)
    assert result["available"] is False
    assert "b.dgx" in result["reason"]
    assert calls[0][0][:2] == ["docker", "info"]


def test_critique_blocks_alembic_repo_not_in_candidates():
    plan = _plan(_alembic_task())
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(route_alembic=True),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
        repo_candidates=[{"url": "https://github.com/other/unrelated"}],
    )
    assert critique.verdict == "revise"
    assert any("not in" in i.message and "repo_candidates" in i.message for i in critique.issues)


def test_critique_blocks_alembic_with_premature_mcp_servers():
    task = _alembic_task()
    task["mcp_servers"] = [
        {
            "name": "synspace",
            "server_id": "synspace",
            "url": "http://example.invalid/mcp",
            "source": "alembic",
            "tools": [{"name": "generate", "description": "x"}],
        }
    ]
    plan = _plan(task)
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(route_alembic=True),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
        repo_candidates=[{"url": "https://github.com/whitead/synspace"}],
    )
    assert critique.verdict == "revise"
    assert any("mcp_servers empty" in i.message for i in critique.issues)


def _fedot_post_build_task() -> dict:
    task = _alembic_task()
    task["post_build_route"] = "fedot_mas"
    return task


def test_critique_blocks_a_fedot_post_build_route_while_fedot_is_off():
    critique = critique_plan(
        _plan(_fedot_post_build_task()),
        settings=ExperimentsSettings(route_alembic=True, route_fedot=False),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
        repo_candidates=[{"url": "https://github.com/whitead/synspace"}],
    )
    assert critique.verdict == "revise"
    issue = next(i for i in critique.issues if "post_build_route" in i.message)
    assert issue.severity == "blocker"
    assert "react_tools" in issue.suggestion


def test_alembic_success_reopens_on_react_tools_while_fedot_is_off():
    """A built server is still used - through ExperimentAgent, not FEDOT.MAS."""
    state: dict = {}
    initialize_runtime(
        state,
        _plan(_fedot_post_build_task()),
        critique={"verdict": "approve", "issues": [], "summary": "forced"},
    )
    approve_plan(state)
    settings = ExperimentsSettings(route_alembic=True, route_fedot=False)
    started = start_task(state, "EXP-1", settings=settings)
    mark_route_returned(state, "McpBuilderAgent")
    recorded = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "success",
            "summary": "Built MCP",
            "outputs": {
                "mcp_url": "http://127.0.0.1:9000/mcp",
                "mcp_endpoint": "http://127.0.0.1:9000/mcp",
                "tools": ["synspace_score"],
            },
            "criteria_checks": [
                {"criterion_id": "EXP-1-C1", "passed": True, "details": "mcp_url present"}
            ],
        },
        settings=settings,
    )
    assert recorded["post_build"]["post_build_route"] == "react_tools"
    task_runtime = state["experiment_runtime"]["tasks"]["EXP-1"]
    assert task_runtime["current_route"] == "react_tools"
    assert task_runtime["task"]["route"] == "react_tools"
    assert task_runtime["route_history"][-1]["route"] == "react_tools"
