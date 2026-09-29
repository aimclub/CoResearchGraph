"""Validation diagnostics are durable data, never live exception objects."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments import review as review_mod
from CoScientist.experiments.critique import (
    PlanValidationError,
    validate_and_critique_plan,
)
from CoScientist.experiments.review import ExperimentReviewSessionAgent

from .helpers import _inventory, _plan, _task


def test_pydantic_value_errors_are_json_native_before_they_reach_state():
    plan = _plan(_task("EXP-1", route="react_tools"))
    payload = plan.model_dump(mode="json")
    payload["tasks"][0]["mcp_servers"][0]["url"] = None

    with pytest.raises(PlanValidationError) as caught:
        validate_and_critique_plan(
            payload,
            settings=ExperimentsSettings(),
            available_tools=_inventory(),
        )

    errors = caught.value.errors
    assert errors and errors[0]["type"] == "value_error"
    assert "MCP server requires url" in errors[0]["msg"]
    assert set(errors[0]) == {"type", "loc", "msg"}
    TypeAdapter(list[dict]).dump_json(errors)


def test_a_schema_valid_revision_clears_stale_schema_errors(monkeypatch):
    plan = _plan(_task("EXP-1", route="coder"))
    state = {
        "experiment_context": {
            "experiment_run_id": plan.experiment_run_id,
            "source_request": plan.source_request,
            "available_mcp_capabilities": _inventory(),
            "operations": [
                {"operation_id": "OP-1", "statement": "Prepare inputs"},
                {"operation_id": "OP-2", "statement": "Run the computation"},
            ],
        },
        "experiment_plan_validation_errors": [{
            "type": "value_error",
            "loc": ["tasks", 0],
            "msg": "stale failure",
        }],
    }
    monkeypatch.setattr(
        review_mod,
        "get_settings",
        lambda: SimpleNamespace(
            experiments=ExperimentsSettings(max_plan_revisions=4)
        ),
    )
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-1")

    response = asyncio.run(agent._review_plan(ctx, plan.model_dump_json()))

    assert response.action.value == "edit"
    assert state["experiment_plan_validation_errors"] is None
    assert state["experiment_plan_critique"]["verdict"] == "revise"
