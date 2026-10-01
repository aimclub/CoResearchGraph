"""The preview script's sample plan has to stay a plan the schema accepts.

``scripts/preview_plan_card.py`` renders the review card without a run, and is
what a demo of the card is shown from. Its sample plan is written by hand, so a
schema change would break it silently — nobody runs a preview script in CI, and
the failure surfaces in front of an audience. The node rendering is not tested
here (node is not a test dependency); the Python half is.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from CoScientist.experiments.schemas import ExperimentPlan

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "preview_plan_card.py"


@pytest.fixture(scope="module")
def preview():
    spec = importlib.util.spec_from_file_location("preview_plan_card", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sample_is_a_plan_the_schema_accepts(preview):
    plan = ExperimentPlan.model_validate(preview._sample_plan())

    assert len(plan.tasks) == 4
    assert {t.route.value for t in plan.tasks} == {
        "fedot_mas", "alembic_build", "react_tools", "coder"}


def test_the_sample_shows_the_shapes_a_reviewer_must_tell_apart(preview):
    """A ready MCP route, a build with no tool yet, a waiting task, an optional
    one — a sample where every card looks the same demonstrates nothing."""
    plan = ExperimentPlan.model_validate(preview._sample_plan())
    by_id = {task.id: task for task in plan.tasks}

    assert by_id["EXP-1"].mcp_servers[0].tools[0].name == "estimate_property"
    assert by_id["EXP-2"].repo_url and not by_id["EXP-2"].mcp_servers
    assert by_id["EXP-3"].depends_on == ["EXP-1", "EXP-2"]
    assert by_id["EXP-3"].input_data and by_id["EXP-3"].warnings
    assert by_id["EXP-4"].optional is True


def test_loading_with_no_file_yields_the_sample_as_a_view(preview):
    view, critique, plan = preview._load(None)

    assert view["kind"] == "experiment_plan"
    assert view["task_count"] == 4
    assert critique["verdict"] == "approve"
    assert plan is not None                      # so --before can render too


def test_a_session_state_dump_is_read_through_its_runtime(preview, tmp_path):
    import json

    plan = preview._sample_plan()
    dump = tmp_path / "state.json"
    dump.write_text(json.dumps({
        "experiment_runtime": {"plan": plan},
        "experiment_plan_critique": {"verdict": "revise", "issues": []},
    }), encoding="utf-8")

    view, critique, _ = preview._load(dump)

    assert view["plan_id"] == plan["plan_id"]
    assert critique["verdict"] == "revise"
    assert view["critique"]["verdict"] == "revise"


def test_a_plan_view_is_taken_as_it_is(preview, tmp_path):
    """The execution graph's decision node carries one; it renders directly."""
    import json

    from CoScientist.experiments.plan_view import plan_to_view

    view = plan_to_view(ExperimentPlan.model_validate(preview._sample_plan()))
    dump = tmp_path / "view.json"
    dump.write_text(json.dumps(view, ensure_ascii=False), encoding="utf-8")

    loaded, _critique, plan = preview._load(dump)

    assert loaded == view
    assert plan is None                          # nothing to render "before" from
