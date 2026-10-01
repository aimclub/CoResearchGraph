"""plan_to_view: the plan as data, so the web card and the graphs can draw it.

The web UI used to receive the plan only as the Markdown ``render_experiment_plan``
writes for the console — a design matrix flattened into pipe-separated rows
inside a ``<pre>``. These tests pin the structured view that replaced it: every
field a reviewer needs is present and typed, and a slot the planner left empty
comes back empty instead of carrying its placeholder text.
"""
from __future__ import annotations

from CoScientist.experiments.plan_view import (
    plan_headline,
    plan_to_view,
    task_to_view,
)
from CoScientist.experiments.schemas import ExperimentPlan

from .helpers import _design, _plan, _task


def test_the_view_carries_the_plan_header():
    plan = _plan(_task("EXP-1", hypothesis_ref="H1"))

    view = plan_to_view(plan)

    assert view["kind"] == "experiment_plan"
    assert view["status"] == "proposed"
    assert view["plan_id"] == plan.plan_id
    assert view["revision"] == plan.revision
    assert view["goal"] == plan.goal
    assert view["methods"] == list(plan.methods)
    assert view["task_count"] == 1
    assert view["total_est_duration_min"] == plan.total_est_duration_min
    assert view["routes"] == ["fedot_mas"]
    assert view["hypotheses"] == [
        {"id": "H1", "statement": "Fixture statement for H1."}
    ]


def test_every_task_becomes_a_matrix_row_and_a_card():
    plan = _plan(_task("EXP-1"), _task("EXP-2", route="coder"))

    view = plan_to_view(plan)

    assert [row["task_id"] for row in view["matrix"]] == ["EXP-1", "EXP-2"]
    assert [task["id"] for task in view["tasks"]] == ["EXP-1", "EXP-2"]
    assert view["routes"] == ["coder", "fedot_mas"]
    row = view["matrix"][0]
    assert row["hypothesis"] == "H1"
    assert row["dataset"] == "ready_mcp_inputs"
    assert row["baselines"] == ["no-tool control"]
    assert row["metrics"] == ["artifact_present (maximize)"]
    assert row["tools"] == ["chem-ready:estimate_property"]
    assert row["artifacts"] == ["metrics_table.json"]
    assert row["route"] == "fedot_mas"


def test_a_task_card_carries_the_whole_design():
    task = task_to_view(_plan(_task("EXP-1")).tasks[0])

    assert task["route"] == "fedot_mas"
    assert task["est_duration_min"] == 1
    assert task["optional"] is False
    assert task["design"]["question"].startswith("Does the ready MCP")
    assert task["design"]["dataset"]["notes"] == "Synthetic fixture inputs for unit tests."
    assert task["mcp_servers"][0]["url"] == "http://127.0.0.1:8000/mcp"
    assert task["mcp_servers"][0]["tools"][0]["name"] == "estimate_property"
    assert task["success_criteria"][0]["criterion_id"] == "EXP-1-C1"
    assert task["expected_artifacts"][0]["role"] == "data"
    assert task["launch_params"] == {"smiles": "CCO"}


def test_a_threshold_criterion_reads_as_one_expression():
    """The reviewer sees "score >= 0.8", not three fields to recombine."""
    raw = _task("EXP-1")
    raw["success_criteria"] = [{
        "criterion_id": "EXP-1-C1",
        "description": "The score clears the bar.",
        "kind": "threshold",
        "metric": "score",
        "operator": ">=",
        "target": 0.8,
        "verification": "Read the metric off the result table.",
    }]

    task = task_to_view(_plan(raw).tasks[0])

    assert task["success_criteria"][0]["threshold"] == "score >= 0.8"


def test_an_unfilled_design_slot_comes_back_empty_not_as_its_placeholder():
    """"unspecified" in a card reads as a decision; nothing reads as nothing.

    Placeholder is whatever ``is_design_placeholder`` calls one, so the card and
    the critic agree on which slots the planner actually filled.
    """
    design = _design("H1")
    design["experiment_question"] = "unspecified"
    design["dataset"] = {"name": "task dataset", "ref": None, "notes": None}
    design["baselines"] = []
    plan = _plan(_task("EXP-1", design=design))

    view = plan_to_view(plan)

    assert view["tasks"][0]["design"]["question"] is None
    assert view["tasks"][0]["design"]["dataset"]["name"] is None
    assert view["matrix"][0]["question"] is None
    assert view["matrix"][0]["baselines"] == []


def test_an_input_is_named_by_a_location_a_reader_can_resolve():
    """A bare S3 key names no bucket, and an upstream artifact names no task."""
    raw = _task("EXP-2", depends_on=["EXP-1"])
    raw["input_data"] = [
        {"data_id": "RECEPTOR", "kind": "s3", "description": "Prepared receptor.",
         "bucket": "coscientist", "s3_key": "receptors/1q41.pdbqt"},
        {"data_id": "TOP200", "kind": "task_artifact", "description": "Upstream ranking.",
         "source_task_id": "EXP-1", "source_artifact_id": "ranking.csv"},
    ]

    inputs = task_to_view(_plan(_task("EXP-1"), raw).tasks[1])["input_data"]

    assert inputs[0]["location"] == "s3://coscientist/receptors/1q41.pdbqt"
    assert inputs[1]["location"] == "EXP-1/ranking.csv"


def test_a_rationale_that_only_repeats_the_description_is_dropped():
    raw = _task("EXP-1")
    raw["rationale"] = raw["description"]

    task = task_to_view(_plan(raw).tasks[0])

    assert task["rationale"] is None
    assert task["description"] == raw["description"]


def test_long_free_text_is_cut_rather_than_shipped_whole():
    raw = _task("EXP-1")
    raw["description"] = "word " * 800

    task = task_to_view(_plan(raw).tasks[0])

    assert len(task["description"]) <= 1200
    assert task["description"].endswith("…")


def test_the_critique_travels_with_the_plan():
    plan = _plan(_task("EXP-1"))
    critique = {
        "verdict": "revise",
        "issues": [{
            "issue_id": "I-1", "severity": "blocker", "category": "coverage",
            "task_id": "EXP-1", "message": "No baseline.",
            "suggestion": "Name one.",
        }],
    }

    view = plan_to_view(plan, critique, status="revision_requested")

    assert view["status"] == "revision_requested"
    assert view["critique"]["verdict"] == "revise"
    assert view["critique"]["issues"][0]["message"] == "No baseline."


def test_no_critique_is_absent_rather_than_an_empty_verdict():
    assert plan_to_view(_plan(_task("EXP-1")))["critique"] is None
    assert plan_to_view(_plan(_task("EXP-1")), {})["critique"] is None


def test_the_view_is_json_serialisable():
    """It travels over the HITL websocket and into a graph node."""
    import json

    view = plan_to_view(_plan(_task("EXP-1"), _task("EXP-2", route="coder")))

    assert json.loads(json.dumps(view))["task_count"] == 2


def test_the_headline_says_what_a_graph_card_has_room_for():
    view = plan_to_view(_plan(_task("EXP-1"), _task("EXP-2")))

    assert plan_headline(view) == "plan rev 1 · 2 tasks · 2 min"
    assert plan_headline(plan_to_view(_plan(_task("EXP-1")))).endswith("1 task · 1 min")


def test_an_alembic_build_task_keeps_the_repo_it_will_build():
    raw = _task("EXP-1", route="alembic_build")
    raw["repo_url"] = "https://github.com/example/tool"
    raw["post_build_route"] = "react_tools"
    raw["code_assessment"] = {
        "requirement": "reuse",
        "evidence": "Existing tool entrypoint covers the operation unchanged.",
        "entrypoints": ["tool.run"],
    }
    plan = ExperimentPlan.model_validate(_plan(raw).model_dump(mode="json"))

    task = plan_to_view(plan)["tasks"][0]

    assert task["repo_url"] == "https://github.com/example/tool"
    assert task["post_build_route"] == "react_tools"
