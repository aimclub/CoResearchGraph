"""Q/H/DL stay one chain: plan, recorded result, graph, report.

A task being done is not a closed requirement. The graph assertions publish a
result the runtime actually stored.
"""
from __future__ import annotations

import pytest

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.critique.validator import critique_plan
from CoScientist.experiments.plan_view import plan_to_view
from CoScientist.experiments.review import render_experiment_plan, render_experiment_results
from CoScientist.experiments.runtime import record_result, start_task
from CoScientist.experiments.runtime.state_machine import get_experiment_plan
from CoScientist.requirements.coverage import project_requirements

from .helpers import _approved_state, _design, _inventory, _plan, _route_return, _success_result, _task
from .test_graph_bridge import _nodes_by_type, _seeded_store

_SETTINGS = ExperimentsSettings(route_fedot=True)


@pytest.fixture(autouse=True)
def _fedot_attached(monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().web, "fedot_fallback_enabled", True)
    monkeypatch.setattr(get_settings().experiments, "route_fedot", True)


def _bare(task_id: str, **design_extra):
    design = _design("")
    design["hypothesis_ref"] = ""
    design.update(design_extra)
    return _task(task_id, hypothesis_ref="", design=design)


def _refs(*rows: dict) -> list[dict]:
    return list(rows)


def _question():
    return {
        "id": "Q-1", "kind": "question", "formulation": "Which compounds bind?",
        "expected": "a grounded answer", "completion": "an answer with evidence",
    }


def _hypothesis():
    return {"hypothesis_id": "H1", "statement": "Compound X inhibits the target."}


def _deliverable():
    return {
        "id": "DL-1", "kind": "deliverable", "formulation": "Ranked candidate table",
        "expected": "the table", "completion": "the table is delivered",
    }


def _grounded(task_id: str, **extra) -> dict:
    payload = _success_result(task_id)
    payload.update(extra)
    return payload


def _finish(plan, task_id: str, payload: dict, context: dict, workspace_path=None):
    state = _approved_state(plan)
    state["experiment_context"] = context
    state["report_language"] = "en"
    started = start_task(state, task_id)
    _route_return(state, "FedotAgent")
    if payload.get("status") != "failure":
        state["fedot_artifacts"] = [{
            "name": f"{task_id.lower()}-result.csv",
            "bucket": "managed-experiments",
            "s3_key": f"experiments/run/{task_id}/result.csv",
            "tool": "estimate_property",
            **({"workspace_path": str(workspace_path)} if workspace_path else {}),
        }]
    stored = record_result(state, task_id, started["attempt_id"], payload)
    return state, stored


def _by_id(rows):
    return {row["id"]: row for row in rows}


def test_invalid_saved_outcome_remains_open_without_breaking_other_requirements():
    from CoScientist.requirements.coverage import assess_target
    from CoScientist.requirements.routing import normalize_statement

    source = "Provide the table."
    part = normalize_statement(source, {"parts": [{"id": "DL-1", "kind": "deliverable",
        "formulation": source, "source": "user_request", "quote": source}]}).parts[0].model_dump()
    outcome = {"status": "success", "summary": "Table saved.",
               "outputs": {"produced": "a recipe and balance"}, "artifacts": [{"artifact_id": "ART-table"}]}
    status, actual, grounds, debt = assess_target("deliverable", [outcome], part)
    assert status == "open" and actual == "Table saved." and grounds == ""
    assert debt == "invalid outcome fields: produced"
    good = {"status": "success", "outputs": {"answer": "10", "answer_grounded": True}}
    assert assess_target("question", [good], {**part, "kind": "question"})[0] == "met"


def test_a_question_closes_only_on_a_grounded_answer():
    task = _bare("EXP-1", target_refs=["Q-1"], step_key="answer")
    plan = _plan(task, hypotheses=[])
    critique = critique_plan(
        plan, settings=_SETTINGS, hypothesis_refs=[],
        requirement_refs=[_question()], available_tools=_inventory(),
    )
    assert [i.message for i in critique.issues if i.severity in {"blocker", "major"}] == []
    state, _stored = _finish(
        plan, "EXP-1", _grounded("EXP-1", summary="Compound CCO binds."),
        {"requirement_refs": [_question()]},
    )
    row = _by_id(project_requirements(
        plan.tasks, [_question()], state["experiment_runtime"]["results"],
    ))["Q-1"]
    assert row["status"] == "met"
    assert row["tasks"] == ["EXP-1"]
    assert "CCO" in row["actual"]
    assert row["debt"] == ""
    report = render_experiment_results(state)
    assert "`Q-1`" in report and "status: met" in report


def test_a_hypothesis_closes_on_a_grounded_verdict_including_refutation():
    task = _task("EXP-1", hypothesis_ref="H1")
    plan = _plan(task)
    confirmed, _stored = _finish(
        plan, "EXP-1",
        _grounded("EXP-1", summary="Inhibition was measured.", outputs={"verdict": "confirmed"}),
        {"hypothesis_refs": [_hypothesis()]},
    )
    row = _by_id(project_requirements(
        plan.tasks, [_hypothesis()], confirmed["experiment_runtime"]["results"],
    ))["H1"]
    assert row["status"] == "met"
    assert row["actual"] == "confirmed"

    refuted_state, _stored = _finish(
        plan, "EXP-1",
        _grounded("EXP-1", summary="The dose response was flat.", outputs={"verdict": "refuted"}),
        {"hypothesis_refs": [_hypothesis()]},
    )
    refuted = _by_id(project_requirements(
        plan.tasks, [_hypothesis()], refuted_state["experiment_runtime"]["results"],
    ))["H1"]
    assert refuted["status"] == "met"
    assert refuted["actual"] == "refuted"

    bare = {
        "task_id": "EXP-1", "status": "success", "summary": "refuted",
        "outputs": {"verdict": "refuted"}, "criteria_checks": [],
    }
    ungrounded = _by_id(project_requirements(plan.tasks, [_hypothesis()], [bare]))["H1"]
    assert ungrounded["status"] == "open"


def test_a_deliverable_stays_partial_until_the_ordered_result_is_complete(tmp_path):
    task = _bare("EXP-1", target_refs=["DL-1"], step_key="deliver")
    plan = _plan(task, hypotheses=[])
    path = tmp_path / "exp-1-result.csv"
    path.write_text("compound,score\nCCO,-1\n")
    state, _stored = _finish(
        plan, "EXP-1", _grounded("EXP-1", summary="The ranking was written."),
        {"requirement_refs": [_deliverable()]}, workspace_path=path,
    )
    met = _by_id(project_requirements(
        plan.tasks, [_deliverable()], state["experiment_runtime"]["results"],
    ))["DL-1"]
    assert met["status"] == "met"

    partial = {
        "task_id": "EXP-1", "status": "partial", "summary": "Only the first page.",
        "artifacts": [{"name": "ranking.csv", "artifact_id": "ART-1", "workspace_path": str(path)}],
        "criteria_checks": [{
            "criterion_id": "DL-volume", "passed": False, "details": "10 of 50 rows",
        }],
    }
    row = _by_id(project_requirements(plan.tasks, [_deliverable()], [partial]))["DL-1"]
    assert row["status"] == "partial"
    assert row["debt"]


def test_a_successful_preparatory_step_does_not_close_the_requirement():
    prep = _bare("EXP-1", target_refs=["DL-1"], step_key="prep")
    final = _bare("EXP-2", target_refs=["DL-1"], step_key="deliver")
    final["depends_on"] = ["EXP-1"]
    plan = _plan(prep, final, hypotheses=[])
    state, _stored = _finish(
        plan, "EXP-1", _grounded("EXP-1", summary="Inputs are prepared."),
        {"requirement_refs": [_deliverable()]},
    )
    row = _by_id(project_requirements(
        plan.tasks, [_deliverable()], state["experiment_runtime"]["results"],
    ))["DL-1"]
    assert row["tasks"] == ["EXP-1", "EXP-2"]
    assert row["preparatory_tasks"] == ["EXP-1"]
    assert row["closing_tasks"] == ["EXP-2"]
    assert row["status"] == "open"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "done"


def test_a_tool_error_does_not_close_a_question_or_refute_a_hypothesis():
    design = _design("H1")
    design["target_refs"] = ["Q-1", "H1"]
    task = _task("EXP-1", design=design)
    plan = _plan(task)
    payload = {
        "status": "failure",
        "error_code": "service_unavailable",
        "error_message": "connection refused",
        "summary": "connection refused",
        "retryable": False,
        "outputs": {"verdict": "refuted"},
        "criteria_checks": [{
            "criterion_id": "EXP-1-C1", "passed": False, "details": "the server did not answer",
        }],
    }
    state, stored = _finish(
        plan, "EXP-1", payload,
        {"requirement_refs": [_question()], "hypothesis_refs": [_hypothesis()]},
    )
    assert stored["task_result"]["status"] == "failure"
    rows = _by_id(state["experiment_runtime"]["requirement_projection"])
    assert rows["Q-1"]["status"] == "open"
    assert rows["H1"]["status"] == "open"
    assert rows["H1"]["actual"] != "refuted"


def test_unknown_targets_are_rejected_and_one_requirement_may_have_many_tasks():
    missing = _bare("EXP-1", target_refs=["Q-9"])
    missed = critique_plan(
        _plan(missing, hypotheses=[]),
        settings=_SETTINGS,
        hypothesis_refs=[],
        requirement_refs=[_question()],
        available_tools=_inventory(),
    )
    assert any("unknown requirement Q-9" in i.message for i in missed.issues)

    first = _bare("EXP-1", target_refs=["Q-1", "DL-1"], step_key="screen")
    second = _bare("EXP-2", target_refs=["Q-1"], step_key="answer")
    shared = critique_plan(
        _plan(first, second, hypotheses=[]),
        settings=_SETTINGS,
        hypothesis_refs=[],
        requirement_refs=[_question(), _deliverable()],
        available_tools=_inventory(),
    )
    majors = [i.message for i in shared.issues if i.severity in {"blocker", "major"}]
    assert majors == []


def test_plan_graph_and_report_share_one_published_result(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_to_graph,
        publish_result_to_graph,
        refresh_requirement_assessment,
    )

    store = _seeded_store(tmp_path)
    root = store.root_id()
    framed = store.commit(
        source="ExperimentModule",
        nodes=[
            {"type": "ResearchQuestion", "ref": "binding_question", "status": "open",
             "attrs": {"formulation": "Which compounds bind?", "stable_id": "Q-1"}},
            {"type": "Deliverable", "ref": "ranked_table", "status": "specified",
             "attrs": {"formulation": "Ranked candidate table", "stable_id": "DL-1"}},
        ],
        edges=[
            {"type": "contains", "from": root, "to": "#binding_question"},
            {"type": "asks_for", "from": "#binding_question", "to": "#ranked_table"},
        ],
        enforce_permissions=False,
    )
    assert framed.ok, framed.errors

    design = _design("H1")
    design["target_refs"] = ["Q-1", "H1", "DL-1"]
    plan = _plan(_task("EXP-1", design=design))
    context = {
        "requirement_refs": [_question(), _deliverable()],
        "hypothesis_refs": [_hypothesis()],
    }
    path = tmp_path / "exp-1-result.csv"
    path.write_text("compound,score\nCCO,-1\n")
    state, stored = _finish(
        plan, "EXP-1",
        _grounded(
            "EXP-1",
            summary="Compound CCO binds and the table is written.",
            outputs={"verdict": "confirmed"},
        ),
        context, workspace_path=path,
    )
    publish_plan_to_graph(store, state)
    publish_result_to_graph(store, state, "EXP-1", stored["task_result"])

    rows = _by_id(state["experiment_runtime"]["requirement_projection"])
    view = plan_to_view(
        plan, requirement_refs=context["requirement_refs"] + context["hypothesis_refs"],
        results=state["experiment_runtime"]["results"],
    )
    view_rows = _by_id(view["requirements"])
    report = render_experiment_results(state)
    text_plan = render_experiment_plan(
        plan, "en", context["requirement_refs"] + context["hypothesis_refs"],
    )
    evidence = [
        node for node in store.full()["nodes"] if node["type"] == "Evidence"
    ]
    assert len(evidence) == 1
    assessment = _by_id(evidence[0]["attrs"]["requirement_assessment"])
    edges = store.full()["edges"]
    question_id = next(
        node["id"] for node in store.full()["nodes"]
        if (node.get("attrs") or {}).get("stable_id") == "Q-1"
    )
    deliverable_id = next(
        node["id"] for node in store.full()["nodes"]
        if (node.get("attrs") or {}).get("stable_id") == "DL-1"
    )
    for ref in ("Q-1", "H1", "DL-1"):
        assert rows[ref]["status"] == "met"
        assert view_rows[ref]["status"] == rows[ref]["status"]
        assert assessment[ref]["status"] == rows[ref]["status"]
        assert f"`{ref}`" in report and f"status: {rows[ref]['status']}" in report
        assert f"`{ref}`" in text_plan
    assert any(
        edge["type"] == "answers" and edge["from"] == evidence[0]["id"] and edge["to"] == question_id
        for edge in edges
    )
    assert any(
        edge["type"] == "relates_to" and edge["from"] == evidence[0]["id"] and edge["to"] == "H1"
        for edge in edges
    )
    assert any(edge["type"] == "satisfies" and edge["to"] == deliverable_id for edge in edges)
    assert get_experiment_plan(state)["requirement_projection"]

    publish_result_to_graph(store, state, "EXP-1", stored["task_result"])
    assert len([node for node in store.full()["nodes"] if node["type"] == "Evidence"]) == 1

    state["experiment_runtime"]["results"][-1]["summary"] = "The bound table answers which compounds bind."
    refresh_requirement_assessment(store, state, "EXP-1")
    refreshed = [
        node for node in store.full()["nodes"] if node["type"] == "Evidence"
    ]
    assert len(refreshed) == 1
    again = _by_id(refreshed[0]["attrs"]["requirement_assessment"])
    assert again["Q-1"]["actual"] == "The bound table answers which compounds bind."
    assert _nodes_by_type(store)["Evidence"] == [evidence[0]["id"]]


def test_a_published_tool_error_does_not_settle_the_hypothesis(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_to_graph,
        publish_result_to_graph,
    )

    store = _seeded_store(tmp_path)
    design = _design("H1")
    design["target_refs"] = ["Q-1", "H1"]
    plan = _plan(_task("EXP-1", design=design))
    payload = {
        "status": "failure",
        "error_code": "service_unavailable",
        "error_message": "connection refused",
        "summary": "connection refused",
        "retryable": False,
        "outputs": {"verdict": "refuted"},
        "criteria_checks": [{
            "criterion_id": "EXP-1-C1", "passed": False, "details": "the server did not answer",
        }],
    }
    state, stored = _finish(
        plan, "EXP-1", payload,
        {"requirement_refs": [_question()], "hypothesis_refs": [_hypothesis()]},
    )
    publish_plan_to_graph(store, state)
    before = next(node["status"] for node in store.full()["nodes"] if node["id"] == "H1")
    publish_result_to_graph(store, state, "EXP-1", stored["task_result"])
    after = next(node["status"] for node in store.full()["nodes"] if node["id"] == "H1")
    assert after == before
    assert after != "refuted"
    evidence = [node for node in store.full()["nodes"] if node["type"] == "Evidence"]
    assert evidence
    assert not any(
        edge["type"] == "relates_to" and edge["from"] == evidence[-1]["id"]
        for edge in store.full()["edges"]
    )
