"""Deterministic plan critique and result rendering."""
from __future__ import annotations

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.critique import critique_plan
from CoScientist.experiments.schemas import ExperimentPlan

from .helpers import (
    _design,
    _inventory,
    _plan,
    _task,
)

def test_deterministic_critique_blocks_disabled_and_unknown_routes():
    plan = _plan(_task("EXP-1"))
    disabled = critique_plan(
        plan,
        settings=ExperimentsSettings(route_fedot=False),
        available_tools=_inventory(),
    )
    assert disabled.verdict == "revise"
    assert any(issue.category == "feasibility" for issue in disabled.issues)
    # The fix is named: the same bound tool, run by ExperimentAgent.
    blocker = next(i for i in disabled.issues if "'fedot_mas' is switched off" in i.message)
    assert blocker.severity == "blocker"
    assert "route=react_tools" in blocker.suggestion

    unknown = critique_plan(
        plan,
        settings=ExperimentsSettings(route_fedot=True),
        available_tools=[],
    )
    assert unknown.verdict == "revise"
    assert any("absent from the capability inventory" in issue.message for issue in unknown.issues)


def test_a_switched_off_medical_task_is_told_to_use_external_research(monkeypatch):
    """The critique used to approve medical tasks whatever MEDICAL__ENABLED said,
    and start_task then refused them."""
    from CoScientist.config import get_settings

    task = _task("EXP-1", route="medical")
    task["design"]["analysis_artifacts"] = [{
        "name": "pico.json", "role": "report",
        "prepare_via": "medical", "path_or_tool": "get_pico",
    }]
    plan = _plan(task)
    settings = ExperimentsSettings(route_fedot=True)
    assert critique_plan(plan, settings=settings, available_tools=_inventory()).verdict == "approve"

    monkeypatch.setattr(get_settings().web, "medical_agent_enabled", False)
    off = critique_plan(plan, settings=settings, available_tools=_inventory())
    assert off.verdict == "revise"
    blocker = next(i for i in off.issues if "'medical' is switched off" in i.message)
    assert blocker.severity == "blocker"
    assert "orchestrator's ResearchAgent" in blocker.suggestion
    assert "route=research" not in blocker.suggestion


def test_a_switched_off_medical_task_gets_one_consistent_answer(monkeypatch):
    """With the route off, its own shape checks would point back at it (bind an
    available_medical_capabilities tool), and route=research is no answer when
    the human scope forbids research."""
    from CoScientist.config import get_settings

    task = _task("EXP-1", route="medical")  # no medical family tool bound
    plan = _plan(task)
    settings = ExperimentsSettings(route_fedot=True)
    monkeypatch.setattr(get_settings().web, "medical_agent_enabled", False)

    off = critique_plan(plan, settings=settings, available_tools=_inventory())
    assert not any("without binding a family tool" in i.message for i in off.issues)
    assert not [i.suggestion for i in off.issues if "available_medical" in i.suggestion]

    # Same order as planner rule 3 with the route off, and never "drop the
    # step" - a task covering a frame operation cannot just go.
    blocker = next(i for i in off.issues if "'medical' is switched off" in i.message)
    assert "orchestrator's ResearchAgent" in blocker.suggestion and "route=coder" in blocker.suggestion
    assert "route=research" not in blocker.suggestion
    assert "drop" not in blocker.suggestion.lower()

    scoped = critique_plan(plan, settings=settings, available_tools=_inventory(),
                           pipeline_scope={"research": False})
    blocker = next(i for i in scoped.issues if "'medical' is switched off" in i.message)
    assert "route=research" not in blocker.suggestion
    assert "route=coder" in blocker.suggestion


def test_the_review_passes_the_sessions_answer_not_the_switch_as_it_is_now():
    """A session built without MedicalAgent must not have a medical plan
    approved because the switch went on afterwards - start_task would refuse it."""
    task = _task("EXP-1", route="medical")
    task["design"]["analysis_artifacts"] = [{
        "name": "pico.json", "role": "report",
        "prepare_via": "medical", "path_or_tool": "get_pico",
    }]
    settings = ExperimentsSettings(route_fedot=True)
    session_without = critique_plan(_plan(task), settings=settings,
                                    available_tools=_inventory(), medical_on=False)
    assert session_without.verdict == "revise"
    session_with = critique_plan(_plan(task), settings=settings,
                                 available_tools=_inventory(), medical_on=True)
    assert session_with.verdict == "approve"


def test_a_coder_task_naming_a_medical_tool_is_left_alone_while_medical_is_off(monkeypatch):
    """With MedicalAgent in the run, Coder must not reimplement its tools; with
    it out, pointing the planner at route=medical would ask for a refused route."""
    from CoScientist.config import get_settings

    coder = _task("EXP-1", route="coder")
    coder["description"] = "Extract PICO from the supplied abstract with get_pico."
    settings = ExperimentsSettings(route_fedot=True)

    on = critique_plan(_plan(coder), settings=settings, available_tools=_inventory())
    family = next(i for i in on.issues if "reimplement that family" in i.message)
    assert "route=medical" in family.suggestion

    monkeypatch.setattr(get_settings().web, "medical_agent_enabled", False)
    off = critique_plan(_plan(coder), settings=settings, available_tools=_inventory())
    assert not any("reimplement that family" in i.message for i in off.issues)
    assert not [i.suggestion for i in off.issues if "medical" in i.suggestion.lower()]


def test_revision_suggestions_never_steer_toward_fedot():
    """Suggestions reach the planner verbatim in its revision round; they used
    to name fedot_mas first, and did so even with the route switched off."""
    coder = _task("EXP-1", route="coder")
    coder["description"] = "Reimplement estimate_property in a script."
    for route_fedot in (False, True):
        critique = critique_plan(
            _plan(coder),
            settings=ExperimentsSettings(route_fedot=route_fedot),
            available_tools=_inventory(),
        )
        assert any("reimplement a ready MCP" in i.message for i in critique.issues)
        assert not [i.suggestion for i in critique.issues if "fedot" in i.suggestion.lower()]


def test_completeness_critique_rejects_when_request_explicitly_requires_unused_tools():
    overview = _task("EXP-1", tool="dataset_overview")
    overview["mcp_servers"][0]["server_id"] = "srv-heracleum"
    overview["mcp_servers"][0]["tools"][0] = {
        "name": "dataset_overview",
        "description": "Overview of the reconstructed metabolite dataset.",
        "input_schema": {},
    }
    plan = ExperimentPlan.model_validate(
        {
            **_plan(overview).model_dump(mode="json"),
            "source_request": (
                "Use dataset_overview, then call chemical_space_cluster, "
                "run predict_ld50, and invoke predict_general_toxicity on the panel."
            ),
        }
    )
    inventory = [
        {
            "tool": "dataset_overview",
            "server_id": "srv-heracleum",
            "description": "Overview of the reconstructed metabolite dataset.",
        },
        {
            "tool": "chemical_space_cluster",
            "server_id": "srv-heracleum",
            "description": "Cluster metabolites by molecular similarity.",
        },
        {
            "tool": "predict_ld50",
            "server_id": "srv-heracleum",
            "description": "Impute mouse LD50 across administration routes.",
        },
        {
            "tool": "predict_general_toxicity",
            "server_id": "srv-heracleum",
            "description": "Hepatotoxicity, DILI, cardiotoxicity, carcinogenicity.",
        },
    ]
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=inventory,
    )
    assert critique.verdict == "revise"
    assert any(issue.category == "completeness" for issue in critique.issues)
    assert any("chemical_space_cluster" in issue.message for issue in critique.issues)


def test_completeness_critique_ignores_incidental_tool_name_mentions():
    """Same-domain tool names in prose must not force revise (Option A)."""
    plan = ExperimentPlan.model_validate(
        {
            **_plan(_task("EXP-1")).model_dump(mode="json"),
            "source_request": (
                "Run a statistical comparison of cleaned vs raw splits. "
                "The registry may contain chemical_space_cluster as a nearby "
                "chemistry tool, but do not stretch it onto this coder analysis."
            ),
        }
    )
    inventory = _inventory() + [
        {
            "tool": "chemical_space_cluster",
            "server_id": "srv-chem",
            "description": "Cluster metabolites by molecular similarity.",
        },
    ]
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(route_fedot=True),
        available_tools=inventory,
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    assert critique.verdict == "approve"
    assert not any(
        issue.severity in {"blocker", "major"} and "chemical_space_cluster" in issue.message
        for issue in critique.issues
    )


def test_completeness_critique_keeps_single_capability_plans():
    plan = _plan(_task("EXP-1"))
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(route_fedot=True),
        available_tools=_inventory(),
    )
    assert critique.verdict == "approve"
    assert not any(issue.category == "completeness" for issue in critique.issues)


def test_completeness_critique_allows_named_tool_alternatives():
    plan = ExperimentPlan.model_validate(
        {
            **_plan(_task("EXP-1", tool="generate_case_mols")).model_dump(mode="json"),
            "source_request": (
                "Generate candidates with generate_case_mols for alzheimer; "
                "otherwise generate_mols."
            ),
        }
    )
    inventory = [
        {
            "tool": "generate_case_mols",
            "server_id": "srv-gen",
            "description": "Case-conditioned molecule generation.",
        },
        {
            "tool": "generate_mols",
            "server_id": "srv-gen",
            "description": "Generic molecule generation.",
        },
    ]
    plan.tasks[0].mcp_servers[0].server_id = "srv-gen"
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(route_fedot=True),
        available_tools=inventory,
        preferred_tools=inventory,
    )
    assert critique.verdict == "approve"
    assert not any(
        issue.severity in {"blocker", "major"} and issue.category == "completeness"
        for issue in critique.issues
    )


def test_orphan_hypotheses_are_critique_majors():
    """Orphans are left for critique — no silent auto-link onto the first task."""
    plan = _plan(
        _task("EXP-1"),
        hypotheses=[
            {"hypothesis_id": "H1", "statement": "Primary claim."},
            {"hypothesis_id": "H2", "statement": "Secondary claim."},
        ],
    )
    # No context hypothesis_refs → plan.hypotheses orphans are critique majors.
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=_inventory(),
    )
    assert critique.verdict == "revise"
    assert any("not linked from tasks" in i.message for i in critique.issues)


def test_critique_requires_hypothesis_coverage_and_blocks_empty_inventory_mcp():
    coder_task = _task("EXP-1", route="coder", hypothesis_ref="H1")
    plan = _plan(
        coder_task,
        hypotheses=[
            {"hypothesis_id": "H1", "statement": "Claim one."},
            {"hypothesis_id": "H2", "statement": "Claim two."},
        ],
    )
    uncovered = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=[],
        hypothesis_refs=[
            {"hypothesis_id": "H1", "statement": "Claim one."},
            {"hypothesis_id": "H2", "statement": "Claim two."},
        ],
    )
    assert uncovered.verdict == "revise"
    assert any("H2" in issue.message for issue in uncovered.issues)

    mcp_no_inventory = _plan(_task("EXP-1", route="fedot_mas"))
    blocked = critique_plan(
        mcp_no_inventory,
        settings=ExperimentsSettings(),
        available_tools=[],
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    assert blocked.verdict == "revise"
    assert any("inventory is empty" in issue.message for issue in blocked.issues)

    ok = critique_plan(
        _plan(_task("EXP-1", route="coder")),
        settings=ExperimentsSettings(),
        available_tools=[],
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    assert ok.verdict == "approve"


def test_render_experiment_plan_includes_design_matrix():
    from CoScientist.experiments.review import render_experiment_plan

    text = render_experiment_plan(_plan(_task("EXP-1", route="coder")))
    assert "Design matrix" in text
    assert "`H1`" in text
    assert "Baselines:" in text
    assert "Metrics:" in text
    assert "Analysis artifacts:" in text
    assert "`coder`" in text


def test_critique_blocks_invented_hypothesis_beyond_refs():
    task = _task("EXP-1", hypothesis_ref="H1")
    plan = _plan(
        task,
        hypotheses=[
            {"hypothesis_id": "H1", "statement": "Fixture H1."},
            {"hypothesis_id": "H9", "statement": "Invented extra."},
        ],
    )
    # H9 is in plan but not covered — also invent beyond refs
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture H1."}],
    )
    assert critique.verdict == "revise"
    assert any("invents ids" in i.message for i in critique.issues)


def test_critique_blocks_placeholder_urls_in_input_data():
    task = _task("EXP-1")
    task["input_data"] = [
        {
            "data_id": "ld50",
            "kind": "url",
            "description": "Public LD50 table",
            "url": "https://example.com/public_ld50_data.csv",
            "required": True,
        }
    ]
    plan = _plan(task)
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=_inventory(),
    )
    assert critique.verdict == "revise"
    assert any(
        i.severity == "blocker" and "placeholder" in i.message.lower()
        for i in critique.issues
    )


def test_critique_blocks_fake_s3_artifacts_in_dataset_notes():
    task = _task("EXP-1", route="coder")
    task["design"] = _design("H1")
    task["design"]["dataset"]["notes"] = "Load from s3://artifacts/fake_run/data.csv"
    plan = _plan(task)
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=_inventory(),
    )
    assert critique.verdict == "revise"
    assert any("s3://artifacts" in i.message for i in critique.issues)


def test_render_experiment_results_prefers_http_then_s3_and_sets_manifest():
    from CoScientist.experiments.review import (
        build_experiment_artifacts_manifest,
        render_experiment_results,
    )

    state = {
        "experiment_task_results": [
            {
                "task_id": "EXP-1",
                "status": "done",
                "summary": "ok",
                "route_used": "fedot_mas",
                "artifacts": [
                    {
                        "artifact_id": "ART-1",
                        "name": "candidates.csv",
                        "bucket": None,
                        "s3_key": None,
                        "workspace_path": None,
                        "external_url": "https://storage.example-cdn.test/runs/a/candidates.csv",
                        "media_type": "text/csv",
                    },
                    {
                        "artifact_id": "ART-2",
                        "name": "metrics.json",
                        "bucket": "bkt",
                        "s3_key": "runs/a/metrics.json",
                        "workspace_path": None,
                        "external_url": None,
                        "media_type": "application/json",
                    },
                ],
            }
        ]
    }
    # This case is about which address wins, not about wording, so it pins the
    # language rather than depending on the session default (Russian).
    state["report_language"] = "en"
    text = render_experiment_results(state)
    assert "Canonical artifact locations" in text
    assert "https://storage.example-cdn.test/runs/a/candidates.csv" in text
    assert "s3://bkt/runs/a/metrics.json" in text
    assert "S3://artifacts" not in text
    manifest = state["experiment_artifacts_manifest"]
    assert len(manifest) == 2
    assert manifest == build_experiment_artifacts_manifest(state)
    assert manifest[1]["location"] == "s3://bkt/runs/a/metrics.json"


def test_critique_flags_uncovered_frame_operations():
    plan = ExperimentPlan.model_validate(_plan(_task("EXP-1")).model_dump(mode="json"))
    plan.tasks[0].design.operation_ref = "OP-1"
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
        operations=[
            {"operation_id": "OP-1", "statement": "Review published methods"},
            {"operation_id": "OP-2", "statement": "Fit six predictive models for the endpoint"},
        ],
    )
    assert any(
        i.severity == "major" and "Frame operations uncovered" in i.message and "OP-2" in i.message
        for i in critique.issues
    )


def test_critique_allows_multiple_hypotheses_on_one_operation_via_also_tests():
    task = _task("EXP-1")
    task["design"]["hypothesis_ref"] = "H1"
    task["design"]["also_tests"] = ["H2", "H3"]
    task["design"]["operation_ref"] = "OP-1"
    plan = _plan(
        task,
        hypotheses=[
            {"hypothesis_id": "H1", "statement": "Claim one."},
            {"hypothesis_id": "H2", "statement": "Claim two."},
            {"hypothesis_id": "H3", "statement": "Claim three."},
        ],
    )
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(),
        available_tools=_inventory(),
        hypothesis_refs=[
            {"hypothesis_id": "H1", "statement": "Claim one."},
            {"hypothesis_id": "H2", "statement": "Claim two."},
            {"hypothesis_id": "H3", "statement": "Claim three."},
        ],
        operations=[{"operation_id": "OP-1", "statement": "Suggest molecules."}],
    )
    assert not any("uncovered by non-optional" in i.message for i in critique.issues)
    assert not any("share the same operation_ref" in i.message for i in critique.issues)


def test_a_conclusion_operation_is_left_to_the_reporting_stage():
    """OP-n "draw the conclusion with numbers" has no task by design: the
    aggregator writes it. It must not send the plan into revision."""
    task = _task("EXP-1", route="coder")
    task["design"]["operation_ref"] = "OP-1"
    critique = critique_plan(
        _plan(task),
        settings=ExperimentsSettings(route_fedot=True),
        available_tools=_inventory(),
        operations=[
            {"operation_id": "OP-1", "statement": "Reproduce the authors' result with their code"},
            {"operation_id": "OP-2", "statement": "Draw the conclusion with numbers per horizon"},
        ],
    )
    assert not any("Frame operations uncovered" in i.message for i in critique.issues)
    note = next(i for i in critique.issues if "left to the reporting stage" in i.message)
    assert note.severity == "minor" and "OP-2" in note.message
    still = critique_plan(
        _plan(task),
        settings=ExperimentsSettings(route_fedot=True),
        available_tools=_inventory(),
        operations=[
            {"operation_id": "OP-1", "statement": "Reproduce the authors' result with their code"},
            {"operation_id": "OP-2", "statement": "Train the control models and compare on the same split"},
        ],
    )
    assert any("Frame operations uncovered" in i.message and "OP-2" in i.message for i in still.issues)


def test_a_conclusion_operation_with_glued_constraints_is_still_reporting():
    """Run 11 of the blind Informer check: the frame parser attached the
    request's constraints paragraph ("epochs may be reduced", "save artifacts")
    to OP-3 "make the conclusion", and the training verb in it made the
    critique demand a task for the conclusion again."""
    from CoScientist.experiments.critique.validator import _is_reporting_operation

    statement = (
        "Сделай вывод: подтверждается ли заявление, с числами по каждому горизонту.\n\n"
        "Ограничения: CPU или одна GPU 4 ГБ, несколько часов. Число эпох можно уменьшить, "
        "если протокол одинаков для всех сравниваемых моделей. Все скрипты и метрики сохраняй как артефакты."
    )
    assert _is_reporting_operation(statement)
    assert not _is_reporting_operation("Обучи модель и сделай вывод по каждому горизонту.")


def test_invented_hypothesis_ids_are_dropped_instead_of_costing_a_revision():
    """The planner copies postponed hypotheses from the research overview next
    to the authoritative refs; the plan is normalised, the critique stays
    about the tasks."""
    from CoScientist.experiments.critique import validate_and_critique_plan

    payload = _plan(
        _task("EXP-1", hypothesis_ref="H1"),
        _task("EXP-2", hypothesis_ref="H2", design={**_design("H2"), "also_tests": ["H1", "H3"]}),
        hypotheses=[
            {"hypothesis_id": "H1", "statement": "Authoritative."},
            {"hypothesis_id": "H2", "statement": "Postponed in the graph."},
            {"hypothesis_id": "H3", "statement": "Postponed in the graph."},
        ],
    ).model_dump(mode="json")
    plan, critique = validate_and_critique_plan(
        payload,
        settings=ExperimentsSettings(route_fedot=True),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Authoritative."}],
    )
    assert [h.hypothesis_id for h in plan.hypotheses] == ["H1"]
    assert plan.tasks[1].design.hypothesis_ref == "H1"
    assert plan.tasks[1].design.also_tests == []
    assert not any("invents ids" in issue.message for issue in critique.issues)


def test_an_ineligible_id_named_only_in_also_tests_is_dropped():
    """The plan's hypothesis list held H1 alone and a task named H2 in
    also_tests; start_task refused the task and amend_task cannot edit
    also_tests, so the executor stopped."""
    from CoScientist.experiments.critique import validate_and_critique_plan

    payload = _plan(
        _task("EXP-1", hypothesis_ref="H1", design={**_design("H1"), "also_tests": ["H2"]}),
        hypotheses=[{"hypothesis_id": "H1", "statement": "Authoritative."}],
    ).model_dump(mode="json")
    plan, _ = validate_and_critique_plan(
        payload,
        settings=ExperimentsSettings(route_fedot=True),
        available_tools=_inventory(),
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Authoritative."}],
    )
    assert plan.tasks[0].design.hypothesis_ref == "H1"
    assert plan.tasks[0].design.also_tests == []


def test_a_directory_named_as_an_artifact_is_refused():
    """record_result registers files; a required grid_data/ is never found and
    the tasks that need it are blocked with the producer terminal."""
    task = _task("EXP-1")
    task["expected_artifacts"] = [
        {"name": "grid_data/", "role": "data", "media_type": "text/csv", "required": True, "description": "grid"},
    ]
    plan = _plan(task)
    critique = critique_plan(plan, settings=ExperimentsSettings(route_fedot=True), available_tools=_inventory())
    assert any("names a directory" in issue.message and issue.severity == "major" for issue in critique.issues)
