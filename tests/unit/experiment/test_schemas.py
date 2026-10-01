"""Plan/task/MCP/DataRef contract coercions."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.critique import critique_plan
from CoScientist.experiments.schemas import ExperimentPlan, ExperimentTask

from .helpers import (
    _inventory,
    _plan,
    _task,
)

def test_experiment_plan_json_schema_is_openai_structured_output_safe():
    schema = ExperimentPlan.model_json_schema()
    bad: list[str] = []

    def walk(obj: object, path: str = "") -> None:
        if isinstance(obj, dict):
            if obj.get("additionalProperties") is True:
                bad.append(path or "$")
            for key, value in obj.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                walk(value, f"{path}[{index}]")

    walk(schema)
    assert bad == []

    task = ExperimentTask.model_validate(
        {
            **_task("EXP-1"),
            "launch_params": '{"case":"cancer","num":5}',
        "mcp_servers": [
            {
                "name": "generative",
                "server_id": "srv-gen",
                "url": "http://127.0.0.1:8000/mcp",
                "source": "registry",
                "health": "unknown",
                "tools": [
                    {
                        "name": "generate_case_mols",
                        "description": "Generate molecules",
                        "input_schema": '{"type":"object"}',
                        "required_for_task": True,
                    }
                ],
            }
        ],
        }
    )
    assert task.launch_params == {"case": "cancer", "num": 5}
    assert task.mcp_servers[0].tools[0].input_schema == {"type": "object"}


def test_four_experiment_contracts_are_registered():
    from CoScientist.assembly.registry import REGISTRY
    from CoScientist.experiments.schemas import (
        ExperimentTask,
        PlanCritique,
        TaskResult,
    )

    assert REGISTRY.output_schema("experiment_plan") is ExperimentPlan
    assert REGISTRY.output_schema("experiment_task") is ExperimentTask
    assert REGISTRY.output_schema("task_result") is TaskResult
    assert REGISTRY.output_schema("plan_critique") is PlanCritique


def test_contracts_reject_presigned_urls_and_non_dag_plans():
    task = _task("EXP-1")
    task["input_data"] = [
        {
            "data_id": "input",
            "kind": "url",
            "description": "bad canonical URL",
            "url": "https://s3.example/x.csv?X-Amz-Signature=secret",
        }
    ]
    with pytest.raises(ValidationError, match="signing"):
        _plan(task)

    first = _task("EXP-1", depends_on=["EXP-2"])
    second = _task("EXP-2", depends_on=["EXP-1"])
    with pytest.raises(ValidationError, match="DAG"):
        _plan(first, second)


def test_mcp_health_rejects_non_enum_values():
    task = _task("EXP-1")
    task["mcp_servers"][0]["health"] = "active"
    with pytest.raises(ValidationError):
        _plan(task)

    task["mcp_servers"][0]["health"] = "healthy"
    assert _plan(task).tasks[0].mcp_servers[0].health == "healthy"


def test_dataref_requires_canonical_fields():
    from CoScientist.experiments.schemas import DataRef

    with pytest.raises(ValidationError):
        DataRef.model_validate({"description": "no kind or location"})

    # producer string + artifact alias → task_artifact (GLM/weak planner shape).
    ref_alias = DataRef.model_validate({"artifact": "activity_data", "producer": "EXP-2"})
    assert ref_alias.kind == "task_artifact"
    assert ref_alias.source_task_id == "EXP-2"
    assert ref_alias.source_artifact_id == "activity_data"

    ref = DataRef.model_validate(
        {
            "data_id": "activity_data",
            "kind": "task_artifact",
            "description": "Activity table from EXP-2",
            "source_task_id": "EXP-2",
            "source_artifact_id": "activity_data",
        }
    )
    assert ref.kind == "task_artifact"
    assert ref.source_task_id == "EXP-2"


def test_criterion_check_coerces_null_evidence_and_details():
    from CoScientist.experiments.schemas.models import CriterionCheck

    check = CriterionCheck.model_validate(
        {
            "criterion_id": "C1",
            "passed": True,
            "evidence_artifact_ids": None,
            "details": None,
        }
    )
    assert check.evidence_artifact_ids == []
    assert check.details == "n/a"


def test_glm_hypothesis_and_dataref_extras_are_coerced():
    """Weak GLM planners emit falsifiable/tests/summary and input_data.ref/name."""
    from CoScientist.experiments.schemas import DataRef, HypothesisSpec

    h = HypothesisSpec.model_validate(
        {
            "hypothesis_id": "H1",
            "summary": "Metabolites can be clustered by similarity.",
            "falsifiable": "Clusters do not separate toxicity.",
            "falsification_criteria": "No clear clusters.",
            "tests": "Run chemical_space_clustering.",
            "status": "pending",
            "type": "structural",
            "test_strategy": "Cluster chemical space.",
        }
    )
    assert h.statement == "Metabolites can be clustered by similarity."
    assert not hasattr(h, "type") or "type" not in h.model_dump()

    ref = DataRef.model_validate(
        {
            "ref": "T2/clustering_results.json",
            "name": "clustering_results.json",
        }
    )
    assert ref.kind == "task_artifact"
    assert ref.source_task_id == "EXP-2"
    assert ref.source_artifact_id == "clustering_results.json"
    assert ref.data_id == "clustering_results.json"

    # producer/ref as bare task ids (no /artifact) must still coerce.
    bare = DataRef.model_validate(
        {"kind": "task_artifact", "ref": "EXP-T1", "producer": "EXP-T1"}
    )
    assert bare.source_task_id == "EXP-1"
    assert bare.source_artifact_id == "artifact"

    # GLM also puts ExpectedArtifact/task fields on DataRef + EXP-Tn ids.
    ref2 = DataRef.model_validate(
        {
            "kind": "task_artifact",
            "source_task_id": "EXP-T1",
            "role": "data",
            "depends_on": ["T1"],
            "path_or_tool": "dataset_overview.json",
            "location": "ignored_when_path_set.json",
        }
    )
    assert ref2.source_task_id == "EXP-1"
    assert ref2.source_artifact_id == "dataset_overview.json"
    assert ref2.data_id == "dataset_overview.json"

    t2 = _task("EXP-T2", depends_on=[])
    t2["input_data"] = [
        {
            "kind": "workspace",
            "workspace_path": "dataset_overview.json",
            "data_id": "overview",
            "description": "Overview file",
            "depends_on": ["EXP-T1"],
            "uri": "dataset_overview.json",
        }
    ]
    plan = ExperimentPlan.model_validate(
        {
            **_plan(_task("EXP-1"), t2).model_dump(mode="json"),
            "tasks": [_task("EXP-1"), t2],
            "total_est_duration_min": 2,
        }
    )
    assert plan.tasks[1].id == "EXP-2"
    assert plan.tasks[1].depends_on == ["EXP-1"]
    assert plan.tasks[1].input_data[0].workspace_path == "dataset_overview.json"


def test_depends_on_normalizes_t_ids():
    task = _task("EXP-2", depends_on=["T1", "TASK-3"])
    plan = _plan(task, _task("EXP-1"), _task("EXP-3"))
    # rebuild with proper deps
    t2 = _task("EXP-2", depends_on=["T1"])
    t1 = _task("EXP-1")
    plan = ExperimentPlan.model_validate(
        {
            **_plan(t1, t2).model_dump(mode="json"),
            "tasks": [t1, t2],
            "total_est_duration_min": 2,
        }
    )
    assert plan.tasks[1].depends_on == ["EXP-1"]


def test_criterion_rejects_invented_kinds_without_alias_map():
    from CoScientist.experiments.schemas import SuccessCriterion

    with pytest.raises(ValidationError):
        SuccessCriterion.model_validate(
            {
                "criterion_id": "C2",
                "description": "Output mentions candidates",
                "kind": "output_contains",
                "verification": "Check textual output.",
            }
        )
    with pytest.raises(ValidationError):
        SuccessCriterion.model_validate(
            {
                "criterion_id": "C4",
                "description": "SMILES are valid",
                "kind": "output_valid_smiles",
                "verification": "Validate SMILES strings.",
            }
        )

    ok = SuccessCriterion.model_validate(
        {
            "criterion_id": "C1",
            "description": "Artifact produced",
            "kind": "artifact_exists",
            "verification": "Check artifact.",
        }
    )
    assert ok.kind == "artifact_exists"
    assert ok.operator is None

    # Non-threshold criteria silently drop stray metric/operator/target.
    cleaned = SuccessCriterion.model_validate(
        {
            "criterion_id": "C5",
            "description": "Artifact produced",
            "kind": "artifact_exists",
            "operator": "==",
            "verification": "Check artifact.",
        }
    )
    assert cleaned.operator is None
    assert cleaned.metric is None
    assert cleaned.target is None


def test_plan_rejects_non_string_hypothesis_and_coerces_risk_objects():
    task = _task("EXP-1")
    plan_data = _plan(task).model_dump(mode="json")
    plan_data["hypothesis"] = ["H1. Conflict criteria.", "H2. SA vs affinity."]
    with pytest.raises(ValidationError):
        ExperimentPlan.model_validate(plan_data)

    plan_data = _plan(task).model_dump(mode="json")
    plan_data["risks"] = [
        {
            "risk_id": "R1",
            "description": "Too few candidates pass filters.",
            "mitigation": "Relax thresholds.",
        }
    ]
    plan = ExperimentPlan.model_validate(plan_data)
    assert plan.risks == ["Too few candidates pass filters. Mitigation: Relax thresholds."]

    plan_data = _plan(task).model_dump(mode="json")
    plan_data["hypothesis"] = "H1 and H2 are in tension."
    plan_data["risks"] = ["Too few candidates pass filters."]
    plan = ExperimentPlan.model_validate(plan_data)
    assert plan.hypothesis == "H1 and H2 are in tension."
    assert plan.risks == ["Too few candidates pass filters."]


def test_task_level_risks_assumptions_popped_and_hoisted():
    """Plan-only fields on tasks must not burn extra_forbidden revise budget."""
    base = _plan(_task("EXP-1", route="fedot_mas")).model_dump(mode="json")
    task = _task("EXP-1", route="fedot_mas")
    task["risks"] = []
    task["assumptions"] = []
    plan = ExperimentPlan.model_validate({**base, "tasks": [task], "risks": [], "assumptions": []})
    assert plan.risks == []
    assert plan.assumptions == []

    task2 = _task("EXP-1", route="fedot_mas")
    task2["risks"] = ["MCP may time out"]
    task2["assumptions"] = ["Inventory covers generation"]
    plan2 = ExperimentPlan.model_validate(
        {**base, "tasks": [task2], "risks": [], "assumptions": []}
    )
    assert plan2.risks == ["MCP may time out"]
    assert plan2.assumptions == ["Inventory covers generation"]


def test_mcp_empty_expected_artifacts_not_required_report():
    """fedot/react with empty arts get a neutral data placeholder, not *_report."""
    for route in ("fedot_mas", "react_tools"):
        payload = _task("EXP-1", route=route)
        payload["expected_artifacts"] = []
        task = ExperimentTask.model_validate(payload)
        assert len(task.expected_artifacts) >= 1
        primary = task.expected_artifacts[0]
        assert primary.role == "data"
        assert primary.required is True
        assert not str(primary.name).endswith("_report")
        assert primary.media_type != "text/markdown"


def test_plan_duration_aligns_when_nested_under_design():
    """Planner often nests est_duration_min under design; do not burn revise budget."""
    base = _plan(_task("EXP-1")).model_dump(mode="json")
    t1 = _task("EXP-1", route="coder")
    t1.pop("est_duration_min", None)
    t1["design"]["est_duration_min"] = 30
    t2 = _task("EXP-2", route="fedot_mas")
    t2.pop("est_duration_min", None)
    t2["design"]["est_duration_min"] = 45
    plan = ExperimentPlan.model_validate(
        {
            **base,
            "tasks": [t1, t2],
            "total_est_duration_min": 450,
            "hypotheses": [
                {"hypothesis_id": "H1", "statement": "Fixture statement for H1."},
            ],
        }
    )
    assert plan.tasks[0].est_duration_min == 30
    assert plan.tasks[1].est_duration_min == 45
    assert plan.total_est_duration_min == 75


def test_bare_task_payload_fails_strict_plan_schema():
    from CoScientist.experiments.critique import (
        PlanValidationError,
        validate_and_critique_plan,
    )

    bare = _task("EXP-1")
    with pytest.raises(PlanValidationError):
        validate_and_critique_plan(
            bare,
            settings=ExperimentsSettings(),
            available_tools=_inventory(),
            experiment_run_id="EXRUN-acceptance",
            source_request="Run a bounded chemistry MCP experiment.",
        )


def test_expected_artifact_role_is_closed_enum_and_image_artifacts_need_tool_mime():
    from CoScientist.experiments.schemas import ExpectedArtifact

    with pytest.raises(ValidationError):
        ExpectedArtifact.model_validate(
            {
                "name": "overview_plot",
                "role": "visualization",
                "description": "Invented figure.",
            }
        )

    task = _task("EXP-1", tool="dataset_overview")
    task["mcp_servers"][0]["tools"][0] = {
        "name": "dataset_overview",
        "description": "Return tabular summary statistics for the dataset.",
        "input_schema": {},
    }
    task["expected_artifacts"] = [
        {
            "name": "dataset_overview",
            "role": "data",
            "media_type": "application/json",
            "description": "Overview JSON.",
        },
        {
            "name": "dataset_overview_plot",
            "role": "plot",
            "media_type": "image/png",
            "required": True,
            "description": "Invented plot the tool cannot produce.",
        },
    ]
    plan = _plan(task)
    inventory = [
        {
            "tool": "dataset_overview",
            "server_id": "srv-chem",
            "description": "Return tabular summary statistics for the dataset.",
        }
    ]
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(route_fedot=True),
        available_tools=inventory,
    )
    # Soft advisory: media/role mismatch must not force plan revision loops.
    assert critique.verdict == "approve"
    assert any(
        issue.severity == "minor"
        and ("image/plot" in issue.message or "image/*" in issue.message)
        for issue in critique.issues
    )


def test_design_field_coercions_are_llm_tolerant():
    from CoScientist.experiments.schemas import TaskDesign

    design = TaskDesign.model_validate(
        {
            "hypothesis_ref": ["H2", "H3"],
            "experiment_question": "Is SA related to affinity?",
            "dataset": {
                "name": "pubchem",
                "ref": "https://pubchem.ncbi.nlm.nih.gov/",
                "notes": None,
            },
            "baselines": [{"name": "null model", "kind": "method", "ref": None}],
            "metrics": [{"name": "corr", "direction": "compare", "threshold": None, "test": "spearman"}],
            "analysis_artifacts": [
                {"name": "plot.png", "role": "plot", "prepare_via": "coder", "path_or_tool": "plot.png"},
                {"name": "table.csv", "role": "data", "prepare_via": "coder", "path_or_tool": "table.csv"},
            ],
        }
    )
    assert design.hypothesis_ref == "H2"
    assert design.also_tests == ["H3"]
    assert design.covered_hypothesis_ids() == {"H2", "H3"}
    assert design.dataset.ref is not None
    assert design.dataset.ref.kind == "url"
    assert design.analysis_artifacts[0].role == "report"
    assert design.analysis_artifacts[1].role == "metrics_table"

    messy = TaskDesign.model_validate(
        {
            "hypothesis_ref": "H1",
            "experiment_question": "Q",
            "dataset": {
                "name": "x",
                "ref": {
                    "data_id": "pubchem_source",
                    "kind": "external",
                    "source": "url",
                    "path_or_uri": "https://pubchem.ncbi.nlm.nih.gov/",
                },
            },
            "baselines": [{"name": "b", "kind": "method"}],
            "metrics": [{"name": "m", "direction": "maximize"}],
            "analysis_artifacts": [{"name": "a.py", "role": "code", "prepare_via": "coder"}],
        }
    )
    assert messy.dataset.ref is not None
    assert messy.dataset.ref.kind == "url"


def test_task_design_is_required_on_experiment_plan_1_0():
    task = _task("EXP-1")
    del task["design"]
    plan = ExperimentPlan.model_validate(
        {
            **_plan(_task("EXP-1")).model_dump(mode="json"),
            "tasks": [task],
        }
    )
    assert plan.tasks[0].design.hypothesis_ref == "H1"
    assert plan.tasks[0].design.baselines == []
    assert plan.tasks[0].design.metrics == []


def test_experiment_task_does_not_default_missing_route_to_coder():
    from pydantic import ValidationError
    from CoScientist.experiments.schemas.models import ExperimentTask

    with pytest.raises(ValidationError):
        ExperimentTask.model_validate(
            {
                "id": "EXP-1",
                "name": "Data collection",
                "description": "Collect activity data",
                "rationale": "Need dataset",
                "design": {
                    "hypothesis_ref": "H1",
                    "experiment_question": "What activity data exists?",
                    "dataset": {"name": "activity"},
                    "baselines": [{"name": "none", "kind": "method"}],
                    "metrics": [{"name": "n", "direction": "maximize"}],
                    "analysis_artifacts": [
                        {"name": "a.py", "role": "code", "prepare_via": "coder"}
                    ],
                },
                "success_criteria": [
                    {
                        "criterion_id": "C1",
                        "description": "csv exists",
                        "kind": "artifact_exists",
                        "verification": "file",
                    }
                ],
                "expected_artifacts": [
                    {"name": "data.csv", "role": "data", "description": "table"}
                ],
                "est_duration_min": 20,
            }
        )


def test_scientific_check_is_optional_on_task_result():
    from CoScientist.experiments.schemas import TaskResult

    base = {
        "schema_version": "task-result/0.1",
        "result_id": "RES-1",
        "plan_id": "PLAN-acceptance",
        "task_id": "EXP-1",
        "attempt_id": "ATT-1",
        "attempt_no": 1,
        "status": "success",
        "planned_route": "coder",
        "route_used": "coder",
        "started_at": datetime(2026, 7, 31, 18, 0, tzinfo=timezone.utc),
        "finished_at": datetime(2026, 7, 31, 18, 1, tzinfo=timezone.utc),
        "summary": "ok",
        "criteria_checks": [
            {
                "criterion_id": "EXP-1-C1",
                "passed": True,
                "details": "ok",
            }
        ],
    }
    plain = TaskResult.model_validate(base)
    assert plain.scientific_check is None
    with_check = TaskResult.model_validate(
        {
            **base,
            "scientific_check": {
                "hypothesis_ref": "H1",
                "status": "inconclusive",
                "details": "Need more samples.",
            },
        }
    )
    assert with_check.scientific_check is not None
    assert with_check.scientific_check.status == "inconclusive"


def test_mcp_server_ref_coerces_singular_tool_field():
    from CoScientist.experiments.schemas.models import MCPServerRef

    server = MCPServerRef.model_validate(
        {
            "source": "registry",
            "server_id": "srv-x",
            "url": "http://127.0.0.1:8000/mcp",
            "tool": "calculate_docking",
            "health": "unknown",
        }
    )
    assert server.tools[0].name == "calculate_docking"


def test_mcp_server_ref_disallows_composite_server_id():
    from pydantic import ValidationError
    from CoScientist.experiments.schemas.models import MCPServerRef

    with pytest.raises(ValidationError, match="composite"):
        MCPServerRef.model_validate(
            {
                "source": "registry",
                "name": "molgen",
                "server_id": "d36e3d994404e957/generate_case_mols",
                "url": "http://127.0.0.1:8000/mcp",
                "tools": ["generate_case_mols"],
            }
        )


def test_baseline_kind_aliases_and_task_shape_coercion():
    from CoScientist.experiments.schemas.models import ExperimentTask

    task = ExperimentTask.model_validate(
        {
            "id": "TASK-1",
            "name": "PubChem fetch",
            "route": "coder",
            "design": {
                "hypothesis_ref": "H3",
                "experiment_question": "Is PubChem activity data heterogeneous?",
                "dataset": {"name": "PubChem bioassay"},
                "baselines": [{"name": "literature review", "kind": "report"}],
                "metrics": [{"name": "n_compounds", "direction": "maximize"}],
                "analysis_artifacts": [
                    {"name": "fetch.py", "role": "code", "prepare_via": "coder"}
                ],
                "est_duration_min": 45,
            },
        }
    )
    assert task.id == "EXP-1"
    assert task.description == "PubChem fetch"
    assert task.est_duration_min == 45
    assert task.design.baselines[0].kind == "prior_result"
    assert task.success_criteria[0].kind == "artifact_exists"
    assert task.expected_artifacts


def test_plan_methods_filled_when_empty():
    from datetime import datetime, timezone
    from CoScientist.experiments.schemas.models import ExperimentPlan

    plan = ExperimentPlan.model_validate(
        {
            "schema_version": "experiment-plan/1.0",
            "plan_id": "p1",
            "experiment_run_id": "EXRUN-1",
            "revision": 1,
            "source_request": "do 4.2",
            "goal": "multitarget design",
            "methods": [],
            "context_digest": "digest",
            "tasks": [
                {
                    "id": "EXP-1",
                    "name": "Docking filter",
                    "description": "Filter by docking",
                    "rationale": "Need affinity",
                    "route": "coder",
                    "design": {
                        "hypothesis_ref": "H1",
                        "experiment_question": "Docking threshold?",
                        "dataset": {"name": "candidates"},
                        "baselines": [{"name": "random", "kind": "method"}],
                        "metrics": [{"name": "hit_rate", "direction": "maximize"}],
                        "analysis_artifacts": [
                            {"name": "dock.py", "role": "code", "prepare_via": "coder"}
                        ],
                    },
                    "success_criteria": [
                        {
                            "criterion_id": "C1",
                            "description": "report exists",
                            "kind": "artifact_exists",
                            "verification": "file present",
                        }
                    ],
                    "expected_artifacts": [
                        {
                            "name": "dock_report",
                            "role": "report",
                            "description": "docking report",
                        }
                    ],
                    "est_duration_min": 20,
                }
            ],
            "total_est_duration_min": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    assert plan.methods
    assert plan.total_est_duration_min == 20

def test_design_placeholders_are_dropped_not_invented():
    from CoScientist.experiments.review import render_experiment_plan
    from CoScientist.experiments.schemas.models import ExperimentTask

    payload = {
        "id": "EXP-1",
        "name": "Tox screen",
        "route": "coder",
        "design": {
            "hypothesis_ref": "H1",
            "experiment_question": "What measurable outcome does this task produce?",
            "dataset": {"name": "task dataset"},
            "baselines": [{"name": "unspecified", "kind": "method"}],
            "metrics": [{"name": "primary_outcome", "direction": "compare"}],
            "analysis_artifacts": [
                {"name": "analysis.py", "role": "code", "prepare_via": "coder"}
            ],
        },
        "success_criteria": [
            {
                "criterion_id": "C1",
                "description": "report exists",
                "kind": "artifact_exists",
                "verification": "file present",
            }
        ],
        "expected_artifacts": [
            {"name": "tox_report", "role": "report", "description": "tox report"}
        ],
        "est_duration_min": 10,
    }
    task = ExperimentTask.model_validate(payload)
    assert task.design.baselines == []
    assert task.design.metrics == []
    assert task.design.analysis_artifacts == []
    assert task.design.dataset.name == ""
    assert task.design.experiment_question == ""

    plan = _plan({**_task("EXP-1", route="coder"), "design": task.design.model_dump(mode="json")})
    critique = critique_plan(
        plan,
        settings=ExperimentsSettings(lenient_planner=False),
        available_tools=[],
        hypothesis_refs=[{"hypothesis_id": "H1", "statement": "Fixture"}],
    )
    assert critique.verdict == "approve"
    assert all(i.severity == "minor" for i in critique.issues if "design." in i.message)
    text = render_experiment_plan(plan)
    assert "comparative reference method" not in text
    assert "primary_outcome" not in text
    assert "analysis.py" not in text
    assert "| EXP-1 |" in text and "—" in text

