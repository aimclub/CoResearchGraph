"""State machine: start/record/retry/fallback/artifact matching."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.runtime import (
    ExperimentRuntimeError,
    amend_task,
    approve_plan,
    fallback_task,
    guard_route_agent_tool,
    initialize_runtime,
    mark_route_returned,
    on_route_agent_returned,
    record_result,
    retry_task,
    skip_task,
    start_task,
)

from .helpers import (
    NOW,
    _approved_state,
    _design,
    _inventory,
    _plan,
    _route_return,
    _success_result,
    _task,
    _tool_context,
)

def test_scenario_a_ready_chemistry_mcp_defaults_to_one_fedot_attempt_and_artifact():
    """§11.6 A: ready MCP -> FEDOT -> one guarded call -> managed artifact."""
    plan = _plan(_task("EXP-1"))
    assert plan.tasks[0].route.value == "fedot_mas"
    state = _approved_state(plan)

    started = start_task(state, "EXP-1")
    assert started["route"] == "fedot_mas"
    assert started["route_agent"] == "FedotAgent"
    attempt_id = started["attempt_id"]
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["attempts"][attempt_id][
        "route_returned"
    ] is False

    _route_return(state, "FedotAgent")
    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/run/plan/EXP-1/attempt/result.csv",
            "tool": "estimate_property",
        }
    ]
    stored = record_result(state, "EXP-1", attempt_id, _success_result("EXP-1"))
    result = stored["task_result"]

    assert result["status"] == "success"
    assert result["route_used"] == "fedot_mas"
    assert len(result["artifacts"]) == 1
    artifact = result["artifacts"][0]
    assert artifact["bucket"] == "managed-experiments"
    assert artifact["plan_id"] == plan.plan_id
    assert artifact["task_id"] == "EXP-1"
    assert artifact["attempt_id"] == attempt_id
    assert state["experiment_runtime"]["phase"] == "reporting"


def test_start_task_rejects_a_hypothesis_outside_the_eligible_context():
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H2")))
    state["experiment_context"] = {
        "hypothesis_refs": [{"hypothesis_id": "H1", "statement": "active"}],
    }

    with pytest.raises(ExperimentRuntimeError) as raised:
        start_task(state, "EXP-1")

    assert raised.value.code == "hypothesis_not_eligible"
    assert "H2" in str(raised.value)


def test_scenario_b_two_sequential_fedot_tasks_and_duplicate_route_refused():
    """§11.6 B: no session hard-stop; second call in one attempt is refused."""
    plan = _plan(
        _task("EXP-1"),
        _task("EXP-2", depends_on=["EXP-1"]),
    )
    state = _approved_state(plan)
    state["fedot_artifacts"] = [
        {
            "name": "old.csv",
            "bucket": "managed-experiments",
            "s3_key": "old/other-attempt.csv",
        }
    ]

    first = start_task(state, "EXP-1")
    _route_return(state, "FedotAgent")
    duplicate = guard_route_agent_tool(
        SimpleNamespace(name="FedotAgent"), {}, _tool_context(state)
    )
    assert duplicate["status"] == "refused"
    assert duplicate["error_code"] == "route_already_returned"
    state["fedot_artifacts"].append(
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-1/result.csv",
        }
    )
    record_result(state, "EXP-1", first["attempt_id"], _success_result("EXP-1"))

    # A legacy session flag must not prevent a distinct ready attempt.
    state["fedot_deliverable_ready"] = True
    second = start_task(state, "EXP-2")
    assert second["route_agent"] == "FedotAgent"
    _route_return(state, "FedotAgent")
    state["fedot_artifacts"].append(
        {
            "name": "exp-2-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-2/result.csv",
        }
    )
    record_result(state, "EXP-2", second["attempt_id"], _success_result("EXP-2"))

    results = state["experiment_task_results"]
    assert [result["task_id"] for result in results] == ["EXP-1", "EXP-2"]
    assert results[0]["artifacts"][0]["s3_key"].endswith("EXP-1/result.csv")
    assert results[1]["artifacts"][0]["s3_key"].endswith("EXP-2/result.csv")
    assert state["experiment_runtime"]["phase"] == "reporting"


def test_record_result_coerces_error_status_to_failure():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "error",
            "summary": "No relevant data found.",
            "criteria_checks": [],
            "error_code": "no_data",
            "error_message": "empty",
            "retryable": True,
        },
    )
    assert stored["status"] == "success"
    assert stored["task_result"]["status"] == "failure"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "retry_pending"


def test_record_result_coerces_partial_success_alias_to_partial():
    """LLM often emits partial_success; closed enum only allows partial."""
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/run/plan/EXP-1/attempt/result.csv",
            "tool": "estimate_property",
        }
    ]
    payload = _success_result("EXP-1")
    payload["status"] = "partial_success"
    payload["summary"] = "KRAS docking ok; HRAS/NRAS timed out."
    stored = record_result(state, "EXP-1", started["attempt_id"], payload)
    assert stored["status"] == "success"
    assert stored["task_result"]["status"] == "partial"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "done_with_warnings"


def test_start_task_generates_fresh_transient_s3_links(monkeypatch):
    from CoScientist.experiments.runtime import coder_artifacts

    monkeypatch.setattr(coder_artifacts, "_read_resolved_bytes", lambda item: b"col1,col2\n1,2\n")

    task = _task("EXP-1", route="coder")
    task["input_data"] = [
        {
            "data_id": "input-csv",
            "kind": "s3",
            "description": "Managed source data",
            "bucket": "inputs",
            "s3_key": "data/source.csv",
        }
    ]
    plan = _plan(task)
    state = _approved_state(plan)
    calls = []

    def presign(bucket: str, key: str, expiration: int) -> str:
        calls.append((bucket, key, expiration))
        return f"https://s3.local/{key}?X-Amz-Signature=fresh-{len(calls)}"

    started = start_task(state, "EXP-1", presign=presign)
    assert calls and calls[0][2] > ExperimentsSettings().coder_timeout_s
    assert "X-Amz-Signature=fresh-1" in started["resolved_inputs"][0]["resolved_url"]
    assert "resolved_url" not in state["experiment_runtime"]["plan"]["tasks"][0][
        "input_data"
    ][0]


def test_retry_fallback_skip_and_amend_transitions():
    # retry
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "failure",
            "summary": "Transient FEDOT timeout.",
            "criteria_checks": [],
            "error_code": "timeout",
            "error_message": "timeout",
            "retryable": True,
        },
    )
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "retry_pending"
    retry_task(state, "EXP-1")
    retried = start_task(state, "EXP-1")
    assert retried["attempt_id"] != started["attempt_id"]

    # After per-route retry budget is spent, still fall back to the next route.
    mark_route_returned(state, "FedotAgent")
    record_result(
        state,
        "EXP-1",
        retried["attempt_id"],
        {
            "status": "failure",
            "summary": "FEDOT still failing after retry.",
            "criteria_checks": [],
            "error_code": "timeout",
            "error_message": "timeout",
            "retryable": True,
        },
    )
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "fallback_pending"
    fb = fallback_task(state, "EXP-1", "FEDOT retries exhausted")
    assert fb["route"] == "react_tools"
    assert fb.get("must_start_task_id") == "EXP-1"
    assert start_task(state, "EXP-1")["route_agent"] == "ExperimentAgent"

    # fallback (non-retryable → immediate next route: react_tools)
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "failure",
            "summary": "No runnable FEDOT server.",
            "criteria_checks": [],
            "error_code": "route_unavailable",
            "error_message": "server unavailable",
            "retryable": False,
        },
    )
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "fallback_pending"
    fallback = fallback_task(state, "EXP-1", "FEDOT route unavailable")
    assert fallback["route"] == "react_tools"
    assert start_task(state, "EXP-1")["route_agent"] == "ExperimentAgent"

    # skip optional
    optional = _task("EXP-1", route="coder", optional=True)
    state = _approved_state(_plan(optional))
    skipped = skip_task(state, "EXP-1", "Optional comparison omitted.")
    assert skipped["task_result"]["status"] == "skipped"

    # amend an unstarted task; criteria changes force review.
    state = _approved_state(_plan(_task("EXP-1", route="coder")))
    amended = amend_task(
        state,
        "EXP-1",
        {
            "success_criteria": [
                {
                    "criterion_id": "EXP-1-C1",
                    "description": "The script exits successfully.",
                    "kind": "execution",
                    "verification": "Check exit code 0.",
                }
            ]
        },
        "Clarify deterministic verification.",
    )
    assert amended["requires_review"] is True
    assert state["experiment_runtime"]["phase"] == "awaiting_review"


def test_react_to_coder_fallback_completes_with_real_workspace_artifact(tmp_path):
    """The v0 demo fallback reaches Coder without enabling MCP-in-Coder mode."""
    state = _approved_state(_plan(_task("EXP-1", route="react_tools")))
    first = start_task(state, "EXP-1")
    assert first["route_agent"] == "ExperimentAgent"
    mark_route_returned(state, "ExperimentAgent")
    record_result(
        state,
        "EXP-1",
        first["attempt_id"],
        {
            "status": "failure",
            "summary": "Ready MCP returned no usable output.",
            "criteria_checks": [],
            "error_code": "empty_result",
            "error_message": "empty result",
            "retryable": False,
        },
    )
    fallback_task(state, "EXP-1", "ReAct MCP result was empty")
    second = start_task(state, "EXP-1")
    assert second["route_agent"] == "CoderAgent"
    assert state["filtered_tools"] == []
    assert state["deployed_mcps"] == []

    artifact_path = tmp_path / "exp-1-result.csv"
    artifact_path.write_text("property,value\nmw,46.07\n", encoding="utf-8")
    mark_route_returned(state, "CoderAgent")
    stored = record_result(
        state,
        "EXP-1",
        second["attempt_id"],
        {
            **_success_result("EXP-1"),
            "artifacts": [
                {
                    "name": "exp-1-result.csv",
                    "workspace_path": str(artifact_path),
                    "durability": "workspace",
                    "tool": "execute_bash",
                }
            ],
        },
    )
    assert stored["task_result"]["route_used"] == "coder"
    assert stored["task_result"]["artifacts"][0]["workspace_path"] == str(
        artifact_path
    )
    assert state["experiment_runtime"]["phase"] == "reporting"


def test_terminal_and_incomplete_results_are_rejected():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    with pytest.raises(ExperimentRuntimeError, match="missing required evidence"):
        record_result(
            state,
            "EXP-1",
            started["attempt_id"],
            {
                "status": "success",
                "summary": "Claimed success without evidence.",
                "criteria_checks": [],
            },
        )

    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-1/result.csv",
        }
    ]
    record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))
    with pytest.raises(ExperimentRuntimeError, match="terminal"):
        start_task(state, "EXP-1")


def test_record_result_requires_canonical_result_keys():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-1/result.csv",
        }
    ]
    mark_route_returned(state, "FedotAgent")

    with pytest.raises(ValidationError):
        record_result(
            state,
            "EXP-1",
            started["attempt_id"],
            {
                "status": "success",
                "summary": "Alias keys must not be accepted.",
                "actual_outputs": {"rows": 1},
                "criteria_checks": [
                    {
                        "criterion_id": "EXP-1-C1",
                        "met": True,
                        "evidence": {"status": "success"},
                        "message": "alias keys",
                    }
                ],
            },
        )

    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "success",
            "summary": "Canonical keys only.",
            "outputs": {"rows": 1},
            "criteria_checks": [
                {
                    "criterion_id": "EXP-1-C1",
                    "passed": True,
                    "observed": {"status": "success"},
                    "details": "The route returned a structured success result.",
                }
            ],
        },
    )

    result = stored["task_result"]
    assert result["outputs"] == {"rows": 1}
    assert result["criteria_checks"] == [
        {
            "criterion_id": "EXP-1-C1",
            "purpose": "execution",
            "passed": True,
            "observed": {"status": "success"},
            "evidence_artifact_ids": [],
            "details": "The route returned a structured success result.",
        }
    ]
    assert stored["phase"] == "reporting"


def test_record_result_repairs_truncated_attempt_id():
    """LLM executors often drop the last hex char of ATT-<uuid>; repair near-miss."""
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    full_id = started["attempt_id"]
    truncated = full_id[:-1]
    assert truncated != full_id

    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-1/result.csv",
        }
    ]
    stored = record_result(
        state,
        "EXP-1",
        truncated,
        _success_result("EXP-1"),
    )
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["attempt_id"] == full_id


def test_record_result_rejects_unrelated_attempt_id():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    with pytest.raises(ExperimentRuntimeError, match="do not match the active attempt"):
        record_result(
            state,
            "EXP-1",
            "ATT-deadbeefdeadbeefdeadbeefdeadbeef",
            _success_result("EXP-1"),
        )
    assert started["attempt_id"] == state["experiment_runtime"]["active_attempt_id"]


def test_start_task_resolves_task_artifact_by_name(tmp_path, monkeypatch):
    """Planner stores source_artifact_id as the filename; runtime ART-* is unknown at plan time."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))

    upstream = _task("EXP-1", artifact_name="diagnosis_findings.json")
    downstream = _task("EXP-2", route="coder", depends_on=["EXP-1"], artifact_name="literature.json")
    downstream["mcp_servers"] = []
    downstream["input_data"] = [
        {
            "data_id": "findings",
            "kind": "task_artifact",
            "description": "Findings from EXP-1",
            "source_task_id": "EXP-1",
            "source_artifact_id": "diagnosis_findings.json",
        }
    ]
    state = _approved_state(_plan(upstream, downstream))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    folder = tmp_path / "experiment_artifacts" / "EXP-1" / started["attempt_id"]
    folder.mkdir(parents=True)
    path = folder / "diagnosis_findings.json"
    path.write_text('{"ok": true}', encoding="utf-8")
    state["fedot_artifacts"] = [
        {
            "name": "diagnosis_findings.json",
            "workspace_path": str(path),
            "media_type": "application/json",
        }
    ]
    record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))

    started2 = start_task(state, "EXP-2")
    assert started2["status"] == "success"
    resolved = started2["resolved_inputs"]
    assert Path(resolved[0]["resolved_workspace_path"]).name == "diagnosis_findings.json"


def test_control_tool_downgrades_incomplete_success_to_terminal_failure():
    from CoScientist.experiments.runtime.tools import ExperimentControlToolset

    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")

    stored = ExperimentControlToolset().record_result(
        "EXP-1",
        started["attempt_id"],
        {
            "status": "partial",
            "summary": "Route returned text but no required artifact.",
            "criteria_checks": [
                {
                    "criterion_id": "EXP-1-C1",
                    "passed": False,
                    "details": "Required execution evidence was absent.",
                }
            ],
            "retryable": False,
        },
        SimpleNamespace(state=state),
    )

    assert stored["status"] == "success"
    assert stored["downgraded_from"] == "partial"
    assert stored["task_result"]["status"] == "failure"
    assert stored["task_result"]["error_code"] == "partial_missing_core_evidence"
    assert stored["task_result"]["retryable"] is False
    assert state["experiment_runtime"]["active_attempt_id"] is None
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "fallback_pending"


def test_record_result_downgrades_fabricated_success_to_partial():
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/run/plan/EXP-1/attempt/result.csv",
            "tool": "estimate_property",
        }
    ]
    payload = _success_result("EXP-1")
    payload["summary"] = "Completed with simulated PubMed hits and hardcoded metabolite list."
    payload["warnings"] = ["Literature data was simulated via a hardcoded list"]

    stored = record_result(state, "EXP-1", started["attempt_id"], payload)
    result = stored["task_result"]
    assert result["status"] == "partial"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "done_with_warnings"
    assert any("downgraded_from_success" in w for w in result["warnings"])


def test_start_task_seeds_upstream_from_resolved_inputs(tmp_path, monkeypatch):
    """Consumer start_task materializes producer CSV into upstream_bindings."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))

    upstream = _task("EXP-1", artifact_name="generated_molecules.csv")
    downstream = _task(
        "EXP-2",
        route="react_tools",
        tool="estimate_property",
        depends_on=["EXP-1"],
        artifact_name="toxicity.json",
    )
    downstream["input_data"] = [
        {
            "data_id": "mols",
            "kind": "task_artifact",
            "description": "Generated molecules",
            "source_task_id": "EXP-1",
            "source_artifact_id": "generated_molecules.csv",
        }
    ]
    state = _approved_state(_plan(upstream, downstream))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    folder = tmp_path / "experiment_artifacts" / "EXP-1" / started["attempt_id"]
    folder.mkdir(parents=True)
    path = folder / "generated_molecules.csv"
    path.write_text("smiles,score\nCCO,0.9\nCCN,0.8\n", encoding="utf-8")
    state["fedot_artifacts"] = [
        {
            "name": "generated_molecules.csv",
            "workspace_path": str(path),
            "media_type": "text/csv",
        }
    ]
    # Ambient wrong table must lose to EM lineage.
    state["fedot_artifact_tables"] = [
        {
            "columns": ["smiles"],
            "rows": [{"smiles": "c1ccccc1"}],
            "format": "csv",
            "url": "https://example.invalid/benzene.csv",
        }
    ]
    record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))

    started2 = start_task(state, "EXP-2")
    assert started2["status"] == "success"
    assert started2["resolved_inputs"][0]["resolved_workspace_path"] == str(path)
    bindings = started2.get("upstream_bindings") or {}
    assert "smiles" in bindings
    assert "CCO" in bindings["smiles"]
    assert "c1ccccc1" not in bindings["smiles"]
    assert state.get("upstream_artifact_inputs")
    tables = state.get("fedot_artifact_tables") or []
    assert tables and any(
        row.get("smiles") == "CCO"
        for t in tables
        for row in (t.get("rows") or [])
        if isinstance(row, dict)
    )


def test_inline_fedot_csv_is_materialized_as_expected_workspace_artifact(
    tmp_path, monkeypatch
):
    from CoScientist.config import get_settings
    from CoScientist.experiments.runtime.inline_artifacts import (
        materialize_inline_result,
    )

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")

    artifacts = materialize_inline_result(
        state,
        {
            "metabolite_analysis_output": (
                "```csv\n"
                '"Name","Cluster","LD50"\n'
                '"Bergapten","E","597.6"\n'
                "```"
            )
        },
    )

    assert len(artifacts) == 1
    artifact_path = artifacts[0]["workspace_path"]
    assert artifact_path.endswith("exp-1-result.csv")
    assert '"Bergapten","E","597.6"' in open(
        artifact_path, encoding="utf-8"
    ).read()

    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        _success_result("EXP-1"),
    )
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["artifacts"][0]["workspace_path"] == artifact_path


def test_record_result_leaves_unchecked_quality_threshold_unassessed(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    task = _task("EXP-1")
    task["success_criteria"] = [
        {
            "criterion_id": "C-metric",
            "description": "LD50 below threshold.",
            "kind": "threshold",
            "metric": "ld50",
            "operator": "<=",
            "target": 100,
            "verification": "Compare predicted LD50 to target.",
        }
    ]
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "workspace_path": str(tmp_path / "exp-1-result.csv"),
            "media_type": "text/csv",
        }
    ]
    (tmp_path / "exp-1-result.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        {
            "status": "success",
            "summary": "Model ran.",
            "outputs": {"exp-1-result.csv": "a,b\n1,2\n"},
        },
    )
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["execution_status"] == "completed"
    assert stored["task_result"]["assessment_status"] == "not_evaluated"


def test_coder_workspace_artifacts_are_promoted_into_lineage(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    task = _task("EXP-1", route="coder", artifact_name="concentrations.csv")
    task["expected_artifacts"] = [
        {
            "name": "concentrations.csv",
            "role": "data",
            "media_type": "text/csv",
            "description": "ODE concentration time series.",
        }
    ]
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")

    sandbox = tmp_path / "ws_session_test"
    sandbox.mkdir()
    (sandbox / "concentrations.csv").write_text("t,A,B,C\n0,1,0,0\n", encoding="utf-8")
    state["coder_workspace_id"] = "ws_session_test"

    on_route_agent_returned(
        SimpleNamespace(name="CoderAgent"),
        {},
        SimpleNamespace(state=state),
        {"status": "success"},
    )
    assert state["coder_artifacts"]
    assert Path(state["coder_artifacts"][0]["workspace_path"]).is_file()

    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        _success_result("EXP-1"),
    )
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["artifacts"][0]["name"] == "concentrations.csv"


def test_soft_artifact_name_match_accepts_stem_variants(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    task = _task("EXP-1", artifact_name="dataset_overview")
    task["expected_artifacts"] = [
        {
            "name": "dataset_overview",
            "role": "data",
            "media_type": "application/json",
            "description": "Overview payload.",
        }
    ]
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")

    folder = tmp_path / "experiment_artifacts" / "EXP-1" / started["attempt_id"]
    folder.mkdir(parents=True)
    path = folder / "dataset_overview.json"
    path.write_text('{"n": 3}', encoding="utf-8")
    state["fedot_artifacts"] = [
        {
            "name": "dataset_overview.json",
            "workspace_path": str(path),
            "media_type": "application/json",
            "producer_tool": "dataset_overview",
        }
    ]

    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        _success_result("EXP-1"),
    )
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["artifacts"][0]["name"] == "dataset_overview"


def test_uuid_s3_csv_binds_to_semantic_expected_name(tmp_path, monkeypatch):
    """MCP generators return UUID filenames; plan expects alzheimer_candidates.csv."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    monkeypatch.setattr(get_settings().s3, "bucket_name", "molecule-generative-mcp")
    task = _task("EXP-1", tool="generate_case_mols", artifact_name="alzheimer_candidates.csv")
    plan = _plan(task)
    state: dict = {}
    initialize_runtime(
        state,
        plan,
        critique={
            "schema_version": "plan-critique/0.1",
            "critique_id": "CRIT-test",
            "plan_id": plan.plan_id,
            "verdict": "approve",
            "issues": [],
            "checked_at": NOW,
        },
    )
    approve_plan(state)
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")

    state["fedot_artifacts"] = [
        {
            "url": "http://10.32.1.114:9000/molecule-generative-mcp/generated/alzheimer/35dcc1e32f934178935c4e9cc2415f49.csv",
            "s3_key": "generated/alzheimer/35dcc1e32f934178935c4e9cc2415f49.csv",
            "bucket": "molecule-generative-mcp",
            "tool": "generate_case_mols",
        }
    ]

    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        _success_result("EXP-1"),
    )
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["artifacts"][0]["name"] == "alzheimer_candidates.csv"
    assert stored["task_result"]["artifacts"][0]["durability"] == "managed"
    assert not any("URL-only" in w for w in stored["task_result"].get("warnings") or [])


def test_managed_data_satisfies_mistyped_required_data_name(tmp_path, monkeypatch):
    """Managed MCP CSV satisfies fantasy required data name (e.g. *.json) — R2 class."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    monkeypatch.setattr(get_settings().s3, "bucket_name", "molecule-generative-mcp")
    task = _task("EXP-1", tool="generate_case_mols", artifact_name="generated_molecules.json")
    task["expected_artifacts"] = [
        {
            "name": "generated_molecules.json",
            "role": "data",
            "media_type": "application/json",
            "required": True,
            "description": "Planner fantasy name; tool returns CSV.",
        }
    ]
    plan = _plan(task)
    state: dict = {}
    initialize_runtime(
        state,
        plan,
        critique={
            "schema_version": "plan-critique/0.1",
            "critique_id": "CRIT-test",
            "plan_id": plan.plan_id,
            "verdict": "approve",
            "issues": [],
            "checked_at": NOW,
        },
    )
    approve_plan(state)
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    state["fedot_artifacts"] = [
        {
            "url": "http://10.32.1.114:9000/molecule-generative-mcp/generated/alzheimer/aabbccdd11223344.csv",
            "s3_key": "generated/alzheimer/aabbccdd11223344.csv",
            "bucket": "molecule-generative-mcp",
            "tool": "generate_case_mols",
            "role": "data",
        }
    ]
    stored = record_result(
        state,
        "EXP-1",
        started["attempt_id"],
        _success_result("EXP-1"),
    )
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["artifacts"][0]["name"] == "generated_molecules.json"
    assert stored["task_result"]["artifacts"][0]["durability"] == "managed"
    assert stored["task_result"]["artifacts"][0]["s3_key"].endswith(".csv")


def test_resolve_fallback_chains_from_settings():
    from CoScientist.experiments.runtime.state_machine import (
        FALLBACK_CHAINS,
        resolve_fallback_chains,
    )

    default = resolve_fallback_chains(ExperimentsSettings())
    assert default["fedot_mas"] == FALLBACK_CHAINS["fedot_mas"] == [
        "fedot_mas",
        "react_tools",
        "coder",
    ]
    custom = resolve_fallback_chains(
        ExperimentsSettings(fallback_fedot_mas=["fedot_mas", "coder"])
    )
    assert custom["fedot_mas"] == ["fedot_mas", "coder"]


def test_record_result_accepts_s3_csv_when_planner_name_differs():
    """S10-like: Fedot wrote gan_default/...csv; planner expected another basename."""
    task = _task("EXP-1", artifact_name="planner_wanted_candidates.csv")
    task["expected_artifacts"].append({
        "name": "comprehensive_report.md",
        "role": "report",
        "media_type": "text/markdown",
        "required": True,
        "description": "Narrative report the planner invented.",
    })
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    state["fedot_artifacts"] = [
        {
            "name": "molecules.csv",
            "bucket": "managed-experiments",
            "s3_key": "gan_default/run/molecules.csv",
            "tool": "generate_mols",
            "role": "data",
            "media_type": "text/csv",
        }
    ]
    payload = {
        "status": "success",
        "summary": "Fedot produced S3 CSV via generate_mols.",
        "criteria_checks": [
            {"criterion_id": "WRONG-ID", "passed": True, "details": "route finished"}
        ],
    }
    stored = record_result(state, "EXP-1", started["attempt_id"], payload)
    assert stored["status"] == "success"
    assert stored["task_result"]["status"] == "success"
    assert stored["task_result"]["error_code"] is None
    assert any(
        a.get("bucket") and a.get("s3_key")
        for a in stored["task_result"]["artifacts"]
    )
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "done"


def test_fallback_reaches_coder_when_bound_inventory_tool_already_tried():
    from CoScientist.experiments.context.builder import RETRIEVED_CAPABILITIES_KEY

    task = _task("EXP-1")
    task["description"] = "Train AutoML models to predict LD50 and impute gaps."
    state = _approved_state(_plan(task))
    state[RETRIEVED_CAPABILITIES_KEY] = [{
        "tool": "estimate_property",
        "server_id": "srv-chem",
        "description": "ready MCP",
    }]
    fail = {
        "status": "failure",
        "summary": "Ready MCP returned no usable output.",
        "criteria_checks": [],
        "error_code": "empty_result",
        "error_message": "empty result",
        "retryable": False,
    }
    first = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    record_result(state, "EXP-1", first["attempt_id"], fail)
    assert fallback_task(state, "EXP-1", "FEDOT empty")["route"] == "react_tools"
    second = start_task(state, "EXP-1")
    assert second["route_agent"] == "ExperimentAgent"
    mark_route_returned(state, "ExperimentAgent")
    record_result(state, "EXP-1", second["attempt_id"], fail)
    fallback = fallback_task(state, "EXP-1", "want coder")
    assert fallback["route"] == "coder"
    assert fallback.get("must_start_task_id") == "EXP-1"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "ready"
    started = start_task(state, "EXP-1")
    assert started["status"] == "success"
    assert started["route_agent"] == "CoderAgent"


def test_start_task_ignores_stuffed_tool_name_when_operation_unnamed():
    from CoScientist.experiments.context.builder import RETRIEVED_CAPABILITIES_KEY

    task = _task("EXP-1", route="coder")
    task["name"] = "Train six LD50 QSAR models via predict_ld50"
    task["description"] = "Call predict_ld50 to pretend the six-route models exist."
    task["mcp_servers"] = []
    task["design"] = _design("H1")
    task["design"]["operation_ref"] = "OP-1"
    plan = _plan(task)
    state: dict = {}
    initialize_runtime(
        state, plan, critique={"verdict": "approve", "issues": [], "summary": "forced"},
    )
    approve_plan(state)
    state["experiment_context"] = {
        "operations": [{"operation_id": "OP-1", "statement": "Fit six predictive models for mouse LD50"}],
    }
    state[RETRIEVED_CAPABILITIES_KEY] = [{
        "tool": "predict_ld50",
        "server_id": "srv-hogweed",
        "url": "http://127.0.0.1:7336/mcp",
        "score": 0.9,
    }]
    out = start_task(state, "EXP-1")
    assert out["status"] == "success"
    assert out["route"] == "coder"


def test_code_modification_is_never_rewritten_to_a_ready_mcp():
    from CoScientist.experiments.context.builder import RETRIEVED_CAPABILITIES_KEY

    task = _task("EXP-1", route="coder")
    task["code_assessment"] = {
        "requirement": "modify",
        "evidence": "The model objective and training loop must be changed.",
        "entrypoints": ["train.py"],
    }
    state = _approved_state(_plan(task))
    state[RETRIEVED_CAPABILITIES_KEY] = _inventory()

    started = start_task(state, "EXP-1")
    assert started["route"] == "coder"
    assert started["route_agent"] == "CoderAgent"


def test_result_tasks_ok_ignores_unused_research_failure():
    from CoScientist.experiments.review import result_tasks_ok

    runtime = {
        "tasks": {
            "EXP-1": {
                "status": "failed",
                "planned_route": "research",
                "task": {"id": "EXP-1", "route": "research", "input_data": []},
            },
            "EXP-2": {
                "status": "done",
                "planned_route": "fedot_mas",
                "task": {"id": "EXP-2", "route": "fedot_mas", "input_data": []},
            },
        }
    }
    assert result_tasks_ok(runtime) is True
    runtime["tasks"]["EXP-2"]["status"] = "failed"
    assert result_tasks_ok(runtime) is False


# ── result review: state persistence and the replan budget ────────────────────
# Regression cover for the loop observed on 2026-09-01, where one run built five
# plans. Two independent defects had to line up:
#   * mark_result_review mutated the runtime dict in place, so ADK never recorded
#     a state delta and build_experiment_context's `phase == "completed"` gate
#     never saw the finished stage;
#   * the round counter lived inside that same runtime, which the builder nulls
#     via _CLEAR_ON_NEW_RUN before every replan, so the budget bounded nothing.
# Both are asserted on the ADK State delta, NOT on reading the dict back: a plain
# read-back passes either way, because _runtime() hands out the live object.

def _reported_state():
    """An approved run that has reached reporting, ready for result review.

    The phase is set directly rather than driven through record_result: these
    tests are about what the review does to the phase and the replan budget, and
    routing a real result through would drag in artifact-evidence plumbing that
    has nothing to do with the loop being covered here.
    """
    state = _approved_state(_plan(_task("EXP-1")))
    state["experiment_runtime"]["phase"] = "reporting"
    return state


def _adk_state(plain: dict):
    """Wrap a plain dict the way ADK does, so state deltas are observable."""
    from google.adk.sessions.state import State
    delta: dict = {}
    return State(value=plain, delta=delta), delta


def test_result_review_approval_records_a_state_delta():
    """The whole point of the fix: assignment, not mutation.

    Reading state["experiment_runtime"]["phase"] back would pass even without
    the fix, because _runtime() returns the live nested dict. Only the delta
    distinguishes a mutation from an assignment, and only the delta is what
    reaches the next agent.
    """
    from CoScientist.experiments.runtime import mark_result_review
    from CoScientist.experiments.runtime.state_machine import RUNTIME_KEY
    st, delta = _adk_state(_reported_state())
    out = mark_result_review(st, approved=True)
    assert out["phase"] == "completed"
    assert RUNTIME_KEY in delta, "phase=completed never reached session state"
    assert delta[RUNTIME_KEY]["phase"] == "completed"


def test_quality_feedback_completes_and_records_a_state_delta(monkeypatch):
    from CoScientist.config import get_settings
    from CoScientist.experiments.runtime import mark_result_review
    from CoScientist.experiments.runtime.state_machine import RUNTIME_KEY
    monkeypatch.setattr(get_settings().experiments, "max_replan_rounds", 1)
    st, delta = _adk_state(_reported_state())
    out = mark_result_review(st, approved=False, feedback="metrics missing")
    assert out["phase"] == "completed"
    assert out["result_review_disposition"] == "changes_suggested"
    assert delta[RUNTIME_KEY]["phase"] == "completed"


def test_quality_feedback_does_not_spend_one_replan_round(monkeypatch):
    from CoScientist.config import get_settings
    from CoScientist.experiments.runtime import mark_result_review
    from CoScientist.experiments.runtime.state_machine import REPLAN_ROUNDS_KEY
    monkeypatch.setattr(get_settings().experiments, "max_replan_rounds", 1)
    state = _reported_state()
    out = mark_result_review(state, approved=False, feedback="metrics missing")
    assert out["replan_rounds"] == 0
    assert out["replan_exhausted"] is False
    assert state.get(REPLAN_ROUNDS_KEY, 0) == 0
    assert "metrics missing" in state["experiment_runtime"]["result_review_feedback"]


def test_feedback_does_not_create_a_replan_for_builder_to_resume(monkeypatch):
    """The path production actually takes.

    build_experiment_context runs as the planner's before_agent and nulls
    experiment_runtime through _CLEAR_ON_NEW_RUN. An earlier version of this fix
    kept the counter inside the runtime and passed its tests only because they
    called initialize_runtime directly, stepping over exactly this wipe.
    """
    from CoScientist.config import get_settings
    from CoScientist.experiments.context.builder import _CLEAR_ON_NEW_RUN
    from CoScientist.experiments.runtime import get_experiment_plan, mark_result_review
    from CoScientist.experiments.runtime.state_machine import REPLAN_ROUNDS_KEY
    monkeypatch.setattr(get_settings().experiments, "max_replan_rounds", 1)
    assert REPLAN_ROUNDS_KEY not in _CLEAR_ON_NEW_RUN, (
        "the counter must not be in the list that a replan clears"
    )
    state = _reported_state()
    mark_result_review(state, approved=False, feedback="first objection")

    for key in _CLEAR_ON_NEW_RUN:          # what the builder does on a replan hop
        if key in state:
            state[key] = None
    initialize_runtime(state, _plan(_task("EXP-1")), critique=None)
    assert get_experiment_plan(state)["replan_rounds"] == 0

    state["experiment_runtime"]["phase"] = "reporting"
    out = mark_result_review(state, approved=False, feedback="second objection")
    assert out["phase"] == "completed"
    assert out["replan_exhausted"] is False


def test_zero_budget_never_replans(monkeypatch):
    from CoScientist.config import get_settings
    from CoScientist.experiments.runtime import mark_result_review
    monkeypatch.setattr(get_settings().experiments, "max_replan_rounds", 0)
    out = mark_result_review(_reported_state(), approved=False, feedback="nope")
    assert out["phase"] == "completed"
    assert out["replan_exhausted"] is False


def test_a_new_ask_starts_without_implicit_result_replans(monkeypatch):
    """A different question is a new run, so it must not inherit spent rounds."""
    from CoScientist.config import get_settings
    from CoScientist.experiments.runtime import get_experiment_plan, mark_result_review
    from CoScientist.experiments.runtime.state_machine import REPLAN_ROUNDS_KEY
    monkeypatch.setattr(get_settings().experiments, "max_replan_rounds", 1)
    state = _reported_state()
    mark_result_review(state, approved=False, feedback="redo")
    assert state.get(REPLAN_ROUNDS_KEY, 0) == 0
    state[REPLAN_ROUNDS_KEY] = 0           # what builder.py does when the ask changes
    initialize_runtime(state, _plan(_task("EXP-1")), critique=None)
    view = get_experiment_plan(state)
    assert view["replan_rounds"] == 0
    assert view["replan_rounds_remaining"] == 1


def test_first_pass_plan_view_reports_a_full_budget(monkeypatch):
    from CoScientist.config import get_settings
    from CoScientist.experiments.runtime import get_experiment_plan
    monkeypatch.setattr(get_settings().experiments, "max_replan_rounds", 1)
    view = get_experiment_plan(_approved_state(_plan(_task("EXP-1"))))
    assert view["replan_rounds"] == 0
    assert view["replan_rounds_remaining"] == 1
    assert "replan_reason" not in view


def test_plan_approval_records_a_state_delta():
    """approve_plan flips phase to execution — on the delta, not just in place.

    Same class of defect as mark_result_review: the control tool that calls this
    hands in an ADK State, and a nested mutation there is invisible to every
    later agent, so the approved plan is re-approved on the next hop.
    """
    from CoScientist.experiments.runtime import approve_plan
    from CoScientist.experiments.runtime.state_machine import RUNTIME_KEY

    plain = _approved_state(_plan(_task("EXP-1")))
    plain["experiment_runtime"]["phase"] = "awaiting_review"
    plain["experiment_runtime"]["approved"] = False
    st, delta = _adk_state(plain)

    result = approve_plan(st)

    assert result["phase"] == "execution"
    assert RUNTIME_KEY in delta, "approve_plan mutated the runtime without reassigning it"
    assert delta[RUNTIME_KEY]["phase"] == "execution"
    assert delta[RUNTIME_KEY]["approved"] is True


# ── FEDOT.MAS: one switch, and the executor's tree is the truth ─────────────

def _experiments_tree(*, fedot_listed: bool = True):
    """The experiments profile, optionally with FedotAgent taken out of the
    executor's subordinates - what an operator does to drop it from the YAML."""
    from CoScientist.assembly.schema import load_config, resolve_config_path

    config = load_config(resolve_config_path("experiments"))
    if not fedot_listed:
        config.agents["ExperimentExecutorAgent"].subordinates.remove("FedotAgent")
    return config


def _use_tree(monkeypatch, config) -> None:
    """Make the state machine read ``config`` as the YAML on disk."""
    from CoScientist.experiments.runtime import state_machine

    monkeypatch.setattr(state_machine, "_config_tree", lambda: config)


def _route_failure() -> dict:
    return {
        "status": "failure",
        "summary": "The tool returned nothing usable.",
        "criteria_checks": [],
        "error_code": "empty_result",
        "error_message": "empty result",
        "retryable": False,
    }


def test_fedot_route_available_is_the_switch_and_the_tree(monkeypatch):
    from CoScientist.assembly.schema import load_config, resolve_config_path
    from CoScientist.config import get_settings
    from CoScientist.experiments.runtime.state_machine import fedot_route_available

    on, off = ExperimentsSettings(route_fedot=True), ExperimentsSettings(route_fedot=False)
    assert fedot_route_available(on, system=_experiments_tree()) is True
    assert fedot_route_available(off, system=_experiments_tree()) is False
    # Out of the executor's subordinates: off, whatever the switch says.
    assert fedot_route_available(on, system=_experiments_tree(fedot_listed=False)) is False
    disabled = _experiments_tree()
    disabled.agents["FedotAgent"].enabled = False
    assert fedot_route_available(on, system=disabled) is False
    # main has no executor, so FedotAgent's own `enabled` decides there.
    main = load_config(resolve_config_path("system"))
    assert fedot_route_available(on, system=main) is True
    monkeypatch.setattr(get_settings().web, "fedot_fallback_enabled", False)
    assert fedot_route_available(on, system=main) is False


def test_an_unreadable_tree_takes_fedot_down(monkeypatch):
    """Fails closed: react_tools is always there to take the task."""
    from CoScientist.experiments.runtime import state_machine

    def broken():
        raise ValueError("unparsable profile")

    monkeypatch.setattr(state_machine, "_config_tree", broken)
    assert state_machine.fedot_route_available(ExperimentsSettings(route_fedot=True)) is False


def test_the_tree_is_reread_when_the_yaml_changes(monkeypatch, tmp_path):
    """build_for_mode() builds every session from the YAML on disk, so an edit
    made while the server runs has to reach the route check as well."""
    import os

    from CoScientist.assembly import schema
    from CoScientist.experiments.runtime import state_machine

    profile = tmp_path / "profile.yaml"
    profile.write_text(schema.resolve_config_path("experiments").read_text(encoding="utf-8"),
                       encoding="utf-8")
    monkeypatch.setenv(schema.CONFIG_ENV_VAR, str(profile))
    monkeypatch.setattr(state_machine, "_TREE_CACHE", {})
    on = ExperimentsSettings(route_fedot=True)
    assert state_machine.fedot_route_available(on) is True

    source = profile.read_text(encoding="utf-8")
    edited = source.replace("      - FedotAgent\n", "", 1)
    assert edited != source
    profile.write_text(edited, encoding="utf-8")
    stat = profile.stat()
    os.utime(profile, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert state_machine.fedot_route_available(on) is False


def test_the_switch_off_reroutes_fedot_to_react_tools(monkeypatch):
    from CoScientist.config import get_settings

    state = _approved_state(_plan(_task("EXP-1")))
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["current_route"] == "fedot_mas"
    monkeypatch.setattr(get_settings().experiments, "route_fedot", False)

    started = start_task(state, "EXP-1")
    assert started["route"] == "react_tools"
    assert started["route_agent"] == "ExperimentAgent"
    history = state["experiment_runtime"]["tasks"]["EXP-1"]["route_history"]
    assert "EXPERIMENTS__ROUTE_FEDOT" in history[-1]["reason"]


def test_removing_fedot_from_the_executor_takes_its_route_down(monkeypatch):
    """The reported case: FedotAgent deleted from the YAML, switches untouched.

    start_task used to hand back route_agent=FedotAgent for an agent nobody
    attached, and enforce_continue_until_reporting then demanded a call to it
    until the run stalled.
    """
    state = _approved_state(_plan(_task("EXP-1")))
    _use_tree(monkeypatch, _experiments_tree(fedot_listed=False))

    started = start_task(state, "EXP-1")
    assert started["route"] == "react_tools"
    assert started["route_agent"] == "ExperimentAgent"


def test_the_fedot_route_is_untouched_while_its_agent_is_attached(monkeypatch):
    state = _approved_state(_plan(_task("EXP-1")))
    _use_tree(monkeypatch, _experiments_tree())

    started = start_task(state, "EXP-1")
    assert started["route"] == "fedot_mas"
    assert started["route_agent"] == "FedotAgent"


def test_start_task_never_hands_out_an_agent_the_running_executor_lacks():
    """The YAML can change under a running session: the tree that session runs
    is the one that has to hold FedotAgent."""
    state = _approved_state(_plan(_task("EXP-1")))

    started = start_task(state, "EXP-1", route_agents=frozenset({"ExperimentAgent", "CoderAgent"}))
    assert started["route"] == "react_tools"
    assert started["route_agent"] == "ExperimentAgent"


@pytest.mark.parametrize("attached, route", [
    (("FedotAgent", "ExperimentAgent"), "fedot_mas"),
    (("ExperimentAgent", "CoderAgent"), "react_tools"),
])
def test_the_control_tool_reads_the_executors_live_agent_tools(attached, route):
    from CoScientist.experiments.runtime.tools import ExperimentControlToolset

    state = _approved_state(_plan(_task("EXP-1")))
    executor = SimpleNamespace(tools=[
        *(SimpleNamespace(agent=SimpleNamespace(name=name)) for name in attached),
        object(),  # a toolset: no .agent
    ])
    tool_context = SimpleNamespace(state=state, _invocation_context=SimpleNamespace(agent=executor))

    started = ExperimentControlToolset().start_task("EXP-1", tool_context)
    assert started["status"] == "success"
    assert started["route"] == route


def test_a_coder_task_naming_an_inventory_tool_goes_to_react_tools():
    """The runtime's implicit coder→MCP rewrite never picks FEDOT.MAS, even
    with it on: one bound tool is ExperimentAgent's job."""
    from CoScientist.experiments.context.builder import RETRIEVED_CAPABILITIES_KEY

    task = _task("EXP-1", route="coder")
    task["description"] = "Compute the property of CCO with estimate_property."
    state: dict = {}
    initialize_runtime(
        state, _plan(task), critique={"verdict": "approve", "issues": [], "summary": "forced"},
    )
    approve_plan(state)
    state[RETRIEVED_CAPABILITIES_KEY] = [{
        "tool": "estimate_property",
        "server_id": "srv-chem",
        "url": "http://127.0.0.1:8000/mcp",
        "score": 0.9,
    }]

    started = start_task(state, "EXP-1", settings=ExperimentsSettings(route_fedot=True))
    assert started["route"] == "react_tools"
    assert started["route_agent"] == "ExperimentAgent"
    history = state["experiment_runtime"]["tasks"]["EXP-1"]["route_history"]
    assert history[-1]["reason"] == "inventory_rewrote_coder"


def test_a_fallback_chain_skips_a_switched_off_route():
    """EXPERIMENTS__FALLBACK_* may name fedot_mas. While it is off the chain goes
    past it, instead of parking the task in fallback_pending on a route that
    fallback_task then refuses as route_disabled."""
    settings = ExperimentsSettings(
        route_fedot=False, fallback_react_tools=["react_tools", "fedot_mas", "coder"],
    )
    state = _approved_state(_plan(_task("EXP-1", route="react_tools")))
    started = start_task(state, "EXP-1", settings=settings)
    mark_route_returned(state, "ExperimentAgent")
    record_result(state, "EXP-1", started["attempt_id"], _route_failure(), settings=settings)

    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "fallback_pending"
    assert fallback_task(state, "EXP-1", "tool empty", settings=settings)["route"] == "coder"


def test_the_fallback_never_offers_fedot_the_running_executor_lacks():
    """Switch on and YAML listing FedotAgent, but this session's executor was
    built without it: the fallback decision reads the same live tree as
    start_task, so it does not park the task on a route that cannot run."""
    settings = ExperimentsSettings(
        route_fedot=True, fallback_react_tools=["react_tools", "fedot_mas", "coder"],
    )
    live = frozenset({"ExperimentAgent", "CoderAgent"})
    state = _approved_state(_plan(_task("EXP-1", route="react_tools")))
    started = start_task(state, "EXP-1", settings=settings, route_agents=live)
    mark_route_returned(state, "ExperimentAgent")
    record_result(
        state, "EXP-1", started["attempt_id"], _route_failure(),
        settings=settings, route_agents=live,
    )

    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "fallback_pending"
    fallback = fallback_task(state, "EXP-1", "tool empty", settings=settings, route_agents=live)
    assert fallback["route"] == "coder"


# ── The medical route follows MedicalAgent the same way ────────────────────

def _medical_task(task_id: str = "EXP-1") -> dict:
    task = _task(task_id, route="medical")
    task["design"]["analysis_artifacts"] = [{
        "name": "pubmed_notes.md", "role": "report",
        "prepare_via": "medical", "path_or_tool": "get_pico",
    }]
    return task


def _forced_state(*tasks: dict) -> dict:
    state: dict = {}
    initialize_runtime(
        state, _plan(*tasks), critique={"verdict": "approve", "issues": [], "summary": "forced"},
    )
    approve_plan(state)
    return state


def test_medical_route_available_follows_the_agent(monkeypatch):
    from CoScientist.assembly.schema import load_config, resolve_config_path
    from CoScientist.config import get_settings
    from CoScientist.experiments.runtime import state_machine
    from CoScientist.experiments.runtime.state_machine import medical_route_available

    tree = _experiments_tree()
    assert medical_route_available(system=tree) is True
    detached = _experiments_tree()
    detached.agents["ExperimentExecutorAgent"].subordinates.remove("MedicalAgent")
    assert medical_route_available(system=detached) is False
    main = load_config(resolve_config_path("system"))
    assert medical_route_available(system=main) is True
    # MEDICAL__ENABLED is MedicalAgent's own `enabled`, read on every call.
    monkeypatch.setattr(get_settings().web, "medical_agent_enabled", False)
    assert medical_route_available(system=tree) is False
    assert medical_route_available(system=main) is False

    def broken():
        raise ValueError("unparsable profile")

    monkeypatch.setattr(state_machine, "_config_tree", broken)
    monkeypatch.setattr(get_settings().web, "medical_agent_enabled", True)
    assert medical_route_available() is False


@pytest.mark.parametrize("switch_off, route_agents", [
    (True, None),
    (False, frozenset({"ExperimentAgent", "CoderAgent", "ResearchAgent"})),
])
def test_a_medical_task_is_blocked_not_stranded_when_the_agent_is_gone(
    monkeypatch, switch_off, route_agents,
):
    """start_task used to refuse it as route_disabled and leave it 'ready' - a
    state neither retry_task nor fallback_task accepts. Nothing can stand in for
    MedicalAgent, so the task is blocked: terminal, and the run moves on."""
    from CoScientist.config import get_settings

    state = _forced_state(_medical_task())
    if switch_off:
        monkeypatch.setattr(get_settings().web, "medical_agent_enabled", False)

    with pytest.raises(ExperimentRuntimeError) as exc:
        start_task(state, "EXP-1", route_agents=route_agents)
    assert exc.value.code == "route_disabled"
    runtime = state["experiment_runtime"]
    assert runtime["tasks"]["EXP-1"]["status"] == "blocked"
    assert runtime["phase"] == "reporting"


def test_a_medical_task_starts_while_its_agent_is_attached():
    state = _forced_state(_medical_task())
    started = start_task(state, "EXP-1")
    assert started["route"] == "medical"
    assert started["route_agent"] == "MedicalAgent"


@pytest.mark.parametrize("medical_on, route", [(True, "medical"), (False, "coder")])
def test_a_coder_task_naming_a_medical_tool_follows_the_switch(monkeypatch, medical_on, route):
    """The runtime rewrite of a coder task onto the family its text names must
    not pick a switched-off medical route - it would be refused one line later."""
    from CoScientist.config import get_settings

    task = _task("EXP-1", route="coder")
    task["description"] = "Extract PICO from the supplied abstract with get_pico."
    state = _forced_state(task)
    monkeypatch.setattr(get_settings().web, "medical_agent_enabled", medical_on)

    started = start_task(state, "EXP-1")
    assert started["route"] == route


def test_a_coder_task_the_runtime_moved_to_medical_goes_back_to_coder(monkeypatch):
    """Only a task PLANNED on medical has nothing to fall back on. One the runtime
    moved there itself (its text named a medical tool) returns to its planned
    coder route, as planned - and its dependents are not blocked with it."""
    from CoScientist.config import get_settings

    first = _task("EXP-1", route="coder")
    first["description"] = "Extract PICO from the supplied abstract with get_pico."
    second = _task("EXP-2", route="coder", depends_on=["EXP-1"])
    state = _forced_state(first, second)
    planned_task = state["experiment_runtime"]["tasks"]["EXP-1"]["task"]

    started = start_task(state, "EXP-1")
    assert started["route"] == "medical"
    mark_route_returned(state, "MedicalAgent")
    record_result(state, "EXP-1", started["attempt_id"],
                  {**_route_failure(), "error_code": "timeout", "retryable": True})
    retry_task(state, "EXP-1")
    monkeypatch.setattr(get_settings().web, "medical_agent_enabled", False)

    again = start_task(state, "EXP-1")
    assert again["route"] == "coder"
    assert again["route_agent"] == "CoderAgent"
    task_runtime = state["experiment_runtime"]["tasks"]["EXP-1"]
    assert task_runtime["task"] == planned_task
    assert any("back to the planned route" in h["reason"] for h in task_runtime["route_history"])
    assert state["experiment_runtime"]["tasks"]["EXP-2"]["status"] != "blocked"


def test_an_amendment_keeps_the_planned_route_of_a_task_the_runtime_moved(monkeypatch):
    """An amend that does not touch the route must not turn the runtime's own
    coder->medical move into a plan: the task still goes back to coder, and
    keeps the amendment when it does."""
    from CoScientist.config import get_settings

    task = _task("EXP-1", route="coder")
    task["description"] = "Extract PICO from the supplied abstract with get_pico."
    state = _forced_state(task)
    started = start_task(state, "EXP-1")
    assert started["route"] == "medical"
    mark_route_returned(state, "MedicalAgent")
    record_result(state, "EXP-1", started["attempt_id"],
                  {**_route_failure(), "error_code": "timeout", "retryable": True})
    retry_task(state, "EXP-1")
    amend_task(state, "EXP-1", {"launch_params": {"smiles": "CCN"}}, "different input")
    task_runtime = state["experiment_runtime"]["tasks"]["EXP-1"]
    assert task_runtime["planned_route"] == "coder"

    monkeypatch.setattr(get_settings().web, "medical_agent_enabled", False)
    again = start_task(state, "EXP-1")
    assert again["route"] == "coder"
    assert task_runtime["task"]["route"] == "coder"
    assert task_runtime["task"]["launch_params"] == {"smiles": "CCN"}


def test_session_route_agents_reads_the_executor_of_the_running_tree():
    """The planner and its critique sit beside the executor; they find it
    through the tree the session was built with."""
    from CoScientist.experiments.runtime.state_machine import session_route_agents

    executor = SimpleNamespace(tools=[
        SimpleNamespace(agent=SimpleNamespace(name="ExperimentAgent")),
        SimpleNamespace(agent=SimpleNamespace(name="CoderAgent")),
        object(),
    ])

    class _Root:
        def find_agent(self, name):
            return executor if name == "ExperimentExecutorAgent" else None

    planner = SimpleNamespace(root_agent=_Root())
    assert session_route_agents(planner) == frozenset({"ExperimentAgent", "CoderAgent"})
    # No executor in this tree (the main profile) or no tree at all: the YAML decides.
    assert session_route_agents(SimpleNamespace(root_agent=SimpleNamespace(find_agent=lambda n: None))) is None
    assert session_route_agents(None) is None


def test_fedot_switched_off_after_the_fallback_chose_it_does_not_strand_the_task():
    """The fallback moved a task onto fedot_mas after react_tools had spent its
    attempts, and then FEDOT went away. start_task must still open an attempt:
    raising attempt_budget_exhausted here would leave the task 'ready', a state
    neither retry_task nor fallback_task accepts, with the executor driven to
    start_task until the run gave out."""
    chain = ["react_tools", "fedot_mas", "coder"]
    on = ExperimentsSettings(route_fedot=True, fallback_react_tools=chain)
    off = ExperimentsSettings(route_fedot=False, fallback_react_tools=chain)
    live = frozenset({"ExperimentAgent", "FedotAgent", "CoderAgent"})
    retryable = {**_route_failure(), "error_code": "timeout", "retryable": True}
    state = _approved_state(_plan(_task("EXP-1", route="react_tools")))
    task_runtime = state["experiment_runtime"]["tasks"]["EXP-1"]
    for attempt in range(2):
        if attempt:
            retry_task(state, "EXP-1", settings=on)
        started = start_task(state, "EXP-1", settings=on, route_agents=live)
        mark_route_returned(state, "ExperimentAgent")
        record_result(state, "EXP-1", started["attempt_id"], retryable, settings=on, route_agents=live)
    assert fallback_task(state, "EXP-1", "react_tools spent", settings=on, route_agents=live)["route"] == "fedot_mas"

    started = start_task(state, "EXP-1", settings=off, route_agents=live)
    assert started["status"] == "success"
    assert started["route_agent"] == "ExperimentAgent"
    mark_route_returned(state, "ExperimentAgent")
    record_result(state, "EXP-1", started["attempt_id"], _route_failure(), settings=off, route_agents=live)
    # This is the third total attempt. No fourth automatic route is exposed.
    assert task_runtime["status"] == "failed"
    with pytest.raises(ExperimentRuntimeError, match="requires fallback_pending"):
        fallback_task(state, "EXP-1", "still empty", settings=off, route_agents=live)
