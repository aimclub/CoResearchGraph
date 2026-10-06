"""Logical output, the file that satisfies it, and repair of that link."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from CoScientist.experiments.runtime import (
    ExperimentRuntimeError,
    experiment_next_actions,
    get_experiment_plan,
    mark_route_returned,
    operation_attempt_count,
    record_result,
    recover_task_outputs,
    request_task_redo,
    start_task,
)
from CoScientist.experiments.runtime.outputs import (
    BINDINGS_KEY,
    PINS_KEY,
    RECOVERY_BUDGET,
    content_verified,
    contract_problem,
    publish_bindings,
    resolve_output,
)
from CoScientist.experiments.runtime.readiness import refresh_readiness
from CoScientist.experiments.schemas.models import ExpectedArtifact

from .helpers import _approved_state, _plan, _success_result, _task

_HASH_NAME = "651a36b4fc2043a5aa009f50b7732181.csv"


@pytest.mark.parametrize("property_verified,expected_status", [(None, "met"), (False, "partial")])
def test_requirement_filename_is_bound_after_registration_and_assessed(tmp_path, property_verified, expected_status):
    from CoScientist.requirements.coverage import projection_from_state
    from CoScientist.requirements.routing import normalize_statement

    raw = _task("EXP-1", artifact_name="measurements.csv")
    raw["design"]["target_refs"] = ["P-data"]
    raw["design"]["target_links"] = [{"requirement_id": "P-data", "role": "delivers"}]
    state = _approved_state(_plan(raw))
    state["normalized_statement"] = normalize_statement("Deliver measurements", {"parts": [{
        "id": "P-data", "kind": "deliverable", "formulation": "Measurements",
        "source": "user_request", "quote": "Deliver measurements",
    }]}).model_dump()
    started = start_task(state, "EXP-1")
    assert started["requirements"] == [{"id": "P-data", "kind": "deliverable",
                                        "formulation": "Measurements", "role": "delivers"}]
    path = tmp_path / "measurements.csv"
    path.write_text("value\n42\n")
    state["fedot_artifacts"] = [{"name": path.name, "workspace_path": str(path)}]
    mark_route_returned(state, "FedotAgent")
    result = _success_result("EXP-1")
    result["outputs"] = {"requirements": {"P-data": {
        "artifact_ids": [path.name], "summary": "Measured values", "grounded": True,
        "property_verified": property_verified,
    }}}
    result["outputs"]["requirements"]["P-data"]["artifact_ids"] = ["missing.csv"]
    with pytest.raises(ExperimentRuntimeError, match="0 artifacts"):
        record_result(state, "EXP-1", started["attempt_id"], result)
    assert state["experiment_runtime"]["results"] == []
    assert state["experiment_runtime"]["active_attempt_id"] == started["attempt_id"]
    result["outputs"]["requirements"]["P-data"]["artifact_ids"] = [path.name]
    recorded = record_result(state, "EXP-1", started["attempt_id"], result)["task_result"]
    artifact_id = next(a["artifact_id"] for a in recorded["artifacts"] if a["name"] == path.name)
    assert recorded["outputs"]["requirements"]["P-data"]["artifact_ids"] == [artifact_id]
    assert result["outputs"]["requirements"]["P-data"]["artifact_ids"] == [path.name]
    assert projection_from_state(state)[0]["status"] == expected_status


@pytest.mark.parametrize("reference", ["shared.csv", "missing.csv"])
def test_requirement_artifact_reference_never_guesses(reference):
    from types import SimpleNamespace
    from CoScientist.experiments.runtime.artifacts import resolve_result_artifact_refs

    artifacts = [SimpleNamespace(artifact_id=f"ART-{i}", name="shared.csv", output_id=None,
                                 workspace_path=f"/attempt/{i}/shared.csv", session_artifact_id=None)
                 for i in range(2)]
    with pytest.raises(ExperimentRuntimeError, match="exact, unique reference"):
        resolve_result_artifact_refs({"requirements": {"P": {"artifact_ids": [reference]}}}, artifacts)
    resolved = resolve_result_artifact_refs({"requirements": {
        "P": {"artifact_ids": ["ART-1"]}, "other": {"summary": "Not delivered"},
    }}, artifacts)
    assert resolved["requirements"]["P"]["artifact_ids"] == ["ART-1"]
    assert "artifact_ids" not in resolved["requirements"]["other"]


def test_published_filename_resolves_its_bound_output_not_the_duplicate_raw_log():
    data = {'artifact_id': 'ART-data', 'name': 'response-hash.json', 'role': 'data', 'output_id': 'measurements'}
    log = {'artifact_id': 'ART-log', 'name': 'response-hash.json', 'role': 'log'}
    runtime = {'results': [{'task_id': 'EXP-1', 'status': 'success', 'artifacts': [log, data]}],
               BINDINGS_KEY: {'EXP-1/measurements': {'task_id': 'EXP-1', 'output_id': 'measurements',
                   'preferred_name': 'measurements.json', 'artifact_id': 'ART-data', 'current': True}}}
    assert resolve_output(runtime, source_task_id='EXP-1', source_artifact_id='response-hash.json') is data
    other = {**data, 'artifact_id': 'ART-other', 'output_id': 'other'}
    runtime['results'][0]['artifacts'].append(other)
    runtime[BINDINGS_KEY]['EXP-1/other'] = {'task_id': 'EXP-1', 'output_id': 'other',
        'preferred_name': 'other.json', 'artifact_id': 'ART-other', 'current': True}
    with pytest.raises(ExperimentRuntimeError, match='matches outputs'):
        resolve_output(runtime, source_task_id='EXP-1', source_artifact_id='response-hash.json')


def _csv_bytes() -> bytes:
    return b"Smiles,QED\nCCCc1c(C)nc(-c2cnn(C)c2)nc1C,0.81\n"


def test_plain_text_is_not_assumed_to_be_a_table(tmp_path):
    path = tmp_path / "answer.txt"
    path.write_text("A complete one-line explanation.")
    expected = ExpectedArtifact(name=path.name, role="data", media_type="text/plain",
                                description="An explanation, without a table contract.")
    assert contract_problem({"workspace_path": str(path)}, expected) is None


def test_binary_content_must_be_opened_before_it_is_verified(tmp_path, monkeypatch):
    path = tmp_path / 'sample.bin'
    path.write_bytes(b'actual bytes')
    expected = ExpectedArtifact(name=path.name, role='data', description='Binary result')
    assert content_verified({'workspace_path': str(path)}, expected) is True

    def unreadable(*args, **kwargs):
        raise PermissionError('not readable')

    monkeypatch.setattr(Path, 'open', unreadable)
    assert content_verified({'workspace_path': str(path)}, expected) is False


@pytest.mark.parametrize("payload", [[{"value": 1}, {}], [{"value": 1}, 2]])
def test_json_checks_required_fields_in_every_record(tmp_path, payload):
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps(payload))
    expected = ExpectedArtifact(name=path.name, role="data", columns=["value"],
                                description="Every measurement must contain value.")
    assert contract_problem({"workspace_path": str(path)}, expected) is not None


@pytest.mark.parametrize("row", ["1", "1,2,3"])
def test_csv_conversion_cannot_pad_or_discard_fields(tmp_path, row):
    from CoScientist.experiments.runtime.outputs import convert_csv_workspace_to_json

    path = tmp_path / "measurements.csv"
    path.write_text("first,second\n" + row + "\n")
    expected = ExpectedArtifact(name=path.name, role="data", media_type="text/csv",
                                description="Measurements with two fields.")
    assert contract_problem({"workspace_path": str(path)}, expected) is not None
    assert convert_csv_workspace_to_json(str(path), dest_name="measurements.json") is None
    assert not (tmp_path / "measurements.json").exists()


def test_json_metadata_cannot_satisfy_csv_even_without_declared_columns(tmp_path):
    from CoScientist.experiments.runtime.outputs import convert_csv_workspace_to_json

    path = tmp_path / "measurements.csv"
    path.write_text(json.dumps({"rows": 12, "columns": ["smiles", "score"],
                                "path": "/remote/measurements.csv"}, indent=2))
    expected = ExpectedArtifact(name="measurements.csv", role="data", media_type="text/csv",
                                description="The actual measurement rows.")
    assert contract_problem({"workspace_path": str(path)}, expected) is not None
    assert content_verified({"workspace_path": str(path)}, expected) is False
    assert convert_csv_workspace_to_json(str(path), dest_name="measurements.json") is None
    assert not (tmp_path / "measurements.json").exists()


def _consumer(task_id: str, *, depends_on: list[str]) -> dict:
    task = _task(task_id, depends_on=depends_on, artifact_name=f"{task_id.lower()}-out.csv")
    task["input_data"] = [
        {
            "data_id": "candidates",
            "kind": "task_artifact",
            "description": "Candidates produced by EXP-1",
            "source_task_id": "EXP-1",
            "source_artifact_id": "th_candidates_15.json",
            "source_output_id": "candidates",
        }
    ]
    return task


def _c2_plan():
    producer = _task("EXP-1", artifact_name="th_candidates_15.json")
    producer["expected_artifacts"] = [
        {
            "name": "th_candidates_15.json",
            "output_id": "candidates",
            "role": "data",
            "media_type": "application/json",
            "required": True,
            "description": "Generated TH activator candidates.",
            "columns": ["Smiles"],
        }
    ]
    return _plan(
        producer,
        _consumer("EXP-2", depends_on=["EXP-1"]),
        _consumer("EXP-3", depends_on=["EXP-1"]),
        _task("EXP-4", depends_on=["EXP-2", "EXP-3"]),
        _task("EXP-5", depends_on=["EXP-4"]),
    )


def _record_csv(state: dict, tmp_path: Path) -> dict:
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    path = tmp_path / _HASH_NAME
    path.write_bytes(_csv_bytes())
    state.setdefault("fedot_artifacts", []).append(
        {
            "name": _HASH_NAME,
            "workspace_path": str(path),
            "role": "data",
            "media_type": "text/csv",
            "tool": "generate_mols",
        }
    )
    stored = record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))
    return stored


def _break_like_saved_c2(state: dict, tmp_path: Path) -> None:
    """Drop the logical link the way the saved run did, and keep the CSV."""
    runtime = state["experiment_runtime"]
    result = runtime["results"][-1]
    csv_path = tmp_path / _HASH_NAME
    pointer = tmp_path / "pointer.json"
    pointer.write_text(json.dumps({"url": "http://example.com/not-the-table.csv"}), encoding="utf-8")
    data = next(item for item in result["artifacts"] if item.get("role") == "data")
    data["name"] = _HASH_NAME
    data["output_id"] = None
    data["workspace_path"] = str(csv_path)
    data["media_type"] = "text/csv"
    data["bucket"] = None
    data["s3_key"] = None
    data["external_url"] = None
    result["artifacts"].append({
        "artifact_id": "ART-pointer",
        "name": "th_candidates_15.json",
        "role": "data",
        "workspace_path": str(pointer),
        "media_type": "application/json",
        "durability": "workspace",
    })
    result["artifacts"].append({
        "artifact_id": "ART-family",
        "name": "family_outputs.json",
        "role": "data",
        "workspace_path": str(pointer),
        "media_type": "application/json",
        "durability": "workspace",
    })
    runtime.pop(BINDINGS_KEY, None)
    for task_id in ("EXP-2", "EXP-3", "EXP-4"):
        runtime["tasks"][task_id]["status"] = "blocked"
        runtime["tasks"][task_id]["blocked_reason"] = {
            "code": "required_upstream_artifact_missing",
            "artifacts": [{"source_task_id": "EXP-1", "source_output_id": "candidates"}],
        }
    runtime["tasks"]["EXP-5"]["status"] = "blocked"
    runtime["tasks"]["EXP-5"]["blocked_reason"] = {"code": "operator_cancelled"}
    runtime["phase"] = "reporting"


def test_saved_c2_csv_is_registered_without_regenerating(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    state = _approved_state(_c2_plan())
    stored = _record_csv(state, tmp_path)
    runtime = state["experiment_runtime"]
    key = runtime["tasks"]["EXP-1"]["operation_key"]
    attempts_before = operation_attempt_count(state, key)
    order_before = list(runtime["tasks"]["EXP-1"]["attempt_order"])
    assert stored["task_result"]["status"] == "success"
    assert runtime["tasks"]["EXP-2"]["status"] == "ready"
    assert runtime["tasks"]["EXP-3"]["status"] == "ready"
    assert runtime["phase"] == "execution"

    _break_like_saved_c2(state, tmp_path)
    assert runtime["tasks"]["EXP-2"]["status"] == "blocked"
    assert experiment_next_actions(state) == [
        {"tool": "recover_task_outputs", "arguments": {"task_id": "EXP-1"}},
    ]
    repaired = recover_task_outputs(state, "EXP-1")

    assert repaired["status"] == "success"
    assert operation_attempt_count(state, key) == attempts_before
    assert runtime["tasks"]["EXP-1"]["attempt_order"] == order_before
    binding = runtime[BINDINGS_KEY]["EXP-1/candidates"]
    assert binding["output_id"] == "candidates"
    assert binding["preferred_name"] == "th_candidates_15.json"
    body = Path(
        next(item["workspace_path"] for item in runtime["results"][-1]["artifacts"]
             if item.get("artifact_id") == binding["artifact_id"])
    ).read_text(encoding="utf-8")
    parsed = json.loads(body)
    assert isinstance(parsed, list) and parsed
    assert "Smiles" in parsed[0]
    assert "CCCc1c" in parsed[0]["Smiles"]
    assert runtime["tasks"]["EXP-2"]["status"] == "ready"
    assert runtime["tasks"]["EXP-3"]["status"] == "ready"
    assert runtime["tasks"]["EXP-4"]["status"] == "pending"
    assert runtime["tasks"]["EXP-5"]["status"] == "blocked"
    assert runtime["tasks"]["EXP-5"]["blocked_reason"]["code"] == "operator_cancelled"
    assert runtime["phase"] == "execution"
    assert [item["tool"] for item in experiment_next_actions(state)] == ["start_task", "start_task"]
    assert not any(item["tool"] == "recover_task_outputs" for item in experiment_next_actions(state))

    again = recover_task_outputs(state, "EXP-1")
    assert again["idempotent"] is True
    assert again["recovery_used"] == repaired["recovery_used"]
    assert operation_attempt_count(state, key) == attempts_before


def test_pause_keeps_reporting_while_a_user_cancel_stays_blocked(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    state = _approved_state(_c2_plan())
    _record_csv(state, tmp_path)
    _break_like_saved_c2(state, tmp_path)
    state["experiment_runtime"]["automation_paused"] = True
    recover_task_outputs(state, "EXP-1")
    runtime = state["experiment_runtime"]
    assert runtime["phase"] == "reporting"
    assert runtime["tasks"]["EXP-2"]["status"] == "ready"
    assert runtime["tasks"]["EXP-5"]["blocked_reason"]["code"] == "operator_cancelled"


def test_resume_keeps_the_pinned_input_version(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    state = _approved_state(_c2_plan())
    _record_csv(state, tmp_path)
    started = start_task(state, "EXP-2")
    assert started["status"] == "success"
    runtime = state["experiment_runtime"]
    pin = runtime[PINS_KEY]["EXP-2/candidates"]
    pinned_id = pin["artifact_id"]
    runtime["results"][-1]["artifacts"].append({
        **runtime["results"][-1]["artifacts"][0],
        "artifact_id": "ART-later",
        "name": "later.json",
    })
    runtime[BINDINGS_KEY]["EXP-1/candidates"]["artifact_id"] = "ART-later"
    resolved = resolve_output(
        runtime,
        source_task_id="EXP-1",
        source_output_id="candidates",
        source_artifact_id="th_candidates_15.json",
        consumer_task_id="EXP-2",
        data_id="candidates",
    )
    assert resolved["artifact_id"] == pinned_id
    assert resolved["artifact_id"] != "ART-later"


def test_republish_does_not_duplicate_the_result_or_the_binding(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    state = _approved_state(_plan(_task("EXP-1", artifact_name="candidates.csv")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    path = tmp_path / "whatever.csv"
    path.write_text("Smiles,QED\nCCO,0.5\n", encoding="utf-8")
    state["fedot_artifacts"] = [{
        "name": "whatever.csv",
        "workspace_path": str(path),
        "role": "data",
        "media_type": "text/csv",
    }]
    payload = _success_result("EXP-1")
    first = record_result(state, "EXP-1", started["attempt_id"], payload)
    second = record_result(state, "EXP-1", started["attempt_id"], payload)
    runtime = state["experiment_runtime"]
    assert second["idempotent"] is True
    assert len(runtime["results"]) == 1
    binding = runtime[BINDINGS_KEY]["EXP-1/candidates"]
    publish_bindings(
        runtime,
        task_id="EXP-1",
        expected=[ExpectedArtifact.model_validate(item) for item in runtime["tasks"]["EXP-1"]["task"]["expected_artifacts"]],
        artifacts=first["task_result"]["artifacts"],
        result_id=binding["result_id"],
        result_version=binding["result_version"],
        attempt_id=binding["attempt_id"],
    )
    assert runtime[BINDINGS_KEY]["EXP-1/candidates"]["history"] == []
    assert len(runtime["results"]) == 1


def test_two_matching_files_are_not_chosen_at_random():
    runtime = {
        "results": [{
            "task_id": "EXP-1",
            "status": "success",
            "artifacts": [
                {"artifact_id": "ART-1", "name": "a.csv", "output_id": "candidates", "role": "data"},
                {"artifact_id": "ART-2", "name": "b.csv", "output_id": "candidates", "role": "data"},
            ],
        }]
    }
    with pytest.raises(ExperimentRuntimeError) as raised:
        resolve_output(runtime, source_task_id="EXP-1", source_output_id="candidates")
    assert raised.value.code == "output_ambiguous"


def test_json_pointer_and_bare_url_do_not_satisfy_a_table(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    pointer = tmp_path / "th_candidates_15.json"
    pointer.write_text(json.dumps({"url": "https://example.com/table.csv"}), encoding="utf-8")
    state = _approved_state(_plan(_task("EXP-1", artifact_name="th_candidates_15.json")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    state["fedot_artifacts"] = [{
        "name": "th_candidates_15.json",
        "workspace_path": str(pointer),
        "role": "data",
        "media_type": "application/json",
    }]
    with pytest.raises(ExperimentRuntimeError) as raised:
        record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))
    assert raised.value.code == "result_incomplete"

    expected = ExpectedArtifact.model_validate({
        "name": "candidates.csv",
        "role": "data",
        "media_type": "text/csv",
        "description": "Table",
        "columns": ["Smiles"],
    })
    assert contract_problem(
        {"external_url": "https://example.com/candidates.csv", "role": "data"},
        expected,
    ) == "a URL is not the required table"


def test_schema_is_not_passed_because_a_file_exists():
    task = _task("EXP-1", artifact_name="candidates.csv")
    task["success_criteria"].append({
        "criterion_id": "EXP-1-SCHEMA",
        "description": "The table has the declared columns.",
        "kind": "schema",
        "verification": "Open the table.",
    })
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    state["fedot_artifacts"] = [{
        "name": "candidates.csv",
        "bucket": "managed-experiments",
        "s3_key": "generated/candidates.csv",
        "role": "data",
        "media_type": "text/csv",
    }]
    payload = _success_result("EXP-1")
    payload["criteria_checks"].append({
        "criterion_id": "EXP-1-SCHEMA",
        "passed": True,
        "details": "file key exists",
    })
    stored = record_result(state, "EXP-1", started["attempt_id"], payload)
    assert stored["task_result"]["status"] == "failure"
    schema = next(
        item for item in stored["task_result"]["criteria_checks"]
        if item["criterion_id"] == "EXP-1-SCHEMA"
    )
    assert schema["passed"] is False
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] != "done"


def test_empty_table_follows_the_contract(tmp_path):
    empty = tmp_path / "candidates.csv"
    empty.write_text("Smiles,QED\n", encoding="utf-8")
    expected = ExpectedArtifact.model_validate({
        "name": "candidates.csv",
        "role": "data",
        "media_type": "text/csv",
        "description": "Table",
        "columns": ["Smiles"],
    })
    artifact = {"workspace_path": str(empty), "role": "data", "name": "candidates.csv"}
    assert contract_problem(artifact, expected) == "table is empty"
    allowed = expected.model_copy(update={"allow_empty": True})
    assert contract_problem(artifact, allowed) is None


def test_download_failure_does_not_spend_a_generation_attempt(tmp_path, monkeypatch):
    from CoScientist.config import get_settings
    from CoScientist.experiments.runtime import artifacts as artifact_mod

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    task = _task("EXP-1", artifact_name="candidates.csv")
    task["expected_artifacts"][0]["columns"] = ["Smiles"]
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    path = tmp_path / "candidates.csv"
    path.write_text("Smiles,QED\nCCO,0.5\n", encoding="utf-8")
    state["fedot_artifacts"] = [{
        "name": "candidates.csv",
        "workspace_path": str(path),
        "role": "data",
        "media_type": "text/csv",
    }]
    record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))
    runtime = state["experiment_runtime"]
    key = runtime["tasks"]["EXP-1"]["operation_key"]
    attempts = operation_attempt_count(state, key)
    data = runtime["results"][-1]["artifacts"][0]
    data["name"] = _HASH_NAME
    data["output_id"] = None
    data["workspace_path"] = None
    data["external_url"] = "https://example.com/651a36b4fc2043a5aa009f50b7732181.csv"
    data["bucket"] = None
    data["s3_key"] = None
    runtime.pop(BINDINGS_KEY, None)

    monkeypatch.setattr(artifact_mod, "_materialize_signed_artifact", lambda *a, **k: None)
    failed = recover_task_outputs(state, "EXP-1")
    assert failed["status"] == "blocked"
    assert failed["code"] == "materialization_unavailable"
    assert failed["recovery_used"] == 1
    assert operation_attempt_count(state, key) == attempts
    assert runtime["tasks"]["EXP-1"]["status"] == "done"
    assert BINDINGS_KEY not in runtime or "EXP-1/candidates" not in runtime.get(BINDINGS_KEY, {})

    def _write(*_args, **_kwargs):
        saved = tmp_path / "recovered.csv"
        saved.write_text("Smiles,QED\nCCO,0.5\n", encoding="utf-8")
        return str(saved)

    monkeypatch.setattr(artifact_mod, "_materialize_signed_artifact", _write)
    repaired = recover_task_outputs(state, "EXP-1")
    assert repaired["status"] == "success"
    assert repaired["recovery_used"] == 2
    assert operation_attempt_count(state, key) == attempts
    assert runtime["tasks"]["EXP-1"]["attempt_order"] == [started["attempt_id"]]


def test_refresh_leaves_a_user_cancel_blocked():
    state = _approved_state(_plan(_task("EXP-1"), _task("EXP-2", depends_on=["EXP-1"])))
    runtime = state["experiment_runtime"]
    runtime["tasks"]["EXP-1"]["status"] = "done"
    runtime["tasks"]["EXP-2"]["status"] = "blocked"
    runtime["tasks"]["EXP-2"]["blocked_reason"] = {"code": "operator_cancelled"}
    refresh_readiness(runtime)
    assert runtime["tasks"]["EXP-2"]["status"] == "blocked"
    assert runtime["tasks"]["EXP-2"]["blocked_reason"]["code"] == "operator_cancelled"


def test_two_data_files_on_recovery_are_ambiguous(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    state = _approved_state(_plan(_task("EXP-1", artifact_name="candidates.csv")))
    started = start_task(state, "EXP-1")
    mark_route_returned(state, "FedotAgent")
    path = tmp_path / "candidates.csv"
    path.write_text("Smiles,QED\nCCO,0.5\n", encoding="utf-8")
    state["fedot_artifacts"] = [{
        "name": "candidates.csv",
        "workspace_path": str(path),
        "role": "data",
        "media_type": "text/csv",
    }]
    record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))
    runtime = state["experiment_runtime"]
    other = tmp_path / "other.csv"
    other.write_text("Smiles,QED\nCCC,0.2\n", encoding="utf-8")
    data = runtime["results"][-1]["artifacts"][0]
    data["output_id"] = None
    data["name"] = "a.csv"
    runtime["results"][-1]["artifacts"].append({
        **data,
        "artifact_id": "ART-other",
        "name": "b.csv",
        "workspace_path": str(other),
    })
    runtime.pop(BINDINGS_KEY, None)
    with pytest.raises(ExperimentRuntimeError) as raised:
        recover_task_outputs(state, "EXP-1")
    assert raised.value.code == "output_ambiguous"
    assert "EXP-1/candidates" not in (runtime.get(BINDINGS_KEY) or {})


def test_bucket_key_is_a_location_not_verified_data():
    expected = ExpectedArtifact.model_validate({
        "name": "candidates.json",
        "role": "data",
        "media_type": "application/json",
        "description": "Generated candidates.",
        "columns": ["Smiles"],
    })
    located = {
        "name": "candidates.json",
        "role": "data",
        "bucket": "managed-experiments",
        "s3_key": "generated/candidates.json",
        "media_type": "application/json",
    }
    assert contract_problem(located, expected) == "bucket/key is a location, not verified data"
    assert content_verified(located, expected) is False
    plain = ExpectedArtifact.model_validate({
        "name": "candidates.csv",
        "role": "data",
        "media_type": "text/csv",
        "description": "Table without a declared schema.",
    })
    assert content_verified(
        {"bucket": "managed-experiments", "s3_key": "generated/candidates.csv", "role": "data"},
        plain,
    ) is False


def test_serialized_mcp_endpoint_is_not_checked_as_a_dataset(tmp_path):
    path = tmp_path / "mcp_endpoint.json"
    path.write_text(json.dumps("http://127.0.0.1:9000/mcp"), encoding="utf-8")
    expected = ExpectedArtifact.model_validate({
        "name": "mcp_endpoint", "role": "mcp_server", "description": "Served MCP URL",
    })
    assert contract_problem({"workspace_path": str(path), "role": "mcp_server"}, expected) is None


def test_json_requires_structure_and_required_fields(tmp_path):
    expected = ExpectedArtifact.model_validate({
        "name": "candidates.json",
        "role": "data",
        "media_type": "application/json",
        "description": "Generated candidates.",
        "columns": ["Smiles"],
    })
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"notes": "no structure"}), encoding="utf-8")
    assert contract_problem(
        {"workspace_path": str(missing), "role": "data", "name": "candidates.json"},
        expected,
    ) == "missing fields: Smiles"
    scalar = tmp_path / "scalar.json"
    scalar.write_text(json.dumps("CC"), encoding="utf-8")
    assert contract_problem(
        {"workspace_path": str(scalar), "role": "data", "name": "candidates.json"},
        expected,
    ) == "JSON is not an object or array"
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([{"Smiles": "CCO"}]), encoding="utf-8")
    assert contract_problem(
        {"workspace_path": str(rows), "role": "data", "name": "candidates.json"},
        expected,
    ) is None
    pointer = tmp_path / "pointer.json"
    pointer.write_text(json.dumps({"url": "https://example.com/table.csv"}), encoding="utf-8")
    assert contract_problem(
        {"workspace_path": str(pointer), "role": "data", "name": "candidates.json"},
        expected,
    ) == "JSON pointer is not the required table"


def test_recover_tool_is_registered_and_repairs_once(tmp_path, monkeypatch):
    import asyncio

    from CoScientist.config import get_settings
    from CoScientist.experiments.runtime.tools import ExperimentControlToolset

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    tools = asyncio.run(ExperimentControlToolset().get_tools())
    assert "recover_task_outputs" in {getattr(tool, "name", "") for tool in tools}
    state = _approved_state(_c2_plan())
    _record_csv(state, tmp_path)
    _break_like_saved_c2(state, tmp_path)

    class _Ctx:
        def __init__(self, values):
            self.state = values

    repaired = ExperimentControlToolset().recover_task_outputs("EXP-1", _Ctx(state))
    assert repaired["status"] == "success"
    assert state["experiment_runtime"]["tasks"]["EXP-2"]["status"] == "ready"
    assert state["experiment_runtime"]["tasks"]["EXP-3"]["status"] == "ready"


def test_session_roundtrip_keeps_bindings_pins_and_recovery_budget(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    state = _approved_state(_c2_plan())
    _record_csv(state, tmp_path)
    _break_like_saved_c2(state, tmp_path)
    runtime = state["experiment_runtime"]
    runtime["tasks"]["EXP-1"]["output_recovery_used"] = 1
    recover_task_outputs(state, "EXP-1")
    started = start_task(state, "EXP-2")
    assert started["status"] == "success"
    saved = json.loads(json.dumps(state, default=str))
    loaded = saved["experiment_runtime"]
    assert loaded[PINS_KEY]["EXP-2/candidates"]["artifact_id"] == loaded[BINDINGS_KEY]["EXP-1/candidates"]["artifact_id"]
    assert loaded["tasks"]["EXP-1"]["output_recovery_used"] == 2
    assert resolve_output(
        loaded,
        source_task_id="EXP-1",
        source_output_id="candidates",
        consumer_task_id="EXP-2",
        data_id="candidates",
    )["artifact_id"] == loaded[PINS_KEY]["EXP-2/candidates"]["artifact_id"]


def test_recovery_does_not_rewrite_a_pinned_input(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    state = _approved_state(_c2_plan())
    _record_csv(state, tmp_path)
    started = start_task(state, "EXP-2")
    assert started["status"] == "success"
    runtime = state["experiment_runtime"]
    pin = runtime[PINS_KEY]["EXP-2/candidates"]
    pinned_id = pin["artifact_id"]
    pinned = next(
        art for art in runtime["results"][-1]["artifacts"] if art.get("artifact_id") == pinned_id
    )
    path = Path(pinned["workspace_path"])
    before = path.read_bytes()
    runtime.pop(BINDINGS_KEY, None)
    runtime["tasks"]["EXP-3"]["status"] = "blocked"
    runtime["tasks"]["EXP-3"]["blocked_reason"] = {
        "code": "required_upstream_artifact_missing",
        "artifacts": [{"source_task_id": "EXP-1", "source_output_id": "candidates"}],
    }
    runtime["phase"] = "execution"
    repaired = recover_task_outputs(state, "EXP-1")
    assert repaired["status"] == "success"
    kept = next(
        art for art in runtime["results"][-1]["artifacts"] if art.get("artifact_id") == pinned_id
    )
    assert kept["workspace_path"] == str(path)
    assert path.read_bytes() == before
    assert resolve_output(
        runtime,
        source_task_id="EXP-1",
        source_output_id="candidates",
        consumer_task_id="EXP-2",
        data_id="candidates",
    )["artifact_id"] == pinned_id


def test_targeted_redo_rebinds_consumers_to_new_data_after_session_reload(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    state = _approved_state(_plan(
        _c2_plan().tasks[0].model_dump(mode="json"),
        _consumer("EXP-2", depends_on=["EXP-1"]),
    ))
    prior = _record_csv(state, tmp_path)["task_result"]
    started = start_task(state, "EXP-2")
    runtime = state["experiment_runtime"]
    old_pin = dict(runtime[PINS_KEY]["EXP-2/candidates"])
    mark_route_returned(state, "FedotAgent")
    consumer_output = tmp_path / "exp-2-out.csv"
    consumer_output.write_text("Smiles,score\nCCO,0.5\n", encoding="utf-8")
    state["fedot_artifacts"].append({
        "name": "exp-2-out.csv", "role": "data", "media_type": "text/csv",
        "workspace_path": str(consumer_output),
    })
    record_result(state, "EXP-2", started["attempt_id"], _success_result("EXP-2"))
    # An unrelated completed consumer's pin must survive targeted redo.
    runtime[PINS_KEY]["EXP-9/external"] = {"consumer_task_id": "EXP-9", "artifact_id": "ART-external"}
    request_task_redo(state, ["EXP-1"], "Regenerate the candidates.")
    assert "EXP-2/candidates" not in runtime[PINS_KEY]
    assert runtime[PINS_KEY]["EXP-9/external"]["artifact_id"] == "ART-external"
    revised_dir = tmp_path / "revision-2"
    revised_dir.mkdir()
    latest = _record_csv(state, revised_dir)["task_result"]
    state = json.loads(json.dumps(state, default=str))
    runtime = state["experiment_runtime"]
    started = start_task(state, "EXP-2")
    assert started["status"] == "success"
    fresh_pin = runtime[PINS_KEY]["EXP-2/candidates"]
    assert fresh_pin["artifact_id"] != old_pin["artifact_id"]
    assert fresh_pin["attempt_id"] == latest["attempt_id"]
    assert resolve_output(runtime, source_task_id="EXP-1", source_output_id="candidates",
                          consumer_task_id="EXP-2", data_id="candidates")["artifact_id"] == fresh_pin["artifact_id"]
    assert any(result["result_id"] == prior["result_id"] for result in runtime["results"])


def test_exhausted_recovery_stops_without_a_loop_and_continues_independent_work(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    state = _approved_state(_plan(
        _c2_plan().tasks[0].model_dump(mode="json"),
        _consumer("EXP-2", depends_on=["EXP-1"]),
        _task("EXP-9"),
    ))
    _record_csv(state, tmp_path)
    runtime = state["experiment_runtime"]
    runtime["results"][-1]["artifacts"][0]["output_id"] = None
    runtime["results"][-1]["artifacts"][0]["name"] = _HASH_NAME
    runtime.pop(BINDINGS_KEY, None)
    runtime["tasks"]["EXP-2"]["status"] = "blocked"
    runtime["tasks"]["EXP-2"]["blocked_reason"] = {
        "code": "required_upstream_artifact_missing",
        "artifacts": [{"source_task_id": "EXP-1", "source_output_id": "candidates"}],
    }
    runtime["tasks"]["EXP-1"]["output_recovery_used"] = RECOVERY_BUDGET
    runtime["tasks"]["EXP-9"]["status"] = "ready"
    runtime["phase"] = "execution"
    assert [item for item in experiment_next_actions(state) if item["tool"] == "recover_task_outputs"] == [
        {"tool": "recover_task_outputs", "arguments": {"task_id": "EXP-1"}},
    ]
    stopped = recover_task_outputs(state, "EXP-1")
    assert stopped["code"] == "output_recovery_budget"
    assert stopped["recovery_used"] == RECOVERY_BUDGET
    assert runtime["tasks"]["EXP-2"]["blocked_reason"]["code"] == "output_recovery_exhausted"
    assert runtime["tasks"]["EXP-9"]["status"] == "ready"
    assert runtime["delivery_status"] == "partial"
    assert len(runtime["output_recovery_stops"]) == 1
    follow = experiment_next_actions(state)
    assert not any(item["tool"] == "recover_task_outputs" for item in follow)
    assert {"tool": "start_task", "arguments": {"task_id": "EXP-9"}} in follow
    again = recover_task_outputs(state, "EXP-1")
    assert again["idempotent"] is True
    assert again["recovery_used"] == RECOVERY_BUDGET
    assert len(runtime["output_recovery_stops"]) == 1
    assert not any(
        item["tool"] == "recover_task_outputs" for item in experiment_next_actions(state)
    )


def test_user_cancel_does_not_start_a_recovery_loop(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    state = _approved_state(_c2_plan())
    _record_csv(state, tmp_path)
    _break_like_saved_c2(state, tmp_path)
    runtime = state["experiment_runtime"]
    for task_id in ("EXP-2", "EXP-3", "EXP-4", "EXP-5"):
        runtime["tasks"][task_id]["status"] = "blocked"
        runtime["tasks"][task_id]["blocked_reason"] = {"code": "operator_cancelled"}
    runtime["phase"] = "execution"
    for _ in range(4):
        actions = experiment_next_actions(state)
        assert not any(item["tool"] == "recover_task_outputs" for item in actions)
    viewed = get_experiment_plan(state)
    assert viewed["next_actions"] == []
    assert "EXP-5:operator_cancelled" in viewed["execution_note"]


def test_manual_review_blocks_recovery_actions(tmp_path, monkeypatch):
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    state = _approved_state(_c2_plan())
    _record_csv(state, tmp_path)
    _break_like_saved_c2(state, tmp_path)
    state["experiment_runtime"]["manual_review_required"] = True
    assert experiment_next_actions(state) == []
