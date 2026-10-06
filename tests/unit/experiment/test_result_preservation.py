"""Regression: record_result must not replace delivered data with model prose."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from CoScientist.config import get_settings
from CoScientist.agents.callbacks.tool_callbacks import record_experiment_tool_observation
from CoScientist.experiments.runtime import ExperimentRuntimeError, record_result, start_task
from .helpers import _approved_state, _plan, _route_return, _success_result, _task


@pytest.fixture
def running(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings().code_exec, "workspace_root", str(tmp_path))
    monkeypatch.setattr(get_settings().experiments, "route_fedot", True)
    monkeypatch.setattr(get_settings().web, "fedot_fallback_enabled", True)
    task = _task("EXP-1", artifact_name="overview.json")
    task["expected_artifacts"][0]["media_type"] = "application/json"
    task["design"]["analysis_artifacts"] = [{
        "name": "overview.json", "role": "metrics_table", "prepare_via": "mcp",
        "path_or_tool": "estimate_property",
    }]
    state = _approved_state(_plan(task))
    started = start_task(state, "EXP-1")
    path = tmp_path / "experiment_artifacts" / "EXP-1" / started["attempt_id"] / "overview.json"
    return state, started, path


def finish(state, started, outputs):
    _route_return(state, "FedotAgent")
    return record_result(state, "EXP-1", started["attempt_id"],
                         _success_result("EXP-1") | {"outputs": outputs})["task_result"]


@pytest.mark.parametrize("outputs", [{}, {"overview.json": "Overview saved; contains 225 metabolites."}])
def test_delivered_file_survives_summary_or_omitted_output(running, outputs):
    state, started, path = running
    data = b'{"metabolites":225,"clusters":["A","B","C","D","E"]}\n'
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    result = finish(state, started, outputs)
    assert result["status"] == "success"
    assert path.read_bytes() == data
    artifact = next(a for a in result["artifacts"] if a["name"] == "overview.json")
    assert artifact["checksum_sha256"] == hashlib.sha256(data).hexdigest()
    assert len(state["experiment_runtime"]["tasks"]["EXP-1"]["attempt_order"]) == 1


def test_callback_captures_explicitly_bound_structured_response(running):
    state, started, path = running
    data = {"metabolites": 225, "clusters": ["A", "B", "C", "D", "E"]}
    record_experiment_tool_observation(SimpleNamespace(name="estimate_property"), {},
        SimpleNamespace(state=state), {"structuredContent": data, "isError": False})
    assert not path.exists()  # The attempt may produce more responses.
    result = finish(state, started, {"overview.json": "A summary instead of the actual data."})
    assert result["status"] == "success"
    assert json.loads(path.read_text()) == data
    raw = next(a for a in state["mcp_artifacts"] if a["role"] == "log")
    assert json.loads(Path(raw["workspace_path"]).read_text()) == data


def test_unbound_response_and_prose_do_not_fulfil_json_output(running):
    state, started, path = running
    state["experiment_runtime"]["tasks"]["EXP-1"]["task"]["design"]["analysis_artifacts"] = []
    record_experiment_tool_observation(SimpleNamespace(name="estimate_property"), {},
        SimpleNamespace(state=state), {"structuredContent": {"count": 225}})
    with pytest.raises(ExperimentRuntimeError, match="missing required evidence"):
        finish(state, started, {"overview.json": "Overview saved."})
    assert not path.exists()


def test_bound_output_preserves_all_calls_and_their_inputs(running):
    state, started, path = running
    responses = []
    for value in (1, 2, 3):
        args, data = {"sample": value}, {"measurement": value * 10}
        responses.append({"tool": "estimate_property", "arguments": args, "response": data})
        record_experiment_tool_observation(SimpleNamespace(name="estimate_property"), args,
            SimpleNamespace(state=state), {"structuredContent": data, "isError": False})
    result = finish(state, started, {"overview.json": "Only the first value matters."})
    assert result["status"] == "success"
    assert json.loads(path.read_text()) == {"tool_responses": responses}
    assert len([a for a in state["mcp_artifacts"] if a["role"] == "log"]) == 3


def test_tool_response_binding_survives_adk_event_delta(running):
    import copy
    from google.adk.sessions.state import State

    state, started, path = running
    parent = copy.deepcopy(state)
    for sample in (1, 2):
        delta = {}
        child = State(copy.deepcopy(parent), delta)
        record_experiment_tool_observation(SimpleNamespace(name="estimate_property"),
            {"sample": sample}, SimpleNamespace(state=child),
            {"structuredContent": {"measurement": sample * 10}})
        parent.update(copy.deepcopy(delta))
    finish(parent, started, {})
    assert [r["arguments"]["sample"] for r in json.loads(path.read_text())["tool_responses"]] == [1, 2]


def test_multiple_csv_responses_are_not_silently_reduced_to_the_first():
    from CoScientist.experiments.runtime.inline_artifacts import _csv_payload
    assert _csv_payload({'responses': ['sample,value\na,1\n', 'sample,value\nb,2\n']}) is None


def test_json_output_is_not_published_with_a_csv_extension(running):
    from CoScientist.experiments.runtime.artifacts import normalise_artifacts

    state, started, path = running
    runtime = state["experiment_runtime"]
    task = runtime["tasks"]["EXP-1"]
    task["task"]["expected_artifacts"][0].update(name="measurements.csv", media_type="text/csv")
    path.parent.mkdir(parents=True)
    path.write_text('{"rows": [{"sample": 1, "value": 10}]}')
    artifacts, _ = normalise_artifacts(
        [{"name": path.name, "workspace_path": str(path), "media_type": "application/json"}],
        runtime=runtime, task_runtime=task,
        attempt=task["attempts"][started["attempt_id"]],
    )
    assert len(artifacts) == 1
    assert artifacts[0].name == "overview.json"
    assert artifacts[0].media_type == "application/json"
    assert json.loads(Path(artifacts[0].workspace_path).read_text())["rows"][0]["value"] == 10


def test_coder_publication_preserves_readable_bytes_for_schema_verification(running, monkeypatch):
    from CoScientist.experiments.runtime.coder_artifacts import promote_coder_workspace_artifacts
    from CoScientist.experiments.runtime.artifacts import normalise_artifacts
    from CoScientist.experiments.runtime.outputs import schema_outputs_verified
    from CoScientist.experiments.schemas import ExperimentTask
    from CoScientist.graph.session_scope import GRAPH_SCOPE_SESSION_KEY, GRAPH_SCOPE_USER_KEY

    state, started, path = running
    state.update({GRAPH_SCOPE_SESSION_KEY: 'session-test', GRAPH_SCOPE_USER_KEY: 'user-test',
                  'coder_workspace_id': 'coder-test'})
    sandbox = Path(get_settings().code_exec.workspace_root) / 'coder-test'
    sandbox.mkdir()
    (sandbox / 'overview.json').write_text('{"measurement": 10}')
    calls = []
    def mirror(**kwargs):
        calls.append(kwargs)
        return {'state': 'stored', 'artifact_id': 'stored.json', 'filename': 'overview.json', 'bucket': 'store', 's3_key': 'stored.json'}
    monkeypatch.setattr('CoScientist.reporting.mirror.mirror_artifact', mirror)
    raw = promote_coder_workspace_artifacts(state)
    runtime = state['experiment_runtime']
    task = runtime['tasks']['EXP-1']
    artifacts, _ = normalise_artifacts(raw, runtime=runtime, task_runtime=task,
        attempt=task['attempts'][started['attempt_id']], state=state)
    assert len(calls) == 1
    assert artifacts[0].workspace_path == str(path)
    assert artifacts[0].session_artifact_id == 'stored.json'
    assert schema_outputs_verified(ExperimentTask.model_validate(task['task']), [a.model_dump() for a in artifacts])
