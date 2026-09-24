"""Structured MCP tool results become the artifacts the planner named (blind Informer run 12)."""
from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.runtime.artifacts import captured_delta
from CoScientist.experiments.runtime.guards import capture_experiment_tool_results, on_route_agent_returned
from CoScientist.experiments.runtime.inline_artifacts import materialize_tool_results, record_tool_result
from CoScientist.experiments.runtime.state_machine import (
    approve_plan, initialize_runtime, record_result, start_task,
)

from .helpers import _plan, _task, _tool_context


def _mcp(text: dict) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(text)}]}


def _train(pred_len: int, mse: float, mae: float) -> dict:
    return _mcp({"setting": f"informer_ETTh1_pl{pred_len}", "metrics": {"mae": mae, "mse": mse}, "epochs": 3})


def test_a_train_result_is_recorded_and_written_as_the_planned_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "CoScientist.experiments.runtime.inline_artifacts.get_settings",
        lambda: SimpleNamespace(code_exec=SimpleNamespace(workspace_root=str(tmp_path)),
                                experiments=SimpleNamespace(max_inline_bytes=10_000_000)),
    )
    state: dict = {}
    assert record_tool_result(state, attempt_id="ATT-1", tool="train_informer",
                              args={"pred_len": 24}, response=_train(24, 0.699, 0.637))
    assert record_tool_result(state, attempt_id="ATT-1", tool="train_informer",
                              args={"pred_len": 48}, response=_train(48, 0.715, 0.650))
    # a plain status string is not a result
    assert not record_tool_result(state, attempt_id="ATT-1", tool="sleep", args={}, response="slept 5s")
    created = materialize_tool_results(
        state, task_id="EXP-1", attempt_id="ATT-1",
        expected_artifacts=[
            {"name": "informer_metrics.csv", "role": "data", "media_type": "text/csv"},
            {"name": "informer2020-mcp-server", "role": "mcp_server"},
            {"name": "run_log.json", "role": "log"},
        ],
    )
    assert [c["name"] for c in created] == ["informer_metrics.csv", "run_log.json"]
    rows = list(csv.DictReader(open(created[0]["workspace_path"], encoding="utf-8")))
    assert [r["pred_len"] for r in rows] == ["24", "48"]
    assert rows[0]["metrics.mse"] == "0.699" and rows[1]["metrics.mae"] == "0.65"
    assert [a["name"] for a in state["fedot_artifacts"]] == ["informer_metrics.csv", "run_log.json"]
    # a second call writes nothing new
    assert materialize_tool_results(state, task_id="EXP-1", attempt_id="ATT-1",
                                    expected_artifacts=[{"name": "informer_metrics.csv", "role": "data"}],
                                    existing_names={"informer_metrics.csv"}) == []


def test_the_route_return_materializes_and_tells_the_executor(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "CoScientist.experiments.runtime.inline_artifacts.get_settings",
        lambda: SimpleNamespace(code_exec=SimpleNamespace(workspace_root=str(tmp_path)),
                                experiments=SimpleNamespace(max_inline_bytes=10_000_000)),
    )
    cfg = ExperimentsSettings(route_fedot=True)
    task = _task("EXP-1", route="react_tools", tool="train_informer", artifact_name="informer_metrics.csv")
    state: dict = {}
    initialize_runtime(state, _plan(task), critique={"verdict": "approve", "issues": [], "summary": "forced"})
    approve_plan(state)
    start_task(state, "EXP-1", settings=cfg)
    ctx = _tool_context(state)
    capture_experiment_tool_results(SimpleNamespace(name="train_informer"), {"pred_len": 24}, ctx, _train(24, 0.699, 0.637))
    capture_experiment_tool_results(SimpleNamespace(name="ExperimentAgent"), {}, ctx, {"result": "ignored"})
    replaced = on_route_agent_returned(SimpleNamespace(name="ExperimentAgent"), {}, ctx, {"result": "Trained 24."})
    assert replaced is not None and "informer_metrics.csv" in replaced["result"]
    runtime = state["experiment_runtime"]
    attempt_id = runtime["tasks"]["EXP-1"]["attempt_order"][-1]
    attempt = runtime["tasks"]["EXP-1"]["attempts"][attempt_id]
    names = [a["name"] for a in captured_delta(state, attempt)]
    assert names == ["informer_metrics.csv"]
    stored = record_result(state, "EXP-1", attempt_id, {
        "status": "success", "summary": "trained",
        "criteria_checks": [{"criterion_id": "EXP-1-C1", "passed": True, "details": "ok"}],
    }, settings=cfg)
    assert stored["status"] == "success"
    assert [a["name"] for a in stored["task_result"]["artifacts"]] == ["informer_metrics.csv"]
    assert runtime["tasks"]["EXP-1"]["status"] == "done"
