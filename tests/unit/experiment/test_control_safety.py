"""Finite experiment-control selection and MCP result contracts."""
from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

from CoScientist.experiments.capabilities.inventory import input_schema_mismatches
from CoScientist.experiments.runtime.guards import (
    assess_experiment_inventory_feasibility,
    force_schema_s3_upload,
    on_route_agent_returned,
    select_experiment_action,
    skip_when_experiment_stage_complete,
)



def _selector_state() -> dict:
    return {"experiment_runtime": {
        "phase": "execution",
        "approved": True,
        "task_order": ["EXP-1"],
        "active_task_id": None,
        "active_attempt_id": None,
        "results": [],
        "tasks": {"EXP-1": {
            "status": "ready",
            "current_route": "coder",
            "planned_route": "coder",
            "attempt_order": [],
            "attempts": {},
            "task": {"id": "EXP-1", "success_criteria": []},
        }},
    }}


def test_normalize_supported_mcp_shapes_without_inventing_plain_text_success():
    from CoScientist.agents.callbacks.tool_callbacks import normalize_tool_observation

    wrapped = normalize_tool_observation({
        "structuredContent": {"rows": [1, 2]},
        "content": [{"type": "text", "text": '{"rows":[1,2]}'}],
        "isError": False,
        "status": "done",
    })
    assert wrapped["status"] == "success"
    assert wrapped["data"] == {"rows": [1, 2]}

    pending = normalize_tool_observation({"status": "queued", "job_id": "job-1"})
    assert pending["pending"] is True and pending["job_id"] == "job-1"

    miss = normalize_tool_observation({
        "isError": True,
        "content": [{"text": "NO_MATCHING_TOOL: docking is unavailable"}],
    })
    assert miss["status"] == "failure"
    assert miss["error_code"] == "no_matching_tool"

    prose = normalize_tool_observation("finished, probably")
    assert prose["status"] == "unknown"


def test_known_input_schema_is_checked_but_unknown_schema_is_permitted():
    schema = {
        "type": "object",
        "required": ["smiles"],
        "properties": {"smiles": {"type": "string"}, "count": {"type": "integer"}},
        "additionalProperties": False,
    }
    mismatch = input_schema_mismatches(schema, {"count": "ten", "extra": 1})
    assert {item["code"] for item in mismatch} == {
        "required_argument_missing", "argument_type_mismatch", "additional_argument_forbidden",
    }
    assert input_schema_mismatches({}, {"anything": object()}) == []
    assert input_schema_mismatches(None, {}) == []


def test_before_call_schema_guard_returns_actionable_refusal_and_injects_upload():
    state = {
        "experiment_runtime": {"phase": "execution"},
        "filtered_tools": [{
            "tool": "dock",
            "input_schema": {
                "type": "object",
                "required": ["smiles"],
                "properties": {
                    "smiles": {"type": "string"},
                    "upload_results_to_s3": {"type": "boolean"},
                    "output_s3_prefix": {"type": "string"},
                },
                "additionalProperties": False,
            },
        }],
    }
    args = {}
    refused = force_schema_s3_upload(
        SimpleNamespace(name="dock"), args, SimpleNamespace(state=state),
    )
    assert refused["error_code"] == "tool_input_schema_mismatch"
    assert refused["next_actions"][0]["required_arguments"] == ["smiles"]
    assert args["upload_results_to_s3"] is True


def test_selector_stops_for_pause_and_for_closed_active_attempt():
    state = _selector_state()
    assert select_experiment_action(state) == ("start_task", {"task_id": "EXP-1"})

    state["experiment_manual_pause"] = True
    assert select_experiment_action(state) is None
    state["experiment_manual_pause"] = False

    runtime = state["experiment_runtime"]
    runtime["active_task_id"] = "EXP-1"
    runtime["active_attempt_id"] = "ATT-closed"
    runtime["tasks"]["EXP-1"]["status"] = "running"
    runtime["tasks"]["EXP-1"]["attempts"]["ATT-closed"] = {
        "attempt_id": "ATT-closed",
        "status": "failure",
        "result_id": "RES-1",
        "route": "coder",
    }
    assert select_experiment_action(state) is None


def test_selector_does_not_repeat_a_refused_unchanged_fallback():
    state = _selector_state()
    runtime = state["experiment_runtime"]
    runtime["tasks"]["EXP-1"]["status"] = "fallback_pending"
    selected = select_experiment_action(state)
    assert selected and selected[0] == "fallback_task"
    assert selected[1]["reason"]

    on_route_agent_returned(
        SimpleNamespace(name="fallback_task"), selected[1], SimpleNamespace(state=state),
        {"status": "error", "error_code": "fallback_exhausted", "message": "none left"},
    )
    assert select_experiment_action(state) is None


def test_discovery_rounds_are_request_scoped_and_finite(monkeypatch):
    import CoScientist.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_settings",
        lambda: SimpleNamespace(experiments=SimpleNamespace(
            max_recovery_discovery_rounds=2,
        )),
    )
    state = {
        "orchestrator_root_goal": "fit one classifier",
        "experiment_runtime": {"phase": "execution"},
    }
    context = SimpleNamespace(state=state, agent_name="ToolPreparerAgent")

    assess_experiment_inventory_feasibility(context)
    assert skip_when_experiment_stage_complete(context) is None
    assess_experiment_inventory_feasibility(context)
    stopped = skip_when_experiment_stage_complete(context)

    assert stopped is not None
    assert "Capability discovery limit reached (2/2)" in stopped.parts[0].text
    assert sum(state["experiment_discovery_rounds_by_request"].values()) == 2


def test_targeted_redo_skips_discovery_and_planning_but_not_executor():
    state = {
        "experiment_runtime": {"phase": "execution"},
        "experiment_targeted_redo_pending": {"selected_task_ids": ["EXP-2"]},
    }
    for agent_name in ("ToolPreparerAgent", "ExperimentPlannerAgent"):
        stopped = skip_when_experiment_stage_complete(SimpleNamespace(
            state=state, agent_name=agent_name,
        ))
        assert stopped is not None
        assert "Targeted result-review redo" in stopped.parts[0].text

    assert skip_when_experiment_stage_complete(SimpleNamespace(
        state=state, agent_name="ExperimentExecutorAgent",
    )) is None


def _before(guard, name, args, state, call_id):
    return asyncio.run(guard.before_tool_callback(
        tool=SimpleNamespace(name=name), tool_args=args,
        tool_context=SimpleNamespace(
            agent_name="ExperimentExecutorAgent", state=state, function_call_id=call_id,
        ),
    ))


def _after(guard, name, args, state, call_id, result):
    return asyncio.run(guard.after_tool_callback(
        tool=SimpleNamespace(name=name), tool_args=args,
        tool_context=SimpleNamespace(
            agent_name="ExperimentExecutorAgent", state=state, function_call_id=call_id,
        ),
        result=result,
    ))


def test_semantic_guard_ignores_reason_and_returns_machine_recovery():
    from CoScientist.agents.loop_guard_plugin import RepeatCallGuardPlugin

    guard = RepeatCallGuardPlugin()
    state = {"experiment_runtime": {
        "phase": "execution", "task_order": ["EXP-1"],
        "active_task_id": None, "active_attempt_id": None,
        "tasks": {"EXP-1": {
            "status": "fallback_pending", "current_route": "react_tools", "attempt_order": [],
        }},
    }}
    for index in range(3):
        args = {"task_id": "EXP-1", "reason": f"wording {index}"}
        assert _before(guard, "fallback_task", args, state, f"call-{index}") is None
        assert _after(
            guard, "fallback_task", args, state, f"call-{index}",
            {"status": "error", "error_code": "fallback_exhausted", "message": "none"},
        ) is None

    blocked = _before(
        guard, "fallback_task", {"task_id": "EXP-1", "reason": "new wording"},
        state, "call-4",
    )
    assert blocked["error_code"] == "same_control_failure_limit"
    assert blocked["terminal_control_guard"] is True
    assert blocked["next_actions"][0]["action"] == "stop_automatic_retries"

    # Read-only observation remains available after the guard trips.
    for index in range(10):
        assert _before(guard, "get_experiment_plan", {}, state, f"read-{index}") is None


def test_five_read_only_no_progress_turns_stop_the_next_mutation_not_reading():
    from CoScientist.agents.loop_guard_plugin import RepeatCallGuardPlugin

    guard = RepeatCallGuardPlugin()
    state = {"experiment_runtime": {
        "phase": "execution", "task_order": [], "tasks": {},
        "active_task_id": None, "active_attempt_id": None,
    }}
    for index in range(5):
        call_id = f"read-{index}"
        assert _before(guard, "get_experiment_plan", {}, state, call_id) is None
        assert _after(
            guard, "get_experiment_plan", {}, state, call_id,
            {"status": "success", "phase": "execution"},
        ) is None
    assert _before(guard, "get_experiment_plan", {}, state, "read-more") is None
    blocked = _before(guard, "start_task", {"task_id": "EXP-1"}, state, "mutate")
    assert blocked["error_code"] == "control_no_progress_limit"


def test_semantic_guard_requests_a_real_resumable_run_pause(tmp_path):
    from CoScientist.agents.loop_guard_plugin import RepeatCallGuardPlugin
    from CoScientist.execution_control import RunController, bind_run

    guard = RepeatCallGuardPlugin()
    controller = RunController(tmp_path / "loop-control.sqlite3", poll_interval=0.01)
    handle = controller.create_run("loop-run")
    state = {"experiment_runtime": {
        "phase": "execution", "task_order": ["EXP-1"],
        "active_task_id": None, "active_attempt_id": None,
        "tasks": {"EXP-1": {
            "status": "fallback_pending", "current_route": "react_tools",
            "attempt_order": [],
        }},
    }}

    with bind_run(handle):
        for index in range(3):
            args = {"task_id": "EXP-1", "reason": f"cosmetic {index}"}
            assert _before(guard, "fallback_task", args, state, f"pause-{index}") is None
            assert _after(
                guard, "fallback_task", args, state, f"pause-{index}",
                {"status": "error", "error_code": "fallback_exhausted"},
            ) is None

    status = controller.status("loop-run")
    assert status.state == "pause_requested"
    assert "execution_error" in status.pause_causes
    decision = status.pending_decisions["execution_error"]
    assert decision["kind"] == "semantic_control_loop_guard"
    assert decision["actions"][1]["action"] == "accept_partial_result_and_report"
    assert state["experiment_control_guard_pause"]["state_revision"]


def test_explicit_stagnation_resolution_preserves_success_and_reports_limits(
    tmp_path, monkeypatch,
):
    from CoScientist.agents.loop_guard_plugin import apply_pending_control_resolution
    from CoScientist.execution_control import RunController, bind_run

    prior_result = {"result_id": "RES-good", "task_id": "EXP-good", "status": "success"}
    state = {"experiment_runtime": {
        "phase": "execution",
        "task_order": ["EXP-good", "EXP-running", "EXP-optional"],
        "active_task_id": "EXP-running",
        "active_attempt_id": "ATT-stuck",
        "results": [prior_result],
        "tasks": {
            "EXP-good": {"status": "done", "attempt_order": [], "attempts": {}},
            "EXP-running": {
                "status": "running", "task": {"optional": False},
                "attempt_order": ["ATT-stuck"],
                "attempts": {"ATT-stuck": {"status": "running", "route": "coder"}},
            },
            "EXP-optional": {
                "status": "ready", "task": {"optional": True},
                "attempt_order": [], "attempts": {},
            },
        },
    }}
    context = SimpleNamespace(state=state, agent_name="ExperimentExecutorAgent")

    # No bound, explicit operator decision means no mutation.
    assert apply_pending_control_resolution(context) is None
    assert state["experiment_runtime"]["phase"] == "execution"

    controller = RunController(tmp_path / "resolve-control.sqlite3")
    handle = controller.create_run("resolve-run")
    controller.update_metadata("resolve-run", {"control_resolution": {
        "decision": "finish_with_limits",
        "source": "human",
        "reason": "same_control_failure_limit",
    }})
    snapshots = []

    def _capture_resolution_snapshot(callback_context, *, boundary, agent=""):
        snapshots.append({
            "phase": callback_context.state["experiment_runtime"]["phase"],
            "decision": controller.status("resolve-run").metadata["control_resolution"],
            "boundary": boundary,
        })

    import CoScientist.agents.run_control_plugin as run_control_plugin
    monkeypatch.setattr(run_control_plugin, "save_continuation", _capture_resolution_snapshot)
    with bind_run(handle):
        applied = apply_pending_control_resolution(context)

    runtime = state["experiment_runtime"]
    assert applied["status"] == "success"
    assert runtime["phase"] == "reporting"
    assert runtime["tasks"]["EXP-good"]["status"] == "done"
    assert runtime["tasks"]["EXP-running"]["status"] == "failed"
    assert runtime["tasks"]["EXP-optional"]["status"] == "skipped"
    assert runtime["results"] == [prior_result]  # never fabricates success/result evidence
    stuck = runtime["tasks"]["EXP-running"]["attempts"]["ATT-stuck"]
    assert stuck["status"] == "failure" and stuck["retryable"] is False
    assert runtime["active_task_id"] is None and runtime["active_attempt_id"] is None
    status = controller.status("resolve-run")
    assert status.metadata["control_resolution"] is None
    assert status.metadata["control_resolution_done"]["resolution"] == "finish_with_limits"
    assert controller.journal_entries("resolve-run")[-2]["event"] == "control_resolution_applied"
    assert snapshots == [{
        "phase": "reporting",
        "decision": {
            "decision": "finish_with_limits",
            "source": "human",
            "reason": "same_control_failure_limit",
        },
        "boundary": "control_resolution_applied",
    }]


def test_finish_stagnation_publishes_runtime_delta_for_adk_like_state():
    from CoScientist.agents.loop_guard_plugin import finish_control_stagnation

    class DeltaOnlyState:
        """Nested values are detached; only __setitem__ publishes a delta."""

        def __init__(self, values):
            self._values = copy.deepcopy(values)

        def get(self, key, default=None):
            return copy.deepcopy(self._values.get(key, default))

        def __setitem__(self, key, value):
            self._values[key] = copy.deepcopy(value)

        def to_dict(self):
            return copy.deepcopy(self._values)

    state = DeltaOnlyState({"experiment_runtime": {
        "phase": "execution",
        "task_order": ["EXP-1"],
        "active_task_id": "EXP-1",
        "active_attempt_id": "ATT-1",
        "results": [],
        "tasks": {"EXP-1": {
            "status": "running",
            "task": {"optional": False},
            "attempt_order": ["ATT-1"],
            "attempts": {"ATT-1": {"status": "running"}},
        }},
    }})

    resolved = finish_control_stagnation(state)
    persisted = state.to_dict()["experiment_runtime"]

    assert resolved["status"] == "success"
    assert persisted["phase"] == "reporting"
    assert persisted["tasks"]["EXP-1"]["status"] == "failed"
    assert persisted["tasks"]["EXP-1"]["attempts"]["ATT-1"]["status"] == "failure"
    assert persisted["active_task_id"] is None
