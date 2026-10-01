"""Executor control-tool guards and rewrite."""
from __future__ import annotations

import json
from types import SimpleNamespace

from CoScientist.experiments.runtime import (
    RECORD_REQUIRED_MESSAGE,
    enforce_continue_until_reporting,
    enforce_pending_record_result,
    guard_route_agent_tool,
    mark_route_returned,
    record_result,
    retry_task,
    start_task,
)

from .helpers import (
    _approved_state,
    _plan,
    _route_return,
    _success_result,
    _task,
    _tool_context,
)

def test_guard_coerces_nested_agent_tool_request_to_string():
    state = _approved_state(_plan(_task("EXP-1")))
    start_task(state, "EXP-1")
    args = {
        "request": {
            "task_id": "EXP-1",
            "attempt_id": "ATT-x",
            "launch_params": {"case": "cancer", "num": 10},
        }
    }
    assert (
        guard_route_agent_tool(
            SimpleNamespace(name="FedotAgent"), args, _tool_context(state)
        )
        is None
    )
    assert isinstance(args["request"], str)
    assert '"case": "cancer"' in args["request"] or '"case":"cancer"' in args["request"]


def test_guard_refuses_start_task_until_record_result():
    state = _approved_state(_plan(_task("EXP-1"), _task("EXP-2", depends_on=["EXP-1"])))
    first = start_task(state, "EXP-1")
    _route_return(state, "FedotAgent")

    refused = guard_route_agent_tool(
        SimpleNamespace(name="start_task"),
        {"task_id": "EXP-2"},
        _tool_context(state),
    )
    assert refused["status"] == "refused"
    assert refused["error_code"] == "record_result_required"
    assert refused["must_record_attempt_id"] == first["attempt_id"]
    assert RECORD_REQUIRED_MESSAGE in refused["message"]

    # Closing tools remain allowed.
    assert (
        guard_route_agent_tool(
            SimpleNamespace(name="record_result"),
            {},
            _tool_context(state),
        )
        is None
    )


def test_enforce_pending_record_result_injects_function_call():
    from google.adk.models import LlmResponse
    from google.genai import types

    state = _approved_state(_plan(_task("EXP-1")))
    started = start_task(state, "EXP-1")
    _route_return(state, "FedotAgent")
    state["experiment_last_route_response"] = "Fedot failed: model restriction."

    prose = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text="All done, here is the experiment summary.")],
        )
    )
    # No captured artifacts → force failure + retryable (not false success).
    forced = enforce_pending_record_result(SimpleNamespace(state=state), prose)
    assert forced is not None
    fc = forced.content.parts[0].function_call
    assert fc.name == "record_result"
    assert fc.args["task_id"] == "EXP-1"
    assert fc.args["attempt_id"] == started["attempt_id"]
    assert fc.args["result"]["status"] == "failure"
    assert fc.args["result"]["retryable"] is True
    assert fc.args["result"]["error_code"] == "route_failed_or_empty"

    # Already calling record_result → leave response alone.
    closing = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="record_result",
                    args={"task_id": "EXP-1", "attempt_id": started["attempt_id"], "result": {}},
                )
            ],
        )
    )
    assert enforce_pending_record_result(SimpleNamespace(state=state), closing) is None

    # With captured artifacts → optimistic success (tools may still downgrade).
    state2 = _approved_state(_plan(_task("EXP-1")))
    started2 = start_task(state2, "EXP-1")
    _route_return(state2, "FedotAgent")
    state2["fedot_artifacts"] = [
        {
            "name": "candidates.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-1/candidates.csv",
        }
    ]
    state2["experiment_last_route_response"] = "Fedot finished with managed table."
    forced_ok = enforce_pending_record_result(SimpleNamespace(state=state2), prose)
    assert forced_ok is not None
    assert forced_ok.content.parts[0].function_call.args["result"]["status"] == "success"
    assert forced_ok.content.parts[0].function_call.args["attempt_id"] == started2["attempt_id"]

    # retry_task before record is refused so FORCE_RECORD can close the attempt.
    state3 = _approved_state(_plan(_task("EXP-1")))
    start_task(state3, "EXP-1")
    _route_return(state3, "FedotAgent")
    refused_retry = guard_route_agent_tool(
        SimpleNamespace(name="retry_task"),
        {"task_id": "EXP-1"},
        _tool_context(state3),
    )
    assert refused_retry is not None
    assert refused_retry["error_code"] == "record_result_required"
    retry_llm = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="retry_task",
                    args={"task_id": "EXP-1"},
                )
            ],
        )
    )
    forced_over_retry = enforce_pending_record_result(SimpleNamespace(state=state3), retry_llm)
    assert forced_over_retry is not None
    assert forced_over_retry.content.parts[0].function_call.name == "record_result"
    assert forced_over_retry.content.parts[0].function_call.args["result"]["retryable"] is True


def test_enforce_continue_until_reporting_starts_next_ready_task():
    from google.adk.models import LlmResponse
    from google.genai import types

    plan = _plan(_task("EXP-1"), _task("EXP-2", depends_on=["EXP-1"]))
    state = _approved_state(plan)
    started = start_task(state, "EXP-1")
    _route_return(state, "FedotAgent")
    state["fedot_artifacts"] = [
        {
            "name": "exp-1-result.csv",
            "bucket": "managed-experiments",
            "s3_key": "experiments/EXP-1/result.csv",
        }
    ]
    record_result(state, "EXP-1", started["attempt_id"], _success_result("EXP-1"))
    assert state["experiment_runtime"]["phase"] == "execution"
    assert state["experiment_runtime"]["tasks"]["EXP-2"]["status"] == "ready"

    prose = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text="EXP-1 succeeded; experiment complete.")],
        )
    )
    forced = enforce_continue_until_reporting(SimpleNamespace(state=state), prose)
    assert forced is not None
    fc = forced.content.parts[0].function_call
    assert fc.name == "start_task"
    assert fc.args["task_id"] == "EXP-2"

    # Tool call already present → do not override.
    with_tool = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="get_experiment_plan",
                    args={},
                )
            ],
        )
    )
    assert enforce_continue_until_reporting(SimpleNamespace(state=state), with_tool) is None


def test_rewrite_mismatched_control_action_fixes_retry_when_fallback_pending():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime import rewrite_mismatched_control_action

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
    retry_task(state, "EXP-1")
    retried = start_task(state, "EXP-1")
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

    wrong = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_function_call(name="retry_task", args={"task_id": "EXP-1"})],
        )
    )
    fixed = rewrite_mismatched_control_action(SimpleNamespace(state=state), wrong)
    assert fixed is not None
    assert fixed.content.parts[0].function_call.name == "fallback_task"
    assert fixed.content.parts[0].function_call.args["task_id"] == "EXP-1"
    assert fixed.content.parts[0].function_call.args.get("reason")


def test_rewrite_mismatched_control_action_suppresses_start_while_running():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime import rewrite_mismatched_control_action

    plan = _plan(_task("EXP-1"), _task("EXP-2", depends_on=[]))
    state = _approved_state(plan)
    start_task(state, "EXP-1")
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "running"
    # EXP-2 may be ready concurrently; model must not start it while EXP-1 runs.
    state["experiment_runtime"]["tasks"]["EXP-2"]["status"] = "ready"

    wrong = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_function_call(name="start_task", args={"task_id": "EXP-2"})],
        )
    )
    fixed = rewrite_mismatched_control_action(SimpleNamespace(state=state), wrong)
    assert fixed is not None
    assert fixed.content.parts[0].function_call.name == "FedotAgent"


def test_rewrite_mismatched_control_action_does_not_force_read_outside_execution():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime import rewrite_mismatched_control_action

    state = _approved_state(_plan(_task("EXP-1")))
    state["experiment_runtime"]["phase"] = "reporting"
    state["experiment_runtime"]["tasks"]["EXP-1"]["status"] = "failed"
    wrong = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="fallback_task",
                    args={"task_id": "EXP-1", "reason": "try again"},
                )
            ],
        )
    )
    fixed = rewrite_mismatched_control_action(SimpleNamespace(state=state), wrong)
    # Reporting is terminal for executor transitions.  Synthesizing a
    # get_experiment_plan call here used to create a read/rewrite loop.
    assert fixed is None


def test_guard_does_not_mutate_route_request_payload():
    state = _approved_state(_plan(_task("EXP-1")))
    start_task(state, "EXP-1")
    payload = {
        "case": "cancer",
        "num": 10,
        "upload_results_to_s3": False,
        "output_s3_prefix": "generated",
    }
    args = {"request": json.dumps(payload)}
    assert (
        guard_route_agent_tool(
            SimpleNamespace(name="FedotAgent"), args, _tool_context(state)
        )
        is None
    )
    assert json.loads(args["request"]) == payload
