"""Ambiguous external tool failures must never become automatic replays."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.plugin_manager import PluginManager

from CoScientist.agents.run_control_plugin import (
    RECOVERY_STATE_KEY,
    RunControlPlugin,
    action_key,
    unresolved_actions,
)
from CoScientist.execution_control import RunController, bind_run


def _context(call_id: str = "call-1"):
    return SimpleNamespace(
        state={"experiment_runtime": {"active_task_id": "T1", "active_attempt_id": "A1"}},
        agent_name="Executor",
        function_call_id=call_id,
    )


def test_remote_timeout_is_serializable_and_requires_operator_inspection(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        plugin = RunControlPlugin()
        tool = SimpleNamespace(name="submit_training")
        context = _context()
        with bind_run(handle):
            await plugin.before_tool_callback(
                tool=tool, tool_args={"dataset": "input.csv"}, tool_context=context
            )
            response = await plugin.on_tool_error_callback(
                tool=tool,
                tool_args={"dataset": "input.csv"},
                tool_context=context,
                error=TimeoutError("remote response timed out after submit"),
            )
            # This is the path ADK takes after an error plugin supplies a safe
            # function response.  It must not relabel the action completed.
            after = await plugin.after_tool_callback(
                tool=tool,
                tool_args={"dataset": "input.csv"},
                tool_context=context,
                result=response,
            )
        assert after is None
        assert response["outcome_unknown"] is True
        assert response["retryable"] is False
        json.dumps(response)
        status = handle.status()
        assert status.pause_causes == ("unknown_completion",)
        assert status.pending_decisions["unknown_completion"]["actions"][0]["tool"] \
            == "submit_training"
        unknown = unresolved_actions(controller.journal_entries(handle.run_id))
        assert len(unknown) == 1
        assert unknown[0]["event"] == "tool_outcome_unknown"

    asyncio.run(scenario())


def test_remote_invalid_json_after_dispatch_requires_operator_inspection(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        plugin = RunControlPlugin()
        tool = SimpleNamespace(name="submit_training")
        context = _context()
        with bind_run(handle):
            await plugin.before_tool_callback(
                tool=tool, tool_args={"dataset": "input.csv"}, tool_context=context
            )
            response = await plugin.on_tool_error_callback(
                tool=tool,
                tool_args={"dataset": "input.csv"},
                tool_context=context,
                error=ValueError("invalid JSON in accepted-job response"),
            )
        assert response["outcome_unknown"] is True
        assert handle.status().pause_causes == ("unknown_completion",)
        assert unresolved_actions(controller.journal_entries(handle.run_id))

    asyncio.run(scenario())


def test_local_validation_failure_remains_available_to_bounded_recovery(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        plugin = RunControlPlugin()
        tool = SimpleNamespace(name="record_result")
        context = _context()
        with bind_run(handle):
            await plugin.before_tool_callback(
                tool=tool, tool_args={"task_id": "wrong"}, tool_context=context
            )
            response = await plugin.on_tool_error_callback(
                tool=tool,
                tool_args={"task_id": "wrong"},
                tool_context=context,
                error=ValueError("attempt id does not match active task"),
            )
        assert response is None
        assert handle.status().state == "running"
        assert handle.status().pause_causes == ()
        entries = controller.journal_entries(handle.run_id)
        assert entries[-2]["event"] == "tool_failed"
        assert not unresolved_actions(entries)

    asyncio.run(scenario())


def test_delegation_timeout_uses_child_recovery_instead_of_opaque_pause(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        plugin = RunControlPlugin()
        tool = SimpleNamespace(name="ResearchAgent", agent=object())
        context = _context()
        with bind_run(handle):
            await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=context)
            response = await plugin.on_tool_error_callback(
                tool=tool, tool_args={}, tool_context=context,
                error=TimeoutError("child timed out"),
            )
        assert response is None
        assert handle.status().pause_causes == ()
        assert not unresolved_actions(controller.journal_entries(handle.run_id))

    asyncio.run(scenario())


def test_declared_read_only_network_timeout_is_safe_to_retry(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        plugin = RunControlPlugin()
        tool = SimpleNamespace(name="tavily_search")
        context = _context()
        with bind_run(handle):
            await plugin.before_tool_callback(
                tool=tool, tool_args={"query": "evidence"}, tool_context=context
            )
            # Even a process loss immediately after a declared read does not
            # require operator authorization: repeating a read has no effect.
            assert not unresolved_actions(
                controller.journal_entries(handle.run_id)
            )
            response = await plugin.on_tool_error_callback(
                tool=tool, tool_args={"query": "evidence"}, tool_context=context,
                error=TimeoutError("read timed out"),
            )
        assert response is None
        assert handle.status().pause_causes == ()
        assert not unresolved_actions(controller.journal_entries(handle.run_id))

    asyncio.run(scenario())


def test_registry_read_is_refetched_instead_of_replayed_after_restart(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        plugin = RunControlPlugin()
        tool = SimpleNamespace(name="read_file")
        context = _context()
        context.state[RECOVERY_STATE_KEY] = True
        args = {"file_path": "analysis.md"}
        identity = action_key(tool, args, context)
        controller.journal(
            handle.run_id,
            "tool_completed",
            action_id=identity,
            data={"tool": "read_file", "result": {"content": "old"}},
        )
        with bind_run(handle):
            replay = await plugin.before_tool_callback(
                tool=tool, tool_args=args, tool_context=context
            )
        assert replay is None
        assert controller.journal_entries(handle.run_id)[-1]["event"] == "tool_dispatched"

    asyncio.run(scenario())


def test_human_channel_dispatch_is_not_an_unknown_external_side_effect(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        plugin = RunControlPlugin()
        tool = SimpleNamespace(name="request_approval")
        context = _context()
        with bind_run(handle):
            await plugin.before_tool_callback(
                tool=tool, tool_args={"message": "Proceed?"}, tool_context=context
            )
            assert not unresolved_actions(controller.journal_entries(handle.run_id))
            response = await plugin.on_tool_error_callback(
                tool=tool,
                tool_args={"message": "Proceed?"},
                tool_context=context,
                error=TimeoutError("human response timed out"),
            )
        assert response is None
        assert handle.status().pause_causes == ()
        assert not unresolved_actions(controller.journal_entries(handle.run_id))

    asyncio.run(scenario())


def test_ordinary_json_coercion_does_not_skip_later_adk_plugins(tmp_path):
    observed = []

    class Observer(BasePlugin):
        def __init__(self):
            super().__init__(name="observer")

        async def after_tool_callback(self, **kwargs):
            observed.append(kwargs["result"])
            return None

    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        manager = PluginManager([RunControlPlugin(), Observer()])
        tool = SimpleNamespace(name="local_report")
        context = _context()
        result = {"created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}
        with bind_run(handle):
            await RunControlPlugin().before_tool_callback(
                tool=tool, tool_args={}, tool_context=context
            )
            altered = await manager.run_after_tool_callback(
                tool=tool, tool_args={}, tool_context=context, result=result
            )
        assert altered is None
        assert observed == [result]
        json.dumps(controller.journal_entries(handle.run_id))

    asyncio.run(scenario())


def test_exception_leaf_is_replaced_by_serializable_diagnostic(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        plugin = RunControlPlugin()
        tool = SimpleNamespace(name="validate_data")
        context = _context()
        result = {"errors": [{"ctx": {"error": ValueError("bad column")}}]}
        with bind_run(handle):
            await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=context)
            encoded = await plugin.after_tool_callback(
                tool=tool, tool_args={}, tool_context=context, result=result
            )
        assert encoded["errors"][0]["ctx"]["error"] == {
            "error_type": "ValueError", "message": "bad column"
        }
        json.dumps(encoded)
        assert handle.status().pause_causes == ()

    asyncio.run(scenario())


def test_pending_human_resolution_is_applied_before_continuation_snapshot(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        controller.update_metadata("run", {
            "control_resolution": {
                "decision": "finish_with_limits", "source": "human",
            },
        })
        state = {
            "experiment_runtime": {
                "phase": "execution",
                "active_task_id": "T1",
                "active_attempt_id": None,
                "task_order": ["T1"],
                "tasks": {
                    "T1": {
                        "status": "running",
                        "task": {"optional": False},
                        "attempts": {},
                    },
                },
            },
        }
        context = SimpleNamespace(
            state=state, agent_name="Executor", function_call_id="call-resolution"
        )
        plugin = RunControlPlugin()
        with bind_run(handle):
            response = await plugin.before_tool_callback(
                tool=SimpleNamespace(name="get_experiment_plan"),
                tool_args={},
                tool_context=context,
            )
        assert state["experiment_runtime"]["phase"] == "reporting"
        assert response["resolution"] == "finish_with_limits"
        saved = handle.status().metadata["continuation"]["state"]
        assert saved["experiment_runtime"]["phase"] == "reporting"
        assert handle.status().metadata["control_resolution"] is None
        assert not any(
            row["event"] == "tool_dispatched"
            for row in controller.journal_entries(handle.run_id)
        )

    asyncio.run(scenario())


def test_pending_resolution_bypasses_stale_model_request_without_spending_budget(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        controller.update_metadata("run", {
            "control_resolution": {
                "decision": "finish_with_limits", "source": "human",
                "reason": "control_no_progress",
            },
        })
        state = {
            "experiment_runtime": {
                "phase": "execution",
                "active_task_id": "T1",
                "active_attempt_id": None,
                "task_order": ["T1"],
                "tasks": {
                    "T1": {
                        "status": "running",
                        "task": {"optional": False},
                        "attempts": {},
                    },
                },
            },
        }
        context = SimpleNamespace(state=state, agent_name="ExperimentExecutorAgent")
        plugin = RunControlPlugin()
        with bind_run(handle):
            response = await plugin.before_model_callback(
                callback_context=context, llm_request=object()
            )
        text = response.content.parts[0].text
        assert text.startswith(
            "Operator closed unfinished tasks with documented limitations; "
            "proceed to result review."
        )
        assert "Limited tasks: T1." in text
        assert response.custom_metadata == {
            "control_resolution": "finish_with_limits",
            "summary": "Automatic control stopped after repeated no-progress transitions.",
            "limited_task_ids": ["T1"],
        }
        assert "status" not in response.custom_metadata
        assert state["experiment_runtime"]["phase"] == "reporting"
        assert handle.status().attempts_used == 0
        saved = handle.status().metadata["continuation"]["state"]
        assert saved["experiment_runtime"]["phase"] == "reporting"

        # The durable instruction is consumed once. A later, freshly-built
        # result-review request is allowed through normally.
        with bind_run(handle):
            assert await plugin.before_model_callback(
                callback_context=context, llm_request=object()
            ) is None

    asyncio.run(scenario())
