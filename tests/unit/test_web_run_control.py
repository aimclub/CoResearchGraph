from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from CoScientist.execution_control import bind_run, before_model_attempt, before_tool_action
from CoScientist.web.app import WebRuntime, create_app
from CoScientist.agents.run_control_plugin import RunControlPlugin, unresolved_actions

web_app = importlib.import_module("CoScientist.web.app")


async def settle(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


def test_manual_pause_resumes_same_coroutine_and_keeps_budget(monkeypatch):
    async def scenario():
        runtime = WebRuntime()
        key = ("user", "session")
        ready, next_step, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []

        async def fake_chat(_runtime, _key, _data):
            await before_model_attempt("first")
            calls.append("first")
            ready.set()
            await next_step.wait()
            await before_tool_action("compute")
            calls.append("tool")
            await before_model_attempt("second")
            calls.append("second")
            finished.set()

        monkeypatch.setattr(web_app, "_handle_chat", fake_chat)
        assert await runtime.start_run(key, {"message": "research"})
        await ready.wait()
        owner = runtime.active_runs[key]
        handle = runtime.execution_handle(key)
        original_start = runtime.run_times[key]["started_at"]
        await runtime.pause_execution(key, handle.run_id)
        next_step.set()
        await settle(lambda: handle.status().state == "paused")
        assert calls == ["first"]
        assert handle.status().attempts_used == 1
        await runtime.resume_execution(key, handle.run_id)
        assert runtime.active_runs[key] is owner
        assert runtime.run_times[key]["started_at"] == original_start
        assert runtime.execution_snapshot(key)["started_at"] == original_start
        await finished.wait()
        await owner
        assert calls == ["first", "tool", "second"]
        assert handle.status().attempts_used == 2
        await runtime.close()

    asyncio.run(scenario())


def test_budget_decision_is_explicit_and_idempotent(monkeypatch):
    async def scenario():
        runtime = WebRuntime()
        key = ("user", "session")
        settings = SimpleNamespace(orchestrator=SimpleNamespace(max_llm_calls=1))
        monkeypatch.setattr(runtime, "settings_snapshot", lambda _key: (0, settings))
        calls = []

        async def fake_chat(*_args):
            await before_model_attempt("first")
            calls.append(1)
            await before_model_attempt("second")
            calls.append(2)

        monkeypatch.setattr(web_app, "_handle_chat", fake_chat)
        assert await runtime.start_run(key, {"message": "research"})
        owner = runtime.active_runs[key]
        handle = runtime.execution_handle(key)
        await settle(lambda: "budget_exhausted" in handle.status().pause_causes)
        assert calls == [1]
        with pytest.raises(ValueError, match="pending decision"):
            await runtime.resume_execution(key, handle.run_id)
        payload = {"decision": "continue", "decision_id": "approval-one"}
        await runtime.decide_execution_budget(key, handle.run_id, payload)
        await runtime.decide_execution_budget(key, handle.run_id, payload)
        with pytest.raises(ValueError, match="another decision"):
            await runtime.decide_execution_budget(key, handle.run_id,
                {"decision": "stop", "decision_id": "approval-one"})
        await owner
        assert calls == [1, 2]
        assert handle.status().budget_total == 101
        assert handle.status().attempts_used == 2
        await runtime.close()

    asyncio.run(scenario())


def test_restart_exposes_pause_and_restores_without_resetting_attempt(monkeypatch):
    async def scenario():
        first = WebRuntime()
        key = ("user", "session")
        settings = SimpleNamespace(orchestrator=SimpleNamespace(max_llm_calls=100))
        handle = first.prepare_execution(key, {"message": "original request"}, settings)
        state = {"experiment_runtime": {"phase": "execution", "active_task_id": "EXP-4",
                                         "active_attempt_id": "ATT-4"}}
        handle.controller.update_metadata(handle.run_id, {"continuation": {"state": state}, "stage_index": 2})
        with bind_run(handle):
            await before_model_attempt("spent")
        second = WebRuntime()
        snapshot = second.execution_snapshot(key)
        assert "server_restart" in snapshot["pause_causes"]
        assert snapshot["budget"]["used"] == 1
        restored = {}

        class Manager:
            async def _set_state(self, name, value):
                restored[name] = value

        async def manager(*_args):
            return Manager()

        resumed = []
        async def start(_key, data):
            resumed.append(data)
            return True

        monkeypatch.setattr(second, "get_manager", manager)
        monkeypatch.setattr(second, "start_run", start)
        await second.resume_execution(key, handle.run_id)
        assert restored["experiment_runtime"]["active_attempt_id"] == "ATT-4"
        assert restored["_checkpoint_resume"]["stage_index"] == 2
        assert restored["_execution_recovery"] is True
        assert resumed[0]["_execution_run_id"] == handle.run_id
        assert resumed[0]["_execution_resume"] is True
        assert handle.status().attempts_used == 1

    asyncio.run(scenario())


def test_unknown_external_dispatch_is_not_automatically_replayed(monkeypatch):
    async def scenario():
        runtime = WebRuntime()
        key = ("user", "session")
        handle = runtime.prepare_execution(key, {"message": "compute"},
            SimpleNamespace(orchestrator=SimpleNamespace(max_llm_calls=100)))
        handle.controller.request_pause(handle.run_id, "manual")
        handle.controller.update_metadata(handle.run_id, {"continuation": {"state": {}}})
        handle.controller.journal(handle.run_id, "tool_dispatched", action_id="remote-1",
                                  data={"tool": "train_model", "delegation": False})
        with pytest.raises(ValueError, match="unknown outcome"):
            await runtime.resume_execution(key, handle.run_id)
        assert "unknown_completion" in handle.status().pause_causes
        assert key not in runtime.active_runs

    asyncio.run(scenario())


def test_completed_tool_is_replayed_from_journal_without_dispatch():
    async def scenario():
        runtime = WebRuntime()
        handle = runtime.execution_controller.create_run()
        plugin = RunControlPlugin()
        tool = SimpleNamespace(name="train_model")
        context = SimpleNamespace(state={"experiment_runtime": {"active_task_id": "EXP-1", "active_attempt_id": "ATT-1"}},
                                  agent_name="CoderAgent", function_call_id="call-1")
        with bind_run(handle):
            assert await plugin.before_tool_callback(tool=tool, tool_args={"epochs": 2}, tool_context=context) is None
            await plugin.after_tool_callback(tool=tool, tool_args={"epochs": 2}, tool_context=context,
                                             result={"model_file": "result.bin"})
            context.state["_execution_recovery"] = True
            context.function_call_id = "call-after-restart"
            result = await plugin.before_tool_callback(tool=tool, tool_args={"epochs": 2}, tool_context=context)
        assert result == {"model_file": "result.bin"}
        entries = handle.controller.journal_entries(handle.run_id)
        assert len([row for row in entries if row["event"] == "tool_dispatched"]) == 1
        assert not unresolved_actions(entries)

    asyncio.run(scenario())


def test_control_api_requires_session_ownership_and_does_not_leak_state():
    app = create_app()
    with TestClient(app) as client:
        user = client.post("/api/users", json={"nickname": "Control tester"}).json()["user"]
        session = client.post(f"/api/users/{user['id']}/sessions", json={"title": "Budget"}).json()["session"]
        key = (user["id"], session["id"])
        runtime = app.state.runtime
        handle = runtime.prepare_execution(key, {"message": "private research"},
            SimpleNamespace(orchestrator=SimpleNamespace(max_llm_calls=100)))
        handle.controller.update_metadata(handle.run_id, {"continuation": {"state": {"private": "do not expose"}}})
        prefix = f"/api/users/{key[0]}/sessions/{key[1]}"
        result = client.get(prefix + "/run-control")
        assert result.status_code == 200
        assert "do not expose" not in result.text
        paused = client.post(prefix + f"/runs/{handle.run_id}/pause", json={})
        assert paused.status_code == 200
        assert paused.json()["run"]["pause_causes"] == ["manual"]
        assert client.get(prefix + "/runs/someone-else").status_code == 404
        assert client.post(prefix + f"/runs/{handle.run_id}/budget/decision",
                           json={"decision": "continue", "decision_id": "unsolicited"}).status_code == 409


def test_legacy_budget_is_clamped_and_nonpositive_budget_rejected():
    from CoScientist.config.settings import OrchestratorSettings
    assert OrchestratorSettings(max_llm_calls=3000).max_llm_calls == 100
    assert OrchestratorSettings(max_llm_calls=12).max_llm_calls == 12
    with pytest.raises(ValueError):
        OrchestratorSettings(max_llm_calls=0)


def test_exception_leaves_in_tool_responses_are_safe_diagnostics():
    async def scenario():
        runtime = WebRuntime()
        handle = runtime.execution_controller.create_run()
        plugin = RunControlPlugin()
        tool = SimpleNamespace(name="validate_data")
        context = SimpleNamespace(state={}, agent_name="Executor", function_call_id="v1")
        with bind_run(handle):
            await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=context)
            response = await plugin.after_tool_callback(tool=tool, tool_args={}, tool_context=context,
                result={"status": "error", "errors": [{"ctx": {"error": ValueError("invalid column")}}]})
        assert response["errors"][0]["ctx"]["error"] == {
            "error_type": "ValueError", "message": "invalid column"}
        assert not unresolved_actions(handle.controller.journal_entries(handle.run_id))
    asyncio.run(scenario())


def test_unknown_tool_result_pauses_and_requires_inspected_retry():
    async def scenario():
        runtime = WebRuntime()
        key = ("user", "session")
        handle = runtime.prepare_execution(key, {"message": "research"},
            SimpleNamespace(orchestrator=SimpleNamespace(max_llm_calls=100)))
        plugin = RunControlPlugin()
        tool = SimpleNamespace(name="train_model")
        context = SimpleNamespace(state={}, agent_name="Executor", function_call_id="train1")
        with bind_run(handle):
            await before_model_attempt("spent")
            await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=context)
            response = await plugin.after_tool_callback(tool=tool, tool_args={}, tool_context=context,
                                                        result={"unsupported": object()})
        assert response["retryable"] is False
        assert "unknown_completion" in handle.status().pause_causes
        action_id = response["action_id"]
        with pytest.raises(ValueError, match="confirmation"):
            await runtime.authorize_unknown_retry(key, handle.run_id, {"action_id": action_id})
        payload = {"action_id": action_id, "notes": "Remote job was never accepted", "confirmed": True}
        await runtime.authorize_unknown_retry(key, handle.run_id, payload)
        await runtime.authorize_unknown_retry(key, handle.run_id, payload)
        assert not unresolved_actions(handle.controller.journal_entries(handle.run_id))
        assert handle.status().pause_causes == ("manual",)
        assert handle.status().attempts_used == 1
        assert handle.status().budget_total == 100
    asyncio.run(scenario())


def test_operator_can_finish_stagnation_without_resetting_budget_or_repeating_success(monkeypatch):
    async def scenario():
        runtime = WebRuntime()
        key = ("user", "session")
        ready = asyncio.Event()
        next_boundary = asyncio.Event()
        state = {"experiment_runtime": {"phase": "execution", "task_order": ["A", "B"],
            "tasks": {"A": {"status": "done", "task": {}}, "B": {"status": "pending", "task": {}}},
            "results": [{"task_id": "A", "summary": "Real previously obtained result"}]}}
        async def fake_chat(*args):
            await before_model_attempt("one")
            ready.set()
            await next_boundary.wait()
            response = await RunControlPlugin().before_model_callback(
                callback_context=SimpleNamespace(state=state, agent_name="ExperimentExecutorAgent"), llm_request=None)
            assert response is not None

        monkeypatch.setattr(web_app, "_handle_chat", fake_chat)
        await runtime.start_run(key, {"message": "research"})
        await ready.wait()
        handle = runtime.execution_handle(key)
        handle.controller.request_pause(handle.run_id, "execution_error", pending_decision={
            "kind": "semantic_control_loop_guard", "error_code": "same_control_failure_limit"})
        handle.controller.request_pause(handle.run_id, "manual")
        next_boundary.set()
        await runtime.finish_stalled_execution(key, handle.run_id, {"confirmed": True})
        assert "manual" in handle.status().pause_causes
        assert state["experiment_runtime"]["phase"] == "execution"
        await runtime.resume_execution(key, handle.run_id)
        owner = runtime.active_runs.get(key)
        if owner:
            await owner
        assert state["experiment_runtime"]["phase"] == "reporting"
        assert state["experiment_runtime"]["tasks"]["A"]["status"] == "done"
        assert state["experiment_runtime"]["tasks"]["B"]["status"] == "failed"
        assert len(state["experiment_runtime"]["results"]) == 1
        assert handle.status().attempts_used == 1
        assert handle.status().budget_total == 100
        assert handle.status().metadata.get("control_resolution") is None
        await runtime.close()
    asyncio.run(scenario())


def test_controlled_hitl_timeout_keeps_same_request_answerable_and_replays_decision(monkeypatch):
    from CoScientist.hitl.models import HITLRequest, HITLAction
    from CoScientist.web.handler import WebHITLHandler

    async def scenario():
        runtime = WebRuntime()
        handle = runtime.execution_controller.create_run()
        handler = WebHITLHandler()
        key = ("user", "session")
        messages = []
        async def send_json(payload):
            messages.append(payload)
        async def no_document(*args):
            return None
        monkeypatch.setattr(handler, "_attach_document", no_document)
        await handler.attach_websocket(SimpleNamespace(send_json=send_json), key)
        request = HITLRequest(agent_name="ResultReview", action_type=HITLAction.APPROVE,
            message="Review", requires_human=True, timeout_seconds=0.01,
            context={"experiment_review_id": "result:v1", "experiment_review_kind": "result",
                     "_session": {"user_id": key[0], "session_id": key[1]}})
        handle.controller.request_pause(handle.run_id, "manual")
        with bind_run(handle):
            task = asyncio.create_task(handler.handle_request(request))
            await settle(lambda: any(row.get("type") == "hitl_timeout" for row in messages))
            assert not task.done()
            timeout = next(row for row in messages if row["type"] == "hitl_timeout")
            assert timeout["paused"] is True
            request_id = next(row["request_id"] for row in messages if row["type"] == "hitl_request")
            assert handler.resolve_request(request_id, {"action": "edit", "approved": False,
                "selected_task_ids": ["A"], "instructions": "use corrected inputs"}, key)
            answer = await task
            assert answer.selected_task_ids == ["A"]
            assert handle.status().pause_causes == ("manual",)
            # A new handler has no original Future, but uses the saved decision.
            restored = await WebHITLHandler().handle_request(request)
            assert restored.selected_task_ids == ["A"]
        assert handle.status().attempts_used == 0
    asyncio.run(scenario())


def test_rejected_chat_and_recovered_invocation_preserve_original_run_time(monkeypatch):
    async def scenario():
        runtime = WebRuntime()
        key = ("user", "session")
        handle = runtime.prepare_execution(key, {"message": "research"},
            SimpleNamespace(orchestrator=SimpleNamespace(max_llm_calls=100)))
        original_start = runtime.execution_snapshot(key)["started_at"]
        runtime.run_times[key] = {"started_at": original_start, "finished_at": None}
        handle.controller.request_pause(handle.run_id, "manual")
        assert not await runtime.start_run(key, {"message": "rejected new request"})
        assert runtime.run_times[key]["started_at"] == original_start

        ready, release = asyncio.Event(), asyncio.Event()
        async def resumed_chat(*args):
            ready.set()
            await release.wait()
        monkeypatch.setattr(web_app, "_handle_chat", resumed_chat)
        handle.controller.resume(handle.run_id, "manual")
        # Simulate loss of the process-local timing table after a restart.
        runtime.run_times.clear()
        assert await runtime.start_run(key, {"message": "research",
            "_execution_run_id": handle.run_id, "_execution_resume": True})
        await ready.wait()
        assert runtime.run_times[key] == {"started_at": original_start, "finished_at": None}
        handle.controller.request_pause(handle.run_id, "execution_error")
        release.set()
        await settle(lambda: key not in runtime.active_runs)
        assert runtime.run_times[key]["finished_at"] is None
        await runtime.stop_run(key)
        from datetime import datetime
        assert datetime.fromisoformat(runtime.run_times[key]["finished_at"]).tzinfo is not None
        await runtime.close()
    asyncio.run(scenario())


def test_blocked_scientific_return_is_durably_paused_not_completed(monkeypatch):
    async def scenario():
        runtime = WebRuntime()
        key = ("outcome-user", "outcome-session")
        handle = runtime.prepare_execution(
            key,
            {"message": "run the study"},
            SimpleNamespace(orchestrator=SimpleNamespace(max_llm_calls=100)),
        )
        runtime.execution_handles[key] = handle

        async def fake_chat(*_args):
            return None

        state = {
            "experiment_plan_review_paused": True,
            "experiment_review_pause_reason": "max_plan_revisions",
            "experiment_module_outcome": {
                "status": "blocked",
                "stage": "plan_review",
                "reason": "max_plan_revisions",
                "accepted": False,
            },
            "_master_active_tasks": [{"id": "TASK-1", "status": "TODO"}],
        }

        async def fresh(*_args):
            return state

        monkeypatch.setattr(web_app, "_handle_chat", fake_chat)
        monkeypatch.setattr(runtime, "_fresh_execution_state", fresh)
        await runtime.run_controlled_chat(key, {"message": "run the study"})

        status = handle.status()
        assert status.state != "completed"
        assert "planning_review" in status.pause_causes
        assert status.pending_decisions["planning_review"]["kind"] == "experiment_plan_recovery"
        assert status.metadata["continuation"]["state"]["_master_active_tasks"][0]["status"] == "TODO"
        await runtime.close()

    asyncio.run(scenario())


def test_explicit_plan_recovery_resume_preserves_budget_and_root_state(monkeypatch):
    async def scenario():
        runtime = WebRuntime()
        key = ("recovery-user", "recovery-session")
        handle = runtime.prepare_execution(
            key,
            {"message": "original research"},
            SimpleNamespace(orchestrator=SimpleNamespace(max_llm_calls=100)),
        )
        runtime.execution_handles[key] = handle
        with bind_run(handle):
            await before_model_attempt("spent-one")
            await before_model_attempt("spent-two")

        saved = {
            "experiment_plan_review_paused": True,
            "experiment_review_pause_reason": "max_plan_revisions",
            "experiment_plan_revision_count": 4,
            "experiment_plan_last_executable_candidate": {"plan_id": "PLAN-1"},
            "_master_active_tasks": [{"id": "TASK-1", "status": "TODO"}],
        }
        handle.controller.update_metadata(handle.run_id, {
            "continuation": {"state": saved, "boundary": "test"},
        })
        handle.controller.request_pause(
            handle.run_id,
            "planning_review",
            pending_decision={
                "kind": "experiment_plan_recovery",
                "reason": "max_plan_revisions",
            },
        )

        restored = {}

        class Manager:
            async def _set_state(self, name, value):
                restored[name] = value

        async def manager(*_args):
            return Manager()

        started = []

        async def start(_key, data):
            started.append(data)
            return True

        monkeypatch.setattr(runtime, "get_manager", manager)
        monkeypatch.setattr(runtime, "start_run", start)
        await runtime.resume_execution(key, handle.run_id)

        status = handle.status()
        assert status.attempts_used == 2
        assert status.budget_total == 100
        assert restored["experiment_plan_revision_count"] == 4
        assert restored["_master_active_tasks"] == [{"id": "TASK-1", "status": "TODO"}]
        assert restored["experiment_plan_recovery_requested"] == {
            "source": "run_control", "reason": "max_plan_revisions",
        }
        assert restored["_execution_resume_pending"] is True
        assert started[0]["_execution_resume"] is True
        await runtime.close()

    asyncio.run(scenario())


def test_exact_root_roadmap_limits_can_be_human_accepted_and_resumed(monkeypatch):
    async def scenario():
        runtime = WebRuntime()
        key = ("limited-user", "limited-session")
        handle = runtime.prepare_execution(
            key, {"message": "full study"},
            SimpleNamespace(orchestrator=SimpleNamespace(max_llm_calls=100)),
        )
        runtime.execution_handles[key] = handle
        with bind_run(handle):
            await before_model_attempt("already-spent")
        state = {
            "experiment_runtime": {"phase": "completed", "task_order": [], "tasks": {}},
            "experiment_module_outcome": {
                "status": "completed", "stage": "result_review",
                "reason": "result_approved", "accepted": True,
            },
            "_master_active_tasks": [
                {"id": "TASK-1", "status": "DONE", "title": "finished"},
                {"id": "TASK-2", "status": "TODO", "title": "not run"},
            ],
        }
        handle.controller.update_metadata(handle.run_id, {"continuation": {"state": state}})
        handle.controller.request_pause(handle.run_id, "scientific_work_incomplete",
            pending_decision={"kind": "scientific_outcome",
                              "reason": "root_roadmap_incomplete",
                              "unfinished_task_ids": ["TASK-2"]})

        restored = {}

        class Manager:
            async def _set_state(self, name, value):
                restored[name] = value

        async def manager(*_args):
            return Manager()

        started = []

        async def start(_key, data):
            started.append(data)
            return True

        monkeypatch.setattr(runtime, "get_manager", manager)
        monkeypatch.setattr(runtime, "start_run", start)
        await runtime.decide_scientific_outcome(key, handle.run_id, {
            "decision": "accept_limited",
            "decision_id": "limited-1",
            "confirmed": True,
            "accepted_item_ids": ["TASK-2"],
        })

        acceptance = restored["scientific_limited_scope_acceptance"]
        assert acceptance["run_id"] == handle.run_id
        assert acceptance["accepted_task_ids"] == ["TASK-2"]
        assert "Do not execute or retry" in started[0]["_execution_resume_instruction"]
        assert handle.status().attempts_used == 1
        from CoScientist.experiments.outcome.reconciliation import (
            DispositionKind, reconcile_scientific_outcome,
        )
        assert reconcile_scientific_outcome(
            restored, current_run_id=handle.run_id,
        ).kind is DispositionKind.COMPLETED_LIMITED
        await runtime.close()

    asyncio.run(scenario())


def test_capacity_resolution_cannot_approve_above_hard_schema_limit(monkeypatch):
    async def scenario():
        runtime = WebRuntime()
        key = ("capacity-user", "capacity-session")
        handle = runtime.prepare_execution(
            key, {"message": "large study"},
            SimpleNamespace(orchestrator=SimpleNamespace(max_llm_calls=100)),
        )
        runtime.execution_handles[key] = handle
        state = {
            "experiment_context": {
                "experiment_run_id": "EXRUN-capacity",
                "operations": [{"operation_id": f"OP-{i}", "statement": "compute"}
                               for i in range(21)],
            },
            "experiment_plan_review_paused": True,
            "experiment_review_pause_reason": "plan_capacity_conflict",
            "experiment_module_outcome": {
                "status": "blocked", "stage": "preflight",
                "reason": "plan_capacity_conflict", "can_raise_limit": False,
                "requested_max_plan_tasks": 20, "accepted": False,
            },
        }
        handle.controller.update_metadata(handle.run_id, {"continuation": {"state": state}})
        handle.controller.request_pause(handle.run_id, "planning_review",
            pending_decision={"kind": "plan_capacity_conflict",
                              "reason": "plan_capacity_conflict",
                              "can_raise_limit": False})

        with pytest.raises(ValueError, match="hard 20-task"):
            await runtime.decide_scientific_outcome(key, handle.run_id, {
                "decision": "approve_capacity", "decision_id": "capacity-1",
                "confirmed": True,
            })
        assert handle.status().attempts_used == 0
        assert "planning_review" in handle.status().pause_causes
        await runtime.close()

    asyncio.run(scenario())


def test_capacity_resolution_reuses_saved_run_until_preflight_consumes_exact_grant(monkeypatch):
    async def scenario():
        from google.adk.models import LlmResponse
        from google.genai import types

        from CoScientist.config import get_settings
        from CoScientist.experiments.context.builder import build_experiment_context
        from CoScientist.experiments.plan_policy import check_experiment_plan_capacity
        from CoScientist.experiments.runtime.coalesce import (
            coalesce_experiment_module_calls,
            prepare_experiment_user_turn,
            suppress_experiment_module_after_completed,
        )

        monkeypatch.setattr(get_settings().experiments, "max_plan_tasks", 8)
        runtime = WebRuntime()
        key = ("capacity-resume-user", "capacity-resume-session")
        handle = runtime.prepare_execution(
            key, {"message": "compute all properties"},
            SimpleNamespace(orchestrator=SimpleNamespace(max_llm_calls=100)),
        )
        runtime.execution_handles[key] = handle
        operations = [
            {"operation_id": f"OP-{i}", "statement": f"Compute property {i}"}
            for i in range(1, 10)
        ]
        state = {
            "experiment_source_request": "compute all properties",
            "experiment_operations": operations,
            "experiment_context": {
                "experiment_run_id": "EXRUN-capacity-resume",
                "source_request": "compute all properties",
                "operations": operations,
            },
            "experiment_plan_revision_count": 3,
            "experiment_plan_last_executable_candidate": {"plan_id": "PLAN-saved"},
            "experiment_plan_review_paused": True,
            "experiment_review_pause_reason": "plan_capacity_conflict",
            "experiment_module_outcome": {
                "status": "blocked", "stage": "preflight",
                "reason": "plan_capacity_conflict", "can_raise_limit": True,
                "requested_max_plan_tasks": 9, "max_plan_tasks": 8,
                "required_operations": [row["operation_id"] for row in operations],
                "accepted": False,
            },
        }
        handle.controller.update_metadata(handle.run_id, {"continuation": {"state": state}})
        handle.controller.request_pause(
            handle.run_id, "planning_review",
            pending_decision={
                "kind": "plan_capacity_conflict", "reason": "plan_capacity_conflict",
                "can_raise_limit": True,
            },
        )

        restored = {}

        class Manager:
            async def _set_state(self, name, value):
                restored[name] = value

        async def manager(*_args):
            return Manager()

        async def start(_key, _data):
            return True

        monkeypatch.setattr(runtime, "get_manager", manager)
        monkeypatch.setattr(runtime, "start_run", start)
        await runtime.decide_scientific_outcome(key, handle.run_id, {
            "decision": "approve_capacity", "decision_id": "capacity-resume-1",
            "confirmed": True,
        })

        # The endpoint keeps the pause typed.  The forced saved-module hop must
        # cross suppression, and only the exact-run preflight may clear it.
        assert restored["experiment_plan_review_paused"] is True
        assert restored["experiment_review_pause_reason"] == "plan_capacity_conflict"
        prepare_experiment_user_turn(SimpleNamespace(
            state=restored, invocation_id="resume-invocation",
            user_content=types.Content(
                role="user", parts=[types.Part(text="compute all properties")],
            ),
        ))
        prose = LlmResponse(content=types.Content(
            role="model", parts=[types.Part(text="Continuing the root roadmap.")],
        ))
        ctx = SimpleNamespace(
            state=restored, agent_name="OrchestratorAgent",
            user_content=types.Content(
                role="user", parts=[types.Part(text="compute all properties")],
            ),
        )
        forced = coalesce_experiment_module_calls(ctx, prose)
        assert forced.content.parts[0].function_call.name == "ExperimentModuleAgent"
        suppress_experiment_module_after_completed(ctx, forced)
        assert forced.content.parts[0].function_call.name == "ExperimentModuleAgent"

        build_experiment_context(SimpleNamespace(state=restored, user_content=None))
        assert restored["experiment_context"]["experiment_run_id"] == "EXRUN-capacity-resume"
        assert restored["experiment_plan_revision_count"] == 3
        assert restored["experiment_plan_last_executable_candidate"] == {"plan_id": "PLAN-saved"}
        assert await check_experiment_plan_capacity(SimpleNamespace(state=restored)) is None
        assert restored["experiment_plan_review_paused"] is False
        assert restored["experiment_review_pause_reason"] is None
        assert restored["experiment_module_outcome"] is None
        assert restored["experiment_context"]["plan_limits"]["max_tasks"] == 9
        await runtime.close()

    asyncio.run(scenario())


def test_root_continue_does_not_redispatch_an_accepted_failed_experiment(monkeypatch):
    async def scenario():
        from google.adk.models import LlmResponse
        from google.genai import types

        from CoScientist.experiments.runtime.coalesce import (
            prepare_experiment_user_turn,
            suppress_experiment_module_after_completed,
        )

        runtime = WebRuntime()
        key = ("root-continue-user", "root-continue-session")
        handle = runtime.prepare_execution(
            key, {"message": "complete the roadmap"},
            SimpleNamespace(orchestrator=SimpleNamespace(max_llm_calls=100)),
        )
        runtime.execution_handles[key] = handle
        state = {
            "experiment_source_request": "complete the roadmap",
            "experiment_runtime": {
                "phase": "completed", "task_order": ["EXP-1"],
                "tasks": {"EXP-1": {"status": "failed"}},
            },
            "experiment_task_results": [
                {"task_id": "EXP-1", "execution_status": "failed"},
            ],
            "experiment_module_outcome": {
                "status": "completed", "stage": "result_review",
                "reason": "result_approved", "accepted": True,
            },
            "_master_active_tasks": [
                {"id": "TASK-EM", "status": "DONE", "assignee": "ExperimentModuleAgent"},
                {"id": "TASK-REPORT", "status": "TODO", "assignee": "ReportAgent"},
            ],
        }
        handle.controller.update_metadata(handle.run_id, {"continuation": {"state": state}})
        handle.controller.request_pause(
            handle.run_id, "scientific_work_incomplete",
            pending_decision={
                "kind": "scientific_outcome", "reason": "root_roadmap_incomplete",
                "unfinished_task_ids": ["TASK-REPORT"],
            },
        )
        restored = {}

        class Manager:
            async def _set_state(self, name, value):
                restored[name] = value

        async def manager(*_args):
            return Manager()

        async def start(_key, _data):
            return True

        monkeypatch.setattr(runtime, "get_manager", manager)
        monkeypatch.setattr(runtime, "start_run", start)
        await runtime.decide_scientific_outcome(key, handle.run_id, {
            "decision": "continue", "decision_id": "root-continue-1",
            "confirmed": True,
        })
        prepare_experiment_user_turn(SimpleNamespace(
            state=restored, invocation_id="root-continue-invocation",
            user_content=types.Content(
                role="user", parts=[types.Part(text="complete the roadmap")],
            ),
        ))
        response = LlmResponse(content=types.Content(
            role="model", parts=[types.Part.from_function_call(
                name="ExperimentModuleAgent", args={"request": "retry the failed task"},
            )],
        ))
        suppress_experiment_module_after_completed(
            SimpleNamespace(state=restored, agent_name="OrchestratorAgent"), response,
        )
        assert not any(
            getattr(getattr(part, "function_call", None), "name", None)
            == "ExperimentModuleAgent"
            for part in response.content.parts
        )
        assert restored.get("experiment_module_runs") is None
        await runtime.close()

    asyncio.run(scenario())
