"""CLI decisions for every durable run-control pause cause."""
from __future__ import annotations

import asyncio

from CoScientist.execution_cli import (
    _BUDGET_CONTINUE,
    _FINISH_LIMITS,
    _INSPECTED_RETRY,
    _STOP,
    _watch_run_control,
)
from CoScientist.execution_control import RunController
from CoScientist.hitl.models import (
    HITLAction,
    HITLDecisionSource,
    HITLResponse,
)


class QueueHandler:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    async def handle_request(self, request):
        self.requests.append(request)
        if not self.responses:
            await asyncio.Future()
        response = self.responses.pop(0)
        return response


def human(option: str, *, notes: str | None = None) -> HITLResponse:
    return HITLResponse(
        action=HITLAction.SELECT,
        selected_option=option,
        instructions=notes,
        approved=True,
        decision_source=HITLDecisionSource.HUMAN,
    )


async def settle(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.005)


def test_semantic_pause_offers_honest_finish_and_persists_human_instruction(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        controller.request_pause(
            "run", "execution_error",
            pending_decision={
                "kind": "semantic_control_loop_guard",
                "error_code": "control_no_progress_limit",
                "state_revision": "rev-7",
            },
        )
        handler = QueueHandler(human(_FINISH_LIMITS))
        watcher = asyncio.create_task(
            _watch_run_control(handle, controller, handler, poll_interval=0.005)
        )
        await settle(lambda: "execution_error" not in handle.status().pause_causes)
        assert handler.requests[0].requires_human is True
        assert handler.requests[0].options == [_FINISH_LIMITS, _STOP]
        assert handle.status().metadata["control_resolution"] == {
            "decision": "finish_with_limits",
            "source": "human",
            "reason": "control_no_progress_limit",
        }
        event = next(
            row for row in controller.journal_entries("run")
            if row["event"] == "control_resolution_requested"
        )
        assert event["action_id"] == "semantic:run:rev-7"
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)

    asyncio.run(scenario())


def test_unknown_external_actions_require_inspection_and_do_not_reset_budget(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run", initial_budget=3)
        await handle.before_model_attempt("spent")
        for action_id, tool in (("a1", "submit_training"), ("a2", "upload_model")):
            controller.journal(
                "run", "tool_outcome_unknown", action_id=action_id,
                data={"tool": tool, "outcome_unknown": True, "delegation": False},
            )
        controller.request_pause(
            "run", "unknown_completion",
            pending_decision={"kind": "external_outcome", "actions": ["a1", "a2"]},
        )
        handler = QueueHandler(human(_INSPECTED_RETRY, notes="Both jobs were absent remotely"))
        watcher = asyncio.create_task(
            _watch_run_control(handle, controller, handler, poll_interval=0.005)
        )
        await settle(lambda: "unknown_completion" not in handle.status().pause_causes)
        assert handler.requests[0].requires_human is True
        assert handler.requests[0].options == [_INSPECTED_RETRY, _STOP]
        authorized = [
            row for row in controller.journal_entries("run")
            if row["event"] == "tool_retry_authorized"
        ]
        assert {row["action_id"] for row in authorized} == {"a1", "a2"}
        assert all(row["data"]["source"] == "human" for row in authorized)
        status = handle.status()
        assert status.attempts_used == 1
        assert status.budget_total == 3
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)

    asyncio.run(scenario())


def test_concurrent_budget_is_decided_before_unknown_outcome(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3", poll_interval=0.005)
        handle = controller.create_run("run", initial_budget=1)
        await handle.before_model_attempt("first")
        blocked_model = asyncio.create_task(handle.before_model_attempt("second"))
        await settle(lambda: "budget_exhausted" in handle.status().pause_causes)
        controller.journal(
            "run", "tool_outcome_unknown", action_id="external",
            data={"tool": "submit_training", "outcome_unknown": True, "delegation": False},
        )
        controller.request_pause("run", "unknown_completion")
        handler = QueueHandler(human(_BUDGET_CONTINUE), human(_INSPECTED_RETRY))
        watcher = asyncio.create_task(
            _watch_run_control(handle, controller, handler, poll_interval=0.005)
        )
        await asyncio.wait_for(blocked_model, 2)
        assert [request.context["kind"] for request in handler.requests[:2]] == [
            "budget_exhausted", "unknown_completion",
        ]
        assert handle.status().budget_total == 101
        assert handle.status().attempts_used == 2
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)

    asyncio.run(scenario())


def test_nonhuman_answer_never_releases_pause_and_watcher_is_cancellable(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        controller.request_pause("run", "execution_error", pending_decision={
            "kind": "semantic_control_loop_guard", "error_code": "stalled",
        })
        automatic = HITLResponse(
            action=HITLAction.SELECT,
            selected_option=_FINISH_LIMITS,
            approved=True,
            decision_source=HITLDecisionSource.MODE_AUTO,
        )
        handler = QueueHandler(automatic)
        watcher = asyncio.create_task(
            _watch_run_control(handle, controller, handler, poll_interval=0.005)
        )
        await settle(lambda: len(handler.requests) >= 1)
        await asyncio.sleep(0.02)
        assert "execution_error" in handle.status().pause_causes
        assert handle.status().metadata.get("control_resolution") is None
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        assert watcher.cancelled()

    asyncio.run(scenario())


def test_stop_decision_terminates_watcher_and_run(tmp_path):
    async def scenario():
        controller = RunController(tmp_path / "control.sqlite3")
        handle = controller.create_run("run")
        controller.request_pause("run", "unknown_completion")
        controller.journal(
            "run", "tool_outcome_unknown", action_id="a1",
            data={"tool": "submit_training", "outcome_unknown": True, "delegation": False},
        )
        handler = QueueHandler(human(_STOP))
        await asyncio.wait_for(
            _watch_run_control(handle, controller, handler, poll_interval=0.005), 1
        )
        assert handle.status().state == "stopped"
        assert handle.status().stop_reason == "operator_stopped_at_unknown_outcome"

    asyncio.run(scenario())
