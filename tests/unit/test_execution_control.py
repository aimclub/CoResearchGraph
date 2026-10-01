"""Durable global execution budget and cooperative lifecycle control."""
import asyncio

import pytest

from CoScientist.execution_control import (
    BUDGET_PAUSE_CAUSE,
    ConcurrentControlUpdate,
    DuplicateAttempt,
    RunController,
    RunStopped,
    before_model_attempt,
    bind_run,
    current_run,
)


def test_run_is_durable_and_create_does_not_reset_budget(tmp_path):
    db = tmp_path / "control.sqlite3"
    first = RunController(db)
    handle = first.create_run("r1", initial_budget=3, metadata={"session_id": "s"})

    async def reserve():
        with bind_run(handle):
            await before_model_attempt("one")

    asyncio.run(reserve())

    restarted = RunController(db)
    restarted.create_run("r1", initial_budget=100, metadata={"session_id": "wrong"})
    status = restarted.status("r1")
    assert status.initial_budget == 3
    assert status.attempts_used == 1
    assert status.metadata == {"session_id": "s"}


def test_context_is_inherited_by_nested_async_tasks(tmp_path):
    controller = RunController(tmp_path / "control.sqlite3")
    handle = controller.create_run(initial_budget=2)

    async def scenario():
        with bind_run(handle):
            assert current_run() is handle
            reservation = await asyncio.create_task(before_model_attempt("nested"))
            assert reservation is not None
        assert current_run() is None

    asyncio.run(scenario())
    assert controller.status(handle.run_id).attempts_used == 1


def test_exhaustion_waits_for_an_idempotent_explicit_100_grant(tmp_path):
    db = tmp_path / "control.sqlite3"
    worker = RunController(db, poll_interval=0.01)
    operator = RunController(db, poll_interval=0.01)
    handle = worker.create_run("r", initial_budget=1)

    async def scenario():
        await handle.before_model_attempt("first")
        waiting = asyncio.create_task(handle.before_model_attempt("second"))
        for _ in range(100):
            if operator.status("r").state == "paused":
                break
            await asyncio.sleep(0.01)
        status = operator.status("r")
        assert status.state == "paused"
        assert status.pause_causes == (BUDGET_PAUSE_CAUSE,)
        assert status.pending_decisions[BUDGET_PAUSE_CAUSE]["grant_size"] == 100
        assert not waiting.done()

        decision_revision = status.control_revision
        operator.grant(
            "r", "user-decision-1", expected_revision=decision_revision
        )
        # An HTTP replay carries the old revision but the same durable decision
        # id.  It is still a no-op, not a stale-update conflict.
        operator.grant(
            "r", "user-decision-1", expected_revision=decision_revision
        )
        reservation = await asyncio.wait_for(waiting, 2)
        assert reservation.attempt_no == 2

    asyncio.run(scenario())
    status = operator.status("r")
    assert status.budget_total == 101
    assert status.attempts_used == 2
    assert status.remaining == 99
    with pytest.raises(ValueError, match="exactly 100"):
        operator.grant("r", "bad-size", amount=101)


def test_pause_causes_are_independent_and_resume_never_resets_usage(tmp_path):
    controller = RunController(tmp_path / "control.sqlite3", poll_interval=0.01)
    handle = controller.create_run("r", initial_budget=4)

    async def scenario():
        await handle.before_model_attempt("first")
        controller.request_pause("r", "manual", pending_decision={"question": "continue?"})
        controller.request_pause("r", "review")
        task = asyncio.create_task(handle.before_tool_action("tool"))
        await asyncio.sleep(0.03)
        assert controller.status("r").state == "paused"
        controller.resume("r", "manual")
        await asyncio.sleep(0.03)
        assert not task.done()
        assert controller.status("r").pause_causes == ("review",)
        controller.resume("r", "review")
        await asyncio.wait_for(task, 1)

    asyncio.run(scenario())
    status = controller.status("r")
    assert status.state == "running"
    assert status.attempts_used == 1
    assert status.pending_decisions == {}


def test_stop_signal_bypasses_ordinary_retry_handlers(tmp_path):
    db = tmp_path / "control.sqlite3"
    worker = RunController(db, poll_interval=0.01)
    operator = RunController(db, poll_interval=0.01)
    handle = worker.create_run("r")
    operator.request_pause("r", "manual")

    async def retry_like_wrapper():
        try:
            await handle.before_model_attempt("never-sent")
        except Exception:  # execution-control signals deliberately bypass this
            return "retried"
        return "sent"

    async def scenario():
        task = asyncio.create_task(retry_like_wrapper())
        await asyncio.sleep(0.03)
        operator.stop("r", "user stopped")
        with pytest.raises(RunStopped, match="user stopped"):
            await asyncio.wait_for(task, 1)

    asyncio.run(scenario())
    assert worker.status("r").attempts_used == 0


def test_reservations_are_atomic_and_action_ids_cannot_replay(tmp_path):
    controller = RunController(tmp_path / "control.sqlite3")
    handle = controller.create_run("r", initial_budget=20)

    async def scenario():
        reservations = await asyncio.gather(
            *(handle.before_model_attempt("parallel") for _ in range(20))
        )
        assert sorted(item.attempt_no for item in reservations) == list(range(1, 21))
        with pytest.raises(DuplicateAttempt):
            await handle.before_model_attempt("x", action_id=reservations[0].action_id)

    asyncio.run(scenario())
    assert controller.status("r").attempts_used == 20


def test_concurrent_same_action_id_allows_exactly_one_provider_attempt(tmp_path):
    controller = RunController(tmp_path / "control.sqlite3")
    handle = controller.create_run("r")

    async def scenario():
        outcomes = await asyncio.gather(
            handle.before_model_attempt("parallel", action_id="same"),
            handle.before_model_attempt("parallel", action_id="same"),
            return_exceptions=True,
        )
        assert sum(not isinstance(item, BaseException) for item in outcomes) == 1
        assert sum(isinstance(item, DuplicateAttempt) for item in outcomes) == 1

    asyncio.run(scenario())
    assert controller.status("r").attempts_used == 1


def test_cancelling_a_paused_gate_does_not_reserve_an_attempt(tmp_path):
    controller = RunController(tmp_path / "control.sqlite3", poll_interval=0.01)
    handle = controller.create_run("r")
    controller.request_pause("r", "manual")

    async def scenario():
        waiting = asyncio.create_task(handle.before_model_attempt("cancelled"))
        await asyncio.sleep(0.03)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

    asyncio.run(scenario())
    assert controller.status("r").attempts_used == 0


def test_control_revision_ignores_metadata_and_attempts_but_guards_decisions(tmp_path):
    controller = RunController(tmp_path / "control.sqlite3")
    handle = controller.create_run("r", initial_budget=2)
    initial = controller.status("r")
    controller.update_metadata("r", {"message": "hello"})
    asyncio.run(handle.before_model_attempt("one"))
    assert controller.status("r").control_revision == initial.control_revision

    paused = controller.request_pause(
        "r", "manual", expected_revision=initial.control_revision
    )
    assert paused.control_revision == initial.control_revision + 1
    with pytest.raises(ConcurrentControlUpdate):
        controller.resume("r", "manual", expected_revision=initial.control_revision)


def test_status_shape_reconnect_lookup_and_journal(tmp_path):
    controller = RunController(tmp_path / "control.sqlite3")
    handle = controller.create_run(
        "r", metadata={"user_id": "u", "session_id": "s"}
    )
    controller.journal("r", "message", action_id="m1", data={"value": 1})
    found = controller.latest_run(user_id="u", session_id="s")
    assert found is not None and found.run_id == handle.run_id
    payload = found.status().to_dict()
    assert payload["metadata"] == {"user_id": "u", "session_id": "s"}
    assert isinstance(payload["pause_causes"], list)
    assert payload["budget"] == {"used": 0, "limit": 100, "remaining": 100}
    assert isinstance(payload["control_revision"], int)
    assert controller.journal_entries("r")[-1]["action_id"] == "m1"


def test_async_notifier_observes_reservations_and_state_changes(tmp_path):
    seen = []

    async def notify(handle, status):
        seen.append((handle.run_id, status.state, status.attempts_used))

    controller = RunController(tmp_path / "control.sqlite3", notifier=notify)

    async def scenario():
        handle = controller.create_run("r")
        await handle.before_model_attempt("one")
        controller.request_pause("r", "manual")
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert ("r", "running", 1) in seen
    assert any(state == "pause_requested" for _, state, _ in seen)


def test_default_journal_read_includes_latest_entries_beyond_first_thousand(tmp_path):
    from CoScientist.agents.run_control_plugin import unresolved_actions

    controller = RunController(tmp_path / "control.sqlite3")
    handle = controller.create_run("r")
    controller.journal(
        "r", "tool_dispatched", action_id="long-action",
        data={"tool": "submit_training", "delegation": False},
    )
    for index in range(1001):
        controller.journal("r", "progress", data={"index": index})
    controller.journal(
        "r", "tool_completed", action_id="long-action",
        data={"tool": "submit_training", "result": {"status": "done"}},
    )

    complete = controller.journal_entries(handle.run_id)
    assert len(complete) > 1000
    assert complete[-1]["event"] == "tool_completed"
    assert not unresolved_actions(complete)
    assert len(controller.journal_entries(handle.run_id, limit=10)) == 10
