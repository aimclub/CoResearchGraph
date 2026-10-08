"""The sandbox agent's own plan, on its way out to a watching UI.

A sandbox task is the longest single thing the system does: one tool call that
can run for hours and reports nothing until it ends. The agent inside the
container keeps a task list of its own, though, and the poll loop already sees
it on every ``/status`` response — so the plan is the only account of what is
happening in there while it happens.

What is worth pinning is when the relay stays *quiet*: the same snapshot comes
back on every poll, and re-announcing it would make a UI redraw every ten
seconds for a plan that has not changed.
"""

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from CoScientist.tools.coder_tools import openhands_sandbox as client
from CoScientist.tools.coder_tools import sandbox_tools


def _task(revision, current="Обучить модель", status="running"):
    return {
        "status": status,
        "plan": {
            "revision": revision,
            "current": current,
            "progress": {"total": 3, "done": 1},
            "items": [{"title": current, "status": "in_progress"}],
        },
    }


# ---------------------------------------------------------------------------
# _PlanRelay
# ---------------------------------------------------------------------------

def test_a_revision_is_announced_once():
    seen = []
    relay = client._PlanRelay(seen.append)

    relay.offer(_task(1))
    relay.offer(_task(1))          # same snapshot, next poll
    relay.offer(_task(2))
    relay.offer(_task(2))

    assert [plan["revision"] for plan in seen] == [1, 2]


def test_no_plan_is_not_an_event():
    seen = []
    relay = client._PlanRelay(seen.append)

    relay.offer({"status": "running"})            # `plan: null` — never planned
    relay.offer({"status": "running", "plan": None})
    relay.offer(None)                             # record pruned server-side

    assert seen == []


def test_the_relay_hands_over_a_copy():
    """A watcher that mutates what it got must not corrupt the next comparison."""
    seen = []
    relay = client._PlanRelay(seen.append)
    task = _task(1)

    relay.offer(task)
    seen[0]["revision"] = 99

    assert task["plan"]["revision"] == 1
    relay.offer(_task(1))
    assert len(seen) == 1


def test_a_broken_watcher_never_fails_the_run():
    def explode(plan):
        raise RuntimeError("watcher is down")

    relay = client._PlanRelay(explode)
    relay.offer(_task(1))          # must not raise
    asyncio.run(relay.aoffer(_task(2)))


def test_no_watcher_costs_nothing():
    relay = client._PlanRelay(None)
    relay.offer(_task(1))
    asyncio.run(relay.aoffer(_task(2)))


def test_a_coroutine_watcher_is_awaited():
    seen = []

    async def collect(plan):
        seen.append(plan["revision"])

    relay = client._PlanRelay(collect)
    asyncio.run(relay.aoffer(_task(7)))

    assert seen == [7]


def test_a_coroutine_watcher_is_refused_by_the_blocking_api(caplog):
    """Mirrors `on_start`: the sync path has no loop to await on."""
    async def collect(plan):
        pass

    relay = client._PlanRelay(collect)
    with caplog.at_level("WARNING"):
        relay.offer(_task(1))

    assert "arun_sandbox_task" in caplog.text


# ---------------------------------------------------------------------------
# The ADK adapter
# ---------------------------------------------------------------------------

@pytest.fixture
def plan_sink():
    received = []

    async def collect(key, info):
        received.append((key, info))

    sandbox_tools.set_sandbox_plan_sink(collect)
    yield received
    sandbox_tools.set_sandbox_plan_sink(None)


def test_the_notifier_routes_the_plan_to_the_host_session(plan_sink, monkeypatch):
    monkeypatch.setattr(sandbox_tools, "_host_session", lambda ctx: ("user_1", "session_1"))
    notify = sandbox_tools._plan_notifier(SimpleNamespace(agent_name="CoderAgent"))

    asyncio.run(notify({"revision": 2, "current": "Обучить модель"}))

    (key, info), = plan_sink
    assert key == ("user_1", "session_1")
    assert info["agent"] == "CoderAgent"
    assert info["plan"]["current"] == "Обучить модель"


def test_with_no_sink_registered_the_notifier_is_inert(monkeypatch):
    """CLI and A2A runs pay nothing for a UI that is not there."""
    sandbox_tools.set_sandbox_plan_sink(None)
    monkeypatch.setattr(sandbox_tools, "_host_session", lambda ctx: None)
    notify = sandbox_tools._plan_notifier(None)

    asyncio.run(notify({"revision": 1}))  # must not raise


def test_every_waiting_entry_point_accepts_a_plan_watcher():
    """`check_sandbox_task` waits as long as `run_sandbox_task` does."""
    for fn in (client.run_sandbox_task, client.arun_sandbox_task,
               client.await_sandbox_task):
        assert "on_plan" in inspect.signature(fn).parameters, fn.__name__


# ---------------------------------------------------------------------------
# The web runtime
# ---------------------------------------------------------------------------

def test_the_web_app_relays_the_plan_to_the_tabs_watching_the_session(monkeypatch):
    import importlib

    web_app = importlib.import_module("CoScientist.web.app")
    runtime = web_app.WebRuntime()
    web_app._wire_sandbox_plan(runtime)
    sink = sandbox_tools._plan_sink
    assert sink is not None

    key = ("user_1", "session_1")
    runtime.sockets[key] = set()
    sent = []

    async def capture(target, payload):
        sent.append((target, payload))

    monkeypatch.setattr(runtime, "send", capture)

    async def scenario():
        await sink(key, {"agent": "CoderAgent", "plan": {"revision": 4}})
        # A session nobody is watching, and a call with no host session at all.
        await sink(("user_1", "other"), {"plan": {"revision": 1}})
        await sink(None, {"plan": {"revision": 1}})

    try:
        asyncio.run(scenario())
    finally:
        sandbox_tools.set_sandbox_plan_sink(None)

    assert [payload for _, payload in sent] == [
        {"type": "sandbox_plan", "agent": "CoderAgent", "plan": {"revision": 4}},
    ]
