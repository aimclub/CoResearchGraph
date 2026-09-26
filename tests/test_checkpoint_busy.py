"""Restore must wait for actual Runner completion, never an elapsed-time guess."""

import asyncio
import weakref
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from google.adk.agents.base_agent import BaseAgent
from google.adk.events.event import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import SecretStr

from CoScientist.a2a import server
from CoScientist.checkpoints import api, plugin
from CoScientist.checkpoints.store import LocalZipStore

TOKEN = "test-only-platform-admin-credential"
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def control(monkeypatch, tmp_path):
    import CoScientist.config

    monkeypatch.setattr(plugin.CheckpointPlugin, "_instances", weakref.WeakSet())

    monkeypatch.setattr(
        CoScientist.config,
        "get_settings",
        lambda: SimpleNamespace(
            checkpoints=SimpleNamespace(api_token=SecretStr(TOKEN))
        ),
    )
    restored = AsyncMock(return_value={"context_id": "new-context"})
    monkeypatch.setattr(api, "restore_checkpoint", restored)
    app = FastAPI()
    app.include_router(
        api.make_checkpoint_router(
            session_service=InMemorySessionService(),
            app_name="other-agent",
            store=LocalZipStore(str(tmp_path)),
        )
    )
    return app, restored


async def restore_status(control):
    app, _ = control
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/checkpoints/saved/restore",
            json={},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        return response.status_code


async def runner_for(agent):
    checkpoint = plugin.CheckpointPlugin()
    checkpoint._save = AsyncMock()  # isolate the restore gate from bundle capture
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="busy-probe", user_id="u")
    runner = server.Runner(
        agent=agent,
        app_name="busy-probe",
        session_service=sessions,
        plugins=[checkpoint],
    )
    return runner, checkpoint, session.id


def events_for(runner, session_id):
    return runner.run_async(
        user_id="u",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text="probe")]),
    )


async def consume(events):
    async with aclosing(events):
        return [event async for event in events]


def blocking_agent(entered, release, *, fail=False, closed=None):
    class BlockingAgent(BaseAgent):
        async def _run_async_impl(self, ctx):
            entered.set()
            try:
                await release.wait()
                if fail:
                    raise ValueError("deliberate agent failure")
                yield Event(
                    author=self.name,
                    invocation_id=ctx.invocation_id,
                    content=types.Content(
                        role="model", parts=[types.Part(text="done")]
                    ),
                )
            finally:
                if closed:
                    closed.set()

    return BlockingAgent(name="BlockingAgent")


async def test_live_invocation_blocks_restore_after_two_hours(control, monkeypatch):
    # Patch the old gate's clock only; the event loop and agent remain live.
    clock = SimpleNamespace(monotonic=lambda: 100.0)
    monkeypatch.setattr(plugin, "time", clock, raising=False)
    entered, release = asyncio.Event(), asyncio.Event()
    runner, checkpoint, session_id = await runner_for(blocking_agent(entered, release))
    task = asyncio.create_task(consume(events_for(runner, session_id)))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        clock.monotonic = lambda: 7301.0
        assert not task.done()
        assert await restore_status(control) == 409
        control[1].assert_not_awaited()
        assert checkpoint.is_busy()
    finally:
        release.set()
        await task
    assert await restore_status(control) == 200


@pytest.mark.parametrize("termination", ["success", "error", "cancel"])
async def test_actual_termination_releases_gate(control, termination):
    entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    runner, checkpoint, session_id = await runner_for(
        blocking_agent(
            entered,
            release,
            fail=termination == "error",
            closed=closed,
        )
    )
    task = asyncio.create_task(consume(events_for(runner, session_id)))
    await asyncio.wait_for(entered.wait(), 5)
    assert await restore_status(control) == 409
    if termination == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        release.set()
        if termination == "error":
            with pytest.raises(ValueError, match="deliberate agent failure"):
                await task
        else:
            await task
    assert closed.is_set()
    assert not checkpoint.is_busy()
    assert await restore_status(control) == 200


async def test_closing_stream_releases_gate_after_agent_cleanup(control):
    closed = asyncio.Event()

    class StreamingAgent(BaseAgent):
        async def _run_async_impl(self, ctx):
            try:
                for _ in range(2):
                    yield Event(author=self.name, invocation_id=ctx.invocation_id)
            finally:
                closed.set()

    runner, checkpoint, session_id = await runner_for(StreamingAgent(name="Streaming"))
    events = events_for(runner, session_id)
    async with aclosing(events):
        await anext(events)
        assert await restore_status(control) == 409
    assert closed.is_set()
    assert not checkpoint.is_busy()
    assert await restore_status(control) == 200


@pytest.mark.parametrize("same_runner", [False, True])
async def test_finishing_one_invocation_does_not_release_another(control, same_runner):
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    if same_runner:
        invocation_index = iter(range(2))

        class ConcurrentAgent(BaseAgent):
            async def _run_async_impl(self, ctx):
                index = next(invocation_index)
                entered[index].set()
                await release[index].wait()
                yield Event(author=self.name, invocation_id=ctx.invocation_id)

        runner, checkpoint, sid = await runner_for(ConcurrentAgent(name="Concurrent"))
        session = await runner.session_service.create_session(
            app_name="busy-probe",
            user_id="u",
        )
        runs = [(runner, checkpoint, sid), (runner, checkpoint, session.id)]
    else:
        runs = [
            await runner_for(blocking_agent(entered[i], release[i])) for i in range(2)
        ]
    tasks = [
        asyncio.create_task(consume(events_for(runner, sid))) for runner, _, sid in runs
    ]
    try:
        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in entered)), 5)
        release[0].set()
        await tasks[0]
        assert not tasks[1].done()
        assert await restore_status(control) == 409
    finally:
        for event in release:
            event.set()
        await asyncio.gather(*tasks)
    assert await restore_status(control) == 200


async def test_cancel_keeps_restore_blocked_during_async_cleanup(control):
    entered, cleaning, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release_cleanup = asyncio.Event()

    class CleaningAgent(BaseAgent):
        async def _run_async_impl(self, ctx):
            entered.set()
            try:
                await asyncio.Event().wait()
                yield Event(author=self.name)
            finally:
                cleaning.set()
                await release_cleanup.wait()
                cleaned.set()

    runner, checkpoint, sid = await runner_for(CleaningAgent(name="Cleaning"))
    task = asyncio.create_task(consume(events_for(runner, sid)))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 5)
        assert not cleaned.is_set()
        assert await restore_status(control) == 409
    finally:
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cleaned.is_set()
    assert not checkpoint.is_busy()
    assert await restore_status(control) == 200
