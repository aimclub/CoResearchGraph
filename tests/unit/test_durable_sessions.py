"""Transactional guarantees of the local ADK session adapter."""
from __future__ import annotations

import asyncio

import pytest
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from pydantic_core import PydanticSerializationError

from CoScientist.web.durable_sessions import DurableSessionService


def _event(event_id: str, delta: dict) -> Event:
    return Event(
        id=event_id,
        invocation_id="inv-1",
        author="test",
        actions=EventActions(state_delta=delta),
    )


def test_bad_delta_is_rejected_before_it_poisoned_the_session(tmp_path):
    async def scenario():
        service = DurableSessionService(tmp_path)
        session = await service.create_session(
            app_name="app", user_id="user", session_id="session", state={"ok": 1}
        )
        target = service._path("app", "user", "session")
        before = target.read_text(encoding="utf-8")

        with pytest.raises(PydanticSerializationError):
            await service.append_event(
                session,
                _event("bad", {"diagnostic": {"error": ValueError("not JSON")}}),
            )

        canonical = service._canonical(session)
        assert canonical.state == {"ok": 1}
        assert canonical.events == []
        assert target.read_text(encoding="utf-8") == before

        await service.append_event(session, _event("good", {"healthy": True}))
        assert canonical.state["healthy"] is True
        assert [event.id for event in canonical.events] == ["good"]

        reloaded = DurableSessionService(tmp_path)
        stored = await reloaded.get_session(
            app_name="app", user_id="user", session_id="session"
        )
        assert stored is not None and stored.state["healthy"] is True
        assert [event.id for event in stored.events] == ["good"]

    asyncio.run(scenario())


def test_persistence_failure_rolls_back_memory_and_leaves_disk_intact(
    tmp_path, monkeypatch
):
    async def scenario():
        service = DurableSessionService(tmp_path)
        session = await service.create_session(
            app_name="app", user_id="user", session_id="session", state={"ok": 1}
        )
        target = service._path("app", "user", "session")
        before = target.read_text(encoding="utf-8")
        persist = service._persist

        def fail(*_args, **_kwargs):
            raise OSError("simulated disk failure")

        monkeypatch.setattr(service, "_persist", fail)
        with pytest.raises(OSError, match="simulated disk failure"):
            await service.append_event(session, _event("not-committed", {"changed": True}))

        canonical = service._canonical(session)
        assert canonical.state == {"ok": 1}
        assert canonical.events == []
        assert target.read_text(encoding="utf-8") == before

        monkeypatch.setattr(service, "_persist", persist)
        await service.append_event(session, _event("committed", {"changed": True}))
        assert canonical.state["changed"] is True
        assert [event.id for event in canonical.events] == ["committed"]

    asyncio.run(scenario())


def test_failed_create_does_not_leave_session_or_shared_state(tmp_path, monkeypatch):
    async def scenario():
        service = DurableSessionService(tmp_path)

        with pytest.raises(PydanticSerializationError):
            await service.create_session(
                app_name="app",
                user_id="user",
                session_id="bad-json",
                state={"app:diagnostic": ValueError("not JSON")},
            )
        assert service.sessions == {}
        assert service.app_state == {}
        assert service.user_state == {}

        def fail(*_args, **_kwargs):
            raise OSError("simulated first-save failure")

        persist = service._persist
        monkeypatch.setattr(service, "_persist", fail)
        with pytest.raises(OSError, match="first-save failure"):
            await service.create_session(
                app_name="app",
                user_id="user",
                session_id="failed-save",
                state={"app:shared": 1, "user:preference": "x", "ok": True},
            )
        assert service.sessions == {}
        assert service.app_state == {}
        assert service.user_state == {}

        monkeypatch.setattr(service, "_persist", persist)
        session = await service.create_session(
            app_name="app",
            user_id="user",
            session_id="failed-save",
            state={"ok": True},
        )
        assert session.id == "failed-save"

    asyncio.run(scenario())


def test_failed_replace_restores_memory_and_disk(tmp_path, monkeypatch):
    async def scenario():
        service = DurableSessionService(tmp_path)
        session = await service.create_session(
            app_name="app", user_id="user", session_id="session", state={"ok": 1}
        )
        target = service._path("app", "user", "session")
        before = target.read_text(encoding="utf-8")
        replacement = session.model_copy(deep=True)
        replacement.state = {"ok": 2}

        def fail(*_args, **_kwargs):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(service, "_persist", fail)
        with pytest.raises(OSError, match="replace failure"):
            await service.replace_session(replacement)

        canonical = service._canonical(session)
        assert canonical.state == {"ok": 1}
        assert target.read_text(encoding="utf-8") == before

    asyncio.run(scenario())
