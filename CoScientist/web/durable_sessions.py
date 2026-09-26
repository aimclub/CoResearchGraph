"""Durable ADK sessions for the local web runtime.

ADK's in-memory service is useful for tests, but its session catalogue is lost
on every web restart.  This adapter keeps ADK's normal async API and stores a
JSON copy after every event.  Files are per session, so a torn write cannot
hide unrelated sessions and the replacement is atomic.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

from google.adk.events.event import Event
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.session import Session

from CoScientist.web.session_store import state_dir

_LOCK = threading.RLock()
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


class DurableSessionService(InMemorySessionService):
    """An InMemorySessionService with an on-disk session backing store."""

    def __init__(self, root: Optional[Path] = None) -> None:
        super().__init__()
        self.root = Path(root or state_dir()) / "adk_sessions"
        # Full events are already written to disk under one process-wide lock.
        # Serialising their in-memory mutation as well makes rollback exact for
        # session-, user- and app-scoped state when persistence fails.
        self._append_lock = asyncio.Lock()

    def _path(self, app_name: str, user_id: str, session_id: str) -> Path:
        return (
            self.root / _SAFE.sub("_", app_name) / _SAFE.sub("_", user_id)
            / f"{_SAFE.sub('_', session_id)}.json"
        )

    def _load(self, app_name: str, user_id: str, session_id: str) -> Optional[Session]:
        key = self._path(app_name, user_id, session_id)
        try:
            payload = json.loads(key.read_text(encoding="utf-8"))
            session = Session.model_validate(payload)
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
            return None
        self.sessions.setdefault(app_name, {}).setdefault(user_id, {})[session_id] = session
        return session

    def _canonical(self, session: Session) -> Session:
        return self.sessions[session.app_name][session.user_id][session.id]

    @staticmethod
    def _serialized(session: Session) -> str:
        return json.dumps(session.model_dump(mode="json"), ensure_ascii=False)

    def _persist(self, session: Session, *, serialized: str | None = None) -> None:
        target = self._path(session.app_name, session.user_id, session.id)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            serialized if serialized is not None else self._serialized(session),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    @staticmethod
    def _restore_session(target: Session, snapshot: Session) -> None:
        target.state = copy.deepcopy(snapshot.state)
        target.events = copy.deepcopy(snapshot.events)
        target.last_update_time = snapshot.last_update_time

    async def create_session(self, **kwargs: Any) -> Session:
        app_name = kwargs["app_name"]
        user_id = kwargs["user_id"]
        session_id = kwargs.get("session_id")
        # Validate every namespace before ADK extracts app:/user: values and
        # mutates its shared maps.  A failed first save must not leave a ghost
        # session or poisoned shared state behind.
        probe = Session(
            app_name=app_name,
            user_id=user_id,
            id=session_id or "serialization-probe",
            state=copy.deepcopy(kwargs.get("state") or {}),
        )
        self._serialized(probe)

        async with self._append_lock:
            if session_id and self._load(app_name, user_id, session_id) is not None:
                # Let ADK raise its normal AlreadyExistsError.
                return await super().create_session(**kwargs)

            sessions_existed = app_name in self.sessions
            users_existed = (
                sessions_existed and user_id in self.sessions.get(app_name, {})
            )
            sessions_before = copy.deepcopy(
                self.sessions.get(app_name, {}).get(user_id)
            )
            app_existed = app_name in self.app_state
            user_app_existed = app_name in self.user_state
            app_before = copy.deepcopy(self.app_state.get(app_name))
            user_before = copy.deepcopy(self.user_state.get(app_name))
            try:
                session = await super().create_session(**kwargs)
                canonical = self._canonical(session)
                serialized = self._serialized(canonical)
                with _LOCK:
                    self._persist(canonical, serialized=serialized)
                return session
            except Exception:
                if users_existed:
                    self.sessions.setdefault(app_name, {})[user_id] = sessions_before
                elif sessions_existed:
                    self.sessions.get(app_name, {}).pop(user_id, None)
                else:
                    self.sessions.pop(app_name, None)
                if app_existed:
                    self.app_state[app_name] = app_before
                else:
                    self.app_state.pop(app_name, None)
                if user_app_existed:
                    self.user_state[app_name] = user_before
                else:
                    self.user_state.pop(app_name, None)
                raise

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        if session_id not in self.sessions.get(app_name, {}).get(user_id, {}):
            self._load(app_name, user_id, session_id)
        return await super().get_session(
            app_name=app_name, user_id=user_id, session_id=session_id, config=config
        )

    async def append_event(self, session: Session, event: Event) -> Event:
        if session.id not in self.sessions.get(session.app_name, {}).get(session.user_id, {}):
            self._load(session.app_name, session.user_id, session.id)
        if event.partial:
            return await super().append_event(session=session, event=event)

        async with self._append_lock:
            canonical = self._canonical(session)
            # Fail before ADK mutates either the caller's session or the
            # canonical history.  Checking the event separately pinpoints a
            # bad delta even when the current session is still healthy.
            self._serialized(canonical)
            event.model_dump(mode="json")

            canonical_before = copy.deepcopy(canonical)
            session_before = (
                copy.deepcopy(session) if session is not canonical else canonical_before
            )
            app_name, user_id = session.app_name, session.user_id
            app_existed = app_name in self.app_state
            user_app_existed = app_name in self.user_state
            app_before = copy.deepcopy(self.app_state.get(app_name))
            user_before = copy.deepcopy(self.user_state.get(app_name))

            try:
                result = await super().append_event(session=session, event=event)
                serialized = self._serialized(canonical)
                with _LOCK:
                    self._persist(canonical, serialized=serialized)
                return result
            except Exception:
                self._restore_session(canonical, canonical_before)
                if session is not canonical:
                    self._restore_session(session, session_before)
                if app_existed:
                    self.app_state[app_name] = app_before
                else:
                    self.app_state.pop(app_name, None)
                if user_app_existed:
                    self.user_state[app_name] = user_before
                else:
                    self.user_state.pop(app_name, None)
                raise

    async def replace_session(self, session: Session) -> None:
        """Persist an intentional state/events replacement, e.g. rollback."""
        self._serialized(session)
        async with self._append_lock:
            canonical = self._canonical(session)
            before = copy.deepcopy(canonical)
            try:
                canonical.state = copy.deepcopy(session.state)
                canonical.events = copy.deepcopy(session.events)
                canonical.last_update_time = session.last_update_time
                # Serialize the canonical object too: callers can pass a stale
                # session with different identity fields, while only these
                # replacement fields are committed.
                serialized = self._serialized(canonical)
                with _LOCK:
                    self._persist(canonical, serialized=serialized)
            except Exception:
                self._restore_session(canonical, before)
                raise

    async def delete_session(self, **kwargs: Any) -> None:
        await super().delete_session(**kwargs)
        self._path(kwargs["app_name"], kwargs["user_id"], kwargs["session_id"]).unlink(
            missing_ok=True
        )


__all__ = ["DurableSessionService"]
