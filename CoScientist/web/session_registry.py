"""User and session catalogue for the development Web UI.

The ADK session service owns agent state and event history.  This registry only
adds the small amount of UI metadata ADK does not provide (nickname, title,
last selected session) — and persists it to disk (see web/session_store.py), so
past conversations can still be opened after a restart instead of vanishing
while their graphs and artifacts stay on disk.
"""

from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from typing import Any, Optional
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalSessionRegistry:
    """Thread-safe, in-memory catalogue of local users and their sessions."""

    def __init__(self, *, persist: bool = True) -> None:
        self._users: dict[str, dict[str, Any]] = {}
        self._nickname_index: dict[str, str] = {}
        self._sessions: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = RLock()
        self._persist_enabled = persist
        if persist:
            self._restore()

    # ── persistence ──────────────────────────────────────────────────────────
    def _restore(self) -> None:
        """Load the catalogue written by a previous process (best-effort)."""
        try:
            from CoScientist.web.session_store import load_registry
            data = load_registry()
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            for user in data.get("users", []):
                uid = user.get("id")
                nick = user.get("nickname")
                if not uid or not nick:
                    continue
                self._users[uid] = dict(user)
                self._nickname_index[str(nick).casefold()] = uid
            for session in data.get("sessions", []):
                uid, sid = session.get("user_id"), session.get("id")
                if uid in self._users and sid:
                    # A session cannot still be running in a fresh process.
                    session = dict(session)
                    if session.get("status") == "processing":
                        session["status"] = "idle"
                    self._sessions[(uid, sid)] = session

    def _save(self) -> None:
        if not self._persist_enabled:
            return
        try:
            from CoScientist.web.session_store import save_registry
            save_registry(self._users, self._sessions)
        except Exception:  # noqa: BLE001
            pass

    def list_users(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(
                (dict(user) for user in self._users.values()),
                key=lambda user: user["nickname"].casefold(),
            )

    def create_user(self, nickname: str) -> dict[str, Any]:
        if not isinstance(nickname, str):
            raise ValueError("Nickname must be a string.")
        nickname = " ".join(nickname.strip().split())
        if not nickname:
            raise ValueError("Nickname must not be empty.")
        if len(nickname) > 64:
            raise ValueError("Nickname must be at most 64 characters.")

        nickname_key = nickname.casefold()
        with self._lock:
            if nickname_key in self._nickname_index:
                raise ValueError(f"Nickname '{nickname}' is already registered.")

            user_id = f"user_{uuid4().hex}"
            user = {
                "id": user_id,
                "nickname": nickname,
                "created_at": _now(),
                "last_session_id": None,
            }
            self._users[user_id] = user
            self._nickname_index[nickname_key] = user_id
            self._save()
            return dict(user)

    def get_user(self, user_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            user = self._users.get(user_id)
            return dict(user) if user else None

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        self.require_user(user_id)
        with self._lock:
            sessions = [
                dict(session)
                for (owner_id, _), session in self._sessions.items()
                if owner_id == user_id
            ]
        return sorted(sessions, key=lambda session: session["updated_at"], reverse=True)

    def create_session(
        self,
        user_id: str,
        title: str = "",
        *,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        self.require_user(user_id)
        session_id = session_id or f"session_{uuid4().hex}"
        if not isinstance(title, str):
            raise ValueError("Session title must be a string.")
        title = " ".join(title.strip().split()) or "New session"
        if len(title) > 120:
            raise ValueError("Session title must be at most 120 characters.")

        with self._lock:
            key = (user_id, session_id)
            if key in self._sessions:
                raise ValueError(f"Session '{session_id}' already exists.")
            timestamp = _now()
            session = {
                "id": session_id,
                "user_id": user_id,
                "title": title,
                "status": "idle",
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            self._sessions[key] = session
            self._users[user_id]["last_session_id"] = session_id
            self._save()
            return dict(session)

    def get_session(self, user_id: str, session_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            session = self._sessions.get((user_id, session_id))
            return dict(session) if session else None

    def touch_session(self, user_id: str, session_id: str, *, status: Optional[str] = None) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get((user_id, session_id))
            if not session:
                raise KeyError(f"Unknown session '{session_id}' for user '{user_id}'.")
            session["updated_at"] = _now()
            if status is not None:
                session["status"] = status
            self._users[user_id]["last_session_id"] = session_id
            self._save()
            return dict(session)

    def rename_session(self, user_id: str, session_id: str, title: str) -> dict[str, Any]:
        if not isinstance(title, str):
            raise ValueError("Session title must be a string.")
        title = " ".join(title.strip().split())
        if not title:
            raise ValueError("Session title must not be empty.")
        if len(title) > 120:
            raise ValueError("Session title must be at most 120 characters.")
        with self._lock:
            session = self._sessions.get((user_id, session_id))
            if not session:
                raise KeyError(f"Unknown session '{session_id}' for user '{user_id}'.")
            session["title"] = title
            session["updated_at"] = _now()
            self._users[user_id]["last_session_id"] = session_id
            self._save()
            return dict(session)

    def require_user(self, user_id: str) -> dict[str, Any]:
        user = self.get_user(user_id)
        if not user:
            raise KeyError(f"Unknown user '{user_id}'.")
        return user

    def require_session(self, user_id: str, session_id: str) -> dict[str, Any]:
        self.require_user(user_id)
        session = self.get_session(user_id, session_id)
        if not session:
            raise KeyError(f"Unknown session '{session_id}' for user '{user_id}'.")
        return session

    def ensure_user(self, nickname: str) -> dict[str, Any]:
        """Find an existing user by nickname (case-insensitive) or create one.

        Used by session import to guarantee the target user exists without
        raising on duplicates.
        """
        if not isinstance(nickname, str):
            raise ValueError("Nickname must be a string.")
        nickname = " ".join(nickname.strip().split())
        if not nickname:
            raise ValueError("Nickname must not be empty.")

        nickname_key = nickname.casefold()
        with self._lock:
            existing_id = self._nickname_index.get(nickname_key)
            if existing_id is not None:
                return dict(self._users[existing_id])
            # Create a new user
            user_id = f"user_{uuid4().hex}"
            user = {
                "id": user_id,
                "nickname": nickname,
                "created_at": _now(),
                "last_session_id": None,
            }
            self._users[user_id] = user
            self._nickname_index[nickname_key] = user_id
            return dict(user)

    def import_session(
        self,
        user_id: str,
        session_id: str,
        title: str = "Imported session",
        *,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> dict[str, Any]:
        """Register a session with explicit id and preserved metadata.

        Unlike ``create_session``, this accepts a pre-generated ``session_id``
        and optional original timestamps — used when restoring a bundle so the
        UI shows when the session was originally created.
        """
        self.require_user(user_id)
        title = " ".join((title or "Imported session").strip().split())[:120] or "Imported session"

        with self._lock:
            key = (user_id, session_id)
            if key in self._sessions:
                raise ValueError(f"Session '{session_id}' already exists.")
            now = _now()
            session = {
                "id": session_id,
                "user_id": user_id,
                "title": title,
                "status": "idle",
                "created_at": created_at or now,
                "updated_at": updated_at or now,
            }
            self._sessions[key] = session
            self._users[user_id]["last_session_id"] = session_id
            return dict(session)
