"""Resolve a stable graph namespace from an ADK user/session context."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

SessionKey = tuple[str, str]
DEFAULT_SESSION_KEY: SessionKey = ("local", "default")
GRAPH_SCOPE_USER_KEY = "graph_scope_user_id"
GRAPH_SCOPE_SESSION_KEY = "graph_scope_session_id"


def session_key(
    context: Any = None,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> SessionKey:
    """Return ``(user_id, session_id)`` for Tool/Callback/Invocation contexts.

    Direct CLI helpers and legacy unit tests may not provide an ADK context; in
    that case they share a deliberately named local default namespace.
    """
    invocation = (
        getattr(context, "_invocation_context", None)
        or getattr(context, "invocation_context", None)
    )
    session = getattr(context, "session", None)
    if session is None:
        session = getattr(invocation, "session", None)

    # AgentTool runs a delegated agent in a fresh, randomly named child ADK
    # session. Its state is copied from the parent, however, so pin the public
    # Web-session scope there and prefer it over the transient child id.
    state = getattr(context, "state", None)
    if state is None:
        state = getattr(session, "state", None)
    try:
        scoped_user = state.get(GRAPH_SCOPE_USER_KEY) if state is not None else None
        scoped_session = (
            state.get(GRAPH_SCOPE_SESSION_KEY) if state is not None else None
        )
    except Exception:  # noqa: BLE001 - context state implementations vary
        scoped_user = scoped_session = None

    resolved_user = user_id or scoped_user or getattr(session, "user_id", None)
    resolved_session = (
        session_id or scoped_session or getattr(session, "id", None)
    )
    if not resolved_user or not resolved_session:
        return DEFAULT_SESSION_KEY

    resolved = str(resolved_user), str(resolved_session)
    if state is not None and not (scoped_user and scoped_session):
        try:
            state[GRAPH_SCOPE_USER_KEY] = resolved[0]
            state[GRAPH_SCOPE_SESSION_KEY] = resolved[1]
        except Exception:  # noqa: BLE001 - read-only contexts still resolve
            pass
    return resolved


def safe_component(value: str) -> str:
    """Make an identifier safe for a path component and graph run id."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned[:96] or "unknown"


def namespace(key: SessionKey) -> str:
    return f"{safe_component(key[0])}__{safe_component(key[1])}"


def storage_dir(base: str | Path, key: SessionKey) -> Path:
    """Directory containing all persistent graph state for one session."""
    return Path(base) / "sessions" / safe_component(key[0]) / safe_component(key[1])
