"""The web ТЗ panel's feed: live ТЗ snapshots and the stored ТЗ.

The panel is part of the core UI; what a ТЗ looks like belongs to the profile
that builds one. A profile registers ``set_tz_state_view`` — how to read its
ТЗ out of the session state — and pushes snapshots with ``publish_tz`` while
the ТЗ is being built. With neither, the panel simply stays empty.

In the microfluidics profile TZSpecAgent runs inside an AgentTool, i.e. in a nested Runner with its own
child session — the web session's state does not see the ТЗ until the whole
module returns. So the agent PUSHES a snapshot of the ТЗ after every change
(each section saved, each operator round, each agent fill, the approval),
the same way tool activity and metrics reach the browser.

The web UI registers a sink with ``set_tz_sink``; it is called as
``await sink(session_key, payload)``. With no sink the publishing is inert,
and a failing sink never breaks a run.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Mapping, Optional

from CoScientist.graph.session_scope import SessionKey, session_key

logger = logging.getLogger(__name__)

TZSink = Callable[[SessionKey, dict], Awaitable[None]]

# session state -> the panel's view of the stored ТЗ ({"sections": ...}), or None
TZStateView = Callable[[Mapping[str, Any]], Optional[dict]]

_sink: Optional[TZSink] = None
_state_view: Optional[TZStateView] = None


def set_tz_sink(sink: Optional[TZSink]) -> None:
    """Register (or clear, with ``None``) the observer for ТЗ snapshots."""
    global _sink
    _sink = sink


async def publish_tz(context: Any, payload: dict) -> None:
    """Send one ТЗ snapshot to the tabs watching the context's session."""
    sink = _sink
    if sink is None:
        return
    try:
        key = session_key(context)
    except Exception:  # noqa: BLE001 — context shapes vary across ADK paths
        return
    payload.setdefault("timestamp", datetime.now().isoformat())
    try:
        await sink(key, payload)
    except Exception as exc:  # noqa: BLE001 — an observer must not fail a run
        logger.warning("ТЗ snapshot sink failed: %s", exc)


def set_tz_state_view(view: Optional[TZStateView]) -> None:
    """Register (or clear) how the active profile reads its ТЗ from state."""
    global _state_view
    _state_view = view


def stored_tz_view(state: Mapping[str, Any]) -> Optional[dict]:
    """The ТЗ kept in a session's state, as the panel renders it; None when
    there is none or no profile registered a view."""
    view = _state_view
    if view is None:
        return None
    try:
        return view(state)
    except Exception as exc:  # noqa: BLE001 — a bad ТЗ must not break the page
        logger.warning("stored ТЗ view failed: %s", exc)
        return None


__all__ = [
    "publish_tz",
    "set_tz_sink",
    "set_tz_state_view",
    "stored_tz_view",
]
