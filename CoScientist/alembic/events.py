"""Optional live-event bus for the web dashboard.

The pipeline and the agent runtime call ``await emit({...})`` at every
meaningful boundary — stage start/stop, each agent tool call/result, per-tool
validation. By default there is **no sink**, so ``emit`` is a cheap no-op: the
CLI and the benchmark runner pay nothing. The web app (``alembic.web.app``)
installs a sink for the duration of one pipeline run to stream those events to
the browser.

The sink lives in a :class:`~contextvars.ContextVar` so two concurrent runs
(e.g. two browser tabs) never clobber each other's stream — a run's child tasks
inherit whichever sink was active in the context that created them.
"""
from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from typing import Awaitable, Callable, Optional

from loguru import logger

# An async callback that receives one JSON-serialisable event dict.
Sink = Callable[[dict], Awaitable[None]]

_sink: ContextVar[Optional[Sink]] = ContextVar("alembic_event_sink", default=None)


def set_sink(sink: Optional[Sink]):
    """Install (``sink``) or clear (``None``) the event sink for this context.
    Returns the token the caller passes to :func:`reset_sink` when the run ends."""
    return _sink.set(sink)


def reset_sink(token) -> None:
    _sink.reset(token)


def has_sink() -> bool:
    return _sink.get() is not None


def safe(value):
    """A deeply JSON-serialisable copy of ``value`` (unserialisable → ``str``)."""
    try:
        return json.loads(json.dumps(value, default=str, ensure_ascii=False))
    except (TypeError, ValueError):
        return str(value)


async def emit(msg: dict) -> None:
    """Push one event to the active sink; a no-op when none is installed.

    A sink may raise :class:`asyncio.CancelledError` to unwind a superseded run
    (the web app does this the moment a new run replaces an in-flight one) — that
    propagates so the pipeline stops at the next boundary. Any *other* sink
    exception is swallowed, so a UI hiccup can never crash the pipeline.
    """
    sink = _sink.get()
    if sink is None:
        return
    try:
        await sink(msg)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — a UI sink must never kill the run
        logger.debug(f"[events] sink raised, ignoring: {exc}")
