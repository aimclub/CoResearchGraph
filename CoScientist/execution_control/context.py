"""Context propagation and low-friction gates for model/tool call sites."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping, Optional

from .controller import AttemptReservation, RunHandle


_CURRENT_RUN: ContextVar[Optional[RunHandle]] = ContextVar(
    "coscientist_execution_run", default=None
)


def current_run() -> Optional[RunHandle]:
    """The run inherited by this task, including newly-created asyncio tasks."""
    return _CURRENT_RUN.get()


def require_bound_run() -> RunHandle:
    handle = current_run()
    if handle is None:
        raise RuntimeError("no RunHandle is bound to the current execution context")
    return handle


@contextmanager
def bind_run(handle: RunHandle) -> Iterator[RunHandle]:
    if not isinstance(handle, RunHandle):
        raise TypeError("bind_run expects a RunHandle")
    token = _CURRENT_RUN.set(handle)
    try:
        yield handle
    finally:
        _CURRENT_RUN.reset(token)


async def before_model_attempt(
    action: str,
    *,
    action_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[AttemptReservation]:
    """Gate and reserve an attempt, or no-op for an intentionally legacy caller."""
    handle = current_run()
    if handle is None:
        return None
    return await handle.before_model_attempt(
        action, action_id=action_id, metadata=metadata
    )


async def before_tool_action(
    action: str, *, metadata: Optional[Mapping[str, Any]] = None
) -> None:
    """Pause/stop gate for a tool boundary; tool calls do not consume LLM budget."""
    handle = current_run()
    if handle is None:
        return
    await handle.before_tool_action(action, metadata=metadata)


__all__ = [
    "before_model_attempt",
    "before_tool_action",
    "bind_run",
    "current_run",
    "require_bound_run",
]
