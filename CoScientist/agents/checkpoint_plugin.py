"""Automatic checkpoints and deterministic fast-forward for linear pipelines.

The web application installs a sink with :func:`set_checkpoint_sink`.  Before
every configured linear stage starts, this plugin gives the sink a complete
copy of the ADK session state.  The observer is deliberately optional: CLI and
A2A runners still get the inexpensive resume gate without depending on Web UI
code or storage.

Restoring a checkpoint writes ``CHECKPOINT_RESUME_STATE_KEY`` into session
state.  During the next invocation stages before the target return a synthetic
successful result, while the target and all later stages execute normally.
This makes resume behaviour a runtime invariant instead of a prompt hint.
"""
from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from CoScientist.graph.session_scope import SessionKey, session_key

logger = logging.getLogger("CoScientist.agents.checkpoint_plugin")

CHECKPOINT_RESUME_STATE_KEY = "_checkpoint_resume"

CheckpointSink = Callable[[SessionKey, dict[str, Any]], Awaitable[None]]
_sink: Optional[CheckpointSink] = None


def set_checkpoint_sink(sink: Optional[CheckpointSink]) -> None:
    """Register the process-wide observer used by the Web runtime."""
    global _sink
    _sink = sink


def _linear_stages() -> list[dict[str, Any]]:
    try:
        from CoScientist.assembly.schema import get_config

        return get_config().linear_stages()
    except Exception as exc:  # noqa: BLE001 - checkpointing must not break a run
        logger.warning("Pipeline stages unavailable: %s", exc)
        return []


class CheckpointPlugin(BasePlugin):
    """Snapshot stage boundaries and skip restored prefix stages."""

    def __init__(self, name: str = "pipeline_checkpoints") -> None:
        super().__init__(name=name)

    async def before_agent_callback(self, *, agent, callback_context):
        stages = _linear_stages()
        if not stages:
            return None

        author = getattr(agent, "name", None)
        stage_index = next(
            (i for i, stage in enumerate(stages) if stage.get("agent") == author),
            None,
        )
        if stage_index is None:
            return None

        state = callback_context.state
        resume = state.get(CHECKPOINT_RESUME_STATE_KEY)
        if isinstance(resume, dict):
            try:
                target_index = int(resume.get("stage_index", -1))
            except (TypeError, ValueError):
                target_index = -1

            if 0 <= stage_index < target_index:
                # Returning Content from before_agent is ADK's supported way to
                # bypass an agent while letting its parent workflow continue.
                return types.Content(
                    role="model",
                    parts=[types.Part(text=(
                        f"Stage {stage_index + 1} ({author}) was restored from "
                        "the checkpoint and was not executed again."
                    ))],
                )

            if stage_index == target_index:
                # Clear before the real target executes.  State has no delete
                # operation because deltas are append-only, so None is the
                # canonical inactive value.
                state[CHECKPOINT_RESUME_STATE_KEY] = None

        sink = _sink
        if sink is None:
            return None

        try:
            key = session_key(callback_context)
            snapshot = copy.deepcopy(state.to_dict())
            snapshot.pop(CHECKPOINT_RESUME_STATE_KEY, None)
            # Invocation-scoped ADK keys must never leak into a new invocation.
            snapshot = {
                k: v for k, v in snapshot.items()
                if not str(k).startswith("temp:")
            }
            await sink(key, {
                "stage_index": stage_index,
                "stage_count": len(stages),
                "agent": author,
                "title": stages[stage_index].get("title") or author,
                "created_at": datetime.now().isoformat(),
                "state": snapshot,
            })
        except Exception as exc:  # noqa: BLE001 - observer must never fail a run
            logger.warning("Could not create pipeline checkpoint: %s", exc)
        return None


__all__ = [
    "CHECKPOINT_RESUME_STATE_KEY",
    "CheckpointPlugin",
    "set_checkpoint_sink",
]
