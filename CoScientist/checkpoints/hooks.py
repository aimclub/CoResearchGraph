"""Explicit checkpoint hooks for the two moments no plugin callback can see.

``SessionAgent`` BUFFERS its final event while the human reviews the proposed
output (hitl/session_agent.py) — the event is not committed, so neither
``on_event_callback`` nor ``after_agent_callback`` fires. The call site in
``SessionAgent._run_async_impl`` invokes this hook right before the review,
passing the buffered event so capture can merge it (T0: "plan formed, review
not yet shown").
"""
from __future__ import annotations

import logging
from typing import Optional

from CoScientist.checkpoints.capture import capture_checkpoint
from CoScientist.checkpoints.model import HitlPending

logger = logging.getLogger(__name__)


def checkpoints_enabled() -> bool:
    try:
        from CoScientist.config import get_settings
        return bool(get_settings().checkpoints.enabled)
    except Exception:  # noqa: BLE001
        return False


async def save_pre_hitl_checkpoint(
    ctx,
    *,
    agent_name: str,
    final_event=None,
    output_text: Optional[str] = None,
) -> None:
    """T0 snapshot before a SessionAgent-style human review. Never raises."""
    if not checkpoints_enabled():
        return
    try:
        pending = HitlPending(
            agent=agent_name,
            kind="session_review",
            payload={"output": (str(output_text) if output_text is not None else "")[:20000]},
        )
        await capture_checkpoint(
            session=ctx.session,
            label="T0_before_hitl",
            reason="pre_approval",
            trigger_event=final_event,
            hitl_pending=pending,
        )
    except Exception:  # noqa: BLE001 — a checkpoint must never break the review loop
        logger.exception("pre-HITL checkpoint failed; run continues")
