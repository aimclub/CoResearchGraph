"""Explicit sleep tool for agents watching a long-running external job.

Gives a ReAct agent a way to deliberately pause between status/log checks on
a job that runs for a long time (e.g. an MCP-server experiment), instead of
re-checking immediately every turn and burning LLM tokens on tight polling.
"""
from __future__ import annotations

import asyncio

MAX_SLEEP_MINUTES = 10.0


async def sleep_tool(minutes: float) -> dict:
    """Pause before your next tool call, instead of checking again immediately.

    Use this after starting or checking a long-running job (e.g. one that runs
    for hours) to space out your status/log checks. Values above the maximum
    are silently capped — call it again afterwards if you need to wait longer.

    Args:
        minutes: How long to sleep, in minutes. Capped at 10.

    Returns:
        A dictionary with the actual number of minutes slept.
    """
    clamped = max(0.0, min(float(minutes), MAX_SLEEP_MINUTES))
    await asyncio.sleep(clamped * 60)
    return {"slept_minutes": clamped}
