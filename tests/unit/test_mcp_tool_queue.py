"""The MCP limiter spaces call starts without serializing call execution."""
from __future__ import annotations

import asyncio
import time

from CoScientist.tools.mcp_tool_queue import MCPToolStartLimiter


def test_calls_start_at_interval_but_run_concurrently():
    async def scenario():
        limiter = MCPToolStartLimiter(interval_seconds=0.03)
        started: list[float] = []
        finished: list[float] = []

        async def operation():
            started.append(time.monotonic())
            await asyncio.sleep(0.12)
            finished.append(time.monotonic())

        await asyncio.gather(
            limiter.run(operation),
            limiter.run(operation),
            limiter.run(operation),
        )
        return started, finished

    started, finished = asyncio.run(scenario())

    assert [b - a for a, b in zip(started, started[1:])] >= [0.02, 0.02]
    # The first call is still running when the second one starts.
    assert started[1] < finished[0]
