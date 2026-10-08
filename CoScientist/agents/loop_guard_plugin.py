"""RepeatCallGuardPlugin — stop an agent from repeating the identical tool call.

Prompt-level "don't thrash" guidance only binds the agent whose prompt carries
it. In practice any agent can fall into a tight loop: we watched a collector
issue the SAME web search 39 times in a row, burning the run's budget and
producing nothing. This guard is deterministic — it counts identical
(agent, tool, args) calls and, past a threshold, refuses with an instruction to
change approach instead of executing the call again.

It never blocks a *different* call, so legitimate polling of a job (check_job
with a new job_id) or paging through distinct queries is unaffected. Repeated
identical polling of the same job_id is allowed too — that is the sanctioned
way to wait — via the POLLING_TOOLS exemption.

Disable with REPEAT_CALL_GUARD=0; tune with REPEAT_CALL_LIMIT (default 4).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, Optional

from google.adk.plugins.base_plugin import BasePlugin

# Tools whose whole purpose is to be called again with the same arguments.
POLLING_TOOLS = {"check_job", "research_triggers", "get_active_tasks"}


def _enabled() -> bool:
    return os.getenv("REPEAT_CALL_GUARD", "1") not in ("0", "false", "False")


def _limit() -> int:
    try:
        return max(2, int(os.getenv("REPEAT_CALL_LIMIT", "4")))
    except ValueError:
        return 4


def _key(agent: str, tool: str, args: Any) -> str:
    try:
        blob = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        blob = str(args)
    return f"{agent}|{tool}|{blob}"


class RepeatCallGuardPlugin(BasePlugin):
    """Refuse the Nth identical tool call and tell the agent to change approach."""

    def __init__(self, name: str = "repeat_call_guard") -> None:
        super().__init__(name=name)
        self._counts: Counter = Counter()

    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> Optional[Dict[str, Any]]:
        if not _enabled():
            return None
        tool_name = str(getattr(tool, "name", "") or "")
        short = tool_name.rsplit("_", 1)[-1] if tool_name else ""
        if tool_name in POLLING_TOOLS or short in POLLING_TOOLS or any(
            tool_name.endswith(p) for p in POLLING_TOOLS
        ):
            return None

        agent = str(getattr(tool_context, "agent_name", "") or "?")
        key = _key(agent, tool_name, tool_args)
        self._counts[key] += 1
        n = self._counts[key]
        if n <= _limit():
            return None

        return {
            "status": "blocked",
            "blocked_by": "repeat_call_guard",
            "repeats": n,
            "message": (
                f"BLOCKED: you have called `{tool_name}` with these exact arguments "
                f"{n} times. Repeating it will not produce a different result. "
                "Change approach: use different arguments, a different tool, or "
                "state plainly what is blocking you and stop. If you are waiting "
                "for a long job, poll it with check_job instead."
            ),
        }


repeat_call_guard_plugin = RepeatCallGuardPlugin()
