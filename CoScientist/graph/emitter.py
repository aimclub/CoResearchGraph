"""ADK plugin that materialises the execution graph from agent events.

Attached to every A2A server's Runner (next to the event logger). Each tool call
becomes a node; sub-agent delegations are `agent_call` nodes, ordinary tools are
`tool_call` nodes. All best-effort — if the graph service is down nothing breaks.

run_id: a shared run is identified by a `run_id` propagated through the A2A
delegation (see run_id propagation). When none is present (direct call) we fall
back to the ADK invocation_id, which is stable for one orchestration request.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from google.adk.plugins.base_plugin import BasePlugin

from CoScientist.graph import client
from CoScientist.utils.s3_refs import find_s3_uris

# A tool call whose name is one of our delegatable agents is a delegation, not
# a leaf tool. The roster comes from system.yaml (every agent that appears as a
# subordinate), resolved lazily and cached so a config problem never breaks
# graph emission.
_agent_names_cache: Optional[set] = None


def _agent_names() -> set:
    global _agent_names_cache
    if _agent_names_cache is None:
        try:
            from CoScientist.assembly.schema import get_config
            _agent_names_cache = get_config().delegatable_names()
        except Exception:  # noqa: BLE001 — best-effort, like the rest of the module
            _agent_names_cache = set()
    return _agent_names_cache

_RUN_ID_STATE_KEY = "deg_run_id"


def _short(value: Any, limit: int = 800) -> str:
    """Trim a value for display. See the note on the same helper in plugin.py:
    the old 300-character cut removed the file references at the end of a result.
    The references now travel in their own fields, so this stays modest."""
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


def _run_id(tool_context: Any) -> str:
    """Shared run id: the propagated DEG run id if set, else the invocation id."""
    try:
        state = getattr(tool_context, "state", None)
        if state is not None and state.get(_RUN_ID_STATE_KEY):
            return str(state[_RUN_ID_STATE_KEY])
    except Exception:  # noqa: BLE001
        pass
    inv = getattr(tool_context, "invocation_id", None)
    return f"run_{inv}" if inv else "run_default"


def _agent(tool_context: Any) -> str:
    return getattr(tool_context, "agent_name", None) or "agent"


def _is_error(result: Any) -> bool:
    if isinstance(result, dict):
        if result.get("status") in ("error", "failed", "timeout"):
            return True
        if result.get("error"):
            return True
    return False


class GraphEmitterPlugin(BasePlugin):
    def __init__(self, name: str = "graph_emitter") -> None:
        super().__init__(name=name)

    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> Optional[dict]:
        run_id = _run_id(tool_context)
        agent = _agent(tool_context)
        root = f"{run_id}:agent:{agent}"
        # The agent's own root node (idempotent upsert).
        await client.emit_node(
            run_id=run_id, node_id=root, kind="agent_call",
            label=agent, executor_agent=agent, status="running", t_start=time.time(),
        )
        fcid = getattr(tool_context, "function_call_id", None) or tool.name
        nid = f"{run_id}:{fcid}"
        is_delegation = tool.name in _agent_names()
        await client.emit_node(
            run_id=run_id, node_id=nid,
            kind="agent_call" if is_delegation else "tool_call",
            label=f"{tool.name}: {_short(tool_args, 200)}",
            executor_agent=tool.name if is_delegation else agent,
            status="running", parent_ids=[root], input=_short(tool_args),
            input_files=find_s3_uris(tool_args), t_start=time.time(),
        )
        await client.emit_edge(run_id=run_id, src=root, dst=nid,
                               type="delegated_to" if is_delegation else "caused_by")
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result) -> Optional[dict]:
        run_id = _run_id(tool_context)
        fcid = getattr(tool_context, "function_call_id", None) or tool.name
        await client.set_status(
            run_id=run_id, node_id=f"{run_id}:{fcid}",
            status="failed" if _is_error(result) else "success",
            output=_short(result, 1500), output_files=find_s3_uris(result),
            t_end=time.time(),
        )
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error) -> Optional[dict]:
        run_id = _run_id(tool_context)
        fcid = getattr(tool_context, "function_call_id", None) or tool.name
        await client.set_status(
            run_id=run_id, node_id=f"{run_id}:{fcid}",
            status="failed", output=_short(str(error), 300), t_end=time.time(),
        )
        return None
