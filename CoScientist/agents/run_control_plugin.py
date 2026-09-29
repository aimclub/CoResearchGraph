"""Cooperative pause boundaries and a durable dispatch journal.

The provider adapter owns the LLM counter.  This plugin never increments it:
it preserves the continuation state and prevents new tool work while paused.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any

from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types
from pydantic_core import to_jsonable_python

from CoScientist.execution_control import before_tool_action, current_run

logger = logging.getLogger(__name__)
RECOVERY_STATE_KEY = "_execution_recovery"
_READ_TOOLS = {"get_experiment_plan", "check_job", "get_active_tasks", "research_triggers"}
_LOCAL_CONTROL_TOOLS = {
    "get_experiment_plan", "start_task", "record_result", "retry_task",
    "fallback_task", "skip_task", "amend_task",
}
_HUMAN_CHANNEL_TOOLS = {"request_approval", "request_selection"}
def _state(context: Any) -> dict[str, Any]:
    state = context.state
    raw = state.to_dict() if hasattr(state, "to_dict") else dict(state)
    # These are invocation-local ADK values, not portable continuation data.
    return {key: value for key, value in raw.items() if not str(key).startswith("temp:")}


def save_continuation(context: Any, *, boundary: str, agent: str = "") -> None:
    handle = current_run()
    if handle is None:
        return
    try:
        saved = to_jsonable_python(_state(context))
    except Exception as exc:
        # An unknown object is not silently stringified into executable state.
        handle.controller.update_metadata(handle.run_id, {
            "continuation_error": f"{type(exc).__name__}: {exc}",
        })
        logger.warning("Cannot preserve run continuation: %s", exc)
        return
    handle.controller.update_metadata(handle.run_id, {
        "continuation": {"state": saved, "boundary": boundary, "agent": agent},
        "continuation_error": None,
    })


def action_key(tool: Any, tool_args: dict[str, Any], context: Any) -> str:
    state = _state(context)
    runtime = state.get("experiment_runtime") or {}
    payload = {
        "tool": str(getattr(tool, "name", "")),
        "agent": str(getattr(context, "agent_name", "")),
        "task_id": runtime.get("active_task_id"),
        "attempt_id": runtime.get("active_attempt_id"),
        "args": tool_args,
    }
    wire = json.dumps(to_jsonable_python(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()


def _latest_actions(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest = {}
    for entry in entries:
        if str(entry.get("event", "")).startswith("tool_") and entry.get("action_id"):
            latest[entry["action_id"]] = entry
    return latest


def unresolved_actions(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A delegation can resume from its child state; an opaque dispatch cannot."""
    return [
        row for row in _latest_actions(entries).values()
        if (row.get("event") in {"tool_dispatched", "tool_outcome_unknown"}
            or (row.get("data") or {}).get("outcome_unknown"))
        and not (row.get("data") or {}).get("delegation")
        and str((row.get("data") or {}).get("tool") or "") not in _HUMAN_CHANNEL_TOOLS
        and not _is_read_only_tool(str((row.get("data") or {}).get("tool") or ""))
    ]


def _exception_payload(value: Any) -> dict[str, str]:
    # Pydantic validation diagnostics can contain the original exception in
    # ctx.error. Preserve it as diagnostics, never as executable state.
    if isinstance(value, BaseException):
        return {"error_type": type(value).__name__, "message": str(value)}
    raise TypeError(f"Unsupported tool response type: {type(value).__name__}")


def _contains_exception(value: Any, seen: set[int] | None = None) -> bool:
    """Whether a response has an exception leaf ADK cannot serialise."""
    if isinstance(value, BaseException):
        return True
    if isinstance(value, (str, bytes, bytearray, int, float, bool, type(None))):
        return False
    seen = seen or set()
    marker = id(value)
    if marker in seen:
        return False
    seen.add(marker)
    if isinstance(value, Mapping):
        return any(
            _contains_exception(key, seen) or _contains_exception(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_exception(item, seen) for item in value)
    return False


def _uncertain_transport_error(error: BaseException) -> bool:
    """A raised opaque-tool error cannot prove the remote side did no work.

    Exception classes are not an acceptance protocol: a client may raise
    ``ValueError`` while decoding the response after the server accepted the
    request.  Read-only, local-control, human-channel, and delegation calls are
    excluded by the caller; every other raised error is conservative-unknown.
    """
    return isinstance(error, BaseException)


def _dispatch_for_call(entries: list[dict[str, Any]], call_id: Any) -> dict[str, Any] | None:
    if call_id is None:
        return None
    return next(
        (
            row for row in reversed(entries)
            if row.get("event") == "tool_dispatched"
            and (row.get("data") or {}).get("function_call_id") == call_id
        ),
        None,
    )


def _error_response(identity: str, error: BaseException) -> dict[str, Any]:
    return {
        "status": "error",
        "error_code": "TOOL_OUTCOME_UNKNOWN",
        "message": (
            "The remote operation may have been accepted before the connection "
            "failed. Inspect its outcome before authorizing a repeat."
        ),
        "error": {"error_type": type(error).__name__, "message": str(error)},
        "action_id": identity,
        "outcome_unknown": True,
        "retryable": False,
    }


def _is_read_only_tool(name: str) -> bool:
    if name in _READ_TOOLS:
        return True
    try:
        from CoScientist.hitl.work_order_risk import Tier, tool_tier

        return tool_tier(name) == Tier.READ
    except Exception:  # noqa: BLE001 - conservative when the registry is unavailable
        return False


class RunControlPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="run_control")

    async def before_agent_callback(self, *, agent, callback_context):
        handle = current_run()
        if handle is None:
            return None
        # Track the current linear stage without importing assembly at startup.
        from CoScientist.assembly.schema import get_config
        name = str(getattr(agent, "name", ""))
        try:
            stages = get_config().linear_stages()
            index = next((i for i, stage in enumerate(stages) if stage.get("agent") == name), None)
            if index is not None:
                handle.controller.update_metadata(handle.run_id, {"stage_index": index})
        except Exception:
            pass
        return None

    async def before_model_callback(self, *, callback_context, llm_request):
        await before_tool_action("model_boundary")
        from CoScientist.agents.loop_guard_plugin import apply_pending_control_resolution

        resolution = apply_pending_control_resolution(callback_context)
        # Snapshot after applying a human resolution, so restart cannot restore
        # the pre-resolution execution loop.
        save_continuation(callback_context, boundary="before_model",
                          agent=str(getattr(callback_context, "agent_name", "")))
        if resolution is not None:
            summary = str(resolution.get("summary") or "").strip()
            limited = [str(item) for item in resolution.get("limited_task_ids") or []]
            text = (
                "Operator closed unfinished tasks with documented limitations; "
                "proceed to result review."
            )
            if summary:
                text += f" {summary}"
            if limited:
                text += " Limited tasks: " + ", ".join(limited) + "."
            # This is a control transition, not a successful research result.
            # Return only factual resolution fields and never the helper's
            # internal status='success' marker.
            return LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text=text)]
                ),
                custom_metadata={
                    "control_resolution": "finish_with_limits",
                    "summary": summary,
                    "limited_task_ids": limited,
                },
            )
        return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        handle = current_run()
        if handle is None:
            return None
        await before_tool_action(str(getattr(tool, "name", "tool")))
        from CoScientist.agents.loop_guard_plugin import apply_pending_control_resolution

        resolution = apply_pending_control_resolution(tool_context)
        save_continuation(tool_context, boundary="before_tool",
                          agent=str(getattr(tool_context, "agent_name", "")))
        if resolution is not None:
            # The call was proposed from the pre-resolution execution state.
            # Do not dispatch it after the operator has ended that loop.
            return resolution
        identity = action_key(tool, tool_args, tool_context)
        name = str(getattr(tool, "name", ""))
        if _state(tool_context).get(RECOVERY_STATE_KEY) and not _is_read_only_tool(name):
            previous = _latest_actions(handle.controller.journal_entries(handle.run_id)).get(identity)
            if previous and previous.get("event") == "tool_completed":
                # Restore exactly the recorded response; do not execute again.
                return (previous.get("data") or {}).get("result")
        handle.controller.journal(handle.run_id, "tool_dispatched", action_id=identity, data={
            "tool": name,
            "agent": str(getattr(tool_context, "agent_name", "")),
            "delegation": hasattr(tool, "agent"),
            "function_call_id": getattr(tool_context, "function_call_id", None),
            "job_id": tool_args.get("job_id"),
        })
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        handle = current_run()
        if handle is None:
            return None
        # A terminal control tool may clear active IDs; use its call-id to find
        # the original action rather than hashing the now-changed runtime.
        call_id = getattr(tool_context, "function_call_id", None)
        rows = handle.controller.journal_entries(handle.run_id)
        identity = next((row["action_id"] for row in reversed(rows)
                         if row.get("event") == "tool_dispatched"
                         and call_id is not None
                         and (row.get("data") or {}).get("function_call_id") == call_id), None)
        identity = identity or action_key(tool, tool_args, tool_context)
        try:
            encoded = to_jsonable_python(result, fallback=_exception_payload)
        except (TypeError, ValueError) as exc:
            name = str(getattr(tool, "name", ""))
            handle.controller.journal(handle.run_id, "tool_outcome_unknown", action_id=identity,
                data={"tool": name, "error_type": type(exc).__name__, "message": str(exc)})
            handle.controller.request_pause(handle.run_id, "unknown_completion",
                pending_decision={"actions": [{"action_id": identity, "tool": name}],
                                  "reason": "tool_result_not_serializable"})
            save_continuation(tool_context, boundary="after_tool_unserializable")
            return {"status": "error", "error_code": "RESULT_SERIALIZATION_ERROR",
                    "message": "Tool may have executed; inspect its outcome before authorizing a repeat.",
                    "action_id": identity, "retryable": False}
        if isinstance(result, Mapping) and result.get("outcome_unknown") is True:
            # on_tool_error_callback already recorded the ambiguous outcome.
            # Returning None lets observers run, but deliberately does not
            # overwrite it with tool_completed (which would make it replayable).
            save_continuation(tool_context, boundary="after_tool_outcome_unknown",
                              agent=str(getattr(tool_context, "agent_name", "")))
            return None
        handle.controller.journal(handle.run_id, "tool_completed", action_id=identity, data={
            "tool": str(getattr(tool, "name", "")), "result": encoded,
        })
        save_continuation(tool_context, boundary="after_tool",
                          agent=str(getattr(tool_context, "agent_name", "")))
        # An exception leaf must not escape into ADK's strict event serializer.
        # Ordinary JSON coercions (datetime, enum, tuple, model) do NOT return
        # an override: ADK stops the whole plugin chain on any non-None value.
        return encoded if _contains_exception(result) else None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        handle = current_run()
        if handle is None:
            return None
        name = str(getattr(tool, "name", ""))
        rows = handle.controller.journal_entries(handle.run_id)
        dispatched = _dispatch_for_call(
            rows, getattr(tool_context, "function_call_id", None)
        )
        identity = (
            dispatched.get("action_id") if dispatched is not None
            else action_key(tool, tool_args, tool_context)
        )
        dispatch_data = (dispatched or {}).get("data") or {}
        delegation = bool(dispatch_data.get("delegation") or hasattr(tool, "agent"))
        uncertain = (
            not _is_read_only_tool(name)
            and name not in _LOCAL_CONTROL_TOOLS
            and name not in _HUMAN_CHANNEL_TOOLS
            and not delegation
            and _uncertain_transport_error(error)
        )
        if uncertain:
            handle.controller.journal(
                handle.run_id,
                "tool_outcome_unknown",
                action_id=identity,
                data={
                    "tool": name,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "outcome_unknown": True,
                    "delegation": False,
                },
            )
            unknown = unresolved_actions(
                handle.controller.journal_entries(handle.run_id)
            )
            handle.controller.request_pause(
                handle.run_id,
                "unknown_completion",
                pending_decision={
                    "actions": [
                        {
                            "action_id": row["action_id"],
                            "tool": (row.get("data") or {}).get("tool"),
                        }
                        for row in unknown
                    ],
                    "reason": "opaque_external_error_after_dispatch",
                },
            )
            save_continuation(tool_context, boundary="after_tool_outcome_unknown")
            # Keep the event serialisable and the agent from immediately
            # retrying.  The next model/tool boundary waits on the pause.
            return _error_response(identity, error)

        handle.controller.journal(
            handle.run_id,
            "tool_failed",
            action_id=identity,
            data={
                "tool": name,
                "error_type": type(error).__name__,
                "message": str(error),
                "delegation": delegation,
            },
        )
        save_continuation(tool_context, boundary="after_tool_error")
        return None
