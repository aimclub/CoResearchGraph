"""Bound exact repeats and semantic experiment control loops.

The generic guard catches identical calls. Experiment control additionally
uses state progress and normalized failures, so changing only a reason or call
id cannot reset the loop. Read-only plan observations and sanctioned polling
are never blocked.

Neither is the coder, and for a reason that does not apply to a search: the
guard keys on (agent, tool, args), and for a tool that acts on a workspace the
arguments are not the whole input — the filesystem is. Running the same test
command after editing a file, re-reading a path a job is still writing, listing
a directory until the artifact lands: the same arguments, a different world
each time, and a genuinely different answer. Blocking those told an agent to
change approach at the point where repeating the call WAS the approach.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from google.adk.plugins.base_plugin import BasePlugin

POLLING_TOOLS = {"check_job", "check_sandbox_task", "check_mcp_build",
                 "research_triggers", "get_active_tasks"}
#: Everything the coder works through — the local/remote execution toolset and
#: the sandbox one. Exempt wholesale: their results depend on a workspace that
#: changes under them, so an identical call is not a repeated question. A
#: toolset hands its tools over as `{prefix}_{name}`, which the suffix match
#: below accounts for.
CODER_TOOLS = {
    "execute_bash", "read_file", "write_file", "list_directory",
    "install_package",
    "run_sandbox_task", "list_sandbox_files", "fetch_sandbox_artifact",
}
#: What the guard lets through without counting.
EXEMPT_TOOLS = POLLING_TOOLS | CODER_TOOLS
READ_ONLY_TOOLS = {"get_experiment_plan"}
CONTROL_TOOLS = {
    "start_task", "record_result", "retry_task", "fallback_task", "skip_task",
    "amend_task", *READ_ONLY_TOOLS,
}
_SEMANTIC_STATE_KEY = "_experiment_semantic_loop_guard"
_SEMANTIC_PAUSE_KEY = "experiment_control_guard_pause"
_VOLATILE_ARGUMENTS = frozenset({
    "reason", "call_id", "function_call_id", "request_id", "trace_id", "timestamp",
})


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


def _semantic_args(args: Any) -> Any:
    """Drop cosmetic fields that must not reset a control-loop counter."""
    if not isinstance(args, dict):
        return args
    return {
        key: _semantic_args(value)
        for key, value in args.items()
        if str(key).lower() not in _VOLATILE_ARGUMENTS
    }


def _state_revision(state: Any) -> str:
    """Use a published revision when present; otherwise fingerprint control state."""
    runtime = state.get("experiment_runtime") if state else None
    if not isinstance(runtime, dict):
        return "no-runtime"
    try:
        from CoScientist.experiments.runtime.state_machine import experiment_state_revision

        return experiment_state_revision(state)
    except Exception:  # noqa: BLE001 - keep the guard usable during early startup
        pass
    if runtime.get("state_revision") is not None:
        return str(runtime["state_revision"])
    tasks = runtime.get("tasks") or {}
    rows = []
    for task_id in runtime.get("task_order") or []:
        task = tasks.get(task_id) or {}
        rows.append((
            str(task_id), str(task.get("status") or ""),
            str(task.get("current_route") or ""), len(task.get("attempt_order") or []),
        ))
    return json.dumps({
        "phase": runtime.get("phase"),
        "active_task_id": runtime.get("active_task_id"),
        "active_attempt_id": runtime.get("active_attempt_id"),
        "tasks": rows,
    }, sort_keys=True, default=str)


def _semantic_limits() -> tuple[int, int]:
    """Mirror ExperimentsSettings without initializing the assembled agent tree."""
    try:
        same = int(os.getenv("EXPERIMENTS__CONTROL_SAME_FAILURE_LIMIT", "3"))
    except ValueError:
        same = 3
    try:
        progress = int(os.getenv("EXPERIMENTS__CONTROL_NO_PROGRESS_LIMIT", "5"))
    except ValueError:
        progress = 5
    return max(1, same), max(1, progress)


def _recovery(cause: str, state: Any) -> Dict[str, Any]:
    runtime = state.get("experiment_runtime") if state else None
    phase = str((runtime or {}).get("phase") or "") if isinstance(runtime, dict) else ""
    if phase == "reporting":
        next_actions = [{"action": "continue_to_result_review"}]
    else:
        next_actions = [
            {"action": "stop_automatic_retries", "reason": cause},
            {"action": "request_human_decision", "options": ["accept_limit", "revise_task"]},
        ]
    return {
        "status": "blocked",
        "blocked_by": "semantic_control_loop_guard",
        "error_code": cause,
        "terminal_control_guard": True,
        "run_pause_cause": "execution_error",
        "resumable": True,
        "next_actions": next_actions,
        "message": (
            "Automatic control stopped because the same state is not making progress. "
            "The current result/blocker remains recorded; choose an explicit recovery action."
        ),
    }


def _request_run_pause(cause: str, state: Any, tool_context: Any) -> bool:
    """Stop the actual run at its next cooperative boundary.

    Returning another ordinary tool error is not terminal: an LLM can spend its
    whole budget repeating it.  The durable run controller is the authoritative
    stop mechanism.  ``execution_error`` is intentionally resumable by the web
    control plane, while the pending decision explains that a plain repeat will
    immediately pause again unless the task/state is revised or partial results
    are accepted.
    """
    revision = _state_revision(state)
    decision = {
        "kind": "semantic_control_loop_guard",
        "error_code": cause,
        "state_revision": revision,
        "message": (
            "Automatic experiment control made no progress. Before resuming, "
            "revise/skip the blocked task or accept the recorded partial result."
        ),
        "actions": [
            {"action": "revise_or_skip_blocked_task"},
            {"action": "accept_partial_result_and_report"},
        ],
    }
    state[_SEMANTIC_PAUSE_KEY] = dict(decision)
    try:
        from CoScientist.execution_control import current_run

        handle = current_run()
        if handle is None:
            return False
        handle.controller.request_pause(
            handle.run_id, "execution_error", pending_decision=decision,
        )
        # The run-control plugin may have snapshotted just before this plugin.
        # Refresh the continuation so a restart cannot lose the terminal guard.
        from CoScientist.agents.run_control_plugin import save_continuation

        save_continuation(
            tool_context,
            boundary="semantic_control_loop_guard",
            agent=str(getattr(tool_context, "agent_name", "") or ""),
        )
        return True
    except Exception:  # noqa: BLE001 - guard must still return its blocker
        return False


def finish_control_stagnation(state: Any) -> Dict[str, Any]:
    """Finish an explicitly operator-accepted stagnation with honest limits.

    This does not infer success and does not retry anything.  Existing terminal
    tasks/results are preserved.  Only unfinished tasks are closed (optional
    tasks as skipped, required tasks as failed), an open attempt is marked
    failed without fabricating a TaskResult, and the runtime advances to normal
    result review.  Callers must gate this helper on an explicit operator
    decision; :func:`apply_pending_control_resolution` is that gate.
    """
    runtime = state.get("experiment_runtime") if state is not None else None
    if not isinstance(runtime, dict):
        return {
            "status": "error", "error_code": "experiment_runtime_missing",
            "message": "No experiment runtime can be finished with limits.",
        }
    already = state.get("experiment_control_resolution_done")
    if isinstance(already, dict) and runtime.get("phase") == "reporting":
        return {"status": "success", "already_applied": True, **already}

    now = datetime.now(timezone.utc).isoformat()
    reason = "Automatic control stopped after repeated no-progress transitions."
    terminal = {"done", "done_with_warnings", "failed", "skipped", "blocked"}
    preserved: list[str] = []
    limited: list[str] = []
    tasks = runtime.get("tasks") or {}
    for task_id in runtime.get("task_order") or list(tasks):
        task = tasks.get(task_id)
        if not isinstance(task, dict):
            continue
        if str(task.get("status") or "") in terminal:
            preserved.append(str(task_id))
            continue
        optional = bool((task.get("task") or {}).get("optional"))
        task["status"] = "skipped" if optional else "failed"
        task["last_message"] = reason
        task["blocked_reason"] = {
            "code": "control_stagnation_limit",
            "operator_resolution": "finish_with_limits",
        }
        limited.append(str(task_id))

    active_task_id = str(runtime.get("active_task_id") or "")
    active_attempt_id = str(runtime.get("active_attempt_id") or "")
    active_task = tasks.get(active_task_id) if active_task_id else None
    active_attempt = (
        (active_task.get("attempts") or {}).get(active_attempt_id)
        if isinstance(active_task, dict) and active_attempt_id else None
    )
    if isinstance(active_attempt, dict) and not active_attempt.get("result_id"):
        active_attempt["status"] = "failure"
        active_attempt["error_code"] = "control_stagnation_limit"
        active_attempt["error_message"] = reason
        active_attempt["retryable"] = False
        active_attempt["finished_at"] = now

    runtime["active_task_id"] = None
    runtime["active_attempt_id"] = None
    runtime["phase"] = "reporting"
    runtime["completion_reason"] = "control_stagnation_limit"
    resolution = {
        "resolution": "finish_with_limits",
        "finished_at": now,
        "limited_task_ids": limited,
        "preserved_terminal_task_ids": preserved,
        "summary": reason,
    }
    runtime.setdefault("control_resolutions", []).append(dict(resolution))
    # google.adk State tracks top-level assignments as deltas. Mutating the
    # nested dict returned by get() is not enough for that change to cross an
    # AgentTool/session boundary, so publish the complete updated runtime.
    state["experiment_runtime"] = runtime
    state["experiment_control_resolution_done"] = dict(resolution)
    state["experiment_execution_summary"] = (
        reason + (f" Limited tasks: {', '.join(limited)}." if limited else "")
    )
    for key in (
        "experiment_active_envelope", "filtered_tools", "deployed_mcps",
        "upstream_artifact_inputs", "experiment_last_route_response",
        "experiment_last_route_observation",
    ):
        state[key] = None
    semantic = dict(state.get(_SEMANTIC_STATE_KEY) or {})
    semantic.update({
        "blocked": None,
        "resolved": "finish_with_limits",
        "resolved_at": now,
        "no_progress": 0,
        "same_failure": 0,
    })
    state[_SEMANTIC_STATE_KEY] = semantic
    state[_SEMANTIC_PAUSE_KEY] = None
    return {"status": "success", **resolution}


def apply_pending_control_resolution(context: Any) -> Optional[Dict[str, Any]]:
    """Consume the durable operator decision at a cooperative run boundary."""
    from CoScientist.execution_control import current_run

    handle = current_run()
    state = getattr(context, "state", None)
    if handle is None or state is None:
        return None
    status = handle.status()
    requested = status.metadata.get("control_resolution")
    if not (
        isinstance(requested, dict)
        and requested.get("decision") == "finish_with_limits"
        and requested.get("source") == "human"
    ):
        return None
    result = finish_control_stagnation(state)
    done = {
        "resolution": "finish_with_limits",
        "source": "human",
        "reason": requested.get("reason"),
        "state_revision": _state_revision(state),
        "result": dict(result),
    }
    # Persist the resolved runtime while the operator decision is still
    # pending. If the process dies after this snapshot but before metadata is
    # consumed, restore will re-apply the idempotent helper and finish safely.
    from CoScientist.agents.run_control_plugin import save_continuation

    save_continuation(
        context,
        boundary="control_resolution_applied",
        agent=str(getattr(context, "agent_name", "") or ""),
    )
    handle.controller.journal(
        handle.run_id, "control_resolution_applied", data=done,
    )
    # Consume only after the state mutation and journal are durable.  The
    # helper is idempotent if a crash occurs between those operations.
    handle.controller.update_metadata(handle.run_id, {
        "control_resolution": None,
        "control_resolution_done": done,
    })
    return result


class RepeatCallGuardPlugin(BasePlugin):
    """Refuse exact repeats and terminally bound no-progress control calls."""

    def __init__(self, name: str = "repeat_call_guard") -> None:
        super().__init__(name=name)
        self._counts: Counter = Counter()
        self._pending: Dict[str, str] = {}
        self._semantic_pending: Dict[str, tuple[str, str]] = {}

    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> Optional[Dict[str, Any]]:
        if not _enabled():
            return None
        tool_name = str(getattr(tool, "name", "") or "")
        short = tool_name.rsplit("_", 1)[-1] if tool_name else ""
        if tool_name in EXEMPT_TOOLS or short in EXEMPT_TOOLS or any(
            tool_name.endswith(p) for p in EXEMPT_TOOLS
        ):
            return None

        state = getattr(tool_context, "state", None)
        if tool_name in CONTROL_TOOLS and state is not None:
            semantic = dict(state.get(_SEMANTIC_STATE_KEY) or {})
            blocked = semantic.get("blocked")
            if isinstance(blocked, dict) and tool_name not in READ_ONLY_TOOLS:
                _request_run_pause(
                    str(blocked.get("error_code") or "control_no_progress_limit"),
                    state,
                    tool_context,
                )
                return dict(blocked)

        agent = str(getattr(tool_context, "agent_name", "") or "?")
        keyed_args = _semantic_args(tool_args) if tool_name in CONTROL_TOOLS else tool_args
        key = _key(agent, tool_name, keyed_args)
        self._counts[key] += 1
        n = self._counts[key]
        call_id = str(getattr(tool_context, "function_call_id", None) or "")
        pending_id = call_id or f"{id(state)}:{agent}:{tool_name}"
        if call_id:
            self._pending[call_id] = key
        if tool_name in CONTROL_TOOLS and state is not None:
            self._semantic_pending[pending_id] = (tool_name, _state_revision(state))
        if n <= _limit() or tool_name in READ_ONLY_TOOLS:
            return None

        return {
            "status": "blocked",
            "blocked_by": "repeat_call_guard",
            "repeats": n,
            "next_actions": [
                {"action": "use_different_arguments_or_tool"},
                {"action": "record_blocker_and_stop"},
            ],
            "message": (
                f"BLOCKED: you have called `{tool_name}` with these exact arguments "
                f"{n} times. Repeating it will not produce a different result. "
                "Change approach: use different arguments, a different tool, or "
                "state plainly what is blocking you and stop. If you are waiting "
                "for a long job, poll it with check_job or check_sandbox_task instead."
            ),
        }

    async def after_tool_callback(
        self, *, tool, tool_args, tool_context, result
    ) -> Optional[Dict[str, Any]]:
        from CoScientist.agents.callbacks.tool_callbacks import (
            is_transient_tool_error,
            normalize_tool_observation,
        )

        if tool_context is None:
            return None
        call_id = str(getattr(tool_context, "function_call_id", None) or "")
        key = self._pending.pop(call_id, None)
        if key and is_transient_tool_error(result) and self._counts[key] > 0:
            self._counts[key] -= 1

        tool_name = str(getattr(tool, "name", "") or "")
        state = getattr(tool_context, "state", None)
        if tool_name not in CONTROL_TOOLS or state is None:
            return None
        pending_id = call_id or (
            f"{id(state)}:{getattr(tool_context, 'agent_name', '?')}:{tool_name}"
        )
        _, before_revision = self._semantic_pending.pop(
            pending_id, (tool_name, _state_revision(state)),
        )
        after_revision = _state_revision(state)
        semantic = dict(state.get(_SEMANTIC_STATE_KEY) or {})
        if before_revision == after_revision:
            semantic["no_progress"] = int(semantic.get("no_progress") or 0) + 1
        else:
            semantic["no_progress"] = 0
            semantic["blocked"] = None

        observation = normalize_tool_observation(result)
        if observation.get("is_error"):
            signature = json.dumps({
                "tool": tool_name,
                "task_id": str((tool_args or {}).get("task_id") or ""),
                "error_code": str(observation.get("error_code") or "tool_error"),
            }, sort_keys=True)
            if signature == semantic.get("last_failure"):
                semantic["same_failure"] = int(semantic.get("same_failure") or 0) + 1
            else:
                semantic["last_failure"] = signature
                semantic["same_failure"] = 1
        elif before_revision != after_revision:
            semantic["last_failure"] = None
            semantic["same_failure"] = 0

        same_limit, progress_limit = _semantic_limits()
        if int(semantic.get("same_failure") or 0) >= same_limit:
            semantic["blocked"] = _recovery("same_control_failure_limit", state)
        elif int(semantic.get("no_progress") or 0) >= progress_limit:
            semantic["blocked"] = _recovery("control_no_progress_limit", state)
        state[_SEMANTIC_STATE_KEY] = semantic
        blocked = semantic.get("blocked")
        if isinstance(blocked, dict):
            _request_run_pause(
                str(blocked.get("error_code") or "control_no_progress_limit"),
                state,
                tool_context,
            )
        # Plugin after_tool callbacks must never swallow the agent callback chain.
        return None


repeat_call_guard_plugin = RepeatCallGuardPlugin()
