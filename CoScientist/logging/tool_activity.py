"""ADK plugin that streams every tool call/result to an observer sink.

Why a plugin and not the event stream: subordinate agents are attached as
``AgentTool`` (see ``assembly/assembler.py``), and an AgentTool delegation runs
its agent in a *nested* Runner with its own child ADK session. Only the
delegation's own function_call/function_response surfaces in the parent
``run_async`` stream — everything the sub-agent does inside (e.g.
``ResearchAgent`` calling ``tavily_search``) never reaches a consumer of the
top-level events. Tool callbacks, on the other hand, fire in nested runners
too, which is exactly why the console trace in ``event_logger`` sees them.

The web UI registers a sink here to show the full picture live. With no sink
registered the plugin is inert, so CLI/A2A runs pay nothing for it.

The sink is invoked as ``await sink(session_key, payload)`` and must never be
able to break a run: every dispatch is guarded.
"""

from __future__ import annotations

import contextvars
import copy
import json
import logging
import re
import sys
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.sessions.state import State

from CoScientist.graph.session_scope import SessionKey, session_key

logger = logging.getLogger("CoScientist.logging.tool_activity")

ToolActivitySink = Callable[[SessionKey, dict], Awaitable[None]]

_PREVIEW_LIMIT = 1500
_FULL_LIMIT = 2_000_000
_DESCRIPTION_LIMIT = 200

_sink: Optional[ToolActivitySink] = None

# The delegation a nested AgentTool run was launched by: the calling agent,
# its runtime instance (session id) and the call id. ADK runs a call's
# before-tool callbacks and the tool itself in one task of their own, and an
# AgentTool drives its nested Runner inside that task, so everything the
# delegated agent does sees this value — and sibling calls never see each
# other's. The nested session has no link back to its caller otherwise.
_delegation_caller: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "tool_activity_delegation_caller", default=None,
)


def set_tool_activity_sink(sink: Optional[ToolActivitySink]) -> None:
    """Register (or clear, with ``None``) the observer for tool activity."""
    global _sink
    _sink = sink


def _render(value: Any, limit: int) -> tuple[Any, bool]:
    """Return ``(rendered, truncated)`` — a JSON-safe rendering of ``value``
    capped at ``limit`` characters.

    Structure is preserved while it stays under the cap — observers render a
    dict of arguments far more readably than its escaped JSON text. Anything
    larger degrades to a truncated string.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    try:
        text = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, default=str,
        )
    except (TypeError, ValueError):
        text = str(value)
    truncated = len(text) > limit
    if truncated:
        text = text[:limit] + " …"
    if isinstance(value, str) or truncated:
        return text, truncated
    try:
        return json.loads(text), False
    except (TypeError, ValueError):
        return text, False


def _preview_and_full(value: Any, limit: int = _PREVIEW_LIMIT) -> tuple[Any, Any, bool]:
    """Return ``(preview, full, truncated)`` for one args/result/error value.

    ``truncated`` reflects only the preview cap — ``full`` is what the
    ToolsViewer fetches on demand when the user asks to see everything, so a
    caller should skip storing/sending it at all when ``truncated`` is False
    (the preview already *is* the complete value).
    """
    preview, truncated = _render(value, limit)
    if not truncated:
        return preview, preview, False
    full, _ = _render(value, _FULL_LIMIT)
    return preview, full, True


def _short_description(tool: Any) -> Optional[str]:
    """The tool's own description, on one line and capped.

    Only the first sentence-ish is useful to a consumer classifying the call,
    and a full MCP description can run to several paragraphs.
    """
    text = getattr(tool, "description", None)
    if not isinstance(text, str):
        return None
    text = " ".join(text.split())
    if not text:
        return None
    return text[:_DESCRIPTION_LIMIT]


def _agent_name(tool_context: Any) -> str:
    return getattr(tool_context, "agent_name", None) or "system"


def _agent_instance(tool_context: Any) -> Optional[str]:
    """Return the runtime session that distinguishes repeated agent runs.

    A parallel pair of AgentTool delegations can run the same configured agent
    name at once.  ADK gives each delegated run its own child session, whereas
    ``agent_name`` alone is necessarily identical.  Keep the public session
    scope out of this value: the UI needs the transient runtime identity to
    render two branches, not another routing key.
    """
    invocation = (
        getattr(tool_context, "_invocation_context", None)
        or getattr(tool_context, "invocation_context", None)
    )
    session = getattr(tool_context, "session", None) or getattr(invocation, "session", None)
    value = getattr(session, "id", None)
    return str(value) if value else None


def _parent_agent_instance(tool_context: Any) -> Optional[str]:
    """Return the runtime identity of a nested callback's caller, if known."""
    return _agent_instance(getattr(tool_context, "_parent_ctx", None))


def _delegating_caller(context: Any, parent: Optional[str]) -> Optional[dict]:
    """The delegation that launched the nested run ``context`` belongs to.

    Only a run in a session other than the caller's is a delegated one, and
    only when ``parent`` names the caller — an agent nested in-process below
    the delegated agent has a parent of its own.
    """
    caller = _delegation_caller.get()
    if caller is None or caller["instance"] == _agent_instance(context):
        return None
    if parent is not None and parent != caller["agent"]:
        return None
    return caller


def _call_id(tool_context: Any) -> Optional[str]:
    """The id ADK assigns to this function call.

    Lets an observer pair a result with its own call instead of guessing by
    tool name — which is wrong as soon as an agent runs the same tool twice.
    """
    call_id = getattr(tool_context, "function_call_id", None)
    return str(call_id) if call_id else None


def _delegation_target(tool: Any, tool_args: Any = None) -> Optional[str]:
    """Detect if a tool is delegating to a sub-agent."""
    agent_attr = getattr(tool, "agent", None)
    if agent_attr is not None and getattr(agent_attr, "name", None):
        return str(agent_attr.name)
    tool_name = getattr(tool, "name", "")
    if tool_name == "transfer_to_agent" and isinstance(tool_args, dict):
        return tool_args.get("agent_name") or tool_args.get("agentName")
    try:
        from CoScientist.assembly.schema import get_config
        config = get_config()
        if tool_name in config.agents:
            return tool_name
    except Exception:
        pass
    return None


def _parent_agent_name(tool_context: Any, author: str) -> Optional[str]:
    """Resolve the parent agent in the execution hierarchy (runtime or static config)."""
    node = getattr(tool_context, "_node", None)
    if node is not None:
        parent_agent = getattr(node, "parent_agent", None)
        if parent_agent and getattr(parent_agent, "name", None):
            return str(parent_agent.name)
    parent_ctx = getattr(tool_context, "_parent_ctx", None)
    if parent_ctx is not None and getattr(parent_ctx, "agent_name", None):
        return str(parent_ctx.agent_name)
    try:
        from CoScientist.assembly.schema import get_config
        h_map = get_config().agent_hierarchy_map()
        return h_map.get("parents", {}).get(author)
    except Exception:
        return None


def _inside_run_of(tool: Any) -> bool:
    """Whether the caller is (somewhere below) ``tool.run_async``.

    Separates the tool's own state reads from those of the before/after-tool
    callbacks, which see the same ``ToolContext`` but are not the tool's
    inputs (a Work Order guard, the link registry…).
    """
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_name == "run_async" and frame.f_locals.get("self") is tool:
            return True
        frame = frame.f_back
    return False


class _RecordingState(State):
    """The call's own ``State``, noting which keys the tool reads.

    Shares the value/delta dicts of the state it replaces, so reads and writes
    behave exactly as before. A key the tool wrote before reading it is its own
    output, not an input, and is not recorded.
    """

    def __init__(self, base: State, tool: Any) -> None:
        super().__init__(
            value=base._value, delta=base._delta, schema=getattr(base, "_schema", None),
        )
        self._tool = tool
        self.reads: dict[str, Any] = {}
        self._written: set[str] = set()

    def _note(self, key: str, value: Any) -> None:
        if key in self.reads or key in self._written or not _inside_run_of(self._tool):
            return
        try:
            # A snapshot: the tool may go on to mutate what it read.
            self.reads[key] = copy.deepcopy(value)
        except Exception:  # noqa: BLE001 - an uncopyable value is still worth showing
            self.reads[key] = value

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        self._note(key, value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self._note(key, None)
            return default
        return self[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._written.add(key)
        super().__setitem__(key, value)

    def update(self, delta: dict[str, Any]) -> None:
        self._written.update(delta)
        super().update(delta)


class _CallCapture:
    """What a call's closing record needs from its opening: the arguments as
    the model wrote them and the recording state installed for the tool."""

    def __init__(self, args: Any, original: Any, recording: Optional[_RecordingState]):
        self.args = args
        self.original = original
        self.recording = recording


_CAPTURE_ATTR = "_coscientist_tool_capture"


def _start_capture(tool: Any, tool_args: Any, tool_context: Any) -> None:
    try:
        args = copy.deepcopy(tool_args)
    except Exception:  # noqa: BLE001
        args = None
    original = getattr(tool_context, "_state", None)
    recording = None
    if isinstance(original, State) and not isinstance(original, _RecordingState):
        try:
            recording = _RecordingState(original, tool)
            tool_context._state = recording
        except Exception as exc:  # noqa: BLE001 - observing must not break a call
            logger.debug("Could not record state reads for %s: %s", getattr(tool, "name", "?"), exc)
            recording = None
    try:
        setattr(tool_context, _CAPTURE_ATTR, _CallCapture(args, original, recording))
    except Exception:  # noqa: BLE001
        pass


# Names that mark a secret, matched against every key of a captured value.
_SECRET_KEY_RE = re.compile(
    r"pass(word|wd)?|secret|token|api[_-]?key|credential|auth|cookie|session[_-]?key|private[_-]?key",
    re.I,
)
_REDACTED = "***"


def _redact(value: Any, depth: int = 0) -> Any:
    """``value`` with secret-looking entries masked, for state and callback-set
    arguments: neither is written by the model, so either may hold a key the
    model itself never saw (ADK keeps OAuth credentials in the state)."""
    if depth > 20:
        return value
    if hasattr(value, "model_dump") and type(value).__name__ == "AuthCredential":
        return _REDACTED
    if isinstance(value, dict):
        return {
            k: _REDACTED if isinstance(k, str) and _SECRET_KEY_RE.search(k) and v not in (None, "")
            else _redact(v, depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(v, depth + 1) for v in value]
    return value


def _value_fields(field: str, value: Any) -> dict:
    preview, full, truncated = _preview_and_full(_redact(value))
    fields = {field: preview, f"{field}_truncated": truncated}
    if truncated:
        fields[f"{field}_full"] = full
    return fields


def _finish_capture(tool_args: Any, tool_context: Any) -> dict:
    """Payload fields describing the call's complete inputs.

    ``effective_args`` — the arguments the tool actually ran with, sent only
    when a before-tool callback rewrote the model's ones; ``state_inputs`` —
    the session-state keys the tool itself read, with the values it saw.
    """
    capture = getattr(tool_context, _CAPTURE_ATTR, None)
    if not isinstance(capture, _CallCapture):
        return {}
    try:
        delattr(tool_context, _CAPTURE_ATTR)
    except Exception:  # noqa: BLE001
        pass
    if capture.recording is not None and getattr(tool_context, "_state", None) is capture.recording:
        tool_context._state = capture.original
    fields: dict = {}
    if capture.args is not None and tool_args != capture.args:
        fields.update(_value_fields("effective_args", tool_args))
    if capture.recording is not None and capture.recording.reads:
        fields.update(_value_fields("state_inputs", capture.recording.reads))
    return fields


async def report_activity(context: Any, payload: dict) -> None:
    """Send one record to the sink on behalf of ``context``'s session.

    The plugin's own entry point, and the one for work that is not an ADK tool
    or agent but should show up as one — the planner's plan critic is a bare
    LLM call that no callback ever sees.
    """
    sink = _sink
    if sink is None:
        return
    try:
        key = session_key(context)
    except Exception:  # noqa: BLE001 - context shapes vary across ADK paths
        return
    payload.setdefault("timestamp", datetime.now().isoformat())
    try:
        await sink(key, payload)
    except Exception as exc:  # noqa: BLE001 - an observer must not fail a run
        logger.warning("Tool activity sink failed: %s", exc)


async def report_delegation(
    context: Any,
    *,
    author: str,
    target: str,
    call_id: str,
    args: Any = None,
    result: Any = None,
    error: Optional[str] = None,
    phase: str,
) -> None:
    """Report a hand-off ``author`` → ``target`` that no AgentTool carries.

    ``phase="call"`` opens it (with ``args``) and starts ``target``;
    ``"result"``/``"error"`` ends ``target`` and closes the call. Observers
    render it exactly like an AgentTool delegation: ``target`` gets its own
    entry in the activity rail and the agent tree, the call its record in the
    ToolsViewer, and the status line says who is working.
    """
    if phase == "call":
        await report_activity(context, {
            "phase": "agent_start", "author": target, "parent": author,
            "agent_class": target,
        })
        preview, full, truncated = _preview_and_full(args)
        payload = {
            "phase": "call", "author": author, "tool": target,
            "call_id": call_id, "args": preview, "args_truncated": truncated,
            "parent": author, "is_delegation": True, "target_agent": target,
        }
        if truncated:
            payload["args_full"] = full
        await report_activity(context, payload)
        return

    await report_activity(context, {"phase": "agent_end", "author": target})
    field = "error" if phase == "error" else "result"
    preview, full, truncated = _preview_and_full(error if phase == "error" else result)
    payload = {
        "phase": phase, "author": author, "tool": target, "call_id": call_id,
        field: preview, f"{field}_truncated": truncated,
        "is_delegation": True, "target_agent": target,
    }
    if truncated:
        payload[f"{field}_full"] = full
    await report_activity(context, payload)


class ToolActivityPlugin(BasePlugin):
    """Report every tool call, result, and error to the registered sink."""

    def __init__(self, name: str = "tool_activity") -> None:
        super().__init__(name=name)

    async def _dispatch(self, tool_context: Any, payload: dict) -> None:
        await report_activity(tool_context, payload)

    async def before_agent_callback(self, *, agent, callback_context) -> None:
        author = getattr(agent, "name", "unknown")
        instance = _agent_instance(callback_context)
        parent = getattr(getattr(agent, "parent_agent", None), "name", None)
        # An in-process sub-agent runs in its parent's own session, so it
        # shares the parent's runtime identity.
        parent_instance = _parent_agent_instance(callback_context) or (instance if parent else None)
        caller = None if parent else _delegating_caller(callback_context, None)
        if caller is not None:
            parent, parent_instance = caller["agent"], caller["instance"]
        elif not parent:
            parent = _parent_agent_name(callback_context, author)
        payload = {
            "phase": "agent_start",
            "author": author,
            "agent_instance": instance,
            "parent": parent,
            "parent_instance": parent_instance,
            "agent_class": getattr(getattr(agent, "__class__", None), "__name__", "Agent"),
        }
        if caller is not None and caller["call_id"]:
            payload["spawn_call_id"] = caller["call_id"]
        await self._dispatch(callback_context, payload)
        return None

    async def after_agent_callback(self, *, agent, callback_context) -> None:
        author = getattr(agent, "name", "unknown")
        payload = {
            "phase": "agent_end",
            "author": author,
            "agent_instance": _agent_instance(callback_context),
        }
        await self._dispatch(callback_context, payload)
        return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> None:
        if _sink is not None:
            _start_capture(tool, tool_args, tool_context)
        preview, full, truncated = _preview_and_full(tool_args)
        author = _agent_name(tool_context)
        target = _delegation_target(tool, tool_args)
        parent = _parent_agent_name(tool_context, author)
        instance = _agent_instance(tool_context)
        call_id = _call_id(tool_context)
        parent_instance = _parent_agent_instance(tool_context)
        if parent_instance is None:
            caller = _delegating_caller(tool_context, parent)
            parent_instance = caller["instance"] if caller else None
        payload = {
            "phase": "call",
            "author": author,
            "agent_instance": instance,
            "tool": getattr(tool, "name", "?"),
            "call_id": call_id,
            "args": preview,
            "args_truncated": truncated,
            "parent": parent,
            "parent_instance": parent_instance,
        }
        if target:
            payload["is_delegation"] = True
            payload["target_agent"] = target
            # Lives only in this call's task; no reset needed.
            _delegation_caller.set({"agent": author, "instance": instance, "call_id": call_id})
        description = _short_description(tool)
        if description:
            payload["description"] = description
        if truncated:
            payload["args_full"] = full
        await self._dispatch(tool_context, payload)
        return None  # never override the tool's own execution

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result) -> None:
        tool_name = getattr(tool, "name", "?")
        # Plan and task tracker tools require their structured payload in full
        # for real-time UI synchronisation; don't truncate them under 1500 chars.
        is_plan_tool = bool(re.search(r"create_plan|task_status|active_tasks|roadmap|add_task|create_task", str(tool_name), re.I))
        limit = 50_000 if is_plan_tool else _PREVIEW_LIMIT
        preview, full, truncated = _preview_and_full(result, limit=limit)
        payload = {
            "phase": "result",
            "author": _agent_name(tool_context),
            "agent_instance": _agent_instance(tool_context),
            "tool": tool_name,
            "call_id": _call_id(tool_context),
            "result": preview,
            "result_truncated": truncated,
        }
        if truncated:
            payload["result_full"] = full
        payload.update(_finish_capture(tool_args, tool_context))
        await self._dispatch(tool_context, payload)
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error) -> None:
        # ADK re-raises a tool error when no plugin supplies a replacement
        # response, so ``after_tool_callback`` never fires for it: this is the
        # only closing record such a call will ever get.
        preview, full, truncated = _preview_and_full(str(error))
        payload = {
            "phase": "error",
            "author": _agent_name(tool_context),
            "agent_instance": _agent_instance(tool_context),
            "tool": getattr(tool, "name", "?"),
            "call_id": _call_id(tool_context),
            "error": preview,
            "error_truncated": truncated,
        }
        if truncated:
            payload["error_full"] = full
        payload.update(_finish_capture(tool_args, tool_context))
        await self._dispatch(tool_context, payload)
        return None
