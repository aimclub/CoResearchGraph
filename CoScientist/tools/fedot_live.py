from __future__ import annotations

import asyncio
import logging
import time
from uuid import uuid4

from google.adk.events import Event
from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext

from CoScientist.tools import fedot_runs

_WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


def _dump(value):
    """Serialize content, never transport credentials or callable tools."""
    if value is None:
        return None
    try:
        return value.model_dump(mode="json", exclude_none=True)
    except Exception:
        return str(value)


class FedotLiveRun:
    def __init__(self, bus, scope, context):
        self.bus, self.scope = bus, scope
        self.meta = fedot_runs.create_run(scope, **context)
        self.run_id = self.meta["run_id"]

    def event(self, payload: dict) -> None:
        try:
            event = fedot_runs.append_event(self.scope, self.run_id, payload)
            self.bus._broadcast(self.scope, event)
        except Exception:
            logging.getLogger(__name__).exception("Cannot persist FEDOT event for %s", self.run_id)

    def publish_config(self, config: dict) -> None:
        self.event({"type": "config", "config": config})


class FedotLiveBroadcaster:
    def __init__(self) -> None:
        self._subscribers: dict[asyncio.Queue, tuple[str, str]] = {}

    def begin_run(self, scope=("local", "default"), **context) -> FedotLiveRun:
        run = FedotLiveRun(self, scope, context)
        run.event({"type": "run_start", **context})
        return run

    def subscribe(self, scope=("local", "default")) -> asyncio.Queue:
        # Internal compatibility for consumers/tests. HTTP uses the durable
        # journal, so other processes and server restarts do not lose events.
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[q] = scope
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.pop(q, None)

    def _broadcast(self, scope: tuple[str, str], payload: dict) -> None:
        for q, owner in list(self._subscribers.items()):
            if owner == scope:
                q.put_nowait(payload)

fedot_live = FedotLiveBroadcaster()


class FedotLivePlugin(BasePlugin):
    def __init__(self, broadcaster: FedotLiveRun, name: str = "fedot_live", *, trace_plugin=None) -> None:
        super().__init__(name)
        self._bus = broadcaster
        self._start: dict[tuple, list[tuple[str, float]]] = {}
        self._trace_plugin = trace_plugin
        self._trace_link = None

    def _capture_trace_link(self):
        # end_trace() clears the SDK's _trace before MAS.run returns. Capture
        # inside the runner lifecycle, after Langfuse's own before_run hook.
        trace = getattr(self._trace_plugin, "_trace", None)
        trace_id, observation_id = getattr(trace, "trace_id", None), getattr(trace, "id", None)
        link = (trace_id, observation_id)
        if isinstance(trace_id, str) and isinstance(observation_id, str) and link != self._trace_link:
            self._trace_link = link
            self._bus.event({"type": "langfuse_link", "trace_id": trace_id, "observation_id": observation_id})

    async def before_run_callback(self, *, invocation_context):
        self._capture_trace_link()

    async def after_run_callback(self, *, invocation_context):
        self._capture_trace_link()

    @staticmethod
    def _key(context, name):
        invocation = getattr(context, "_invocation_context", None)
        return (getattr(context, "invocation_id", ""),
                getattr(invocation, "branch", ""), name)

    def _is_workflow(self, name: str) -> bool:
        return name.startswith(_WORKFLOW_PREFIXES)

    def _context(self, context, name):
        key = self._key(context, name)
        stack = self._start.get(key, [])
        return {"invocation_id": key[0], "branch": key[1],
                "agent_span_id": stack[-1][0] if stack else None}

    async def before_model_callback(self, *, callback_context, llm_request):
        self._bus.event({"type": "model_start", "agent": callback_context.agent_name,
                         **self._context(callback_context, callback_context.agent_name),
                         "request": {"model": llm_request.model,
                                     "contents": [_dump(c) for c in llm_request.contents],
                                     "system_instruction": _dump(llm_request.config.system_instruction)}})

    async def after_model_callback(self, *, callback_context, llm_response):
        self._bus.event({"type": "model_end", "agent": callback_context.agent_name,
                         **self._context(callback_context, callback_context.agent_name),
                         "response": _dump(llm_response)})

    async def on_model_error_callback(self, *, callback_context, llm_request, error):
        self._bus.event({"type": "model_error", "agent": callback_context.agent_name,
                         **self._context(callback_context, callback_context.agent_name),
                         "error": str(error)})

    async def before_agent_callback(self, *, agent, callback_context):  # noqa: ANN001
        if self._is_workflow(agent.name):
            return None
        span_id = uuid4().hex
        parent_span_id = None
        parent = getattr(agent, "parent_agent", None)
        while parent is not None:
            parent_span_id = self._context(callback_context, parent.name)["agent_span_id"]
            if parent_span_id:
                break
            parent = getattr(parent, "parent_agent", None)
        self._start.setdefault(self._key(callback_context, agent.name), []).append((span_id, time.monotonic()))
        instruction = str(getattr(agent, "instruction", "") or "")
        try:
            state = dict(callback_context.state.to_dict())
        except Exception:  # noqa: BLE001 — best-effort, tracing must never break the run
            state = {}
        incoming = {k: v for k, v in state.items() if k != "user_query"}
        self._bus.event({
            "type": "agent_start",
            "agent": agent.name,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            **self._context(callback_context, agent.name),
            "instruction": instruction,
            "incoming": incoming,
        })
        return None

    async def after_agent_callback(self, *, agent, callback_context):  # noqa: ANN001
        stack = self._start.get(self._key(callback_context, agent.name), [])
        if not stack or self._is_workflow(agent.name):
            return None
        span_id, t0 = stack.pop()
        output_key = getattr(agent, "output_key", None)
        produced = ""
        if output_key:
            try:
                produced = str(callback_context.state.to_dict().get(output_key, ""))
            except Exception:  # noqa: BLE001
                produced = ""
        self._bus.event({
            "type": "agent_done",
            "agent": agent.name,
            "span_id": span_id,
            **self._context(callback_context, agent.name),
            "ms": int((time.monotonic() - t0) * 1000),
            "output_key": output_key,
            "output": produced,
        })
        return None

    async def on_event_callback(  # noqa: ANN001
        self, *, invocation_context: InvocationContext, event: Event
    ):
        if event.partial:
            return None
        author = event.author or "?"
        key = (getattr(event, "invocation_id", ""), getattr(event, "branch", ""), author)
        stack = self._start.get(key, [])
        context = {"invocation_id": key[0], "branch": key[1],
                   "agent_span_id": stack[-1][0] if stack else None}

        usage = event.usage_metadata
        tokens = 0
        if usage is not None:
            tokens = (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0)

        for fc in event.get_function_calls():
            args = fc.args or {}
            target = args.get("agent_name") or args.get("agent") or ""
            self._bus.event({
                "type": "tool", "agent": author, "tool": fc.name,
                **context,
                "target": str(target), "args": args, "call_id": fc.id,
            })
        for fr in event.get_function_responses():
            resp = fr.response if fr.response is not None else ""
            is_error = isinstance(fr.response, dict) and fr.response.get("isError") is True
            self._bus.event({
                "type": "tool_result", "agent": author, "tool": fr.name,
                **context,
                "error": bool(is_error), "text": resp, "call_id": fr.id,
            })

        text = ""
        if event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts if p.text)
        if text.strip():
            self._bus.event({"type": "text", "agent": author, **context, "text": text.strip(), "tokens": tokens})
        elif tokens:
            self._bus.event({"type": "tokens", "agent": author, "tokens": tokens})
        return None
