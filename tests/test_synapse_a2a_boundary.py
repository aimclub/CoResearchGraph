"""Typed boundary coverage for continuation lookup and stream cleanup."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    CancelTaskRequest,
    GetTaskRequest,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TaskResubscriptionRequest,
    TaskState,
    TaskStatus,
)

TRACE_ID = "4af7651916cd43dd8448eb211c80319c"
ROOT_SPAN_ID = "f7ad6b7169203331"
CONTEXT_ID = "ctx-task-store"
TASK_ID = "task-existing"


def _card() -> AgentCard:
    return AgentCard(
        name="boundary",
        description="boundary",
        url="http://boundary/",
        version="1",
        capabilities=AgentCapabilities(streaming=True),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[],
    )


def _task() -> Task:
    return Task(
        id=TASK_ID,
        context_id=CONTEXT_ID,
        status=TaskStatus(state=TaskState.working),
    )


def _message_request(streaming: bool = False):
    cls = SendStreamingMessageRequest if streaming else SendMessageRequest
    return cls.model_validate(
        {
            "jsonrpc": "2.0",
            "id": "request-1",
            "method": "message/stream" if streaming else "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "role": "user",
                    "messageId": "message-1",
                    "taskId": TASK_ID,
                    "parts": [{"kind": "text", "text": "continue"}],
                }
            },
        }
    )


class _RecordingRequestHandler:
    def __init__(self, trace_context):
        self.trace_context = trace_context
        self.observed: list[tuple[str, str | None]] = []
        self.closed = asyncio.Event()

    def _record(self, operation: str):
        self.observed.append((operation, self.trace_context.active_run_id()))

    async def on_message_send(self, params, context):
        self._record("send")
        return _task()

    async def on_message_send_stream(self, params, context):
        self._record("stream")
        try:
            yield _task()
            await asyncio.Event().wait()
        finally:
            self._record("stream-close")
            self.closed.set()

    async def on_get_task(self, params, context):
        self._record("get")
        return _task()

    async def on_cancel_task(self, params, context):
        self._record("cancel")
        return _task()

    async def on_resubscribe_to_task(self, params, context):
        self._record("resubscribe")
        try:
            yield _task()
            await asyncio.Event().wait()
        finally:
            self._record("resubscribe-close")


def test_task_store_recovers_context_and_stream_close_resets_scope():
    from CoScientist.a2a.synapse_tracing import SynapseJSONRPCHandler
    from CoScientist.checkpoints import synapse, trace_context

    async def exercise():
        store = InMemoryTaskStore()
        await store.save(_task())
        downstream = _RecordingRequestHandler(trace_context)
        handler = SynapseJSONRPCHandler(_card(), downstream, task_store=store)
        call_context = SimpleNamespace(state={"headers": {}})
        synapse.clear_runs()
        synapse.register_run(
            CONTEXT_ID,
            "run-task-store",
            f"00-{TRACE_ID}-{ROOT_SPAN_ID}-01",
        )

        await handler.on_message_send(_message_request(), call_context)
        stream = handler.on_message_send_stream(
            _message_request(streaming=True), call_context
        )
        await anext(stream)
        await stream.aclose()
        await asyncio.wait_for(downstream.closed.wait(), timeout=1)

        await handler.on_get_task(
            GetTaskRequest.model_validate(
                {
                    "jsonrpc": "2.0",
                    "id": "g",
                    "method": "tasks/get",
                    "params": {"id": TASK_ID},
                }
            ),
            call_context,
        )
        await handler.on_cancel_task(
            CancelTaskRequest.model_validate(
                {
                    "jsonrpc": "2.0",
                    "id": "c",
                    "method": "tasks/cancel",
                    "params": {"id": TASK_ID},
                }
            ),
            call_context,
        )
        resubscribe = handler.on_resubscribe_to_task(
            TaskResubscriptionRequest.model_validate(
                {
                    "jsonrpc": "2.0",
                    "id": "r",
                    "method": "tasks/resubscribe",
                    "params": {"id": TASK_ID},
                }
            ),
            call_context,
        )
        await anext(resubscribe)
        await resubscribe.aclose()

        assert trace_context.active_run_id() is None
        return downstream.observed

    observed = asyncio.run(exercise())
    assert observed == [
        ("send", "run-task-store"),
        ("stream", "run-task-store"),
        ("stream-close", "run-task-store"),
        ("get", "run-task-store"),
        ("cancel", "run-task-store"),
        ("resubscribe", "run-task-store"),
        ("resubscribe-close", "run-task-store"),
    ]
