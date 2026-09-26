"""Typed A2A tracing boundary and outbound RemoteA2aAgent propagation."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, Mapping
from contextlib import aclosing
from typing import Optional

from a2a.client.middleware import ClientCallContext
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers.jsonrpc_handler import JSONRPCHandler
from a2a.server.tasks import TaskStore
from a2a.types import (
    CancelTaskRequest,
    CancelTaskResponse,
    GetTaskRequest,
    GetTaskResponse,
    SendMessageResponse,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
    TaskResubscriptionRequest,
)
from google.adk.a2a.agent.config import (
    A2aRemoteAgentConfig,
    ParametersConfig,
    RequestInterceptor,
)

from CoScientist.checkpoints.trace_context import outbound_carrier, scope_for_request

logger = logging.getLogger(__name__)


def _headers(context: Optional[ServerCallContext]) -> dict[str, str]:
    if context is None or not isinstance(context.state, dict):
        return {}
    headers = context.state.get("headers")
    return dict(headers) if isinstance(headers, Mapping) else {}


class SynapseJSONRPCHandler(JSONRPCHandler):
    """Enter canonical Run scope before the instrumented SDK handler methods."""

    def __init__(self, *args, task_store: TaskStore, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._synapse_task_store = task_store

    async def _task_context_id(
        self, task_id: str, context: Optional[ServerCallContext]
    ) -> Optional[str]:
        try:
            task = await self._synapse_task_store.get(task_id, context)
            return task.context_id if task is not None else None
        except Exception as exc:  # noqa: BLE001 - tracing lookup is best-effort
            logger.warning("synapse: task trace lookup failed for %s: %s", task_id, exc)
            return None

    async def on_message_send(self, request, context=None) -> SendMessageResponse:
        context_id = request.params.message.context_id
        if context_id is None and request.params.message.task_id is not None:
            context_id = await self._task_context_id(
                request.params.message.task_id, context
            )
        with scope_for_request(context_id, _headers(context)):
            return await super().on_message_send(request, context)

    async def on_message_send_stream(
        self, request: SendStreamingMessageRequest, context=None
    ) -> AsyncIterable[SendStreamingMessageResponse]:
        context_id = request.params.message.context_id
        if context_id is None and request.params.message.task_id is not None:
            context_id = await self._task_context_id(
                request.params.message.task_id, context
            )
        with scope_for_request(context_id, _headers(context)):
            async with aclosing(
                super().on_message_send_stream(request, context)
            ) as events:
                async for event in events:
                    yield event

    async def on_get_task(
        self, request: GetTaskRequest, context=None
    ) -> GetTaskResponse:
        context_id = await self._task_context_id(request.params.id, context)
        with scope_for_request(context_id, _headers(context)):
            return await super().on_get_task(request, context)

    async def on_cancel_task(
        self, request: CancelTaskRequest, context=None
    ) -> CancelTaskResponse:
        context_id = await self._task_context_id(request.params.id, context)
        with scope_for_request(context_id, _headers(context)):
            return await super().on_cancel_task(request, context)

    async def on_resubscribe_to_task(
        self, request: TaskResubscriptionRequest, context=None
    ) -> AsyncIterable[SendStreamingMessageResponse]:
        context_id = await self._task_context_id(request.params.id, context)
        with scope_for_request(context_id, _headers(context)):
            async with aclosing(
                super().on_resubscribe_to_task(request, context)
            ) as events:
                async for event in events:
                    yield event


async def _inject_run_context(ctx, a2a_request, params: ParametersConfig):
    carrier = outbound_carrier()
    if not carrier:
        return a2a_request, params

    original = params.client_call_context
    state = dict(original.state or {}) if original is not None else {}
    raw_http_kwargs = state.get("http_kwargs")
    if raw_http_kwargs is not None and not isinstance(raw_http_kwargs, Mapping):
        logger.warning(
            "synapse: outbound trace headers skipped for invalid http_kwargs"
        )
        return a2a_request, params
    http_kwargs = dict(raw_http_kwargs or {})
    raw_headers = http_kwargs.get("headers")
    if raw_headers is not None and not isinstance(raw_headers, Mapping):
        logger.warning("synapse: outbound trace headers skipped for invalid headers")
        return a2a_request, params
    headers = dict(raw_headers or {})
    propagated = {key.lower() for key in carrier}
    headers = {
        key: value for key, value in headers.items() if key.lower() not in propagated
    }
    headers.update(carrier)
    http_kwargs["headers"] = headers
    state["http_kwargs"] = http_kwargs
    copied = ClientCallContext(state=state)
    return a2a_request, params.model_copy(update={"client_call_context": copied})


def make_remote_agent_config() -> A2aRemoteAgentConfig:
    """Configure supported W3C propagation without mutating session state."""
    return A2aRemoteAgentConfig(
        request_interceptors=[RequestInterceptor(before_request=_inject_run_context)]
    )
