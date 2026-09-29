"""Authenticated research-graph scope handoff for trusted A2A delegations.

Public A2A callers may choose their own context ID, but that must never let
them choose another research graph. Only the internal RemoteA2aAgent signs the
parent scope; the child binds its ADK session after verifying that signature.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
from collections.abc import Mapping

from a2a.client.middleware import ClientCallContext
from google.adk.a2a.agent.config import (
    A2aRemoteAgentConfig,
    ParametersConfig,
    RequestInterceptor,
)
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor

from CoScientist.graph.session_scope import (
    GRAPH_SCOPE_SESSION_KEY,
    GRAPH_SCOPE_USER_KEY,
    SessionKey,
    session_key,
)

logger = logging.getLogger(__name__)
_SCOPE_HEADER = "x-coscientist-graph-scope"
_PROOF_HEADER = "x-coscientist-graph-scope-proof"
_SECRET_ENV = "COSCIENTIST_A2A_GRAPH_SCOPE_SECRET"
# run_all hosts all agents in one process. Separate processes must configure
# the same private secret explicitly; a random fallback cannot cross processes.
_PROCESS_SECRET = secrets.token_bytes(32)


def _secret() -> bytes:
    configured = os.getenv(_SECRET_ENV)
    return configured.encode("utf-8") if configured else _PROCESS_SECRET


def _encode_scope(scope: SessionKey) -> str:
    payload = json.dumps(scope, ensure_ascii=False, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def _decode_scope(encoded: str) -> SessionKey:
    if len(encoded) > 2048:
        raise ValueError("oversized graph scope")
    try:
        value = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid graph scope") from exc
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            not isinstance(item, str) or not item or len(item) > 256 for item in value
        )
    ):
        raise ValueError("invalid graph scope")
    return value[0], value[1]


def _proof(scope: str, target: str, message_id: str) -> str:
    payload = f"coscientist-graph-scope-v1\n{target}\n{message_id}\n{scope}"
    return hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()


async def _inject_scope(ctx, message, params: ParametersConfig, target: str):
    """Sign the parent's resolved graph scope without editing science state."""
    scope = _encode_scope(session_key(ctx))
    original = params.client_call_context
    state = dict(original.state or {}) if original is not None else {}
    raw_kwargs = state.get("http_kwargs")
    if raw_kwargs is not None and not isinstance(raw_kwargs, Mapping):
        logger.warning("graph scope: invalid outbound http_kwargs")
        return message, params
    kwargs = dict(raw_kwargs or {})
    raw_headers = kwargs.get("headers")
    if raw_headers is not None and not isinstance(raw_headers, Mapping):
        logger.warning("graph scope: invalid outbound headers")
        return message, params
    headers = {
        key: value
        for key, value in (raw_headers or {}).items()
        if str(key).lower() not in (_SCOPE_HEADER, _PROOF_HEADER)
    }
    headers[_SCOPE_HEADER] = scope
    headers[_PROOF_HEADER] = _proof(scope, target, message.message_id)
    kwargs["headers"] = headers
    state["http_kwargs"] = kwargs
    return message, params.model_copy(
        update={"client_call_context": ClientCallContext(state=state)}
    )


def make_graph_scope_config(
    target: str, *, trace: bool = False
) -> A2aRemoteAgentConfig:
    """Configure signed graph scope, followed by optional Synapse tracing."""
    if trace:
        from CoScientist.a2a.synapse_tracing import make_remote_agent_config

        config = make_remote_agent_config()
    else:
        config = A2aRemoteAgentConfig()

    async def inject(ctx, message, params):
        return await _inject_scope(ctx, message, params, target)

    config.request_interceptors = [
        RequestInterceptor(before_request=inject),
        *(config.request_interceptors or []),
    ]
    return config


def _scope_from_request(context, target: str) -> SessionKey | None:
    call_context = context.call_context
    state = getattr(call_context, "state", None)
    raw_headers = state.get("headers") if isinstance(state, Mapping) else None
    headers = (
        {str(key).lower(): str(value) for key, value in raw_headers.items()}
        if isinstance(raw_headers, Mapping)
        else {}
    )
    encoded = headers.get(_SCOPE_HEADER)
    proof = headers.get(_PROOF_HEADER)
    if encoded is None and proof is None:
        return None
    message_id = getattr(context.message, "message_id", None)
    if not encoded or not proof or not message_id:
        raise ValueError("incomplete graph scope proof")
    if not hmac.compare_digest(proof, _proof(encoded, target, message_id)):
        raise ValueError("invalid graph scope proof")
    return _decode_scope(encoded)


class GraphScopeA2aAgentExecutor(A2aAgentExecutor):
    """Bind a child ADK session to a verified parent scope before execution."""

    def __init__(self, *, runner):
        # ADK 2.9's opt-in executor bypasses _prepare_session. Until that path
        # offers the same proof check, do not let a public extension header
        # select it for a session already bound to a trusted graph scope.
        super().__init__(runner=runner, use_legacy=True)

    async def _prepare_session(self, context, run_request, runner):
        scope = _scope_from_request(context, runner.app_name)
        session = await super()._prepare_session(context, run_request, runner)
        existing_user = session.state.get(GRAPH_SCOPE_USER_KEY)
        existing_session = session.state.get(GRAPH_SCOPE_SESSION_KEY)
        if existing_user or existing_session:
            existing = (existing_user, existing_session)
            own_scope = (run_request.user_id, run_request.session_id)
            if (scope is None and existing != own_scope) or (
                scope is not None and scope != existing
            ):
                raise ValueError("graph scope proof required for this A2A session")
        if scope is not None:
            run_request.state_delta = {
                GRAPH_SCOPE_USER_KEY: scope[0],
                GRAPH_SCOPE_SESSION_KEY: scope[1],
            }
        return session
