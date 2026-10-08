"""HITL tools for agents served over A2A.

Over A2A there is no console and no websocket to the human: the agent runs as an
HTTP service and the *caller* is the one who can answer. Blocking inside the tool
(what ConsoleHITLHandler does) would hang the server, which is why HITL did not
work in A2A mode.

A2A has a native mechanism for exactly this, and ADK implements it: a call to a
**long-running** tool is emitted to the client and puts the A2A task into
``input-required`` (see google/adk/a2a/converters/long_running_functions.py).
The run then yields instead of blocking; the client answers by sending back a
FunctionResponse for that call id, and the run resumes with the human's answer.

So these tools are wrapped in ``LongRunningFunctionTool`` and return NOTHING:
ADK skips the auto-FunctionResponse only for a long-running tool that returns a
falsy value, and that is precisely what makes the run yield instead of racing on
with an invented answer (see ``_announce`` for the exact ADK condition). Their
names/signatures match the in-process tools (CoScientist/hitl/tool.py) so
prompts, docs and the unknown-tool guard are unchanged — only delivery differs.

Scope: this native mechanism is for the agent the client talks to directly (the
A2A ROOT). A pause inside a NON-root agent (reached through the orchestrator's
AgentTool) never reaches the caller: AgentTool returns the sub-run's final text,
and a paused sub-agent has none, so the parent sees an empty result and carries
on. Non-root agents keep the handler path (see hitl/tool.get_hitl_tools).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext

from CoScientist.graph.session_scope import session_key

logger = logging.getLogger(__name__)


def _announce(agent_name: str, question: str, **extra: Any) -> None:
    """Log the question and return NOTHING — returning nothing is what pauses the run.

    ADK only skips synthesizing an immediate FunctionResponse when a long-running
    tool returns a FALSY value (flows/llm_flows/functions.py):

        if (tool.is_long_running or tool._defers_response) and not function_response:
            return None   # no auto-FR -> the run yields, A2A task -> input-required

    So a tool that returns a "pending" dict does NOT pause: ADK builds the FR at
    once and the model happily continues — i.e. it would invent the human's answer.
    The question still reaches the client: it travels in the function-call ARGS of
    the emitted call, which A2A delivers in the task's status message.
    """
    logger.info("[HITL/A2A] %s asks the caller: %s | %s", agent_name, question[:300], extra)
    return None


async def request_approval(
    agent_name: str,
    message: str,
    tool_context: ToolContext = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Ask the human to approve an action (yes / no, or a free-text answer).

    Use before expensive, outward-facing or hard-to-reverse actions, and whenever
    a decision is genuinely the human's. The human may reply yes/no OR with
    free-text instructions ("other") — follow whatever they send.

    Args:
        agent_name: Name of the agent requesting approval.
        message: What needs approval, in plain language.
        context: Optional extra context for the human.

    Returns:
        Nothing directly — the run PAUSES here and the human's answer
        ({'approved': bool, 'feedback': str}) arrives as this call's response.
        Do not call again and do not assume an answer; continue from what arrives.
    """
    user_id, session_id = session_key(tool_context) if tool_context is not None else (None, None)
    return _announce(agent_name, message, context=dict(context or {}),
                     session=(user_id, session_id))


async def request_selection(
    agent_name: str,
    message: str,
    options: List[str],
    tool_context: ToolContext = None,
) -> None:
    """Ask the human to choose among options you generated (or answer their own way).

    Use when several alternatives (hypotheses, plans, thresholds) need a human
    decision. The human may pick one of `options` OR reply with their own answer
    ("other") — honour whichever they send.

    Args:
        agent_name: Name of the agent requesting the choice.
        message: What to choose and why it matters.
        options: 2-4 concrete options.

    Returns:
        Nothing directly — the run PAUSES here and the human's choice
        ({'selected': str, 'approved': bool, 'feedback': str}) arrives as this
        call's response. Do not call again; continue from what arrives.
    """
    user_id, session_id = session_key(tool_context) if tool_context is not None else (None, None)
    return _announce(agent_name, message, options=list(options or []),
                     session=(user_id, session_id))


def get_a2a_hitl_tools() -> list:
    """The HITL tools to attach when serving an agent over A2A."""
    return [LongRunningFunctionTool(request_approval),
            LongRunningFunctionTool(request_selection)]
