"""
Critic callbacks for the orchestrator.

Two callback FACTORIES are wired onto the OrchestratorAgent via system.yaml:

  * `make_pre_action_critique(instruction)`   -> after_model_callback
        Runs after the orchestrator LLM has decided on its next action(s)
        but BEFORE those actions execute. Inspects the chosen function
        calls (which sub-agent, with what args) in light of the task and
        history, and votes:
          - APPROVE -> let the calls execute as-is
          - REVISE  -> mutate the args in place and let the calls execute
          - REJECT  -> replace the response with a text message that
                       describes why the plan was rejected; the
                       orchestrator will then re-decide on its next turn

  * `make_post_action_critique(instruction)`  -> after_tool_callback
        Runs after a sub-agent (tool) returns. Evaluates the result and
        annotates it with a `_critic` directive when the result is
        insufficient or wrong, leaving the original payload intact.

A third factory is wired onto the PLANNER instead, independently of the two
above (system.yaml -> ``PlannerAgent.critic``):

  * `make_plan_critique(instruction)`         -> SessionAgent.plan_critic
        Reviews the roadmap the planner just registered, BEFORE it is accepted
        (and before the human sees it). Returns feedback text to send back for
        one rewrite, or None to accept the plan. The revision budget lives in
        the SessionAgent (`critic_max_rounds`, default 1), not here.

All three critics are themselves LLM calls returning strict JSON. They are
factories (not module-level callbacks) because their system prompts embed the
CURRENT roster — the assembler renders the prompt from the same config that
wires the agents and passes it in.
"""

from __future__ import annotations
from google.adk.agents import callback_context

from opik import track

import asyncio
import json
import logging
import os
import time
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import litellm
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from CoScientist.agents.callbacks.json_output import _extract_json
from CoScientist.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


# ---------------------------------------------------------------------------
# Verdict enums
# ---------------------------------------------------------------------------
class PreVerdict(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


class PostVerdict(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    WRONG = "wrong"


class PlanVerdict(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"


# ---------------------------------------------------------------------------
# Trajectory parsing (from session history on the callback context)
# ---------------------------------------------------------------------------
def _session_contents(callback_context: CallbackContext) -> List[types.Content]:
    """
    Best-effort extraction of the session's message history from the
    callback context. Falls back to an empty list if the internal layout
    differs across ADK versions.
    """
    inv = getattr(callback_context, "_invocation_context", None) or getattr(
        callback_context, "invocation_context", None
    )
    if inv is None:
        return []
    session = getattr(inv, "session", None)
    if session is None:
        return []
    events = getattr(session, "events", None) or []
    contents: List[types.Content] = []
    for ev in events:
        c = getattr(ev, "content", None)
        if c is not None and getattr(c, "parts", None):
            contents.append(c)
    return contents


def _extract_completed_trajectory(
    contents: List[types.Content],
) -> List[Dict[str, Any]]:
    """
    Pair every function_call with its matching function_response. Calls
    without a response yet are skipped — they belong to the in-flight
    decision the pre-critic is currently evaluating.
    """
    responses_by_id: Dict[str, Any] = {}
    for c in contents:
        if c.role != "user" or not c.parts:
            continue
        for p in c.parts:
            fr = getattr(p, "function_response", None)
            if fr is not None:
                fr_id = getattr(fr, "id", "")
                # Skip responses without an id — otherwise they all collapse to
                # the "" key and overwrite each other, mis-pairing calls.
                if fr_id:
                    responses_by_id[fr_id] = getattr(fr, "response", None)

    trajectory: List[Dict[str, Any]] = []
    pending_thought: Optional[str] = None

    for c in contents:
        if c.role != "model" or not c.parts:
            continue
        for p in c.parts:
            if getattr(p, "text", None) and getattr(p, "thought", False):
                pending_thought = p.text
                continue
            fc = getattr(p, "function_call", None)
            if fc is None:
                continue
            call_id = getattr(fc, "id", "") or ""
            if call_id not in responses_by_id:
                pending_thought = None
                continue
            trajectory.append(
                {
                    "thought": pending_thought,
                    "tool": getattr(fc, "name", ""),
                    "args": dict(getattr(fc, "args", {}) or {}),
                    "response": responses_by_id[call_id],
                }
            )
            pending_thought = None
    return trajectory


def _extract_pending_calls(llm_response: LlmResponse) -> List[Dict[str, Any]]:
    """
    Pull (thought?, function_call) pairs out of the orchestrator's freshly
    produced LlmResponse. These are the calls about to execute.
    """
    if llm_response is None or llm_response.content is None:
        return []
    parts = llm_response.content.parts or []
    pending_thought: Optional[str] = None
    calls: List[Dict[str, Any]] = []
    for p in parts:
        if getattr(p, "text", None) and getattr(p, "thought", False):
            pending_thought = p.text
            continue
        fc = getattr(p, "function_call", None)
        if fc is None:
            continue
        calls.append(
            {
                "thought": pending_thought,
                "tool": getattr(fc, "name", ""),
                "args": dict(getattr(fc, "args", {}) or {}),
                "_part": p,  # kept so REVISE can mutate args in place
            }
        )
        pending_thought = None
    return calls


# ---------------------------------------------------------------------------
# Formatting / truncation
# ---------------------------------------------------------------------------
def _truncate(value: Any, limit: int = 1500) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(s) <= limit:
        return s
    return s[:limit] + f"...[truncated {len(s) - limit} chars]"


def _format_trajectory(trajectory: List[Dict[str, Any]]) -> str:
    if not trajectory:
        return "(no completed prior steps)"
    lines: List[str] = []
    for i, step in enumerate(trajectory, 1):
        lines.append(f"--- Completed step {i} ---")
        if step.get("thought"):
            lines.append(f"Reasoning: {_truncate(step['thought'], 500)}")
        lines.append(f"Tool called: {step['tool']}")
        lines.append(f"Args: {_truncate(step['args'], 400)}")
        lines.append(f"Result: {_truncate(step['response'], 1000)}")
    return "\n".join(lines)


def _format_pending_calls(calls: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, call in enumerate(calls, 1):
        lines.append(f"--- Proposed action {i} ---")
        if call.get("thought"):
            lines.append(f"Reasoning: {_truncate(call['thought'], 500)}")
        lines.append(f"Tool to call: {call['tool']}")
        lines.append(f"Args: {_truncate(call['args'], 600)}")
    return "\n".join(lines)


def _usage_note(resp: Any) -> str:
    """`completion=N (reasoning=M) of cap` — what the answer actually spent."""
    cap = settings.critic.max_tokens
    usage = getattr(resp, "usage", None)
    if usage is None:
        return f"no usage reported, cap={cap}"
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None)
    return (
        f"completion_tokens={getattr(usage, 'completion_tokens', '?')}"
        + (f" (reasoning={reasoning})" if reasoning else "")
        + f" of cap={cap}"
    )


# ---------------------------------------------------------------------------
# LLM critic invocation
# ---------------------------------------------------------------------------
async def _invoke_critic_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Returns the critic's parsed JSON verdict; on any failure approves.

    Uses the async litellm API so the critic's network call does not block the
    orchestrator's event loop (the callbacks run inside it), under a hard
    wall-clock deadline so a stalled provider cannot park the run forever. 
    A timed-out or otherwise failed critic approves:
    a verdict nobody could produce must not hold up the work.
    """
    # Shares the agent tree's notion of a retryable upstream fault, and its
    # proxy pre-flight. Imported lazily to keep this module out of
    # `agents.common`'s import graph.
    from CoScientist.agents.common import (
        RetryingLiteLlm,
        _is_transient,
        _reasoning_kwargs,
    )

    cfg = settings.critic
    model = cfg.model or settings.llm.main_model
    # The critic builds its model here rather than through the assembler, so
    # system.yaml's `reasoning:` never reaches it — it follows settings.critic.
    reasoning = _reasoning_kwargs(model, cfg.reasoning)

    for attempt in range(1, cfg.max_attempts + 1):
        started = time.perf_counter()
        try:
            # `timeout=` is only httpx's per-read budget; `asyncio.timeout` is
            # what actually caps a provider that stalls mid-response.
            async with asyncio.timeout(cfg.timeout):
                resp = await litellm.acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=cfg.max_tokens,
                    timeout=cfg.http_timeout,
                    num_retries=0,
                    **reasoning,
                )

            from CoScientist.logging.metrics import record_completion
            record_completion(resp, model=model, agent="Critic")

            choice = resp["choices"][0]
            raw = choice["message"]["content"]
            payload = _extract_json(raw or "")
            if isinstance(payload, dict):
                return payload
            logger.warning(
                "[Critic] unparseable verdict (%r); finish_reason=%r, %s; defaulting to permissive verdict.",
                _truncate(raw, 300),
                choice.get("finish_reason"),
                _usage_note(resp),
            )
            return {"verdict": "approve"}
        except Exception as e:  # noqa: BLE001 — a critic never takes the run down
            deadline_blown = isinstance(e, TimeoutError)
            retryable = (
                attempt < cfg.max_attempts
                and not deadline_blown
                and _is_transient(e)
            )
            # Elapsed time and route are what tell a stalled provider apart from
            # a stalled transport after the fact: the deadline reports a bare
            # `TimeoutError()`, which on its own names neither.
            route = (
                f"via proxy {settings.services.proxy_url}"
                if settings.web.use_proxy
                else "direct"
            )
            logger.warning(
                "[Critic] LLM call failed after %.1fs (%s, attempt %d/%d, %r); %s",
                time.perf_counter() - started,
                route,
                attempt,
                cfg.max_attempts,
                e,
                "retrying." if retryable else "defaulting to permissive verdict.",
            )
            if not retryable:
                return {"verdict": "approve"}
            await asyncio.sleep(min(1.5 ** attempt, 8.0))

    return {"verdict": "approve"}


# ---------------------------------------------------------------------------
# Pre-action critic  (after_model_callback)
# ---------------------------------------------------------------------------
def _apply_revisions(
    pending: List[Dict[str, Any]], revised_calls: List[Dict[str, Any]]
) -> None:
    """
    Mutate the LlmResponse's function_call parts in place using the critic's
    revised args. Match by index — the critic is told to return one entry
    per proposed action in the same order.

    The critic may only override the VALUES of args the orchestrator already
    chose; it cannot drop existing keys or introduce new ones. Every sub-agent
    is an ``AgentTool`` that requires a ``request`` key, so a naive replace
    would let the critic strip it (KeyError at call time) or inject keys the
    tool does not accept. We therefore keep the original args and apply only
    overrides for keys that already exist.
    """
    for i, call in enumerate(pending):
        if i >= len(revised_calls):
            break
        rev = revised_calls[i]
        new_args = rev.get("args")
        if not isinstance(new_args, dict):
            continue
        part = call["_part"]
        fc = getattr(part, "function_call", None)
        if fc is None:
            continue
        original = dict(getattr(fc, "args", {}) or {})
        overrides = {k: v for k, v in new_args.items() if k in original}
        ignored = [k for k in new_args if k not in original]
        if ignored:
            logger.info(
                "[Critic] ignoring revision keys not in original args for %s: %s",
                call.get("tool"),
                ignored,
            )
        safe_args = {**original, **overrides}
        if safe_args == original:
            continue
        try:
            fc.args = safe_args
        except Exception:
            try:
                object.__setattr__(fc, "args", safe_args)
            except Exception:
                pass

def _format_retrieved_tools(state: Any) -> str:
    """Render the tools the orchestrator's retrieve_tools surfaced, so the critic
    can judge whether a chosen agent actually MATCHES an available tool.

    Reads `accumulated_tools` (what the orchestrator's own retrieve_tools calls
    accumulated this turn). Returns '' when nothing was retrieved.
    """
    try:
        tools = state.get("accumulated_tools") or []
    except Exception:  # noqa: BLE001 — state shape varies across ADK versions
        return ""
    if not tools:
        return ""
    lines: List[str] = []
    for t in tools[:20]:
        if not isinstance(t, dict):
            continue
        name = t.get("tool") or t.get("name") or t.get("tool_name") or "?"
        desc = t.get("description") or t.get("desc") or ""
        lines.append(f"- {name}: {_truncate(desc, 160)}")
    return "\n".join(lines)


def _first_text(content) -> str:
    """First text part of a Content, or '' if none (e.g. an image-only message)."""
    if content is None or not getattr(content, "parts", None):
        return ""
    for p in content.parts:
        if getattr(p, "text", None):
            return p.text
    return ""


def make_pre_action_critique(instruction: str) -> Callable:
    """Build the orchestrator's after_model_callback with the given critic prompt."""

    @track(name="pre_action_critique")
    async def pre_action_critique(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        """after_model_callback for the OrchestratorAgent."""
        pending = _extract_pending_calls(llm_response)

        # No tool calls in this response -> nothing to critique.
        if not pending:
            return None

        #######
        # Auto-approve task management tools to save LLM calls and prevent false rejections.
        MANAGEMENT_TOOLS = {"update_task_status", "request_approval"}
        if all(call.get("tool") in MANAGEMENT_TOOLS for call in pending):
            logger.info("pre_action_critique auto-approved management tools: %s", [c.get("tool") for c in pending])
            return None

        contents = _session_contents(callback_context)
        # user_content may be absent or start with a non-text part (image/DICOM upload).
        user_task = _first_text(getattr(callback_context, "user_content", None))
        trajectory = _extract_completed_trajectory(contents)

        # Tools the orchestrator's retrieve_tools surfaced — lets the critic judge
        # whether a chosen compute agent actually matches an available tool.
        retrieved = _format_retrieved_tools(callback_context.state)
        retrieved_block = (
            f"RETRIEVED TOOLS (from retrieve_tools this turn):\n{retrieved}\n\n"
            if retrieved else ""
        )

        user_prompt = (
            f"ORIGINAL TASK:\n{user_task}\n\n"
            f"COMPLETED TRAJECTORY:\n{_format_trajectory(trajectory)}\n\n"
            f"{retrieved_block}"
            f"PROPOSED NEXT ACTION(S) (not yet executed):\n{_format_pending_calls(pending)}\n\n"
            "Decide whether to approve, revise, or reject these proposed actions. "
            "Respond as strict JSON."
        )

        payload = await _invoke_critic_llm(instruction, user_prompt)
        verdict_raw = (payload.get("verdict") or "approve").lower().strip()
        feedback = (payload.get("feedback") or "").strip()
        revised_calls = payload.get("revised_calls") or []

        # Audit trail
        state = callback_context.state
        history = state.get("critic_pre_history", [])
        history.append(
            {
                "verdict": verdict_raw,
                "feedback": feedback,
                "proposed": [{"tool": c["tool"], "args": c["args"]} for c in pending],
            }
        )
        state["critic_pre_history"] = history

        if verdict_raw == PreVerdict.APPROVE.value:
            logger.info("pre action critic approved proposed action(s)")
            return None

        if verdict_raw == PreVerdict.REVISE.value:
            if revised_calls:
                _apply_revisions(pending, revised_calls)
            if feedback and llm_response.content and llm_response.content.parts:
                llm_response.content.parts.insert(
                    0,
                    types.Part(text=f"[CRITIC REVISION]: {feedback}", thought=True),
                )
            return None

        if verdict_raw == PreVerdict.REJECT.value:
            msg = (
                "I am rejecting my own proposed action(s). "
                f"Reason: {feedback or 'the plan does not advance the task'}. "
                "I will reconsider which agent to call and with what arguments, "
                "given the original task and the completed trajectory so far."
            )
            return LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text=msg)])
            )

        return None

    return pre_action_critique


# ---------------------------------------------------------------------------
# Post-action critic  (after_tool_callback)
# ---------------------------------------------------------------------------
def make_post_action_critique(instruction: str) -> Callable:
    """Build the orchestrator's after_tool_callback with the given critic prompt."""

    @track(name="post_action_critique")
    async def post_action_critique(
        tool: BaseTool,
        args: Dict[str, Any],
        tool_context: ToolContext,
        tool_response: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """after_tool_callback for the OrchestratorAgent."""
        user_prompt = (
            f"TOOL CALLED: {tool.name}\n"
            f"ARGS: {_truncate(args, 800)}\n"
            f"RESULT: {_truncate(tool_response, 3000)}\n\n"
            "Evaluate whether this result is sufficient to advance the task, "
            "needs refinement, or is wrong. Respond as strict JSON."
        )

        payload = await _invoke_critic_llm(instruction, user_prompt)
        verdict_raw = (payload.get("verdict") or "sufficient").lower().strip()
        feedback = (payload.get("feedback") or "").strip()
        state = tool_context.state
        history = state.get("critic_post_history", [])
        history.append(
            {"tool": tool.name, "verdict": verdict_raw, "feedback": feedback}
        )
        state["critic_post_history"] = history

        if verdict_raw == PostVerdict.SUFFICIENT.value:
            return None

        annotated = (
            deepcopy(tool_response)
            if isinstance(tool_response, dict)
            else {"result": tool_response}
        )

        if verdict_raw == PostVerdict.INSUFFICIENT.value:
            annotated["_critic"] = {
                "verdict": "insufficient",
                "directive": "REFINE",
                "feedback": feedback
                or "Result is incomplete; refine the query or call a different agent.",
            }
            return annotated

        if verdict_raw == PostVerdict.WRONG.value:
            annotated["_critic"] = {
                "verdict": "wrong",
                "directive": "REPLAN",
                "feedback": feedback
                or "Result does not address the task; re-plan from scratch.",
            }
            return annotated

        return None

    return post_action_critique


# ---------------------------------------------------------------------------
# Plan critic  (SessionAgent.plan_critic — the PLANNER's own critic)
# ---------------------------------------------------------------------------
def make_plan_critique(instruction: str) -> Callable:
    """Build the planner's plan critic with the given critic prompt.

    Not an ADK callback: an after_model/after_agent callback can only rewrite
    or replace an output, and a plan critique is worthless unless the PLANNER
    itself redoes the roadmap. The returned coroutine is handed to the
    SessionAgent, which owns the generate → review → revise loop and caps it
    (`critic_max_rounds`, default 1 — the critic gets one say).

    Contract: ``await plan_critique(task, plan) -> feedback | None``. A None
    (approve, empty feedback, or a failed LLM call) accepts the plan as-is.
    """

    @track(name="plan_critique")
    async def plan_critique(task: str, plan: str) -> Optional[str]:
        user_prompt = (
            f"ORIGINAL TASK:\n{task or '(not available)'}\n\n"
            f"PROPOSED PLAN (as registered, in execution order):\n{_truncate(plan, 6000)}\n\n"
            "Decide whether to approve this plan or send it back for its one "
            "revision. Respond as strict JSON."
        )


        payload = await _invoke_critic_llm(instruction, user_prompt)
        verdict_raw = (payload.get("verdict") or "approve").lower().strip()
        feedback = (payload.get("feedback") or "").strip()


        # Anything but an explicit, substantiated "revise" accepts the plan:
        # a revision round the critic cannot justify only costs a rewrite.
        if verdict_raw != PlanVerdict.REVISE.value or not feedback:
            return None
        return feedback

    return plan_critique