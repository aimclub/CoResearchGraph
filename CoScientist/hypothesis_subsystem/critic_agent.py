"""
Critic agents for the orchestrator.

Two callbacks are wired onto the OrchestratorAgent:

  * `pre_action_critique`   -> after_model_callback
        Runs after the orchestrator LLM has decided on its next action(s)
        but BEFORE those actions execute. Inspects the chosen function
        calls (which sub-agent, with what args) in light of the task and
        history, and votes:
          - APPROVE -> let the calls execute as-is
          - REVISE  -> mutate the args in place and let the calls execute
          - REJECT  -> replace the response with a text message that
                       describes why the plan was rejected; the
                       orchestrator will then re-decide on its next turn

  * `post_action_critique`  -> after_tool_callback
        Runs after a sub-agent (tool) returns. Evaluates the result and
        annotates it with a `_critic` directive when the result is
        insufficient or wrong, leaving the original payload intact.

Both critics are themselves LLM calls returning strict JSON.
"""

from __future__ import annotations

from opik import track

import json
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional

import litellm
from CoScientist.hypothesis_subsystem.moosechem_tool import _extract_json
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from CoScientist.config import get_settings


_settings = get_settings()
_CRITIC_MODEL = _settings.llm.main_model

# Pre-action critic system prompt — used by pre_action_critique() below.
_PRE_ACTION_CRITIC_INSTRUCTION = """
You are the PRE-ACTION CRITIC for a scientific multi-agent orchestrator.
Evaluate the orchestrator's proposed next action(s) and vote APPROVE, REVISE, or REJECT.
Return strict JSON with keys: verdict, feedback, revised_calls (for revise).
"""

# Post-action critic system prompt — used by post_action_critique() below.
_POST_ACTION_CRITIC_INSTRUCTION = """
You are the POST-ACTION CRITIC for a scientific multi-agent orchestrator.
Evaluate a sub-agent's result and vote sufficient, insufficient, or wrong.
Return strict JSON with keys: verdict, feedback.
"""


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
                responses_by_id[getattr(fr, "id", "")] = getattr(fr, "response", None)

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


# ---------------------------------------------------------------------------
# LLM critic invocation
# ---------------------------------------------------------------------------
def _invoke_critic_llm(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Returns parsed JSON dict; on any failure returns {} (permissive default)."""
    try:
        resp = litellm.completion(
            model=_CRITIC_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        raw = resp["choices"][0]["message"]["content"]
        return _extract_json(raw)
    except Exception as e:
        print(f"[Critic] LLM call failed ({e!r}); defaulting to permissive verdict.")
        return {}


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
        try:
            fc.args = new_args
        except Exception:  
            try:
                object.__setattr__(fc, "args", new_args)
            except Exception:  
                pass

@track(name="pre_action_critique")
def pre_action_critique(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """after_model_callback for the OrchestratorAgent."""
    pending = _extract_pending_calls(llm_response)

    # No tool calls in this response -> nothing to critique.
    if not pending:
        return None

    contents = _session_contents(callback_context)
    user_task = callback_context.user_content.parts[0].text
    trajectory = _extract_completed_trajectory(contents)

    user_prompt = (
        f"ORIGINAL TASK:\n{user_task}\n\n"
        f"COMPLETED TRAJECTORY:\n{_format_trajectory(trajectory)}\n\n"
        f"PROPOSED NEXT ACTION(S) (not yet executed):\n{_format_pending_calls(pending)}\n\n"
        "Decide whether to approve, revise, or reject these proposed actions. "
        "Respond as strict JSON."
    )

    print(f"pre action critic invoked with such prompt: {user_prompt}")

    payload = _invoke_critic_llm(_PRE_ACTION_CRITIC_INSTRUCTION, user_prompt)
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

    print(f"pre action critic returned: {payload}")
    if verdict_raw == PreVerdict.APPROVE.value:
        print(f"pre action critic returned None")
        return None

    if verdict_raw == PreVerdict.REVISE.value:
        if revised_calls:
            _apply_revisions(pending, revised_calls)
        if feedback and llm_response.content and llm_response.content.parts:
            llm_response.content.parts.insert(
                0,
                types.Part(text=f"[CRITIC REVISION]: {feedback}", thought=True),
            )
        print(f"pre action critic returned revision: {feedback}")
        return None

    if verdict_raw == PreVerdict.REJECT.value:
        msg = (
            "I am rejecting my own proposed action(s). "
            f"Reason: {feedback or 'the plan does not advance the task'}. "
            "I will reconsider which agent to call and with what arguments, "
            "given the original task and the completed trajectory so far."
        )
        print(f"pre action critic rejected with msg {msg}")
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=msg)])
        )

    return None


# ---------------------------------------------------------------------------
# Post-action critic  (after_tool_callback)
# ---------------------------------------------------------------------------
@track(name="post_action_critique")
def post_action_critique(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """after_tool_callback for the OrchestratorAgent."""
    user_prompt = (
        f"TOOL CALLED: {tool.name}\n"
        f"ARGS: {_truncate(args, 800)}\n"
        f"RESULT: {_truncate(tool_response, 2500)}\n\n"
        "Evaluate whether this result is sufficient to advance the task, "
        "needs refinement, or is wrong. Respond as strict JSON."
    )

    print(f'Post action critic invoked with {user_prompt}')
    payload = _invoke_critic_llm(_POST_ACTION_CRITIC_INSTRUCTION, user_prompt)
    verdict_raw = (payload.get("verdict") or "sufficient").lower().strip()
    feedback = (payload.get("feedback") or "").strip()
    print(f'Post action critic returned with {payload}')
    state = tool_context.state
    history = state.get("critic_post_history", [])
    history.append(
        {"tool": tool.name, "verdict": verdict_raw, "feedback": feedback}
    )
    state["critic_post_history"] = history

    if verdict_raw == PostVerdict.SUFFICIENT.value:
        print(f'post action critic returned None')
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
        print(f'post action critic returned insufficient {annotated}')
        return annotated

    if verdict_raw == PostVerdict.WRONG.value:
        annotated["_critic"] = {
            "verdict": "wrong",
            "directive": "REPLAN",
            "feedback": feedback
            or "Result does not address the task; re-plan from scratch.",
        }
        print(f'post action critic returned wrong {annotated}')
        return annotated

    return None


# ============================================================================
# HypothesisCriticAgent — internal subsystem critic (used by LoopCoordinator)
# ============================================================================
# These classes provide the contract that HypothesisLoopCoordinator imports.
# They are separate from the orchestrator-level pre/post action critics above.

from dataclasses import dataclass, field as dc_field

_HYPOTHESIS_CRITIC_SYSTEM_PROMPT = """You are a rigorous scientific hypothesis critic. Your job is to evaluate a single
hypothesis across five dimensions and return a structured JSON verdict.

## Dimensions (score each 0-2)
- **verifiability** (0-2): Are refutation conditions concrete, measurable, falsifiable?
  2 = can be tested with EXISTING tools (if a tool catalog is provided).
  1 = falsifiable in principle but no matching tool available.
  0 = unfalsifiable / tautological.
- **tool_coverage** (0-2): How many of the hypothesis's required tools match available
  validation tools? 2 = all tools available, 1 = partial match, 0 = none.
- **consistency** (0-2): Does reasoning align with evidence? No logical gaps?
- **specificity** (0-2): Are variables well-defined with units/scales? Domain clearly scoped?
- **novelty** (0-2): Is the claim original vs known approaches? Distinguished from alternatives?

## Passing threshold (RELAXED)
A hypothesis passes when BOTH conditions hold:
- Sum of all five scores ≥ 6 out of 10 — the hypothesis is good "overall".
- No single dimension scores 0 (min ≥ 1) — no fatal failure on any criterion.

If sum ≥ 6 but one dimension is 0 → `tools_available: true`, passed: false (REVISE — fix the zero).
If sum < 6 → `tools_available: false`, passed: false (REJECT — fundamental issues).
Only when BOTH sum ≥ 6 AND min ≥ 1 → `tools_available: true`, passed: true (APPROVE).

## Output format (strict JSON only, no markdown)
{
  "passed": true,
  "scores": {"verifiability": 2, "tool_coverage": 2, "consistency": 1, "specificity": 2, "novelty": 0},
  "feedback": "Specific actionable critique. For revise: what to fix. For reject: why unfixable.",
  "tools_available": true,
  "tool_request": {}
}"""


@dataclass
class HypothesisInput:
    """Input for the HypothesisCriticAgent — minimal subset of a full Hypothesis."""
    id: str
    claim: str
    domain: str
    variables: str       # JSON string with independent/dependent variable summaries
    verification_plan: str
    tools: list
    strategy_type: str = ""


@dataclass
class HypothesisCriticResult:
    """Structured output from HypothesisCriticAgent.critique_one()."""
    passed: bool
    scores: dict          # {"verifiability": 0-2, "consistency": 0-2, ...}
    feedback: str = ""
    tools_available: bool = True
    tool_request: dict = dc_field(default_factory=dict)


class RAGClient:
    """Minimal RAG client stub.

    Provides a :meth:`query` method so HypothesisCriticAgent can optionally
    enrich its evaluation with external context. The default implementation
    returns an empty list — deploy with a real RAG backend as needed.
    """

    def query(self, text: str, top_k: int = 3) -> list:
        """Return relevant context chunks for the given query text."""
        return []


@track(name="hypothesis_critique_one")
class HypothesisCriticAgent:
    """Internal subsystem critic — evaluates one hypothesis at a time.

    Called by :class:`HypothesisLoopCoordinator` during the Generator↔Critic
    refinement loop. Uses the same litellm path as the orchestrator critics
    but with a hypothesis-specific evaluation prompt.

    Args:
        rag_client: Optional RAG client for evidence enrichment.
        model: LLM model identifier (defaults to _CRITIC_MODEL).
    """

    def __init__(self, rag_client: RAGClient | None = None, model: str | None = None):
        self._rag = rag_client or RAGClient()
        self._model = model or _CRITIC_MODEL

    def critique_one(self, hypothesis: HypothesisInput) -> HypothesisCriticResult:
        """Evaluate a single hypothesis and return a structured verdict.

        Args:
            hypothesis: The hypothesis to evaluate.

        Returns:
            HypothesisCriticResult with passed/scores/feedback.
        """
        variables_str = hypothesis.variables or "{}"
        user_prompt = (
            f"HYPOTHESIS ID: {hypothesis.id}\n"
            f"CLAIM: {hypothesis.claim}\n"
            f"DOMAIN: {hypothesis.domain}\n"
            f"VARIABLES: {variables_str}\n"
            f"VERIFICATION PLAN: {hypothesis.verification_plan}\n"
            f"TOOLS: {', '.join(hypothesis.tools) if hypothesis.tools else 'none'}\n"
            f"STRATEGY: {hypothesis.strategy_type}\n\n"
            "Evaluate across all five dimensions. Return strict JSON."
        )

        try:
            resp = litellm.completion(
                model=self._model,
                messages=[
                    {"role": "system", "content": _HYPOTHESIS_CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            raw = resp["choices"][0]["message"]["content"]
            payload = _extract_json(raw)

            return HypothesisCriticResult(
                passed=bool(payload.get("passed", False)),
                scores=payload.get("scores", {}),
                feedback=(payload.get("feedback") or "").strip(),
                tools_available=bool(payload.get("tools_available", True)),
                tool_request=payload.get("tool_request", {}),
            )
        except Exception as exc:
            print(f"[HypothesisCriticAgent] LLM call failed ({exc!r}); defaulting to permissive pass.")
            return HypothesisCriticResult(
                passed=True,
                scores={
                    "verifiability": 2, "tool_coverage": 2, "consistency": 2,
                    "specificity": 2, "novelty": 2,
                },
                feedback=f"Critic LLM error: {exc}",
            )