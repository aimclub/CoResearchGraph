"""Orchestrator helpers for ExperimentModuleAgent call shaping.

Two structural (never keyword-based) safety nets live here:

1. ``coalesce_experiment_module_calls`` — if the orchestrator accidentally fans
   out several ExperimentModuleAgent calls in one turn, merge them into a single
   self-contained brief so the module builds ONE ExperimentPlan.

2. ``suppress_experiment_module_after_completed`` — prevent re-entering the module
   after result HITL accepted the stage or if planning budget is exhausted.
"""

from __future__ import annotations

import logging
from hashlib import sha256
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types

logger = logging.getLogger(__name__)

_EM_NAME = "ExperimentModuleAgent"
#: Why the last plan or result review did not end in an approval. Owned by
#: CoScientist/experiments/review.py (imported lazily, like everything else
#: this module reaches for: importing review at module scope would pull the
#: whole experiment package into the orchestrator's callback chain).
_PAUSE_REASON_STATE_KEY = "experiment_review_pause_reason"
_TIMED_OUT_REASONS = {"plan_review_timeout", "result_review_timeout"}
# Set by research_init, or by ContextInit from the user's original_request.
_ROOT_GOAL_STATE_KEY = "orchestrator_root_goal"
_FRAME_STATE_KEY = "research_frame"
_TARGETED_REDO_PENDING_KEY = "experiment_targeted_redo_pending"
_RESUME_PENDING_STATE_KEY = "_execution_resume_pending"
_PLAN_RECOVERY_STATE_KEY = "experiment_plan_recovery_requested"
_CAPACITY_RECOVERY_STATE_KEY = "experiment_capacity_recovery_requested"
_RECOVERY_DISPATCH_STATE_KEY = "experiment_recovery_dispatch_pending"

_TURN_CLEAR_KEYS = (
    "experiment_context", "experiment_plan", "experiment_plan_view", "experiment_plan_critique",
    "experiment_plan_record_id",
    "experiment_runtime", "experiment_task_results", "experiment_summary",
    "experiment_artifacts_manifest", "experiment_last_route_response",
    "experiment_active_envelope", "experiment_plan_validation_errors",
    "experiment_plan_candidate", "experiment_plan_fallback_pending",
    "experiment_module_outcome",
    "experiment_plan_review_paused", _PAUSE_REASON_STATE_KEY,
    "experiment_plan_revision_count", "experiment_inventory_blocker_hits",
    "experiment_no_matching_tool", "experiment_execution_summary",
    "experiment_repo_candidates", "experiment_module_runs",
    "experiment_module_dispatched",
    "experiment_plan_last_schema_valid_candidate",
    "experiment_plan_last_executable_candidate",
    "experiment_plan_capacity_override",
    _CAPACITY_RECOVERY_STATE_KEY,
    _RECOVERY_DISPATCH_STATE_KEY,
    "scientific_limited_scope_acceptance",
    _PLAN_RECOVERY_STATE_KEY,
)


def _invocation_id(callback_context: CallbackContext) -> str:
    inv = getattr(callback_context, "_invocation_context", None) or getattr(
        callback_context, "invocation_context", None
    )
    return str(
        getattr(inv, "invocation_id", None)
        or getattr(callback_context, "invocation_id", None)
        or ""
    ).strip()


def prepare_experiment_user_turn(callback_context: CallbackContext) -> None:
    """Reset stale execution state once, before orchestrator suppression guards."""
    state = getattr(callback_context, "state", None)
    if state is None or not hasattr(state, "get") or not hasattr(state, "__setitem__"):
        return None
    user_text = _text_parts(getattr(callback_context, "user_content", None))
    if not user_text:
        return None
    invocation_id = _invocation_id(callback_context)
    turn_id = invocation_id or (
        "legacy:" + sha256(user_text.encode("utf-8")).hexdigest()
    )
    if state.get(_RESUME_PENDING_STATE_KEY):
        # A controller continuation is not a new scientific request. Bind the
        # new ADK invocation to the existing turn and consume only this marker;
        # root tasks, candidates, attempt ledgers and budgets remain untouched.
        state[_RESUME_PENDING_STATE_KEY] = False
        state["experiment_user_turn_id"] = turn_id
        state["experiment_force_new_run"] = False
        logger.info("EXPERIMENT_USER_TURN_RESUMED turn_id=%s", turn_id)
        return None
    if str(state.get("experiment_user_turn_id") or "") == turn_id:
        return None

    previous = {
        "user_turn_id": state.get("experiment_user_turn_id"),
        "source_request": state.get("experiment_source_request"),
        "context": state.get("experiment_context"),
        "plan": state.get("experiment_plan"),
        "plan_record_id": state.get("experiment_plan_record_id"),
        "runtime": state.get("experiment_runtime"),
        "task_results": state.get("experiment_task_results"),
        "artifacts_manifest": state.get("experiment_artifacts_manifest"),
        "summary": state.get("experiment_summary"),
    }
    if any(previous.get(key) for key in (
        "source_request", "plan", "runtime", "task_results", "artifacts_manifest", "summary"
    )):
        history = list(state.get("experiment_run_history") or [])
        history.append(previous)
        state["experiment_run_history"] = history[-20:]

    for key in _TURN_CLEAR_KEYS:
        if key in state:
            state[key] = None
    state["experiment_user_turn_id"] = turn_id
    state["experiment_source_request"] = user_text
    state["experiment_force_new_run"] = True
    logger.info("EXPERIMENT_USER_TURN_PREPARED turn_id=%s", turn_id)
    return None


def _spend(state: object, current_runs: int) -> None:
    """Let this dispatch through, one unit lighter.

    Always returns None, so it can be returned straight from the decision:
    an after_model callback that returns None leaves the response alone.
    """
    if hasattr(state, "__setitem__"):
        state["experiment_module_runs"] = current_runs + 1
    return None


def _text_parts(content: object) -> str:
    parts = getattr(content, "parts", None) if content is not None else None
    return "\n".join(
        t.strip() for p in (parts or [])
        if (t := getattr(p, "text", "") or "").strip()
    ).strip()


def _canonical_ask(callback_context: CallbackContext, fallback: str) -> str:
    """User's original ask — never a reworded Research/McpBuilder brief.

    Order: research_init goal, ContextInit frame.original_request, user_content, brief.
    """
    state = getattr(callback_context, "state", None)
    getter = getattr(state, "get", None) if state is not None else None
    if callable(getter):
        root = getter(_ROOT_GOAL_STATE_KEY)
        if isinstance(root, str) and root.strip():
            return root.strip()
        raw = getter(_FRAME_STATE_KEY)
        if isinstance(raw, dict):
            text = raw.get("original_request")
            if isinstance(text, str) and text.strip():
                return text.strip()
        text = getattr(raw, "original_request", None) if raw is not None else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    user = _text_parts(getattr(callback_context, "user_content", None))
    return user or fallback


def coalesce_experiment_module_calls(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """If the orchestrator fans out N ExperimentModuleAgent calls, keep one.

    Merges every ``request`` into a single self-contained brief so the module
    builds one ExperimentPlan instead of N interleaved runtimes.
    """
    state = getattr(callback_context, "state", None)
    content = getattr(llm_response, "content", None)
    parts = list(getattr(content, "parts", None) or [])
    runtime = state.get("experiment_runtime") if hasattr(state, "get") else None
    targeted_redo = (
        bool(state.get(_TARGETED_REDO_PENDING_KEY))
        if hasattr(state, "get") else False
    )
    recovery_requested = bool(
        (state.get(_PLAN_RECOVERY_STATE_KEY)
         or state.get(_CAPACITY_RECOVERY_STATE_KEY))
        if hasattr(state, "get") else False
    )
    if recovery_requested:
        already_called = any(
            getattr(getattr(part, "function_call", None), "name", None) == _EM_NAME
            for part in parts
        )
        state[_RECOVERY_DISPATCH_STATE_KEY] = True
        state[_CAPACITY_RECOVERY_STATE_KEY] = None
        if not already_called:
            request = _canonical_ask(callback_context, "Continue the saved experiment review.")
            return LlmResponse(content=types.Content(
                role=getattr(content, "role", None) or "model",
                parts=[types.Part.from_function_call(
                    name=_EM_NAME,
                    args={"request": request + (
                        "\n\nResume the saved deterministic review state. Do not "
                        "regenerate the root roadmap, reset revision counters, or "
                        "replay completed work."
                    )},
                )],
            ))
    if targeted_redo and isinstance(runtime, dict) and runtime.get("phase") == "execution":
        already_called = any(
            getattr(getattr(part, "function_call", None), "name", None) == _EM_NAME
            for part in parts
        )
        if not already_called:
            redo = state.get(_TARGETED_REDO_PENDING_KEY) or {}
            selected = list(redo.get("selected_task_ids") or []) if isinstance(redo, dict) else []
            affected = list(redo.get("affected_task_ids") or []) if isinstance(redo, dict) else []
            request = _canonical_ask(callback_context, "Continue the approved experiment.")
            suffix = (
                "\n\nResume only the explicitly approved targeted redo. "
                f"Selected tasks: {selected or 'recorded selection'}; "
                f"affected tasks: {affected or 'runtime-derived dependents'}. "
                "Preserve completed independent tasks and the cumulative attempt ledger."
            )
            state["experiment_module_dispatched"] = True
            logger.info("EXPERIMENT_TARGETED_REDO_FORCED selected=%s affected=%s", selected, affected)
            return LlmResponse(content=types.Content(
                role=getattr(content, "role", None) or "model",
                parts=[types.Part.from_function_call(
                    name=_EM_NAME, args={"request": request + suffix},
                )],
            ))
    if not parts:
        return None

    em_idxs: list[int] = []
    requests: list[str] = []
    for i, part in enumerate(parts):
        fc = getattr(part, "function_call", None)
        if fc is None or getattr(fc, "name", None) != _EM_NAME:
            continue
        em_idxs.append(i)
        args = dict(getattr(fc, "args", None) or {})
        req = args.get("request")
        if isinstance(req, str) and req.strip():
            requests.append(req.strip())

    # Marks that the module was asked for at all; the BUDGET is spent in
    # suppress_experiment_module_after_completed, which is where it is
    # checked. Counting here meant a dispatch paid for itself before being
    # judged, and the judge then refused it for being over budget.
    if em_idxs and hasattr(state, "__setitem__"):
        state["experiment_module_dispatched"] = True

    if len(em_idxs) <= 1:
        return None

    canonical = _canonical_ask(callback_context, "")
    merged = canonical or (
        "Complete the following computational experiment as ONE stage. "
        "Build a single ExperimentPlan covering all items below in order "
        "(with depends_on / artifact handoff as needed):\n\n"
        + "\n\n".join(r for r in requests)
    )
    keep_i = em_idxs[0]
    keep_fc = getattr(parts[keep_i], "function_call", None)
    if keep_fc is not None:
        keep_fc.args = dict(getattr(keep_fc, "args", None) or {})
        keep_fc.args["request"] = merged

    drop = set(em_idxs[1:])
    content.parts = [p for i, p in enumerate(parts) if i not in drop]
    agent = getattr(callback_context, "agent_name", None) or "orchestrator"
    logger.warning(
        "[%s] coalesced %d ExperimentModuleAgent calls into 1",
        agent,
        len(em_idxs),
    )
    return None  # in-place mutation is enough


def suppress_experiment_module_after_completed(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """after_model: do not re-enter the module after result HITL accepted the
    stage, or if planning is paused, or if the dispatch budget is spent — and
    spend a unit of that budget for a dispatch this lets through."""
    state = getattr(callback_context, "state", None)
    getter = getattr(state, "get", None) if state is not None else None
    if not callable(getter):
        return None

    # Find the dispatch first: with nothing to judge there is nothing to spend
    # either, and this callback runs after every model turn.
    content = getattr(llm_response, "content", None)
    parts = list(getattr(content, "parts", None) or [])
    em_idxs = [
        i for i, part in enumerate(parts)
        if getattr(getattr(part, "function_call", None), "name", None) == _EM_NAME
    ]
    if not em_idxs:
        return None

    runtime = getter("experiment_runtime")
    plan_paused = bool(getter("experiment_plan_review_paused"))
    fallback_pending = bool(getter("experiment_plan_fallback_pending"))
    targeted_redo_pending = bool(getter(_TARGETED_REDO_PENDING_KEY))
    recovery_dispatch_pending = bool(getter(_RECOVERY_DISPATCH_STATE_KEY))
    is_completed = isinstance(runtime, dict) and runtime.get("phase") == "completed"
    from CoScientist.experiments.outcome.reconciliation import accepted_stage_outcome
    accepted_terminal = accepted_stage_outcome(state, _EM_NAME)
    module_outcome = getter("experiment_module_outcome")
    typed_result_unaccepted = bool(
        is_completed
        and isinstance(module_outcome, dict)
        and not accepted_terminal
        and (
            str(module_outcome.get("stage") or "") == "result_review"
            or str(module_outcome.get("status") or "")
            in {"awaiting_human", "blocked", "rejected", "paused"}
        )
    )
    from CoScientist.config import get_settings
    try:
        current_runs = int(getter("experiment_module_runs") or 0)
    except (TypeError, ValueError):
        current_runs = 0
    max_em_runs = get_settings().experiments.max_replans
    budget_exhausted = current_runs >= max_em_runs

    if recovery_dispatch_pending:
        # One explicitly requested continuation may cross the ordinary module
        # redispatch gate without consuming an automatic replan unit.
        state[_RECOVERY_DISPATCH_STATE_KEY] = None
        return None
    if targeted_redo_pending:
        # Explicit result-HITL redo is not an automatic replan and is consumed
        # at the next ExperimentModuleAgent return.  Do not charge or block it
        # with the automatic module-run budget.
        return None
    if accepted_terminal:
        # Result approval is the terminality authority, even when the accepted
        # scientific result contains failed/skipped tasks.  Such a result is a
        # limited outcome, not permission for an autonomous full-stage retry.
        # Explicit targeted-redo and recovery markers were handled above.
        pass
    elif typed_result_unaccepted:
        # Execution ending is not the same as result acceptance.  Keep this
        # bounded review state intact; a new autonomous experiment hop cannot
        # manufacture the missing human/typed decision.
        pass
    elif fallback_pending:
        pass
    elif plan_paused:
        pass
    elif budget_exhausted:
        pass
    elif is_completed:
        from CoScientist.experiments.review import result_tasks_ok
        if not result_tasks_ok(runtime):
            return _spend(state, current_runs)
    else:
        return _spend(state, current_runs)

    # Refused: discard the model's text as well as the call. Keeping prose such
    # as "I will retry now" after removing its function call produced a false
    # promise. Preserve unrelated calls, then append one truthful explanation
    # and publish the refusal as structured state.
    reason = str(getter(_PAUSE_REASON_STATE_KEY) or "")
    if accepted_terminal:
        refusal_reason = "accepted_experiment_stage_already_terminal"
        summary = (
            "The accepted experiment stage is already terminal; no duplicate "
            "ExperimentModuleAgent call was started."
        )
    elif typed_result_unaccepted:
        refusal_reason = str(module_outcome.get("reason") or "result_review_unaccepted")
        summary = (
            "Experiment execution ended, but its result review is still unaccepted"
            + (f" ({refusal_reason})" if refusal_reason else "")
            + ". No duplicate ExperimentModuleAgent call was started."
        )
    elif fallback_pending:
        refusal_reason = "plan_revision_budget_exhausted_pending_human"
        summary = (
            "Experiment plan is waiting for an explicit human decision after "
            "automatic revisions were exhausted. No new plan or execution was started."
        )
    elif plan_paused:
        refusal_reason = reason or "plan_review_paused"
        summary = (
            "Experiment plan review is paused"
            + (f" ({refusal_reason})" if refusal_reason else "")
            + ". No retry or second plan was started."
        )
    elif budget_exhausted:
        refusal_reason = reason or "experiment_module_dispatch_budget_exhausted"
        if reason in _TIMED_OUT_REASONS:
            what = "plan" if reason == "plan_review_timeout" else "result"
            waiting = "is waiting" if what == "plan" else "was waiting"
            summary = (
                f"The experiment {what} {waiting} for a human approval; "
                f"automatic redispatch was refused after {current_runs}/{max_em_runs} attempts used."
            )
        else:
            summary = (
                f"Automatic ExperimentModuleAgent redispatch was refused after "
                f"the maximum attempt budget ({current_runs}/{max_em_runs}). The scientific work is not "
                "thereby complete."
            )
    else:
        refusal_reason = "accepted_experiment_stage_already_terminal"
        summary = (
            "The accepted experiment stage is already terminal; no duplicate "
            "ExperimentModuleAgent call was started."
        )

    if hasattr(state, "__setitem__"):
        state["experiment_module_dispatch_refusal"] = {
            "status": "refused",
            "reason": refusal_reason,
            "requested_calls": len(em_idxs),
            "module_runs": current_runs,
            "module_run_limit": max_em_runs,
        }
        prior = getter("experiment_module_outcome")
        if budget_exhausted and not typed_result_unaccepted and not (
            isinstance(prior, dict)
            and prior.get("status") in {"completed", "not_applicable"}
        ):
            state["experiment_module_outcome"] = {
                "status": "blocked",
                "stage": "plan_review",
                "reason": refusal_reason,
                "accepted": False,
            }

    dropped = set(em_idxs)
    kept = [
        part for index, part in enumerate(parts)
        if index not in dropped
        and not str(getattr(part, "text", "") or "").strip()
    ]
    kept.append(types.Part(text=(
        f"[EXPERIMENT_MODULE_DISPATCH_REFUSED reason={refusal_reason}] {summary}"
    )))
    content.parts = kept
    logger.warning(
        "[%s] suppressed ExperimentModuleAgent: runs=%d/%d plan_paused=%s "
        "fallback_pending=%s pause_reason=%s",
        getattr(callback_context, "agent_name", None) or "orchestrator",
        current_runs,
        max_em_runs,
        plan_paused,
        fallback_pending,
        getter(_PAUSE_REASON_STATE_KEY) or "-",
    )
    return None


__all__ = [
    "prepare_experiment_user_turn",
    "coalesce_experiment_module_calls",
    "suppress_experiment_module_after_completed",
]
