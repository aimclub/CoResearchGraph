"""HITL: human picks which orchestrator lanes to run (no query classification)."""

from __future__ import annotations

from typing import Any, Optional

from google.genai import types as genai_types

from CoScientist.config import get_settings
from CoScientist.graph.session_scope import session_key
from CoScientist.hitl.handler import AbstractHITLHandler
from CoScientist.hitl.models import HITLAction, HITLRequest

# Lane order is the only sequencing the orchestrator may use.
LANES = (
    ("research", "ResearchAgent"),
    ("hypotheses", "HypothesesAgent"),
    ("experiments", "ExperimentModuleAgent"),
)

OPTION_CUSTOM = "свой набор"
OPTION_SKIP = "пропустить"
SELECT_OPTIONS = (OPTION_CUSTOM, OPTION_SKIP)

STATE_KEY = "pipeline_scope"
DIRECTIVE_KEY = "pipeline_scope_directive"
FORM_BLOCK = "пункты пайплайна"

# Same tokens as WebSettings env flags, plus "да" for the Russian form.
_TRUTHY = frozenset({"true", "1", "yes", "да"})

_SELECT_MESSAGE = (
    "Какие пункты пайплайна запустить?\n"
    "«свой набор» — отметить research / hypotheses / experiments "
    "(по умолчанию все нет).\n"
    "«пропустить» — оркестратор решит сам, как сейчас."
)
_FORM_INTRO = (
    "По умолчанию все пункты выключены (нет). Переключите нужные на да. "
    "Если все нет — оркестратор решит сам."
)

# Same four-basket classifier the orchestrator uses when the human skips HITL.
# Custom HITL replaces this whole block with one basket — never stacked on top.
_COMMIT_LINE = (
    "   - Subordinates (ResearchAgent, HypothesesAgent) manage their own research graph commits. "
    "Do NOT call research_commit yourself with Evidence or Hypothesis nodes."
)
BASKET_FULL = (
    "   - Full-cycle scientific research (requesting literature + hypotheses + computational experiment/data/generation, or multi-step research plans with generation/models/validation) →\n"
    "     Execute all three stages in sequence without stopping or skipping:\n"
    "     1. ResearchAgent — review literature, known ligands/chemistry, and background data.\n"
    "     2. HypothesesAgent — formulate and commit scientific hypotheses into the research graph.\n"
    "     3. ExperimentModuleAgent — immediately call ExperimentModuleAgent after HypothesesAgent finishes to plan and execute the computational experiment stage. Pass the full goal and context to ExperimentModuleAgent, including collected literature evidence, data refs and missing inputs. Literature collection stays with ResearchAgent outside the module; never ask the module to repeat it or substitute Coder for literature search. Do NOT finish your turn or provide a final answer before ExperimentModuleAgent runs!"
)
BASKET_LITERATURE = (
    "   - Literature search / paper reviews / scientific knowledge questions (no experiment requested) →\n"
    "     Call ResearchAgent. When ResearchAgent completes, synthesize the literature findings\n"
    "     in your final answer and STOP. Do NOT call HypothesesAgent or ExperimentModuleAgent."
)
BASKET_HYPOTHESES = (
    "   - Hypotheses formulation / ideas only (no experiment requested) →\n"
    "     Call HypothesesAgent. When HypothesesAgent completes, summarize the formulated hypotheses\n"
    "     in your final answer and STOP. Do NOT call ExperimentModuleAgent."
)
BASKET_COMPUTE = (
    "   - Pure computational tasks, molecule generation/docking, simulations, code (where literature/hypotheses are not asked or already given) →\n"
    "     Call ExperimentModuleAgent directly once."
)
SCOPE_BASKETS_ALL = (
    "Match the delegation strictly to the user's requested scope:\n"
    f"{BASKET_FULL}\n"
    f"{BASKET_LITERATURE}\n"
    f"{BASKET_HYPOTHESES}\n"
    f"{BASKET_COMPUTE}\n"
    f"{_COMMIT_LINE}"
)
# Skip HITL: retrieve lives here so the static orchestrator prompt is not dual skip/custom.
SKIP_DIRECTIVE = (
    "ALWAYS call `retrieve_tools` FIRST, then pick one basket from the directive below. "
    "If suitable tools exist the task IS executable as an experiment — send those stages "
    "to ExperimentModuleAgent.\n"
    "   Retrieved tools accumulate — do not repeat near-identical queries, and "
    "never invent server ids (`get_server_info` only takes ids it returned).\n"
    f"{SCOPE_BASKETS_ALL}"
)


def as_bool(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY


def flatten_form_values(form_values: Optional[dict]) -> dict[str, Any]:
    """Collapse `{block: {field: value}}` (and a flat dict) into field answers."""
    if not form_values:
        return {}
    answers: dict[str, Any] = {}
    for key, val in form_values.items():
        if isinstance(val, dict):
            answers.update(val)
        else:
            answers[key] = val
    return answers


def pack_scope(form_values: Optional[dict]) -> Optional[dict]:
    answers = flatten_form_values(form_values)
    scope = {key: as_bool(answers.get(key)) for key, _ in LANES}
    if not any(scope.values()):
        return None
    scope["source"] = "hitl"
    return scope


def named_agents(scope: dict) -> list[str]:
    return [agent for key, agent in LANES if scope.get(key)]


def render_directive(scope: dict) -> str:
    """Custom HITL: orchestrate only the named lanes, in LANES order."""
    names = named_agents(scope)
    skipped = [agent for key, agent in LANES if not scope.get(key)]
    stages = "\n".join(f"     {i}. {name}" for i, name in enumerate(names, 1))
    skip_line = f"     Do NOT call {', '.join(skipped)}." if skipped else ""
    if "ExperimentModuleAgent" in names:
        retrieve = (
            "Call `retrieve_tools` before ExperimentModuleAgent. "
            "retrieve_tools must not add or drop agents.\n"
        )
    else:
        retrieve = "Do not call `retrieve_tools` to pick extra agents.\n"
    return (
        "Human-fixed scope. Do not re-classify the user text. "
        "Orchestrate using only the agents named below, in that order.\n"
        + retrieve
        + "   - Run these stages in sequence, then STOP:\n"
        + stages
        + (("\n" + skip_line) if skip_line else "")
        + "\n"
        + _COMMIT_LINE
    )


def scope_form() -> dict:
    return {
        "title": "Пункты пайплайна",
        "intro": _FORM_INTRO,
        "submit_label": "Продолжить",
        "allow_skip": False,
        "blocks": [
            {
                "title": FORM_BLOCK,
                "usage": "по умолчанию нет; переключите на да",
                "fields": [
                    {
                        "name": key,
                        "kind": "toggle",
                        "value": "нет",
                        "status": "нет",
                        "open": False,
                    }
                    for key, _ in LANES
                ],
            }
        ],
    }


DONE_KEY = "pipeline_scope_done"
_ROOT_GOAL_KEY = "orchestrator_root_goal"
_FRAME_KEY = "research_frame"


def next_named_agent(scope: dict, done: Optional[list] = None) -> Optional[str]:
    finished = {str(name) for name in (done or [])}
    for name in named_agents(scope):
        if name not in finished:
            return name
    return None


def _text_parts(content: object) -> str:
    parts = getattr(content, "parts", None) if content is not None else None
    return "\n".join(
        t.strip()
        for p in (parts or [])
        if (t := getattr(p, "text", "") or "").strip()
    ).strip()


def _canonical_ask(callback_context) -> str:
    state = getattr(callback_context, "state", None)
    getter = getattr(state, "get", None) if state is not None else None
    if callable(getter):
        root = getter(_ROOT_GOAL_KEY)
        if isinstance(root, str) and root.strip():
            return root.strip()
        raw = getter(_FRAME_KEY)
        if isinstance(raw, dict):
            text = raw.get("original_request")
            if isinstance(text, str) and text.strip():
                return text.strip()
        text = getattr(raw, "original_request", None) if raw is not None else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    return _text_parts(getattr(callback_context, "user_content", None))


def _function_names(llm_response) -> list[str]:
    content = getattr(llm_response, "content", None) if llm_response is not None else None
    parts = getattr(content, "parts", None) if content is not None else None
    names: list[str] = []
    for part in parts or []:
        name = getattr(getattr(part, "function_call", None), "name", None)
        if name:
            names.append(str(name))
    return names


def _force_named_agent(callback_context, name: str):
    from google.adk.models import LlmResponse

    ask = _canonical_ask(callback_context) or (
        "Continue the human-fixed pipeline scope."
    )
    return LlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part.from_function_call(name=name, args={"request": ask})],
        )
    )


def enforce_pipeline_scope_hops(callback_context, llm_response=None):
    """after_model: custom HITL — force the next named AgentTool if the model strayed."""
    state = getattr(callback_context, "state", None)
    getter = getattr(state, "get", None) if state is not None else None
    if not callable(getter):
        return None
    scope = getter(STATE_KEY)
    if not isinstance(scope, dict):
        return None
    from CoScientist.execution_control import current_run
    from CoScientist.experiments.outcome.reconciliation import (
        DispositionKind,
        reconcile_scientific_outcome,
    )

    handle = current_run()
    disposition = reconcile_scientific_outcome(
        state, current_run_id=handle.run_id if handle else None,
    )
    if disposition.kind is DispositionKind.COMPLETED_LIMITED:
        return None
    nxt = next_named_agent(scope, getter(DONE_KEY) or [])
    if not nxt:
        return None
    names = _function_names(llm_response)
    allowed = set(named_agents(scope))
    lane_names = {agent for _, agent in LANES}
    if any(name in lane_names and name not in allowed for name in names):
        return _force_named_agent(callback_context, nxt)
    if nxt in names:
        return None
    if "retrieve_tools" in names and nxt == "ExperimentModuleAgent":
        return None
    return _force_named_agent(callback_context, nxt)


def mark_pipeline_scope_lane(tool, args, tool_context, tool_response=None):
    """after_tool: close a named lane only after an accepted typed outcome."""
    name = getattr(tool, "name", "") or ""
    if name not in {agent for _, agent in LANES}:
        return None
    state = getattr(tool_context, "state", None)
    getter = getattr(state, "get", None) if state is not None else None
    if not callable(getter):
        return None

    # Result HITL may request one explicit, targeted redo.  The first module
    # return must therefore not close the experiments lane.  Consume the
    # request here, at the actual AgentTool boundary, and leave a one-shot
    # marker for the orchestrator callback.  A second return clears it even if
    # that AgentTool failed before reaching another review, preventing a loop.
    redo_key = "experiment_targeted_redo_pending"
    if name == "ExperimentModuleAgent":
        outcome = getter("experiment_module_outcome")
        targeted = (
            isinstance(outcome, dict)
            and outcome.get("resume_required") is True
            and outcome.get("reason") == "targeted_redo_requested"
        )
        if targeted:
            consumed = dict(outcome)
            consumed["resume_required"] = False
            consumed["resume_dispatched"] = True
            state["experiment_module_outcome"] = consumed
            state[redo_key] = {
                "selected_task_ids": list(consumed.get("selected_task_ids") or []),
                "affected_task_ids": list(consumed.get("affected_task_ids") or []),
            }
            done = [item for item in list(getter(DONE_KEY) or []) if item != name]
            state[DONE_KEY] = done
            return None
        if getter(redo_key):
            state[redo_key] = None

        from CoScientist.experiments.outcome.reconciliation import (
            accepted_stage_outcome,
        )

        if not accepted_stage_outcome(state, name):
            # A normal AgentTool return can mean plan pause, unresolved result
            # review, or an operational failure.  None is an accepted stage
            # outcome, so a stale DONE marker must not authorize completion.
            state[DONE_KEY] = [
                item for item in list(getter(DONE_KEY) or []) if item != name
            ]
            return None

    if not isinstance(getter(STATE_KEY), dict):
        return None
    done = list(getter(DONE_KEY) or [])
    if name not in done:
        done.append(name)
        state[DONE_KEY] = done
    return None


def _session_context(callback_context) -> dict:
    user_id, session_id = session_key(callback_context)
    return {"_session": {"user_id": user_id, "session_id": session_id}}


def make_ask_pipeline_scope_callback(handler: AbstractHITLHandler):
    """before_agent: ask once per session, write scope + directive, never cancel."""

    async def before_agent_callback(
        callback_context, llm_request=None
    ) -> Optional[genai_types.Content]:
        web = get_settings().web
        if not web.scope_hitl or not web.hitl_enabled:
            return None

        state = callback_context.state
        if state.get(STATE_KEY) or state.get(DIRECTIVE_KEY):
            return None

        agent_name = getattr(callback_context, "agent_name", "OrchestratorAgent")
        ctx = _session_context(callback_context)

        select = await handler.handle_request(
            HITLRequest(
                agent_name=agent_name,
                action_type=HITLAction.SELECT,
                message=_SELECT_MESSAGE,
                options=list(SELECT_OPTIONS),
                context=ctx,
                invoked_via="callback",
            )
        )
        if select.selected_option != OPTION_CUSTOM:
            state[DIRECTIVE_KEY] = SKIP_DIRECTIVE
            return None

        form = await handler.handle_request(
            HITLRequest(
                agent_name=agent_name,
                action_type=HITLAction.APPROVE,
                message="Отметьте пункты пайплайна.",
                form=scope_form(),
                context=ctx,
                invoked_via="callback",
            )
        )
        scope = pack_scope(form.form_values)
        if not scope:
            state[DIRECTIVE_KEY] = SKIP_DIRECTIVE
            return None

        state[STATE_KEY] = scope
        state[DIRECTIVE_KEY] = render_directive(scope)
        return None

    return before_agent_callback
