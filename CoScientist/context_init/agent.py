"""The pre-stage context-initialization agent.

``ContextInitSessionAgent`` is a ``SessionAgent`` that:

  1. runs its LLM to draft a ``ResearchFrame`` from the raw research question
     (``output_schema = research_frame``, stored under ``output_key``);
  2. shows the operator a STRUCTURED WEB FORM (one field per framing entity)
     through the HITL bridge, and folds the operator's answers back onto the
     frame — untouched fields keep the agent's drafted values (soft gate);
  3. seeds the confirmed frame into the Research Context Graph (the privileged
     init path) BEFORE the orchestrator runs (without duplicating the frame in chat).

In headless mode (no HITL handler) the base loop skips the review and step 3
still runs — the agent's drafted frame is seeded as-is.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types

from CoScientist.context_init.commit import seed_frame
from CoScientist.context_init.models import (
    BLOCK_I18N,
    FIELD_I18N,
    FrameOperation,
    ResearchFrame,
)
from CoScientist.context_init.operations import (
    OPS_FORM_BLOCK,
    fill_operations_if_missing,
)
from CoScientist.graph.research.store import get_research_graph
from CoScientist.graph.session_scope import session_key
from CoScientist.hitl.field_status import OPERATOR_STATUS, is_open
from CoScientist.hitl.models import (
    HITLAction,
    HITLDecisionSource,
    HITLRequest,
    HITLResponse,
)
from CoScientist.hitl.session_agent import SessionAgent

logger = logging.getLogger(__name__)

FRAME_STATE_KEY = "research_frame"
# ``research_frame`` is the agent output and may be present before the operator
# accepts it.  This separate marker is written only after the frame has been
# seeded, so a failed or interrupted first turn can still be retried.
FRAME_COMPLETED_STATE_KEY = "research_frame_initialized"
_FORM_INTRO = ("Заполните рамку исследования. Пустые поля агент заполнит "
               "рабочими значениями. Рамка задаёт стратегию: литературный "
               "поиск, дорогой или дешёвый эксперимент.")
_FORM_INTRO_EN = ("Fill in the research frame. The agent fills empty fields "
                  "with working values. The frame sets the strategy: literature "
                  "search, expensive or cheap experiment.")
_HITL_MESSAGE = "Подтвердите рамку исследования перед запуском."
_HITL_MESSAGE_EN = "Confirm the research frame before the run starts."
_OPS_BLOCK_USAGE = ("что исследование обязано дать на выходе. Это задачи "
                    "ИССЛЕДОВАНИЯ, а не эксперимента: каждая из них "
                    "разворачивается в один или несколько экспериментов на "
                    "этапе планирования. Отчёт задачей не считается")
_OPS_BLOCK_USAGE_EN = ("What the research must deliver. These are tasks of the "
                       "RESEARCH, not of an experiment: each becomes one or "
                       "more experiments at planning time. The report is not "
                       "one of them.")
_OPS_FIELD_PLACEHOLDER = {
    "en": ("Enter one deliverable the research must produce, or leave it empty "
           "so the agent derives the tasks from the ask."),
    "ru": ("Укажите один результат, который должно дать исследование, или "
           "оставьте поле пустым — агент выведет задачи из запроса."),
}


def _task_label(operation_id: str) -> Dict[str, str]:
    """«Задача 1 · OP-1» — the word for the reader, the id for the plan.

    The id is not decoration: the experiment planner writes it into
    `design.operation_ref`, and the plan critic refuses a plan that leaves an
    operation uncovered. An operator who sees «OP-1» in a plan card has to be
    able to find it here, so it stays — behind the word that says what it is.
    """
    number = operation_id.split("-")[-1].strip() or "?"
    return {"en": f"Task {number} · {operation_id}",
            "ru": f"Задача {number} · {operation_id}"}


def coerce_frame(value: Any) -> ResearchFrame:
    """Best-effort ResearchFrame from a model output / state value."""
    if isinstance(value, ResearchFrame):
        frame = value.normalized()
    else:
        if isinstance(value, str):
            value = json.loads(value)
        frame = ResearchFrame.model_validate(value).normalized()
    return fill_operations_if_missing(frame)


def frame_to_form(frame: ResearchFrame) -> Dict[str, Any]:
    """Build the HITLRequest.form payload the web UI renders as a form."""
    frame = frame.normalized()
    blocks: List[Dict[str, Any]] = []
    for b in frame.blocks:
        block_i18n = BLOCK_I18N.get(b.title, {})
        fields = []
        for f in b.fields:
            field_i18n = FIELD_I18N.get(f.name, {})
            fields.append({
                "name": f.name, "value": f.value, "status": f.status,
                "open": is_open(f.status),
                "label": field_i18n.get("label", {"en": f.name, "ru": f.name}),
                "placeholder": field_i18n.get("placeholder", {"en": "", "ru": ""}),
            })
        blocks.append({
            "title": b.title, "usage": b.usage,
            "title_i18n": block_i18n.get("title", {"en": b.title, "ru": b.title}),
            "usage_i18n": block_i18n.get("usage", {"en": b.usage, "ru": b.usage}),
            "fields": fields,
        })
    ops_fields = [
        {"name": op.operation_id, "value": op.statement,
         "status": "задано заказчиком", "open": False,
         "label": _task_label(op.operation_id),
         "placeholder": _OPS_FIELD_PLACEHOLDER}
        for op in frame.operations
    ]
    if not ops_fields:
        ops_fields = [{
            "name": "OP-1", "value": "", "status": "не задано", "open": True,
            "label": _task_label("OP-1"),
            "placeholder": _OPS_FIELD_PLACEHOLDER,
        }]
    blocks.append({
        "title": OPS_FORM_BLOCK,
        "usage": _OPS_BLOCK_USAGE,
        "title_i18n": {"en": "Research tasks", "ru": OPS_FORM_BLOCK},
        "usage_i18n": {"en": _OPS_BLOCK_USAGE_EN, "ru": _OPS_BLOCK_USAGE},
        "fields": ops_fields,
    })
    return {
        "kind": "research_frame",
        "title": "Рамка исследования",
        "title_i18n": {"en": "Research frame", "ru": "Рамка исследования"},
        "intro": _FORM_INTRO,
        "intro_i18n": {"en": _FORM_INTRO_EN, "ru": _FORM_INTRO},
        "message_i18n": {"en": _HITL_MESSAGE_EN, "ru": _HITL_MESSAGE},
        "blocks": blocks,
    }


def apply_form_values(frame: ResearchFrame,
                      form_values: Optional[Dict[str, Any]]) -> ResearchFrame:
    """Fold operator answers ({block: {field: value}}) onto the frame; the
    touched fields become «уточнено оператором», the rest keep their status."""
    if not form_values:
        return frame
    frame = frame.normalized()
    for b in frame.blocks:
        answers = form_values.get(b.title) or {}
        if not isinstance(answers, dict):
            continue
        for f in b.fields:
            val = answers.get(f.name)
            if val is None or not str(val).strip():
                continue
            f.value = str(val).strip()
            f.status = OPERATOR_STATUS
    answers = form_values.get(OPS_FORM_BLOCK) or {}
    if isinstance(answers, dict) and any(str(v).strip() for v in answers.values()):
        def _op_sort(name: str) -> int:
            match = re.match(r"OP-(\d+)$", str(name).strip(), re.I)
            return int(match.group(1)) if match else 10**6
        rows: List[FrameOperation] = []
        for name in sorted(answers, key=_op_sort):
            val = str(answers.get(name) or "").strip()
            if not val:
                continue
            rows.append(FrameOperation(operation_id=f"OP-{len(rows) + 1}", statement=val))
        if rows:
            frame.operations = rows
    return frame


def apply_review_response(frame: ResearchFrame,
                          response: HITLResponse) -> ResearchFrame:
    """Apply form values only when a person actually reviewed the frame.

    Auto mode materialises a form-shaped response from the proposed defaults so
    downstream consumers can proceed deterministically.  Those values were not
    entered or confirmed by an operator and therefore must keep their original
    provenance statuses.
    """
    if response.decision_source != HITLDecisionSource.HUMAN:
        return frame.normalized()
    return apply_form_values(frame, response.form_values)


def render_frame_summary(frame: ResearchFrame) -> str:
    """Compact readable summary (console review / chat publication)."""
    lines: List[str] = ["## Рамка исследования", ""]
    for b in frame.blocks:
        set_fields = b.set_fields()
        mark = "✓" if set_fields else "—"
        lines.append(f"{mark} **{b.title}**"
                     + (f": {len(set_fields)} поле(й)" if set_fields else " (пусто)"))
        for f in set_fields:
            lines.append(f"    - {f.name}: {f.value}")
    if frame.operations:
        lines.append(f"✓ **{OPS_FORM_BLOCK}**: {len(frame.operations)} слот(ов)")
        for op in frame.operations:
            lines.append(f"    - {_task_label(op.operation_id)['ru']}: "
                         f"{op.statement}")
    return "\n".join(lines)


def frame_is_initialized(state: Dict[str, Any]) -> bool:
    """Whether this session has already completed its one-time frame stage."""
    return bool(state.get(FRAME_COMPLETED_STATE_KEY))


class ContextInitSessionAgent(SessionAgent):
    """SessionAgent that confirms the frame via a web form and seeds the graph."""

    unfinished_max_rounds: int = 1

    def _frame(self, ctx: InvocationContext, output_text=None) -> ResearchFrame:
        frame = coerce_frame(ctx.session.state.get(self.output_key) or output_text)
        content = getattr(ctx, "user_content", None)
        source = "".join(part.text or "" for part in (getattr(content, "parts", None) or []))
        # The source belongs to the caller. The model must not rewrite the
        # text against which its own provenance and counts are checked.
        return frame.model_copy(update={"original_request": source}) if source else frame

    @staticmethod
    def _statement(frame: ResearchFrame):
        from CoScientist.requirements.routing import normalize_statement

        accepted_fields = {
            f"{block.title}.{field.name}": {
                "value": field.value,
                "accepted": field.status in {"задано заказчиком", OPERATOR_STATUS},
            }
            for block in frame.blocks for field in block.fields
        }
        return normalize_statement(
            frame.original_request, frame.statement_draft, frame_fields=accepted_fields,
        )

    def _unfinished_feedback_author(self) -> str:
        return self.name

    def _unfinished_feedback(self, ctx: InvocationContext) -> str | None:
        repair_input = {}
        try:
            frame = self._frame(ctx)
            repair_input = {"original_request": frame.original_request,
                            "statement_draft": frame.statement_draft.model_dump() if frame.statement_draft else None}
            statement = self._statement(frame)
            if frame.statement_draft is not None and not statement.rejected:
                # A real clarification is for the user, not another model pass.
                if statement.parts or frame.statement_draft.clarification is not None:
                    return None
            diagnostics = statement.rejected or [{"reason": "statement_draft needs requested parts or a sourced clarification"}]
        except (ValueError, TypeError) as exc:
            diagnostics = [{"reason": str(exc)}]
        return (
            "The statement schema/provenance validation failed. Correct the full ResearchFrame JSON once; "
            "this is internal validation, not a new user request. Preserve original_request and all user "
            "requirements and conditional clauses. Do not invent missing counts: unspecified values are null, "
            "and numeric volume fields require literal digits in their source quotes. Keep quantities written "
            "in words verbatim in criteria and leave the corresponding volume field null. "
            "For every reported volume error, set that named volume field to null and retain its exact source "
            "quantity in the relevant part's criteria; never rewrite the source quote with invented digits. "
            "Quotes must preserve the source exactly, including URLs without adding Markdown. hypothesis_mode is null for questions "
            "and deliverables. Keep open method "
            "choices in planning_notes; clarification is only a question about an unidentifiable requested outcome. "
            "Validation errors: " + json.dumps(diagnostics, ensure_ascii=False)
            + "\nDraft to repair against its trusted source: " + json.dumps(repair_input, ensure_ascii=False)
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Run the framing stage only once per persisted ADK session.

        The pipeline wrapper invokes every pre-stage for every chat turn.  Once
        this agent successfully finishes, later user messages are continuations
        of the same research and must go straight to the orchestrator.
        """
        if frame_is_initialized(ctx.session.state):
            logger.info(
                "research frame already initialized; skipping ContextInitAgent "
                "for session %s",
                session_key(ctx)[1],
            )
            return

        async for event in super()._run_async_impl(ctx):
            yield event

    def _review_output(self, output_text) -> str:
        try:
            return render_frame_summary(coerce_frame(output_text))
        except Exception:  # noqa: BLE001 — review must never crash the run
            return super()._review_output(output_text)

    async def _review_decision(self, ctx: InvocationContext, output_text) -> HITLResponse:
        try:
            frame = self._frame(ctx, output_text)
        except Exception as exc:  # noqa: BLE001 — fall back to plain approval
            logger.warning("frame form skipped (parse failed): %s", exc)
            return HITLResponse(action=HITLAction.APPROVE, approved=True)

        user_id, session_id = session_key(ctx)
        request = HITLRequest(
            agent_name=self.name,
            action_type=HITLAction.APPROVE,
            message=_HITL_MESSAGE,
            form=frame_to_form(frame),
            # `output` is what the web handler publishes as the request's
            # document; without it the frame would be the one review with no
            # readable body to open.
            context={
                "_session": {"user_id": user_id, "session_id": session_id},
                "output": render_frame_summary(frame),
            },
            invoked_via="internal_loop",
        )
        response = await self.hitl_handler.handle_request(request)

        # Fold the operator's answers in and store the merged frame back, so the
        # base loop finishes (approved, no instructions) with the updated frame.
        merged = apply_review_response(frame, response)
        if self.output_key:
            ctx.session.state[self.output_key] = merged.model_dump()
        return HITLResponse(action=HITLAction.APPROVE, approved=True)

    def _post_final_events(self, ctx: InvocationContext, output_text):
        try:
            frame = self._frame(ctx, output_text)
        except Exception as exc:  # noqa: BLE001 — seeding must not kill the run
            logger.warning("frame not seeded (parse failed): %s", exc)
            yield Event(invocation_id=ctx.invocation_id, author=self.name, branch=ctx.branch,
                actions=EventActions(state_delta={FRAME_COMPLETED_STATE_KEY: False,
                    "research_frame_initialization_error": str(exc)}))
            return
        statement = self._statement(frame)
        frame = frame.model_copy(update={"normalized_statement": statement.model_dump()})
        catalog_result = {"ok": False, "reason": statement.uncertainty}
        result = {"ok": False}
        try:
            store = get_research_graph(ctx)
            result = seed_frame(store, frame)
            if result.get("ok"):
                from CoScientist.requirements.graph_io import commit_statement

                catalog_result = commit_statement(store, statement)
                if catalog_result.get("ok") and hasattr(store, "set_framing_snapshot"):
                    store.set_framing_snapshot(frame.model_dump())
        except Exception as exc:  # noqa: BLE001
            logger.warning("frame graph seeding failed: %s", exc)
            catalog_result = {"ok": False, "reason": str(exc)}

        ok = bool(result.get("ok") and catalog_result.get("ok") and not statement.uncertain)
        stats = result.get("graph_stats") or {}
        if ok:
            logger.info(
                "research frame seeded in graph (%d nodes, %d edges)",
                stats.get("nodes", 0), stats.get("edges", 0),
            )
        else:
            logger.warning("frame initialization incomplete: frame=%s catalog=%s", result, catalog_result)

        state_delta = {FRAME_STATE_KEY: frame.model_dump()}
        if ask := (frame.original_request or "").strip():
            state_delta["orchestrator_root_goal"] = ask
        if frame.operations:
            state_delta["experiment_operations"] = [op.model_dump() for op in frame.operations]
        state_delta["normalized_statement"] = statement.model_dump()
        state_delta["requirement_refs"] = [
            {**part.model_dump(), "volume": statement.volume.model_dump()}
            for part in statement.parts if not part.retired
        ]
        state_delta[FRAME_COMPLETED_STATE_KEY] = ok
        state_delta["research_frame_initialization_error"] = "" if ok else (
            statement.uncertainty or catalog_result.get("reason")
            or result.get("reason") or "The research frame or requirement catalog could not be saved."
        )
        # Leave the stage eligible for retry until both frame and catalog succeed.
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(state_delta=state_delta),
            content=None if ok else types.Content(role="model", parts=[types.Part(
                text="Research initialization incomplete: " + state_delta["research_frame_initialization_error"])]),
        )


__all__ = [
    "ContextInitSessionAgent",
    "FRAME_COMPLETED_STATE_KEY",
    "FRAME_STATE_KEY",
    "apply_form_values",
    "apply_review_response",
    "coerce_frame",
    "frame_is_initialized",
    "frame_to_form",
    "render_frame_summary",
]
