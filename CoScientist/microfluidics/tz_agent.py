"""ТЗ session agent + document publication for the microfluidics profile.

``TZSessionAgent`` is the SessionAgent used for TZSpecAgent:

  * during the HITL review loop the human is shown the RENDERED ТЗ document
    (the reference Markdown format), not the raw JSON the schema needs;
  * once the ТЗ is accepted (approved by the human, or produced directly in
    headless mode) the agent PUBLISHES the document: saves it to
    ``tz_documents/TZ_<timestamp>.md``, stores it in session state under
    ``structured_tz_document`` and emits a chat message with the file path,
    the web link (/api/tz-document) and the full document text — right before
    the pipeline moves on to planning / literature search.

``save_tz_document`` remains available as an after_agent callback for
profiles that want persistence without the chat message.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types

from CoScientist.hitl.models import HITLAction, HITLRequest, HITLResponse
from CoScientist.hitl.session_agent import SessionAgent
from CoScientist.microfluidics.render import render_tz_document
from CoScientist.graph.session_scope import session_key

logger = logging.getLogger(__name__)

TZ_DOCUMENT_STATE_KEY = "structured_tz_document"
TZ_DOCUMENTS_DIR = Path("tz_documents")


def persist_tz_document(tz, out_dir: Optional[Path] = None) -> Tuple[str, Optional[Path]]:
    """Render the ТЗ document and write it to disk.

    Returns ``(document_markdown, saved_path)``; ``saved_path`` is None when
    the file could not be written (the document itself is still returned).
    Raises only if the ТЗ cannot be rendered at all.
    """
    document = render_tz_document(tz)
    out_dir = out_dir or TZ_DOCUMENTS_DIR
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"TZ_{stamp}.md"
        path.write_text(document, encoding="utf-8")
        logger.info("ТЗ document saved to %s", path)
        return document, path
    except OSError as exc:
        logger.warning("Could not write the ТЗ document file: %s", exc)
        return document, None


class TZSessionAgent(SessionAgent):
    """SessionAgent with a two-phase operator review of the ТЗ.

    1. The rendered document is shown for approval (Accept / Revise / Reject).
    2. After the document is accepted, the operator is interviewed about the
       remaining «не задано» fields — ONE HITL window PER QUESTION (ported
       from VibePAV's questionnaire), each answerable with a value, «не знаю»
       (field stays open), «на усмотрение агента», or «завершить опрос».
       Collected answers trigger ONE rewrite of the ТЗ, then the updated
       document comes back for the final approval.

    The interview runs once per invocation; every path degrades to "continue
    with open questions" so the pipeline can never get stuck."""

    def _review_output(self, output_text) -> str:
        from CoScientist.microfluidics.questionnaire import build_questionnaire

        try:
            document = render_tz_document(output_text)
        except Exception as exc:  # noqa: BLE001 — review must never crash the run
            logger.warning("TZ review render failed (%s); showing raw output", exc)
            return str(output_text)
        try:
            open_blocks = len(build_questionnaire(output_text))
        except Exception:  # noqa: BLE001 — the note is an aid only
            open_blocks = 0
        if open_blocks:
            document += (
                "\n---\n\n"
                "**Дальше:** правки к самому документу отправьте сейчас через "
                "Revise. После согласования документа система однократно "
                f"задаст уточняющие вопросы по незаполненным блокам "
                f"(сейчас таких блоков: {open_blocks}) — по одному окну на вопрос.\n"
            )
        return document

    async def _review_decision(self, ctx: InvocationContext, output_text) -> HITLResponse:
        from CoScientist.microfluidics.questionnaire import build_questionnaire

        response = await super()._review_decision(ctx, output_text)

        # Interview only once the DOCUMENT itself is accepted as-is.
        doc_accepted = (
            response.approved
            and response.action != HITLAction.EDIT
            and not response.free_input
            and not response.instructions
        )
        if not doc_accepted:
            return response

        # Once per invocation: the post-rewrite approval goes straight through.
        interview_key = f"_tz_interview_done_{ctx.invocation_id}"
        if ctx.session.state.get(interview_key):
            return response
        ctx.session.state[interview_key] = True

        try:
            tz = ctx.session.state.get(self.output_key) or output_text
            questions = build_questionnaire(tz)
        except Exception as exc:  # noqa: BLE001 — no interview is a safe fallback
            logger.warning("TZ interview skipped (questionnaire failed): %s", exc)
            return response
        if not questions:
            return response

        answers = await self._interview(questions, ctx)
        # Nothing that changes the ТЗ — no rewrite needed, keep the approval.
        if not answers or all(a == "не знаю" for _b, a in answers.values()):
            return response

        feedback = (
            "Ответы оператора на опросник по незаполненным полям ТЗ: "
            + "; ".join(
                f"{qid} (блок «{block}»): {answer}"
                for qid, (block, answer) in answers.items()
            )
            + ". Вопросы, не упомянутые здесь, оставь со статусом «не задано»."
        )
        logger.info("TZ interview collected %d answer(s) — rewriting the ТЗ", len(answers))
        return HITLResponse(
            action=HITLAction.EDIT, approved=False, instructions=feedback
        )

    async def _interview(self, questions, ctx: InvocationContext) -> dict:
        """Ask ONE HITL window per question; return {qid: (block, answer)}."""
        from CoScientist.microfluidics.questionnaire import (
            OPT_AGENT,
            OPT_SKIP,
            OPT_STOP,
            QUESTION_OPTIONS,
            render_question_card,
        )

        answers: dict = {}
        silent_in_a_row = 0
        total = len(questions)
        for i, q in enumerate(questions, 1):
            user_id, session_id = session_key(ctx)
            request = HITLRequest(
                agent_name=self.name,
                action_type=HITLAction.PROVIDE_INPUT,
                message=f"Уточнение ТЗ — вопрос {i} из {total}: «{q.block}»",
                options=list(QUESTION_OPTIONS),
                context={
                    "output": render_question_card(q, i, total),
                    "_session": {
                        "user_id": user_id,
                        "session_id": session_id,
                    },
                },
                invoked_via="internal_loop",
            )
            response = await self.hitl_handler.handle_request(request)

            text = (response.instructions or response.free_input or "").strip()
            option = (response.selected_option or "").strip()

            if option == OPT_STOP or (response.action == HITLAction.REJECT and not text):
                logger.info("TZ interview stopped by the operator at %d/%d", i, total)
                break
            if option == OPT_SKIP:
                answers[q.qid] = (q.block, "не знаю")
                silent_in_a_row = 0
                continue
            if option == OPT_AGENT:
                answers[q.qid] = (q.block, "на усмотрение агента")
                silent_in_a_row = 0
                continue
            if text:
                answers[q.qid] = (q.block, text)
                silent_in_a_row = 0
                continue
            # A bare approve with no content = an unattended timeout
            # auto-approve. Two in a row => the operator is gone; stop asking.
            silent_in_a_row += 1
            if silent_in_a_row >= 2:
                logger.warning(
                    "TZ interview: two unanswered questions in a row — "
                    "continuing with open fields"
                )
                break
        return answers

    def _post_final_events(self, ctx: InvocationContext, output_text):
        """Publish the accepted ТЗ document into the chat (file + state + text)."""
        tz = None
        if self.output_key:
            tz = ctx.session.state.get(self.output_key)
        if not tz:
            tz = output_text
        try:
            document, path = persist_tz_document(tz)
        except Exception as exc:  # noqa: BLE001 — publishing must not kill the run
            logger.warning("TZ document publication failed: %s", exc)
            return

        header = "📄 Структурированное ТЗ сформировано."
        if path is not None:
            header += f"\nФайл: {path.resolve()}"
        header += "\nОткрыть в браузере: /api/tz-document"
        text = f"{header}\n\n{document}"

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            actions=EventActions(state_delta={TZ_DOCUMENT_STATE_KEY: document}),
        )


def save_tz_document(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """after_agent callback: persist the approved ТЗ as a Markdown document
    (state + file) WITHOUT emitting a chat message."""
    tz = callback_context.state.get("structured_tz")
    if not tz:
        return None
    try:
        document, _path = persist_tz_document(tz)
    except Exception as exc:  # noqa: BLE001 — a render bug must not kill the run
        logger.warning("TZ document render failed: %s", exc)
        return None
    callback_context.state[TZ_DOCUMENT_STATE_KEY] = document
    return None


__all__ = [
    "TZSessionAgent",
    "TZ_DOCUMENT_STATE_KEY",
    "persist_tz_document",
    "save_tz_document",
]
