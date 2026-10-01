"""ТЗ session agent + document publication for the microfluidics profile.

``TZSessionAgent`` is the SessionAgent used for TZSpecAgent:

  * the ТЗ is assembled SECTION BY SECTION through ``fill_tz_section``
    (``tz_builder.py``), which keeps it in ``state["structured_tz"]`` — the
    model's final answer is only a short "done" line, so every reviewer reads
    the ТЗ from state. An agent that stops before the last section is sent
    back to continue from the section it stopped at;
  * with ``parallel_build`` (the default) the sections are not filled by this
    agent's own LLM one after another: one worker per ``SECTION_GROUPS`` part
    fills its sections AT THE SAME TIME, each into its own state key, and the
    parts are put together into ``state["structured_tz"]``. The wall time is
    the slowest part's instead of the sum of all 16 sections. The fields the
    operator leaves empty are filled the same way — each part's worker fills
    the ones in its sections, all at once;
  * the operator reviews the ТЗ as a FORM in the web ТЗ panel (``tz_review.py``):
    sections with empty fields first, filled ones below. Values typed by the
    operator go straight into the ТЗ; fields left EMPTY are filled by the
    agent (``fill_agent_fields``, «заполнено агентом») and the ТЗ comes back
    for another round with those fields highlighted — until the operator
    submits a form with nothing left empty;
  * every change is pushed to the panel live (``CoScientist/hitl/tz_panel.py``): the agent runs
    in a nested AgentTool session the web session cannot see;
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

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional, Tuple

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.tools import FunctionTool
from google.adk.utils.context_utils import Aclosing
from google.genai import types
from pydantic import PrivateAttr

from CoScientist.graph.session_scope import session_key
from CoScientist.hitl.models import HITLAction, HITLRequest, HITLResponse
from CoScientist.hitl.session_agent import SessionAgent
from CoScientist.microfluidics.render import render_tz_document
from CoScientist.microfluidics.tz_builder import (
    AGENT_FILL_STATE_KEY,
    AGENT_FILL_TOOL_NAME,
    FILL_TOOL_NAME,
    SECTION_GROUPS,
    TOTAL_SECTIONS,
    TZ_BUILD_STATE_KEY,
    TZ_STATE_KEY,
    SectionGroup,
    agent_fill_feedback,
    agent_fill_instruction,
    agent_fill_pending,
    agent_fill_request,
    apply_group_fills,
    assemble_groups,
    group_fill_delta,
    group_fill_key,
    group_fill_request,
    group_part_key,
    groups_agent_fill_feedback,
    groups_unfinished_feedback,
    load_tz,
    make_group_agent_fill_tool,
    make_group_fill_tool,
    request_text,
    tz_edition,
    unfinished_feedback,
    unfinished_fill_groups,
    unfinished_groups,
)
from CoScientist.hitl.tz_panel import publish_tz
from CoScientist.microfluidics.tz_review import (
    agent_fill_message,
    apply_operator_values,
    tz_form,
    tz_view,
)

logger = logging.getLogger(__name__)

TZ_DOCUMENT_STATE_KEY = "structured_tz_document"
TZ_DOCUMENTS_DIR = Path("tz_documents")

_TZ_TOOLS = {FILL_TOOL_NAME, AGENT_FILL_TOOL_NAME}
_REWRITE_PROMPT = "Оператор прислал правки к ТЗ:\n\n{feedback}\n\nТЗ собирается заново: {how}"
_REWRITE_SEQUENTIAL = (
    "заполни все разделы через fill_tz_section, начиная с раздела 1, перенося "
    "прежние значения и применяя правки."
)
_REWRITE_PARALLEL = (
    "каждый агент заново заполняет свою часть через fill_tz_section — с первого "
    "своего раздела, перенося прежние значения и применяя правки, которые к ней "
    "относятся."
)
_AGENT_FILL_PARALLEL = (
    "Их заполняют агенты частей ТЗ одновременно — каждый поля своих разделов, "
    "список и порядок у каждого в его инструкции."
)


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


async def _merge_runs(
    runs: Dict[str, AsyncGenerator[Event, None]],
) -> AsyncGenerator[Event, None]:
    """Interleave the events of concurrent agent runs as they come.

    As in ADK's ParallelAgent, a run waits until its event has been consumed
    (appended to the session) before it goes on, so each tool call reads
    current state. Unlike it, a run that fails is logged and stops alone: the
    others finish, and its unsaved sections are left to the completeness check.
    """
    queue: asyncio.Queue = asyncio.Queue()
    finished = object()

    async def pump(name: str, run: AsyncGenerator[Event, None]) -> None:
        try:
            async with Aclosing(run) as agen:
                async for event in agen:
                    consumed = asyncio.Event()
                    await queue.put((event, consumed))
                    await consumed.wait()
        except Exception:  # noqa: BLE001 — one worker must not stop the others
            logger.exception("ТЗ worker %s failed", name)
        finally:
            await queue.put((finished, None))

    tasks = [asyncio.create_task(pump(name, run)) for name, run in runs.items()]
    try:
        remaining = len(tasks)
        while remaining:
            event, consumed = await queue.get()
            if event is finished:
                remaining -= 1
                continue
            yield event
            consumed.set()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _touches_tz(event: Event) -> bool:
    """Did this event carry the answer of a tool that changes the ТЗ?"""
    parts = event.content.parts if event.content and event.content.parts else []
    return any(
        p.function_response is not None and p.function_response.name in _TZ_TOOLS
        for p in parts
    )


class TZSessionAgent(SessionAgent):
    """SessionAgent that reviews the ТЗ through the web ТЗ panel, round by round.

    Each round the operator gets the whole ТЗ as a form. Typed values are
    applied at once; if fields were left empty the agent fills them and the
    next round shows the result, agent values highlighted. A round with nothing
    left empty accepts the ТЗ. No answer (timeout, a handler without forms)
    accepts it as it is; a free-text correction (console "Edit") refills the
    whole ТЗ from section 1. At most ``max_form_rounds`` forms per invocation,
    so the pipeline can never get stuck."""

    # The agent composes complete instructions itself (fill what was left /
    # rewrite with these corrections); the base wrapper would contradict them.
    correction_prompt: str = "{feedback}"
    max_form_rounds: int = 5
    # Filling a 16-section form takes longer than a yes/no: the web handler's
    # auto-approve waits at least this long for it.
    form_timeout_seconds: float = 1800
    # Assemble the ТЗ — and later fill the fields the operator left empty — with
    # one worker per SECTION_GROUPS part, all at once, instead of this agent's
    # LLM walking the sections one after another. The review stays here.
    parallel_build: bool = True
    _workers: Dict[str, LlmAgent] = PrivateAttr(default_factory=dict)

    def _current_tz(self, ctx: InvocationContext, fallback):
        """The ТЗ assembled by fill_tz_section; ``fallback`` when there is none."""
        return ctx.session.state.get(TZ_STATE_KEY) or fallback

    def _proposed_output(self, ctx: InvocationContext, output_text):
        return self._current_tz(ctx, output_text)

    def _unfinished_feedback(self, ctx: InvocationContext):
        state = ctx.session.state
        if self.parallel_build:
            fill_feedback, build_feedback = groups_agent_fill_feedback, groups_unfinished_feedback
        else:
            fill_feedback, build_feedback = agent_fill_feedback, unfinished_feedback
        return (
            fill_feedback(state, ctx.invocation_id)
            or build_feedback(state, ctx.invocation_id)
        )

    def _rewrite_state_delta(self, ctx: InvocationContext) -> dict:
        state = ctx.session.state
        request = agent_fill_pending(state, ctx.invocation_id)
        if request is not None:
            # The operator's answers and what is left to the agent ride on the
            # feedback event, so they are persisted (and reach the parent
            # session through the AgentTool) even before the agent's first call.
            if self.parallel_build:
                return {
                    TZ_STATE_KEY: state.get(TZ_STATE_KEY),
                    **{group_fill_key(g): state.get(group_fill_key(g)) for g in SECTION_GROUPS},
                }
            return {TZ_STATE_KEY: state.get(TZ_STATE_KEY), AGENT_FILL_STATE_KEY: request}
        # Clearing the edition marker makes fill_tz_section start over at
        # section 1; the previous ТЗ stays in state until then.
        return {TZ_BUILD_STATE_KEY: None}

    def _review_output(self, output_text) -> str:
        try:
            return render_tz_document(output_text)
        except Exception as exc:  # noqa: BLE001 — review must never crash the run
            logger.warning("TZ review render failed (%s); showing raw output", exc)
            return str(output_text)

    # ── parallel assembly ────────────────────────────────────────────────────
    def _needs_build(self, ctx: InvocationContext) -> bool:
        """Is the ТЗ (still) to be assembled by the parallel workers?"""
        state = ctx.session.state
        invocation_id = ctx.invocation_id
        if not self.parallel_build or agent_fill_pending(state, invocation_id):
            return False
        return (
            tz_edition(state, invocation_id) is None
            or bool(unfinished_groups(state, invocation_id))
        )

    def _produce(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if self._needs_build(ctx):
            return self._build_in_parallel(ctx)
        if self.parallel_build and unfinished_fill_groups(ctx.session.state, ctx.invocation_id):
            return self._fill_in_parallel(ctx)
        # Sequential mode: this agent's own LLM builds the ТЗ / fills the fields.
        return super()._produce(ctx)

    def _worker(self, group: SectionGroup) -> LlmAgent:
        """The worker filling ``group`` — same model, its own prompt and tool."""
        worker = self._workers.get(group.key)
        if worker is None:
            from CoScientist.microfluidics.prompts import microfluidics_tz_worker

            worker = LlmAgent(
                name=f"{self.name}_{group.key}",
                description=f"заполняет часть ТЗ «{group.title}»",
                model=self.model,
                instruction=microfluidics_tz_worker(group),
                tools=[FunctionTool(make_group_fill_tool(group))],
                generate_content_config=self.generate_content_config,
                planner=self.planner,
                disallow_transfer_to_parent=True,
                disallow_transfer_to_peers=True,
            )
            self._adopt(worker)
            self._workers[group.key] = worker
        return worker

    def _fill_worker(self, group: SectionGroup) -> LlmAgent:
        """The worker filling the operator-left fields of ``group``. Its prompt
        is rendered before every call: the current ТЗ and what is left."""
        key = f"{group.key}_fill"
        worker = self._workers.get(key)
        if worker is None:
            from CoScientist.microfluidics.prompts import microfluidics_tz_fill_worker

            def instruction(context) -> str:
                state = context.state
                tz = load_tz(state.get(TZ_STATE_KEY))
                if tz is not None:
                    tz = apply_group_fills(state, context.invocation_id, tz)
                share = group_fill_request(state, context.invocation_id, group)
                return microfluidics_tz_fill_worker(group, tz, share)

            worker = LlmAgent(
                name=f"{self.name}_{key}",
                description=f"дозаполняет поля части ТЗ «{group.title}»",
                model=self.model,
                instruction=instruction,
                tools=[FunctionTool(make_group_agent_fill_tool(group))],
                generate_content_config=self.generate_content_config,
                planner=self.planner,
                disallow_transfer_to_parent=True,
                disallow_transfer_to_peers=True,
            )
            self._adopt(worker)
            self._workers[key] = worker
        return worker

    def _adopt(self, worker: LlmAgent) -> None:
        """Make ``worker`` this agent's subordinate. It is not in
        ``sub_agents`` (the workers are built on demand), so without this the
        tool-activity plugin sees no parent and the ToolsViewer shows every
        worker as a separate root instead of a branch under this agent. Both
        transfer flags are off, so ADK offers no transfer back to it."""
        worker.parent_agent = self

    def _worker_ctx(self, ctx: InvocationContext, worker: LlmAgent) -> InvocationContext:
        """A branch of its own for ``worker`` (as ParallelAgent does): it sees
        the request and this agent's turns — e.g. the operator's corrections —
        but not what the other workers are doing."""
        worker_ctx = ctx.model_copy()
        branch = f"{self.name}.{worker.name}"
        worker_ctx.branch = f"{ctx.branch}.{branch}" if ctx.branch else branch
        return worker_ctx

    async def _commit(self, ctx: InvocationContext, delta: dict) -> None:
        """Persist ``delta`` right away. It cannot ride on this pass's final
        event: that one is held back until the review is over."""
        await ctx.session_service.append_event(ctx.session, Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(state_delta=delta),
        ))

    async def _build_in_parallel(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Fill the ТЗ with every unfinished group's worker at once.

        The workers' tool calls pass through as they come (the live panel fills
        in from all parts at once); their closing lines do not — the pass ends
        with this agent's own final response once the parts are put together.
        An incomplete ТЗ is sent back by the completeness check, and the next
        pass re-runs only the workers that did not finish."""
        state = ctx.session.state
        invocation_id = ctx.invocation_id
        if tz_edition(state, invocation_id) is None:
            # A new edition (first pass, or a rewrite after review): what the
            # workers saved in an earlier round does not count.
            await self._commit(ctx, {group_part_key(g): None for g in SECTION_GROUPS})

        groups = unfinished_groups(state, invocation_id)
        logger.info(
            "%s: filling the ТЗ in parallel — %s", self.name,
            ", ".join(f"«{g.title}» ({len(g.sections)})" for g in groups),
        )
        async with Aclosing(self._run_workers(ctx, [self._worker(g) for g in groups])) as agen:
            async for event in agen:
                yield event

        tz = assemble_groups(state, invocation_id, request_text(ctx))
        await self._commit(ctx, {
            TZ_STATE_KEY: tz.model_dump(),
            TZ_BUILD_STATE_KEY: {"invocation_id": invocation_id},
        })
        logger.info("%s: ТЗ assembled — %d/%d sections", self.name, len(tz.blocks), TOTAL_SECTIONS)
        yield Event(
            invocation_id=invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(
                text=f"ТЗ собрано: {len(tz.blocks)}/{TOTAL_SECTIONS} разделов."
            )]),
        )

    async def _fill_in_parallel(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Fill the operator-left fields with every unfinished group's worker at
        once, then write their values into the ТЗ. Fields still empty are sent
        back by the completeness check; the next pass re-runs only those
        workers."""
        state = ctx.session.state
        invocation_id = ctx.invocation_id
        groups = unfinished_fill_groups(state, invocation_id)
        logger.info(
            "%s: filling the operator-left fields in parallel — %s", self.name,
            ", ".join(f"«{g.title}»" for g in groups),
        )
        workers = [self._fill_worker(g) for g in groups]
        async with Aclosing(self._run_workers(ctx, workers)) as agen:
            async for event in agen:
                yield event

        tz = load_tz(state.get(TZ_STATE_KEY))
        if tz is not None:
            tz = apply_group_fills(state, invocation_id, tz)
            await self._commit(ctx, {TZ_STATE_KEY: tz.model_dump()})
        request = agent_fill_pending(state, invocation_id)
        left = len(request["pending"]) if request else 0
        logger.info("%s: operator-left fields filled — %d section(s) left", self.name, left)
        yield Event(
            invocation_id=invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(
                text="Поля, оставленные оператором, дозаполнены."
                if not left else f"Поля дозаполнены не все: осталось разделов — {left}."
            )]),
        )

    async def _run_workers(
        self, ctx: InvocationContext, workers: list
    ) -> AsyncGenerator[Event, None]:
        """Run ``workers`` concurrently, each on its own branch, passing their
        events through as they come — except their closing lines: the pass
        ends with this agent's own final response."""
        runs = {w.name: w.run_async(self._worker_ctx(ctx, w)) for w in workers}
        async with Aclosing(_merge_runs(runs)) as agen:
            async for event in agen:
                if event.is_final_response():
                    if event.error_code:
                        logger.warning(
                            "%s: %s ended with %s: %s", self.name, event.author,
                            event.error_code, event.error_message,
                        )
                    continue
                yield event

    # ── live panel ───────────────────────────────────────────────────────────
    def _live_tz(self, ctx: InvocationContext):
        """The ТЗ as it stands — while the workers fill it, with their work."""
        state = ctx.session.state
        if self._needs_build(ctx):
            tz = assemble_groups(state, ctx.invocation_id, request_text(ctx))
            return tz if tz.blocks else None
        tz = load_tz(state.get(TZ_STATE_KEY))
        if tz is not None and self.parallel_build:
            tz = apply_group_fills(state, ctx.invocation_id, tz)
        return tz

    async def _publish(self, ctx: InvocationContext, phase: str, tz=None) -> None:
        state = ctx.session.state
        tz = tz if tz is not None else self._live_tz(ctx)
        if tz is None:
            return
        request = agent_fill_pending(state, ctx.invocation_id)
        agent_fill = None
        if request is not None:
            total = int(request.get("total") or 0)
            agent_fill = {"done": total - len(request["pending"]), "total": total}
        await publish_tz(ctx, {
            "phase": phase,
            "agent": self.name,
            "progress": {"filled": len(tz.blocks), "total": TOTAL_SECTIONS},
            "agent_fill": agent_fill,
            "round": state.get(self._round_key(ctx)),
            **tz_view(tz, request),
        })

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        async with Aclosing(super()._run_async_impl(ctx)) as agen:
            async for event in agen:
                if self.parallel_build and event.branch is None:
                    # ADK's node runtime gives a branch-less event the branch
                    # of the event before it — after the workers, one of
                    # theirs, hiding e.g. the operator's corrections from the
                    # other workers. This agent's own events take the branch
                    # the workers' branches descend from, visible to all.
                    event.branch = self.name
                yield event
                # The consumer has appended the event by now: state is current.
                if _touches_tz(event):
                    phase = (
                        "agent_filling"
                        if agent_fill_pending(ctx.session.state, ctx.invocation_id)
                        else "filling"
                    )
                    await self._publish(ctx, phase)
        await self._publish(ctx, "final")

    # ── operator review ──────────────────────────────────────────────────────
    @staticmethod
    def _round_key(ctx: InvocationContext) -> str:
        return f"_tz_form_round_{ctx.invocation_id}"

    async def _review_decision(self, ctx: InvocationContext, output_text) -> HITLResponse:
        state = ctx.session.state
        tz = load_tz(self._current_tz(ctx, output_text))
        if tz is None or not tz.blocks:
            # Nothing assembled to put in a form — plain review of what there is.
            return await super()._review_decision(ctx, output_text)

        round_no = int(state.get(self._round_key(ctx)) or 0) + 1
        if round_no > self.max_form_rounds:
            logger.warning(
                "%s: %d review rounds done — ТЗ accepted as it is", self.name,
                self.max_form_rounds,
            )
            return HITLResponse(action=HITLAction.APPROVE, approved=True)
        state[self._round_key(ctx)] = round_no
        # Whatever the agent did not manage to fill shows up as empty again.
        state[AGENT_FILL_STATE_KEY] = None
        state.update(group_fill_delta(None))
        await self._publish(ctx, "review", tz)

        user_id, session_id = session_key(ctx)
        form = tz_form(tz, round_no)
        request = HITLRequest(
            agent_name=self.name,
            action_type=HITLAction.APPROVE,
            message=f"Проверка ТЗ оператором (раунд {round_no}). {form['intro']}",
            form=form,
            context={
                "output": self._review_output(tz),
                "_session": {"user_id": user_id, "session_id": session_id},
            },
            invoked_via="internal_loop",
            timeout_seconds=self.form_timeout_seconds,
        )
        response = await self.hitl_handler.handle_request(request)

        if response.form_values is None:
            feedback = (response.instructions or response.free_input or "").strip()
            if response.action == HITLAction.EDIT and feedback:
                # A handler without forms (console) sent free-text corrections.
                how = _REWRITE_PARALLEL if self.parallel_build else _REWRITE_SEQUENTIAL
                return HITLResponse(
                    action=HITLAction.EDIT, approved=False,
                    instructions=_REWRITE_PROMPT.format(feedback=feedback, how=how),
                )
            # Timeout / skip: the operator is not there — keep the ТЗ as it is.
            return HITLResponse(action=HITLAction.APPROVE, approved=True)

        answers = apply_operator_values(tz, response.form_values)
        # In-run for the tools; persisted by the next event's state delta.
        state[TZ_STATE_KEY] = answers.tz.model_dump()
        fill = agent_fill_request(ctx.invocation_id, answers.left_to_agent)
        logger.info(
            "%s: round %d — %d value(s) from the operator, %d marked not "
            "required, %d left to the agent",
            self.name, round_no, answers.set_by_operator, answers.not_required,
            answers.left_count,
        )
        if fill is None:
            await self._publish(ctx, "review", answers.tz)
            return HITLResponse(action=HITLAction.APPROVE, approved=True)

        if self.parallel_build:
            # Each group's worker gets the fields in its own sections.
            state.update(group_fill_delta(fill))
            instruction = _AGENT_FILL_PARALLEL
        else:
            state[AGENT_FILL_STATE_KEY] = fill
            instruction = agent_fill_instruction(fill)
        await self._publish(ctx, "agent_filling", answers.tz)
        return HITLResponse(
            action=HITLAction.EDIT, approved=False,
            instructions=agent_fill_message(answers, instruction),
        )

    def _post_final_events(self, ctx: InvocationContext, output_text):
        """Publish the accepted ТЗ document into the chat (file + state + text)."""
        tz = self._current_tz(ctx, output_text)
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

        delta = {
            TZ_DOCUMENT_STATE_KEY: document,
            AGENT_FILL_STATE_KEY: None,
            **group_fill_delta(None),
        }
        model = load_tz(tz)
        if model is not None:
            # Operator answers may not have been followed by any tool call.
            delta[TZ_STATE_KEY] = model.model_dump()
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            actions=EventActions(state_delta=delta),
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
