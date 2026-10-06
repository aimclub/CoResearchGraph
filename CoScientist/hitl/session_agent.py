import asyncio
import logging
import os
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.utils.context_utils import Aclosing

from CoScientist.hitl.handler import AbstractHITLHandler
from CoScientist.hitl.models import HITLAction, HITLRequest, HITLResponse
from CoScientist.graph.session_scope import session_key

import json
import yaml
from CoScientist.tools.task_tracker import task_tracker_instance

logger = logging.getLogger("CoScientist.hitl.session_agent")


class _ReviewYamlDumper(yaml.SafeDumper):
    """YAML for a human reader: list items indented under their key,
    multi-line strings as ``|`` blocks."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def _represent_str(dumper: yaml.SafeDumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_ReviewYamlDumper.add_representer(str, _represent_str)


def render_review_yaml(data) -> str:
    """Render a structured output (dict/list) as readable YAML.

    Key order is kept (the schema's order reads best), Cyrillic stays as is,
    and long lines are not hard-wrapped — the UI wraps them itself.
    """
    return yaml.dump(
        data,
        Dumper=_ReviewYamlDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=float("inf"),
    ).rstrip()


def render_task_plan(tasks) -> str:
    """Render the registered delegation plan without model/tool narration."""
    if not isinstance(tasks, list) or not tasks:
        return ""
    lines = [f"План: {len(tasks)} задач(и)"]
    for index, task in enumerate(tasks, 1):
        title = task.get("title") or f"Задача {index}"
        assignee = task.get("assignee") or "не назначен"
        lines.extend((f"\n{index}. {title}", f"Исполнитель: {assignee}"))
        description = (task.get("description") or "").strip()
        if description:
            lines.append(description)
        depends_on = task.get("parent_id")
        if depends_on:
            lines.append(f"Зависит от: {depends_on}")
    return "\n".join(lines)


# Chat text for one plan-critic verdict, keyed by verdict then report language
# ("report_language" session state — see agents/callbacks/report_language.py).
# "failed" covers both an exception and the critic's OWN silent fallback to
# "approve" on a timed-out/unparseable LLM call (agents/callbacks/critic.py) —
# from here the two are indistinguishable, so a failure is announced as one.
_CRITIC_VERDICT_TEXT = {
    "approve": {
        "ru": "✅ **Одобрено** (раунд {round}/{max}) — критик не нашёл, что поправить.",
        "en": "✅ **Approved** (round {round}/{max}) — nothing for the critic to flag.",
    },
    "revise": {
        "ru": "🔁 **Отправлено на доработку** (раунд {round}/{max})\n\n{feedback}",
        "en": "🔁 **Sent back for revision** (round {round}/{max})\n\n{feedback}",
    },
    "failed": {
        "ru": "⚠️ Критик не ответил (сбой или таймаут) — план принят без проверки.",
        "en": "⚠️ The critic did not answer (failed or timed out) — the plan was accepted unreviewed.",
    },
}


def _user_task(ctx: InvocationContext) -> str:
    """The user's request as plain text ('' when the turn opened with a file)."""
    content = getattr(ctx, "user_content", None)
    if content is None or not getattr(content, "parts", None):
        return ""
    return "".join(part.text or "" for part in content.parts)


class SessionAgent(LlmAgent):
    """A planner that generates a roadmap and asks the human.
    If the human requests changes, it automatically feeds the changes back
    to itself and generates a new roadmap, looping until approved.

    An LLM critic can review the output first, in the same loop and on the same
    feed-it-back mechanism, but on a fixed budget: it is a reviewer that never
    tires, so unlike the human it does not get to keep asking. Wired separately
    from HITL (system.yaml ``critic:``), so it also reviews autonomous runs.
    """
    hitl_handler: Optional[AbstractHITLHandler] = None
    # async (task, output) -> feedback to act on, or None to accept as-is.
    # Built by the assembler from `critic:` (agents/callbacks/critic.py).
    plan_critic: Optional[Callable[[str, str], Awaitable[Optional[str]]]] = None
    # Critic revision rounds per run, from Settings -> PlannerAgent. One: the
    # critic gets a single say, then the rewrite stands — a self-critique loop
    # that can run forever will. There is always a budget; only its size moves.
    critic_max_rounds: int = 1
    # Times an agent that stopped half way (see `_unfinished_feedback`) is sent
    # back to finish, per review pass. After that its output is reviewed as is.
    unfinished_max_rounds: int = 3
    # Who the critic's review is reported as in the web UI (activity rail,
    # agent tree, ToolsViewer, status line). Not a YAML agent, so never internal.
    critic_agent_name: str = "PlanCriticAgent"
    correction_prompt: str = "The human reviewed your output and provided this feedback/correction:\n\n{feedback}\n\nYou MUST rewrite your output incorporating this feedback."
    critic_correction_prompt: str = "A plan critic reviewed your output and asked for one revision:\n\n{feedback}\n\nProduce the output again ONCE, in full, fixing exactly what the critic named — the previous version was discarded. Registering it normalises it (ids are renumbered, adjacent steps with the same executor assignee are merged); that is expected, so do not register again to undo it. This is the last round: there is no second review."

    def _review_output(self, output_text) -> str:
        """How the proposed output is presented to the human reviewer.

        Structured outputs (dict/list from an output_schema, or the same as a
        JSON string) are shown as YAML — far easier to read than JSON.
        Subclasses may override to show a rendered document instead (e.g. the
        microfluidics ТЗ agent renders Markdown)."""
        data = output_text
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                return output_text
        if isinstance(data, (dict, list)):
            try:
                return render_review_yaml(data)
            except yaml.YAMLError:
                pass
        return str(output_text)

    def _proposed_output(self, ctx: InvocationContext, output_text) -> Any:
        """What a reviewer (critic or human) actually judges.

        For the planner that is the REGISTERED plan — the tasks that will be
        executed — rather than the model's closing narration about it. Falls
        back to the agent's own output when nothing was registered.
        """
        if self.name == "PlannerAgent":
            registered_plan = render_task_plan(ctx.session.state.get("active_tasks"))
            if registered_plan:
                return registered_plan
        return output_text

    def _critic_output(self, ctx: InvocationContext, output_text) -> str:
        """The critic's input can differ from the document shown to a human."""
        return self._review_output(self._proposed_output(ctx, output_text))

    def _critic_task(self, ctx: InvocationContext) -> str:
        return _user_task(ctx)

    def _can_run_critic(self, ctx: InvocationContext) -> bool:
        return True

    def _critic_review_limit(self) -> int:
        return self.critic_max_rounds

    def _unfinished_feedback_author(self) -> str:
        return "user"

    def _unfinished_feedback(self, ctx: InvocationContext) -> Optional[str]:
        """Feedback that sends the agent back to FINISH its output — not to
        redo it — when it ended its turn before the output was complete.

        For agents that assemble their output over several tool calls (the
        microfluidics ТЗ is filled section by section). Checked after every
        pass, before any reviewer sees the output. Default: always complete."""
        return None

    def _produce(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """One pass producing the output: the agent's own LLM turn by default.

        Subclasses may produce it another way — e.g. the microfluidics ТЗ agent
        fans the sections out to parallel workers. The last final response
        yielded is the pass's output, reviewed like the LLM's."""
        return super()._run_async_impl(ctx)

    def _rewrite_state_delta(self, ctx: InvocationContext) -> dict:
        """State to carry with a reviewer's (critic or human) feedback — e.g. a
        reset so the next pass starts over instead of continuing. Default:
        none."""
        return {}

    async def _feed_back(
        self,
        ctx: InvocationContext,
        feedback_prompt: str,
        state_delta: Optional[dict] = None,
        author: str = "user",
    ) -> AsyncGenerator[Event, None]:
        """Hand review feedback to the agent as a user turn and let it re-run.

        ``author`` is who the feedback comes from: the human by default, or the
        critic. A non-user author reaches the model as ADK's
        "For context: [<author>] said: ..." turn, so the critic's objection is
        not mistaken for the user's own words.

        Yielding the event is what puts it in front of the model: the consumer
        (the Runner) appends everything we yield to the session before asking us
        for the next event, and the next run builds its contents from there.
        Appending it ourselves as well duplicated the message in the agent's
        context — the runner's session object is the very one we hold.
        ``state_delta`` rides on the same event, so the reset and the feedback
        land together.

        The fallback covers a consumer that does not write to the session (a
        bare ``run_async`` in a test): without the event the re-run would not
        see the feedback at all and would just reproduce its previous answer.
        """
        event = Event(
            invocation_id=ctx.invocation_id,
            author=author,
            branch=ctx.branch,
            content=types.Content(
                role="user", parts=[types.Part(text=feedback_prompt)]
            ),
            actions=EventActions(state_delta=dict(state_delta or {})),
        )
        yield event
        if not any(e is event for e in reversed(ctx.session.events)):
            ctx.session.events.append(event)
            ctx.session.state.update(state_delta or {})
        # Clear the end-of-agent flag so the agent is allowed to run again.
        ctx.set_agent_state(self.name)

    def _post_final_events(self, ctx: InvocationContext, output_text):
        """Extra events to emit AFTER the final output is accepted (approved
        by the human, or produced directly when no HITL handler is wired).

        Subclasses may yield follow-up events — e.g. the microfluidics ТЗ
        agent publishes the rendered ТЗ document into the chat before the
        pipeline moves on. Default: nothing."""
        return iter(())

        return iter(())

    def _should_run_review(self) -> bool:
        """Whether this turn enters ``_review_decision``.

        Default: only when a handler is wired *and* the global HITL switch is
        on. Experiment plan/result review overrides this so deterministic
        validate + ``initialize_runtime`` still run in headless smokes
        (``HITL__ENABLED`` off, ``COSCIENTIST_EXPERIMENT_HITL_AUTO_APPROVE=1``).
        """
        from CoScientist.config import get_settings
        return bool(self.hitl_handler and get_settings().web.hitl_enabled)

    async def _emit_final(
        self, ctx: InvocationContext, final_event: Event, output_text
    ) -> AsyncGenerator[Event, None]:
        """Yield the final event, then any subclass follow-up events.

        When ``_post_final_events`` actually publishes something (a rendered
        summary/document standing in for the raw model output — e.g.
        ContextInitSessionAgent's frame summary, the ТЗ agent's document),
        showing ``final_event``'s raw text too would double the chat message.
        ``final_event`` is still yielded unchanged otherwise — its
        state_delta (e.g. ``output_key``) and its place in session history
        must survive — only its visible text is blanked, which the chat feed
        treats as nothing to display.
        """
        extras = list(self._post_final_events(ctx, output_text))
        if extras and final_event.content and final_event.content.parts:
            # Каждую текстовую часть, а не первую: у рассуждающей модели первой
            # идёт мысль (её и так не показывают), а ответ — второй. Погасив
            # только нулевую, мы стёрли бы невидимое и оставили в ленте сырой
            # JSON рядом с объявлением о документе.
            for part in final_event.content.parts:
                if getattr(part, "text", None):
                    part.text = ""
        yield final_event
        for extra in extras:
            yield extra

    async def _review_decision(self, ctx: InvocationContext, output_text) -> HITLResponse:
        """One review round with the human; returns the final decision.

        Default: a single approve/edit request showing the proposed output.
        Subclasses may run a multi-step dialogue instead (e.g. the ТЗ agent
        first reviews the document, then interviews the operator question by
        question) as long as they return one HITLResponse."""
        review_output = self._proposed_output(ctx, output_text)

        user_id, session_id = session_key(ctx)
        request = HITLRequest(
            agent_name=self.name,
            action_type=HITLAction.APPROVE,
            message=(
                f"Agent '{self.name}' proposes its result. Please review."
            ),
            context={
                "output": self._review_output(review_output),
                "_session": {
                    "user_id": user_id,
                    "session_id": session_id,
                },
            },
            invoked_via="internal_loop",
        )
        return await self.hitl_handler.handle_request(request)

    async def _run_plan_critic(
        self, ctx: InvocationContext, critic_input, round_no: int
    ) -> Optional[str]:
        """One critic review: feedback to act on, or None to accept.

        The critic is a bare LLM call, not an ADK agent, so no callback reports
        it — without the records below it ran invisibly, and a run where it
        approved looked exactly like a run where it never ran, or hung. It is
        reported two ways, both as ``critic_agent_name``: a hand-off in the
        activity rail/ToolsViewer (open on the call, closed on the verdict),
        and — since a delegation's own result is never shown, only its
        preview — the verdict itself as a chat message, the same way any
        other agent's deliverable reaches the user.
        """
        from CoScientist.logging.tool_activity import report_delegation
        from CoScientist.logging.agent_output import report_output

        task = self._critic_task(ctx)
        plan = self._critic_output(ctx, critic_input)
        report = dict(
            author=self.name,
            target=self.critic_agent_name,
            call_id=f"plan_critic-{ctx.invocation_id}-{round_no}",
        )
        lang = "ru" if ctx.session.state.get("report_language") == "ru" else "en"

        async def announce(verdict: str, feedback_text: str = "") -> None:
            template = _CRITIC_VERDICT_TEXT[verdict][lang]
            await report_output(ctx, {
                "agent": self.critic_agent_name,
                "caller": self.name,
                "call_id": report["call_id"],
                "content": template.format(
                    round=round_no, max=self._critic_review_limit(),
                    feedback=feedback_text,
                ),
            })

        await report_delegation(
            ctx, phase="call",
            args={"task": task, "plan": plan, "round": round_no,
                  "max_rounds": self._critic_review_limit()},
            **report,
        )
        try:
            feedback = await self.plan_critic(task, plan)
        except Exception as exc:  # noqa: BLE001
            # A reviewer must never take the run down with it.
            logger.exception(
                "%s: plan critic failed — output accepted unreviewed", self.name,
            )
            await report_delegation(ctx, phase="error", error=repr(exc), **report)
            await announce("failed")
            return None

        await report_delegation(
            ctx, phase="result",
            result={"verdict": "revise" if feedback else "approve",
                    "feedback": feedback or ""},
            **report,
        )
        await announce("revise" if feedback else "approve", feedback or "")
        return feedback

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:

        critic_rounds = 0
        unfinished_rounds = 0

        while True:
            output_text = ""
            final_event = None
            last_model_text = ""

            # Never review a stale plan from an earlier attempt/session turn if
            # the current planner run fails before create_plan succeeds.
            if self.name == "PlannerAgent":
                # Session services return copies and persist only state deltas;
                # mutating ``ctx.session.state`` directly can let an old plan
                # reappear on the next invocation. Commit the empty plan first.
                await ctx.session_service.append_event(
                    ctx.session,
                    Event(
                        invocation_id=ctx.invocation_id,
                        author=self.name,
                        branch=ctx.branch,
                        actions=EventActions(state_delta={"active_tasks": []}),
                    ),
                )

            async with Aclosing(self._produce(ctx)) as agen:
                async for event in agen:
                    # ADK puts output_key in the event delta; the Runner only
                    # persists it after we yield. Final answers are held here
                    # for review, so make the current draft available locally.
                    if self.output_key and self.output_key in event.actions.state_delta:
                        ctx.session.state[self.output_key] = event.actions.state_delta[self.output_key]
                    event_text = "".join(
                        part.text or ""
                        for part in (event.content.parts if event.content else [])
                        if not getattr(part, "thought", False)
                    )
                    if event_text.strip():
                        last_model_text = event_text
                    if event.is_final_response() and (
                        event.content is not None or event.error_code
                        or (self.output_key and self.output_key in event.actions.state_delta)
                    ):
                        final_event = event
                        # Earlier text events contain reasoning and tool-call
                        # narration. Prefer the final response for HITL when present.
                        output_text = event_text
                    else:
                        yield event

            if self.output_key:
                # ADK State cannot delete keys; clearers set them to None. Treat
                # None as "missing" so we keep the live final-event text.
                # Structured-output planners may also land the payload only in
                # state (no is_final_response event) — still review that value.
                stored = ctx.session.state.get(self.output_key)
                if stored is not None:
                    if isinstance(stored, (dict, list)):
                        output_text = json.dumps(stored, ensure_ascii=False)
                    else:
                        output_text = str(stored)

            if not output_text.strip() and last_model_text.strip():
                # Some providers emit the structured plan as a non-final text
                # event; still feed it to deterministic review.
                output_text = last_model_text

            usable = (output_text or "").strip()

            if final_event is not None and final_event.error_code:
                # A provider failure is not a new draft. Never re-review the
                # previous output_key after a rejected answer failed to rewrite.
                if self.output_key:
                    ctx.session.state[self.output_key] = None
                    final_event.actions.state_delta[self.output_key] = None
                yield final_event
                break

            # ── Completeness check ───────────────────────────────────────
            # An agent that builds its output over several tool calls may end
            # its turn half way. Send it back to finish before anyone reviews
            # a half-built result; like a critic rejection, the premature
            # final answer never reaches the chat.
            if final_event is not None:
                unfinished = self._unfinished_feedback(ctx)
                if unfinished and unfinished_rounds < self.unfinished_max_rounds:
                    unfinished_rounds += 1
                    logger.info(
                        "%s: output unfinished (round %d/%d): %s",
                        self.name, unfinished_rounds, self.unfinished_max_rounds,
                        unfinished,
                    )
                    async for event in self._feed_back(
                        ctx, unfinished, author=self._unfinished_feedback_author(),
                    ):
                        yield event
                    continue
                if unfinished:
                    logger.warning(
                        "%s: output still unfinished after %d round(s) — "
                        "passing it on as is", self.name, unfinished_rounds,
                    )

            # ── Critic review ────────────────────────────────────────────
            # Runs before the human sees anything and regardless of whether
            # HITL is on at all, on a budget of `critic_max_rounds` rewrites.
            # No feedback (approved, unparseable, or the call failed) accepts
            # the output: an objection nobody can state is not worth a replan.
            if (
                final_event is not None
                and self.plan_critic is not None
                and self._can_run_critic(ctx)
                and critic_rounds < self.critic_max_rounds
            ):
                critic_input = output_text
                if self.output_key:
                    critic_input = ctx.session.state.get(self.output_key, output_text)

                critic_rounds += 1
                feedback = await self._run_plan_critic(
                    ctx, critic_input, critic_rounds
                )

                if feedback:
                    logger.info(
                        "%s: plan critic requested a revision (round %d/%d): %s",
                        self.name, critic_rounds, self.critic_max_rounds, feedback,
                    )
                    # The rejected output never reaches the chat — only the
                    # rewrite does, exactly as with a human rejection.
                    unfinished_rounds = 0
                    async for event in self._feed_back(
                        ctx,
                        self.critic_correction_prompt.format(feedback=feedback),
                        self._rewrite_state_delta(ctx),
                        author=self.critic_agent_name,
                    ):
                        yield event
                    continue

                logger.info("%s: plan critic approved the output", self.name)

            if not self._should_run_review():
                # No HITL (or a subclass that opted out): pass the model output
                # through. Do not treat a missing final-event flag as "skip
                # review" — some providers emit the plan as a non-final text
                # event; subclasses that must review still get the usable text
                # below after synthesizing a final event.
                logger.info(
                    "%s: HITL disabled (hitl_enabled=False) — "
                    "output passed through without human review", self.name,
                )
                if final_event is not None:
                    async for event in self._emit_final(ctx, final_event, output_text):
                        yield event
                break

            if not usable:
                # Empty final / failed structured output — do not invent a review turn.
                if final_event is not None:
                    yield final_event
                break

            if final_event is None:
                final_event = Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    branch=ctx.branch,
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=usable)],
                    ),
                )

            # Perform HITL check (subclasses may run a multi-step dialogue).
            response = await self._review_decision(ctx, usable)

            if response.timed_out or response.stop_review_loop:
                # A specialised review agent has recorded a pause or terminal
                # decision in its runtime. Stop instead of silently approving
                # or regenerating forever.
                if final_event is not None:
                    yield final_event
                break

            if response.approved:
                # REPLACING the output is what «provide input» means: the
                # operator edits the text (or the roadmap JSON) and the edited
                # version becomes the answer. A plain APPROVE that happens to
                # carry a note must not do that — the note is a remark ABOUT the
                # output, not the output. It used to be impossible to hit,
                # because approve-plus-note was rewritten into `edit` in the
                # browser; the moment a positive answer started meaning approval
                # it became a live hazard, and a live one: the experiment
                # planner's `output_key` is `experiment_plan`, so one sentence
                # of feedback replaced the whole approved plan in session state
                # while the original plan went on running.
                if response.action == HITLAction.PROVIDE_INPUT:
                    edited_text = (
                        response.instructions
                        if response.instructions is not None
                        else response.free_input
                        if response.free_input is not None
                        else ""
                    )
                    if final_event is not None and final_event.content and final_event.content.parts:
                        final_event.content.parts[0].text = edited_text
                    if self.output_key:
                        ctx.session.state[self.output_key] = edited_text

                    try:
                        parsed = json.loads(edited_text)
                        if isinstance(parsed, list):
                            class DummyContext:
                                def __init__(self, state):
                                    self.state = state
                            prepared_state = {}
                            task_tracker_instance.create_plan(
                                parsed, DummyContext(prepared_state)
                            )
                            await ctx.session_service.append_event(
                                ctx.session,
                                Event(
                                    invocation_id=ctx.invocation_id,
                                    author=self.name,
                                    branch=ctx.branch,
                                    actions=EventActions(state_delta={
                                        "active_tasks": prepared_state.get(
                                            "active_tasks", []
                                        )
                                    }),
                                ),
                            )
                    except Exception:
                        pass

                if response.action == HITLAction.EDIT:
                    pass
                else:
                    # HITL approved — now emit the (possibly updated) final event and exit
                    if final_event is not None:
                        async for event in self._emit_final(ctx, final_event, output_text):
                            yield event
                    break

            # Rejected or "Edit" requested — feed feedback back into the agent
            feedback = response.instructions or response.free_input or "No feedback provided."

            unfinished_rounds = 0
            async for event in self._feed_back(
                ctx,
                self.correction_prompt.format(feedback=feedback),
                self._rewrite_state_delta(ctx),
            ):
                yield event
