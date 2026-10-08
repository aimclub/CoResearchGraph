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
from CoScientist.tools.task_tracker import task_tracker_instance

logger = logging.getLogger("CoScientist.hitl.session_agent")


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
    correction_prompt: str = "The human reviewed your output and provided this feedback/correction:\n\n{feedback}\n\nYou MUST rewrite your output incorporating this feedback. Write your answer in the report language of this session. The session state key `report_language` gives it: en = English, ru = Russian. If it is empty, use English."
    critic_correction_prompt: str = "A plan critic reviewed your output and asked for one revision:\n\n{feedback}\n\nProduce the output again ONCE, in full, fixing exactly what the critic named — the previous version was discarded. Registering it normalises it (ids are renumbered, adjacent steps with the same executor assignee are merged); that is expected, so do not register again to undo it. This is the last round: there is no second review. Write your answer in the report language of this session. The session state key `report_language` gives it: en = English, ru = Russian. If it is empty, use English."

    def _review_output(self, output_text) -> str:
        """How the proposed output is presented to the human reviewer.

        Structured outputs (dict/list from an output_schema) are shown as
        readable JSON. Subclasses may override to show a rendered document
        instead (e.g. the microfluidics ТЗ agent renders Markdown)."""
        if isinstance(output_text, (dict, list)):
            try:
                return json.dumps(output_text, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
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

    async def _feed_back(
        self, ctx: InvocationContext, feedback_prompt: str
    ) -> AsyncGenerator[Event, None]:
        """Hand review feedback to the agent as a user turn and let it re-run.

        Yielding the event is what puts it in front of the model: the consumer
        (the Runner) appends everything we yield to the session before asking us
        for the next event, and the next run builds its contents from there.
        Appending it ourselves as well duplicated the message in the agent's
        context — the runner's session object is the very one we hold.

        The fallback covers a consumer that does not write to the session (a
        bare ``run_async`` in a test): without the event the re-run would not
        see the feedback at all and would just reproduce its previous answer.
        """
        event = Event(
            invocation_id=ctx.invocation_id,
            author="user",
            branch=ctx.branch,
            content=types.Content(
                role="user", parts=[types.Part(text=feedback_prompt)]
            ),
        )
        yield event
        if not any(e is event for e in reversed(ctx.session.events)):
            ctx.session.events.append(event)
        # Clear the end-of-agent flag so the agent is allowed to run again.
        ctx.set_agent_state(self.name)

    def _post_final_events(self, ctx: InvocationContext, output_text):
        """Extra events to emit AFTER the final output is accepted (approved
        by the human, or produced directly when no HITL handler is wired).

        Subclasses may yield follow-up events — e.g. the microfluidics ТЗ
        agent publishes the rendered ТЗ document into the chat before the
        pipeline moves on. Default: nothing."""
        return iter(())

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
            final_event.content.parts[0].text = ""
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

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:

        critic_rounds = 0

        while True:
            output_text = ""
            final_event = None

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

            async with Aclosing(super()._run_async_impl(ctx)) as agen:
                async for event in agen:
                    if event.is_final_response():
                        final_event = event
                        # Earlier text events contain reasoning and tool-call
                        # narration. Only the final response may reach HITL.
                        output_text = "".join(
                            part.text or ""
                            for part in (event.content.parts if event.content else [])
                        )
                    else:
                        yield event

            # ── Critic review ────────────────────────────────────────────
            # Runs before the human sees anything and regardless of whether
            # HITL is on at all, on a budget of `critic_max_rounds` rewrites.
            # No feedback (approved, unparseable, or the call failed) accepts
            # the output: an objection nobody can state is not worth a replan.
            if (
                final_event is not None
                and self.plan_critic is not None
                and critic_rounds < self.critic_max_rounds
            ):
                critic_input = output_text
                if self.output_key:
                    critic_input = ctx.session.state.get(self.output_key, output_text)

                critic_rounds += 1
                try:
                    feedback = await self.plan_critic(
                        _user_task(ctx),
                        self._review_output(self._proposed_output(ctx, critic_input)),
                    )
                except Exception:  # noqa: BLE001
                    # A reviewer must never take the run down with it.
                    logger.exception(
                        "%s: plan critic failed — output accepted unreviewed",
                        self.name,
                    )
                    feedback = None

                if feedback:
                    logger.info(
                        "%s: plan critic requested a revision (round %d/%d): %s",
                        self.name, critic_rounds, self.critic_max_rounds, feedback,
                    )
                    # The rejected output never reaches the chat — only the
                    # rewrite does, exactly as with a human rejection.
                    async for event in self._feed_back(
                        ctx, self.critic_correction_prompt.format(feedback=feedback)
                    ):
                        yield event
                    continue

                logger.info("%s: plan critic approved the output", self.name)

            from CoScientist.config import get_settings
            hitl_on = bool(self.hitl_handler and get_settings().web.hitl_enabled)

            if not hitl_on or final_event is None:
                # No HITL or not a final event (e.g. tool call): just pass and exit
                if not hitl_on:
                    logger.info(
                        "%s: HITL disabled (hitl_enabled=False) — "
                        "output passed through without human review", self.name,
                    )
                if final_event is not None:
                    async for event in self._emit_final(ctx, final_event, output_text):
                        yield event
                break

            if self.output_key:
                output_text = ctx.session.state.get(self.output_key, output_text)

            # Perform HITL check (subclasses may run a multi-step dialogue).
            response = await self._review_decision(ctx, output_text)

            if response.approved:
                if response.instructions and response.action != HITLAction.EDIT:
                    edited_text = response.instructions
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

                if not response.free_input and response.action != HITLAction.EDIT:
                    # HITL approved — now emit the (possibly updated) final event and exit
                    if final_event is not None:
                        async for event in self._emit_final(ctx, final_event, output_text):
                            yield event
                    break

            # Rejected or "Edit" requested — feed feedback back into the agent
            feedback = response.instructions or response.free_input or "No feedback provided."

            async for event in self._feed_back(
                ctx, self.correction_prompt.format(feedback=feedback)
            ):
                yield event

