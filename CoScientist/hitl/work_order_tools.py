"""Work Order tools: declare the contract, amend it, report step progress, and
submit the Work Report the human checks before the result goes on.

One toolset instance per agent (the assembler builds it with the agent's real
tool names, so a contract can only name tools the agent actually has).

How a contract is confirmed depends on its risk tier (work_order_risk.py):

  read         a declaration is reviewed like compute; an amendment that only
               adds read tools is a notice card and returns at once;
  compute      a HITL request; a configured positive veto window auto-approves
               it when that window runs out;
  side_effect  a HITL request, never auto-approved by a veto window.

How LONG any of them waits is the run's HITL mode (CoScientist/hitl/mode.py):
`auto` answers without asking, `basic` waits ten minutes, `debug` waits for the
human. Silence approves in none of them — it used to, for side effects only,
which made the riskiest tier the one the clock could sign off.

The Work Report is confirmed at the tier of the order it reports on. Sent back
for rework, the agent keeps working in the same run and reports again.

In step-review mode (``step_review=True``) every step also names what it will
SEND to its tools, and each finished step goes before the human with what was
sent, what was expected and what was found, plus the calls the system recorded
for it. The human accepts the step, sends it back, or stops the order.

With HITL or Work Orders switched off, the contract is still recorded (the guard
keeps working as a scope check) and approved without asking anyone.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from CoScientist.config import get_settings
from CoScientist.graph.session_scope import session_key
from CoScientist.hitl.models import HITLAction, HITLRequest, HITLResponse
from CoScientist.agents.callbacks.report_language import session_report_language
from CoScientist.hitl.work_order import (
    Artifact,
    Assumption,
    Finding,
    SideEffect,
    StepReview,
    WorkOrder,
    WorkReport,
    WorkStep,
    diff_work_orders,
    load_order,
    performed_side_effects,
    render_work_order,
    render_work_report,
    render_work_step_review,
    report_warnings,
    save_order,
)
from CoScientist.hitl.work_order_risk import (
    EXEMPT_TOOLS,
    TOOL_SIDE_EFFECTS,
    SideEffectKind,
    Tier,
    exempt_tools,
    max_tier,
    tool_tier,
)

logger = logging.getLogger("CoScientist.hitl.work_order")

_STEP_STATUSES = ("pending", "in_progress", "done", "skipped")
_DONE_VERDICTS = ("met", "partial", "not_met")
_CONFIDENCES = ("high", "medium", "low")
_ARTIFACT_KINDS = ("file", "dataset", "graph_node", "link", "other")


def work_order_active() -> bool:
    web = get_settings().web
    return bool(web.hitl_enabled and web.work_order_enabled)


def _error(message: str) -> Dict[str, Any]:
    return {"status": "error", "message": message}


def session_context(tool_context: Any) -> Dict[str, str]:
    user_id, session_id = session_key(tool_context)
    return {"user_id": user_id, "session_id": session_id}


def _marked(response: Any, field: str) -> set:
    """The ids the operator ticked, whatever action they ticked them with.

    One reader for both fields and every path. They used to be read only on the
    one action each was thought to belong to, so a mark made alongside anything
    else was discarded without a word.
    """
    if response is None or not isinstance(getattr(response, "form_values", None), dict):
        return set()
    return {str(i) for i in (response.form_values.get(field) or [])}


def _rejected_ids(response: Any) -> set:
    """Assumptions the operator refused to accept."""
    return _marked(response, "rejected_assumption_ids")


def _disputed_ids(response: Any) -> set:
    """Findings the operator marked as wrong."""
    return _marked(response, "disputed_finding_ids")


def _claim_plan_step(state: Any, agent: str, order: Any) -> None:
    """An approved order takes the agent's current plan step to in_progress.

    Best-effort by this module's contract: the human has already approved the
    order, and a bookkeeping failure must not undo that.
    """
    try:
        from CoScientist.tools.task_tracker import (
            current_task_for_agent,
            set_task_status,
        )

        if getattr(order, "plan_task_id", ""):
            task_id = order.plan_task_id
        else:
            task = current_task_for_agent(state, agent)
            if task is None:
                return
            task_id = str(task.get("id") or "")
            if not task_id:
                return
            order.plan_task_id = task_id
        set_task_status(state, task_id, "IN_PROGRESS",
                        notes="work order approved", agent=agent)
        _show_the_step_started(state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("claiming a plan step for %s failed: %s", agent, exc)


def _show_the_step_started(state: Any) -> None:
    """…and say so on the graph, now, rather than at the next orchestrator tick.

    This is the moment a step actually begins, and it was the moment nothing
    recorded. The plan mirror runs only on the orchestrator's own hooks, while
    the tracker goes IN_PROGRESS → DONE inside a single sub-agent turn — so by
    the time the mirror looked, the step had already finished and the graph
    drew it «не начат» and then «выполнен», with nothing in between. That is
    the whole of the operator's complaint that the graph never says which stage
    is running, for every step that is not an experiment.

    Best-effort, like everything else on this path: a bookkeeping failure must
    not undo an order the human has already approved.
    """
    try:
        from types import SimpleNamespace

        from CoScientist.agents.callbacks.tool_callbacks import (
            sync_plan_to_research_graph,
        )
        from CoScientist.graph.research.store import get_research_graph

        tasks = state.get("_master_active_tasks") or []
        if not tasks:
            return
        # The scope lives in the state itself, and `session_key` reads it off
        # `context.state` — so a bare holder for it is the whole context this
        # needs, and it resolves to the same store the agents commit to.
        sync_plan_to_research_graph(
            tasks, get_research_graph(SimpleNamespace(state=state)), state,
            str(state.get("user_query", "")),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("mirroring the claimed plan step failed: %s", exc)


def _close_plan_step(state: Any, agent: str, order: Any, status: str,
                     note: str) -> None:
    """The report's outcome moves the step the order claimed, by id.

    By id and not by "the agent's current step": between the claim and the
    report the agent may have been handed more, and closing the wrong one is
    worse than closing none.
    """
    try:
        from CoScientist.tools.task_tracker import set_task_status

        task_id = str(getattr(order, "plan_task_id", "") or "")
        if not task_id:
            return
        set_task_status(state, task_id, status, notes=note, agent=agent)
    except Exception as exc:  # noqa: BLE001
        logger.warning("closing plan step for %s failed: %s", agent, exc)


class WorkOrderToolset:
    """declare_work_order / update_work_order / update_work_step /
    submit_work_report for one agent."""

    def __init__(
        self,
        agent_name: str,
        valid_tool_names: Optional[Iterable[str]] = None,
        handler: Any = None,
        internal_tools: Iterable[str] = (),
        step_review: bool = False,
    ) -> None:
        self.agent_name = agent_name
        self.step_review = step_review
        # Allowed without declaring and kept apart from the contract's tools, so
        # they neither raise its tier nor show on the card unless the viewer
        # asks. The agent may still name them — that is no error.
        self.exempt = exempt_tools(internal_tools)
        # None: the agent's tool surface is resolved at runtime — names can't be checked.
        self.valid_tool_names = (
            None if valid_tool_names is None
            else set(valid_tool_names) - self.exempt
        )
        self._handler_override = handler

    # ── plumbing ────────────────────────────────────────────────────────────
    @property
    def handler(self):
        if self._handler_override is not None:
            return self._handler_override
        from CoScientist.agents.common import hitl_handler
        return hitl_handler

    def tools(self) -> List[FunctionTool]:
        return [
            FunctionTool(self.declare_work_order),
            FunctionTool(self.update_work_order),
            FunctionTool(self.update_work_step),
            FunctionTool(self.submit_work_report),
        ]

    def _unknown_tools(self, names: Iterable[str]) -> List[str]:
        if self.valid_tool_names is None:
            return []
        return [n for n in names if n not in self.valid_tool_names]


    def _parse_steps(self, raw: Optional[List[Any]], start: int = 1) -> List[WorkStep] | str:
        steps: List[WorkStep] = []
        for offset, item in enumerate(raw or []):
            if isinstance(item, str):
                item = {"title": item}
            if not isinstance(item, dict) or not str(item.get("title") or "").strip():
                return f"Step at index {offset} needs a non-empty 'title'."
            tools = item.get("tools") or []
            if isinstance(tools, str):
                tools = [tools]
            tools = [str(t) for t in tools]
            external = [t for t in tools if t not in self.exempt]
            inputs = str(item.get("inputs") or "").strip()
            if self.step_review and external and not inputs:
                return (
                    f"Step at index {offset} ('{str(item['title']).strip()}') calls "
                    f"{external} but has no 'inputs': state what you will send to "
                    f"these tools (queries, names, SMILES, parameters)."
                )
            steps.append(WorkStep(
                id=f"S{start + offset}",
                title=str(item["title"]).strip(),
                tools=external,
                internal_tools=self._internal_named(tools),
                inputs=inputs,
                expected_outcome=str(item.get("expected_outcome") or ""),
            ))
        return steps

    def _internal_named(self, names: Iterable[str], into: Optional[List[str]] = None) -> List[str]:
        """The internal tools among ``names``, appended to ``into`` without
        repeats. Protocol tools are not listed: nobody needs to see those."""
        found = list(into or [])
        for name in names:
            if name in self.exempt and name not in EXEMPT_TOOLS and name not in found:
                found.append(name)
        return found

    @staticmethod
    def _parse_assumptions(raw: Optional[List[Any]]) -> List[Assumption] | str:
        assumptions: List[Assumption] = []
        for offset, item in enumerate(raw or []):
            # A {"text": ...} dict is the older shape; models still send it now and then.
            if isinstance(item, dict):
                item = item.get("text")
            text = str(item or "").strip()
            if not text:
                return f"Assumption at index {offset} must be a non-empty string."
            assumptions.append(Assumption(id=f"A{offset + 1}", text=text))
        return assumptions


    @staticmethod
    def _implied_side_effects(tools: Iterable[str], effects: List[SideEffect]) -> None:
        """A tool whose very nature is a side effect declares it implicitly."""
        declared = {e.kind for e in effects}
        for tool in tools:
            kind = TOOL_SIDE_EFFECTS.get(tool)
            if kind is not None and kind not in declared:
                effects.append(SideEffect(kind=kind, detail=f"implied by {tool}"))
                declared.add(kind)

    async def _review(
        self,
        order: WorkOrder,
        tier: Tier,
        tool_context: Any,
        *,
        trigger: str,
        extra_context: Optional[Dict[str, Any]] = None,
        force_blocking: bool = False,
    ) -> Optional[HITLResponse]:
        """Put the contract (or amendment) in front of the human by tier.

        Returns None when nobody is asked (off, or a read-tier amendment
        notice): the caller treats that as approved.
        """
        session = session_context(tool_context)
        if not work_order_active():
            return None
        # The contract is read by the same person the report is written for.
        lang = session_report_language(tool_context)
        # A read-only amendment is just a notice; the contract itself always goes
        # to the human (under the veto window) — otherwise the agent starts first.
        if tier == Tier.READ and trigger != "work_order" and not force_blocking:
            try:
                await self.handler.notify({
                    "kind": "amended",
                    "agent_name": self.agent_name,
                    "tier": tier.value,
                    "work_order": order.model_dump(mode="json"),
                    "text": render_work_order(order, lang),
                    **(extra_context or {}),
                    "_session": session,
                })
            except Exception:  # noqa: BLE001 - a notice must never fail the run
                logger.exception("%s: work order notice failed", self.agent_name)
            return None

        request = HITLRequest(
            agent_name=self.agent_name,
            action_type=HITLAction.APPROVE,
            message=(
                f"Agent '{self.agent_name}' declares a work order "
                f"(rev {order.revision}, tier {tier.value}). Please review."
            ),
            context={
                "work_order": order.model_dump(mode="json"),
                "tier": tier.value,
                "output": render_work_order(order, lang),
                **(extra_context or {}),
                "_session": session,
            },
            invoked_via="tool",
            trigger=trigger,
            timeout_seconds=None,
        )
        return await self.handler.handle_request(request)

    @staticmethod
    def _review_timeout(tier: Tier) -> Optional[float]:
        """Compatibility shim: the global HITL mode owns every review window."""
        return None

    async def review_step(
        self, order: WorkOrder, step: WorkStep, tool_context: Any
    ) -> Optional[HITLResponse]:
        """Put one finished step before the human: sent / expected / found.

        Returns None when Work Orders are off: the caller treats that as accepted.
        """
        if not work_order_active():
            return None
        review = step.review or StepReview()
        request = HITLRequest(
            agent_name=self.agent_name,
            action_type=HITLAction.APPROVE,
            message=(
                f"Agent '{self.agent_name}' finished step {step.id} of its work order "
                f"(rev {order.revision}, round {review.round}). Please check what it "
                f"sent and what it found."
            ),
            context={
                "work_order": order.model_dump(mode="json"),
                "step": step.model_dump(mode="json"),
                "step_calls": list(step.calls),
                "tier": order.tier.value,
                "output": render_work_step_review(order, step),
                "_session": session_context(tool_context),
            },
            invoked_via="tool",
            trigger="work_step",
            timeout_seconds=self._review_timeout(order.tier),
        )
        return await self.handler.handle_request(request)

    async def review_report(self, order: WorkOrder, tool_context: Any) -> Optional[HITLResponse]:
        """Put ``order.report`` before the human at the order's tier.

        Returns None when Work Orders are off: the caller treats that as accepted.
        """
        if not work_order_active():
            return None
        lang = session_report_language(tool_context)
        report = order.report or WorkReport()
        request = HITLRequest(
            agent_name=self.agent_name,
            action_type=HITLAction.APPROVE,
            message=(
                f"Agent '{self.agent_name}' reports the result of its work order "
                f"(rev {order.revision}, round {report.round}). Please check it."
            ),
            context={
                "work_order": order.model_dump(mode="json"),
                "work_report": report.model_dump(mode="json"),
                "tier": order.tier.value,
                "warnings": report_warnings(order),
                "journal": {
                    "tool_calls": dict(order.tool_calls),
                    "side_effects": performed_side_effects(order),
                    "amendments": list(order.amendments),
                    "deviations": list(order.deviations),
                },
                # The session is what turns an artifact reference into a
                # link; the renderer cannot ask for it on its own.
                "output": render_work_report(
                    order, lang, scope=session_key(tool_context)),
                "_session": session_context(tool_context),
            },
            invoked_via="tool",
            trigger="work_report",
            timeout_seconds=None,
        )
        return await self.handler.handle_request(request)

    @staticmethod
    def _parse_findings(raw: Optional[List[Any]]) -> List[Finding] | str:
        findings: List[Finding] = []
        for offset, item in enumerate(raw or []):
            if isinstance(item, str):
                item = {"text": item}
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                return f"Finding at index {offset} needs a non-empty 'text'."
            confidence = str(item.get("confidence") or "medium").lower()
            if confidence not in _CONFIDENCES:
                return f"Finding at index {offset}: 'confidence' must be one of {list(_CONFIDENCES)}."
            findings.append(Finding(
                id=f"F{offset + 1}",
                text=str(item["text"]).strip(),
                evidence=str(item.get("evidence") or "").strip(),
                confidence=confidence,  # type: ignore[arg-type]
                step_id=str(item.get("step_id") or ""),
            ))
        return findings

    @staticmethod
    def _parse_artifacts(raw: Optional[List[Any]]) -> List[Artifact] | str:
        artifacts: List[Artifact] = []
        for offset, item in enumerate(raw or []):
            if isinstance(item, str):
                item = {"ref": item}
            if not isinstance(item, dict) or not str(item.get("ref") or "").strip():
                return f"Artifact at index {offset} needs a non-empty 'ref'."
            kind = str(item.get("kind") or "other").lower()
            if kind not in _ARTIFACT_KINDS:
                kind = "other"
            artifacts.append(Artifact(
                kind=kind,  # type: ignore[arg-type]
                ref=str(item["ref"]).strip(),
                description=str(item.get("description") or "").strip(),
            ))
        return artifacts

    # ── tools ───────────────────────────────────────────────────────────────
    async def declare_work_order(
        self,
        goal: str,
        done_criteria: str,
        assumptions: List[str],
        steps: List[Dict[str, Any]],
        planned_tools: List[str],
        tool_context: ToolContext,
        expected_outcome: str = "",
        fallback: str = "",
    ) -> Dict[str, Any]:
        """Declare your work order BEFORE your first external action.

        The human sees it and may approve, adjust or reject it. After approval
        you may only call the tools you declared; to need more, call
        update_work_order with a reason.

        Args:
            goal: What this run must achieve, in one or two sentences.
            done_criteria: How you will know you are done.
            assumptions: Every assumption your plan relies on, one string per
                item. Keep assumptions atomic (one condition per item) and non-trivial.
            steps: Ordered steps, each {"title", "tools": [tool names],
                "inputs", "expected_outcome"}. "inputs" is what you will send
                to the step's tools (queries, names, SMILES, parameters);
                "expected_outcome" is what you expect to get back.
            planned_tools: All tool names you intend to call.
            expected_outcome: What result you expect (be concrete: counts,
                ranges, metrics).
            fallback: What you will do if the plan does not work.

        Returns:
            {"status": "approved" | "revise" | "rejected" | "error", ...}.
        """
        state = tool_context.state
        current = load_order(state, self.agent_name)
        if current is not None and current.status == "approved":
            return _error(
                "A work order is already approved for this run. To change it, "
                "call update_work_order with a reason."
            )
        if current is not None and current.status == "rejected":
            return {
                "status": "rejected",
                "message": "The human rejected your work order. Do not act; "
                           "finish and report why the task was not carried out.",
            }
        if not str(goal or "").strip():
            return _error("'goal' must not be empty.")

        parsed_assumptions = self._parse_assumptions(assumptions)
        if isinstance(parsed_assumptions, str):
            return _error(parsed_assumptions)
        parsed_steps = self._parse_steps(steps)
        if isinstance(parsed_steps, str):
            return _error(parsed_steps)
        named = [str(t) for t in planned_tools or []]
        tools: List[str] = []
        for name in named + [t for s in parsed_steps for t in s.tools]:
            if name not in tools and name not in self.exempt:
                tools.append(name)
        internal = self._internal_named(
            named + [t for s in parsed_steps for t in s.internal_tools]
        )
        unknown = self._unknown_tools(tools)
        if unknown:
            return _error(
                f"Unknown tools {unknown}. You can only plan tools you have: "
                f"{sorted(self.valid_tool_names or [])}."
            )
        effects: List[SideEffect] = []
        self._implied_side_effects(tools, effects)

        order = WorkOrder(
            agent=self.agent_name,
            goal=str(goal).strip(),
            done_criteria=str(done_criteria or ""),
            assumptions=parsed_assumptions,
            steps=parsed_steps,
            planned_tools=tools,
            internal_tools=internal,
            side_effects=effects,
            expected_outcome=str(expected_outcome or ""),
            fallback=str(fallback or ""),
            step_review=self.step_review,
        )
        tier = order.recompute_tier()

        response = await self._review(order, tier, tool_context, trigger="work_order")
        return self._settle_declaration(state, order, response)

    def _settle_declaration(
        self, state: Any, order: WorkOrder, response: Optional[HITLResponse]
    ) -> Dict[str, Any]:
        feedback = (response.instructions or response.free_input or "").strip() if response else ""

        if response is not None and response.action == HITLAction.EDIT:
            result: Dict[str, Any] = {
                "status": "revise",
                "feedback": feedback or "No feedback provided.",
                "message": "The human asked for changes. Declare a revised work "
                           "order with declare_work_order before acting.",
            }
            # An operator who unticks an assumption and writes a note is doing
            # one thing, not two, and the note is what turns this into a revise.
            # Only the approve path used to read the unticks, so on this path
            # they reached nobody — the agent was sent back to the drawing board
            # without being told which premise had been refused.
            rejected = _rejected_ids(response)
            named = [a.text for a in order.assumptions if a.id in rejected]
            if named:
                result["rejected_assumptions"] = named
                result["message"] += (" The human REJECTED the listed assumptions: "
                                      "the revised order must not rest on them.")
            return result
        if response is not None and getattr(response, "timed_out", False):
            # Nobody answered. The action is NOT approved — silence consents to
            # nothing — but this is not the human refusing, and it must not be
            # recorded as one: a persisted `rejected` order makes every later
            # `declare_work_order` return that rejection without asking, so one
            # missed card killed the agent for the rest of the run even after
            # the operator came back. Nothing is saved, so declaring again asks
            # again.
            return {
                "status": "unanswered",
                "reason": "Nobody answered the work order within the review window.",
                "message": "Nobody confirmed your work order, so you may not act on "
                           "it. Do not treat this as a refusal of the plan: declare "
                           "again if the work is still needed, or finish and report "
                           "that the contract was never confirmed.",
            }
        if response is not None and not response.approved:
            order.status = "rejected"
            order.operator_notes = feedback
            save_order(state, order)
            return {
                "status": "rejected",
                "reason": feedback or "No reason given.",
                "message": "The human rejected your work order. Do not act; "
                           "finish and report why the task was not carried out.",
            }

        rejected_ids = _rejected_ids(response)
        for assumption in order.assumptions:
            assumption.rejected = assumption.id in rejected_ids
        order.status = "approved"
        order.operator_notes = feedback
        _claim_plan_step(state, self.agent_name, order)
        save_order(state, order)

        result: Dict[str, Any] = {
            "status": "approved",
            "mode": "autonomous" if not work_order_active() else order.tier.value,
            "revision": order.revision,
            "steps": [{"id": s.id, "title": s.title} for s in order.steps],
            "message": "Proceed within this work order. Mark each step with "
                       "update_work_step as you start and finish it.",
        }
        rejected = [a.text for a in order.assumptions if a.rejected]
        if rejected:
            result["rejected_assumptions"] = rejected
            result["message"] += (" The human REJECTED the listed assumptions: do not "
                                  "rely on them — adapt the plan accordingly.")
        if feedback:
            result["operator_notes"] = feedback
            result["message"] += " Follow the operator notes."
        return result

    async def update_work_order(
        self,
        reason: str,
        tool_context: ToolContext,
        add_tools: Optional[List[str]] = None,
        add_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Amend your approved work order when you need to go beyond it.

        Call this when a tool call was blocked by the work order guard, or
        before doing something the contract does not cover. The human sees the
        change as a diff.

        Args:
            reason: Why the amendment is needed (what you learned).
            add_tools: Tool names to add to the plan.
            add_steps: Steps to append, each {"title", "tools", "inputs",
                "expected_outcome"}.

        Returns:
            {"status": "approved" | "revise" | "rejected" | "error", ...}.
        """
        state = tool_context.state
        old = load_order(state, self.agent_name)
        if old is None or old.status != "approved":
            return _error("There is no approved work order to amend — call declare_work_order first.")
        if not str(reason or "").strip():
            return _error("'reason' must explain why the amendment is needed.")

        new_steps = self._parse_steps(add_steps, start=len(old.steps) + 1)
        if isinstance(new_steps, str):
            return _error(new_steps)

        new = old.model_copy(deep=True)
        named = [str(t) for t in add_tools or []]
        for name in named + [t for s in new_steps for t in s.tools]:
            if name not in new.planned_tools and name not in self.exempt:
                new.planned_tools.append(name)
        new.internal_tools = self._internal_named(
            named + [t for s in new_steps for t in s.internal_tools], into=new.internal_tools
        )
        unknown = self._unknown_tools(new.planned_tools)
        if unknown:
            return _error(
                f"Unknown tools {unknown}. You can only plan tools you have: "
                f"{sorted(self.valid_tool_names or [])}."
            )
        new.steps.extend(new_steps)
        self._implied_side_effects(new.planned_tools, new.side_effects)
        new.revision = old.revision + 1

        diff = diff_work_orders(old, new)
        if not any(diff.values()):
            if new.internal_tools != old.internal_tools:
                # Nothing to review: internal tools were never blocked.
                old.internal_tools = new.internal_tools
                save_order(state, old)
                return {
                    "status": "approved",
                    "revision": old.revision,
                    "message": "No amendment needed: those tools are always allowed.",
                }
            return _error("The amendment changes nothing. Name the tools or steps you need.")

        delta_tier = max_tier(
            [tool_tier(t) for t in diff["added_tools"]]
            + ([Tier.SIDE_EFFECT] if diff["added_side_effects"] else [])
        )
        new.recompute_tier()
        force_blocking = len(old.amendments) >= get_settings().web.work_order_max_amendments

        response = await self._review(
            new, delta_tier, tool_context,
            trigger="work_order_amendment",
            extra_context={"diff": diff, "reason": str(reason).strip()},
            force_blocking=force_blocking,
        )
        feedback = (response.instructions or response.free_input or "").strip() if response else ""

        if response is not None and response.action == HITLAction.EDIT:
            return {
                "status": "revise",
                "feedback": feedback or "No feedback provided.",
                "message": "The human asked for a different amendment. Adjust it "
                           "and call update_work_order again, or stay within the current order.",
            }
        if response is not None and response.timed_out:
            return {
                "status": "unanswered",
                "reason": "Nobody answered the amendment within the review window.",
                "message": "The current work order remains active and unchanged. "
                           "Submit the amendment again before using its new tools or steps.",
            }
        if response is not None and not response.approved:
            return {
                "status": "rejected",
                "reason": feedback or "No reason given.",
                "message": "The amendment was rejected. Continue within the current "
                           "work order, or finish and report what could not be done.",
            }

        new.amendments.append({
            "revision": new.revision,
            "reason": str(reason).strip(),
            "diff": diff,
            "operator_notes": feedback,
        })
        new.status = "approved"
        save_order(state, new)
        result = {
            "status": "approved",
            "revision": new.revision,
            "message": "Amendment approved. Proceed within the updated work order.",
        }
        if feedback:
            result["operator_notes"] = feedback
        return result

    async def update_work_step(
        self,
        step_id: str,
        status: str,
        tool_context: ToolContext,
        note: str = "",
        result: str = "",
    ) -> Dict[str, Any]:
        """Report progress on a step of your approved work order.

        Mark a step in_progress BEFORE calling its tools, and done (or skipped)
        when it is finished. With step review on, a finished step goes before
        the human, who sees what you sent, what you expected and what you found.

        Args:
            step_id: The step id from the approved order (e.g. "S1").
            status: One of pending, in_progress, done, skipped.
            note: Short note: what came out, or why it was skipped.
            result: For done: what the step actually produced — concrete
                values, counts, ids — to set against its expected outcome.

        Returns:
            {"status": "ok" | "accepted" | "revise" | "rejected" | "error", ...}.
        """
        state = tool_context.state
        order = load_order(state, self.agent_name)
        if order is None or order.status != "approved":
            return _error("There is no approved work order — call declare_work_order first.")
        if status not in _STEP_STATUSES:
            return _error(f"'status' must be one of {list(_STEP_STATUSES)}.")
        step = order.step(step_id)
        if step is None:
            return _error(f"Unknown step {step_id!r}; steps are {[s.id for s in order.steps]}.")
        reviewed = order.step_review and status in ("done", "skipped")
        if reviewed and status == "done" and not str(result or "").strip():
            return _error(
                f"Step {step_id} is reviewed by the human: pass 'result' — what the "
                f"step actually produced (values, counts, ids)."
            )
        step.status = status  # type: ignore[assignment]
        step.note = str(note or "")
        if result or status == "done":
            step.result = str(result or "")
        save_order(state, order)

        if not reviewed:
            await self._notify_progress(order, step, tool_context)
            return {"status": "ok", "step": step.id, "step_status": step.status}

        if step.review is None:
            step.review = StepReview()
        response = await self.review_step(order, step, tool_context)
        settled = self._settle_step(state, order, step, response)
        # The order card shows the step row too: refresh it with the verdict.
        await self._notify_progress(order, step, tool_context)
        return settled

    async def _notify_progress(self, order: WorkOrder, step: WorkStep, tool_context: Any) -> None:
        if not work_order_active():
            return
        try:
            await self.handler.notify({
                "kind": "progress",
                "agent_name": self.agent_name,
                "revision": order.revision,
                "step": step.model_dump(mode="json"),
                "_session": session_context(tool_context),
            })
        except Exception:  # noqa: BLE001 - a notice must never fail the run
            logger.exception("%s: work step notice failed", self.agent_name)

    @staticmethod
    def _settle_step(
        state: Any, order: WorkOrder, step: WorkStep, response: Optional[HITLResponse]
    ) -> Dict[str, Any]:
        review = step.review or StepReview()
        feedback = (response.instructions or response.free_input or "").strip() if response else ""

        if response is not None and response.action == HITLAction.EDIT:
            review.history.append({
                "round": review.round,
                "result": step.result,
                "note": step.note,
                "calls": list(step.calls),
                "notes": feedback,
            })
            review.status = "revise"
            review.notes = feedback
            review.round += 1
            step.status = "in_progress"
            step.calls = []
            step.review = review
            save_order(state, order)
            return {
                "status": "revise",
                "step_id": step.id,
                "feedback": feedback or "No feedback provided.",
                "message": (
                    f"The human sent step {step.id} back. Redo it as the feedback "
                    f"says, then mark it done again with the new result. If you "
                    f"need tools beyond your work order, call update_work_order first."
                ),
            }

        if response is not None and response.timed_out:
            review.status = "pending"
            review.notes = ""
            step.status = "in_progress"
            step.review = review
            save_order(state, order)
            return {
                "status": "unanswered",
                "step_id": step.id,
                "reason": "Nobody answered the step review within the review window.",
                "message": "The step remains in progress and must be submitted for review again.",
            }

        if response is not None and not response.approved:
            review.status = "rejected"
            review.notes = feedback
            step.review = review
            order.status = "rejected"
            order.operator_notes = feedback
            save_order(state, order)
            return {
                "status": "rejected",
                "step_id": step.id,
                "reason": feedback or "No reason given.",
                "message": (
                    "The human stopped your work order at this step. Do not call "
                    "any more tools; finish and state plainly what was done, what "
                    "was not, and why."
                ),
            }

        review.status = "accepted"
        review.notes = feedback
        step.review = review
        save_order(state, order)
        result: Dict[str, Any] = {
            "status": "accepted",
            "step_id": step.id,
            "message": "The human accepted this step. Go on to the next one.",
        }
        if feedback:
            result["operator_notes"] = feedback
            result["message"] += " Follow the operator notes."
        return result

    async def submit_work_report(
        self,
        summary: str,
        findings: List[Dict[str, Any]],
        done_verdict: str,
        tool_context: ToolContext,
        done_evidence: str = "",
        actual_outcome: str = "",
        artifacts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Report what you did and found BEFORE your final answer.

        The human checks the report against your work order and may accept it,
        send you back to rework it, or reject it.

        Args:
            summary: The result in 2-4 sentences.
            findings: What you found, one item per finding: {"text", "evidence"
                (links, DOIs, file paths, graph node ids backing it),
                "confidence": high | medium | low, "step_id" (e.g. "S2")}.
            done_verdict: Whether the done criteria are met: met | partial | not_met.
            done_evidence: Why you judge the done criteria so (counts, checks).
            actual_outcome: The concrete result to set against the expected
                outcome (counts, ranges, metrics).
            artifacts: What you produced, each {"kind": file | dataset |
                graph_node | link | other, "ref", "description"}. Only what exists.

        Returns:
            {"status": "accepted" | "revise" | "rejected" | "error", ...}.
        """
        state = tool_context.state
        order = load_order(state, self.agent_name)
        if order is None or order.status != "approved":
            return _error("There is no approved work order to report on — call declare_work_order first.")
        if order.report is not None and order.report.status in ("accepted", "rejected"):
            return _error(
                f"Your work report was already {order.report.status}. "
                "Give your final answer now."
            )
        if not str(summary or "").strip():
            return _error("'summary' must not be empty.")
        verdict = str(done_verdict or "").strip().lower()
        if verdict not in _DONE_VERDICTS:
            return _error(f"'done_verdict' must be one of {list(_DONE_VERDICTS)}.")
        parsed_findings = self._parse_findings(findings)
        if isinstance(parsed_findings, str):
            return _error(parsed_findings)
        parsed_artifacts = self._parse_artifacts(artifacts)
        if isinstance(parsed_artifacts, str):
            return _error(parsed_artifacts)

        previous_round = order.report.round if order.report is not None else 0
        order.report = WorkReport(
            summary=str(summary).strip(),
            findings=parsed_findings,
            done_verdict=verdict,  # type: ignore[arg-type]
            done_evidence=str(done_evidence or "").strip(),
            actual_outcome=str(actual_outcome or "").strip(),
            artifacts=parsed_artifacts,
            round=previous_round + 1,
        )
        response = await self.review_report(order, tool_context)
        feedback = (response.instructions or response.free_input or "").strip() if response else ""
        report = order.report

        disputed = sorted(_disputed_ids(response))
        if response is not None and response.action == HITLAction.EDIT:
            report.status = "revise"
            report.operator_notes = feedback
            report.disputed_finding_ids = disputed
            order.reports.append(report.model_dump(mode="json"))
            save_order(state, order)
            result: Dict[str, Any] = {
                "status": "revise",
                "round": report.round,
                "feedback": feedback or "No feedback provided.",
                "message": "The human sent your result back for rework. Address the "
                           "feedback, then call submit_work_report again. If you need "
                           "tools beyond your work order, call update_work_order first.",
            }
            if disputed:
                result["disputed_findings"] = [
                    {"id": f.id, "text": f.text} for f in report.findings if f.id in disputed
                ]
                result["message"] += " The human marked the listed findings as wrong: recheck them."
            return result

        if response is not None and getattr(response, "timed_out", False):
            # Same distinction as the declaration: an unanswered report is not a
            # rejected one. Nothing is persisted and the plan step is left where
            # it is — failing it would close a step nobody judged.
            return {
                "status": "unanswered",
                "reason": "Nobody answered the work report within the review window.",
                "message": "Nobody checked your result. Do not treat this as a "
                           "rejection: submit the work report again before moving "
                           "past this review.",
            }
        if response is not None and not response.approved:
            report.status = "rejected"
            report.operator_notes = feedback
            # The step did not complete and the agent is told not to continue,
            # so it is failed, not left running.
            _close_plan_step(state, self.agent_name, order, "FAILED",
                             "work report rejected")
            save_order(state, order)
            return {
                "status": "rejected",
                "reason": feedback or "No reason given.",
                "message": "The human rejected your result. Do not continue; finish and "
                           "state plainly that the result was not accepted and why.",
            }

        report.status = "accepted"
        report.operator_notes = feedback
        # Accepting the result and doubting a particular finding are not the
        # same act, and an operator may well do both. Only the rework path read
        # these marks, so a finding ticked as wrong on an otherwise accepted
        # report was recorded nowhere and mentioned to nobody.
        report.disputed_finding_ids = disputed
        _close_plan_step(state, self.agent_name, order, "DONE",
                         "work report accepted")
        save_order(state, order)
        result = {
            "status": "accepted",
            "round": report.round,
            "message": "The human accepted your report. Give your final answer based on it.",
        }
        if feedback:
            result["operator_notes"] = feedback
            result["message"] += " Take the operator notes into account."
        if disputed:
            result["disputed_findings"] = [
                {"id": f.id, "text": f.text} for f in report.findings
                if f.id in disputed
            ]
            result["message"] += (" The human accepted the report but marked the "
                                  "listed findings as wrong: do not carry them "
                                  "into the answer as established.")
        return result


def make_work_order_tools(
    agent_name: str,
    valid_tool_names: Optional[Iterable[str]] = None,
    internal_tools: Iterable[str] = (),
    step_review: bool = False,
) -> List[FunctionTool]:
    return WorkOrderToolset(
        agent_name, valid_tool_names, internal_tools=internal_tools, step_review=step_review,
    ).tools()
