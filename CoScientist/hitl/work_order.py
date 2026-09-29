"""Work Order — the contract an executor agent declares before it acts.

Before its first external action an agent states WHAT it is going to do and on
WHICH assumptions: goal, done criteria, assumptions, steps, tools and the
outcome it expects (side effects follow from the tools). The human reviews that contract (by risk
tier, see work_order_risk.py) and the guard (work_order_guard.py) then keeps
the agent inside it. Needing more is not forbidden — it is an amendment the
agent must justify, and the human sees it as a diff.

State lives in flat, per-agent keys that are always written whole: AgentTool
forwards a sub-agent's ``state_delta`` to the parent, and mutating a nested dict
in place produces no delta at all.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from CoScientist.hitl.work_order_risk import SideEffectKind, Tier, order_tier

StepStatus = Literal["pending", "in_progress", "done", "skipped"]
OrderStatus = Literal["pending", "approved", "rejected"]
ReportStatus = Literal["pending", "accepted", "revise", "rejected"]
DoneVerdict = Literal["met", "partial", "not_met"]
Confidence = Literal["high", "medium", "low"]
ArtifactKind = Literal["file", "dataset", "graph_node", "link", "other"]
StepReviewStatus = Literal["pending", "accepted", "revise", "rejected"]

# How much of a tool answer the step journal keeps: enough to judge the step,
# small enough that the flat per-agent state key stays cheap to rewrite whole.
STEP_CALL_ARGS_CHARS = 500
STEP_CALL_RESULT_CHARS = 2000


def order_key(agent_name: str) -> str:
    return f"work_order:{agent_name}"


class StepReview(BaseModel):
    """The human's verdict on one finished step (step-review mode only)."""
    status: StepReviewStatus = "pending"
    round: int = 1
    notes: str = ""
    # Earlier rounds the human sent back: what was claimed and called then.
    history: List[Dict[str, Any]] = Field(default_factory=list)


class WorkStep(BaseModel):
    id: str
    title: str
    tools: List[str] = Field(default_factory=list)
    # Internal tools the agent named for this step: allowed anyway, kept out of
    # `tools` (and so out of the tier), shown only when the viewer asks for them.
    internal_tools: List[str] = Field(default_factory=list)
    # What the agent will SEND to its tools in this step (queries, names,
    # SMILES, parameters) — the human sees it before anything leaves.
    inputs: str = ""
    expected_outcome: str = ""
    status: StepStatus = "pending"
    note: str = ""
    # What the agent says the step produced (concrete values, ids).
    result: str = ""
    # Recorded by the system, not claimed by the agent: every tool call made
    # while this step was in progress — {tool, args, result_excerpt, is_error}.
    calls: List[Dict[str, Any]] = Field(default_factory=list)
    review: Optional[StepReview] = None


class Assumption(BaseModel):
    id: str
    text: str
    rejected: bool = False


class SideEffect(BaseModel):
    kind: SideEffectKind
    detail: str = ""


class Finding(BaseModel):
    id: str
    text: str
    evidence: str = ""
    confidence: Confidence = "medium"
    step_id: str = ""


class Artifact(BaseModel):
    kind: ArtifactKind = "other"
    ref: str
    description: str = ""


class WorkReport(BaseModel):
    """What the agent says it found and made, put before the human at the end.

    Everything here is the agent's claim; the order around it (steps, tool calls,
    amendments, deviations) is what the system recorded — the card shows both.
    """
    summary: str = ""
    findings: List[Finding] = Field(default_factory=list)
    done_verdict: DoneVerdict = "not_met"
    done_evidence: str = ""
    actual_outcome: str = ""
    artifacts: List[Artifact] = Field(default_factory=list)
    status: ReportStatus = "pending"
    round: int = 1
    operator_notes: str = ""
    disputed_finding_ids: List[str] = Field(default_factory=list)
    # Built by the after_agent fallback from the order alone: the agent never reported.
    fallback: bool = False


class WorkOrder(BaseModel):
    agent: str
    goal: str
    # The outer plan's step this order carries out (TASK-n), claimed when the
    # order is approved. Empty when the run has no plan, or when every step of
    # this agent's is already finished.
    plan_task_id: str = ""
    done_criteria: str = ""
    assumptions: List[Assumption] = Field(default_factory=list)
    steps: List[WorkStep] = Field(default_factory=list)
    planned_tools: List[str] = Field(default_factory=list)
    # Internal tools the agent named anywhere in the order (see WorkStep).
    internal_tools: List[str] = Field(default_factory=list)
    side_effects: List[SideEffect] = Field(default_factory=list)
    expected_outcome: str = ""
    fallback: str = ""
    # Every finished step goes before the human (sent / expected / found).
    step_review: bool = False
    tier: Tier = Tier.READ
    status: OrderStatus = "pending"
    revision: int = 1
    amendments: List[Dict[str, Any]] = Field(default_factory=list)
    deviations: List[Dict[str, Any]] = Field(default_factory=list)
    operator_notes: str = ""
    # Calls the guard let through, per tool (exempt and internal tools not counted).
    tool_calls: Dict[str, int] = Field(default_factory=dict)
    report: Optional[WorkReport] = None
    # Earlier report rounds the human sent back, oldest first.
    reports: List[Dict[str, Any]] = Field(default_factory=list)

    def side_effect_kinds(self) -> set:
        return {s.kind for s in self.side_effects}

    def recompute_tier(self) -> Tier:
        self.tier = order_tier(self.planned_tools, self.side_effect_kinds())
        return self.tier

    def step(self, step_id: str) -> Optional[WorkStep]:
        return next((s for s in self.steps if s.id == step_id), None)

    def step_for_call(self, tool_name: str) -> Optional[WorkStep]:
        """The in-progress step a call of ``tool_name`` belongs to.

        The in-progress step that declares the tool wins; otherwise the only
        in-progress step; with several and none declaring it — no step.
        """
        active = [s for s in self.steps if s.status == "in_progress"]
        declaring = [s for s in active if tool_name in s.tools]
        if declaring:
            return declaring[0]
        return active[0] if len(active) == 1 else None


def load_order(state: Any, agent_name: str) -> Optional[WorkOrder]:
    raw = state.get(order_key(agent_name)) if state is not None else None
    if not raw:
        return None
    try:
        return WorkOrder.model_validate(raw)
    except Exception:  # noqa: BLE001 - a corrupt entry behaves as "no order"
        return None


def save_order(state: Any, order: WorkOrder) -> None:
    state[order_key(order.agent)] = order.model_dump(mode="json")


_STEP_MARK = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]", "skipped": "[-]"}


# The frame around an agent's own words. The agent writes its goal and its
# steps in the session's language; these labels used to be English regardless,
# so a Russian study read "Goal: Сформировать…". One table, two columns, kept
# parallel line for line.
_WO_WORDS = {
    "en": {
        "order": "Work Order", "report": "Work Report", "rev": "rev", "round": "round",
        "tier": "tier", "goal": "Goal", "done": "Done when", "assumptions": "Assumptions",
        "steps": "Steps", "expect": "expected", "send": "send", "tools": "Tools",
        "effects": "Side effects", "outcome": "Expected outcome", "fallback": "If it fails",
        "rejected": "rejected", "summary": "Summary", "expected": "Expected",
        "actual": "Actual", "findings": "Findings", "evidence": "evidence",
        "artifacts": "Artifacts", "calls": "Tool calls",
        "performed": "Side effects performed", "amendments": "Amendments",
        "blocked": "Blocked calls",
    },
    "ru": {
        "order": "План работы агента", "report": "Отчёт о работе", "rev": "Ревизия",
        "round": "раунд", "tier": "Уровень", "goal": "Цель",
        "done": "Критерий готовности", "assumptions": "Условия и ограничения",
        "steps": "Шаги", "expect": "Ожидается", "send": "Передать", "tools": "Инструменты",
        "effects": "Побочные эффекты", "outcome": "Ожидаемый результат",
        "fallback": "Если не получится", "rejected": "Отклонено",
        "summary": "Итог", "expected": "Ожидалось", "actual": "Получено",
        "findings": "Находки", "evidence": "Подтверждение",
        "artifacts": "Артефакты", "calls": "Вызовы инструментов",
        "performed": "Выполненные побочные эффекты", "amendments": "Дополнения",
        "blocked": "Заблокированные вызовы",
    },
}


def _words(lang) -> dict:
    from CoScientist.agents.callbacks.report_language import normalize_report_language

    return _WO_WORDS[normalize_report_language(lang)]


def render_work_order(order: WorkOrder, lang: str = "en") -> str:
    """The contract as Markdown, in the session's language.

    Markdown rather than plain text because this is what the chat writes into a
    document and renders in a panel; the console handler shows the same source,
    which reads no worse for the headings.
    """
    w = _words(lang)
    lines = [
        f"# {w['order']} — {order.agent}",
        "",
        f"*{w['rev']} {order.revision} · {w['tier']} {order.tier.value}*",
        "",
        f"## {w['goal']}", "", order.goal or "—",
    ]
    if order.done_criteria:
        lines += ["", f"## {w['done']}", "", order.done_criteria]
    if order.assumptions:
        lines += ["", f"## {w['assumptions']}", ""]
        for a in order.assumptions:
            mark = f" _({w['rejected']})_" if a.rejected else ""
            lines.append(f"- **{a.id}** {a.text}{mark}")
    if order.steps:
        lines += ["", f"## {w['steps']}", ""]
        for s in order.steps:
            tools = f" — `{'`, `'.join(s.tools)}`" if s.tools else ""
            lines.append(f"- {_STEP_MARK.get(s.status, '[ ]')} **{s.id}** {s.title}{tools}")
            if s.inputs:
                lines.append(f"  - {w['send']}: {s.inputs}")
            if s.expected_outcome:
                lines.append(f"  - {w['expect']}: {s.expected_outcome}")
    if order.planned_tools:
        lines += ["", f"## {w['tools']}", "", "`" + "`, `".join(order.planned_tools) + "`"]
    if order.side_effects:
        lines += ["", f"## {w['effects']}", ""]
        lines += [f"- {s.kind.value}" + (f" ({s.detail})" if s.detail else "")
                  for s in order.side_effects]
    if order.expected_outcome:
        lines += ["", f"## {w['outcome']}", "", order.expected_outcome]
    if order.fallback:
        lines += ["", f"## {w['fallback']}", "", order.fallback]
    return "\n".join(lines)


def render_work_step_review(order: WorkOrder, step: WorkStep) -> str:
    """Plain-text step review (sent / expected / found) for the console and logs."""
    review = step.review or StepReview()
    lines = [
        f"Work Step: {order.agent} {step.id} (rev {order.revision}, round {review.round}, "
        f"tier {order.tier.value})",
        f"Step: {step.title} [{step.status}]",
    ]
    if step.tools:
        lines.append(f"Tools: {', '.join(step.tools)}")
    lines.append(f"Sent: {step.inputs or '-'}")
    lines.append(f"Expected: {step.expected_outcome or '-'}")
    lines.append(f"Found: {step.result or '-'}")
    if step.note:
        lines.append(f"Note: {step.note}")
    if step.calls:
        lines.append("\nCalls:")
        for call in step.calls:
            mark = " [ERROR]" if call.get("is_error") else ""
            lines.append(f"  {call.get('tool', '?')}{mark}({call.get('args', '')})")
            excerpt = str(call.get("result_excerpt") or "")
            if excerpt:
                lines.append(f"      -> {excerpt[:300]}")
    else:
        lines.append("Calls: none recorded")
    return "\n".join(lines)


def unreviewed_steps(order: WorkOrder) -> List[str]:
    """Finished steps that used tools but were never accepted by the human."""
    if not order.step_review:
        return []
    return [
        s.id for s in order.steps
        if s.status == "done" and (s.tools or s.calls)
        and (s.review is None or s.review.status != "accepted")
    ]


def performed_side_effects(order: WorkOrder) -> List[str]:
    """Side effects the run actually had: tools whose nature is one, and were called."""
    from CoScientist.hitl.work_order_risk import TOOL_SIDE_EFFECTS

    kinds: List[str] = []
    for tool in order.tool_calls:
        kind = TOOL_SIDE_EFFECTS.get(tool)
        if kind is not None and kind.value not in kinds:
            kinds.append(kind.value)
    return kinds


def report_warnings(order: WorkOrder) -> List[Dict[str, Any]]:
    """What the human should look at first: where the claim and the record disagree."""
    warnings: List[Dict[str, Any]] = []
    report = order.report
    if report is not None and report.fallback:
        warnings.append({"code": "no_report"})
    open_steps = [s.id for s in order.steps if s.status in ("pending", "in_progress")]
    if open_steps:
        warnings.append({"code": "open_steps", "steps": open_steps})
    if report is not None and not report.fallback:
        if report.done_verdict != "met":
            warnings.append({"code": "done_not_met", "verdict": report.done_verdict})
        unsupported = [f.id for f in report.findings if not f.evidence.strip()]
        if unsupported:
            warnings.append({"code": "findings_without_evidence", "findings": unsupported})
    if order.deviations:
        warnings.append({"code": "deviations", "count": len(order.deviations)})
    unreviewed = unreviewed_steps(order)
    if unreviewed:
        warnings.append({"code": "unreviewed_steps", "steps": unreviewed})
    return warnings


def _artifact_line(artifact: Any, scope: Any = None) -> str:
    """One artifact as a reader can use it: a link when we hold the bytes.

    The rule `_href` and `resolve_ref` already follow — a link that opens
    nothing is worse than a name — so the reference is printed as written
    whenever this session does not hold the file. The agent writes `ref` in its
    own words, and it is as often a workspace path as an artifact id.
    """
    desc = f" — {artifact.description}" if artifact.description else ""
    citation = None
    if scope:
        try:
            from CoScientist.utils.report_links import artifact_citation

            citation = artifact_citation(scope, artifact.ref)
        except Exception:  # noqa: BLE001 — a link is not worth a failed render
            citation = None
    if citation:
        return f"- `{artifact.kind}` [{citation['name']}]({citation['ref']}){desc}"
    return f"- `{artifact.kind}` {artifact.ref}{desc}"


def render_work_report(order: WorkOrder, lang: str = "en",
                       scope: Any = None) -> str:
    """What the agent says it did, against the contract it signed.

    `scope` is the session, and only with it can an artifact reference become a
    link. Without one the output is byte-identical to what it was — which is
    what the console HITL and every existing caller still get.
    """
    w = _words(lang)
    report = order.report or WorkReport()
    lines = [
        f"# {w['report']} — {order.agent}",
        "",
        f"*{w['rev']} {order.revision} · {w['round']} {report.round} · {w['tier']} {order.tier.value}*",
        "",
        f"## {w['goal']}", "", order.goal or "—",
    ]
    if report.summary:
        lines += ["", f"## {w['summary']}", "", report.summary]
    if order.done_criteria:
        lines += ["", f"## {w['done']}", "",
                  f"{order.done_criteria} → **{report.done_verdict}**"
                  + (f"\n\n{report.done_evidence}" if report.done_evidence else "")]
    if order.expected_outcome or report.actual_outcome:
        lines += ["", f"## {w['outcome']}", "",
                  f"| | |", "|---|---|",
                  f"| {w['expected']} | {_cell(order.expected_outcome)} |",
                  f"| {w['actual']} | {_cell(report.actual_outcome)} |"]
    if report.findings:
        lines += ["", f"## {w['findings']}", ""]
        for f in report.findings:
            lines.append(f"- **{f.id}** {f.text} _[{f.confidence}]_")
            if f.evidence:
                lines.append(f"  - {w['evidence']}: {f.evidence}")
    if order.steps:
        lines += ["", f"## {w['steps']}", ""]
        for s in order.steps:
            note = f" — {s.note}" if s.note else ""
            lines.append(f"- {_STEP_MARK.get(s.status, '[ ]')} **{s.id}** {s.title}{note}")
    if report.artifacts:
        lines += ["", f"## {w['artifacts']}", ""]
        for a in report.artifacts:
            lines.append(_artifact_line(a, scope))
    if order.tool_calls:
        lines += ["", f"## {w['calls']}", "",
                  ", ".join(f"`{t}` ×{n}" for t, n in order.tool_calls.items())]
    effects = performed_side_effects(order)
    if effects:
        lines += ["", f"## {w['performed']}", "", ", ".join(effects)]
    if order.amendments:
        lines += ["", f"## {w['amendments']}", "", str(len(order.amendments))]
    if order.deviations:
        lines += ["", f"## {w['blocked']}", "",
                  ", ".join(f"`{d.get('tool', '?')}`" for d in order.deviations)]
    return "\n".join(lines)


def _cell(value) -> str:
    """One table cell: no pipe may survive into it, no newline may break the row."""
    return str(value or "—").replace("|", "\\|").replace("\n", " ")


def diff_work_orders(old: WorkOrder, new: WorkOrder) -> Dict[str, Any]:
    """What an amendment adds or raises — the part the human has to judge."""
    old_steps = {s.id for s in old.steps}
    old_effects = {(s.kind, s.detail) for s in old.side_effects}
    return {
        "added_tools": [t for t in new.planned_tools if t not in old.planned_tools],
        "added_side_effects": [
            s.model_dump(mode="json") for s in new.side_effects
            if (s.kind, s.detail) not in old_effects
        ],
        "added_steps": [s.model_dump(mode="json") for s in new.steps if s.id not in old_steps],
    }
