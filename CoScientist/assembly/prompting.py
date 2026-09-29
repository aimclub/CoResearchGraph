"""PromptContext — everything a prompt template needs to render itself.

Templates receive one :class:`PromptContext` and render the unified sections:

  <<TOOLS>>    the agent's available tools — generated from the ToolDocs of the
               tools that are ACTUALLY attached (incl. HITL tools when attached),
               so prompts can never drift from the wiring
  <<AGENTS>>   bullet list of the agent's enabled subordinates
  <<ROUTING>>  routing guidance ("which subordinate for which job")
  <<HITL>>     usage guidance for the HITL tools (empty when not attached)

plus helpers for conditional sections (``has_tool``, ``is_enabled``,
``sibling_roster`` for the planner).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from CoScientist.assembly.registry import ToolDoc, ToolEntry, render_tool_docs
from CoScientist.assembly.schema import AgentConfig, SystemConfig

TOOLS_GUARD = (
    "IMPORTANT: ONLY call the tools listed above. Never call any other tool name —\n"
    "if a capability isn't in this list, you do not have it."
)

_HITL_SECTION = """\
### Human-in-the-loop

A human supervises this work — treat them as a collaborator, not a rubber stamp.
Two ways to involve them:
- `request_approval(agent_name, message)` — a yes / no question. The human may
  answer plainly OR reply with free-text ("other") — that free-text is an
  instruction, follow it. Returns {approved, feedback}.
- `request_selection(agent_name, message, options, default_option=None)` — offer 2–4 concrete options
  and let the human choose (e.g. among hypotheses, plans, thresholds). The human
  may pick one of the options OR give their own answer in the feedback ("other");
  honor whichever they provide. Returns {selected, approved, feedback}.

Pass your own name as `agent_name`. If a request is denied, don't retry the same
thing — adjust using the feedback."""

# Agents with a Work Order get their plan approved through it instead.
_HITL_WHEN_TO_ASK = """\
**When to ask.** State your intended course of action and get it approved BEFORE
you carry it out — not only for expensive or irreversible steps. Concretely, ask
whenever you have decided:
- which approach, method or tool you will use, where another was available;
- what you will build, change or run next, in enough detail that the human can
  disagree with it ("I will rebuild the dataset from the GOLEM trajectories with
  the corrected fitness, then retrain from scratch — proceed?");
- that something is finished, refuted, or good enough to hand on;
- that you will skip, drop or defer something you were asked to do.

Ask once per decision, not once per tool call: a plan of action is one question,
and the calls that carry it out are not. Do not ask about routine reads, or
about anything a previous answer in this session already settled — a human who
is asked to confirm the obvious stops reading the questions."""

_HITL_WHEN_TO_ASK_WORK_ORDER = """\
**When to ask.** Your plan is approved through the work order below — do not ask
for it separately. Ask only about decisions that come up along the way and the
order did not settle (e.g. which of several found options to take, whether a
result is good enough to hand on). Do not ask about routine reads, or about
anything a previous answer in this session already settled."""


_HITL_RESEARCH_COOP = """\
#### Co-building the research graph with the human
The research graph is built by BOTH the agents and the human. Pause at each
epistemic checkpoint, let the human validate and extend it, then record their
input in the graph:
- When Hypotheses are proposed, ask the human to validate them (keep / drop /
  edit) via `request_selection` or `request_approval`.
- A Hypothesis MUST have acceptance criteria — the metric + threshold that would
  confirm or refute it — BEFORE it is verified. If the human did not provide any,
  you MUST ask for them explicitly ("what result would confirm or refute this
  hypothesis?") and record the answer as its ConfirmationCriteria. Never start
  verification on a hypothesis that has no criteria.
- Likewise invite the human to confirm or adjust the verification methods, the
  question's scope, and the final conclusion.
Find the gaps (e.g. a hypothesis with no criteria) and resolve them by asking the
human — not by inventing the answer. Commit the human's input with
`research_commit`; if your role may not create that node type, state their
decision in your text answer so the orchestrator records it."""

# No curly braces in this text: the instruction goes through ADK's session-state
# injection, which treats a braced identifier as a state key.
_WORK_ORDER_SECTION = """\
### Work Order — declare what you will do before you do it

The human must be able to see your plan and the assumptions behind it BEFORE
you act.

Language: the human reads the work order in Russian. Write every human-readable
value in Russian — `goal`, `done_criteria`, assumption texts, step titles and
expected outcomes, `expected_outcome`, `fallback`, the `reason` of
`update_work_order`, the `note` of `update_work_step`, and the `summary`,
finding texts, evidence descriptions, `done_evidence`, `actual_outcome` and
artifact descriptions of `submit_work_report` — even when the task,
the context or these instructions are in English. Keep as they are: tool names,
step and assumption ids, numbers, units, formulas,
identifiers (database ids, file paths, links), paper titles and author names.

Protocol:
1. Orient yourself first if you need to: reading your own context (research
   context, active tasks, directory listings) is allowed before declaring.
2. Call `declare_work_order` before your first external action (search, download,
   code run, graph write). Provide:
   - `goal`: what this run must achieve in 1-2 sentences.
   - `done_criteria`: concrete condition for completion.
   - `assumptions`: list of strings, one assumption per item.
     Rules for assumptions:
     * ATOMIC: each item is a single, falsifiable constraint (1 checkbox for the human). Never bundle multiple conditions into one item.
     * NON-TRIVIAL: do NOT state obvious facts (e.g. "инструменты работают", "данные существуют", "Python доступен").
     * CONCRETE DOMAIN CONSTRAINTS:
       - Scope & cohort: exact date ranges (e.g. 2020-2025), organisms, clinical phases, target IDs.
       - Data & normalization: specific sources, units of measurement (e.g. nM vs µM), activity cutoffs (e.g. IC50 < 100 nM).
       - Methodology & filters: study designs accepted (RCTs only, peer-reviewed), exclusions (exclude case reports).
       - Volume & sufficiency: e.g. "10 самых цитируемых статей достаточно для первичного обзора".
   - `steps`: ordered steps `[{"title": "...", "tools": [...], "expected_outcome": "..."}]`.
   - `planned_tools`: all tool names you intend to call.
   - `expected_outcome`: concrete results (counts, ranges, metrics).
   - `fallback`: alternative approach if the plan fails.
3. Read the result:
   - `approved` — proceed. Assumptions the human rejected are listed: do not rely
     on them. Operator notes are instructions: follow them.
   - `revise` — declare again, taking the feedback into account.
   - `rejected` — do not act; finish and report why the task was not done.
4. As you work, mark steps with `update_work_step` (in_progress, then done or
   skipped with a short note on what came out).
5. A call outside the approved order (undeclared tool) is BLOCKED. Do not retry
   it: if you really need it, call `update_work_order` with the reason — what you
   learned that the plan did not foresee — and what tools you need added. Otherwise
   continue within the order.
6. BEFORE your final answer call `submit_work_report`. The human checks it
   against your order:
   - `summary`: the result in 2-4 sentences.
   - `findings`: one atomic finding per item, `{"text", "evidence", "confidence",
     "step_id"}`. Evidence is what backs it — links, DOIs, file paths, graph node
     ids, numbers. A finding without evidence is flagged to the human.
   - `done_verdict`: `met` / `partial` / `not_met` against your `done_criteria`,
     with `done_evidence`. Be honest: `partial` with reasons beats a false `met`.
   - `actual_outcome`: the concrete result to set against `expected_outcome`.
   - `artifacts`: only what really exists — files, datasets, graph nodes, links.
   Close your steps with `update_work_step` first: open steps are flagged too.
   Read the result:
   - `accepted` — give your final answer based on the report.
   - `revise` — rework what the feedback (and any disputed findings) points at,
     amend the order if you need more tools, then submit the report again.
   - `rejected` — stop; state plainly that the result was not accepted and why.

Keep the order honest and specific: a human who reads "search the literature"
learns nothing; "искать в PubMed РКИ по X с 2015 года, без описаний клинических
случаев" is something they can correct."""

# Appended to the protocol for agents with `work_order_step_review: true`.
# No curly braces here either (ADK state injection).
_WORK_ORDER_STEP_REVIEW_SECTION = """\
### Step review — the human checks every step

The human confirms each step of your order before you go on: what you sent,
what you expected and what you found. So:
- Every step that calls a tool has `inputs` — exactly what you will send to
  the tool (search queries, substance names, SMILES, route ids, parameter
  values) — and `expected_outcome` — what you expect back, as concretely as
  you can (fields, counts, ranges, units). A step without `inputs` is refused.
- One step, one purpose: keep calls that answer different questions in
  different steps, so each can be judged on its own.
- Before a step's first tool call, mark it `in_progress` with
  `update_work_step`. A tool call made while no step is in progress is BLOCKED.
- When the step is finished, call `update_work_step` with status `done`, a
  short `note` and `result` — what the step actually produced: numbers with
  units, statuses, ids, what is missing. The system shows the human every call
  of the step next to your result, so report what came back, not what you hoped.
- Read the answer:
  - `accepted` — go on to the next step; follow operator notes if any.
  - `revise` — redo this step as the feedback says (it is in progress again),
    then mark it `done` with the new result.
  - `rejected` — the human stopped the order: call no more tools, submit your
    work report with `not_met` and say what was done and what was not."""

_WORK_ORDER_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("websearch",),
     "For searches, the query formulations and the source selection criteria "
     "(recency, study types, venues) are assumptions — list them as atomic items "
     '(e.g. "Только рецензируемые статьи 2020–2025 гг.").'),
    (("papers_search",),
     "For paper downloads, state which papers or how many you will fetch, and why those as assumptions "
     '(e.g. "Скачать полные тексты 5 наиболее релевантных статей").'),
    (("medical",),
     "For clinical evidence, state the population, study designs and date range "
     "you will accept as separate atomic assumptions "
     '(e.g. "Только взрослые пациенты, без педиатрических исследований", '
     '"Только рандомизированные контролируемые исследования (РКИ)").'),
    (("dynamic_tools",),
     "For MCP tools, name the exact tools you will call in `planned_tools` and the steps; "
     "the input values you choose (models, parameters, units, thresholds, number of runs) "
     "are assumptions "
     '(e.g. "Аффинность связывания в ккал/моль").'),
    (("coder", "sandbox"),
     "For data work, the data sources, filters, units and expected volumes are "
     "assumptions "
     '(e.g. "Активность соединений выражена в нМ", '
     '"Основной источник активностей — ChEMBL v33 или новее").'),
    (("research_graph",),
     "If you will write the research graph, name in the steps which nodes you will "
     "create or change."),
]


def register_work_order_hint(tool_keys: tuple[str, ...], hint: str) -> None:
    """Add a Work Order hint shown to agents that have any of ``tool_keys``
    (profiles register the hints for their own tools)."""
    _WORK_ORDER_HINTS.append((tuple(tool_keys), hint))


# The orchestrator alone holds `research_triggers`, so only its copy of the
# protocol may name it. Naming it for every writer told worker agents to call a
# tool they are not given, which is exactly the kind of instruction that makes a
# model invent a call and then apologise for it.
_HITL_RESEARCH_COOP_ORCHESTRATOR = """Use `research_triggers` to find those gaps and resolve them by asking the
human."""


@dataclass
class PromptContext:
    config: AgentConfig
    system: SystemConfig
    # Tool entries actually attached to the agent, in attachment order
    # (includes the synthetic HITL entry when HITL tools are attached).
    tool_entries: List[ToolEntry] = field(default_factory=list)
    hitl_attached: bool = False
    work_order_attached: bool = False

    # ── queries ──────────────────────────────────────────────────────────────
    def has_tool(self, key: str) -> bool:
        return any(e.key == key for e in self.tool_entries)

    def is_enabled(self, agent_name: str) -> bool:
        return (
            agent_name in self.system.agents
            and self.system.agent(agent_name).is_enabled()
        )

    @property
    def subordinates(self) -> List[AgentConfig]:
        return self.system.enabled_subordinates(self.config.name)

    def has_subordinate(self, agent_name: str) -> bool:
        return any(s.name == agent_name for s in self.subordinates)

    def _delegators_beside_me(self) -> List[AgentConfig]:
        """Enabled agents that sit beside me under the same composite parent and
        delegate to subordinates of their own.

        The planner is a CHILD of a composite (it runs before the orchestrator
        rather than under it), so its roster cannot be found through
        ``parents_of``, which only walks subordinates. Found by SHAPE rather
        than by name: the profiles name their orchestrator differently
        (OrchestratorAgent, LiteratureOrchestrator), and a roster that silently
        empties on a rename is worse than no roster at all.
        """
        out = []
        for parent in self.system.agents.values():
            if self.config.name not in parent.children or not parent.is_enabled():
                continue
            for name in parent.children:
                agent = self.system.agent(name)
                if name != self.config.name and agent.subordinates and agent.is_enabled():
                    out.append(agent)
        return out

    def siblings(self) -> List[AgentConfig]:
        """Enabled co-subordinates: my parents' other enabled subordinates.

        For the planner this is exactly the roster the orchestrator can
        delegate to (minus the planner itself) — the agents a plan may assign
        steps to.
        """
        if self.config.name == "PlannerAgent":
            delegators = self._delegators_beside_me()
            if delegators:
                return self.system.enabled_subordinates(delegators[0].name)

        seen, out = set(), []
        for parent in self.system.parents_of(self.config.name):
            if not parent.is_enabled():
                continue
            for sub in self.system.enabled_subordinates(parent.name):
                if sub.name != self.config.name and sub.name not in seen:
                    seen.add(sub.name)
                    out.append(sub)
        return out

    @property
    def docs(self) -> List[ToolDoc]:
        return [d for e in self.tool_entries for d in e.resolved_docs()]

    # ── section renderers ────────────────────────────────────────────────────
    def render_tools(self) -> str:
        """The standard 'available tools' section (header + bullets + guard)."""
        if not self.docs:
            return ""
        return (
            "You have access to the following tools:\n\n"
            f"{render_tool_docs(self.docs)}\n\n"
            f"{TOOLS_GUARD}"
        )

    def render_agents(self) -> str:
        return "\n".join(
            f"* **{a.name}** — {a.description}" for a in self.subordinates
        )

    def render_routing(self) -> str:
        return "\n".join(
            f"    - {a.name} — {a.routing}" for a in self.subordinates if a.routing
        )

    def render_critic_roster(self, agents: Optional[List[AgentConfig]] = None) -> str:
        """Compact `name: description` roster for a critic prompt.

        Defaults to the agent's own subordinates — the orchestrator's critics
        judge which sub-agent it picked. Pass an explicit roster for a critic
        that judges someone else's targets: the plan critic checks assignees,
        which are the planner's SIBLINGS (`ctx.siblings()`), not its own.
        """
        roster = self.subordinates if agents is None else agents
        return "\n".join(f"  - {a.name}: {a.description}" for a in roster)

    def render_hitl(self) -> str:
        if not self.hitl_attached:
            return ""
        section = _HITL_SECTION + "\n\n" + (
            _HITL_WHEN_TO_ASK_WORK_ORDER if self.work_order_attached else _HITL_WHEN_TO_ASK
        )
        # Agents that also write the research graph get the co-building protocol
        # (validate hypotheses with the human; a hypothesis needs acceptance
        # criteria before verification — ask for them if the human didn't give any).
        if self.has_tool("research_graph") or self.has_tool("research_graph_orchestrator"):
            section += "\n\n" + _HITL_RESEARCH_COOP
            if self.has_tool("research_graph_orchestrator"):
                section += "\n" + _HITL_RESEARCH_COOP_ORCHESTRATOR
        if self.work_order_attached:
            section += "\n\n" + _WORK_ORDER_SECTION
            if self.config.work_order_step_review:
                section += "\n\n" + _WORK_ORDER_STEP_REVIEW_SECTION
            hints = [
                hint for keys, hint in _WORK_ORDER_HINTS
                if any(self.has_tool(key) for key in keys)
            ]
            if hints:
                section += "\n\n" + "\n".join(f"- {hint}" for hint in hints)
        return section

    def render_sibling_roster(self) -> str:
        """Planner-style roster of co-subordinates, from their `planning` text."""
        blocks = []
        for a in self.siblings():
            blocks.append(f"- {a.name} – {(a.planning or a.description).strip()}")
        return "\n\n".join(blocks)


__all__ = ["PromptContext", "TOOLS_GUARD"]
