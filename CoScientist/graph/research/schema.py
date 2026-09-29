"""Meta-model schema for the Research Context Graph.

The single source of truth for node types, statuses, status transitions, edge
types (with allowed endpoint type pairs) and per-agent write permissions.
Both the write-validation (store.commit) and the agent prompt sections
(templates.render_research_protocol) render from these tables, so what an agent
is told it may write and what the graph actually accepts can never drift.

Canonical identifiers are English; the spec's Russian terms are accepted on
input via RU_ALIASES and normalized before validation (the EN↔RU mapping is
documented in docs/research_graph.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

# ── Node types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NodeTypeSpec:
    name: str
    prefix: str                      # readable id prefix: H1, E3, VM2, …
    layer: int                       # meta-model layer (1,2,3,4,6)
    statuses: Tuple[str, ...]        # first entry = default initial status
    creatable: Tuple[str, ...]       # statuses allowed at creation time
    attr_docs: Dict[str, str] = field(default_factory=dict)
    subtypes: Tuple[str, ...] = ()   # allowed attrs.subtype values
    subtype_required: bool = False
    #: subtype -> attributes that subtype cannot be created without. Declared
    #: here so `permitted_summary` can tell an agent what it must supply: a
    #: requirement the validator enforces and the prompt never mentions is a
    #: refusal the agent cannot act on.
    required_attrs_by_subtype: Dict[str, Tuple[str, ...]] = field(default_factory=dict)


NODE_TYPES: Dict[str, NodeTypeSpec] = {s.name: s for s in [
    # ── Layer 1 — epistemic objects ──────────────────────────────────────────
    NodeTypeSpec(
        "ResearchQuestion", "Q", 1,
        statuses=("open", "decomposed", "closed"), creatable=("open",),
        attr_docs={
            "formulation": "the question itself",
            "domain": "subject area",
            "specificity": "how narrow/precise the question is",
            "gap": "the knowledge gap it addresses",
            "decomposition": "sub-questions it breaks into",
            "target_setting": "целевая постановка (what kind of answer is sought)",
            "research_form": "fundamental / exploratory / applied",
            "trl": "УГТ — technology readiness level",
            "completion_criteria": "when the research is done: exhaustive / "
                                   "pragmatic / resource / economic",
            "ai_application_model": "autonomy level: lab-assistant / assistant / "
                                    "copilot / architect",
        },
    ),
    NodeTypeSpec(
        "Hypothesis", "H", 1,
        # `inconclusive` is the verdict a validator reaches when the branch was
        # tested and the evidence did not settle it. Without it that outcome was
        # written as `postponed`, which means "in the backlog, not attempted" —
        # so a hypothesis that had been worked on for an hour was recorded as
        # never having been tried, and the graph reported an idle study.
        statuses=("formulated", "under_verification", "confirmed", "refuted",
                  "inconclusive", "postponed"),
        creatable=("formulated", "postponed"),
        attr_docs={
            "formulation": "the testable statement",
            "rationale": "why it is plausible",
            "priority": "verification priority (e.g. 1..5 or high/medium/low)",
            "inconclusive_reason": "what the evidence failed to settle — required "
                                   "when a verdict of inconclusive is written",
            "not_tested_reason": "why this hypothesis was left untested — set it "
                                 "instead of verifying, when a verdict already "
                                 "obtained makes the test unnecessary",
        },
    ),
    NodeTypeSpec(
        "Evidence", "E", 1,
        statuses=("obtained", "validated", "rejected"), creatable=("obtained",),
        attr_docs={
            "subtype": "REQUIRED — literature / experimental / computational / expert / meta",
            "content": "the observation or fact",
            "measured_on": "REQUIRED for computational/experimental evidence — "
                           "WHAT was actually measured, named exactly: the repo "
                           "and commit, the dataset, the deployed service. Say so "
                           "here if it is a stand-in for what the hypothesis names "
                           "(e.g. 'local reimplementation; upstream repo 404')",
            "reliability": "weight / reliability estimate",
            "result_kind": "reader-facing classification such as a partial "
                           "result or execution error; not a hypothesis verdict",
            "failure_reason": "why an attempted execution failed; execution "
                              "errors must not be linked as scientific support/refutation",
            "source_ref": "paper DOI, PMC id, dataset, run id, … — one per "
                          "source, separated by ';' when there are several",
            "doi": "set by the graph when a cited paper is matched to a stored file",
            "pmcid": "same, for a PubMed Central id",
        },
        subtypes=("literature", "experimental", "computational", "expert", "meta"),
        subtype_required=True,
        required_attrs_by_subtype={"computational": ("measured_on",),
                                   "experimental": ("measured_on",)},
    ),
    NodeTypeSpec(
        "Conclusion", "CL", 1,
        statuses=("draft", "approved"), creatable=("draft",),
        attr_docs={
            # The conclusion is what a reader takes AWAY from the study, and
            # what the next study starts from — so it carries the chain, not
            # just the answer. One paragraph of prose was all it used to hold,
            # which meant the reader had to re-walk the graph to learn how the
            # answer had been reached, and a follow-up study had nothing to
            # begin with.
            "synthesis": "THE ANSWER, in one or two sentences — this is the "
                         "card's headline, so it must stand alone",
            "how_established": "the chain that produced it, by stage: what the "
                               "reading established, what was run, with which "
                               "instrument, and the numbers it returned",
            "against_criteria": "each ConfirmationCriteria by id, the value "
                                "measured against it, and whether it was met",
            "validity_bounds": "limits of validity: the population, the "
                               "conditions, the model, what it does NOT cover",
            "open_questions": "what the next study should do first — the "
                              "measurement that was missing, the bar that was "
                              "not reached, the branch nobody tested",
            "new_question": "optional follow-up question text",
        },
    ),
    # ── Derived, never written ───────────────────────────────────────────────
    # Two cards the reader needs that no agent authors: the framing the study
    # started from, and the story its verdicts add up to. Both are projected by
    # store.to_view from nodes that already exist (the context star, and the
    # hypothesis/conclusion chain), so materializing them would create a second
    # source of truth that goes stale on the next Constraint. They are declared
    # here anyway, so the type name, the id prefix and the display words live in
    # the one table everything else reads — and `creatable=()` plus their
    # absence from every AgentPerm.create makes validate_node_draft refuse any
    # agent that tries to write one.
    NodeTypeSpec(
        "Framing", "F", 1,
        statuses=("derived",), creatable=(),
        attr_docs={"_": "DERIVED — projected from the context star "
                        "(Constraint/Resource/EmpiricalBase/ConfirmationCriteria/"
                        "CostModel) by store.to_view; no agent may create it"},
    ),
    NodeTypeSpec(
        "Outcome", "OC", 1,
        statuses=("derived",), creatable=(),
        attr_docs={"_": "DERIVED — the chain of hypothesis verdicts, assembled "
                        "by store.to_view; no agent may create it"},
    ),
    # ── Layer 2 — methodological frame ───────────────────────────────────────
    # The PLAN, as opposed to the record. A plan step is an intention: what is
    # to be done, in what order, by whom. A VerificationMethod is the answer to
    # a different question — by WHAT MEANS was this established, and against
    # which bar — and it can only exist once there is a claim to test.
    #
    # They were the same node until now, and that is the single biggest thing
    # wrong with the graph the operator reads: the mirror wrote the planner's
    # task list as `VerificationMethod`, so "Метод проверки 2" hung off the
    # research QUESTION, carried no instrument, and could not be told apart
    # from a method an agent had actually designed. A step and a method are
    # linked by `realises`, and a reader can now see both the intention and
    # what was made of it.
    NodeTypeSpec(
        "PlanStep", "PS", 2,
        # The task tracker's own vocabulary, lowercased. A step the mirror first
        # sees already finished is created finished, so every status is creatable.
        statuses=("todo", "in_progress", "done", "blocked"),
        creatable=("todo", "in_progress", "done", "blocked"),
        attr_docs={
            "title": "the step as the plan words it — the card's headline",
            "description": "what the step asks for",
            "plan_task_id": "the id the plan gave it (TASK-n)",
            "assignee": "the agent the plan assigned it to",
            "notes": "anything the plan attached to the step",
        },
    ),
    # One task of the experiment module's own plan: the detailed grain under a
    # general step. Prefix "XT" rather than "ET" — the E space is already
    # crowded (E, EB, EJ, EM) and "ET3" reads as an Evidence variant to a
    # person even though the code is unambiguous.
    NodeTypeSpec(
        "ExperimentTask", "XT", 2,
        # The runtime's own vocabulary, collapsed to what a reader needs:
        # pending/ready → planned, running/retry_pending/fallback_pending →
        # running, done/done_with_warnings → done. All creatable, for
        # PlanStep's reason: a re-publish after a replan may first meet a task
        # that has already finished.
        statuses=("planned", "running", "done", "failed", "skipped"),
        creatable=("planned", "running", "done", "failed", "skipped"),
        attr_docs={
            "title": "the task as the plan names it — the card's headline",
            "description": "what the task asks for",
            "rationale": "why the plan included it",
            "experiment_task_id": "the id the experiment plan gave it (EXP-n)",
            "plan_task_id": "the OUTER plan's step it elaborates (TASK-n)",
            "plan_id": "which experiment plan, and which revision of it",
            "plan_revision": "the revision number the human approved",
            "experiment_run_id": "the run the plan belongs to (EXRUN-…)",
            "route": "how it will be executed — react_tools / fedot_mas / "
                     "coder / alembic_build / research / medical",
            "question": "the experimental question this task answers",
            "hypothesis_refs": "the claims it tests, as the plan names them "
                               "(the graph link itself is on the method)",
            "operation_ref": "the framing operation it covers (OP-n)",
            "dataset": "the data it runs on, and where that lives",
            "baselines": "what the result is compared against",
            "metrics": "what is measured, and which direction is better",
            "success_criteria": "the bar, as the plan set it: metric, "
                                "comparison and target",
            "expected_artifacts": "what it must produce to count as done",
            "tools": "the MCP servers and tools it will call, 'server:tool'",
            "input_data": "what it consumes, including which earlier task "
                          "produced it",
            "depends_on": "the tasks that must finish first",
            "cost": "the plan's own estimate of how long it takes",
            "limitations": "what the plan already knows is wrong with it",
            "optional": "written only when the plan marked it optional",
            "failure_reason": "why it failed — required when you move it to "
                              "failed, or the card says a thing went wrong "
                              "and not what",
        },
    ),
    NodeTypeSpec(
        "VerificationMethod", "VM", 2,
        # A method is not work, and it cannot be "запланирован" or "выполнен":
        # those are states of a TASK. A method is the MEANS — the list of
        # instruments a claim is to be settled by — so the only thing that can
        # be said about it is whether it was offered, whether the study
        # actually leaned on it, and whether it was left aside. `not_used`
        # covers both "nobody ran it" and "it was run and settled nothing":
        # from the reader's side those are the same fact about the method, and
        # `failure_reason` says which of the two it was.
        statuses=("proposed", "used", "not_used"), creatable=("proposed",),
        attr_docs={
            "method_type": "computational / laboratory / analytical / statistical "
                           "/ expert / literature_review",
            "description": "WHAT this method is, in one line — it is the card's "
                           "headline, so two methods that differ must differ here",
            # The point of the card. A method that names no instrument is a
            # restated task: it says a thing will be checked and never says by
            # what, which is precisely what a reader comes to a method for.
            "instruments": "REQUIRED — WHAT IT IS RUN WITH, named exactly and "
                           "listed one per entry, separated by ';': MCP tools "
                           "as 'server:tool', agents by their name, libraries "
                           "and services by theirs. Not a description — a list",
            "procedure": "HOW it is run: the concrete steps, tools and settings",
            "inputs": "what it needs",
            "outputs": "what it yields",
            "cost": "estimated cost",
            "limitations": "known weaknesses",
            # Written by the plan mirror (agents/callbacks/tool_callbacks.py), not
            # by a model, and undeclared until now — which is why a reader could
            # not tell two methods mirrored from two plan steps apart.
            "plan_task_id": "the plan step this method mirrors (TASK-n)",
            "assignee": "the agent the plan assigned the step to",
            # A method set aside with no reason is indistinguishable from one
            # nobody got to, which is exactly how a run reads when it is over.
            "failure_reason": "WHY it settled nothing — the error, the missing "
                              "input, the limit hit, or plainly that the study "
                              "never reached it. Required when you move it to "
                              "`not_used`",
        },
    ),
    NodeTypeSpec(
        "ConfirmationCriteria", "CC", 2,
        statuses=("not_met", "met"), creatable=("not_met",),
        attr_docs={
            "threshold": "quantitative/qualitative bar — FROZEN once evidence "
                         "is aimed at the hypothesis; to revise the standard, "
                         "write a new criterion saying what it replaces",
            "confirmations_needed": "number of independent confirmations",
            "reproducibility": "reproducibility requirement",
        },
    ),
    # ── Layer 3 — feasibility context ────────────────────────────────────────
    NodeTypeSpec(
        "Tool", "T", 3,
        statuses=("available", "needs_adaptation", "being_created", "creation_failed"),
        creatable=("available", "needs_adaptation", "being_created"),
        attr_docs={
            "name": "tool/service name",
            "tool_type": "computational / laboratory / analytical / informational",
            "requirements": "what it needs to run",
            # Without this a Tool is only a name. A worker told to "use GOLEM"
            # and given no path has to guess where it is, and what it does
            # instead is write its own — which is how a run silently stops using
            # the library it was supposed to build on.
            "location": "WHERE it is: repo URL, local path, MCP server or API "
                        "endpoint — required for anything the coder must read or run",
            "failure_reason": "WHY building it failed. Required when you move "
                              "it to `creation_failed`",
        },
    ),
    NodeTypeSpec(
        "Resource", "R", 3,
        statuses=("available", "exhausted"), creatable=("available",),
        attr_docs={
            "resource_type": "GPU-hours / tokens / reagents / time / expert-hours",
            "remaining": "numeric remainder",
            "limit": "numeric budget",
        },
    ),
    NodeTypeSpec(
        "EmpiricalBase", "EB", 3,
        statuses=("created",), creatable=("created",),
        attr_docs={
            "base_type": "dataset / corpus / knowledge_base",
            "volume": "size",
            "source_ref": "where it lives",
        },
    ),
    NodeTypeSpec(
        "Constraint", "C", 3,
        statuses=("active",), creatable=("active",),
        attr_docs={
            "subtype": "REQUIRED — profile / methodological_norms / theoretical_framework / "
                       "domain_standards / ethics / expert_knowledge / roles",
            "content": "the constraint itself",
        },
        subtypes=("profile", "methodological_norms", "theoretical_framework",
                  "domain_standards", "ethics", "expert_knowledge", "roles"),
        subtype_required=True,
    ),
    # ── Layer 4 — artifacts ──────────────────────────────────────────────────
    NodeTypeSpec("CodeArtifact", "CA", 4, statuses=("created",), creatable=("created",),
                 attr_docs={"path": "repo/path", "version": "version", "description": "what it is"}),
    NodeTypeSpec("GeneratedData", "GD", 4, statuses=("created",), creatable=("created",),
                 attr_docs={"path": "where the data lives", "schema": "columns/format", "volume": "size"}),
    NodeTypeSpec("Report", "RP", 4, statuses=("created",), creatable=("created",),
                 attr_docs={"content": "text or link"}),
    NodeTypeSpec("Publication", "PB", 4, statuses=("created",), creatable=("created",),
                 attr_docs={"content": "text or link"}),
    NodeTypeSpec("Spec", "SP", 4, statuses=("created",), creatable=("created",),
                 attr_docs={"content": "ТЗ text or link"}),
    NodeTypeSpec("EfficiencyJustification", "EJ", 4, statuses=("created",), creatable=("created",),
                 attr_docs={"content": "metrics, comparison, recommendations"}),
    # ── Layer 6 — economics ──────────────────────────────────────────────────
    NodeTypeSpec("CostModel", "CM", 6, statuses=("created",), creatable=("created",),
                 attr_docs={"rule": "cost formula/rule for a step"}),
    NodeTypeSpec("EfficiencyMetric", "EM", 6, statuses=("created",), creatable=("created",),
                 attr_docs={"time": "speed-up", "cost": "savings", "quality": "reproducibility/accuracy"}),
]}
# Layer 5 (navigation logic: transition triggers, completion criteria, AI
# application model) is deliberately NOT materialized as node types — per the
# spec it lives in orchestrator logic (queries.py) and in status_history logs.


# ── Status transitions ───────────────────────────────────────────────────────
# (from, to) pairs allowed per type; a status absent as `from` is terminal.
STATUS_TRANSITIONS: Dict[str, FrozenSet[Tuple[str, str]]] = {
    "ResearchQuestion": frozenset({("open", "decomposed"), ("open", "closed"),
                                   ("decomposed", "closed")}),
    "Hypothesis": frozenset({("formulated", "under_verification"),
                             ("formulated", "postponed"),
                             ("under_verification", "confirmed"),
                             ("under_verification", "refuted"),
                             ("under_verification", "inconclusive"),
                             ("under_verification", "postponed"),
                             ("inconclusive", "under_verification"),
                             ("postponed", "formulated")}),
    "Evidence": frozenset({("obtained", "validated"), ("obtained", "rejected")}),
    "Conclusion": frozenset({("draft", "approved")}),
    # Offered → leaned on, or left aside. `not_used → used` is the retry: a
    # method the study first gave up on can still turn out to be the one that
    # settles the claim. There is no way back out of `used`, because a method
    # that produced evidence stays a method the study was built on.
    "VerificationMethod": frozenset({("proposed", "used"),
                                     ("proposed", "not_used"),
                                     ("not_used", "used")}),
    "ConfirmationCriteria": frozenset({("not_met", "met"), ("met", "not_met")}),
    # The tracker's own moves. A finished step can be reopened, because a
    # re-plan may put a step back in play, and a blocked one can be released.
    "PlanStep": frozenset({("todo", "in_progress"), ("todo", "done"),
                           ("todo", "blocked"), ("in_progress", "done"),
                           ("in_progress", "blocked"), ("in_progress", "todo"),
                           ("blocked", "in_progress"), ("blocked", "todo"),
                           ("done", "in_progress")}),
    # `failed → planned` is not cosmetic: retry_task and fallback_task put the
    # runtime status back to ready, and a card stuck on "failed" would then
    # contradict a task that is running again.
    "ExperimentTask": frozenset({("planned", "running"), ("planned", "skipped"),
                                 ("planned", "failed"), ("planned", "done"),
                                 ("running", "done"), ("running", "failed"),
                                 ("running", "skipped"), ("failed", "planned"),
                                 ("failed", "skipped"), ("done", "running")}),
    "Tool": frozenset({("needs_adaptation", "available"),
                       ("needs_adaptation", "being_created"),
                       ("being_created", "available"),
                       ("being_created", "creation_failed"),
                       ("creation_failed", "being_created")}),
    "Resource": frozenset({("available", "exhausted"), ("exhausted", "available")}),
}


# ── Edge types ───────────────────────────────────────────────────────────────
_ARTIFACT_TYPES = ("CodeArtifact", "GeneratedData", "Report", "Publication",
                   "Spec", "EfficiencyJustification")

EDGE_TYPES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "motivates": (("ResearchQuestion", "Hypothesis"),),
    # A plan arrives before the hypotheses do: the deterministic plan mirror
    # writes one method per registered task, and at that moment there may be
    # nothing to hang it on but the question itself. A method floating with no
    # parent reads as a bug, so the question may be what a method tests.
    "tested_by": (("Hypothesis", "VerificationMethod"),
                  ("ResearchQuestion", "VerificationMethod")),
    "requires": (("Hypothesis", "Tool"),),
    "uses": (("VerificationMethod", "Tool"),),
    "consumes": (("VerificationMethod", "Resource"),),
    "produces": (("VerificationMethod", "Evidence"),
                 ("Conclusion", "ResearchQuestion")),
    "supports": (("Evidence", "Hypothesis"),),
    "refutes": (("Evidence", "Hypothesis"),),
    "refines": (("Evidence", "Hypothesis"),),
    # How a study actually moves: a hypothesis is judged, and a modified one
    # takes its place. Without this edge the iterations sit side by side under
    # the question and the reader cannot tell a second attempt from a second
    # branch. `from` is the hypothesis that was judged, `to` the one that
    # replaced it — the arrow points the way the research went. attrs carry
    # {"verdict": confirmed|refuted|inconclusive, "reason": what changed}.
    "supersedes": (("Hypothesis", "Hypothesis"),),
    # Execution dependency, deliberately separate from provenance-oriented
    # supersedes.  The successor stays dormant until the predecessor reaches
    # attrs.required_status (v1 supports only refuted).
    "conditional_successor": (("Hypothesis", "Hypothesis"),),
    "based_on": (("Conclusion", "Evidence"),),
    "determines_sufficiency": (("ConfirmationCriteria", "Conclusion"),),
    # Not in the spec's edge table, but the docx says criteria are "formulated
    # for a hypothesis" and the closable-path trigger needs the linkage.
    "formulated_for": (("ConfirmationCriteria", "Hypothesis"),),
    # What a plan step turned into. The arrow runs from the RECORD to the
    # INTENTION — "this method realises that step" — so a reader following the
    # research forward never walks into the plan by accident, and a step with
    # nothing pointing at it is visibly unrealised. A step can be realised by
    # more than the method: "formulate a testable hypothesis" is realised by the
    # hypothesis itself, and "write the report" by the conclusion.
    # A finer intention under a coarser one. Deliberately NOT `realises`:
    # that one means record → intention ("this work carried out that"), and an
    # ExperimentTask is not a record. Reusing it would rebuild the very
    # conflation the PlanStep type exists to undo.
    "elaborates": (("ExperimentTask", "PlanStep"),),
    "realises": (("VerificationMethod", "PlanStep"),
                 ("Hypothesis", "PlanStep"),
                 ("Evidence", "PlanStep"),
                 ("Conclusion", "PlanStep"),
                 # …and the record may attach to the detailed task instead of
                 # the general step, which is where it actually came from.
                 ("VerificationMethod", "ExperimentTask"),
                 ("Evidence", "ExperimentTask")),
    "regulates": (("Constraint", "VerificationMethod"),
                  ("Constraint", "ConfirmationCriteria")),
    "constrains": (("Constraint", "Hypothesis"),
                   ("Constraint", "VerificationMethod")),
    # A study that reached no Conclusion still produced a write-up, and it has
    # to hang off something — otherwise the one node a reader most wants is the
    # one node with no edge to find it by.
    # A Spec written BEFORE the study — the техническое задание the framing
    # stage issues — has no conclusion and no evidence to derive from, and the
    # question is the only thing it is about.
    "derived_from": tuple((a, t) for a in _ARTIFACT_TYPES
                          for t in ("Conclusion", "Evidence"))
    + (("Report", "ResearchQuestion"), ("Spec", "ResearchQuestion")),
    "contextualizes": (("Constraint", "ResearchQuestion"),),
    "defines_scope": (("ResearchQuestion", "EmpiricalBase"),),
    "relates_to": (("Evidence", "ResearchQuestion"), ("Evidence", "Hypothesis")),
    # Small addition so Layer-6 nodes are attachable to what they measure.
    "applies_to": tuple((m, t) for m in ("CostModel", "EfficiencyMetric")
                        for t in ("ResearchQuestion", "Hypothesis",
                                  "VerificationMethod", "Conclusion")),
}


# ── Russian aliases (spec vocabulary accepted on input) ──────────────────────
RU_ALIASES: Dict[str, str] = {
    # node types
    "исследовательский_вопрос": "ResearchQuestion", "вопрос": "ResearchQuestion",
    "гипотеза": "Hypothesis",
    "свидетельство": "Evidence",
    "заключение": "Conclusion",
    "метод_проверки": "VerificationMethod", "методы_проверки": "VerificationMethod",
    "задача_эксперимента": "ExperimentTask", "шаг_эксперимента": "ExperimentTask",
    "условия_подтверждения": "ConfirmationCriteria", "условие_подтверждения": "ConfirmationCriteria",
    "инструмент": "Tool",
    "ресурс": "Resource",
    "эмпирическая_база": "EmpiricalBase",
    "ограничение": "Constraint",
    "код": "CodeArtifact",
    "порождённые_данные": "GeneratedData", "порожденные_данные": "GeneratedData",
    "отчёт": "Report", "отчет": "Report",
    "публикация": "Publication",
    "тз": "Spec",
    "обоснование_эффективности": "EfficiencyJustification",
    "модель_стоимости": "CostModel",
    "метрика_эффективности": "EfficiencyMetric",
    # edge types
    "мотивирует": "motivates",
    "проверяется_через": "tested_by",
    "требует": "requires",
    "использует": "uses",
    "потребляет": "consumes",
    "порождает": "produces",
    "поддерживает": "supports",
    "опровергает": "refutes",
    "уточняет": "refines",
    "условный_преемник": "conditional_successor",
    "основано_на": "based_on",
    "определяет_достаточность": "determines_sufficiency",
    "сформулировано_для": "formulated_for", "формулируется_для": "formulated_for",
    "регулирует": "regulates",
    "ограничивает": "constrains",
    "формируется_из": "derived_from",
    "контекстуализирует": "contextualizes",
    "определяет_область": "defines_scope",
    "относится_к": "relates_to",
    "применяется_к": "applies_to",
    # statuses
    "открыт": "open", "декомпозирован": "decomposed", "закрыт": "closed",
    "сформулирована": "formulated", "на_проверке": "under_verification",
    "подтверждена": "confirmed", "опровергнута": "refuted", "отложена": "postponed",
    "получено": "obtained", "валидировано": "validated", "отвергнуто": "rejected",
    "черновик": "draft", "утверждено": "approved",
    "запланирован": "planned", "выполняется": "running",
    "выполнен": "done", "провален": "failed",
    # A method's own words. Kept apart from the task vocabulary above on
    # purpose: «выполнен» must keep meaning a task that ran, so a model that
    # writes it about a method is refused rather than quietly understood.
    "предложен": "proposed", "использован": "used",
    "не_использован": "not_used", "неиспользован": "not_used",
    "не_выполнены": "not_met", "выполнены": "met",
    "доступен": "available", "нужна_адаптация": "needs_adaptation",
    "создаётся": "being_created", "создается": "being_created",
    "не_удалось_создать": "creation_failed",
    "исчерпан": "exhausted",
    "активен": "active",
    "создан": "created",
    # subtypes
    "литературное": "literature", "экспериментальное": "experimental",
    "вычислительное": "computational", "экспертное": "expert", "мета": "meta",
    "профиль_исследования": "profile", "профиль": "profile",
    "методологические_нормы": "methodological_norms",
    "теоретические_рамки": "theoretical_framework",
    "доменные_стандарты": "domain_standards",
    "этика": "ethics", "этика/регуляторика": "ethics", "регуляторика": "ethics",
    "экспертное_знание": "expert_knowledge",
    "роли": "roles",
}

# Case-insensitive node-type lookup ("hypothesis" → "Hypothesis").
_TYPE_BY_LOWER = {name.lower(): name for name in NODE_TYPES}


def _norm_token(value: str) -> str:
    return str(value or "").strip().replace(" ", "_").replace("-", "_").lower()


def normalize_node_type(value: str) -> str:
    """Canonical node type name, or the cleaned input if unknown."""
    tok = _norm_token(value)
    return RU_ALIASES.get(tok) or _TYPE_BY_LOWER.get(tok) or str(value or "").strip()


def normalize_token(value: str) -> str:
    """Canonical edge type / status / subtype, or the cleaned input if unknown."""
    tok = _norm_token(value)
    return RU_ALIASES.get(tok, tok)


# ── Per-agent write permissions ──────────────────────────────────────────────

@dataclass(frozen=True)
class AgentPerm:
    """What one agent may write. Edges and transitions are held at full
    granularity — (edge, from_type, to_type) and (type, from, to) triples — so
    e.g. the orchestrator may create produces(Conclusion→ResearchQuestion) but
    not produces(VerificationMethod→Evidence)."""
    create: FrozenSet[str]
    update_attrs: FrozenSet[str]
    transitions: FrozenSet[Tuple[str, str, str]]
    edges: FrozenSet[Tuple[str, str, str]]
    #: Single (type, attribute) pairs an agent may write on a node it may not
    #: otherwise touch. The protocol asks for reasons — why a branch was left
    #: untested, what the evidence failed to settle — from the role that knows
    #: them, and that role is rarely the owner of the node. Without this the
    #: instruction is one no agent can carry out: the triggers told the
    #: orchestrator to commit `attrs.not_tested_reason` and the store refused
    #: it, so the study stayed open and the agent learned to shrug.
    update_fields: FrozenSet[Tuple[str, str]] = frozenset()


def _edges(*specs) -> FrozenSet[Tuple[str, str, str]]:
    """Expand edge specs: a bare edge-type string means all its allowed pairs;
    a (edge_type, from_type, to_type) triple restricts to that pair."""
    out = set()
    for spec in specs:
        if isinstance(spec, str):
            out.update((spec, f, t) for f, t in EDGE_TYPES[spec])
        else:
            edge, f, t = spec
            if (f, t) not in EDGE_TYPES[edge]:
                raise ValueError(f"unknown edge pair {spec}")
            out.add(spec)
    return frozenset(out)


def _transitions(*specs) -> FrozenSet[Tuple[str, str, str]]:
    """Expand transition specs: a bare type name means all its transitions;
    a (type, from, to) triple restricts to that transition."""
    out = set()
    for spec in specs:
        if isinstance(spec, str):
            out.update((spec, f, t) for f, t in STATUS_TRANSITIONS.get(spec, ()))
        else:
            typ, f, t = spec
            if (f, t) not in STATUS_TRANSITIONS.get(typ, ()):
                raise ValueError(f"unknown transition {spec}")
            out.add(spec)
    return frozenset(out)


# Types that research_init seeds as the "context star". These may be created by
# the init path REGARDLESS of the caller's general create-set (privileged, trusted
# seeding — the agent is not choosing types freely, the tool constructs them), so
# they are absent from the orchestrator's mid-run create-set above.
INIT_SEED_TYPES = frozenset({"ResearchQuestion", "Tool", "Resource",
                             "EmpiricalBase", "Constraint",
                             "ConfirmationCriteria", "CostModel"})

# Attributes the graph writes ABOUT a node, never accepts as a claim IN it.
#
# `research_commit` puts no whitelist on attribute names, and a worker sees a
# node's attrs in its context slice — so an agent can, and in time will, send
# one of these back. The attrs merge is a shallow `{**stored, **incoming}`
# (`_commit_locked`), which replaces a key wholesale: one partial write and an
# append-only participation record is down to whatever that agent happened to
# say. The key is dropped rather than the commit refused, because the commit is
# worth more than the key, and the caller is warned.
#
# `_provenance` is deliberately NOT here: `agent_tools._enrich_evidence` writes
# it through the ordinary create path before the commit, so reserving it would
# silently throw away every piece of evidence's provenance.
RESERVED_ATTRS = frozenset({
    "contributors", "contributors_more",
    "report_artifact_id", "report_stamp", "report_lang",
})


# Spec §2 roles mapped onto the agents that actually exist in system.yaml:
# init-agent + validator/critic duties → OrchestratorAgent; hypothesis
# generation (incl. methods + criteria, docx Module 2) → HypothesesAgent;
# researcher → ResearchAgent (+ MedicalAgent for the clinical domain);
# coder → CoderAgent / DatasetCollectorAgent / ExperimentAgent; the human
# acts through HITL approvals ("human" pseudo-agent, reserved for that bridge).
AGENT_PERMISSIONS: Dict[str, AgentPerm] = {
    "OrchestratorAgent": AgentPerm(
        # Coordinator + framing + SCHEDULING + approval — NOT the judge. It seeds
        # the context star only via research_init (privileged); mid-run it creates
        # its own outputs (artifacts, economics) and spawned sub-questions, starts/
        # postpones verification, approves conclusions, and wires constraints. The
        # VERDICT (under_verification→confirmed/refuted) and the Conclusion belong
        # to the ValidatorAgent, so they are absent here.
        create=frozenset({"ResearchQuestion", "Evidence", "Report", "Publication", "Spec",
                          "EfficiencyJustification", "CostModel", "EfficiencyMetric"}),
        update_attrs=frozenset({"Resource", "ResearchQuestion", "EmpiricalBase", "Tool"}),
        transitions=_transitions(
            "ResearchQuestion", "Resource",
            ("Tool", "needs_adaptation", "available"),
            ("Tool", "needs_adaptation", "being_created"),
            ("Tool", "being_created", "available"),
            ("Conclusion", "draft", "approved"),               # approval
            ("Hypothesis", "formulated", "under_verification"),  # start verification
            ("Hypothesis", "formulated", "postponed"),
            ("Hypothesis", "postponed", "formulated"),           # scheduling only
            # `inconclusive` is in the lifecycle but nothing held the way OUT of
            # it, so a branch the judge could not settle was parked there for
            # good. The background validator now WRITES that verdict whenever a
            # confirmation is refused, which makes a dead end that used to be
            # nearly unreachable ordinary. Reopening one is scheduling, same as
            # reviving a postponed branch: new evidence arrived, put it back
            # under verification and let the judge look again.
            ("Hypothesis", "inconclusive", "under_verification")),
        edges=_edges("contextualizes", "defines_scope", "derived_from", "applies_to",
                     "motivates", "regulates", "constrains",
                     "relates_to", "supports", "refutes", "refines", "supersedes",
                     ("produces", "Conclusion", "ResearchQuestion")),
        # It decides what gets tested, so it is the one that can say why a
        # branch was not. Only that: the formulation and the verdict stay with
        # the agents that own them.
        update_fields=frozenset({("Hypothesis", "not_tested_reason")}),
    ),
    # Spec Module 4 — the judge. Given ONE hypothesis's evidence slice it weighs
    # the evidence against the criteria, sets the verdict, and writes the
    # Conclusion. Kept separate from the orchestrator so the judgment runs in a
    # small focused context instead of bloating the coordinator.
    "ValidatorAgent": AgentPerm(
        create=frozenset({"Conclusion"}),
        update_attrs=frozenset(),
        transitions=_transitions(
            ("Hypothesis", "under_verification", "confirmed"),
            ("Hypothesis", "under_verification", "refuted"),
            # A branch that was tested and did not settle is inconclusive, not
            # postponed: writing it as postponed says the work never happened.
            ("Hypothesis", "under_verification", "inconclusive"),
            "ConfirmationCriteria", "Evidence"),
        # supports/refutes/refines: the validator assigns the POLARITY of evidence
        # that reached the hypothesis only as relates_to (focus auto-link).
        edges=_edges("based_on", "determines_sufficiency",
                     "supports", "refutes", "refines"),
        # It writes the inconclusive verdict, so it writes what the evidence
        # failed to settle — the schema asks for that reason and nobody could
        # supply it.
        update_fields=frozenset({("Hypothesis", "inconclusive_reason")}),
    ),
    "HypothesesAgent": AgentPerm(
        # May also declare the Tools its methods need (as needs_adaptation — a
        # NEED, not a confirmed capability), so its requires/uses edges resolve;
        # the orchestrator/coder later flips them to available/being_created. A
        # needs_adaptation tool keeps the hypothesis correctly BLOCKED until then.
        create=frozenset({"Hypothesis", "VerificationMethod", "ConfirmationCriteria",
                          "Tool"}),
        update_attrs=frozenset(),
        transitions=_transitions(("Hypothesis", "formulated", "postponed")),
        # It writes the modified hypothesis, so it is the one that can say which
        # hypothesis that modification replaces.
        edges=_edges("motivates", "tested_by", "requires", "formulated_for",
                     "uses", "consumes", "supersedes", "conditional_successor"),
    ),
    "ResearchAgent": AgentPerm(
        # The same hole the coder and the experimenter had, on the literature
        # side. It gathered the sources and wrote the Evidence, but could
        # neither open the method that gathered them nor say that the method
        # produced them: `produces` was not in its edges and VerificationMethod
        # was not in its create. So a literature finding could only hang off the
        # question by `relates_to` — and with no hypotheses yet, that was its
        # ONLY legal attachment — while the "collect the literature" method the
        # plan mirror had written stayed `planned` forever, with the evidence it
        # produced floating beside it. A literature review IS a verification
        # method, so it gets the node, the lifecycle and the produces edge.
        create=frozenset({"Evidence", "EmpiricalBase", "VerificationMethod"}),
        update_attrs=frozenset({"EmpiricalBase"}),
        transitions=_transitions("Evidence", "VerificationMethod"),
        edges=_edges("relates_to", "supports", "refutes", "refines",
                     "defines_scope", "tested_by", "uses",
                     ("produces", "VerificationMethod", "Evidence")),
    ),
    "MedicalAgent": AgentPerm(
        # Same as the ResearchAgent above: a PubMed review is a method, and the
        # findings it returns belong to it and not to the bare question.
        create=frozenset({"Evidence", "VerificationMethod"}),
        update_attrs=frozenset(),
        transitions=_transitions("Evidence", "VerificationMethod"),
        edges=_edges("relates_to", "supports", "refutes", "refines",
                     "tested_by", "uses",
                     ("produces", "VerificationMethod", "Evidence")),
    ),
    "CoderAgent": AgentPerm(
        # It already owned the VerificationMethod lifecycle (transitions below)
        # but could not open one, so work the plan never named ran with no
        # method to point at and its evidence hung off the hypothesis with
        # nothing in between.
        create=frozenset({"Tool", "CodeArtifact", "GeneratedData", "Evidence",
                          "VerificationMethod"}),
        update_attrs=frozenset({"Tool"}),
        transitions=_transitions("Tool", "VerificationMethod"),
        edges=_edges("uses", "consumes", "derived_from", "supports", "refutes",
                     "refines", "relates_to", "tested_by",
                     ("produces", "VerificationMethod", "Evidence")),
    ),
    "DatasetCollectorAgent": AgentPerm(
        create=frozenset({"GeneratedData", "EmpiricalBase"}),
        update_attrs=frozenset({"EmpiricalBase"}),
        transitions=frozenset(),
        edges=_edges("derived_from", "defines_scope", "relates_to"),
    ),
    "ExperimentAgent": AgentPerm(
        # Same hole as the coder's: it ran the method and could move it through
        # planned→running→done/failed, but could not create the node it was
        # moving. Creating a type also carries the right to enrich it
        # (_stage_merge), so `failure_reason` needs no separate grant.
        create=frozenset({"Evidence", "GeneratedData", "VerificationMethod"}),
        update_attrs=frozenset(),
        transitions=_transitions("VerificationMethod"),
        edges=_edges("uses", "consumes", "supports", "refutes", "refines",
                     "relates_to", "derived_from", "tested_by",
                     ("produces", "VerificationMethod", "Evidence")),
    ),
    # NOT an agent: the deterministic mirror that turns each registered plan
    # task into a planned VerificationMethod (agents/callbacks/tool_callbacks.py).
    # It is named as its own write-source rather than borrowing the planner's or
    # the orchestrator's, because a reader who sees "plan-mirror" on a method
    # knows no model chose it — it is the roadmap, one card per step. It writes
    # methods and attaches them; it never runs or judges anything.
    "plan-mirror": AgentPerm(
        create=frozenset({"PlanStep"}),
        update_attrs=frozenset({"PlanStep"}),
        # It follows the tracker, so it moves a step through the tracker's own
        # states. It still judges nothing and runs nothing.
        transitions=_transitions("PlanStep"),
        edges=_edges("realises"),
    ),
    # The same idea one grain down: the experiment module's approved plan,
    # mirrored task by task. A source of its own, so a reader who sees it knows
    # no model chose these cards.
    "experiment-plan-mirror": AgentPerm(
        create=frozenset({"ExperimentTask"}),
        update_attrs=frozenset({"ExperimentTask"}),
        # …and the one thing it may say about the step ABOVE its tasks: that
        # work on it has begun. Usually that is a step nobody has started. A
        # retry may also resume a block that these same experiment tasks caused;
        # graph_bridge checks the block's source and reason before asking for
        # that move, so operator/replan/report blocks remain protected. The
        # outer plan's own mirror only runs on an orchestrator tick, and by then
        # the tracker has usually moved the step from «не начат» straight to
        # «выполнен» — so a step that was being worked on for minutes was never
        # once drawn as being worked on. The module knows the moment a task
        # starts; this lets it say so, and nothing else about the step.
        transitions=_transitions("ExperimentTask",
                                 ("PlanStep", "todo", "in_progress"),
                                 ("PlanStep", "blocked", "in_progress")),
        edges=_edges("elaborates",
                     ("realises", "VerificationMethod", "ExperimentTask"),
                     ("realises", "Evidence", "ExperimentTask")),
    ),
    # The module's own deterministic writes — the methods, tools, evidence and
    # data it records once a task has actually run. Those commits go through the
    # privileged path (`enforce_permissions=False`), so this entry is mostly
    # documentation — except that `_stage_merge` consults the table whatever the
    # flag says, so WITHOUT it the bridge's re-approval path (an id-only attrs
    # merge onto the method it already wrote) is refused on every replan, and
    # refused silently, because the bridge swallows the error by contract.
    "ExperimentModule": AgentPerm(
        create=frozenset({"VerificationMethod", "Tool", "Evidence",
                          "GeneratedData"}),
        update_attrs=frozenset({"VerificationMethod"}),
        transitions=_transitions("VerificationMethod", "Tool",
                                 ("Hypothesis", "formulated", "postponed"),
                                 ("Hypothesis", "postponed", "formulated")),
        edges=_edges("tested_by", "uses", "produces", "relates_to",
                     "derived_from"),
    ),
    # The final write-up, published by code rather than by a model. The
    # aggregator that produces the text holds the READ-ONLY research surface,
    # and widening an LLM's rights to record something deterministic buys
    # nothing — `finalize_report` already has the markdown in hand. Same shape
    # as the plan mirrors above: a named source with exactly the rights it uses.
    # `update_attrs` matters because re-running finalize must enrich the node it
    # already wrote instead of adding a second one, and `_stage_merge` consults
    # this table even when enforcement is off.
    "report-writer": AgentPerm(
        create=frozenset({"Report"}),
        update_attrs=frozenset({"Report"}),
        transitions=frozenset(),
        edges=_edges("derived_from"),
    ),
    # NOT an agent: the deterministic writer behind a node's own write-up. It
    # stamps ONE attribute family onto the node it summarised and nothing else —
    # no status, no edges, no new nodes — so its rights are stated a (type,
    # attribute) pair at a time rather than by owning the type. The body lives
    # in the session's artifact store; only its id is on the node, because
    # `_truncate_attrs` caps an ordinary attribute at 2 000 characters and a
    # write-up would reach the page as its first two paragraphs.
    "node-report": AgentPerm(
        create=frozenset(),
        update_attrs=frozenset(),
        transitions=frozenset(),
        edges=frozenset(),
        update_fields=frozenset(
            (node_type, attribute)
            for node_type in ("Evidence", "PlanStep", "ExperimentTask",
                              "VerificationMethod")
            for attribute in ("report_artifact_id", "report_stamp", "report_lang")
        ),
    ),
    # NOT an agent: the code that turns a DOI or a PMC id an agent cited into
    # the file this session holds. It stamps the citation's own attributes onto
    # the Evidence that carried it and nothing else — no status, no edges, no
    # new nodes — so its rights are stated one (type, attribute) pair at a time
    # rather than by owning the type. Matched on identifiers, never on a title:
    # two papers share a title far more often than they share a DOI.
    "paper-linker": AgentPerm(
        create=frozenset(),
        update_attrs=frozenset(),
        transitions=frozenset(),
        edges=frozenset(),
        update_fields=frozenset({
            ("Evidence", "session_artifact_id"), ("Evidence", "doi"),
            ("Evidence", "pmcid"), ("Evidence", "paper_title"),
            ("Evidence", "paper_year"),
        }),
    ),
    # The pre-stage context-initialization agent seeds the framing frame at the
    # start of a run. It writes the whole context star through the PRIVILEGED
    # init path (store.init_research, enforce_permissions=False), so these rights
    # matter only if it ever writes through research_commit directly; they are
    # kept aligned with INIT_SEED_TYPES for clarity and for schema tests.
    "ContextInitAgent": AgentPerm(
        # `Spec` is the техническое задание, written after the frame is
        # confirmed — through an ordinary commit, not through `init_research`,
        # which starts a NEW study and would archive the graph it was just
        # seeded into. So this grant is load-bearing, not decorative.
        create=frozenset({"ResearchQuestion", "Constraint", "Tool", "Resource",
                          "EmpiricalBase", "ConfirmationCriteria", "CostModel",
                          "Spec"}),
        update_attrs=frozenset({"ResearchQuestion", "Spec"}),
        transitions=frozenset(),
        edges=_edges("contextualizes", "defines_scope", "applies_to",
                     "derived_from"),
    ),
    # The human writes through the HITL bridge (web endpoint / approval flow),
    # never through an LLM toolset. Expert evidence is the human's own layer-1
    # contribution in the meta-model ("Свидетельство … подтипы: … экспертное;
    # Человек в контуре — экспертные"), and it is checked to actually carry
    # subtype=expert in validate_node_draft. Without the edges the human could
    # create a Constraint but never attach it to anything, leaving it orphaned.
    "human": AgentPerm(
        create=frozenset({"Constraint", "Evidence"}),
        update_attrs=frozenset(),
        transitions=_transitions(("Conclusion", "draft", "approved")),
        edges=_edges("contextualizes", "regulates", "relates_to",
                     ("refines", "Evidence", "Hypothesis"),
                     ("supports", "Evidence", "Hypothesis"),
                     ("refutes", "Evidence", "Hypothesis")),
    ),
}


# ── Validation helpers (pure functions — unit-testable without a store) ──────

def _perm(agent: str) -> Optional[AgentPerm]:
    return AGENT_PERMISSIONS.get(agent)


def validate_node_draft(agent: str, node_type: str, status: str,
                        attrs: Dict, enforce_permissions: bool = True) -> List[str]:
    """Errors for creating a node of `node_type` with `status` as `agent`.

    `enforce_permissions=False` keeps the STRUCTURAL checks (known type, valid
    initial status, required subtype) but skips the per-agent ACL — used by the
    privileged research_init seeding path."""
    errors: List[str] = []
    spec = NODE_TYPES.get(node_type)
    perm = _perm(agent)
    if spec is None:
        allowed = ", ".join(sorted(perm.create)) if perm else "none"
        return [f"unknown node type '{node_type}'. "
                f"Allowed for you ({agent}): {allowed}."]
    if enforce_permissions:
        if perm is None:
            return [f"agent '{agent}' has no write access to the research graph."]
        if node_type not in perm.create:
            errors.append(
                f"agent '{agent}' may not create '{node_type}' nodes "
                f"(your types: {', '.join(sorted(perm.create))}). "
                f"This node type is owned by a different role — do NOT create it; "
                f"state the need in your text answer and the responsible agent "
                f"will add it.")
    if status not in spec.creatable:
        errors.append(
            f"'{status}' is not a valid initial status for {node_type}. "
            f"Allowed at creation: {', '.join(spec.creatable)}.")
    if spec.subtype_required:
        subtype = normalize_token(str((attrs or {}).get("subtype", "")))
        if subtype not in spec.subtypes:
            errors.append(
                f"{node_type} requires attrs.subtype — one of: "
                f"{', '.join(spec.subtypes)}.")
        # Measured evidence must say what it was measured on. A run once
        # benchmarked a locally written stand-in because the repository the
        # hypothesis named returned 404, and reported the result as a comparison
        # of the named systems: the substitution was mentioned in the coder's
        # report and nowhere in the record, so nothing downstream could see it.
        for needed in spec.required_attrs_by_subtype.get(subtype, ()):
            if not str((attrs or {}).get(needed, "")).strip():
                errors.append(
                    f"{subtype} {node_type} requires attrs.{needed} — "
                    f"{spec.attr_docs.get(needed, 'see the schema')}")
        # The human may file evidence, but only the kind a human actually is:
        # expert judgement. Routing a computational or literature result through
        # the HITL bridge would launder its provenance.
        if enforce_permissions and agent == "human" and node_type == "Evidence" \
                and subtype != "expert":
            errors.append(
                "the human may only file Evidence with subtype='expert'; a "
                f"'{subtype or 'missing'}' result must be recorded by the agent "
                "that produced it.")
    return errors


def validate_edge(agent: str, edge_type: str, from_type: str,
                  to_type: str, enforce_permissions: bool = True) -> List[str]:
    """Errors for creating an edge of `edge_type` between those node types."""
    perm = _perm(agent)
    pairs = EDGE_TYPES.get(edge_type)
    if pairs is None:
        own = sorted({e for e, _, _ in perm.edges}) if perm else []
        return [f"unknown edge type '{edge_type}'. "
                f"Your edge types: {', '.join(own)}."]
    if (from_type, to_type) not in pairs:
        allowed = "; ".join(f"{f} → {t}" for f, t in pairs)
        hint = " Swap from/to?" if (to_type, from_type) in pairs else ""
        return [f"'{edge_type}' must connect {allowed}; "
                f"got {from_type} → {to_type}.{hint}"]
    if enforce_permissions:
        if perm is None:
            return [f"agent '{agent}' has no write access to the research graph."]
        if (edge_type, from_type, to_type) not in perm.edges:
            own = sorted({e for e, _, _ in perm.edges})
            return [f"agent '{agent}' may not create '{edge_type}' edges "
                    f"({from_type} → {to_type}). Your edge types: {', '.join(own)}."]
    return []


def validate_transition(agent: str, node_type: str, from_status: str,
                        to_status: str, enforce_permissions: bool = True) -> List[str]:
    """Errors for moving a `node_type` node from `from_status` to `to_status`."""
    perm = _perm(agent)
    allowed = STATUS_TRANSITIONS.get(node_type, frozenset())
    if (from_status, to_status) not in allowed:
        nexts = sorted(t for f, t in allowed if f == from_status)
        reachable = ", ".join(nexts) if nexts else "none (terminal)"
        extra = (" Refuted branches stay in the graph as negative results."
                 if (node_type, from_status) == ("Hypothesis", "refuted") else "")
        return [f"{node_type} cannot go '{from_status}' → '{to_status}'. "
                f"Allowed from '{from_status}': {reachable}.{extra}"]
    if enforce_permissions:
        if perm is None:
            return [f"agent '{agent}' has no write access to the research graph."]
        if (node_type, from_status, to_status) not in perm.transitions:
            own = sorted({f"{t}({f}→{s})" for t, f, s in perm.transitions})
            mine = ", ".join(own) if own else "none"
            return [f"agent '{agent}' may not change {node_type} status. "
                    f"Your transitions: {mine}. Ask the orchestrator."]
    return []


def permitted_summary(agent: str) -> Dict[str, List[str]]:
    """Human/LLM-readable summary of an agent's write rights — rendered into
    that agent's prompt section, so prompt and enforcement share one table."""
    perm = _perm(agent)
    if perm is None:
        return {"create": [], "edges": [], "transitions": [], "update_attrs": []}
    create = []
    for t in sorted(perm.create):
        spec = NODE_TYPES[t]
        sub = f" (attrs.subtype: {'/'.join(spec.subtypes)})" if spec.subtype_required else ""
        must = "; ".join(
            f"{st} REQUIRES attrs.{', attrs.'.join(keys)}"
            for st, keys in sorted(spec.required_attrs_by_subtype.items()))
        create.append(f"{t}{sub}" + (f" — {must}" if must else ""))
    edges_by_type: Dict[str, List[str]] = {}
    for edge, f, t in sorted(perm.edges):
        edges_by_type.setdefault(edge, []).append(f"{f}→{t}")
    edges = [f"{edge} ({', '.join(pairs)})" for edge, pairs in edges_by_type.items()]
    trans_by_type: Dict[str, List[str]] = {}
    for typ, f, t in sorted(perm.transitions):
        trans_by_type.setdefault(typ, []).append(f"{f}→{t}")
    transitions = [f"{typ}: {', '.join(pairs)}" for typ, pairs in trans_by_type.items()]
    fields_by_type: Dict[str, List[str]] = {}
    for typ, attr in sorted(perm.update_fields):
        fields_by_type.setdefault(typ, []).append(f"attrs.{attr}")
    return {
        "create": create,
        "edges": edges,
        "transitions": transitions,
        "update_attrs": sorted(perm.update_attrs)
        + [f"{typ} ({', '.join(attrs)} only)"
           for typ, attrs in sorted(fields_by_type.items())],
    }
