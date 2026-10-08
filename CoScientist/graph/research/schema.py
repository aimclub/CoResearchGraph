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
            "source_ref": "paper DOI, dataset, run id, …",
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
            "synthesis": "the synthesized finding",
            "validity_bounds": "limits of validity",
            "new_question": "optional follow-up question text",
        },
    ),
    # ── Layer 2 — methodological frame ───────────────────────────────────────
    NodeTypeSpec(
        "VerificationMethod", "VM", 2,
        statuses=("planned", "running", "done", "failed"), creatable=("planned",),
        attr_docs={
            "method_type": "computational / laboratory / analytical / statistical / expert",
            "inputs": "what it needs",
            "outputs": "what it yields",
            "cost": "estimated cost",
            "limitations": "known weaknesses",
        },
    ),
    NodeTypeSpec(
        "ConfirmationCriteria", "CC", 2,
        statuses=("not_met", "met"), creatable=("not_met",),
        attr_docs={
            "threshold": "quantitative/qualitative bar",
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
    "VerificationMethod": frozenset({("planned", "running"), ("planned", "failed"),
                                     ("running", "done"), ("running", "failed"),
                                     ("failed", "planned")}),
    "ConfirmationCriteria": frozenset({("not_met", "met"), ("met", "not_met")}),
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
    "tested_by": (("Hypothesis", "VerificationMethod"),),
    "requires": (("Hypothesis", "Tool"),),
    "uses": (("VerificationMethod", "Tool"),),
    "consumes": (("VerificationMethod", "Resource"),),
    "produces": (("VerificationMethod", "Evidence"),
                 ("Conclusion", "ResearchQuestion")),
    "supports": (("Evidence", "Hypothesis"),),
    "refutes": (("Evidence", "Hypothesis"),),
    "refines": (("Evidence", "Hypothesis"),),
    "based_on": (("Conclusion", "Evidence"),),
    "determines_sufficiency": (("ConfirmationCriteria", "Conclusion"),),
    # Not in the spec's edge table, but the docx says criteria are "formulated
    # for a hypothesis" and the closable-path trigger needs the linkage.
    "formulated_for": (("ConfirmationCriteria", "Hypothesis"),),
    "regulates": (("Constraint", "VerificationMethod"),
                  ("Constraint", "ConfirmationCriteria")),
    "constrains": (("Constraint", "Hypothesis"),
                   ("Constraint", "VerificationMethod")),
    "derived_from": tuple((a, t) for a in _ARTIFACT_TYPES
                          for t in ("Conclusion", "Evidence")),
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
            ("Hypothesis", "postponed", "formulated")),          # scheduling only
        edges=_edges("contextualizes", "defines_scope", "derived_from", "applies_to",
                     "motivates", "regulates", "constrains",
                     "relates_to", "supports", "refutes", "refines",
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
        edges=_edges("motivates", "tested_by", "requires", "formulated_for",
                     "uses", "consumes"),
    ),
    "ResearchAgent": AgentPerm(
        create=frozenset({"Evidence", "EmpiricalBase"}),
        update_attrs=frozenset({"EmpiricalBase"}),
        transitions=_transitions("Evidence"),
        edges=_edges("relates_to", "supports", "refutes", "refines", "defines_scope"),
    ),
    "MedicalAgent": AgentPerm(
        create=frozenset({"Evidence"}),
        update_attrs=frozenset(),
        transitions=_transitions("Evidence"),
        edges=_edges("relates_to", "supports", "refutes", "refines"),
    ),
    "CoderAgent": AgentPerm(
        create=frozenset({"Tool", "CodeArtifact", "GeneratedData", "Evidence"}),
        update_attrs=frozenset({"Tool"}),
        transitions=_transitions("Tool", "VerificationMethod"),
        edges=_edges("uses", "consumes", "derived_from", "supports", "refutes",
                     "refines", "relates_to", ("produces", "VerificationMethod", "Evidence")),
    ),
    "DatasetCollectorAgent": AgentPerm(
        create=frozenset({"GeneratedData", "EmpiricalBase"}),
        update_attrs=frozenset({"EmpiricalBase"}),
        transitions=frozenset(),
        edges=_edges("derived_from", "defines_scope", "relates_to"),
    ),
    "ExperimentAgent": AgentPerm(
        create=frozenset({"Evidence", "GeneratedData"}),
        update_attrs=frozenset(),
        transitions=_transitions("VerificationMethod"),
        edges=_edges("uses", "consumes", "supports", "refutes", "refines",
                     "relates_to", "derived_from",
                     ("produces", "VerificationMethod", "Evidence")),
    ),
    # The pre-stage context-initialization agent seeds the framing frame at the
    # start of a run. It writes the whole context star through the PRIVILEGED
    # init path (store.init_research, enforce_permissions=False), so these rights
    # matter only if it ever writes through research_commit directly; they are
    # kept aligned with INIT_SEED_TYPES for clarity and for schema tests.
    "ContextInitAgent": AgentPerm(
        create=frozenset({"ResearchQuestion", "Constraint", "Tool", "Resource",
                          "EmpiricalBase", "ConfirmationCriteria", "CostModel"}),
        update_attrs=frozenset({"ResearchQuestion"}),
        transitions=frozenset(),
        edges=_edges("contextualizes", "defines_scope", "applies_to"),
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
