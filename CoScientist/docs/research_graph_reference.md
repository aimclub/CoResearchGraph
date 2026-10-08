# Research Context Graph — Encyclopedia of Nodes & Edges

Complete reference for every node type and edge type in the research context
graph (`CoScientist/graph/research/`). The authoritative source is
`graph/research/schema.py`; this document is generated from and kept in sync
with it. For the concept/lifecycle overview see
[research_graph.md](research_graph.md).

**How to read a node entry:** *purpose* · *id prefix & layer* · *attributes* ·
*status lifecycle* (allowed transitions; anything not listed is forbidden and a
terminal status has no outgoing transitions) · *created by / enriched by /
transitioned by* (which agents may do it) · *edges in/out*.

Node record shape: `{id, type, attrs, status, source, created_at, updated_at, status_history}`.
Edge record shape: `{id, type, from, to, attrs, source, created_at}`.
IDs are auto-assigned per type: `Q1`, `H2`, `E3`, `VM1`, `CC1`, `T1`, …

---

## Layer 1 — Epistemic objects (what science produces)

### `ResearchQuestion` — id `Q`
The root of a research; everything hangs off it. One graph = one root question
(sub-questions can be spawned by a Conclusion).
- **attrs:** `formulation` (the question), `domain`, `gap` (the knowledge gap),
  `target_setting` (what kind of answer is sought), `research_form`
  (fundamental / exploratory / applied + TRL).
- **status:** `open` → `decomposed` → `closed` (also `open`→`closed`).
- **created by:** Orchestrator · **enriched by:** Orchestrator ·
  **transitioned by:** Orchestrator.
- **edges out:** `motivates`→Hypothesis, `defines_scope`→EmpiricalBase.
- **edges in:** `contextualizes` from Constraint, `produces` from Conclusion
  (a new question), `relates_to` from Evidence, `applies_to` from CostModel/EfficiencyMetric.

### `Hypothesis` — id `H`
A testable claim. The central object that ties every layer together: it is
motivated by a question, tested by methods, needs tools, judged by criteria, and
supported/refuted by evidence.
- **attrs:** `formulation`, `rationale`, `priority`.
- **status:** `formulated` → `under_verification` → {`confirmed` | `refuted` |
  `postponed`}; `formulated`→`postponed`; `postponed`→`formulated`.
  `under_verification` **locks the branch** (a second start is rejected).
  `confirmed`/`refuted` are terminal — refuted branches stay as negative results.
- **created by:** HypothesesAgent · **transitioned by:** HypothesesAgent (only
  `formulated`→`postponed`) and Orchestrator (all, incl. the confirm/refute
  verdict — **only the orchestrator adjudicates a hypothesis**).
- **edges out:** `tested_by`→VerificationMethod, `requires`→Tool.
- **edges in:** `motivates` from Question, `formulated_for` from Criteria,
  `supports`/`refutes`/`refines`/`relates_to` from Evidence, `constrains` from
  Constraint, `applies_to` from CostModel/EfficiencyMetric.

### `Evidence` — id `E`
An atomic observation/fact. The unit that moves a hypothesis.
- **attrs:** `subtype` **(required)** — one of `literature` / `experimental` /
  `computational` / `expert` / `meta`; `content`, `reliability`, `source_ref`.
- **status:** `obtained` → {`validated` | `rejected`} (both terminal).
- **created by:** ResearchAgent, MedicalAgent (literature/expert), CoderAgent,
  ExperimentAgent (computational/experimental) · **transitioned by:** the
  creators above + Orchestrator.
- **edges out:** `supports`/`refutes`/`refines`→Hypothesis, `relates_to`→Question/Hypothesis.
- **edges in:** `produces` from VerificationMethod, `based_on` from Conclusion,
  `derived_from` from any artifact.

### `Conclusion` — id `CL`
The synthesis over a hypothesis and its evidence: confirmation, refutation,
partial, or a new-question spawn. Includes validity bounds.
- **attrs:** `synthesis`, `validity_bounds`, `new_question` (optional).
- **status:** `draft` → `approved`.
- **created by:** Orchestrator · **transitioned by:** Orchestrator and `human`
  (the human-in-the-loop `draft`→`approved` sign-off).
- **edges out:** `based_on`→Evidence, `produces`→ResearchQuestion (a new root).
- **edges in:** `determines_sufficiency` from Criteria, `derived_from` from
  artifacts, `applies_to` from CostModel/EfficiencyMetric.

---

## Layer 2 — Methodological frame (how knowledge is obtained & judged)

### `VerificationMethod` — id `VM`
A concrete procedure that yields evidence for a hypothesis.
- **attrs:** `method_type` (computational / laboratory / analytical /
  statistical / expert), `inputs`, `outputs`, `cost`, `limitations`.
- **status:** `planned` → `running` → {`done` | `failed`}; `planned`→`failed`;
  `failed`→`planned` (retry).
- **created by:** HypothesesAgent (it proposes how to test its hypothesis) ·
  **transitioned by:** CoderAgent, ExperimentAgent (the executors that run it).
- **edges out:** `uses`→Tool, `consumes`→Resource, `produces`→Evidence.
- **edges in:** `tested_by` from Hypothesis, `regulates`/`constrains` from
  Constraint, `applies_to` from CostModel/EfficiencyMetric.

### `ConfirmationCriteria` — id `CC`
The formal bar at which evidence is deemed sufficient. The gate for closing a
hypothesis into a Conclusion.
- **attrs:** `threshold`, `confirmations_needed`, `reproducibility`.
- **status:** `not_met` ↔ `met`.
- **created by:** HypothesesAgent · **transitioned by:** Orchestrator (it judges
  whether the collected evidence meets the bar).
- **edges out:** `formulated_for`→Hypothesis, `determines_sufficiency`→Conclusion.
- **edges in:** `regulates` from Constraint.

---

## Layer 3 — Feasibility context (what is possible and what constrains it)

### `Tool` — id `T`
A capability that transforms inputs into results but is not consumed:
computational library, lab instrument, external API, trained model.
- **attrs:** `name`, `tool_type` (computational / laboratory / analytical /
  informational), `requirements`.
- **status:** `available` | `needs_adaptation` → `being_created` →
  {`available` | `creation_failed`}; `creation_failed`→`being_created` (retry).
  A tool that is not `available` keeps every hypothesis that `requires` it
  **BLOCKED** (see the `ready_hypotheses` trigger).
- **created by:** HypothesesAgent (declares a NEED as `needs_adaptation`),
  CoderAgent (builds/integrates one), Orchestrator (declares known available
  tools at `research_init`) · **enriched by:** CoderAgent ·
  **transitioned by:** CoderAgent (the build lifecycle).
- **edges in:** `requires` from Hypothesis, `uses` from VerificationMethod.
- ⚠️ *Design smell:* three creators — see Design Notes.

### `Resource` — id `R`
Anything with a finite budget that is consumed: GPU-hours, tokens, reagents,
time, expert-hours.
- **attrs:** `resource_type`, `remaining` (numeric), `limit` (numeric).
- **status:** `available` ↔ `exhausted`. The `resources_low` trigger also fires
  when `remaining/limit < 0.1`.
- **created by / enriched by / transitioned by:** Orchestrator (declared at init;
  spent as the research proceeds).
- **edges in:** `consumes` from VerificationMethod.

### `EmpiricalBase` — id `EB`
The body of available observations: datasets, literature corpus, prior results,
a trusted knowledge base.
- **attrs:** `base_type` (dataset / corpus / knowledge_base), `volume`, `source_ref`.
- **status:** `created` (no lifecycle).
- **created by:** Orchestrator (at init), ResearchAgent, DatasetCollectorAgent ·
  **enriched by:** ResearchAgent, DatasetCollectorAgent (they grow it).
- **edges in:** `defines_scope` from ResearchQuestion.

### `Constraint` — id `C`
A cognitive/normative boundary on the research.
- **attrs:** `subtype` **(required)** — `profile` (the research profile /
  modality / target-setting) / `methodological_norms` / `theoretical_framework`
  / `domain_standards` / `ethics` / `expert_knowledge` / `roles`; `content`.
- **status:** `active` (no lifecycle).
- **created by:** Orchestrator (profile, norms, standards at init), `human`
  (expert knowledge, ethics — via HITL).
- **edges out:** `contextualizes`→Question, `regulates`→VerificationMethod/Criteria,
  `constrains`→Hypothesis/VerificationMethod.
- ⚠️ *Design smell:* `regulates`/`constrains` currently have **no permitted
  creator** — see Design Notes.

---

## Layer 4 — Artifacts (what gets materialized)

All statusless (`created`) and all attached to what they were built from via
`derived_from`→Conclusion/Evidence.

### `CodeArtifact` — id `CA`
Scripts, trained models, pipelines. **attrs:** `path`, `version`, `description`.
**created by:** CoderAgent.

### `GeneratedData` — id `GD`
Datasets produced during the research. **attrs:** `path`, `schema`, `volume`.
**created by:** CoderAgent, DatasetCollectorAgent, ExperimentAgent.

### `Report` — id `RP`
Internal / GOST / free-form report. **attrs:** `content` (text or link).
**created by:** Orchestrator.

### `Publication` — id `PB`
Conference/journal article. **attrs:** `content`. **created by:** Orchestrator.

### `Spec` — id `SP`
Technical specification (ТЗ) for deploying results. **attrs:** `content`.
**created by:** Orchestrator.

### `EfficiencyJustification` — id `EJ`
Document showing automation was justified: metrics, comparison, recommendations.
**attrs:** `content`. **created by:** Orchestrator.

---

## Layer 6 — Economics

### `CostModel` — id `CM`
Formalizes the cost of a research step. **attrs:** `rule` (cost formula/rule).
**status:** `created`. **created by:** Orchestrator. **edges out:** `applies_to`
→Question/Hypothesis/VerificationMethod/Conclusion.

### `EfficiencyMetric` — id `EM`
Aggregated time/cost/quality assessment. **attrs:** `time`, `cost`, `quality`.
**status:** `created`. **created by:** Orchestrator. **edges out:** `applies_to`
→same targets as CostModel.

> Layer 5 of the meta-model (transition triggers, information dependencies,
> completion criteria, AI-application mode) is **not** materialized as nodes —
> it lives in `queries.py` (the orchestrator's trigger queries) and in each
> node's `status_history`.

---

## Edge encyclopedia

Every edge is directed `from → to` and only connects the type pairs listed.
"created by" is which agents may author it.

| Edge | from → to | Meaning | Created by |
|---|---|---|---|
| `motivates` | Question → Hypothesis | the question gives rise to the hypothesis | Hypotheses, Orchestrator |
| `tested_by` | Hypothesis → VerificationMethod | which method tests the hypothesis | Hypotheses |
| `requires` | Hypothesis → Tool | the hypothesis needs this tool to be verified | Hypotheses |
| `uses` | VerificationMethod → Tool | the method runs on this tool | Hypotheses, Coder, Experiment |
| `consumes` | VerificationMethod → Resource | the method spends this budget | Hypotheses, Coder, Experiment |
| `produces` | VerificationMethod → Evidence; Conclusion → Question | a step creates a new node | Coder, Experiment (VM→E); Orchestrator (CL→Q) |
| `supports` | Evidence → Hypothesis | evidence backs the hypothesis | Research, Medical, Coder, Experiment |
| `refutes` | Evidence → Hypothesis | evidence contradicts it (→ review trigger) | Research, Medical, Coder, Experiment |
| `refines` | Evidence → Hypothesis | evidence narrows/qualifies it | Research, Medical, Coder, Experiment |
| `based_on` | Conclusion → Evidence | the conclusion is synthesized from this evidence | Orchestrator |
| `determines_sufficiency` | Criteria → Conclusion | the criterion gates this conclusion | Orchestrator |
| `formulated_for` | Criteria → Hypothesis | the criterion belongs to this hypothesis | Hypotheses |
| `regulates` | Constraint → VerificationMethod / Criteria | norms govern the method/criterion | **⚠ nobody** |
| `constrains` | Constraint → Hypothesis / VerificationMethod | ethics/theory limit it | **⚠ nobody** |
| `derived_from` | artifact → Conclusion / Evidence | the artifact is built from this | Coder, DatasetCollector, Experiment, Orchestrator |
| `contextualizes` | Constraint(profile) → Question | frame for interpreting the question | Orchestrator |
| `defines_scope` | Question → EmpiricalBase | which data is relevant to the question | Research, DatasetCollector, Orchestrator |
| `relates_to` | Evidence → Question / Hypothesis | attaches free-standing evidence | Research, Medical, Coder, DatasetCollector, Experiment |
| `applies_to` | CostModel / EfficiencyMetric → Question/Hypothesis/VM/Conclusion | what the economic node measures | Orchestrator |

---

## Who-writes-what (summary)

| Agent | Creates | Transitions | Enriches | Edges |
|---|---|---|---|---|
| **Orchestrator** | Question, Tool, Resource, EmpiricalBase, Constraint, Conclusion, Report, Publication, Spec, EfficiencyJustification, CostModel, EfficiencyMetric | Question, Hypothesis (all incl. confirm/refute), Evidence, Conclusion, Criteria, Resource | Question, Resource | motivates, contextualizes, defines_scope, based_on, determines_sufficiency, produces(CL→Q), derived_from, applies_to |
| **HypothesesAgent** | Hypothesis, VerificationMethod, ConfirmationCriteria, Tool | Hypothesis (→postponed) | — | motivates, tested_by, requires, formulated_for, uses, consumes |
| **ResearchAgent** | Evidence, EmpiricalBase | Evidence | EmpiricalBase | supports, refutes, refines, relates_to, defines_scope |
| **MedicalAgent** | Evidence | Evidence | — | supports, refutes, refines, relates_to |
| **CoderAgent** | Tool, CodeArtifact, GeneratedData, Evidence | Tool, VerificationMethod | Tool | uses, consumes, produces(VM→E), derived_from, supports, refutes, refines, relates_to |
| **DatasetCollectorAgent** | GeneratedData, EmpiricalBase | — | EmpiricalBase | derived_from, defines_scope, relates_to |
| **ExperimentAgent** | Evidence, GeneratedData | VerificationMethod | — | produces(VM→E), uses, consumes, supports, refutes, refines, relates_to, derived_from |
| **human** (HITL) | Constraint | Conclusion (→approved) | — | — |

---

## Design notes & open questions

These are surfaced by inverting the permission table; they are candidates for
the **selective-context** refinement (give each agent a view/permission set
narrow enough that it never *tries* what it can't do).

1. **`Tool` has three creators (Orchestrator, Hypotheses, Coder).** Intended
   split: Hypotheses declares a *need* (`needs_adaptation`), Coder *builds* one
   (`being_created`→`available`), Orchestrator declares *known available* tools —
   but only at `research_init`. The orchestrator's general `create Tool`
   permission lets it also invent tools mid-run, which is out of role. Fix:
   make `research_init` a privileged seeding path and drop Tool (and Resource /
   EmpiricalBase / Constraint) from the orchestrator's *general* create-set.

2. **`regulates` and `constrains` have no permitted creator.** Constraints can be
   created but never linked to the methods/criteria/hypotheses they govern.
   Fix: grant these edges to whoever owns constraint wiring (Orchestrator, and
   possibly `human`).

3. **`defines_scope` has three creators; `relates_to` has five.** Broad, but
   low-risk (both are attachment edges). Leave unless it causes confusion.

4. **Orchestrator ↔ Evidence.** The orchestrator repeatedly *tries* to create
   Evidence (it holds all workers' text). It is currently forbidden (workers own
   Evidence). Decision pending: allow it as a "recorder of record" (pragmatic) or
   keep the wall and rely on re-delegation (spec-faithful).

5. **Evidence self-validation.** ResearchAgent/MedicalAgent may transition their
   own Evidence `obtained`→`validated`. If independent validation matters, move
   validation to the Orchestrator only.

The broader principle for the next phase: today the model **proposes** and the
schema **rejects** (reactive guardrail). Selective context aims to make each
agent's prompt + tool surface + graph slice so role-specific that the wrong
write is never attempted in the first place (proactive).
