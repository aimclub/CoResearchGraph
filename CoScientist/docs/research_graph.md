# Research Context Graph

A typed, schema-validated **property graph** that is the shared working memory
of one research investigation (one root question). Agents write their results
into it as typed nodes instead of passing dialog history to each other; the
orchestrator reads it to decide what to do next; worker agents receive a
**context slice** (the node they work on + its neighborhood), not the whole
graph.

Implements `developer/research_context_graph_spec.md` (the 6-layer meta-model of
the scientific process). Lives in `CoScientist/graph/research/`.

> **Not** the execution graph. `CoScientist/graph/*` (see `execution_graph.md`)
> is an *auto-recorded observational trace* of what agents did. This graph is a
> *generative blackboard* the agents author on purpose. They coexist; both are
> visible in the web `/graph` viewer. Both are scoped to one `(user_id,
> session_id)`. The semantic Knowledge Memory is intentionally different: it is
> global and accumulates reusable facts from every session and local user.

## Why

Agents were sharing state by relaying text. That loses structure, can't be
queried ("is any hypothesis ready to verify?"), and can't be trusted (an LLM can
claim anything in prose). The research graph makes the shared state **typed and
validated at the write boundary** — a hypothesis is a `Hypothesis` node with a
status lifecycle, evidence is `Evidence` with a subtype and edges to the
hypotheses it supports/refutes, and an agent can only write the node types its
role allows. That validation is the first barrier against hallucinated state.

## Data model

Node: `{id, type, attrs, status, source, created_at, updated_at, status_history}`.
Edge: `{id, type, from, to, attrs, source, created_at}`. Readable ids are
auto-assigned per type: `Q1`, `H2`, `E3`, `VM1`, `CC1`, `T1`, …

### Node types (layer · statuses; first = default at creation)

| Type | id | Statuses |
|---|---|---|
| ResearchQuestion | Q | open → decomposed → closed |
| Hypothesis | H | formulated → under_verification → confirmed / refuted / postponed |
| Evidence | E | obtained → validated / rejected (requires `attrs.subtype`: literature/experimental/computational/expert/meta) |
| Conclusion | CL | draft → approved |
| VerificationMethod | VM | planned → running → done / failed |
| ConfirmationCriteria | CC | not_met ↔ met |
| Tool | T | available / needs_adaptation → being_created → available / creation_failed |
| Resource | R | available ↔ exhausted |
| EmpiricalBase | EB | created |
| Constraint | C | active (requires `attrs.subtype`: profile/methodological_norms/theoretical_framework/domain_standards/ethics/expert_knowledge/roles) |
| CodeArtifact / GeneratedData / Report / Publication / Spec / EfficiencyJustification | CA/GD/RP/PB/SP/EJ | created |
| CostModel / EfficiencyMetric | CM/EM | created |

Layer 5 of the meta-model (transition triggers, completion criteria, AI
application mode) is **not** materialized as nodes — it is orchestrator logic,
implemented in `queries.py` and in each node's `status_history`.

### Edge types (allowed `from → to`)

`motivates` (Q→H), `tested_by` (H→VM), `requires` (H→T), `uses` (VM→T),
`consumes` (VM→R), `produces` (VM→E, CL→Q), `supports`/`refutes`/`refines`
(E→H), `based_on` (CL→E), `determines_sufficiency` (CC→CL), `formulated_for`
(CC→H), `regulates` (C→VM/CC), `constrains` (C→H/VM), `derived_from`
(artifact→CL/E), `contextualizes` (C→Q), `defines_scope` (Q→EB), `relates_to`
(E→Q/H), `applies_to` (CM/EM→Q/H/VM/CL).

### Russian ↔ English

Canonical identifiers are English. The spec's Russian terms are accepted on
input and normalized (`schema.RU_ALIASES`): e.g. `Гипотеза`→`Hypothesis`,
`на_проверке`→`under_verification`, `проверяется_через`→`tested_by`,
`опровергает`→`refutes`, `создаётся`→`being_created`. Input is also case- and
`-`/`_`-folded (`under-verification` → `under_verification`).

## Write permissions (spec §2)

Every write is attributed to the **calling agent** via `ToolContext.agent_name`
(not a parameter — the LLM can't spoof it) and checked against this table. A
write of a disallowed type/edge/transition is rejected before anything is saved.

| Agent | Creates | Status changes | Edges |
|---|---|---|---|
| OrchestratorAgent | ResearchQuestion, Tool, Resource, EmpiricalBase, Constraint, Conclusion, artifacts, CostModel, EfficiencyMetric | Question, Hypothesis, Resource, Conclusion, Criteria, Evidence(validate/reject) | contextualizes, defines_scope, based_on, determines_sufficiency, produces(CL→Q), derived_from, applies_to, motivates |
| HypothesesAgent | Hypothesis, VerificationMethod, ConfirmationCriteria | Hypothesis formulated→postponed | motivates, tested_by, requires, formulated_for, uses, consumes |
| ResearchAgent | Evidence, EmpiricalBase (+enrich EB) | Evidence | relates_to, supports, refutes, refines, defines_scope |
| MedicalAgent | Evidence | Evidence | relates_to, supports, refutes, refines |
| CoderAgent | Tool, CodeArtifact, GeneratedData, Evidence (+enrich Tool) | Tool, VerificationMethod | uses, consumes, produces(VM→E), derived_from, supports/refutes/refines, relates_to |
| DatasetCollectorAgent | GeneratedData, EmpiricalBase (+enrich EB) | — | derived_from, defines_scope, relates_to |
| ExperimentAgent | Evidence, GeneratedData | VerificationMethod | produces(VM→E), uses, consumes, supports/refutes/refines, relates_to, derived_from |
| human (HITL bridge) | Constraint | Conclusion draft→approved | — |

The `human` row is written through the HITL approval flow, not an LLM toolset.

## Tools (ADK)

Workers get `research_commit`, `research_context_slice`, `research_overview`,
`research_provenance`. The orchestrator additionally gets `research_init`,
`research_triggers`, `research_set_focus`. (Two ToolEntry keys, `research_graph`
and `research_graph_orchestrator`, so a worker's prompt never documents a tool
it doesn't have.)

`research_commit(nodes, edges, status_updates)` is the transactional write
(spec §5.3) — everything validated together, applied all-or-nothing:

```jsonc
nodes: [
  {"type": "Evidence", "attrs": {"subtype": "computational", "content": "AUC=0.91"}, "ref": "e1"},
  {"id": "EB1", "attrs": {"volume": "20k"}}          // enrich an existing node (no "type")
]
edges: [ {"type": "supports", "from": "#e1", "to": "H2"} ]   // "#ref" points at a node made in this call
status_updates: [ {"id": "H2", "status": "under_verification", "reason": "..."} ]
```

On failure the tool returns `{ok: false, errors: [...], hint: "...NOTHING was saved"}`
with instructive per-item messages so the model self-corrects and retries.

## How the orchestrator uses it (triggers, spec §3)

Before each step the orchestrator consults `research_triggers`, which evaluates
named queries over the graph (`queries.py`): **READY** hypotheses (all their
tools available), **BLOCKED**, **BACKLOG**, **REFUTE SIGNAL**, **CLOSABLE**
(evidence in + criteria met → write a Conclusion), **PENDING** conclusions,
**TOOLS** not ready, **RESOURCES LOW**, open **QUESTIONS**, **PROGRESS**. Moving
a hypothesis to `under_verification` **locks the branch** (a second start is
rejected) so two agents don't verify the same hypothesis. Refuted/postponed
branches are never deleted — they stay as negative results so work isn't
repeated.

### One hypothesis at a time

A generator agent proposes several hypotheses; the run verifies **one**. Two
mechanisms enforce it, so no single LLM decision can start a parallel batch:

* **Commit invariant** (`store._normalize_hypothesis_selection`): if one commit
  creates several hypotheses as `formulated`, only the picked one
  (`attrs.selected`, else the highest `attrs.priority`, else the first) stays
  active — the rest are created `postponed` and the commit returns a warning
  saying so. Nothing is dropped: they are the ranked backlog.
* **Trigger serialization** (`queries.ready_hypotheses`): `items` holds at most
  ONE hypothesis (the same ranking), the others come back under `queued` and are
  rendered as "do NOT verify in parallel"; anything already
  `under_verification` is rendered first as the active branch. When nothing is
  ready or active, `postponed_hypotheses` surfaces the **BACKLOG** so the
  orchestrator revives the next one (`postponed`→`formulated`).

`HypothesesAgent` is prompted to make that choice itself and to answer with a
`SELECTED HYPOTHESIS: <id> — …` line plus the backlog; the two mechanisms above
are the deterministic net under it.

## Context slices (spec §3)

Workers do not get the whole graph. `research_set_focus(node_id)` (orchestrator)
records the node the next worker should work on; the worker's
`{research_context?}` prompt block is that node's `get_context_slice`
(node + 1–2 hop neighborhood, depth capped by settings). Workers can pull more
with `research_context_slice`/`research_overview`/`research_provenance`.

## Lifecycle & persistence

Phases follow the spec: init (root + context star) → reconnaissance (literature
Evidence) → branching (Hypotheses + Methods + Criteria) → deepening (Tools built,
Evidence produced) → synthesis (Conclusions) → codification (artifacts). The
graph is snapshotted atomically to
`graph_runs/sessions/<user>/<session>/research_active.json` after every write
and **loaded when that scope is opened**. It survives the Web Stop button and
accumulates across all prompts in one session, so a session can contain one
complete research. `research_init` archives the previous graph inside the same
session directory. Nodes are never deleted.

## Web viewer

The active session opens `/graph?user_id=...&session_id=...`; the view selector
loads its **research graph (blackboard)** from
`/api/users/{user_id}/sessions/{session_id}/graph?view=research`. Nodes are
colored by type and shaped by kind; click a node for its attributes.

## Settings (`RESEARCH_GRAPH__*`)

`enabled` (default true — when false the tools and prompt sections vanish
entirely), `dir` (`./graph_runs`), `active_file` (`research_active.json`),
`slice_depth_max` (2), `slice_char_budget` (4000), `context_char_budget` (4000),
`reset_on_session` (false — keep the graph across prompts in its session).

## Limitation: A2A

The default Web/CLI deployment uses a thread-safe in-process registry keyed by
the public `(user_id, session_id)`. ADK `AgentTool` child sessions inherit that
scope through session state, so delegated agents use the same blackboard. Under
A2A (`remote_subagents=True`) each agent is a separate process and needs a shared
external scoped backend; that remains out of scope here.
