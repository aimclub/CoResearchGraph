# Dynamic Execution Graph (DEG)

A materialised graph of how the multi-agent system solved a task. It serves two
distinct consumers from **one** store:

| | Fact 1 — Interpretability | Fact 2 — Agent context |
|---|---|---|
| When | offline, after/between runs | online, within a run |
| Reads | the **whole** graph + outcomes | a **projection** (slice) |
| Goal | understand → distil / cache / optimise | raise per-step quality |
| Demands | persistence, cross-run comparability, a **quality signal on nodes** | cheap, relevant, budgeted views |

Because Fact 1 needs a quality signal, every node carries the **critic verdict**
(approve/revise/reject, sufficient/insufficient/wrong) and a status — the critic
is the graph's reward channel. Successful subgraphs become distillation/cache
material; repeated-failure paths become optimisation targets.

## Layers

- **Root layer (static) — Capability/Module graph.** Nodes = our agents
  (Orchestrator, Hypotheses, Research, TaskExecutor, Coder, Medical) and the MCP
  servers/tools in the registry. Answers *what the system can do*. (This is where
  the earlier "inject available capabilities" idea belongs — as structure, not
  prompt text.)
- **Reasoning layer (dynamic) — Execution graph growing from the root.** Nodes =
  research steps; each links to its executor module (root) and its parent step.
  Answers *what the system did*.

## Configuration (decided)

- **Active** graph-guided planning is the priority (orchestrator reads graph
  state to decide), not passive-only logging.
- In the default Web/CLI deployment, an in-process registry holds one execution
  graph per public `(user_id, session_id)` and snapshots it to
  `graph_runs/sessions/<user>/<session>/execution.json`. Every prompt in that
  session appends another goal tree. A new session gets a different graph;
  refresh, reconnect and Stop preserve the existing one.
- The central Graph service/emitter remains the integration surface for a fully
  distributed A2A mesh; cross-process scope propagation is still required there.
- **Raw agent-call granularity** first (nodes = delegations + tool calls from the
  `event_logger` stream). Semantic abstraction (research-intention nodes) is a
  later layer.

## Schema (raw MVP, forward-compatible with semantic abstraction)

```
Node:
  id: str                 # f"{run_id}:{n}"
  run_id: str
  kind: goal | agent_call | tool_call | decision | reflection
  executor_agent: str | None
  label: str              # request text / command / short summary
  status: running | success | failed | interrupted | pruned
  parent_ids: list[str]
  input: Any | None
  output: str | None
  verdict: str | None     # critic verdict — reward signal (Fact 1)
  t_start: float
  t_end: float | None
  semantic: {type, goal, entity} | None   # filled by the abstraction layer (phase 2)

Edge:
  src, dst: str
  type: caused_by | delegated_to | produced | failed_into
        | depends_on | validated_by | branches_to
  run_id
```

## Graph identity over A2A — `run_id` propagation (the systems contribution)

The graph is one **within a public research session**, not across all Web users.
When agents live on separate A2A servers with separate internal sessions, a
stable public scope/run id must ride through each delegation so a sub-agent's
events land in that session's graph. `context_id` changes per call, so:

- the orchestrator derives a stable `run_id` from its own session at task start;
- it attaches `run_id` to each A2A delegation via the `RemoteA2aAgent`
  request-metadata hook;
- a sub-agent server's emitter reads `run_id` from the incoming request metadata
  (falls back to its own session id when called directly).

In a monolithic MAS this problem does not exist — it is specific to our A2A mesh.

## Projection Layer — Fact 2 (the context-engineering core)

> relevance = **role × position**. An agent's context is
> `project(graph, role, current_node, budget)`, never the whole graph.

Three composable mechanisms:

1. **Role views.** Orchestrator (the only planner) → planning summary: frontier /
   failed / validated / completed. Sub-agent → local view: ancestral path (why it
   was called) + in-scope validated findings + reflections; *not* the frontier or
   sibling branches.
2. **Table-of-contents (hierarchical collapse).** Done subtrees collapse to one
   line ("Literature search KRAS → done: 3 scaffolds"); the active path stays
   expanded — like a book ToC with the current section detailed.
3. **Ego-graph by edge type.** From the current node walk `caused_by`/`depends_on`
   upward (why), `validated_by` (reuse), `failed_into` (avoid), within a token
   budget.

Pipeline: status-filter → ego-neighbourhood → ToC render. **Rule-based** first
(reproducibility is an evaluation metric); LLM collapse of done subtrees optional
later.

**Delivery over A2A.** Orchestrator pulls its own summary in `before_model`. For
sub-agents the orchestrator **pushes** the projected local view into the
delegation envelope — delegation = `task + projection-for-this-agent` — so
sub-agents need not call the service. The service stays source-of-truth for
Fact 1 and the orchestrator's self-pull.

## Components

- **Graph Memory Service** — FastAPI + NetworkX (per `run_id`), JSON snapshots for
  persistence/replay; Neo4j later, same API. Endpoints: `/node`, `/edge`,
  `/node/{id}/status`, `/summary`, `/graph`.
- **Graph Emitter** — derived from the existing `event_logger` plugin; translates
  before_tool / after_tool / after_model events into node/edge/status ops.
- **Projection Layer** — `project(role, node, budget)` (above).
- **Active hook** — orchestrator `before_model` appends the summary (the seam the
  capability injection used).

## Phased plan

1. Storage + emitter + `run_id` (raw graph from a real run) + active summary hook.
2. Step Abstraction (raw nodes → research-intention nodes; rule + LLM; node-type
   whitelist against semantic drift). The reasoning-novelty lives here.
3. Projection Layer refinement (ToC + ego-graph, budgeted) on live graphs.
4. Constraints/branching (retry limits, pruning, rollback) + reflection nodes.
5. Evaluation — baselines: vanilla / passive / active. Metrics: task success,
   repeated failures, token consumption, explainability, reproducibility.
