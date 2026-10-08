# CoScientist Knowledge Graph

## Concept

The multi-agent system now builds an explicit **graph of the research process**
instead of leaving it implicit in agent chatter. The thesis (Proposition 1 —
"MAS for science, differentiated by the graph structure of research"): making the
process a first-class typed graph buys us three things —

1. **Interpretability** — you can see exactly what each agent did and why.
2. **Reproducibility** — the graph is a deterministic, replayable protocol of a run.
3. **Performance (graph memory)** — facts established in earlier runs are retrieved
   and fed back into later runs, so the system builds on prior findings instead of
   rediscovering them. This is the lever we expect to move the benchmark metrics.

It is exposed through three related **views** with deliberately different
lifetimes:

| View | Scope | Answers | How it's built |
|------|-------|---------|----------------|
| **execution** | one user session | *what the system did* | recorded live from agent events; all prompts in the session append to it |
| **knowledge** | one user session | *what this session learned* | deterministic projection of exact scoped provenance (no extra LLM call) |
| **memory** | the whole CoScientist installation | *what we know overall* | LLM entity/relation extraction, accumulated across users and sessions |

## Architecture

```
USER QUERY
  │  (execution graph, live)
  ▼
goal ── OrchestratorAgent ── agent_call: TaskExecutorAgent ── agent_call: ToolPipelineAgent ── agent_call: ExperimentAgent ── tool_call: dock(...)
              │                                                                                                                       │
              └── result  ◀────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                    │
                    │  (semantic extraction, LLM, on the final answer)
                    ▼
        Entities/Relations  ──ingest──►  GLOBAL KNOWLEDGE MEMORY (persistent)
                                              │  target:GSK-3beta, molecule:CCO {docking:-9.2}, …
                                              │
   any later session ◀── inject "Known from prior work: …" ◀── relevant(query)
```

**The reuse loop is the point:** one session produces a final answer → we extract
typed domain entities/relations from it → merge them into a persistent memory
graph shared by the installation (dedup by canonical key, accumulate
mentions/sources/provenance). Before any later session acts, we retrieve entities
relevant to the new query and inject them into context. Provenance retains the
producing `user_id`, `session_id`, research/run/goal/result ids, agent and
timestamp, so globally reused facts remain auditable. That is the "graph memory"
the ablation study targets
(base vs +memory vs +MCP vs +Fedot).

## One data-driven template (no class-per-entity)

Everything is `Node` + `Edge` (`graph/models.py`). A node's *kind* and an entity's
*domain type* are **strings in data**, never Python subclasses, so new entity or
relation types need zero code changes:

- `Node.kind`: system | agent | goal | agent_call | tool_call | result | **entity**
- an entity node carries its domain type in `semantic.type` (`target`, `molecule`,
  `metric`, `paper`, `hypothesis`, `method`, …) and its fields in `input` (attrs).
- `Edge.type` is a free string — control-flow (`caused_by`, `delegated_to`,
  `produced`) and knowledge relations (`has_property`, `about`, `supports`, …).

The LLM extraction returns exactly this template:
`{entities:[{key,type,name,attrs}], relations:[{src,dst,type}]}` (`graph/semantic.py`).

## How agents interact with it

- **Root on start:** the orchestrator and planner get the graph root up front
  (every agent + capabilities + this session's trace) **plus relevant global
  prior-run memory** — `inject_graph_root` → `{graph_root?}` in their prompts.
- **Read anytime:** every reasoning agent has graph tools
  (`read_research_graph` / `get_graph_history` / `get_agents_info`).
- **Grows automatically:** `GraphMemoryPlugin` (an ADK plugin on the Runner)
  records goals, delegations, tool calls and results — no agent bookkeeping.

## Visualization

Live web page `/graph` (vis-network, vendored — works offline): nodes are
**coloured by status** (running/success/failed) and **shaped by type**; click any
node for full detail incl. the prompt an agent was called with; a **view switch**
toggles execution / knowledge / memory; updates live (1.5 s poll). Also a CLI:
`python -m CoScientist graph viz|show|dot --view {execution,knowledge,memory}`.
For a session snapshot, pass both `--user-id <id> --session-id <id>`; the
unscoped `--run` option remains for legacy flat snapshots.

The storage and API boundaries are explicit:

```text
graph_runs/knowledge_memory.json                         # global, durable
graph_runs/sessions/<user>/<session>/execution.json     # session-local
graph_runs/sessions/<user>/<session>/research_active.json # session-local
```

- `GET /api/knowledge` returns the global memory graph without requiring a Web
  session. The session graph endpoint keeps `view=memory` as a compatibility
  alias to the same global graph.
- `GET /api/users/{user}/sessions/{session}/graph?view=execution|research`
  always resolves exactly one registered session.
- `view=knowledge` projects only facts whose provenance points to that session;
  identical query text from another user cannot leak into the projection.
- Browser refresh, reconnect and Stop preserve both session stores and the
  global memory; the knowledge view is recomputed from them. An explicit session
  reset can clear/archive that session's execution/research data, but it never
  clears the global memory.

When upgrading from the earlier per-user implementation, the first global
memory resolution scans `graph_runs/users/*/knowledge_memory.json` and merges
valid files into the canonical graph. Source files are retained unchanged. The
canonical JSON records imported source paths in
`_meta.migrated_user_memories`, so restarts do not duplicate counts or
provenance; malformed sources are skipped and can be repaired and retried.

The local JSON backend is a **single-process/single-writer** implementation.
The in-process lock makes concurrent sessions safe inside the default one-worker
Web service, but several Uvicorn workers, CLI writers or A2A processes must not
share one `KG_MEMORY_PATH`: they can overwrite each other's snapshots. A
transactional shared backend is tracked with the SQLite persistence TODO.

## Code map

```
graph/
  models.py        Node/Edge/Semantic schema (the one template)
  store.py         NetworkX-backed store + JSON snapshots (per run)
  memory.py        in-process session graph + root seeding (agent roster)
  plugin.py        GraphMemoryPlugin — grows the execution graph from events,
                   and (when enabled) triggers semantic extraction on the result
  agent_tools.py   read-only graph tools given to every agent
  knowledge.py     execution → knowledge-view projection (deterministic, no LLM)
  semantic.py      Entity/Relation/Extraction template + LLM extractor (Option B)
  memory_store.py  KnowledgeMemory — one persistent global entity/relation graph
  viz.py           to_dot / interactive HTML (colour by status, shape by type)
agents/callbacks/tool_callbacks.py
                   inject_graph_root — root + relevant prior-run memory into prompts
web/app.py         session-scoped graph API plus the global /api/knowledge endpoint
web/templates/graph.html   live, clickable, view-switching graph UI
```

Wiring: the `graph` tool and `inject_graph_root` callback are registered in
`assembly/bindings.py` and attached in `agents/system.yaml`; the plugin is added
to the Runner in `main.py`.

## Flags

| flag | default | effect |
|------|---------|--------|
| `GRAPH__ENABLED=false` | on | master switch, also exposed as **Settings → Graphs → Knowledge Graph** in the web UI: the `graph` reader toolset drops off every agent (and out of their prompts), `inject_graph_root` yields nothing, and `GraphMemoryPlugin` records nothing, so the Graph view stays empty |
| `LOG_AGENT_EVENTS=0` | on | turn the graph/logging off |
| `KG_SEMANTIC_ENABLED=0` | **on** | LLM entity extraction into memory (1 small LLM call per query); set 0 to disable |
| `KG_SEMANTIC_MODEL` | main model | model used for extraction |
| `KG_MEMORY_PATH` | `graph_runs/knowledge_memory.json` | canonical global knowledge-memory file |
| `GRAPH_SNAPSHOT_DIR` | `./graph_runs` | base directory for per-session execution snapshots |

## Status & next steps

- Execution + knowledge views and the cross-run memory loop are implemented and
  tested (deterministically; semantic extraction verified with a mocked LLM).
- Semantic extraction is on by default; set `KG_SEMANTIC_ENABLED=0` when an
  environment must avoid the additional extraction call.
- For the paper/ablation: vary graph-growing strategy (plan-all-at-once vs
  incremental vs hybrid) and toggle the memory layer; measure task metrics,
  redundant-call count, and result variance (reproducibility) on the chem
  benchmarks where targets/scores are quantitatively checkable.
