# Developing CoScientist — the YAML-driven agent system

This is the hands-on guide for changing the multi-agent system: adding agents,
tools, callbacks, prompts, HITL, and A2A exposure. For running the A2A stack
see [a2a.md](a2a.md); for the execution graph see
[execution_graph.md](execution_graph.md).

---

## 1. Mental model

The system is **declared** in one file and **built** by one package:

```
CoScientist/agents/system.yaml      WHAT the system is
        │                           (agents, hierarchy, tools, callbacks,
        │                            HITL flags, prompts, models, A2A)
        ▼
CoScientist/assembly/               HOW it gets built
  schema.py      validates the YAML (pydantic; refs, cycles, ports, root)
  registry.py    name -> implementation tables
  bindings.py    fills the tables: every tool/callback/class/schema by name,
                 plus each tool's prompt documentation (ToolDoc)
  prompting.py   PromptContext — renders <<TOOLS>>/<<AGENTS>>/<<ROUTING>>/<<HITL>>
  assembler.py   build_system() — constructs the ADK agent tree
        │
        ▼
google.adk agents (LlmAgent / SequentialAgent / ParallelAgent / custom)
```

Three consumers build from the same declaration, so they can never disagree:

| Consumer | Entry point |
|---|---|
| In-process system (`adk web`, `main.py`) | `CoScientist.agents` → `build_system()` |
| A2A orchestrator (remote sub-agents) | `build_system(remote_subagents=True)` |
| A2A servers, cards, ports, `run_all` | `a2a/serve.py`, `a2a/run_all.py`, `a2a/config.py` |

**The core invariant:** an agent's prompt advertises *exactly* what is wired.
Tool sections are rendered from the `ToolDoc`s of the tools actually attached;
agent rosters from the subordinates actually attached; the HITL section appears
only when the HITL tools are attached. The assembler *verifies* this at build
time and refuses to start on a mismatch — drift between prompts and wiring is a
startup error, not a runtime surprise.

Rule of thumb: **never hand-write a tool name or an agent name inside prompt
text.** If you need one, it must come from a placeholder or a `ctx` helper.

---

## 2. Quick start

All commands run from the repo root (`/app`), not from `CoScientist/`:

```bash
# Validate the YAML and print the build plan (no LLM calls):
python -m CoScientist.assembly

# Run the assembly invariants test-suite:
pytest tests/unit/test_assembly.py -q

# Inspect any agent's rendered prompt:
python -c "
from dotenv import load_dotenv; load_dotenv()
from CoScientist.assembly import build_system
print(build_system().agent('ResearchAgent').instruction)"

# Run the system:
adk web                       # in-process
A2A_MODE=1 adk web            # A2A orchestrator (sub-agents must be running)
python -m CoScientist.a2a.run_all
```

Both `adk web` modes print every agent's thoughts, tool calls and tool results
to the console (the same event-logger plugin the A2A servers use) and append
the same trace to `AGENT_LOG_FILE` (default `/app/agent_events.log`); disable
with `LOG_AGENT_EVENTS=0`.

The historical imports still work — `from CoScientist.agents import
research_agent, orchestrator_agent, ...` re-exports the assembled instances.

---

## 3. system.yaml field reference

```yaml
agents:
  MyAgent:
    class: llm              # llm | sequential | parallel | custom:<registered name>
    root: false             # exactly ONE agent has root: true
    enabled: true           # bool or "${dotted.settings.path}" (see §4.4)
    model: main             # main | coder | literal litellm model string
    description: ...        # THE one description (parent prompt roster,
                            # AgentTool declaration, A2A card)
    routing: ...            # "when to pick me" bullet in the parent's prompt
    planning: ...           # how the planner's roster describes me
    prompt: my_template     # registered prompt template name
    tools: [my_toolset]     # registered tool names (llm only)
    subordinates: [...]     # agents attached as AgentTool (llm only)
    children: [...]         # execution units (sequential/parallel only)
    callbacks:              # registered callback names by kind
      before_model: []
      after_model: []
      before_tool: []
      after_tool: []
      before_agent: []
      after_agent: []
    hitl: false             # see §4.5
    critic: false           # LLM critic reviews my output once — see §4.5a
                            # (custom:session only; bool or "${settings.path}")
    report_output: false    # post my final answer to the chat (see below)
    output_key: my_results  # ADK session-state key for the agent's output
    output_schema: ...      # registered pydantic schema name (structured output)
    planner: plan_react     # registered ADK planner name
    options: {}             # extra constructor kwargs (mostly for custom classes)
    a2a:                    # optional — expose as a standalone A2A service
      key: my_agent         # snake key: serve argument + <KEY>_PORT env prefix
      port: 8007
      env: {}               # env defaults applied before tool modules import
      skill: {id: ..., name: ..., description: ..., tags: []}
```

Semantics worth knowing:

- **`enabled: false` does not delete the agent.** It is still *built* (and can
  still be served standalone over A2A); it just detaches from every parent —
  it disappears from rosters, routing, the planner's roster, and the critic's
  roster automatically.
- `subordinates` order = roster order in the prompt = tool order.
- **`report_output: true` puts the agent's final answer in the chat.** A
  subordinate runs as an `AgentTool`, so its deliverable (the hypotheses, the
  research summary) comes back inside the *caller's* function_response and is
  never spoken in the event stream — the user only saw the orchestrator's
  retelling of it. `AgentOutputPlugin` (`logging/agent_output.py`) closes that
  gap: it reports the answer of every flagged agent to the web UI, which renders
  it as a message authored by that agent. Reserve it for agents whose output IS
  a deliverable — pipeline internals (rerankers, retrievers) would only add
  noise.
- A composite (`sequential`/`parallel`) cannot have `tools`, `prompt`, `model`
  or `subordinates` — the schema rejects it.
- Unknown keys anywhere are rejected (`extra="forbid"`), so typos fail loudly.

---

## 4. Recipes

### 4.1 Give an existing agent a new tool

1. **Register the tool** in `assembly/bindings.py` — a factory plus a `ToolDoc`
   per tool the entry contributes:

   ```python
   def _weather():
       from CoScientist.tools.weather_tools import weather_toolset_instance
       return weather_toolset_instance

   REGISTRY.register_tool(ToolEntry(
       key="weather",
       factory=_weather,
       docs=(
           ToolDoc(
               name="get_forecast",                      # exact callable name
               signature="get_forecast(location, days)", # shown in the prompt
               purpose="Fetch a weather forecast for a location.",
               usage=("Prefer ISO country codes for ambiguous city names.",),
           ),
       ),
   ))
   ```

   Flags:
   - `optional=True` — factory may return `None` (unconfigured service); the
     tool then silently drops out of the agent **and** its prompt.
   - `runtime_resolved=True` — for MCP toolsets whose real tool list comes from
     the remote server; exempts the entry from the attached-vs-documented name
     check (the docs are trusted as written).

   Keep factories lazy (import inside the function) — bindings must be
   importable without constructing MCP sessions.

2. **Add the name to the agent** in `system.yaml`:

   ```yaml
   ResearchAgent:
     tools: [websearch, paper_analysis, papers_search, weather]
   ```

3. Done. The prompt's `<<TOOLS>>` section now includes `get_forecast` — you do
   not edit the prompt. Verify: `python -m CoScientist.assembly && pytest
   tests/unit/test_assembly.py -q`.

The `ToolDoc`s in bindings are the ONLY place tool descriptions for prompts
live. If you change a tool's behavior, update its docstring (the ADK schema)
*and* its ToolDoc — the test suite can't read your mind about semantics, only
about names.

### 4.2 Add a brand-new agent (end to end)

1. Tools → §4.1.

2. **Prompt template** in `agents/prompts/templates.py`:

   ```python
   @_register("weather_reporter")
   def weather_reporter(ctx: PromptContext) -> str:
       return render_template('''
   You are a weather reporting agent.

   <<TOOLS>>

   ## Workflow
   1. Resolve the location the user means.
   2. Fetch the forecast and summarize it in two sentences.

   <<HITL>>
   ''', TOOLS=ctx.render_tools(), HITL=ctx.render_hitl())
   ```

3. **Declare it** in `system.yaml` and put it on the orchestrator's roster:

   ```yaml
   WeatherAgent:
     class: llm
     description: fetches and summarizes weather forecasts.
     routing: when the task needs current or forecast weather data.
     planning: |-
       use when:
           * the step needs weather or forecast data
     prompt: weather_reporter
     tools: [weather]
     output_key: weather_results

   OrchestratorAgent:
     subordinates: [..., WeatherAgent]
   ```

   Everything downstream updates by itself: the orchestrator's `<<AGENTS>>` and
   `<<ROUTING>>`, the pre-action critic's roster, the planner's AVAILABLE
   AGENTS section, and the execution-graph emitter's delegation detection.

4. **Validate**: `python -m CoScientist.assembly`, then the tests.

5. *(Optional)* expose over A2A → §4.9.

### 4.3 Add a callback

Implement it under `agents/callbacks/`, export it from
`agents/callbacks/__init__.py`, then register in `assembly/bindings.py` with
its **kind** (`before_model`, `after_model`, `before_tool`, `after_tool`,
`before_agent`, `after_agent`):

```python
_cb("my_callback", "after_tool", factory=lambda ctx: _my_callback())
```

Reference it in YAML under the matching kind — listing it under the wrong kind
is a build error. If the callback's behavior depends on the assembled config
(like the critics, whose LLM prompt embeds the orchestrator's roster), register
a context factory instead: `factory=lambda ctx: make_my_callback(...)` — `ctx`
is the agent's `PromptContext`.

### 4.4 Toggle agents from settings

`enabled` accepts a settings reference resolved against
`CoScientist.config.get_settings()`:

```yaml
PlannerAgent:
  enabled: ${orchestrator.use_planner}    # ORCHESTRATOR__USE_PLANNER in .env
```

Flipping the setting re-shapes every prompt that mentions the agent on the next
build — e.g. the orchestrator's planning step switches between "call the
PlannerAgent first" and "there is NO planner tool" automatically.

To toggle a *single tool on a single agent*, register a gated alias of the tool
whose factory returns `None` when the setting is off, and mark it `optional`:

```python
def _planner_retrieval():
    if not get_settings().web.planner_retrieval_enabled:
        return None
    return _retrieval()

REGISTRY.register_tool(ToolEntry(
    key="planner_retrieval", factory=_planner_retrieval,
    optional=True, docs=_RETRIEVAL_DOCS,
))
```

The agent lists the alias (`tools: [planner_retrieval, ...]`) and the prompt
template branches on `ctx.has_tool("planner_retrieval")`, so a disabled tool
disappears from the agent AND from its prompt in one step. `planner_retrieval`
and `planner_graph` (Settings → PlannerAgent in the web UI) work exactly this
way; every other agent keeps the ungated `retrieval` / `graph` entries.

To gate a tool **everywhere at once**, skip the alias and put the check in the
entry's own factory, marking the entry `optional`. Three switches in the web UI
work this way:

| Switch | Setting | Effect when off |
|--------|---------|-----------------|
| Graphs → Knowledge Graph | `web.knowledge_graph_enabled` | `graph` drops off every agent, `inject_graph_root` yields nothing, and `GraphMemoryPlugin` stops recording |
| Graphs → Research Graph | `research_graph.enabled` | `research_graph` / `research_graph_orchestrator` drop out with their prompt sections |
| CoderAgent → Coder Execution Mode | `web.coder_mode` (`local`/`openhands`) | When set to `openhands`, `coder` local toolset drops out, leaving the coder family with OpenHands `sandbox` tools only |

Prose that names specific tools has to branch too, or the prompt will advertise
a tool the agent cannot call and `guard_unknown_tools` will fire on every
attempt. The coder prompt swaps its whole persona on `ctx.has_tool("coder")` —
with the local toolset it is an engineer with the full operating manual, without
it a thin **sandbox relay** whose only job is to forward the incoming task
verbatim to `run_sandbox_task` (attaching `dataset_url`, choosing
`new_sandbox`) and pass the sandbox agent's report straight back; it is
deliberately given no engineering guidance, since writing instructions for the
sandbox agent is the sandbox agent's job. The orchestrator's KNOWLEDGE GRAPH
section renders only under `ctx.has_tool("graph")`.

When a tool's *documentation* depends on what else is configured, pass a
callable as `docs` — `ToolEntry.resolved_docs()` calls it at build time. The
`sandbox` entry uses this to stop cross-referencing `execute_bash` once the
local coder tools are switched off.

To make a **constructor kwarg** follow a setting, put the reference in
`options` — values there take `"${settings.path}"` too (resolved at build time
by `AgentConfig.resolved_options()`, keeping their own type):

```yaml
PlannerAgent:
  options:
    critic_max_rounds: ${web.planner_critic_rounds}   # Settings -> PlannerAgent
```

That is the whole path from a web-UI field to a runtime knob: the UI posts to
`/api/settings` → `web/app.py` writes `settings.web` → the next system build
resolves the reference. No code change per knob, and one place (`_settings_payload`)
that a GET and the echo of a POST both answer from, so they cannot drift.

### 4.5 HITL

`hitl: true` means "this agent uses human-in-the-loop *when HITL is globally
enabled*" (`HITL__ENABLED` in `.env`). What it does depends on the class:

- `class: llm` — attaches the `request_approval` / `request_selection` tools
  **and** renders their bullets into `<<TOOLS>>` plus the `<<HITL>>` guidance
  section. With HITL off, both the tools and the text disappear.
- `class: custom:session` (the planner) — passes the global `hitl_handler` so
  the agent runs its generate → human-review → revise loop.

Your template must contain `<<HITL>>` (it renders empty when off).

### 4.5a Critics

Three critics exist, wired independently — you can run any subset:

| Critic | Wiring | Judges |
|---|---|---|
| Pre-action | `callbacks.after_model: [pre_action_critique]` on the orchestrator | the delegation it is about to make (approve / revise args / reject) |
| Post-action | `callbacks.after_tool: [post_action_critique]` | the result a sub-agent returned (annotates it with a `_critic` directive) |
| Plan | `critic: true` on a `custom:session` agent (the planner) | the roadmap the planner registered |

All three are LLM calls returning strict JSON, built by *context factories* so
their prompts embed the roster they are judging against — the orchestrator's
subordinates for the first two, the planner's siblings (the agents a plan may
assign to) for the plan critic. Each prompt section that documents a critic is
gated on that critic actually being wired, so a prompt never announces a review
that cannot happen.

The plan critic is not a callback: no callback can make the planner redo its
roadmap. `make_plan_critique()` returns `async (task, plan) -> feedback | None`
and the assembler hands it to `SessionAgent`, which already owns the
generate → review → revise loop and feeds the critique back as a user turn — the
same mechanism as a human rejection, so the planner replans instead of patching
prose. Differences from HITL: it runs whether or not HITL is enabled (before the
human sees anything), and its budget is **one round** (`critic_max_rounds`) — a
reviewer that never tires would otherwise loop forever. After the rewrite the
plan stands, reviewed or not. A critic that fails or cannot articulate an
objection accepts the plan; the review must never take a run down with it.

Both knobs live in the web UI under Settings → PlannerAgent (and in `.env` for
headless runs):

| UI control | Setting | `.env` |
|---|---|---|
| Plan Critic | `web.planner_critic_enabled` → `PlannerAgent.critic` | `PLANNER__CRITIC_ENABLED` |
| Revision Rounds | `web.planner_critic_rounds` → `options.critic_max_rounds` | `PLANNER__CRITIC_ROUNDS` |

The budget is settable but never absent: the UI floor is one round, and a
rejected `criticRounds < 1` keeps the previous value — a zero-round critic is
just a critic that never runs, which is what the switch is for. Both apply to
new sessions (the system is built once per session), and the critic costs an
extra LLM call per planning run plus a whole planning round whenever it objects.

### 4.6 Composite pipelines

`sequential` / `parallel` agents list `children` in execution order:

```yaml
ToolPipelineAgent:
  class: sequential
  children: [ToolPreparerAgent, ExperimentAgent]
```

Children are full agents declared in the same file. Note that pipeline stages
usually communicate through ADK session state: one agent's `output_key` is the
next agent's `{state_key}` prompt injection (see §5) — renaming an
`output_key` means updating the prompts and callbacks that read it.

A composite can itself be a `subordinate`: listed that way it is attached as an
AgentTool and an LLM agent above it decides *whether* to run the whole pipeline.
That is how `TaskExecutorAgent` works — an LLM router whose subordinates are
`ToolPipelineAgent` (the discover→deploy→run MCP pipeline) and `CoderAgent`, so
the choice between "a ready tool exists" and "this needs engineering" is made
inside the executor instead of by the orchestrator.

### 4.7 Custom agent classes

For behavior YAML can't express, write a class and register it:

```python
# agents/custom_agents.py (BaseAgent) or your own module (LlmAgent subclass)
REGISTRY.register_agent_class("my_kind", MyAgentClass)   # in bindings.py
```

```yaml
MyAgent:
  class: custom:my_kind
  options: {plan_file_path: roadmap.txt}   # extra constructor kwargs
```

`LlmAgent` subclasses get the full treatment (model, prompt, tools, callbacks,
output_key/schema, planner, HITL); plain `BaseAgent` subclasses get
`name`/`description`/`options`.

### 4.8 Structured output & planners

```python
REGISTRY.register_output_schema("my_ranking", MyRankingModel)  # bindings.py
REGISTRY.register_planner("plan_react", PlanReActPlanner)
```

```yaml
MyRanker:
  output_schema: my_ranking   # ADK: structured-output agents take no tools
  planner: plan_react
```

### 4.9 Expose over A2A

Add the `a2a:` section (see §3). That alone gives you:

- `python -m CoScientist.a2a.serve my_agent` — a standalone server with a
  generated AgentCard (name/description from the agent, skill from the YAML);
- inclusion in `python -m CoScientist.a2a.run_all`;
- a `RemoteA2aAgent` in the A2A orchestrator (if the agent is a subordinate);
- port override via `MY_AGENT_PORT`.

Use `a2a.env` for env vars that must exist *before tool modules import* (e.g.
the coder's shared workspace id) — `serve.py`/`run_all.py` apply them with
`setdefault` first.

### 4.10 Models

`model:` is `main` (default), `coder`, or a literal litellm model string. The
named ones come from `settings.llm.main_model` / `settings.llm.coder_model`
(`LLM__MAIN_MODEL` / `LLM__CODER_MODEL` in `.env`) and are wrapped in
`RetryingLiteLlm` (transient-error retries) either way.

---

## 5. Writing prompt templates

Templates live in `agents/prompts/templates.py`, registered under the name the
YAML references. A template is a function `(ctx: PromptContext) -> str`; for
fully static text use `_static("name", '''...''')`.

**Placeholders** (filled via `render_template`, `<<NAME>>` sentinels):

| Placeholder | Renders |
|---|---|
| `<<TOOLS>>` | "You have access to the following tools:" + one bullet per ToolDoc of every *attached* tool (incl. HITL tools) + the only-listed-tools guard |
| `<<AGENTS>>` | `* **Name** — description` per enabled subordinate |
| `<<ROUTING>>` | `- Name — routing` per enabled subordinate |
| `<<HITL>>` | HITL usage guidance, or empty when HITL tools aren't attached |

**`ctx` helpers** for conditional sections:

| Helper | Use |
|---|---|
| `ctx.has_tool("papers_search")` | gate workflow steps on an optional tool being configured |
| `ctx.has_subordinate("PlannerAgent")` | gate text on a roster member (e.g. the planning step) |
| `ctx.is_enabled("TaskExecutorAgent")` | gate on any agent's enabled flag |
| `ctx.siblings()` | my parents' other enabled subordinates (e.g. coder's scope boundary) |
| `ctx.render_sibling_roster()` | planner-style roster from each sibling's `planning` text |
| `ctx.render_critic_roster()` | compact roster for critic prompts (pass `ctx.siblings()` for a critic judging peers, as the plan critic does) |

Two templating systems coexist — don't confuse them:

- `<<NAME>>` — filled **once, at build time**, by the assembler. Chosen over
  `str.format` so literal `{}` (JSON examples) never needs escaping.
- `{state_key}` — single braces are **ADK session-state injection at run
  time** (e.g. `{filtered_tools}`, `{accumulated_tools}`). Leave them verbatim;
  they must match a producer's `output_key` or callback state write.

House rules:

1. No hand-written tool/agent names in text — placeholders and `ctx` only.
   (The few intentional exceptions are gated with `has_subordinate`, so the
   text vanishes with the agent.)
2. A leftover `<<...>>` after rendering is a build error — you'll hear about a
   forgotten substitution immediately.
3. Blank-line runs left by empty placeholders are collapsed automatically; no
   need to micro-manage whitespace.

---

## 6. What the drift protection actually catches

At `build_system()` time, per agent:

| Error | Meaning |
|---|---|
| `Unknown tool 'x'. Known: ...` | YAML references a name nobody registered in bindings |
| `MyAgent: required tool 'x' is not available` | factory returned `None` and the entry isn't `optional` |
| `MyAgent: callback 'x' is a after_tool callback, listed under before_model` | callback kind mismatch |
| `MyAgent: prompt/tool mismatch — attached but undocumented: [...]; documented but not attached: [...]` | a function tool is wired without a ToolDoc, or a ToolDoc exists for a tool that isn't attached |
| `MyAgent: prompt 'x' left placeholders unfilled: ['<<TOOLS>>']` | template didn't substitute a placeholder |
| `Agent dependency cycle: A -> B -> A` / `unknown agent reference` / `Exactly one agent must have root` / duplicate a2a keys/ports | schema validation |

Limits to know: MCP toolsets (`runtime_resolved=True`) fetch their tool surface
from the remote server at runtime, so for them the name check is skipped — keep
their ToolDocs honest by hand. And no static check can verify *semantics*; if a
tool's behavior changes, update its ToolDoc (§4.1).

---

## 7. Testing & debugging

```bash
python -m CoScientist.assembly             # validate + build plan
pytest tests/unit/test_assembly.py -q      # invariants (no LLM calls)
```

The test-suite asserts, among other things: every YAML reference resolves; the
orchestrator's prompt names exactly its enabled subordinates; every attached
function tool is named in its agent's prompt; the critic prompt embeds the
current roster; the planner uses real ADK names; the orchestrator documents
only the critics that are wired. If you add an agent, these tests cover it
automatically — add agent-specific assertions only for agent-specific behavior.

Debugging tips:

- **See what the model sees:** `build_system().agent("X").instruction`.
- **Critic prompts** aren't on an agent — render via the registry:

  ```python
  from CoScientist.assembly.prompting import PromptContext
  from CoScientist.assembly.registry import REGISTRY
  from CoScientist.assembly.schema import get_config
  cfg = get_config()
  print(REGISTRY.prompt("pre_action_critic")(PromptContext(config=cfg.root, system=cfg)))
  # the plan critic renders against the PLANNER's context, not the root's:
  planner = PromptContext(config=cfg.agent("PlannerAgent"), system=cfg)
  print(REGISTRY.prompt("plan_critic")(planner))
  ```

- **Config is cached per process** (`get_config()` is `lru_cache`d): a running
  `run_all`/`adk web` won't see YAML edits — restart it. In tests, build from
  an explicit path with `build_system(config_path=...)` if you need a variant.
- **Settings come from `.env`** (nested with `__`, e.g. `HITL__ENABLED`,
  `ORCHESTRATOR__USE_PLANNER`) — when behavior looks wrong, check the resolved
  values, not the field defaults.
- Run from `/app`, never from inside `CoScientist/` (the package's `logging`
  module would shadow the stdlib).

---

## 8. File map

```
CoScientist/
  agents/
    system.yaml          ← the system declaration (start here)
    common.py            make_llm()/make_coder_llm(), HITL handler, settings
    custom_agents.py     custom BaseAgent classes
    callbacks/           callback implementations (wired by name via bindings)
    prompts/
      templates.py       ← prompt templates (registered by name)
      builder.py         render_template() + PromptBuilder
  assembly/
    schema.py            YAML schema + loader (get_config / load_config)
    registry.py          registry types (ToolEntry, ToolDoc, CallbackEntry)
    bindings.py          ← name → implementation + ToolDocs (edit when adding)
    prompting.py         PromptContext (placeholder renderers)
    assembler.py         build_system() + consistency checks
  a2a/
    serve.py             serve one agent by a2a key
    run_all.py           serve everything
    config.py            ports/URLs derived from the YAML
    server.py            make_agent_card() + make_a2a_app()
    orchestrator.py      the remote-mode orchestrator
tests/unit/test_assembly.py   assembly invariants
```

A typical change touches at most three places: `bindings.py` (new names),
`templates.py` (new prompt), `system.yaml` (the wiring) — and nothing else.
