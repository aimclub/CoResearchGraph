# Usage & cost metrics

What a session spent, broken down by agent: model calls, tokens, money — plus
the OpenHands sandbox's own bill (its agent's tokens, GPU seconds, electricity)
when the CoderAgent raised one.

One ledger per public `(user_id, session_id)`, in
[`CoScientist/logging/metrics.py`](../logging/metrics.py).

## What feeds it

| Source | Covers | How |
|---|---|---|
| `UsageMetricsPlugin` | every model call in the agent tree | ADK `after_model_callback`; plugins are inherited by `AgentTool` sub-runners, so a subordinate's own traffic is billed to the subordinate |
| `record_completion()` | the raw `litellm` calls off the tree — critic, semantic extraction, research validator | called at the three call sites; the session comes from the ambient binding |
| `record_sandbox_run()` | one finished sandbox run | `sandbox_tools._metrics_sink` forwards the client's metrics record |

The plugin is registered in [`main.py`](../main.py), [`agent.py`](../agent.py)
and [`a2a/server.py`](../a2a/server.py).

### Sessions for code that has no context

Background jobs spawned mid-run (semantic extraction, the validator) are too
deep to be handed a session key. The plugin's `before_run_callback` binds the
key into a `ContextVar`, and `asyncio.create_task` copies the current context —
so a job spawned during a run bills the run that spawned it.

## Metrics never reach a model

They are host metadata and are deliberately kept out of everything the model
reads:

* an agent that can see its own bill optimises the bill instead of the task —
  it cuts steps short and abandons experiments to "save budget";
* it would cost context tokens on every tool result to say so.

Concretely: nothing here is written into agent-visible state, and the sandbox
client keeps the record out of the dict `run_sandbox_task` returns (that dict
becomes prompt text). Do not interpolate `sandbox_metrics` — the session-state
key the client mirrors its journal into — into any instruction template.

## Getting the numbers out

```python
from CoScientist.logging.metrics import snapshot, format_report

data = snapshot(key=(user_id, session_id))
print(format_report(data))
```

* **Console** — `CoScientistManager.run()` logs the report at the end of every
  run. Disable with `LOG_USAGE_METRICS=0`.
* **HTTP** — `GET /api/users/{user_id}/sessions/{session_id}/metrics`, with
  `?report=1` for the console rendering as well.
* **Live** — the Web UI's *Usage & Cost* panel. The runtime registers a sink
  with `set_metrics_sink()` and gets a cumulative snapshot pushed at most once
  every `METRICS_PUSH_INTERVAL` seconds (default 2), plus a forced one when a
  run ends — a stopped or crashed run still spent money.

### Snapshot shape

```jsonc
{
  "session": {"user_id": "...", "session_id": "..."},
  "llm":     {"calls", "prompt_tokens", "completion_tokens", "cached_tokens",
              "total_tokens", "cost_usd", "seconds",
              "unpriced_calls", "unpriced_models"},
  "sandbox": {"runs", "wall_seconds", "agent_seconds", "queue_seconds",
              "cpu_core_seconds", "gpu_seconds", "energy_wh", "llm_calls",
              "total_tokens", "api_cost_usd", "energy_cost_usd", "total_cost_usd"},
  "totals":  {"cost_usd", "api_cost_usd", "energy_cost_usd", "llm_calls",
              "total_tokens", "energy_wh", "cpu_core_seconds", "gpu_seconds",
              "complete"},
  "agents":  [{"agent", "llm", "sandbox", "cost_usd", "models": [...]}]
}
```

`totals.cost_usd` = this process's model spend + the sandbox's total (API plus
energy). Agents are sorted by cost.

## Pricing, and when it is honest

Prices come dynamically from OpenRouter's live API (cached in memory for 1 hour) and litellm's model table.

When an OpenRouter model call is made, the price lookup queries OpenRouter's live pricing API directly. For other providers, prices are retrieved from litellm's model table.

A model that cannot be priced by any source is counted in **tokens** and reported as
`unpriced_calls`; `totals.complete` goes `false` and both the console report and
the UI say the total is a floor. Zero there means *unknown*, never *free*.

## Sandbox metrics

The client ([`openhands_sandbox.py`](../tools/coder_tools/openhands_sandbox.py))
fetches `GET /api/v1/metrics?task_id=<id>` once a run reaches a terminal status
and files the record in a per-session journal:

```python
from CoScientist.tools.coder_tools.openhands_sandbox import (
    get_sandbox_metrics, clear_sandbox_metrics,
)

m = get_sandbox_metrics(session_id=sid)   # or tool_context=tool_context
m["metrics"]   # last run's full record — may be None, the key always exists
m["runs"]      # every sandbox of the session
m["totals"]    # summed
```

Points worth knowing:

* **Only terminal runs are journalled.** `running`/`timeout` would give a
  non-final snapshot; pass `live=True` to poll one deliberately (progress UI).
* **Failures are journalled too** — `error` and `cancelled` runs show what was
  burned for nothing.
* **A follow-up replaces its sandbox's entry**, never adds one: the server
  reports cumulative figures per container, so adding would count the whole run
  again. `totals["runs"]` counts sandboxes, not tool calls.
* **Collection cannot fail a run.** The request happens after the task is over;
  if it fails you get the ordinary result and no journal entry.
* **Check the source fields, not just the numbers** — `compute.cpu_source`,
  `compute.energy.source`, `api.cost_source`. `unavailable` on the last one
  means the server has no price for that model (tokens are still right).
* Long-lived processes should call `clear_sandbox_metrics()` when a user
  session ends; it hands back the totals it discards. The in-process ledger is
  released for you when `CoScientistManager.close()` runs.
