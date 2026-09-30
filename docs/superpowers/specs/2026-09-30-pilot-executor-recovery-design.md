# Pilot executor delegation recovery

## Context

In the live Synapse run `3c2f2465-2344-4dc8-8ba2-3dbd83f57a17`, the pilot
retrieved the prepared Heracleum MCP tools and received a `ResearchAgent`
response. The orchestrator then requested a final answer without ever calling
`TaskExecutorAgent`. The existing fail-closed callback correctly rejected that
answer, so the run produced no scientific computations or report.

## Behavior

Only the `synapse_pilot` profile uses `require_pilot_delegations`. On a
non-partial final response, the callback continues to reject a missing or
unsuccessful `retrieve_tools` or `ResearchAgent` call. If those succeeded but
no successful `TaskExecutorAgent` response exists, it returns a real ADK
`TaskExecutorAgent` function call targeting the first required, still-missing
Heracleum MCP tool. It uses the existing `_request_missing_science` helper,
which requires a discovered `server_id` and records one targeted attempt per
tool and invocation.

The returned call is not a receipt or fabricated result. The orchestrator
must observe the executor's actual response. If the MCP target was not
discovered, or the targeted attempt does not produce verified science, the
existing fail-closed behavior ends the run with an error. A final report still
requires every pilot science tool to have a verified result. Intermediate
tool calls, partial responses, successful runs, and other profiles retain
their current behavior. No Synapse API or event contract changes.

## Alternatives considered

- Retry the same run without changing code: may succeed, but leaves a known
  intermittent path where the model ends before scientific execution.
- Strengthen the prompt: it already explicitly requests the executor; another
  instruction does not provide a deterministic execution guarantee.
- Reuse the pilot's bounded recovery callback: selected because it enforces
  actual tool execution without inventing evidence or widening the workflow.

## Verification

Add a regression test for a final response after successful retrieval and
research with no executor. Assert the callback returns a targeted
`TaskExecutorAgent` function call. Keep tests showing that missing retrieval,
missing research, missing server IDs, and a failed targeted attempt still
raise errors. Run the pilot callback and A2A regression suites, rebuild the
local CoScientist image from the PR branch, and launch one fresh Synapse
Heracleum project. A `completed` UI status alone is insufficient: inspect the
ADK call/response pairs, MCP receipts, and grounded Russian report.
