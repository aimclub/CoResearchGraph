# Pilot exact-tool discovery

## Observed failure

In the local scientific pilot, an explicit `TaskExecutorAgent` request named
`dataset_overview_heracleum_tox` and included its discovered server ID. The
request reached `ToolPipelineAgent`, but its `ToolRetrieverAgent` ended without
calling `retrieve_tools`. The downstream reranker then rejected the explicit
target because it was absent from that executor invocation's tool inventory:
`Explicit target tool dataset_overview_heracleum_tox was not retrieved`.

A separate broad executor request also failed after some MCP tools returned
data, with `ExperimentAgent cannot finish without a successful scientific MCP
call`. This design does not claim to resolve that receipt-accounting failure.

## Behavior

Only the `synapse_pilot` profile gains an after-model callback on
`ToolRetrieverAgent`. When its non-partial response is final and the incoming
task names an explicit `Target tool: <name>`, the callback checks the actual
`accumulated_tools` state. A matching tool with a server ID satisfies the
requirement. If it is absent, the callback returns one real ADK
`retrieve_tools(query=<name>)` function call. The result must pass through the
existing retrieval tool and populate the normal state; the callback never
invents tool metadata or a scientific result.

The callback records one exact-name attempt per invocation. If the target is
still absent after that attempt, it raises a clear error rather than looping
or allowing the reranker to continue with an unrelated tool. Requests without
an explicit target, partial model responses, and normal model-originated tool
calls retain their current behavior. No Synapse contract or other CoScientist
profile changes.

## Alternatives

- Add another prompt instruction: smaller text change, but the retriever
  already had a mandatory-call instruction and skipped it in the observed run.
- Call the RAG retriever directly inside the reranker callback: avoids an ADK
  round trip, but bypasses the normal observable tool-call path.
- Use the existing ADK tool path with one callback-generated exact query:
  selected because the call and response remain visible and bounded.

## Verification and scope

Write a failing regression for a final retriever response with a named target
missing from state, then verify the generated call and one-attempt limit.
Cover a present target, missing server ID, no target, partial response, and
model-originated tool call. Run an in-memory ADK Runner test with a stubbed
`retrieve_tools` transport so it observes the real call/response path. Build
the pilot image and test one targeted executor invocation before attempting
another full scientific project. Keep PR #401 draft until the remaining
receipt-accounting failure and end-to-end report are separately verified.
