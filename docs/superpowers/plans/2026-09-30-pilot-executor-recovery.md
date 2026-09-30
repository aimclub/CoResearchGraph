# Pilot Executor Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the isolated Synapse pilot request a real `TaskExecutorAgent` call when the orchestrator tries to finish after successful retrieval and research but before scientific execution.

**Architecture:** Extend only the `synapse_pilot` after-model callback. Reuse its existing `_request_missing_science` function, which resolves a discovered MCP server ID, emits an ADK function call, and limits attempts to one per tool and invocation. Preserve rejection when retrieval or research is missing, when discovery lacks the target server ID, and when the executor does not return verified science.

**Tech Stack:** Python, Google ADK `LlmResponse`, pytest, Docker Compose, Synapse project API.

## Global Constraints

- Change only the CoScientist pilot callback and its regression tests; do not change Synapse API or events.
- Never treat a generated function call as evidence: a successful observed executor response and scientific MCP receipt remain required.
- Leave regular CoScientist profiles and intermediate/partial responses unchanged.
- Run one new full Heracleum project after rebuilding only the CoScientist service; report the result from observed calls and receipts, not UI status alone.

---

### Task 1: Recover a missing first executor delegation

**Files:**
- Modify: `tests/test_synapse_pilot_delegation.py:488`
- Modify: `tests/test_synapse_pilot_delegation.py:577`
- Modify: `CoScientist/agents/callbacks/pilot_delegation.py:278`

**Interfaces:**
- Consumes: `require_pilot_delegations(callback_context, llm_response)`, `_request_missing_science(callback_context, name)`, and the test helpers `_context`, `_event`, `_model_response`.
- Produces: `LlmResponse` with a `TaskExecutorAgent` function call targeting `dataset_overview_heracleum_tox`, or a fail-closed `RuntimeError`.

- [ ] **Step 1: Write the failing test.** Add this test beside the existing missing-profile test:

```python
def test_pilot_requests_first_science_tool_when_final_skips_executor():
    context = _context(
        _event("retrieve_tools"),
        _event("ResearchAgent"),
        state={"accumulated_tools": [{
            "tool": "dataset_overview_heracleum_tox",
            "server_id": "heracleum-server",
        }]},
    )

    correction = require_pilot_delegations(context, _model_response())
    call = correction.content.parts[0].function_call
    assert call.name == "TaskExecutorAgent"
    assert "Target tool: dataset_overview_heracleum_tox" in call.args["request"]
    assert "server_id=heracleum-server" in call.args["request"]
```

- [ ] **Step 2: Verify the regression fails.** Run `pytest -q tests/test_synapse_pilot_delegation.py::test_pilot_requests_first_science_tool_when_final_skips_executor`. Expected: `RuntimeError` names the missing `TaskExecutorAgent`.

- [ ] **Step 3: Implement the minimal callback branch.** Insert immediately before the existing `if missing: raise RuntimeError(...)` block:

```python
    if missing == ["TaskExecutorAgent"]:
        return _request_missing_science(callback_context, _PILOT_SCIENCE_TOOLS[0])
```

- [ ] **Step 4: Verify green and retain fail-closed coverage.** Rerun the new test. Change the existing decorator to `@pytest.mark.parametrize("missing", REQUIRED[:2])`; leave the existing test body unchanged. Add these cases beside it:

```python
def test_pilot_rejects_missing_executor_without_discovered_server():
    context = _context(_event("retrieve_tools"), _event("ResearchAgent"))
    with pytest.raises(RuntimeError, match="dataset_overview_heracleum_tox.*not discovered"):
        require_pilot_delegations(context, _model_response())


def test_pilot_rejects_unverified_executor_after_one_targeted_attempt():
    context = _context(
        _event("retrieve_tools"), _event("ResearchAgent"),
        state={"accumulated_tools": [{
            "tool": "dataset_overview_heracleum_tox",
            "server_id": "heracleum-server",
        }]},
    )
    assert require_pilot_delegations(context, _model_response()) is not None
    with pytest.raises(RuntimeError, match="no verified result"):
        require_pilot_delegations(context, _model_response())


def test_pilot_does_not_count_failed_executor_response():
    context = _context(
        _event("retrieve_tools"), _event("ResearchAgent"),
        _event("TaskExecutorAgent", result={"status": "failed"}),
        state={"accumulated_tools": [{
            "tool": "dataset_overview_heracleum_tox",
            "server_id": "heracleum-server",
        }]},
    )
    correction = require_pilot_delegations(context, _model_response())
    assert correction.content.parts[0].function_call.name == "TaskExecutorAgent"
```

Run `pytest -q tests/test_synapse_pilot_delegation.py tests/test_pilot_handoff_adk.py tests/test_pilot_profile_a2a_handoff.py tests/test_synapse_a2a_boundary.py tests/test_synapse_native_a2a.py`. Expected: all pass.

- [ ] **Step 5: Review and commit.** Run `git diff --check`, `ruff check CoScientist/agents/callbacks/pilot_delegation.py tests/test_synapse_pilot_delegation.py`, and `black --check CoScientist/agents/callbacks/pilot_delegation.py tests/test_synapse_pilot_delegation.py` if those executables are present in the project environment. Inspect the diff and commit only the callback/test change on `feature/heracleum-report-demo`; push the same branch to PR #401. Do not create a new PR.

### Task 2: Verify the live pilot

**Files:**
- No source files; local Docker image and one new Synapse project only.

**Interfaces:**
- Consumes: committed CoScientist branch, isolated `synapse-coscientist-demo` stack, existing Heracleum prompt/workflow and `approval_mode=auto`.
- Produces: one project ID, run ID, trace ID, and evidence-based verdict.

- [ ] **Step 1: Build the image.** Build `docker/Dockerfile.a2a` from the committed branch; tag it with the branch commit and as `coscientist-local-demo:local`.
- [ ] **Step 2: Restart only CoScientist.** Recreate `coscientist-real` with `--no-deps --no-build`; confirm healthy A2A cards and the `synapse_pilot` profile. Do not restart Synapse or Mongo.
- [ ] **Step 3: Start one fresh Heracleum project.** Reuse the approved full prompt and pilot workflow in the local Synapse API, with `approval_mode=auto`.
- [ ] **Step 4: Check the evidence.** Compare ADK call/response events and MCP receipts with the report. A successful result needs observed retrieval, research, `TaskExecutorAgent`, and all four required science computations. If it fails, record the exact failing boundary without further speculative patches.

