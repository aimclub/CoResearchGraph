# Pilot Exact-Tool Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pilot retriever perform one observable exact-name lookup before rejecting an explicitly named scientific tool.

**Architecture:** An `after_model` callback on `ToolRetrieverAgent` inspects only structured `Target tool:` requests. If a final model response would leave the requested tool absent from `accumulated_tools`, it emits a real ADK `retrieve_tools` call once; subsequent absence is an error. The hook is registered only in `synapse_pilot`.

**Tech Stack:** Python 3.12, Google ADK callbacks and Runner, pytest, CoScientist YAML profiles.

## Global Constraints

- Change only `synapse_pilot`; other CoScientist profiles and Synapse stay unchanged.
- Do not invent a retrieved tool, MCP result, or scientific report.
- Do not claim the separate successful-MCP-receipt failure is fixed.
- Work in existing `feature/heracleum-report-demo` / draft PR #401; create no new branch or PR.

---

### Task 1: Exact-name retrieval on a skipped model call

**Files:**
- Create: `tests/unit/test_pilot_exact_retrieval.py`
- Create: `CoScientist/agents/callbacks/pilot_exact_retrieval.py`
- Modify: `CoScientist/assembly/bindings.py` (pilot callback registration)
- Modify: `CoScientist/agents/synapse_pilot.yaml` (`ToolRetrieverAgent` override)

**Interfaces:**
- Consumes `callback_context.user_content`, `callback_context.state["accumulated_tools"]`, `LlmResponse.partial` and `LlmResponse.content.parts`.
- Produces `ensure_pilot_exact_tool_retrieved(callback_context, llm_response) -> LlmResponse | None`; callback key `ensure_pilot_exact_tool_retrieved`.

- [ ] **Step 1: Write the first failing behavior test.** Create `tests/unit/test_pilot_exact_retrieval.py` with this first test:

```python
from types import SimpleNamespace

from google.adk.models import LlmResponse
from google.genai import types

from CoScientist.agents.callbacks.pilot_exact_retrieval import ensure_pilot_exact_tool_retrieved


def context(state=None, task="Target tool: dataset_overview_heracleum_tox (server_id=server-a)"):
    return SimpleNamespace(
        state={} if state is None else state,
        invocation_id="run-1",
        user_content=types.Content(parts=[types.Part(text=task)]),
    )


def final(text="Done"):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def test_missing_explicit_tool_requests_exact_retrieval():
    result = ensure_pilot_exact_tool_retrieved(context(), final())
    call = result.content.parts[0].function_call
    assert call.name == "retrieve_tools"
    assert dict(call.args) == {"query": "dataset_overview_heracleum_tox"}
```

- [ ] **Step 2: Verify red.** From the CoScientist worktree run:

```powershell
docker run --rm -v 'C:\Users\mrbas\.codex\worktrees\heracleum-report-demo\CoScientist:/workspace' -w /workspace -e PYTHONDONTWRITEBYTECODE=1 -e STORAGE__LOGGING_PATH=/tmp/cosci-test-logs -e LLM__MAIN_MODEL=gpt-4o-mini --entrypoint python coscientist-local-demo:local -m pytest -q -p no:cacheprovider tests/unit/test_pilot_exact_retrieval.py::test_missing_explicit_tool_requests_exact_retrieval
```

Expected: missing module/import, not a dependency or fixture error.

- [ ] **Step 3: Implement the callback and pilot binding.** Create `CoScientist/agents/callbacks/pilot_exact_retrieval.py`:

```python
import re

from google.adk.models import LlmResponse
from google.genai import types


_TARGET = re.compile(r"\bTarget tool:\s*([A-Za-z_][A-Za-z_0-9]*)\b")
_SERVER = re.compile(r"\bserver_id=([A-Za-z0-9_-]+)\b")
_ATTEMPT_KEY = "_pilot_exact_tool_attempt"


def ensure_pilot_exact_tool_retrieved(callback_context, llm_response):
    if llm_response.partial:
        return None
    parts = getattr(llm_response.content, "parts", None) or []
    if any(getattr(part, "function_call", None) for part in parts):
        return None
    content = getattr(callback_context, "user_content", None)
    task = "\n".join(
        part.text or "" for part in (getattr(content, "parts", None) or [])
    )
    target_match = _TARGET.search(task)
    if target_match is None:
        return None
    target = target_match.group(1)
    server_match = _SERVER.search(task)
    server_id = server_match.group(1) if server_match else None
    inventory = callback_context.state.get("accumulated_tools") or []
    if any(
        isinstance(row, dict)
        and row.get("tool") == target
        and row.get("server_id")
        and (server_id is None or str(row["server_id"]) == server_id)
        for row in inventory
    ):
        return None
    attempt = f"{getattr(callback_context, 'invocation_id', '')}:{target}"
    if callback_context.state.get(_ATTEMPT_KEY) == attempt:
        raise RuntimeError(f"Explicit target tool {target} was not retrieved after exact lookup")
    callback_context.state[_ATTEMPT_KEY] = attempt
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part.from_function_call(name="retrieve_tools", args={"query": target})
    ]))
```

Register with `_cb("ensure_pilot_exact_tool_retrieved", "after_model", func=ensure_pilot_exact_tool_retrieved)` in `CoScientist/assembly/bindings.py` beside the pilot callback registrations, importing from the new module. In `CoScientist/agents/synapse_pilot.yaml`, add:

```yaml
  ToolRetrieverAgent:
    callbacks:
      after_model: [ensure_pilot_exact_tool_retrieved]
```

- [ ] **Step 4: Verify green.** Run the Step 2 command again; expected: one PASS.

- [ ] **Step 5: Add and red-green edge behaviors, one at a time.** Add independently named tests using the `context` and `final` helpers above, each with a concrete assertion:

```python
def test_matching_tool_and_server_needs_no_extra_lookup():
    state = {"accumulated_tools": [{"tool": "dataset_overview_heracleum_tox", "server_id": "server-a"}]}
    assert ensure_pilot_exact_tool_retrieved(context(state), final()) is None


def test_same_name_on_wrong_server_still_requires_lookup():
    state = {"accumulated_tools": [{"tool": "dataset_overview_heracleum_tox", "server_id": "server-b"}]}
    assert ensure_pilot_exact_tool_retrieved(context(state), final()).content.parts[0].function_call.name == "retrieve_tools"


def test_no_explicit_target_does_not_change_response():
    assert ensure_pilot_exact_tool_retrieved(context(task="Find tools"), final()) is None


def test_second_missing_result_raises_instead_of_looping():
    ctx = context()
    ensure_pilot_exact_tool_retrieved(ctx, final())
    import pytest
    with pytest.raises(RuntimeError, match="after exact lookup"):
        ensure_pilot_exact_tool_retrieved(ctx, final())
```

Also add one test each for a missing `server_id`, `partial=True`, and a model-originated `function_call` using the same helpers. Run each newly introduced behavior red if code does not yet implement it, then green with the Step 2 command targeting the full new file.

- [ ] **Step 6: Add the ADK Runner regression.** Extend the new test file with this observable call/response test (and necessary imports). It uses a scripted model and a stubbed external retrieval function, not a mocked callback:

```python
import asyncio

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from pydantic import PrivateAttr


class FinalOnlyModel(BaseLlm):
    _calls: int = PrivateAttr(default=0)

    def __init__(self):
        super().__init__(model="scripted-final-only")

    async def generate_content_async(self, llm_request, stream=False):
        self._calls += 1
        yield final()


def test_adk_executes_exact_lookup_once_before_failing_closed():
    seen = []

    async def retrieve_tools(query: str) -> dict:
        seen.append(query)
        return {"status": "ok", "result": []}

    async def run():
        sessions = InMemorySessionService()
        await sessions.create_session(app_name="exact_retrieval", user_id="user", session_id="session")
        agent = LlmAgent(
            name="ToolRetrieverProbe", model=FinalOnlyModel(), instruction="Find tools.",
            tools=[retrieve_tools], after_model_callback=ensure_pilot_exact_tool_retrieved,
        )
        runner = Runner(agent=agent, app_name="exact_retrieval", session_service=sessions)
        events = []
        try:
            async for event in runner.run_async(
                user_id="user", session_id="session",
                new_message=types.Content(role="user", parts=[types.Part(text=(
                    "Target tool: dataset_overview_heracleum_tox (server_id=server-a)"
                ))]),
            ):
                events.append(event)
        except RuntimeError as exc:
            assert "after exact lookup" in str(exc)
        else:
            raise AssertionError("missing tool was accepted")
        return events, await sessions.get_session(
            app_name="exact_retrieval", user_id="user", session_id="session"
        )

    events, session = asyncio.run(run())
    assert seen == ["dataset_overview_heracleum_tox"]
    assert any(
        response.name == "retrieve_tools"
        for event in events for response in event.get_function_responses()
    )
    assert not session.state.get("accumulated_tools")
```

- [ ] **Step 7: Verify all relevant tests and static checks.** Repeat the Step 2 Docker command with the test path replaced by `tests/unit/test_pilot_exact_retrieval.py tests/unit/test_executor_redirect.py tests/test_synapse_pilot_unknown_tools.py tests/unit/experiment/test_profile.py tests/unit/test_assembly.py`. Run `git diff --check` and `python -m compileall -q CoScientist/agents/callbacks/pilot_exact_retrieval.py`. Expected: zero test failures and zero diff/compile errors.

- [ ] **Step 8: Commit and update PR #401.** Commit only the tested callback, binding, pilot profile, and tests. Push the existing branch; keep PR draft. Update the PR body with a sanitized statement of what changed, what was verified, and the unresolved receipt-accounting/full-report limitations—without local run IDs, private endpoints, credentials, or detailed internal traces.

### Task 2: Pilot verification without disturbing the working stand

**Files:** None unless a failing regression exposes a new, separately scoped defect.

**Interfaces:** Consumes the built pilot image and a targeted executor request; produces a recorded observation, not an automatic completion claim.

- [ ] **Step 1: Build a disposable image from the PR #401 branch.** Do not replace the live service until the image builds and the tests above pass. Expected: build exit code zero.

- [ ] **Step 2: Run one targeted executor call against the disposable image using the configured pilot endpoints.** Observe a real `retrieve_tools` call and response for the explicit target; if the retrieved inventory lacks the target, report the clear failure. Do not fabricate success or launch a full two-hour scientific project as part of this small fix.

- [ ] **Step 3: Report exact verified scope.** Distinguish unit/ADK tests, targeted live call, and the still-unverified full scientific report and MCP receipt accounting.
