"""The planner's own critic: the LLM verdict and the one-round review loop.

The critic is wired separately from the orchestrator's pre/post-action critics
(system.yaml -> PlannerAgent.critic) and runs inside SessionAgent's review loop,
so what has to hold is: an objection makes the planner replan, and it can only
ever do that ONCE per run — a self-critique loop with no budget never stops.

Run from the repo root:  pytest tests/unit/test_plan_critic.py -q
"""
import asyncio
from typing import AsyncGenerator, List

import pytest
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.genai import types

from CoScientist.agents.callbacks import critic as critic_module
from CoScientist.agents.callbacks import make_plan_critique
from CoScientist.hitl.session_agent import SessionAgent


# ── the critic verdict ───────────────────────────────────────────────────────

def _stub_llm(monkeypatch, payload):
    async def fake_invoke(system_prompt, user_prompt):
        fake_invoke.prompts.append(user_prompt)
        return payload

    fake_invoke.prompts = []
    monkeypatch.setattr(critic_module, "_invoke_critic_llm", fake_invoke)
    return fake_invoke


def test_approved_plan_returns_no_feedback(monkeypatch):
    _stub_llm(monkeypatch, {"verdict": "approve", "feedback": ""})
    critique = make_plan_critique("SYSTEM")

    assert asyncio.run(critique("task", "1. Do the thing")) is None


def test_revise_returns_the_feedback_and_sees_task_and_plan(monkeypatch):
    stub = _stub_llm(
        monkeypatch,
        {"verdict": "revise", "feedback": "TASK-2 has no assignee on the roster."},
    )
    critique = make_plan_critique("SYSTEM")

    feedback = asyncio.run(critique("Find inhibitors", "1. Search\n2. Compute"))

    assert feedback == "TASK-2 has no assignee on the roster."
    assert "Find inhibitors" in stub.prompts[0]
    assert "1. Search" in stub.prompts[0]


def test_revise_without_feedback_accepts_the_plan(monkeypatch):
    """An objection the critic cannot put into words costs a replan for nothing."""
    _stub_llm(monkeypatch, {"verdict": "revise", "feedback": "  "})
    critique = make_plan_critique("SYSTEM")

    assert asyncio.run(critique("task", "plan")) is None


def test_unparseable_verdict_accepts_the_plan(monkeypatch):
    _stub_llm(monkeypatch, {})
    critique = make_plan_critique("SYSTEM")

    assert asyncio.run(critique("task", "plan")) is None


# ── the review loop ──────────────────────────────────────────────────────────

class _FakeSession:
    def __init__(self):
        self.state = {}
        self.events: List[Event] = []


class _FakeSessionService:
    """Applies state_delta the way a real session service does."""

    async def append_event(self, session, event):
        delta = getattr(event.actions, "state_delta", None) or {}
        session.state.update(delta)
        session.events.append(event)


class _FakeContext:
    def __init__(self, task: str):
        self.session = _FakeSession()
        self.session_service = _FakeSessionService()
        self.invocation_id = "inv-1"
        self.branch = None
        self.user_content = types.Content(role="user", parts=[types.Part(text=task)])
        self.agent_states = []

    def set_agent_state(self, name):
        self.agent_states.append(name)


def _plan(title: str):
    return [
        {"id": "TASK-1", "title": title, "description": "d", "assignee": "ResearchAgent"}
    ]


@pytest.fixture
def planner_runs(monkeypatch):
    """Stub the underlying LlmAgent: each run registers the next plan in the list."""
    def _install(plans):
        remaining = list(plans)
        calls = []

        async def fake_run(self, ctx) -> AsyncGenerator[Event, None]:
            title = remaining.pop(0)
            calls.append(title)
            ctx.session.state["active_tasks"] = _plan(title)
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(
                    role="model", parts=[types.Part(text=f"registered {title}")]
                ),
            )

        monkeypatch.setattr(LlmAgent, "_run_async_impl", fake_run)
        return calls

    return _install


def _planner(critic, **kwargs) -> SessionAgent:
    return SessionAgent(
        name="PlannerAgent", model="stub-model", plan_critic=critic, **kwargs
    )


def _run(agent, ctx) -> List[Event]:
    async def collect():
        return [event async for event in agent._run_async_impl(ctx)]

    return asyncio.run(collect())


def _texts(events):
    return [
        "".join(p.text or "" for p in (e.content.parts if e.content else []))
        for e in events
    ]


def test_critic_objection_makes_the_planner_replan_once(planner_runs):
    calls = planner_runs(["first plan", "second plan"])
    seen = []

    async def critic(task, plan):
        seen.append((task, plan))
        return "TASK-1 misses the deliverable." if len(seen) == 1 else None

    ctx = _FakeContext("Find inhibitors")
    events = _run(_planner(critic), ctx)

    # Exactly two planning runs, and the critic reviewed only the first — its
    # budget is spent, so the rewrite is accepted unreviewed.
    assert calls == ["first plan", "second plan"]
    assert len(seen) == 1
    assert seen[0][0] == "Find inhibitors"
    assert "first plan" in seen[0][1]  # the REGISTERED plan, not the narration

    # The discarded plan never reaches the chat; the feedback turn and the
    # accepted rewrite do.
    texts = _texts(events)
    assert not any("registered first plan" in t for t in texts)
    assert any("TASK-1 misses the deliverable." in t for t in texts)
    assert texts[-1] == "registered second plan"
    assert ctx.session.state["active_tasks"] == _plan("second plan")


def test_a_critic_that_never_approves_still_stops_after_one_round(planner_runs):
    calls = planner_runs(["first plan", "second plan"])
    rounds = []

    async def critic(task, plan):
        rounds.append(plan)
        return "still not good enough"

    _run(_planner(critic), _FakeContext("task"))

    assert calls == ["first plan", "second plan"]
    assert len(rounds) == 1


def test_approved_plan_runs_the_planner_once(planner_runs):
    calls = planner_runs(["only plan"])

    async def critic(task, plan):
        return None

    events = _run(_planner(critic), _FakeContext("task"))

    assert calls == ["only plan"]
    assert _texts(events) == ["registered only plan"]


def test_no_critic_wired_leaves_the_loop_untouched(planner_runs):
    calls = planner_runs(["only plan"])

    events = _run(_planner(None), _FakeContext("task"))

    assert calls == ["only plan"]
    assert _texts(events) == ["registered only plan"]


def test_failing_critic_accepts_the_plan_instead_of_killing_the_run(planner_runs):
    calls = planner_runs(["only plan"])

    async def critic(task, plan):
        raise RuntimeError("provider down")

    events = _run(_planner(critic), _FakeContext("task"))

    assert calls == ["only plan"]
    assert _texts(events) == ["registered only plan"]


def test_critic_round_budget_is_configurable(planner_runs):
    """`critic_max_rounds` is the knob the web UI writes; 1 is the default."""
    calls = planner_runs(["p1", "p2", "p3"])

    async def critic(task, plan):
        return "again"

    _run(_planner(critic, critic_max_rounds=2), _FakeContext("task"))

    assert calls == ["p1", "p2", "p3"]


# ── the re-registration guard ────────────────────────────────────────────────
#
# `create_plan` normalises what it is given (renumbers ids, merges adjacent
# steps with the same executor assignee), and the planner prompt tells the model
# to check the plan it gets back. A critic asking for "a separate analysis step"
# next to an existing executor step therefore sets up an unwinnable chase: the
# merge happens again on every rewrite. The guard ends the turn instead.

def _model_response(*parts):
    from google.adk.models import LlmResponse

    return LlmResponse(content=types.Content(role="model", parts=list(parts)))


def _create_plan_call():
    return types.Part(
        function_call=types.FunctionCall(name="create_plan", args={"tasks": []})
    )


class _FakeCallbackContext:
    def __init__(self, state):
        self.state = state
        self.agent_name = "PlannerAgent"


def _guard():
    from CoScientist.agents.callbacks import make_plan_registration_guard

    return make_plan_registration_guard()


def test_the_first_registration_of_a_run_goes_through():
    ctx = _FakeCallbackContext({})  # SessionAgent cleared active_tasks

    assert _guard()(ctx, _model_response(_create_plan_call())) is None


def test_a_retry_after_a_rejected_create_plan_still_goes_through():
    """A create_plan that errored registers nothing, so the retry is not a loop."""
    ctx = _FakeCallbackContext({"active_tasks": []})

    assert _guard()(ctx, _model_response(_create_plan_call())) is None


def test_re_registering_a_registered_plan_ends_the_turn():
    ctx = _FakeCallbackContext(
        {"active_tasks": [{"title": "Train the model", "assignee": "TaskExecutorAgent"}]}
    )

    response = _guard()(ctx, _model_response(_create_plan_call()))

    # A text response with no function call is a final response, so the LLM
    # flow stops looping instead of running the tool again.
    assert response is not None
    assert not response.content.parts[0].function_call
    text = response.content.parts[0].text
    assert "Train the model" in text and "TaskExecutorAgent" in text


def test_other_calls_are_untouched_once_the_plan_is_registered():
    ctx = _FakeCallbackContext({"active_tasks": [{"title": "t", "assignee": "a"}]})
    other = types.Part(
        function_call=types.FunctionCall(name="retrieve_tools", args={"query": "x"})
    )

    assert _guard()(ctx, _model_response(other)) is None
    assert _guard()(ctx, _model_response(types.Part(text="done"))) is None


# ── end to end, through the real ADK flow ────────────────────────────────────
#
# The two defects this file exists for were both invisible to the stubbed loop
# above: the planner looping on create_plan, and the feedback reaching the model
# twice. Both only show up once a real LlmAgent flow, a real tool and a real
# Runner are in play, so this drives all three with a scripted model.

class _ScriptedLlm(BaseLlm):
    """Registers a plan, and re-registers whenever the plan it gets back is not
    the one it sent — which is what create_plan's merge guarantees."""

    calls: int = 0
    feedback_seen_per_call: list = []
    runaway_at: int = 12

    async def generate_content_async(self, llm_request, stream: bool = False):
        self.calls += 1
        if self.calls > self.runaway_at:
            raise AssertionError(f"model called {self.calls}x — runaway loop")

        contents = llm_request.contents or []
        texts = [
            p.text or "" for c in contents for p in (c.parts or [])
            if getattr(p, "text", None)
        ]
        # What the model ACTUALLY receives, not what the session stored.
        seen = sum(1 for t in texts if "plan critic reviewed" in t)
        self.feedback_seen_per_call.append(seen)

        responses = [
            p.function_response for c in contents for p in (c.parts or [])
            if getattr(p, "function_response", None)
        ]
        if responses:
            registered = responses[-1].response.get("plan") or []
            if seen and len(registered) < 2:  # the analysis step was merged away
                yield _plan_call(_revised_tasks())
                return
            yield _model_text("Plan registered.")
            return

        yield _plan_call(_revised_tasks() if seen else _initial_tasks())


def _initial_tasks():
    return [{"id": "TASK-1", "title": "Assess datasets", "description": "d",
             "assignee": "HypothesesAgent", "parent_id": None}]


def _revised_tasks():
    """Two adjacent executor tasks — create_plan merges them into one."""
    return [
        {"id": "TASK-1", "title": "Analyse overlap", "description": "explicit step",
         "assignee": "TaskExecutorAgent", "parent_id": None},
        {"id": "TASK-2", "title": "Train", "description": "train",
         "assignee": "TaskExecutorAgent", "parent_id": "TASK-1"},
    ]


def _plan_call(tasks):
    from google.adk.models import LlmResponse

    return LlmResponse(content=types.Content(role="model", parts=[types.Part(
        function_call=types.FunctionCall(
            id="fc-1", name="create_plan", args={"tasks": tasks})
    )]))


def _model_text(text):
    from google.adk.models import LlmResponse

    return LlmResponse(content=types.Content(
        role="model", parts=[types.Part(text=text)]))


def _run_real_planner():
    from google.adk.runners import InMemoryRunner
    from CoScientist.agents.callbacks import make_plan_registration_guard
    from CoScientist.tools.task_tracker import create_plan_tool

    async def critic(task, plan):
        if not critic.done:
            critic.done = True
            return "Step 1 needs an executor; the analysis must be explicit."
        return None

    critic.done = False
    model = _ScriptedLlm(model="scripted", feedback_seen_per_call=[])
    agent = SessionAgent(
        name="PlannerAgent",
        model=model,
        instruction="Register the plan with create_plan.",
        tools=[create_plan_tool()],
        output_key="planner_roadmap",
        plan_critic=critic,
        after_model_callback=make_plan_registration_guard(),
    )

    async def drive():
        runner = InMemoryRunner(agent=agent, app_name="t")
        session = await runner.session_service.create_session(app_name="t", user_id="u")
        async for _ in runner.run_async(
            user_id="u", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="Plan it")]),
        ):
            pass
        return await runner.session_service.get_session(
            app_name="t", user_id="u", session_id=session.id
        )

    return model, asyncio.run(drive())


def test_the_planner_finishes_its_turn_after_a_critic_revision():
    """Without the registration guard this never terminates: the critic asks for
    a separate step, create_plan merges it back, and the planner re-registers
    forever chasing a difference the tracker itself creates."""
    model, session = _run_real_planner()

    assert model.calls <= model.runaway_at
    assert session.state.get("active_tasks"), "the revised plan must be registered"


def test_the_critic_feedback_reaches_the_model_exactly_once():
    """Yielding the feedback event is what puts it in the session; appending it
    by hand as well showed the planner the same instruction twice."""
    model, _ = _run_real_planner()

    assert max(model.feedback_seen_per_call) == 1, model.feedback_seen_per_call


def test_feedback_still_reaches_a_run_whose_consumer_writes_no_session(planner_runs):
    """A bare run_async (no Runner) has nothing appending events — the agent
    must put the feedback in itself rather than silently re-answer unchanged."""
    planner_runs(["first plan", "second plan"])

    async def objects_once(task, plan):
        return "fix it"  # the round budget stops the loop after one

    ctx = _FakeContext("task")
    _run(_planner(objects_once), ctx)

    feedback = [
        e for e in ctx.session.events
        for p in (e.content.parts if e.content else [])
        if getattr(p, "text", None) and "plan critic reviewed" in p.text
    ]
    assert len(feedback) == 1


# ── the web settings surface ─────────────────────────────────────────────────

@pytest.fixture
def web_settings(monkeypatch):
    """The runtime settings the UI writes, restored after the test."""
    from CoScientist.config import get_settings

    web = get_settings().web
    for field in ("planner_critic_enabled", "planner_critic_rounds"):
        monkeypatch.setattr(web, field, getattr(web, field))
    return web


def test_web_ui_settings_round_trip(web_settings):
    from CoScientist.web.app import _apply_frontend_settings, _settings_payload

    _apply_frontend_settings(
        {"plannerAgent": {"criticEnabled": True, "criticRounds": 3}}
    )

    assert web_settings.planner_critic_enabled is True
    assert web_settings.planner_critic_rounds == 3
    planner = _settings_payload()["plannerAgent"]
    assert planner["criticEnabled"] is True
    assert planner["criticRounds"] == 3


def test_web_ui_rejects_a_budget_below_one_round(web_settings):
    """A zero-round critic is one that never runs — that is the switch's job."""
    from CoScientist.web.app import _apply_frontend_settings

    _apply_frontend_settings({"plannerAgent": {"criticRounds": 2}})
    _apply_frontend_settings({"plannerAgent": {"criticRounds": 0}})

    assert web_settings.planner_critic_rounds == 2
