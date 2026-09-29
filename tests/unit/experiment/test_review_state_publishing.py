"""The review agent's decisions have to survive the AgentTool boundary.

The experiment module is attached to the orchestrator as an ADK ``AgentTool``.
AgentTool runs it against a FRESH ``InMemorySessionService`` session seeded from
a copy of the caller's state, and forwards exactly one thing back — the
``state_delta`` on the events the module yields:

    async for event in agen:
        if event.actions.state_delta:
            tool_context.state.update(event.actions.state_delta)

``SessionAgent`` writes through ``ctx.session.state``, a plain dict with no
delta tracking, so without an explicit publish the review's decisions die with
the throwaway child session. Callbacks are not affected — ``CallbackContext``
hands out a real ADK ``State`` — and that asymmetry is the whole defect: the
planner context builder's ``experiment_runtime = None`` wipe reached the
caller, the approval that followed it did not, and the next module hop replanned
and re-ran the same experiment from scratch.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.adk.events.event import Event

from CoScientist.experiments.review import (
    _REVIEW_OWNED_STATE_KEYS,
    ExperimentReviewSessionAgent,
)


def _ctx(state: dict) -> SimpleNamespace:
    return SimpleNamespace(
        session=SimpleNamespace(state=state), invocation_id="inv-1", branch=None,
    )


def _agent(kind: str = "result") -> ExperimentReviewSessionAgent:
    return ExperimentReviewSessionAgent(name="Reviewer", review_kind=kind)


def _event() -> Event:
    return Event(invocation_id="inv-1", author="Reviewer")


def _forward(delta: dict, caller: dict) -> None:
    """What AgentTool does with a yielded event's delta."""
    caller.update(delta)


def test_completed_phase_reaches_the_caller():
    """The exact loop: caller keeps runtime=None unless the phase is published."""
    caller = {"experiment_runtime": None}          # what the context builder wiped
    child = dict(caller)                            # AgentTool seeds a copy
    child["experiment_runtime"] = {"phase": "completed", "plan_id": "PLAN-1"}

    event = _event()
    _agent()._publish_state(_ctx(child), event)
    _forward(event.actions.state_delta, caller)

    assert caller["experiment_runtime"]["phase"] == "completed"


def test_nothing_published_means_the_caller_still_sees_the_wipe():
    """Pin the pre-fix behaviour, so a regression is unambiguous."""
    caller = {"experiment_runtime": None}
    child = dict(caller)
    child["experiment_runtime"] = {"phase": "completed"}

    _forward(_event().actions.state_delta, caller)  # a bare event carries nothing

    assert caller["experiment_runtime"] is None


def test_publishes_every_key_the_review_owns():
    state = {key: f"value-of-{key}" for key in _REVIEW_OWNED_STATE_KEYS}
    state["unrelated_key"] = "not ours"

    event = _event()
    _agent()._publish_state(_ctx(state), event)

    assert set(event.actions.state_delta) == set(_REVIEW_OWNED_STATE_KEYS)
    assert "unrelated_key" not in event.actions.state_delta


def test_absent_keys_are_not_invented():
    """Publishing a key the child never had would push a null over the caller's value."""
    event = _event()
    _agent()._publish_state(_ctx({"experiment_runtime": {"phase": "completed"}}), event)

    assert set(event.actions.state_delta) == {"experiment_runtime"}


def test_a_key_adk_already_published_wins():
    """output_key is filled by the model turn; our stale copy must not clobber it."""
    state = {"experiment_summary": "stale summary from the previous run"}
    event = _event()
    event.actions.state_delta["experiment_summary"] = "the summary just written"

    _agent()._publish_state(_ctx(state), event)

    assert event.actions.state_delta["experiment_summary"] == "the summary just written"


def test_replan_budget_is_published():
    """The budget is what stops the loop; it has to cross the boundary too."""
    from CoScientist.experiments.runtime.state_machine import REPLAN_ROUNDS_KEY

    caller: dict = {}
    child = {REPLAN_ROUNDS_KEY: 1, "experiment_runtime": {"phase": "completed"}}

    event = _event()
    _agent()._publish_state(_ctx(child), event)
    _forward(event.actions.state_delta, caller)

    assert caller[REPLAN_ROUNDS_KEY] == 1


def test_publishing_clears_the_pending_flag():
    """One decision publishes once; every later event stays empty."""
    agent = _agent()
    agent._state_publish_pending = True
    agent._publish_state(_ctx({"experiment_runtime": {"phase": "completed"}}), _event())
    assert agent._state_publish_pending is False


@pytest.mark.parametrize("kind", ["plan", "result"])
def test_review_decision_arms_the_publish_even_when_it_raises(kind):
    """Arming happens in a finally, so a decision that half-ran is not skipped.

    This covers arming only. If the exception escapes the agent's generator no
    event is yielded and nothing is delivered — by then the invocation is
    failing anyway. What matters is that the flag is not tied to the success
    path, so a review that pauses or returns early still publishes.
    """
    import asyncio

    agent = _agent(kind)

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("critique exploded")

    agent._review_plan = _boom       # type: ignore[method-assign]
    agent._review_result = _boom     # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        asyncio.run(agent._review_decision(_ctx({}), "{}"))
    assert agent._state_publish_pending is True
