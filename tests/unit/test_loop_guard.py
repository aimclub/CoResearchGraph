"""An agent must not be able to repeat one identical tool call forever."""
import asyncio
import types

from CoScientist.agents.loop_guard_plugin import RepeatCallGuardPlugin


def _call(guard, tool, args, agent="DatasetCollectorAgent"):
    return asyncio.run(guard.before_tool_callback(
        tool=types.SimpleNamespace(name=tool), tool_args=args,
        tool_context=types.SimpleNamespace(agent_name=agent)))


def test_identical_call_is_blocked_after_the_limit():
    guard = RepeatCallGuardPlugin()
    args = {"query": "same thing"}
    allowed = [_call(guard, "tavily_search", args) for _ in range(4)]
    blocked = _call(guard, "tavily_search", args)
    assert all(r is None for r in allowed)
    assert blocked and blocked["blocked_by"] == "repeat_call_guard"
    assert "Change approach" in blocked["message"]


def test_different_arguments_are_never_blocked():
    guard = RepeatCallGuardPlugin()
    for i in range(10):
        assert _call(guard, "tavily_search", {"query": f"q{i}"}) is None


def test_polling_the_same_job_is_exempt():
    """Waiting on a long job is the sanctioned pattern, not a loop."""
    guard = RepeatCallGuardPlugin()
    for _ in range(12):
        assert _call(guard, "check_job", {"job_id": "j1"}) is None


def test_counts_are_per_agent():
    guard = RepeatCallGuardPlugin()
    args = {"query": "x"}
    for _ in range(5):
        _call(guard, "tavily_search", args, agent="A")
    assert _call(guard, "tavily_search", args, agent="B") is None
