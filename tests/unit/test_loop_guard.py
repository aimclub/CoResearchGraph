"""An agent must not be able to repeat one identical tool call forever."""
import asyncio
import types

from CoScientist.agents.loop_guard_plugin import RepeatCallGuardPlugin
from CoScientist.agents.callbacks.tool_callbacks import TavilySearchLimiter


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


def test_tavily_fallback_budget_is_independent_per_agent():
    limiter = TavilySearchLimiter(max_searches=2)
    state = {}

    def call(agent, tool="tavily_search"):
        return limiter.limit_searches(
            types.SimpleNamespace(name=tool), {},
            types.SimpleNamespace(agent_name=agent, state=state),
        )

    assert call("ResearchAgent") is None
    assert call("ResearchAgent") is None
    assert call("ResearchAgent") is not None
    assert call("EconomicsAgent") is None
    assert call("ReactorAgent") is None
    assert call("EconomicsAgent", "search_by_structure") is None


def test_every_coder_tool_survives_repetition():
    """The coder's tools act on a workspace, so identical args are not a loop.

    The guard compares (agent, tool, args). For a search that triple is the
    whole question, and asking it again is thrash. For `execute_bash` the
    filesystem is half the input: the same test command after an edit is a
    different call with the same arguments, and blocking it told the agent to
    change approach when repeating the call WAS the approach.
    """
    from CoScientist.agents.loop_guard_plugin import CODER_TOOLS

    for tool in sorted(CODER_TOOLS):
        guard = RepeatCallGuardPlugin()
        results = [_call(guard, tool, {"cmd": "pytest -q"}, agent="CoderAgent")
                   for _ in range(12)]
        assert all(r is None for r in results), f"{tool} was blocked"


def test_a_prefixed_coder_tool_is_exempt_too():
    """A toolset hands its tools over as `{prefix}_{name}`, and the guard sees that."""
    guard = RepeatCallGuardPlugin()
    results = [_call(guard, "coder_execute_bash", {"cmd": "ls"}, agent="CoderAgent")
               for _ in range(12)]
    assert all(r is None for r in results)


def test_the_exemption_did_not_swallow_the_guard():
    """Everything else is still counted — the point of the plugin stands."""
    guard = RepeatCallGuardPlugin()
    args = {"query": "same thing"}
    for _ in range(4):
        assert _call(guard, "tavily_search", args) is None
    assert _call(guard, "tavily_search", args) is not None


def test_the_exempt_names_are_the_ones_the_toolsets_actually_expose():
    """A renamed tool would silently fall back under the guard."""
    import inspect

    from CoScientist.agents.loop_guard_plugin import CODER_TOOLS, EXEMPT_TOOLS
    from CoScientist.tools.coder_tools.coder_tools import CoderToolset
    from CoScientist.tools.coder_tools import sandbox_tools

    toolset = CoderToolset()
    live = {name for name in dir(toolset)
            if not name.startswith("_")
            and inspect.iscoroutinefunction(getattr(toolset, name, None))}
    # Plumbing, not tools the model calls.
    live -= {"close", "get_tools_with_prefix", "process_llm_request", "get_tools"}
    live |= {f.__name__ for f in sandbox_tools.get_sandbox_tools()}

    missing = live - EXEMPT_TOOLS
    assert not missing, f"coder tools left under the guard: {sorted(missing)}"
    stale = CODER_TOOLS - live
    assert not stale, f"exempted names no toolset exposes: {sorted(stale)}"
