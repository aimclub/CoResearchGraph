"""Assembly-level tests for the hypothesis subsystem integration.

These build the real system from CoScientist/agents/system.yaml (no LLM calls)
and pin the two integration bugs that used to live at the assembler boundary:

1. ``before_agent_callback`` arriving as a LIST (two callbacks in system.yaml)
   used to be invoked as a single callable by the hand-rolled composer inside
   ``HypothesisSubsystemAgent`` -> TypeError on first delegation.
2. ``guard_unknown_tools`` used to have an empty whitelist because the internal
   tools were built inside the class, invisible to the assembler -> every real
   tool call was rejected.

Both are now fixed by declaring the internal tools in system.yaml and
normalizing the callback list into ADK's canonical form.

Run from the repo root:  pytest tests/unit/test_hypothesis_assembly.py -q
"""
from dotenv import load_dotenv

load_dotenv()

from google.adk.models import LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402

from CoScientist.assembly import build_system  # noqa: E402
from CoScientist.assembly.schema import get_config  # noqa: E402


class _CallbackContext:
    """Minimal stand-in for CallbackContext — the guard only reads agent_name."""

    agent_name = "HypothesesAgent"


def _llm_response_with_tool(name: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(name=name, args={}))],
        )
    )


def test_hypotheses_agent_assembles_with_internal_tools_and_callbacks():
    """The assembler wires the hypothesis subsystem's internal tools and
    normalizes its before_agent callbacks into ADK's canonical list form."""
    system = build_system(get_config())
    agent = system.agent("HypothesesAgent")

    # 1. The internal strategy tools are attached under their real ADK names
    #    (FunctionTool derives its name from the wrapped function __name__).
    tool_names = sorted(t.name for t in agent.tools)
    assert tool_names == ["generate_via_moosechem", "run_critic_loop"]

    # 2. before_agent_callback must be a list (not a single wrapper that tries
    #    to call a list) so ADK runs the chain: inject_state FIRST, then the two
    #    system.yaml callbacks (before_get_task, inject_research_context).
    assert isinstance(agent.before_agent_callback, list)
    callback_names = [
        getattr(c, "__name__", type(c).__name__)
        for c in agent.canonical_before_agent_callbacks
    ]
    assert callback_names[0] == "inject_state"
    assert callback_names[1:] == ["before_get_task", "inject_research_context"]


def test_guard_unknown_tools_accepts_internal_tools():
    """The after_model guard's whitelist is populated from the declared tools,
    so legitimate internal tool calls pass and only unknown tools are caught."""
    system = build_system(get_config())
    agent = system.agent("HypothesesAgent")

    guard = agent.after_model_callback
    assert guard is not None

    # A real internal tool must NOT be intercepted.
    for name in ("generate_via_moosechem", "run_critic_loop"):
        assert guard(_CallbackContext(), _llm_response_with_tool(name)) is None

    # An unknown tool must be caught and replaced with a corrective response.
    caught = guard(_CallbackContext(), _llm_response_with_tool("bogus_tool"))
    assert caught is not None
    assert "bogus_tool" in caught.content.parts[0].text
