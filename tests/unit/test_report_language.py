"""Report language: the per-session parameter that drives the final report.

The language reaches the Result Aggregator as ADK session state, not as a
build-time constant, so these tests cover the two halves of that path: the
callback that renders the block, and the prompt that carries the placeholder.
"""
import re

from CoScientist.agents.callbacks.report_language import (
    DEFAULT_REPORT_LANGUAGE,
    REPORT_LANGUAGE_BLOCK_STATE_KEY,
    REPORT_LANGUAGE_STATE_KEY,
    inject_report_language,
    normalize_report_language,
)

CYRILLIC = re.compile(r"[Ѐ-ӿ]")


class _FakeContext:
    """Stand-in for ADK's CallbackContext — the callback only touches .state."""

    def __init__(self, state=None):
        self.state = dict(state or {})


def test_normalize_falls_back_for_anything_unsupported():
    assert normalize_report_language("en") == "en"
    assert normalize_report_language(" RU ") == "ru"
    # A missing key, an empty value, and an unsupported code all collapse to the
    # default, so the CLI and A2A paths keep the behavior they had before.
    for bad in (None, "", "   ", "de", 7):
        assert normalize_report_language(bad) == DEFAULT_REPORT_LANGUAGE


def test_callback_renders_the_matching_block_and_returns_none():
    for value, expected_marker in (
        (None, "report in Russian"),
        ("ru", "report in Russian"),
        ("en", "report in English"),
        ("de", "report in Russian"),
    ):
        state = {} if value is None else {REPORT_LANGUAGE_STATE_KEY: value}
        context = _FakeContext(state)
        # None keeps the agent's other before_agent callbacks running — ADK
        # short-circuits its callback list on the first non-None result.
        assert inject_report_language(context) is None
        assert expected_marker in context.state[REPORT_LANGUAGE_BLOCK_STATE_KEY]


def test_english_block_keeps_the_tool_generated_headings():
    """format_results builds "## Figures" in code; English must not rewrite it."""
    context = _FakeContext({REPORT_LANGUAGE_STATE_KEY: "en"})
    inject_report_language(context)
    block = context.state[REPORT_LANGUAGE_BLOCK_STATE_KEY]
    assert "## Objective" in block
    assert not CYRILLIC.search(block)

    context = _FakeContext({REPORT_LANGUAGE_STATE_KEY: "ru"})
    inject_report_language(context)
    russian = context.state[REPORT_LANGUAGE_BLOCK_STATE_KEY]
    assert "## Иллюстрации" in russian
    assert "## Цель" in russian


def test_callback_composes_with_the_research_context_callback():
    """Both before_agent callbacks write to the SAME state, one after the other.

    The aggregator runs inject_research_context then inject_report_language. If
    either replaced .state instead of mutating it, the second write would drop
    the first key and one of the prompt's two placeholders would render empty.
    """
    from CoScientist.graph.research.agent_tools import make_inject_research_context

    context = _FakeContext({REPORT_LANGUAGE_STATE_KEY: "en"})
    assert make_inject_research_context(is_root=False)(context) is None
    assert inject_report_language(context) is None
    assert "research_context" in context.state
    assert "report in English" in context.state[REPORT_LANGUAGE_BLOCK_STATE_KEY]


def test_aggregator_prompt_is_language_neutral():
    """The regression guard: no language may be hardcoded back into the prompt.

    The prompt must also keep its two state placeholders. A bare ``{key}``
    without the ``?`` would raise KeyError mid-run on a session that predates
    the key.
    """
    from CoScientist.agents import result_aggregator_agent

    instruction = result_aggregator_agent.instruction
    assert "{report_language_block?}" in instruction
    assert "{research_context?}" in instruction
    assert not CYRILLIC.search(instruction)
    # A leftover build-time sentinel would mean render_template missed a value.
    assert not re.search(r"<<[A-Z_]+>>", instruction)
