"""The meta-agent config recovery plugin.

Covers the dominant FEDOT failure mode on this stand: the model answers, but the
answer is not a clean JSON text part, so ADK never fills output_key and
_execute_meta_call raises "<agent> did not produce '<key>' in session state" —
82 of 88 config-generation failures measured on 2026-09-01.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from google.genai import types

from CoScientist.tools.fedot_mas_patch import MetaJsonRecoveryPlugin

CONFIG = {
    "agents": [{"name": "a1", "instruction": "run it", "output_key": "o1"}],
    "pipeline": {"type": "agent", "agent_name": "a1"},
}


class _MetaAgent:
    """A meta agent: ADK stores its answer against a schema."""

    name = "pool_generator"
    output_schema = object()


class _Worker:
    """A pipeline worker: free-text answer, nothing to store."""

    name = "interval_estimator"
    output_schema = None


def _ctx(agent):
    return SimpleNamespace(
        agent_name=agent.name,
        get_invocation_context=lambda: SimpleNamespace(agent=agent),
    )


META = _ctx(_MetaAgent())


def _response(parts):
    return SimpleNamespace(content=types.Content(role="model", parts=parts))


def _run(plugin, response, ctx=None):
    return asyncio.run(
        plugin.after_model_callback(callback_context=ctx or META, llm_response=response)
    )


def test_recovers_json_parked_in_a_thought_part():
    """GLM, the model configured for the meta agents, does exactly this."""
    plugin = MetaJsonRecoveryPlugin()
    out = _run(plugin, _response([
        types.Part(text="Let me design the pool.", thought=True),
        types.Part(text=json.dumps(CONFIG), thought=True),
    ]))
    assert out is not None
    assert json.loads(out.content.parts[0].text) == CONFIG
    assert plugin.repaired == ["pool_generator"]


def test_recovers_json_wrapped_in_a_fence_with_preamble():
    plugin = MetaJsonRecoveryPlugin()
    out = _run(plugin, _response([
        types.Part(text="Here is the pipeline:\n```json\n" + json.dumps(CONFIG) + "\n```\nDone."),
    ]))
    assert out is not None
    assert json.loads(out.content.parts[0].text) == CONFIG


def test_leaves_a_clean_answer_untouched():
    plugin = MetaJsonRecoveryPlugin()
    assert _run(plugin, _response([types.Part(text=json.dumps(CONFIG))])) is None
    assert plugin.repaired == []


def test_leaves_a_tool_call_untouched():
    plugin = MetaJsonRecoveryPlugin()
    call = types.Part.from_function_call(name="some_tool", args={})
    assert _run(plugin, _response([call])) is None


def test_invents_nothing_when_there_is_no_json():
    """A genuine refusal must still surface as the original error, not a fake config."""
    plugin = MetaJsonRecoveryPlugin()
    assert _run(plugin, _response([
        types.Part(text="I cannot design a pipeline for this task."),
    ])) is None
    assert plugin.repaired == []


def test_empty_response_is_a_pass_through():
    plugin = MetaJsonRecoveryPlugin()
    assert _run(plugin, SimpleNamespace(content=None)) is None


def test_leaves_a_pipeline_worker_alone():
    """Plugins reach the pipeline agents too, and a worker's prose must survive.

    Regression for 2026-09-02, when an unscoped version of this plugin rewrote
    worker answers — "rebuilt experiment_executor config from the text part" and
    "rebuilt interval_estimator config from the thought part" — replacing a prose
    report that merely contained JSON with the JSON alone, discarding the
    narrative the caller actually reads.
    """
    plugin = MetaJsonRecoveryPlugin()
    worker_answer = _response([
        types.Part(
            text="I ran the tool and got:\n```json\n"
            + json.dumps(CONFIG)
            + "\n```\nR2 for the test sample was 0.91."
        ),
    ])
    assert _run(plugin, worker_answer, ctx=_ctx(_Worker())) is None
    assert plugin.repaired == []
    # …while the same answer from a schema-bound meta agent is still recovered.
    assert _run(plugin, worker_answer) is not None


def test_unreachable_agent_is_left_alone():
    """Fail closed: no agent visible means no rewrite."""
    plugin = MetaJsonRecoveryPlugin()
    bare = SimpleNamespace(agent_name="x")
    prose = _response([types.Part(text='prose ```json\n{"a": 1}\n```')])
    assert _run(plugin, prose, ctx=bare) is None


def _truncated_response(parts):
    """A response the provider cut off at the output limit.

    Shaped the way LiteLLM delivers it: any non-STOP finish reason also sets
    error_code and error_message on the SAME response that carries the content
    (google/adk/models/lite_llm.py). Leaving those out would make the assertions
    below pass on a response that never had the flag in the first place.
    """
    return SimpleNamespace(
        content=types.Content(role="model", parts=parts),
        finish_reason=types.FinishReason.MAX_TOKENS,
        error_code=types.FinishReason.MAX_TOKENS,
        error_message="Maximum tokens reached",
    )


def test_a_truncated_config_says_so_in_the_log(caplog):
    """Nothing is recoverable from half a JSON object — but silence is worse.

    On 2026-09-17 routing_meta_agent was cut off at the output limit on both
    fedot_tool calls of EXP-3. The module fell back to react_tools, and
    logs/app.log carried not one mention of why: the reason survived only inside
    the TaskResult summary, where nobody watching a run would look.
    """
    plugin = MetaJsonRecoveryPlugin()
    half = json.dumps(CONFIG)[: len(json.dumps(CONFIG)) // 2]

    with caplog.at_level("WARNING"):
        out = _run(plugin, _truncated_response([types.Part(text=half)]))

    assert out is None, "a half-written config must not be passed off as recovered"
    assert plugin.repaired == []
    text = caplog.text
    assert "MAX_TOKENS" in text
    assert "FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS" in text, (
        "the log must name the knob that fixes it"
    )


def test_an_ordinary_unrecoverable_answer_stays_quiet_about_tokens(caplog):
    """Only truncation earns the token warning; prose with no JSON is a
    different failure and must not be blamed on the output limit."""
    plugin = MetaJsonRecoveryPlugin()

    with caplog.at_level("WARNING"):
        out = _run(plugin, _response([types.Part(text="I could not build a config.")]))

    assert out is None
    assert "FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS" not in caplog.text


def test_a_recoverable_answer_is_still_recovered_when_truncated():
    """Truncation that still left a complete object behind is recoverable, and
    the salvage must win over the warning."""
    plugin = MetaJsonRecoveryPlugin()

    out = _run(plugin, _truncated_response([
        types.Part(text="Here is the config:\n```json\n" + json.dumps(CONFIG) + "\n```\ntrailing"),
    ]))

    assert out is not None
    assert plugin.repaired == ["pool_generator"]


def test_a_recovered_config_does_not_still_carry_the_error_flag():
    """The repair has to survive the runner that reads it.

    LiteLLM sets error_code for ANY non-STOP finish reason, and
    fedotmas/meta/_adk_runner.py raises on any error_code at all — no MAX_TOKENS
    carve-out, unlike its own sibling fedotmas/core/runner.py. So a config we
    rebuilt in full still killed the run on the flag alone. Observed on
    2026-09-17: recovery fired inside two of the three failing fedot_tool
    windows of EXP-3 and the calls failed anyway.
    """
    plugin = MetaJsonRecoveryPlugin()
    response = _truncated_response([
        types.Part(text=json.dumps(CONFIG), thought=True),
    ])
    assert response.finish_reason == types.FinishReason.MAX_TOKENS

    out = _run(plugin, response)

    assert out is not None
    assert json.loads(out.content.parts[0].text) == CONFIG
    assert getattr(out, "error_code", None) is None, "the runner would raise on this"
    assert getattr(out, "error_message", None) is None
    # The original is untouched — the plugin works on a copy.
    assert response.finish_reason == types.FinishReason.MAX_TOKENS


def test_a_narrow_escape_is_reported_as_one(caplog):
    """A config that arrived complete but flagged MAX_TOKENS is one token away
    from the failure mode, and the operator should hear about it."""
    plugin = MetaJsonRecoveryPlugin()

    with caplog.at_level("WARNING"):
        _run(plugin, _truncated_response([types.Part(text=json.dumps(CONFIG), thought=True)]))

    assert "FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS" in caplog.text
