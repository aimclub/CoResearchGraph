"""Cost accounting across the agent tree, and the sandbox's own bill.

Two things are being pinned down here. First, that every model call lands on
the agent that made it and on the session that owns the run — including calls
made inside an ``AgentTool`` sub-runner, whose child session must not become a
separate line item. Second, that the sandbox's metrics travel on their own
channel: into the ledger, never into the dict the model reads back.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from CoScientist.graph.session_scope import (
    GRAPH_SCOPE_SESSION_KEY,
    GRAPH_SCOPE_USER_KEY,
)
from CoScientist.logging import metrics
from CoScientist.tools.coder_tools import openhands_sandbox as client

KEY = ("user_1", "session_1")
PRICED = "gpt-4o"          # in litellm's table
UNPRICED = "vendor/model-that-does-not-exist"


@pytest.fixture(autouse=True)
def clean_ledger():
    metrics.LEDGER.reset(KEY)
    metrics.set_metrics_sink(None)
    yield
    metrics.LEDGER.reset(KEY)
    metrics.set_metrics_sink(None)


def model_context(agent_name="ResearchAgent", invocation_id="inv_1"):
    """A sub-agent callback context as AgentTool builds it: child session,
    parent state (that pin is what keeps the child out of its own bucket)."""
    return SimpleNamespace(
        agent_name=agent_name,
        invocation_id=invocation_id,
        state={
            GRAPH_SCOPE_USER_KEY: KEY[0],
            GRAPH_SCOPE_SESSION_KEY: KEY[1],
        },
        session=SimpleNamespace(user_id="child", id="child_session_xyz"),
    )


def usage(prompt=1000, completion=100, cached=0, total=None):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=completion,
        cached_content_token_count=cached,
        thoughts_token_count=0,
        total_token_count=total if total is not None else prompt + completion,
    )


def response(**kwargs):
    return SimpleNamespace(
        usage_metadata=kwargs.pop("usage_metadata", usage()),
        model_version=kwargs.pop("model_version", PRICED),
        partial=kwargs.pop("partial", False),
        **kwargs,
    )


def run_call(plugin, context, llm_response, model=PRICED):
    async def scenario():
        await plugin.before_model_callback(
            callback_context=context, llm_request=SimpleNamespace(model=model),
        )
        await plugin.after_model_callback(
            callback_context=context, llm_response=llm_response,
        )

    asyncio.run(scenario())


# ── per-agent, per-session accounting ────────────────────────────────────────

def test_subagent_calls_are_billed_to_the_subagent_and_the_parent_session():
    plugin = metrics.UsageMetricsPlugin()
    run_call(plugin, model_context("ResearchAgent"), response())
    run_call(plugin, model_context("CoderAgent"), response())
    run_call(plugin, model_context("CoderAgent"), response())

    data = metrics.LEDGER.snapshot(KEY)
    by_agent = {agent["agent"]: agent for agent in data["agents"]}

    assert set(by_agent) == {"ResearchAgent", "CoderAgent"}
    assert by_agent["CoderAgent"]["llm"]["calls"] == 2
    assert by_agent["ResearchAgent"]["llm"]["calls"] == 1
    assert data["llm"]["calls"] == 3
    assert data["session"] == {"user_id": KEY[0], "session_id": KEY[1]}


def test_tokens_and_money_come_from_the_response():
    plugin = metrics.UsageMetricsPlugin()
    run_call(plugin, model_context(), response(
        usage_metadata=usage(prompt=1000, completion=100, cached=800),
    ))

    llm = metrics.LEDGER.snapshot(KEY)["llm"]
    assert llm["prompt_tokens"] == 1000
    assert llm["completion_tokens"] == 100
    assert llm["cached_tokens"] == 800
    # 200 fresh + 800 cached (half price) input, plus output — cached input is
    # cheaper, so the total must be below the undiscounted price.
    assert 0 < llm["cost_usd"] < metrics.price_call(PRICED, 1000, 100)[0]


def test_streamed_partials_are_not_counted():
    plugin = metrics.UsageMetricsPlugin()
    context = model_context()

    async def scenario():
        await plugin.before_model_callback(
            callback_context=context, llm_request=SimpleNamespace(model=PRICED),
        )
        # Chunks carry no usage; litellm attaches it once, to the aggregate.
        for _ in range(5):
            await plugin.after_model_callback(
                callback_context=context,
                llm_response=response(usage_metadata=None, partial=True),
            )
        await plugin.after_model_callback(
            callback_context=context, llm_response=response(),
        )

    asyncio.run(scenario())

    assert metrics.LEDGER.snapshot(KEY)["llm"]["calls"] == 1


def test_the_routed_slug_prices_the_call_not_the_provider_echo():
    """OpenRouter answers as ``qwen/qwen3-…`` while the agent is configured with
    ``openrouter/qwen/qwen3-…``. Only the second has a price, so pricing must
    read the request's name, not the response's."""
    routed = "openrouter/qwen/qwen3-235b-a22b-2507"
    echoed = "qwen/qwen3-235b-a22b-2507"
    assert metrics.price_call(echoed, 1000, 100)[1] == "unavailable"

    plugin = metrics.UsageMetricsPlugin()
    run_call(
        plugin, model_context(), response(model_version=echoed), model=routed,
    )

    data = metrics.LEDGER.snapshot(KEY)
    assert data["llm"]["cost_usd"] > 0
    assert data["totals"]["complete"] is True
    assert data["agents"][0]["models"][0]["model"] == routed


def test_the_echoed_name_is_used_when_it_is_the_one_with_a_price():
    """Symmetric case — and the real one for the coder model: litellm prices
    ``deepseek/deepseek-v4-flash`` but not the ``openrouter/``-routed slug."""
    plugin = metrics.UsageMetricsPlugin()
    run_call(
        plugin, model_context(), response(model_version=PRICED), model=UNPRICED,
    )

    data = metrics.LEDGER.snapshot(KEY)
    model = data["agents"][0]["models"][0]
    assert data["llm"]["cost_usd"] > 0
    assert model["model"] == PRICED
    # Flagged: that is the underlying provider's list price, not the router's.
    assert model["cost_source"] == "litellm-alias"


def test_an_unpriceable_call_is_labelled_with_the_configured_slug():
    """Neither name prices: report the one an override key must match."""
    name, cost, source = metrics.price_first(
        [UNPRICED, "provider/echo"], 1000, 100,
    )
    assert (name, cost, source) == (UNPRICED, 0.0, "unavailable")


def test_a_model_without_a_price_is_counted_but_not_priced():
    plugin = metrics.UsageMetricsPlugin()
    run_call(
        plugin, model_context(), response(model_version=UNPRICED), model=UNPRICED,
    )

    data = metrics.LEDGER.snapshot(KEY)
    assert data["llm"]["total_tokens"] == 1100
    assert data["llm"]["cost_usd"] == 0.0
    assert data["llm"]["unpriced_calls"] == 1
    # The flag is the point: zero here means "unknown", not "free" — and the
    # model is named so the reader knows what to price.
    assert data["totals"]["complete"] is False
    assert data["llm"]["unpriced_models"] == [UNPRICED]


def test_price_overrides_close_the_gap(monkeypatch):
    monkeypatch.setattr(metrics, "_overrides", None)
    monkeypatch.setenv(
        "LLM_PRICE_OVERRIDES",
        '{"%s": {"input": 1.0, "output": 2.0, "cache_read": 0.1}}' % UNPRICED,
    )

    cost, source = metrics.price_call(UNPRICED, 1_000_000, 1_000_000, 0)
    assert source == "override"
    assert cost == pytest.approx(3.0)

    cached, _ = metrics.price_call(UNPRICED, 1_000_000, 0, 1_000_000)
    assert cached == pytest.approx(0.1)
    monkeypatch.setattr(metrics, "_overrides", None)


def test_a_failed_call_does_not_leak_its_start_record():
    plugin = metrics.UsageMetricsPlugin()
    context = model_context()

    async def scenario():
        await plugin.before_model_callback(
            callback_context=context, llm_request=SimpleNamespace(model=PRICED),
        )
        await plugin.on_model_error_callback(
            callback_context=context,
            llm_request=SimpleNamespace(model=PRICED),
            error=RuntimeError("502"),
        )

    asyncio.run(scenario())
    assert plugin._pending == {}


def test_live_pushes_are_rate_limited_but_the_last_one_always_lands():
    """A run makes hundreds of model calls; a browser needs a running total,
    not a frame per call. The final number must never be the one dropped."""
    sent: list[dict] = []

    async def sink(key, payload):
        sent.append(payload)

    metrics.set_metrics_sink(sink)
    plugin = metrics.UsageMetricsPlugin()

    async def scenario():
        for _ in range(5):
            await plugin.after_model_callback(
                callback_context=model_context(), llm_response=response(),
            )
        await asyncio.sleep(0)
        assert len(sent) == 1  # the rest fell inside the interval

        metrics.publish(KEY, force=True)
        await asyncio.sleep(0)
        assert len(sent) == 2
        assert sent[-1]["llm"]["calls"] == 5

    asyncio.run(scenario())


def test_a_failing_sink_cannot_break_a_run():
    async def broken(key, payload):
        raise RuntimeError("telemetry down")

    metrics.set_metrics_sink(broken)
    plugin = metrics.UsageMetricsPlugin()

    async def scenario():
        await plugin.after_model_callback(
            callback_context=model_context(), llm_response=response(),
        )
        await asyncio.sleep(0)  # let the fire-and-forget delivery run

    asyncio.run(scenario())
    assert metrics.LEDGER.snapshot(KEY)["llm"]["calls"] == 1


# ── sandbox ──────────────────────────────────────────────────────────────────

RECORD = {
    "task_id": "b9e357f5",
    "wall_clock": {"total_seconds": 645.2, "agent_seconds": 612.0, "queue_seconds": 0.0},
    "compute": {
        "cpu_seconds": 1080.0,
        "gpu_seconds": 468.0,
        "energy": {"total_energy_wh": 39.05},
    },
    "api": {"llm_calls": 24, "total_tokens": 232800, "cost_usd": 0.0432},
    "cost": {"api_cost_usd": 0.0432, "energy_cost_usd": 0.0022, "total_cost_usd": 0.0454},
}


@pytest.fixture(autouse=True)
def clean_journal():
    client._METRICS.clear("sbx-session")
    yield
    client._METRICS.clear("sbx-session")


def test_sandbox_run_digest_reads_every_section():
    digest = client.sandbox_run_digest(RECORD)
    assert digest["runs"] == 1
    assert digest["agent_seconds"] == 612.0
    assert digest["cpu_core_seconds"] == 1080.0
    assert digest["gpu_seconds"] == 468.0
    assert digest["energy_wh"] == 39.05
    assert digest["total_cost_usd"] == 0.0454


def test_a_partial_record_digests_to_zeros_not_an_error():
    assert client.sandbox_run_digest({})["total_cost_usd"] == 0.0
    assert client.sandbox_run_digest(None)["runs"] == 1


def test_total_cost_falls_back_to_its_parts():
    digest = client.sandbox_run_digest({
        "cost": {"api_cost_usd": 0.04, "energy_cost_usd": 0.002},
    })
    assert digest["total_cost_usd"] == pytest.approx(0.042)


def test_a_followup_replaces_its_sandbox_instead_of_doubling_it():
    """The server reports cumulative figures per container, so a second call
    into the same sandbox must overwrite, not add."""
    client._METRICS.put("sbx-session", "b9e357f5", RECORD)
    grown = {**RECORD, "cost": {**RECORD["cost"], "total_cost_usd": 0.09}}
    client._METRICS.put("sbx-session", "b9e357f5", grown)

    totals = client._METRICS.totals("sbx-session")
    assert totals["runs"] == 1
    assert totals["total_cost_usd"] == pytest.approx(0.09)


def test_totals_add_up_across_sandboxes():
    client._METRICS.put("sbx-session", "one", RECORD)
    client._METRICS.put("sbx-session", "two", RECORD)

    totals = client._METRICS.totals("sbx-session")
    assert totals["runs"] == 2
    assert totals["llm_calls"] == 48
    assert totals["total_cost_usd"] == pytest.approx(0.0908)


def test_metrics_are_collected_only_once_the_run_is_over():
    # Still running: asking now would journal a non-final snapshot.
    assert client._should_collect(True, "running", "b9e357f5") is False
    assert client._should_collect(True, "timeout", "b9e357f5") is False
    # Finished — including the failures, which is when the number matters most.
    assert client._should_collect(True, "cooldown", "b9e357f5") is True
    assert client._should_collect(True, "error", "b9e357f5") is True
    assert client._should_collect(True, "cancelled", "b9e357f5") is True
    # Switched off, or nothing to ask about.
    assert client._should_collect(False, "cooldown", "b9e357f5") is False
    assert client._should_collect(True, "cooldown", None) is False


def test_get_sandbox_metrics_reads_the_journal():
    client._METRICS.put("sbx-session", "b9e357f5", RECORD)

    view = client.get_sandbox_metrics(session_id="sbx-session")
    assert view["metrics"]["task_id"] == "b9e357f5"
    assert view["runs"] == [RECORD]
    assert view["totals"]["runs"] == 1


def test_metrics_key_exists_even_when_there_is_nothing():
    view = client.get_sandbox_metrics(session_id="never-ran")
    assert view["metrics"] is None
    assert view["runs"] == []


def test_clearing_hands_the_totals_back_before_dropping_them():
    client._METRICS.put("sbx-session", "b9e357f5", RECORD)

    discarded = client.clear_sandbox_metrics(session_id="sbx-session")
    assert discarded["discarded_runs"] == 1
    assert discarded["discarded_totals"]["total_cost_usd"] == pytest.approx(0.0454)
    assert client.get_sandbox_metrics(session_id="sbx-session")["metrics"] is None


def test_a_broken_metrics_sink_does_not_fail_the_run():
    def broken(record):
        raise RuntimeError("telemetry down")

    client._publish_metrics("sbx-session", "b9e357f5", RECORD, None, broken)
    # Journalled regardless of what the subscriber did with it.
    assert client._METRICS.get("sbx-session") == RECORD


def test_sandbox_cost_lands_on_the_agent_that_started_it():
    metrics.record_sandbox_run(
        KEY, "CoderAgent", "b9e357f5", client.sandbox_run_digest(RECORD),
    )
    run_call(metrics.UsageMetricsPlugin(), model_context("CoderAgent"), response())

    data = metrics.LEDGER.snapshot(KEY)
    coder = data["agents"][0]
    assert coder["agent"] == "CoderAgent"
    assert coder["sandbox"]["runs"] == 1
    # The session total spans both bills: this process's tokens and the
    # sandbox's own agent, GPU and electricity.
    assert data["totals"]["cost_usd"] == pytest.approx(
        coder["llm"]["cost_usd"] + 0.0454,
    )
    assert data["totals"]["energy_wh"] == 39.05


def test_the_run_result_never_carries_metrics(monkeypatch):
    """A tool result is prompt text: a model that can read its own bill starts
    optimising the bill. The numbers must reach the host by another road."""
    monkeypatch.setattr(client, "_prepare", lambda **kw: client._Submission(
        base_url="http://sandbox", api_url="http://sandbox/api/v1",
        session="sbx-session", target_id=None,
        url="http://sandbox/api/v1/run", body={},
    ))
    monkeypatch.setattr(client.httpx, "post", lambda *a, **kw: SimpleNamespace(
        status_code=200, raise_for_status=lambda: None,
        json=lambda: {"task_id": "b9e357f5", "reused": False},
    ))
    monkeypatch.setattr(client, "_wait_for_completion", lambda **kw: {
        "status": "cooldown", "succeeded": True, "summary": "done",
    })
    monkeypatch.setattr(client, "_fetch_metrics", lambda api_url, sandbox_id: RECORD)

    seen: list[dict] = []
    result = client.run_sandbox_task(
        "train a model", session_id="sbx-session", verbose=False,
        metrics_sink=seen.append,
    )

    assert result["status"] == "cooldown"
    assert not [k for k in result if "metric" in k or "cost" in k]
    assert "0.0454" not in str(result)
    # …and it did reach the subscriber and the journal.
    assert seen and seen[0]["task_id"] == "b9e357f5"
    assert client._METRICS.totals("sbx-session")["total_cost_usd"] == pytest.approx(0.0454)


# ── reporting ────────────────────────────────────────────────────────────────

def test_report_shows_agents_the_sandbox_and_the_caveat():
    plugin = metrics.UsageMetricsPlugin()
    run_call(plugin, model_context("OrchestratorAgent"), response())
    run_call(plugin, model_context("CoderAgent", "inv_2"),
             response(model_version=UNPRICED), model=UNPRICED)
    metrics.record_sandbox_run(
        KEY, "CoderAgent", "b9e357f5", client.sandbox_run_digest(RECORD),
    )

    report = metrics.format_report(metrics.LEDGER.snapshot(KEY))
    assert "OrchestratorAgent" in report and "CoderAgent" in report
    assert "sandbox" in report
    assert "468 GPU-s" in report
    assert "no known price" in report and UNPRICED in report


def test_an_empty_session_reports_zero_rather_than_failing():
    data = metrics.LEDGER.snapshot(("nobody", "nothing"))
    assert data["agents"] == []
    assert data["totals"]["cost_usd"] == 0.0
    assert data["totals"]["complete"] is True
    assert metrics.format_report(data)
