"""The blind-review guard keeps a validation run from reading its own answer."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from CoScientist.agents.callbacks.spoiler_guard import LOG_STATE_KEY, REDACTED, SpoilerGuard


def _guard(**kw):
    base = dict(blocklist=["DLinear", "Zeng", "Are Transformers Effective"], cutoff_year=2021, judge=False)
    base.update(kw)
    return SpoilerGuard(**base)


def _ctx():
    return SimpleNamespace(state={})


def _tool(name):
    return SimpleNamespace(name=name)


def test_a_query_naming_a_listed_term_is_refused_and_logged():
    g, ctx = _guard(), _ctx()
    out = g.guard_query(_tool("tavily_search"), {"query": "Informer vs DLinear ETTh1"}, ctx)
    assert out["status"] == "blocked" and "DLinear" in out["error"]
    assert ctx.state[LOG_STATE_KEY][0]["kind"] == "query_blocked"


def test_a_clean_query_passes_and_openalex_gets_the_cutoff():
    g, ctx = _guard(), _ctx()
    args = {"query": "Informer long sequence forecasting ETTh1"}
    assert g.guard_query(_tool("search_papers"), args, ctx) is None
    assert args["publication_year"] == "<2022"
    assert g.guard_query(_tool("tavily_search"), {"query": "Informer ETTh1"}, ctx) is None


def test_tools_that_do_not_search_are_left_alone():
    g, ctx = _guard(), _ctx()
    assert g.guard_query(_tool("research_overview"), {"q": "DLinear"}, ctx) is None
    assert g.guard_query(_tool("run_sandbox_task"), {"cmd": "echo DLinear"}, ctx) is None


def test_structured_results_are_dropped_by_term_and_by_date_in_place():
    g, ctx = _guard(), _ctx()
    payload = {"results": [
        {"title": "Informer: beyond efficient transformer", "url": "u1", "published_date": "2021-05-01", "content": "ProbSparse attention"},
        {"title": "Are Transformers Effective for Time Series Forecasting?", "url": "u2", "published_date": "2022-05-26", "content": "a linear model"},
        {"title": "ETT dataset description", "url": "u3", "published_date": "2023-01-01", "content": "seven features"},
    ], "count": 3}
    response = {"content": [{"type": "text", "text": json.dumps(payload)}]}
    assert asyncio.run(g.guard_result(_tool("tavily_search"), {}, ctx, response)) is None
    left = json.loads(response["content"][0]["text"])
    assert [r["url"] for r in left["results"]] == ["u1"]
    assert left["count"] == 1
    kinds = [e["kind"] for e in ctx.state[LOG_STATE_KEY]]
    assert kinds == ["result_dropped", "date_dropped"]


def test_plain_text_is_redacted_sentence_by_sentence():
    g, ctx = _guard(), _ctx()
    response = {"text": "Informer uses ProbSparse attention. Zeng et al. later showed a linear model wins. The data has seven features."}
    asyncio.run(g.guard_result(_tool("tavily_extract"), {}, ctx, response))
    assert response["text"] == f"Informer uses ProbSparse attention. {REDACTED} The data has seven features."
    assert ctx.state[LOG_STATE_KEY][0] == {**ctx.state[LOG_STATE_KEY][0], "kind": "redacted", "sentences": 1}


def test_the_judge_drops_a_block_that_gives_the_outcome_away(monkeypatch):
    g, ctx = _guard(judge=True), _ctx()

    async def fake_judge(blocks):
        return {i for i, b in enumerate(blocks) if "simpler baseline matched it" in b}

    monkeypatch.setattr(g, "_judge", fake_judge)
    payload = [{"title": "A", "url": "a", "content": "Informer architecture overview."},
               {"title": "B", "url": "b", "content": "In 2023 a simpler baseline matched it on ETTh1."}]
    response = {"content": [{"type": "text", "text": json.dumps(payload)}]}
    asyncio.run(g.guard_result(_tool("search_papers"), {}, ctx, response))
    assert [r["url"] for r in json.loads(response["content"][0]["text"])] == ["a"]
    assert ctx.state[LOG_STATE_KEY][-1]["kind"] == "judge_dropped"


def test_a_failing_judge_lets_blocks_through_and_says_so(monkeypatch, caplog):
    g, ctx = _guard(judge=True), _ctx()

    async def broken(blocks):
        raise RuntimeError("no model")

    import CoScientist.agents.callbacks.spoiler_guard as mod
    monkeypatch.setattr(mod.litellm if hasattr(mod, "litellm") else mod, "acompletion", broken, raising=False)
    flagged = asyncio.run(g._judge(["some block"]))
    assert flagged == set()


def test_the_blind_profile_builds_with_the_guard_first():
    from CoScientist.assembly.schema import load_config, resolve_config_path
    from CoScientist.assembly.assembler import build_system

    cfg = load_config(resolve_config_path("blind"))
    research = cfg.agents["ResearchAgent"]
    assert research.callbacks.before_tool[0] == "guard_spoiler_query"
    assert research.callbacks.after_tool[0] == "guard_spoiler_result"
    assert "WebSearchLimiter" in research.callbacks.before_tool
    collector = cfg.agents["DatasetCollectorAgent"]
    assert collector.callbacks.before_tool[0] == "guard_spoiler_query"
    build_system(cfg)
