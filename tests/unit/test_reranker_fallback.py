"""The reranker-failure path: a malformed ranking must not look like an abstain.

`filtered_tools` comes out empty for three very different reasons, and only one
of them means "no tool fits this task". These tests pin the other two — an
unreadable reply and a readable-but-unusable one — to recovery, not to the
CoderAgent redirect that would silently discard tools retrieval found correctly.
"""
from types import SimpleNamespace

import pytest

from CoScientist.agents.callbacks.tool_callbacks import (
    RERANK_EMPTY_RANKING,
    RERANK_NO_CANDIDATES,
    RERANK_PARSE_FAILED,
    RERANK_RECOVERED,
    RERANK_SCORED,
    TOOL_MATCH_STATE_KEY,
    after_fullset_reranker_agent,
    after_tool_reranker_agent,
    redirect_when_no_tools,
    rerank_fallback_active,
)

ACC = [
    {"tool_index": 1, "tool": "run_transformer", "server_id": "s1"},
    {"tool_index": 2, "tool": "dock_ligand", "server_id": "s2"},
]


def _ctx(state):
    return SimpleNamespace(state=state)


def _verdict(state, reranked, *, acc=None, shortlist_scores=None):
    state.update({"reranked_tools": reranked, "accumulated_tools": list(ACC if acc is None else acc)})
    if shortlist_scores is not None:
        state["shortlist_scores"] = shortlist_scores
    after_tool_reranker_agent(_ctx(state))
    return state[TOOL_MATCH_STATE_KEY]


# ── the reranker's answer was unreadable ─────────────────────────────────────

@pytest.mark.parametrize("payload", [
    '{"tools": [{"index": 1, "score": 0.9',      # truncated mid-object
    'Sorry, I cannot rank these tools.',          # prose, no JSON at all
    '[{"index": 1, "score": 0.9}]',               # a JSON array, not an object
    None,                                          # the agent produced nothing
])
def test_unreadable_ranking_keeps_the_candidates_for_the_fallback(payload):
    state = {}
    verdict = _verdict(state, payload)

    assert verdict["reason"] == RERANK_PARSE_FAILED
    assert verdict["candidates"] == 2
    # The pool MUST survive: FedotAgent builds its server set out of it.
    assert state["accumulated_tools"] == ACC
    assert rerank_fallback_active(state) is True
    # ...and this is emphatically not an abstain.
    assert redirect_when_no_tools(_ctx(state)) is None


def test_readable_but_unusable_ranking_is_its_own_verdict():
    """Right JSON, wrong key: the prompt names `tool_index` in its constraints
    and `index` in its output format, so a model emitting the former is a
    realistic failure — and by score alone it is indistinguishable from
    "everything is irrelevant"."""
    state = {}
    verdict = _verdict(state, '{"tools": [{"tool_index": 1, "score": 0.9}]}')

    assert verdict["reason"] == RERANK_EMPTY_RANKING
    assert rerank_fallback_active(state) is True
    assert redirect_when_no_tools(_ctx(state)) is None


# ── recovery layer 1: the local cross-encoder ────────────────────────────────

def test_cross_encoder_scores_rank_the_tools_when_the_model_cannot():
    """The shortlist pass already scored every candidate against this same task,
    on one scale. Spending that instead of the fallback is free."""
    state = {}
    verdict = _verdict(state, "not json at all", shortlist_scores={1: 0.81, 2: 0.04})

    assert verdict["reason"] == RERANK_RECOVERED
    assert [t["tool"] for t in state["filtered_tools"]] == ["run_transformer"]
    assert verdict["matched"] is True
    # Judged after all — so the pool is consumed and the fallback stands down.
    assert state["accumulated_tools"] == []
    assert rerank_fallback_active(state) is False


def test_recovery_that_finds_nothing_relevant_still_abstains():
    """A cross-encoder verdict IS a verdict: all-irrelevant means CoderAgent,
    not FEDOT.MAS."""
    state = {}
    verdict = _verdict(state, "not json at all", shortlist_scores={1: 0.02, 2: 0.01})

    assert verdict["reason"] == RERANK_RECOVERED
    assert state["filtered_tools"] == []
    assert rerank_fallback_active(state) is False
    assert redirect_when_no_tools(_ctx(state)) is not None


# ── the paths that must keep behaving exactly as before ──────────────────────

def test_a_real_low_score_verdict_still_abstains():
    state = {}
    verdict = _verdict(state, {"tools": [{"index": 1, "score": 0.05}, {"index": 2, "score": 0.02}]})

    assert verdict["reason"] == RERANK_SCORED
    assert verdict["matched"] is False
    assert rerank_fallback_active(state) is False
    assert redirect_when_no_tools(_ctx(state)) is not None


def test_a_real_match_is_untouched():
    state = {}
    verdict = _verdict(state, {"tools": [{"index": 1, "score": 0.8}]})

    assert verdict["reason"] == RERANK_SCORED
    assert [t["tool"] for t in state["filtered_tools"]] == ["run_transformer"]
    assert state["accumulated_tools"] == []
    assert rerank_fallback_active(state) is False


def test_nothing_retrieved_is_not_a_fallback_case():
    """No candidates means there is nothing to hand FEDOT.MAS either."""
    state = {}
    verdict = _verdict(state, "unreadable", acc=[])

    assert verdict["reason"] == RERANK_NO_CANDIDATES
    assert rerank_fallback_active(state) is False
    assert redirect_when_no_tools(_ctx(state)) is not None


def test_the_fallback_switch_turns_the_whole_path_off(monkeypatch):
    import CoScientist.config as config_mod

    state = {}
    _verdict(state, "unreadable")
    assert rerank_fallback_active(state) is True

    settings = config_mod.get_settings()
    monkeypatch.setattr(settings.web, "fedot_fallback_enabled", False)
    assert rerank_fallback_active(state) is False
    # With nowhere to route it, an unjudged verdict falls back to the old abstain.
    assert redirect_when_no_tools(_ctx(state)) is not None


# ── the sibling reranker used to take the whole run down with it ─────────────

@pytest.mark.parametrize("payload", [
    '{"mcp_scores": [{"index": 0, "score": tru',        # truncated JSON string
    {"mcp_scores": [{"idx": 0, "score": True}]},        # no `index` key
    {"mcp_scores": "none"},                              # not even a list
    {},                                                   # nothing at all
    "utter nonsense",
])
def test_malformed_web_mcp_ranking_never_raises(payload):
    """This callback used to index straight into the payload, so a shape the
    schema would have rejected raised AttributeError/KeyError/TypeError out of an
    after_agent callback and killed the session."""
    state = {
        "reranked_web_servers": payload,
        "accumulated_web_mcps": [{"index": 0, "name": "web-mcp"}],
    }
    after_fullset_reranker_agent(_ctx(state))
    assert state["filtered_mcps"] == []
    assert state["accumulated_web_mcps"] == []


def test_web_mcp_ranking_still_deploys_what_it_selects():
    state = {
        "reranked_web_servers": '{"mcp_scores": [{"index": 0, "score": true}, {"index": 1, "score": false}]}',
        "accumulated_web_mcps": [{"index": 0, "name": "keep"}, {"index": 1, "name": "drop"}],
    }
    after_fullset_reranker_agent(_ctx(state))
    assert [m["name"] for m in state["filtered_mcps"]] == ["keep"]


# ── the switch itself ────────────────────────────────────────────────────────

class _Marker:
    """Stands in for a child agent: records that it ran, yields one event."""

    def __init__(self, name):
        self.name = name

    async def run_async(self, ctx):
        yield SimpleNamespace(author=self.name, content=f"answer from {self.name}")


def _run_switch(state):
    import asyncio

    from CoScientist.agents.custom_agents import ExecutorSwitchAgent

    switch = ExecutorSwitchAgent(name="ExecutorSwitchAgent")
    # Bypass pydantic's BaseAgent validation on sub_agents: the switch only ever
    # calls run_async on them, and real LlmAgents would need a live runner.
    object.__setattr__(switch, "sub_agents", [_Marker("ExperimentAgent"), _Marker("FedotAgent")])
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="i1")

    async def drain():
        return [e async for e in switch._run_async_impl(ctx)]

    return asyncio.run(drain())


def test_switch_runs_the_react_executor_on_a_normal_verdict():
    state = {}
    _verdict(state, {"tools": [{"index": 1, "score": 0.8}]})
    events = _run_switch(state)
    # Exactly one child speaks — AgentTool returns the LAST content event, so a
    # second, stood-down child would overwrite this answer.
    assert [e.author for e in events] == ["ExperimentAgent"]


def test_switch_runs_the_fedot_fallback_when_the_ranking_was_unusable():
    state = {}
    _verdict(state, '{"tools": [{"index": 1, "score": 0.9')
    events = _run_switch(state)
    assert [e.author for e in events] == ["FedotAgent"]


# ── what fedot_tool is actually handed ───────────────────────────────────────

def _stub_fedot(monkeypatch):
    """Replace fedot_tool's Postgres + MAS with recorders; return the recorder."""
    import asyncio

    from CoScientist.tools import fedotmas_tools as ft

    seen = {}

    class _Server:
        def __init__(self, sid):
            self.name, self.url, self.description, self.protocol = sid, f"http://{sid}", "", "http"

    class _PG:
        def __init__(self, *a, **k):
            pass

        async def initialize(self):
            return None

        async def close(self):
            return None

        async def get_server(self, sid):
            return _Server(sid)

    class _MAS:
        def __init__(self, mcp_servers=None, plugins=None):
            seen["servers"] = sorted(mcp_servers or {})

        async def run(self, task, timeout=None):
            seen["timeout"] = timeout
            return "done"

    monkeypatch.setattr(ft, "PostgresClient", _PG)
    monkeypatch.setattr(ft, "MAS", _MAS)
    monkeypatch.setattr(ft, "HttpMCPServer", lambda **kw: kw)

    def call(state):
        tc = SimpleNamespace(state=state)
        asyncio.run(ft.fedot_toolset.fedot_tool("do the thing", tool_context=tc))
        return seen

    return call


def test_fedot_gets_the_rerankers_pick_when_there_is_one(monkeypatch):
    call = _stub_fedot(monkeypatch)
    seen = call({"filtered_tools": [{"server_id": "s1"}], "accumulated_tools": [{"server_id": "s9"}]})
    assert seen["servers"] == ["s1"]
    # A chosen tool set is the normal path: no fallback budget imposed.
    assert seen["timeout"] is None


def test_fedot_gets_the_whole_pool_when_the_reranker_failed(monkeypatch):
    """The point of the fallback: an empty `filtered_tools` must not mean an
    empty server list, or this path would rescue nothing."""
    call = _stub_fedot(monkeypatch)
    seen = call({"filtered_tools": [], "accumulated_tools": [{"server_id": "s1"}, {"server_id": "s2"}]})
    assert seen["servers"] == ["s1", "s2"]
    # Nobody chose this run — a malformed reply did — so it is time-bounded.
    assert seen["timeout"] == 900
