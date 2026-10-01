"""The "view summary" button: one agent's trace, a small model, a few lines."""
import asyncio

import pytest

from CoScientist.graph import agent_summary as m


def _node():
    return {
        "id": "agent:CoderAgent@inv#2", "kind": "agent", "executor_agent": "CoderAgent",
        "status": "success", "run": 2, "runs": 3, "t_start": 0.0, "t_end": 75.0,
        "input": "request: train the CVAE",
        "output": "## done\n\nTrained for 10 epochs, loss 0.22. " + "x" * 9000,
        "calls": [
            {"tool": "execute_bash", "status": "success", "duration": 40.0,
             "input": "command: python train.py", "output": "epoch 10/10 loss=0.22",
             "output_files": []},
            {"tool": "execute_bash", "status": "failed", "duration": 12.0,
             "input": "command: pytest -q", "output": "2 failed, 14 passed", "output_files": []},
            {"tool": "upload_artifact", "status": "success", "duration": 2.0,
             "input": "path: /workspace/fig.png", "output": "uri: s3://b/fig.png",
             "output_files": ["s3://b/fig.png"]},
        ],
        "artifacts": [{"uri": "s3://b/fig.png", "kind": "file", "tool": "upload_artifact"}],
    }


def test_trace_reads_top_down_and_is_cut_for_length():
    trace = m.trace_of(_node())
    lines = trace.splitlines()
    assert lines[0] == "AGENT: CoderAgent"
    assert "RUN: 2 of 3 in this request" in lines
    assert "STATUS: success (1m 15s)" in lines
    assert "TASK: request: train the CVAE" in lines
    assert "TOOL CALLS (3):" in lines
    assert "2. execute_bash [failed, 12.0s]" in lines
    assert "   result: 2 failed, 14 passed" in lines
    assert "   files: s3://b/fig.png" in lines
    assert "ARTIFACTS: s3://b/fig.png" in lines
    report = next(l for l in lines if l.startswith("FINAL REPORT: "))
    assert report.endswith("…") and len(report) < 6_100, "the report is cut, not dumped"
    # A cut value is one line; the model reads a list, not a document.
    assert "\n\n" not in trace


def test_a_summary_is_remembered_until_the_trace_changes(monkeypatch):
    calls = []

    async def fake(system, user):
        calls.append((system, user))
        return f"summary #{len(calls)}", "tiny-model"

    monkeypatch.setattr(m, "_complete", fake)
    m._CACHE.clear()

    async def scenario():
        node = _node()
        first = await m.summarize(node, lang="ru", scope="u/s")
        again = await m.summarize(node, lang="ru", scope="u/s")
        other_lang = await m.summarize(node, lang="en", scope="u/s")
        node["calls"].append({"tool": "execute_bash", "status": "success", "input": "ls"})
        moved_on = await m.summarize(node, lang="ru", scope="u/s")
        forced = await m.summarize(node, lang="ru", scope="u/s", force=True)
        return first, again, other_lang, moved_on, forced

    first, again, other_lang, moved_on, forced = asyncio.run(scenario())
    # `stamp` rides along now: it is the digest of the trace the account was
    # written from, and it is what lets a reader be told an account is stale
    # instead of being shown a stale one as current.
    assert {k: first[k] for k in ("summary", "model", "cached")} == {
        "summary": "summary #1", "model": "tiny-model", "cached": False}
    assert first["stamp"], "the account says which trace it describes"
    assert again["summary"] == "summary #1" and again["cached"] is True
    assert other_lang["summary"] == "summary #2", "each language is its own summary"
    assert moved_on["summary"] == "summary #3", "a changed trace is summarized afresh"
    assert forced["summary"] == "summary #4" and forced["cached"] is False
    assert "Russian" in calls[0][0] and "English" in calls[1][0]
    assert calls[0][1].startswith("AGENT: CoderAgent")


def test_model_choice_falls_back_from_dedicated_to_summary_url_to_main(monkeypatch):
    from types import SimpleNamespace

    def settings(**llm):
        base = dict(agent_summary_model=None, summary_url=None, main_model="main", main_url="http://main")
        base.update(llm)
        return SimpleNamespace(llm=SimpleNamespace(**base))

    import CoScientist.config as cfg
    monkeypatch.setattr(cfg, "get_settings", lambda: settings(agent_summary_model="small"))
    assert m.model_name() == ("small", "http://main")
    monkeypatch.setattr(cfg, "get_settings", lambda: settings(summary_url="http://x/v1;flash-lite"))
    assert m.model_name() == ("flash-lite", "http://x/v1")
    monkeypatch.setattr(cfg, "get_settings", lambda: settings())
    assert m.model_name() == ("main", "http://main")


def test_the_button_endpoint_summarizes_one_agent_of_one_request(tmp_path, monkeypatch):
    import time
    import uuid

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web"))
    from starlette.testclient import TestClient

    from CoScientist.graph.memory import get_knowledge_graph
    from CoScientist.web.app import create_app

    seen = []

    async def fake(system, user):
        seen.append(user)
        return "- ran a search\n- found 3 papers", "tiny-model"

    monkeypatch.setattr(m, "_complete", fake)
    m._CACHE.clear()

    with TestClient(create_app()) as client:
        user = client.post("/api/users", json={"nickname": f"t-{uuid.uuid4().hex[:8]}"}).json()["user"]
        session = client.post(f"/api/users/{user['id']}/sessions", json={"title": "run"}).json()["session"]
        graph = get_knowledge_graph(user_id=user["id"], session_id=session["id"])
        now = time.time()
        graph.add_node(id="goal:i1", kind="goal", turn_id="i1", label="find papers",
                       status="success", t_start=now, t_end=now + 9)
        graph.add_node(id="agent:ResearchAgent@i1", kind="agent", turn_id="i1",
                       executor_agent="ResearchAgent", status="success",
                       input="request: find papers", output="found 3", t_start=now + 1, t_end=now + 8)
        graph.add_edge("goal:i1", "agent:ResearchAgent@i1", type="caused_by")
        graph.add_node(id="tool:1", kind="tool_call", turn_id="i1", label="search",
                       executor_agent="ResearchAgent", status="success",
                       input="query: cvae", output="3 hits", t_start=now + 2, t_end=now + 3)
        graph.add_edge("agent:ResearchAgent@i1", "tool:1", type="caused_by")

        url = f"/api/users/{user['id']}/sessions/{session['id']}/graph/agent_summary"
        r = client.post(url, json={"node_id": "agent:ResearchAgent@i1", "turn": "i1", "lang": "en"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert {k: body[k] for k in ("summary", "model", "cached")} == {
            "summary": "- ran a search\n- found 3 papers",
            "model": "tiny-model", "cached": False}
        # The model saw the folded trace: the agent's own call, not the whole graph.
        assert "1. search [success" in seen[0] and "args: query: cvae" in seen[0]

        second = client.post(url, json={"node_id": "agent:ResearchAgent@i1", "turn": "i1", "lang": "en"})
        assert second.json()["cached"] is True and len(seen) == 1, "same trace, same language: no second call"
        assert client.post(url, json={"node_id": "tool:1", "turn": "i1"}).status_code == 404
        assert client.post(url, json={"node_id": "agent:Nobody@i1"}).status_code == 404
        assert client.post(url, json={}).status_code == 422


def test_a_model_named_without_a_provider_goes_through_the_openai_route():
    """summary_url names the model the way the endpoint does; litellm needs
    a provider it knows, and an OpenAI-compatible base is the route for it."""
    assert m._routable("google/gemini-2.0-flash-lite-001", "https://openrouter.ai/api/v1") \
        == "openai/google/gemini-2.0-flash-lite-001"
    assert m._routable("openrouter/deepseek/deepseek-v4.1-flash", "https://openrouter.ai/api/v1") \
        == "openrouter/deepseek/deepseek-v4.1-flash", "a routable name is left alone"


def test_a_retired_small_model_falls_back_to_the_main_model(monkeypatch):
    """A config still naming a model the endpoint no longer serves must not
    leave the panel empty: the main model writes the summary instead."""
    import litellm
    from types import SimpleNamespace

    import CoScientist.config as cfg
    monkeypatch.setattr(cfg, "get_settings", lambda: SimpleNamespace(llm=SimpleNamespace(
        agent_summary_model="openrouter/retired-model", summary_url=None,
        main_model="openrouter/main-model", main_url="http://main", openai_api_key="k",
        request_timeout=5)))
    tried = []

    async def fake_completion(**kw):
        tried.append(kw["model"])
        if "retired" in kw["model"]:
            raise litellm.NotFoundError("No endpoints found", model=kw["model"], llm_provider="openrouter")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="written by main"))])

    monkeypatch.setattr(litellm, "acompletion", fake_completion)
    text, model = asyncio.run(m._complete("sys", "trace"))
    assert (text, model) == ("written by main", "openrouter/main-model")
    assert tried == ["openrouter/retired-model", "openrouter/main-model"]
