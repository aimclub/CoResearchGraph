"""An agent's account of its run, kept past the end of the process.

It used to live in an in-process LRU of 256 entries, so every restart threw away
every account a study had paid a model to write, and re-opening a finished run
bought them again one card at a time. An imported session bundle arrived with
none at all.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from CoScientist.graph import agent_summary, summary_store


@pytest.fixture()
def key(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    agent_summary._CACHE.clear()
    return ("u", "s")


def _node(calls=1):
    return {"id": "agent:ResearchAgent@i1", "kind": "agent",
            "executor_agent": "ResearchAgent", "status": "success",
            "input": "find literature on furanocoumarins",
            "calls": [{"tool": "tavily_search", "status": "success",
                       "output": f"hit {i}"} for i in range(calls)]}


def _answers(monkeypatch, text="an account of the run"):
    said = []

    async def _complete(system, user):
        said.append(user)
        return text, "tiny-model"

    monkeypatch.setattr(agent_summary, "_complete", _complete)
    return said


# ── the file ────────────────────────────────────────────────────────────────
def test_an_account_survives_the_process(key, monkeypatch):
    """The whole point: the LRU is gone, the record is not."""
    asked = _answers(monkeypatch)
    store = summary_store.for_session(key)

    first = asyncio.run(agent_summary.summarize(_node(), lang="en", store=store))
    assert first["cached"] is False and len(asked) == 1

    agent_summary._CACHE.clear()          # the restart

    again = asyncio.run(agent_summary.summarize(_node(), lang="en", store=store))
    assert again["summary"] == first["summary"]
    assert again["cached"] is True
    assert len(asked) == 1, "the model was not asked a second time"


def test_a_changed_trace_is_a_different_account(key, monkeypatch):
    """A running agent moves on, and an account of what it had done twenty
    calls ago, presented as current, is worse than none."""
    asked = _answers(monkeypatch)
    store = summary_store.for_session(key)

    one = asyncio.run(agent_summary.summarize(_node(calls=1), lang="en", store=store))
    two = asyncio.run(agent_summary.summarize(_node(calls=3), lang="en", store=store))

    assert one["stamp"] != two["stamp"]
    assert len(asked) == 2


def test_regenerating_replaces_what_was_kept(key, monkeypatch):
    store = summary_store.for_session(key)
    _answers(monkeypatch, "first account")
    asyncio.run(agent_summary.summarize(_node(), lang="en", store=store))

    _answers(monkeypatch, "second account")
    forced = asyncio.run(agent_summary.summarize(_node(), lang="en", store=store,
                                                 force=True))
    assert forced["summary"] == "second account"
    agent_summary._CACHE.clear()
    kept = asyncio.run(agent_summary.summarize(_node(), lang="en", store=store))
    assert kept["summary"] == "second account"


def test_each_language_is_its_own_account(key, monkeypatch):
    asked = _answers(monkeypatch)
    store = summary_store.for_session(key)
    asyncio.run(agent_summary.summarize(_node(), lang="en", store=store))
    asyncio.run(agent_summary.summarize(_node(), lang="ru", store=store))
    assert len(asked) == 2


def test_the_newest_account_is_readable_without_the_trace(key, monkeypatch):
    """What the node report needs: what is KNOWN about an agent, not what is
    current. It compares the stamp itself to decide whether to say so."""
    _answers(monkeypatch, "an account")
    store = summary_store.for_session(key)
    asyncio.run(agent_summary.summarize(_node(), lang="en", store=store))

    kept = summary_store.latest(key, "agent:ResearchAgent@i1", lang="en")
    assert kept["summary"] == "an account"
    assert kept["trace_stamp"] and kept["written_at"]
    assert summary_store.latest(key, "agent:Nobody@i1") is None


# ── it must never break a run ───────────────────────────────────────────────
def test_a_damaged_file_is_not_a_failed_run(key, monkeypatch):
    summary_store.store_path(key).parent.mkdir(parents=True, exist_ok=True)
    summary_store.store_path(key).write_text("{ this is not json",
                                             encoding="utf-8")
    assert summary_store.latest(key, "agent:ResearchAgent@i1") is None

    asked = _answers(monkeypatch)
    result = asyncio.run(agent_summary.summarize(
        _node(), lang="en", store=summary_store.for_session(key)))
    assert result["summary"] and len(asked) == 1


def test_a_store_that_throws_does_not_lose_the_answer(key, monkeypatch):
    class _Broken:
        def get(self, *a, **k):
            raise OSError("disk gone")

        def put(self, *a, **k):
            raise OSError("disk gone")

    _answers(monkeypatch, "the account")
    result = asyncio.run(agent_summary.summarize(_node(), lang="en",
                                                 store=_Broken()))
    assert result["summary"] == "the account"


def test_no_store_at_all_is_the_old_behaviour(key, monkeypatch):
    asked = _answers(monkeypatch)
    asyncio.run(agent_summary.summarize(_node(), lang="en"))
    assert len(asked) == 1
    assert not summary_store.all_entries(key), "nothing written without a store"


def test_an_empty_account_is_not_recorded(key):
    summary_store.put(key, "agent:X@i1", lang="en", trace_stamp="abc",
                      summary="   ")
    assert summary_store.all_entries(key) == {}


# ── it travels ──────────────────────────────────────────────────────────────
def test_accounts_travel_with_the_session(key, tmp_path):
    """An imported bundle must not re-buy what the exporting run already paid
    for. The scope is deliberately not part of the key — an import lands under
    a new user and session id, and keying on it would match nothing."""
    from CoScientist.web.session_bundle import _restore_graph_files

    summary_store.put(key, "agent:ResearchAgent@i1", lang="en",
                      trace_stamp="abc123", summary="what it did",
                      model="tiny-model")
    packed = {"version": 1, "entries": summary_store.all_entries(key)}

    _restore_graph_files("other_user", "other_session", None, None, packed)

    moved = summary_store.latest(("other_user", "other_session"),
                                 "agent:ResearchAgent@i1")
    assert moved["summary"] == "what it did"


def test_the_file_does_not_grow_without_bound(key, monkeypatch):
    monkeypatch.setattr(summary_store, "_MAX_ENTRIES", 5)
    for i in range(9):
        summary_store.put(key, f"agent:A@{i}", lang="en",
                          trace_stamp=f"stamp{i}", summary=f"account {i}")
    entries = summary_store.all_entries(key)
    assert len(entries) == 5
    # The oldest go; the newest stay.
    assert {e["node_id"] for e in entries.values()} == {
        f"agent:A@{i}" for i in range(4, 9)}


def test_the_file_is_written_atomically(key, monkeypatch):
    """A half-written index reads exactly like a corrupt one."""
    summary_store.put(key, "agent:A@1", lang="en", trace_stamp="s",
                      summary="one")
    path = summary_store.store_path(key)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
    assert not list(path.parent.glob("*.tmp")), "no leftovers"
