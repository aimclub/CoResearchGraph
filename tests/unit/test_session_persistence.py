"""Past conversations must survive a restart and be reopenable in the UI."""
import importlib

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path))
    import CoScientist.web.session_store as mod
    importlib.reload(mod)
    return mod


@pytest.fixture()
def Registry(store):
    import CoScientist.web.session_registry as mod
    importlib.reload(mod)
    return mod.LocalSessionRegistry


def test_users_and_sessions_survive_a_restart(Registry):
    first = Registry()
    user = first.create_user("ivan")
    session = first.create_session(user["id"], title="Молекулы")

    restarted = Registry()  # a fresh process would build a new registry
    users = restarted.list_users()
    assert [u["nickname"] for u in users] == ["ivan"]
    sessions = restarted.list_sessions(user["id"])
    assert [s["id"] for s in sessions] == [session["id"]]
    assert sessions[0]["title"] == "Молекулы"


def test_a_run_left_processing_comes_back_idle(Registry):
    first = Registry()
    user = first.create_user("ivan")
    session = first.create_session(user["id"])
    first.touch_session(user["id"], session["id"], status="processing")

    # No run can still be alive in a new process — a stuck "processing" badge
    # would make the UI look busy forever.
    restored = Registry().get_session(user["id"], session["id"])
    assert restored["status"] == "idle"


def test_rename_is_persisted(Registry):
    first = Registry()
    user = first.create_user("ivan")
    session = first.create_session(user["id"], title="old")
    first.rename_session(user["id"], session["id"], "new title")
    assert Registry().get_session(user["id"], session["id"])["title"] == "new title"


def test_transcript_is_appended_and_reloaded(store):
    for i in range(3):
        assert store.append_event("u1", "s1", {"type": "agent_event", "n": i})
    assert store.has_events("u1", "s1")
    events = store.load_events("u1", "s1")
    assert [e["n"] for e in events] == [0, 1, 2]
    assert [e["n"] for e in store.load_events("u1", "s1", limit=2)] == [1, 2]


def test_a_torn_last_line_does_not_break_reading(store):
    store.append_event("u1", "s1", {"type": "agent_event", "n": 0})
    with open(store.events_path("u1", "s1"), "a", encoding="utf-8") as f:
        f.write('{"type": "agent_event", "n": 1')  # killed mid-write
    assert [e["n"] for e in store.load_events("u1", "s1")] == [0]


def test_missing_transcript_is_empty_not_an_error(store):
    assert store.load_events("nobody", "nothing") == []
    assert store.has_events("nobody", "nothing") is False


def test_ids_with_path_separators_cannot_escape_the_state_dir(store, tmp_path):
    store.append_event("../../etc", "../passwd", {"x": 1})
    written = list(tmp_path.rglob("*.jsonl"))
    assert written and all(tmp_path in p.parents for p in written)
