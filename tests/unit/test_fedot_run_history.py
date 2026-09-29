"""No FEDOT execution may be substituted with a different session's latest run."""
import asyncio
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from CoScientist.config import get_settings
from CoScientist.tools import fedot_runs
from CoScientist.tools.fedot_live import FedotLiveBroadcaster, FedotLivePlugin


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web"))
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("RESEARCH_GRAPH_DIR", str(tmp_path / "research"))
    monkeypatch.setattr(get_settings().web, "fedot_fallback_enabled", True)


def record(scope, task="first", status="success"):
    run = FedotLiveBroadcaster().begin_run(scope, task=task, engine="mas")
    run.publish_config({"coordinator": {"name": task}, "workers": []})
    run.event({"type": "text", "agent": task, "text": task * 5000})
    run.event({"type": "run_end", "status": status, "result": task})
    return run.run_id


def test_all_runs_survive_a_fresh_broadcaster_and_are_session_scoped():
    ids = {record(("alice", "a"), str(i)) for i in range(8)}
    other = record(("alice", "b"), "foreign")
    assert {r["run_id"] for r in fedot_runs.list_runs(("alice", "a"))} == ids
    assert fedot_runs.get_run(("alice", "a"), other) is None
    assert fedot_runs.list_runs(("bob", "a")) == []
    events, _ = fedot_runs.read_events(("alice", "b"), other)
    assert events[2]["text"] == "foreign" * 5000
    assert [e["seq"] for e in events] == [1, 2, 3, 4]


def test_another_process_can_read_persisted_history():
    run_id = record(("u", "s"))
    # Load the pure store in a fresh interpreter, without initialising agents.
    code = "import runpy,json,sys; m=runpy.run_path(sys.argv[1]); print(json.dumps(m['snapshot'](('u','s'))))"
    result = subprocess.run([sys.executable, "-c", code, str(Path(fedot_runs.__file__))],
                            capture_output=True, text=True, encoding="utf-8", check=True, timeout=15)
    restored = json.loads(result.stdout)
    assert restored["runs"][0]["meta"]["run_id"] == run_id
    assert restored["runs"][0]["events"][1]["type"] == "config"


def test_interleaved_runs_cannot_overwrite_each_others_graphs():
    bus = FedotLiveBroadcaster()
    first = bus.begin_run(("u", "s"))
    second = bus.begin_run(("u", "s"))
    second.publish_config({"name": "second"})
    first.publish_config({"name": "first"})
    second.event({"type": "run_end", "status": "success"})
    first.event({"type": "run_end", "status": "timeout"})
    for run, name in ((first, "first"), (second, "second")):
        events, _ = fedot_runs.read_events(("u", "s"), run.run_id)
        assert events[1]["config"] == {"name": name}
        assert all(e["run_id"] == run.run_id for e in events)


def test_scope_components_cannot_alias_or_escape():
    record(("a/b", "../session"))
    assert fedot_runs.list_runs(("a_b", "../session")) == []
    for bad in ("../run", "", "a" * 33):
        with pytest.raises(ValueError):
            fedot_runs.get_run(("u", "s"), bad)


def test_in_process_subscribers_are_also_scoped():
    bus = FedotLiveBroadcaster()
    first, second = bus.subscribe(("u", "a")), bus.subscribe(("u", "b"))
    run = bus.begin_run(("u", "a"), task="mine")
    assert first.get_nowait()["run_id"] == run.run_id
    assert second.empty()


def test_snapshot_preserves_runs_graphs_and_original_provenance():
    source, target = ("alice", "source"), ("bob", "imported")
    run_id = record(source)
    running = FedotLiveBroadcaster().begin_run(source, task="in progress")
    snapshot = fedot_runs.snapshot(source)
    fedot_runs.restore(target, snapshot)
    meta = fedot_runs.get_run(target, run_id)
    assert (meta["user_id"], meta["session_id"]) == target
    assert meta["origin"] == {"user_id": "alice", "session_id": "source", "run_id": run_id}
    assert fedot_runs.get_run(target, running.run_id)["status"] == "snapshot"
    before, _ = fedot_runs.read_events(source, run_id)
    after, _ = fedot_runs.read_events(target, run_id)
    assert after == [{**e, "user_id": target[0], "session_id": target[1]} for e in before]
    with pytest.raises(ValueError, match="overwrite"):
        fedot_runs.restore(target, snapshot)


def test_tail_reader_does_not_consume_a_partial_concurrent_write():
    scope = ("u", "s")
    run_id = record(scope)
    events, offset = fedot_runs.read_events(scope, run_id)
    path = fedot_runs._path(scope, run_id) / "events.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"type": "text"')
    assert fedot_runs.read_events(scope, run_id, offset) == ([], offset)
    with path.open("ab") as stream:
        stream.write(b'}\n')
    assert fedot_runs.read_events(scope, run_id, offset)[0] == [{"type": "text"}]


@pytest.fixture()
def client():
    from CoScientist.web.app import create_app
    with TestClient(create_app()) as client:
        yield client


def new_session(client, user_id=None):
    if user_id is None:
        user_id = client.post("/api/users", json={"nickname": "history"}).json()["user"]["id"]
    session = client.post(f"/api/users/{user_id}/sessions", json={"title": "FEDOT history"}).json()["session"]
    return user_id, session["id"]


def test_api_stream_and_trace_only_serve_selected_session(client):
    scope = new_session(client)
    other = new_session(client, scope[0])
    ids = [record(scope, "A"), record(scope, "B")]
    foreign = record(other, "SECRET")
    base = f"/api/users/{scope[0]}/sessions/{scope[1]}/fedot/runs"
    assert {r["run_id"] for r in client.get(base).json()["runs"]} == set(ids)
    assert client.get(base + "/" + foreign).status_code == 404
    params = {"user_id": scope[0], "session_id": scope[1], "run_id": ids[0]}
    trace = client.get("/api/fedot-langfuse-trace", params=params)
    assert trace.status_code == 200
    assert trace.json()["source"] == "local"
    assert "SECRET" not in trace.text
    stream = client.get("/api/fedot-live-stream", params=params)
    assert stream.status_code == 200 and "event: complete" in stream.text
    assert "SECRET" not in stream.text and ids[1] not in stream.text
    replay = client.get("/api/fedot-live-stream", params=params, headers={"Last-Event-ID": "2"})
    assert '"seq": 1' not in replay.text and '"seq": 2' not in replay.text
    assert '"seq": 3' in replay.text
    for url in ("/api/fedot-live-stream", "/api/fedot-langfuse-trace"):
        assert client.get(url, params={**params, "run_id": foreign}).status_code == 404
        assert client.get(url, params={**params, "session_id": "missing"}).status_code == 404
        assert client.get(url, params={"user_id": scope[0]}).status_code == 400


def test_full_session_zip_round_trip_and_legacy_bundle(client, monkeypatch):
    from CoScientist.web.session_bundle import export_session, import_session
    from CoScientist.tools import alembic_tools
    monkeypatch.setattr(alembic_tools, "export_jobs_snapshot", lambda: [])
    scope = new_session(client)
    ids = {record(scope, "one"), record(scope, "two", "timeout")}
    record(new_session(client, scope[0]), "DO NOT EXPORT")
    runtime = client.app.state.runtime
    bundle = asyncio.run(export_session(runtime, scope))
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        section = archive.read("fedot/runs.json")
        assert b"DO NOT EXPORT" not in section
        assert len(json.loads(section)["runs"]) == 2
        legacy_buffer = io.BytesIO()
        with zipfile.ZipFile(legacy_buffer, "w") as old:
            for name in archive.namelist():
                if name != "fedot/runs.json":
                    old.writestr(name, archive.read(name))
    imported = asyncio.run(import_session(runtime, scope[0], bundle))
    assert imported["user"]["id"] == scope[0]
    target = (scope[0], imported["session"]["id"])
    assert {r["run_id"] for r in fedot_runs.list_runs(target)} == ids
    assert all(r["imported"] for r in fedot_runs.list_runs(target))
    old = asyncio.run(import_session(runtime, scope[0], legacy_buffer.getvalue()))
    assert fedot_runs.list_runs((scope[0], old["session"]["id"])) == []


def test_model_trace_keeps_content_but_not_transport_credentials():
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types
    run = FedotLiveBroadcaster().begin_run(("u", "s"))
    plugin = FedotLivePlugin(run)
    context = SimpleNamespace(agent_name="worker")
    request = LlmRequest(model="test", contents=[types.Content(parts=[types.Part(text="input")])],
                         config=types.GenerateContentConfig(http_options=types.HttpOptions(headers={"Authorization": "SECRET"})))
    asyncio.run(plugin.before_model_callback(callback_context=context, llm_request=request))
    asyncio.run(plugin.after_model_callback(callback_context=context, llm_response=LlmResponse(content=types.Content(parts=[types.Part(text="output")]))))
    events, _ = fedot_runs.read_events(("u", "s"), run.run_id)
    assert "input" in json.dumps(events) and "output" in json.dumps(events)
    assert "SECRET" not in json.dumps(events)


@pytest.mark.parametrize("status", ["success", "timeout", "cancelled", "error"])
def test_tool_lifecycle_records_each_attempt_and_scope(monkeypatch, status):
    from CoScientist.tools.fedotmas_tools import FedotMASToolset
    toolset = FedotMASToolset()
    async def execute(task, context, live):
        if status == "cancelled":
            raise asyncio.CancelledError()
        if status == "error":
            raise RuntimeError("preparation failed")
        live.publish_config({"workers": []})
        return {"status": status, "result": {"state": {"answer": task}}, "artifacts": []}
    monkeypatch.setattr(toolset, "_execute_fedot", execute)
    context = SimpleNamespace(state={"graph_scope_user_id": "parent", "graph_scope_session_id": "session"},
                              _invocation_context=SimpleNamespace(session=SimpleNamespace(user_id="child", id="ephemeral")))
    for _ in range(2):
        try:
            asyncio.run(toolset.fedot_tool("same task", context))
        except (asyncio.CancelledError, RuntimeError):
            pass
    runs = fedot_runs.list_runs(("parent", "session"))
    assert len(runs) == 2 and all(r["status"] == status for r in runs)
    for meta in runs:
        events, _ = fedot_runs.read_events(("parent", "session"), meta["run_id"])
        assert events[0]["type"] == "run_start" and events[-1]["type"] == "run_end"
    assert fedot_runs.list_runs(("child", "ephemeral")) == []


def test_nonfinite_scientific_values_remain_valid_browser_json():
    run = FedotLiveBroadcaster().begin_run(("u", "s"))
    run.event({"type": "tool_result", "text": {"value": float("nan"), "limit": float("inf")}})
    events, _ = fedot_runs.read_events(("u", "s"), run.run_id)
    assert events[-1]["text"] == {"value": "nan", "limit": "inf"}


@pytest.mark.parametrize("value", [{}, {"version": 1, "runs": [None]},
                                    {"version": 1, "runs": [{"meta": {"run_id": "../escape"}}]}])
def test_malformed_history_is_rejected_before_import(value):
    with pytest.raises(ValueError):
        fedot_runs.restore(("u", "s"), value)
    assert fedot_runs.list_runs(("u", "s")) == []


def test_repeated_agent_names_do_not_nest_under_previous_attempt():
    from CoScientist.web.fedot_routes import local_trace
    run = FedotLiveBroadcaster().begin_run(("u", "s"), engine="mas")
    for span in ("first", "second"):
        run.event({"type": "agent_start", "agent": "worker", "span_id": span})
        run.event({"type": "agent_done", "agent": "worker", "span_id": span, "output": span})
    events, _ = fedot_runs.read_events(("u", "s"), run.run_id)
    trace = local_trace(fedot_runs.get_run(("u", "s"), run.run_id), events)
    agents = [o for o in trace["observations"] if o["type"] == "AGENT"]
    assert [o["output"] for o in agents] == ["first", "second"]
    assert all(o["parent_id"] == run.run_id for o in agents)


def test_langfuse_link_is_captured_before_sdk_finalization_and_only_once():
    run = FedotLiveBroadcaster().begin_run(("u", "s"))
    sdk = SimpleNamespace(_trace=SimpleNamespace(trace_id="exact-trace", id="fedot-root"))
    plugin = FedotLivePlugin(run, trace_plugin=sdk)
    asyncio.run(plugin.before_run_callback(invocation_context=None))
    asyncio.run(plugin.after_run_callback(invocation_context=None))
    sdk._trace = None
    events, _ = fedot_runs.read_events(("u", "s"), run.run_id)
    links = [e for e in events if e["type"] == "langfuse_link"]
    assert len(links) == 1
    assert (links[0]["trace_id"], links[0]["observation_id"]) == ("exact-trace", "fedot-root")
