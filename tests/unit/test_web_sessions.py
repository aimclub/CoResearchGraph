import asyncio
import importlib
import json

from fastapi.testclient import TestClient

from CoScientist.web.app import APP_NAME, create_app

web_app = importlib.import_module("CoScientist.web.app")


def _create_user(client, nickname):
    response = client.post("/api/users", json={"nickname": nickname})
    assert response.status_code == 201
    return response.json()["user"]


def _create_session(client, user_id, title):
    response = client.post(
        f"/api/users/{user_id}/sessions",
        json={"title": title},
    )
    assert response.status_code == 201
    return response.json()["session"]


def test_websocket_reconnect_receives_same_session_snapshot():
    app = create_app()
    with TestClient(app) as client:
        user = _create_user(client, "Gleb")
        session = _create_session(client, user["id"], "Work")
        key = (user["id"], session["id"])
        app.state.runtime.agent_events[key].append({
            "type": "user_message",
            "message": "previous question",
            "timestamp": "2026-01-01T10:00:00",
        })

        url = f"/ws?user_id={user['id']}&session_id={session['id']}"
        with client.websocket_connect(url) as websocket:
            assert websocket.receive_json()["type"] == "connected"
            snapshot = websocket.receive_json()
            assert snapshot["type"] == "session_snapshot"
            assert snapshot["user"]["nickname"] == "Gleb"
            assert snapshot["messages"][0]["message"] == "previous question"

        with client.websocket_connect(url) as websocket:
            websocket.receive_json()
            snapshot = websocket.receive_json()
            assert snapshot["session"]["id"] == session["id"]
            assert snapshot["messages"][0]["message"] == "previous question"


def test_runtime_rejects_a_second_run_and_stop_cleans_only_its_session(monkeypatch):
    async def scenario():
        runtime = web_app.WebRuntime()
        first_key = ("user_a", "session_a")
        second_key = ("user_b", "session_b")
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked_chat(_runtime, key, data):
            assert key == first_key
            assert data["message"] == "first"
            started.set()
            await release.wait()

        monkeypatch.setattr(web_app, "_handle_chat", blocked_chat)
        assert await runtime.start_run(first_key, {"message": "first"})
        await started.wait()
        owner = runtime.active_runs[first_key]

        assert not await runtime.start_run(first_key, {"message": "second"})
        assert runtime.active_runs[first_key] is owner

        first_wait = asyncio.Event()
        second_wait = asyncio.Event()
        runtime.pending_hitl["first"] = {
            "event": first_wait,
            "response": None,
            "session_key": first_key,
        }
        runtime.pending_hitl["second"] = {
            "event": second_wait,
            "response": None,
            "session_key": second_key,
        }

        assert await runtime.stop_run(first_key)
        assert first_key not in runtime.active_runs
        assert first_wait.is_set()
        assert "first" not in runtime.pending_hitl
        assert not second_wait.is_set()
        assert "second" in runtime.pending_hitl

        web_app._cancel_pending_hitl(runtime)

    asyncio.run(scenario())


def test_finished_run_cannot_discard_a_new_owner():
    async def scenario():
        runtime = web_app.WebRuntime()
        key = ("user_a", "session_a")
        old_owner = asyncio.create_task(asyncio.sleep(0))
        await old_owner
        release = asyncio.Event()
        new_owner = asyncio.create_task(release.wait())
        runtime.active_runs[key] = new_owner

        assert not await runtime.discard_run(key, old_owner)
        assert runtime.active_runs[key] is new_owner

        new_owner.cancel()
        await asyncio.gather(new_owner, return_exceptions=True)

    asyncio.run(scenario())


def test_chat_controls_follow_server_status_broadcasts():
    chat_js = (web_app.WEB_DIR / "static" / "js" / "chat.js").read_text(encoding="utf-8")
    ws_js = (web_app.WEB_DIR / "static" / "js" / "ws.js").read_text(encoding="utf-8")
    sessions_js = (web_app.WEB_DIR / "static" / "js" / "sessions.js").read_text(encoding="utf-8")
    submit_handler = chat_js.split(
        "document.getElementById('chat-form').addEventListener", 1
    )[1].split("function stopChat", 1)[0]
    stop_handler = chat_js.split("function stopChat", 1)[1].split(
        "function applyReportLanguage", 1
    )[0]

    assert "function applyRunStatus(status, version = null)" in sessions_js
    assert "parsedVersion < runStatusVersion" in sessions_js
    assert "case 'status':" in ws_js
    assert "applyRunStatus(data.status, data.run_status_version);" in ws_js
    assert "case 'chat_accepted':" in ws_js
    assert "addUserMsg(msg);" not in submit_handler
    assert "input.value = '';" not in submit_handler
    assert "send-btn').disabled" not in submit_handler
    assert "send-btn').disabled" not in stop_handler
    assert "stop-btn').classList" not in stop_handler


def test_initial_snapshot_is_ordered_before_live_socket_broadcasts():
    class SlowSocket:
        def __init__(self):
            self.messages = []
            self.first_send_started = asyncio.Event()
            self.release_first_send = asyncio.Event()

        async def send_json(self, payload):
            if not self.messages:
                self.first_send_started.set()
                await self.release_first_send.wait()
            self.messages.append(payload)

    async def scenario():
        runtime = web_app.WebRuntime()
        key = ("user_a", "session_a")
        socket = SlowSocket()
        attach = asyncio.create_task(runtime.attach_with_snapshot(
            key,
            socket,
            user={"id": key[0], "nickname": "A"},
            session={"id": key[1]},
            active_tasks=[],
        ))
        await socket.first_send_started.wait()
        live = asyncio.create_task(runtime.send(
            key,
            runtime.status_payload(key, "processing", "newer status"),
        ))
        await asyncio.sleep(0)
        socket.release_first_send.set()
        await asyncio.gather(attach, live)

        assert [message["type"] for message in socket.messages] == [
            "connected",
            "session_snapshot",
            "status",
        ]

    asyncio.run(scenario())


def test_stop_releases_control_lock_before_slow_socket_delivery():
    async def scenario():
        runtime = web_app.WebRuntime()
        key = ("user_a", "session_a")
        owner = asyncio.create_task(asyncio.Event().wait())
        runtime.active_runs[key] = owner
        send_started = asyncio.Event()
        release_send = asyncio.Event()

        async def slow_send(_key, _payload):
            send_started.set()
            await release_send.wait()

        runtime.send = slow_send
        stopping = asyncio.create_task(runtime.stop_run(key))
        await send_started.wait()

        lock = runtime.control_lock(key)
        await asyncio.wait_for(lock.acquire(), timeout=0.1)
        lock.release()
        release_send.set()
        assert await stopping

    asyncio.run(scenario())


def test_runtime_rejects_new_runs_after_shutdown_begins():
    async def scenario():
        runtime = web_app.WebRuntime()
        runtime._closing = True
        assert not await runtime.start_run(
            ("user_a", "session_a"),
            {"message": "must not start"},
        )
        assert runtime.active_runs == {}

    asyncio.run(scenario())


def test_sandbox_links_reach_only_the_owning_tab_as_they_happen():
    """The sandbox call runs for as long as the job does, so the console links
    have to be pushed when the container comes up — not when the tool returns."""
    from CoScientist.tools.coder_tools import sandbox_tools

    class Socket:
        def __init__(self):
            self.messages = []

        async def send_json(self, payload):
            self.messages.append(payload)

    async def scenario():
        runtime = web_app.WebRuntime()
        web_app._wire_sandbox_links(runtime)
        try:
            owner, other = ("user_a", "session_a"), ("user_b", "session_b")
            owner_socket, other_socket = Socket(), Socket()
            runtime.attach_socket(owner, owner_socket)
            runtime.attach_socket(other, other_socket)

            info = {"sandbox_id": "s1", "watch_url": "http://box/live",
                    "vscode_url": "http://box/code", "reused": False}
            await sandbox_tools._start_sink(owner, info)

            assert len(owner_socket.messages) == 1
            event = owner_socket.messages[0]
            assert event["type"] == "agent_event"
            assert "http://box/live" in event["content"]
            assert "http://box/code" in event["content"]
            assert not other_socket.messages          # session isolation

            # A reconnecting tab must still find the links in its history.
            assert runtime.agent_events[owner] == [event]

            # A follow-up task reuses the container; don't repost the links.
            await sandbox_tools._start_sink(owner, {**info, "reused": True})
            assert len(owner_socket.messages) == 1

            # A genuinely new sandbox is announced again.
            await sandbox_tools._start_sink(owner, {**info, "sandbox_id": "s2"})
            assert len(owner_socket.messages) == 2
        finally:
            sandbox_tools.set_sandbox_start_sink(None)

    asyncio.run(scenario())


def test_key_agent_output_reaches_the_owning_tab_and_its_history():
    """A subordinate's deliverable is posted as its own chat message.

    It arrives at the caller as an AgentTool result, so without this it would
    only ever be visible as a truncated preview in the tool-activity rail.
    """
    from CoScientist.logging import agent_output

    class Socket:
        def __init__(self):
            self.messages = []

        async def send_json(self, payload):
            self.messages.append(payload)

    async def scenario():
        runtime = web_app.WebRuntime()
        web_app._wire_agent_output(runtime)
        try:
            owner, other = ("user_a", "session_a"), ("user_b", "session_b")
            owner_socket, other_socket = Socket(), Socket()
            runtime.attach_socket(owner, owner_socket)
            runtime.attach_socket(other, other_socket)

            await agent_output._sink(owner, {
                "agent": "HypothesesAgent",
                "caller": "OrchestratorAgent",
                "content": "H1: ...\nH2: ...",
                "timestamp": "2026-07-30T12:00:00",
            })

            (event,) = owner_socket.messages
            assert event["type"] == "agent_output"
            assert event["agent"] == "HypothesesAgent"
            assert event["content"] == "H1: ...\nH2: ..."
            assert not other_socket.messages          # session isolation

            # A reconnecting tab must find the deliverable in its history.
            assert runtime.agent_events[owner] == [event]

            # A session nobody is watching (e.g. the CLI scope) is dropped.
            await agent_output._sink(("cli", "default"), {
                "agent": "HypothesesAgent", "content": "H3",
            })
            assert ("cli", "default") not in runtime.agent_events
        finally:
            agent_output.set_agent_output_sink(None)

    asyncio.run(scenario())


def test_dataset_link_is_validated_stored_and_broadcast_per_session():
    """The chat's "+" attachment: only a .zip, per session, visible on reconnect."""
    app = create_app()
    runtime = app.state.runtime
    with TestClient(app) as client:
        user = _create_user(client, "Gleb")
        session = _create_session(client, user["id"], "Work")
        other = _create_session(client, user["id"], "Other")
        key = (user["id"], session["id"])
        url = f"/ws?user_id={user['id']}&session_id={session['id']}"

        with client.websocket_connect(url) as websocket:
            websocket.receive_json()                       # connected
            assert websocket.receive_json()["dataset_url"] == ""

            # A link that is not an http(s) .zip never reaches the session.
            for bad in ("ftp://host/data.zip", "https://host/data.tar.gz", "nonsense"):
                websocket.send_json({"type": "set_dataset_url", "dataset_url": bad})
                rejected = websocket.receive_json()
                assert rejected["type"] == "dataset_url_rejected"
                assert rejected["message"]
            assert key not in runtime.dataset_urls

            good = "https://example.org/data/dataset.zip"
            websocket.send_json({"type": "set_dataset_url", "dataset_url": good})
            accepted = websocket.receive_json()
            assert accepted == {"type": "dataset_url", "dataset_url": good}
            assert runtime.dataset_urls[key] == good

        # Mirrored into ADK state, where the coder's prompt/tool callbacks read it.
        adk_session = runtime.session_service.sessions[APP_NAME][user["id"]][session["id"]]
        assert adk_session.state[web_app.DATASET_URL_STATE_KEY] == good

        # A reconnecting tab gets it back; a sibling session is unaffected.
        with client.websocket_connect(url) as websocket:
            websocket.receive_json()
            assert websocket.receive_json()["dataset_url"] == good
        with client.websocket_connect(
            f"/ws?user_id={user['id']}&session_id={other['id']}"
        ) as websocket:
            websocket.receive_json()
            assert websocket.receive_json()["dataset_url"] == ""

        # Detaching clears both the runtime mirror and the agent-visible state.
        with client.websocket_connect(url) as websocket:
            websocket.receive_json()
            websocket.receive_json()
            websocket.send_json({"type": "set_dataset_url", "dataset_url": ""})
            assert websocket.receive_json() == {"type": "dataset_url", "dataset_url": ""}
        assert key not in runtime.dataset_urls
        assert adk_session.state[web_app.DATASET_URL_STATE_KEY] == ""


def test_report_language_is_validated_stored_and_broadcast_per_session():
    """The composer's language picker: a closed enum, per session, on reconnect."""
    app = create_app()
    runtime = app.state.runtime
    with TestClient(app) as client:
        user = _create_user(client, "Gleb")
        session = _create_session(client, user["id"], "Work")
        other = _create_session(client, user["id"], "Other")
        key = (user["id"], session["id"])
        url = f"/ws?user_id={user['id']}&session_id={session['id']}"

        with client.websocket_connect(url) as websocket:
            websocket.receive_json()                       # connected
            # Empty, not "ru": the server must not pick for the browser, or the
            # UI default could never follow the interface language.
            assert websocket.receive_json()["report_language"] == ""

            # Anything outside the enum is refused before it reaches the session.
            for bad in ("de", "russian", "en-US"):
                websocket.send_json({
                    "type": "set_report_language", "report_language": bad,
                })
                rejected = websocket.receive_json()
                assert rejected["type"] == "report_language_rejected"
                assert rejected["message"]
            assert key not in runtime.report_languages

            websocket.send_json({"type": "set_report_language", "report_language": "en"})
            accepted = websocket.receive_json()
            assert accepted == {"type": "report_language", "report_language": "en"}
            assert runtime.report_languages[key] == "en"

        # Mirrored into ADK state, where inject_report_language reads it.
        adk_session = runtime.session_service.sessions[APP_NAME][user["id"]][session["id"]]
        assert adk_session.state[web_app.REPORT_LANGUAGE_STATE_KEY] == "en"

        # A reconnecting tab gets it back; a sibling session is unaffected.
        with client.websocket_connect(url) as websocket:
            websocket.receive_json()
            assert websocket.receive_json()["report_language"] == "en"
        with client.websocket_connect(
            f"/ws?user_id={user['id']}&session_id={other['id']}"
        ) as websocket:
            websocket.receive_json()
            assert websocket.receive_json()["report_language"] == ""

        # Clearing drops the mirror and empties the agent-visible state, which
        # hands the session back to the callback's default.
        with client.websocket_connect(url) as websocket:
            websocket.receive_json()
            websocket.receive_json()
            websocket.send_json({"type": "set_report_language", "report_language": ""})
            assert websocket.receive_json() == {
                "type": "report_language", "report_language": "",
            }
        assert key not in runtime.report_languages
        assert adk_session.state[web_app.REPORT_LANGUAGE_STATE_KEY] == ""


def test_truncated_tool_result_is_stashed_and_fetchable_on_demand():
    """A truncated tool result never goes out whole on the socket, but the
    ToolsViewer's "Show full result" can still fetch it afterwards — the
    untruncated value is kept server-side, keyed by call_id, instead.
    """
    from CoScientist.logging import tool_activity

    class Socket:
        def __init__(self):
            self.messages = []

        async def send_json(self, payload):
            self.messages.append(payload)

    app = create_app()
    runtime = app.state.runtime
    with TestClient(app) as client:
        user = _create_user(client, "Nadia")
        session = _create_session(client, user["id"], "Big results")
        key = (user["id"], session["id"])
        socket = Socket()
        runtime.attach_socket(key, socket)

        full_result = {"status": "success", "result": ["tool"] * 500}
        asyncio.run(tool_activity._sink(key, {
            "phase": "result",
            "author": "ToolRetrieverAgent",
            "tool": "retrieve_tools",
            "call_id": "fc_42",
            "result": "{\"status\": \"success\" …",
            "result_truncated": True,
            "result_full": full_result,
        }))

        # The broadcast preview must not carry the full payload.
        (event,) = socket.messages
        assert event["result_truncated"] is True
        assert "result_full" not in event
        assert runtime.tool_full_values[key]["fc_42"]["result"] == full_result

        response = client.get(
            f"/api/users/{user['id']}/sessions/{session['id']}/tool-activity/fc_42"
        )
        assert response.status_code == 200
        assert response.json()["result"] == full_result

        # An id nobody stashed a full value under (never truncated, wrong
        # session, made up) is a 404, not an empty success.
        missing = client.get(
            f"/api/users/{user['id']}/sessions/{session['id']}/tool-activity/no-such-call"
        )
        assert missing.status_code == 404
