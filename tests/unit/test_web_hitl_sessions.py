import asyncio

import pytest

from CoScientist.hitl.models import HITLAction, HITLRequest
from CoScientist.web.handler import WebHITLHandler


@pytest.fixture(autouse=True)
def _session_store(tmp_path, monkeypatch):
    """Keep the documents these requests publish out of the working tree.

    `handle_request` writes each request's body into the session's artifact
    directory; without this the suite grows `graph_runs/sessions/user_a/` in
    the repo every time it runs.
    """
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")


class _Socket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)

async def _delivered(socket, *, ticks: int = 200):
    """Wait for the card to reach this socket.

    `handle_request` publishes the request's body as a session document before
    it broadcasts — a thread hop, not a single loop tick — so a bare
    `asyncio.sleep(0)` no longer proves anything either way.
    """
    for _ in range(ticks):
        if socket.messages:
            return socket.messages
        await asyncio.sleep(0.005)
    raise AssertionError("no HITL card was delivered")


def test_hitl_request_and_response_are_scoped_to_session():
    async def scenario():
        handler = WebHITLHandler()
        first_key = ("user_a", "session_a")
        second_key = ("user_b", "session_b")
        first_socket = _Socket()
        second_socket = _Socket()
        await handler.attach_websocket(first_socket, first_key)
        await handler.attach_websocket(second_socket, second_key)

        request_task = asyncio.create_task(handler.handle_request(HITLRequest(
            agent_name="PlannerAgent",
            action_type=HITLAction.APPROVE,
            message="Approve plan",
            context={
                "output": "plan",
                "_session": {"user_id": first_key[0], "session_id": first_key[1]},
            },
        )))
        await _delivered(first_socket)

        assert len(first_socket.messages) == 1
        assert second_socket.messages == []
        payload = first_socket.messages[0]
        assert "_session" not in payload["context"]

        assert not handler.resolve_request(
            payload["request_id"],
            {"action": "approve", "approved": True},
            second_key,
        )
        assert not request_task.done()
        assert handler.resolve_request(
            payload["request_id"],
            {"action": "approve", "approved": True},
            first_key,
        )
        response = await request_task
        assert response.approved

    asyncio.run(scenario())


def test_unresolved_hitl_is_redelivered_only_until_it_is_resolved():
    async def scenario():
        handler = WebHITLHandler()
        key = ("user_a", "session_a")
        first_socket = _Socket()
        await handler.attach_websocket(first_socket, key)

        request_task = asyncio.create_task(handler.handle_request(HITLRequest(
            agent_name="PlannerAgent",
            action_type=HITLAction.APPROVE,
            message="Approve plan",
            context={"_session": {"user_id": key[0], "session_id": key[1]}},
        )))
        await asyncio.sleep(0)
        request_id = first_socket.messages[0]["request_id"]

        handler.detach_websocket(first_socket, key)
        reconnect_socket = _Socket()
        await handler.attach_websocket(reconnect_socket, key)
        assert [message["request_id"] for message in reconnect_socket.messages] == [
            request_id
        ]

        assert handler.resolve_request(
            request_id,
            {"action": "approve", "approved": True},
            key,
        )
        assert (await request_task).approved

        late_socket = _Socket()
        await handler.attach_websocket(late_socket, key)
        assert late_socket.messages == []

    asyncio.run(scenario())


def test_a_required_human_is_not_bypassed_by_auto_mode(monkeypatch):
    monkeypatch.setenv("HITL__MODE", "auto")

    async def scenario():
        handler = WebHITLHandler()
        key = ("user_a", "session_a")
        socket = _Socket()
        await handler.attach_websocket(socket, key)
        task = asyncio.create_task(handler.handle_request(HITLRequest(
            agent_name="ExperimentPlannerAgent",
            action_type=HITLAction.APPROVE,
            message="Approve the exhausted plan",
            context={"_session": {"user_id": key[0], "session_id": key[1]}},
            timeout_seconds=10,
            requires_human=True,
        )))

        await _delivered(socket)
        assert not task.done()
        payload = socket.messages[0]
        assert payload["requires_human"] is True
        assert payload["timeout_seconds"] == 10
        assert handler.resolve_request(
            payload["request_id"],
            {"action": "approve", "approved": True},
            key,
        )
        response = await task
        assert response.approved
        assert response.decision_source.value == "human"

    asyncio.run(scenario())


def test_logical_review_id_is_stable_and_concurrent_replays_share_one_card():
    async def run_once(*, duplicate: bool):
        handler = WebHITLHandler()
        key = ("user_a", "session_a")
        socket = _Socket()
        await handler.attach_websocket(socket, key)
        request = HITLRequest(
            agent_name="ExperimentPlannerAgent",
            action_type=HITLAction.APPROVE,
            message="Approve the exhausted plan",
            context={
                "experiment_review_id": "plan-fallback:run-1:plan-1:r4:digest",
                "_session": {"user_id": key[0], "session_id": key[1]},
            },
            timeout_seconds=10,
            requires_human=True,
        )
        tasks = [asyncio.create_task(handler.handle_request(request))]
        if duplicate:
            tasks.append(asyncio.create_task(handler.handle_request(request)))
        await _delivered(socket)
        await asyncio.sleep(0)
        assert len(socket.messages) == 1
        request_id = socket.messages[0]["request_id"]
        assert handler.resolve_request(
            request_id,
            {"action": "approve", "approved": True},
            key,
        )
        responses = await asyncio.gather(*tasks)
        assert all(response.approved for response in responses)
        return request_id

    async def scenario():
        first = await run_once(duplicate=True)
        # A new handler models a process restart.  The same durable review
        # identity must map to the same browser-card identity.
        second = await run_once(duplicate=False)
        assert first == second

    asyncio.run(scenario())


def test_cancelled_hitl_is_removed_and_not_redelivered_on_reconnect():
    async def scenario():
        handler = WebHITLHandler()
        key = ("user_a", "session_a")
        first_socket = _Socket()
        await handler.attach_websocket(first_socket, key)

        request_task = asyncio.create_task(handler.handle_request(HITLRequest(
            agent_name="PlannerAgent",
            action_type=HITLAction.APPROVE,
            message="Approve plan",
            context={"_session": {"user_id": key[0], "session_id": key[1]}},
        )))
        await asyncio.sleep(0)
        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task

        assert handler.pending_summary() == []
        assert first_socket.messages[-1]["type"] == "hitl_cancelled"

        reconnect_socket = _Socket()
        await handler.attach_websocket(reconnect_socket, key)
        assert reconnect_socket.messages == []

    asyncio.run(scenario())


def _request(key, timeout_seconds=None):
    return HITLRequest(
        agent_name="ResearchAgent",
        action_type=HITLAction.APPROVE,
        message="Work order",
        context={"_session": {"user_id": key[0], "session_id": key[1]}},
        trigger="work_order",
        timeout_seconds=timeout_seconds,
    )


def test_global_wait_overrides_the_legacy_request_timeout(monkeypatch):
    """A configured veto window is shorter than the run's mode allows: the
    request's own timeout_seconds must win. Running out no longer approves —
    silence is a refusal everywhere, and a run that must proceed unattended
    says so with HITL__MODE=auto instead."""
    import CoScientist.hitl.mode as mode_mod

    monkeypatch.setattr(mode_mod, "wait_seconds", lambda: 0.05)

    async def scenario():
        handler = WebHITLHandler()
        key = ("user_a", "session_a")
        socket = _Socket()
        await handler.attach_websocket(socket, key)

        response = await asyncio.wait_for(
            handler.handle_request(_request(key, timeout_seconds=300)), timeout=2
        )
        assert response.approved is False
        assert response.timed_out is True
        assert socket.messages[0]["timeout_seconds"] == 0.05
        assert socket.messages[-1]["type"] == "hitl_timeout"

    asyncio.run(scenario())


def test_hold_stops_the_auto_approve_countdown():
    async def scenario():
        handler = WebHITLHandler()
        key = ("user_a", "session_a")
        other_key = ("user_b", "session_b")
        socket = _Socket()
        await handler.attach_websocket(socket, key)

        task = asyncio.create_task(handler.handle_request(_request(key, timeout_seconds=0.1)))
        await asyncio.sleep(0)
        request_id = socket.messages[0]["request_id"]

        assert not await handler.hold_request(request_id, other_key)
        assert await handler.hold_request(request_id, key)
        assert socket.messages[-1] == {"type": "hitl_hold", "request_id": request_id}

        await asyncio.sleep(0.25)
        assert not task.done(), "a held request must wait for the human"

        # A reconnecting tab gets the request without a countdown.
        late = _Socket()
        await handler.attach_websocket(late, key)
        assert late.messages[0]["timeout_seconds"] == 0
        assert late.messages[0]["held"] is True

        assert handler.resolve_request(request_id, {"action": "reject", "approved": False}, key)
        response = await task
        assert not response.approved

    asyncio.run(scenario())


def test_notify_reaches_only_its_session_and_is_logged():
    async def scenario():
        handler = WebHITLHandler()
        key = ("user_a", "session_a")
        other_key = ("user_b", "session_b")
        mine, other = _Socket(), _Socket()
        await handler.attach_websocket(mine, key)
        await handler.attach_websocket(other, other_key)

        await handler.notify({
            "kind": "progress",
            "agent_name": "ResearchAgent",
            "step": {"id": "S1", "status": "done"},
            "_session": {"user_id": key[0], "session_id": key[1]},
        })

        assert other.messages == []
        assert mine.messages[0]["type"] == "work_order_notice"
        assert mine.messages[0]["kind"] == "progress"
        assert "_session" not in mine.messages[0]
        assert handler.get_event_log(key)[0]["type"] == "work_order_notice"
        assert handler.get_event_log(other_key) == []

    asyncio.run(scenario())


def test_hitl_request_and_its_answer_are_recorded_in_the_session_transcript(monkeypatch):
    """A reload, export or import rebuilds the chat from the transcript, so an
    answered HITL card and the answer must both be in it — and only in its own
    session's transcript."""
    import CoScientist.hitl.mode as mode_mod

    window = [5.0]
    monkeypatch.setattr(mode_mod, "wait_seconds", lambda: window[0])

    async def scenario():
        handler = WebHITLHandler()
        recorded = []
        handler.set_recorder(lambda session_key, event: recorded.append((session_key, event)))
        key = ("user_a", "session_a")
        socket = _Socket()
        await handler.attach_websocket(socket, key)

        task = asyncio.create_task(handler.handle_request(_request(key, timeout_seconds=5)))
        await asyncio.sleep(0)
        request_id = socket.messages[0]["request_id"]
        await handler.hold_request(request_id, key)
        assert handler.resolve_request(request_id, {
            "action": "approve",
            "approved": True,
            "instructions": "go",
            "form_values": {"rejected_assumption_ids": ["A1"]},
        }, key)
        await task

        assert [session_key for session_key, _ in recorded] == [key, key]
        request_event, response_event = (event for _, event in recorded)
        assert request_event["type"] == "hitl_request"
        assert request_event["request_id"] == request_id
        # Recorded as sent: a later hold must not rewrite the history copy.
        assert request_event["timeout_seconds"] == 5
        assert "held" not in request_event
        assert response_event["type"] == "hitl_response"
        assert response_event["request_id"] == request_id
        assert response_event["instructions"] == "go"
        assert response_event["form_values"] == {"rejected_assumption_ids": ["A1"]}

        window[0] = 0.05
        timed_out = await asyncio.wait_for(
            handler.handle_request(_request(key, timeout_seconds=0.05)), timeout=2
        )
        # Silence is a refusal, and the transcript says so: an expiry that used
        # to be recorded as an approval was indistinguishable, on a reload, from
        # a human having pressed the button.
        assert timed_out.approved is False
        assert timed_out.timed_out is True
        assert recorded[-1][1]["type"] == "hitl_timeout"
        assert recorded[-1][1]["paused"] is True

    asyncio.run(scenario())


def test_the_structured_experiment_plan_reaches_the_browser():
    """The plan card is drawn from ``context.experiment_plan``.

    The handler forwards the request context verbatim minus ``_session``, so
    this pins that the plan is not stripped on the way out — the browser falls
    back to the Markdown blob when it is missing, which is the view this
    replaced.
    """
    async def scenario():
        handler = WebHITLHandler()
        key = ("user_a", "session_a")
        socket = _Socket()
        await handler.attach_websocket(socket, key)

        plan = {"kind": "experiment_plan", "revision": 2, "task_count": 3,
                "matrix": [{"task_id": "EXP-1"}], "tasks": [{"id": "EXP-1"}]}
        task = asyncio.create_task(handler.handle_request(HITLRequest(
            agent_name="ExperimentPlannerAgent",
            action_type=HITLAction.APPROVE,
            message="Review and explicitly approve the experiment plan.",
            context={
                "output": "# Experiment plan · revision 2",
                "experiment_review_kind": "plan",
                "experiment_plan": plan,
                "_session": {"user_id": key[0], "session_id": key[1]},
            },
        )))
        await _delivered(socket)

        context = socket.messages[0]["context"]
        assert context["experiment_plan"] == plan
        assert context["output"].startswith("# Experiment plan")
        assert "_session" not in context

        handler.resolve_request(socket.messages[0]["request_id"],
                                {"action": "approve", "approved": True}, key)
        await task

    asyncio.run(scenario())


def test_a_review_arrives_as_a_summary_and_a_document():
    """The card gets a way in; the wall of text goes to a file.

    The body used to be the message. A seven-task plan is thirty kilobytes, and
    printed into the feed it buries every message around it, so the request
    carries a summary and the id of a document the panel opens instead.
    """
    async def scenario():
        handler = WebHITLHandler()
        key = ("user_doc", "session_doc")
        socket = _Socket()
        await handler.attach_websocket(socket, key)

        plan = {"revision": 2, "task_count": 7, "goal": "Построить профиль токсичности."}
        # Document-sized: a body that fits in a chat message stays in the chat
        # message (reporting.documents.MIN_DOCUMENT_CHARS).
        body = ("# План эксперимента · ревизия 2\n\nСемь задач, оценка 195 минут.\n\n"
                + "## Задачи\n\n" + "".join(
                    f"- **EXP-{i}** Задача номер {i}, структурная кластеризация.\n"
                    for i in range(1, 9)))
        task = asyncio.create_task(handler.handle_request(HITLRequest(
            agent_name="ExperimentPlannerAgent",
            action_type=HITLAction.APPROVE,
            message="Review the plan.",
            context={
                "output": body,
                "experiment_plan": plan,
                "_session": {"user_id": key[0], "session_id": key[1]},
            },
        )))
        await _delivered(socket)
        payload = socket.messages[0]

        document = payload["document"]
        assert document["kind"] == "plan"
        assert document["title"] == "План эксперимента · ревизия 2"
        # The plan states its own goal in the operator's language; that beats
        # anything derived from the rendered document.
        assert payload["summary"] == "Построить профиль токсичности."
        assert len(payload["summary"]) < len(body)

        # The document is a real file in this session, openable by the route.
        from CoScientist.reporting import session_files as sf

        assert sf.has_artifact(key, document["artifact_id"])
        stored = sf.resolve_path(key, document["artifact_id"]).read_text(encoding="utf-8")
        assert stored == body.strip()
        record = sf.load_manifest(key[1], key[0])[document["artifact_id"]]
        assert record["media_type"].startswith("text/")

        # The body stays in `context` too: nothing downstream loses the text.
        assert payload["context"]["output"] == body

        handler.resolve_request(payload["request_id"],
                                {"action": "approve", "approved": True}, key)
        await task

    asyncio.run(scenario())


def test_a_request_with_no_body_is_unchanged():
    """No body, no document, no dead button."""
    async def scenario():
        handler = WebHITLHandler()
        key = ("user_bare", "session_bare")
        socket = _Socket()
        await handler.attach_websocket(socket, key)

        task = asyncio.create_task(handler.handle_request(HITLRequest(
            agent_name="CoderAgent",
            action_type=HITLAction.APPROVE,
            message="Run this?",
            context={"_session": {"user_id": key[0], "session_id": key[1]}},
        )))
        await _delivered(socket)
        payload = socket.messages[0]

        assert "document" not in payload
        assert "summary" not in payload

        handler.resolve_request(payload["request_id"],
                                {"action": "approve", "approved": True}, key)
        await task

    asyncio.run(scenario())
