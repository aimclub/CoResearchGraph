import asyncio

import pytest

from CoScientist.hitl.models import HITLAction, HITLRequest
from CoScientist.web.handler import WebHITLHandler


class _Socket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


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
        await asyncio.sleep(0)

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
