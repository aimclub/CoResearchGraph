import asyncio
import re
from unittest.mock import AsyncMock, patch

from google.adk.sessions import InMemorySessionService

from CoScientist.main import CoScientistManager


class _Runner:
    def __init__(
        self,
        *,
        session_service,
        app=None,
        agent=None,
        app_name=None,
    ):
        self.app = app
        self.agent = agent or getattr(app, "root_agent", None)
        self.app_name = app_name or getattr(app, "name", None)
        self.session_service = session_service


def test_manager_generates_random_local_ids():
    manager = CoScientistManager()
    assert re.fullmatch(r"user_[0-9a-f]{32}", manager.user_id)
    assert re.fullmatch(r"session_[0-9a-f]{32}", manager.session_id)


def test_managers_share_service_but_keep_sessions_separate():
    async def scenario():
        service = InMemorySessionService()
        with patch("CoScientist.main.Runner", _Runner):
            first = CoScientistManager(
                user_id="user_a",
                session_id="session_a",
                session_service=service,
            )
            second = CoScientistManager(
                user_id="user_a",
                session_id="session_b",
                session_service=service,
            )
            await first.initialize()
            await second.initialize()
            await first.initialize()  # idempotent

        first_session = await service.get_session(
            app_name="coscientist_app", user_id="user_a", session_id="session_a"
        )
        second_session = await service.get_session(
            app_name="coscientist_app", user_id="user_a", session_id="session_b"
        )
        assert first.session_service is second.session_service is service
        assert first.runner.session_service is second.runner.session_service is service
        assert first_session.state["active_tasks"] == []
        assert second_session.state["active_tasks"] == []

    asyncio.run(scenario())


def test_manager_close_awaits_runner_before_resetting_lifecycle_state():
    async def scenario():
        manager = CoScientistManager(
            user_id="user_close",
            session_id="session_close",
        )
        runner = _Runner(session_service=InMemorySessionService())

        async def assert_runner_is_still_attached_while_closing():
            assert manager.runner is runner
            assert manager._initialized is True

        runner.close = AsyncMock(
            side_effect=assert_runner_is_still_attached_while_closing
        )
        manager.runner = runner
        manager._initialized = True

        with patch("CoScientist.main.cleanup_uploaded_papers"):
            await manager.close()

        runner.close.assert_awaited_once_with()
        assert manager.runner is None
        assert manager._initialized is False

    asyncio.run(scenario())


def test_initial_state_is_seeded_without_dropping_the_defaults():
    """Callers that drive the system programmatically need state in place
    before the first turn; the manager's own session scoping stays authoritative."""

    async def scenario():
        service = InMemorySessionService()
        with patch("CoScientist.main.Runner", _Runner):
            manager = CoScientistManager(
                user_id="user_a",
                session_id="session_a",
                session_service=service,
                initial_state={"deployed_mcps": [{"url": "http://h:8000/mcp"}]},
            )
            await manager.initialize()
            return await service.get_session(
                app_name=manager.app_name,
                user_id="user_a",
                session_id="session_a",
            )

    state = asyncio.run(scenario()).state
    assert state["deployed_mcps"] == [{"url": "http://h:8000/mcp"}]
    assert state["active_tasks"] == []


def test_a_caller_cannot_redirect_the_session_scope():
    """Scope keys decide whose graph a run writes to — not the caller's to set."""

    async def scenario():
        service = InMemorySessionService()
        from CoScientist.graph.session_scope import GRAPH_SCOPE_USER_KEY

        with patch("CoScientist.main.Runner", _Runner):
            manager = CoScientistManager(
                user_id="user_a",
                session_id="session_a",
                session_service=service,
                initial_state={GRAPH_SCOPE_USER_KEY: "someone_else"},
            )
            await manager.initialize()
            session = await service.get_session(
                app_name=manager.app_name,
                user_id="user_a",
                session_id="session_a",
            )
            return session.state[GRAPH_SCOPE_USER_KEY]

    assert asyncio.run(scenario()) == "user_a"
