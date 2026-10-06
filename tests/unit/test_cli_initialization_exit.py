"""Interactive CLI exit status reflects persisted initialization failure."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("error,expected", [("", 0), ("Catalog constraints missing", 1)])
def test_cli_initialization_failure_exit_status(monkeypatch, error, expected):
    from CoScientist import cli, main
    from CoScientist.utils import interrupt

    closed = []

    async def run(query):
        assert query == "research request"
        return "frame result"

    async def get_session(**kwargs):
        assert kwargs == {"app_name": "app", "user_id": "user", "session_id": "session"}
        return SimpleNamespace(state={"research_frame_initialization_error": error})

    async def close():
        closed.append(True)

    async def create_manager():
        return SimpleNamespace(
            run=run, close=close, app_name="app", user_id="user", session_id="session",
            session_service=SimpleNamespace(get_session=get_session),
        )

    queries = iter(["research request", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(queries))
    monkeypatch.setattr(main, "create_manager", create_manager)
    monkeypatch.setattr(interrupt, "install_sigint_exit", lambda: None)
    monkeypatch.setattr(cli, "_configure_utf8_stdio", lambda: None)
    assert cli.main(["cli"]) == expected
    assert closed == [True]
