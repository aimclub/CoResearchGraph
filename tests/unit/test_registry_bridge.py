"""A built MCP server has to reach two places: the durable catalogue (so later
runs find it) and this run's state (so the executor can call it now). They
happen at different moments — the catalogue when the build finishes, the state
when a session polls it — so they are tested apart. No network, no database:
the rag_tools manager is injected.
"""

import asyncio
import json

import pytest

from CoScientist.tools.registry_bridge import register_mcp_server, resolve_into_state


class _Server:
    def __init__(self, url, name):
        self.url = url
        self.name = name
        self.status = "ok"


class _Manager:
    """Records what was registered; stands in for a live rag_tools manager."""

    def __init__(self):
        self.added = []
        self.closed = False

    async def add_server(self, *, name, protocol, url, description, headers, sync_tools):
        self.added.append(
            {"name": name, "url": url, "sync_tools": sync_tools, "headers": headers}
        )
        return _Server(url, name)

    async def close(self):
        self.closed = True


def test_registering_indexes_the_server_and_its_tools():
    manager = _Manager()

    server = asyncio.run(
        register_mcp_server("http://host:8000/mcp", "medsam", manager=manager)
    )

    assert manager.added == [
        {
            "name": "medsam",
            "url": "http://host:8000/mcp",
            "sync_tools": True,  # indexing the tools is the point of registering
            "headers": None,
        }
    ]
    assert server.url == "http://host:8000/mcp"
    assert not manager.closed  # an injected manager is the caller's to close


def test_a_missing_url_is_refused():
    with pytest.raises(ValueError):
        asyncio.run(register_mcp_server("", "medsam", manager=_Manager()))


def test_resolving_makes_the_tool_callable_this_run():
    state = {}

    resolve_into_state(state, _Server("http://host:8000/mcp", "medsam"))

    assert state["deployed_mcps"] == [
        {"url": "http://host:8000/mcp", "name": "medsam"}
    ]


def test_resolving_the_same_server_twice_adds_one_entry():
    """Retrieval may add the same server, so this has to compose with it."""
    state = {"deployed_mcps": [{"url": "http://host:8000/mcp", "name": "medsam"}]}

    resolve_into_state(state, "http://host:8000/mcp", "medsam")

    assert len(state["deployed_mcps"]) == 1


def test_resolving_keeps_servers_that_are_already_there():
    state = {"deployed_mcps": [{"url": "http://other:9000/mcp", "name": "other"}]}

    resolve_into_state(state, _Server("http://host:8000/mcp", "medsam"))

    assert [d["name"] for d in state["deployed_mcps"]] == ["other", "medsam"]


# ── the call site ────────────────────────────────────────────────────────────
# A finished build must reach the catalogue on its own. The agent's own prompt
# tells it to start a build, report the job_id and go do something else, so
# nothing may depend on it coming back to poll.


class _Ctx:
    def __init__(self):
        self.state = {}


_DONE_LOG = """\
  image     : alembic-tool:gget
  container : alembic-serve-gget-b29990
  url       : http://build-host:20162/mcp
"""

_LOOPBACK_DONE_LOG = """\
  image     : alembic-tool:gget
  container : alembic-serve-gget-b29990
  url       : http://localhost:20162/mcp
"""


class _FakeProc:
    """A build subprocess that writes its log and exits with ``returncode``."""

    def __init__(self, stdout, text, returncode):
        self.pid = 4242
        self._stdout, self._text, self._rc = stdout, text, returncode

    def wait(self):
        self._stdout.write(self._text)
        self._stdout.flush()
        return self._rc


def _fake_build(monkeypatch, tmp_path, *, log=_DONE_LOG, returncode=0):
    """A job record whose build subprocess is faked out."""
    from CoScientist.tools import alembic_tools

    # _runner persists the job's meta file; keep it out of the real build list.
    monkeypatch.setattr(alembic_tools, "LOG_DIR", tmp_path / "builds")
    monkeypatch.setattr(
        alembic_tools.subprocess,
        "Popen",
        lambda *a, stdout=None, **kw: _FakeProc(stdout, log, returncode),
    )
    rec = {
        "job_id": "gget-938c68",
        "repo_url": "https://github.com/pachterlab/gget",
        "status": "running",
        "started_at": 0.0,
        "log_file": str(tmp_path / "build.log"),
        "workdir": str(tmp_path / "work"),
    }
    monkeypatch.setattr(alembic_tools, "_JOBS", {"gget-938c68": rec})
    return alembic_tools, rec


def _stub_registry(monkeypatch, calls, *, boom=None, status=None):
    async def _register(mcp_url, name, description="", **kw):
        if boom is not None:
            raise boom
        calls.append({"url": mcp_url, "name": name})
        return _Server(mcp_url, name) if status is None else _Errored(mcp_url, name)

    monkeypatch.setattr(
        "CoScientist.tools.registry_bridge.register_mcp_server", _register
    )


class _Errored(_Server):
    """A server that was added but whose tool sync failed."""

    def __init__(self, url, name):
        super().__init__(url, name)
        from rag_tools.storage.models import ToolStatus

        self.status = ToolStatus.ERROR


def test_a_finished_build_registers_itself_with_nobody_watching(monkeypatch, tmp_path):
    calls = []
    alembic_tools, rec = _fake_build(monkeypatch, tmp_path)
    _stub_registry(monkeypatch, calls)

    alembic_tools._runner(rec)  # the build thread's body, run inline

    assert rec["status"] == "done"
    assert calls == [{"url": "http://build-host:20162/mcp", "name": "gget"}]
    assert rec["registered"] is True


def test_a_failed_build_registers_nothing(monkeypatch, tmp_path):
    calls = []
    alembic_tools, rec = _fake_build(monkeypatch, tmp_path, log="boom\n", returncode=1)
    _stub_registry(monkeypatch, calls)

    alembic_tools._runner(rec)

    assert rec["status"] == "failed"
    assert calls == []


def test_a_registry_outage_does_not_fail_the_build(monkeypatch, tmp_path):
    """The build succeeded. A catalogue that is down is recorded, not raised."""
    alembic_tools, rec = _fake_build(monkeypatch, tmp_path)
    _stub_registry(monkeypatch, [], boom=ConnectionError("registry unreachable"))

    alembic_tools._runner(rec)

    assert rec["status"] == "done"
    assert rec["registered"] is False
    assert "registry unreachable" in rec["registration_error"]


def test_polling_a_registered_build_reports_it_and_makes_it_callable(
    monkeypatch, tmp_path
):
    calls = []
    alembic_tools, rec = _fake_build(monkeypatch, tmp_path)
    _stub_registry(monkeypatch, calls)
    alembic_tools._runner(rec)
    ctx = _Ctx()

    out = asyncio.run(alembic_tools.check_mcp_build("gget-938c68", ctx))

    assert len(calls) == 1  # the poll does not register a second time
    assert out["registered"] is True
    assert ctx.state["deployed_mcps"] == [
        {"url": "http://build-host:20162/mcp", "name": "gget"}
    ]


def test_a_failed_registration_is_still_reported_on_a_later_poll(monkeypatch, tmp_path):
    alembic_tools, rec = _fake_build(monkeypatch, tmp_path)
    _stub_registry(monkeypatch, [], boom=ConnectionError("registry unreachable"))
    alembic_tools._runner(rec)

    first = asyncio.run(alembic_tools.check_mcp_build("gget-938c68", _Ctx()))
    second = asyncio.run(alembic_tools.check_mcp_build("gget-938c68", _Ctx()))

    for out in (first, second):
        assert out["status"] == "done"
        assert out["registered"] is False
        assert "registry unreachable" in out["registration_error"]


def test_a_build_that_never_ran_the_thread_is_registered_on_its_first_poll(
    monkeypatch, tmp_path
):
    """A record can reach a poll without having gone through the build thread
    (restored from disk, thread killed). The poll is the fallback."""
    calls = []
    alembic_tools, rec = _fake_build(monkeypatch, tmp_path)
    _stub_registry(monkeypatch, calls)
    (tmp_path / "build.log").write_text(_DONE_LOG, encoding="utf-8")
    rec.update(status="done", finished_at=1.0, mcp_url="http://build-host:20162/mcp")

    out = asyncio.run(alembic_tools.check_mcp_build("gget-938c68", _Ctx()))

    assert calls == [{"url": "http://build-host:20162/mcp", "name": "gget"}]
    assert out["registered"] is True


def test_a_loopback_build_is_not_pushed_to_the_shared_catalogue(monkeypatch, tmp_path):
    """A localhost URL in a shared registry is a broken entry — other machines
    resolve it to their own loopback. The build is still callable in-process,
    just kept out of the catalogue with a hint on how to opt in."""
    calls = []
    alembic_tools, rec = _fake_build(monkeypatch, tmp_path, log=_LOOPBACK_DONE_LOG)
    _stub_registry(monkeypatch, calls)
    ctx = _Ctx()

    alembic_tools._runner(rec)
    out = asyncio.run(alembic_tools.check_mcp_build("gget-938c68", ctx))

    assert calls == []  # nothing pushed to the shared catalogue
    assert rec["registered"] is False
    assert "A2A_HOST" in rec["registration_error"]
    assert out["registration_error"] == rec["registration_error"]
    # still callable this run — the executor uses deployed_mcps directly
    assert ctx.state["deployed_mcps"] == [
        {"url": "http://localhost:20162/mcp", "name": "gget"}
    ]


def test_a_server_whose_tools_could_not_be_indexed_is_not_called_registered(
    monkeypatch, tmp_path
):
    """Retrieval scores tools. A server row with none behind it will never
    surface, so reporting it as registered tells the agent something false."""
    alembic_tools, rec = _fake_build(monkeypatch, tmp_path)
    _stub_registry(monkeypatch, [], status="error")

    alembic_tools._runner(rec)

    assert rec["status"] == "done"
    assert rec["registered"] is False
    assert "tools could not be indexed" in rec["registration_error"]


def test_the_catalogue_outcome_survives_a_restart(monkeypatch, tmp_path):
    """After a restart the build list is rebuilt from the meta file. A user who
    never sees why the server is missing from the catalogue cannot fix it."""
    alembic_tools, rec = _fake_build(monkeypatch, tmp_path, log=_LOOPBACK_DONE_LOG)
    _stub_registry(monkeypatch, [])

    alembic_tools._runner(rec)
    monkeypatch.setattr(alembic_tools, "_JOBS", {})  # as after a restart

    meta = json.loads((tmp_path / "builds" / "gget-938c68.json").read_text())
    snap = alembic_tools.web_build_snapshot("gget-938c68")

    assert meta["registered"] is False
    assert snap["mcp_url"] == "http://localhost:20162/mcp"
    assert "A2A_HOST" in snap["registration_error"]
