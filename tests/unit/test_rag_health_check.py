"""Unit tests for the rag_tools registry health check (no live infra).

The health check connects through the same datastore clients Retrieve_tools
uses; here we drive its core (`probe_registry` / `run_health_check`) with fake
async clients so the logic — empty-registry query, ok/fail verdict, and the
transient-failure retry — is verified without Postgres/Qdrant running.

The repo has no pytest-asyncio, so each async call is driven with asyncio.run.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

# Loaded by path: scripts/ is not a package on the pythonpath. Register it in
# sys.modules before exec so dataclasses can resolve its string annotations.
_MOD_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "rag_tools" / "health_check.py"
)
_spec = importlib.util.spec_from_file_location("rag_health_check", _MOD_PATH)
hc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = hc
_spec.loader.exec_module(hc)


class FakePostgres:
    def __init__(self, servers=(), *, fail_on=None):
        self._servers = list(servers)
        self._fail_on = fail_on
        self.closed = False

    async def initialize(self):
        if self._fail_on == "initialize":
            raise ConnectionRefusedError("postgres not up")

    async def list_servers(self):
        return list(self._servers)

    async def close(self):
        self.closed = True


class FakeQdrant:
    def __init__(self, *, status="ok", collections=0, fail_on=None):
        self._health = {"status": status, "collections": collections}
        self._fail_on = fail_on
        self.closed = False

    async def connect(self):
        if self._fail_on == "connect":
            raise ConnectionRefusedError("qdrant not up")

    async def health_check(self):
        return dict(self._health)

    async def close(self):
        self.closed = True


def test_probe_empty_registry_is_healthy():
    health = asyncio.run(hc.probe_registry(FakePostgres(), FakeQdrant()))
    assert health.ok()
    assert health.server_count == 0
    assert health.qdrant_collections == 0


def test_probe_reports_registered_server_count():
    health = asyncio.run(
        hc.probe_registry(FakePostgres(servers=["s1", "s2"]), FakeQdrant(collections=3))
    )
    assert health.ok()
    assert health.server_count == 2
    assert health.qdrant_collections == 3


def test_probe_closes_both_clients_even_on_success():
    pg, qd = FakePostgres(), FakeQdrant()
    asyncio.run(hc.probe_registry(pg, qd))
    assert pg.closed and qd.closed


def test_qdrant_down_status_is_unhealthy():
    health = asyncio.run(hc.probe_registry(FakePostgres(), FakeQdrant(status="down")))
    assert not health.ok()
    assert not health.qdrant_ok


def test_run_health_check_retries_then_succeeds(monkeypatch):
    """A container that is up-but-not-ready fails the first probe and passes the
    next — the retry loop must recover instead of erroring out."""
    calls = {"n": 0}

    def fake_build(_settings):
        calls["n"] += 1
        fail = "initialize" if calls["n"] == 1 else None
        return FakePostgres(fail_on=fail), FakeQdrant()

    monkeypatch.setattr(hc, "_build_clients", fake_build)

    health = asyncio.run(hc.run_health_check(object(), retries=3, delay=0.0))
    assert health.ok()
    assert calls["n"] == 2  # failed once, succeeded on the second attempt


def test_run_health_check_times_out_a_hung_probe(monkeypatch):
    """A client whose connect hangs must not stall the check forever — the
    per-attempt timeout fires and (with retries=1) surfaces a TimeoutError."""

    class HangingPostgres(FakePostgres):
        async def initialize(self):
            await asyncio.sleep(60)  # never completes within the timeout

    monkeypatch.setattr(
        hc, "_build_clients", lambda _s: (HangingPostgres(), FakeQdrant())
    )
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        asyncio.run(hc.run_health_check(object(), retries=1, delay=0.0, timeout=0.05))


def test_run_health_check_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(
        hc,
        "_build_clients",
        lambda _s: (FakePostgres(fail_on="initialize"), FakeQdrant()),
    )
    with pytest.raises(ConnectionRefusedError):
        asyncio.run(hc.run_health_check(object(), retries=2, delay=0.0))
