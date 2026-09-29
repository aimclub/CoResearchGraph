"""The live bridge narrates a FEDOT run; it must not be able to change one.

The /fedot-demo page draws a pipeline from the config the engine generates,
and the config is exactly the thing `run()` does not hand back — it generates
and immediately builds. Taking `run()` apart to get at it is what broke here
first: `tests/unit/test_reranker_fallback.py` stubs the engine with a single
`run(task, timeout=...)`, so calling `generate_config` + `build_and_run`
instead raised AttributeError inside the broad `except`, and a run that
should have reported its timeout reported "error" instead. A caller that only
implements `run` is not a test artefact either — it is the contract this
toolset has always used.

So the generator is wrapped and `run()` still called. These tests pin both
halves: the bridge sees the config, and an engine without a generator still
runs.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from CoScientist.config import get_settings
from CoScientist.tools import fedotmas_tools as ft
from CoScientist.tools.fedot_live import fedot_live


@pytest.fixture(autouse=True)
def _enable_fedot_for_bridge_tests(monkeypatch):
    monkeypatch.setattr(get_settings().web, "fedot_fallback_enabled", True)


class _Server:
    def __init__(self, sid):
        self.name, self.url, self.description, self.protocol = (
            sid, f"http://{sid}", "", "http")


class _PG:
    def __init__(self, *a, **k):
        pass

    async def initialize(self):
        return None

    async def close(self):
        return None

    async def get_server(self, sid):
        return _Server(sid)


class _Config:
    """Stands in for a MASConfig/MAWConfig: pydantic-shaped enough."""

    def __init__(self, agents=("a1",)):
        self.agents = list(agents)

    def model_dump(self):
        return {"agents": self.agents}


def _engine(*, generator=True, config=None):
    """An engine that records what it was asked to do."""
    seen = {}

    class _Engine:
        def __init__(self, mcp_servers=None, plugins=None):
            seen["servers"] = sorted(mcp_servers or {})
            seen["plugins"] = [type(p).__name__ for p in (plugins or [])]

        async def run(self, task, timeout=None):
            seen["timeout"] = timeout
            if generator:
                # What the real `run()` does first, and the only reason the
                # wrapper is reached at all.
                seen["config"] = await self.generate_config(task)
            return "done"

        def _finalize_langfuse(self):
            seen["finalized"] = True

    if generator:
        async def generate_config(self, task):  # noqa: ANN001
            seen["generated"] = task
            return config if config is not None else _Config()

        _Engine.generate_config = generate_config

    return _Engine, seen


def _run(monkeypatch, engine_cls, state=None):
    monkeypatch.setattr(ft, "PostgresClient", _PG)
    monkeypatch.setattr(ft, "PatchedMAS", engine_cls)
    monkeypatch.setattr(ft, "PatchedMAW", engine_cls)
    monkeypatch.setattr(ft, "HttpMCPServer", lambda **kw: kw)
    tc = SimpleNamespace(state=state or {"filtered_tools": [{"server_id": "s1"}]})
    return asyncio.run(ft.fedot_toolset.fedot_tool("do the thing", tool_context=tc))


def _drain(queue):
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


@pytest.fixture()
def bus():
    """A subscriber, removed again whatever the test does."""
    queue = fedot_live.subscribe()
    _drain(queue)  # a previous run's config is replayed on subscribe
    try:
        yield queue
    finally:
        fedot_live.unsubscribe(queue)


def test_a_direct_tool_call_is_refused_before_engine_start_when_disabled(monkeypatch):
    monkeypatch.setattr(get_settings().web, "fedot_fallback_enabled", False)
    monkeypatch.setattr(get_settings().experiments, "route_fedot", False)
    engine_cls, seen = _engine()

    result = _run(monkeypatch, engine_cls)

    assert result["error_code"] == "capability_disabled"
    assert "run" not in seen


def test_the_page_is_told_the_shape_before_the_first_agent_runs(monkeypatch, bus):
    engine_cls, seen = _engine()
    ret = _run(monkeypatch, engine_cls)
    assert ret["status"] == "success", ret

    kinds = [e["type"] for e in _drain(bus)]
    # Order is the point: a config that arrives after run_end draws nothing.
    assert kinds[:2] == ["run_start", "config"], kinds
    assert kinds[-1] == "run_end"
    assert "FedotLivePlugin" in seen["plugins"]


def test_an_engine_without_a_generator_still_runs(monkeypatch, bus):
    """The regression: `run()` remains the one call this toolset makes."""
    engine_cls, seen = _engine(generator=False)
    ret = _run(monkeypatch, engine_cls)
    assert ret["status"] == "success", ret
    assert "timeout" in seen, "run() was not called"
    assert [e["type"] for e in _drain(bus)] == ["run_start", "run_end"]


def test_a_config_the_bus_chokes_on_does_not_fail_the_run(monkeypatch, bus):
    class _Unserialisable(_Config):
        def model_dump(self):
            raise TypeError("not today")

    engine_cls, _ = _engine(config=_Unserialisable())
    ret = _run(monkeypatch, engine_cls)
    assert ret["status"] == "success", ret
    assert [e["type"] for e in _drain(bus)] == ["run_start", "run_end"]


def test_the_config_check_runs_on_the_engine_it_was_written_for(monkeypatch, bus):
    """`run_config_guardrails` reads `config.pipeline`, which a MASConfig has
    not got — on the default engine it would raise on every single run."""
    calls = []
    monkeypatch.setattr(ft, "run_config_guardrails",
                        lambda cfg: calls.append(cfg) or [])

    monkeypatch.setattr(get_settings().experiments, "fedot_engine", "mas")
    engine_cls, _ = _engine()
    assert _run(monkeypatch, engine_cls)["status"] == "success"
    assert calls == [], "the MAS config was handed to the MAW check"

    monkeypatch.setattr(get_settings().experiments, "fedot_engine", "maw")
    engine_cls, _ = _engine()
    assert _run(monkeypatch, engine_cls)["status"] == "success"
    assert len(calls) == 1


def test_a_design_fault_stops_the_run_before_the_pipeline_is_paid_for(monkeypatch, bus):
    monkeypatch.setattr(get_settings().experiments, "fedot_engine", "maw")
    monkeypatch.setattr(ft, "run_config_guardrails",
                        lambda cfg: ["Unused agents not referenced in pipeline"])
    engine_cls, seen = _engine()
    ret = _run(monkeypatch, engine_cls)
    assert ret["status"] == "error" and "Unused agents" in ret["error"], ret
    # And nothing was drawn for a pipeline that never existed.
    assert "config" not in [e["type"] for e in _drain(bus)]


def test_an_older_engine_without_the_check_is_not_a_failed_run(monkeypatch, bus):
    monkeypatch.setattr(get_settings().experiments, "fedot_engine", "maw")
    monkeypatch.setattr(ft, "run_config_guardrails", None)
    engine_cls, _ = _engine()
    assert _run(monkeypatch, engine_cls)["status"] == "success"
