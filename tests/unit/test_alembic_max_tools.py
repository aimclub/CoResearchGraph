"""The number of tools a build may expose is a per-build setting."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from CoScientist.alembic import config

# The instructions package imports itself as top-level `alembic`, the name it
# has inside the build container, so the prompt module is loaded by path.
_spec = importlib.util.spec_from_file_location(
    "_explorer_prompt",
    Path(config.__file__).parent / "instructions" / "explorer.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
explorer_instruction = _module.explorer_instruction


@pytest.mark.parametrize("raw, expected", [
    (None, None), ("", None), ("8", 8), (" 8 ", 8), ("1", 1), ("30", 30),
    ("0", None), ("31", None), ("-3", None), ("eight", None), ("7.5", None),
])
def test_requested_max_tools(raw, expected):
    assert config.requested_max_tools(raw) == expected


def test_explorer_rule_keeps_the_old_range_by_default():
    assert config.explorer_tool_count_rule(None) == "Propose 2-5 tools, best first."
    assert config.explorer_tool_count_rule(8) == "Propose 2-8 tools, best first."
    assert config.explorer_tool_count_rule(1) == "Propose exactly 1 tool, the most useful one."


def test_explorer_prompt_carries_the_token_the_agent_fills():
    assert explorer_instruction.count("__TOOL_COUNT_RULE__") == 1
    assert "2-5 tools" not in explorer_instruction


@pytest.fixture
def launched(monkeypatch, tmp_path):
    """build_mcp_server up to the point it would start a subprocess."""
    from CoScientist.tools import alembic_tools as at

    monkeypatch.setattr(at, "LOG_DIR", tmp_path)
    monkeypatch.setattr(at, "JOB_METADATA_DIR", tmp_path / "jobs")
    monkeypatch.setattr(at, "_JOBS", {})
    monkeypatch.setattr(at, "alembic_preflight", lambda *a, **k: {"available": True})
    monkeypatch.setattr(at, "_repo_exists", lambda *a, **k: (True, ""))
    monkeypatch.setattr(at, "_first_reusable_same_repo", lambda *a, **k: None)
    monkeypatch.setattr(at, "_reuse_from_host", lambda *a, **k: None)
    monkeypatch.setattr(at, "_pull_from_hub", lambda *a, **k: None)
    monkeypatch.setattr("CoScientist.capabilities.alembic_enabled", lambda: True)
    started = []
    # The build thread's target; asyncio's own worker threads stay real.
    monkeypatch.setattr(at, "_runner", started.append)
    return at, started


def test_max_tools_is_recorded_on_the_build(launched):
    at, started = launched
    out = asyncio.run(at.build_mcp_server("https://github.com/aimclub/FEDOT", max_tools=8))

    assert out["status"] == "running", out
    assert started and started[0]["max_tools"] == 8


@pytest.mark.parametrize("bad", [0, 31, "many"])
def test_max_tools_out_of_range_is_refused(launched, bad):
    at, started = launched
    out = asyncio.run(at.build_mcp_server("https://github.com/aimclub/FEDOT", max_tools=bad))

    assert out["status"] == "error" and "max_tools" in out["error"]
    assert not started


def test_runner_passes_the_cap_to_start_chain(monkeypatch, tmp_path):
    from CoScientist.tools import alembic_tools as at

    monkeypatch.setattr(at, "LOG_DIR", tmp_path)
    seen = {}

    class _Proc:
        pid = 1

        def wait(self):
            return 0

    def _popen(cmd, **kw):
        seen.update(kw["env"])
        return _Proc()

    monkeypatch.setattr(at.subprocess, "Popen", _popen)
    monkeypatch.setattr(at, "_finalize", lambda *a, **k: None)
    monkeypatch.setattr(at, "_write_job_meta", lambda *a, **k: None)

    async def _no_catalogue(*a, **k):
        return None

    monkeypatch.setattr(at, "_register_in_catalogue", _no_catalogue)
    rec = {"job_id": "FEDOT-abc123", "repo_url": "https://github.com/aimclub/FEDOT",
           "status": "running", "log_file": str(tmp_path / "FEDOT-abc123.log"),
           "workdir": str(tmp_path / "FEDOT-abc123" / "workdir"), "max_tools": 8}
    at._runner(rec)

    assert seen.get("ALEMBIC_MAX_TOOLS") == "8"
