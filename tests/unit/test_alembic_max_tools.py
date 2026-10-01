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


def test_coder_budget_grows_with_the_plan():
    assert config.coder_max_steps(0) == config.MAX_STEPS
    assert config.coder_max_steps(5) == config.MAX_STEPS
    assert config.coder_max_steps(8) == 8 * config.CODER_STEPS_PER_TOOL


# ── resuming a finished build ────────────────────────────────────────────────
@pytest.fixture
def failed_build(monkeypatch, tmp_path):
    from CoScientist.tools import alembic_tools as at

    monkeypatch.setattr(at, "LOG_DIR", tmp_path)
    monkeypatch.setattr(at, "JOB_METADATA_DIR", tmp_path / "jobs")
    workdir = tmp_path / "FEDOT-abc123" / "workdir"
    workdir.mkdir(parents=True)
    rec = {"job_id": "FEDOT-abc123", "repo_url": "https://github.com/aimclub/FEDOT",
           "status": "failed", "error": "coder gate", "finished_at": 1.0, "returncode": 1,
           "log_file": str(tmp_path / "FEDOT-abc123.log"), "workdir": str(workdir),
           "max_tools": 8}
    monkeypatch.setattr(at, "_JOBS", {rec["job_id"]: rec})
    started = []
    monkeypatch.setattr(at, "_runner", started.append)
    return at, rec, started


def test_resume_reruns_the_build_from_a_stage(failed_build):
    at, rec, started = failed_build
    out = at.resume_build("FEDOT-abc123", "coder")

    assert out["ok"] and out["resume_from"] == "coder"
    assert rec["status"] == "running" and rec["resume_from"] == "coder"
    assert "error" not in rec and "finished_at" not in rec
    assert started == [rec]


@pytest.mark.parametrize("stage, status, why", [
    ("compile", "failed", "stage must be one of"),
    ("coder", "running", "still running"),
])
def test_resume_refuses(failed_build, stage, status, why):
    at, rec, started = failed_build
    rec["status"] = status
    out = at.resume_build("FEDOT-abc123", stage)

    assert not out["ok"] and why in out["error"]
    assert not started


def test_resume_needs_the_workdir(failed_build):
    import shutil

    at, rec, started = failed_build
    shutil.rmtree(rec["workdir"])
    out = at.resume_build("FEDOT-abc123", "coder")

    assert not out["ok"] and "workdir" in out["error"]


def test_resumed_runner_passes_the_stage_and_appends_the_log(monkeypatch, tmp_path):
    from CoScientist.tools import alembic_tools as at

    monkeypatch.setattr(at, "LOG_DIR", tmp_path)
    log = tmp_path / "FEDOT-abc123.log"
    log.write_text("explorer and environment stages\n", encoding="utf-8")
    seen = {}

    class _Proc:
        pid = 1

        def wait(self):
            return 0

    def _popen(cmd, **kw):
        seen["cmd"] = cmd
        return _Proc()

    async def _no_catalogue(*a, **k):
        return None

    monkeypatch.setattr(at.subprocess, "Popen", _popen)
    monkeypatch.setattr(at, "_finalize", lambda *a, **k: None)
    monkeypatch.setattr(at, "_write_job_meta", lambda *a, **k: None)
    monkeypatch.setattr(at, "_register_in_catalogue", _no_catalogue)
    rec = {"job_id": "FEDOT-abc123", "repo_url": "https://github.com/aimclub/FEDOT",
           "status": "running", "log_file": str(log),
           "workdir": str(tmp_path / "FEDOT-abc123" / "workdir"), "resume_from": "coder"}
    at._runner(rec)

    assert seen["cmd"][-2:] == ["--resume", "coder"]
    text = log.read_text(encoding="utf-8")
    assert text.startswith("explorer and environment stages")
    assert f"{at._RESUME_MARK} coder" in text


def test_resume_clears_the_recovered_mark(failed_build):
    at, rec, started = failed_build
    rec["_recovered"] = True
    rec["pid"] = 999999
    at.resume_build("FEDOT-abc123", "coder")

    assert "_recovered" not in rec and "pid" not in rec


def test_log_status_reads_only_the_resumed_run():
    from CoScientist.tools import alembic_tools as at

    earlier = 'coder "status": "complete" ... pipeline failed (exit 137)\n'
    assert at._status_from_log(earlier + f"\n{at._RESUME_MARK} coder\nSTAGE 3") == "running"
    assert at._status_from_log(earlier + f"\n{at._RESUME_MARK} coder\n{at._SERVE_BANNER}") == "done"
    assert at._status_from_log(earlier) == "done"


def test_resume_remembers_the_server_it_replaces(failed_build):
    at, rec, started = failed_build
    rec.update(status="done", container="alembic-serve-FEDOT-old", server_id="old-row")
    at.resume_build("FEDOT-abc123", "validator")

    assert rec["replaces"] == {"container": "alembic-serve-FEDOT-old", "server_id": "old-row"}
    assert "container" not in rec and "server_id" not in rec


def test_replaced_server_is_retired_after_the_new_one_serves(monkeypatch, tmp_path):
    from CoScientist.tools import alembic_tools as at

    monkeypatch.setattr(at, "LOG_DIR", tmp_path)
    monkeypatch.setattr(at, "JOB_METADATA_DIR", tmp_path / "jobs")
    removed, unregistered = [], []
    monkeypatch.setattr(at, "_docker", lambda *a, **k: removed.append(a))
    monkeypatch.setattr(at, "_unregister", lambda sid: unregistered.append(sid))
    rec = {"job_id": "FEDOT-abc123", "status": "done", "container": "new-c", "server_id": "new-row",
           "replaces": {"container": "old-c", "server_id": "old-row"}}
    at._retire_replaced(rec)

    assert removed == [("rm", "-f", "old-c")]
    assert unregistered == ["old-row"]
    assert "replaces" not in rec


def test_nothing_is_retired_when_the_server_is_the_same(monkeypatch, tmp_path):
    from CoScientist.tools import alembic_tools as at

    monkeypatch.setattr(at, "LOG_DIR", tmp_path)
    monkeypatch.setattr(at, "JOB_METADATA_DIR", tmp_path / "jobs")
    removed, unregistered = [], []
    monkeypatch.setattr(at, "_docker", lambda *a, **k: removed.append(a))
    monkeypatch.setattr(at, "_unregister", lambda sid: unregistered.append(sid))
    rec = {"job_id": "FEDOT-abc123", "status": "done", "container": "c", "server_id": "row",
           "replaces": {"container": "c", "server_id": "row"}}
    at._retire_replaced(rec)

    assert removed == [] and unregistered == []
