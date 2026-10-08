"""Tests for the Alembic A2A tools (GitHub repo -> served MCP server).

These exercise the job registry directly — no real Docker build runs. The
background subprocess launch (``_runner``) is monkeypatched out so a "running"
job never actually finishes, which lets tests assert on the reuse logic
deterministically.
"""
import asyncio
import time

from dotenv import load_dotenv

load_dotenv()

import CoScientist.assembly.bindings  # noqa: E402,F401  (registration side effect)
from CoScientist.assembly.registry import REGISTRY  # noqa: E402

from CoScientist.tools import alembic_tools  # noqa: E402
from CoScientist.tools.alembic_tools import (  # noqa: E402
    ALEMBIC_TOOLS,
    build_mcp_server,
    check_mcp_build,
    list_mcp_builds,
)


def setup_function():
    alembic_tools._JOBS.clear()


def _noop_runner(rec):
    """Stand-in for _runner: never touches the record, so the job stays
    "running" forever — exactly what the reuse tests need, without spawning a
    real subprocess."""


def _make_rec(job_id, repo_url, status="done", **extra):
    rec = {
        "job_id": job_id,
        "repo_url": repo_url,
        "status": status,
        "started_at": time.time() - 60,
        "finished_at": time.time(),
        "log_file": "/no/such/file.log",  # _read_log tolerates a missing file
    }
    rec.update(extra)
    return rec


# ── build_mcp_server: input validation ──────────────────────────────────────

def test_build_mcp_server_rejects_invalid_repo_url():
    result = asyncio.run(build_mcp_server("not-a-git-url"))
    assert result["status"] == "error"
    assert "repo_url" in result["error"]
    assert alembic_tools._JOBS == {}  # no job was started


# ── build_mcp_server: reuse instead of rebuilding ────────────────────────────

def test_repeated_build_reuses_running_job(monkeypatch):
    monkeypatch.setattr(alembic_tools, "_runner", _noop_runner)
    repo_url = "https://github.com/whitead/synspace"

    first = asyncio.run(build_mcp_server(repo_url))
    assert first["status"] == "running"
    job_id = first["job_id"]

    second = asyncio.run(build_mcp_server(repo_url))
    assert second["job_id"] == job_id
    assert second["status"] == "running"
    assert "already running" in second["note"]
    assert len(alembic_tools._JOBS) == 1  # no second job created


def test_repeated_build_reuses_done_job():
    repo_url = "https://github.com/whitead/synspace"
    rec = _make_rec(
        "synspace-abc123", repo_url, status="done",
        mcp_url="http://localhost:9001/mcp",
        image="alembic-tool:synspace", container="synspace_container",
    )
    alembic_tools._JOBS[rec["job_id"]] = rec

    result = asyncio.run(build_mcp_server(repo_url))
    assert result["job_id"] == rec["job_id"]
    assert result["status"] == "done"
    assert result["mcp_url"] == "http://localhost:9001/mcp"
    assert "already built" in result["note"] or "reusing" in result["note"]
    assert len(alembic_tools._JOBS) == 1  # reused, not rebuilt


def test_force_rebuild_starts_a_new_job(monkeypatch):
    monkeypatch.setattr(alembic_tools, "_runner", _noop_runner)
    repo_url = "https://github.com/whitead/synspace"
    rec = _make_rec("synspace-old", repo_url, status="done")
    alembic_tools._JOBS[rec["job_id"]] = rec

    result = asyncio.run(build_mcp_server(repo_url, force_rebuild=True))
    assert result["status"] == "running"
    assert result["job_id"] != rec["job_id"]
    assert len(alembic_tools._JOBS) == 2


# ── log parsing: _finalize / _snapshot ───────────────────────────────────────

def test_finalize_parses_url_image_container_and_stage(tmp_path):
    log = tmp_path / "build.log"
    log.write_text(
        "STAGE 1 — clone\n"
        "STAGE 2 — env\n"
        "STAGE 3 — generate\n"
        "STAGE 4 — validate\n"
        "STAGE 5 — serve\n"
        "url: http://localhost:9001/mcp\n"
        "image: alembic-tool:demo\n"
        "container: demo_container_abc\n",
        encoding="utf-8",
    )
    rec = {
        "job_id": "demo-1",
        "repo_url": "https://github.com/demo/demo",
        "status": "running",
        "started_at": time.time(),
        "log_file": str(log),
    }

    alembic_tools._finalize(rec, returncode=0)
    assert rec["status"] == "done"
    assert rec["mcp_url"] == "http://localhost:9001/mcp"
    assert rec["image"] == "alembic-tool:demo"
    assert rec["container"] == "demo_container_abc"

    snap = alembic_tools._snapshot(rec)
    assert snap["stage"] == "5/5 serve"
    assert snap["mcp_url"] == "http://localhost:9001/mcp"
    assert snap["image"] == "alembic-tool:demo"
    assert snap["container"] == "demo_container_abc"


def test_finalize_marks_failed_on_nonzero_returncode(tmp_path):
    log = tmp_path / "build.log"
    log.write_text("STAGE 2 — env\nsome error happened\n", encoding="utf-8")
    rec = {
        "job_id": "demo-2",
        "repo_url": "https://github.com/demo/demo",
        "status": "running",
        "started_at": time.time(),
        "log_file": str(log),
    }

    alembic_tools._finalize(rec, returncode=1)
    assert rec["status"] == "failed"

    snap = alembic_tools._snapshot(rec)
    assert "some error happened" in snap["error"]


# ── check_mcp_build ──────────────────────────────────────────────────────────

def test_check_mcp_build_unknown_job_id_is_an_error():
    result = asyncio.run(check_mcp_build("no-such-job"))
    assert result["status"] == "error"
    assert "no-such-job" in result["error"]


def test_check_mcp_build_reports_known_job():
    rec = _make_rec("known-1", "https://github.com/demo/demo", status="done",
                     mcp_url="http://localhost:1/mcp", image="i", container="c")
    alembic_tools._JOBS[rec["job_id"]] = rec

    result = asyncio.run(check_mcp_build("known-1"))
    assert result["status"] == "done"
    assert result["mcp_url"] == "http://localhost:1/mcp"


# ── list_mcp_builds ───────────────────────────────────────────────────────────

def test_list_mcp_builds_returns_every_known_job():
    alembic_tools._JOBS["a"] = _make_rec("a", "https://github.com/x/a", status="done")
    alembic_tools._JOBS["b"] = _make_rec("b", "https://github.com/x/b", status="failed")

    result = asyncio.run(list_mcp_builds())
    job_ids = {b["job_id"] for b in result["builds"]}
    assert job_ids == {"a", "b"}


# ── registry consistency ─────────────────────────────────────────────────────

def test_alembic_tool_registry_docs_match_attached_functions():
    entry = REGISTRY.tool("alembic")
    doc_names = {d.name for d in entry.docs}
    real_names = {f.__name__ for f in ALEMBIC_TOOLS}
    assert doc_names == real_names
