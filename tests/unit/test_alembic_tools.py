"""Tests for the Alembic A2A tools (GitHub repo -> served MCP server).

These exercise the job registry directly — no real Docker build runs. The
background subprocess launch (``_runner``) is monkeypatched out so a "running"
job never actually finishes, which lets tests assert on the reuse logic
deterministically.
"""
import asyncio
import json
import subprocess
import time
import types
from pathlib import Path

import pytest

import pytest
from dotenv import load_dotenv

load_dotenv()

import CoScientist.assembly.bindings  # noqa: E402,F401  (registration side effect)
from CoScientist.assembly.registry import REGISTRY  # noqa: E402
from CoScientist.config import get_settings  # noqa: E402

from CoScientist.tools import alembic_tools  # noqa: E402
from CoScientist.tools.alembic_tools import (  # noqa: E402
    ALEMBIC_TOOLS,
    build_mcp_server,
    check_mcp_build,
    list_mcp_builds,
)


_IMPORT_ARTIFACTS = alembic_tools._import_image_artifacts


def setup_function():
    alembic_tools._JOBS.clear()


@pytest.fixture(autouse=True)
def _no_host(monkeypatch, tmp_path):
    """build_mcp_server looks for reusable servers on the host. Keep it away from
    the network, the real Docker daemon and the real build logs."""
    monkeypatch.setattr(alembic_tools, "_repo_exists", lambda url, timeout=20: (True, "ok"))
    monkeypatch.setattr(alembic_tools, "LOG_DIR", tmp_path / "builds")
    monkeypatch.delenv("A2A_HOST", raising=False)
    monkeypatch.setattr(alembic_tools, "_served_tools", _no_tools)
    monkeypatch.setattr(alembic_tools, "_import_image_artifacts",
                        lambda job_id, container, repo: (None, None))
    monkeypatch.setattr(alembic_tools, "_pull_from_hub", lambda repo_url, scope: None)
    _host(monkeypatch)


async def _no_tools(mcp_url):
    return []


def _host(monkeypatch, containers=None, tags=None, ports=None):
    """A fake daemon: ``containers`` name -> {image_id, running}, ``tags`` tag ->
    image id, ``ports`` running container -> published port."""
    inv = {"images": {i: "1GB" for i in (tags or {}).values()},
           "tags": dict(tags or {}), "containers": dict(containers or {})}
    for c in inv["containers"].values():
        inv["images"].setdefault(c["image_id"], "1GB")
    monkeypatch.setattr(alembic_tools, "docker_inventory", lambda: inv)
    monkeypatch.setattr(alembic_tools, "_container_state", lambda name: {
        "exists": name in inv["containers"],
        "running": inv["containers"].get(name, {}).get("running", False),
        "image_id": inv["containers"].get(name, {}).get("image_id"),
        "port": (ports or {}).get(name)})
    return inv


@pytest.fixture(autouse=True)
def _enable_alembic_for_tool_mechanics(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings().experiments, "route_alembic", True)
    # The durable registry is the host's: a test must neither read the builds
    # of this machine nor leave its own fake jobs there for the next process.
    monkeypatch.setattr(alembic_tools, "JOB_METADATA_DIR", tmp_path / "jobs")
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


def test_build_mcp_server_refuses_a_direct_call_when_capability_is_off(monkeypatch):
    monkeypatch.setattr(get_settings().experiments, "route_alembic", False)
    result = asyncio.run(build_mcp_server("https://github.com/whitead/synspace"))
    assert result["status"] == "error"
    assert result["error_code"] == "capability_disabled"
    assert alembic_tools._JOBS == {}


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


def test_repeated_build_reuses_done_job(monkeypatch):
    _host(monkeypatch, containers={"synspace_container": {"image_id": "i", "running": True}})
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


def test_keyed_build_reuses_done_job_for_same_repo(monkeypatch):
    """A new EM run_id must not rebuild a repo whose MCP is already served.
    The server must be up (or start again): a finished build whose server is
    gone is rebuilt, the same as for an unkeyed call."""
    monkeypatch.setattr(alembic_tools, "_runner", _noop_runner)
    monkeypatch.setattr(alembic_tools, "_load_jobs_from_disk", lambda merge=True: None)
    _host(monkeypatch, containers={"alembic-serve-unomd-old": {"image_id": "i", "running": True}})
    repo_url = "https://github.com/ingcoder/unomd"
    rec = _make_rec(
        "unomd-abc123", repo_url, status="done",
        mcp_url="http://localhost:21147/mcp",
        image="alembic-tool:unomd", container="alembic-serve-unomd-old",
        idempotency_key="old-run:EXP-1:https://github.com/ingcoder/unomd",
    )
    alembic_tools._JOBS[rec["job_id"]] = rec

    result = asyncio.run(build_mcp_server(
        repo_url,
        idempotency_key="new-run:EXP-1:https://github.com/ingcoder/unomd",
    ))
    assert result["job_id"] == rec["job_id"]
    assert result["status"] == "done"
    assert result["mcp_url"] == "http://localhost:21147/mcp"
    assert len(alembic_tools._JOBS) == 1


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
        "container: alembic-serve-demo_container_abc\n",
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
    assert rec["container"] == "alembic-serve-demo_container_abc"

    snap = alembic_tools._snapshot(rec)
    assert snap["stage"] == "5/5 serve"
    assert snap["mcp_url"] == "http://localhost:9001/mcp"
    assert snap["image"] == "alembic-tool:demo"
    assert snap["container"] == "alembic-serve-demo_container_abc"


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


def test_check_mcp_build_reports_known_job(monkeypatch):
    _host(monkeypatch, containers={"c": {"image_id": "i", "running": True}})
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


# ── build_mcp_server: reuse what the host already has ────────────────────────

_REPO = "https://github.com/whitead/synspace"


def _disk_build(job_id, repo_url=_REPO, **fields):
    """A finished build known only from its meta file, as after a restart."""
    meta = {"job_id": job_id, "repo_url": repo_url, "status": "done",
            "started_at": time.time() - 600, "finished_at": time.time() - 300, **fields}
    alembic_tools.LOG_DIR.mkdir(parents=True, exist_ok=True)
    (alembic_tools.LOG_DIR / f"{job_id}.json").write_text(json.dumps(meta))


def test_a_build_from_an_earlier_process_that_still_serves_is_reused(monkeypatch):
    _disk_build("synspace-aaa111", container="alembic-serve-synspace-c0ffee",
                image_id="sha256:img")
    _host(monkeypatch,
          containers={"alembic-serve-synspace-c0ffee": {"image_id": "sha256:img", "running": True}},
          ports={"alembic-serve-synspace-c0ffee": "25001"})
    monkeypatch.setattr(alembic_tools, "start_build_server",
                        lambda job_id: pytest.fail("restarted a server that runs"))
    monkeypatch.setattr(alembic_tools, "_runner", _noop_runner)

    result = asyncio.run(build_mcp_server(_REPO))

    assert result["status"] == "done"
    assert result["job_id"] == "synspace-aaa111"
    # The current port, not whatever address the build logged back then.
    assert result["mcp_url"] == "http://localhost:25001/mcp"
    assert list(alembic_tools._JOBS) == ["synspace-aaa111"]


def test_a_build_whose_server_is_down_is_served_again_from_its_image(monkeypatch):
    """The earlier cascade queued a record without a workdir, and _runner died on
    it with the build stuck at running. Starting goes through the web controls."""
    _disk_build("synspace-bbb222", container="alembic-serve-synspace-dead00",
                image_id="sha256:img")
    _host(monkeypatch, tags={"alembic-tool:synspace-bbb222": "sha256:img"})
    started = []

    def start(job_id):
        started.append(job_id)
        return {"ok": True, "container": "alembic-serve-synspace-beef01",
                "mcp_url": "http://localhost:25002/mcp", "registered": False}

    monkeypatch.setattr(alembic_tools, "start_build_server", start)
    monkeypatch.setattr(alembic_tools, "_runner", _noop_runner)

    result = asyncio.run(build_mcp_server(_REPO))

    assert started == ["synspace-bbb222"]
    assert result["status"] == "done"
    assert result["container"] == "alembic-serve-synspace-beef01"
    assert result["mcp_url"] == "http://localhost:25002/mcp"


def test_an_image_that_does_not_start_leads_to_a_new_build(monkeypatch):
    _disk_build("synspace-bbb222", image_id="sha256:img")
    _host(monkeypatch, tags={"alembic-tool:synspace-bbb222": "sha256:img"})
    monkeypatch.setattr(alembic_tools, "start_build_server",
                        lambda job_id: {"ok": False, "error": "port taken"})
    monkeypatch.setattr(alembic_tools, "_runner", _noop_runner)

    result = asyncio.run(build_mcp_server(_REPO))

    assert result["status"] == "running"
    assert result["job_id"] != "synspace-bbb222"


def test_a_server_started_outside_this_tool_is_reused(monkeypatch):
    _host(monkeypatch,
          containers={"alembic-serve-synspace-abcdef": {"image_id": "sha256:x", "running": True}},
          ports={"alembic-serve-synspace-abcdef": "25003"})
    monkeypatch.setattr(alembic_tools, "_runner", _noop_runner)

    result = asyncio.run(build_mcp_server(_REPO))

    assert result["status"] == "done"
    assert result["container"] == "alembic-serve-synspace-abcdef"
    assert "outside this tool" in result["note"]


def test_a_server_of_a_repository_with_a_longer_name_is_not_reused(monkeypatch):
    _host(monkeypatch,
          containers={"alembic-serve-synspace-abcdef": {"image_id": "sha256:x", "running": True}},
          ports={"alembic-serve-synspace-abcdef": "25003"})
    monkeypatch.setattr(alembic_tools, "_runner", _noop_runner)

    result = asyncio.run(build_mcp_server("https://github.com/org/syn"))

    assert result["status"] == "running"


def test_force_rebuild_ignores_what_the_host_has(monkeypatch):
    _disk_build("synspace-aaa111", container="alembic-serve-synspace-c0ffee",
                image_id="sha256:img")
    _host(monkeypatch,
          containers={"alembic-serve-synspace-c0ffee": {"image_id": "sha256:img", "running": True}},
          ports={"alembic-serve-synspace-c0ffee": "25001"})
    monkeypatch.setattr(alembic_tools, "_runner", _noop_runner)

    result = asyncio.run(build_mcp_server(_REPO, force_rebuild=True))

    assert result["status"] == "running"


# ── which live servers belong to a session ───────────────────────────────────

def _scoped_context(user, session):
    return types.SimpleNamespace(state={"graph_scope_user_id": user,
                                        "graph_scope_session_id": session})


def test_the_build_is_told_what_kind_of_tool_is_wanted(monkeypatch, tmp_path):
    """A hint steers the explorer; a task spec pins the tools the build must
    produce. Both go on the record the runner passes to the pipeline as
    environment; a spec given as a file travels as its content."""
    spec = tmp_path / "tasks.yaml"
    spec.write_text("- name: seqio_convert\n  description: convert\n", encoding="utf-8")
    _host(monkeypatch)
    env = {}

    def runner(rec):
        env.update({"hints": rec.get("hints"), "task_spec": rec.get("task_spec")})

    monkeypatch.setattr(alembic_tools, "_runner", runner)

    result = asyncio.run(build_mcp_server(
        _REPO, hints="  something that samples analogues  ", task_spec=str(spec)))

    assert result["status"] == "running"
    assert env == {"hints": "something that samples analogues",
                   "task_spec": "- name: seqio_convert\n  description: convert\n"}


def test_a_task_spec_that_cannot_be_read_stops_the_build(monkeypatch):
    _host(monkeypatch)
    monkeypatch.setattr(alembic_tools, "_runner",
                        lambda rec: pytest.fail("a build started on an unreadable spec"))
    monkeypatch.setattr(alembic_tools, "_TASK_SPEC_MAX_CHARS", 10)

    result = asyncio.run(build_mcp_server(_REPO, task_spec="x" * 50))

    assert result["status"] == "error" and "task spec could not be read" in result["error"]


def test_a_reused_build_is_attached_only_to_the_session_that_reused_it(monkeypatch):
    _host(monkeypatch, containers={"c1": {"image_id": "i", "running": True}})
    alembic_tools._JOBS["synspace-abc123"] = _make_rec(
        "synspace-abc123", _REPO, mcp_url="http://localhost:9001/mcp", container="c1")

    asyncio.run(build_mcp_server(_REPO, tool_context=_scoped_context("u", "s1")))

    assert alembic_tools.live_build_servers(("u", "s1")) == {"http://localhost:9001/mcp": "synspace"}
    assert alembic_tools.live_build_servers(("u", "s2")) == {}
    assert alembic_tools.live_build_servers(None) == {}


def test_a_server_another_session_built_does_not_hide_a_tool_gap(monkeypatch):
    """Any running build used to count as a usable tool, so one served repository
    turned off the redirect to CoderAgent/McpBuilderAgent for every task."""
    from CoScientist.agents.callbacks.tool_callbacks import (
        TOOL_MATCH_STATE_KEY,
        redirect_when_no_tools,
    )

    _host(monkeypatch, containers={"c1": {"image_id": "i", "running": True}})
    rec = _make_rec("synspace-abc123", _REPO, mcp_url="http://localhost:9001/mcp",
                    container="c1", scopes=[["u", "other"]])
    alembic_tools._JOBS[rec["job_id"]] = rec
    ctx = _scoped_context("u", "s1")
    ctx.state[TOOL_MATCH_STATE_KEY] = {"matched": False, "best_score": 0.1}

    assert redirect_when_no_tools(ctx) is not None  # abstains: nothing of its own

    rec["scopes"].append(["u", "s1"])
    ctx.state.pop("fedot_results", None)
    assert redirect_when_no_tools(ctx) is None


def test_a_finished_build_reports_the_names_of_its_tools(monkeypatch):
    """Without the names the caller guessed one ("sample_space" for synspace's
    chemical_space) and the tool call went nowhere."""
    async def served(mcp_url):
        return [{"name": "chemical_space", "description": "Local chemical space around a molecule"}]

    monkeypatch.setattr(alembic_tools, "_served_tools", served)
    _host(monkeypatch, containers={"c1": {"image_id": "i", "running": True}})
    alembic_tools._JOBS["synspace-abc123"] = _make_rec(
        "synspace-abc123", _REPO, mcp_url="http://localhost:9001/mcp", container="c1")

    result = asyncio.run(build_mcp_server(_REPO))

    assert [tool["name"] for tool in result["tools"]] == ["chemical_space"]


def test_a_build_another_chat_made_is_attached_after_check_mcp_build(monkeypatch):
    """The web process keeps builds across chats. A later chat reaches the known
    build through list_mcp_builds + check_mcp_build, and its server was not
    attached to that chat's executor, so the task fell through to CoderAgent."""
    _host(monkeypatch, containers={"c1": {"image_id": "i", "running": True}})
    rec = _make_rec("synspace-abc123", _REPO, mcp_url="http://localhost:9001/mcp",
                    container="c1", scopes=[["u", "first-chat"]])
    alembic_tools._JOBS[rec["job_id"]] = rec

    asyncio.run(check_mcp_build("synspace-abc123", tool_context=_scoped_context("u", "second-chat")))

    assert alembic_tools.live_build_servers(("u", "second-chat")) == {"http://localhost:9001/mcp": "synspace"}
    assert alembic_tools.live_build_servers(("u", "first-chat")) == {"http://localhost:9001/mcp": "synspace"}
    assert alembic_tools.live_build_servers(("u", "third-chat")) == {}


# ── a known build whose server was stopped ───────────────────────────────────

def _stopped_build(monkeypatch, start_ok=True):
    """synspace-abc123 in memory with its container c1 stopped (the Stop button).
    A successful start brings it back as c2 on a new port."""
    inv = _host(monkeypatch, containers={"c1": {"image_id": "i", "running": False}})
    rec = _make_rec("synspace-abc123", _REPO, mcp_url="http://localhost:9001/mcp",
                    container="c1", scopes=[["u", "first-chat"]])
    alembic_tools._JOBS[rec["job_id"]] = rec
    started = []

    def start(job_id):
        started.append(job_id)
        if not start_ok:
            return {"ok": False, "error": "this build has no image of its own on this host"}
        fields = {"container": "c2", "mcp_url": "http://localhost:9002/mcp"}
        alembic_tools._JOBS[job_id].update(fields)
        inv["containers"]["c2"] = {"image_id": "i", "running": True}
        return {"ok": True, **fields}

    monkeypatch.setattr(alembic_tools, "start_build_server", start)
    monkeypatch.setattr(alembic_tools, "_runner", _noop_runner)
    return started


def test_check_mcp_build_starts_a_stopped_server_again(monkeypatch):
    """The record stayed "done" after Stop, check_mcp_build handed out the dead
    address, the server was not attached and CoderAgent redid the task."""
    started = _stopped_build(monkeypatch)
    listed = asyncio.run(list_mcp_builds())["builds"]
    assert listed[0]["server_running"] is False

    result = asyncio.run(check_mcp_build("synspace-abc123",
                                         tool_context=_scoped_context("u", "second-chat")))

    assert started == ["synspace-abc123"]
    assert result["status"] == "done"
    assert result["mcp_url"] == "http://localhost:9002/mcp"
    assert "started it again" in result["note"]
    assert alembic_tools.live_build_servers(("u", "second-chat")) == {"http://localhost:9002/mcp": "synspace"}


def test_check_mcp_build_gives_no_address_of_a_server_that_does_not_start(monkeypatch):
    _stopped_build(monkeypatch, start_ok=False)

    result = asyncio.run(check_mcp_build("synspace-abc123",
                                         tool_context=_scoped_context("u", "second-chat")))

    assert result["status"] == "done"
    assert "mcp_url" not in result
    assert result["server_running"] is False
    assert "force_rebuild" in result["note"]
    assert alembic_tools.live_build_servers(("u", "second-chat")) == {}


def test_build_mcp_server_starts_a_stopped_server_of_a_known_build(monkeypatch):
    started = _stopped_build(monkeypatch)

    result = asyncio.run(build_mcp_server(_REPO, tool_context=_scoped_context("u", "second-chat")))

    assert started == ["synspace-abc123"]
    assert result["job_id"] == "synspace-abc123"
    assert result["mcp_url"] == "http://localhost:9002/mcp"
    assert "started it again" in result["note"]
    assert list(alembic_tools._JOBS) == ["synspace-abc123"]  # no new build


def test_a_known_build_whose_server_does_not_start_is_built_anew(monkeypatch):
    _stopped_build(monkeypatch, start_ok=False)

    result = asyncio.run(build_mcp_server(_REPO))

    assert result["status"] == "running"
    assert result["job_id"] != "synspace-abc123"


def test_a_started_server_is_handed_out_once_it_answers(monkeypatch):
    """A container that stays up for the settle time can still be loading its
    libraries; its address came with no tool names and the executor missed it."""
    _stopped_build(monkeypatch)
    monkeypatch.setattr(alembic_tools, "_SERVE_READY_POLL", 0)
    calls = []

    async def served(mcp_url):
        calls.append(mcp_url)
        return None if len(calls) < 3 else [{"name": "chemical_space", "description": ""}]

    monkeypatch.setattr(alembic_tools, "_served_tools", served)

    result = asyncio.run(check_mcp_build("synspace-abc123"))

    assert calls == ["http://localhost:9002/mcp"] * 3
    assert result["mcp_url"] == "http://localhost:9002/mcp"
    assert [tool["name"] for tool in result["tools"]] == ["chemical_space"]


def test_a_started_server_that_does_not_answer_gives_no_address(monkeypatch):
    _stopped_build(monkeypatch)
    monkeypatch.setattr(alembic_tools, "_SERVE_READY_TIMEOUT", 0)

    async def silent(mcp_url):
        return None

    monkeypatch.setattr(alembic_tools, "_served_tools", silent)

    checked = asyncio.run(check_mcp_build("synspace-abc123",
                                          tool_context=_scoped_context("u", "second-chat")))
    built = asyncio.run(build_mcp_server(_REPO))

    for result in (checked, built):
        assert "mcp_url" not in result
        assert result["server_running"] is False
        assert "check_mcp_build('synspace-abc123') again" in result["note"]
    assert list(alembic_tools._JOBS) == ["synspace-abc123"]  # waits, no new build


def test_a_server_served_again_after_a_restart_waits_until_it_answers(monkeypatch):
    _disk_build("synspace-bbb222", container="alembic-serve-synspace-dead00",
                image_id="sha256:img")
    _host(monkeypatch, tags={"alembic-tool:synspace-bbb222": "sha256:img"})
    monkeypatch.setattr(alembic_tools, "start_build_server", lambda job_id: {
        "ok": True, "container": "alembic-serve-synspace-beef01",
        "mcp_url": "http://localhost:25002/mcp"})
    monkeypatch.setattr(alembic_tools, "_runner", _noop_runner)
    monkeypatch.setattr(alembic_tools, "_SERVE_READY_TIMEOUT", 0)

    async def silent(mcp_url):
        return None

    monkeypatch.setattr(alembic_tools, "_served_tools", silent)

    result = asyncio.run(build_mcp_server(_REPO))

    assert result["job_id"] == "synspace-bbb222"
    assert "mcp_url" not in result
    assert "again in a minute" in result["note"]


# ── a server from a tool image built outside the builds tool ─────────────────

class _ImageFiles:
    """docker cp out of a container whose image holds ``files`` (path -> text)."""

    def __init__(self, files):
        self.files = files
        self.copied = []

    def __call__(self, *args, timeout=120):
        if args[0] != "cp":
            return subprocess.CompletedProcess(args, 1, "", "unexpected docker call")
        src, dest = args[1].split(":", 1)[1], Path(args[2])
        self.copied.append(src)
        found = {p: t for p, t in self.files.items() if p == src or p.startswith(src + "/")}
        for path, text in found.items():
            target = dest if path == src else dest / path[len(src) + 1:]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        return subprocess.CompletedProcess(args, 0 if found else 1, "", "")


_EXTERNAL = "alembic-serve-synspace-abcdef"


def _external_server(monkeypatch, files):
    _host(monkeypatch, containers={_EXTERNAL: {"image_id": "sha256:x", "running": True}},
          ports={_EXTERNAL: "25003"})
    monkeypatch.setattr(alembic_tools, "_import_image_artifacts", _IMPORT_ARTIFACTS)
    docker = _ImageFiles(files)
    monkeypatch.setattr(alembic_tools, "_docker", docker)
    monkeypatch.setattr(alembic_tools, "_runner", _noop_runner)
    return docker


def test_a_server_from_a_tool_image_gets_its_artifacts_and_a_history(monkeypatch):
    """The image carries its reports and generated tools, but the builds page
    reads them from a build's own workdir, so such a server showed no tools to
    call, no validation and an empty history."""
    root = "/work/.alembic/synspace"
    counts = {"tools_total": 4, "tools_passed": 3, "tools_perfect": 2}
    docker = _external_server(monkeypatch, {
        f"{root}/reports/plan.json": json.dumps({"tools": [{"name": "chemical_space"}]}),
        f"{root}/reports/validation.json": json.dumps({"tools": [], "counts": counts}),
        f"{root}/output/server.py": "x",
        f"{root}/output/.venv/bin/python": "venv",
        f"{root}/pipeline.log": "agent output",
    })

    result = asyncio.run(build_mcp_server(_REPO))

    job_id = result["job_id"]
    workdir = alembic_tools.web_build_workdir(job_id)
    assert (workdir / "synspace" / "reports" / "plan.json").exists()
    assert (workdir / "synspace" / "output" / "server.py").exists()
    assert not any(".venv" in src or "pipeline.log" in src for src in docker.copied)
    assert (result["origin"], result["tool_counts"]) == ("image", counts)
    assert "outside the builds tool" in alembic_tools.web_build_log_file(job_id).read_text()

    alembic_tools._JOBS.clear()  # a restarted process finds it on disk
    [listed] = alembic_tools.web_list_builds()
    assert (listed["job_id"], listed["origin"], listed["tool_counts"]) == (job_id, "image", counts)
    assert listed["mcp_url"] == "http://localhost:25003/mcp"


def test_a_server_whose_image_has_no_alembic_artifacts_is_marked_unknown(monkeypatch):
    _external_server(monkeypatch, {})

    result = asyncio.run(build_mcp_server(_REPO))

    assert result["origin"] == "unknown"
    assert "tool_counts" not in result
    assert alembic_tools.web_build_workdir(result["job_id"]) is None
    assert "no alembic artifacts" in alembic_tools.web_build_log_file(result["job_id"]).read_text()


# ── running servers nobody recorded show up in the builds list ───────────────

def _unrecorded(monkeypatch, ps_lines, files):
    """docker ps lists ``ps_lines``; docker cp copies ``files`` out of any container."""
    copy = _ImageFiles(files)

    def docker(*args, timeout=120):
        if args[0] == "ps":
            return subprocess.CompletedProcess(args, 0, "\n".join(ps_lines), "")
        return copy(*args, timeout=timeout)

    monkeypatch.setattr(alembic_tools, "_docker", docker)
    monkeypatch.setattr(alembic_tools, "_import_image_artifacts", _IMPORT_ARTIFACTS)
    monkeypatch.setattr(alembic_tools, "_last_discovery", 0.0)


def test_a_running_server_nobody_recorded_shows_up_in_the_builds_list(monkeypatch):
    """It reached the builds list only once an agent reused it. An older image
    has no alembic.repo_url label; its plan names the repository."""
    name = "alembic-serve-mordred-7c4adc"
    _host(monkeypatch, containers={name: {"image_id": "sha256:m", "running": True}},
          ports={name: "26963"})
    root = "/work/.alembic/mordred"
    _unrecorded(monkeypatch, [f"{name}|"], {
        f"{root}/reports/plan.json": json.dumps(
            {"repo_url": "https://github.com/mordred-descriptor/mordred", "tools": []}),
        f"{root}/reports/validation.json": json.dumps({"counts": {"tools_total": 4, "tools_passed": 4}}),
    })

    assert alembic_tools.adopt_unclaimed_servers() == ["mordred-external-7c4adc"]
    assert alembic_tools.adopt_unclaimed_servers() == []  # the list polls: one pass per interval
    monkeypatch.setattr(alembic_tools, "_last_discovery", 0.0)
    assert alembic_tools.adopt_unclaimed_servers() == []  # recorded already

    [listed] = alembic_tools.web_list_builds()
    assert (listed["repo_url"], listed["origin"], listed["mcp_url"]) == (
        "https://github.com/mordred-descriptor/mordred", "image", "http://localhost:26963/mcp")
    log = alembic_tools.web_build_log_file(listed["job_id"]).read_text()
    assert f"found {name}" in log


def test_only_unrecorded_servers_that_name_their_repository_are_added(monkeypatch):
    known = "alembic-serve-synspace-af3ffc"
    anonymous = "alembic-serve-mystery-000001"
    labelled = "alembic-serve-gget-b29990"
    _host(monkeypatch,
          containers={n: {"image_id": "sha256:" + n[-6:], "running": True}
                      for n in (known, anonymous, labelled)},
          ports={known: "27951", anonymous: "25010", labelled: "25011"})
    alembic_tools._JOBS["synspace-53de13"] = _make_rec(
        "synspace-53de13", _REPO, mcp_url="http://localhost:27951/mcp", container=known)
    _unrecorded(monkeypatch, [f"{known}|", f"{anonymous}|",
                              f"{labelled}|https://github.com/pachterlab/gget"], {})

    assert alembic_tools.adopt_unclaimed_servers() == ["gget-external-b29990"]
    gget = alembic_tools.web_build_snapshot("gget-external-b29990")
    assert (gget["repo_url"], gget["origin"]) == ("https://github.com/pachterlab/gget", "unknown")
    assert not (alembic_tools.LOG_DIR / "mystery-external-000001").exists()


def test_a_known_build_serving_from_a_new_container_gets_no_second_record(monkeypatch):
    """Start replaced a container whose S3 settings changed. Until the build's
    record named the new container, the builds list took it for a server nobody
    recorded and added mordred-external-<hex> next to mordred-babc36."""
    fresh = "alembic-serve-mordred-1a28bf"
    mordred = "https://github.com/mordred-descriptor/mordred"
    _host(monkeypatch, containers={fresh: {"image_id": "sha256:babc", "running": True}},
          ports={fresh: "27969"}, tags={"alembic-tool:mordred-babc36": "sha256:babc"})
    alembic_tools._JOBS["mordred-babc36"] = _make_rec(
        "mordred-babc36", mordred, mcp_url="http://localhost:27969/mcp",
        container="alembic-serve-mordred-e1bae8", image_id="sha256:babc")
    _unrecorded(monkeypatch, [f"{fresh}|"], {
        "/work/.alembic/mordred/reports/plan.json": json.dumps({"repo_url": mordred})})

    assert alembic_tools.adopt_unclaimed_servers() == []
    assert list(alembic_tools._JOBS) == ["mordred-babc36"]
