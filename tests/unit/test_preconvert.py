"""Unit tests for the alembic pre-conversion batch (docker-free).

Drives the batch (``run_batch_async``) with injected fakes for the three real
steps (reachability precheck, alembic convert, registry register) so the batch
logic — manifest parsing, dry-run planning, ALEMBIC_HINTS wiring, and the
per-item failure isolation (never abort) — is verified without Docker or a live
registry. Model on test_registry_bridge.py's FakeManager style.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MOD_PATH = (
    Path(__file__).resolve().parents[2] / "CoScientist" / "alembic" / "preconvert.py"
)
_spec = importlib.util.spec_from_file_location("preconvert", _MOD_PATH)
pc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pc
_spec.loader.exec_module(pc)


def _manifest(tmp_path, tools, benchmark="tmbench"):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"benchmark": benchmark, "tools": tools}), encoding="utf-8")
    return p


# ── manifest parsing ─────────────────────────────────────────────────────────


def test_load_manifest_parses_items_and_defaults_name(tmp_path):
    path = _manifest(
        tmp_path,
        [
            {"repo": "https://github.com/org/foo.git", "hint": "a sketch"},
            {"repo": "https://github.com/org/bar", "name": "custom", "hint": ""},
        ],
    )
    m = pc.load_manifest(path)
    assert m.benchmark == "tmbench"
    assert [i.name for i in m.items] == ["foo", "custom"]  # basename default
    assert m.items[0].hint == "a sketch"


def test_load_manifest_parses_task_spec_and_mount_dir(tmp_path):
    path = _manifest(
        tmp_path,
        [
            {
                "repo": "https://github.com/PriorLabs/TabPFN",
                "name": "tabpfn_predict",
                "task_spec": "$TOOLMAKER_ROOT/benchmark/tasks/tabpfn_predict.yaml",
                "mount_dir": "$TOOLMAKER_ROOT/benchmark/data",
            },
            {"repo": "https://github.com/org/plain", "hint": "just a hint"},
        ],
    )
    m = pc.load_manifest(path)
    assert m.items[0].task_spec.endswith("tabpfn_predict.yaml")
    assert m.items[0].mount_dir.endswith("benchmark/data")
    # a plain item keeps task_spec/mount_dir None
    assert m.items[1].task_spec is None and m.items[1].mount_dir is None


def test_plan_threads_gold_task_and_mount(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLMAKER_ROOT", "/tm")
    m = pc.load_manifest(
        _manifest(
            tmp_path,
            [
                {
                    "repo": "https://github.com/PriorLabs/TabPFN",
                    "name": "tabpfn_predict",
                    "hint": "a tabular predictor",
                    "task_spec": "$TOOLMAKER_ROOT/benchmark/tasks/tabpfn_predict.yaml",
                    "mount_dir": "$TOOLMAKER_ROOT/benchmark/data",
                }
            ],
        )
    )
    plan = pc._plan(m.items[0])
    # gold task → ALEMBIC_TASKS (expanded); soft hint still present
    assert plan["env"]["ALEMBIC_TASKS"] == "/tm/benchmark/tasks/tabpfn_predict.yaml"
    assert plan["env"]["ALEMBIC_HINTS"] == "a tabular predictor"
    # data dir forwarded as --mount-dir (expanded)
    assert "--mount-dir /tm/benchmark/data" in plan["command"]


def test_gpus_forwarded_to_build_command(tmp_path):
    m = pc.load_manifest(
        _manifest(
            tmp_path,
            [
                {"repo": "https://github.com/x/gpu", "name": "g", "gpus": "all"},
                {"repo": "https://github.com/x/cpu", "name": "c"},
            ],
        )
    )
    gpu, cpu = m.items
    assert gpu.gpus == "all" and cpu.gpus is None
    # GPU item forwards --gpus all to start_chain; CPU item omits it entirely
    assert "--gpus all" in " ".join(pc.build_command(gpu))
    assert "--gpus" not in " ".join(pc.build_command(cpu))
    assert pc._plan(gpu)["gpus"] == "all"


def test_stage_volume_forwarded_to_build_command(tmp_path):
    m = pc.load_manifest(
        _manifest(
            tmp_path,
            [
                {
                    "repo": "https://github.com/x/heavy",
                    "name": "h",
                    "stage_volume": "tmbench-crc",
                },
                {"repo": "https://github.com/x/plain", "name": "p"},
            ],
        )
    )
    heavy, plain = m.items
    assert heavy.stage_volume == "tmbench-crc" and plain.stage_volume is None
    # a reuse volume forwards --stage-volume; a plain item omits it entirely
    assert "--stage-volume tmbench-crc" in " ".join(pc.build_command(heavy))
    assert "--stage-volume" not in " ".join(pc.build_command(plain))


def test_task_spec_inlined_as_content_when_file_exists(tmp_path):
    # server.py is generated inside a container on the build daemon (often
    # remote), where a *local* task-spec path does not exist — so the spec must
    # cross the boundary as inline content, not a path.
    spec = tmp_path / "tabpfn_predict.yaml"
    spec.write_text(
        "name: tabpfn_predict\narguments: {train_csv: str}\n", encoding="utf-8"
    )
    assert pc._task_spec_env(str(spec)) == spec.read_text(encoding="utf-8")
    # a non-file value (already inline, or an in-container path) passes through
    assert pc._task_spec_env("/does/not/exist.yaml") == "/does/not/exist.yaml"


def test_plan_threads_gold_task_content_for_real_file(tmp_path, monkeypatch):
    spec = tmp_path / "tabpfn_predict.yaml"
    spec.write_text("name: tabpfn_predict\n", encoding="utf-8")
    monkeypatch.setenv("TOOLMAKER_ROOT", str(tmp_path))
    m = pc.load_manifest(
        _manifest(
            tmp_path,
            [
                {
                    "repo": "https://github.com/PriorLabs/TabPFN",
                    "name": "tabpfn_predict",
                    "task_spec": "$TOOLMAKER_ROOT/tabpfn_predict.yaml",
                }
            ],
        )
    )
    plan = pc._plan(m.items[0])
    # env carries the CONTENT (survives the daemon boundary); the reported
    # task_spec field keeps the human-readable resolved path.
    assert plan["env"]["ALEMBIC_TASKS"] == "name: tabpfn_predict\n"
    assert plan["task_spec"] == str(spec)


def test_load_manifest_rejects_missing_repo(tmp_path):
    path = _manifest(tmp_path, [{"name": "x", "hint": "y"}])
    with pytest.raises(ValueError, match="repo"):
        pc.load_manifest(path)


def test_load_manifest_rejects_empty_tools(tmp_path):
    path = _manifest(tmp_path, [])
    with pytest.raises(ValueError, match="non-empty"):
        pc.load_manifest(path)


# ── dry-run: plans only, never builds ────────────────────────────────────────


def test_dry_run_plans_without_building(tmp_path):
    m = pc.load_manifest(
        _manifest(
            tmp_path, [{"repo": "https://github.com/org/foo", "hint": "sketch it"}]
        )
    )

    def _boom_converter(*a, **k):  # must never be called in a dry-run
        raise AssertionError("converter called during dry-run")

    records = asyncio.run(
        pc.run_batch_async(
            m,
            dry_run=True,
            converter=_boom_converter,
            precheck=lambda _r: (True, ""),
        )
    )
    assert len(records) == 1
    rec = records[0]
    assert rec["status"] == "planned"
    # The sketch is wired to ALEMBIC_HINTS and the build command is start_chain.
    assert rec["plan"]["env"] == {"ALEMBIC_HINTS": "sketch it"}
    assert "start_chain.py" in rec["plan"]["command"]
    assert rec["plan"]["command"].endswith("https://github.com/org/foo")


# ── convert → register happy path (fakes) ────────────────────────────────────


class FakeServer:
    def __init__(self, server_id, status="active"):
        self.server_id = server_id
        self.status = type("S", (), {"value": status})()


def test_batch_converts_then_registers(tmp_path):
    m = pc.load_manifest(
        _manifest(
            tmp_path,
            [
                {"repo": "https://github.com/org/foo", "hint": "h1"},
                {"repo": "https://github.com/org/bar", "hint": "h2"},
            ],
        )
    )
    seen_hints = []

    def fake_convert(item, *, timeout=None):
        seen_hints.append(item.hint)
        return {"returncode": 0, "mcp_url": f"http://localhost/{item.name}/mcp"}

    registered = []

    async def fake_register(mcp_url, name, *, manager=None):
        registered.append((mcp_url, name))
        return FakeServer(server_id=f"id-{name}")

    records = asyncio.run(
        pc.run_batch_async(
            m,
            converter=fake_convert,
            registrar=fake_register,
            precheck=lambda _r: (True, ""),
        )
    )
    assert {r["status"] for r in records} == {"registered"}
    assert [r["mcp_url"] for r in records] == [
        "http://localhost/foo/mcp",
        "http://localhost/bar/mcp",
    ]
    assert [r["server_id"] for r in records] == ["id-foo", "id-bar"]
    assert sorted(seen_hints) == ["h1", "h2"]  # per-item hint reached the converter


def test_failed_sync_is_registered_with_errors(tmp_path):
    m = pc.load_manifest(
        _manifest(tmp_path, [{"repo": "https://github.com/org/foo", "hint": "h"}])
    )

    async def fake_register(mcp_url, name, *, manager=None):
        return FakeServer(server_id="id", status="error")  # sync failed downstream

    records = asyncio.run(
        pc.run_batch_async(
            m,
            converter=lambda item, timeout=None: {
                "returncode": 0,
                "mcp_url": "http://x/mcp",
            },
            registrar=fake_register,
            precheck=lambda _r: (True, ""),
        )
    )
    assert records[0]["status"] == "registered_with_errors"


# ── per-item failure isolation: one bad item never aborts the batch ──────────


def test_unreachable_repo_is_skipped_not_fatal(tmp_path):
    m = pc.load_manifest(
        _manifest(
            tmp_path,
            [
                {"repo": "https://github.com/org/dead", "hint": "h"},
                {"repo": "https://github.com/org/live", "hint": "h"},
            ],
        )
    )

    def precheck(repo):
        return (False, "unreachable/empty repo") if "dead" in repo else (True, "")

    async def fake_register(mcp_url, name, *, manager=None):
        return FakeServer(server_id=f"id-{name}")

    records = asyncio.run(
        pc.run_batch_async(
            m,
            converter=lambda item, timeout=None: {
                "returncode": 0,
                "mcp_url": "http://x/mcp",
            },
            registrar=fake_register,
            precheck=precheck,
        )
    )
    by_name = {r["name"]: r for r in records}
    assert by_name["dead"]["status"] == "skipped"
    assert by_name["live"]["status"] == "registered"


def test_converter_exception_is_isolated(tmp_path):
    m = pc.load_manifest(
        _manifest(
            tmp_path,
            [
                {"repo": "https://github.com/org/boom", "hint": "h"},
                {"repo": "https://github.com/org/ok", "hint": "h"},
            ],
        )
    )

    def convert(item, *, timeout=None):
        if "boom" in item.repo:
            raise RuntimeError("docker exploded")
        return {"returncode": 0, "mcp_url": "http://x/mcp"}

    async def fake_register(mcp_url, name, *, manager=None):
        return FakeServer(server_id="id")

    records = asyncio.run(
        pc.run_batch_async(
            m,
            converter=convert,
            registrar=fake_register,
            precheck=lambda _r: (True, ""),
        )
    )
    by_name = {r["name"]: r for r in records}
    assert by_name["boom"]["status"] == "failed"
    assert "docker exploded" in by_name["boom"]["error"]
    assert by_name["ok"]["status"] == "registered"  # the batch kept going


def test_no_mcp_url_is_a_failure(tmp_path):
    m = pc.load_manifest(
        _manifest(tmp_path, [{"repo": "https://github.com/org/foo", "hint": "h"}])
    )
    records = asyncio.run(
        pc.run_batch_async(
            m,
            converter=lambda item, timeout=None: {
                "returncode": 0,
                "mcp_url": None,
                "error": "serving skipped",
            },
            registrar=None,
            precheck=lambda _r: (True, ""),
        )
    )
    assert records[0]["status"] == "failed"
    assert "serving skipped" in records[0]["error"]


def test_build_manager_default_delegates_to_bridge(monkeypatch):
    """Without --local-embedder the batch reuses registry_bridge's APIEmbedder
    manager (so the index matches Retrieve_tools)."""
    import CoScientist.tools.registry_bridge as rb

    called = {"n": 0}

    async def fake_default():
        called["n"] += 1
        return "MANAGER"

    monkeypatch.setattr(rb, "_default_manager", fake_default)
    result = asyncio.run(pc.build_manager(local_embedder=False))
    assert result == "MANAGER" and called["n"] == 1


def test_summarize_counts_statuses():
    counts = pc.summarize(
        [{"status": "registered"}, {"status": "registered"}, {"status": "failed"}]
    )
    assert counts == {"registered": 2, "failed": 1}


# ── shared-manager safety (regression: the live Phase-1 failure) ──────────────


def test_concurrent_registration_is_serialized(tmp_path):
    """All items share ONE rag_tools manager (a single asyncpg connection); a
    reg_lock must serialize the writes, else parallel items race the shared
    session (asyncpg "another operation is in progress")."""
    m = pc.load_manifest(
        _manifest(
            tmp_path,
            [
                {"repo": "https://github.com/org/a", "hint": "h"},
                {"repo": "https://github.com/org/b", "hint": "h"},
            ],
        )
    )
    active = {"n": 0, "max": 0}

    async def fake_register(mcp_url, name, *, manager=None):
        active["n"] += 1
        active["max"] = max(active["max"], active["n"])
        await asyncio.sleep(0.01)  # hold the "connection" so a race would overlap
        active["n"] -= 1
        return FakeServer(server_id=f"id-{name}")

    records = asyncio.run(
        pc.run_batch_async(
            m,
            parallel=2,
            converter=lambda item, *, timeout=None: {
                "returncode": 0,
                "mcp_url": f"http://h/{item.name}/mcp",
            },
            registrar=fake_register,
            precheck=lambda _r: (True, ""),
        )
    )
    assert {r["status"] for r in records} == {"registered"}
    assert active["max"] == 1  # never two registrations in flight at once


def test_run_batch_builds_uses_and_closes_manager_in_one_loop(monkeypatch):
    """Regression: main() used three separate asyncio.run() calls, so the manager's
    asyncpg pool was created in one loop and used in another (closed) loop. run_batch
    must build, use, and close the manager in a SINGLE loop."""
    seen = {}

    class FakeMgr:
        def __init__(self):
            seen["build_loop"] = asyncio.get_running_loop()

        async def close(self):
            seen["close_loop"] = asyncio.get_running_loop()

    async def fake_build(local_embedder=False):
        return FakeMgr()

    reg_loops = []

    async def fake_register(mcp_url, name, *, manager=None):
        reg_loops.append(asyncio.get_running_loop())
        assert isinstance(manager, FakeMgr)  # the shared manager is threaded through
        return FakeServer(server_id=f"id-{name}")

    monkeypatch.setattr(pc, "build_manager", fake_build)
    manifest = pc.Manifest(
        benchmark="t",
        items=[
            pc.ManifestItem(repo="r1", name="a", hint=None),
            pc.ManifestItem(repo="r2", name="b", hint=None),
        ],
    )
    records = asyncio.run(
        pc.run_batch(
            manifest,
            parallel=2,
            converter=lambda item, *, timeout=None: {
                "returncode": 0,
                "mcp_url": f"http://h/{item.name}/mcp",
            },
            registrar=fake_register,
            precheck=lambda _r: (True, ""),
        )
    )
    assert [r["status"] for r in records] == ["registered", "registered"]
    # one loop throughout: build, every registration, and close share it
    assert seen["build_loop"] is seen["close_loop"]
    assert all(loop is seen["build_loop"] for loop in reg_loops)
