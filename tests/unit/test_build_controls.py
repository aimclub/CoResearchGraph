"""Serving controls on the builds list: start, restart and stop a build's MCP
server, and delete its image. Docker is a scripted fake; the catalogue is stubbed.
"""

import asyncio
import json
import subprocess
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from CoScientist.alembic.web import build_api
from CoScientist.tools import alembic_tools

_JOB = "gget-938c68"
_REPO = "https://github.com/pachterlab/gget"
_SERVE = "alembic-serve-gget-b29990"
_OWN = "sha256:own"
_OTHER = "sha256:other"


class _Docker:
    """A scripted docker CLI: images by tag, containers by name, published ports."""

    def __init__(self, images=None, containers=None, ports=None, fail=()):
        self.images = dict(images or {})
        self.containers = {k: dict(v) for k, v in (containers or {}).items()}
        self.ports = dict(ports or {})
        self.fail = set(fail)
        self.calls = []

    def __call__(self, *args, timeout=120):
        self.calls.append(args)

        def out(text="", rc=0, err=""):
            return subprocess.CompletedProcess(args, rc, text, err)

        verb = args[0]
        if verb in self.fail:
            return out(rc=1, err=f"{verb} failed")
        if verb == "images":
            # `docker images` without -a hides an untagged image that only a
            # stopped container still uses, so only tagged images are listed.
            return out("\n".join(f"{i}|{t}|3.35GB" for t, i in self.images.items()))
        if verb == "ps" and "--filter" in args:
            image = args[args.index("--filter") + 1].split("=", 1)[1]
            return out("\n".join(n for n, c in self.containers.items() if c["image_id"] == image))
        if verb == "ps":
            return out("\n".join(self.containers))
        if verb == "inspect":
            fmt, names = args[2], args[3:]
            rows = []
            for name in names:
                c = self.containers.get(name)
                if c is None:
                    return out(rc=1, err="no such container")
                running = str(c["running"]).lower()
                rows.append(f"/{name}|{c['image_id']}|{running}" if fmt.startswith("{{.Name}}")
                            else f"{running}|{c['image_id']}")
            return out("\n".join(rows))
        if verb == "port":
            c = self.containers.get(args[1])
            port = self.ports.get(args[1])
            return out(f"0.0.0.0:{port}" if c and c["running"] and port else "")
        if verb in ("start", "restart"):
            self.containers[args[1]]["running"] = True
            return out(args[1])
        if verb == "stop":
            self.containers[args[1]]["running"] = False
            return out(args[1])
        if verb == "rm":
            self.containers.pop(args[-1], None)
            return out()
        if verb == "image":  # image inspect -f <format> <ref>
            ref = args[-1]
            iid = self.images.get(ref) or (ref if ref in self.images.values() else None)
            if iid is None:
                return out(rc=1)
            if "RepoTags" in args[3]:
                return out(" ".join(t for t, i in self.images.items() if i == iid))
            return out(iid)
        if verb == "rmi":
            for tag in args[1:]:
                self.images.pop(tag, None)
            return out()
        if verb == "logs":
            return out("[entrypoint] No server.py found at /work/.alembic/gget/output/server.py")
        return out()


class _ExitsAtStart(_Docker):
    """A container whose server dies as soon as docker starts it."""

    def __call__(self, *args, timeout=120):
        if args[0] == "start":
            self.calls.append(args)
            return subprocess.CompletedProcess(args, 0, args[1], "")
        return super().__call__(*args, timeout=timeout)


@pytest.fixture
def build(monkeypatch, tmp_path):
    monkeypatch.setattr(alembic_tools, "LOG_DIR", tmp_path)
    monkeypatch.setattr(alembic_tools, "_JOBS", {})
    monkeypatch.setattr(alembic_tools, "_SERVE_SETTLE_SECONDS", 0)
    monkeypatch.delenv("A2A_HOST", raising=False)

    def serve_new_container_must_not_run(job, ref):
        raise AssertionError("a new container was started")

    monkeypatch.setattr(alembic_tools, "_serve_new_container", serve_new_container_must_not_run)

    def make(docker, **meta):
        record = {"job_id": _JOB, "repo_url": _REPO, "status": "done", "container": _SERVE, **meta}
        (tmp_path / f"{_JOB}.json").write_text(json.dumps(record), encoding="utf-8")
        monkeypatch.setattr(alembic_tools, "_docker", docker)
        return lambda: json.loads((tmp_path / f"{_JOB}.json").read_text(encoding="utf-8"))

    return make


class _Registered:
    status = "ok"

    def __init__(self, url):
        self.server_id = "id-" + url.rsplit(":", 1)[1].split("/")[0]


def _catalogue(monkeypatch, *, remove_error=None):
    seen = {"removed": [], "registered": []}

    def unregister(server_id):
        seen["removed"].append(server_id)
        return remove_error

    async def register(mcp_url, name, description="", **kw):
        seen["registered"].append(mcp_url)
        return _Registered(mcp_url)

    monkeypatch.setattr(alembic_tools, "_unregister", unregister)
    monkeypatch.setattr("CoScientist.tools.registry_bridge.register_mcp_server", register)
    return seen


def _running(image=_OWN, running=True):
    return {_SERVE: {"image_id": image, "running": running}}


# ── stop ─────────────────────────────────────────────────────────────────────


def test_stop_takes_a_registered_server_out_of_the_catalogue(build, monkeypatch):
    docker = _Docker(images={f"alembic-tool:{_JOB}": _OWN}, containers=_running())
    meta = build(docker, registered=True, server_id="id-1", mcp_url="http://build-host:1/mcp")
    seen = _catalogue(monkeypatch)

    res = alembic_tools.stop_build_server(_JOB)

    assert res["ok"] is True
    assert ("stop", _SERVE) in docker.calls
    assert seen["removed"] == ["id-1"]
    assert meta().get("registered") is False
    assert not meta().get("server_id")


def test_a_failed_stop_leaves_the_catalogue_alone(build, monkeypatch):
    docker = _Docker(containers=_running(), fail={"stop"})
    meta = build(docker, registered=True, server_id="id-1")
    seen = _catalogue(monkeypatch)

    res = alembic_tools.stop_build_server(_JOB)

    assert res["ok"] is False
    assert seen["removed"] == []
    assert meta()["registered"] is True


def test_a_catalogue_removal_that_failed_is_retried_by_the_next_stop(build, monkeypatch):
    docker = _Docker(containers=_running())
    meta = build(docker, registered=True, server_id="id-1")
    _catalogue(monkeypatch, remove_error="ConnectionError: catalogue down")

    first = alembic_tools.stop_build_server(_JOB)

    assert first["ok"] is True and "not removed" in first["warning"]
    assert meta()["registered"] is True and meta()["server_id"] == "id-1"

    seen = _catalogue(monkeypatch)
    alembic_tools.stop_build_server(_JOB)

    assert seen["removed"] == ["id-1"]
    assert docker.calls.count(("stop", _SERVE)) == 1  # already stopped: only the catalogue is retried
    assert meta().get("registered") is False


# ── start / restart ──────────────────────────────────────────────────────────


def test_start_brings_back_the_stopped_container_and_rechecks_registration(build, monkeypatch):
    """A2A_HOST set after the build: the restart is where the server can join the catalogue."""
    monkeypatch.setenv("A2A_HOST", "build-host")
    docker = _Docker(images={f"alembic-tool:{_JOB}": _OWN}, containers=_running(running=False),
                     ports={_SERVE: "25280"})
    meta = build(docker, registered=False, mcp_url="http://localhost:25280/mcp",
                 registration_error="served on a loopback address")
    seen = _catalogue(monkeypatch)

    res = alembic_tools.start_build_server(_JOB)

    assert res["ok"] is True
    assert ("start", _SERVE) in docker.calls
    assert seen["registered"] == ["http://build-host:25280/mcp"]
    assert meta()["mcp_url"] == "http://build-host:25280/mcp"
    assert meta()["registered"] is True and meta()["server_id"] == "id-25280"
    assert not meta().get("registration_error")


def test_a_loopback_start_stays_out_of_the_catalogue(build, monkeypatch):
    docker = _Docker(images={f"alembic-tool:{_JOB}": _OWN}, containers=_running(running=False),
                     ports={_SERVE: "25280"})
    meta = build(docker, mcp_url="http://localhost:25280/mcp")
    seen = _catalogue(monkeypatch)

    res = alembic_tools.start_build_server(_JOB)

    assert res["ok"] is True
    assert seen["registered"] == []
    assert meta()["registered"] is False
    assert "A2A_HOST" in meta()["registration_error"]


def test_restart_on_a_new_address_drops_the_old_catalogue_row(build, monkeypatch):
    """server_id hashes the url, so a new address is a new row and the old one would linger."""
    monkeypatch.setenv("A2A_HOST", "build-host")
    docker = _Docker(images={f"alembic-tool:{_JOB}": _OWN}, containers=_running(),
                     ports={_SERVE: "2222"})
    meta = build(docker, registered=True, server_id="id-1111", mcp_url="http://build-host:1111/mcp")
    seen = _catalogue(monkeypatch)

    res = alembic_tools.restart_build_server(_JOB)

    assert res["ok"] is True
    assert seen["removed"] == ["id-1111"]
    assert seen["registered"] == ["http://build-host:2222/mcp"]
    assert meta()["server_id"] == "id-2222"


def test_a_server_that_exits_right_after_start_is_reported_with_its_logs(build, monkeypatch):
    docker = _ExitsAtStart(images={f"alembic-tool:{_JOB}": _OWN}, containers=_running(running=False),
                           ports={_SERVE: "25280"})
    build(docker)
    seen = _catalogue(monkeypatch)

    res = alembic_tools.start_build_server(_JOB)

    assert res["ok"] is False
    assert "exited right after start" in res["error"]
    assert "No server.py" in res["error"]
    assert seen["registered"] == []


def test_a_build_without_its_own_image_is_not_started(build, monkeypatch):
    """alembic-tool:<repo> moves to every newer build of the repo; starting from it
    would serve some other build under this build's name."""
    docker = _Docker(images={"alembic-tool:gget": _OTHER})
    build(docker, image="alembic-tool:gget")
    _catalogue(monkeypatch)

    res = alembic_tools.start_build_server(_JOB)

    assert res["ok"] is False
    assert "no image of its own" in res["error"]
    assert not any(call[0] in ("start", "run") for call in docker.calls)


# ── delete image ─────────────────────────────────────────────────────────────


def test_deleting_the_image_removes_its_containers_and_every_tag(build, monkeypatch):
    docker = _Docker(
        images={f"alembic-tool:{_JOB}": _OWN, "alembic-tool:gget": _OWN, "alembic-tool:other": _OTHER},
        containers={**_running(), "unrelated": {"image_id": _OTHER, "running": True}},
    )
    meta = build(docker, registered=True, server_id="id-1")
    seen = _catalogue(monkeypatch)

    res = alembic_tools.delete_build_image(_JOB)

    assert res["ok"] is True
    assert seen["removed"] == ["id-1"]
    assert ("rm", "-f", _SERVE) in docker.calls
    assert "unrelated" in docker.containers
    assert set(docker.images) == {"alembic-tool:other"}
    assert meta()["image_deleted"] is True
    assert alembic_tools.job_image(meta(), alembic_tools.docker_inventory()) is None


def _another_build(tmp_path, job_id, container, **meta):
    record = {"job_id": job_id, "repo_url": _REPO, "status": "done", "container": container, **meta}
    (tmp_path / f"{job_id}.json").write_text(json.dumps(record), encoding="utf-8")
    return lambda: json.loads((tmp_path / f"{job_id}.json").read_text(encoding="utf-8"))


def _shared_image_docker():
    """Two older builds of one repo, both served from the same untagged-per-job image."""
    return _Docker(images={"alembic-tool:gget": _OWN},
                   containers={_SERVE: {"image_id": _OWN, "running": False},
                               "alembic-serve-gget-older": {"image_id": _OWN, "running": False}})


def test_an_image_shared_with_another_build_is_not_deleted_without_confirmation(
    build, monkeypatch, tmp_path
):
    """Deleting the image of one of two such builds took it, and its container,
    from the other build as well."""
    docker = _shared_image_docker()
    build(docker)
    _another_build(tmp_path, "gget-older", "alembic-serve-gget-older")
    _catalogue(monkeypatch)

    res = alembic_tools.delete_build_image(_JOB)

    assert res["ok"] is False
    assert res["shared_with"] == ["gget-older"]
    assert "alembic-tool:gget" in docker.images
    assert len(docker.containers) == 2


def test_a_confirmed_shared_deletion_marks_every_build_that_lost_the_image(
    build, monkeypatch, tmp_path
):
    docker = _shared_image_docker()
    meta = build(docker)
    other = _another_build(tmp_path, "gget-older", "alembic-serve-gget-older",
                           registered=True, server_id="id-older")
    seen = _catalogue(monkeypatch)

    res = alembic_tools.delete_build_image(_JOB, include_shared=True)

    assert res["ok"] is True
    assert res["also_deleted_for"] == ["gget-older"]
    assert docker.images == {} and docker.containers == {}
    assert seen["removed"] == ["id-older"]
    assert meta()["image_deleted"] is True
    assert other()["image_deleted"] is True
    assert other()["registered"] is False


def test_an_image_stays_while_its_catalogue_entry_cannot_be_removed(build, monkeypatch):
    docker = _Docker(images={f"alembic-tool:{_JOB}": _OWN}, containers=_running())
    meta = build(docker, registered=True, server_id="id-1")
    _catalogue(monkeypatch, remove_error="ConnectionError: catalogue down")

    res = alembic_tools.delete_build_image(_JOB)

    assert res["ok"] is False
    assert f"alembic-tool:{_JOB}" in docker.images
    assert _SERVE in docker.containers
    assert meta()["registered"] is True


# ── web endpoints ────────────────────────────────────────────────────────────


def _client():
    app = FastAPI()
    app.include_router(build_api.router)
    return TestClient(app)


def test_controls_can_be_switched_off_for_a_shared_deploy(monkeypatch):
    monkeypatch.setenv("ALEMBIC_WEB_CONTROLS", "0")

    r = _client().post(f"/api/builds/{_JOB}/stop")

    assert r.status_code == 403


def test_an_action_runs_in_the_background_and_the_list_reports_its_outcome(monkeypatch):
    """A restart takes about 9 s, and docker changes network interfaces meanwhile;
    Firefox drops a request left waiting that long. So the POST answers at once."""
    monkeypatch.delenv("ALEMBIC_WEB_CONTROLS", raising=False)
    monkeypatch.setattr(build_api, "_action_state", {})
    monkeypatch.setattr(alembic_tools, "web_build_snapshot", lambda jid: {"job_id": jid, "status": "done"})
    monkeypatch.setattr(alembic_tools, "web_list_builds", lambda: [{"job_id": _JOB, "status": "done"}])
    monkeypatch.setattr(alembic_tools, "docker_inventory",
                        lambda: {"images": {}, "tags": {}, "containers": {}})
    release = threading.Event()

    def stop(job_id):
        release.wait(5)
        return {"ok": True, "job": job_id}

    monkeypatch.setattr(alembic_tools, "stop_build_server", stop)
    app = FastAPI()
    app.include_router(build_api.router)

    with TestClient(app) as client:
        accepted = client.post(f"/api/builds/{_JOB}/stop")
        running = client.get("/api/builds").json()["builds"][0]["action"]
        second = client.post(f"/api/builds/{_JOB}/stop")
        release.set()
        for _ in range(100):
            action = client.get("/api/builds").json()["builds"][0]["action"]
            if not action["running"]:
                break
            time.sleep(0.05)
        unknown = client.post(f"/api/builds/{_JOB}/explode")

    assert accepted.status_code == 202
    assert running["name"] == "stop" and running["running"] is True
    assert second.status_code == 409
    assert action["result"] == {"ok": True, "job": _JOB}
    assert unknown.status_code == 404


def test_the_shared_deletion_confirmation_reaches_the_control(monkeypatch):
    monkeypatch.delenv("ALEMBIC_WEB_CONTROLS", raising=False)
    monkeypatch.setattr(build_api, "_action_state", {})
    monkeypatch.setattr(alembic_tools, "web_build_snapshot", lambda jid: {"job_id": jid, "status": "done"})
    received = []

    def delete(job_id, include_shared=False):
        received.append(include_shared)
        return {"ok": True}

    monkeypatch.setattr(alembic_tools, "delete_build_image", delete)
    app = FastAPI()
    app.include_router(build_api.router)

    with TestClient(app) as client:
        accepted = client.post(f"/api/builds/{_JOB}/delete-image?include_shared=1")
        for _ in range(100):
            if received:
                break
            time.sleep(0.05)

    assert accepted.status_code == 202
    assert received == [True]


def test_unregistering_removes_the_row_by_server_id():
    from CoScientist.tools.registry_bridge import unregister_mcp_server

    class _Manager:
        def __init__(self):
            self.removed = []

        async def remove_server(self, server_id):
            self.removed.append(server_id)
            return {}

        async def close(self):
            raise AssertionError("an injected manager is the caller's to close")

    manager = _Manager()
    asyncio.run(unregister_mcp_server("id-1", manager=manager))

    assert manager.removed == ["id-1"]


# ── clear non-runnable ───────────────────────────────────────────────────────


def _history(tmp_path, job_id, **meta):
    record = {"job_id": job_id, "repo_url": _REPO, **meta}
    (tmp_path / f"{job_id}.json").write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / f"{job_id}.log").write_text("build log\n", encoding="utf-8")


def test_clearing_removes_only_builds_that_cannot_be_served(build, monkeypatch, tmp_path):
    docker = _Docker(images={f"alembic-tool:{_JOB}": _OWN},
                     containers={**_running(), "alembic-serve-gget-dead": {"image_id": "sha256:gone",
                                                                           "running": False}})
    build(docker)  # runnable through its own tag
    _history(tmp_path, "gget-noimage", status="done", container="alembic-serve-gget-dead")
    (tmp_path / "gget-noimage" / "workdir").mkdir(parents=True)
    _history(tmp_path, "gget-failed", status="failed")
    _history(tmp_path, "gget-running", status="running")
    _catalogue(monkeypatch)

    res = alembic_tools.clear_non_runnable_builds()

    assert res["ok"] is True
    assert sorted(res["removed"]) == ["gget-failed", "gget-noimage"]
    assert ("rm", "alembic-serve-gget-dead") in docker.calls
    assert ("rmi", "sha256:gone") in docker.calls  # the untagged image only it still used
    assert not (tmp_path / "gget-noimage").exists()
    assert not (tmp_path / "gget-noimage.log").exists()
    assert not (tmp_path / "gget-failed.json").exists()
    assert (tmp_path / "gget-running.json").exists()
    assert (tmp_path / f"{_JOB}.json").exists()


def test_a_build_whose_catalogue_entry_cannot_be_removed_stays(build, monkeypatch, tmp_path):
    build(_Docker(), status="failed", registered=True, server_id="id-1")
    _catalogue(monkeypatch, remove_error="ConnectionError: catalogue down")

    res = alembic_tools.clear_non_runnable_builds()

    assert res["ok"] is False
    assert res["kept"][0]["job_id"] == _JOB
    assert "catalogue" in res["kept"][0]["reason"]
    assert (tmp_path / f"{_JOB}.json").exists()


def test_clearing_runs_in_the_background_and_the_list_reports_it(monkeypatch):
    monkeypatch.delenv("ALEMBIC_WEB_CONTROLS", raising=False)
    monkeypatch.setattr(build_api, "_action_state", {})
    monkeypatch.setattr(alembic_tools, "web_list_builds", lambda: [])
    monkeypatch.setattr(alembic_tools, "docker_inventory",
                        lambda: {"images": {}, "tags": {}, "containers": {}})
    monkeypatch.setattr(alembic_tools, "clear_non_runnable_builds",
                        lambda: {"ok": True, "removed": ["gget-failed"], "kept": []})
    app = FastAPI()
    app.include_router(build_api.router)

    with TestClient(app) as client:
        accepted = client.post("/api/builds/cleanup")
        for _ in range(100):
            cleanup = client.get("/api/builds").json()["cleanup"]
            if cleanup and not cleanup["running"]:
                break
            time.sleep(0.05)

    assert accepted.status_code == 202
    assert cleanup["result"]["removed"] == ["gget-failed"]
