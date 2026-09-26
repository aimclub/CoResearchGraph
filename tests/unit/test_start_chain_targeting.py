"""start_chain builds where it is told and advertises an address that works.

start_chain imports the alembic package under its container layout, so the
module is loaded standalone with those dependencies stubbed — the same reason
tests/unit/_codegen_loader.py exists.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import types
from pathlib import Path

import pytest

_START_CHAIN = (
    Path(__file__).resolve().parents[2] / "CoScientist" / "alembic" / "start_chain.py"
)


def _load():
    for name in ("alembic", "alembic.common", "alembic.remote", "alembic.targets"):
        sys.modules.pop(name, None)
    pkg = types.ModuleType("alembic")
    pkg.__path__ = [str(_START_CHAIN.parent)]
    sys.modules["alembic"] = pkg
    spec = importlib.util.spec_from_file_location("alembic_start_chain_under_test", _START_CHAIN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load()


def _ns(**kw):
    base = dict(mount_dir=None, context=None, stage_volume=None, advertise_host=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_no_data_directory_means_no_mount(monkeypatch):
    assert sc._mount_args("repo", _ns()) == []


def test_a_local_build_binds_the_host_path(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd))

    args = sc._mount_args("repo", _ns(mount_dir=str(tmp_path)))

    assert args == ["-v", f"{tmp_path}:/mount/data:ro"]
    assert ran == []  # nothing to stage


def test_a_remote_build_stages_the_data_first(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd) or _Ok())
    monkeypatch.setattr(sc, "context_endpoint", lambda ctx: "ssh://user@gpu-box:22")

    args = sc._mount_args("repo", _ns(mount_dir=str(tmp_path), context="gpu"))

    assert any("cp" in cmd for cmd in ran)  # the data was copied over
    assert args[0] == "-v" and args[1].endswith(":/mount/data:ro")
    assert str(tmp_path) not in args[1]  # a local path would not exist there


def test_a_named_volume_that_already_exists_is_reused(tmp_path, monkeypatch):
    """Staging a large dataset again on every build is the cost this avoids."""
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd) or _Ok())
    monkeypatch.setattr(sc, "context_endpoint", lambda ctx: "ssh://user@gpu-box:22")
    monkeypatch.setattr(sc, "_volume_exists", lambda ctx, vol: True)

    args = sc._mount_args("repo", _ns(mount_dir=str(tmp_path), context="gpu", stage_volume="ct"))

    assert ran == []
    assert args == ["-v", "ct:/mount/data:ro"]


class _Ok:
    returncode = 0


def test_a_soft_hint_reaches_the_build_container(monkeypatch, tmp_path):
    """--hints is documented in the README; it has to actually arrive."""
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd) or _Ok())
    ns = argparse.Namespace(
        platform=None, gpus=None, mount_dir=None, context=None, stage_volume=None,
        advertise_host=None, env_file=tmp_path / "absent.env", resume=None, until=None,
        hints="a tool for drug and disease associations",
    )

    sc.build_image("https://github.com/org/repo", ns)

    build = " ".join(ran[0])
    assert "ALEMBIC_HINTS=a tool for drug and disease associations" in build


def test_no_hint_adds_nothing(monkeypatch, tmp_path):
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd) or _Ok())
    ns = argparse.Namespace(
        platform=None, gpus=None, mount_dir=None, context=None, stage_volume=None,
        advertise_host=None, env_file=tmp_path / "absent.env", resume=None, until=None,
        hints=None,
    )

    sc.build_image("https://github.com/org/repo", ns)

    assert "ALEMBIC_HINTS" not in " ".join(ran[0])


def test_the_api_version_pin_reaches_the_docker_call(monkeypatch):
    """An old daemon rejects the client's default version. The pin has to be on
    the call, since exporting it would break every newer daemon."""
    seen = {}
    monkeypatch.setattr(sc.subprocess, "run", lambda cmd, **kw: seen.update(kw) or _Ok())
    monkeypatch.setattr(sc, "_API_VERSION", "1.43")

    sc._run(["docker", "info"])

    assert seen["env"]["DOCKER_API_VERSION"] == "1.43"


def test_without_a_pin_the_docker_call_keeps_the_ambient_environment(monkeypatch):
    monkeypatch.delenv("DOCKER_API_VERSION", raising=False)
    seen = {}
    monkeypatch.setattr(sc.subprocess, "run", lambda cmd, **kw: seen.update(kw) or _Ok())
    monkeypatch.setattr(sc, "_API_VERSION", None)

    sc._run(["docker", "info"])

    assert "DOCKER_API_VERSION" not in seen["env"]


def _s3_ns(monkeypatch, tmp_path, env_text, **kw):
    for name in sc.SERVE_ONLY_ENV:
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(env_text, encoding="utf-8")
    return _ns(env_file=env_file, **kw)


def test_a_loopback_s3_endpoint_reaches_the_serve_container_through_the_host(monkeypatch, tmp_path):
    ns = _s3_ns(monkeypatch, tmp_path, "S3__ENDPOINT_URL=http://0.0.0.0:9000\n")

    args, env = sc._s3_endpoint_args(ns)

    assert args == ["--add-host", "host.docker.internal:host-gateway"]
    assert env == {
        "S3__ENDPOINT_URL": "http://host.docker.internal:9000",
        "S3__EXTERNAL_ENDPOINT_URL": "http://0.0.0.0:9000",
    }


def test_a_remote_s3_endpoint_is_passed_as_is(monkeypatch, tmp_path):
    ns = _s3_ns(monkeypatch, tmp_path, "S3__ENDPOINT_URL=https://storage.yandexcloud.net\n")
    assert sc._s3_endpoint_args(ns) == ([], {})


def test_an_explicit_external_endpoint_is_kept(monkeypatch, tmp_path):
    ns = _s3_ns(monkeypatch, tmp_path,
                "ENDPOINT_URL=http://localhost:9000\nS3__EXTERNAL_ENDPOINT_URL=http://minio.lan:9000\n")

    _, env = sc._s3_endpoint_args(ns)

    assert env == {"ENDPOINT_URL": "http://host.docker.internal:9000"}


def test_a_loopback_s3_endpoint_on_a_remote_daemon_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(sc, "context_endpoint", lambda ctx: "ssh://user@gpu-box:22")
    ns = _s3_ns(monkeypatch, tmp_path, "S3__ENDPOINT_URL=http://localhost:9000\n", context="gpu")
    assert sc._s3_endpoint_args(ns) == ([], {})


def test_an_env_override_replaces_the_value_instead_of_repeating_it(monkeypatch, tmp_path):
    ns = _s3_ns(monkeypatch, tmp_path, "S3__ENDPOINT_URL=http://0.0.0.0:9000\n")

    args = sc._env_args(ns.env_file, extra_env=sc.SERVE_ONLY_ENV,
                        overrides={"S3__ENDPOINT_URL": "http://host.docker.internal:9000"})

    endpoint = [a for a in args if a.startswith("S3__ENDPOINT_URL=")]
    assert endpoint == ["S3__ENDPOINT_URL=http://host.docker.internal:9000"]


class _Pipe:
    def close(self):
        pass


class _Tar:
    """Stands in for the tar process that streams the host workdir."""

    def __init__(self, *a, **kw):
        self.stdout = _Pipe()

    def wait(self):
        return 0


def _build_ns(tmp_path, **kw):
    base = dict(
        platform=None, gpus=None, mount_dir=None, context=None, stage_volume=None,
        advertise_host=None, env_file=tmp_path / "absent.env", resume=None, until=None,
        hints=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_the_host_workdir_ends_up_inside_the_tool_image(monkeypatch, tmp_path):
    """docker commit leaves bind mounts out. Without the bake step the image
    has no server.py and the serve container exits at start."""
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd) or _Ok())
    monkeypatch.setattr(sc.subprocess, "Popen", _Tar)
    monkeypatch.setenv("ALEMBIC_HOST_WORKDIR", str(tmp_path / "work"))

    image = sc.build_image("https://github.com/org/repo", _build_ns(tmp_path))

    lines = [" ".join(map(str, c)) for c in ran]
    commits = [i for i, line in enumerate(lines) if " commit " in line]
    copy = next(i for i, line in enumerate(lines) if " cp - alembic-bake-" in line)
    assert f"{tmp_path / 'work'}:/work/.alembic" in lines[0]
    assert len(commits) == 2 and commits[0] < copy < commits[1]
    assert lines[commits[1]].endswith(f" {image}")
    assert any("chown" in line for line in lines)  # root-owned files go back to the user


def test_a_remote_build_does_not_mount_a_local_workdir(monkeypatch, tmp_path):
    """The path would resolve on the remote daemon, where it does not exist."""
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd) or _Ok())
    monkeypatch.setattr(sc.subprocess, "Popen", _Tar)
    monkeypatch.setenv("ALEMBIC_HOST_WORKDIR", str(tmp_path / "work"))

    sc.build_image("https://github.com/org/repo", _build_ns(tmp_path, context="gpu"))

    lines = [" ".join(map(str, c)) for c in ran]
    assert ":/work/.alembic" not in lines[0]
    assert not any("alembic-bake-" in line for line in lines)


class _Inspect:
    returncode = 0

    def __init__(self, running):
        self.stdout = running


def _serve(monkeypatch, tmp_path, running):
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: _Ok())
    monkeypatch.setattr(sc.subprocess, "run", lambda cmd, **kw: _Inspect(running))
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    sc.serve_image("https://github.com/org/repo", "alembic-tool:repo", _build_ns(tmp_path))


def test_a_server_that_dies_at_start_fails_the_build(monkeypatch, tmp_path):
    """"MCP server up" used to be printed right after docker run -d. A
    container that exits at once (no server.py) has to fail the build."""
    with pytest.raises(SystemExit):
        _serve(monkeypatch, tmp_path, "false\n")


def test_a_server_that_stays_up_is_reported(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sc, "SERVE_SETTLE_SECONDS", 0.01)

    _serve(monkeypatch, tmp_path, "true\n")

    assert "MCP server up." in capsys.readouterr().out


def test_a_web_build_also_gets_its_own_image_tag(monkeypatch, tmp_path):
    """alembic-tool:<repo> moves with every build of the repo; the job tag keeps
    pointing at this build, so it can be served again later."""
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd) or _Ok())
    monkeypatch.delenv("ALEMBIC_HOST_WORKDIR", raising=False)
    monkeypatch.setenv("ALEMBIC_JOB_ID", "repo-abc123")

    image = sc.build_image("https://github.com/org/repo", _build_ns(tmp_path))

    assert image == "alembic-tool:repo-abc123"
    assert ran[-1][-3:] == ["tag", "alembic-tool:repo", "alembic-tool:repo-abc123"]


def test_without_a_job_the_image_keeps_only_the_repo_tag(monkeypatch, tmp_path):
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd) or _Ok())
    monkeypatch.delenv("ALEMBIC_HOST_WORKDIR", raising=False)
    monkeypatch.delenv("ALEMBIC_JOB_ID", raising=False)

    image = sc.build_image("https://github.com/org/repo", _build_ns(tmp_path))

    assert image == "alembic-tool:repo"
    assert not any("tag" in cmd for cmd in ran)


class _Fail:
    returncode = 1


def _main(monkeypatch, *argv, image_found=True):
    served = {}
    monkeypatch.setattr(sys, "argv", ["start_chain", "https://github.com/org/repo", *argv])
    monkeypatch.setattr(sc, "detect_gpu", lambda *a, **kw: False)
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: _Ok() if image_found else _Fail())
    monkeypatch.setattr(sc, "ensure_base_image", lambda *a, **kw: pytest.fail("built the base image"))
    monkeypatch.setattr(sc, "build_image", lambda *a, **kw: pytest.fail("ran the pipeline"))
    monkeypatch.setattr(sc, "serve_image", lambda url, image, ns: served.update(image=image))
    sc.main()
    return served


def test_serve_only_serves_the_given_image_without_a_build(monkeypatch):
    served = _main(monkeypatch, "--serve-only", "--image", "alembic-tool:repo-abc123")

    assert served == {"image": "alembic-tool:repo-abc123"}


def test_serve_only_defaults_to_the_repo_image(monkeypatch):
    assert _main(monkeypatch, "--serve-only") == {"image": "alembic-tool:repo"}


def test_serve_only_without_the_image_fails(monkeypatch):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--serve-only", image_found=False)


def test_the_committed_image_records_where_it_came_from(monkeypatch, tmp_path):
    """An image that travels (a registry, the hub) has to name its repository
    and build and say how its tools validated."""
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd) or _Ok())
    monkeypatch.setattr(sc.subprocess, "Popen", _Tar)
    work = tmp_path / "work"
    (work / "repo" / "reports").mkdir(parents=True)
    (work / "repo" / "reports" / "validation.json").write_text(
        json.dumps({"counts": {"tools_total": 4, "tools_passed": 3, "tools_perfect": 2}}))
    monkeypatch.setenv("ALEMBIC_HOST_WORKDIR", str(work))
    monkeypatch.setenv("ALEMBIC_JOB_ID", "repo-abc123")

    sc.build_image("https://github.com/org/repo", _build_ns(tmp_path))

    commit = next(cmd for cmd in ran if "commit" in cmd)
    labels = {commit[i + 1] for i, arg in enumerate(commit)
              if arg == "--change" and commit[i + 1].startswith("LABEL ")}
    assert labels == {'LABEL alembic.repo_url="https://github.com/org/repo"',
                      'LABEL alembic.job_id="repo-abc123"', 'LABEL alembic.tools_total="4"',
                      'LABEL alembic.tools_passed="3"', 'LABEL alembic.tools_perfect="2"'}


def test_without_a_mounted_workdir_the_pipeline_log_is_emptied_before_the_commit(
        monkeypatch, tmp_path):
    """The build container has exited by commit time, so an rm through docker
    exec did nothing and a hand-run build kept pipeline.log in its image."""
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append((cmd, kw)) or _Ok())
    monkeypatch.delenv("ALEMBIC_HOST_WORKDIR", raising=False)
    monkeypatch.delenv("ALEMBIC_JOB_ID", raising=False)

    sc.build_image("https://github.com/org/repo", _build_ns(tmp_path))

    cmds = [cmd for cmd, _ in ran]
    assert not any("exec" in cmd for cmd in cmds)
    blank = next(i for i, cmd in enumerate(cmds) if cmd[-3:-1] == ["cp", "-"])
    assert cmds[blank][-1].endswith(":/work/.alembic/repo")
    with tarfile.open(fileobj=io.BytesIO(ran[blank][1]["input"])) as tar:
        [member] = tar.getmembers()
    assert (member.name, member.size) == ("pipeline.log", 0)
    assert blank < next(i for i, cmd in enumerate(cmds) if "commit" in cmd)


def test_serve_env_prints_what_a_serve_container_would_get_with_keys_fingerprinted(
        monkeypatch, tmp_path, capsys):
    """The builds page compares these with a stopped container's settings. A key
    itself never leaves start_chain."""
    for name in sc.SERVE_ONLY_ENV:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("S3__ENDPOINT_URL=http://localhost:19000\nS3__BUCKET_NAME=agent-vault\n"
                   "S3__SECRET_KEY=topsecret\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["start_chain", "https://github.com/org/repo",
                                      "--serve-env", "--env-file", str(env)])
    monkeypatch.setattr(sc, "detect_gpu", lambda *a, **kw: pytest.fail("asked docker"))

    sc.main()

    out = capsys.readouterr().out
    settings = json.loads(out)
    assert settings["S3__ENDPOINT_URL"] == "http://host.docker.internal:19000"
    assert settings["S3__EXTERNAL_ENDPOINT_URL"] == "http://localhost:19000"
    assert settings["S3__BUCKET_NAME"] == "agent-vault"
    assert settings["S3__SECRET_KEY"] == "sha256:" + hashlib.sha256(b"topsecret").hexdigest()[:16]
    assert "topsecret" not in out


def test_serve_image_publishes_the_port_it_is_given(monkeypatch, tmp_path):
    """A container replaced for new S3 settings keeps its predecessor's port, so
    the server keeps its address."""
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd) or _Ok())
    monkeypatch.setattr(sc.subprocess, "run", lambda cmd, **kw: _Inspect("true\n"))
    monkeypatch.setattr(sc, "SERVE_SETTLE_SECONDS", 0.01)
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)

    sc.serve_image("https://github.com/org/repo", "alembic-tool:repo", _build_ns(tmp_path, port=27969))

    assert "27969:8000" in ran[0]
