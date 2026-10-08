"""Reaching the Docker daemon a build should run on.

No Docker and no GPU here: the one probe takes an injectable runner.
"""

import subprocess

from CoScientist.alembic.targets import detect_gpu, docker_cli, docker_env


class _Result:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


def _runner(answers, seen=None):
    def run(cmd, **kw):
        if seen is not None:
            seen.append(cmd)
        return answers.get(cmd[0], _Result(returncode=1))

    return run


def test_a_gpu_is_detected_from_dockers_own_runtimes():
    """That is what actually gates `docker run --gpus`."""
    assert detect_gpu(runner=_runner({"docker": _Result("[nvidia runc]")}))


def test_nvidia_smi_is_the_fallback():
    assert detect_gpu(runner=_runner({"nvidia-smi": _Result("GPU 0: NVIDIA A100")}))


def test_no_gpu_when_neither_probe_says_so():
    assert not detect_gpu(runner=_runner({"docker": _Result("[runc]")}))


def test_a_broken_probe_reads_as_no_gpu():
    """Better to omit --gpus than to pass it where it will fail."""

    def boom(cmd, **kw):
        raise subprocess.SubprocessError("docker missing")

    assert not detect_gpu(runner=boom)


def test_the_local_daemon_takes_a_bare_docker():
    assert docker_cli() == ["docker"]
    assert docker_cli(None) == ["docker"]


def test_a_remote_daemon_is_addressed_by_context():
    assert docker_cli("gpu") == ["docker", "--context", "gpu"]


def test_an_old_daemon_gets_its_api_version_pinned():
    assert docker_env({}, api_version="1.43")["DOCKER_API_VERSION"] == "1.43"


def test_without_a_pin_the_environment_is_untouched():
    """A pin one daemon needs makes a newer one look unreachable."""
    assert docker_env({"PATH": "/bin"}) == {"PATH": "/bin"}


def test_the_gpu_probe_asks_the_daemon_the_build_will_run_on():
    """A GPU here says nothing about the machine the build is headed for."""
    seen = []
    detect_gpu("nss-calc", runner=_runner({"docker": _Result("[runc]")}, seen))

    assert seen[0][:3] == ["docker", "--context", "nss-calc"]


def test_nvidia_smi_is_not_consulted_about_another_machine():
    """It reports this host's driver, which is not the question being asked."""
    seen = []
    found = detect_gpu(
        "nss-calc", runner=_runner({"nvidia-smi": _Result("GPU 0: NVIDIA A100")}, seen)
    )

    assert not found
    assert all(cmd[0] != "nvidia-smi" for cmd in seen)
