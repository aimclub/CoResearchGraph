"""Addressing the Docker daemon a build runs on.

A build does not have to run on the machine that started it. It can go to a
remote Docker context: the box with the GPU, or the one with the cores. Each
such daemon needs slightly different handling, and this module holds the three
pieces of that: which ``docker`` argv prefix reaches it, whether ``--gpus`` will
work here, and whether its API version has to be pinned.

The API-version pin belongs to a single call rather than the whole process. A
daemon old enough to reject the client's default needs one, and a newer daemon
rejects the pin, so a process-wide ``DOCKER_API_VERSION`` makes one of the two
unreachable.

The one probe, :func:`detect_gpu`, takes an injectable ``runner``, so all of
this is testable without Docker or a GPU.
"""

from __future__ import annotations

import os
import subprocess


def detect_gpu(
    context: str | None = None, *, env: dict[str, str] | None = None,
    runner=subprocess.run,
) -> bool:
    """True if the daemon a build will run on exposes an NVIDIA GPU to Docker.

    Probes ``docker info`` for the nvidia runtime, which is what actually gates
    ``docker run --gpus``. Any probe failure (no Docker, no driver, timeout)
    reads as no GPU, so ``--gpus`` is forwarded only where it will work.

    The probe must ask the *target* daemon. Asking the local one and then
    forwarding the answer to a remote build gets it wrong in both directions: a
    GPU here and none there makes every remote run fail on ``--gpus``, and a GPU
    there and none here leaves it unused. ``nvidia-smi`` is a local fallback
    only, since it says nothing about another machine.
    """
    probes = [([*docker_cli(context), "info", "--format", "{{.Runtimes}}"], "nvidia")]
    if not context:
        probes.append((["nvidia-smi", "-L"], "gpu"))
    for cmd, needle in probes:
        try:
            result = runner(cmd, capture_output=True, text=True, check=False,
                            timeout=20, env=env)
        except (OSError, subprocess.SubprocessError):
            continue
        if getattr(result, "returncode", 1) == 0 and needle in (result.stdout or "").lower():
            return True
    return False


def docker_cli(context: str | None = None) -> list[str]:
    """The ``docker`` argv prefix for a context: ``docker [--context X]``.

    ``None`` or empty means the local daemon and a bare ``docker``.
    """
    return ["docker", "--context", context] if context else ["docker"]


def docker_env(
    base_env: dict[str, str] | None = None, *, api_version: str | None = None
) -> dict[str, str]:
    """The environment for one docker call, pinning ``DOCKER_API_VERSION`` only
    when this daemon needs it. With nothing to pin the environment is returned
    unchanged, since a pin the daemon does not want is what makes it look
    unreachable.
    """
    env = dict(base_env if base_env is not None else os.environ)
    if api_version:
        env["DOCKER_API_VERSION"] = api_version
    return env
