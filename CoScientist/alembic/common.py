from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

BASE_IMAGE = "alembic-base:latest"

# A digest of the sources baked into the base image, stamped on it as a label.
# Without it a base image built weeks ago is reused as if it were current, and
# the pipeline silently runs old code — the failure mode is invisible, because
# everything works, just not with the code in the checkout. The paths below must
# track what the Dockerfile COPYs in.
_LABEL_KEY = "alembic.source_sha"
_BAKED_PATHS = (
    "CoScientist/alembic",
    "docker/alembic/requirements.txt",
    "docker/alembic/entrypoint.py",
    "docker/alembic/serve.py",
)


def get_repo_name(repo_url: str) -> str:
    """Last path segment of a repo URL, without a trailing ``.git``."""
    return re.sub(r"\.git$", "", repo_url.rstrip("/").split("/")[-1])


def _docker(context: str | None) -> list[str]:
    """``docker [--context X]``, imported late.

    ``alembic.targets`` resolves only where the package is on sys.path (inside
    the build container, or through start_chain's path setup). This module is
    also imported plainly as ``CoScientist.alembic.common``, so the import
    happens where docker is actually being run rather than at module load.
    """
    from alembic.targets import docker_cli

    return docker_cli(context)


def _image_exists(name: str, context: str | None = None) -> bool:
    """True if a docker image with this name/tag exists on that daemon."""
    return subprocess.run(
        [*_docker(context), "image", "inspect", name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def _iter_baked_files(project_root: Path) -> Iterator[Path]:
    """Every file baked into the base image, in a deterministic order.

    ``__pycache__``/``.pyc`` are skipped so a stray bytecode file cannot
    perturb the digest.
    """
    for rel in _BAKED_PATHS:
        base = project_root / rel
        if base.is_file():
            yield base
        elif base.is_dir():
            for path in sorted(base.rglob("*")):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix != ".pyc"
                ):
                    yield path


def _source_hash(dockerfile: Path, project_root: Path) -> str:
    """Digest of the Dockerfile plus everything it copies into the base image.

    The same checkout hashes identically on any machine, so the value doubles as
    a build identifier.
    """
    digest = hashlib.sha256()
    digest.update(b"Dockerfile\0")
    try:
        digest.update(dockerfile.read_bytes())
    except OSError:
        pass
    for path in _iter_baked_files(project_root):
        digest.update(str(path.relative_to(project_root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _image_label(name: str, key: str, context: str | None = None) -> str | None:
    """Value of image label ``key``, or ``None`` if the image or label is absent."""
    result = subprocess.run(
        [*_docker(context), "image", "inspect", "--format",
         f'{{{{ index .Config.Labels "{key}" }}}}', name],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    # docker renders a missing label as "<no value>".
    return value if value and value != "<no value>" else None


def ensure_base_image(dockerfile: Path, project_root: Path,
                platform: str | None = None, rebuild: bool = False,
                context: str | None = None) -> None:
    """Make sure the daemon the build will run on has a current base image.

    Out of date means the sources baked into the present image differ from the
    checkout — detected through the ``alembic.source_sha`` label. Rebuilding on
    that difference is what stops a cached image from quietly running old
    pipeline code.

    ``context`` names the daemon the build is headed for, and the check has to
    ask that one. Asking the local daemon about a build that will run elsewhere
    answers the wrong question, and the answer is reassuring, so old pipeline
    code runs on the remote machine with nothing said.

    A remote daemon is checked but never built on: the base image installs
    system packages, which the build hosts cannot reach the network for. A
    remote image that is missing or stale is therefore an error with the command
    to fix it, not something to paper over.
    """
    want = _source_hash(dockerfile, project_root)
    if context:
        have = _image_label(BASE_IMAGE, _LABEL_KEY, context)
        if have == want and not rebuild:
            print(f"[alembic] base image on {context} is up to date ({want}) — reusing.")
            return
        state = (f"built from {have}" if have else
                 "unlabelled" if _image_exists(BASE_IMAGE, context) else "missing")
        sys.exit(
            f"[alembic] base image {BASE_IMAGE} on context {context!r} is {state}, "
            f"the checkout is {want}.\n"
            f"[alembic] It cannot be built there (the build installs system "
            f"packages and those hosts have no network during a build). Build it "
            f"here and ship it:\n"
            f"[alembic]   python CoScientist/alembic/start_chain.py --help  # local build first\n"
            f"[alembic]   docker save {BASE_IMAGE} | docker --context {context} load"
        )
    if not rebuild and _image_exists(BASE_IMAGE):
        have = _image_label(BASE_IMAGE, _LABEL_KEY)
        if have == want:
            print(f"[alembic] base image {BASE_IMAGE} up to date ({want}) — reusing.")
            return
        print(
            f"[alembic] base image {BASE_IMAGE} is stale "
            f"(built from {have or 'unlabelled sources'}, checkout is {want}) "
            "— rebuilding.",
            flush=True,
        )
    cmd = ["docker", "build"]
    if platform:
        cmd += ["--platform", platform]
    cmd += ["--label", f"{_LABEL_KEY}={want}",
            "-t", BASE_IMAGE, "-f", str(dockerfile), str(project_root)]
    print(f"[alembic] building base image: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
