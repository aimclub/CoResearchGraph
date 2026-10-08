"""Delivering a built tool to a machine that cannot reach it over the network.

When the target cannot call a served MCP endpoint directly, the tool is shipped
to it and served locally there. Two ways, both ending at a tool the target can
call:

1. **Through a registry** — tag the committed ``alembic-tool:<name>`` image,
   push it, then pull and serve it on the target. The simple path when the
   target can reach a registry.

2. **As a bundle of portable artefacts** — pack the source and generated code
   without the venvs (see :mod:`alembic.portable`), ship that, and rebuild the
   image on the target. The fallback when there is no registry access, and the
   thing small enough to keep around for standing a tool up later.

Credentials come from the environment and are never hardcoded; a missing
credential returns ``None`` so the caller can fall back to the bundle path
rather than fail. The docker argv sequences and the packing are injectable, so
all of it is testable without Docker, a registry, or real credentials.

.. note::

   There is no command-line entry point for this yet. The functions here are
   the mechanism and are covered by tests, but nothing in the tree calls them,
   so shipping a tool to another machine currently means driving them from
   Python by hand. A CLI is the missing piece.

   For the same reason this path has never been run for real: the tests cover
   it with an injected runner, and no push to a registry, no rebuild on a
   target, no serve on the far side has been done end to end. Both are open:
   write the entry point, then verify it against a live registry and a second
   machine.
"""

from __future__ import annotations

import os
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

from alembic.portable import (
    ARTEFACT_PATHS,
    RebuildResult,
    extract_artifacts,
    rebuild_tool,
)

# ── Docker Hub credentials (from env/secrets) ─────────────────────────────────


@dataclass(frozen=True)
class HubCreds:
    """Docker Hub credentials + namespace, read from env/secrets (never hardcoded)."""

    username: str
    token: str
    namespace: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> HubCreds | None:
        """Load creds from ``DOCKERHUB_USERNAME`` + ``DOCKERHUB_TOKEN`` (+ optional
        ``DOCKERHUB_NAMESPACE``, default = username). ``None`` if not configured,
        so a caller can fall back to the artefact path cleanly."""
        e = env if env is not None else os.environ
        user = (e.get("DOCKERHUB_USERNAME") or "").strip()
        token = (e.get("DOCKERHUB_TOKEN") or e.get("DOCKERHUB_PASSWORD") or "").strip()
        if not user or not token:
            return None
        ns = (e.get("DOCKERHUB_NAMESPACE") or user).strip()
        return cls(username=user, token=token, namespace=ns)


def hub_image_ref(namespace: str, name: str, tag: str = "latest") -> str:
    """The Docker Hub repository reference for a converted tool.

    A registry **repository name must be lowercase** (``docker push`` rejects
    anything else with *"repository name must be lowercase"*), but tool names come
    from the upstream repo basename and are frequently capitalised — ``MedSAM``,
    ``PathFinderCRC``, ``TabPFN``. The local ``alembic-tool:<name>`` tag keeps the
    original casing (an image *tag* may be mixed-case); only the hub repository
    path is lowercased here.
    """
    return f"{namespace.lower()}/alembic-tool-{name.lower()}:{tag}"


# ── Docker Hub strategy: argv builders (pure) ─────────────────────────────────


def login_command(creds: HubCreds) -> list[str]:
    """`docker login` reading the token from stdin (never on the argv)."""
    return ["docker", "login", "--username", creds.username, "--password-stdin"]


def push_commands(local_image: str, remote_ref: str) -> list[list[str]]:
    """Tag the local tool image to the hub ref and push it."""
    return [
        ["docker", "tag", local_image, remote_ref],
        ["docker", "push", remote_ref],
    ]


def pull_and_serve_commands(
    remote_ref: str, *, context: str | None = None, port: int = 8000
) -> list[list[str]]:
    """On the target: pull the image from the hub and serve it. ``context`` runs
    it on a remote Docker daemon; the served URL is advertised by start_chain."""
    base = ["docker", "--context", context] if context else ["docker"]
    cname = "alembic-hub-serve"
    return [
        [*base, "pull", remote_ref],
        [*base, "run", "-d", "--name", cname, "-p", f"{port}:8000", remote_ref],
    ]


# ── artefact-bundle strategy ──────────────────────────────────────────────────


def bundle_name(name: str) -> str:
    return f"alembic-artifacts-{name}.tar.gz"


def pack_artifacts(
    name: str,
    out_dir: Path,
    *,
    extractor=None,
    workdir: Path | None = None,
) -> Path:
    """Extract a tool's low-weight artefacts (no venvs) and pack them into
    a ``.tar.gz`` bundle under ``out_dir`` — the minimal, portable, in-repo-able
    form you can rebuild the MCP from anywhere. Returns the bundle path."""
    import tempfile

    extractor = extractor or extract_artifacts
    scratch = workdir or Path(tempfile.mkdtemp(prefix="hub_pack_"))
    ctx = extractor(name, scratch, runner=subprocess.run)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / bundle_name(name)
    # Archive the venv-free context relative to itself so it unpacks as a clean
    # build context (.alembic/<name>/... + docker/alembic serve files).
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(ctx, arcname=".")
    return bundle


def unpack_bundle(bundle: Path, dest_dir: Path) -> Path:
    """Unpack a bundle into ``dest_dir`` and return the build-context root."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        _safe_extractall(tar, dest_dir)
    return dest_dir


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract, refusing any member that would escape ``dest`` (path traversal)."""
    base = dest.resolve()
    for member in tar.getmembers():
        target = (base / member.name).resolve()
        if not str(target).startswith(str(base)):
            raise ValueError(f"unsafe path in bundle: {member.name}")
    # members validated above; "data" filter also strips unsafe metadata (py3.12+).
    tar.extractall(dest, filter="data")


def rebuild_from_bundle(
    bundle: Path,
    name: str,
    dest_dir: Path,
    *,
    builder=rebuild_tool,
    tag_prefix: str = "alembic-hub",
) -> RebuildResult:
    """Unpack a bundle on the target and rebuild the tool image from it via
    ``serve.Dockerfile``. Returns the :class:`RebuildResult`."""
    ctx = unpack_bundle(bundle, dest_dir)
    return builder(name, ctx, runner=subprocess.run, tag_prefix=tag_prefix)


# ── orchestration ─────────────────────────────────────────────────────────────


@dataclass
class HubShipResult:
    name: str
    strategy: str  # "dockerhub" | "artifact"
    ok: bool
    ref: str = ""  # hub image ref or bundle path
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "ok": self.ok,
            "ref": self.ref,
            "detail": self.detail,
        }


def choose_strategy(creds: HubCreds | None, *, prefer: str = "auto") -> str:
    """Pick the delivery strategy: explicit ``dockerhub``/``artifact`` wins; ``auto``
    uses Docker Hub when creds exist, else the artefact fallback."""
    if prefer in ("dockerhub", "artifact"):
        return prefer
    return "dockerhub" if creds is not None else "artifact"


def push_to_hub(
    name: str,
    *,
    creds: HubCreds,
    local_image: str | None = None,
    tag: str = "latest",
    runner=subprocess.run,
) -> HubShipResult:
    """Tag + login + push a tool image to Docker Hub. Login token is fed via stdin;
    a non-zero step aborts with the failing step recorded."""
    local = local_image or f"alembic-tool:{name}"
    ref = hub_image_ref(creds.namespace, name, tag)
    login = runner(
        login_command(creds),
        input=creds.token,
        capture_output=True,
        text=True,
        check=False,
    )
    if getattr(login, "returncode", 1) != 0:
        return HubShipResult(
            name, "dockerhub", ok=False, ref=ref, detail="login failed"
        )
    for cmd in push_commands(local, ref):
        r = runner(cmd, capture_output=True, text=True, check=False)
        if getattr(r, "returncode", 1) != 0:
            return HubShipResult(
                name, "dockerhub", ok=False, ref=ref, detail=f"{cmd[1]} failed"
            )
    return HubShipResult(name, "dockerhub", ok=True, ref=ref, detail="pushed")


def ship_via_hub(
    name: str,
    *,
    prefer: str = "auto",
    out_dir: Path | None = None,
    creds: HubCreds | None = None,
    runner=subprocess.run,
) -> HubShipResult:
    """Ship one tool via the chosen strategy. ``dockerhub`` pushes the image (needs
    creds); ``artifact`` packs the low-weight bundle (needs neither creds nor a
    network) — the documented fallback for registry-less targets."""
    creds = creds if creds is not None else HubCreds.from_env()
    strategy = choose_strategy(creds, prefer=prefer)
    if strategy == "dockerhub":
        if creds is None:
            return HubShipResult(
                name, "dockerhub", ok=False, detail="no DOCKERHUB_* credentials in env"
            )
        return push_to_hub(name, creds=creds, runner=runner)
    out = out_dir or Path("hub_artifacts")
    bundle = pack_artifacts(name, out)
    return HubShipResult(
        name, "artifact", ok=bundle.exists(), ref=str(bundle), detail="packed"
    )


__all__ = [
    "ARTEFACT_PATHS",
    "HubCreds",
    "HubShipResult",
    "bundle_name",
    "choose_strategy",
    "hub_image_ref",
    "login_command",
    "pack_artifacts",
    "pull_and_serve_commands",
    "push_commands",
    "push_to_hub",
    "rebuild_from_bundle",
    "ship_via_hub",
    "unpack_bundle",
]
