"""Getting a built tool onto a machine that cannot call it over the network.

Two routes: through an image registry, or as a bundle of portable artefacts
rebuilt on the target. No Docker, no registry, no real credentials — the docker
steps are argv sequences and the packing works on a temp directory.

The modules import under the container's package layout, so they are loaded
with CoScientist/ on sys.path, the same way the build container sees them.
"""

import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "CoScientist"))

from alembic.hub import (  # noqa: E402
    HubCreds,
    bundle_name,
    choose_strategy,
    hub_image_ref,
    pack_artifacts,
    push_commands,
    unpack_bundle,
)


# ── credentials ──────────────────────────────────────────────────────────────


def test_credentials_come_from_the_environment():
    creds = HubCreds.from_env({"DOCKERHUB_USERNAME": "me", "DOCKERHUB_TOKEN": "t"})

    assert creds.username == "me"
    assert creds.namespace == "me"  # defaults to the user


def test_missing_credentials_are_not_an_error():
    """The caller falls back to the bundle route instead of failing."""
    assert HubCreds.from_env({}) is None


def test_a_repository_reference_is_lowercased():
    """Registries reject anything else, and tool names follow the repo name —
    MedSAM, TabPFN — so this rejection is the common case, not the rare one."""
    assert hub_image_ref("Me", "MedSAM") == "me/alembic-tool-medsam:latest"


def test_pushing_tags_the_local_image_first():
    ref = "me/alembic-tool-medsam:latest"
    commands = push_commands("alembic-tool:medsam", ref)

    assert commands[0] == ["docker", "tag", "alembic-tool:medsam", ref]
    assert commands[-1] == ["docker", "push", ref]


# ── strategy ─────────────────────────────────────────────────────────────────


def test_a_registry_is_used_when_credentials_exist():
    creds = HubCreds(username="me", token="t", namespace="me")

    assert choose_strategy(creds) == "dockerhub"


def test_without_credentials_the_bundle_route_is_chosen():
    assert choose_strategy(None) == "artifact"


# ── the bundle ───────────────────────────────────────────────────────────────


def test_a_bundle_round_trips(tmp_path):
    context = tmp_path / "ctx" / "medsam"
    (context / ".alembic" / "medsam" / "output").mkdir(parents=True)
    (context / ".alembic" / "medsam" / "output" / "server.py").write_text("x", encoding="utf-8")

    bundle = pack_artifacts(
        "medsam", tmp_path / "out", extractor=lambda name, dest, **kw: context
    )
    restored = unpack_bundle(bundle, tmp_path / "back")

    assert bundle.name == bundle_name("medsam")
    assert (restored / ".alembic" / "medsam" / "output" / "server.py").exists()


def test_a_bundle_cannot_write_outside_its_destination(tmp_path):
    """A bundle may arrive from elsewhere; a path traversal in it must not land."""
    bundle = tmp_path / "evil.tar.gz"
    victim = tmp_path / "outside.txt"
    victim.write_text("original", encoding="utf-8")
    payload = tmp_path / "payload.txt"
    payload.write_text("replaced", encoding="utf-8")
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(payload, arcname="../outside.txt")

    try:
        unpack_bundle(bundle, tmp_path / "dest")
    except Exception:
        pass

    assert victim.read_text(encoding="utf-8") == "original"
