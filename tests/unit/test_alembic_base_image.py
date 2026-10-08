"""The alembic base image must not silently be out of date.

Reusing a base image just because it exists means the pipeline can run code
from weeks ago while everything looks fine. The image is stamped with a digest
of the sources baked into it, and reused only when that digest matches the
checkout. No Docker here: the three shell-outs are replaced.
"""

from pathlib import Path

import pytest

from CoScientist.alembic import common


@pytest.fixture
def project(tmp_path):
    """A checkout with the files the base image bakes in."""
    (tmp_path / "CoScientist" / "alembic").mkdir(parents=True)
    (tmp_path / "CoScientist" / "alembic" / "main.py").write_text("v1", encoding="utf-8")
    (tmp_path / "docker" / "alembic").mkdir(parents=True)
    dockerfile = tmp_path / "docker" / "alembic" / "Dockerfile"
    dockerfile.write_text("FROM python:3.11\n", encoding="utf-8")
    return tmp_path, dockerfile


def _patch(monkeypatch, *, exists, label):
    """Record what would have been run; answer the two lookups."""
    builds = []
    monkeypatch.setattr(common, "_image_exists", lambda name, context=None: exists)
    monkeypatch.setattr(
        common, "_image_label", lambda name, key, context=None: label
    )
    monkeypatch.setattr(
        common.subprocess, "run", lambda cmd, **kw: builds.append(cmd) or _Ok()
    )
    return builds


class _Ok:
    returncode = 0


def test_a_matching_image_is_reused(project, monkeypatch):
    root, dockerfile = project
    current = common._source_hash(dockerfile, root)
    builds = _patch(monkeypatch, exists=True, label=current)

    common.ensure_base_image(dockerfile, root)

    assert builds == []


def test_an_image_built_from_older_sources_is_rebuilt(project, monkeypatch):
    root, dockerfile = project
    builds = _patch(monkeypatch, exists=True, label="0123456789abcdef")

    common.ensure_base_image(dockerfile, root)

    assert len(builds) == 1


def test_an_unlabelled_image_is_rebuilt(project, monkeypatch):
    """Images from before the label existed cannot be trusted to be current."""
    root, dockerfile = project
    builds = _patch(monkeypatch, exists=True, label=None)

    common.ensure_base_image(dockerfile, root)

    assert len(builds) == 1


def test_a_missing_image_is_built_and_stamped(project, monkeypatch):
    root, dockerfile = project
    builds = _patch(monkeypatch, exists=False, label=None)

    common.ensure_base_image(dockerfile, root)

    cmd = builds[0]
    assert "--label" in cmd
    assert cmd[cmd.index("--label") + 1] == (
        f"{common._LABEL_KEY}={common._source_hash(dockerfile, root)}"
    )


def test_editing_a_baked_file_changes_the_digest(project):
    root, dockerfile = project
    before = common._source_hash(dockerfile, root)

    (root / "CoScientist" / "alembic" / "main.py").write_text("v2", encoding="utf-8")

    assert common._source_hash(dockerfile, root) != before


def test_editing_the_dockerfile_changes_the_digest(project):
    root, dockerfile = project
    before = common._source_hash(dockerfile, root)

    dockerfile.write_text("FROM python:3.12\n", encoding="utf-8")

    assert common._source_hash(dockerfile, root) != before


def test_stray_bytecode_does_not_change_the_digest(project):
    root, dockerfile = project
    before = common._source_hash(dockerfile, root)

    cache = root / "CoScientist" / "alembic" / "__pycache__"
    cache.mkdir()
    (cache / "main.cpython-311.pyc").write_bytes(b"\x00\x01")

    assert common._source_hash(dockerfile, root) == before


def test_the_digest_is_the_same_on_any_machine(project):
    """It is derived from file contents and repo-relative paths, nothing local."""
    root, dockerfile = project
    first = common._source_hash(dockerfile, root)

    assert first == common._source_hash(Path(dockerfile), Path(root))


# ── a build headed for another daemon ────────────────────────────────────────


def test_the_check_asks_the_daemon_the_build_will_run_on(project, monkeypatch):
    """Asking the local daemon about a remote build answers the wrong question,
    and answers it reassuringly."""
    root, dockerfile = project
    asked = []
    monkeypatch.setattr(
        common, "_image_label",
        lambda name, key, context=None: asked.append(context) or common._source_hash(
            dockerfile, root
        ),
    )

    common.ensure_base_image(dockerfile, root, context="nss-calc")

    assert asked == ["nss-calc"]


def test_a_current_image_on_the_remote_is_reused(project, monkeypatch):
    root, dockerfile = project
    current = common._source_hash(dockerfile, root)
    builds = _patch(monkeypatch, exists=True, label=current)

    common.ensure_base_image(dockerfile, root, context="nss-calc")

    assert builds == []


def test_a_stale_image_on_the_remote_stops_the_run(project, monkeypatch):
    """It cannot be built there, so carrying on would run old pipeline code on
    the remote machine with nothing said."""
    root, dockerfile = project
    builds = _patch(monkeypatch, exists=True, label="0123456789abcdef")

    with pytest.raises(SystemExit) as exit_info:
        common.ensure_base_image(dockerfile, root, context="nss-calc")

    assert builds == []
    message = str(exit_info.value)
    assert "nss-calc" in message
    assert "docker save" in message  # says how to fix it


def test_a_missing_image_on_the_remote_stops_the_run(project, monkeypatch):
    root, dockerfile = project
    _patch(monkeypatch, exists=False, label=None)

    with pytest.raises(SystemExit) as exit_info:
        common.ensure_base_image(dockerfile, root, context="nss-calc")

    assert "missing" in str(exit_info.value)
