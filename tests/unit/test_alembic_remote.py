"""A tool built on another machine has to be reachable from this one.

Two things break silently across machines: the served address (localhost means
nothing to anyone else) and the data directory (a bind mount resolves on the
daemon's filesystem, so a local path simply does not exist on a remote daemon).
No Docker here — the one lookup takes an injectable runner.
"""

from CoScientist.alembic import remote


class _Result:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


# ── the advertised address ───────────────────────────────────────────────────


def test_a_remote_endpoint_gives_the_reachable_host():
    assert remote.host_from_endpoint("ssh://user@gpu-box:22") == "gpu-box"
    assert remote.host_from_endpoint("tcp://10.0.0.5:2376") == "10.0.0.5"


def test_a_local_socket_has_no_reachable_host():
    assert remote.host_from_endpoint("unix:///var/run/docker.sock") is None
    assert remote.host_from_endpoint(None) is None


def test_the_remote_host_is_advertised_over_a_loopback_default():
    assert remote.resolve_advertise_host(context_host="gpu-box") == "gpu-box"


def test_an_explicit_host_wins():
    assert (
        remote.resolve_advertise_host(explicit="public.example", context_host="gpu-box")
        == "public.example"
    )


def test_a_loopback_hint_is_not_advertised():
    """Advertising localhost to another machine is the bug this exists for."""
    assert remote.resolve_advertise_host(a2a_host="127.0.0.1") == "localhost"


def test_the_url_is_composed_for_the_caller():
    assert remote.advertised_url("gpu-box", 20001) == "http://gpu-box:20001/mcp"


def test_the_endpoint_lookup_survives_a_missing_context():
    assert remote.context_endpoint("nope", runner=lambda *a, **k: _Result(returncode=1)) is None
    assert remote.context_endpoint(None) is None


# ── the data directory ───────────────────────────────────────────────────────


def test_the_local_daemon_needs_no_staging():
    assert not remote.needs_remote_staging(None)


def test_a_remote_daemon_needs_staging():
    assert remote.needs_remote_staging("gpu", "ssh://user@gpu-box:22")


def test_an_unknown_endpoint_is_assumed_remote():
    """Guessing local would mount a path that is not there."""
    assert remote.needs_remote_staging("gpu", None)


def test_staging_copies_the_directory_contents_into_a_volume():
    commands = remote.build_stage_commands("gpu", "alembic-tool:x", "vol", "/data/ct/")

    joined = [" ".join(c) for c in commands]
    assert joined[0] == "docker --context gpu volume create vol"
    assert any("cp /data/ct/. " in c for c in joined)  # contents, not the dir itself
    assert joined[-1].startswith("docker --context gpu rm -f")


def test_a_remote_run_mounts_the_staged_volume():
    args = remote.serve_mount_args(context="gpu", mount_dir="/data/ct", volume="vol")

    assert args == ["-v", f"vol:{remote.MOUNT_TARGET}:ro"]


def test_a_local_run_binds_the_host_path():
    args = remote.serve_mount_args(context=None, mount_dir="/data/ct", volume=None)

    assert args == ["-v", f"/data/ct:{remote.MOUNT_TARGET}:ro"]


def test_no_data_directory_means_no_mount():
    assert remote.serve_mount_args(context="gpu", mount_dir=None, volume="vol") == []
