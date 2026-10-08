"""Reaching a tool that was built on another machine.

A tool built and served on a remote Docker context is only useful to an
executor on another host if two things hold:

1. it is advertised at a **network-reachable address** — not the hardcoded
   ``http://localhost:{port}`` that only works on the serving host; and
2. any provided ``--mount-dir`` data is **on the daemon that runs the container**
   — a Docker bind mount (``-v host:container``) resolves on the *daemon's*
   filesystem, so a local path does not exist on a remote daemon and must be
   staged there (copied into a daemon-side volume) instead.

This module holds the pure logic for both — deriving the advertised host/URL from
the chosen context's endpoint (reusing ``A2A_HOST`` as the fallback), and building
the docker command sequences that stage a local dir into a remote volume + mount
it. The one lookup that needs Docker (``context_endpoint``) takes an injectable
``runner``, so everything here is testable without a second host.
"""

from __future__ import annotations

import subprocess
from urllib.parse import urlparse

# Loopback-ish hosts that are NOT reachable from another machine.
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", ""})
MOUNT_TARGET = "/mount/data"  # matches start_chain.py / tools/paths.py


# ── advertised address ────────────────────────────────────────────────────────


def host_from_endpoint(endpoint: str | None) -> str | None:
    """The reachable hostname of a Docker context endpoint, or ``None`` if local.

    ``ssh://user@b.dgx:22`` → ``b.dgx``; ``tcp://10.0.0.5:2376`` → ``10.0.0.5``;
    a ``unix://``/``npipe://`` socket (the local daemon) → ``None``.
    """
    if not endpoint:
        return None
    parsed = urlparse(endpoint)
    if parsed.scheme in ("unix", "npipe", ""):
        return None
    return parsed.hostname or None


def context_endpoint(context: str | None, *, runner=subprocess.run) -> str | None:
    """Look up a context's Docker endpoint URL (``docker context inspect``).

    Returns ``None`` for the local daemon (no context) or if the lookup fails.
    """
    if not context:
        return None
    try:
        r = runner(
            [
                "docker",
                "context",
                "inspect",
                context,
                "--format",
                "{{.Endpoints.docker.Host}}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(r, "returncode", 1) != 0:
        return None
    return (r.stdout or "").strip() or None


def resolve_advertise_host(
    *,
    explicit: str | None = None,
    context_host: str | None = None,
    a2a_host: str | None = None,
    default: str = "localhost",
) -> str:
    """Pick the host to advertise a served MCP at, in precedence order.

    An explicit ``--advertise-host`` always wins; then the remote context's own
    host; then ``A2A_HOST`` when it names a real (non-loopback) host — reusing the
    existing A2A plumbing; otherwise ``default`` (``localhost``).
    """
    if explicit:
        return explicit
    if context_host and context_host not in _LOCAL_HOSTS:
        return context_host
    if a2a_host and a2a_host not in _LOCAL_HOSTS:
        return a2a_host
    return default


def advertised_url(
    host: str, port: int | str, *, path: str = "/mcp", scheme: str = "http"
) -> str:
    """Compose the network-reachable MCP URL an executor on another host calls."""
    return f"{scheme}://{host}:{port}{path}"


# ── remote mount staging ──────────────────────────────────────────────────────


def needs_remote_staging(context: str | None, endpoint: str | None = None) -> bool:
    """Whether ``--mount-dir`` must be staged into a daemon-side volume.

    A local bind mount is fine on the local daemon; on a remote daemon the local
    path does not exist, so the data has to be copied over. When the endpoint is
    unknown but a context is set, assume remote (stage) to stay correct.
    """
    if not context:
        return False
    if endpoint is None:
        return True
    return host_from_endpoint(endpoint) is not None


def stage_volume_name(repo: str, token: str) -> str:
    """Deterministic-per-run name for the daemon-side data volume."""
    return f"alembic-data-{repo}-{token}"


def build_stage_commands(
    context: str | None,
    helper_image: str,
    volume: str,
    local_dir: str,
) -> list[list[str]]:
    """Docker argv sequence that copies ``local_dir`` into a remote ``volume``.

    Creates the volume, makes a throwaway container off ``helper_image`` (already
    present on the target — it is the just-built tool image, so no extra pull)
    with the volume mounted, ``docker cp``s the local data into it (cp streams the
    local path to the daemon over the context connection), then removes the helper.
    """
    base = ["docker", "--context", context] if context else ["docker"]
    helper = f"alembic-stage-{volume}"
    return [
        [*base, "volume", "create", volume],
        [
            *base,
            "create",
            "--name",
            helper,
            "-v",
            f"{volume}:{MOUNT_TARGET}",
            helper_image,
            "help",
        ],
        # trailing "/." copies the *contents* of local_dir into the volume root
        [*base, "cp", f"{local_dir.rstrip('/')}/.", f"{helper}:{MOUNT_TARGET}"],
        [*base, "rm", "-f", helper],
    ]


def serve_mount_args(
    *, context: str | None, mount_dir: str | None, volume: str | None
) -> list[str]:
    """The ``-v`` args for the serve container: local bind, or staged volume.

    No mount dir → no args. Remote context → mount the staged ``volume``. Local
    daemon → bind the host path directly (read-only), as before.
    """
    if not mount_dir:
        return []
    if context and volume:
        return ["-v", f"{volume}:{MOUNT_TARGET}:ro"]
    return ["-v", f"{mount_dir}:{MOUNT_TARGET}:ro"]


__all__ = [
    "MOUNT_TARGET",
    "advertised_url",
    "build_stage_commands",
    "context_endpoint",
    "host_from_endpoint",
    "needs_remote_staging",
    "resolve_advertise_host",
    "serve_mount_args",
    "stage_volume_name",
]
