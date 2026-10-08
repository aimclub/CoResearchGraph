#!/usr/bin/env python3
"""Check the infrastructure before starting a long run.

A run that builds and serves tools across machines has a lot of ways to fail an
hour in for a reason that was knowable at the start: an unset key, a host that
is not reachable, a Docker daemon that rejects the client's API version, a tool
server that is not answering. This prints a green/red checklist of exactly
those, and exits non-zero if anything is red.

It checks:

* **environment** — the named variables are set (never their values);
* **hosts** — a TCP connect to each ``host:port``;
* **docker contexts** — the daemon answers, with the API version an older one
  needs;
* **MCP servers** — the tools you expect to use answer at their addresses.

Every probe is injectable, so the aggregation is testable without a network,
Docker, or a VPN.

Usage::

    python scripts/preflight.py \\
        --env OPENROUTER_API_KEY \\
        --host build-host:22 \\
        --context gpu:1.43 \\
        --mcp http://build-host:20001/mcp
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field

_TCP_TIMEOUT = 5
_HTTP_TIMEOUT = 8
_DOCKER_TIMEOUT = 20


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        mark = "✓" if self.ok else "✗"  # ✓ / ✗
        tail = f" — {self.detail}" if self.detail else ""
        return f"  [{mark}] {self.name}{tail}"


# ── probes (impure, injectable) ───────────────────────────────────────────────


def tcp_probe(host: str, port: int, *, timeout: int = _TCP_TIMEOUT) -> tuple[bool, str]:
    """True if a TCP connection to ``host:port`` opens within ``timeout``."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"{host}:{port} reachable"
    except OSError as exc:
        return False, f"{host}:{port} unreachable ({exc.__class__.__name__})"


def docker_probe(
    context: str, api_version: str | None, *, runner=subprocess.run
) -> tuple[bool, str]:
    """True if the docker ``context`` daemon answers ``version`` (API pin applied)."""
    env = dict(os.environ)
    if api_version:
        env["DOCKER_API_VERSION"] = api_version
    try:
        r = runner(
            [
                "docker",
                "--context",
                context,
                "version",
                "--format",
                "{{.Server.Version}}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=_DOCKER_TIMEOUT,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"context {context}: {exc.__class__.__name__}"
    if getattr(r, "returncode", 1) != 0:
        return False, f"context {context}: {(r.stderr or '').strip()[:120]}"
    pin = f" (API {api_version})" if api_version else ""
    return True, f"context {context}: daemon {(r.stdout or '').strip()}{pin}"


def http_probe(url: str, *, timeout: int = _HTTP_TIMEOUT) -> tuple[bool, str]:
    """True if ``url`` answers (any HTTP status — an MCP endpoint that responds is
    up; a refused connection is not)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return True, f"{url} → HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return True, f"{url} → HTTP {exc.code}"  # responded = server is up
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, f"{url} unreachable ({exc.__class__.__name__})"


# ── spec + aggregation (pure) ─────────────────────────────────────────────────


@dataclass
class PreflightSpec:
    required_env: list[str] = field(default_factory=list)
    hosts: list[tuple[str, int]] = field(default_factory=list)
    contexts: list[tuple[str, str | None]] = field(default_factory=list)
    mcp_urls: list[str] = field(default_factory=list)


@dataclass
class PreflightReport:
    results: list[CheckResult]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def render(self) -> str:
        n_ok = sum(1 for r in self.results if r.ok)
        head = f"[preflight] {n_ok}/{len(self.results)} checks passed — " + (
            "READY" if self.ok else "NOT READY"
        )
        return "\n".join([head, *(r.line() for r in self.results)])


def check_env(
    required: list[str], env: dict[str, str] | None = None
) -> list[CheckResult]:
    e = env if env is not None else os.environ
    out = []
    for var in required:
        present = bool((e.get(var) or "").strip())
        out.append(CheckResult(f"env {var}", present, "set" if present else "MISSING"))
    return out


def run_preflight(
    spec: PreflightSpec,
    *,
    env: dict[str, str] | None = None,
    tcp=tcp_probe,
    docker=docker_probe,
    http=http_probe,
) -> PreflightReport:
    """Run every configured check with the given (injectable) probes → report."""
    results: list[CheckResult] = list(check_env(spec.required_env, env))
    for host, port in spec.hosts:
        ok, detail = tcp(host, port)
        results.append(CheckResult(f"reach {host}:{port}", ok, detail))
    for context, api in spec.contexts:
        ok, detail = docker(context, api)
        results.append(CheckResult(f"docker {context}", ok, detail))
    for url in spec.mcp_urls:
        ok, detail = http(url)
        results.append(CheckResult(f"mcp {url}", ok, detail))
    return PreflightReport(results)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _host_port(s: str) -> tuple[str, int]:
    host, _, port = s.rpartition(":")
    if not host:
        raise argparse.ArgumentTypeError(f"expected host:port, got {s!r}")
    return host, int(port)


def _context_api(s: str) -> tuple[str, str | None]:
    ctx, sep, api = s.partition(":")
    return ctx, (api if sep and api else None)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-run checklist for a remote run ."
    )
    p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="VAR",
        help="required env var (repeatable)",
    )
    p.add_argument(
        "--host",
        action="append",
        default=[],
        metavar="HOST:PORT",
        type=_host_port,
        help="host:port that must be reachable (repeatable)",
    )
    p.add_argument(
        "--context",
        action="append",
        default=[],
        metavar="CTX[:APIVER]",
        type=_context_api,
        help="docker context (+ optional API pin), repeatable",
    )
    p.add_argument(
        "--mcp",
        action="append",
        default=[],
        metavar="URL",
        help="MCP server URL that must respond (repeatable)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = PreflightSpec(
        required_env=args.env,
        hosts=args.host,
        contexts=args.context,
        mcp_urls=args.mcp,
    )
    report = run_preflight(spec)
    print(report.render())
    return report.exit_code


__all__ = [
    "CheckResult",
    "PreflightReport",
    "PreflightSpec",
    "check_env",
    "docker_probe",
    "http_probe",
    "run_preflight",
    "tcp_probe",
]


if __name__ == "__main__":
    sys.exit(main())
