"""Launch all CoScientist A2A agent servers in a single process.

Usage (from /app):
    python -m CoScientist.a2a.run_all

The served agents, their ports, cards and env defaults all come from the
``a2a`` sections in system.yaml. Each sub-agent runs on its own port; the root
agent (orchestrator) is served in remote mode — it delegates to the sub-agent
servers over A2A. Port defaults can be overridden via environment variables
(see config.py).
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal

# Apply every agent's pre-import env defaults BEFORE the agent/tool modules
# load (e.g. the coder's shared A2A workspace id and git fail-fast knobs).
from CoScientist.assembly.schema import get_config

for _agent_cfg in get_config().a2a_agents():
    for _name, _value in _agent_cfg.a2a.env.items():
        os.environ.setdefault(_name, _value)

# Served over A2A (see a2a/serve.py): attach the non-blocking, long-running HITL
# tools so a human question becomes an `input-required` task for the caller
# instead of blocking on a console this process does not have.
os.environ.setdefault("COSCIENTIST_A2A_MODE", "1")

import uvicorn

from CoScientist.a2a.config import AGENT_PORTS
from CoScientist.a2a.server import make_a2a_app, make_agent_card
from CoScientist.assembly import build_system
from CoScientist.utils.interrupt import SharedSignalServer, request_exit

logger = logging.getLogger(__name__)

# How long graceful shutdown waits for an in-flight request (e.g. a long
# orchestrator LLM chain) before the serve loop stops anyway.
_SHUTDOWN_TIMEOUT = int(os.getenv("A2A_SHUTDOWN_TIMEOUT", "8"))


def _build_apps() -> list[tuple[str, object, int]]:
    """(label, app, port) for every a2a-exposed agent in system.yaml."""
    config = get_config()
    system = build_system(config)
    specs: list[tuple[str, object, int]] = []
    for agent_cfg in config.a2a_agents():
        if not agent_cfg.is_enabled():
            continue
        key = agent_cfg.a2a.key
        if agent_cfg.root:
            # Served in remote mode: delegations go to the sub-agent servers.
            agent = build_system(config, remote_subagents=True).root
        else:
            agent = system.agent(agent_cfg.name)
        app = make_a2a_app(agent, make_agent_card(agent_cfg), key)
        specs.append((agent_cfg.name, app, AGENT_PORTS[key]))
    return specs


def _make_server(app, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    # Don't let a long in-flight request block Ctrl+C forever.
    config.timeout_graceful_shutdown = _SHUTDOWN_TIMEOUT
    # One shared handler is installed in main(); a plain uvicorn.Server would
    # replace it, and Ctrl+C would then stop only one of the servers.
    return SharedSignalServer(config)


async def main() -> None:
    specs = _build_apps()
    servers = [_make_server(app, port) for _, app, port in specs]

    # Shared shutdown path: first Ctrl+C asks every server to stop gracefully
    # (and arms a hard-exit watchdog); a second one exits immediately.
    loop = asyncio.get_running_loop()

    def _request_shutdown() -> None:
        # Past the servers' own graceful timeout, so the watchdog only
        # catches what that timeout cannot: threads blocking interpreter exit.
        request_exit(grace=_SHUTDOWN_TIMEOUT + 2)
        logger.info("Shutdown requested; stopping all A2A servers...")
        for server in servers:
            server.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:  # e.g. on Windows
            signal.signal(sig, lambda *_: _request_shutdown())

    print("Starting CoScientist A2A agents:")
    for label, _, port in specs:
        print(f"  {label:<22} → http://localhost:{port}/")
    print()

    await asyncio.gather(*(server.serve() for server in servers))


if __name__ == "__main__":
    asyncio.run(main())
