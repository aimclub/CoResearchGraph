"""A2A host/port/URL maps, derived from the agent declarations in system.yaml.

Each agent with an ``a2a`` section gets an entry keyed by its ``a2a.key``. The
default port comes from the YAML and can be overridden per agent with the
``<KEY>_PORT`` env var (e.g. ``RESEARCH_PORT``); the host with ``A2A_HOST``.
"""
import os

from CoScientist.assembly.schema import get_config

A2A_HOST = os.getenv("A2A_HOST", "localhost")


def _agent_ports() -> dict[str, int]:
    return {
        agent.a2a.key: int(os.getenv(f"{agent.a2a.key.upper()}_PORT", str(agent.a2a.port)))
        for agent in get_config().a2a_agents()
    }


AGENT_PORTS: dict[str, int] = _agent_ports()

AGENT_URLS: dict[str, str] = {
    name: f"http://{A2A_HOST}:{port}/"
    for name, port in AGENT_PORTS.items()
}

# A2A well-known agent card URLs used by RemoteA2aAgent
AGENT_CARD_URLS: dict[str, str] = {
    name: f"http://{A2A_HOST}:{port}/.well-known/agent.json"
    for name, port in AGENT_PORTS.items()
}
