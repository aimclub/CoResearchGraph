"""Serve ONE agent from system.yaml as a standalone A2A service.

Usage (from /app):
    python -m CoScientist.a2a.serve <a2a-key>
    python -m CoScientist.a2a.serve research
    python -m CoScientist.a2a.serve orchestrator   # sub-agents must be running

The agent, its card, port and pre-import env defaults all come from the
agent's ``a2a`` section in system.yaml — there are no per-agent server modules.
"""
import argparse
import os


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", help="a2a key of the agent to serve (see system.yaml)")
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args(argv)

    # Only the light-weight config is imported before env defaults are applied:
    # some agents need env set before their tool modules load (e.g. the coder's
    # shared A2A workspace id).
    from CoScientist.assembly.schema import get_config

    agent_cfg = get_config().a2a_agent_by_key(args.key)
    for name, value in agent_cfg.a2a.env.items():
        os.environ.setdefault(name, value)

    # Served over A2A: HITL must not block on a console that isn't there. Set
    # BEFORE build_system() so the assembler attaches the A2A (long-running)
    # HITL tools, which put the task into `input-required` for the caller to
    # answer instead of hanging the server.
    os.environ.setdefault("COSCIENTIST_A2A_MODE", "1")

    import uvicorn

    from CoScientist.a2a.config import AGENT_PORTS
    from CoScientist.a2a.server import make_a2a_app, make_agent_card
    from CoScientist.assembly import build_system

    if agent_cfg.root:
        # The orchestrator delegates to its sub-agents over A2A.
        agent = build_system(remote_subagents=True).root
    else:
        agent = build_system().agent(agent_cfg.name)

    app = make_a2a_app(agent, make_agent_card(agent_cfg), agent_cfg.a2a.key)
    uvicorn.run(app, host=args.host, port=AGENT_PORTS[agent_cfg.a2a.key], log_level="info")


if __name__ == "__main__":
    main()
