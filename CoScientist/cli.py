"""Unified command-line entry point for CoScientist.

One dispatcher for every way to run the system — prefer this over the
individual module entry points:

    python -m CoScientist web                      # web UI (uvicorn)
    python -m CoScientist cli                       # interactive REPL
    python -m CoScientist a2a all                   # all A2A agent servers
    python -m CoScientist a2a serve <key>           # one agent over A2A
    python -m CoScientist a2a bench --agent .. --text ..

Every heavy import is done lazily inside its handler. Some run modes (the A2A
servers) apply per-agent environment defaults BEFORE their agent/tool modules
load, so nothing agent-related may be imported at module top here. Keeping the
top of this module import-light also makes ``--help`` fast.

The legacy entry modules (``CoScientist.main``, ``CoScientist.web.server``,
``CoScientist.a2a.*``) still work and now forward here / are delegated to from
here, so existing commands and muscle memory keep working.
"""
import argparse
import sys
from typing import Optional

from dotenv import load_dotenv


def _configure_utf8_stdio() -> None:
    """Keep scientific Unicode readable on Windows and in redirected output."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            # Captured/test streams and already-closed streams may not support
            # reconfiguration. Output should remain best-effort in that case.
            pass


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------
def run_web(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Serve the FastAPI web interface via uvicorn."""
    import uvicorn

    # Pass the import string (not the app object) so ``--reload`` can re-import
    # the worker. The app itself lives in CoScientist.web.server:app.
    uvicorn.run(
        "CoScientist.web.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        # Above the 5 s polling of the builds page. At uvicorn's default of 5 s
        # a POST could land on a connection the server was closing, and the
        # browser reported a NetworkError for an action that never arrived.
        timeout_keep_alive=30,
    )


def run_repl() -> None:
    """Interactive terminal REPL against the in-process agent system."""
    _configure_utf8_stdio()
    import asyncio

    from CoScientist.main import create_manager

    async def _loop() -> None:
        manager = await create_manager()
        print("CoScientist (ADK) initialized — type 'exit' to quit.")
        print("(The web UI is also available: python -m CoScientist web)\n")
        try:
            while True:
                query = input("Enter query (or 'exit'): ")
                if query.lower() in {"exit", "quit"}:
                    break
                result = await manager.run(query)
                print("\n=== Final Response ===")
                print(result)
                print()
        finally:
            await manager.close()

    asyncio.run(_loop())


def _run_a2a(a2a_cmd: str, rest: list) -> None:
    """Delegate to the A2A modules, which own the env-before-import ordering.

    ``rest`` is forwarded verbatim to the underlying module's argparse, so e.g.
    ``a2a serve research --host 0.0.0.0`` and
    ``a2a bench --agent research --text "hi"`` keep their original flags.
    """
    if a2a_cmd == "all":
        import asyncio

        from CoScientist.a2a import run_all

        asyncio.run(run_all.main())
    elif a2a_cmd == "serve":
        from CoScientist.a2a import serve

        serve.main(rest)
    elif a2a_cmd == "bench":
        import asyncio

        from CoScientist.a2a import benchmark

        asyncio.run(benchmark.main(rest))


def _run_graph(
    graph_cmd: str,
    run_id: str,
    out: Optional[str],
    view: str = "execution",
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Execution-graph utilities over legacy or user/session snapshots."""
    from CoScientist.graph import viz

    if bool(user_id) != bool(session_id):
        raise ValueError("--user-id and --session-id must be provided together")
    if user_id and session_id:
        import os

        from CoScientist.graph.session_scope import storage_dir

        directory = storage_dir(
            os.getenv("GRAPH_SNAPSHOT_DIR", "./graph_runs"),
            (user_id, session_id),
        )
        full = viz.load_snapshot("execution", snapshot_dir=str(directory))
    else:
        full = viz.load_snapshot(run_id)

    graph_label = session_id or run_id

    if graph_cmd == "viz":
        out = out or f"knowledge_graph_{graph_label}_{view}.html"
        path = viz.render_html(full, out)
        print(f"Graph ({view}) written to {path}")
    elif graph_cmd == "dot":
        print(viz.to_dot(full))
    else:  # show
        nodes, edges = full.get("nodes", []), full.get("edges", [])
        scope = (
            f"user={user_id} session={session_id}"
            if user_id and session_id
            else f"run={run_id}"
        )
        print(f"{scope} view={view}  nodes={len(nodes)}  edges={len(edges)}")
        for n in nodes:
            label = (n.get("label") or "")[:80]
            print(f"  [{n.get('kind'):<10}] {n.get('id'):<30} {label}  ({n.get('status')})")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m CoScientist",
        description="CoScientist unified entry point.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    web = sub.add_parser("web", help="Run the web UI (uvicorn).")
    web.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    web.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    web.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")

    sub.add_parser("cli", help="Interactive terminal REPL.")

    # The a2a sub-command takes a mode plus arbitrary trailing flags that are
    # forwarded to the underlying module (collected via parse_known_args below,
    # so flags like --agent/--text/--host survive regardless of position —
    # argparse.REMAINDER would drop a leading "--flag").
    a2a = sub.add_parser(
        "a2a",
        help="Run A2A agent servers / benchmark client.",
        description=(
            "a2a all            serve all A2A agents in one process\n"
            "a2a serve <key>    serve one agent over A2A [--host HOST]\n"
            "a2a bench ...      smoke-test/perf client (--agent <key> --text <msg> ...)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    a2a.add_argument("a2a_cmd", choices=["all", "serve", "bench"], help="A2A run mode.")

    graph = sub.add_parser("graph", help="Knowledge-graph utilities.")
    graph.add_argument("graph_cmd", choices=["viz", "show", "dot"], help="viz=HTML, show=text, dot=Graphviz.")
    graph.add_argument("--run", default="session", help="run_id snapshot to load (default: session).")
    graph.add_argument("--user-id", default=None, help="user id for a scoped Web/CLI snapshot.")
    graph.add_argument("--session-id", default=None, help="session id for a scoped Web/CLI snapshot.")
    graph.add_argument("--out", default=None, help="output file (viz writes HTML here).")
    graph.add_argument("--view", choices=["execution"], default="execution",
                       help="execution=the session call log.")
    return parser


def main(argv=None) -> int:
    _configure_utf8_stdio()
    load_dotenv()
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)

    if args.cmd in ("web", "cli") and rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")

    if args.cmd == "web":
        run_web(args.host, args.port, args.reload)
    elif args.cmd == "cli":
        run_repl()
    elif args.cmd == "a2a":
        _run_a2a(args.a2a_cmd, rest)
    elif args.cmd == "graph":
        if bool(args.user_id) != bool(args.session_id):
            parser.error("--user-id and --session-id must be provided together")
        _run_graph(
            args.graph_cmd,
            args.run,
            args.out,
            args.view,
            user_id=args.user_id,
            session_id=args.session_id,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
