"""Smoke-check the normcontrol MCP and the NIR builder, end to end.

Two things can be wrong before a run ever asks for a report: the server is not
reachable (or is deployed without Times New Roman, which fails every render),
and the document we build does not satisfy the contract. This checks both
without starting an agent.

    # is the server up, and does it expose the NIR tools?
    MCP__NORMCONTROL_URL=http://localhost:8001/mcp python scripts/nir_smoke.py

    # also build a document from a finished run and render it
    python scripts/nir_smoke.py --session session_b8b88d83d09b463e90f61fd283241a6d

The session form reads the on-disk record directly — the research graph, the
frame, the execution graph and the report folder — so it needs no ADK session
and can be pointed at any completed run.

Exit code is 0 only when every requested stage passed, so this is usable as a
deployment gate.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CoScientist.reporting.nir import assets, build, client, contract  # noqa: E402
from CoScientist.reporting.nir.evidence import NirEvidence  # noqa: E402

REQUIRED_TOOLS = {client.TOOL_VALIDATE, client.TOOL_RENDER}


def _graph_root() -> Path:
    return Path(os.getenv("GRAPH_SNAPSHOT_DIR") or "graph_runs") / "sessions"


def _reports_root() -> Path:
    return Path(os.getenv("REPORTS_ROOT") or "logs/reports")


def _find_session(session_id: str) -> Path:
    """The per-session graph directory, wherever its user scope happens to be."""
    for candidate in _graph_root().glob(f"*/{session_id}"):
        if candidate.is_dir():
            return candidate
    raise SystemExit(f"session {session_id} not found under {_graph_root()}")


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_evidence(session_id: str) -> NirEvidence:
    """Rebuild a NirEvidence from what a finished run left on disk."""
    base = _find_session(session_id)
    report_dir = _reports_root() / session_id

    ev = NirEvidence(user_id=base.parent.name, session_id=session_id)
    graph = _load(base / "research_active.json") or {}
    ev.nodes = graph.get("nodes") or []
    ev.edges = graph.get("edges") or []

    frame = _load(base / "research_frame.json") or {}
    inner = frame.get("frame") if isinstance(frame.get("frame"), dict) else frame
    ev.original_request = str(inner.get("original_request") or "")

    execution = _load(base / "execution.json") or {}
    ev.agent_reports = [
        {"agent": str(n.get("executor_agent") or ""), "text": n["output"],
         "status": str(n.get("status") or "")}
        for n in execution.get("nodes") or []
        if n.get("kind") == "agent" and isinstance(n.get("output"), str) and n["output"].strip()
    ]

    ev.report_dir = report_dir
    for attribute in ("figures", "tables", "files"):
        directory = report_dir / attribute
        if directory.is_dir():
            setattr(ev, attribute, sorted(p for p in directory.iterdir() if p.is_file()))
    return ev


async def check_server() -> bool:
    url = client.normcontrol_url()
    if not url:
        print("FAIL  MCP__NORMCONTROL_URL is not set")
        return False
    print(f"      endpoint {url}")
    names = await client.list_tools()
    if names is None:
        print("FAIL  server unreachable")
        return False
    missing = REQUIRED_TOOLS - set(names)
    if missing:
        print(f"FAIL  server is up but does not expose {sorted(missing)}")
        return False
    print(f"OK    server exposes {len(names)} tools, including the NIR pair")
    return True


async def check_document(session_id: str, render: bool) -> bool:
    ev = load_evidence(session_id)
    print(f"      {len(ev.nodes)} graph nodes, {len(ev.agent_reports)} agent reports, "
          f"{len(ev.figures)} figures")

    bundle = assets.build_assets(ev.figures)
    for dropped in bundle.dropped:
        print(f"      dropped: {dropped}")
    outline = build.build_outline(ev, bundle)
    print(f"      {len(outline.sections)} sections: "
          + ", ".join(plan.id for plan in outline.sections))

    # Deliberately no prose: this checks the floor. Whatever the agent fails to
    # write, the builder must still produce a document the contract accepts.
    values, problems = build.build_nir_values(
        ev, build.NirRequisites(), build.NirProse(), outline
    )
    for problem in problems:
        print(f"      note: {problem}")

    ok, size = assets.fits(values, bundle.assets)
    print(f"      request {size / 1024 / 1024:.2f} MB "
          f"(limit {contract.MAX_REQUEST_BYTES // 1024 // 1024} MB)")
    if not ok:
        print("FAIL  request exceeds the server limit")
        return False

    result = await client.nir_validate(values, assets=bundle.assets)
    if not result.get("ok"):
        print(f"FAIL  validate: {result.get('error')}")
        for error in result.get("errors") or []:
            print(f"        {error}")
        return False
    print("OK    the server accepts the document")
    for warning in result.get("warnings") or []:
        code = warning.get("code") if isinstance(warning, dict) else warning
        print(f"      warning: {code}")

    if not render:
        return True

    rendered = await client.nir_render(values, assets=bundle.assets)
    if not rendered.get("ok"):
        print(f"FAIL  render: {rendered.get('error')}")
        # Both of these mean the document was fine and the deployment was not.
        if rendered.get("error") == contract.ERROR_FONTS_MISSING:
            print("      the server is deployed without Times New Roman — "
                  "use its Dockerfile or set GOSTDOC_FONT_DIR")
        elif rendered.get("error") == contract.ERROR_STORAGE_FAILED:
            print("      the server built the DOCX and could not store it — "
                  "its S3/MinIO is unreachable or unconfigured")
        return False
    print(f"OK    DOCX built: {rendered.get('size_bytes')} bytes, "
          f"link expires {rendered.get('expires_at')}")
    print(f"      {rendered.get('output_docx')}")
    return True


async def main(args: argparse.Namespace) -> int:
    passed = await check_server()
    if passed and args.session:
        passed = await check_document(args.session, args.render)
    print("\nSMOKE PASSED" if passed else "\nSMOKE FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", help="a finished session id to build a document from")
    parser.add_argument("--render", action="store_true",
                        help="also build the DOCX (takes the server's render lock)")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
