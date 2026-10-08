"""Collect everything one run produced into a single directory.

The pieces a run leaves behind live in four different places — the research and
execution graphs under ``graph_runs/sessions``, the UI's event journal under the
web state dir, the flat agent log wherever AGENT_LOG_JSONL pointed, and the cost
metrics only in the server's memory until something asks for them. A run is not
reproducible if any of them is missing, and the missing one is always noticed
after the process is gone.

Usage:
    python scripts/collect_session.py <user_id> <session_id> [--out DIR] \
        [--base http://127.0.0.1:8000]

Writes DIR/manifest.json plus metrics.json, graphs/, events, logs and a rendered
slide. Everything is copied, never moved: the run's own files stay where the
system expects them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")


def _copy(src: Path, dst_dir: Path, note: dict, key: str) -> None:
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        note[key] = {"file": src.name, "bytes": src.stat().st_size}
    else:
        note[key] = {"missing": str(src)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("user_id")
    ap.add_argument("session_id")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", default="http://127.0.0.1:8000",
                    help="running web server, for the cost metrics")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or f"/app/runs/{args.session_id}_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"collected_at": stamp, "user_id": args.user_id,
                      "session_id": args.session_id, "parts": {}}
    parts = manifest["parts"]

    # ── graphs: the research record and the execution trace ──────────────────
    sess = Path("graph_runs/sessions") / args.user_id / args.session_id
    for name in ("research_active.json", "research_frame.json", "execution.json"):
        _copy(sess / name, out / "graphs", parts, name)

    # ── the UI's event journal (this is what a replay would read) ────────────
    state_dir = Path(os.getenv("WEB_STATE_DIR", "graph_runs/web_state"))
    ev_dir = state_dir / args.user_id / args.session_id
    if ev_dir.is_dir():
        dst = out / "events"
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(ev_dir.glob("*.jsonl")):
            shutil.copy2(f, dst / f.name)
            n += 1
        parts["session_events"] = {"files": n, "dir": str(ev_dir)}
    else:
        parts["session_events"] = {"missing": str(ev_dir)}

    # ── the flat agent log, wherever this run was told to write it ───────────
    for env, key in (("AGENT_LOG_JSONL", "agent_events.jsonl"),
                     ("AGENT_LOG_FILE", "agent_events.log")):
        default = f"/app/{key}"
        _copy(Path(os.getenv(env, default)), out / "logs", parts, key)

    # ── cost metrics: only the running server has them ───────────────────────
    url = (f"{args.base.rstrip('/')}/api/users/{args.user_id}"
           f"/sessions/{args.session_id}/metrics?report=1")
    try:
        import httpx
        r = httpx.get(url, timeout=15.0)
        r.raise_for_status()
        (out / "metrics.json").write_text(r.text, encoding="utf-8")
        parts["metrics"] = {"file": "metrics.json", "source": url}
    except Exception as exc:  # noqa: BLE001
        parts["metrics"] = {"failed": f"{type(exc).__name__}: {exc}", "url": url,
                            "hint": "the server must still be running for this"}

    # ── the graph as a slide, so the record is readable without the UI ───────
    research = out / "graphs" / "research_active.json"
    if research.exists():
        try:
            from CoScientist.graph.research.slide_render import render_slide, render_html
            data = json.loads(research.read_text(encoding="utf-8"))
            svg = render_slide(data)
            (out / "research_graph.svg").write_text(svg, encoding="utf-8")
            (out / "research_graph.html").write_text(
                render_html(svg, title="research graph"), encoding="utf-8")
            parts["slide"] = {"file": "research_graph.svg"}
        except Exception as exc:  # noqa: BLE001
            parts["slide"] = {"failed": f"{type(exc).__name__}: {exc}"}

        # ── and the criteria audit, which is the point of the whole exercise ──
        try:
            import tempfile
            from CoScientist.graph.research.store import ResearchGraphStore
            from CoScientist.graph.research.queries import criteria_coverage
            tmp = tempfile.mkdtemp()
            shutil.copy(research, Path(tmp) / "g.json")
            cov = criteria_coverage(ResearchGraphStore(directory=tmp, active_file="g.json"))
            (out / "criteria_coverage.json").write_text(
                json.dumps(cov, ensure_ascii=False, indent=1), encoding="utf-8")
            parts["criteria_coverage"] = {
                "recorded_metrics": cov["recorded_metrics"],
                "gaps": [{"criteria": i["criteria"], "missing": i["missing"]}
                         for i in cov["items"]]}
        except Exception as exc:  # noqa: BLE001
            parts["criteria_coverage"] = {"failed": f"{type(exc).__name__}: {exc}"}

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"collected into {out}")
    for k, v in parts.items():
        flag = "!" if ("missing" in v or "failed" in v) else " "
        print(f" {flag} {k}: {v}")
    missing = [k for k, v in parts.items() if "missing" in v or "failed" in v]
    if missing:
        print(f"\nНЕ СОБРАНО: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
