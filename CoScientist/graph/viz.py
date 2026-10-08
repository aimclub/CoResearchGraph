"""Visualisation for the knowledge / execution graph.

* ``to_dot(full)``       — Graphviz DOT text (no system binary needed to produce).
* ``render_html(full)``  — self-contained interactive HTML via pyvis.
* ``render(run_id)``     — load a snapshot JSON and write an HTML file.

A "full" graph is the dict returned by GraphStore.full / knowledge_graph.full:
``{"run_id", "nodes": [...], "edges": [...]}``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Run-step nodes are coloured by STATUS; the static structure (system root and
# it acted) keeps its own identity colour so it doesn't read as "done".
_STATUS_COLOR = {
    "running": "#f1c40f",  # amber
    "success": "#2ecc71",  # green
    "failed": "#e74c3c",   # red
    "pruned": "#7f8c8d",   # grey
}
_IDENTITY_COLOR = {"system": "#f4c542", "agent": "#7eb6ff"}

# Shape encodes the node KIND so type is readable without relying on colour.
_SHAPE = {
    "system": "hexagon",
    "agent": "dot",
    "goal": "diamond",
    "agent_call": "ellipse",
    "tool_call": "box",
    "result": "star",
}



def _color(node: Dict[str, Any]) -> str:
    kind = node.get("kind", "")
    if kind in _IDENTITY_COLOR:
        return _IDENTITY_COLOR[kind]
    return _STATUS_COLOR.get(node.get("status", ""), "#dcdde1")


def _shape(node: Dict[str, Any]) -> str:
    return _SHAPE.get(node.get("kind", ""), "box")


def _label(node: Dict[str, Any]) -> str:
    # tool_call: show WHAT tool; agent nodes: show WHICH agent. Who called a tool
    # is obvious from its parent agent node, so we don't repeat the caller.
    if node.get("kind") in ("tool_call", "result", "goal"):
        base = node.get("label") or node.get("id", "")
    else:
        base = node.get("executor_agent") or node.get("label") or node.get("id", "")
    return str(base)[:40]


def _title(node: Dict[str, Any]) -> str:
    bits = [f"{k}: {node.get(k)}" for k in ("id", "kind", "executor_agent", "status") if node.get(k)]
    if node.get("label"):
        bits.append(f"label: {node['label']}")
    if node.get("output"):
        bits.append(f"output: {str(node['output'])[:300]}")
    return "\n".join(bits)


# vis shape -> graphviz shape (graphviz has no "dot"/"ellipse" name clash issues)
_DOT_SHAPE = {"hexagon": "hexagon", "dot": "circle", "diamond": "diamond",
              "ellipse": "ellipse", "box": "box", "star": "star"}


def to_dot(full: Dict[str, Any]) -> str:
    lines = ["digraph KnowledgeGraph {", '  rankdir=LR;', '  node [style=filled];']
    for n in full.get("nodes", []):
        nid = n.get("id", "")
        lbl = (_label(n) or nid).replace('"', "'")
        shape = _DOT_SHAPE.get(_shape(n), "box")
        lines.append(f'  "{nid}" [label="{lbl}", fillcolor="{_color(n)}", shape={shape}];')
    for e in full.get("edges", []):
        lines.append(f'  "{e.get("src")}" -> "{e.get("dst")}" [label="{e.get("type", "")}"];')
    lines.append("}")
    return "\n".join(lines)


def render_html(full: Dict[str, Any], out_path: str) -> str:
    """Write an interactive HTML view; returns the path. Falls back to an
    embedded-JSON page if pyvis is unavailable."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from pyvis.network import Network

        net = Network(height="800px", width="100%", directed=True, bgcolor="#1e1e1e", font_color="#eee")
        net.barnes_hut()
        for n in full.get("nodes", []):
            net.add_node(
                n.get("id"), label=_label(n), title=_title(n), color=_color(n),
                shape=_shape(n),
            )
        existing = {n.get("id") for n in full.get("nodes", [])}
        for e in full.get("edges", []):
            if e.get("src") in existing and e.get("dst") in existing:
                net.add_edge(e.get("src"), e.get("dst"), label=e.get("type", ""))
        net.save_graph(str(out))
        return str(out)
    except Exception:  # noqa: BLE001 — fall back to a dependency-free page
        out.write_text(_fallback_html(full), encoding="utf-8")
        return str(out)


def _fallback_html(full: Dict[str, Any]) -> str:
    data = json.dumps(full, ensure_ascii=False, default=str)
    return (
        "<!doctype html><meta charset='utf-8'><title>Knowledge graph</title>"
        "<h3>Knowledge graph snapshot</h3>"
        "<pre style='white-space:pre-wrap;font:13px monospace'>"
        + data.replace("<", "&lt;")
        + "</pre>"
    )


def load_snapshot(run_id: str = "session", snapshot_dir: Optional[str] = None) -> Dict[str, Any]:
    snapshot_dir = snapshot_dir or os.getenv("GRAPH_SNAPSHOT_DIR", "./graph_runs")
    path = Path(snapshot_dir) / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No graph snapshot at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def render(run_id: str = "session", out_path: Optional[str] = None,
           snapshot_dir: Optional[str] = None) -> str:
    full = load_snapshot(run_id, snapshot_dir)
    out_path = out_path or f"knowledge_graph_{run_id}.html"
    return render_html(full, out_path)
