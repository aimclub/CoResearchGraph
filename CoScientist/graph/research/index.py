# -*- coding: utf-8 -*-
"""Cross-run index of finished researches — so a new run can reuse old ones.

Each research graph already persists per session
(``graph_runs/sessions/<user>/<session>/research_active.json``), but nothing
ever read those back: every run started from zero and re-derived knowledge the
system had already produced. This module keeps one small, DERIVED catalogue —
``graph_runs/research_index.json`` — mapping each past research to what it
settled: its question, hypotheses with verdicts, conclusions, the verification
methods and tools it used.

The index is derived, never authoritative: delete it and ``rebuild()`` restores
it from the snapshots on disk. Matching is deterministic token overlap (the same
convention the knowledge memory uses) so a demo can SHOW why a prior matched —
no embedding service required.

CLI:
    python -m CoScientist.graph.research.index --rebuild
    python -m CoScientist.graph.research.index --search "молекулы эволюция SA"
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_LOCK = threading.RLock()
_TOKEN_RE = re.compile(r"[^\wа-яё]+", re.IGNORECASE)
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "are", "was",
    "которая", "которые", "который", "чтобы", "если", "быть", "было", "этой", "этот",
    "для", "как", "что", "при", "или", "все", "его", "она", "они", "нужно", "можно",
}


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.split((text or "").lower())
            if len(t) > 2 and t not in _STOP]


def index_path() -> Path:
    base = Path(os.getenv("RESEARCH_GRAPH_DIR", "./graph_runs"))
    return base / os.getenv("RESEARCH_INDEX_FILE", "research_index.json")


def _attr(node: Dict[str, Any], *keys: str) -> str:
    attrs = node.get("attrs") or {}
    for k in keys:
        v = attrs.get(k)
        if v not in (None, "", [], {}):
            return str(v)
    return ""


def summarize(data: Dict[str, Any], *, path: str = "", user_id: str = "",
              session_id: str = "") -> Optional[Dict[str, Any]]:
    """Derive one index record from a serialized research graph.

    Returns None for an empty / rootless graph (there are such leftovers on disk
    and they carry nothing worth reusing).
    """
    nodes = data.get("nodes") or []
    if not nodes:
        return None
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for n in nodes:
        by_type.setdefault((n.get("type") or "").lower(), []).append(n)

    root_id = data.get("root_id")
    root = next((n for n in nodes if n.get("id") == root_id), None)
    if root is None:
        root = (by_type.get("researchquestion") or [None])[0]
    question = _attr(root, "formulation") if root else ""
    if not question:
        return None

    hyps = by_type.get("hypothesis", [])
    concls = by_type.get("conclusion", [])
    methods = by_type.get("verificationmethod", [])
    evidence = by_type.get("evidence", [])
    tools = by_type.get("tool", [])

    statuses = [h.get("status") or "" for h in hyps]
    record = {
        "research_id": data.get("research_id") or Path(path).stem or "research",
        "question": question,
        "domain": _attr(root, "domain") if root else "",
        "user_id": user_id,
        "session_id": session_id,
        "path": path,
        "created_at": data.get("created_at"),
        "indexed_at": time.time(),
        "counts": {
            "hypotheses": len(hyps),
            "confirmed": sum(s == "confirmed" for s in statuses),
            "refuted": sum(s == "refuted" for s in statuses),
            "evidence": len(evidence),
            "conclusions": len(concls),
            "nodes": len(nodes),
        },
        # What a later run actually wants to reuse: the claims and their verdicts,
        # and HOW they were checked (methods/tools carry the operational know-how).
        "hypotheses": [{"id": h.get("id"), "status": h.get("status"),
                        "formulation": _attr(h, "formulation", "statement")[:400]}
                       for h in hyps],
        "conclusions": [{"id": c.get("id"), "status": c.get("status"),
                         "synthesis": _attr(c, "synthesis", "conclusion")[:400]}
                        for c in concls],
        "methods": [{"id": m.get("id"),
                     "procedure": _attr(m, "procedure", "method", "description",
                                        "method_type")[:400]}
                    for m in methods],
        # Carry WHERE each tool is, not just what it is called. A later run gets
        # this instead of the graph itself, and a name alone ("GOLEM") leaves its
        # coder — which starts in a clean environment — to guess or, as happened,
        # to write its own replacement.
        "tools": sorted({
            (lambda n, loc: f"{n} — {loc}" if loc else n)(
                _attr(t, "name") or t.get("id"), _attr(t, "location")[:200])
            for t in tools if t}),
    }
    text = " ".join([question, _attr(root, "domain") if root else ""]
                    + [h["formulation"] for h in record["hypotheses"]]
                    + [c["synthesis"] for c in record["conclusions"]]
                    + [m["procedure"] for m in record["methods"]])
    record["tokens"] = sorted(set(_tokens(text)))
    return record


class ResearchIndex:
    """Small JSON catalogue of past researches (atomic writes, corrupt-tolerant)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else index_path()
        self._records: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    # ── storage ──────────────────────────────────────────────────────────────
    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for rec in data.get("researches", []):
                    rid = rec.get("research_id")
                    if rid:
                        self._records[rid] = rec
        except Exception:  # noqa: BLE001 — a corrupt index must never break a run
            self._records = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"updated_at": time.time(),
                       "researches": list(self._records.values())}
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2),
                           encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001 — indexing is best-effort
            pass

    # ── api ──────────────────────────────────────────────────────────────────
    def upsert(self, record: Optional[Dict[str, Any]]) -> bool:
        if not record or not record.get("research_id"):
            return False
        with _LOCK:
            self._load()
            self._records[record["research_id"]] = record
            self._save()
        return True

    def all(self) -> List[Dict[str, Any]]:
        with _LOCK:
            self._load()
            return list(self._records.values())

    def search(self, query: str, *, limit: int = 3,
               exclude_id: str = "") -> List[Dict[str, Any]]:
        """Rank past researches by token overlap with `query`.

        Deterministic and explainable: every hit carries the tokens that matched,
        so a demo can show WHY this prior was surfaced.
        """
        q = set(_tokens(query))
        if not q:
            return []
        hits = []
        for rec in self.all():
            if exclude_id and rec.get("research_id") == exclude_id:
                continue
            toks = set(rec.get("tokens") or [])
            shared = q & toks
            if not shared:
                continue
            qtoks = set(_tokens(rec.get("question", "")))
            score = len(shared) / len(q) + 0.5 * len(q & qtoks) / max(len(q), 1)
            # settled work (a verdict was reached) is worth more than an open one
            counts = rec.get("counts") or {}
            if counts.get("confirmed") or counts.get("refuted"):
                score += 0.25
            hits.append({**rec, "score": round(score, 3),
                         "matched_tokens": sorted(shared)[:12]})
        hits.sort(key=lambda r: r["score"], reverse=True)
        return hits[:limit]

    def rebuild(self, base: Optional[Path] = None) -> int:
        """Re-derive the whole index from the snapshots on disk."""
        base = Path(base or os.getenv("RESEARCH_GRAPH_DIR", "./graph_runs"))
        found = 0
        with _LOCK:
            self._load()
            self._records = {}
            for f in sorted(base.glob("sessions/*/*/research_active.json")):
                parts = f.parts
                user = next((p for p in parts if p.startswith("user_")), "")
                sess = next((p for p in parts if p.startswith("session_")), "")
                try:
                    rec = summarize(json.loads(f.read_text(encoding="utf-8")),
                                    path=str(f), user_id=user, session_id=sess)
                except Exception:  # noqa: BLE001
                    rec = None
                if rec:
                    self._records[rec["research_id"]] = rec
                    found += 1
            self._save()
        return found


_INDEX: Optional[ResearchIndex] = None


def get_research_index() -> ResearchIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = ResearchIndex()
    return _INDEX


def index_research(data: Dict[str, Any], *, path: str = "", user_id: str = "",
                   session_id: str = "") -> bool:
    """Index one research graph (best-effort; never raises into a run)."""
    try:
        return get_research_index().upsert(
            summarize(data, path=path, user_id=user_id, session_id=session_id))
    except Exception:  # noqa: BLE001
        return False


def format_priors(hits: Iterable[Dict[str, Any]], budget: int = 1400) -> str:
    """Render prior researches as a compact digest for an agent's context."""
    out: List[str] = []
    for h in hits:
        c = h.get("counts") or {}
        lines = [f"• Прошлое исследование «{h.get('question', '')[:110]}» "
                 f"(похожесть {h.get('score')}, гипотез {c.get('hypotheses', 0)}: "
                 f"✓{c.get('confirmed', 0)} ✗{c.get('refuted', 0)})"]
        for hyp in (h.get("hypotheses") or [])[:3]:
            lines.append(f"    – [{hyp.get('status')}] {(hyp.get('formulation') or '')[:130]}")
        for m in (h.get("methods") or [])[:2]:
            lines.append(f"    – метод: {(m.get('procedure') or '')[:130]}")
        for cl in (h.get("conclusions") or [])[:2]:
            lines.append(f"    – вывод: {(cl.get('synthesis') or '')[:130]}")
        if h.get("tools"):
            lines.append(f"    – инструменты: {', '.join(h['tools'][:6])}")
        out.append("\n".join(lines))
    text = "\n".join(out)
    return text[:budget] + ("…" if len(text) > budget else "")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="re-derive from snapshots")
    ap.add_argument("--search", default="", help="query past researches")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args(argv)
    idx = get_research_index()
    if args.rebuild:
        print(f"indexed {idx.rebuild()} researches -> {idx.path}")
    if args.search:
        hits = idx.search(args.search, limit=args.limit)
        print(f"{len(hits)} match(es) for {args.search!r}\n")
        print(format_priors(hits, budget=4000) or "(nothing)")
    if not args.rebuild and not args.search:
        print(f"{len(idx.all())} researches indexed at {idx.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
