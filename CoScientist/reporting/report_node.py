"""Put the run's write-up into the research graph, as a node a reader can open.

The graph ended a study with two cards and no document. ``Итог`` is derived, and
carries one line — ``"<hypothesis> — <status>"`` — with no input, no provenance
and no history; its emptiness is structural, not a bug. Meanwhile an 18 KB
``report.md`` sat on disk, written by the aggregator, referenced from nowhere.

So the write-up gets a card of its own. The panel already renders markdown
(``reportBlock``, marked + DOMPurify, scrollable); it had simply never been
handed anything longer than a headline.

Published by code, not by a model. ``ResultAggregatorAgent`` holds the read-only
research surface, and granting an LLM write rights to record something this
deterministic would widen the ACL for no gain — ``finalize_report`` already has
the text. The write goes in under the ``report-writer`` source with enforcement
on, so the permission is real rather than bypassed.

Best-effort by contract: a study that cannot record its report still has its
report on disk and in the chat.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SOURCE = "report-writer"

#: What the card is called when the document has no heading of its own.
DEFAULT_TITLE = "Результаты исследования"


def _anchor(nodes: Dict[str, Dict[str, Any]], root: Optional[str]) -> Optional[str]:
    """What the report hangs off: the newest Conclusion, else the question.

    A study that reached no conclusion still produced a write-up, and a node
    with no edge is a node nobody finds.
    """
    conclusions = [
        nid for nid, data in nodes.items() if data.get("type") == "Conclusion"
    ]
    if conclusions:
        return sorted(conclusions)[-1]
    if root:
        return root
    questions = [
        nid for nid, data in nodes.items() if data.get("type") == "ResearchQuestion"
    ]
    return sorted(questions)[0] if questions else None


def publish_report_node(
    user_id: str,
    session_id: str,
    markdown: str,
    *,
    artifact_id: Optional[str] = None,
    title: str = "",
) -> Optional[str]:
    """Record the final write-up as a ``Report`` node. Returns its id, or None.

    Idempotent: a second finalize of the same session updates the node it wrote
    the first time rather than adding another.
    """
    text = str(markdown or "").strip()
    if not (text and user_id and session_id):
        return None

    try:
        from CoScientist.graph.research.store import get_research_graph
        from CoScientist.utils.report_links import remint_report_urls

        store = get_research_graph(user_id=user_id, session_id=session_id)
        raw = store.full() or {}
        nodes = {n["id"]: n for n in (raw.get("nodes") or []) if n.get("id")}

        attrs: Dict[str, Any] = {
            # S3 URLs are reminted here, because a presigned link that expires
            # is wrong in the record as well as on the screen. `cos-artifact:`
            # references are deliberately NOT resolved: they carry no session,
            # which is what lets an exported bundle draw this report's figures
            # after an import under a new id. The view layer resolves them from
            # whatever scope is reading (see `_readable_body`).
            "content": remint_report_urls(text),
            "title": title or DEFAULT_TITLE,
            "name": "report.md",
        }
        if artifact_id:
            # Lets the chip beside the card open the file itself. Under the name
            # `_href` actually looks for — `artifact_id` is the experiment
            # runtime's own namespace and is not consulted there, so writing it
            # under that key produced a chip that opened nothing.
            attrs["session_artifact_id"] = artifact_id

        existing = next(
            (nid for nid, data in nodes.items() if data.get("type") == "Report"), None
        )
        if existing:
            result = store.commit(source=_SOURCE, nodes=[{"id": existing, "attrs": attrs}])
            return existing if getattr(result, "ok", False) else None

        anchor = _anchor(nodes, raw.get("root_id"))
        if anchor is None:
            logger.info("report node: nothing to attach to; skipping")
            return None

        edges: List[Dict[str, Any]] = [
            {"type": "derived_from", "from": "#rp", "to": anchor}
        ]
        result = store.commit(
            source=_SOURCE,
            nodes=[{"type": "Report", "ref": "rp", "attrs": attrs}],
            edges=edges,
        )
        if not getattr(result, "ok", False):
            logger.warning(
                "report node: commit refused (%s)", getattr(result, "errors", None)
            )
            return None
        created = [n.get("id") for n in (getattr(result, "committed", {}) or {}).get("nodes", [])]
        logger.info("report node: published %s (%d chars)", created, len(text))
        return created[0] if created else None
    except Exception as exc:  # noqa: BLE001 — a report must not sink on this
        logger.warning("report node: could not publish (%s)", exc)
        return None


__all__ = ["publish_report_node", "DEFAULT_TITLE"]
