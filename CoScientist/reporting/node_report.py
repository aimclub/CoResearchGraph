"""What one node of the research graph established, and who established it.

The execution log already answers "what did this agent do". This answers a
different question, the one a reader of the graph actually has in front of a
card: what does THIS node say, what is it resting on, who took part, and where
are the files.

TWO RULES, AND THEY ARE THE WHOLE DESIGN.

**The facts are assembled from the record; the model only writes the joins.**
`build_facts` reads the node, its status history, its participation record, its
artifacts and its neighbours — all of it already written down — and
`render_skeleton` prints them. The model is then given that text and asked for
two or three connecting paragraphs, forbidden to name a number, a file, a link
or an agent that is not already there. The server concatenates the two. A model
that fails, times out or is not configured leaves the skeleton, which is
complete and true on its own: there is no state in which this publishes an
invention.

**An agent's own account is cited, never copied.** The execution graph holds
one document per agent run, written by `agent_summary`. Copying it here would
make two texts that drift apart and say different things about the same run.
So a contributor's account contributes its opening line and a link to the run
that produced it — one document per thing, joined by `exec_id`.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: The kinds a node report is offered for. `Evidence` is the "observation" a
#: reader means; `PlanStep` and `ExperimentTask` are the two grains of plan
#: step; `VerificationMethod` is the card a reader opens between a hypothesis
#: and its evidence. `Report` is excluded — its own body already fills the
#: panel — and so are the projected `Framing`/`Outcome` cards, which have no
#: stored attributes to report on.
REPORTABLE = frozenset({"Evidence", "PlanStep", "ExperimentTask",
                        "VerificationMethod"})

SOURCE = "node-report"
KIND = "node_report"

_MAX_TOKENS = 900
_EXCERPT = 240

_WORDS = {
    "ru": {
        "title": "Отчёт по узлу", "facts": "Факты", "says": "Содержание",
        "status": "Состояние", "history": "Как менялось",
        "took_part": "Кто участвовал", "observed": "наблюдалось",
        "planned": "назначен планом, не подтверждено",
        "accounts": "Что сказали участники", "sources": "Источники",
        "files": "Файлы", "links": "Связи", "none": "нет",
        "no_account": "отчёт о работе этого агента ещё не написан",
        "read_more": "полный отчёт о запуске",
    },
    "en": {
        "title": "Node report", "facts": "Facts", "says": "Content",
        "status": "Status", "history": "How it moved",
        "took_part": "Who took part", "observed": "observed",
        "planned": "planned, not observed",
        "accounts": "What the participants reported", "sources": "Sources",
        "files": "Files", "links": "Connections", "none": "none",
        "no_account": "no account of this agent's run has been written yet",
        "read_more": "the full account of the run",
    },
}

_SYSTEM = (
    "You are given a complete and true account of one node of a research "
    "graph, already assembled from the record. Write, in {language}, two or "
    "three short paragraphs that say what this node establishes, what it rests "
    "on, and what remains open.\n"
    "You may NOT introduce a number, a file name, a link, an agent name, a "
    "date or a claim that is not in the text above. If the record is thin, say "
    "so in one sentence rather than padding. Do not repeat the lists verbatim "
    "— the reader has them. No heading, no preamble, no closing line."
)


def _words(lang: str) -> Dict[str, str]:
    return _WORDS.get(lang, _WORDS["ru"])


def _cut(value: Any, limit: int) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _node_of(view: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    return next((n for n in view.get("nodes") or []
                 if n.get("id") == node_id), None)


def build_facts(view: Dict[str, Any], node_id: str, *,
                summaries: Any = None, lang: str = "ru") -> Optional[Dict[str, Any]]:
    """Everything the record says about one node. Reads nothing else.

    `summaries` is the durable store of agent accounts, consulted but never
    written to: generating one account per contributor inside this call would
    turn opening a card into half a dozen model calls.
    """
    node = _node_of(view, node_id)
    if node is None:
        return None
    kind = str(node.get("kind") or "")
    contributors = list(node.get("contributors") or [])

    accounts: List[Dict[str, str]] = []
    seen_runs = set()
    for row in contributors:
        if not row.get("observed"):
            continue
        run = str(row.get("exec_id") or "")
        if not run or run in seen_runs:
            continue
        seen_runs.add(run)
        kept = None
        if summaries is not None:
            try:
                kept = summaries.latest(run, lang=lang)
            except Exception:  # noqa: BLE001 — a missing account is a fact too
                kept = None
        accounts.append({
            "agent": str(row.get("agent") or ""),
            "exec_id": run,
            # The opening line and a link, never the whole thing: the run's own
            # account lives in the execution log, and a second copy here would
            # drift from it.
            "excerpt": _cut((kept or {}).get("summary", ""), _EXCERPT),
        })

    return {
        "id": node_id,
        "kind": kind,
        "type_word": str(node.get("type_word") or kind),
        "index": node.get("index"),
        "label": str(node.get("label") or ""),
        "status": str(node.get("status") or ""),
        "status_word": str(node.get("status_word") or node.get("status") or ""),
        "fields": dict(node.get("input") or {}),
        "why": str(node.get("why") or ""),
        "history": list(node.get("status_history") or []),
        "contributors": contributors,
        "accounts": accounts,
        "attachments": list(node.get("attachments") or []),
        "chips": list(node.get("chips") or []),
        "links": _links_of(view, node_id),
    }


def _links_of(view: Dict[str, Any], node_id: str) -> List[Dict[str, str]]:
    """What this node connects to, in the reader's words."""
    by_id = {n.get("id"): n for n in view.get("nodes") or []}
    out = []
    for edge in view.get("edges") or []:
        src, dst = edge.get("src"), edge.get("dst")
        if node_id not in (src, dst):
            continue
        other = dst if src == node_id else src
        node = by_id.get(other)
        if not node:
            continue
        out.append({"type": str(edge.get("type") or ""),
                    "direction": "out" if src == node_id else "in",
                    "id": str(other),
                    "label": _cut(node.get("label"), 90)})
    return out


def _sources_of(facts: Dict[str, Any]) -> List[Dict[str, str]]:
    """Every work the node's own fields cite, by any identifier in them."""
    from CoScientist.reporting import references as refs

    text = " ".join(str(v) for v in facts["fields"].values())
    return [{"raw": r.raw or r.value, "url": r.url, "kind": r.kind}
            for r in refs.parse_references(text)]


def render_skeleton(facts: Dict[str, Any], lang: str = "ru") -> str:
    """The record, as markdown. Complete and true with no model involved."""
    w = _words(lang)
    head = f"{facts['type_word']} {facts['index'] or ''}".strip()
    lines = [f"# {w['title']} — {head} ({facts['id']})", ""]

    if facts["label"]:
        lines += [f"## {w['says']}", "", facts["label"], ""]
    lines += [f"## {w['facts']}", "",
              f"- **{w['status']}:** {facts['status_word']}"]
    for name, value in facts["fields"].items():
        if isinstance(value, (str, int, float)) and str(value).strip():
            lines.append(f"- **{name}:** {_cut(value, 400)}")
    if facts["why"]:
        lines.append(f"- **why:** {_cut(facts['why'], 400)}")
    lines.append("")

    if facts["history"]:
        lines += [f"## {w['history']}", ""]
        for move in facts["history"]:
            was = move.get("from_word") or move.get("from") or ""
            now = move.get("to_word") or move.get("to") or ""
            step = f"- {was} → {now}" if was else f"- {now}"
            if move.get("source"):
                step += f" ({move['source']})"
            if move.get("reason"):
                step += f" — {_cut(move['reason'], 200)}"
            lines.append(step)
        lines.append("")

    lines += [f"## {w['took_part']}", ""]
    if facts["contributors"]:
        for row in facts["contributors"]:
            # The distinction the whole participation record exists for: the
            # plan naming someone is an intention, not an observation.
            mark = w["observed"] if row.get("observed") else w["planned"]
            lines.append(f"- **{row.get('agent', '')}** — {row.get('basis', '')} "
                         f"({mark})")
    else:
        lines.append(f"- {w['none']}")
    lines.append("")

    if facts["accounts"]:
        lines += [f"## {w['accounts']}", ""]
        for account in facts["accounts"]:
            lines.append(f"### {account['agent']}")
            lines.append("")
            lines.append(account["excerpt"] or f"*{w['no_account']}*")
            lines.append("")
            lines.append(f"→ {w['read_more']}: `{account['exec_id']}`")
            lines.append("")

    sources = _sources_of(facts)
    if sources:
        lines += [f"## {w['sources']}", ""]
        for source in sources:
            lines.append(f"- [{source['raw']}]({source['url']})"
                         if source["url"] else f"- {source['raw']}")
        lines.append("")

    files = [a for a in facts["attachments"] + facts["chips"] if a.get("href")]
    if files:
        lines += [f"## {w['files']}", ""]
        for item in files:
            lines.append(f"- [{item.get('label') or item.get('id')}]({item['href']})")
        lines.append("")

    if facts["links"]:
        lines += [f"## {w['links']}", ""]
        for link in facts["links"]:
            arrow = "→" if link["direction"] == "out" else "←"
            lines.append(f"- {arrow} {link['type']}: {link['id']} {link['label']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def stamp(facts: Dict[str, Any], lang: str) -> str:
    """What makes an existing report stale.

    A NAMED list of inputs, and deliberately not `updated_at` or the attribute
    dict: writing the report puts `report_artifact_id` into attrs and moves
    `updated_at`, so a stamp over either would declare the report stale in the
    same breath as writing it.
    """
    parts = [
        facts["id"], facts["status"], str(len(facts["history"])), lang,
        # What the node SAYS, not only what happened to it. Left out at first,
        # and that was the bigger half of the hole: an Evidence whose text was
        # corrected kept the same stamp, so `write_report` returned the old
        # write-up — about the old text — and called it current. The report
        # attributes cannot leak in here: `_project_nodes` keeps them out of
        # `input`, and a test holds that shut.
        facts["label"],
        ";".join(f"{k}={v}" for k, v in sorted(facts["fields"].items())),
        ";".join(sorted(f"{c.get('agent')}|{c.get('basis')}"
                        for c in facts["contributors"])),
        ";".join(sorted(str(a.get("href") or "") for a in facts["attachments"])),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


async def narrate(skeleton: str, lang: str) -> str:
    """The model's part: the joins, and nothing the skeleton does not say."""
    from CoScientist.graph.agent_summary import _complete

    language = {"ru": "Russian", "en": "English"}.get(lang, "Russian")
    text, _model = await _complete(_SYSTEM.format(language=language), skeleton)
    return str(text or "").strip()


async def write_report(view: Dict[str, Any], node_id: str, *, scope,
                       store: Any, summaries: Any = None, lang: str = "ru",
                       force: bool = False) -> Optional[Dict[str, Any]]:
    """Assemble, narrate, publish, and stamp the node. Never raises.

    Returns ``{report, artifact_id, stamp, cached}`` or None when the node is
    not one a report is offered for.
    """
    facts = build_facts(view, node_id, summaries=summaries, lang=lang)
    if facts is None:
        return None
    node = _node_of(view, node_id) or {}
    if (node.get("type_word") and facts["kind"]) and not _reportable(view, node_id):
        return None

    current = stamp(facts, lang)
    if not force and str(node.get("report_stamp") or "") == current \
            and node.get("report"):
        return {"report": node["report"], "artifact_id": "",
                "stamp": current, "cached": True}

    skeleton = render_skeleton(facts, lang)
    body = skeleton
    try:
        narrative = await narrate(skeleton, lang)
        if narrative:
            body = f"{skeleton}\n---\n\n{narrative}\n"
    except Exception as exc:  # noqa: BLE001
        # The skeleton stands on its own. A model that is down, slow or not
        # configured costs the reader the prose, never the facts.
        logger.info("node report %s: no narrative (%s)", node_id, exc)

    artifact_id = _publish(scope, facts, body, lang)
    if artifact_id:
        _stamp_node(store, node_id, artifact_id, current, lang)
    return {"report": body, "artifact_id": artifact_id, "stamp": current,
            "cached": False}


def _reportable(view: Dict[str, Any], node_id: str) -> bool:
    node = _node_of(view, node_id) or {}
    kind = str(node.get("kind") or "")
    return any(kind == t.lower() for t in REPORTABLE)


def _publish(scope, facts: Dict[str, Any], body: str, lang: str) -> str:
    try:
        from CoScientist.reporting.documents import publish_document

        head = f"{facts['type_word']} {facts['index'] or facts['id']}".strip()
        document = publish_document(
            scope, markdown=body, kind=KIND, agent=SOURCE,
            title=f"{_words(lang)['title']} — {head}",
            filename=f"node-{facts['id'].lower()}-report.md")
        return document.artifact_id if document else ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("node report %s not published: %s", facts["id"], exc)
        return ""


def _stamp_node(store: Any, node_id: str, artifact_id: str, current: str,
                lang: str) -> None:
    """Point the node at its write-up. The body never goes into attrs."""
    try:
        store.commit(source=SOURCE, allow_reserved=True, nodes=[{
            "id": node_id,
            "attrs": {"report_artifact_id": artifact_id,
                      "report_stamp": current, "report_lang": lang}}])
    except Exception as exc:  # noqa: BLE001
        logger.warning("node report %s not stamped: %s", node_id, exc)


__all__ = ["REPORTABLE", "KIND", "SOURCE", "build_facts", "render_skeleton",
           "stamp", "narrate", "write_report"]
