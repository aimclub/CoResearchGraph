"""Long results become documents; the chat keeps a summary and a button.

An agent's deliverable, an experiment plan, a work report — each of them runs to
tens of kilobytes, and printing one into the feed buries every message around
it. The convention this module generalises was already written down for the
ГОСТ ТЗ, in ``context_init/tz_agent.announcement``:

    Короткое намеренно. Тело документа в ленту не выгружается — карточку и
    ссылку читают глазами, а стена текста в чате хоронит и то и другое.

So the body is written as Markdown into the session's own artifact store, the
chat gets two or three lines and a button, and the panel opens the file. Because
the store is the same one every other artifact uses, a document costs nothing
extra: it is content-addressed, it is deduplicated, it rides along in an
exported session, and a ``cos-artifact:`` reference to it survives an import
under a new session id.

Nothing here raises. A document that cannot be written is a document the chat
message does without — the message still goes out, whole, exactly as before.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from CoScientist.graph.session_scope import SessionKey

logger = logging.getLogger(__name__)

#: `mimetypes.guess_type("a.md")` answers `(None, None)` on CPython 3.12, and a
#: record with no media type is served as `application/octet-stream` with
#: `Content-Disposition: attachment` — i.e. the panel would get a download
#: instead of a document. It is stated here rather than guessed.
MARKDOWN_MEDIA_TYPE = "text/markdown; charset=utf-8"

#: What a document is of. Goes into the record as `doc:<kind>`, which is what
#: the viewer's "session documents" list filters on.
KIND_PREFIX = "doc:"

#: The summary in the feed is two or three lines. Past this it stops being a
#: summary and starts being the thing it is meant to replace.
SUMMARY_LIMIT = 260

#: Titles share the cap the research graph uses for the same job.
TITLE_LIMIT = 120

#: Below this a message is already its own summary, and a document would be a
#: file and a button for one line the reader can see without either. The critic
#: verdict ("🔁 Отправлено на доработку (раунд 1/1)") is the case that showed it.
MIN_DOCUMENT_CHARS = 400

_FILENAMES = {
    "plan": "experiment-plan.md",
    "result": "experiment-results.md",
    "work_order": "work-order.md",
    "work_report": "work-report.md",
    "frame": "research-frame.md",
    "review": "review.md",
    "answer": "agent-answer.md",
    "node_report": "node-report.md",
}

#: Kinds published WITHOUT the `doc:` prefix, so they stay out of the chat
#: page's "session documents" list. A node's write-up belongs to the node — it
#: is opened from the graph, beside the card it describes — and a study with
#: forty nodes would otherwise bury the plan and the work reports under forty
#: rows the reader did not ask for.
UNLISTED_KINDS = frozenset({"node_report"})

#: Markdown that carries no prose: a heading rule, a table row, a fence.
_NOT_PROSE = re.compile(r"^\s*(?:[-=]{3,}|\|.*\||```|~~~|<!--)")
#: Leading list bullets and blockquote marks, so a summary reads as a sentence.
_LEAD_MARK = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|>\s*)")
#: Inline emphasis and code ticks, which are noise once the markup is gone.
_INLINE = re.compile(r"[*_`]{1,3}")
#: `[label](target)` → `label`.
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


@dataclass(frozen=True)
class ChatDocument:
    """One published document, as the chat needs to describe it."""

    artifact_id: str
    title: str
    summary: str
    kind: str

    def as_payload(self) -> Dict[str, str]:
        """The shape that rides on the websocket message.

        Deliberately no URL: the browser builds it from the session it is
        looking at, which is what lets an imported bundle resolve the same
        reference under a different session id.
        """
        return {"artifact_id": self.artifact_id, "title": self.title, "kind": self.kind}


def _clean(line: str) -> str:
    """One Markdown line as a reader would say it aloud."""
    text = _LINK.sub(r"\1", line)
    text = _INLINE.sub("", text)
    text = _LEAD_MARK.sub("", text)
    return text.strip()


def _cut(text: str, limit: int) -> str:
    """Trim to `limit`, preferring a sentence end and then a word boundary."""
    if len(text) <= limit:
        return text
    window = text[: limit + 1]
    for stop in (". ", "! ", "? ", "; "):
        cut = window.rfind(stop)
        if cut >= limit // 2:
            return window[: cut + 1].strip()
    cut = window.rfind(" ")
    return (window[:cut] if cut >= limit // 2 else text[:limit]).rstrip() + "…"


def summarise_markdown(markdown: Any, *, limit: int = SUMMARY_LIMIT) -> tuple[str, str]:
    """`(title, summary)` for a Markdown document.

    The title rule is the one `graph.research.store._headline` uses for a
    `Report` node — the first heading, else the first line — so a document and
    the graph card that will later point at it are named the same way.
    """
    text = str(markdown or "")
    title = ""
    summary_lines: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            # A blank line ends the summary only once it has something in it,
            # so a document that opens with a heading is not summarised empty.
            if summary_lines:
                break
            continue
        if line.startswith("#"):
            if not title:
                title = _cut(_clean(line.lstrip("#")), TITLE_LIMIT)
                continue
            # A second heading means the opening section is over.
            if summary_lines:
                break
            continue
        if _NOT_PROSE.match(line):
            if summary_lines:
                break
            continue
        said = _clean(line)
        if said:
            summary_lines.append(said)
        if sum(len(x) for x in summary_lines) > limit:
            break

    summary = _cut(" ".join(summary_lines), limit)
    if not title:
        title = _cut(summary, TITLE_LIMIT) if summary else ""
    return title, summary


def short_summary(text: Any, *, limit: int = SUMMARY_LIMIT) -> str:
    """One line for the feed out of a plain field — a goal, an agent's summary.

    Not `summarise_markdown`: that reads a document looking for its opening
    section. This is for a value the caller already has and only needs shortened
    to something that fits under a message header.
    """
    said = " ".join(_clean(line) for line in str(text or "").splitlines())
    return _cut(" ".join(said.split()), limit)


def publish_document(
    key: Optional[SessionKey],
    *,
    markdown: Any,
    kind: str,
    title: str = "",
    summary: str = "",
    agent: str = "",
    filename: str = "",
) -> Optional[ChatDocument]:
    """Write one Markdown document into the session's store.

    `title` and `summary` are what the caller already knows — a plan headline,
    an agent's own two-sentence result — and only what it does not know is
    derived from the text. Returns None when there is nothing to write or the
    write did not succeed; the caller then sends its message unchanged.
    """
    text = str(markdown or "").strip()
    if not key or len(text) < MIN_DOCUMENT_CHARS:
        return None

    derived_title, derived_summary = summarise_markdown(text)
    title = str(title or "").strip() or derived_title
    summary = str(summary or "").strip() or derived_summary
    if not title:
        return None

    try:
        from CoScientist.reporting import session_files

        record = session_files.put_bytes(
            key,
            text.encode("utf-8"),
            filename=filename or _FILENAMES.get(kind, f"{kind}.md"),
            label=_cut(title, TITLE_LIMIT),
            source_tool=agent,
            source_kind=(kind if kind in UNLISTED_KINDS
                         else f"{KIND_PREFIX}{kind}"),
            media_type=MARKDOWN_MEDIA_TYPE,
            agent=agent or None,
        )
    except Exception as exc:  # noqa: BLE001 — a document must not break a message
        logger.warning("document not written (%s): %s", kind, exc)
        return None

    if record.get("state") != session_files.STATE_STORED:
        logger.info(
            "document not stored (%s): %s", kind, record.get("reason") or "unknown"
        )
        return None

    artifact_id = str(record.get("artifact_id") or "")
    if not artifact_id:
        return None
    return ChatDocument(
        artifact_id=artifact_id,
        title=_cut(title, TITLE_LIMIT),
        summary=_cut(summary, SUMMARY_LIMIT),
        kind=kind,
    )


__all__ = [
    "ChatDocument",
    "KIND_PREFIX",
    "MIN_DOCUMENT_CHARS",
    "MARKDOWN_MEDIA_TYPE",
    "SUMMARY_LIMIT",
    "TITLE_LIMIT",
    "publish_document",
    "short_summary",
    "summarise_markdown",
]
