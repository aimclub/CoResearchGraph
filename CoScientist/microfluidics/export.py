"""after_agent export: persist the ТЗ + literature queries as Markdown and HTML.

Runs after the query-generation step, when both the structured ТЗ
(``structured_tz``) and the target literature queries (``tz_literature_queries``)
are in session state. Writes a shareable ``TZ_export_<timestamp>.md`` / ``.html``
pair under ``tz_documents/`` and reports the saved paths into the chat, so the
operator can save and hand the ТЗ and its literature-agent queries off.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from CoScientist.microfluidics.render import (
    render_tz_and_queries_html,
    render_tz_and_queries_markdown,
)

logger = logging.getLogger(__name__)

TZ_DOCUMENTS_DIR = Path("tz_documents")


def export_tz_and_queries(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """Persist the ТЗ and its literature queries as Markdown AND HTML files."""
    state = callback_context.state
    tz = state.get("structured_tz")
    if not tz:
        return None
    queries = state.get("tz_literature_queries")

    try:
        markdown = render_tz_and_queries_markdown(tz, queries)
        html_doc = render_tz_and_queries_html(tz, queries)
    except Exception as exc:  # noqa: BLE001 — export must never crash the run
        logger.warning("TZ+queries export render failed: %s", exc)
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        TZ_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        md_path = TZ_DOCUMENTS_DIR / f"TZ_export_{stamp}.md"
        html_path = TZ_DOCUMENTS_DIR / f"TZ_export_{stamp}.html"
        md_path.write_text(markdown, encoding="utf-8")
        html_path.write_text(html_doc, encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write TZ export files: %s", exc)
        return None

    logger.info("ТЗ + queries exported: %s | %s", md_path, html_path)
    text = (
        "📄 ТЗ и целевые запросы к литературному агенту сохранены для передачи:\n"
        f"- Markdown: {md_path.resolve()}\n"
        f"- HTML: {html_path.resolve()}"
    )
    return types.Content(role="model", parts=[types.Part(text=text)])


__all__ = ["export_tz_and_queries", "TZ_DOCUMENTS_DIR"]
