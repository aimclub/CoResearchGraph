"""In-process tools for NirReportAgent: outline, draft, submit.

The normcontrol MCP is reached through a wrapper rather than attached as an
``McpToolset``, because its ``assets`` argument cannot pass through a model: one
287 KB figure is ~383 000 base64 characters. The wrapper fills ``assets`` from
the files ``collect_artifacts`` already downloaded, so the agent only ever
handles prose.

Three calls, deliberately:

* ``nir_report_outline`` — what the report will contain and what evidence backs
  each section. Read-only, no network.
* ``nir_report_draft`` — the agent's prose, assembled into a document and
  checked. Still no network, so the agent can iterate on style and on missing
  sections for free before anything is rendered.
* ``nir_report_submit`` — validate remotely, render, then bring the DOCX home:
  local file, our S3, and a link that outlives the server's 24-hour one.

Everything runs in the session scope from ``session_key``, not ``session.id``.
Inside an AgentTool the raw session id is a fresh random one, and the report
folder, the graph and the artifact index all live under the public scope.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from CoScientist.config import get_settings
from CoScientist.config.report import ReportConfig
from CoScientist.graph.session_scope import session_key
from CoScientist.reporting.collect import report_dir_for
from CoScientist.reporting.nir import assets as nir_assets
from CoScientist.reporting.nir import (
    build, client, contract, hitl_form, register_check, slop_check,
)
from CoScientist.reporting.nir.evidence import collect_nir_evidence

logger = logging.getLogger(__name__)

#: Where the DOCX lands inside the report folder, and its object key prefix.
DOCX_NAME = "nir_report.docx"
_S3_FEATURE = "nir"

#: Survives one turn of the agent between draft and submit.
_STATE_DRAFT = "nir_report_draft"


def _report_dir(tool_context: ToolContext) -> Path:
    state = _state(tool_context)
    cfg = ReportConfig.from_mapping(state.get("report_config"))
    _, session_id = session_key(tool_context)
    return report_dir_for(session_id, cfg.reports_root)


def _state(tool_context: ToolContext) -> Dict[str, Any]:
    from CoScientist.reporting.nir.evidence import _state_to_dict

    return _state_to_dict(getattr(tool_context, "state", None))


def _request(tool_context: ToolContext) -> Dict[str, Any]:
    value = _state(tool_context).get(hitl_form.STATE_REQUEST_KEY)
    return value if isinstance(value, dict) else {}


def _not_requested() -> Dict[str, Any]:
    return {
        "status": "skipped",
        "message": (
            "Отчёт о НИР не запрашивался оператором — этот инструмент вызывать не нужно."
        ),
    }


async def nir_report_outline(tool_context: ToolContext) -> Dict[str, Any]:
    """Show what the GOST report will contain and which evidence backs each part.

    Call this FIRST. It returns the section ids you must write prose for, the
    facts recorded for each of them (graph nodes, agent reports, measured
    numbers), the figures that need captions, and the sources already collected.
    Write only from what it returns; do not add facts from memory."""
    if not _request(tool_context).get("enabled"):
        return _not_requested()

    report_dir = _report_dir(tool_context)
    evidence = await asyncio.to_thread(
        collect_nir_evidence, tool_context, report_dir, _state(tool_context)
    )
    bundle = await asyncio.to_thread(nir_assets.build_assets, evidence.figures)
    outline = build.build_outline(evidence, bundle)

    return {
        "status": "success",
        "question": outline.question,
        "sections": [
            {
                "id": plan.id,
                "title_hint": plan.title,
                "evidence": plan.digest,
                "evidence_chars": plan.evidence_chars(),
                "figures": plan.figure_briefs(),
                "tables": len(plan.tables),
            }
            for plan in outline.sections
        ],
        # The agents' own reports. Not pinned to any one section: this is where
        # most of the run's concrete detail lives, and any section may use it.
        "materials": outline.materials,
        "sources_collected": len(build._references(evidence, None)),
        "figures_encoded": len(bundle.assets),
        "figures_dropped": bundle.dropped,
        "gaps": outline.gaps,
        "requisites": _request(tool_context).get("requisites", {}),
    }


async def nir_report_draft(
    tool_context: ToolContext,
    research_title: str,
    report_title: str,
    abstract_text: str,
    keywords: List[str],
    introduction_paragraphs: List[str],
    conclusion_paragraphs: List[str],
    section_texts: Dict[str, List[str]],
    figure_captions: Optional[Dict[str, str]] = None,
    section_titles: Optional[Dict[str, str]] = None,
    terms: Optional[List[Dict[str, str]]] = None,
    abbreviations: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Assemble your prose into a GOST document and check it, without sending it.

    ``section_texts`` maps a section id from the outline to its paragraphs.
    ``figure_captions`` maps a figure id ("fig-1") to its caption. You supply
    text only — structure, figure placement, tables, sources and appendices are
    built here.

    ``section_titles`` maps a section id to the heading you want for it —
    write one for every section, naming its subject rather than its number.

    Returns what needs fixing: unwritten sections, missing keywords, an
    over-long abstract, style warnings about AI-sounding prose, and
    ``register_warnings`` for internal notation that leaked into the text
    (node ids, plan ids, function or file names, untranslated schema words).
    Fix them and call this again — it costs nothing. Then call
    nir_report_submit."""
    if not _request(tool_context).get("enabled"):
        return _not_requested()

    report_dir = _report_dir(tool_context)
    evidence = await asyncio.to_thread(
        collect_nir_evidence, tool_context, report_dir, _state(tool_context)
    )
    bundle = await asyncio.to_thread(nir_assets.build_assets, evidence.figures)
    outline = build.build_outline(evidence, bundle)

    prose = build.NirProse(
        research_title=research_title,
        report_title=report_title,
        abstract_text=abstract_text,
        keywords=list(keywords or []),
        introduction_paragraphs=list(introduction_paragraphs or []),
        conclusion_paragraphs=list(conclusion_paragraphs or []),
        section_texts={k: list(v) for k, v in (section_texts or {}).items()},
        figure_captions=dict(figure_captions or {}),
        section_titles=dict(section_titles or {}),
        terms=list(terms or []),
        abbreviations=list(abbreviations or []),
    )
    requisites = hitl_form.requisites_from_state(_request(tool_context).get("requisites"))
    values, problems = build.build_nir_values(evidence, requisites, prose, outline)

    style_input = {"введение": prose.introduction_paragraphs,
                   "заключение": prose.conclusion_paragraphs}
    style_input.update(prose.section_texts)
    # Everything else the author wrote. The first pass checked only the body and
    # reported clean while the abstract, the headings and the figure captions
    # went unread — each of which prints on the page like any other sentence.
    written_elsewhere = {
        "реферат": [prose.abstract_text],
        "ключевые слова": list(prose.keywords),
        "заголовки": list(prose.section_titles.values()),
        "подписи к рисункам": list(prose.figure_captions.values()),
        "наименование": [prose.research_title, prose.report_title],
    }
    style = slop_check.render(slop_check.check_document(style_input))
    # Internal notation that survived into the prose. Advisory: submit builds
    # the document either way, because a report with a stray identifier beats
    # no report, and the identifier check can collide with real chemistry.
    register = register_check.render(
        register_check.check_document(
            {**style_input, **written_elsewhere}, evidence.nodes
        )
    )

    fits, size = nir_assets.fits(values, bundle.assets)
    if not fits:
        problems.append(
            f"запрос {size // (1024 * 1024)} МБ превышает лимит сервера "
            f"{contract.MAX_REQUEST_BYTES // (1024 * 1024)} МБ"
        )

    # Kept so submit renders exactly what was checked here, rather than
    # rebuilding from prose the agent may have described differently.
    tool_context.state[_STATE_DRAFT] = values

    return {
        "status": "success",
        "problems": problems,
        "style_warnings": style,
        "register_warnings": register,
        "abstract_chars": len(abstract_text or ""),
        "abstract_limit": contract.ABSTRACT_MAX_CHARS,
        "keywords": len(prose.keywords),
        "request_size_mb": round(size / (1024 * 1024), 2),
        "ready": not (problems or register),
    }


async def nir_report_submit(tool_context: ToolContext) -> Dict[str, Any]:
    """Validate the draft on the server, build the DOCX and return a download link.

    Call nir_report_draft first. On a validation failure this returns the
    server's field-level errors — fix them with another nir_report_draft call
    and submit again. On success it returns a permanent link to include in your
    answer, plus any warnings the server raised (placeholders, draft pagination)
    which you must report to the user rather than hide."""
    request = _request(tool_context)
    if not request.get("enabled"):
        return _not_requested()

    values = _state(tool_context).get(_STATE_DRAFT)
    if not isinstance(values, dict):
        return {
            "status": "error",
            "message": "Черновик не найден — сначала вызовите nir_report_draft.",
        }

    report_dir = _report_dir(tool_context)
    figures = sorted((report_dir / "figures").glob("*")) if (report_dir / "figures").is_dir() else []
    bundle = await asyncio.to_thread(nir_assets.build_assets, figures)
    mode = request.get("mode") or get_settings().nir.mode
    page_count = (request.get("requisites") or {}).get("page_count")

    validation = await client.nir_validate(
        values, assets=bundle.assets, mode=mode, page_count=page_count
    )
    if not validation.get("ok"):
        return {
            "status": "invalid",
            "error": validation.get("error"),
            "errors": validation.get("errors") or [],
            "warnings": validation.get("warnings") or [],
            "message": _explain(validation),
        }

    rendered = await client.nir_render(
        values, assets=bundle.assets, mode=mode, page_count=page_count
    )
    if not rendered.get("ok"):
        return {
            "status": "error",
            "error": rendered.get("error"),
            "errors": rendered.get("errors") or [],
            "message": _explain(rendered),
        }

    stored = await asyncio.to_thread(_keep, tool_context, report_dir, rendered)
    result = {
        "status": "success",
        "warnings": [
            w.get("code", str(w)) if isinstance(w, dict) else str(w)
            for w in rendered.get("warnings") or []
        ],
        "mode": mode,
        **stored,
    }
    tool_context.state[hitl_form.STATE_RESULT_KEY] = {
        k: v for k, v in result.items() if k != "status"
    }
    return result


def _explain(payload: Dict[str, Any]) -> str:
    """A sentence the agent can pass to the user, for the failures worth naming."""
    error = payload.get("error")
    if error == contract.ERROR_FONTS_MISSING:
        return (
            "Сервер нормоконтроля развёрнут без шрифтов Times New Roman — "
            "это проблема развёртывания, документ тут ни при чём."
        )
    if error == contract.ERROR_SERVER_BUSY:
        return "Сервер занят другим отчётом; повторные попытки исчерпаны."
    if error == contract.ERROR_STORAGE_FAILED:
        return (
            "Документ собран, но сервер не смог сохранить его в своё хранилище — "
            "на его стороне не настроен или недоступен S3. Содержание отчёта "
            "здесь ни при чём; нужно поправить развёртывание сервера."
        )
    if error == "not_configured":
        return payload.get("message", "Сервер отчётов о НИР не настроен.")
    if error == "unreachable":
        return payload.get("message", "Сервер отчётов о НИР недоступен.")
    if error == contract.ERROR_REQUEST_TOO_LARGE:
        return "Документ вместе с иллюстрациями превысил допустимый размер запроса."
    return payload.get("message") or f"Сервер отказал: {error}"


def _free_path(preferred: Path) -> Path:
    """``preferred``, or a numbered sibling when it cannot be written.

    A previous report left open in Word holds a lock on the file; the write then
    fails, and the run falls back to the server's 24-hour link — losing the
    document to save a filename. Observed the second time this ran, with Word
    still on the first report. Writing beside it keeps the deliverable.
    """
    try:
        if not preferred.exists():
            return preferred
        # exists() is not writability: ask the filesystem.
        with preferred.open("ab"):
            return preferred
    except OSError:
        pass
    for index in range(2, 50):
        candidate = preferred.with_name(f"{preferred.stem}_{index}{preferred.suffix}")
        if not candidate.exists():
            return candidate
    return preferred


def _keep(tool_context: ToolContext, report_dir: Path, rendered: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch the DOCX and give it an address that outlives the server's link.

    The ``output_docx`` the MCP returns is presigned for 24 hours and points at
    *their* bucket, so it is useless as the report's permanent reference. The
    file is downloaded, written beside report.md, uploaded to our own storage,
    and surfaced as ``/api/artifact/<bucket>/<key>`` — a route that mints a
    fresh URL per request, so the link still works from another browser next
    week (``web/app.py``'s artifact route).
    """
    from CoScientist.reporting.collect import _download

    out: Dict[str, Any] = {
        "task_id": rendered.get("task_id"),
        "expires_at": rendered.get("expires_at"),
    }
    url = rendered.get("output_docx")
    files_dir = report_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    local = _free_path(files_dir / DOCX_NAME)

    if not (url and _download(url, local, timeout=120)):
        logger.warning("nir: could not download the rendered DOCX")
        out["download_link"] = url
        out["message"] = (
            "Документ собран, но скачать его на нашу сторону не удалось; "
            "ссылка сервера действует 24 часа."
        )
        return out

    out["local_path"] = str(local)
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    out["sha256"] = digest
    expected = rendered.get("sha256")
    if expected and expected != digest:
        # Worth saying out loud: a mismatch means the bytes on disk are not the
        # document the server validated.
        out["message"] = "Контрольная сумма скачанного файла не совпала с ответом сервера."
        logger.warning("nir: sha256 mismatch for %s", local)

    reference = _store(tool_context, local)
    if reference:
        from CoScientist.utils.report_links import artifact_link

        bucket, key = reference
        out["bucket"], out["s3_key"] = bucket, key
        out["download_link"] = artifact_link(bucket, key)
    else:
        # No S3 anywhere: the file still exists, and a relative path is what the
        # rest of the report folder uses for its own artifacts.
        out["download_link"] = f"files/{DOCX_NAME}"
        out["message"] = (
            "Документ сохранён локально; загрузить его в хранилище не удалось, "
            "поэтому ссылка работает только рядом с папкой отчёта."
        )
    return out


def _store(tool_context: ToolContext, local: Path) -> Optional[tuple]:
    """``(bucket, key)`` for the DOCX: vault first, direct S3 second, else None.

    The vault is preferred because its keys live under ``ephemeral/`` and
    ``finalize_report`` promotes them to ``permanent/``, which is what survives
    the bucket's lifecycle rule. A direct upload has no such promotion, so it is
    the fallback rather than the default.
    """
    user_id, session_id = session_key(tool_context)
    try:
        from CoScientist.tools.vault_client import call_vault_sync, vault_url

        if vault_url():
            # get_upload_link declares user_id/session_id, and SessionScopePlugin
            # fills those only at the ADK tool boundary — this call is framework
            # code and has to pass the scope itself.
            link = call_vault_sync(
                "get_upload_link",
                user_id=user_id,
                session_id=session_id,
                filename=DOCX_NAME,
                feature=_S3_FEATURE,
            )
            if link and link.get("upload_url") and link.get("s3_key"):
                import requests

                response = requests.put(
                    link["upload_url"], data=local.read_bytes(), timeout=120
                )
                response.raise_for_status()
                return link.get("bucket"), link.get("s3_key")
    except Exception as exc:  # noqa: BLE001 - fall through to the direct upload
        logger.warning("nir: vault upload failed (%s)", exc)

    from CoScientist.reporting.s3_upload import upload_and_ref

    return upload_and_ref(local, f"reports/{session_id}/{_S3_FEATURE}")


nir_report_outline_tool = FunctionTool(nir_report_outline)
nir_report_draft_tool = FunctionTool(nir_report_draft)
nir_report_submit_tool = FunctionTool(nir_report_submit)

#: Registered under the "nir_report" tool key (see assembly/bindings.py).
nir_report_tools = [
    nir_report_outline_tool,
    nir_report_draft_tool,
    nir_report_submit_tool,
]


__all__ = [
    "nir_report_outline",
    "nir_report_draft",
    "nir_report_submit",
    "nir_report_tools",
    "DOCX_NAME",
]
