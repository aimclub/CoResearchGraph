"""Finalize a run into the report folder deliverable.

Runs in the manager driver AFTER the aggregator agent has produced its narrative
markdown. Writes ``report.md``, renders LaTeX per :class:`ReportConfig`, and
writes ``MANIFEST.json`` by scanning what actually landed on disk.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from CoScientist.config.report import ReportConfig
from CoScientist.reporting.collect import SOURCES_FILENAME, report_dir_for
from CoScientist.reporting.latex import render_latex

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """What :meth:`CoScientistManager.run` returns.

    ``markdown`` is the assembled report text (for a chat bubble / stdout).
    ``report_dir`` is the on-disk folder deliverable. ``manifest`` lists its
    contents. ``report_dir``/``manifest`` are ``None`` if no folder was written.
    """

    markdown: str
    report_dir: Optional[Path] = None
    manifest: Optional[Dict[str, Any]] = None
    #: The session artifact holding `report.md`, when one was stored. This is
    #: what the chat's document panel opens; the graph's Report node carries
    #: the same id.
    report_artifact_id: Optional[str] = None

    def __str__(self) -> str:  # so legacy `print(result)` / str() still reads well
        return self.markdown


def _scope_of(session_id: str, state: Dict[str, Any]) -> tuple[str, str]:
    """``(user_id, session_id)`` from ADK state, falling back to the run's own id."""
    from CoScientist.graph.session_scope import (
        GRAPH_SCOPE_SESSION_KEY,
        GRAPH_SCOPE_USER_KEY,
    )

    return (
        str(state.get(GRAPH_SCOPE_USER_KEY) or ""),
        str(state.get(GRAPH_SCOPE_SESSION_KEY) or session_id),
    )


def _readable(markdown: str, session_id: str, state: Dict[str, Any]) -> str:
    """Report text with every stored reference turned into a working link.

    Never raises: the deliverable outranks any one link.
    """
    try:
        from CoScientist.utils.report_links import (
            remint_report_urls,
            resolve_artifact_refs,
        )

        scope = _scope_of(session_id, state)
        return remint_report_urls(resolve_artifact_refs(markdown, scope), scope)
    except Exception as exc:  # noqa: BLE001
        logger.warning("report: could not resolve links (%s)", exc)
        return markdown


def _publish_to_research_graph(
    session_id: str, markdown: str, state: Dict[str, Any], path: Path
) -> Optional[str]:
    """Give the write-up a card in the research graph, and a file beside it.

    The graph used to end on a one-line derived summary while the document it
    summarised sat on disk, referenced from nowhere. Best-effort: the report is
    already written by the time this runs.
    """
    try:
        from CoScientist.graph.session_scope import (
            GRAPH_SCOPE_SESSION_KEY,
            GRAPH_SCOPE_USER_KEY,
        )

        user_id = str(state.get(GRAPH_SCOPE_USER_KEY) or "")
        scoped_session = str(state.get(GRAPH_SCOPE_SESSION_KEY) or session_id)
        if not user_id:
            logger.info("report node: no user scope in state; skipping")
            return None

        artifact_id = None
        try:
            from CoScientist.reporting.mirror import mirror_artifact

            record = mirror_artifact(
                user_id=user_id, session_id=scoped_session, path=path,
                filename="report.md", label="Отчёт по исследованию",
                tool="format_results", source_kind="report",
            )
            artifact_id = record.get("artifact_id")
        except Exception as exc:  # noqa: BLE001
            logger.warning("report node: could not mirror report.md (%s)", exc)

        from CoScientist.reporting.report_node import publish_report_node

        publish_report_node(user_id, scoped_session, markdown, artifact_id=artifact_id)
        return artifact_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("report node: publish step failed (%s)", exc)
    return None


def finalize_report(
    session_id: str,
    final_markdown: str,
    report_config: ReportConfig,
    state: Optional[Dict[str, Any]] = None,
) -> RunResult:
    """Write report.md + LaTeX + MANIFEST.json; return a :class:`RunResult`."""
    report_dir = report_dir_for(session_id, report_config.reports_root)
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        # `report.md` is read by people and by LaTeX, neither of which knows the
        # `cos-artifact:` scheme. The graph card keeps the portable form (see
        # publish_report_node); the file on disk gets URLs.
        markdown = _readable(final_markdown or "", session_id, state or {})
        (report_dir / "report.md").write_text(markdown, encoding="utf-8")

        references = _extract_references(state or {})
        latex_files = render_latex(
            markdown, report_dir, report_config.latex, references
        )
        nir = _record_nir(report_dir, state or {})
        report_artifact_id = _publish_to_research_graph(
            session_id, final_markdown or "", state or {}, report_dir / "report.md"
        )
        promoted = _promote_sources(report_dir)
        manifest = _build_manifest(
            session_id, report_dir, report_config, latex_files, promoted, nir
        )
        (report_dir / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info("report: wrote deliverable to %s (latex=%s)", report_dir, report_config.latex)
        return RunResult(markdown=final_markdown, report_dir=report_dir,
                         manifest=manifest, report_artifact_id=report_artifact_id)
    except Exception as exc:  # never let report packaging sink a completed run
        logger.error("report: failed to finalize %s (%s)", report_dir, exc)
        return RunResult(markdown=final_markdown, report_dir=None, manifest=None)


def _record_nir(report_dir: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    """Fold the NIR DOCX into the artifact sources, before promotion runs.

    The tool that built the document cannot write this file itself:
    ``collect_artifacts`` rewrites ``SOURCES_FILENAME`` wholesale, so a second
    ``format_results`` call — which the aggregator is free to make — would erase
    the entry. Here there is exactly one writer and a fixed order, so the key
    reaches ``_promote_sources`` and moves from ``ephemeral/`` to ``permanent/``
    like every other artifact the report shows.

    Returns the manifest block, or {} when the run produced no NIR report.
    """
    nir = state.get("nir_report")
    if not isinstance(nir, dict):
        return {}

    local = nir.get("local_path")
    bucket, key = nir.get("bucket"), nir.get("s3_key")
    if local and bucket and key:
        sources_path = report_dir / SOURCES_FILENAME
        sources: Dict[str, Any] = {}
        if sources_path.exists():
            try:
                loaded = json.loads(sources_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    sources = loaded
            except Exception as exc:  # noqa: BLE001
                logger.warning("report: cannot merge into %s (%s)", sources_path, exc)
        try:
            relative = str(Path(local).relative_to(report_dir)).replace("\\", "/")
        except ValueError:
            relative = f"files/{Path(local).name}"
        sources[relative] = {"bucket": bucket, "s3_key": key}
        try:
            sources_path.write_text(json.dumps(sources, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - promotion is not the report
            logger.warning("report: cannot write %s (%s)", sources_path, exc)

    return {
        k: nir.get(k)
        for k in ("task_id", "bucket", "s3_key", "local_path", "sha256",
                  "download_link", "expires_at", "mode", "warnings")
        if nir.get(k) is not None
    }


def _promote_sources(report_dir: Path) -> Dict[str, str]:
    """Copy every collected artifact out of ``ephemeral/`` and into ``permanent/``.

    A worker only ever uploads under ``ephemeral/``, where the bucket lifecycle
    rule deletes it after EPHEMERAL_TTL_DAYS. The report outlives that, so the
    objects it shows have to move. ``collect_artifacts`` left the mapping from
    each local file to its object in ``SOURCES_FILENAME``.

    Only objects that S3 already holds are promoted. ``report.md``, the LaTeX
    output, and the files a local sandbox left on this disk were never uploaded,
    and a worker may not write to ``permanent/`` directly.

    Returns report-relative path -> new ``permanent/`` key. Empty on any failure:
    a vault that is down costs the deliverable its durability, not its existence.
    """
    sources_path = report_dir / SOURCES_FILENAME
    if not sources_path.exists():
        return {}
    try:
        sources = json.loads(sources_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report: cannot read %s (%s)", sources_path, exc)
        return {}
    # A stale or hand-edited file can hold any shape. Check it here: an escape
    # from this function lands in the caller's except, which reports the whole
    # deliverable as missing while report.md sits complete on disk.
    if not isinstance(sources, dict):
        logger.warning("report: %s is not a mapping, skipping promotion", sources_path)
        return {}

    from CoScientist.tools.vault_client import call_vault_sync, vault_url

    if not vault_url():
        logger.info("report: MCP__VAULT_URL is not set, artifacts stay ephemeral")
        return {}

    promoted: Dict[str, str] = {}
    for rel_path, ref in sorted(sources.items()):
        key = ref.get("s3_key") if isinstance(ref, dict) else None
        if not isinstance(key, str) or not key.startswith("ephemeral/"):
            continue
        result = call_vault_sync("promote_artifact", s3_key=key)
        new_key = (result or {}).get("s3_key")
        if new_key:
            promoted[rel_path] = new_key
        else:
            logger.warning("report: could not promote %s for %s", key, rel_path)

    logger.info("report: promoted %d of %d artifact(s)", len(promoted), len(sources))
    return promoted


def _build_manifest(
    session_id: str,
    report_dir: Path,
    report_config: ReportConfig,
    latex_files: List[Path],
    promoted: Optional[Dict[str, str]] = None,
    nir: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    def listing(subdir: str) -> List[str]:
        d = report_dir / subdir
        if not d.exists():
            return []
        return sorted(str(p.relative_to(report_dir)) for p in d.rglob("*") if p.is_file())

    manifest: Dict[str, Any] = {
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report": "report.md" if (report_dir / "report.md").exists() else None,
        "figures": listing("figures"),
        "tables": listing("tables"),
        "files": listing("files"),
        "sections": listing("sections"),
        # Report-relative path -> the permanent/ key that outlives the run. A
        # file with no entry here exists only inside this folder.
        "promoted": promoted or {},
        "latex": {
            "mode": report_config.latex,
            "files": sorted(str(p.relative_to(report_dir)) for p in latex_files),
        },
    }
    # Absent unless the operator asked for a GOST report, so a manifest from a
    # run without one is byte-identical to what this wrote before.
    if nir:
        manifest["nir"] = nir
    return manifest


def _extract_references(state: Dict[str, Any]) -> List[str]:
    """Structured references from session state.

    An explicit ``references`` list wins: something took the trouble to write
    it. Failing that, the papers the run actually looked up — title, year and
    DOI per record, kept by ``PaperCapturePlugin``. This is the metadata
    the long-standing TODO here was waiting for: paper research used to keep its
    results as free text (``search_results``), so there was nothing to build a
    bibliography from.
    """
    refs = state.get("references")
    if isinstance(refs, list) and refs:
        out: List[str] = []
        for r in refs:
            if isinstance(r, str):
                out.append(r)
            elif isinstance(r, dict):
                out.append(r.get("citation") or r.get("title") or json.dumps(r))
        return out
    try:
        from CoScientist.reporting.paper_library import references

        return references(state)
    except Exception:  # noqa: BLE001 — a bibliography must not sink a report
        return []


__all__ = ["finalize_report", "RunResult"]
