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

    def __str__(self) -> str:  # so legacy `print(result)` / str() still reads well
        return self.markdown


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
        (report_dir / "report.md").write_text(final_markdown or "", encoding="utf-8")

        references = _extract_references(state or {})
        latex_files = render_latex(
            final_markdown or "", report_dir, report_config.latex, references
        )
        promoted = _promote_sources(report_dir)
        manifest = _build_manifest(
            session_id, report_dir, report_config, latex_files, promoted
        )
        (report_dir / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info("report: wrote deliverable to %s (latex=%s)", report_dir, report_config.latex)
        return RunResult(markdown=final_markdown, report_dir=report_dir, manifest=manifest)
    except Exception as exc:  # never let report packaging sink a completed run
        logger.error("report: failed to finalize %s (%s)", report_dir, exc)
        return RunResult(markdown=final_markdown, report_dir=None, manifest=None)


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
) -> Dict[str, Any]:
    def listing(subdir: str) -> List[str]:
        d = report_dir / subdir
        if not d.exists():
            return []
        return sorted(str(p.relative_to(report_dir)) for p in d.rglob("*") if p.is_file())

    return {
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report": "report.md" if (report_dir / "report.md").exists() else None,
        "figures": listing("figures"),
        "tables": listing("tables"),
        "sections": listing("sections"),
        # Report-relative path -> the permanent/ key that outlives the run. A
        # file with no entry here exists only inside this folder.
        "promoted": promoted or {},
        "latex": {
            "mode": report_config.latex,
            "files": sorted(str(p.relative_to(report_dir)) for p in latex_files),
        },
    }


def _extract_references(state: Dict[str, Any]) -> List[str]:
    """Best-effort structured references from session state.

    TODO(bibliography): paper-research currently stores results as free text
    (``search_results``), so there is no reliable citation metadata to build a
    real ``references.bib`` from. When paper-research is changed to retain raw
    paper/citation records in state (e.g. a ``references`` list of dicts), read
    them here. Until then this returns whatever plain-string references are
    already present and otherwise nothing.
    """
    refs = state.get("references")
    if isinstance(refs, list):
        out: List[str] = []
        for r in refs:
            if isinstance(r, str):
                out.append(r)
            elif isinstance(r, dict):
                out.append(r.get("citation") or r.get("title") or json.dumps(r))
        return out
    return []


__all__ = ["finalize_report", "RunResult"]
