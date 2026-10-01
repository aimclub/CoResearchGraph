"""The report's artifact sections list what the run PRODUCED, not its paperwork.

The chat's document panel (reporting/documents.py) writes work orders, agents'
reports and answers, the research frame and the plan review into the same
session store the run's outputs live in — and finalize mirrors report.md there
as well. Collected as artifacts, they filled the Files section of a live report
(session 4638232b, 2026-09-24): fifteen entries, fourteen of them paperwork, one
of them a paper the run found.

Run from the repo root:  pytest tests/unit/test_report_skips_paperwork.py -q
"""
from __future__ import annotations

import pytest

from CoScientist.config import get_settings
from CoScientist.reporting import collect, s3_upload
from CoScientist.reporting import session_files as sf


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path / "g"))
    monkeypatch.setattr(get_settings().s3, "use_s3", False)
    s3_upload._reset_for_tests()
    key = ("u1", "s1")
    # What the run produced.
    sf.put_bytes(key, b"\x89PNG\r\n\x1a\n" + b"p" * 32, filename="fig_ld50.png",
                 source_tool="predict_ld50", source_kind="mcp_url")
    sf.put_bytes(key, b"%PDF-1.4 a paper", filename="paper.pdf",
                 source_tool="search_papers", source_kind="mcp_url")
    # What the run wrote about itself.
    for n, kind in enumerate(("doc:work_order", "doc:work_report", "doc:answer",
                              "doc:frame", "doc:review", "node_report")):
        sf.put_bytes(key, f"# paperwork {n}".encode(), filename=f"{kind.split(':')[-1]}.md",
                     source_tool="ResearchAgent", source_kind=kind)
    sf.put_bytes(key, b"# the report itself", filename="report.md",
                 source_tool="format_results", source_kind="report")
    yield key
    s3_upload._reset_for_tests()


def _collect(tmp_path, key):
    return collect.collect_artifacts(
        "s1", reports_root=tmp_path / "reports", workspace_root=tmp_path / "nowhere",
        index_key=key)


def test_only_what_the_run_produced_is_listed(tmp_path, store):
    out = _collect(tmp_path, store)
    assert out["block_counts"] == {"figures": 1, "tables": 0, "files": 1}
    md = out["blocks_markdown"]
    assert "fig_ld50.png" in md and "paper.pdf" in md
    for name in ("work_order.md", "work_report.md", "answer.md", "frame.md",
                 "review.md", "node_report.md", "report.md"):
        assert name not in md, name


def test_paperwork_is_not_reported_as_missing_either(tmp_path, store, monkeypatch):
    """A document that failed to store is not a result the reader was owed."""
    sf.note(store, state=sf.STATE_FAILED, reason=sf.REASON_UNREADABLE,
            filename="work-report.md", source_kind="doc:work_report")
    _collect(tmp_path, store)
    skipped = tmp_path / "reports" / "s1" / "sections" / "skipped.md"
    assert not skipped.exists() or "work-report.md" not in skipped.read_text(encoding="utf-8")


def test_every_document_kind_is_recognised():
    """The exclusion follows the documents module, so a new kind needs no edit here."""
    from CoScientist.reporting.documents import KIND_PREFIX, UNLISTED_KINDS, _FILENAMES

    for kind in _FILENAMES:
        source_kind = kind if kind in UNLISTED_KINDS else KIND_PREFIX + kind
        assert collect._is_session_paperwork({"source_kind": source_kind}), kind
    assert not collect._is_session_paperwork({"source_kind": "mcp_url"})
    assert not collect._is_session_paperwork({"source_kind": "experiment"})
    assert not collect._is_session_paperwork({})
