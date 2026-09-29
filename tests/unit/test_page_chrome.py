"""A publisher's furniture is not an illustration of the run.

Every URL below is real: they come from one session's artifact manifest, where
reading two article pages brought home eighteen viewer icons of 141–720 bytes,
two journal logos, a publisher logo, an avatar placeholder and an author's
photograph — 23 of 36 stored artifacts — all of which would have been offered
to the reader in the report's Figures section.

The seven genuine figures of the paper were in the same manifest, from the same
host. So the cut cannot be by size: a journal banner in that session was 22 007
bytes and a real figure was 22 477. It is by path segment, because that is where
the two actually differ — and `img` and `images` are different segments.
"""
from __future__ import annotations

import pytest

from CoScientist.reporting.collect import _is_page_chrome
from CoScientist.reporting.mirror import _skip

CHROME = [
    # The PMC article viewer's own icons — eighteen of these in one session.
    "https://cdn.ncbi.nlm.nih.gov/pmc/pd-medc-pmc-cloudpmc-viewer/production/"
    "a2b04810/var/data/static/us_flag.svg",
    "https://cdn.ncbi.nlm.nih.gov/pmc/pd-medc-pmc-cloudpmc-viewer/production/"
    "a2b04810/var/data/static/icon-dot-gov.svg",
    "https://cdn.ncbi.nlm.nih.gov/pmc/banners/logo-plants.png",
    "https://pub.mdpi-res.com/img/design/mdpi-pub-logo-black-small1.svg?da3a8dca",
    "https://pub.mdpi-res.com/img/journals/separations-logo.png?640eac0d26d2f4d3",
    "https://pub.mdpi-res.com/bundles/mdpisciprofileslink/img/unknown-user.png",
    "https://www.mdpi.com/profiles/508968/thumb/Antonio_Tagarelli.png",
    "https://pub.mdpi-res.com/img/table.png",
    "https://example.org/assets/favicon.ico",
]

FIGURES = [
    # The paper's own figures, same host, same session, 22 KB – 823 KB.
    "https://pub.mdpi-res.com/separations/separations-12-00175/article_deploy/"
    "html/images/separations-12-00175-g001.png",
    "https://pub.mdpi-res.com/separations/separations-12-00175/article_deploy/"
    "html/images/separations-12-00175-g004-550.jpg",
    "https://www.mdpi.com/separations/separations-12-00175/article_deploy/html/"
    "images/separations-12-00175-sch001.png",
    # And what a run of our own produces.
    "https://minio.internal/bucket/figures/roc_curve.png?X-Amz-Signature=abc",
    "https://minio.internal/bucket/tables/metrics.csv",
]


@pytest.mark.parametrize("url", CHROME)
def test_a_publishers_furniture_is_refused(url):
    assert _is_page_chrome(url)
    assert _skip(url, {}), "and it is never even downloaded"


@pytest.mark.parametrize("url", FIGURES)
def test_what_the_paper_and_the_run_actually_produced_gets_through(url):
    assert not _is_page_chrome(url)
    assert not _skip(url, {})


def test_img_and_images_are_different_segments():
    """The distinction the whole rule rests on."""
    assert _is_page_chrome("https://h/img/table.png")
    assert not _is_page_chrome("https://h/article/html/images/figure-1.png")


def test_a_host_that_keeps_its_furniture_loose_still_has_furniture():
    assert _is_page_chrome("https://h/site-logo.svg")
    assert _is_page_chrome("https://h/a/b/placeholder-figure.png")
    assert _is_page_chrome("https://h/banner.png")


@pytest.mark.parametrize("name", [
    "logotype-analysis.png",      # `logo` is a substring here, not a word
    "phylogenetic-tree.svg",
    "bannerjee-2024-fig2.png",    # an author's name, not a banner
    "iconography-of-plates.png",
])
def test_a_marker_must_be_a_word_not_a_substring(name):
    """A figure whose own name happens to contain one of these must survive.

    Matching on the substring would drop real content to a rule written for a
    site's letterhead — and a lost figure is silent.
    """
    assert not _is_page_chrome(f"https://h/article/images/{name}")


@pytest.mark.parametrize("url", ["", None, "not a url", "https://h"])
def test_nothing_shaped_like_a_url_is_not_an_error(url):
    assert _is_page_chrome(url) is False


# ── read, versus made ───────────────────────────────────────────────────────
class _Tool:
    def __init__(self, name):
        self.name = name


def _mirrored(monkeypatch, tool_name, url, tmp_path):
    """Mirror one URL through the real capture body and return its record."""
    import requests

    from CoScientist.reporting import mirror

    class _Answer:
        status_code = 200
        headers = {"Content-Type": "image/png", "Content-Length": "9"}

        def iter_content(self, _n):
            yield b"\x89PNG\r\n\x1a\n\x00"

        def close(self):
            pass

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Answer())
    records = mirror.mirror_tool_result(
        _Tool(tool_name), None, {"figure": {"artifact": url}}, ("u", "s"))
    return records[0] if records else {}


FIGURE = "https://pub.mdpi-res.com/separations/x/article_deploy/html/images/g001.png"
MADE = "https://minio.internal/bucket/figures/roc_curve.png"


def test_a_figure_from_someone_elses_paper_is_not_an_illustration_of_this_study(
        monkeypatch, tmp_path):
    """The journal's own plate, uncredited, in a document handed to someone else.

    It stays stored and openable — it belongs with the paper — but the report's
    Figures section is for what THIS run produced.
    """
    from CoScientist.reporting.collect import SOURCE_ASSET_KIND, _is_source_material

    record = _mirrored(monkeypatch, "tavily_extract", FIGURE, tmp_path)
    assert record["source_kind"] == SOURCE_ASSET_KIND
    assert record["state"] == "stored", "kept; only its section changes"
    assert _is_source_material(record, FIGURE), "and out of the report"


def test_a_figure_the_run_drew_itself_still_reaches_the_report(monkeypatch, tmp_path):
    """The other half, and the reason the rule is about provenance.

    No rule over hosts or file names separates these two: measured over the
    recorded sessions the publisher's CDN served a 22 007-byte journal banner
    and a 22 477-byte figure of the paper. What separates them is which tool
    brought them back.
    """
    from CoScientist.reporting.collect import _is_source_material

    record = _mirrored(monkeypatch, "chemical_space_tsne", MADE, tmp_path)
    assert record["source_kind"] == "mcp_url"
    assert not _is_source_material(record, MADE)


def _report_of(monkeypatch, tmp_path, tool_name, result):
    """Drive the real path end to end: mirror → the durable index → the report."""
    import requests

    from CoScientist.reporting import collect, mirror
    from CoScientist.tools.mcp_artifact_plugin import McpArtifactCapturePlugin

    fetched = []

    class _Answer:
        status_code = 200
        headers = {"Content-Type": "image/png", "Content-Length": "72"}
        content = b"\x89PNG\r\n\x1a\n" + b"z" * 64

        def iter_content(self, _n):
            yield self.content

        def close(self):
            pass

    def _get(url, *a, **k):
        fetched.append(url)
        return _Answer()

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")
    monkeypatch.setattr(requests, "get", _get)

    ctx = type("Ctx", (), {"state": {"graph_scope_user_id": "u",
                                     "graph_scope_session_id": "s"}})()
    tool = _Tool(tool_name)
    records = mirror.mirror_tool_result(tool, ctx, result, ("u", "s"))
    McpArtifactCapturePlugin._record_index(
        tool, ctx, collect.find_artifact_urls(result), [], records)
    fetched.clear()
    out = collect.collect_artifacts("s", state=ctx.state, index_key=("u", "s"))
    import os
    return {
        "figures": [os.path.basename(p) for p in (out.get("figures") or [])],
        "fetched": [u.rsplit("/", 1)[-1].split("?")[0] for u in fetched],
    }


def test_refusing_the_download_must_not_let_chrome_into_the_report(
        monkeypatch, tmp_path):
    """The inversion this rule created on its first attempt.

    Refusing chrome at MIRROR time is not enough, and was worse than nothing: no
    mirror record means no entry in `mirrored_by_url`, and that entry is exactly
    what stops the report's URL pass fetching the file again. So the icons were
    declined once, downloaded a second time from the publisher during
    collection, and printed as illustrations of the run — while the paper's real
    figures, which did mirror, were excluded as source material. Both outcomes
    the opposite of the intent.
    """
    logo = "https://cdn.ncbi.nlm.nih.gov/pmc/banners/logo-plants.png"
    page = {"content": [{"type": "text", "text": (
        '{"results":[{"url":"https://www.mdpi.com/a","raw_content":'
        f'"text ![logo]({logo}) and ![fig]({FIGURE})"}}]}}')}]}

    out = _report_of(monkeypatch, tmp_path, "tavily_extract", page)

    assert out["fetched"] == [], "nothing is re-downloaded at report time"
    assert "logo-plants.png" not in " ".join(out["figures"])
    assert not any("g001" in f for f in out["figures"]), (
        "the journal's plate is not an illustration of this study either")


def test_an_index_written_before_this_rule_still_produces_a_clean_report(
        monkeypatch, tmp_path):
    """Every session already on disk carries the furniture in its index.

    Across the recorded sessions 70 of 219 index rows are a publisher's
    letterhead, written before any of this existed and carrying no `source_kind`
    at all. Filtering only where the rows are WRITTEN would leave every one of
    those reports to be assembled with icons in its illustrations — so the
    collector judges each row as it reads it.
    """
    import os

    import requests

    from CoScientist.reporting import collect
    from CoScientist.reporting.artifact_index import record

    fetched = []

    class _Answer:
        status_code = 200
        headers = {"Content-Type": "image/png"}
        content = b"\x89PNG\r\n\x1a\n" + b"z" * 64

        def iter_content(self, _n):
            yield self.content

        def raise_for_status(self):
            """`collect._download` calls this; a fake without it fails silently
            and the test then 'passes' for the wrong reason."""

        def close(self):
            pass

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(requests, "get",
                        lambda url, *a, **k: (fetched.append(url), _Answer())[1])

    # Exactly the shape of a real pre-change row: no source_kind, no artifact_id.
    record([{"url": "https://cdn.ncbi.nlm.nih.gov/pmc/banners/logo-plants.png",
             "tool": "tavily_extract", "label": "artifact",
             "bucket": None, "s3_key": None},
            {"url": "https://minio.internal/b/plots/roc_curve.png",
             "tool": "predict_ld50", "label": "artifact",
             "bucket": None, "s3_key": None}],
           user_id="u", session_id="s")

    out = collect.collect_artifacts("s", state={}, index_key=("u", "s"))
    figures = [os.path.basename(p) for p in (out.get("figures") or [])]

    assert not any("logo" in f for f in figures), figures
    assert any("roc_curve" in f for f in figures), (
        "and the run's own plot is still collected")
    assert not any("logo" in u for u in fetched), "not even fetched"


def test_a_figure_the_run_drew_still_reaches_the_report(monkeypatch, tmp_path):
    """The regression guard on the ordinary path — the one that must not break."""
    drawn = ("https://minio.internal/bucket/plots/tsne_chemical_space.png"
             "?X-Amz-Signature=ab")
    out = _report_of(monkeypatch, tmp_path, "chemical_space_tsne",
                     {"metadata": {"figure": {"artifact": drawn}}})
    assert out["figures"] == ["tsne_chemical_space.png"]


def test_a_tool_that_reads_is_told_from_a_tool_that_draws():
    """The discriminator itself, over the two sets it has to separate."""
    from CoScientist.reporting.collect import is_reading_tool

    assert is_reading_tool("tavily_extract")
    assert not is_reading_tool("chemical_space_tsne")
    assert not is_reading_tool(None)


def test_the_state_list_the_plugin_writes_is_filtered_too(monkeypatch, tmp_path):
    """The SECOND door into the report, and the one the filter missed.

    `mcp_artifacts` in session state is swept by the collector independently of
    the durable index. The guard was written for it and landed after the early
    `return`, where it could never run — so the index beside it was clean and
    this list went out carrying the publisher's letterhead.
    """
    import asyncio

    import requests

    from CoScientist.tools.mcp_artifact_plugin import McpArtifactCapturePlugin

    class _Answer:
        status_code = 200
        headers = {"Content-Type": "image/png", "Content-Length": "72"}

        def iter_content(self, _n):
            yield b"\x89PNG\r\n\x1a\n" + b"z" * 64

        def close(self):
            pass

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Answer())

    logo = "https://pub.mdpi-res.com/img/journals/separations-logo.png"
    figure = ("https://pub.mdpi-res.com/separations/separations-12-00175/"
              "article_deploy/html/images/separations-12-00175-g001.png")
    ctx = type("Ctx", (), {"state": {"graph_scope_user_id": "u",
                                     "graph_scope_session_id": "s"}})()

    asyncio.run(McpArtifactCapturePlugin().after_tool_callback(
        tool=_Tool("tavily_extract"), tool_args={}, tool_context=ctx,
        result={"images": [logo, figure]}))

    kept = [a.get("url") for a in ctx.state.get("mcp_artifacts") or []]
    assert figure in kept, "the paper's own figure is recorded"
    assert logo not in kept, "the journal's letterhead is not"
