"""The papers a run finds on the open web become files it holds.

Every case here comes from one real session. The ResearchAgent made seven Tavily
calls and none to the papers server; its evidence cited `PMC12610272`, a paper
the run had read and had no copy of. The result envelopes below are the real
shapes: Tavily hands its payload back as ONE JSON string at `content[0].text`,
which is why a structural walk never reaches `results[i].title`.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from CoScientist.reporting import paper_library as pl
from CoScientist.reporting import session_files as sf
from CoScientist.tools.paper_capture_plugin import (
    PaperCapturePlugin,
    is_paper_link,
    papers_in,
    result_items,
)

PDF = b"%PDF-1.7\n" + b"x" * 512


@pytest.fixture()
def key(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")
    return ("u", "s")


@pytest.fixture(autouse=True)
def _no_leaked_lookups():
    """`pmc` remembers a failed lookup for ten minutes, in a process global.

    Without this, one test's failed resolution silences the next test's — which
    is exactly what happened: the dedupe case below passed in file order and
    failed run on its own, because an earlier test had already filled the
    negative cache and the second lookup never left the process.
    """
    from CoScientist.reporting import pmc

    pmc.forget()
    yield
    pmc.forget()


def _tavily(*items):
    """The envelope Tavily actually returns."""
    return {"content": [{"type": "text", "text": json.dumps({
        "query": "furanocoumarins", "results": list(items)})}]}


PMC = {"title": "Heracleum sosnowskyi in the Context of Sustainable Development",
       "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12821576",
       "content": "psoralen and derivatives…", "score": 0.88}
MDPI = {"title": "Furanocoumarins in citrus", "url": "https://www.mdpi.com/2223-7747/15/3/346",
        "content": "…", "score": 0.7}
MINISTRY = {"title": "SKLM opinion on furocoumarins",
            "url": "https://www.dfg.de/resource/blob/169002/sklm-furocoumarine-en-2006.pdf",
            "content": "…", "score": 0.6}
WIKI = {"title": "Psoralen", "url": "https://en.wikipedia.org/wiki/Psoralen",
        "content": "…", "score": 0.5}


# ── what counts as a paper ──────────────────────────────────────────────────
@pytest.mark.parametrize("url", [
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12821576",
    "https://doi.org/10.3390/plants14213253",
    "https://link.springer.com/article/10.1186/s12870-025-07042-3",
    "https://www.mdpi.com/2223-7747/15/3/346",
    "https://arxiv.org/abs/2201.01234",
])
def test_a_publisher_or_index_link_is_a_paper(url):
    assert is_paper_link(url)


@pytest.mark.parametrize("url", [
    # A ministry's report is a document, not a paper — and it stays filed as one.
    "https://www.dfg.de/resource/blob/169002/sklm-furocoumarine-en-2006.pdf",
    "https://en.wikipedia.org/wiki/Psoralen",
    "https://www.thegoodscentscompany.com/data/rw1037841.html",
    "https://github.com/some/repo",
    "",
])
def test_everything_else_is_not(url):
    assert not is_paper_link(url)


# ── reading the envelope ────────────────────────────────────────────────────
def test_the_title_survives_the_json_string():
    """The whole reason `result_items` parses strings on the way down."""
    items = {i["url"]: i for i in result_items(_tavily(PMC, MDPI))}
    assert items[PMC["url"]]["title"].startswith("Heracleum sosnowskyi")
    assert items[MDPI["url"]]["title"] == "Furanocoumarins in citrus"


def test_only_the_papers_are_taken():
    records = papers_in(_tavily(PMC, MDPI, MINISTRY, WIKI), "tavily_search")
    assert {r["pdf_url"] for r in records} == {PMC["url"], MDPI["url"]}
    assert all(r["title"] for r in records), "a paper arrives with its name"
    assert all(r["is_oa"] is None for r in records), (
        "a search result says nothing about the licence — unknown, not assumed")


def test_the_papers_server_still_answers_in_records():
    """The other path is unchanged: that tool already returns a record."""
    records = papers_in({"metadata": {"papers": [{
        "title": "A paper", "doi": "10.1021/acs.jafc.2c05393",
        "publication_year": 2022, "is_oa": True,
        "pdf_url": "https://example.org/p.pdf"}]}}, "search_papers")
    assert records[0]["doi"] == "10.1021/acs.jafc.2c05393"
    assert records[0]["is_oa"] is True


def test_a_result_with_nothing_in_it_is_not_an_error():
    assert papers_in({"content": [{"type": "text", "text": "not json"}]},
                     "tavily_search") == []
    assert papers_in(None, "tavily_search") == []
    assert result_items("plain text") == []


# ── bringing them home ──────────────────────────────────────────────────────
class _Tool:
    def __init__(self, name):
        self.name = name


class _Ctx:
    """Enough of a ToolContext: the plugin only reads state and the scope."""

    def __init__(self, key):
        self.state = {"graph_scope_user_id": key[0],
                      "graph_scope_session_id": key[1]}
        self.agent_name = "ResearchAgent"


class _Response:
    def __init__(self, payload=PDF):
        self.status_code = 200
        self._payload = payload
        self.headers = {"Content-Length": str(len(payload)),
                        "Content-Type": "application/pdf"}

    def iter_content(self, _size):
        yield self._payload

    def close(self):
        pass


def _run(plugin, tool_name, result, ctx):
    return asyncio.run(plugin.after_tool_callback(
        tool=_Tool(tool_name), tool_args={}, tool_context=ctx, result=result))


def test_a_paper_found_on_the_web_is_brought_home(key, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
    ctx = _Ctx(key)

    assert _run(PaperCapturePlugin(), "tavily_search", _tavily(PMC), ctx) is None, (
        "a capture plugin must never decide whether the agent's callbacks run")

    paper = pl.library(ctx.state)[0]
    assert paper["session_artifact_id"], "the bytes are ours now"
    stored = sf.load_manifest(key[1], key[0])[paper["session_artifact_id"]]
    assert stored["source_kind"] == pl.SOURCE_KIND
    assert stored["label"].startswith("Heracleum sosnowskyi")


def test_a_pdf_already_mirrored_is_re_filed_not_re_fetched(key, monkeypatch):
    """`McpArtifactCapturePlugin` runs first and has already taken the bytes.

    A second download would cost another request to someone else's server and
    produce the same content-addressed id, so the copy is simply re-filed.
    """
    import requests

    url = MDPI["url"]
    already = sf.put_bytes(key, PDF, filename="paper.pdf", source_url=url,
                           media_type="application/pdf", source_kind="mcp_url")

    def _refuse(*a, **k):
        raise AssertionError("re-fetched a paper the session already held")

    monkeypatch.setattr(requests, "get", _refuse)
    ctx = _Ctx(key)
    _run(PaperCapturePlugin(), "tavily_search", _tavily(MDPI), ctx)

    stored = sf.load_manifest(key[1], key[0])[already["artifact_id"]]
    assert stored["source_kind"] == pl.SOURCE_KIND, "re-filed as what it is"
    assert pl.library(ctx.state)[0]["session_artifact_id"] == already["artifact_id"]


def test_a_second_search_does_not_fetch_the_same_paper_again(key, monkeypatch):
    """The bytes are bought once.

    Counted over the PAPER's own address, not over every request the capture
    makes: an open-access lookup asks two services about the id, and counting
    those made this assertion depend on whether an earlier test had already
    filled the lookup cache — it passed in file order and failed alone.
    """
    import requests

    fetched = []

    def _get(url, *a, **k):
        if str(url).startswith(PMC["url"]):
            fetched.append(url)
        return _Response()

    monkeypatch.setattr(requests, "get", _get)
    ctx = _Ctx(key)
    _run(PaperCapturePlugin(), "tavily_search", _tavily(PMC), ctx)
    _run(PaperCapturePlugin(), "tavily_search", _tavily(PMC), ctx)
    assert len(fetched) == 1, fetched
    assert len(pl.library(ctx.state)) == 1


def test_a_work_cited_in_the_snippet_is_not_taken_for_the_page_itself(key):
    """The identity of a result is what its ADDRESS says, not what it mentions.

    An abstract cites other works by DOI and by PMC id as a matter of course.
    Read as this page's own handles, they made `merge` fold the cited work into
    this record on the shared key, `_attach_paper` stamp an Evidence with a DOI
    belonging to neither, and the bibliography print the pair. One wrong
    attribution of someone else's work is worse than a hundred left unmatched.
    """
    mdpi = dict(MDPI, content=("Phototoxicity was reported in PMC12610272 and "
                               "the dose data come from doi:10.3390/jox16010006"))
    found = papers_in(_tavily(mdpi, PMC), "tavily_search")

    by_url = {r["pdf_url"]: r for r in found}
    assert by_url[MDPI["url"]]["refs"] == [], "the MDPI page named neither of them"
    assert by_url[MDPI["url"]]["doi"] == ""
    assert by_url[PMC["url"]]["refs"] == ["pmc:12821576"], (
        "and the PMC article keeps the id its own address carries")

    # The two stay two works, and the PMC article is still fetched.
    state = {}
    queued = pl.merge(state, found)
    assert len(pl.library(state)) == 2
    assert len(queued) == 2


def test_the_only_work_a_page_names_is_taken_as_its_own(key):
    """The case the rule must not throw away: a landing page with no identifier
    in its address whose citation header carries exactly one DOI."""
    mdpi = dict(MDPI, content="Plants 2025, 15(3), 346. doi:10.3390/plants15030346")
    record = papers_in(_tavily(mdpi), "tavily_search")[0]
    assert record["refs"] == ["doi:10.3390/plants15030346"]
    assert record["doi"] == "10.3390/plants15030346"


def test_one_work_named_twice_in_a_result_is_queued_once(key, monkeypatch):
    """`merge` returns the STORED record, and several hits reach the same one.

    Queued twice, the same PDF was bought twice — and `_take` mapped the
    outcomes back with `list.index`, which answers with the first match for
    both, so one slot was overwritten and the other stayed empty.
    """
    import requests

    fetched = []

    def _get(url, *a, **k):
        fetched.append(str(url))
        return _Response()

    monkeypatch.setattr(requests, "get", _get)
    same = dict(PMC, url="https://europepmc.org/article/PMC/PMC12821576",
                title="The same work, met at another address")
    ctx = _Ctx(key)
    _run(PaperCapturePlugin(), "tavily_search", _tavily(PMC, same), ctx)

    library = pl.library(ctx.state)
    assert len(library) == 1, "one work"
    assert len([u for u in fetched if "PMC12821576" in u and "articles" in u]) == 1
    assert library[0]["session_artifact_id"], "and its outcome was recorded"


def test_every_paper_gets_its_own_outcome_slot(key, monkeypatch):
    """`list.index` answers with the FIRST match, and it matches by value.

    Two records that look alike then write into one slot and leave the other
    empty, so `note_stored` is told a paper failed when it was fetched.
    """
    import requests

    from CoScientist.tools.paper_capture_plugin import _take

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
    record = {"doi": "10.3390/plants15030346", "doi_raw": "", "title": "Twin",
              "year": 2025, "is_oa": True, "refs": ["doi:10.3390/plants15030346"],
              "pdf_url": "https://example.org/twin.pdf", "presigned_url": "",
              "bucket": None, "s3_key": None, "tool": "tavily_search"}

    outcomes = _take([dict(record), dict(record)], key, "ResearchAgent")

    assert len(outcomes) == 2
    assert all(o.get("artifact_id") for o in outcomes), outcomes


def test_a_tool_we_do_not_watch_is_ignored(key):
    ctx = _Ctx(key)
    assert _run(PaperCapturePlugin(), "execute_bash", _tavily(PMC), ctx) is None
    assert pl.library(ctx.state) == []


def test_a_failure_anywhere_still_returns_none(key, monkeypatch):
    """The contract every capture site holds: never break a tool call."""
    import requests

    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("network")))
    ctx = _Ctx(key)
    assert _run(PaperCapturePlugin(), "tavily_search", _tavily(PMC), ctx) is None
    paper = pl.library(ctx.state)[0]
    assert paper["session_artifact_id"] == "", "not recorded as held"
    assert paper["title"], "but the citation is kept"


def test_the_reading_tools_are_named_once():
    """Two modules ask which tools read the web; they must not drift apart.

    `collect` files what a reading tool brings back as source material; this
    plugin watches the same tools for papers. Two lists would disagree exactly
    when a new search tool was added — silently, and in opposite directions.
    """
    from CoScientist.reporting.collect import WEB_READING_TOOLS

    from CoScientist.tools.paper_capture_plugin import _SEARCH_TOOLS

    assert _SEARCH_TOOLS is WEB_READING_TOOLS
