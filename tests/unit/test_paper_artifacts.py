"""A found paper becomes a file, and the evidence that cites it can open it.

The failure these cover is one a live session showed: the ResearchAgent found a
relevant paper, and everything downstream — the markdown document on the right
of the chat, the Evidence node in the research graph — carried one thing, the
DOI, as a bare string. No file, no link, nothing to click.

Two halves. `paper_library` keeps the citation (title, year, DOI, address) that
`print_research_agent_tool_call` used to throw away, keeping only an S3 key;
`PaperCapturePlugin` brings the bytes into the session store. The join between
them is any identifier the record holds — a DOI, a PMC id — and a DOI is
**normalized** first, because three sources spell the same one three ways.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from CoScientist.reporting import paper_library as pl
from CoScientist.reporting import session_files as sf

PDF = b"%PDF-1.7\n" + b"x" * 512


@pytest.fixture()
def key(tmp_path, monkeypatch):
    """A session whose storage root is short.

    Deliberately not a deep pytest tmp dir: the artifact path runs
    ``<root>/sessions/<user>/<session>/artifacts/files/<id>`` and on Windows a
    long root pushes it past 260 characters, silently.
    """
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")
    # `collect_artifacts` writes the report folder under REPORTS_ROOT, which
    # defaults into the checkout. Left alone it both pollutes the repo and makes
    # the test depend on what an earlier run left behind: a second `scores.csv`
    # comes back as `scores-<id>.csv`, because `_unique_dest` disambiguates.
    monkeypatch.setenv("REPORTS_ROOT", str(tmp_path / "reports"))
    return ("u", "s")


class _Response:
    """Enough of `requests.Response` for `mirror._fetch`."""

    def __init__(self, payload=PDF, status=200, content_type="application/pdf"):
        self.status_code = status
        self._payload = payload
        self.headers = {"Content-Length": str(len(payload)),
                        "Content-Type": content_type}

    def iter_content(self, _size):
        yield self._payload

    def close(self):
        pass


def _search_result(**over):
    """What `search_papers` hands back — the tool that does NOT download."""
    paper = {"title": "Furanocoumarin toxicity in citrus",
             "doi": "https://doi.org/10.1021/acs.jafc.2c05393",
             "publication_year": 2022, "cited_by_count": 14, "is_oa": True,
             "pdf_url": "https://example.org/papers/furano.pdf"}
    paper.update(over)
    return {"answer": "Found papers:\n- Furanocoumarin…",
            "metadata": {"papers": [paper]}}


# ── the DOI, in every spelling it arrives in ────────────────────────────────
@pytest.mark.parametrize("written", [
    "10.1021/acs.jafc.2c05393",
    "10.1021/ACS.JAFC.2C05393",
    "doi:10.1021/acs.jafc.2c05393",
    "https://doi.org/10.1021/acs.jafc.2c05393",
    "http://dx.doi.org/10.1021/acs.jafc.2c05393",
    "  https://doi.org/10.1021/acs.jafc.2c05393.  ",
])
def test_one_paper_however_its_doi_was_written(written):
    """A DOI is case-insensitive by specification and prefixed by habit.

    Matching the raw string means an Evidence whose `source_ref` says
    `doi:10.1021/X` never finds the paper stored under `https://doi.org/10.1021/x`
    — the join silently misses and the file stays unreachable.
    """
    assert pl.normalize_doi(written) == "10.1021/acs.jafc.2c05393"


@pytest.mark.parametrize("junk", ["", None, "see the appendix", "10.bad/x", "arXiv:2201.1"])
def test_what_is_not_a_doi_is_not_forced_into_one(junk):
    assert pl.normalize_doi(junk) == ""


# ── the bytes ───────────────────────────────────────────────────────────────
def test_a_found_paper_is_brought_home(key, monkeypatch):
    """`search_papers` downloads nothing; this is what fixes that."""
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
    state = {}
    records = pl.records_from_tool("search_papers", _search_result())
    wanted = pl.merge(state, records, agent="ResearchAgent")
    assert len(wanted) == 1

    from CoScientist.tools.paper_capture_plugin import mirror_papers

    outcome = mirror_papers(wanted, key, "ResearchAgent")[0]
    pl.note_stored(state, wanted[0], outcome)

    assert outcome["state"] == sf.STATE_STORED
    stored = sf.load_manifest(key[1], key[0])[outcome["artifact_id"]]
    assert stored["source_kind"] == pl.SOURCE_KIND
    assert stored["media_type"] == "application/pdf"
    # The name a reader sees in the download, not a hash or a URL tail.
    assert stored["filename"].startswith("furanocoumarin-toxicity-in-citrus")
    assert pl.library(state)[0]["session_artifact_id"] == outcome["artifact_id"]


def test_a_paper_with_no_copy_to_fetch_keeps_its_citation(key):
    """Not every result has an address. Losing the bytes must not lose the paper.

    The record still carries title, year and DOI, so the bibliography and the
    graph have everything except the file — which is the honest outcome.
    """
    state = {}
    records = pl.records_from_tool(
        "search_papers", _search_result(pdf_url="", is_oa=False))
    assert pl.merge(state, records) == [], "nothing to fetch"
    kept = pl.library(state)[0]
    assert kept["doi"] == "10.1021/acs.jafc.2c05393"
    assert kept["title"] and kept["year"] == 2022
    assert kept["is_oa"] is False, "the licence answer is recorded, not assumed"
    assert kept["session_artifact_id"] == ""


def test_the_same_paper_is_not_fetched_twice(key, monkeypatch):
    """A repeated search must not re-download what the session already holds."""
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
    from CoScientist.tools.paper_capture_plugin import mirror_papers

    state = {}
    wanted = pl.merge(state, pl.records_from_tool("search_papers", _search_result()))
    pl.note_stored(state, wanted[0], mirror_papers(wanted, key, "a")[0])

    again = pl.merge(state, pl.records_from_tool("search_papers", _search_result()))
    assert again == [], "already stored — nothing more to do"
    assert len(pl.library(state)) == 1, "and still one paper, not two"


def test_the_two_tools_describe_one_paper_between_them(key):
    """`search_papers` knows the title and year; the download tool knows where
    the bytes are. They arrive in either order and must not become two records."""
    state = {}
    pl.merge(state, pl.records_from_tool("search_papers", _search_result(pdf_url="")))
    pl.merge(state, pl.records_from_tool("download_papers_from_search", {
        "metadata": {"papers": [{
            "paper_title": "Furanocoumarin toxicity in citrus",
            "doi": "10.1021/ACS.JAFC.2C05393", "publication_year": 2022,
            "bucket": "b", "s3_key": "ephemeral/u/s/papers_search_results/f.pdf",
            "presigned_url": "https://minio/b/f.pdf?sig=1"}]}}))
    assert len(pl.library(state)) == 1
    paper = pl.library(state)[0]
    assert paper["s3_key"].endswith("f.pdf")
    assert paper["title"] == "Furanocoumarin toxicity in citrus"
    # The expiring link is what we fetch from; it is never the paper's address.
    assert paper["presigned_url"].endswith("sig=1")
    assert paper["pdf_url"] == ""


# ── still not a deliverable ─────────────────────────────────────────────────
def test_the_rule_recognises_a_paper_by_what_it_was_filed_as():
    """`_SOURCE_KEY_MARKERS` recognised the papers server's own upload prefix.

    A copy the session mirrors is content-addressed and has no prefix left, so
    the rule matches on `source_kind` too. The exclusion gets stronger here, not
    weaker — that rationale is written at collect.py:45.
    """
    from CoScientist.reporting.collect import _is_source_material

    assert _is_source_material({"source_kind": "paper"}, "https://h/furano.pdf")
    assert _is_source_material({}, "s3://b/ephemeral/u/s/papers_search_results/f.pdf")
    # A PDF the run itself produced is still a result.
    assert not _is_source_material({"source_kind": "experiment"}, "https://h/plot.pdf")


def test_a_mirrored_paper_is_still_not_a_deliverable(key):
    """The report's Files section lists what the run produced, not what it read.

    The rule above is necessary and was not sufficient: `collect_artifacts`
    makes two passes, and only the URL pass ever consulted it. The session-store
    pass copied everything it found — which, once papers were deliberately
    mirrored into that store, put every downloaded paper straight into the
    report. Asserting the predicate alone missed it; this asserts the outcome.
    """
    from CoScientist.reporting.collect import collect_artifacts

    sf.put_bytes(key, PDF, filename="furano.pdf", media_type="application/pdf",
                 source_kind=pl.SOURCE_KIND, label="Furanocoumarin toxicity")
    sf.put_bytes(key, b"id,score\n1,0.9\n", filename="scores.csv",
                 media_type="text/csv", source_kind="experiment")

    collected = collect_artifacts(key[1], state={}, index_key=key)

    names = [Path(p).name for p in (collected.get("files") or [])
             + (collected.get("tables") or []) + (collected.get("figures") or [])]
    assert "furano.pdf" not in names, "a paper is a source, not a result"
    assert "scores.csv" in names, "and the run's own output still gets through"


def test_a_paper_that_did_not_download_is_not_recorded_as_held(key):
    """`session_files.note` mints an id for a failure too, so that the record
    still dedupes and has somewhere to be listed. Taking that id would mark a
    paper we do not hold as held: never retried, and stamped onto Evidence as
    an artifact that resolves to nothing."""
    state = {}
    records = pl.records_from_tool("search_papers", _search_result())
    wanted = pl.merge(state, records)
    failed = sf.note(key, state=sf.STATE_FAILED, reason=sf.REASON_LINK_EXPIRED,
                     filename="furano.pdf")
    assert failed["artifact_id"], "the failure does carry an id"

    pl.note_stored(state, wanted[0], failed)

    paper = pl.library(state)[0]
    assert paper["session_artifact_id"] == ""
    assert paper["artifact_state"] == sf.STATE_FAILED
    assert paper["artifact_reason"] == sf.REASON_LINK_EXPIRED
    # And it is offered again the next time the agent searches.
    assert pl.merge(state, pl.records_from_tool("search_papers", _search_result()))


def test_a_paper_stays_in_the_session(key, monkeypatch):
    """`mirror_off_host=False`: the off-host copy would outlive the run that
    justified fetching it, and retention here is a policy question."""
    import requests

    from CoScientist.reporting import mirror

    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "True")
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
    called = []
    monkeypatch.setattr(mirror, "_also_to_s3", lambda *a: called.append(a))

    mirror.mirror_artifact(None, user_id=key[0], session_id=key[1],
                           url="https://example.org/p.pdf", filename="p.pdf",
                           source_kind="paper", mirror_off_host=False)
    assert called == []
    mirror.mirror_artifact(None, user_id=key[0], session_id=key[1],
                           url="https://example.org/q.pdf", filename="q.pdf",
                           source_kind="experiment")
    assert len(called) == 1, "everything else still gets its second address"


# ── the join into the research graph ────────────────────────────────────────
def test_evidence_is_given_the_paper_it_cited(key, monkeypatch):
    """The agent writes the DOI it read and cannot know we hold the file.

    Joining them at commit time is what turns the DOI in the panel from a string
    into a paper with a download beside it.
    """
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
    from CoScientist.tools.paper_capture_plugin import mirror_papers
    from CoScientist.graph.research.agent_tools import _enrich_evidence

    state = {}
    wanted = pl.merge(state, pl.records_from_tool("search_papers", _search_result()))
    outcome = mirror_papers(wanted, key, "ResearchAgent")[0]
    pl.note_stored(state, wanted[0], outcome)

    ctx = type("Ctx", (), {"state": state, "agent_name": "ResearchAgent"})()
    nodes = _enrich_evidence(
        [{"type": "Evidence",
          "attrs": {"subtype": "literature", "content": "citrus furanocoumarins",
                    "source_ref": "doi:10.1021/ACS.JAFC.2C05393"}}],
        "ResearchAgent", ctx)
    attrs = nodes[0]["attrs"]
    assert attrs["session_artifact_id"] == outcome["artifact_id"]
    assert attrs["paper_title"] == "Furanocoumarin toxicity in citrus"
    assert attrs["paper_year"] == 2022


def test_evidence_citing_a_pmc_id_finds_its_paper(key):
    """The join that had never once fired in a recorded run.

    Of the twenty-five distinct `source_ref` values on disk, eleven carry a PMC
    id and fourteen carry no DOI at all — and the matcher only understood DOIs.
    The citation is also a LIST, and the handle we hold may be third in it.
    """
    from CoScientist.graph.research.agent_tools import _enrich_evidence

    state = {}
    pl.merge(state, [{
        "doi": "", "doi_raw": "", "title": "Heracleum sosnowskyi, a review",
        "year": 2026, "is_oa": None,
        "refs": ["pmc:12610272"],
        "pdf_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12610272",
        "presigned_url": "", "bucket": None, "s3_key": None, "tool": "tavily_search"}])
    pl.note_stored(state, pl.library(state)[0],
                   {"state": sf.STATE_STORED, "artifact_id": "heracleum-ab12cd34.pdf"})

    ctx = type("Ctx", (), {"state": state, "agent_name": "ResearchAgent"})()
    nodes = _enrich_evidence([{"type": "Evidence", "attrs": {
        "subtype": "literature", "content": "furanocoumarin profile",
        # The exact shape a live run wrote, parenthetical title and all.
        "source_ref": "PMC12610272 (Analysis of the Toxicological Profile of "
                      "H. sosnowskyi metabolites using in silico methods), 2026"}}],
        "ResearchAgent", ctx)

    attrs = nodes[0]["attrs"]
    assert attrs["session_artifact_id"] == "heracleum-ab12cd34.pdf"
    assert attrs["pmcid"] == "PMC12610272"
    assert attrs["paper_title"] == "Heracleum sosnowskyi, a review"


def test_a_paper_met_twice_under_two_names_is_one_record(key):
    """A run meets a work by its DOI in one turn and its PMC id in the next."""
    state = {}
    pl.merge(state, [{"doi": "10.3390/plants14213253", "doi_raw": "",
                      "title": "Rassabina & Fedorov", "year": 2025, "is_oa": True,
                      "refs": ["doi:10.3390/plants14213253"],
                      "pdf_url": "https://example.org/a.pdf", "presigned_url": "",
                      "bucket": None, "s3_key": None, "tool": "search_papers"}])
    pl.merge(state, [{"doi": "10.3390/plants14213253", "doi_raw": "", "title": "",
                      "year": None, "is_oa": None,
                      "refs": ["doi:10.3390/plants14213253", "pmc:12610272"],
                      "pdf_url": "https://example.org/a.pdf", "presigned_url": "",
                      "bucket": None, "s3_key": None, "tool": "tavily_search"}])

    assert len(pl.library(state)) == 1, "one work, one record"
    assert set(pl.library(state)[0]["refs"]) == {
        "doi:10.3390/plants14213253", "pmc:12610272"}, "and now known by both"
    assert pl.find(state, "PMC12610272") is pl.find(state, "10.3390/plants14213253")


def test_what_the_agent_wrote_is_never_overwritten(key):
    """An attribute already present is the agent's claim, not ours."""
    from CoScientist.graph.research.agent_tools import _enrich_evidence

    state = {}
    pl.merge(state, pl.records_from_tool("search_papers", _search_result()))
    pl.note_stored(state, pl.library(state)[0],
                   {"state": "stored", "artifact_id": "ours.pdf"})
    ctx = type("Ctx", (), {"state": state, "agent_name": "ResearchAgent"})()
    nodes = _enrich_evidence(
        [{"type": "Evidence",
          "attrs": {"subtype": "literature", "content": "x",
                    "source_ref": "10.1021/acs.jafc.2c05393",
                    "session_artifact_id": "theirs.pdf", "paper_title": "As they read it"}}],
        "ResearchAgent", ctx)
    assert nodes[0]["attrs"]["session_artifact_id"] == "theirs.pdf"
    assert nodes[0]["attrs"]["paper_title"] == "As they read it"


def test_a_commit_that_mentions_no_paper_is_untouched(key):
    """The enrichment must be invisible to everything that is not a citation."""
    from CoScientist.graph.research.agent_tools import _enrich_evidence

    ctx = type("Ctx", (), {"state": {}, "agent_name": "CoderAgent"})()
    before = [{"type": "Evidence",
               "attrs": {"subtype": "computational", "measured_on": "run 4",
                         "content": "score -9.1", "source_ref": "RES-7"}}]
    after = _enrich_evidence([dict(n) for n in before], "CoderAgent", ctx)
    assert after[0]["attrs"] == before[0]["attrs"]


def test_the_node_offers_its_own_file(tmp_path, monkeypatch):
    """An Evidence citing a stored paper offers it in the panel.

    `attachments` were built only from the nodes a card folds in, so a node's
    own file had nowhere to appear — `_href` already knew how to link it and
    nothing was asking.
    """
    from CoScientist.graph.research.store import ResearchGraphStore

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")
    store = ResearchGraphStore(directory=str(tmp_path))
    store.ensure_root("Are citrus furanocoumarins toxic?")
    store._scope = ("u1", "s1")
    record = sf.put_bytes(store._scope, PDF, filename="furano.pdf",
                          media_type="application/pdf", source_kind="paper")
    store.commit(
        source="ResearchAgent",
        nodes=[{"type": "Evidence", "ref": "e",
                "attrs": {"subtype": "literature", "content": "toxic above 5 mg/kg",
                          "source_ref": "10.1021/acs.jafc.2c05393",
                          "session_artifact_id": record["artifact_id"]}}],
    )
    drawn = {n["id"]: n for n in store.to_view()["nodes"]}
    own = drawn["E1"]["attachments"][0]
    assert own["id"] == "E1#file", "never the bare node id — a folded child owns that"
    assert own["href"] == f"/api/users/u1/sessions/s1/artifacts/{record['artifact_id']}"
    assert own["label"] == "furano.pdf"


def test_a_node_whose_file_we_do_not_hold_offers_nothing(tmp_path, monkeypatch):
    """A stale id must not render as a download that 404s on click."""
    from CoScientist.graph.research.store import ResearchGraphStore

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    store = ResearchGraphStore(directory=str(tmp_path))
    store.ensure_root("Q?")
    store._scope = ("u1", "s1")
    store.commit(source="ResearchAgent",
                 nodes=[{"type": "Evidence", "ref": "e",
                         "attrs": {"subtype": "literature", "content": "x",
                                   "session_artifact_id": "gone-00000000.pdf"}}])
    drawn = {n["id"]: n for n in store.to_view()["nodes"]}
    assert drawn["E1"]["attachments"] == []


# ── the bibliography that had nothing to build from ─────────────────────────
def test_the_bibliography_comes_from_what_was_read(key):
    """`finalize._extract_references` carried a TODO waiting for exactly this."""
    from CoScientist.reporting.finalize import _extract_references

    state = {}
    pl.merge(state, pl.records_from_tool("search_papers", _search_result()))
    assert _extract_references(state) == [
        "Furanocoumarin toxicity in citrus (2022) https://doi.org/10.1021/acs.jafc.2c05393"
    ]


def test_an_explicit_reference_list_still_wins(key):
    """Something took the trouble to write it; do not second-guess it."""
    from CoScientist.reporting.finalize import _extract_references

    state = {"references": ["Written by hand"]}
    pl.merge(state, pl.records_from_tool("search_papers", _search_result()))
    assert _extract_references(state) == ["Written by hand"]
