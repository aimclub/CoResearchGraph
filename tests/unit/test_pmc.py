"""Where an open copy of a PubMed Central article lives — asked, never guessed.

The case against the obvious design is in this repository's own artifact
manifest. A live session tried `https://pmc.ncbi.nlm.nih.gov/articles/pdf/
plants-14-03253.pdf` and `.../jox-16-00006.pdf`; both came back 4xx and were
filed as expired links, which is a misleading thing for anyone to read later.
NCBI refuses a plain programmatic GET of the article tree, so the address is
requested from services that exist to answer that question.

No test here touches the network.
"""
from __future__ import annotations

import json

import pytest

from CoScientist.reporting import pmc


@pytest.fixture(autouse=True)
def _no_pacing_and_no_memory(monkeypatch):
    """Drop the rate limit and the negative cache between tests."""
    monkeypatch.setattr(pmc, "_MIN_INTERVAL", 0.0)
    pmc.forget()
    yield
    pmc.forget()


class _Answer:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text if text else json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _europe(*urls, is_oa="Y"):
    return {"resultList": {"result": [{
        "isOpenAccess": is_oa,
        "fullTextUrlList": {"fullTextUrl": list(urls)},
    }]}}


def _pdf(url, availability="Open access"):
    return {"documentStyle": "pdf", "availability": availability, "url": url}


def _serve(monkeypatch, by_url):
    """Answer each service from a table; anything unexpected is a failure."""
    calls = []

    def _get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        for prefix, answer in by_url.items():
            if url.startswith(prefix):
                return answer() if callable(answer) else answer
        raise AssertionError(f"unexpected request to {url}")

    import requests

    monkeypatch.setattr(requests, "get", _get)
    return calls


# ── the identifier ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("written", [
    "PMC12610272", "pmc12610272", "12610272",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12610272",
    "PMC12610272 (Analysis of the Toxicological Profile…)",
])
def test_the_id_is_read_in_every_shape(written):
    assert pmc.normalize_pmcid(written) == "12610272"


@pytest.mark.parametrize("junk", ["", None, "PMC12", "no digits here"])
def test_what_is_not_an_id_yields_nothing(junk):
    assert pmc.normalize_pmcid(junk) == ""


# ── the preferred answer ────────────────────────────────────────────────────
def test_europe_pmc_answers_the_address_and_the_licence(monkeypatch):
    """Both halves in one call — which is the reason it is asked first.

    `is_oa` is what makes the licence status of a stored paper visible instead
    of assumed, and a search result can never supply it.
    """
    calls = _serve(monkeypatch, {pmc.EUROPE_PMC: _Answer(
        payload=_europe(_pdf("https://europepmc.org/articles/PMC12610272?pdf=render")))})

    found = pmc.resolve({"kind": "pmc", "value": "12610272"})

    assert found.url.endswith("pdf=render")
    assert found.is_oa is True
    assert found.via == "europepmc"
    assert calls[0][1]["query"] == "PMCID:PMC12610272"


def test_an_open_copy_is_preferred_over_a_paywalled_one(monkeypatch):
    """The same article is often listed twice — once behind a paywall."""
    _serve(monkeypatch, {pmc.EUROPE_PMC: _Answer(payload=_europe(
        _pdf("https://publisher.example/paywalled.pdf", availability="Subscription"),
        _pdf("https://europepmc.org/open.pdf", availability="Open access")))})

    assert pmc.resolve({"kind": "pmc", "value": "12610272"}).url == (
        "https://europepmc.org/open.pdf")


# ── the fallback, and its honest negative ───────────────────────────────────
def test_the_ncbi_service_answers_when_europe_pmc_does_not(monkeypatch):
    """And its FTP answer is rewritten to the HTTPS tree the mirror can read."""
    _serve(monkeypatch, {
        pmc.EUROPE_PMC: _Answer(payload={"resultList": {"result": []}}),
        pmc.OA_SERVICE: _Answer(text=(
            '<OA><records><record id="PMC12610272">'
            '<link format="pdf" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/'
            'aa/bb/plants.PMC12610272.pdf"/></record></records></OA>')),
    })

    found = pmc.resolve({"kind": "pmc", "value": "12610272"})

    assert found.url.startswith("https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/")
    assert found.via == "ncbi-oa"


def test_not_open_access_is_an_answer_not_a_failure(monkeypatch):
    calls = _serve(monkeypatch, {
        pmc.EUROPE_PMC: _Answer(payload={"resultList": {"result": []}}),
        pmc.OA_SERVICE: _Answer(text=(
            '<OA><error code="idIsNotOpenAccess">not open access</error></OA>')),
    })

    found = pmc.resolve({"kind": "pmc", "value": "12610272"})

    assert found.url == ""
    assert found.is_oa is False
    assert not found, "falsy, so a caller keeps the citation and drops the file"

    # And it is remembered: one study cites the same paper from several
    # findings, and each of them asking is how a polite client is throttled.
    before = len(calls)
    again = pmc.resolve({"kind": "pmc", "value": "12610272"})
    assert len(calls) == before, "no second round of requests"
    assert again.via == "cached"


def test_the_memory_can_be_dropped(monkeypatch):
    _serve(monkeypatch, {pmc.EUROPE_PMC: _Answer(payload={"resultList": {"result": []}}),
                         pmc.OA_SERVICE: _Answer(text='<OA><error code="x"/></OA>')})
    pmc.resolve({"kind": "pmc", "value": "12610272"})
    pmc.forget("PMC12610272")
    calls = _serve(monkeypatch, {pmc.EUROPE_PMC: _Answer(
        payload=_europe(_pdf("https://europepmc.org/open.pdf")))})
    assert pmc.resolve({"kind": "pmc", "value": "12610272"}).url
    assert calls, "asked again"


# ── a PubMed id gets converted first ────────────────────────────────────────
def test_a_pmid_is_converted_before_anything_is_asked(monkeypatch):
    calls = _serve(monkeypatch, {
        pmc.IDCONV: _Answer(payload={"records": [{"pmcid": "PMC12610272"}]}),
        pmc.EUROPE_PMC: _Answer(payload=_europe(_pdf("https://europepmc.org/p.pdf"))),
    })

    assert pmc.resolve({"kind": "pmid", "value": "38913445"}).url.endswith("p.pdf")
    assert calls[0][0].startswith(pmc.IDCONV)
    assert calls[0][1]["ids"] == "38913445"
    assert calls[0][1]["tool"], "NCBI asks every client to identify itself"


def test_a_pmid_with_no_pmc_record_asks_nothing_further(monkeypatch):
    calls = _serve(monkeypatch, {pmc.IDCONV: _Answer(payload={"records": []})})
    assert pmc.resolve({"kind": "pmid", "value": "38913445"}) is None
    assert len(calls) == 1


# ── nothing here may break a run ────────────────────────────────────────────
def test_a_service_that_is_down_is_not_an_exception(monkeypatch):
    def _explode(*a, **k):
        raise OSError("network down")

    import requests

    monkeypatch.setattr(requests, "get", _explode)
    assert pmc.resolve({"kind": "pmc", "value": "12610272"}).url == ""


def test_a_malformed_answer_is_not_an_exception(monkeypatch):
    _serve(monkeypatch, {pmc.EUROPE_PMC: _Answer(text="<not json>"),
                         pmc.OA_SERVICE: _Answer(text="<<<broken xml")})
    assert pmc.resolve({"kind": "pmc", "value": "12610272"}).url == ""


def test_something_that_is_not_an_identifier_asks_nothing(monkeypatch):
    _serve(monkeypatch, {})  # any request would fail the test
    assert pmc.resolve({"kind": "doi", "value": "10.3390/plants14213253"}) is None
    assert pmc.resolve(None) is None


# ── and the capture path uses it ────────────────────────────────────────────
def test_a_pmc_article_page_is_fetched_as_a_pdf_not_as_a_web_page(monkeypatch):
    """The failure this whole module exists to prevent.

    A web search hands back the ARTICLE page. Fetching that stores an HTML page
    under the name of a paper.
    """
    from CoScientist.tools.paper_capture_plugin import _open_copy

    _serve(monkeypatch, {pmc.EUROPE_PMC: _Answer(
        payload=_europe(_pdf("https://europepmc.org/articles/PMC12821576?pdf=render")))})

    paper = {"refs": ["pmc:12821576"], "is_oa": None,
             "pdf_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12821576"}
    _open_copy(paper)

    assert paper["oa_url"].endswith("pdf=render")
    assert paper["is_oa"] is True, "and the licence answer came with it"


def test_a_paper_with_no_pmc_id_is_left_exactly_as_it_was(monkeypatch):
    from CoScientist.tools.paper_capture_plugin import _open_copy

    _serve(monkeypatch, {})
    paper = {"refs": ["doi:10.3390/plants14213253"], "is_oa": None,
             "pdf_url": "https://example.org/a.pdf"}
    _open_copy(paper)
    assert "oa_url" not in paper
