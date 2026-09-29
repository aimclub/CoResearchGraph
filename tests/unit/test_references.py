"""What a citation string names — and, as much, what it does not.

The corpus is real: every string in `CORPUS` is a distinct `attrs.source_ref`
value taken off disk from recorded runs. It is the reason the DOI-only matcher
that came before never fired once in any recorded run — eleven of the
twenty-five carry a PMC id and fourteen carry no DOI at all.

The rejections carry as much weight as the matches. Attaching the wrong file to
a finding is worse than attaching none, and the corpus is full of numbers that
look like identifiers and are not: a PubChem compound id, an ISSN with an
article path, a ScienceDirect PII, a repository accession, a bare review number.
"""
from __future__ import annotations

import pytest

from CoScientist.reporting.references import (
    Ref,
    cite,
    keys_of,
    normalize_doi,
    parse_references,
    refs_of,
)


# ── the DOI, in every spelling it arrives in ────────────────────────────────
@pytest.mark.parametrize("written", [
    "10.3390/plants14213253",
    "10.3390/PLANTS14213253",
    "doi:10.3390/plants14213253",
    "DOI 10.3390/plants14213253",
    "https://doi.org/10.3390/plants14213253",
    "http://dx.doi.org/10.3390/plants14213253",
    "doi:10.3390/plants14213253;",
    "10.3390/plants14213253 (Rassabina & Fedorov 2025)",
    "cdnsciencepub 10.3390/plants14213253",
])
def test_one_doi_however_it_was_written(written):
    """A DOI is case-insensitive by specification and prefixed by habit."""
    assert keys_of(written) == ["doi:10.3390/plants14213253"]


# ── PMC, the identifier that was being lost ─────────────────────────────────
@pytest.mark.parametrize("written", [
    "PMC12610272",
    "pmc12610272",
    "PMC12610272 (Analysis of the Toxicological Profile…), 2026",
    "PMC12610272 (кластер E, Table 2, Fig.3)",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12610272",
])
def test_one_pmc_id_however_it_was_written(written):
    assert keys_of(written) == ["pmc:12610272"]


# ── a citation is a LIST ────────────────────────────────────────────────────
def test_every_identifier_is_returned_not_the_first():
    """Ten of the twenty-five real values name two to six sources.

    `re.search` kept the first and lost the rest, which is how a run that had
    read three papers was recorded as having read one.
    """
    assert keys_of("doi:10.3390/plants14213253; doi:10.3390/jox16010006; "
                   "Malikov&Saidkhodzhaev 2004") == [
        "doi:10.3390/plants14213253", "doi:10.3390/jox16010006"]
    assert keys_of("DOI 10.3390/plants14213253; PMC12610272") == [
        "doi:10.3390/plants14213253", "pmc:12610272"]


# ── what must never be taken for a work ─────────────────────────────────────
@pytest.mark.parametrize("written,why", [
    ("PubChem CID 2355 (IARC)", "a compound, not a work"),
    ("PubChem CIDs (псорален 6199, бергаптен 2355, ксантотоксин 4114)", "compounds"),
    ("MDPI Genes 2039-4713/16/1/6", "an ISSN and an article path"),
    ("MDPI Plants 2025, 15(3):346", "a volume and a page"),
    ("SciDirect S0009279722000850", "a ScienceDirect PII"),
    ("EPubl. Vilnius elaba:240835311", "a repository accession"),
    ("review 389134454", "a bare integer behind a word"),
    ("wikipedia Psoralen; SKLM Furocoumarines 2006", "encyclopaedia and a report"),
    ("EFSA/Schlatter 2004 (Toxicological Assessment of Furocoumarins)", "author-year"),
    ("Gurina Med Journal; Malikov&Saidkhodzhaev 2004", "author-year"),
    ("pubmed query X", "the word pubmed is not a PMID"),
    ("https://www.thegoodscentscompany.com/data/rw1037841.html", "a trade page"),
    ("", "nothing at all"),
])
def test_what_is_not_a_work_is_not_taken_for_one(written, why):
    assert parse_references(written) == [], why


def test_a_pmid_needs_its_own_label():
    """A PubMed id is a bare integer, so only its label makes it one.

    Without that rule `review 389134454` in the corpus becomes a citation.
    """
    assert keys_of("PMID: 38913445") == ["pmid:38913445"]
    assert keys_of("PMID38913445") == ["pmid:38913445"]
    assert parse_references("38913445") == []


def test_arxiv_in_both_its_shapes():
    assert keys_of("arXiv:2201.01234") == ["arxiv:2201.01234"]
    assert keys_of("arXiv:2201.01234v3") == ["arxiv:2201.01234"]
    assert keys_of("https://arxiv.org/abs/2201.01234") == ["arxiv:2201.01234"]
    assert keys_of("arXiv:math.GT/0309136") == ["arxiv:math.GT/0309136"]


# ── the whole corpus at once ────────────────────────────────────────────────
CORPUS = [
    ("PMC12610272", ["pmc:12610272"]),
    ("PMC12610272; MDPI 2223-7747/15/3/346; PMC12363131; MDPI 2039-4713/16/1/6; "
     "cdnsciencepub 10.1139/cjb-2017-0043",
     ["doi:10.1139/cjb-2017-0043", "pmc:12610272", "pmc:12363131"]),
    ("wikipedia Psoralen; SKLM Furocoumarines 2006; PMC7269730", ["pmc:7269730"]),
    ("PMC7269730; review 389134454; IARC", ["pmc:7269730"]),
    ("doi:10.3390/plants14213253", ["doi:10.3390/plants14213253"]),
    ("MDPI Plants 2025, 15(3):346; PMC12821576", ["pmc:12821576"]),
    ("10.1186/s12870-025-07042-3 (BMC Plant Biology, 2025)",
     ["doi:10.1186/s12870-025-07042-3"]),
    ("PMC12610272; 10.1186/s12870-025-07042-3",
     ["doi:10.1186/s12870-025-07042-3", "pmc:12610272"]),
    ("Rassabina & Fedorov 2025, Plants 14(21):3253; Frumin 2024, Russian J. "
     "General Chem., Springer 10.1134/S1070363223130315",
     ["doi:10.1134/s1070363223130315"]),
    ("PubChem CID; E1 (этот граф); литература по фуранокумаринам", []),
]


@pytest.mark.parametrize("written,expected", CORPUS)
def test_the_real_corpus(written, expected):
    assert sorted(keys_of(written)) == sorted(expected)


# ── what a reader can open ──────────────────────────────────────────────────
def test_every_identifier_knows_where_it_opens():
    assert Ref("doi", "10.3390/x").url == "https://doi.org/10.3390/x"
    assert Ref("pmc", "12610272").url == (
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC12610272/")
    assert Ref("pmid", "38913445").url == "https://pubmed.ncbi.nlm.nih.gov/38913445/"
    assert Ref("arxiv", "2201.01234").url == "https://arxiv.org/abs/2201.01234"


def test_the_doi_is_the_address_a_reader_is_given():
    """DOI first: it is the one identifier that outlives an index."""
    refs = parse_references("PMC12610272; 10.3390/plants14213253")
    assert cite(refs) == "https://doi.org/10.3390/plants14213253"
    assert cite(parse_references("PMC12610272")).endswith("PMC12610272/")
    assert cite([]) == ""


def test_identifiers_gathered_across_several_strings():
    found = refs_of("PMC12610272", "10.3390/plants14213253", "PMC12610272")
    assert [r.key for r in found] == ["pmc:12610272", "doi:10.3390/plants14213253"]


def test_what_was_written_is_kept_beside_what_was_matched():
    """The raw spelling is what a reader will search the page for."""
    ref = parse_references("DOI 10.3390/PLANTS14213253")[0]
    assert ref.value == "10.3390/plants14213253"
    assert ref.raw == "10.3390/PLANTS14213253"


# ── the narrow helper the rest of the code still uses ───────────────────────
@pytest.mark.parametrize("junk", ["", None, "see the appendix", "10.bad/x",
                                  "arXiv:2201.1", "PMC12610272"])
def test_normalize_doi_stays_narrow(junk):
    """It answers about DOIs and nothing else — `parse_references` is the wide
    door. Widening this one in place would have quietly changed what every
    existing caller means by it."""
    assert normalize_doi(junk) == ""
