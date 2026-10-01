"""What a citation string actually names.

An agent writes `attrs.source_ref` in prose, and the real values on disk say
what that means in practice:

    PMC12610272
    PMC12610272 (Analysis of the Toxicological Profile…), 2026
    doi:10.3390/plants14213253; doi:10.3390/jox16010006; Malikov&Saidkhodzhaev 2004
    MDPI Plants 2025, 15(3):346; PMC12821576
    https://pmc.ncbi.nlm.nih.gov/articles/PMC12610272; https://en.wikipedia.org/…
    Rassabina & Fedorov 2025, Plants 14(21):3253, DOI 10.3390/plants14213253

Two facts follow, and both broke the DOI-only matcher that came before.

**A source_ref is a LIST.** Ten of the twenty-five distinct values on disk name
two to six sources. Searching for one identifier keeps the first and loses the
rest, so everything here returns all of them.

**Most of them are not DOIs.** Eleven of twenty-five carry a PMC id and fourteen
carry no DOI at all — which is why not one literature Evidence, in any recorded
run, was ever matched to a stored paper.

What is deliberately NOT an identifier matters as much. `PubChem CID 2355` names
a compound, `MDPI 2039-4713/16/1/6` is an ISSN and an article path,
`S0009279722000850` is a ScienceDirect PII, `elaba:240835311` is a repository
number and `review 389134454` is a bare integer. Treating any of them as a work
would attach the wrong file to the wrong finding, which is worse than attaching
nothing: each pattern here therefore demands its own prefix or the `10.` that
only a DOI has.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

#: Trailing characters that end a sentence, not an identifier. A DOI runs to
#: the next space, so `doi:10.3390/plants14213253;` would otherwise carry the
#: semicolon into the key and never match the same DOI written plainly.
_TRAILING = ".,;:)]}>'\"" + "«»"

# A DOI is the one identifier with a shape of its own: the `10.` registrant
# prefix is what tells it apart from every ISSN path and article number in the
# corpus above. Case-insensitive by specification.
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s;,]+)", re.IGNORECASE)

# PMC ids are self-labelling, which is why they survive being written bare,
# inside a sentence, or as the tail of a URL.
_PMC_RE = re.compile(r"\bPMC(\d{5,9})\b", re.IGNORECASE)

# A PubMed id is a bare integer, so it is accepted ONLY behind its own label.
# Without that rule `review 389134454` in the corpus becomes a citation.
_PMID_RE = re.compile(r"\bPMID:?\s*(\d{6,9})\b", re.IGNORECASE)

# Both the label (`arXiv:2201.01234`) and the address
# (`https://arxiv.org/abs/2201.01234`), and both id generations. The version
# suffix is dropped on purpose: `v1` and `v3` are the same work.
_ARXIV_HEAD = r"\barxiv(?:\.org)?[:/\s]+(?:abs/|pdf/)?\s*"
_ARXIV_RE = re.compile(
    _ARXIV_HEAD + r"(\d{4}\.\d{4,5})(?:v\d+)?\b", re.IGNORECASE)
_ARXIV_OLD_RE = re.compile(
    _ARXIV_HEAD + r"([a-z-]+(?:\.[A-Z]{2})?/\d{7})\b", re.IGNORECASE)

_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/",
                 "https://dx.doi.org/", "http://dx.doi.org/", "doi:", "doi ")


@dataclass(frozen=True)
class Ref:
    """One identifier a citation names.

    ``key`` is what everything else matches on — the kind and the normalized
    value together, so a DOI and a PMC id can never collide. ``raw`` is what was
    actually written, kept because that is the string a reader will search for.
    """

    kind: str
    value: str
    raw: str = ""

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.value}"

    @property
    def url(self) -> str:
        """Where a reader can open it. Empty when there is no public address."""
        if self.kind == "doi":
            return f"https://doi.org/{self.value}"
        if self.kind == "pmc":
            return f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{self.value}/"
        if self.kind == "pmid":
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.value}/"
        if self.kind == "arxiv":
            return f"https://arxiv.org/abs/{self.value}"
        return ""


def normalize_doi(value: Any) -> str:
    """A DOI reduced to the one spelling everything matches on.

    Lowercased because the specification says a DOI is case-insensitive, and
    OpenAlex, Crossref and an agent's own prose disagree about case often enough
    that matching without this loses real hits.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    for prefix in _DOI_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):]
            break
    match = _DOI_RE.search(text)
    return match.group(1).rstrip(_TRAILING).lower() if match else ""


def parse_references(text: Any) -> List[Ref]:
    """Every identifier a citation string names, in the order they appear.

    Never guesses. A token that could be an identifier but carries no label and
    no DOI prefix is left alone — see the module docstring for the five kinds of
    number in the real data that are not works.
    """
    raw = str(text or "")
    if not raw.strip():
        return []
    found: List[Ref] = []
    seen = set()

    def add(kind: str, value: str, written: str) -> None:
        value = value.strip().rstrip(_TRAILING)
        if not value:
            return
        ref = Ref(kind=kind, value=value.lower() if kind == "doi" else value,
                  raw=written.strip())
        if ref.key not in seen:
            seen.add(ref.key)
            found.append(ref)

    for match in _DOI_RE.finditer(raw):
        add("doi", match.group(1), match.group(0))
    for match in _PMC_RE.finditer(raw):
        add("pmc", match.group(1), match.group(0))
    for match in _PMID_RE.finditer(raw):
        add("pmid", match.group(1), match.group(0))
    for pattern in (_ARXIV_RE, _ARXIV_OLD_RE):
        for match in pattern.finditer(raw):
            add("arxiv", match.group(1), match.group(0))
    return found


def keys_of(text: Any) -> List[str]:
    """The match keys a citation string yields."""
    return [ref.key for ref in parse_references(text)]


def first(text: Any, kind: str) -> Optional[Ref]:
    """The first identifier of one kind, or None."""
    return next((r for r in parse_references(text) if r.kind == kind), None)


def refs_of(*values: Any) -> List[Ref]:
    """Every identifier across several strings, de-duplicated by key."""
    out: List[Ref] = []
    seen = set()
    for value in values:
        for ref in parse_references(value):
            if ref.key not in seen:
                seen.add(ref.key)
                out.append(ref)
    return out


def cite(refs: Iterable[Ref]) -> str:
    """One readable address for a set of identifiers, DOI first."""
    order = {"doi": 0, "pmc": 1, "pmid": 2, "arxiv": 3}
    best = sorted(refs, key=lambda r: order.get(r.kind, 9))
    return next((r.url for r in best if r.url), "")


def as_dicts(refs: Iterable[Ref]) -> List[Dict[str, str]]:
    """Plain records, for anything that has to survive JSON."""
    return [{"kind": r.kind, "value": r.value, "raw": r.raw, "url": r.url}
            for r in refs]


__all__ = ["Ref", "parse_references", "normalize_doi", "keys_of", "first",
           "refs_of", "cite", "as_dicts"]
