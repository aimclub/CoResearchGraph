"""Where an open-access copy of a PubMed Central article actually lives.

DO NOT BUILD THE URL. The obvious template —
``https://pmc.ncbi.nlm.nih.gov/articles/pdf/<name>.pdf`` — does not work, and
the evidence is in this repository's own artifact manifest: a live session tried
exactly that for ``plants-14-03253.pdf`` and ``jox-16-00006.pdf`` and both came
back 4xx. NCBI refuses a plain programmatic GET of the article tree, and the
mirror filed the refusal as an expired link, which is a misleading thing to read
afterwards.

So the address is asked for, from services that exist to answer that question.
Europe PMC is preferred because it answers both halves at once — where the PDF
is AND whether the article is open access — and that second half is what we
record beside every stored paper so a licence status is visible rather than
assumed. NCBI's own OA service is the fallback, and it has the virtue of a
blunt, honest negative: ``idIsNotOpenAccess`` means there is nothing to fetch,
which is worth recording and not worth retrying.

Nothing here touches the graph, ADK or session state. One concern: an
identifier in, an address out, or nothing.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OA_SERVICE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

#: NCBI asks for ≤3 requests a second without an API key, and asks every client
#: to identify itself. Both are conditions of use, not optimisations.
_MIN_INTERVAL = float(os.getenv("PMC__MIN_INTERVAL", "0.35"))
_TOOL = os.getenv("NCBI__TOOL", "coscientist")
_EMAIL = os.getenv("NCBI__EMAIL", "")

_TIMEOUT = int(os.getenv("PMC__TIMEOUT", "20"))
#: How long a "no open copy" answer stands. A study cites the same paper from
#: several findings; asking about each one separately is how a polite client
#: becomes an impolite one.
_MISS_TTL = float(os.getenv("PMC__MISS_TTL", "600"))

_lock = threading.Lock()
_last_call = 0.0
_misses: Dict[str, float] = {}

_PMC_LABELLED = re.compile(r"\bPMC(\d{5,9})\b", re.IGNORECASE)
_PMC_BARE = re.compile(r"\d{5,9}")


@dataclass(frozen=True)
class Copy:
    """An open copy of an article, or the fact that there is none.

    ``url`` empty with ``is_oa`` False is a real answer — the service said the
    article is not open access — and it is recorded rather than retried.
    """

    url: str = ""
    is_oa: Optional[bool] = None
    via: str = ""

    def __bool__(self) -> bool:
        return bool(self.url)


def _pace() -> None:
    """Hold the agreed interval between calls. Global, because the limit is."""
    global _last_call
    with _lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _identified(params: Dict[str, str]) -> Dict[str, str]:
    out = dict(params, tool=_TOOL)
    if _EMAIL:
        out["email"] = _EMAIL
    return out


def _get(url: str, params: Dict[str, str], *, retries: int = 2) -> Optional[Any]:
    """One GET, paced, with a single backoff on the two retryable answers.

    The shape `papers_processing_refactoring.utils.openalex._request_json` uses,
    minus its raises: a resolver that cannot answer returns nothing, because
    every caller already has a paper worth keeping without the file.
    """
    import requests

    for attempt in range(retries):
        _pace()
        try:
            response = requests.get(url, params=params, timeout=_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            logger.debug("pmc: %s unreachable (%s)", url, exc)
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
                continue
            return None
        if response.status_code == 200:
            return response
        if (response.status_code == 429 or response.status_code >= 500) \
                and attempt + 1 < retries:
            time.sleep(2 ** attempt)
            continue
        logger.debug("pmc: %s answered %s", url, response.status_code)
        return None
    return None


def normalize_pmcid(value: Any) -> str:
    """``PMC12610272`` → ``12610272``. Accepts the bare number and the URL.

    Deliberately NOT "the first run of digits". A DOI carries one — the eight
    digits in `10.3390/plants14213253` would otherwise be read as a PMC id and
    send a request about an article nobody named.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    labelled = _PMC_LABELLED.search(text)
    if labelled:
        return labelled.group(1)
    return text if _PMC_BARE.fullmatch(text) else ""


def _from_europe_pmc(pmcid: str) -> Optional[Copy]:
    """Europe PMC: the address and the licence in one answer."""
    response = _get(EUROPE_PMC, {"query": f"PMCID:PMC{pmcid}",
                                 "resultType": "core", "format": "json"})
    if response is None:
        return None
    try:
        results = ((response.json() or {}).get("resultList") or {}).get("result") or []
    except Exception:  # noqa: BLE001
        return None
    if not results:
        return None
    record = results[0]
    is_oa = str(record.get("isOpenAccess") or "").upper() == "Y"
    urls = ((record.get("fullTextUrlList") or {}).get("fullTextUrl") or [])
    pdfs = [u for u in urls if str(u.get("documentStyle") or "").lower() == "pdf"]
    # An open-access copy first: the same article is often listed twice, once
    # behind a publisher's paywall and once in the archive.
    pdfs.sort(key=lambda u: str(u.get("availability") or "") != "Open access")
    for entry in pdfs:
        url = str(entry.get("url") or "").strip()
        if url:
            return Copy(url=url, is_oa=is_oa, via="europepmc")
    return Copy(url="", is_oa=is_oa, via="europepmc")


def _from_ncbi_oa(pmcid: str) -> Optional[Copy]:
    """NCBI's own open-access service, and its blunt negative answer."""
    response = _get(OA_SERVICE, _identified({"id": f"PMC{pmcid}", "format": "pdf"}))
    if response is None:
        return None
    try:
        root = ET.fromstring(response.text)
    except Exception:  # noqa: BLE001
        return None
    error = root.find(".//error")
    if error is not None:
        code = (error.get("code") or "").strip()
        # Not a failure: the service is telling us there is nothing to fetch.
        return Copy(url="", is_oa=False if code == "idIsNotOpenAccess" else None,
                    via="ncbi-oa")
    for link in root.findall(".//link"):
        href = (link.get("href") or "").strip()
        if not href:
            continue
        # The service answers in FTP. The same tree is served over HTTPS, which
        # is what `mirror._fetch` can actually read.
        if href.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
            href = "https://ftp.ncbi.nlm.nih.gov/" + href[len("ftp://ftp.ncbi.nlm.nih.gov/"):]
        if href.startswith("https://"):
            return Copy(url=href, is_oa=True, via="ncbi-oa")
    return None


def pmcid_for_pmid(pmid: Any) -> str:
    """The PMC id of a PubMed record, when the article has one."""
    digits = re.sub(r"\D", "", str(pmid or ""))
    if not digits:
        return ""
    response = _get(IDCONV, _identified({"ids": digits, "format": "json"}))
    if response is None:
        return ""
    try:
        records = (response.json() or {}).get("records") or []
    except Exception:  # noqa: BLE001
        return ""
    return normalize_pmcid(records[0].get("pmcid")) if records else ""


def resolve(ref: Any) -> Optional[Copy]:
    """Where to fetch this article, or None when nobody could say.

    ``ref`` is a `references.Ref`, a dict from one, or a bare id. A remembered
    negative is returned immediately: the same paper is cited by several
    findings, and each of them asking is how a client gets throttled.
    """
    kind = getattr(ref, "kind", None) or (ref.get("kind") if isinstance(ref, dict) else "")
    value = getattr(ref, "value", None) or (ref.get("value") if isinstance(ref, dict) else ref)

    # Only the two kinds this service knows about. A DOI or an arXiv id has no
    # answer here, and asking would be a request about an article nobody named.
    if kind and kind not in ("pmc", "pmid"):
        return None
    pmcid = pmcid_for_pmid(value) if kind == "pmid" else normalize_pmcid(value)
    if not pmcid:
        return None

    now = time.monotonic()
    missed_at = _misses.get(pmcid)
    if missed_at is not None and now - missed_at < _MISS_TTL:
        return Copy(url="", is_oa=False, via="cached")

    for resolver in (_from_europe_pmc, _from_ncbi_oa):
        try:
            found = resolver(pmcid)
        except Exception as exc:  # noqa: BLE001 — a resolver never breaks a run
            logger.debug("pmc: %s failed on PMC%s (%s)",
                         resolver.__name__, pmcid, exc)
            continue
        if found and found.url:
            return found
    _misses[pmcid] = now
    return Copy(url="", is_oa=False, via="none")


def forget(pmcid: Any = None) -> None:
    """Drop the remembered negatives. For tests and for an operator retrying."""
    if pmcid is None:
        _misses.clear()
    else:
        _misses.pop(normalize_pmcid(pmcid), None)


__all__ = ["Copy", "resolve", "normalize_pmcid", "pmcid_for_pmid", "forget",
           "EUROPE_PMC", "OA_SERVICE", "IDCONV"]
