"""Recognise the papers a run finds, and keep them as papers.

A run finds literature two ways and only one of them was ever watched. The
papers server (`search_papers`, `download_papers_from_search`) returns a tidy
record, and a callback collected it. But in practice the ResearchAgent reaches
for the web: a live session made seven Tavily calls and not one call to the
papers server, and the evidence it committed cited `PMC12610272` — a paper the
run had genuinely read and had no copy of.

Two things were wrong, and this plugin fixes the first.

**It is a plugin, not an after_tool callback.** ADK runs plugin callbacks first
and the agent's own only when no plugin answered, so the truncation plugin —
which answered on every result over 12 000 characters — was silently cancelling
the agent's chain on exactly the big web-search results that carry papers. That
is fixed in `agents/truncation_plugin.py`; being a plugin here means this never
depended on it in the first place, and it also reaches tools called inside an
AgentTool child.

**It re-files rather than re-fetches.** `McpArtifactCapturePlugin` runs before
this one and has already mirrored any `.pdf` link a result carried. A second
download of the same bytes would produce the same content-addressed id at the
cost of another request to someone else's server, so a paper already in the
store is simply re-filed as one.

What counts as a paper is decided by `link_registry.classify_url`, the
classifier the link table already renders into every agent's prompt — so this
cannot drift from what a reader is told a link is. A PDF on a publisher's host
is a paper; the same PDF on a ministry's host is a document, and stays one.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from google.adk.plugins import BasePlugin

from CoScientist.reporting.collect import WEB_READING_TOOLS

logger = logging.getLogger(__name__)

#: Hosts that publish papers but are not in `link_registry._PAPER_HOSTS`.
#: Deliberately kept here and not added there: that tuple decides the word
#: rendered into every agent's `{links_context?}` block, and widening it would
#: change what models read about a link.
_EXTRA_PAPER_HOSTS = (
    "mdpi.com", "frontiersin.org", "plos.org", "tandfonline.com",
    "sagepub.com", "cambridge.org", "academic.oup.com", "bmj.com",
    "jstor.org", "semanticscholar.org", "europepmc.org", "researchgate.net",
    "elsevier.com", "ssrn.com", "preprints.org", "osti.gov",
)

#: The tools whose results are worth walking. Everything else a run calls
#: returns its own artifacts, which `McpArtifactCapturePlugin` already owns.
#: One definition, shared with the rule that keeps what a run READ out of the
#: report's illustrations — the two must never drift apart.
_SEARCH_TOOLS = WEB_READING_TOOLS
_PAPERS_TOOLS = ("search_papers", "download_papers_from_search")

_URL_KEYS = ("url", "link", "href", "pdf_url", "landing_page_url")
_TITLE_KEYS = ("title", "paper_title", "name", "display_name")
_TEXT_KEYS = ("content", "snippet", "abstract", "raw_content", "description")
_MAX_DEPTH = 6


def _host(url: str) -> str:
    try:
        host = urlsplit(str(url)).netloc.lower()
    except Exception:  # noqa: BLE001
        return ""
    return host[4:] if host.startswith("www.") else host


def _published_by(url: str) -> bool:
    """Whether the host is one that publishes papers."""
    from CoScientist.agents.callbacks.link_registry import _PAPER_HOSTS

    host = _host(url)
    if not host:
        return False
    return any(host == h or host.endswith("." + h)
               for h in tuple(_PAPER_HOSTS) + _EXTRA_PAPER_HOSTS)


def is_paper_link(url: str) -> bool:
    """Whether this URL names a scientific work, from its shape alone.

    Conservative by construction: the host decides. `role == "paper"` is the
    classifier's own answer for an index or publisher it knows, and the host
    list covers the publishers it does not — including the ones whose article
    pages carry no extension at all, where the classifier can only say "web
    page" (an MDPI article is `/2223-7747/15/3/346`).

    Note the classifier answers `"PDF document"` before it ever looks at the
    host, so a Springer PDF arrives with that role and is admitted here by its
    host. Which is exactly what keeps a ministry's furanocoumarin report and a
    supplier's application note filed as the documents they are: same role,
    host nobody publishes papers from.
    """
    try:
        from CoScientist.agents.callbacks.link_registry import classify_url

        role, _label = classify_url(url)
    except Exception:  # noqa: BLE001
        return False
    return role == "paper" or _published_by(url)


def result_items(result: Any, _depth: int = 0) -> List[Dict[str, str]]:
    """Every ``{url, title, snippet}`` a tool result names, in any envelope.

    The title is why this exists. The link registry records every URL a tool
    returned and deliberately keeps no title, and Tavily hands its payload back
    as ONE JSON string at ``content[0].text`` — so a structural walk over the
    envelope never reaches ``results[i].title``. Parsing the JSON strings on the
    way down is the only way a paper arrives with its name.
    """
    if _depth > _MAX_DEPTH:
        return []
    out: List[Dict[str, str]] = []
    if isinstance(result, str):
        text = result.lstrip()
        if text[:1] in "{[":
            try:
                return result_items(json.loads(result), _depth + 1)
            except Exception:  # noqa: BLE001
                return []
        return []
    if isinstance(result, list):
        for item in result:
            out.extend(result_items(item, _depth + 1))
        return out
    if not isinstance(result, dict):
        return []

    url = next((str(result[k]) for k in _URL_KEYS
                if isinstance(result.get(k), str) and result[k].strip()), "")
    if url:
        out.append({
            "url": url.strip(),
            "title": next((str(result[k]).strip() for k in _TITLE_KEYS
                           if isinstance(result.get(k), str) and result[k].strip()), ""),
            "snippet": next((str(result[k])[:600] for k in _TEXT_KEYS
                             if isinstance(result.get(k), str) and result[k].strip()), ""),
        })
    for value in result.values():
        if isinstance(value, (dict, list, str)):
            out.extend(result_items(value, _depth + 1))
    return out


def _identity_of(url: str, title: Any, text: Any) -> List[Any]:
    """Which identifiers are THIS result's own — not ones it merely mentions.

    The address is authoritative: a doi.org, PMC or arXiv URL *is* the work's
    name, and when it carries one nothing else is read. The page's prose is not
    authoritative at all. A scientific abstract cites other works by DOI and by
    PMC id as a matter of course, and those were being recorded as this record's
    own handles — after which `paper_library.merge` folded the cited work into
    this one on the shared key, `_attach_paper` stamped an Evidence with a DOI
    belonging to neither, and the bibliography printed the pair. One wrong
    attribution of someone else's work is worse than a hundred papers left
    unmatched, which is why this errs the other way.

    So the text is read only when the address names nothing, and only when it
    names exactly ONE work: a page whose prose mentions two identifiers cannot
    tell us which of them it is about. That leaves the common case — a landing
    page whose citation header carries its own DOI — and refuses the ambiguous
    one.
    """
    from CoScientist.reporting import references as refs

    by_address = refs.refs_of(url)
    if by_address:
        return by_address
    named = refs.refs_of(title, text)
    return named if len(named) == 1 else []


def papers_in(result: Any, tool_name: str) -> List[Dict[str, Any]]:
    """The paper records one tool result is telling us about."""
    from CoScientist.reporting import paper_library as pl

    if tool_name in _PAPERS_TOOLS:
        # The papers server already answers in records; nothing to infer.
        return pl.records_from_tool(tool_name, result)

    seen, records = set(), []
    for item in result_items(result):
        url = item["url"]
        if url in seen or not is_paper_link(url):
            continue
        seen.add(url)
        found = _identity_of(url, item["title"], item["snippet"])
        records.append({
            "doi": next((r.value for r in found if r.kind == "doi"), ""),
            # Every handle the result names. A PMC id lives in the URL far more
            # often than in a field, which is the whole reason a web result can
            # be matched to an agent's citation at all.
            "refs": [r.key for r in found],
            "doi_raw": "",
            "title": item["title"],
            "year": None,
            # The licence is not knowable from a search result. Recorded as
            # unknown rather than guessed: `is_oa` is an answer, not a default.
            "is_oa": None,
            "pdf_url": url,
            "presigned_url": "",
            "bucket": None,
            "s3_key": None,
            "tool": tool_name,
        })
    for record in records:
        if not record["title"]:
            record["title"] = record["doi"] or record["pdf_url"]
    return records


def mirror_papers(papers: List[Dict[str, Any]], scope: Any,
                   agent: str) -> List[Dict[str, Any]]:
    """Fetch each paper into the session store. One thread hop for the batch.

    The presigned link is preferred over the publisher's: it points at the copy
    the papers server already made, so it is the one that reliably answers.
    """
    from CoScientist.reporting import paper_library as pl
    from CoScientist.reporting.mirror import mirror_artifact

    out: List[Dict[str, Any]] = []
    for paper in papers:
        try:
            out.append(mirror_artifact(
                None, user_id=scope[0], session_id=scope[1],
                url=(paper.get("oa_url") or paper.get("presigned_url")
                     or paper.get("pdf_url") or None),
                bucket=paper.get("bucket"), s3_key=paper.get("s3_key"),
                filename=pl.filename_for(paper),
                label=str(paper.get("title") or "")[:120],
                tool=str(paper.get("tool") or ""),
                source_kind=pl.SOURCE_KIND, agent=agent,
                # Stays in the session folder and nowhere else. A paper is
                # someone else's work under someone else's licence, and the
                # off-host copy would outlive the run that justified fetching it.
                mirror_off_host=False,
            ))
        except Exception as exc:  # noqa: BLE001 — one bad paper is not the batch
            logger.warning("paper capture: %s failed (%s)",
                           paper.get("doi") or paper.get("title"), exc)
            out.append({"state": "failed", "reason": str(exc)[:200]})
    return out


def _open_copy(paper: Dict[str, Any]) -> None:
    """Ask where the open copy of a PubMed Central article is, and record it.

    A web search hands back the ARTICLE page — `…/articles/PMC12821576` — and
    fetching that stores a web page under the name of a paper. Building the PDF
    address instead does not work: NCBI refuses a programmatic GET of the
    article tree, and this repository's own manifest holds two such attempts
    filed as expired links. So the address is asked for.

    Also records `is_oa`, which is the one field that speaks to the licence and
    which a search result cannot supply.
    """
    keys = [k for k in (paper.get("refs") or []) if k.startswith(("pmc:", "pmid:"))]
    if not keys or paper.get("oa_url"):
        return
    try:
        from CoScientist.reporting import pmc

        kind, value = keys[0].split(":", 1)
        found = pmc.resolve({"kind": kind, "value": value})
    except Exception as exc:  # noqa: BLE001 — a resolver never breaks a run
        logger.debug("paper_capture: could not resolve %s (%s)", keys[0], exc)
        return
    if not found:
        return
    if found.is_oa is not None and paper.get("is_oa") is None:
        paper["is_oa"] = found.is_oa
    if found.url:
        paper["oa_url"] = found.url


def _take(papers: List[Dict[str, Any]], scope, agent: str) -> List[Dict[str, Any]]:
    """Bring each paper home — re-filing a copy the session already has."""
    from CoScientist.reporting import paper_library as pl
    from CoScientist.reporting import session_files as sf

    outcomes: List[Dict[str, Any]] = []
    fetch: List[Dict[str, Any]] = []
    #: The slot in `outcomes` each queued paper answers for. Kept explicitly
    #: rather than looked up with `papers.index(paper)`: `index` compares by
    #: VALUE and returns the first match, so two records that are equal — or
    #: literally the same object, which `merge` can hand us — would both write
    #: into one slot and leave the other empty for the whole batch.
    slots: List[int] = []
    for paper in papers:
        _open_copy(paper)
        address = (paper.get("oa_url") or paper.get("presigned_url")
                   or paper.get("pdf_url") or "")
        known = sf.artifact_id_for_url(scope, address) if address else None
        if known:
            # Already downloaded by the artifact plugin, under a kind that says
            # nothing about what it is. One manifest write, no second request.
            record = sf.retag(scope, known, source_kind=pl.SOURCE_KIND,
                              label=str(paper.get("title") or "")[:120])
            outcomes.append(record or {"state": sf.STATE_STORED,
                                       "artifact_id": known})
        else:
            slots.append(len(outcomes))
            fetch.append(paper)
            outcomes.append(None)

    if fetch:
        fetched = mirror_papers(fetch, scope, agent)
        for slot, record in zip(slots, fetched):
            outcomes[slot] = record
    return [o or {} for o in outcomes]


class PaperCapturePlugin(BasePlugin):
    """Turn the papers a run reads into files the session holds."""

    def __init__(self, name: str = "paper_capture") -> None:
        super().__init__(name)

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):  # noqa: ANN001
        tool_name = getattr(tool, "name", None) or ""
        if tool_name not in _SEARCH_TOOLS + _PAPERS_TOOLS:
            return None
        try:
            from CoScientist.graph.session_scope import session_key
            from CoScientist.reporting import paper_library as pl

            records = papers_in(result, tool_name)
            if not records:
                return None
            agent = getattr(tool_context, "agent_name", None) or "ResearchAgent"
            wanted = pl.merge(tool_context.state, records, agent=agent)
            if not wanted:
                return None
            capped = wanted[:pl.MAX_PER_CALL]
            # Resolved on the loop: `session_key` writes the pair back into ADK
            # state, and the downloads run in a worker thread.
            scope = session_key(tool_context)
            outcomes = await asyncio.to_thread(_take, capped, scope, agent)
            for paper, outcome in zip(capped, outcomes):
                pl.note_stored(tool_context.state, paper, outcome)
            held = sum(1 for o in outcomes if o.get("artifact_id"))
            logger.info(
                "paper_capture: %s → %d paper(s) known, %d handled, %d held%s",
                tool_name, len(pl.library(tool_context.state)), len(capped),
                held,
                f" ({len(wanted) - len(capped)} over the per-call cap)"
                if len(wanted) > len(capped) else "")
        except Exception as exc:  # noqa: BLE001 — capture must never break a call
            logger.warning("paper_capture failed on %s: %s", tool_name, exc)
        return None  # never decide whether the agent's own callbacks run


__all__ = ["PaperCapturePlugin", "is_paper_link", "papers_in", "result_items",
           "mirror_papers"]
