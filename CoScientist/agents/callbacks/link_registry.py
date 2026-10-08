"""Registry for the links a user puts in their request.

A URL that lives only in the conversation text gets recopied by a model at
every hand-off — orchestrator to worker, worker to sandbox — and reproducing an
80-character presigned URL verbatim is precisely what models are unreliable at:
the query string is clipped, two links merge into one, a plausible-but-wrong
path appears. Prompt wording does not fix this, because the failure is in token
reproduction rather than in instruction following.

So a model never writes a URL. It writes a short reference — ``[[link7f3a]]`` —
and CODE substitutes the real URL at the moment the text leaves the agent: in
the arguments of every tool call (``before_tool``) and in the agent's own answer
(``after_model``). What travels between agents is therefore always a URL that
code wrote, never one a model retyped.

That is what makes this work identically in-process and over A2A. The seam
between two agents is the same in both modes — the ``request`` argument of an
``AgentTool`` call — because A2A only swaps the wrapped agent for a
``RemoteA2aAgent``. Nothing here depends on session state crossing a process
boundary, so it also works for peers that have no ADK session at all: the
OpenHands sandbox, an MCP server, a non-ADK A2A service.

The receiving agent then re-extracts those URLs into its OWN registry and gets
them rendered back into its prompt via ``{links_context?}``, so it can use
references for its own outbound calls. Re-extracting at every hop is safe
precisely BECAUSE the incoming URLs were written by code: there is no mistyped
link to enshrine. And because an id is a digest of the url (see
``link_id_for``), an agent that meets the same link independently derives the
same id for it — no coordination, no shared counter, no state crossing a
process boundary. What travels between agents is still the url, not the id;
the matching ids are a convenience for reading traces, not a channel.

Extraction is not limited to the incoming message, either. A URL the agent is
about to SEE is a URL it might need to reference, wherever it came from: the
dataset the user attached in the web UI, a fact recalled from a prior session's
memory, an Evidence node already committed to the graph. All of those get
rendered into the prompt by other before_agent callbacks before this one runs
(see ``_CONTEXT_SOURCE_KEYS``), and every one of them is scanned too — otherwise
the model would see a raw URL sitting right there in its context with no
reference to call it by, and type it out anyway.

Telling two links apart is a separate problem from keeping them, and how it is
solved depends on where the link came from. A link in prose gets a ``mention``:
the slice of the sentence around it, with the other links in it replaced by
their references. "training set — [[link7f3a]], test — [[link91c2]]" is what
makes the two distinguishable; a bare pair of URLs is not.

A link a tool returned has no sentence around it, but it has a position in the
result, which says the same thing in fewer characters — so it gets an
``origin`` instead: ``run_sandbox_task.watch_url`` against
``run_sandbox_task.vscode_url``, where the URLs alone are two `web page`s on
one host and nothing tells the live console from the editor.

Five callbacks, on the two directions text moves:

INGRESS — what the model is allowed to see:
  * ``redact_link_urls`` — before_model. Swaps every registered URL back to its
    reference in the copy of the conversation handed to the model, so no raw
    URL is ever sitting in front of it to be copied. The session's events and
    the registry keep the real URL.
  * ``user_links`` — before_agent. Extracts every URL in the incoming message
    AND in the other context blocks already rendered for this turn into the
    registry, and renders the table into ``state['links_context']``. Runs on
    every agent, root or not — must run LAST in the before_agent list. This is
    a snapshot taken once, before the agent's own turn begins.

EGRESS — what leaves the agent, where the real URL is required:
  * ``resolve_link_refs`` — before_tool. Expands ``[[linkXXXX]]`` anywhere in a
    string argument, and repairs a URL the model retyped short.
  * ``register_tool_result_links`` — after_tool. The snapshot's counterpart: a
    URL a TOOL returned mid-turn (a sandbox artifact, a generated figure) gets
    registered too, so the agent's next model call already has a reference for
    it instead of having to retype what it just read.
  * ``expand_link_refs`` — after_model. Expands ``[[linkXXXX]]`` AND repairs a
    retyped URL in the agent's own response — the same repair
    ``resolve_link_refs`` does for tool arguments, applied here so a link
    handed to the orchestrator's caller (a report, an ``output_key``, a final
    answer) gets it too.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

USER_LINKS_STATE_KEY = "user_links"
LINKS_CONTEXT_STATE_KEY = "links_context"

# How much of the user's sentence to keep on each side of a link.
MENTION_RADIUS = 110

_TRAILING_PUNCT = ".,;:!?…»«\"'“”„<>"
_URL_CHARS = r"[^\s<>\"'`\[\]{}\\^|]"
_EXTRA_URL_SCHEMES = (
    () if os.getenv("LINK_REGISTRY_EXTRA_SCHEMES", "1") == "0" else ("s3", "gs")
)
_EXTRA_SCHEME_SRC = (
    r"(?:%s)://" % "|".join(_EXTRA_URL_SCHEMES) if _EXTRA_URL_SCHEMES else r"(?!)"
)
_SCHEMED_SRC = r"\b(?:https?://|%s|www\.)" % _EXTRA_SCHEME_SRC + _URL_CHARS + "+"
_BARE_TLDS = (
    "com", "org", "edu", "gov", "mil", "int",
    "ru", "ua", "kz", "uk", "de", "fr", "es", "pt", "br", "mx", "jp", "cn",
    "kr", "ca", "au", "nz", "tr", "gr", "dk", "fi", "cz", "hu", "ro",
    "xyz", "cloud", "tech", "online", "wiki", "science", "software", "рф",
)
_PUNYCODE_TLD = r"xn--[a-z0-9]{2,}"
_PATHED_TLDS = _BARE_TLDS + (
    "io", "ai", "co", "net", "dev", "app", "me", "info", "biz", "site",
    "us", "in", "it", "is", "be", "at", "no", "se", "nl", "ch", "pl", "eu",
    "sh", "run", "page", "tv", "cc", "id", "to", "ly", "one", "live",
)
_LABEL = r"[^\W_](?:[\w-]{0,61}[^\W_])?"
_BARE_HOST_SRC = (
    r"(?<![\w@.+-])(?:%s\.)+"
    r"(?:(?:%s|%s)(?!\.?\w)|(?:%s)(?=[/?#]))(?:[/?#]%s*)?"
    % (_LABEL, "|".join(_BARE_TLDS), _PUNYCODE_TLD,
       "|".join(_PATHED_TLDS), _URL_CHARS)
)

_URL_RE = re.compile("(?i)" + _SCHEMED_SRC)
_URL_RE_BARE = re.compile("(?i)(?:%s)|(?:%s)" % (_SCHEMED_SRC, _BARE_HOST_SRC))
LINK_ID_PREFIX = "link"
_ID_HEX_LEN = 4
_LINK_REF_RE = re.compile(r"\[\[\s*(link[0-9a-f]{%d,})\s*\]\]" % _ID_HEX_LEN,
                          re.IGNORECASE)


def link_ref(link_id: str) -> str:
    """The reference form a model writes for ``link_id``."""
    return f"[[{link_id}]]"


def link_id_for(url: str, taken: Optional[Dict[str, Any]] = None) -> str:
    """The id for ``url``, derived from the URL rather than from a counter.

    Keyed on the NORMALISED url, so the same link written two ways still lands
    on one id. ``taken`` is the registry to avoid clashing inside: on the rare
    digest collision between two genuinely different urls, the id is length-
    ened until it is free. That is deterministic given the same registry, but
    it is the one case where two agents holding different link sets can end up
    naming one url differently — correctness (never two urls under one id)
    wins over cosmetic agreement.
    """
    normalized = normalize_url(url)
    digest = hashlib.blake2s(normalized.encode("utf-8")).hexdigest()
    for length in range(_ID_HEX_LEN, len(digest) + 1, 2):
        candidate = f"{LINK_ID_PREFIX}{digest[:length]}"
        entry = (taken or {}).get(candidate)
        if entry is None or entry.get("normalized") == normalized:
            return candidate
    return f"{LINK_ID_PREFIX}{digest}"


_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "yclid", "_ga", "ref_src",
})

_ARCHIVE_EXT = (".zip", ".tar.gz", ".tgz", ".tar", ".tar.bz2", ".7z", ".rar", ".gz", ".bz2")
_TABLE_EXT = (".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".jsonl", ".ndjson", ".h5", ".hdf5", ".npy", ".npz")
_TEXT_EXT = (".txt", ".json", ".yaml", ".yml", ".xml", ".log", ".md")
_CHEM_BIO_EXT = (
    ".pdb", ".ent", ".cif", ".mmcif", ".sdf", ".mol", ".mol2",
    ".fasta", ".fa", ".fna", ".faa", ".smi", ".smiles", ".xyz", ".mae",
)
_MODEL_EXT = (".pt", ".pth", ".bin", ".onnx", ".safetensors", ".ckpt", ".pkl", ".pickle")
_CODE_EXT = (".py", ".sh", ".bash", ".r", ".ipynb")
_DOC_EXT = (".pdf", ".doc", ".docx")
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp")
_PAPER_HOSTS = (
    "doi.org", "arxiv.org", "biorxiv.org", "medrxiv.org", "chemrxiv.org",
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "openalex.org",
    "sciencedirect.com", "springer.com", "nature.com", "wiley.com",
    "acs.org", "rsc.org", "pubs.acs.org",
)
_REPO_HOSTS = ("github.com", "gitlab.com", "bitbucket.org")
_BLOB_HOSTS = ("amazonaws.com", "storage.googleapis.com", "drive.google.com",
               "dropbox.com", "figshare.com", "zenodo.org")


# ── URL parsing ──────────────────────────────────────────────────────────────
def _trim_trailing(url: str) -> str:
    """Drop the punctuation that ended the sentence, not the URL."""
    while url:
        tail = url[-1]
        if tail == ")":
            if url.count("(") >= url.count(")"):
                break
            url = url[:-1]
            continue
        if tail in _TRAILING_PUNCT:
            url = url[:-1]
            continue
        break
    return url


def with_scheme(url: str) -> str:
    """`www.x.org/a` → `https://www.x.org/a`; anything schemed is returned as is."""
    return url if "://" in url else f"https://{url}"


def find_urls(text: str, bare_hosts: bool = False) -> List[Tuple[str, int, int]]:
    """Every URL in ``text`` as ``(url, start, end)`` spans, in order.

    ``bare_hosts`` additionally matches a scheme-less host — `example.com`,
    `huggingface.co/datasets/x`. Off by default, and turned on in exactly two
    places: the incoming message, which is the only text that can carry a link
    nobody has normalised yet, and the matching done by redaction and egress
    repair, where a span is inert unless the url is ALREADY registered.

    Everything else is scanned with it off — a tool result, a rendered context
    block — because a url that code wrote always carries its scheme, while a
    line of code that merely looks like a host does not. What bounds the cost
    of the loose read is not who wrote the text but `_BARE_TLDS`: the tiering
    is what keeps `scipy.io` and `README.md` out of the registry.
    """
    out: List[Tuple[str, int, int]] = []
    pattern = _URL_RE_BARE if bare_hosts else _URL_RE
    for match in pattern.finditer(text or ""):
        url = _trim_trailing(match.group(0))
        if url:
            out.append((url, match.start(), match.start() + len(url)))
    return out


def normalize_url(url: str) -> str:
    """Dedup key: the same link written two ways must collapse to one id.

    Case-folds the host, drops `www.`, a trailing slash, the fragment and the
    tracking parameters, and sorts what query parameters remain. A Cyrillic
    host is folded to its punycode form.
    """
    parts = urlsplit(with_scheme(url.strip()))
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if not host.isascii():
        # The port is split off first: `idna` encodes per label, and handed
        # it produces a punycode label for that whole string.
        name, sep, port = host.rpartition(":")
        if not (sep and port.isdigit()):
            name, sep, port = host, "", ""
        try:
            host = name.encode("idna").decode("ascii") + sep + port
        except Exception:  # noqa: BLE001 — a name idna rejects (userinfo, an
            pass          # underscore) simply keeps the form it came in.
    path = parts.path.rstrip("/") or "/"
    query = sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    )
    return urlunsplit((parts.scheme.lower(), host, path, urlencode(query), ""))


def classify_url(url: str) -> Tuple[str, str]:
    """``(role, label)`` for one URL, from its shape alone.

    Derived by code from the URL rather than asked of a model, because the whole
    point is that this description cannot drift from what the link actually is.
    """
    parts = urlsplit(with_scheme(url))
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = unquote(parts.path)
    segments = [s for s in path.split("/") if s]
    name = segments[-1] if segments else ""
    lowered = path.lower()
    query_lower = parts.query.lower()

    if lowered.endswith(_ARCHIVE_EXT):
        return "dataset archive", name
    if lowered.endswith(_TABLE_EXT):
        return "data file", name
    if lowered.endswith(_CHEM_BIO_EXT):
        return "chemical/biological data", name
    if lowered.endswith(_MODEL_EXT):
        return "model checkpoint", name
    if lowered.endswith(_TEXT_EXT):
        return "text file", name
    if lowered.endswith(_CODE_EXT):
        return "code script", name
    if lowered.endswith(".pdf"):
        return "PDF document", name
    if lowered.endswith(_DOC_EXT):
        return "document", name
    if lowered.endswith(_IMAGE_EXT):
        return "image file", name
    if any(host == h or host.endswith("." + h) for h in _REPO_HOSTS):
        return "code repository", "/".join(segments[:2]) or host
    if host.endswith("huggingface.co"):
        return "HuggingFace resource", "/".join(segments[:3]) or host
    if any(host == h or host.endswith("." + h) for h in _PAPER_HOSTS):
        return "paper", path.strip("/") or host
    if any(host.endswith(h) for h in _BLOB_HOSTS) or "x-amz-" in query_lower or "x-goog-" in query_lower:
        return "file download", name or host
    clean_path = path.strip("/")
    if clean_path:
        label = f"{host}/{clean_path}" if len(clean_path) <= 40 else (f"{host}/…/{name}" if name else host)
    else:
        label = host
    return "web page", label


# ── registry ─────────────────────────────────────────────────────────────────
def _is_truncation(candidate: str, full: str) -> bool:
    """Is ``candidate`` a shortened rendering of the registered ``full`` URL?"""
    if candidate == full:
        return False
    for ellipsis in ("...", "…"):
        if ellipsis in candidate:
            head = candidate.split(ellipsis, 1)[0]
            return len(head) >= 15 and full.startswith(head)
    if "?" not in full:
        # Without a query string there is nothing a model plausibly drops, and
        # treating a path prefix as a truncation would rewrite /dataset into
        # /dataset/train.csv.
        return False
    if full.split("?", 1)[0] == candidate.rstrip("?"):
        return True   # query string dropped wholesale
    return "?" in candidate and full.startswith(candidate) and len(candidate) >= 20


def _canonical(url: str, registry: Dict[str, Any]) -> Optional[str]:
    """The registered URL this one means, or None if it is not one of ours."""
    normalized = normalize_url(url)
    for entry in registry.values():
        if entry.get("normalized") == normalized:
            return entry["url"]
    candidate = with_scheme(url)
    matches = [e["url"] for e in registry.values()
               if _is_truncation(candidate, e["url"])]
    return matches[0] if len(matches) == 1 else None


def _mention_map(text: str, spans: List[Tuple[str, int, int]],
                 id_of: Dict[str, str]) -> Dict[str, str]:
    """The sentence around each link, with the links shown as their references.

    Built from one masked copy of the message so a fragment holding two links
    reads "train — [[link7f3a]], test — [[link91c2]]" — which is the mapping an
    agent needs, and the one thing a bare list of URLs cannot convey.
    """
    pieces: List[str] = []
    positions: Dict[str, int] = {}
    cursor = 0
    for url, start, end in spans:
        pieces.append(text[cursor:start])
        link_id = id_of.get(normalize_url(url))
        if not link_id:
            # Check canonical for truncated URLs
            for norm_key, lid in id_of.items():
                if _is_truncation(url, norm_key):
                    link_id = lid
                    break
        if link_id:
            positions.setdefault(link_id, sum(len(p) for p in pieces))
            pieces.append(link_ref(link_id))
        else:
            pieces.append(url)
        cursor = end
    pieces.append(text[cursor:])
    masked = "".join(pieces)

    mentions: Dict[str, str] = {}
    for link_id, at in positions.items():
        left = max(0, at - MENTION_RADIUS)
        right = min(len(masked), at + len(link_ref(link_id)) + MENTION_RADIUS)
        snippet = " ".join(masked[left:right].split())
        if left > 0:
            snippet = "…" + snippet
        if right < len(masked):
            snippet = snippet + "…"
        mentions[link_id] = snippet
    return mentions


def register_user_links(state: Any, text: str, bare_hosts: bool = False,
                        with_mentions: bool = True) -> Dict[str, Any]:
    """Extract every URL in ``text`` into the registry; return the registry.

    Runs for EVERY agent, not only the root — ``text`` is the incoming message
    plus whatever else is already rendered into this turn's context (see
    ``user_links``). Below the root the message's URLs were written by
    ``resolve_link_refs`` rather than by a model — so there is no mistyped link
    to enshrine, and the receiver ends up able to use references for its own
    outbound calls.

    Idempotent by normalised URL, so re-running over the same message (a retry,
    a second turn quoting the first) never renumbers a link that is already in
    this agent's table.

    ``bare_hosts`` is passed straight to ``find_urls``: on for the incoming
    message, off for text that was rendered from state or returned by a tool.

    ``with_mentions`` is off for the same machine-written text. A mention is a
    slice of the SENTENCE around a link, and it earns its place when a person
    wrote that sentence — "обучающая — [[link7f3a]], тестовая — [[link91c2]]"
    is the only thing that tells two presigned urls apart. Slicing a search
    tool's JSON body yields no such thing, only `…\"follow_up_questions\":null`
    around the url, repeated for every result. The role and label already say
    what those links are.
    """
    registry: Dict[str, Any] = dict(state.get(USER_LINKS_STATE_KEY) or {})
    spans = find_urls(text, bare_hosts=bare_hosts)
    if not spans:
        return registry

    id_of = {entry["normalized"]: link_id for link_id, entry in registry.items()}
    fresh: List[str] = []
    for url, _start, _end in spans:
        normalized = normalize_url(url)
        if normalized in id_of:
            continue
        canonical = _canonical(url, registry)
        if canonical:
            canonical_norm = normalize_url(canonical)
            if canonical_norm in id_of:
                id_of[normalized] = id_of[canonical_norm]
                continue
        link_id = link_id_for(url, registry)
        id_of[normalized] = link_id
        role, label = classify_url(url)
        registry[link_id] = {
            "id": link_id,
            "url": with_scheme(url),
            "normalized": normalized,
            "role": role,
            "label": label,
            # How this link is told apart from its neighbours: the sentence a
            # person wrote around it, or the field a tool returned it in. One
            # of the two, never both — see `_describe_by_origin`.
            "mention": "",
            "origin": "",
        }
        fresh.append(link_id)

    for link_id, mention in (_mention_map(text, spans, id_of) if with_mentions
                             else {}).items():
        entry = registry.get(link_id)
        # Only the message that introduced a link describes it; a later turn
        # quoting the same URL must not overwrite the sentence that gave it
        # its meaning.
        if entry is not None and not entry.get("mention"):
            entry["mention"] = mention

    state[USER_LINKS_STATE_KEY] = registry
    if fresh:
        logger.info("user_links: registered %s (%d total)",
                    ", ".join(fresh), len(registry))
    return registry


def render_links_block(registry: Dict[str, Any]) -> str:
    """The `{links_context?}` prompt section: every link, every agent, every turn.

    Deliberately omits the URL itself. Telling a model not to retype a string
    that is sitting right there in its own context is a request it loses to
    the text in front of it; leaving the string out is not a request at all —
    there is nothing to copy, so `[[linkN]]` is the only way to refer to the
    link that exists. The real URL still lives in the registry and is
    substituted in mechanically by `resolve_link_refs` / `expand_link_refs`;
    it is simply never rendered into anything a model reads.
    """
    if not registry:
        return ""
    # Registration order — ids are digests now, so there is no number to sort
    # by, and the order a link was met in is the most useful one anyway.
    ordered = list(registry.values())
    lines = [
        f"## Links available in this task ({len(ordered)})",
        "The actual URLs are not shown — refer to a link by its reference, "
        "exactly as written below, anywhere you would have written the link: in "
        "prose, in a task you delegate, in a tool argument, in your answer. It "
        "is substituted for the real URL before your text leaves this agent.",
        "",
    ]
    # A short request produces the same window for every link. Print it once as
    # the shared quote rather than under each entry — same information, and the
    # per-entry lines stay scannable.
    mentions = {e.get("mention", "") for e in ordered}
    shared = mentions.pop() if len(mentions) == 1 else ""
    if shared:
        lines += [f"The request said (links shown as their references): \"{shared}\"", ""]

    for entry in ordered:
        label = entry.get("label") or ""
        head = f"- **{link_ref(entry['id'])}** — {entry.get('role', 'link')}"
        if label:
            head += f" `{label}`"
        lines.append(head)
        # A link that HAS a sentence is described by it, whether that sentence
        # is printed here or once above as the shared quote — falling through
        # to the origin would describe it twice, in two different registers.
        mention = entry.get("mention")
        if mention and not shared:
            lines.append(f"  - mentioned as: \"{mention}\"")
        elif not mention and entry.get("origin"):
            lines.append(f"  - returned as `{entry['origin']}`")
    lines.append("")
    return "\n".join(lines)


# ── callbacks ────────────────────────────────────────────────────────────────
def _user_text(callback_context: CallbackContext) -> str:
    content = getattr(callback_context, "user_content", None)
    if content is None or not getattr(content, "parts", None):
        return ""
    return "\n".join(part.text for part in content.parts if getattr(part, "text", None))


# Other before_agent callbacks running earlier in the SAME chain render these
# state keys into text the agent is about to see in its own prompt — and any of
# them can legitimately carry a real URL that no model typed: the dataset the
# user attached in the web UI (`dataset_context`), a fact recalled from a prior
# session (`graph_root`'s GLOBAL KNOWLEDGE MEMORY), an Evidence/Conclusion
# already committed to the graph (`research_context`). Skipping them would
# leave the model with no `[[linkN]]` for a link sitting right there in its
# context, so — despite its instructions — it retypes the raw URL by hand,
# which is exactly the failure this registry exists to prevent.
#
# Reading them here is only correct because `user_links` runs LAST in every
# agent's `before_agent` list (system.yaml) — by the time it runs,
# inject_graph_root / inject_research_context / inject_dataset_context have
# already populated their keys for THIS turn.
_CONTEXT_SOURCE_KEYS = ("graph_root", "research_context", "dataset_context")


def _context_text(state: Any) -> str:
    return "\n".join(str(state[key]) for key in _CONTEXT_SOURCE_KEYS if state.get(key))


def user_links(callback_context: CallbackContext) -> None:
    """before_agent: extract every link in this turn's context and render the table.

    The same on every agent — there is no root special case, so nothing here
    depends on which agent carries `root: true` in the current start mode.
    "This turn's context" is the incoming message plus whatever the earlier
    before_agent callbacks already rendered (see `_CONTEXT_SOURCE_KEYS`) — not
    only the message, so a link is registered wherever the agent can see it,
    not only where the caller happened to repeat it.

    Both halves are best-effort: a run must never die because of a URL, and
    since this now executes on every agent, a single malformed registry entry
    would otherwise take down the whole system rather than one call.
    """
    state = callback_context.state
    try:
        # Scanned separately, and only the message is read loosely. At the
        # root that message is the user's, typed in whatever form they felt
        # like — `example.com` included. Below the root it is the caller's
        # task text, whose urls `resolve_link_refs` already schemed; it is
        # still read loosely because it is also the one place a link can
        # arrive that nothing in this process has normalised — the caller may
        # have quoted a scheme-less host out of its own prose, and over A2A
        # the caller may not run this registry at all.
        #
        # The context blocks get no such treatment: they were rendered from
        # state by code, so their urls always carry a scheme, and reading them
        # loosely would only buy the chance of registering a line of code that
        # happens to look like a host.
        message = _user_text(callback_context)
        if message:
            register_user_links(state, message, bare_hosts=True)
        context = _context_text(state)
        if context:
            register_user_links(state, context)
    except Exception as exc:  # noqa: BLE001
        logger.error("user_links capture failed: %s", exc)
    try:
        state[LINKS_CONTEXT_STATE_KEY] = render_links_block(
            state.get(USER_LINKS_STATE_KEY) or {}
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("user_links render failed: %s", exc)
        state[LINKS_CONTEXT_STATE_KEY] = ""
    return None


def expand_refs(text: str, registry: Dict[str, Any]) -> str:
    """Replace every ``[[linkN]]`` in ``text`` with the URL it refers to.

    An unknown reference is left as written rather than dropped: a visible
    `[[link7f3a]]` in the output is a legible bug report, whereas silently deleting
    it would hand the next agent a sentence with the object missing.
    """
    def substitute(match: "re.Match[str]") -> str:
        entry = registry.get(match.group(1).lower())
        return entry["url"] if entry else match.group(0)

    return _LINK_REF_RE.sub(substitute, text)


def _resolve_value(value: str, registry: Dict[str, Any]) -> str:
    """One outbound string: references expanded, then retyped URLs repaired."""
    value = expand_refs(value, registry)
    # Loose matching is safe here: a span is only touched when it resolves to
    # something already IN the registry, so a bare `scipy.io` nobody
    # registered leaves exactly as the model wrote it — while the user's own
    # `example.com`, which IS registered, goes out with its scheme.
    spans = find_urls(value, bare_hosts=True)
    if not spans:
        return value
    # Right to left, so an earlier span's offsets stay valid as we splice.
    for url, start, end in reversed(spans):
        canonical = _canonical(url, registry)
        if canonical and canonical != url:
            value = value[:start] + canonical + value[end:]
    return value


def _map_strings(value: Any, transform: Any) -> Any:
    """``value`` of ANY shape with ``transform`` applied to every string in it.

    Both directions need this, and for the same reason: a URL is rarely a
    whole value. Outbound, ``tavily_extract`` takes ``urls=["…"]`` and an MCP
    tool takes a nested options object — a reference sitting one level down
    used to travel untouched, and the tool was handed the literal
    ``[[linkcff8]]`` (`Validation Error: Invalid URL format`). Inbound, a
    tool's result is a JSON structure, and the urls in it sit at whatever
    depth the tool chose.

    Returns the ORIGINAL object when nothing changed, so a caller can test for
    a rewrite with ``is`` and never has to run ``!=`` over a value of unknown
    type. Containers are rebuilt rather than edited in place: a tuple cannot
    be edited, and a list that something else still holds should not be.
    """
    if isinstance(value, str):
        if not value.strip():
            return value
        new = transform(value)
        return new if new != value else value
    if isinstance(value, dict):
        items = {k: _map_strings(v, transform) for k, v in value.items()}
        return items if any(items[k] is not v for k, v in value.items()) else value
    if isinstance(value, (list, tuple)):
        items = [_map_strings(v, transform) for v in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        return tuple(items) if isinstance(value, tuple) else items
    return value


def _resolve_arg(value: Any, registry: Dict[str, Any]) -> Any:
    """One outbound argument, with every reference in it expanded."""
    return _map_strings(value, lambda text: _resolve_value(text, registry))


def resolve_link_refs(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> None:
    """before_tool: expand ``[[linkXXXX]]`` and repair a URL typed short.

    This is the egress point that matters most: the ``request`` argument of an
    ``AgentTool`` call is how one agent hands work to another, and it is the
    same argument whether the callee runs in this process or behind A2A. The
    receiver therefore always gets a real URL, even when it is the OpenHands
    sandbox or some other peer that knows nothing about this registry.

    Expansion is safe in ANY argument, prose included, because `[[…]]` cannot
    be confused with ordinary text the way a bare `l1` could — and it reaches
    every string in an argument, not only an argument that IS one: the list in
    ``urls=[…]`` and the strings inside a nested options object are where a
    reference most often sits (see `_resolve_arg`).

    ADK hands this callback the same dict it then calls the tool with, so
    editing in place is what reaches the tool. Returning None always: this
    corrects a call, it never blocks one.
    """
    registry = tool_context.state.get(USER_LINKS_STATE_KEY) or {}
    if not registry or not isinstance(args, dict):
        return None
    for key, value in list(args.items()):
        try:
            resolved = _resolve_arg(value, registry)
        except Exception as exc:  # noqa: BLE001 — a bad URL must not kill the call
            logger.error("resolve_link_refs failed on %s: %s", key, exc)
            continue
        if resolved is not value:
            args[key] = resolved
            logger.info("resolve_link_refs: %s(%s=…) rewritten from the link registry",
                        getattr(tool, "name", "?"), key)
    return None


def _flatten_tool_result(value: Any) -> str:
    """One text blob to scan for URLs — whatever shape a tool's result is."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if isinstance(k, str):
                parts.append(k)
            parts.append(_flatten_tool_result(v))
        return "\n".join(parts)
    if isinstance(value, (list, tuple, set)):
        return "\n".join(_flatten_tool_result(v) for v in value)
    try:
        if hasattr(value, "model_dump"):
            return _flatten_tool_result(value.model_dump())
    except Exception:  # noqa: BLE001
        pass
    return str(value)


def _to_refs(text: str, ref_of: Dict[str, str]) -> str:
    """``text`` with every REGISTERED url swapped for its reference.

    Matched by span and by normalised url rather than by substring, because
    the form a url takes in the text is not the form the registry stored: the
    user's `example.com` is registered as `https://example.com`, and a plain
    `str.replace` of the stored string finds nothing in the very message the
    link came from. Normalising both sides also makes a trailing slash, a
    `www.`, a case-folded host or an utm parameter stop deciding whether a
    link is redacted.

    A span that is not in the registry is left exactly as written — there is
    no reference to put in its place, and dropping it would cost the model
    information it cannot get back. Right to left, so the earlier spans'
    offsets stay valid as we splice.
    """
    for url, start, end in reversed(find_urls(text, bare_hosts=True)):
        ref = ref_of.get(normalize_url(url))
        if ref:
            text = text[:start] + ref + text[end:]
    return text


def _redact_part(part: Any, ref_of: Dict[str, str]) -> bool:
    """Swap urls for references in ONE part; True if anything changed.

    A part is text OR a function response, and both carry links. The message
    is the obvious half; the function response is the one that was being
    missed, and it is the bigger leak of the two — a search tool comes back
    with a JSON body full of result urls, `part.text` is None for it, and the
    model was reading every one of them raw next to a table that pointedly
    withheld the same urls.
    """
    text = getattr(part, "text", None)
    if text:
        swapped = _to_refs(text, ref_of)
        if swapped != text:
            part.text = swapped
            return True
        return False
    function_response = getattr(part, "function_response", None)
    response = getattr(function_response, "response", None)
    if response is None:
        return False
    swapped = _map_strings(response, lambda value: _to_refs(value, ref_of))
    if swapped is response:
        return False
    function_response.response = swapped
    return True


def redact_link_urls(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """before_model: swap every registered URL for its reference in what the
    model is about to read.

    The counterpart to the egress callbacks, and the piece that actually stops
    a model retyping a link. The table in the system prompt withholds URLs, but
    the MESSAGE does not: the user types a link by hand, and — more often — we
    put one there ourselves, because `resolve_link_refs` expands
    ``[[link7f3a]]`` into the real URL in the outbound `request`, and that text
    becomes the callee's incoming message. So every agent was being handed a
    raw URL next to an instruction not to copy one, which is a contest the
    instruction loses.

    Expansion on the way out is still right — it is what makes a link survive
    A2A, the OpenHands sandbox and any non-ADK peer, none of which can resolve
    a digest. This just means the model never has to see the result.

    Covers both kinds of part — the message and a tool's function response
    (see `_redact_part`), since a search tool hands back a JSON body of result
    urls and that is where most of the raw links in a turn actually are.

    Rewrites ONLY `llm_request.contents`, which ADK builds per call as
    `copy.deepcopy(event.content)` (flows/llm_flows/contents.py). The session's
    events, the registry and everything the egress callbacks read are
    untouched, so what travels between agents is still the real URL.
    """
    registry = callback_context.state.get(USER_LINKS_STATE_KEY) or {}
    if not registry:
        return None
    ref_of: Dict[str, str] = {}
    for entry in registry.values():
        if not isinstance(entry, dict) or not entry.get("url") or not entry.get("id"):
            continue
        try:
            key = entry.get("normalized") or normalize_url(entry["url"])
        except Exception as exc:  # noqa: BLE001 — one bad entry, not a dead run
            logger.error("redact_link_urls skipped a registry entry: %s", exc)
            continue
        ref_of[key] = link_ref(entry["id"])
    if not ref_of:
        return None
    redacted = 0
    for content in llm_request.contents or []:
        for part in getattr(content, "parts", None) or []:
            try:
                redacted += int(_redact_part(part, ref_of))
            except Exception as exc:  # noqa: BLE001
                logger.error("redact_link_urls failed on a part: %s", exc)
    if redacted:
        logger.info("redact_link_urls: %s — %d part(s) now carry "
                    "references instead of raw URLs",
                    getattr(callback_context, "agent_name", "?"), redacted)
    return None


def _url_key_paths(value: Any) -> Dict[str, str]:
    """``{normalised url: the key path that held it}`` for a tool's result.

    What tells two links apart is the wording around them, and a tool result
    has no wording — but it has structure, which says the same thing in fewer
    characters. ``watch_url`` and ``vscode_url`` are exactly the distinction a
    model needs between the sandbox's two links, and both would otherwise
    render as `web page` on one host, indistinguishable.

    This replaces slicing the serialised JSON for a window of text, which for
    the sandbox happened to catch the neighbouring key and for a search tool
    caught `…\\"follow_up_questions\\":null…` — the same noise repeated under
    every result.

    A field that holds ONLY a url names that url; a paragraph that happens to
    mention it does not, so an exact match always wins over an embedded one.
    """
    found: Dict[str, Tuple[str, bool]] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            if not path:
                return
            for url, _start, _end in find_urls(node):
                normalized = normalize_url(url)
                exact = node.strip() == url
                current = found.get(normalized)
                if current is None or (exact and not current[1]):
                    found[normalized] = (path, exact)
        elif isinstance(node, dict):
            for key, child in node.items():
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                walk(child, f"{path}[{index}]")

    walk(value, "")
    return {normalized: path for normalized, (path, _exact) in found.items()}


def _describe_by_origin(registry: Dict[str, Any], tool_response: Any,
                        tool_name: str) -> None:
    """Record where each freshly returned link came back from.

    Written to ``origin`` rather than to ``mention`` because the two are not
    the same claim: a mention quotes the sentence a person wrote, an origin
    names the field a tool filled. Only the first tool to return a link
    describes it, on the same grounds that only the first sentence to mention
    one does.
    """
    paths = _url_key_paths(tool_response)
    if not paths:
        return
    for entry in registry.values():
        if not isinstance(entry, dict) or entry.get("origin"):
            continue
        path = paths.get(entry.get("normalized"))
        if path:
            entry["origin"] = f"{tool_name}.{path}" if tool_name else path


def register_tool_result_links(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Any
) -> None:
    """after_tool: register a URL a tool RETURNED, not one the model wrote.

    `user_links` runs once, before an agent's own turn begins — a snapshot of
    the incoming message and whatever context was already rendered. A link
    that first appears mid-turn, in what a tool comes back with — the sandbox's
    freshly uploaded artifact, an MCP tool's generated figure, a search
    result's article link — is invisible to that snapshot. Nothing else
    registers it, so without this hook the model has no `[[linkN]]` for a link
    it just read and can only pass it on by retyping it — the exact failure
    this registry exists to prevent, just arriving from the other direction.

    The registry and `{links_context?}` are re-rendered here, in the SAME
    turn, so the agent's very next model call already sees a reference for the
    link it just received, before it has any reason to write the raw URL out.

    Each link is described by WHERE in the result it came back — the tool and
    the key path, `run_sandbox_task.watch_url` — rather than by the slice of
    serialised JSON around it (see `_url_key_paths`).
    """
    try:
        text = _flatten_tool_result(tool_response)
        if not text or not find_urls(text):
            return
        state = tool_context.state
        before = set(state.get(USER_LINKS_STATE_KEY) or {})
        register_user_links(state, text, with_mentions=False)
        registry = state.get(USER_LINKS_STATE_KEY) or {}
        _describe_by_origin(registry, tool_response, getattr(tool, "name", ""))
        state[LINKS_CONTEXT_STATE_KEY] = render_links_block(registry)
        fresh = sorted(set(registry) - before)
        if fresh:
            logger.info("register_tool_result_links: %s returned %s",
                        getattr(tool, "name", "?"), ", ".join(fresh))
    except Exception as exc:  # noqa: BLE001 — capture must never break a tool call
        logger.error("register_tool_result_links failed: %s", exc)


def expand_link_refs(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> None:
    """after_model: expand ``[[linkXXXX]]`` AND repair a retyped URL in what the
    agent says.

    Tool arguments are not the only way text leaves an agent. An answer becomes
    a report, an ``output_key`` another agent reads, or the reply the user sees
    — none of which pass through `before_tool`. Without this, a reference the
    model wrote by following its instructions would reach a human as the literal
    string `[[link7f3a]]`.

    Uses the SAME `_resolve_value` as `resolve_link_refs` — not just
    `expand_refs` — so a model that ignores its instructions and retypes the
    URL directly (no `[[…]]` at all) still gets it corrected here if it comes
    out clipped or merged with another registered link. This is the ONLY
    egress point for an agent's own final answer: if this stopped at reference
    expansion, a hand-retyped link in a final answer would sail through
    unrepaired even though the identical mistake in a tool argument would not.

    Mutates the response in place and returns None, which tells ADK to carry on
    with the object we just edited. This only ever rewrites text; it never
    replaces or suppresses a response.
    """
    registry = callback_context.state.get(USER_LINKS_STATE_KEY) or {}
    if not registry:
        return None
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return None
    for part in parts:
        text = getattr(part, "text", None)
        if not text:
            continue
        try:
            resolved = _resolve_value(text, registry)
        except Exception as exc:  # noqa: BLE001
            logger.error("expand_link_refs failed: %s", exc)
            continue
        if resolved != text:
            part.text = resolved
            logger.info("expand_link_refs: %s rewrote a link in its answer "
                        "from the link registry", getattr(callback_context, "agent_name", "?"))
    return None
