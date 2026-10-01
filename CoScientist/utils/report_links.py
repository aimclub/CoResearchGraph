"""Rewrite S3 links in report markdown to local web-app links.

Agents paste raw S3 URLs into report prose. A presigned URL expires. A plain
endpoint URL points at the internal MinIO address, which a browser cannot
reach. An ``s3://bucket/key`` reference is not clickable at all.

This module replaces each of them with a relative
``/api/artifact/<bucket>/<key>`` link. The web app mints a fresh presigned URL
on each request to that route, so a report link never expires for the user and
never exposes the internal endpoint.

A link the module cannot parse is left untouched.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional, Set, Tuple
from urllib.parse import quote, unquote, urlparse

from CoScientist.utils.s3_refs import split_s3_uri

logger = logging.getLogger(__name__)

ARTIFACT_ROUTE = "/api/artifact"

#: The reference scheme the session store writes into graph attrs and report
#: prose. Spelled out rather than imported so this module stays importable from
#: ``session_files`` without a cycle; a test keeps the two in step.
SCHEME_PREFIX = "cos-artifact:"

# A URL in markdown or prose. Stops at whitespace, quotes, and the delimiters
# of a markdown link or image.
_HTTP_URL_RE = re.compile(r"https?://[^\s\"'<>\)\]]+")
_S3_URI_RE = re.compile(r"s3://[^\s\"'<>\)\]]+")

# Sentence punctuation after a bare URL is not part of the URL.
_TRAILING_PUNCTUATION = ".,;:"


def artifact_link(bucket: str, key: str) -> str:
    """Build the local link the report carries for one object."""
    return f"{ARTIFACT_ROUTE}/{quote(bucket, safe='')}/{quote(key, safe='/')}"


def session_artifact_link(user_id: str, session_id: str, artifact_id: str) -> str:
    """The link to a file this session mirrored into its own directory.

    Built from the scope that is *asking*, never from the scope that captured.
    That is what lets an imported bundle resolve ``cos-artifact:`` references
    under a new session id with nothing rewritten.
    """
    return (
        f"/api/users/{quote(user_id, safe='')}"
        f"/sessions/{quote(session_id, safe='')}"
        f"/artifacts/{quote(artifact_id, safe='')}"
    )


#: What an artifact is, for the purpose of writing it into prose.
KIND_FIGURE = "figure"
KIND_TABLE = "table"
KIND_FILE = "file"

_FIGURE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".avif")
_TABLE_EXTS = (".csv", ".tsv")


def _kind_of(name: str, media_type: Optional[str]) -> str:
    """figure / table / file, from the real name and the stored media type.

    The name is consulted first because it is the fact we are most sure of: a
    figure mirrored from MinIO routinely arrives with
    ``Content-Type: binary/octet-stream`` and would otherwise be filed as a
    generic download and never embedded.
    """
    low = str(name or "").lower()
    if low.endswith(_FIGURE_EXTS):
        return KIND_FIGURE
    if low.endswith(_TABLE_EXTS):
        return KIND_TABLE
    kind = str(media_type or "").split(";")[0].strip().lower()
    if kind.startswith("image/"):
        return KIND_FIGURE
    if kind in ("text/csv", "text/tab-separated-values"):
        return KIND_TABLE
    return KIND_FILE


def artifact_citation(
    scope: Optional[Tuple[str, str]], ref_or_id: object
) -> Optional[Dict[str, Any]]:
    """Everything needed to write one artifact into prose, or None.

    The single place that reads an artifact's NAME and its LINK, and it reads
    both out of the *same* manifest record. That is the whole point: four call
    sites used to decide the two halves independently, and they disagreed — a
    chip labelled ``metabolite_smiles.json`` whose href downloaded a PDF.

    Returns ``None`` when the bytes are not in this session's store, for the
    same reason :func:`resolve_ref` does: a link that 404s on click reads as a
    broken page rather than as an artifact nobody mirrored.
    """
    if not scope or not scope[0] or not scope[1]:
        return None

    from CoScientist.reporting.session_files import load_manifest, parse_ref

    text = str(ref_or_id or "").strip()
    artifact_id = parse_ref(text) or text
    if not artifact_id:
        return None
    record = load_manifest(scope[1], scope[0]).get(artifact_id)
    if not record or record.get("state") != "stored":
        return None

    name = str(record.get("filename") or record.get("label") or artifact_id)
    media_type = record.get("media_type")
    return {
        "id": artifact_id,
        "name": name,
        "kind": _kind_of(name, media_type),
        # The session-free form, for anything that gets persisted.
        "ref": f"{SCHEME_PREFIX}{artifact_id}",
        # A URL that works right now, for anything rendered on the spot.
        "href": session_artifact_link(scope[0], scope[1], artifact_id),
        "media_type": media_type,
        "label": str(record.get("label") or name),
        "source_tool": record.get("source_tool") or "",
    }


def artifact_markdown(citation: Optional[Dict[str, Any]], caption: str = "") -> str:
    """One artifact as markdown: an embed for a figure, a link for the rest.

    The destination is the session-free ``cos-artifact:`` reference, not a URL.
    This text is persisted — into ``report.md``, into the ``Report`` node, into
    an exported bundle — and a baked-in ``/api/users/<u>/sessions/<s>/...``
    would break the moment that bundle is opened under a new session id, which
    is the one thing the indirection exists to prevent.
    :func:`resolve_artifact_refs` turns it into a URL at the delivery boundary.

    ``caption`` is what a person calls the thing; the file name travels beside
    it so the reader can tell what will land in their downloads folder. Both
    come from the caller's own knowledge and the citation respectively — never
    from two different guesses about the same file.
    """
    if not citation:
        return ""
    ref, name = citation["ref"], citation["name"]
    title = str(caption or "").strip() or name
    if citation["kind"] == KIND_FIGURE:
        # The alt text is the caption: a broken embed then still says what the
        # picture was, which is what the reader needs when it does not load.
        # The file name goes underneath only when it adds something the heading
        # does not already say.
        tail = "" if title == name else f"\n\n`{name}`"
        return f"### {title}\n\n![{title}]({ref}){tail}"
    return f"### {title} — [{name}]({ref})" if title != name else f"### [{name}]({ref})"


def resolve_artifact_refs(
    text: str, scope: Optional[Tuple[str, str]] = None
) -> str:
    """Replace every ``cos-artifact:<id>`` in prose with a followable URL.

    The stored form carries no session on purpose, so an exported bundle opens
    on another machine under a new id with nothing rewritten. The price is that
    ``cos-artifact:`` is not a scheme any browser knows — and the chat's
    sanitizer drops an ``src`` it cannot classify — so the reference has to
    become a URL on the way out, built from the scope that is *asking*.

    Never raises: a message with an odd reference must still reach the user.
    """
    if not isinstance(text, str) or not text or SCHEME_PREFIX not in text:
        return text
    try:
        from CoScientist.reporting.session_files import find_refs

        seen = {}
        for artifact_id in find_refs(text):
            if artifact_id in seen:
                continue
            seen[artifact_id] = resolve_ref(f"{SCHEME_PREFIX}{artifact_id}", scope)
        for artifact_id, url in seen.items():
            if url:
                text = text.replace(f"{SCHEME_PREFIX}{artifact_id}", url)
        return text
    except Exception:  # noqa: BLE001 — never break a message over a reference
        logger.warning("resolve_artifact_refs failed; text unchanged", exc_info=True)
        return text


def resolve_ref(value: object, scope: Optional[Tuple[str, str]] = None) -> Optional[str]:
    """A browser-followable URL for one stored reference, or None.

    The one place that decides what is clickable. Returning None matters as much
    as returning a link: a research-graph attachment whose ``href`` is
    ``D:\\projects26\\...`` renders as an anchor that navigates nowhere, which
    reads like a bug in the page rather than a file that was never mirrored.
    Twelve of twelve attachments in a real session looked like that.
    """
    text = str(value or "").strip()
    if not text:
        return None

    from CoScientist.reporting.session_files import has_artifact, parse_ref

    artifact_id = parse_ref(text)
    if artifact_id:
        if not scope:
            return None
        # Ask the store, do not trust the string. A reference written by an
        # older run — or by code that mistook another namespace's id for ours —
        # names a file that was never stored, and a link that 404s on click
        # reads as a broken page rather than as an artifact nobody mirrored.
        if not has_artifact((scope[0], scope[1]), artifact_id):
            return None
        return session_artifact_link(scope[0], scope[1], artifact_id)

    if text.startswith("s3://"):
        parsed = split_s3_uri(text)
        if parsed:
            return artifact_link(*parsed)
        return None

    if text.startswith(("http://", "https://", "/api/")):
        return text

    # A local path, a bare filename, a free-text description: not a link.
    return None


def _configured_hosts() -> Set[str]:
    """The host:port pairs of the configured S3 endpoints, read at call time."""
    hosts = set()
    for var in ("S3__ENDPOINT_URL", "S3__EXTERNAL_ENDPOINT_URL"):
        value = os.getenv(var)
        if not value:
            continue
        netloc = urlparse(value).netloc
        if netloc:
            hosts.add(netloc)
    return hosts


def _bucket_key_from_url(url: str, known_hosts: Set[str]) -> Optional[Tuple[str, str]]:
    """Extract (bucket, key) from an S3 URL, or None when it is not one.

    A URL counts only when its host is one of the configured S3 endpoints.
    A signature alone is not enough: a presigned URL for a foreign bucket
    (an agent can quote one) would rewrite to a local link this S3 cannot
    resolve. Anything else is an ordinary web link.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    if parsed.netloc not in known_hosts:
        return None
    bucket, sep, key = parsed.path.lstrip("/").partition("/")
    if not sep or not bucket or not key:
        return None
    key = unquote(key)
    if ".." in key.split("/"):
        return None
    return bucket, key


def _mirrored_link(url: str, scope: Optional[Tuple[str, str]]) -> Optional[str]:
    """A link to our own copy of whatever this URL pointed at, if we have one.

    For the URLs the host gate above can never accept: a tool server's own
    MinIO is not this deployment's S3, so ``/api/artifact/`` could not re-sign
    it and rewriting it there would produce a link that 404s. But the capture
    plugin took the bytes while the signature was alive, so the session store
    can answer instead — and the signature in the text stops mattering.
    """
    if not scope or not scope[0] or not scope[1]:
        return None
    try:
        from CoScientist.reporting.session_files import artifact_id_for_url

        artifact_id = artifact_id_for_url((scope[0], scope[1]), url)
        return session_artifact_link(scope[0], scope[1], artifact_id) if artifact_id else None
    except Exception:  # noqa: BLE001
        return None


def _rewrite_http(match: "re.Match", known_hosts: Set[str],
                  scope: Optional[Tuple[str, str]] = None) -> str:
    url = match.group(0).rstrip(_TRAILING_PUNCTUATION)
    tail = match.group(0)[len(url):]
    parsed = _bucket_key_from_url(url, known_hosts)
    if parsed is None:
        # Not one of ours to re-sign. It may still be one we copied.
        mirrored = _mirrored_link(url, scope)
        return mirrored + tail if mirrored else match.group(0)
    bucket, key = parsed
    return artifact_link(bucket, key) + tail


def _rewrite_s3(match: "re.Match") -> str:
    uri = match.group(0).rstrip(_TRAILING_PUNCTUATION)
    parsed = split_s3_uri(uri)
    if parsed is None:
        return match.group(0)
    bucket, key = parsed
    if ".." in key.split("/"):
        return match.group(0)
    return artifact_link(bucket, key) + match.group(0)[len(uri):]


def remint_report_urls(
    text: str, scope: Optional[Tuple[str, str]] = None
) -> str:
    """Replace S3 links in report text with local ``/api/artifact/`` links.

    ``scope`` is optional and additive: with it, a URL whose host this
    deployment does not own is looked up in that session's artifact store and,
    if we mirrored it, replaced by a link to our copy. Without it the behaviour
    is exactly what it always was.

    Never raises: a report with an odd link must still reach the user.
    """
    if not isinstance(text, str) or not text:
        return text
    try:
        known_hosts = _configured_hosts()
        text = _HTTP_URL_RE.sub(lambda m: _rewrite_http(m, known_hosts, scope), text)
        text = _S3_URI_RE.sub(_rewrite_s3, text)
        return text
    except Exception:  # noqa: BLE001 — never break a report over a link
        logger.warning("remint_report_urls failed; returning the report unchanged", exc_info=True)
        return text


__all__ = [
    "ARTIFACT_ROUTE", "SCHEME_PREFIX",
    "KIND_FIGURE", "KIND_TABLE", "KIND_FILE",
    "artifact_link", "session_artifact_link", "resolve_ref",
    "artifact_citation", "artifact_markdown", "resolve_artifact_refs",
    "remint_report_urls",
]
