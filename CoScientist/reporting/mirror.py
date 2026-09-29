"""Bring an artifact's bytes home, at the moment its link still works.

This is the one place that turns somebody else's expiring URL into a file we
own. Both capture sites call it — the App-level plugin for ordinary tool calls
and the ``after_tool`` callback for the ones that run inside an AgentTool child.
They used to hold two copies of this logic, which is how they drifted: one
recorded to disk and the other did not.

Nothing here raises, and every path returns a record. The contract the capture
plugin already states — *capture must never break a tool call* — is what makes
that non-negotiable.

Failure is a record too. ``skipped``/``failed`` with a reason code, so a figure
that did not survive says why instead of being absent.
"""
from __future__ import annotations

import hashlib
import html
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from CoScientist.graph.session_scope import SessionKey, session_key
from CoScientist.reporting import session_files as sf

logger = logging.getLogger(__name__)

#: How many artifacts one tool call may mirror. Matches ``s3_refs._MAX_REFS`` —
#: a result carrying more references than this is a listing, not a deliverable.
MAX_PER_CALL = 50

_MB = 1024 * 1024


def _settings():
    from CoScientist.config import get_settings

    return get_settings().artifacts


# ── the gate ────────────────────────────────────────────────────────────────
def _blocked(key: SessionKey, cfg) -> Optional[str]:
    """Why this session may not mirror anything more, or None."""
    if not cfg.enabled:
        return sf.REASON_DISABLED
    if sf.stored_count(key) >= cfg.max_files_per_session:
        return sf.REASON_TOO_MANY
    if sf.stored_bytes(key) >= cfg.max_session_mb * _MB:
        return sf.REASON_SESSION_QUOTA
    return None


# ── fetching ────────────────────────────────────────────────────────────────
def _fetch(url: str, *, limit: int, timeout: int) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
    """``(payload, media_type, reason)`` — exactly one of payload/reason is set.

    Streamed and capped: a tool may hand back a link to a multi-gigabyte
    checkpoint, and reading it into memory to then reject it helps nobody.

    ``html.unescape`` because MCP servers escape ``&`` between query parameters,
    and the literal ``&amp;`` invalidates an AWS SigV4 signature — MinIO answers
    403 to a URL that would otherwise work. ``collect._download`` carries the
    same fix and the same comment.
    """
    import requests

    try:
        response = requests.get(html.unescape(url), stream=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("mirror: cannot reach %s (%s)", url[:120], exc)
        return None, None, sf.REASON_UNREADABLE

    try:
        if response.status_code in (404, 410):
            return None, None, sf.REASON_NOT_FOUND
        if response.status_code in (401, 403):
            # A 403 has two quite different meanings and they used to be
            # recorded as one. On a SIGNED link it is the usual shape of an
            # expired signature — the reason this code existed. On an ordinary
            # link it is the host refusing us, which is what NCBI does to a
            # programmatic GET of its article tree; two papers in a live session
            # were filed as expired links when nothing had expired.
            #
            # The signature is the discriminator, and it is in the URL.
            return None, None, (sf.REASON_LINK_EXPIRED if _is_signed(url)
                                else sf.REASON_REFUSED)
        if response.status_code >= 400:
            return None, None, f"download_failed:{response.status_code}"

        declared = response.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > limit:
            return None, None, sf.REASON_OVERSIZE

        chunks: List[bytes] = []
        total = 0
        for chunk in response.iter_content(65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > limit:
                return None, None, sf.REASON_OVERSIZE
            chunks.append(chunk)
        media_type = (response.headers.get("Content-Type") or "").split(";")[0].strip()
        return b"".join(chunks), media_type or None, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("mirror: cannot read %s (%s)", url[:120], exc)
        return None, None, sf.REASON_UNREADABLE
    finally:
        response.close()


#: What a capability-with-an-expiry looks like, across the storage services this
#: system meets. `collect._is_artifact_url` recognises the same two markers.
_SIGNED_MARKERS = ("x-amz-signature=", "x-amz-credential=", "x-goog-signature=",
                   "signature=", "&expires=", "?expires=", "se=", "sig=")


def _is_signed(url: Optional[str]) -> bool:
    """Whether this address carries a signature that can expire."""
    query = str(url or "").split("?", 1)[-1].lower() if "?" in str(url or "") else ""
    return any(marker in query for marker in _SIGNED_MARKERS)


def _read_local(path: Path, *, limit: int) -> tuple[Optional[bytes], Optional[str]]:
    try:
        if not path.is_file():
            return None, sf.REASON_UNREADABLE
        if path.stat().st_size > limit:
            return None, sf.REASON_OVERSIZE
        return path.read_bytes(), None
    except OSError as exc:
        logger.warning("mirror: cannot read %s (%s)", path, exc)
        return None, sf.REASON_UNREADABLE


# ── the entry point ─────────────────────────────────────────────────────────
def mirror_artifact(
    context: Any = None,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    url: Optional[str] = None,
    path: Any = None,
    payload: Optional[bytes] = None,
    bucket: Optional[str] = None,
    s3_key: Optional[str] = None,
    filename: str = "",
    label: str = "",
    tool: str = "",
    source_kind: str = "",
    agent: Optional[str] = None,
    mirror_off_host: bool = True,
) -> Dict[str, Any]:
    """Copy one artifact into the session's store. Always returns a record.

    Sources, in the order they are tried: bytes already in hand, a local file, a
    live URL, or a ``bucket``/``s3_key`` pair whose link is minted through the
    vault first.

    ``mirror_off_host=False`` keeps the copy in the session and nowhere else.
    The second address is normally a kindness — the artifact outlives the
    session — but for a file whose retention is a policy question, and a
    downloaded paper is one, that decision belongs at the call site that judged
    it, not in a global setting.
    """
    cfg = _settings()
    key = session_key(context, user_id=user_id, session_id=session_id)
    name = filename or _name_from(url, path)

    reason = _blocked(key, cfg)
    if reason:
        return sf.note(key, state=sf.STATE_SKIPPED, reason=reason, filename=name,
                       label=label, source_tool=tool, source_kind=source_kind,
                       source_url=url)

    limit = cfg.max_file_mb * _MB
    media_type: Optional[str] = None

    # Too big is a policy decision, not a breakage: it gets `skipped`, like the
    # quota gates above. Everything else at this stage really did fail.
    def _outcome(why: str) -> str:
        return sf.STATE_SKIPPED if why == sf.REASON_OVERSIZE else sf.STATE_FAILED

    if payload is None and path is not None:
        payload, reason = _read_local(Path(path), limit=limit)
        if reason:
            return sf.note(key, state=_outcome(reason), reason=reason, filename=name,
                           label=label, source_tool=tool,
                           source_kind=source_kind, source_url=url)

    if payload is None:
        target = url or _vault_url(bucket, s3_key)
        if not target:
            return sf.note(key, state=sf.STATE_FAILED, reason=sf.REASON_UNREADABLE,
                           filename=name, label=label, source_tool=tool,
                           source_kind=source_kind)
        payload, media_type, reason = _fetch(
            target, limit=limit, timeout=cfg.download_timeout
        )
        if reason:
            return sf.note(key, state=_outcome(reason), reason=reason, filename=name,
                           label=label, source_tool=tool,
                           source_kind=source_kind, source_url=url)

    record = sf.put_bytes(
        key, payload,
        filename=name, label=label, source_tool=tool, source_kind=source_kind,
        source_url=url, media_type=media_type, agent=agent,
        bucket=bucket, s3_key=s3_key,
    )

    # A second, off-host address. The local copy is the home, so a failure here
    # is not a failure of the mirror.
    if (
        mirror_off_host
        and cfg.mirror_to_s3
        and record.get("state") == sf.STATE_STORED
        and not record.get("bucket")
    ):
        _also_to_s3(key, record)
    return record


def _name_from(url: Optional[str], path: Any) -> str:
    if path is not None:
        return Path(str(path)).name
    if url:
        from urllib.parse import unquote, urlsplit

        return unquote(urlsplit(url).path.rsplit("/", 1)[-1]) or "artifact"
    return "artifact"


def _vault_url(bucket: Optional[str], s3_key: Optional[str]) -> Optional[str]:
    """A fresh download link for a durable reference, through the vault.

    The same call ``result_formatter_tool._download_link`` makes. It resolves
    against the vault's own bucket, so an artifact in a foreign bucket comes
    back empty — which is why the URL branch is tried first.
    """
    if not s3_key:
        return None
    try:
        from CoScientist.tools.vault_client import call_vault_sync, vault_url

        if not vault_url():
            return None
        payload = call_vault_sync("get_download_link", s3_key=s3_key)
        if isinstance(payload, dict):
            return payload.get("presigned_url") or payload.get("url")
    except Exception as exc:  # noqa: BLE001
        logger.warning("mirror: vault link failed for %s (%s)", s3_key, exc)
    return None


def _also_to_s3(key: SessionKey, record: Dict[str, Any]) -> None:
    try:
        from CoScientist.graph.session_scope import safe_component
        from CoScientist.reporting.s3_upload import upload_and_ref

        local = sf.resolve_path(key, record["artifact_id"])
        if local is None:
            return
        prefix = f"sessions/{safe_component(key[0])}/{safe_component(key[1])}/artifacts"
        reference = upload_and_ref(local, prefix)
        if reference:
            bucket, s3_key = reference
            sf.record_entries(
                [{**record, "bucket": bucket, "s3_key": s3_key}],
                user_id=key[0], session_id=key[1],
            )
            record["bucket"], record["s3_key"] = bucket, s3_key
    except Exception as exc:  # noqa: BLE001 — the local copy already succeeded
        logger.warning("mirror: S3 copy failed for %s (%s)", record.get("filename"), exc)


# ── the shared capture body ─────────────────────────────────────────────────
def mirror_tool_result(
    tool: Any, tool_context: Any, result: Any,
    scope: Optional[SessionKey] = None,
) -> List[Dict[str, Any]]:
    """Mirror every artifact a tool result points at. Never raises.

    One implementation for both capture sites. Returns the records so the caller
    can write them into whatever index it owns.

    ``scope`` is resolved by the caller, on the event loop, because
    ``session_key`` *writes* the resolved pair back into ADK state the first
    time it runs — and this function is meant to be called from a worker thread.
    """
    from CoScientist.reporting.collect import find_artifact_urls
    from CoScientist.utils.s3_refs import find_s3_artifacts

    try:
        urls = find_artifact_urls(result)
        referenced = find_s3_artifacts(result)
    except Exception:  # noqa: BLE001
        return []
    if not urls and not referenced:
        return []

    user_id, session_id = scope if scope else (None, None)
    context = None if scope else tool_context

    tool_name = getattr(tool, "name", None) or ""
    # An upload link is a PUT capability: it cannot fetch, and the object may not
    # exist yet. The agent was handed somewhere to put one, not something to take.
    if tool_name == "get_upload_link":
        return []

    # What a tool READ is filed apart from what a tool MADE. Both arrive here as
    # a URL in a result, and only the provenance tells them apart: a figure
    # lifted off an article page is the journal's, not this study's, and the
    # report must not offer it as an illustration of the work.
    from CoScientist.reporting.collect import SOURCE_ASSET_KIND, is_reading_tool

    read_kind = SOURCE_ASSET_KIND if is_reading_tool(tool_name) else ""

    records: List[Dict[str, Any]] = []
    claimed = set()
    for item in referenced[:MAX_PER_CALL]:
        url = item.get("url")
        if url:
            claimed.add(url)
        if _skip(url or "", item):
            continue
        records.append(mirror_artifact(
            context, user_id=user_id, session_id=session_id,
            url=url, bucket=item.get("bucket"), s3_key=item.get("s3_key"),
            filename=Path(str(item.get("s3_key") or "")).name,
            tool=tool_name, source_kind=read_kind or "mcp_ref",
        ))

    for url in urls[:MAX_PER_CALL]:
        if url in claimed or _skip(url, {}):
            continue
        records.append(mirror_artifact(
            context, user_id=user_id, session_id=session_id,
            url=url, tool=tool_name, source_kind=read_kind or "mcp_url",
        ))

    stored = sum(1 for r in records if r.get("state") == sf.STATE_STORED)
    if records:
        logger.info(
            "mirror: %s -> %d stored, %d not (%s)",
            tool_name or "?", stored, len(records) - stored,
            ", ".join(sorted({str(r.get("reason")) for r in records if r.get("reason")})) or "-",
        )
    return records


def _skip(url: str, item: Dict[str, Any]) -> bool:
    """Ingest material is not a result.

    The narrow ``_IMAGE_EXTS + _TABLE_EXTS`` filter this replaces is why a
    ``.json`` of predictions, an ``.html`` docking view and every coder output
    never reached the user. The rule is now a deny-list: source documents a
    search step pulled in, and nothing else.
    """
    from CoScientist.reporting.collect import _is_page_chrome, _is_source_material

    try:
        # Page furniture is refused before the request, not after: an icon is
        # never a result of a run, and downloading eighteen of them off one
        # article page spends this session's file quota on a publisher's logo.
        return _is_page_chrome(url) or _is_source_material(item, url)
    except Exception:  # noqa: BLE001
        return False


__all__ = ["mirror_artifact", "mirror_tool_result", "MAX_PER_CALL"]
