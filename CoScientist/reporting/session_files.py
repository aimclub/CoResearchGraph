"""The session's own copy of every file a run produced.

A tool that renders a figure hands back a presigned URL to *its* storage. One
such link, measured on a live run, was valid for six minutes: issued at 12:29:56
and expired at 12:35:29. We stored the string and never fetched the bytes, so by
the time anyone opened the graph the artifact was gone for good. Across six
sessions, 77 of 88 indexed artifacts had no durable reference at all.

So the bytes come here, at the moment the link still works, and this directory —
beside the graph snapshots, inside what export already packs — is the primary
copy. S3 is a secondary address, not the home.

    <GRAPH_SNAPSHOT_DIR>/sessions/<user>/<session>/
        artifacts.json              # the existing index (unchanged)
        artifacts/manifest.json     # one record per artifact_id
        artifacts/files/<id>        # the bytes

**The reference carries no session.** Graph attrs, the index and report markdown
all store ``cos-artifact:<artifact_id>``, and the URL is built from whatever
scope is rendering. That is what lets an exported bundle open on another machine
under a new session id without rewriting a single link — which is the whole
reason for the indirection.

Ids are content-addressed (``<slug>-<sha8><ext>``), so the same figure captured
twice under two different signatures is one record and one file.

Nothing here raises. A file that cannot be stored becomes a record saying so,
with a reason code: a report that silently lost an illustration reads exactly
like one that never had it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import re
import tempfile
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from CoScientist.graph.session_scope import (
    SessionKey,
    safe_component,
    session_key,
    storage_dir,
)

logger = logging.getLogger(__name__)

ARTIFACTS_DIRNAME = "artifacts"
FILES_DIRNAME = "files"
MANIFEST_FILENAME = "manifest.json"

#: The scheme stored in graphs and markdown. Deliberately not a URL: a URL would
#: bake in a host and a session, and both change when a bundle is imported.
SCHEME = "cos-artifact"
_REF_RE = re.compile(rf"{SCHEME}:([A-Za-z0-9][A-Za-z0-9._-]{{0,127}})")

#: What the web route will accept back. Kept here so the writer and the reader
#: can never disagree about which names are legal.
ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: Anything outside this is replaced. The vault enforces the same shape on its
#: own keys (see ``tools/workspace_sync``); matching it here means a mirrored
#: file can be re-uploaded without a second round of renaming.
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SLUG = 48

#: States a record can be in. ``stored`` has bytes on disk; the other two do not
#: and say why.
STATE_STORED = "stored"
STATE_SKIPPED = "skipped"
STATE_FAILED = "failed"

#: Reasons are codes, not sentences, so the UI can translate them the way the
#: research graph already translates its gap codes.
REASON_OVERSIZE = "oversize"
REASON_SESSION_QUOTA = "session_quota"
REASON_TOO_MANY = "too_many_files"
#: A signed link that no longer verifies. Kept narrow on purpose: it used to
#: cover every 4xx, so a host that simply refused us was recorded as an expiry
#: — and that sentence is printed verbatim into the report's "not collected"
#: section, where it tells the reader the wrong thing about why a paper is
#: missing. NCBI answering 403 to a programmatic GET is the case that exposed it.
REASON_LINK_EXPIRED = "link_expired"
#: The host answered, and said no. Nothing about it will change on a retry with
#: the same client.
REASON_REFUSED = "refused"
#: There is nothing at that address.
REASON_NOT_FOUND = "not_found"
REASON_UNREADABLE = "unreadable"
REASON_EMPTY = "empty"
REASON_DISABLED = "disabled"
REASON_BUNDLE_QUOTA = "bundle_quota"


# ── locations ───────────────────────────────────────────────────────────────
def _root() -> str:
    return os.getenv("GRAPH_SNAPSHOT_DIR", "./graph_runs")


def store_dir(key: SessionKey) -> Path:
    return storage_dir(_root(), key) / ARTIFACTS_DIRNAME


def files_dir(key: SessionKey) -> Path:
    return store_dir(key) / FILES_DIRNAME


def manifest_path(key: SessionKey) -> Path:
    return store_dir(key) / MANIFEST_FILENAME


# ── references ──────────────────────────────────────────────────────────────
def ref(artifact_id: str) -> str:
    """``"fig-3f9ac2b1.png"`` -> ``"cos-artifact:fig-3f9ac2b1.png"``."""
    return f"{SCHEME}:{artifact_id}"


def parse_ref(value: Any) -> Optional[str]:
    """The artifact id inside a reference, or None when this is not one."""
    text = str(value or "").strip()
    if not text.startswith(f"{SCHEME}:"):
        return None
    match = _REF_RE.fullmatch(text)
    return match.group(1) if match else None


def find_refs(text: Any) -> List[str]:
    """Every artifact id referenced in a blob of prose or JSON."""
    return _REF_RE.findall(str(text or ""))


# ── naming ──────────────────────────────────────────────────────────────────
def _slug(filename: str) -> str:
    """An ASCII stem a filesystem, a URL and the vault all accept.

    Tools name their output after whatever they please — a Cyrillic caption, a
    space, a parenthesis. Folding first keeps the name recognisable instead of
    replacing it wholesale with a hash.
    """
    stem = Path(str(filename or "")).stem
    folded = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    cleaned = _UNSAFE_RE.sub("_", folded).strip("._-")
    cleaned = re.sub(r"_{2,}", "_", cleaned)[:_MAX_SLUG].strip("._-")
    return cleaned or "artifact"


#: Media types a server hands back when it has not looked at the bytes. MinIO
#: answers ``binary/octet-stream`` for objects uploaded without a type, and a
#: PNG stored under that type is served as a download instead of being drawn —
#: the web route decides `inline` from exactly this field.
_GENERIC_MEDIA_TYPES = frozenset({
    "", "application/octet-stream", "binary/octet-stream",
    "application/binary", "*/*",
})


def _best_media_type(media_type: Optional[str], filename: str) -> Optional[str]:
    """The declared type, unless it says nothing the file name does not say better."""
    declared = str(media_type or "").split(";")[0].strip().lower()
    if declared in _GENERIC_MEDIA_TYPES:
        return mimetypes.guess_type(filename or "")[0] or (declared or None)
    return media_type


def _extension(filename: str, media_type: Optional[str]) -> str:
    suffix = Path(str(filename or "")).suffix.lower()
    if suffix and len(suffix) <= 10:
        return suffix
    if media_type:
        guessed = mimetypes.guess_extension(str(media_type).split(";")[0].strip())
        if guessed:
            return guessed
    return ".bin"


def make_artifact_id(
    payload_sha256: str,
    filename: str,
    media_type: Optional[str] = None,
    taken: Iterable[str] = (),
) -> str:
    """``<slug>-<sha8><ext>``, unique among ``taken``.

    The hash is what makes two captures of one figure the same record. The slug
    is there so a human reading the manifest or the URL can tell what the file
    is without opening it.
    """
    base = f"{_slug(filename)}-{payload_sha256[:8]}{_extension(filename, media_type)}"
    used = set(taken)
    if base not in used:
        return base
    # Different bytes, same 8-hex prefix. Astronomically unlikely, but the full
    # sha256 is in the record, so a collision is detectable rather than silent.
    stem, suffix = base[: -len(Path(base).suffix)], Path(base).suffix
    for index in range(2, 100):
        candidate = f"{stem}-{index}{suffix}"
        if candidate not in used:
            return candidate
    return base


# ── manifest ────────────────────────────────────────────────────────────────
def _read_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 — a broken manifest must not sink a run
        logger.warning("artifact store: cannot read %s (%s)", path, exc)
        return {}
    records = data.get("artifacts") if isinstance(data, dict) else data
    if isinstance(records, dict):
        return {k: v for k, v in records.items() if isinstance(v, dict)}
    if isinstance(records, list):
        return {
            r["artifact_id"]: r
            for r in records
            if isinstance(r, dict) and r.get("artifact_id")
        }
    return {}


def load_manifest(
    session_id: str, user_id: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Every record of one session, keyed by ``artifact_id``.

    ``user_id`` is optional for the same reason it is in ``artifact_index``: the
    report collector is called from a tool that knows only the session id.
    """
    if user_id:
        paths = [manifest_path((user_id, session_id))]
    else:
        root = Path(_root()) / "sessions"
        paths = sorted(
            root.glob(
                f"*/{safe_component(session_id)}/{ARTIFACTS_DIRNAME}/{MANIFEST_FILENAME}"
            )
        )
    for path in paths:
        records = _read_manifest(path)
        if records:
            return records
    return {}


def _write_manifest(path: Path, records: Dict[str, Dict[str, Any]]) -> None:
    """Replace the manifest atomically, the way the artifact index does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"artifacts": records}, ensure_ascii=False, indent=2)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def record_entries(
    entries: List[Dict[str, Any]],
    context: Any = None,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Merge records into one session's manifest. Never raises."""
    if not entries:
        return
    try:
        key = session_key(context, user_id=user_id, session_id=session_id)
        path = manifest_path(key)
        records = _read_manifest(path)
        changed = False
        for entry in entries:
            aid = entry.get("artifact_id")
            if not aid:
                continue
            previous = records.get(aid)
            if previous == entry:
                continue
            # A retry that succeeded replaces a failure; a failure never
            # overwrites bytes that are already on disk.
            if (
                previous
                and previous.get("state") == STATE_STORED
                and entry.get("state") != STATE_STORED
            ):
                continue
            records[aid] = entry
            changed = True
        if changed:
            _write_manifest(path, records)
    except Exception as exc:  # noqa: BLE001 — capture must never break a tool call
        logger.warning("artifact store: cannot record (%s)", exc)


# ── writing ─────────────────────────────────────────────────────────────────
def _now() -> float:
    return time.time()


def note(
    key: SessionKey,
    *,
    state: str,
    reason: str,
    filename: str = "",
    **meta: Any,
) -> Dict[str, Any]:
    """A record for a file that did NOT make it, and why.

    Given an id so it still dedupes and still has somewhere to be listed. The
    id is derived from the name and the reason rather than from bytes we do not
    have.
    """
    seed = hashlib.sha256(f"{filename}|{reason}".encode("utf-8")).hexdigest()
    record = {
        "artifact_id": make_artifact_id(seed, filename or "artifact", None),
        "state": state,
        "reason": reason,
        "filename": str(filename or ""),
        "sha256": None,
        "size_bytes": meta.pop("size_bytes", None),
        "media_type": None,
        "captured_at": _now(),
        **{k: v for k, v in meta.items() if v is not None},
    }
    record_entries([record], user_id=key[0], session_id=key[1])
    return record


def put_bytes(
    key: SessionKey,
    payload: bytes,
    *,
    filename: str,
    label: str = "",
    source_tool: str = "",
    source_kind: str = "",
    source_url: Optional[str] = None,
    media_type: Optional[str] = None,
    agent: Optional[str] = None,
    bucket: Optional[str] = None,
    s3_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Store bytes in this session's directory and return the record.

    Content-addressed, so storing the same bytes twice is one file and one
    record. Never raises: a filesystem error comes back as a ``failed`` record.
    """
    if not payload:
        return note(key, state=STATE_SKIPPED, reason=REASON_EMPTY, filename=filename,
                    label=label, source_tool=source_tool, source_url=source_url)

    digest = hashlib.sha256(payload).hexdigest()
    manifest = load_manifest(key[1], key[0])

    # Already stored under this exact content? Keep the first record; only fill
    # in a durable S3 address if this call learned one.
    for existing in manifest.values():
        if existing.get("sha256") == digest and existing.get("state") == STATE_STORED:
            if bucket and s3_key and not existing.get("bucket"):
                updated = {**existing, "bucket": bucket, "s3_key": s3_key}
                record_entries([updated], user_id=key[0], session_id=key[1])
                return updated
            return existing

    guessed = _best_media_type(media_type, filename)
    artifact_id = make_artifact_id(digest, filename, guessed, manifest.keys())

    try:
        target_dir = files_dir(key)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / artifact_id
        fd, tmp = tempfile.mkstemp(dir=str(target_dir), suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
            os.replace(tmp, target)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise
    except Exception as exc:  # noqa: BLE001
        # Loud on purpose. This fired silently once during development because
        # the session directory sat deep enough for the artifact path to cross
        # Windows' 260-character limit, and a figure that vanishes with no line
        # in the log is indistinguishable from a figure nobody made.
        logger.warning(
            "artifact store: cannot write %s to %s (%s: %s)",
            filename, files_dir(key), type(exc).__name__, exc,
        )
        return note(key, state=STATE_FAILED, reason=REASON_UNREADABLE,
                    filename=filename, label=label, source_tool=source_tool,
                    source_url=source_url)

    record = {
        "artifact_id": artifact_id,
        "state": STATE_STORED,
        "reason": None,
        "sha256": digest,
        "size_bytes": len(payload),
        "media_type": guessed,
        "filename": str(filename or artifact_id),
        "label": label or Path(str(filename or "")).name or artifact_id,
        "source_tool": source_tool or None,
        "source_kind": source_kind or None,
        # Provenance only. Never the link we show: it is the thing that expires.
        "source_url": source_url or None,
        "bucket": bucket,
        "s3_key": s3_key,
        "agent": agent,
        "captured_at": _now(),
    }
    record_entries([record], user_id=key[0], session_id=key[1])
    return record


# ── reading ─────────────────────────────────────────────────────────────────
def resolve_path(key: SessionKey, artifact_id: str) -> Optional[Path]:
    """The file behind an id, or None.

    Traversal-safe by construction *and* by check: the id must match
    ``ARTIFACT_ID_RE`` (which admits no ``/`` and no leading dot, so ``..`` is
    already out), and the resolved path must still sit inside this session's
    files directory. The web route adds a third gate by refusing a path
    parameter that contains a separator at all.
    """
    if not artifact_id or not ARTIFACT_ID_RE.match(artifact_id):
        return None
    base = files_dir(key)
    try:
        target = (base / artifact_id).resolve()
        if not target.is_file():
            return None
        if base.resolve() not in target.parents:
            return None
    except OSError:
        return None
    return target


def has_artifact(key: SessionKey, artifact_id: str) -> bool:
    """Whether this session actually holds that artifact.

    Asked before anything is rendered as a link. A reference is just a string,
    and one written by an older run — or by code that mistook another
    namespace's id for ours — names a file that was never stored. Offering it as
    a link produces a 404 on click, which reads as a broken page rather than as
    an artifact nobody mirrored.

    Cached on the manifest's mtime: the research panel resolves one reference
    per attachment, and re-reading the file for each would make opening a card
    an O(attachments) pile of syscalls.
    """
    if not artifact_id:
        return False
    path = manifest_path(key)
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return False
    return artifact_id in _manifest_cached(str(path), stamp)


@lru_cache(maxsize=32)
def _manifest_cached(path: str, _mtime_ns: int) -> frozenset:
    """Ids in one manifest. ``_mtime_ns`` is the cache key, not an argument."""
    return frozenset(_read_manifest(Path(path)))


def artifact_id_for_url(key: SessionKey, url: str) -> Optional[str]:
    """The id of the artifact we mirrored FROM this URL, if we mirrored one.

    The bridge between a link that is dying and the copy that is not. A tool
    server hands back a presigned URL to *its* storage — a host this
    deployment's S3 settings say nothing about, so the usual rewrite leaves it
    alone and four hundred characters of expiring signature end up in front of
    a reader. The bytes, though, were taken at capture time and are sitting in
    this session's store; this is how the text finds them.

    Cached on the manifest's mtime, like :func:`has_artifact`: the lookup runs
    once per URL in every chat message, and re-reading the file each time would
    make it a syscall per link.
    """
    text = str(url or "").strip()
    if not text:
        return None
    path = manifest_path(key)
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    return _by_source_url(str(path), stamp).get(text)


def retag(key: SessionKey, artifact_id: str, *, source_kind: str,
          label: str = "") -> Optional[Dict[str, Any]]:
    """Re-file an artifact this session already holds. Never raises.

    The bytes are fetched once, by whoever got there first, and what a file IS
    can be learned afterwards. A paper found through a web search arrives as an
    ordinary mirrored URL; only once a citation is matched to it does anyone
    know it is a paper — and being a paper decides whether it is offered as a
    deliverable in the report or kept as the source it is.

    Re-filing beats mirroring again: the store is content-addressed, so a second
    fetch would produce the same id and the same bytes at the cost of another
    download from someone else's server.
    """
    aid = str(artifact_id or "").strip()
    if not aid:
        return None
    try:
        path = manifest_path(key)
        records = _read_manifest(path)
        record = records.get(aid)
        if not record or record.get("source_kind") == source_kind:
            return record
        updated = {**record, "source_kind": source_kind}
        if label:
            updated["label"] = label
        records[aid] = updated
        _write_manifest(path, records)
        return updated
    except Exception as exc:  # noqa: BLE001 — capture must never break a call
        logger.warning("artifact store: cannot re-file %s (%s)", aid, exc)
        return None


@lru_cache(maxsize=32)
def _by_source_url(path: str, _mtime_ns: int) -> Dict[str, str]:
    """``{source_url: artifact_id}`` for the stored records of one manifest."""
    return {
        record["source_url"]: artifact_id
        for artifact_id, record in _read_manifest(Path(path)).items()
        if record.get("state") == STATE_STORED and record.get("source_url")
    }


def stored_bytes(key: SessionKey) -> int:
    """How much this session has already mirrored, for the quota check."""
    return sum(
        int(r.get("size_bytes") or 0)
        for r in load_manifest(key[1], key[0]).values()
        if r.get("state") == STATE_STORED
    )


def stored_count(key: SessionKey) -> int:
    return sum(
        1
        for r in load_manifest(key[1], key[0]).values()
        if r.get("state") == STATE_STORED
    )


__all__ = [
    "ARTIFACTS_DIRNAME", "ARTIFACT_ID_RE", "FILES_DIRNAME", "MANIFEST_FILENAME",
    "SCHEME",
    "STATE_STORED", "STATE_SKIPPED", "STATE_FAILED",
    "REASON_OVERSIZE", "REASON_SESSION_QUOTA", "REASON_TOO_MANY",
    "REASON_LINK_EXPIRED", "REASON_REFUSED", "REASON_NOT_FOUND",
    "REASON_UNREADABLE", "REASON_EMPTY", "REASON_DISABLED",
    "REASON_BUNDLE_QUOTA",
    "store_dir", "files_dir", "manifest_path",
    "ref", "parse_ref", "find_refs", "make_artifact_id",
    "load_manifest", "record_entries", "note", "put_bytes",
    "resolve_path", "has_artifact", "artifact_id_for_url", "retag",
    "stored_bytes", "stored_count",
]
