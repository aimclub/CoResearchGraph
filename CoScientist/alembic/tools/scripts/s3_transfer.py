"""Self-contained S3 transfer helpers for the generated server.py.

Copied verbatim into ``output/helpers/s3_transfer.py`` by codegen
(``write_server``) so the served MCP boundary can hand back S3 references
instead of paths that only exist inside the build container. Deliberately
stdlib + a LAZILY imported ``boto3`` — never at module level — because this
module is loaded inside ``.venv-server``, which the wrapper only guarantees
carries fastmcp/mcp (+ boto3 when S3 is configured); importing boto3 eagerly
here would break every server that has no S3 configured.

v2 scoping model: every generated tool declares optional ``user_id``/
``session_id`` parameters that ``SessionScopePlugin`` (CoScientist/tools/
session_scope_plugin.py) fills in at the ADK tool-call boundary when the
caller leaves them blank — that is the primary path. This module also
exposes ``scope_from_headers``, which reads the ``X-Coscientist-User`` /
``X-Coscientist-Session`` request headers; it exists as a dormant fallback
for ``server.py`` to reach for when the params come back empty (an old
client bypassing the plugin, stdio transport, or an old fastmcp with no
header access) — nothing in this codebase currently sends those headers.
Both paths fall back to a shared ``("local", "default")`` scope when they
have nothing. This is a namespacing convenience, NOT a security
boundary — a presigned URL grants access to whoever holds it.

Error asymmetry (by design, see README): a failed *input* download
(``resolve_input``/``_download_s3``/``_download_http``) raises and fails the
whole tool call — a tool cannot silently run on the wrong/missing data. A
failed *output* upload (``maybe_upload``) only logs to stderr and returns
``None`` — an otherwise-successful tool call must not be turned into a
failure just because publishing its result to S3 did not work.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

# All four must be set (and non-empty) for S3 handling to switch on at all.
# Each is read via _env(): the vault contract's ``S3__<name>`` spelling first
# (pydantic env_nested_delimiter, see config/settings.py:S3Settings — the
# spelling mcp-servers/vault-mcp-server itself is configured with here; it
# also accepts MINIO_*-named fallbacks of its own, which this module has no
# reason to know about), the bare name second as a DEPRECATED legacy fallback
# — never merged, the first spelling that is set and non-empty wins — kept
# only until every .env still using the bare names is updated. A .env that
# already configures S3 for the main app needs no duplicate bare-name
# entries; an old bare-only .env keeps working in the meantime.
S3_ENV = ("ENDPOINT_URL", "ACCESS_KEY", "SECRET_KEY", "BUCKET_NAME")
# Case-insensitive convention a tool param name is checked against to decide
# whether its value may be an input/output file reference.
FILE_SUFFIXES = ("_path", "_file")

_DEFAULT_PRESIGN_EXPIRATION = 3600  # 1 hour — bucket+s3_key is the durable reference;
# the framework can always reissue the URL (collect.py:resolve_url), and every other
# MCP server in this repo presigns on the same hour (chemical-mcp-server's
# _URL_TTL_SECONDS, mcp-servers/chemical-mcp-server/server/utils/vault.py).
_MIN_PRESIGN_EXPIRATION = 1
_MAX_PRESIGN_EXPIRATION = 604800  # 7 days — S3 SigV4's own presign ceiling

_DEFAULT_REGION = "us-east-1"

_DEFAULT_HTTP_TIMEOUT = 300  # seconds
_DEFAULT_HTTP_MAX_BYTES = 1024 * 1024 * 1024  # 1 GiB
_DEFAULT_UPLOAD_MAX_BYTES = 1024 * 1024 * 1024  # 1 GiB
# A result longer than this (as JSON) goes to S3 whole and the caller gets a
# shortened copy: a client cutting it for a model's context loses whole fields.
_DEFAULT_RESULT_MAX_CHARS = 8000
_PREVIEW_ITEMS = 20       # list items and dict keys kept at first
_PREVIEW_CHARS = 1000     # characters kept of a long string at first
_PREVIEW_MAX_NOTES = 50
_HTTP_CHUNK_SIZE = 1024 * 1024

# Same bounds as CoScientist/paper_parser/s3_connection.py (#349). botocore's
# defaults are 60 s to connect, 60 s to read and up to 5 attempts, so an
# endpoint that drops packets blocks a tool call for minutes, past the client's
# MCP read timeout. read_timeout is per socket read, so a large transfer still
# completes.
_CONNECT_TIMEOUT_SECONDS = 5
_READ_TIMEOUT_SECONDS = 30
_MAX_ATTEMPTS = 3


def _env(name: str) -> str | None:
    """Read an ``S3_ENV`` setting: the vault contract's ``S3__<name>``
    spelling first, the bare name second as a DEPRECATED legacy fallback
    (never merged — the first spelling that is set and non-empty wins)."""
    return os.environ.get(f"S3__{name}") or os.environ.get(name)


def s3_enabled() -> bool:
    """True iff every ``S3_ENV`` variable is set and non-empty (under either
    its bare or its ``S3__``-prefixed spelling, per variable)."""
    return all(_env(name) for name in S3_ENV)


def is_file_param(name: str) -> bool:
    """Case-insensitive ``*_path`` / ``*_file`` convention (repos vary on
    casing for their own param names)."""
    return name.lower().endswith(FILE_SUFFIXES)


def safe_component(value: str) -> str:
    """Make an identifier safe for an S3 key component (repo/tool/field
    name). Mirrors ``graph/session_scope.py:safe_component`` — duplicated
    rather than imported, this module must stay free of any CoScientist
    import. NOT used for user/session — see ``safe_id`` below, which the
    vault validates those two segments against a stricter regex. NOT used
    for a downloaded file's basename either — see ``_safe_filename`` below,
    which a non-ASCII name would lose its extension to under this stricter
    ASCII-only cleanup."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned[:96] or "unknown"


def safe_id(value: str, default: str) -> str:
    """Make an id fit the vault's ``user_id``/``session_id`` key segments
    (mcp-servers/vault-mcp-server/vault_server.py:55 ``_ID_RE``): each must
    match ``^[a-zA-Z0-9_-]{1,64}$``. Mirrors
    ``mcp-servers/chemical-mcp-server/server/utils/vault.py:safe_id``,
    duplicated for the same reason ``safe_component`` is. Stricter than
    ``safe_component`` (no dot, 64 chars not 96) — required for the two
    segments the vault validates as an id; ``safe_component`` stays the
    right tool for repo/tool/field, which the vault treats as free path
    text."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "")).strip("_")
    return cleaned[:64] or default


_MAX_FILENAME_STEM = 180
_MAX_FILENAME_SUFFIX = 60


def _safe_filename(raw: str) -> str:
    """Sanitize a downloaded basename for the local filesystem WITHOUT
    mangling it the way ``safe_component()`` would — e.g.
    ``safe_component('данные.csv')`` collapses to ``'csv'`` (the non-ASCII
    stem is replaced then ``.strip("._")`` eats it), silently handing a tool
    that dispatches on suffix (pandas/PIL/torch) the wrong data under a
    misleadingly named path. Only path separators and NUL are touched.
    Traversal is not a concern here: the caller already took ``Path(...).name``
    (stripping any directory components) and writes into a fresh, isolated
    ``_download_dir()``.

    Also bounded to a safe filesystem length: an S3 key can run up to 1024
    bytes, well past the ~255-byte filename limit most filesystems enforce
    (``OSError: [Errno 36] File name too long``) — truncate the stem/suffix
    separately so the extension a tool dispatches on survives."""
    name = re.sub(r"[/\\\x00]", "_", raw)
    if name in ("", ".", ".."):
        return "download"
    stem, suffix = Path(name).stem, Path(name).suffix
    return stem[:_MAX_FILENAME_STEM] + suffix[:_MAX_FILENAME_SUFFIX]


_MAX_KEY_FILENAME_STEM = 96
_MAX_KEY_FILENAME_SUFFIX = 32


def _key_safe_filename(raw: str) -> str:
    """Sanitize a local basename for the S3 KEY it is uploaded under —
    NOT for the local filesystem, see ``_safe_filename`` above, which stays
    permissive of non-ASCII so a downloaded file keeps its real name on disk.
    The vault's key regex only allows ``[a-zA-Z0-9_/.-]`` in the path portion
    of a key (mcp-servers/vault-mcp-server/vault_server.py:56 ``_KEY_RE``);
    a basename with a space, ``+``, or non-ASCII character (``данные.csv``,
    ``a b+c.csv``) would otherwise build a key the vault rejects — silently,
    with only a stderr warning — the next time it is asked to reissue the
    presigned URL (``get_download_link``) or promote the object to
    ``permanent/`` (``promote_artifact``).

    Stem and suffix are sanitized separately, same reasoning as
    ``_safe_filename``: a long or heavily-mangled stem must not eat the
    extension a consumer keys off of — though this is a best effort, not a
    guarantee: an entirely non-ASCII suffix (``.中文``) sanitizes to the
    literal two characters ``"._"`` rather than a meaningful extension, since
    only the stem (not the suffix) is ``.strip("._")``-ed back down to
    ``"download"`` when nothing usable survives."""
    name = re.sub(r"[/\\\x00]", "_", raw)
    if name in ("", ".", ".."):
        return "download"
    stem, suffix = Path(name).stem, Path(name).suffix
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "download"
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix)
    return stem[:_MAX_KEY_FILENAME_STEM] + suffix[:_MAX_KEY_FILENAME_SUFFIX]


def scope_from_headers(headers: dict | None) -> tuple[str, str]:
    """``(user, session)`` from the ``X-Coscientist-User`` / ``X-Coscientist-
    Session`` headers, read case-insensitively; falls back to
    ``("local", "default")`` when either is missing (no client headers, stdio
    transport, or an old fastmcp with no header access). Sanitized with
    ``safe_id``, not ``safe_component`` — these are the same two key
    segments the vault validates as an id (see ``safe_id``'s docstring)."""
    if headers:
        lowered = {k.lower(): v for k, v in headers.items()}
        user = lowered.get("x-coscientist-user")
        session = lowered.get("x-coscientist-session")
        if user and session:
            return safe_id(user, "local"), safe_id(session, "default")
    return "local", "default"


def call_prefix(scope: tuple[str, str], repo_name: str, tool: str) -> str:
    """S3 key prefix for one tool call, laid out per the vault contract:
    ``ephemeral/<user>/<session>/alembic_<repo>/<tool>/<8-hex>``.

    ``ephemeral/`` is the top segment the bucket's lifecycle rule filters on
    (S3 prefix filters are literal strings and cannot match a segment in the
    middle of a key) — every MCP server writes only under it, and promotion
    to ``permanent/`` is the framework's job, not this module's (see
    mcp-servers/vault-mcp-server/vault_server.py:promote_artifact). Feature
    segment ``alembic_<repo>`` mirrors
    mcp-servers/chemical-mcp-server/server/utils/vault.py:scoped_prefix,
    which nests one segment (``chemical_mcp``) under the user/session pair;
    here the repo name is folded into that segment so different repos served
    by this same alembic image don't share one prefix. ``user``/``session``
    are sanitized with ``safe_id`` (the vault validates those two segments
    as an id, stricter than the free path text ``repo_name``/``tool`` get
    from ``safe_component`` — see both functions' docstrings)."""
    user, session = scope
    return (
        f"ephemeral/{safe_id(user, 'local')}/{safe_id(session, 'default')}/"
        f"alembic_{safe_component(repo_name)}/{safe_component(tool)}/{uuid.uuid4().hex[:8]}"
    )


def _client_factory(endpoint_url: str | None = None):
    """boto3 S3 client built from ``S3_ENV`` (+ optional ``S3_REGION``). A
    module attribute (not inlined into its callers) so tests can swap it out
    without moto or real credentials — see
    ``paper_parser/s3_connection.py:S3BucketService`` for the pattern this
    mirrors. ``region_name`` is required by botocore even for a fully custom
    ``endpoint_url`` — omitting it raises ``NoRegionError``.

    ``endpoint_url`` overrides ``ENDPOINT_URL``; ``maybe_upload`` passes the
    external endpoint to presign links for the caller's side."""
    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or _env("ENDPOINT_URL"),
        aws_access_key_id=_env("ACCESS_KEY"),
        aws_secret_access_key=_env("SECRET_KEY"),
        # `or`, not a get() default: the build blanks S3_REGION in the committed
        # image (ENV S3_REGION=), and an empty region signs links that strict
        # S3 services reject.
        region_name=os.environ.get("S3_REGION") or _DEFAULT_REGION,
        config=Config(
            signature_version="s3v4",
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
            read_timeout=_READ_TIMEOUT_SECONDS,
            retries={"max_attempts": _MAX_ATTEMPTS, "mode": "standard"},
        ),
    )


def _external_endpoint() -> str | None:
    """The endpoint the caller reaches S3 at, when it differs from the one this
    server uses. Named as in mcp-servers/vault-mcp-server. A SigV4 signature
    covers the host, so a link must be presigned for the host that will open
    it. start_chain.py sets this when it points a serve container at the host's
    loopback MinIO through host.docker.internal."""
    external = os.environ.get("S3__EXTERNAL_ENDPOINT_URL")
    return external if external and external != _env("ENDPOINT_URL") else None


def _presign_expiration() -> int:
    try:
        value = int(os.environ.get("S3_PRESIGN_EXPIRATION", _DEFAULT_PRESIGN_EXPIRATION))
    except ValueError:
        return _DEFAULT_PRESIGN_EXPIRATION
    return max(_MIN_PRESIGN_EXPIRATION, min(value, _MAX_PRESIGN_EXPIRATION))


def _http_timeout() -> int:
    try:
        return int(os.environ.get("S3_HTTP_TIMEOUT", _DEFAULT_HTTP_TIMEOUT))
    except ValueError:
        return _DEFAULT_HTTP_TIMEOUT


def _http_max_bytes() -> int:
    try:
        return int(os.environ.get("S3_HTTP_MAX_BYTES", _DEFAULT_HTTP_MAX_BYTES))
    except ValueError:
        return _DEFAULT_HTTP_MAX_BYTES


def _upload_max_bytes() -> int:
    try:
        return int(os.environ.get("S3_UPLOAD_MAX_BYTES", _DEFAULT_UPLOAD_MAX_BYTES))
    except ValueError:
        return _DEFAULT_UPLOAD_MAX_BYTES


def _result_max_chars() -> int:
    try:
        return int(os.environ.get("ALEMBIC_RESULT_MAX_CHARS", _DEFAULT_RESULT_MAX_CHARS))
    except ValueError:
        return _DEFAULT_RESULT_MAX_CHARS


def _download_dir(scratch_dir: Path) -> Path:
    """A fresh, unique subdirectory of ``scratch_dir`` for ONE download — two
    different keys/URLs that happen to share a basename (``s3://b1/dir1/
    data.csv`` and ``s3://b2/dir2/data.csv``) must never land on the same
    local path and silently overwrite one another."""
    dest = scratch_dir / uuid.uuid4().hex[:8]
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _download_s3(uri: str, scratch_dir: Path) -> str:
    """``s3://bucket/key`` -> local path under its own subdirectory of
    ``scratch_dir``. The bucket comes from the URI itself, not
    ``BUCKET_NAME`` (the caller may reference any bucket the credentials can
    read). Errors propagate (wrapped with the URI so the failure is
    attributable) — a tool must not silently run on missing input."""
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    local_path = _download_dir(scratch_dir) / _safe_filename(Path(key).name)
    try:
        _client_factory().download_file(bucket, key, str(local_path))
    except Exception as exc:
        local_path.unlink(missing_ok=True)
        raise RuntimeError(f"failed to fetch {uri}: {exc}") from exc
    return str(local_path)


def _check_content_length(uri: str, response, max_bytes: int) -> None:
    """Reject an HTTP download up front when the server advertises a size
    over the cap, instead of only catching it mid-stream (still enforced by
    the read loop below for servers that lie about or omit the header).

    ``getattr(response, "headers", None) and response.headers.get(...)`` looks
    equivalent but is not: urllib's ``response.headers`` is an
    ``email.message.Message``, which defines ``__len__`` — a response with
    ZERO headers is a falsy-but-real ``Message``, so ``and`` would short-
    circuit to that Message object itself instead of ``None``, and
    ``int(Message)`` raises ``TypeError`` (uncaught below), turning a
    perfectly valid download into a crash. Get ``headers`` first and check
    ``is not None`` explicitly."""
    headers = getattr(response, "headers", None)
    length = headers.get("Content-Length") if headers is not None else None
    if length is None:
        return
    try:
        advertised = int(length)
    except (TypeError, ValueError):
        return
    if advertised > max_bytes:
        raise ValueError(
            f"download from {uri!r} advertises {advertised} bytes, over the "
            f"{max_bytes}-byte limit (S3_HTTP_MAX_BYTES)")


class _DeadlineExceeded(TimeoutError):
    """Raised internally when the overall ``S3_HTTP_TIMEOUT`` wall-clock
    budget (as opposed to urllib's own per-read socket timeout) is exceeded.
    Kept as its own subclass so it is never double-wrapped: it already
    carries the URI in its message and must propagate as-is, while a bare
    ``socket.timeout``/``TimeoutError`` from urllib itself (a stalled
    connection, not our deadline check) still needs ``{uri}`` added to be
    attributable — it is still a plain ``TimeoutError``, not this subclass,
    so it falls through to the generic wrap below. Still catchable by callers
    as ``TimeoutError`` either way."""


def _download_http(uri: str, scratch_dir: Path) -> str:
    """``http(s)://...`` -> local path under its own subdirectory of
    ``scratch_dir``, no credentials attached (a plain fetch of a public/
    presigned URL). Bounded three ways: an ``S3_HTTP_TIMEOUT``-second socket
    timeout on each connect/read, the SAME budget re-applied as an overall
    wall-clock deadline across the read loop (a peer trickling data one byte
    at a time never trips a per-read timeout but must still not run for
    hours), and an ``S3_HTTP_MAX_BYTES`` cap checked against ``Content-Length``
    up front and against bytes actually received as they arrive. A partially
    written file is removed before the error propagates."""
    # A URL path is percent-encoded: данные.csv arrives as %D0%B4...csv.
    local_path = _download_dir(scratch_dir) / _safe_filename(
        unquote(Path(urlparse(uri).path).name))
    timeout = _http_timeout()
    max_bytes = _http_max_bytes()
    deadline = time.monotonic() + timeout
    try:
        with urllib.request.urlopen(uri, timeout=timeout) as response:
            _check_content_length(uri, response, max_bytes)
            written = 0
            with open(local_path, "wb") as fh:
                while True:
                    if time.monotonic() > deadline:
                        raise _DeadlineExceeded(
                            f"download from {uri!r} exceeded the {timeout}s overall "
                            f"deadline (S3_HTTP_TIMEOUT)")
                    chunk = response.read(_HTTP_CHUNK_SIZE)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(
                            f"download from {uri!r} exceeded the {max_bytes}-byte "
                            f"limit (S3_HTTP_MAX_BYTES)")
                    fh.write(chunk)
    except (_DeadlineExceeded, ValueError):
        local_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        # Includes a bare socket timeout (socket.timeout IS TimeoutError as
        # of Python 3.10) and any other urllib/OS failure — none of those
        # carry the URI on their own, so it's added here.
        local_path.unlink(missing_ok=True)
        raise RuntimeError(f"failed to fetch {uri}: {exc}") from exc
    return str(local_path)


def resolve_input(value: object, scratch_dir: Path) -> object:
    """``s3://...`` and ``http(s)://...`` values (scheme matched case-
    insensitively) are downloaded into ``scratch_dir`` and replaced by the
    local path; anything else (a local path, a non-string value) is returned
    unchanged. Existence of a local path is deliberately never checked here —
    that stays the tool function's job, same as before this module existed."""
    if isinstance(value, str):
        lowered = value.lower()
        if lowered.startswith("s3://"):
            return _download_s3(value, scratch_dir)
        if lowered.startswith(("http://", "https://")):
            return _download_http(value, scratch_dir)
    return value


def prepare_kwargs(kwargs: dict, scratch_dir: Path) -> dict:
    """``resolve_input`` applied to every ``*_path``/``*_file`` kwarg; every
    other kwarg passes through untouched."""
    return {
        key: (resolve_input(value, scratch_dir) if is_file_param(key) else value)
        for key, value in kwargs.items()
    }


def maybe_upload(local_path: str, prefix: str, field_key: str) -> dict | None:
    """Upload ``local_path`` to ``<prefix>/<field_key>/<basename>`` and
    presign it. ``field_key`` (the result field this file came from) is part
    of the key so two different fields sharing a basename never collide and
    overwrite each other in the bucket. The basename itself goes through
    ``_key_safe_filename`` (not the raw ``Path(local_path).name``) so a
    space/``+``/non-ASCII character in it can't build a key the vault's own
    regex later rejects (see ``_key_safe_filename``'s docstring) — this is
    purely for the S3 KEY, the local path on disk is untouched. When
    sanitizing actually changed the name (e.g. two different fully non-ASCII
    stems both collapse to ``"download"``), a 6-hex digest of the RAW
    basename is appended so the two still get distinct keys under the same
    ``field_key`` — an ASCII basename that passes through untouched is left
    alone. Returns
    ``{"bucket", "s3_key",
    "presigned_url"}`` on success, ``None`` on any failure (S3 not
    configured, upload error) — logged to stderr so the reason is visible in
    ``docker logs``, but never raised: a publish failure must not take down
    an otherwise-successful tool call. ``bucket`` is returned alongside the
    key because ``publish_result`` below nests this whole dict under the
    result — the framework's artifact walker (CoScientist/utils/s3_refs.py:
    _walk) only recognises a durable reference when ``bucket`` and ``s3_key``
    sit together in one dict. Only the exception TYPE and a truncated message
    are logged, never ``str(exc)`` verbatim — a botocore error can embed
    request internals (e.g. ``AWSAccessKeyId=...`` on a presigned-URL-shaped
    error), and this line lands in ``docker logs``."""
    if not s3_enabled():
        return None
    try:
        # A tool returning a checkpoint path must not hold the call for the
        # whole upload, and the report collector later downloads what it finds.
        size, limit = Path(local_path).stat().st_size, _upload_max_bytes()
        if size > limit:
            print(f"[s3] upload skipped for {local_path}: {size} bytes is over the "
                  f"{limit}-byte limit (S3_UPLOAD_MAX_BYTES)", file=sys.stderr)
            return None
        client = _client_factory()
        bucket = _env("BUCKET_NAME")
        raw_name = Path(local_path).name
        safe_name = _key_safe_filename(raw_name)
        if safe_name != raw_name:
            digest = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:6]
            p = Path(safe_name)
            safe_name = f"{p.stem}_{digest}{p.suffix}"
        key = f"{prefix}/{safe_component(field_key)}/{safe_name}"
        client.upload_file(local_path, bucket, key)
        return _presigned(client, bucket, key)
    except Exception as exc:  # noqa: BLE001 - a publish failure must not fail the call
        print(f"[s3] upload failed for {local_path}: {type(exc).__name__}: {str(exc)[:200]}",
              file=sys.stderr)
        return None


def _presigned(client, bucket: str, key: str) -> dict:
    """The durable reference to an uploaded object and a link to download it."""
    external = _external_endpoint()
    presigner = _client_factory(external) if external else client
    expires_in = _presign_expiration()
    url = presigner.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )
    return {"bucket": bucket, "s3_key": key, "presigned_url": url,
            "expires_in": expires_in}


def upload_json(text: str, prefix: str, field_key: str) -> dict | None:
    """Upload a JSON document to ``<prefix>/<field_key>/result.json`` and presign
    it. Same contract as ``maybe_upload``: ``None`` on any failure, logged by
    exception type only, never raised."""
    if not s3_enabled():
        return None
    try:
        body = text.encode("utf-8")
        limit = _upload_max_bytes()
        if len(body) > limit:
            print(f"[s3] upload skipped for the {field_key} JSON: {len(body)} bytes is over "
                  f"the {limit}-byte limit (S3_UPLOAD_MAX_BYTES)", file=sys.stderr)
            return None
        client = _client_factory()
        bucket = _env("BUCKET_NAME")
        key = f"{prefix}/{safe_component(field_key)}/result.json"
        client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        return _presigned(client, bucket, key)
    except Exception as exc:  # noqa: BLE001 - a publish failure must not fail the call
        print(f"[s3] upload failed for the {field_key} JSON: {type(exc).__name__}: "
              f"{str(exc)[:200]}", file=sys.stderr)
        return None


def _preview(value: object, items: int, chars: int, path: str, notes: dict) -> object:
    """A copy of ``value`` with lists and dicts cut to ``items`` entries and
    strings to ``chars`` characters; each cut is recorded in ``notes`` under its
    path. An S3 reference (a dict with ``bucket`` and ``s3_key``) and a
    ``<field>_s3`` entry are kept whole: they are how the caller reaches the data."""
    if isinstance(value, dict):
        if "bucket" in value and "s3_key" in value:
            return value
        keys = list(value)
        kept = keys
        if len(keys) > items:
            kept = keys[:items] + [k for k in keys[items:] if str(k).endswith("_s3")]
            notes[path or "result"] = {"keys": len(keys), "kept": len(kept)}
        return {k: _preview(value[k], items, chars, f"{path}.{k}" if path else str(k), notes)
                for k in kept}
    if isinstance(value, list):
        if len(value) > items:
            notes[path or "result"] = {"items": len(value), "kept": items}
            value = value[:items]
        return [_preview(v, items, chars, f"{path}[{i}]", notes) for i, v in enumerate(value)]
    if isinstance(value, str) and len(value) > chars:
        notes[path or "result"] = {"chars": len(value), "kept": chars}
        return value[:chars]
    return value


def shrink_result(result: object, prefix: str) -> object:
    """Put a result over ``ALEMBIC_RESULT_MAX_CHARS`` (as JSON) in S3 whole and
    return a copy with lists, dicts and strings cut until it fits, plus
    ``result_truncated`` (what was cut) and ``result_s3`` (the full JSON).

    A result that fits, a non-dict, one already using those names, or a failed
    upload comes back unchanged."""
    if not isinstance(result, dict) or "result_s3" in result or "result_truncated" in result:
        return result
    limit = _result_max_chars()
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return result
    uploaded = upload_json(text, prefix, "result")
    if not uploaded:
        return result
    items, chars = _PREVIEW_ITEMS, _PREVIEW_CHARS
    while True:
        notes: dict = {}
        short = _preview(result, items, chars, "", notes)
        listed = dict(list(notes.items())[:_PREVIEW_MAX_NOTES])
        if len(notes) > len(listed):
            listed["..."] = f"{len(notes) - len(listed)} more paths shortened"
        short["result_truncated"] = {
            "full_chars": len(text),
            "shortened": listed,
            "rest": ("The complete result is the JSON file in result_s3. Download it with "
                     "result_s3.presigned_url, or by bucket and s3_key once the link has "
                     "expired, and extract only the fields you need."),
        }
        short["result_s3"] = uploaded
        fits = len(json.dumps(short, ensure_ascii=False, default=str)) <= limit
        if fits or (items == 1 and chars == 100):
            return short
        items, chars = max(1, items // 2), max(100, chars // 2)


def _is_publishable(key: str, value: object, deny_roots: tuple,
                    repo_root: Path | None = None, since: float | None = None) -> bool:
    """True iff ``value`` is an existing regular file, named like a file
    param, and NOT inside any of ``deny_roots``: the per-call scratch dir (a
    tool that echoes its ``input_path`` back would otherwise re-upload the
    caller's own input and hand back a local path that the post-call scratch
    cleanup is about to delete) and the mounted benchmark data.

    Inside ``repo_root`` (the cloned repo) only a file written at or after
    ``since`` is publishable. Source and data files a tool echoes back stay
    out of the bucket, while an output the call wrote there goes up:
    generated tools often resolve a relative out path against the repo."""
    if not (is_file_param(key) and isinstance(value, str)):
        return False
    path = Path(value)
    if not path.is_file():
        return False
    try:
        resolved = path.resolve()
        if any(resolved.is_relative_to(root.resolve()) for root in deny_roots):
            return False
        if repo_root is not None and resolved.is_relative_to(repo_root.resolve()):
            # One second of slack for filesystems that store whole-second mtimes.
            return since is not None and path.stat().st_mtime >= since - 1
        return True
    except OSError:
        return False


def publish_result(result: object, prefix: str, deny_roots,
                   repo_root: Path | None = None, since: float | None = None) -> object:
    """Recursively walk a dict/list tool result. For every ``*_path``/
    ``*_file`` string entry that is an existing, publishable local file,
    upload it under ``prefix`` and add one nested ``<key>_s3`` entry:
    ``{"bucket", "s3_key", "presigned_url"}`` — the original local path is
    left untouched (it is still correct inside the build container).

    A single nested dict, not the flat ``<key>_s3_key`` / ``<key>
    _presigned_url`` siblings this used to add: the framework's artifact
    walker (CoScientist/utils/s3_refs.py:_walk) only records a durable
    reference from a dict that carries BOTH ``bucket`` and ``s3_key`` as
    literal keys — a ``bucket``-less sibling pair sitting next to each other
    at the same level is invisible to it, since a bare ``s3_key`` cannot say
    which bucket holds it. The walker recurses into nested dicts, so this one
    is found the same way every MCP server's own return contract is (see
    mcp-servers/chemical-mcp-server/server/utils/vault.py:contract).

    ``deny_roots`` is a single ``Path`` or an iterable of them; files under
    any deny root are never uploaded, and files under ``repo_root`` only when
    written at or after ``since`` (see ``_is_publishable``). A result still
    over ``ALEMBIC_RESULT_MAX_CHARS`` afterwards is shortened (``shrink_result``)."""
    roots = (deny_roots,) if isinstance(deny_roots, Path) else tuple(deny_roots)
    # Shortened last, so the full copy in S3 carries the <key>_s3 entries too.
    # Here and not in server.py: a server generated earlier calls publish_result
    # and gets this with a new helpers/s3_transfer.py alone.
    return shrink_result(_publish(result, prefix, roots, repo_root, since), prefix)


def _publish(result: object, prefix: str, roots: tuple,
             repo_root: Path | None, since: float | None) -> object:
    if isinstance(result, dict):
        out = {}
        for key, value in result.items():
            out[key] = _publish(value, prefix, roots, repo_root, since)
            # Guarded against `result` (the tool's own dict), not `out`: dict
            # iteration order means a tool-returned `<key>_s3` sibling could
            # sit either before or after `key` in `result` — checking `out`
            # would only catch the "before" ordering and still clobber the
            # tool's own field in the other one.
            if (_is_publishable(key, value, roots, repo_root, since)
                    and f"{key}_s3" not in result):
                uploaded = maybe_upload(value, prefix, key)
                if uploaded:
                    out[f"{key}_s3"] = uploaded
        return out
    if isinstance(result, list):
        return [_publish(item, prefix, roots, repo_root, since) for item in result]
    return result


# The build installs boto3 into .venv-server, but a failed install only logs a
# warning there and the build goes on. Say so once at server start, so that the
# per-call "No module named 'boto3'" errors have a visible cause in docker logs.
if s3_enabled() and importlib.util.find_spec("boto3") is None:
    print("[s3] S3 is configured but boto3 is not installed in this venv: "
          "s3:// inputs will fail and outputs will not be uploaded", file=sys.stderr)
