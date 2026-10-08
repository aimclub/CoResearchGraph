"""Find durable S3 file references in tool arguments and tool results.

Every MCP server in this repository returns ``bucket`` and ``s3_key`` next to a
``presigned_url``. The pair is the durable reference: the object lives on, while
the URL expires (one hour for most servers, the object lifetime for the vault).
So the execution graph, the artifact index, and the report manifest all store
``s3://<bucket>/<key>`` and mint a fresh URL when they need one.

A tool result reaches this module in several shapes. An MCP call returns a
``CallToolResult`` whose payload is a JSON string inside ``content[].text``. A
native tool returns a plain dict. Some servers nest the pair under a
``metadata`` key. The walk below handles all of them.

A dict that carries ``s3_key`` but no ``bucket`` is skipped. A key alone does not
say which bucket holds it, so it cannot be resolved later.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# Matches an s3://bucket/key string already embedded in a payload. The dataset
# collection server returns this form directly.
# The key stops at whitespace, a quote, or a bracket. A trailing period is
# sentence punctuation far more often than the last character of a key.
_S3_URI_RE = re.compile(r"s3://[a-zA-Z0-9][a-zA-Z0-9.\-_]{1,254}/[^\s\"'<>,\]}]*[^\s\"'<>,.\]}]")

# A node records the files a call touched, not a file listing. A tool that
# returns hundreds of keys is a listing, and the graph is the wrong place for it.
_MAX_REFS = 50

# Nested tool envelopes are a few levels deep. This only stops a cycle in a
# non-JSON object reached through its __dict__.
_MAX_DEPTH = 12

# The URL field that sits beside a bucket/s3_key pair, in order of preference.
_URL_FIELDS = ("presigned_url", "upload_url", "url")


def s3_uri(bucket: str, key: str) -> str:
    """Build the durable reference for one object."""
    return f"s3://{bucket}/{key.lstrip('/')}"


def split_s3_uri(uri: str) -> Optional[tuple]:
    """Split ``s3://bucket/key`` back into ``(bucket, key)``, or None."""
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        return None
    rest = uri[len("s3://"):]
    bucket, sep, key = rest.partition("/")
    if not sep or not bucket or not key:
        return None
    return bucket, key


def find_s3_uris(obj: Any) -> List[str]:
    """Collect ``s3://bucket/key`` references from a nested structure.

    Accepts a tool-result envelope, a tool-args dict, or a bare string. Returns
    unique references in the order they appear, at most :data:`_MAX_REFS`.
    """
    uris: List[str] = []
    _walk(obj, [], uris, 0)
    return uris


def find_s3_artifacts(obj: Any) -> List[Dict[str, Any]]:
    """Collect full artifact records: ``bucket``, ``s3_key``, and the URL the
    server returned beside them.

    The URL is a capability with an expiry, not a reference. Callers may use it
    while it is fresh, and must resolve the key when it is not.
    """
    records: List[Dict[str, Any]] = []
    _walk(obj, records, [], 0)
    return records


def _walk(obj: Any, records: List[Dict[str, Any]], uris: List[str], depth: int) -> None:
    if depth > _MAX_DEPTH or len(uris) >= _MAX_REFS or len(records) >= _MAX_REFS:
        return

    if isinstance(obj, dict):
        bucket, key = obj.get("bucket"), obj.get("s3_key")
        if isinstance(bucket, str) and isinstance(key, str) and bucket and key:
            uri = s3_uri(bucket, key)
            _add_uri(uris, uri)
            if not any(r.get("s3_uri") == uri for r in records) and len(records) < _MAX_REFS:
                url = next(
                    (obj[f] for f in _URL_FIELDS
                     if isinstance(obj.get(f), str) and obj[f].startswith("http")),
                    None,
                )
                records.append({"bucket": bucket, "s3_key": key, "s3_uri": uri, "url": url})
        for value in obj.values():
            _walk(value, records, uris, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _walk(value, records, uris, depth + 1)
    elif isinstance(obj, str):
        for match in _S3_URI_RE.findall(obj):
            _add_uri(uris, match)
        # An MCP result carries its payload as a JSON string. Parse and recurse,
        # or every bucket/s3_key pair a server returns stays invisible.
        if obj.lstrip()[:1] in "{[":
            try:
                _walk(json.loads(obj), records, uris, depth + 1)
            except Exception:  # noqa: BLE001 — a non-JSON string is normal
                pass
    elif obj is not None and not isinstance(obj, (int, float, bool)):
        # A Pydantic CallToolResult and friends: walk the fields they expose.
        fields = getattr(obj, "__dict__", None)
        if isinstance(fields, dict):
            _walk(fields, records, uris, depth + 1)
        else:
            _walk(repr(obj), records, uris, depth + 1)


def _add_uri(uris: List[str], uri: str) -> None:
    if len(uris) < _MAX_REFS and uri not in uris:
        uris.append(uri)


__all__ = ["s3_uri", "split_s3_uri", "find_s3_uris", "find_s3_artifacts"]
