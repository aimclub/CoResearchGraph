"""read_result: take values out of a tool result a server stored whole in S3.

A server that shortens a big result (alembic's s3_transfer.shrink_result) returns
``result_s3`` beside the shortened copy; this reads the complete JSON and returns
only the values asked for.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_BYTES = int(os.getenv("READ_RESULT_MAX_BYTES", str(64 * 1024 * 1024)))
_TIMEOUT_SECONDS = 30
_LINK_TTL_SECONDS = 600
_MAX_MATCHES = 5
_MAX_DEPTH = 8
_MAX_SCANNED_ITEMS = 10000
_LIST_ITEMS = 20
_STR_CHARS = 1000


def _url_for(ref: str) -> str:
    """A download URL for ``ref``: a presigned link as is, else a fresh link for an ``s3_key``."""
    ref = html.unescape((ref or "").strip())
    if ref.lower().startswith(("http://", "https://")):
        return ref
    from CoScientist.reporting import s3_upload

    service = s3_upload._get_service()
    if service is None:
        raise RuntimeError("S3 is not configured here, so an s3_key cannot be read; "
                           "pass result_s3.presigned_url instead")
    return service.generate_presigned_url(ref.lstrip("/"), expiration=_LINK_TTL_SECONDS)


def _fetch(url: str) -> bytes:
    import requests

    with requests.get(url, timeout=_TIMEOUT_SECONDS, stream=True) as resp:
        resp.raise_for_status()
        chunks, size = [], 0
        for chunk in resp.iter_content(1024 * 1024):
            size += len(chunk)
            if size > _MAX_BYTES:
                raise ValueError(f"the result is over {_MAX_BYTES} bytes (READ_RESULT_MAX_BYTES)")
            chunks.append(chunk)
    return b"".join(chunks)


def _short(value: Any, depth: int = 0) -> Any:
    """``value`` with long lists and strings cut, so one field cannot flood the answer."""
    if isinstance(value, list):
        if len(value) > _LIST_ITEMS:
            return {"first": [_short(v, depth + 1) for v in value[:_LIST_ITEMS]],
                    "items": len(value)}
        return [_short(v, depth + 1) for v in value]
    if isinstance(value, dict):
        if depth >= 3:
            return f"dict[{len(value)}]"
        return {k: _short(v, depth + 1) for k, v in list(value.items())[:_LIST_ITEMS]}
    if isinstance(value, str) and len(value) > _STR_CHARS:
        return value[:_STR_CHARS] + f"… [{len(value)} chars]"
    return value


def _describe(value: Any) -> str:
    if isinstance(value, (list, dict, str)):
        return f"{type(value).__name__}[{len(value)}]"
    return type(value).__name__


def _join(path: str, key: Any) -> str:
    return f"{path}.{key}" if path else str(key)


def _find(obj: Any, name: str, path: str, depth: int, out: List[Dict[str, Any]]) -> None:
    """Record where ``name`` appears: as a dict key (its value), or as an entry
    of a list whose sibling lists of the same length hold the values."""
    if depth > _MAX_DEPTH or len(out) >= _MAX_MATCHES:
        return
    if isinstance(obj, dict):
        if name in obj:
            out.append({"value": _short(obj[name]), "where": _join(path, name)})
        lists = {k: v for k, v in obj.items() if isinstance(v, list)}
        for key, items in lists.items():
            if len(out) >= _MAX_MATCHES:
                return
            try:
                index = items.index(name)
            except ValueError:
                continue
            siblings = {k: v for k, v in lists.items() if k != key and len(v) == len(items)}
            if len(siblings) == 1:
                [(other, values)] = siblings.items()
                out.append({"value": _short(values[index]), "where": f"{_join(path, other)}[{index}]",
                            "matched": f"{_join(path, key)}[{index}]"})
            elif siblings:
                out.append({"values": {k: _short(v[index]) for k, v in siblings.items()},
                            "matched": f"{_join(path, key)}[{index}]"})
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                _find(value, name, _join(path, key), depth + 1, out)
    elif isinstance(obj, list):
        for index, item in enumerate(obj[:_MAX_SCANNED_ITEMS]):
            if len(out) >= _MAX_MATCHES:
                return
            if isinstance(item, (dict, list)):
                _find(item, name, f"{path}[{index}]", depth + 1, out)


def _load(ref: str) -> Any:
    body = _fetch(_url_for(ref))
    try:
        return json.loads(body)
    except ValueError as exc:
        raise ValueError(f"the file is not JSON ({exc}); it starts with "
                         f"{body[:120].decode('utf-8', 'replace')!r}") from exc


async def read_result(
    ref: str,
    find: Optional[List[str]] = None,
    keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Take values out of a tool result stored whole in S3, without reading all of it.

    Use it on a shortened tool result, one that carries ``result_truncated`` and
    ``result_s3``, when the values you need are not in the shortened part.

    Args:
        ref: ``result_s3.presigned_url`` (or its link reference) or ``result_s3.s3_key``.
        find: Names to look up anywhere in the result. A name found as a dict key
            gives that key's value. A name found in a list (e.g. ``names``) gives the
            entry at the same position of the other lists of that length beside it
            (e.g. ``values``). Up to 5 matches per name.
        keys: Top-level fields to return; long lists and strings are shortened.

    Returns:
        ``found`` (name -> match, or a list of matches), ``not_found``, ``fields``,
        ``missing_keys``, and ``structure`` (the type and size of every top-level
        field); or ``error``.
    """
    try:
        data = await asyncio.to_thread(_load, ref)
    except Exception as exc:  # noqa: BLE001 - reported to the agent, never raised
        logger.warning("read_result: could not read %s: %s", (ref or "")[:120], exc)
        return {"error": f"{type(exc).__name__}: {exc}"}
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        out["structure"] = {k: _describe(v) for k, v in list(data.items())[:50]}
    else:
        out["structure"] = _describe(data)
    if find:
        found, not_found = {}, []
        for name in find:
            matches: List[Dict[str, Any]] = []
            _find(data, name, "", 0, matches)
            if matches:
                found[name] = matches[0] if len(matches) == 1 else matches
            else:
                not_found.append(name)
        out["found"], out["not_found"] = found, not_found
    if keys:
        source = data if isinstance(data, dict) else {}
        out["fields"] = {k: _short(source[k]) for k in keys if k in source}
        out["missing_keys"] = [k for k in keys if k not in source]
    return out
