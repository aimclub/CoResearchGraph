"""Scoped S3 keys and the shared return contract.

Every object this server writes goes under one key layout:

    ephemeral/<user_id>/<session_id>/chemical_mcp/<feature>/<filename>

The server used to write everything under a global ``chemical_mcp/`` prefix, so
all users shared one namespace. Nothing matched a lifecycle rule either, so no
object ever expired. The ``ephemeral/`` top segment is what the bucket rule
filters on: S3 prefix filters are literal strings and cannot match a segment in
the middle of a key.

Every tool that touches an object returns the same three fields: ``bucket``,
``s3_key``, and ``presigned_url``. The bucket and the key are the durable
reference. The URL expires in an hour.
"""
from __future__ import annotations

import re

from ..service_resources import s3_service

_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_URL_TTL_SECONDS = 3600


def safe_id(value: str | None, default: str) -> str:
    """Make an id fit the vault key layout: ``^[a-zA-Z0-9_-]{1,64}$``."""
    cleaned = _ID_RE.sub("_", str(value or "")).strip("_")
    return cleaned[:64] or default


def scoped_prefix(user_id: str | None, session_id: str | None, feature: str) -> str:
    """The prefix this server writes one kind of artifact under."""
    return (
        f"ephemeral/{safe_id(user_id, 'unknown_user')}"
        f"/{safe_id(session_id, 'unknown_session')}"
        f"/chemical_mcp/{feature}"
    )


def contract(s3_key: str) -> dict:
    """The return contract for one stored object."""
    bucket = s3_service.bucket_name
    if not bucket:
        # Every consumer needs the pair. CoScientist.utils.s3_refs drops a record
        # that carries a key and no bucket, because a key alone does not say where
        # the object is. main() refuses to start without a bucket name, so this is
        # the second line of defense, for an s3_service built some other way.
        raise RuntimeError(
            "S3 bucket name is not configured, so the object at "
            f"{s3_key} has no durable reference. Set S3__BUCKET_NAME."
        )
    return {
        "bucket": bucket,
        "s3_key": s3_key,
        "presigned_url": s3_service.generate_presigned_url(
            s3_key, expiration=_URL_TTL_SECONDS
        ),
    }


def upload(user_id: str | None, session_id: str | None, feature: str,
           filename: str, data: bytes) -> dict:
    """Store bytes under the scoped prefix and return the contract."""
    s3_key = s3_service.upload_bytes(
        scoped_prefix(user_id, session_id, feature), filename, data
    )
    return contract(s3_key)


__all__ = ["safe_id", "scoped_prefix", "contract", "upload"]
