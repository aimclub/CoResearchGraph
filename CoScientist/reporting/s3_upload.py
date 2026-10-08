"""Upload run artifacts to S3 and hand back presigned URLs.

The web UI cannot open the relative local paths that reports and agent outputs
emit for figures and tables. This helper uploads an artifact file to the
configured bucket and returns a presigned GET URL the browser can open.

The S3 service is built lazily on first use: the module-level client in
``paper_parser/s3_connection.py`` is constructed at import time and fails when
S3 is not configured. Every failure path here returns None — S3 disabled,
missing credentials, upload error — and never raises, so the caller always
keeps its local fallback.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

_service = None
_service_broken = False


def _get_service():
    """Build the shared S3BucketService on first use. None when unusable."""
    global _service, _service_broken
    if _service is not None:
        return _service
    if _service_broken:
        return None
    try:
        from CoScientist.config import get_settings

        s3 = get_settings().s3
        if not s3.use_s3:
            return None
        if not (s3.endpoint_url and s3.access_key and s3.secret_key and s3.bucket_name):
            logger.warning("s3_upload: S3 is on but the credentials are incomplete")
            return None
        from CoScientist.paper_parser.s3_connection import S3BucketService

        _service = S3BucketService(
            endpoint=s3.endpoint_url,
            access_key=s3.access_key,
            secret_key=s3.secret_key,
            bucket_name=s3.bucket_name,
        )
    except Exception as exc:  # noqa: BLE001 - S3 must never break a run
        logger.warning("s3_upload: could not build the S3 service (%s)", exc)
        _service_broken = True
        return None
    return _service


def upload_and_presign(
    local_path: Union[str, Path],
    s3_key_prefix: str,
) -> Optional[str]:
    """Upload one local file to S3 and return a presigned GET URL for it.

    The S3 key is ``<s3_key_prefix>/<file name>``, so the object keeps the
    file name and extension (the frontend previews image links by extension).
    The URL stays valid for ``settings.s3.presign_ttl`` seconds. Returns None
    on any failure.
    """
    try:
        path = Path(local_path)
        if not path.is_file():
            return None
        service = _get_service()
        if service is None:
            return None
        from CoScientist.config import get_settings

        prefix = str(s3_key_prefix).strip("/")
        service.upload_file_object(prefix, path.name, str(path))
        key = f"{prefix}/{path.name}" if prefix else path.name
        return service.generate_presigned_url(
            key, expiration=get_settings().s3.presign_ttl
        )
    except Exception as exc:  # noqa: BLE001 - an upload error falls back to local
        logger.warning("s3_upload: upload failed for %s (%s)", local_path, exc)
        return None


def _reset_for_tests() -> None:
    """Drop the cached service so tests can re-configure S3."""
    global _service, _service_broken
    _service = None
    _service_broken = False


__all__ = ["upload_and_presign"]
