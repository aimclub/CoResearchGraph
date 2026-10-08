import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastmcp import FastMCP

mcp = FastMCP("VaultServer")


def _env(primary: str, fallback: str, default: str) -> str:
    return os.environ.get(primary) or os.environ.get(fallback) or default


S3_ENDPOINT = _env('S3__ENDPOINT_URL', 'MINIO_ENDPOINT', 'http://minio:9000')
EXTERNAL_ENDPOINT = _env('S3__EXTERNAL_ENDPOINT_URL', 'EXTERNAL_MINIO_ENDPOINT', 'http://localhost:9000')
ACCESS_KEY = _env('S3__ACCESS_KEY', 'MINIO_ACCESS_KEY', 'agent-user')
SECRET_KEY = _env('S3__SECRET_KEY', 'MINIO_SECRET_KEY', 'agent-secret-key')
BUCKET_NAME = _env('S3__BUCKET_NAME', 'MINIO_BUCKET', 'agent-vault')

EPHEMERAL_TTL_DAYS = int(os.environ.get('EPHEMERAL_TTL_DAYS', '7'))
EPHEMERAL_TTL_SECONDS = EPHEMERAL_TTL_DAYS * 86400
SIGV4_MAX_EXPIRES = 604800  # 7 days. Hard SigV4 limit for presigned URLs.

VAULT_SURFACE = os.environ.get('VAULT_SURFACE', 'all').lower()
if VAULT_SURFACE not in ('worker', 'framework', 'all'):
    raise ValueError(f"VAULT_SURFACE must be worker, framework or all, got '{VAULT_SURFACE}'")

# Bounds for one S3 call. The botocore defaults are 60 s to connect, 60 s to
# read, and up to 5 attempts, so an endpoint that drops packets blocks a tool
# for about 300 s. The MCP client gives up at 300 s and closes the stream, and
# the server then logs a ClosedResourceError for a response nobody waits for.
_client_config = Config(
    signature_version='s3v4',
    s3={'addressing_style': 'path'},
    connect_timeout=5,
    read_timeout=30,
    retries={'max_attempts': 3, 'mode': 'standard'},
)

# Internal client for direct operations.
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='us-east-1',
    config=_client_config,
)

# External client for pre-signing, so the Host matches what the caller sees.
signing_client = boto3.client(
    's3',
    endpoint_url=EXTERNAL_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='us-east-1',
    config=_client_config,
)

_ID_RE = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')
_KEY_RE = re.compile(r'^(ephemeral|permanent)/[a-zA-Z0-9_-]{1,64}/[a-zA-Z0-9_-]{1,64}/[a-zA-Z0-9_/.-]+$')
_FILENAME_RE = re.compile(r'^[a-zA-Z0-9_.-]{1,128}$')


def _err(message: str) -> str:
    return json.dumps({"error": message})


def _check_id(value: str, name: str) -> None:
    if not value or not _ID_RE.match(value):
        raise ValueError(f"invalid {name}: must match {_ID_RE.pattern}")


def _check_key(key: str) -> None:
    if not key or not _KEY_RE.match(key) or '..' in key.split('/'):
        raise ValueError(
            "invalid s3_key: must match <retention>/<user_id>/<session_id>/<path> "
            "with no '..' segments"
        )


def _build_key(user_id: str, session_id: str, filename: str, feature: Optional[str]) -> str:
    _check_id(user_id, 'user_id')
    _check_id(session_id, 'session_id')
    if not filename or not _FILENAME_RE.match(filename):
        raise ValueError(f"invalid filename: must match {_FILENAME_RE.pattern}")
    parts = ['ephemeral', user_id, session_id]
    if feature:
        _check_id(feature, 'feature')
        parts.append(feature)
    parts.append(filename)
    return '/'.join(parts)


def _contract(key: str, url: str, expires_in: Optional[int], url_field: str = 'presigned_url') -> str:
    return json.dumps({
        "bucket": BUCKET_NAME,
        "s3_key": key,
        url_field: url,
        "expires_in": expires_in,
    }, indent=2)


def _plain_url(key: str) -> str:
    return f"{EXTERNAL_ENDPOINT}/{BUCKET_NAME}/{key}"


def _remaining_ttl_seconds(key: str) -> int:
    head = s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
    age = (datetime.now(timezone.utc) - head['LastModified']).total_seconds()
    return int(EPHEMERAL_TTL_SECONDS - age)


def _description(key: str) -> str:
    try:
        tagging = s3_client.get_object_tagging(Bucket=BUCKET_NAME, Key=key)
        tags = {t['Key']: t['Value'] for t in tagging.get('TagSet', [])}
        return tags.get('Description', '')
    except ClientError:
        return ''


def _list_session_objects(user_id: str, session_id: str):
    for retention in ('ephemeral', 'permanent'):
        prefix = f'{retention}/{user_id}/{session_id}/'
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
            for obj in page.get('Contents', []):
                yield retention, obj


def _tool(surface):
    """Registers the function as an MCP tool if the surface is active."""
    def wrap(fn):
        if VAULT_SURFACE in ('all', surface):
            return mcp.tool()(fn)
        return fn
    return wrap


# --- Worker surface ---

@_tool('worker')
def get_upload_link(user_id: str, session_id: str, filename: str, feature: Optional[str] = None) -> str:
    """Gets a temporary URL to upload a file to the vault. Uploads land under ephemeral/.
    Use a plain HTTP PUT with the file bytes as the body. No custom headers are needed."""
    try:
        key = _build_key(user_id, session_id, filename, feature)
        expires_in = min(EPHEMERAL_TTL_SECONDS, SIGV4_MAX_EXPIRES)
        upload_url = signing_client.generate_presigned_url(
            ClientMethod='put_object',
            # Do not sign a content type. It would join X-Amz-SignedHeaders, and
            # then a PUT without that exact header fails with SignatureDoesNotMatch.
            Params={'Bucket': BUCKET_NAME, 'Key': key},
            ExpiresIn=expires_in,
        )
        return _contract(key, upload_url, expires_in, url_field='upload_url')
    except (ValueError, ClientError) as e:
        return _err(str(e))


@_tool('worker')
def get_download_link(s3_key: str) -> str:
    """Gets an HTTP link to read an artifact. Ephemeral objects return a presigned URL
    that expires with the object. Permanent objects return a plain URL that never expires."""
    try:
        _check_key(s3_key)
        if s3_key.startswith('permanent/'):
            return _contract(s3_key, _plain_url(s3_key), None)

        try:
            remaining = _remaining_ttl_seconds(s3_key)
        except ClientError:
            return _err(f"object not found: {s3_key}")
        if remaining <= 0:
            return _err(f"object expired or eligible for expiry: {s3_key}")

        url = signing_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': s3_key},
            ExpiresIn=min(remaining, SIGV4_MAX_EXPIRES),
        )
        return _contract(s3_key, url, min(remaining, SIGV4_MAX_EXPIRES))
    except ValueError as e:
        return _err(str(e))


# --- Framework surface ---

@_tool('framework')
def promote_artifact(s3_key: str) -> str:
    """Copies an ephemeral artifact to permanent/ and returns the new key. The source
    object stays until the lifecycle rule reclaims it. Call this at finalize time,
    after all workers complete."""
    try:
        _check_key(s3_key)
        if not s3_key.startswith('ephemeral/'):
            return _err(f"only ephemeral objects can be promoted: {s3_key}")
        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        except ClientError:
            return _err(f"object not found: {s3_key}")

        new_key = 'permanent/' + s3_key[len('ephemeral/'):]
        s3_client.copy_object(
            Bucket=BUCKET_NAME,
            Key=new_key,
            CopySource={'Bucket': BUCKET_NAME, 'Key': s3_key},
        )
        return _contract(new_key, _plain_url(new_key), None)
    except (ValueError, ClientError) as e:
        return _err(str(e))


@_tool('framework')
def cleanup_session(user_id: str, session_id: str, confirm: bool = False) -> str:
    """Deletes every object of a session under both ephemeral/ and permanent/.
    Dry-run by default: returns the object count without deleting. Pass confirm=True
    to delete."""
    try:
        _check_id(user_id, 'user_id')
        _check_id(session_id, 'session_id')
        keys = [obj['Key'] for _, obj in _list_session_objects(user_id, session_id)]

        if not confirm:
            return json.dumps({
                "dry_run": True,
                "object_count": len(keys),
                "objects": keys[:50],
                "truncated": len(keys) > 50,
            }, indent=2)

        deleted = 0
        errors = []
        for i in range(0, len(keys), 1000):
            batch = keys[i:i + 1000]
            # delete_objects reports a per-object refusal in Errors and still
            # answers 200. Counting the batch would call a denied delete a
            # success, and the caller would believe the session is gone.
            resp = s3_client.delete_objects(
                Bucket=BUCKET_NAME,
                Delete={'Objects': [{'Key': k} for k in batch]},
            )
            deleted += len(resp.get('Deleted', []))
            errors.extend(resp.get('Errors', []))

        result = {"dry_run": False, "deleted": deleted, "requested": len(keys)}
        if errors:
            result["failed"] = len(errors)
            result["errors"] = [
                {"key": e.get('Key'), "code": e.get('Code')} for e in errors[:50]
            ]
        return json.dumps(result, indent=2)
    except (ValueError, ClientError) as e:
        return _err(str(e))


@_tool('framework')
def update_artifact_metadata(s3_key: str, description: str) -> str:
    """Sets a short description on an artifact. The value is truncated to 256
    characters (the S3 tag value limit)."""
    try:
        _check_key(s3_key)
        try:
            tagging = s3_client.get_object_tagging(Bucket=BUCKET_NAME, Key=s3_key)
            tags = {t['Key']: t['Value'] for t in tagging.get('TagSet', [])}
        except ClientError:
            return _err(f"object not found: {s3_key}")

        tags['Description'] = description[:256]
        s3_client.put_object_tagging(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Tagging={'TagSet': [{'Key': k, 'Value': v} for k, v in tags.items()]},
        )
        return json.dumps({"s3_key": s3_key, "description": tags['Description']}, indent=2)
    except ValueError as e:
        return _err(str(e))


@_tool('framework')
def get_session_manifest(user_id: str, session_id: str) -> str:
    """Returns a derived manifest of every artifact of a session: keys, sizes,
    retention class, and descriptions. Computed from listings, not stored."""
    try:
        _check_id(user_id, 'user_id')
        _check_id(session_id, 'session_id')
        artifacts = []
        for retention, obj in _list_session_objects(user_id, session_id):
            artifacts.append({
                "s3_key": obj['Key'],
                "retention": retention,
                "size": obj['Size'],
                "last_modified": obj['LastModified'].isoformat(),
                "description": _description(obj['Key']),
            })
        return json.dumps({
            "user_id": user_id,
            "session_id": session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifacts,
        }, indent=2)
    except (ValueError, ClientError) as e:
        return _err(str(e))


@_tool('framework')
def list_artifacts(user_id: str, session_id: str, feature: Optional[str] = None) -> str:
    """Lists the artifacts of one session with tags. Optionally filters by feature."""
    try:
        _check_id(user_id, 'user_id')
        _check_id(session_id, 'session_id')
        if feature:
            _check_id(feature, 'feature')

        artifacts = {}
        for retention, obj in _list_session_objects(user_id, session_id):
            key = obj['Key']
            if feature and key.split('/')[3:-1] != [feature]:
                continue
            artifacts[key] = {
                "retention": retention,
                "size": obj['Size'],
                "last_modified": obj['LastModified'].isoformat(),
                "description": _description(key),
            }
        return json.dumps(artifacts, indent=2)
    except (ValueError, ClientError) as e:
        return _err(str(e))


@mcp.resource("vault://{access_key*}")
def read_resource(access_key: str) -> str:
    """Reads an artifact directly from the vault. The key must be a valid vault key."""
    try:
        _check_key(access_key)
        resp = s3_client.get_object(Bucket=BUCKET_NAME, Key=access_key)
        return resp['Body'].read().decode('utf-8')
    except (ValueError, ClientError) as e:
        return f"Error reading resource: {str(e)}"


if __name__ == "__main__":
    host = os.environ.get('MCP_HOST', '0.0.0.0')
    port = int(os.environ.get('MCP_PORT', 8000))

    mcp.run(
        transport="http",
        host=host,
        port=port,
        log_level="info",
    )
