from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .client import normalize_doi, openalex_id


FINAL_STATUSES = frozenset({
    'ingested',
    'already_in_rag',
    'domain_filtered',
    'not_open_access',
    'missing_openalex_id',
    'no_usable_pdf',
})


def publication_key(publication):
    """Return a stable registry key for a Sapphire publication dictionary.

    Prefer the canonical OpenAlex ID, followed by DOI. A deterministic hash of
    the complete record covers malformed records without either identifier.
    """
    identifier = openalex_id(publication.get('openalex_id'))
    if identifier:
        return f'openalex:{identifier}'
    doi = normalize_doi(publication.get('doi'))
    if doi:
        return f'doi:{doi}'
    payload = json.dumps(publication, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)
    return f'hash:{hashlib.sha256(payload.encode("utf-8")).hexdigest()}'


class PublicationRegistry:
    """Persist Sapphire publication outcomes in a local SQLite database."""

    def __init__(self, path):
        """Open the registry at path and create its schema when absent."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA busy_timeout=5000')
        self.connection.execute(
            '''
            CREATE TABLE IF NOT EXISTS publications (
                registry_key TEXT PRIMARY KEY,
                openalex_id TEXT,
                doi TEXT,
                name TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                payload_json TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            '''
        )
        self.connection.execute(
            'CREATE INDEX IF NOT EXISTS idx_publications_status ON publications(status)'
        )
        self.connection.commit()

    def close(self):
        """Commit pending changes and close the SQLite connection."""
        self.connection.commit()
        self.connection.close()

    def __enter__(self):
        """Return this registry for use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the registry when leaving a context manager block."""
        self.close()

    def register(self, publication):
        """Insert or refresh a publication and return its stable registry key.

        Existing processing state, attempt count, and first-seen timestamp are
        preserved while identifiers, title, and the source payload are refreshed.
        """
        key = publication_key(publication)
        now = datetime.now(timezone.utc).isoformat()
        values = (
            key,
            openalex_id(publication.get('openalex_id')),
            normalize_doi(publication.get('doi')),
            publication.get('name') if isinstance(publication.get('name'), str) else None,
            json.dumps(publication, ensure_ascii=False, sort_keys=True, default=str),
            now,
            now,
        )
        self.connection.execute(
            '''
            INSERT INTO publications (
                registry_key, openalex_id, doi, name,
                payload_json, first_seen_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(registry_key) DO UPDATE SET
                openalex_id=excluded.openalex_id,
                doi=excluded.doi,
                name=excluded.name,
                payload_json=excluded.payload_json,
                updated_at=excluded.updated_at
            ''',
            values,
        )
        self.connection.commit()
        return key

    def status(self, key):
        """Return the stored status for key, or None when it is unknown."""
        row = self.connection.execute(
            'SELECT status FROM publications WHERE registry_key = ?', (key,)
        ).fetchone()
        return row['status'] if row else None

    def start_attempt(self, key):
        """Mark a publication as processing and increment its attempt counter."""
        self.connection.execute(
            '''
            UPDATE publications
            SET status = 'processing', attempts = attempts + 1,
                last_error = NULL, updated_at = ?
            WHERE registry_key = ?
            ''',
            (datetime.now(timezone.utc).isoformat(), key),
        )
        self.connection.commit()

    def finish(self, key, status, error=None):
        """Store the latest outcome and optional error for a publication."""
        self.connection.execute(
            '''
            UPDATE publications
            SET status = ?, last_error = ?, updated_at = ?
            WHERE registry_key = ?
            ''',
            (status, error, datetime.now(timezone.utc).isoformat(), key),
        )
        self.connection.commit()
