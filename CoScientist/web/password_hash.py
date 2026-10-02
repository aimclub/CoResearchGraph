"""The stored-credential format, in one place.

This module is a leaf on purpose. It imports nothing from ``CoScientist``, so
``deploy/make_password_hash.py`` can load it by path and reuse the format
without importing the package — which builds the whole agent system and needs a
complete LLM configuration to do it. An operator who is still writing the
``.env`` file has neither.

Both readers of the format therefore agree by construction. When they did not,
the failure was quiet and expensive: the script printed a line the server
rejected, the gate failed closed, and the deploy health check reported 503 with
no reason an operator could act on.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import secrets
from typing import Optional

#: Stored-credential format: ``pbkdf2_sha256:<rounds>:<salt>:<digest>``, the
#: last two base64url without padding. Separated by ":" and not the usual "$":
#: a "$" makes the value expand to nothing the moment somebody pastes the line
#: into a shell, and the result still looks like a hash.
SCHEME = "pbkdf2_sha256"

#: Work factor. Deliberately below the figure recommended for human-chosen
#: passwords: the README already requires 20 or more random characters, and
#: against that much entropy stretching buys almost nothing. The digest is here
#: so a leaked .env carries no password, not to survive an offline attack on a
#: weak one.
ROUNDS = 200_000

#: The environment key that holds the value this module builds.
ENV_KEY = "AUTH__PASSWORD_HASH"


def b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str, *, rounds: int = ROUNDS,
                  salt: Optional[bytes] = None) -> str:
    """Build the value an operator puts in ``AUTH__PASSWORD_HASH``."""
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return ":".join((SCHEME, str(rounds), b64(salt), b64(digest)))


def parse_hash(stored: str) -> Optional[tuple[int, bytes, bytes]]:
    """Split a stored digest, or return None when it is not one.

    Parsing is lazy, never a pydantic validator. ``Settings()`` runs at module
    import, so a typo checked there would stop the server before any code can
    report why. A None here makes ``is_configured`` false, which routes the
    problem through the 503 banner the operator can actually read.
    """
    parts = stored.strip().split(":")
    if len(parts) != 4 or parts[0] != SCHEME:
        return None
    try:
        rounds = int(parts[1])
        salt = unb64(parts[2])
        digest = unb64(parts[3])
    except (ValueError, binascii.Error):
        return None
    if rounds < 1 or not salt or not digest:
        return None
    return rounds, salt, digest


def verify_password(given: str, stored: str) -> bool:
    """True when *given* matches the digest in *stored*."""
    parsed = parse_hash(stored)
    if parsed is None:
        return False
    rounds, salt, expected = parsed
    digest = hashlib.pbkdf2_hmac("sha256", given.encode("utf-8"), salt, rounds)
    return hmac.compare_digest(digest, expected)
