#!/usr/bin/env python3
"""Print the AUTH__PASSWORD_HASH line for a password.

Standalone on purpose. ``python -m CoScientist auth-hash`` does the same thing,
but importing the package builds the whole agent system first, so it needs a
complete LLM configuration to run. This script needs nothing but a Python 3
interpreter, which is what an operator has while the .env file is still being
written.

    python3 deploy/make_password_hash.py

The format is pinned by tests/unit/test_web_auth.py, which checks that what
this script prints is what the server accepts.
"""
import base64
import getpass
import hashlib
import secrets
import sys

SCHEME = "pbkdf2_sha256"
ROUNDS = 200_000


def build(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ROUNDS)
    enc = lambda raw: base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return ":".join((SCHEME, str(ROUNDS), enc(salt), enc(digest)))


def main() -> int:
    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Repeat: "):
        print("The two entries differ. Nothing written.", file=sys.stderr)
        return 1
    if len(password) < 20:
        print(
            "Warning: shorter than 20 characters. One password guards the whole "
            "deployment and the login throttle does not stop guessing. "
            "See deploy/README.md.",
            file=sys.stderr,
        )
    print(f"AUTH__PASSWORD_HASH={build(password)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
