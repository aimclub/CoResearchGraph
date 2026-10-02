#!/usr/bin/env python3
"""Print the AUTH__PASSWORD_HASH line for a password.

Run it from the repository root:

    python3 deploy/make_password_hash.py

Standalone on purpose. A normal ``import CoScientist`` builds the whole agent
system, so it needs a complete LLM configuration — which an operator does not
have while the .env file is still being written. This script therefore loads one
leaf module by path. That module holds the only copy of the format, so what this
prints and what the server accepts cannot drift apart.
"""
import getpass
import importlib.util
import sys
from pathlib import Path

_LEAF = Path(__file__).resolve().parent.parent / "CoScientist" / "web" / "password_hash.py"


def _load_format():
    """Load CoScientist/web/password_hash.py without importing the package.

    ``spec_from_file_location`` does not run ``CoScientist/__init__.py``, which
    is the whole point — see the module docstring.
    """
    spec = importlib.util.spec_from_file_location("coscientist_password_hash", _LEAF)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load the password format from {_LEAF}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    fmt = _load_format()
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
    print(f"{fmt.ENV_KEY}={fmt.hash_password(password)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
