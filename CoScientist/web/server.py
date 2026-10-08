"""ASGI app for the CoScientist web interface.

``app`` is the uvicorn entry point (``CoScientist.web.server:app``). For running
it, prefer the unified CLI: ``python -m CoScientist web [--host --port --reload]``.
Executing this module directly is kept as a thin backward-compatible shim.
Usage:
    python -m CoScientist.web.server
    # or
    python CoScientist/web/server.py

Environment:
    COSCIENTIST_CONFIG    — system profile (e.g. "microfluidics"); default system.yaml
    COSCIENTIST_WEB_PORT  — port to serve on (default 8000), so a second
                            profile instance can run next to the default one
"""

import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from CoScientist.web.app import create_app

app = create_app()

if __name__ == "__main__":
    from CoScientist.cli import run_web

    run_web()
