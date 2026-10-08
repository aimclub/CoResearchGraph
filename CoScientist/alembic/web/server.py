"""Run the alembic pipeline dashboard locally.

Usage:
    python -m alembic.web.server
    # or
    python CoScientist/alembic/web/server.py

The pipeline writes its workdir under ``.alembic/`` relative to the current
working directory (same as the CLI), so launch this from the same place you
would run ``python CoScientist/alembic/main.py``.
"""
import sys
from pathlib import Path

# The inner ``CoScientist/`` package contains a ``logging/`` sub-package that
# would shadow the stdlib once that dir is on sys.path[0]. Import stdlib
# ``logging`` FIRST so it is cached — otherwise uvicorn/asyncio's own
# ``import logging`` (triggered after the path insert below) resolves to the
# shadow and the server dies at boot. ``logging`` is the *only* stdlib-name
# collision in that package, so this one pre-import is sufficient.
import logging  # noqa: F401  (cache stdlib logging before the path insert)

# CoScientist/alembic/web/server.py -> CoScientist  (so `import alembic.*` works)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import uvicorn

from alembic.web.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "alembic.web.server:app",
        host="127.0.0.1",
        port=8100,
        reload=False,
        log_level="info",
        # Above the 5 s polling of the builds page; see run_web in CoScientist/cli.py.
        timeout_keep_alive=30,
    )
