"""Load alembic's codegen module on the host.

``codegen`` imports ``alembic.tools.paths``: inside the build container
``CoScientist/alembic`` is copied in as a top-level ``alembic`` package, so that
import only resolves there. Putting ``CoScientist/`` on ``sys.path`` to make it
resolve here would also shadow the stdlib, so the module is loaded standalone
with its one package dependency stubbed instead.
"""

import importlib.util
import sys
import types
from pathlib import Path

_CODEGEN = (
    Path(__file__).resolve().parents[2]
    / "CoScientist"
    / "alembic"
    / "tools"
    / "codegen.py"
)


def load_codegen():
    pkg = types.ModuleType("alembic")
    pkg.__path__ = []
    tools = types.ModuleType("alembic.tools")
    tools.__path__ = []
    paths = types.ModuleType("alembic.tools.paths")
    paths.RUN_FUNCTION_SCRIPT = Path("/nonexistent/run_function.py")
    paths.S3_TRANSFER_SCRIPT = Path("/nonexistent/s3_transfer.py")
    paths.MOUNT_DATA = Path("/mount/data")
    paths.output_dir = lambda: Path("/nonexistent/output")
    sys.modules.update(
        {"alembic": pkg, "alembic.tools": tools, "alembic.tools.paths": paths}
    )
    spec = importlib.util.spec_from_file_location("alembic_codegen_under_test", _CODEGEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod
