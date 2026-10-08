"""Run one generated tool function and print its result as sentinel-marked JSON.

    python run_function.py <output_dir> <tool_name> <json_kwargs>

Imports ``tools.<tool_name>`` from <output_dir>, calls the function of the same
name with the JSON kwargs, and prints ``<<<ALEMBIC_RESULT>>>`` followed by one
JSON object — so banners / progress bars the repo code prints to stdout never
corrupt the parse. This single script is the execution path for BOTH the
validator's direct invocations and the generated FastMCP server (they must
never disagree).

Names are underscore-prefixed to avoid clashing with anything the imported
tool module pulls into scope.
"""
import importlib as _importlib
import json as _json
import sys as _sys
import traceback as _tb
from pathlib import Path as _P

_SENTINEL = "<<<ALEMBIC_RESULT>>>"


def _emit(_obj):
    print(_SENTINEL)
    print(_json.dumps(_obj, default=str))


_out_dir  = _P(_sys.argv[1]).resolve()
_toolname = _sys.argv[2]
_kwargs   = _json.loads(_sys.argv[3]) if len(_sys.argv) > 3 else {}

_sys.path.insert(0, str(_out_dir))

try:
    _mod = _importlib.import_module(f"tools.{_toolname}")
    _fn  = getattr(_mod, _toolname, None)
    if _fn is None:
        _emit({"ok": False,
               "error": f"tools/{_toolname}.py defines no function named {_toolname!r}"})
        raise SystemExit(0)
    _result = _fn(**_kwargs)
    _emit({"ok": True, "result": _result})
except SystemExit:
    raise
except Exception as _e:
    _emit({"ok": False, "error": f"{type(_e).__name__}: {_e}",
           "traceback": _tb.format_exc()})
