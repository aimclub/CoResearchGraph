"""The debugger replaces a planned sample input that the code rightly refuses.

iris load_cubes was planned with a .cdl text template where the loader reads a
binary .nc: the tests passed, the direct call crashed, and the debugger could
only say so. set_sample_args saves args the tool actually answers.

The modules import under the container's package layout, so they are loaded
with CoScientist/ on sys.path, the same way the build container sees them.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "CoScientist"))

from alembic import contract  # noqa: E402
from alembic.contract import EnvSpec, Plan, ToolSpec  # noqa: E402
from alembic.tools import invoke  # noqa: E402

_CDL = {"source": "lib/iris/tests/stock/file_headers/xios_2D_face_half_levels.cdl"}
_NC = {"source": "/work/.alembic/iris/output/samples/mesh.nc"}


@pytest.fixture
def plan(monkeypatch):
    state = {"plan": Plan("https://github.com/SciTools/iris", EnvSpec(),
                          [ToolSpec("load_cubes", "iris:load_cubes", sample_args=dict(_CDL))]),
             "saved": 0}
    monkeypatch.setattr(contract, "load_plan", lambda repo_url=None: state["plan"])

    def save(p):
        state["saved"] += 1

    monkeypatch.setattr(contract, "save_plan", save)
    return state


def _answers(monkeypatch, result):
    calls = []
    monkeypatch.setattr(invoke, "_invoke_tool_function_sync",
                        lambda name, args: calls.append((name, args)) or result)
    return calls


def test_args_the_tool_answers_are_saved_in_the_plan(plan, monkeypatch):
    calls = _answers(monkeypatch, {"ok": True, "result": {"cubes": 2}})

    out = invoke._set_sample_args_sync("load_cubes", '{"source": "/work/.alembic/iris/output/samples/mesh.nc"}')

    assert out["saved"] is True
    assert calls == [("load_cubes", _NC)]
    assert plan["plan"].tools[0].sample_args == _NC and plan["saved"] == 1


@pytest.mark.parametrize("result", [
    {"ok": False, "error": "OSError: NetCDF: Unknown file format"},
    # A missing input file is not a result: the call never ran.
    {"ok": True, "runtime_success": True, "reason": "not invoked: input file for source not available"},
])
def test_args_without_a_result_are_not_saved(plan, monkeypatch, result):
    _answers(monkeypatch, result)

    out = invoke._set_sample_args_sync("load_cubes", _NC)

    assert out["saved"] is False and "not saved" in out["error"]
    assert plan["plan"].tools[0].sample_args == _CDL and plan["saved"] == 0


@pytest.mark.parametrize("args, expected", [
    ('{"source": "simple_3d"}', {"source": "simple_3d"}),
    ("{}", {}), (None, {}), ({"n": 1}, {"n": 1}),
])
def test_args_given_as_a_json_string_are_read(args, expected):
    """The debugger passed '{}' as a string; reading it as a dict crashed the
    whole debugger round (iris: "'str' object has no attribute 'items'")."""
    assert invoke._json_args(args) == expected


def test_args_that_are_not_a_json_object_are_an_error_not_a_crash():
    assert invoke._invoke_tool_function_sync("load_cube", "not json")["ok"] is False
    assert "JSON object" in invoke._invoke_tool_function_sync("load_cube", "[1, 2]")["error"]


def test_a_tool_the_plan_does_not_have_is_refused(plan, monkeypatch):
    calls = _answers(monkeypatch, {"ok": True, "result": 1})

    out = invoke._set_sample_args_sync("load_cube", _NC)

    assert out["saved"] is False and calls == []
