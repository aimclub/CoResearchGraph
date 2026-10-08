"""Tool cards are read from the build's files. The coder may give a tool other
parameters than the plan does, and the card has to show the code's signature.
"""

import json

from CoScientist.alembic.web import artifacts

_REPO = "https://github.com/whitead/synspace"


def _workdir(tmp_path, code):
    base = tmp_path / "synspace"
    (base / "output" / "tools").mkdir(parents=True)
    (base / "output" / "tools" / "find_prop.py").write_text(code, encoding="utf-8")
    (base / "reports").mkdir()
    (base / "reports" / "plan.json").write_text(json.dumps({"tools": [
        {"name": "find_prop", "params": ["smi", "mols", "props"], "purpose": "look up"}]}))
    return tmp_path


def test_the_card_shows_the_generated_signature(tmp_path):
    wd = _workdir(tmp_path, "def find_prop(smi, steps=(1, 1), filter=True):\n    return {}\n")

    card = artifacts.build_tools(wd, _REPO)["tools"][0]

    assert card["sig"] == "smi, steps=(1, 1), filter=True"


def test_without_readable_code_the_card_falls_back_to_the_plan(tmp_path):
    wd = _workdir(tmp_path, "def find_prop(:\n")

    assert artifacts.build_tools(wd, _REPO)["tools"][0]["sig"] == "smi, mols, props"


def test_call_args_keep_only_what_the_function_accepts(tmp_path):
    wd = _workdir(tmp_path, "def find_prop(smi, steps=(1, 1)):\n    return {}\n")

    args = artifacts.call_args_for(wd, _REPO, "find_prop",
                                   {"smi": "CCO", "mols": "mols_from_chemical_space"})

    assert args == {"smi": "CCO"}


def test_a_function_taking_kwargs_gets_every_arg(tmp_path):
    wd = _workdir(tmp_path, "def find_prop(smi, **kw):\n    return {}\n")
    args = {"smi": "CCO", "mols": "m"}

    assert artifacts.call_args_for(wd, _REPO, "find_prop", args) == args


def test_declared_params_are_unknown_without_readable_code(tmp_path):
    wd = _workdir(tmp_path, "def find_prop(smi, *, steps=1):\n    return {}\n")

    assert artifacts.declared_params(wd, _REPO, "find_prop") == {"smi", "steps"}
    assert artifacts.declared_params(wd, _REPO, "missing") is None
