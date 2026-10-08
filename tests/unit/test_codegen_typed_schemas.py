"""Generated tools must declare real argument types.

A repo function with no annotations produced an MCP schema with no ``type``,
and a model calling it would pass a list as a JSON string — every such call
crashed. Types are taken from, in order: the function's own annotation, the
type of its literal default, and the invocation recorded in the plan.
"""

from _codegen_loader import load_codegen

cg = load_codegen()


def _tool(tmp_path, name, src):
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    (tools / f"{name}.py").write_text(src, encoding="utf-8")


def _types(sig):
    return {name: annotation for name, annotation, _ in sig["params"]}


def test_an_annotation_is_kept(tmp_path):
    _tool(tmp_path, "t", "def t(path: str, n: int):\n    pass\n")

    assert _types(cg.tool_signature("t", tmp_path)) == {"path": "str", "n": "int"}


def test_a_typing_generic_survives(tmp_path):
    """List[str] used to be dropped to untyped, which is the bug all over again."""
    _tool(tmp_path, "t", "def t(names: List[str]):\n    pass\n")

    assert _types(cg.tool_signature("t", tmp_path)) == {"names": "List[str]"}


def test_a_literal_default_gives_the_type(tmp_path):
    _tool(tmp_path, "t", "def t(cells=['all'], strict=True, k=3):\n    pass\n")

    assert _types(cg.tool_signature("t", tmp_path)) == {
        "cells": "list",
        "strict": "bool",
        "k": "int",
    }


def test_the_recorded_invocation_is_the_last_resort(tmp_path):
    """Nothing in the source says what this takes — the plan does."""
    _tool(tmp_path, "t", "def t(cells):\n    pass\n")

    sig = cg.tool_signature("t", tmp_path, sample_args={"cells": ["a", "b"]})

    assert _types(sig) == {"cells": "list"}


def test_the_annotation_wins_over_the_recorded_invocation(tmp_path):
    _tool(tmp_path, "t", "def t(cells: str):\n    pass\n")

    sig = cg.tool_signature("t", tmp_path, sample_args={"cells": ["a"]})

    assert _types(sig) == {"cells": "str"}


def test_a_param_nothing_can_type_stays_untyped(tmp_path):
    _tool(tmp_path, "t", "def t(thing):\n    pass\n")

    assert _types(cg.tool_signature("t", tmp_path)) == {"thing": None}


def test_the_generated_server_can_resolve_the_types_it_declares(tmp_path):
    """Preserved capitalised annotations need the typing names imported."""
    sig = {"name": "t", "params": [("names", "List[str]", None)], "doc": "d"}

    server = cg.render_server("repo", [sig])

    assert "from typing import" in server
    assert "List" in server.split("from typing import", 1)[1].split("\n", 1)[0]
    assert "names: List[str]" in server
