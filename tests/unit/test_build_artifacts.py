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


def test_a_build_without_a_plan_reads_its_tools_out_of_the_server(tmp_path):
    """A server pulled from the MCP hub can be older than plan.json: the builds
    page showed it with no tools at all, though its server.py names every one."""
    base = tmp_path / "cytopus"
    (base / "output").mkdir(parents=True)
    (base / "reports").mkdir()
    (base / "output" / "server.py").write_text(
        'mcp = FastMCP("cytopus")\n\n'
        '@mcp.tool()\ndef list_kb(kind: str = "celltypes") -> dict:\n'
        '    """Inspect the KnowledgeBase.\n\n    More prose.\n    """\n    return {}\n\n'
        '@mcp.tool(name="export")\nasync def export_kb(out_path: str) -> dict:\n    return {}\n\n'
        'def _helper(x):\n    return x\n', encoding="utf-8")
    # The oldest builds wrote the verdicts as markdown only.
    (base / "reports" / "validation.md").write_text(
        "## Tool Invocations\n- **list_kb** — PASSED\n- **export_kb** — FAILED\n", encoding="utf-8")

    out = artifacts.build_tools(tmp_path, "https://github.com/wallet-maker/cytopus.git")

    assert out["title"] == "cytopus · MCP server"
    assert [(t["name"], t["sig"], t["status"]) for t in out["tools"]] == [
        ("list_kb", 'kind: str=\'celltypes\'', "pass"),
        ("export_kb", "out_path: str", "fail")]
    assert out["tools"][0]["desc"] == "Inspect the KnowledgeBase."


def test_an_example_the_function_would_refuse_is_filtered(tmp_path):
    """The plan is written before the code, and the coder renames parameters.
    A server pulled from the hub has no recorded runs, so the Call form falls
    back to the plan: synspace's find_prop then arrived with mols/props, which
    its function does not take, and every call failed."""
    wd = _workdir(tmp_path, "def find_prop(smi=None, mol=None):\n    return {}\n")
    plan = json.loads((tmp_path / "synspace" / "reports" / "plan.json").read_text())
    plan["tools"][0]["sample_args"] = {"smi": "CCO", "mols": [], "props": []}
    (tmp_path / "synspace" / "reports" / "plan.json").write_text(json.dumps(plan))

    [example] = artifacts.build_examples(wd, _REPO)["examples"]

    assert example["args"] == {"smi": "CCO"}


def test_a_build_without_a_log_shows_the_invocations_from_its_report(tmp_path):
    """The build log stays on the host; the validator's calls travel in the image."""
    wd = _workdir(tmp_path, "def find_prop(smi=None):\n    return {}\n")
    (tmp_path / "synspace" / "reports" / "validation.json").write_text(json.dumps({"tools": [
        {"name": "find_prop", "status": "passed", "invocations": [
            {"args": {"smi": "CCO"}, "ok": False, "error": "TypeError: boom"},
            {"args": {"smi": "CCO"}, "ok": True, "error": None}]}]}))

    card = artifacts.build_tools(wd, _REPO)["tools"][0]

    assert card["runs"] == [{"input": {"smi": "CCO"}, "passed": False, "error": "TypeError: boom"},
                            {"input": {"smi": "CCO"}, "passed": True, "error": None}]


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
