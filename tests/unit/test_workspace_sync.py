"""Pushing a remote sandbox workspace into S3.

With CODE_EXEC__URL set, the run's figures sit on the code-exec host and the
local disk walk in collect.py finds nothing. The code-exec API cannot serve a
file, so the sandbox uploads with a presigned link the vault mints.
"""
from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import pytest

from CoScientist.tools import workspace_sync


class _FakeToolset:
    """Stands in for CoderToolset. Records every command it was asked to run."""

    def __init__(self, listing: str, repos: str = "", fail: bool = False):
        self.listing, self.repos, self.fail = listing, repos, fail
        self.commands = []

    async def _run_sync(self, command, workspace_id, max_wait=30):
        self.commands.append(command)
        if "-name .git" in command:
            return {"status": "success", "stdout": self.repos}
        if command.startswith("find "):
            return {"status": "success", "stdout": self.listing}
        if self.fail:
            return {"status": "error", "stderr": "no such file"}
        return {"status": "success", "stdout": ""}


@pytest.fixture()
def wired(monkeypatch, tmp_path):
    """A remote code-exec host, a configured vault, and an index in tmp_path."""
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setattr(workspace_sync, "vault_url", lambda: "http://vault/mcp")

    settings = workspace_sync.get_settings()
    monkeypatch.setattr(settings.code_exec, "url", "http://code-exec:8131")
    monkeypatch.setattr(settings.web, "coder_workspace_id", None)

    recorded = []
    monkeypatch.setattr(workspace_sync.artifact_index, "record",
                        lambda entries, ctx=None, **kw: recorded.extend(entries))
    return recorded


_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,128}$")  # vault_server._FILENAME_RE
_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")          # vault_server._ID_RE


def _links(monkeypatch, calls):
    """A vault stand-in that validates its arguments the way the real one does.

    get_upload_link takes user_id and session_id and has no defaults for them,
    and it rejects a filename outside its charset. A fake that swallows any
    kwargs would hide a call the real server refuses.
    """
    async def fake(tool, **args):
        for name in ("user_id", "session_id"):
            assert name in args, f"the vault requires {name}"
            assert _ID_RE.match(args[name]), f"bad {name}: {args[name]!r}"
        assert _FILENAME_RE.match(args["filename"]), f"bad filename: {args['filename']!r}"
        calls.append(args)
        return {"bucket": "agent-vault",
                "s3_key": f"ephemeral/u1/s1/workspace/{args['filename']}",
                "upload_url": f"http://minio/{args['filename']}?sig=1"}

    monkeypatch.setattr(workspace_sync, "call_vault", fake)


def _run(toolset, monkeypatch, state=None):
    monkeypatch.setattr(
        "CoScientist.tools.coder_tools.coder_tools.CoderToolset",
        lambda *a, **k: toolset,
    )
    return asyncio.run(workspace_sync.sync_workspace_to_s3(
        SimpleNamespace(state={}), state if state is not None else {"coder_workspace_id": "ws_abc"}
    ))


def test_every_figure_and_table_is_uploaded_and_indexed(wired, monkeypatch):
    calls = []
    _links(monkeypatch, calls)
    toolset = _FakeToolset("./plots/loss.png\n./results.csv\n./notes.txt\n")

    assert _run(toolset, monkeypatch) == {"plots/loss.png", "results.csv"}
    assert [c["filename"] for c in calls] == ["plots_loss.png", "results.csv"]
    assert all(c["feature"] == "workspace" for c in calls)
    # notes.txt is neither a figure nor a table.
    assert len(wired) == 2
    assert wired[0]["s3_key"] == "ephemeral/u1/s1/workspace/plots_loss.png"


def test_an_index_entry_carries_no_url(wired, monkeypatch):
    """An upload link is a PUT capability. It cannot fetch the object back, and
    a download link minted now would expire before the report reads it."""
    _links(monkeypatch, [])
    _run(_FakeToolset("./plot.png\n"), monkeypatch)
    assert "url" not in wired[0]
    assert wired[0]["bucket"] and wired[0]["s3_key"]


def test_two_files_with_one_basename_get_two_keys(wired, monkeypatch):
    """A flat key would let the second upload overwrite the first."""
    calls = []
    _links(monkeypatch, calls)
    _run(_FakeToolset("./plots/loss.png\n./runs/2/loss.png\n"), monkeypatch)
    assert [c["filename"] for c in calls] == ["plots_loss.png", "runs_2_loss.png"]


def test_a_cloned_repo_subtree_is_skipped(wired, monkeypatch):
    """A coder step may clone a whole library. Its bundled example images are
    not run outputs, and collecting them buries the real figures."""
    calls = []
    _links(monkeypatch, calls)
    toolset = _FakeToolset(
        "./rdkit_repo/docs/logo.png\n./node_modules/x/icon.png\n./plot.png\n",
        repos="./rdkit_repo/.git\n",
    )
    _run(toolset, monkeypatch)
    assert [c["filename"] for c in calls] == ["plot.png"]


def test_the_upload_needs_no_curl(wired, monkeypatch):
    """The code-exec image is python:3.12-slim plus build-essential, vim and
    git. It has no curl, and it always has python3."""
    _links(monkeypatch, [])
    toolset = _FakeToolset("./plot.png\n")
    _run(toolset, monkeypatch)
    upload = [c for c in toolset.commands if not c.startswith("find ")]
    assert len(upload) == 1
    assert upload[0].startswith("python3 -c ")
    assert "curl" not in upload[0]


def test_a_failed_upload_is_left_out_of_the_index(wired, monkeypatch):
    _links(monkeypatch, [])
    assert _run(_FakeToolset("./plot.png\n", fail=True), monkeypatch) == set()
    assert wired == []


def test_local_mode_syncs_nothing(wired, monkeypatch):
    """The files are already on this disk and collect.py walks them. Uploading
    would put every figure in the report twice."""
    monkeypatch.setattr(workspace_sync.get_settings().code_exec, "url", None)
    monkeypatch.setattr(workspace_sync, "call_vault",
                        lambda *a, **k: pytest.fail("must not reach the vault"))
    assert _run(_FakeToolset("./plot.png\n"), monkeypatch) == set()


def test_no_configured_vault_syncs_nothing(wired, monkeypatch):
    monkeypatch.setattr(workspace_sync, "vault_url", lambda: None)
    assert _run(_FakeToolset("./plot.png\n"), monkeypatch) == set()


def test_a_session_with_no_coder_step_syncs_nothing(wired, monkeypatch):
    """The workspace id is pinned in session state, not derived from the session
    id. No id means no sandbox was ever created."""
    monkeypatch.setattr(workspace_sync, "call_vault",
                        lambda *a, **k: pytest.fail("must not reach the vault"))
    assert _run(_FakeToolset("./plot.png\n"), monkeypatch, state={}) == set()


def test_an_env_pinned_workspace_is_still_found(wired, monkeypatch):
    """The A2A shared workspace is pinned through CODER_WORKSPACE_ID, and the
    coder toolset never writes that id to state. Reading state alone misses it."""
    calls = []
    _links(monkeypatch, calls)
    monkeypatch.setenv("CODER_WORKSPACE_ID", "shared-run")

    assert _run(_FakeToolset("./plot.png\n"), monkeypatch, state={}) == {"plot.png"}
    assert calls[0]["filename"] == "plot.png"


def test_the_pin_wins_over_the_state_id(wired, monkeypatch):
    _links(monkeypatch, [])
    monkeypatch.setenv("CODER_WORKSPACE_ID", "shared-run")
    toolset = _FakeToolset("./plot.png\n")
    _run(toolset, monkeypatch, state={"coder_workspace_id": "ws_other"})
    assert workspace_sync._workspace_id({"coder_workspace_id": "ws_other"}) == "ws_shared-run"


def test_the_scope_the_vault_needs_is_passed_explicitly(wired, monkeypatch):
    """get_upload_link requires user_id and session_id. SessionScopePlugin fills
    those at the ADK tool boundary, and this call does not go through one."""
    calls = []
    _links(monkeypatch, calls)
    ctx = SimpleNamespace(_invocation_context=SimpleNamespace(
        session=SimpleNamespace(id="sess-1", user_id="alice@example.com", state={})))
    monkeypatch.setattr(
        "CoScientist.tools.coder_tools.coder_tools.CoderToolset",
        lambda *a, **k: _FakeToolset("./plot.png\n"))

    asyncio.run(workspace_sync.sync_workspace_to_s3(ctx, {"coder_workspace_id": "ws_a"}))
    # Sanitized to the vault id charset, the same way every other call is.
    assert calls[0]["user_id"] == "alice_example_com"
    assert calls[0]["session_id"] == "sess-1"


@pytest.mark.parametrize("name,expected", [
    ("roc curve (fold 1).png", "roc_curve_fold_1_.png"),
    ("ünïcode.png", "n_code.png"),
    ("a/b/c.csv", "a_b_c.csv"),
])
def test_a_filename_is_made_to_fit_the_vault_charset(name, expected):
    """The vault rejects anything outside ^[a-zA-Z0-9_.-]{1,128}$, and a sandbox
    writes names it rejects."""
    assert workspace_sync._flatten(name) == expected


def test_a_long_filename_is_truncated_but_keeps_its_extension():
    """The report tells a figure from a table by the extension."""
    flat = workspace_sync._flatten("x" * 300 + ".png")
    assert len(flat) <= 128 and flat.endswith(".png")


def test_a_failed_repository_probe_syncs_nothing(wired, monkeypatch):
    """Without that list nothing prunes a cloned library, and its bundled
    example images would fill the report."""
    _links(monkeypatch, [])

    class _Broken(_FakeToolset):
        async def _run_sync(self, command, workspace_id, max_wait=30):
            if "-name .git" in command:
                return {"status": "timeout", "stderr": "took too long"}
            return await super()._run_sync(command, workspace_id, max_wait)

    assert _run(_Broken("./plot.png\n"), monkeypatch) == set()
