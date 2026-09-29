"""Getting the sandbox's output into the report the aggregator writes.

The aggregator was never withholding these files. The collector knew two
sources — the artifact index and the workspace on this disk — and the sandbox
container is neither, so the only files of its that ever reached a report were
the ones its own agent chose to upload mid-task. Everything else went away with
the container.
"""
from pathlib import Path

import pytest

from CoScientist.tools import sandbox_sweep


class _Sandbox:
    """A sandbox holding one tree, answering the two calls the sweep makes."""

    def __init__(self, tree, sizes=None):
        self.tree = tree
        self.sizes = sizes or {}
        self.downloaded = []

    def resolve_sandbox_url(self, *a, **k):
        return "http://sbx"

    def resolve_session_key(self, session_id, tool_context=None):
        return session_id or "s1"

    def read_binding(self, session, tool_context=None):
        return "task-1"

    def list_sandbox_files(self, path, **kwargs):
        if path not in self.tree:
            return {"status": "error", "error": "gone"}
        return {"status": "ok", "path": path, "entries": [
            {"name": name, "path": f"{path}/{name}",
             "type": "dir" if f"{path}/{name}" in self.tree else "file",
             "size": self.sizes.get(f"{path}/{name}", 10)}
            for name in self.tree[path]
        ]}

    def download_sandbox_file(self, remote, local, **kwargs):
        self.downloaded.append(remote)
        Path(local).write_bytes(b"x" * 10)
        return {"status": "ok", "size_bytes": 10}


@pytest.fixture
def sandbox(monkeypatch):
    def install(tree, sizes=None):
        stub = _Sandbox(tree, sizes)
        monkeypatch.setattr(
            "CoScientist.tools.coder_tools.openhands_sandbox.resolve_sandbox_url",
            stub.resolve_sandbox_url, raising=False)
        monkeypatch.setattr(
            "CoScientist.tools.coder_tools.openhands_sandbox.resolve_session_key",
            stub.resolve_session_key, raising=False)
        monkeypatch.setattr(
            "CoScientist.tools.coder_tools.openhands_sandbox.read_binding",
            stub.read_binding, raising=False)
        monkeypatch.setattr(
            "CoScientist.tools.coder_tools.openhands_sandbox.list_sandbox_files",
            stub.list_sandbox_files, raising=False)
        monkeypatch.setattr(
            "CoScientist.tools.coder_tools.openhands_sandbox.download_sandbox_file",
            stub.download_sandbox_file, raising=False)
        return stub
    return install


def test_the_results_a_run_left_in_the_sandbox_reach_the_report(sandbox, tmp_path):
    """Plots and CSVs the sandbox agent never published are the common case."""
    stub = sandbox({
        "/workspace": ["results", "train.py", "loss.png"],
        "/workspace/results": ["metrics.csv", "final.json"],
    })

    staged = sandbox_sweep.sweep_sandbox_workspace(None, {}, "s1", tmp_path)

    assert staged == {"_sandbox/loss.png", "_sandbox/results/metrics.csv",
                      "_sandbox/results/final.json"}
    # Source is not a result: the report is not a code listing.
    assert "/workspace/train.py" not in stub.downloaded
    # And they land where the collector walks for this session.
    assert (tmp_path / "ws_s1" / "_sandbox" / "results" / "metrics.csv").exists()


def test_a_working_tree_is_not_hauled_across_wholesale(sandbox, tmp_path):
    """A workspace holds a venv, a cloned library and a pip cache."""
    stub = sandbox({
        "/workspace": [".venv", ".git", "node_modules", "__pycache__",
                       "conversations", "GOLEM", "plot.png"],
        "/workspace/.venv": ["lib.png"],
        "/workspace/.git": ["logo.png"],
        "/workspace/node_modules": ["icon.png"],
        "/workspace/__pycache__": ["x.pkl"],
        "/workspace/conversations": ["base_state.json"],
        # A cloned library: its bundled example plots are not this run's.
        "/workspace/GOLEM": [".git", "example.png"],
        "/workspace/GOLEM/.git": [],
    })

    staged = sandbox_sweep.sweep_sandbox_workspace(None, {}, "s1", tmp_path)

    assert staged == {"_sandbox/plot.png"}
    assert stub.downloaded == ["/workspace/plot.png"]


def test_project_furniture_is_not_mistaken_for_a_result(sandbox, tmp_path):
    """A package.json shares its extension with a metrics dump and is not one."""
    sandbox({"/workspace": ["package.json", "meta.json", "metrics.json"]})

    staged = sandbox_sweep.sweep_sandbox_workspace(None, {}, "s1", tmp_path)

    assert staged == {"_sandbox/metrics.json"}


def test_a_checkpoint_too_large_to_move_is_left_where_it_is(sandbox, tmp_path):
    stub = sandbox({"/workspace": ["model.ckpt", "loss.png"]},
                   sizes={"/workspace/model.ckpt": 10 ** 11})

    sandbox_sweep.sweep_sandbox_workspace(None, {}, "s1", tmp_path)

    assert stub.downloaded == ["/workspace/loss.png"]


def test_a_file_the_sandbox_agent_already_published_is_not_staged_twice(
        sandbox, tmp_path):
    """It is collected from its S3 reference; staging it would duplicate it."""
    stub = sandbox({"/workspace": ["loss.png", "acc.png"]})

    staged = sandbox_sweep.sweep_sandbox_workspace(
        None, {}, "s1", tmp_path, already_named={"loss.png"})

    assert staged == {"_sandbox/acc.png"}
    assert stub.downloaded == ["/workspace/acc.png"]


def test_a_session_that_never_started_a_sandbox_costs_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "CoScientist.tools.coder_tools.openhands_sandbox.resolve_sandbox_url",
        lambda *a, **k: "http://sbx", raising=False)
    monkeypatch.setattr(
        "CoScientist.tools.coder_tools.openhands_sandbox.read_binding",
        lambda *a, **k: None, raising=False)

    def refuse(*a, **k):  # pragma: no cover - the point is that it is not called
        raise AssertionError("no sandbox, no listing")

    monkeypatch.setattr(
        "CoScientist.tools.coder_tools.openhands_sandbox.list_sandbox_files",
        refuse, raising=False)

    assert sandbox_sweep.sweep_sandbox_workspace(None, {}, "s1", tmp_path) == set()


def test_an_unreachable_sandbox_does_not_take_the_report_down(monkeypatch, tmp_path):
    """The report is the deliverable; its attachments are not worth failing it."""
    monkeypatch.setattr(
        "CoScientist.tools.coder_tools.openhands_sandbox.resolve_sandbox_url",
        lambda *a, **k: "http://sbx", raising=False)
    monkeypatch.setattr(
        "CoScientist.tools.coder_tools.openhands_sandbox.read_binding",
        lambda *a, **k: "task-1", raising=False)
    monkeypatch.setattr(
        "CoScientist.tools.coder_tools.openhands_sandbox.list_sandbox_files",
        lambda *a, **k: {"status": "error", "error": "connection refused"},
        raising=False)

    assert sandbox_sweep.sweep_sandbox_workspace(None, {}, "s1", tmp_path) == set()


def test_the_aggregator_is_told_these_files_exist_and_will_not_again():
    """Nothing collects them a second time — the container is gone by then."""
    from CoScientist.agents.prompts import templates  # noqa: F401
    from CoScientist.assembly.registry import REGISTRY

    text = REGISTRY.prompts["result_aggregator"](None)
    assert "format_results" in text
    lowered = text.lower()
    assert "files" in lowered
    assert "sandbox" in lowered or "container" in lowered
