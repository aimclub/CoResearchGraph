"""S3 presigned links for report artifacts and agent chat output."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from CoScientist.config import get_settings
from CoScientist.graph.session_scope import (
    GRAPH_SCOPE_SESSION_KEY,
    GRAPH_SCOPE_USER_KEY,
)
from CoScientist.logging import agent_output
from CoScientist.reporting import collect, s3_upload


@pytest.fixture
def s3_off(monkeypatch):
    monkeypatch.setattr(get_settings().s3, "use_s3", False)
    s3_upload._reset_for_tests()
    yield
    s3_upload._reset_for_tests()


@pytest.fixture
def fake_upload(monkeypatch):
    """S3 on, with the network cut: record uploads, return a fake URL."""
    monkeypatch.setattr(get_settings().s3, "use_s3", True)
    calls: list[tuple[str, str]] = []

    def _fake(local_path, prefix):
        from pathlib import Path

        calls.append((str(local_path), prefix))
        return f"https://s3.test/{prefix}/{Path(local_path).name}?sig=fake"

    monkeypatch.setattr(s3_upload, "upload_and_presign", _fake)
    monkeypatch.setattr(collect, "upload_and_presign", _fake)
    yield calls
    s3_upload._reset_for_tests()


def _workspace_with_figure(tmp_path, session_id="sess1"):
    ws = tmp_path / "workspace" / f"ws_{session_id}"
    (ws / "results").mkdir(parents=True)
    (ws / "results" / "plot.png").write_bytes(b"\x89png")
    (ws / "results" / "data.csv").write_text("a,b\n1,2\n")
    return ws


# ── collect_artifacts ─────────────────────────────────────────────────────────


def test_collect_emits_relative_paths_when_s3_off(tmp_path, s3_off):
    _workspace_with_figure(tmp_path)
    out = collect.collect_artifacts(
        "sess1",
        reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "workspace",
    )

    assert "![plot](figures/plot.png)" in out["blocks_markdown"]
    assert "[download](tables/data.csv)" in out["blocks_markdown"]


def test_collect_emits_presigned_urls_when_s3_on(tmp_path, fake_upload):
    _workspace_with_figure(tmp_path)
    out = collect.collect_artifacts(
        "sess1",
        reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "workspace",
    )

    md = out["blocks_markdown"]
    assert "![plot](https://s3.test/reports/sess1/figures/plot.png?sig=fake)" in md
    assert "[download](https://s3.test/reports/sess1/tables/data.csv?sig=fake)" in md
    assert "(figures/plot.png)" not in md
    # The S3 key keeps the file name, so the URL ends with the extension.
    prefixes = {prefix for _, prefix in fake_upload}
    assert prefixes == {"reports/sess1/figures", "reports/sess1/tables"}


def test_upload_helper_returns_none_when_s3_off(tmp_path, s3_off):
    f = tmp_path / "plot.png"
    f.write_bytes(b"\x89png")

    assert s3_upload.upload_and_presign(f, "reports/sess1/figures") is None


def test_upload_helper_returns_none_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings().s3, "use_s3", True)

    assert s3_upload.upload_and_presign(tmp_path / "nope.png", "reports/x") is None


# ── AgentOutputPlugin path rewriting ──────────────────────────────────────────


@pytest.fixture
def coder_workspace(tmp_path, monkeypatch):
    """A local coder workspace rooted in tmp_path."""
    from CoScientist.tools.coder_tools import coder_tools

    monkeypatch.setattr(coder_tools._CFG, "url", None)
    monkeypatch.setattr(coder_tools._CFG, "workspace_root", str(tmp_path / "workspace"))
    return _workspace_with_figure(tmp_path)


@pytest.fixture
def sink(monkeypatch):
    received: list[tuple[tuple[str, str], dict]] = []

    async def collect_sink(key, payload):
        received.append((key, payload))

    monkeypatch.setattr(agent_output, "_sink", collect_sink)
    monkeypatch.setattr(
        agent_output, "reported_agents", lambda: frozenset({"CoderAgent"})
    )
    return received


def _context():
    return SimpleNamespace(
        agent_name="OrchestratorAgent",
        function_call_id="fc_1",
        state={
            GRAPH_SCOPE_USER_KEY: "user_1",
            GRAPH_SCOPE_SESSION_KEY: "sess1",
            "coder_workspace_id": "ws_sess1",
        },
        session=SimpleNamespace(user_id="user_1", id="sess1"),
    )


def _run(plugin, result, context):
    return asyncio.run(plugin.after_tool_callback(
        tool=SimpleNamespace(name="CoderAgent"),
        tool_args={},
        tool_context=context,
        result=result,
    ))


def test_existing_workspace_paths_become_presigned_urls(
    sink, coder_workspace, fake_upload
):
    _run(
        agent_output.AgentOutputPlugin(),
        "Done. See results/plot.png and results/data.csv.",
        _context(),
    )

    (_, payload), = sink
    assert (
        "https://s3.test/sessions/sess1/workspace/results/plot.png?sig=fake"
        in payload["content"]
    )
    assert "results/plot.png" not in payload["content"].replace(
        "https://s3.test/sessions/sess1/workspace/results/plot.png?sig=fake", ""
    )
    assert {p for _, p in fake_upload} == {
        "sessions/sess1/workspace/results",
    }


def test_output_is_unchanged_when_s3_off(sink, coder_workspace, s3_off):
    _run(
        agent_output.AgentOutputPlugin(),
        "Done. See results/plot.png.",
        _context(),
    )

    (_, payload), = sink
    assert payload["content"] == "Done. See results/plot.png."


def test_missing_files_and_urls_are_not_rewritten(sink, coder_workspace, fake_upload):
    _run(
        agent_output.AgentOutputPlugin(),
        "No results/gone.png yet; source: https://example.com/a/plot.png",
        _context(),
    )

    (_, payload), = sink
    assert "results/gone.png" in payload["content"]
    assert "https://example.com/a/plot.png" in payload["content"]
    assert fake_upload == []
