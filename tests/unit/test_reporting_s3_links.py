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
    """S3 on, with the network cut: record uploads, return a fake reference.

    ``upload_and_ref``, not ``upload_and_presign``: collection stopped putting
    signatures into report prose, because a signature expires and the prose is
    read later. It uploads for durability and links through ``/api/artifact/``.
    """
    monkeypatch.setattr(get_settings().s3, "use_s3", True)
    calls: list[tuple[str, str]] = []

    def _fake_ref(local_path, prefix):
        from pathlib import Path

        calls.append((str(local_path), prefix))
        return "test-bucket", f"{prefix}/{Path(local_path).name}"

    def _fake_presign(local_path, prefix):
        from pathlib import Path

        calls.append((str(local_path), prefix))
        return f"https://s3.test/{prefix}/{Path(local_path).name}?sig=fake"

    # Both, because the two consumers want different things: report collection
    # wants a durable reference, while the coder-output rewriter still hands a
    # reader a link to open right now.
    monkeypatch.setattr(s3_upload, "upload_and_ref", _fake_ref)
    monkeypatch.setattr(collect, "upload_and_ref", _fake_ref)
    monkeypatch.setattr(s3_upload, "upload_and_presign", _fake_presign)
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


def test_collect_links_through_the_artifact_route_when_s3_on(tmp_path, fake_upload):
    """Never a signature in prose a person reads later.

    The report used to carry the presigned URL the upload handed back. It
    expires within the hour — and against a MinIO only the cluster can reach,
    which a browser cannot follow even before then. The object is durable; the
    link goes through the route that re-signs it per request.
    """
    _workspace_with_figure(tmp_path)
    out = collect.collect_artifacts(
        "sess1",
        reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "workspace",
    )

    md = out["blocks_markdown"]
    assert "![plot](/api/artifact/test-bucket/reports/sess1/figures/plot.png)" in md
    assert "[download](/api/artifact/test-bucket/reports/sess1/tables/data.csv)" in md
    assert "(figures/plot.png)" not in md
    assert "sig=" not in md and "X-Amz" not in md
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


# ── mirrored artifacts reach the report ──────────────────────────────────────


def _mirrored_session(tmp_path, monkeypatch):
    """A session whose store already holds a figure, a table and a file."""
    from CoScientist.reporting import session_files as sf

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path / "graph"))
    key = ("u1", "s1")
    sf.put_bytes(key, b"\x89PNG\r\n\x1a\n" + b"z" * 32, filename="fig_tsne.png",
                 source_tool="chemical_space_tsne", media_type="binary/octet-stream")
    sf.put_bytes(key, b"a,b\n1,2\n", filename="ld50.csv", source_tool="predict_ld50")
    sf.put_bytes(key, b'{"n": 1}', filename="clusters.json",
                 source_tool="record_result_outputs")
    return key


def test_every_collected_artifact_gets_a_block(tmp_path, monkeypatch, s3_off):
    """The bug the user reported, as an invariant.

    A live run returned ``figures_count: 3 / formatted_markdown: ""``. The
    mirrored pass appended to the path lists and never to the block lists, so
    the counts were right and the markdown was empty — and the aggregator, told
    there were three figures and shown none, invented three
    ``figures/<name>.png`` paths that no route serves.

    A path list must never gain an entry that the block lists do not.
    """
    key = _mirrored_session(tmp_path, monkeypatch)
    out = collect.collect_artifacts(
        "s1", reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "nowhere", index_key=key,
    )

    assert (len(out["figures"]), len(out["tables"]), len(out["files"])) == (1, 1, 1)
    assert out["block_counts"] == {"figures": 1, "tables": 1, "files": 1}
    assert out["blocks_markdown"].strip()
    for kind in ("figures", "tables", "files"):
        assert len(out[kind]) == out["block_counts"][kind]


def test_the_blocks_carry_session_free_references(tmp_path, monkeypatch, s3_off):
    """Not a path, not a signature — a reference that survives an export.

    The bundle packs the bytes and restores them under a NEW session id without
    rewriting anything, which only works while the text says `cos-artifact:<id>`
    instead of a URL with a session baked into it.
    """
    key = _mirrored_session(tmp_path, monkeypatch)
    md = collect.collect_artifacts(
        "s1", reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "nowhere", index_key=key,
    )["blocks_markdown"]

    assert "![" in md and "cos-artifact:" in md
    assert "](figures/" not in md and "](tables/" not in md
    assert "X-Amz" not in md and "sig=" not in md
    assert "/sessions/s1/" not in md, "a session id in the text breaks on import"
    # The name shown is the name of the file behind the link.
    assert "fig_tsne.png" in md and "clusters.json" in md


def test_a_generic_content_type_does_not_demote_a_figure(tmp_path, monkeypatch,
                                                         s3_off):
    """MinIO answers `binary/octet-stream` for objects uploaded without a type.

    Taken at face value that makes a PNG a generic download: the web route reads
    exactly this field to decide `inline`, so the picture arrives as an
    attachment instead of being drawn.
    """
    key = _mirrored_session(tmp_path, monkeypatch)
    out = collect.collect_artifacts(
        "s1", reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "nowhere", index_key=key,
    )
    assert out["block_counts"]["figures"] == 1
    assert "![" in out["blocks_markdown"].split("## Files")[0]
