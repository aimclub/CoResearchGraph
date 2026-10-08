"""The vault consumption path: the client, promotion, and the workspace sync.

Part 1 made every server produce a durable reference. These tests cover the
other half — turning that reference back into a file, and moving the objects a
report needs out of the prefix the lifecycle rule reclaims.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from CoScientist.config.report import ReportConfig
from CoScientist.reporting import artifact_index, collect, finalize
from CoScientist.tools import vault_client


# --- the client ------------------------------------------------------------

def _envelope(payload: dict):
    """What an MCP call returns: a JSON string inside content[].text."""
    return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])


def _fake_transport(monkeypatch, payload: dict):
    """Replace the MCP transport, keeping the real envelope parsing."""
    from contextlib import asynccontextmanager

    import mcp
    import mcp.client.streamable_http as streamable

    class _Session:
        def __init__(self, *a): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def initialize(self): return None
        async def call_tool(self, name, args): return _envelope(payload)

    @asynccontextmanager
    async def _client(url, **kw):
        yield (None, None, None)

    monkeypatch.setattr(streamable, "streamablehttp_client", _client)
    monkeypatch.setattr(mcp, "ClientSession", _Session)


def test_the_reply_is_unwrapped_from_the_double_envelope():
    """The vault returns a JSON string, and MCP wraps that string again."""
    assert vault_client._payload(_envelope({"bucket": "b", "s3_key": "k"})) == {
        "bucket": "b", "s3_key": "k"
    }


@pytest.mark.parametrize("result", [
    SimpleNamespace(content=[]),
    SimpleNamespace(content=[SimpleNamespace(text="not json")]),
    SimpleNamespace(content=None),
])
def test_an_unreadable_reply_is_not_a_payload(result):
    assert vault_client._payload(result) is None


def test_a_vault_refusal_returns_none(monkeypatch):
    """The vault reports a refusal in the body and answers 200. A caller that
    read the body as a success would record a key that does not exist."""
    monkeypatch.setattr(vault_client, "vault_url", lambda: "http://vault/mcp")
    _fake_transport(monkeypatch, {"error": "only ephemeral objects can be promoted"})

    assert asyncio.run(vault_client.call_vault("promote_artifact", s3_key="x")) is None


def test_a_successful_call_returns_the_parsed_payload(monkeypatch):
    monkeypatch.setattr(vault_client, "vault_url", lambda: "http://vault/mcp")
    _fake_transport(monkeypatch, {"bucket": "agent-vault", "s3_key": "permanent/u/s/p.png"})

    payload = asyncio.run(vault_client.call_vault("promote_artifact", s3_key="x"))
    assert payload["s3_key"] == "permanent/u/s/p.png"


def test_no_configured_url_is_not_an_error(monkeypatch):
    """The vault is optional. A deployment without one still runs."""
    monkeypatch.setattr(vault_client, "vault_url", lambda: None)
    assert asyncio.run(vault_client.call_vault("get_download_link", s3_key="k")) is None


def test_an_unreachable_vault_returns_none_instead_of_raising(monkeypatch):
    monkeypatch.setattr(vault_client, "vault_url", lambda: "http://127.0.0.1:1/mcp")
    assert asyncio.run(vault_client.call_vault("get_download_link", s3_key="k")) is None


def test_the_sync_wrapper_refuses_to_run_on_the_event_loop():
    """asyncio.run inside a live loop deadlocks it. Say so instead."""
    async def main():
        with pytest.raises(RuntimeError, match="event loop"):
            vault_client.call_vault_sync("get_download_link", s3_key="k")

    asyncio.run(main())


# --- re-minting a dead link ------------------------------------------------

@pytest.fixture()
def index_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    return tmp_path


def _entry(url="http://expired/plot.png"):
    return {"bucket": "agent-vault", "s3_key": "ephemeral/u1/s1/figures/plot.png",
            "tool": "chem", "label": "plot.png", "url": url}


def test_a_dead_url_is_reminted_from_the_durable_reference(index_root, tmp_path, monkeypatch):
    """The cached presigned URL expired. The object did not."""
    fresh = "http://minio/plot.png?sig=new"
    downloaded = []

    def fake_download(url, dest):
        downloaded.append(url)
        if url != fresh:
            return False  # the expired link 403s
        dest.write_bytes(b"x")
        return True

    monkeypatch.setattr(collect, "_download", fake_download)
    artifact_index.record([_entry()], user_id="u1", session_id="s1")

    result = collect.collect_artifacts(
        session_id="s1", state={}, reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "ws", index_key=("u1", "s1"),
        resolve_url=lambda uri: fresh,
    )

    assert downloaded == ["http://expired/plot.png", fresh]
    assert len(result["figures"]) == 1


def test_an_entry_with_no_url_at_all_is_resolved(index_root, tmp_path, monkeypatch):
    """The workspace sync records a key and no URL: an upload link cannot fetch."""
    seen = []
    monkeypatch.setattr(collect, "_download",
                        lambda url, dest: seen.append(url) or dest.write_bytes(b"x") or True)
    entry = _entry(url=None)
    entry.pop("url")
    artifact_index.record([entry], user_id="u1", session_id="s1")

    result = collect.collect_artifacts(
        session_id="s1", state={}, reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "ws", index_key=("u1", "s1"),
        resolve_url=lambda uri: f"http://minio/{uri.rsplit('/', 1)[-1]}",
    )

    assert seen == ["http://minio/plot.png"]
    assert len(result["figures"]) == 1


def test_a_resolver_that_raises_never_breaks_the_report(index_root, tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "_download", lambda url, dest: False)
    artifact_index.record([_entry()], user_id="u1", session_id="s1")

    def boom(uri):
        raise RuntimeError("vault down")

    result = collect.collect_artifacts(
        session_id="s1", state={}, reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "ws", index_key=("u1", "s1"), resolve_url=boom,
    )
    assert result["figures"] == []


# --- the mapping that crosses the stage boundary ---------------------------

def test_collection_records_where_each_file_came_from(index_root, tmp_path, monkeypatch):
    """collect_artifacts downloads to a local path and drops the key. finalize
    runs later and needs it back, so the mapping goes on disk."""
    monkeypatch.setattr(collect, "_download",
                        lambda url, dest: dest.write_bytes(b"x") or True)
    artifact_index.record([_entry(url="http://minio/plot.png")],
                          user_id="u1", session_id="s1")

    result = collect.collect_artifacts(
        session_id="s1", state={}, reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "ws", index_key=("u1", "s1"),
    )

    sources = json.loads(
        (Path(result["report_dir"]) / collect.SOURCES_FILENAME).read_text()
    )
    assert list(sources.values()) == [
        {"bucket": "agent-vault", "s3_key": "ephemeral/u1/s1/figures/plot.png"}
    ]
    assert list(sources)[0].startswith("figures/")


def test_a_url_only_artifact_produces_no_source_entry(index_root, tmp_path, monkeypatch):
    """Some servers return no key. There is nothing to promote for those."""
    monkeypatch.setattr(collect, "_download",
                        lambda url, dest: dest.write_bytes(b"x") or True)
    result = collect.collect_artifacts(
        session_id="s1",
        state={"mcp_artifacts": [{"url": "http://x/plot.png", "tool": "t"}]},
        reports_root=tmp_path / "reports", workspace_root=tmp_path / "ws",
    )
    assert not (Path(result["report_dir"]) / collect.SOURCES_FILENAME).exists()


# --- promotion at finalize -------------------------------------------------

def _write_sources(report_dir: Path, mapping: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / collect.SOURCES_FILENAME).write_text(json.dumps(mapping))


def test_finalize_promotes_each_collected_object(tmp_path, monkeypatch):
    calls = []

    def fake_call(tool, **args):
        calls.append((tool, args))
        return {"s3_key": "permanent/" + args["s3_key"][len("ephemeral/"):]}

    monkeypatch.setattr(vault_client, "vault_url", lambda: "http://vault/mcp")
    monkeypatch.setattr(vault_client, "call_vault_sync", fake_call)

    cfg = ReportConfig(reports_root=str(tmp_path / "reports"), latex="skip")
    _write_sources(
        finalize.report_dir_for("s1", cfg.reports_root),
        {"figures/plot.png": {"bucket": "b", "s3_key": "ephemeral/u1/s1/figures/plot.png"}},
    )

    result = finalize.finalize_report("s1", "# report", cfg)

    assert calls == [("promote_artifact", {"s3_key": "ephemeral/u1/s1/figures/plot.png"})]
    assert result.manifest["promoted"] == {
        "figures/plot.png": "permanent/u1/s1/figures/plot.png"
    }


def test_a_file_that_was_never_in_s3_is_not_promoted(tmp_path, monkeypatch):
    """A local sandbox leaves files on this disk only. A worker may not write
    to permanent/ directly, so uploading them here is out of scope."""
    monkeypatch.setattr(vault_client, "vault_url", lambda: "http://vault/mcp")
    monkeypatch.setattr(vault_client, "call_vault_sync",
                        lambda *a, **k: pytest.fail("must not call the vault"))

    cfg = ReportConfig(reports_root=str(tmp_path / "reports"), latex="skip")
    _write_sources(finalize.report_dir_for("s1", cfg.reports_root),
                   {"figures/local.png": {"bucket": "b", "s3_key": "permanent/x/y.png"}})

    assert finalize.finalize_report("s1", "# r", cfg).manifest["promoted"] == {}


def test_a_vault_that_is_down_still_leaves_a_full_manifest(tmp_path, monkeypatch):
    """Losing durability must not lose the deliverable."""
    monkeypatch.setattr(vault_client, "vault_url", lambda: "http://vault/mcp")
    monkeypatch.setattr(vault_client, "call_vault_sync", lambda *a, **k: None)

    cfg = ReportConfig(reports_root=str(tmp_path / "reports"), latex="skip")
    _write_sources(finalize.report_dir_for("s1", cfg.reports_root),
                   {"figures/p.png": {"bucket": "b", "s3_key": "ephemeral/u/s/p.png"}})

    result = finalize.finalize_report("s1", "# report", cfg)
    assert result.report_dir is not None
    assert result.manifest["report"] == "report.md"
    assert result.manifest["promoted"] == {}


def test_finalize_without_a_vault_url_writes_the_usual_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(vault_client, "vault_url", lambda: None)
    cfg = ReportConfig(reports_root=str(tmp_path / "reports"), latex="skip")
    _write_sources(finalize.report_dir_for("s1", cfg.reports_root),
                   {"figures/p.png": {"bucket": "b", "s3_key": "ephemeral/u/s/p.png"}})

    manifest = finalize.finalize_report("s1", "# report", cfg).manifest
    assert manifest["promoted"] == {}
    assert manifest["session_id"] == "s1"


def test_a_synced_file_is_not_collected_from_disk_as_well(tmp_path, monkeypatch):
    """The code-exec server may run on this host, and both sides use
    code_exec.workspace_root. Walking the same file again would put the figure
    in the report twice, once from S3 and once from disk."""
    monkeypatch.setattr(collect, "_download",
                        lambda url, dest: dest.write_bytes(b"x") or True)
    ws = tmp_path / "ws" / "ws_s1"
    ws.mkdir(parents=True)
    (ws / "loss.png").write_bytes(b"x")

    result = collect.collect_artifacts(
        session_id="s1",
        state={"mcp_artifacts": [{"url": "http://minio/ws_loss.png", "tool": "workspace"}]},
        reports_root=tmp_path / "reports", workspace_root=tmp_path / "ws",
        synced_files={"loss.png"},
    )
    assert len(result["figures"]) == 1
    assert result["blocks_markdown"].count("### ") == 1


def test_a_file_the_sync_failed_to_upload_is_still_collected(tmp_path):
    """Naming the synced files, rather than switching the walk off, keeps a
    failed upload reachable from disk."""
    ws = tmp_path / "ws" / "ws_s1"
    ws.mkdir(parents=True)
    (ws / "sent.png").write_bytes(b"x")
    (ws / "failed.png").write_bytes(b"x")

    result = collect.collect_artifacts(
        session_id="s1", state={}, reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "ws", synced_files={"sent.png"},
    )
    assert [Path(f).name for f in result["figures"]] == ["failed.png"]


def test_the_disk_walk_still_runs_by_default(tmp_path):
    """Local mode uploads nothing, so the walk is the only source."""
    ws = tmp_path / "ws" / "ws_s1"
    ws.mkdir(parents=True)
    (ws / "loss.png").write_bytes(b"x")

    result = collect.collect_artifacts(
        session_id="s1", state={}, reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "ws",
    )
    assert len(result["figures"]) == 1


# --- what an upload link must not become -----------------------------------

def _capture(tool_name, result, tool_context):
    from CoScientist.tools.mcp_artifact_plugin import McpArtifactCapturePlugin
    plugin = McpArtifactCapturePlugin()
    asyncio.run(plugin.after_tool_callback(
        tool=SimpleNamespace(name=tool_name), tool_args={},
        tool_context=tool_context, result=result,
    ))


def test_an_upload_link_is_indexed_without_its_url(index_root, monkeypatch):
    """A PUT-signed URL cannot fetch the object, and the object may not exist
    yet — the agent was only handed somewhere to put one."""
    recorded = []
    monkeypatch.setattr("CoScientist.tools.mcp_artifact_plugin.record",
                        lambda entries, ctx=None, **kw: recorded.extend(entries))
    _capture("get_upload_link",
             {"bucket": "agent-vault", "s3_key": "ephemeral/u/s/workspace/plot.png",
              "upload_url": "http://minio/plot.png?X-Amz-Signature=put"},
             SimpleNamespace(state={}))

    assert len(recorded) == 1
    assert recorded[0]["url"] is None
    assert recorded[0]["s3_key"] == "ephemeral/u/s/workspace/plot.png"


def test_a_worker_file_that_is_not_a_figure_or_table_is_not_indexed(index_root, monkeypatch):
    """An agent parks whatever it likes in the vault. An unknown extension is
    collected as a table, so a checkpoint would be rendered as one."""
    recorded = []
    monkeypatch.setattr("CoScientist.tools.mcp_artifact_plugin.record",
                        lambda entries, ctx=None, **kw: recorded.extend(entries))
    _capture("get_upload_link",
             {"bucket": "agent-vault", "s3_key": "ephemeral/u/s/workspace/model.pkl",
              "upload_url": "http://minio/model.pkl?X-Amz-Signature=put"},
             SimpleNamespace(state={}))

    assert recorded == []


def test_a_normal_tool_result_still_keeps_its_url(index_root, monkeypatch):
    """Only get_upload_link is special. Every other server returns a URL that
    downloads, and the report uses it inside the same run."""
    recorded = []
    monkeypatch.setattr("CoScientist.tools.mcp_artifact_plugin.record",
                        lambda entries, ctx=None, **kw: recorded.extend(entries))
    _capture("visualize_molecule",
             {"bucket": "b", "s3_key": "ephemeral/u/s/chem/mol.png",
              "presigned_url": "http://minio/mol.png?sig=1"},
             SimpleNamespace(state={}))

    assert recorded[0]["url"] == "http://minio/mol.png?sig=1"


# --- a malformed side file must not cost the deliverable -------------------

@pytest.mark.parametrize("content", ['["not", "a", "mapping"]', '"a string"',
                                     '{"figures/p.png": "not a dict"}'])
def test_a_malformed_source_file_leaves_the_report_intact(tmp_path, monkeypatch, content):
    """An escape from promotion lands in finalize_report's except, which reports
    the whole deliverable as missing while report.md sits complete on disk."""
    monkeypatch.setattr(vault_client, "vault_url", lambda: "http://vault/mcp")
    cfg = ReportConfig(reports_root=str(tmp_path / "reports"), latex="skip")
    report_dir = finalize.report_dir_for("s1", cfg.reports_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / collect.SOURCES_FILENAME).write_text(content)

    result = finalize.finalize_report("s1", "# report", cfg)
    assert result.report_dir is not None
    assert result.manifest["promoted"] == {}
