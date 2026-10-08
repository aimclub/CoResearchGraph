"""The durable-reference layer: extraction, the graph fields, and the artifact index.

Together these guard the invariant the whole S3 integration rests on. A file
reference must survive as ``s3://bucket/key``. A presigned URL must never become
the thing a snapshot remembers, because it expires while the object stays.
"""
from __future__ import annotations

import json

import pytest

from CoScientist.graph.models import Node, StatusUpdate
from CoScientist.graph.store import GraphStore
from CoScientist.reporting import artifact_index
from CoScientist.utils.s3_refs import (
    find_s3_artifacts,
    find_s3_uris,
    s3_uri,
    split_s3_uri,
)


# --- extraction ------------------------------------------------------------

def test_finds_reference_inside_an_mcp_result_envelope():
    """An MCP call returns its payload as a JSON string inside content[].text.
    Without parsing that string every reference a server returns is invisible."""
    result = {
        "content": [{"type": "text", "text": json.dumps({
            "bucket": "agent-vault",
            "s3_key": "ephemeral/u1/s1/figures/plot.png",
            "presigned_url": "http://minio/agent-vault/x?sig=1",
        })}]
    }
    assert find_s3_uris(result) == ["s3://agent-vault/ephemeral/u1/s1/figures/plot.png"]


def test_finds_reference_nested_under_metadata():
    payload = {"metadata": {"docking_html": {"bucket": "b", "s3_key": "k/f.html"}}}
    assert find_s3_uris(payload) == ["s3://b/k/f.html"]


@pytest.mark.parametrize("prose", [
    "saved to s3://my-bucket/data/set.zip, done",
    # A period at the end of a sentence is punctuation. A key that keeps it
    # resolves to nothing.
    "saved to s3://my-bucket/data/set.zip.",
])
def test_finds_a_bare_s3_uri_in_prose(prose):
    assert find_s3_uris(prose) == ["s3://my-bucket/data/set.zip"]


def test_a_presigned_url_alone_is_never_a_reference():
    """The URL expires. A node that stored only this would hold a dead link."""
    assert find_s3_uris({"presigned_url": "https://minio/x/plot.png?sig=1"}) == []


def test_a_key_without_a_bucket_is_skipped():
    """A key alone does not say which bucket holds it, so it cannot be resolved."""
    assert find_s3_uris({"s3_key": "ephemeral/u/s/x.png"}) == []


def test_references_are_deduplicated_in_order():
    payload = [
        {"bucket": "b", "s3_key": "second"},
        {"bucket": "b", "s3_key": "first"},
        {"bucket": "b", "s3_key": "second"},
    ]
    assert find_s3_uris(payload) == ["s3://b/second", "s3://b/first"]


@pytest.mark.parametrize("value", [None, 42, True, "", [], {}])
def test_extraction_survives_any_scalar(value):
    assert find_s3_uris(value) == []


def test_artifacts_carry_the_url_that_sat_beside_the_key():
    """The report downloads with this URL while it is fresh. It is a cache, and
    the bucket and key beside it are the reference."""
    result = {"bucket": "b", "s3_key": "k.png", "presigned_url": "http://x?sig=1"}
    assert find_s3_artifacts(result) == [
        {"bucket": "b", "s3_key": "k.png", "s3_uri": "s3://b/k.png", "url": "http://x?sig=1"}
    ]


def test_upload_link_is_recognized_as_the_url_field():
    result = {"bucket": "b", "s3_key": "k.png", "upload_url": "http://x?sig=1"}
    assert find_s3_artifacts(result)[0]["url"] == "http://x?sig=1"


def test_uri_round_trips():
    assert split_s3_uri(s3_uri("b", "/k/f.png")) == ("b", "k/f.png")
    assert split_s3_uri("https://example.com/x") is None
    assert split_s3_uri("s3://bucket-only") is None


# --- the graph node fields -------------------------------------------------

def test_an_old_snapshot_without_the_new_fields_still_loads():
    node = Node(id="n1", run_id="r", kind="tool_call", label="t")
    assert node.input_files == [] and node.output_files == []


def test_status_update_records_output_files_and_never_wipes_them():
    store = GraphStore()
    store.add_node(Node(id="n1", run_id="r", kind="tool_call", label="t",
                        input_files=["s3://b/in.csv"]))
    store.set_status("n1", StatusUpdate(run_id="r", status="success",
                                        output_files=["s3://b/out.png"]))
    # A later update that carries no files must leave the references alone.
    store.set_status("n1", StatusUpdate(run_id="r", verdict="ok"))

    node = store.full("r")["nodes"][0]
    assert node["input_files"] == ["s3://b/in.csv"]
    assert node["output_files"] == ["s3://b/out.png"]


# --- the durable artifact index --------------------------------------------

@pytest.fixture()
def index_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    return tmp_path


def _entry(key="ephemeral/u1/s1/figures/plot.png", url="http://x?sig=1"):
    return {"bucket": "agent-vault", "s3_key": key, "tool": "visualize_molecule",
            "label": "plot.png", "url": url}


def test_index_round_trips(index_root):
    artifact_index.record([_entry()], user_id="u1", session_id="s1")
    assert artifact_index.load("s1", "u1") == [_entry()]


def test_index_is_found_by_session_id_alone(index_root):
    """The report collector runs in a tool that knows only the session id."""
    artifact_index.record([_entry()], user_id="u1", session_id="s1")
    assert len(artifact_index.load("s1")) == 1
    assert artifact_index.load("other-session") == []


def test_recapturing_an_object_refreshes_the_url_without_a_duplicate(index_root):
    """Dedupe is on the durable reference. A second presigned URL for the same
    object is the same artifact, and it is the URL that still works: the stored
    one is older and expires first."""
    artifact_index.record([_entry(url="http://x?sig=1")], user_id="u1", session_id="s1")
    artifact_index.record([_entry(url="http://x?sig=2")], user_id="u1", session_id="s1")

    entries = artifact_index.load("s1", "u1")
    assert len(entries) == 1
    assert entries[0]["url"] == "http://x?sig=2"


def test_recording_never_raises_on_a_broken_index(index_root):
    artifact_index.record([_entry()], user_id="u1", session_id="s1")
    artifact_index.index_path(("u1", "s1")).write_text("{not json", encoding="utf-8")
    # A corrupt file must not sink the run, and it must not sink the report.
    artifact_index.record([_entry(key="k2.png")], user_id="u1", session_id="s1")
    assert artifact_index.load("s1", "u1") == [_entry(key="k2.png")]


def test_the_report_collector_prefers_the_index_and_falls_back_to_state(index_root, tmp_path, monkeypatch):
    from CoScientist.reporting import collect

    downloaded = []

    def fake_download(url, dest):
        downloaded.append(url)
        dest.write_bytes(b"x")
        return True

    monkeypatch.setattr(collect, "_download", fake_download)
    artifact_index.record([_entry(url="http://from-index/plot.png")],
                          user_id="u1", session_id="s1")

    result = collect.collect_artifacts(
        session_id="s1",
        state={"mcp_artifacts": [{"url": "http://from-state/other.png", "tool": "t"}]},
        reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "ws",
        index_key=("u1", "s1"),
    )

    # The index comes first, and session state still works for a run that
    # started before the index existed.
    assert downloaded == ["http://from-index/plot.png", "http://from-state/other.png"]
    assert len(result["figures"]) == 2


def test_an_expired_url_on_an_indexed_artifact_is_counted_as_unresolved(index_root, tmp_path, monkeypatch, caplog):
    """A presigned URL from an earlier run is present but dead. The key is still
    good, so the gap must be logged, not swallowed."""
    from CoScientist.reporting import collect

    monkeypatch.setattr(collect, "_download", lambda url, dest: False)
    artifact_index.record([_entry(url="http://expired/plot.png")],
                          user_id="u1", session_id="s1")

    with caplog.at_level("WARNING"):
        result = collect.collect_artifacts(
            session_id="s1", state={}, reports_root=tmp_path / "reports",
            workspace_root=tmp_path / "ws", index_key=("u1", "s1"),
        )

    assert result["figures"] == []
    assert "no usable URL" in caplog.text


def test_a_source_pdf_never_lands_in_the_report(index_root, tmp_path, monkeypatch):
    """The papers server uploads every PDF it finds and returns a link to each.
    Those are source material, and the report holds figures and tables."""
    from CoScientist.reporting import collect

    downloaded = []
    monkeypatch.setattr(collect, "_download",
                        lambda url, dest: downloaded.append(url) or dest.write_bytes(b"x") or True)

    result = collect.collect_artifacts(
        session_id="s1",
        state={"mcp_artifacts": [{"url": "http://minio/b/paper.pdf?sig=1", "tool": "papers"}]},
        reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "ws",
    )

    assert downloaded == []
    assert result["tables"] == []
