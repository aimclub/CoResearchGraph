"""Sandbox uploads as durable report artifacts.

A sandbox run reports its uploaded files in ``s3_uploads`` with a ``key`` but
no bucket, and a presigned URL that dies in about an hour. These tests pin the
two halves of the fix: the client adds the durable bucket/s3_key pair when it
reads the uploads, and the report collector puts a non-media artifact — a
checkpoint, an archive, a produced PDF — into the Files section with a
download link.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from CoScientist.reporting import artifact_index, collect
from CoScientist.tools.coder_tools import openhands_sandbox as client
from CoScientist.utils.s3_refs import find_s3_artifacts


def _upload(key="ephemeral/u1/s1/results/model.pt"):
    """One s3_uploads entry, exactly as the sandbox server reports it."""
    return {
        "filename": "model.pt",
        "url": f"http://minio:9000/agent-vault/{key}?X-Amz-Signature=abc",
        "size": 4096,
        "key": key,
    }


# --- normalization at the client boundary ------------------------------------

def test_an_upload_is_normalized_to_a_durable_reference():
    entry, = client._normalize_uploads([_upload()])

    assert entry["s3_key"] == "ephemeral/u1/s1/results/model.pt"
    # The bucket comes from the presigned URL itself: <endpoint>/<bucket>/<key>.
    assert entry["bucket"] == "agent-vault"
    # The server contract keeps its field.
    assert entry["key"] == "ephemeral/u1/s1/results/model.pt"
    assert entry["filename"] == "model.pt"
    assert entry["size"] == 4096


def test_the_bucket_falls_back_to_the_deployment_config(monkeypatch):
    """A virtual-hosted-style URL names no bucket in its path."""
    monkeypatch.setenv("SANDBOX_S3_BUCKET", "cfg-bucket")
    upload = _upload()
    upload["url"] = "https://agent-vault.minio.example/ephemeral/u1/s1/results/model.pt?sig=1"

    entry, = client._normalize_uploads([upload])

    assert entry["bucket"] == "cfg-bucket"
    assert entry["s3_key"] == upload["key"]


def test_an_entry_that_names_no_object_passes_through():
    assert client._normalize_uploads([{"filename": "x"}]) == [{"filename": "x"}]
    assert client._normalize_uploads(None) == []
    assert client._normalize_uploads("not a list") == []


def test_a_normalized_upload_is_what_the_capture_plugin_looks_for():
    """find_s3_artifacts requires bucket AND s3_key. The raw server entry fails
    that check and its URL dies with the run. The normalized entry is the
    durable record the artifact index stores."""
    raw = [_upload()]
    assert find_s3_artifacts({"s3_uploads": raw}) == []

    record, = find_s3_artifacts({"s3_uploads": client._normalize_uploads(raw)})
    assert record["bucket"] == "agent-vault"
    assert record["s3_key"] == "ephemeral/u1/s1/results/model.pt"
    assert record["url"].startswith("http://minio:9000/")


def test_the_poll_result_carries_the_normalized_uploads():
    state = client._PollState(timeout=None, verbose=False)
    result = state.on_poll({
        "status": "completed", "summary": "done", "s3_uploads": [_upload()],
    })

    entry, = result["s3_uploads"]
    assert entry["bucket"] == "agent-vault"
    assert entry["s3_key"] == "ephemeral/u1/s1/results/model.pt"


# --- the report side ----------------------------------------------------------

@pytest.fixture()
def index_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture()
def fake_download(monkeypatch):
    seen = []

    def download(url, dest):
        seen.append(url)
        dest.write_bytes(b"x")
        return True

    monkeypatch.setattr(collect, "_download", download)
    return seen


def _collect(tmp_path, **kwargs):
    return collect.collect_artifacts(
        session_id="s1", state={}, reports_root=tmp_path / "reports",
        workspace_root=tmp_path / "ws", index_key=("u1", "s1"), **kwargs,
    )


def _index_entry(key, url=None, tool="run_sandbox_task"):
    return {
        "bucket": "agent-vault", "s3_key": key, "tool": tool,
        "label": key.rsplit("/", 1)[-1],
        "url": url or f"http://minio/agent-vault/{key}?sig=1",
    }


def test_a_checkpoint_lands_in_the_files_section(index_root, tmp_path, fake_download):
    artifact_index.record(
        [_index_entry("ephemeral/u1/s1/results/model.pt")], user_id="u1", session_id="s1",
    )

    result = _collect(tmp_path)

    assert len(result["files"]) == 1
    assert "## Files" in result["blocks_markdown"]
    # The object is already in S3, so the link points at the artifact route
    # for the original object instead of a local path that is dead in the UI.
    assert (
        "[download](/api/artifact/agent-vault/ephemeral/u1/s1/results/model.pt)"
        in result["blocks_markdown"]
    )
    # The durable reference crosses into finalize's promotion input.
    sources = json.loads(
        (Path(result["report_dir"]) / collect.SOURCES_FILENAME).read_text()
    )
    assert sources == {
        "files/run_sandbox_task_model.pt": {
            "bucket": "agent-vault", "s3_key": "ephemeral/u1/s1/results/model.pt",
        }
    }


def test_the_figures_and_tables_sections_are_unchanged(index_root, tmp_path, fake_download):
    artifact_index.record([
        _index_entry("ephemeral/u1/s1/plot.png", tool="chem"),
        _index_entry("ephemeral/u1/s1/data.csv", tool="chem"),
        _index_entry("ephemeral/u1/s1/results.zip"),
    ], user_id="u1", session_id="s1")

    result = _collect(tmp_path)
    md = result["blocks_markdown"]

    assert "## Figures" in md
    assert "## Data tables" in md
    assert "## Files" in md
    assert len(result["figures"]) == 1
    assert len(result["tables"]) == 1
    assert len(result["files"]) == 1


def test_a_produced_pdf_is_a_file_not_source_material(index_root, tmp_path, fake_download):
    """A PDF the run made is a deliverable. Only bulk search results stay out."""
    artifact_index.record(
        [_index_entry("ephemeral/u1/s1/report.pdf")], user_id="u1", session_id="s1",
    )

    result = _collect(tmp_path)

    assert len(result["files"]) == 1
    assert "## Files" in result["blocks_markdown"]


def test_a_dead_file_url_is_reminted_from_the_key(index_root, tmp_path, monkeypatch):
    """A file entry flows through the same resolve_url path as a figure."""
    seen = []

    def download(url, dest):
        seen.append(url)
        if "sig=old" in url:
            return False  # the expired link 403s
        dest.write_bytes(b"x")
        return True

    monkeypatch.setattr(collect, "_download", download)
    key = "ephemeral/u1/s1/results/model.pt"
    artifact_index.record(
        [_index_entry(key, url=f"http://minio/agent-vault/{key}?sig=old")],
        user_id="u1", session_id="s1",
    )

    result = _collect(tmp_path, resolve_url=lambda uri: f"http://minio/{uri.rsplit('/', 1)[-1]}?sig=new")

    assert seen == [
        f"http://minio/agent-vault/{key}?sig=old",
        "http://minio/model.pt?sig=new",
    ]
    assert len(result["files"]) == 1


def test_the_workspace_walk_collects_files_but_not_code(index_root, tmp_path, monkeypatch):
    """The disk walk is uncurated, so only known deliverable types qualify."""
    # Keep S3 off: the repo .env on a configured machine would really upload.
    monkeypatch.setattr(collect, "upload_and_ref", lambda *a, **k: None)
    ws = tmp_path / "ws" / "ws_s1"
    ws.mkdir(parents=True)
    (ws / "model.pt").write_bytes(b"x")
    (ws / "results.tar.gz").write_bytes(b"x")
    (ws / "train.py").write_bytes(b"x")
    (ws / "notes.txt").write_bytes(b"x")

    result = _collect(tmp_path)

    assert sorted(Path(f).name for f in result["files"]) == ["model.pt", "results.tar.gz"]
    assert "## Files" in result["blocks_markdown"]
    # S3 is off in this test, so the link falls back to the local POSIX path.
    assert "[download](files/model.pt)" in result["blocks_markdown"]
