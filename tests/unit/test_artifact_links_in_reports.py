"""An artifact a reader is told about should be one a reader can open.

Two renderers printed a file as a bare string: the Work Report's artifact list
and the experiment results' canonical addresses. Both now offer a link — under
one rule, and it is the rule `_href` and `resolve_ref` already follow: a link
when this session holds the bytes, the plain reference when it does not,
because a link that opens nothing reads as a broken page.

The experiment results are the delicate one. That section is headed "do not
invent URLs", and the backticked address is what stops a model doing exactly
that — it is the string the runtime accepts back. So the link goes BESIDE the
address, never instead of it.
"""
from __future__ import annotations

import pytest

from CoScientist.hitl.work_order import (
    Artifact,
    WorkOrder,
    WorkReport,
    render_work_report,
)
from CoScientist.reporting import session_files as sf


@pytest.fixture()
def key(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")
    return ("u", "s")


def _reported(*artifacts):
    order = WorkOrder(agent="CoderAgent", goal="cluster the metabolites")
    order.report = WorkReport(summary="done", artifacts=list(artifacts))
    return order


# ── the Work Report ─────────────────────────────────────────────────────────
def test_an_artifact_the_session_holds_becomes_a_link(key):
    record = sf.put_bytes(key, b"id,score\n1,0.9\n", filename="clusters.csv",
                          media_type="text/csv")
    order = _reported(Artifact(kind="dataset", ref=record["artifact_id"],
                               description="k-means output"))

    text = render_work_report(order, "en", scope=key)

    assert f"[clusters.csv](cos-artifact:{record['artifact_id']})" in text
    assert "— k-means output" in text, "the description survives"


def test_a_reference_we_cannot_open_is_printed_as_written(key):
    """The agent writes `ref` in its own words, and it is as often a workspace
    path as an artifact id. A link that 404s is worse than a name."""
    order = _reported(Artifact(kind="file", ref="workspace/notes.txt"))
    text = render_work_report(order, "en", scope=key)
    assert "- `file` workspace/notes.txt" in text
    assert "](" not in text


def test_without_a_session_the_output_is_exactly_what_it_was(key):
    """Every existing caller — the console HITL, the tests — passes no scope."""
    record = sf.put_bytes(key, b"x" * 20, filename="plot.png",
                          media_type="image/png")
    order = _reported(Artifact(kind="file", ref=record["artifact_id"]))

    assert render_work_report(order, "en") == render_work_report(order, "en",
                                                                 scope=None)
    assert f"- `file` {record['artifact_id']}" in render_work_report(order, "en")


# ── the experiment results ──────────────────────────────────────────────────
def _state(key, **over):
    from CoScientist.graph.session_scope import (
        GRAPH_SCOPE_SESSION_KEY,
        GRAPH_SCOPE_USER_KEY,
    )

    state = {
        GRAPH_SCOPE_USER_KEY: key[0],
        GRAPH_SCOPE_SESSION_KEY: key[1],
        "report_language": "en",
        "experiment_task_results": [{
            "task_id": "EXP-1", "status": "success", "summary": "ran",
            "route_used": "coder",
            "artifacts": [{"artifact_id": "A1", "name": "roc.png",
                           "external_url": over.get("url", "")}],
        }],
    }
    state.update({k: v for k, v in over.items() if k != "url"})
    return state


def test_the_canonical_address_always_stands(key):
    """The heading says "do not invent URLs" and that address is why."""
    from CoScientist.experiments.review import render_experiment_results

    text = render_experiment_results(_state(key, url="s3://bucket/runs/roc.png"))
    assert "`s3://bucket/runs/roc.png`" in text


def test_a_file_the_session_mirrored_gets_a_link_beside_the_address(key):
    from CoScientist.experiments.review import render_experiment_results

    url = "https://minio.internal/bucket/runs/roc.png?X-Amz-Signature=ab"
    record = sf.put_bytes(key, b"\x89PNG\r\n\x1a\n" + b"z" * 20,
                          filename="roc.png", media_type="image/png",
                          source_url=url)

    text = render_experiment_results(_state(key, url=url))

    assert f"`{url}`" in text, "the address the runtime accepts back"
    assert f"[roc.png](/api/users/u/sessions/s/artifacts/{record['artifact_id']})" \
        in text, "and something a person can press"


def test_a_location_we_do_not_hold_is_left_alone(key):
    from CoScientist.experiments.review import render_experiment_results

    text = render_experiment_results(
        _state(key, url=r"D:\workspace\run\roc.png"))
    assert "](" not in text, "a local path is a fact about a machine, not a link"


def test_a_state_with_no_session_renders_as_before(key):
    """A CLI run and most tests carry no scope; the renderer must not care."""
    from CoScientist.experiments.review import render_experiment_results

    state = _state(key, url="s3://bucket/runs/roc.png")
    from CoScientist.graph.session_scope import (
        GRAPH_SCOPE_SESSION_KEY,
        GRAPH_SCOPE_USER_KEY,
    )
    state.pop(GRAPH_SCOPE_USER_KEY)
    state.pop(GRAPH_SCOPE_SESSION_KEY)

    text = render_experiment_results(state)
    assert "`s3://bucket/runs/roc.png`" in text
    assert "](" not in text


def test_rendering_never_writes_the_scope_back_into_the_state(key):
    """`session_key` resolves AND writes; a renderer must only read."""
    from CoScientist.experiments.review import render_experiment_results

    state = _state(key, url="s3://bucket/runs/roc.png")
    before = dict(state)
    render_experiment_results(state)
    assert set(state) - set(before) <= {"experiment_artifacts_manifest"}
