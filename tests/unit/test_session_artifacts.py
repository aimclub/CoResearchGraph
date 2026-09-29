"""The session's own copy of what a run produced.

Every case here is a failure that happened on a real run. A tool hands back a
link to its own storage; one measured on a live session was valid for six
minutes (issued 12:29:56, expired 12:35:29). We stored the string, never fetched
the bytes, and by the time anyone opened the graph the figure was gone. Across
six sessions, 77 of 88 indexed artifacts had no durable reference at all.
"""
from __future__ import annotations

import json

import pytest

from CoScientist.reporting import artifact_index, session_files as sf
from CoScientist.reporting.collect import clean_url, _MEDIA_URL_RE
from CoScientist.utils.report_links import resolve_ref

PNG = b"\x89PNG\r\n\x1a\n" + b"z" * 64


@pytest.fixture()
def key(tmp_path, monkeypatch):
    """A session whose storage root is short.

    Deliberately not the pytest tmp dir alone: the artifact path runs
    ``<root>/sessions/<user>/<session>/artifacts/files/<id>``, and on Windows a
    deep root pushes that past the 260-character limit. That is how this failed
    the first time, silently.
    """
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")
    return ("u", "s")


# ── the URL that framed itself ──────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("![fig](http://h/b/f.png?X-Amz-Signature=abc)", "http://h/b/f.png?X-Amz-Signature=abc"),
    ("см. `http://h/b/f.png?sig=1`", "http://h/b/f.png?sig=1"),
    ("**http://h/b/f.png?s=2**", "http://h/b/f.png?s=2"),
    ("ссылка http://h/b/f.png?s=3, далее", "http://h/b/f.png?s=3"),
    ("итог http://h/b/f.png?s=4.", "http://h/b/f.png?s=4"),
])
def test_prose_punctuation_is_not_part_of_the_url(text, expected):
    """Markdown around a link used to be captured as part of it.

    Of 82 artifacts across six live sessions, 36 ended in ``)``, ``` ` ```,
    ``.``, ``,``, ``**`` or ``****`` — each one a link that resolves to nothing.

    Two stages on purpose. The pattern refuses the delimiters that can only be
    structure (``)`` ``]`` ``}`` ``` ` ``` ``*``), because those would otherwise
    run into the middle of a match; ``clean_url`` then trims the sentence
    punctuation that merely follows a URL, which is not worth forbidding inside
    a query string to catch.
    """
    assert clean_url(_MEDIA_URL_RE.findall(text)[0]) == expected


def test_a_truncated_url_is_not_a_url():
    """``_short()`` marks its cut with an ellipsis, and ``http://10.3…`` was
    captured and stored as though it were an address."""
    assert clean_url("http://10.3…") == ""
    assert _MEDIA_URL_RE.findall("figure=http://10.3…") == []


def test_cleaning_collapses_the_duplicates_it_created(key, monkeypatch):
    """One figure, captured beside four different punctuation marks, is one
    artifact — not five downloads of the same bytes."""
    url = "http://h/b/fig.png?sig=1"
    artifact_index.record(
        [{"bucket": None, "s3_key": None, "tool": "t", "label": "artifact", "url": u}
         for u in (url, url + ")", url + "`", url + ".", url + "**")],
        user_id=key[0], session_id=key[1],
    )
    assert [e["url"] for e in artifact_index.load(key[1], key[0])] == [url]


# ── the store ───────────────────────────────────────────────────────────────
def test_the_same_bytes_are_one_artifact(key):
    """The point of addressing by content: the same figure handed back under
    two signatures is one record and one file."""
    first = sf.put_bytes(key, PNG, filename="fig.png", source_tool="a")
    again = sf.put_bytes(key, PNG, filename="fig_other_name.png", source_tool="b")
    assert again["artifact_id"] == first["artifact_id"]
    assert sf.stored_count(key) == 1


def test_a_name_no_filesystem_wants_still_gets_an_id(key):
    """Tools name output after a caption — Cyrillic, spaces, parentheses."""
    record = sf.put_bytes(key, b"a,b\n1,2\n", filename="карта кластеров (v2).csv")
    assert sf.ARTIFACT_ID_RE.match(record["artifact_id"])
    # The real name is kept, because that is what a reader is shown.
    assert record["filename"] == "карта кластеров (v2).csv"


def test_what_was_not_stored_says_why(key):
    """A figure that silently disappears reads exactly like one nobody made."""
    record = sf.note(key, state=sf.STATE_SKIPPED, reason=sf.REASON_OVERSIZE,
                     filename="huge.pkl")
    assert record["state"] == sf.STATE_SKIPPED
    assert record["reason"] == sf.REASON_OVERSIZE
    assert record["artifact_id"] in sf.load_manifest(key[1], key[0])


def test_a_failure_never_overwrites_bytes_already_on_disk(key):
    stored = sf.put_bytes(key, PNG, filename="fig.png")
    sf.record_entries([{**stored, "state": sf.STATE_FAILED, "reason": "link_expired"}],
                      user_id=key[0], session_id=key[1])
    assert sf.load_manifest(key[1], key[0])[stored["artifact_id"]]["state"] == sf.STATE_STORED


@pytest.mark.parametrize("bad", ["../manifest.json", "..", "/etc/passwd", ".hidden",
                                 "a/b", "", "nosuch-00000000.png"])
def test_no_id_escapes_the_session_directory(key, bad):
    sf.put_bytes(key, PNG, filename="fig.png")
    assert sf.resolve_path(key, bad) is None


# ── what becomes a link ─────────────────────────────────────────────────────
def test_a_reference_carries_no_session(key):
    """The whole reason for the indirection: an imported bundle resolves under
    a session id that did not exist when the bytes were captured.

    Import writes the same bytes into the new session's store, and the very same
    reference string — unrewritten, straight out of the graph node — then
    resolves against it. Resolution is bytes-first: the reference alone is not
    enough, which is what keeps a stale one from becoming a 404.
    """
    record = sf.put_bytes(key, PNG, filename="fig.png")
    reference = sf.ref(record["artifact_id"])
    assert sf.parse_ref(reference) == record["artifact_id"]

    # Before import there is nothing to resolve against.
    imported = ("other_user", "other_session")
    assert resolve_ref(reference, imported) is None

    # What `_restore_artifacts` does: the same bytes, under the new scope.
    sf.put_bytes(imported, PNG, filename="fig.png")
    assert resolve_ref(reference, imported) == (
        f"/api/users/other_user/sessions/other_session/artifacts/{record['artifact_id']}"
    )


@pytest.mark.parametrize("value", [
    r"D:\projects26\code_a\workspace\clusters.json",
    "/workspace/figures/out.png",
    "data/smiles.csv",
    "Литературные данные о метаболитах",
    "",
])
def test_a_thing_that_is_not_a_link_is_not_offered_as_one(value):
    """An anchor that navigates nowhere reads as a broken page. Twelve of twelve
    attachments in a real session were exactly that."""
    assert resolve_ref(value, ("u", "s")) is None


def test_a_durable_s3_reference_resolves_without_a_session():
    assert resolve_ref("s3://buck/a/b.png") == "/api/artifact/buck/a/b.png"


def test_a_reference_to_a_file_we_do_not_have_is_not_a_link(key):
    """A reference is a string; holding the bytes is a fact.

    An observation node carried ``cos-artifact:ART-37453f…`` — an id from the
    experiment runtime's own namespace, never stored under that name. Formatting
    it into a URL gave the reader a 📦 that 404s on click, which looks like a
    broken page rather than a file nobody mirrored.
    """
    sf.put_bytes(key, PNG, filename="real.png")
    assert resolve_ref("cos-artifact:ART-37453f5d8f4b41c4a9ac1db5e8f0c14f", key) is None


def test_the_runtime_id_and_our_id_are_different_fields():
    """``artifact_id`` belongs to the experiment runtime (``ART-<uuid>``);
    ours is ``session_artifact_id``. Reading one as the other is how the dead
    reference above got written in the first place."""
    from CoScientist.experiments.runtime.graph_bridge import _artifact_location

    runtime_only = {"artifact_id": "ART-37453f5d8f4b41c4a9ac1db5e8f0c14f",
                    "workspace_path": r"D:\projects26\out\dataset_overview.json"}
    # Falls through to the path — never invents a store reference.
    assert _artifact_location(runtime_only).endswith("dataset_overview.json")
    assert not _artifact_location(runtime_only).startswith("cos-artifact:")

    mirrored = {**runtime_only, "session_artifact_id": "dataset_overview-cf93f46b.json"}
    assert _artifact_location(mirrored) == "cos-artifact:dataset_overview-cf93f46b.json"


def test_an_experiment_artifact_with_only_a_local_path_is_mirrored(key, tmp_path):
    """The case the user reported: `dataset_overview.json` under Вложения,
    clicking to nothing, because its canonical location was a path on the
    machine that ran the task."""
    from CoScientist.graph.session_scope import (
        GRAPH_SCOPE_SESSION_KEY,
        GRAPH_SCOPE_USER_KEY,
    )
    from CoScientist.experiments.runtime.artifacts import _mirror_to_session

    source = tmp_path / "dataset_overview.json"
    source.write_text('{"n_compounds": 225}', encoding="utf-8")
    state = {GRAPH_SCOPE_USER_KEY: key[0], GRAPH_SCOPE_SESSION_KEY: key[1]}

    artifact_id = _mirror_to_session(
        state, name="dataset_overview.json", workspace_path=str(source),
        bucket=None, s3_key=None, external_url=None, tool="dataset_overview",
    )
    assert artifact_id and sf.resolve_path(key, artifact_id) is not None
    assert resolve_ref(sf.ref(artifact_id), key) == (
        f"/api/users/{key[0]}/sessions/{key[1]}/artifacts/{artifact_id}"
    )


def test_mirroring_is_skipped_when_there_is_no_session_to_mirror_into(tmp_path):
    """A CLI run or a unit test has no ADK scope; that must be a no-op, not a
    crash inside artifact normalisation."""
    from CoScientist.experiments.runtime.artifacts import _mirror_to_session

    source = tmp_path / "x.json"
    source.write_text("{}", encoding="utf-8")
    assert _mirror_to_session({}, name="x.json", workspace_path=str(source),
                              bucket=None, s3_key=None, external_url=None,
                              tool="t") is None


def test_a_figure_is_not_claimed_as_the_dataset_that_was_never_produced():
    """Straight from a real session, and the explanation of what the user saw.

    A task failed — its own summary read "Артефакт dataset_overview.json не
    получен" — but a cluster-map PNG had been captured in the same attempt. The
    role-only fallback claimed that PNG as the expected `dataset_overview.json`,
    so the graph showed an attachment under a name that described a different
    file entirely.
    """
    from CoScientist.experiments.runtime.artifacts import match_expected_artifact

    class Expected:
        def __init__(self, name, role="data"):
            self.name, self.role = name, role

    expected = [Expected("dataset_overview.json")]
    assert match_expected_artifact("fig1_cluster_map_8a56f120.png", expected) is None

    # What the fallback IS for, and must keep doing: a planner names the
    # expected artifact before the tool runs, so the suffix is often a guess.
    assert match_expected_artifact("output", expected) is expected[0]
    assert match_expected_artifact("generated_molecules.csv", expected) is expected[0]
    # And the ordinary cases keep working.
    assert match_expected_artifact("dataset_overview.json", expected) is expected[0]
    tabular = [Expected("res.txt")]
    assert match_expected_artifact("res.csv", tabular) is tabular[0]


def test_a_paper_is_not_claimed_as_the_dataset_the_plan_asked_for():
    """The `.json` chip that downloaded a PDF, straight from a live session.

    EXP-1 declared `metabolite_smiles.json`. Its literature search pulled in
    `3d3304f….pdf` from Semantic Scholar, the loose fallback claimed that PDF as
    the expected artifact and renamed it — so the research graph offered a chip
    labelled `metabolite_smiles.json` whose href served a paper. Worse than
    cosmetic: the rename also let a search hit satisfy a required artifact and
    pass a success criterion the run had not met.

    The guard used to cover pictures only, which is why this class of bug
    survived one category over.
    """
    from CoScientist.experiments.runtime.artifacts import match_expected_artifact

    class Expected:
        def __init__(self, name, role="data"):
            self.name, self.role = name, role

    expected = [Expected("metabolite_smiles.json")]
    assert match_expected_artifact(
        "3d3304f36857139f913de6c4cd866b42a492.pdf", expected
    ) is None
    # The same argument for the other families a search or a build drags in.
    assert match_expected_artifact("paper.docx", expected) is None
    assert match_expected_artifact("checkpoint.zip", expected) is None
    assert match_expected_artifact("model.pt", expected) is None
    # Still loose WITHIN a family, which is the whole point of the branch.
    assert match_expected_artifact("clusters.csv", expected) is expected[0]
    assert match_expected_artifact("results.parquet", expected) is expected[0]


def test_the_stored_name_wins_over_the_name_the_plan_asked_for(key, tmp_path,
                                                               monkeypatch):
    """A content hit hands back an earlier record — under an earlier name.

    ``put_bytes`` is content-addressed and first-wins by design. So when the
    same bytes were already mirrored under another name, the caller gets an id
    that ends in THAT name's extension. If it keeps calling the artifact what
    the plan asked for, the label and the file disagree from then on.
    """
    from CoScientist.graph.session_scope import (
        GRAPH_SCOPE_SESSION_KEY,
        GRAPH_SCOPE_USER_KEY,
    )
    from CoScientist.experiments.runtime.artifacts import _mirror_record_to_session

    state = {GRAPH_SCOPE_USER_KEY: key[0], GRAPH_SCOPE_SESSION_KEY: key[1]}
    first = sf.put_bytes(key, b"%PDF-1.4 paper", filename="paper.pdf",
                         source_tool="tavily_search")
    assert first["artifact_id"].endswith(".pdf")

    same_bytes = tmp_path / "metabolite_smiles.json"
    same_bytes.write_bytes(b"%PDF-1.4 paper")
    record = _mirror_record_to_session(
        state, name="metabolite_smiles.json", workspace_path=str(same_bytes),
        bucket=None, s3_key=None, external_url=None, tool="record_result_outputs",
    )
    assert record is not None
    assert record["artifact_id"] == first["artifact_id"]
    # The caller can SEE the disagreement, which is the whole point.
    assert record["filename"] == "paper.pdf"
