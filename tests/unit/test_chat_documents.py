"""A long result becomes a document; the chat keeps a summary and a button.

Every case here is a way this can go quietly wrong. The first one is the worst:
a Markdown file written without an explicit media type is served as an
octet-stream attachment, so the panel that was supposed to render it offers a
download instead — and nothing in the write path complains.
"""
from __future__ import annotations

import mimetypes

import pytest

from CoScientist.reporting import session_files as sf
from CoScientist.reporting.documents import (
    KIND_PREFIX,
    MARKDOWN_MEDIA_TYPE,
    SUMMARY_LIMIT,
    TITLE_LIMIT,
    ChatDocument,
    publish_document,
    summarise_markdown,
)

# Long enough to be a document. A real plan runs to tens of kilobytes; anything
# that fits in a chat message is left in the chat message (MIN_DOCUMENT_CHARS).
PLAN = """# План эксперимента · ревизия 2

Автоматизировать построение токсикологического профиля метаболитов Heracleum
sosnowskyi: собрать перечень с SMILES, кластеризовать, предсказать LD50 для
мыши по всем путям введения, оценить домен применимости моделей и стоимость
синтеза трёх перспективных соединений.

## Гипотезы

- **H1** Фуранокумариновый кластер окажется наиболее токсичным: медиана
  предсказанной LD50 минимальна среди всех кластеров и ниже 100 мг/кг.
- **H2** В наиболее токсичном кластере доминирующим риском будет
  кардиотоксичность, а не канцерогенность.

## Задачи

- **EXP-1** Сбор литературного перечня метаболитов и canonical SMILES.
- **EXP-2** Структурная кластеризация по ECFP4 и Tanimoto.
- **EXP-3** Предсказание LD50 по кластерам.
"""


@pytest.fixture()
def key(tmp_path, monkeypatch):
    """A session whose storage root is short.

    The artifact path runs ``<root>/sessions/<user>/<session>/artifacts/files/
    <id>``; a deep root pushes that past Windows' 260-character limit, which is
    how the artifact store failed the first time, silently.
    """
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")
    return ("u", "s")


# ── the media type nothing guesses ──────────────────────────────────────────
def test_python_still_cannot_guess_a_markdown_media_type():
    """The reason `publish_document` states it instead of letting the store ask.

    If this ever starts answering `text/markdown`, the explicit constant is
    merely redundant rather than load-bearing — but until then, dropping it
    turns every document into a download.
    """
    assert mimetypes.guess_type("a.md")[0] in (None, "text/markdown")


def test_a_document_carries_the_media_type_that_makes_it_render(key):
    doc = publish_document(key, markdown=PLAN, kind="plan")
    record = sf.load_manifest(key[1], key[0])[doc.artifact_id]
    assert record["media_type"] == MARKDOWN_MEDIA_TYPE
    # This is what the download route reads to decide inline vs attachment.
    assert record["media_type"].startswith("text/")


def test_the_document_is_marked_as_a_document(key):
    """`source_kind` is what the viewer's session list filters on."""
    doc = publish_document(key, markdown=PLAN, kind="plan", agent="ExperimentPlannerAgent")
    record = sf.load_manifest(key[1], key[0])[doc.artifact_id]
    assert record["source_kind"] == KIND_PREFIX + "plan"
    assert record["source_tool"] == "ExperimentPlannerAgent"
    assert record["label"] == doc.title


def test_the_bytes_that_come_back_are_the_document(key):
    doc = publish_document(key, markdown=PLAN, kind="plan")
    path = sf.resolve_path(key, doc.artifact_id)
    assert path.read_text(encoding="utf-8") == PLAN.strip()


# ── identity: one text, one file ────────────────────────────────────────────
def test_the_same_text_published_twice_is_one_document(key):
    """Content addressing, so a redelivered request does not litter the store."""
    first = publish_document(key, markdown=PLAN, kind="plan")
    second = publish_document(key, markdown=PLAN, kind="plan")
    assert first.artifact_id == second.artifact_id
    assert sf.stored_count(key) == 1


def test_two_revisions_are_two_documents(key):
    first = publish_document(key, markdown=PLAN, kind="plan")
    second = publish_document(key, markdown=PLAN.replace("ревизия 2", "ревизия 3"), kind="plan")
    assert first.artifact_id != second.artifact_id
    assert sf.stored_count(key) == 2


# ── failure is not an exception ─────────────────────────────────────────────
def test_no_session_means_no_document_and_no_error():
    assert publish_document(None, markdown=PLAN, kind="plan") is None


def test_empty_text_means_no_document(key):
    assert publish_document(key, markdown="", kind="plan") is None
    assert publish_document(key, markdown=None, kind="answer") is None


def test_a_store_that_refuses_does_not_raise(key, monkeypatch):
    """The message must still go out when the document cannot be written."""
    def boom(*args, **kwargs):
        raise OSError("disk is having a day")

    monkeypatch.setattr(sf, "put_bytes", boom)
    assert publish_document(key, markdown=PLAN, kind="plan") is None


def test_a_skipped_record_is_not_offered_as_a_document(key, monkeypatch):
    """A quota refusal returns a record, but not one anything can open."""
    monkeypatch.setattr(
        sf, "put_bytes",
        lambda *a, **k: {"artifact_id": "x-1.md", "state": sf.STATE_SKIPPED,
                         "reason": sf.REASON_SESSION_QUOTA},
    )
    assert publish_document(key, markdown=PLAN, kind="plan") is None


# ── the summary ─────────────────────────────────────────────────────────────
def test_the_title_is_the_first_heading_and_the_summary_the_prose_under_it():
    title, summary = summarise_markdown(PLAN)
    assert title == "План эксперимента · ревизия 2"
    assert summary.startswith("Автоматизировать построение")
    # The hypotheses section is a different section; it is not the summary.
    assert "H1" not in summary


def test_a_document_with_no_heading_falls_back_to_its_first_line():
    title, summary = summarise_markdown("Кластеризация выполнена.\n\nЧетыре кластера.")
    assert title == "Кластеризация выполнена."
    assert summary.startswith("Кластеризация выполнена.")


def test_markup_is_not_read_aloud():
    """A summary is a sentence, not a line of Markdown."""
    _, summary = summarise_markdown("# Итог\n\n- **Найдено** 4 кластера, см. [отчёт](http://x/y).")
    assert summary == "Найдено 4 кластера, см. отчёт."


def test_a_table_is_not_a_summary():
    title, summary = summarise_markdown("# Матрица\n\n| Задача | Гипотеза |\n|---|---|\n| EXP-1 | H1 |")
    assert title == "Матрица"
    assert summary == ""


def test_the_summary_stops_being_a_summary_past_its_limit():
    _, summary = summarise_markdown("# T\n\n" + "Очень длинное предложение. " * 60)
    assert len(summary) <= SUMMARY_LIMIT + 1


def test_a_caller_that_knows_better_keeps_its_own_words(key):
    """A plan headline and an agent's own two sentences beat anything derived."""
    doc = publish_document(
        key, markdown=PLAN, kind="plan",
        title="План эксперимента", summary="Ревизия 2: 7 задач, ≈195 мин.",
    )
    assert doc.title == "План эксперимента"
    assert doc.summary == "Ревизия 2: 7 задач, ≈195 мин."


def test_a_long_title_is_capped(key):
    doc = publish_document(key, markdown="# " + "Слово " * 80 + "\n\nтекст", kind="answer")
    assert len(doc.title) <= TITLE_LIMIT + 1


# ── what rides on the websocket ─────────────────────────────────────────────
def test_the_payload_carries_no_url(key):
    """The browser builds the URL from the session it is looking at.

    A baked-in URL would carry the session that captured the document, and an
    imported bundle resolves under a different one.
    """
    doc = publish_document(key, markdown=PLAN, kind="plan")
    payload = doc.as_payload()
    assert set(payload) == {"artifact_id", "title", "kind"}
    assert "http" not in payload["artifact_id"]
    assert "/" not in payload["artifact_id"]


def test_the_artifact_id_is_what_the_web_route_accepts(key):
    """`resolve_path` re-checks this, so a document that fails it is unopenable."""
    doc = publish_document(key, markdown=PLAN, kind="plan")
    assert sf.ARTIFACT_ID_RE.match(doc.artifact_id)
    assert sf.has_artifact(key, doc.artifact_id)
    assert doc.artifact_id.endswith(".md")


def test_every_kind_gets_a_readable_file_name(key):
    for kind in ("plan", "result", "work_order", "work_report", "frame", "review", "answer"):
        doc = publish_document(key, markdown=f"# {kind}\n\n" + f"текст {kind}. " * 40, kind=kind)
        record = sf.load_manifest(key[1], key[0])[doc.artifact_id]
        assert record["filename"].endswith(".md"), kind
        assert isinstance(doc, ChatDocument)


# ── end to end: the panel has to be able to open it ─────────────────────────
def test_the_route_serves_a_document_as_readable_text(tmp_path, monkeypatch):
    """A document written here must come back as text, not as a download.

    This is the whole chain: `publish_document` → the manifest → the session
    artifact route. It caught the media-type trap once already; without the
    explicit `text/markdown` the response is an octet-stream attachment.
    """
    from fastapi.testclient import TestClient

    from CoScientist.web.app import create_app

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")

    app = create_app()
    with TestClient(app) as client:
        user = client.post("/api/users", json={"nickname": "Gleb"}).json()["user"]
        session = client.post(
            f"/api/users/{user['id']}/sessions", json={"title": "Work"}
        ).json()["session"]
        key = (user["id"], session["id"])

        doc = publish_document(
            key, markdown=PLAN, kind="plan", agent="ExperimentPlannerAgent"
        )
        assert doc is not None

        response = client.get(
            f"/api/users/{user['id']}/sessions/{session['id']}/artifacts/{doc.artifact_id}"
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert response.headers["content-disposition"].startswith("inline")
        assert response.text == PLAN.strip()

        # And the listing has to say enough for the panel to title and filter it.
        listing = client.get(
            f"/api/users/{user['id']}/sessions/{session['id']}/artifacts"
        ).json()
        entry = next(a for a in listing["artifacts"] if a["artifact_id"] == doc.artifact_id)
        assert entry["source_kind"] == KIND_PREFIX + "plan"
        assert entry["label"] == doc.title
        assert entry["agent"] == "ExperimentPlannerAgent"
        assert entry["href"].endswith(doc.artifact_id)


def test_a_document_survives_an_export_and_an_import(tmp_path, monkeypatch):
    """Opened on another machine, under another session id, with nothing rewritten."""
    from fastapi.testclient import TestClient

    from CoScientist.web.app import create_app

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")

    app = create_app()
    with TestClient(app) as client:
        user = client.post("/api/users", json={"nickname": "Gleb"}).json()["user"]
        session = client.post(
            f"/api/users/{user['id']}/sessions", json={"title": "Work"}
        ).json()["session"]
        doc = publish_document((user["id"], session["id"]), markdown=PLAN, kind="plan")

        exported = client.post(
            f"/api/users/{user['id']}/sessions/{session['id']}/export"
        )
        assert exported.status_code == 200

        restored = client.post(
            f"/api/users/{user['id']}/import-session",
            files={"file": ("s.cossession.zip", exported.content, "application/zip")},
        )
        assert restored.status_code in (200, 201), restored.text
        landed = restored.json()
        # Import always lands under the ITMO_DEV user, whatever the URL said,
        # so this ends up under a different user AND a different session than
        # the one that captured the document. Nothing was rewritten to make
        # that work: the reference carries no scope, and the URL is built from
        # whichever scope is asking.
        new_user = landed["user"]["id"]
        new_session = landed["session"]["id"]
        assert (new_user, new_session) != (user["id"], session["id"])

        again = client.get(
            f"/api/users/{new_user}/sessions/{new_session}/artifacts/{doc.artifact_id}"
        )
        assert again.status_code == 200, again.text
        assert again.text == PLAN.strip()
        assert again.headers["content-type"].startswith("text/markdown")
