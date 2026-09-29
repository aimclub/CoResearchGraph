"""Why a file is missing, recorded accurately.

The reason code is not internal bookkeeping: `collect_artifacts` prints it
verbatim into the report's "Не собрано" section, so whatever is stored here is
what a reader is told about why something is absent.

Every 4xx used to be filed as `link_expired`. In a live session NCBI answered
403 to a programmatic GET of two article PDFs, and the report would have said
their links had expired — when nothing had expired and nothing would change on
a retry. A wrong reason is worse than a vague one: it sends whoever reads it
looking for a fresh link that does not exist.
"""
from __future__ import annotations

import pytest

from CoScientist.reporting import mirror
from CoScientist.reporting import session_files as sf

SIGNED = ("https://minio.internal/bucket/fig.png"
          "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=8b55bd44&X-Amz-Expires=3600")
PLAIN = "https://pmc.ncbi.nlm.nih.gov/articles/pdf/plants-14-03253.pdf"


@pytest.fixture()
def key(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")
    return ("u", "s")


class _Answer:
    def __init__(self, status):
        self.status_code = status
        self.headers = {}

    def iter_content(self, _size):
        yield b""

    def close(self):
        pass


def _answers(monkeypatch, status):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Answer(status))


@pytest.mark.parametrize("url,signed", [(SIGNED, True), (PLAIN, False)])
def test_a_signature_is_what_can_expire(url, signed):
    assert mirror._is_signed(url) is signed


def test_a_refused_host_is_not_an_expired_link(key, monkeypatch):
    """The case that exposed this: NCBI refusing a bot, twice, in one session."""
    _answers(monkeypatch, 403)
    record = mirror.mirror_artifact(None, user_id=key[0], session_id=key[1],
                                    url=PLAIN, filename="plants.pdf",
                                    source_kind="paper")
    assert record["reason"] == sf.REASON_REFUSED
    assert record["state"] == sf.STATE_FAILED


def test_an_expired_signature_still_says_so(key, monkeypatch):
    """The original meaning of the code, kept: a presigned link that timed out."""
    _answers(monkeypatch, 403)
    record = mirror.mirror_artifact(None, user_id=key[0], session_id=key[1],
                                    url=SIGNED, filename="fig.png")
    assert record["reason"] == sf.REASON_LINK_EXPIRED


def test_nothing_at_that_address_says_so(key, monkeypatch):
    _answers(monkeypatch, 404)
    record = mirror.mirror_artifact(None, user_id=key[0], session_id=key[1],
                                    url=SIGNED, filename="gone.png")
    assert record["reason"] == sf.REASON_NOT_FOUND, (
        "a 404 is not an expiry even on a signed link")


def test_a_failed_paper_is_still_recorded_as_a_paper(key, monkeypatch):
    """Without `source_kind` on the failure paths, a paper that did not download
    was indistinguishable from a figure that did not — in the live manifest five
    PDFs sat there as `source_kind: None`, unanswerable and unretryable."""
    _answers(monkeypatch, 403)
    record = mirror.mirror_artifact(None, user_id=key[0], session_id=key[1],
                                    url=PLAIN, filename="plants.pdf",
                                    source_kind="paper")
    stored = sf.load_manifest(key[1], key[0])[record["artifact_id"]]
    assert stored["source_kind"] == "paper"
    assert stored["reason"] == sf.REASON_REFUSED
