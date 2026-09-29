"""Report links: raw S3 URLs become local links, and the endpoint mints fresh.

A report outlives the presigned URLs an agent pastes into it. These tests pin
the two halves of the fix: the rewrite that turns every S3 reference into a
relative ``/api/artifact/<bucket>/<key>`` link, and the route that mints a
fresh download URL for each click.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from CoScientist.utils.report_links import artifact_link, remint_report_urls

web_app = importlib.import_module("CoScientist.web.app")

# The dead link from logs/reproduction/tox-antitargets_20260701T141836Z_result.md,
# as an agent pasted it into a report: internal host, one-hour signature.
DEAD_URL = (
    "http://10.32.11.45:9000/tox-bucket/tox_antitargets/"
    "fig5_antitarget_subsets_178a3d1d.png"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=agent-user%2F20260701%2Fus-east-1%2Fs3%2Faws4_request"
    "&X-Amz-Date=20260701T142430Z&X-Amz-Expires=3600"
    "&X-Amz-SignedHeaders=host"
    "&X-Amz-Signature=954bcded7432e3e3fca436c20c93dfb311bd508c0ea4413b15e8325d2e71e090"
)
DEAD_LINK = "/api/artifact/tox-bucket/tox_antitargets/fig5_antitarget_subsets_178a3d1d.png"


@pytest.fixture(autouse=True)
def clean_s3_env(monkeypatch):
    monkeypatch.delenv("S3__ENDPOINT_URL", raising=False)
    monkeypatch.delenv("S3__EXTERNAL_ENDPOINT_URL", raising=False)
    web_app._ARTIFACT_URL_CACHE.clear()
    yield
    web_app._ARTIFACT_URL_CACHE.clear()


# --- the rewrite ------------------------------------------------------------

def test_a_presigned_url_in_a_markdown_link_is_rewritten(monkeypatch):
    monkeypatch.setenv("S3__ENDPOINT_URL", "http://10.32.11.45:9000")
    report = f"**Generated molecules** — [download full CSV]({DEAD_URL})"
    assert remint_report_urls(report) == (
        f"**Generated molecules** — [download full CSV]({DEAD_LINK})"
    )


def test_a_presigned_url_for_a_foreign_host_is_untouched(monkeypatch):
    """An agent can quote a presigned URL that points at someone else's S3.
    Rewriting it to /api/artifact/ would produce a link this S3 cannot
    resolve, so the link stays as it is."""
    monkeypatch.setenv("S3__ENDPOINT_URL", "http://10.32.11.45:9000")
    report = (
        "source: https://s3.eu-west-1.amazonaws.com/other-bucket/data.csv"
        "?X-Amz-Signature=abc123&X-Amz-Expires=3600"
    )
    assert remint_report_urls(report) == report


def test_a_plain_url_to_a_configured_endpoint_is_rewritten(monkeypatch):
    monkeypatch.setenv("S3__ENDPOINT_URL", "http://10.32.11.45:9000")
    url = "http://10.32.11.45:9000/agent-vault/permanent/u1/s1/plot.png"
    report = f"The figure is at {url}."
    assert remint_report_urls(report) == (
        f"The figure is at {artifact_link('agent-vault', 'permanent/u1/s1/plot.png')}."
    )


def test_a_plain_url_to_the_external_endpoint_is_rewritten(monkeypatch):
    monkeypatch.setenv("S3__ENDPOINT_URL", "http://10.32.11.45:9000")
    monkeypatch.setenv("S3__EXTERNAL_ENDPOINT_URL", "https://s3.example.org")
    url = "https://s3.example.org/agent-vault/permanent/u1/s1/plot.png"
    assert remint_report_urls(f"see {url}") == (
        f"see {artifact_link('agent-vault', 'permanent/u1/s1/plot.png')}"
    )


def test_an_s3_uri_in_prose_is_rewritten():
    report = "saved to s3://my-bucket/data/set.zip, done"
    assert remint_report_urls(report) == (
        f"saved to {artifact_link('my-bucket', 'data/set.zip')}, done"
    )


@pytest.mark.parametrize("report", [
    "see https://example.com/docs/page.html for details",
    "download https://files.pythonhosted.org/packages/ab/cool-1.0.zip",
])
def test_an_ordinary_web_link_is_untouched(report):
    assert remint_report_urls(report) == report


@pytest.mark.parametrize("report", [
    "the host is http://10.32.11.45:9000 and nothing else",
    "see s3://bucket-only",
    "no links at all",
])
def test_an_unparseable_reference_is_untouched(report):
    assert remint_report_urls(report) == report


def test_a_key_with_traversal_is_untouched():
    report = "s3://b/../secret"
    assert remint_report_urls(report) == report


@pytest.mark.parametrize("value", [None, "", 42])
def test_non_text_input_passes_through(value):
    assert remint_report_urls(value) == value


def test_the_rewrite_is_idempotent(monkeypatch):
    monkeypatch.setenv("S3__ENDPOINT_URL", "http://10.32.11.45:9000")
    once = remint_report_urls(f"[csv]({DEAD_URL})")
    assert remint_report_urls(once) == once


# --- the endpoint -----------------------------------------------------------

class _StubS3Service:
    def __init__(self, url="https://s3.example.org/fresh.png", raises=None):
        self.url = url
        self.raises = raises
        self.calls = []

    def generate_presigned_url(self, s3_key, method="get_object", expiration=360,
                               bucket_name=None, response_content_type=None):
        self.calls.append((s3_key, method, expiration, bucket_name, response_content_type))
        if self.raises is not None:
            raise self.raises
        return self.url


def _client():
    return TestClient(web_app.create_app())


def test_a_non_vault_bucket_is_signed_directly(monkeypatch):
    stub = _StubS3Service()
    monkeypatch.setattr(web_app, "s3_service", stub)

    with _client() as client:
        response = client.get(
            "/api/artifact/tox-bucket/tox_antitargets/fig5.png",
            follow_redirects=False,
        )

    assert response.status_code == 302
    assert response.headers["location"] == "https://s3.example.org/fresh.png"
    assert stub.calls == [("tox_antitargets/fig5.png", "get_object", 3600, "tox-bucket", "image/png")]


def test_a_minted_url_overrides_the_content_type(monkeypatch):
    """An object uploaded without a ContentType comes back as
    binary/octet-stream, and a browser refuses to render an SVG from that.
    The minted URL overrides the response Content-Type from the extension."""
    stub = _StubS3Service()
    monkeypatch.setattr(web_app, "s3_service", stub)

    with _client() as client:
        response = client.get(
            "/api/artifact/coder-sandbox/sandbox-1/diagrams/canvas.svg",
            follow_redirects=False,
        )

    assert response.status_code == 302
    assert stub.calls == [
        ("sandbox-1/diagrams/canvas.svg", "get_object", 3600, "coder-sandbox", "image/svg+xml")
    ]


def test_a_vault_object_is_minted_through_the_vault(monkeypatch):
    calls = []

    def fake_call_vault_sync(tool_name, **args):
        calls.append((tool_name, args))
        return {"presigned_url": "https://s3.example.org/vault.png"}

    monkeypatch.setenv("S3__BUCKET_NAME", "agent-vault")
    monkeypatch.setattr(web_app, "vault_url", lambda: "http://vault:7331")
    monkeypatch.setattr(web_app, "call_vault_sync", fake_call_vault_sync)

    with _client() as client:
        response = client.get(
            "/api/artifact/agent-vault/ephemeral/u1/s1/plot.png",
            follow_redirects=False,
        )

    assert response.status_code == 302
    assert response.headers["location"] == "https://s3.example.org/vault.png"
    assert calls == [("get_download_link", {"s3_key": "ephemeral/u1/s1/plot.png"})]


def test_a_second_click_hits_the_cache(monkeypatch):
    stub = _StubS3Service()
    monkeypatch.setattr(web_app, "s3_service", stub)

    with _client() as client:
        for _ in range(3):
            response = client.get(
                "/api/artifact/tox-bucket/tox_antitargets/fig5.png",
                follow_redirects=False,
            )
            assert response.status_code == 302

    assert len(stub.calls) == 1


def test_a_refused_mint_is_a_404(monkeypatch):
    monkeypatch.setenv("S3__BUCKET_NAME", "agent-vault")
    monkeypatch.setattr(web_app, "vault_url", lambda: "http://vault:7331")
    monkeypatch.setattr(web_app, "call_vault_sync", lambda *a, **k: None)

    with _client() as client:
        response = client.get(
            "/api/artifact/agent-vault/ephemeral/u1/s1/gone.png",
            follow_redirects=False,
        )

    assert response.status_code == 404


def test_a_failing_mint_is_a_502(monkeypatch):
    stub = _StubS3Service(raises=ConnectionError("endpoint down"))
    monkeypatch.setattr(web_app, "s3_service", stub)

    with _client() as client:
        response = client.get(
            "/api/artifact/tox-bucket/tox_antitargets/fig5.png",
            follow_redirects=False,
        )

    assert response.status_code == 502
    assert "unreachable" in response.json()["detail"]
