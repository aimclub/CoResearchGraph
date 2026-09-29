"""The two FEDOT rows in the activity rail, and where they lead.

PR 368 added the rows and nothing added their labels, so a Russian operator
got `FEDOT.MAS Demo (agent graph)` in Latin — the array's own fallback string,
which `applyLanguage` leaves in place when the key is missing. Every other row
in that rail is translated, so the two new ones read as unfinished.

The viewer is now vendored and both pages require a session/run to read data.
Even with the old standalone backend unavailable, the page and assets work.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
JS = ROOT / "CoScientist" / "web" / "static" / "js"


def _read(path: Path) -> str:
    return io.open(path, encoding="utf-8").read()


# ── the rail ────────────────────────────────────────────────────────────────

def test_every_row_in_the_rail_has_a_translated_label():
    """`initAgentNav` renders `data-i18n="agent.<name>.desc"` for each row and
    calls `applyLanguage()`; a name with no key keeps the English fallback
    baked into the array."""
    rail = _read(JS / "activity_rail.js")
    i18n = _read(JS / "i18n.js")

    block = rail[rail.index("const AGENTS = ["):rail.index("];", rail.index("const AGENTS = ["))]
    names = re.findall(r"\{\s*name:\s*\"([A-Za-z_]+)\"", block)
    assert "FedotTrace" in names and "FedotDemo" in names, names

    missing = [n for n in names
               if f"'agent.{n}.desc'" not in i18n and n != "__settings__"]
    assert not missing, f"no agent.<name>.desc in i18n.js for: {missing}"

    # Both languages, or one of them silently shows the other's text.
    for name in ("FedotTrace", "FedotDemo"):
        row = i18n[i18n.index(f"'agent.{name}.desc'"):]
        row = row[:row.index("\n")]
        assert "en:" in row and "ru:" in row, row


def test_a_rail_row_that_opens_a_page_is_actually_handled():
    """A row with an href still goes through `onAgentClick`, and a name that
    falls off the end of that chain opens nothing at all."""
    rail = _read(JS / "activity_rail.js")
    handler = rail[rail.index("function onAgentClick"):]
    handler = handler[:handler.index("\n    }")]
    for name in ("FedotTrace", "FedotDemo"):
        assert f'name === "{name}"' in handler, f"{name} has no branch"


# ── where the demo row leads when the demo is not running ───────────────────

class _Refusing:
    """An httpx client that cannot reach the gui-demo, like a stopped one."""

    def __init__(self, *_a, **_k):
        pass

    def build_request(self, *_a, **_k):
        return object()

    async def send(self, *_a, **_k):
        raise httpx.ConnectError("nobody is listening")

    async def aclose(self):
        return None


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path / "execution"))
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web_state"))
    monkeypatch.setattr(httpx, "AsyncClient", _Refusing)

    from CoScientist.web.app import create_app

    with TestClient(create_app()) as c:
        yield c


def test_demo_is_vendored_and_does_not_need_another_backend(client):
    page = client.get("/fedot-demo/")
    assert page.status_code == 200
    assert 'id="fedot-runs"' in page.text
    for name in ("app.js", "styles.css"):
        response = client.get("/fedot-demo/" + name)
        assert response.status_code == 200
        assert "no-cache" in response.headers["cache-control"]
    assert client.post("/fedot-demo/api/run", json={}).status_code == 405


def test_demo_redirect_preserves_session_and_run(client):
    response = client.get("/fedot-demo?user_id=u&session_id=s&run_id=r", follow_redirects=False)
    assert response.headers["location"] == "/fedot-demo/?user_id=u&session_id=s&run_id=r"


def test_trace_and_stream_reject_unscoped_access(client):
    page = client.get("/fedot-trace")
    assert page.status_code == 200 and "FEDOT.MAS Trace" in page.text
    assert "/static/css/fonts.css" in page.text
    assert "/static/js/fedot_runs.js" in page.text
    for url in ("/api/fedot-langfuse-trace", "/api/fedot-live-stream"):
        assert client.get(url).status_code == 400


def test_the_front_end_is_never_served_from_a_stale_cache(client):
    """The rail rows above were reported missing from a page whose server had
    them: `index.html` goes out `no-store`, but the modules it loads went out
    with no `Cache-Control` at all, so a browser was free to keep its own copy
    for as long as its heuristic allowed."""
    for path in ("/static/js/activity_rail.js", "/static/js/i18n.js",
                 "/static/css/main.css", "/static/vis-network.min.js"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "no-cache" in r.headers.get("cache-control", ""), path
        assert r.headers.get("etag"), f"{path} has nothing to revalidate WITH"

    # And revalidation is what it costs — not the file again.
    first = client.get("/static/js/activity_rail.js")
    again = client.get("/static/js/activity_rail.js",
                       headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304 and not again.content

    # The page itself is stricter still, and stays that way.
    page = client.get("/")
    assert "no-store" in page.headers.get("cache-control", "")
