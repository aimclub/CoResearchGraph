"""A changed frontend script must reach the browser on a plain reload.

The page is no-store; its /static/ assets carry their file version in the URL
and are served no-cache (revalidated by ETag). Without that, Chrome kept an
old status_indicator.js as heuristically fresh and a fixed indicator never
showed up."""
import re

from fastapi.testclient import TestClient

from CoScientist.web.app import WEB_DIR, _versioned_static_refs, create_app


def test_page_references_versioned_assets_served_no_cache():
    with TestClient(create_app()) as client:
        html = client.get("/").text
        refs = re.findall(r'(?:src|href)="(/static/[^"]+)"', html)
        assert refs and all("?v=" in ref for ref in refs)

        asset = client.get(refs[0])
        assert asset.status_code == 200
        assert asset.headers["cache-control"] == "no-cache"
        again = client.get(refs[0], headers={"if-none-match": asset.headers["etag"]})
        assert again.status_code == 304


def test_version_follows_the_file(tmp_path):
    (tmp_path / "a.js").write_text("1")
    html = '<script src="/static/a.js"></script><script src="/static/missing.js"></script>'

    first = _versioned_static_refs(html, tmp_path)
    (tmp_path / "a.js").write_text("22")
    import os
    os.utime(tmp_path / "a.js", ns=(1, 2_000_000_000_000_000_000))
    second = _versioned_static_refs(html, tmp_path)

    assert re.search(r'/static/a\.js\?v=\d+"', first)
    assert first != second
    assert '"/static/missing.js"' in second  # unknown files are left alone
    assert WEB_DIR.exists()
