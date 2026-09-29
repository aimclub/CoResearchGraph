"""Unit tests for standalone repo_searcher (no planner wiring)."""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

from CoScientist.experiments.capabilities.repo_searcher import (
    RepoCandidate,
    build_search_queries,
    extract_keywords,
    search_repos,
    validate_candidate,
)


def test_extract_keywords_drops_stopwords():
    tokens = extract_keywords("Generate drug-like molecules with RDKit clustering for toxicity")
    assert "rdkit" in tokens
    assert "molecules" in tokens or "toxicity" in tokens
    assert "with" not in tokens
    assert "for" not in tokens





def test_build_search_queries_does_not_and_jam_package_name():
    """Orch-expanded pubchempy asks must still search the package name alone."""
    qs = build_search_queries(
        "Implement a python script using the pubchempy PubChem library package "
        "to fetch compound data and print the molecular formula."
    )
    bodies = [
        re.sub(r"\s*language:\S+|\s*archived:\S+", "", q).strip()
        for q in qs
    ]
    assert "pubchempy" in bodies



def test_validate_rejects_archived_and_denylist():
    cand = RepoCandidate(
        url="https://github.com/evil/malware-kit",
        owner="evil",
        repo_name="malware-kit",
        language="Python",
        archived=True,
    )
    validate_candidate(cand, denylist=["evil/malware"])
    assert cand.validation.ok is False
    assert cand.validation.checks["not_archived"] is False
    assert cand.validation.checks["not_denylisted"] is False




class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Minimal async client stub matching httpx.AsyncClient.get usage."""

    def __init__(self, search_payload: dict, repo_payloads: dict[str, dict] | None = None):
        self.search_payload = search_payload
        self.repo_payloads = repo_payloads or {}
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url: str, params: dict | None = None):
        self.calls.append((url, params or {}))
        if "/search/repositories" in url:
            return _FakeResponse(self.search_payload)
        # /repos/owner/repo
        parts = url.rstrip("/").split("/")
        key = f"{parts[-2]}/{parts[-1]}"
        if key in self.repo_payloads:
            return _FakeResponse(self.repo_payloads[key])
        return _FakeResponse({"message": "Not Found"}, status_code=404, text="Not Found")

    async def aclose(self) -> None:
        return None


def test_search_repos_ranks_and_filters_with_fake_client():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "total_count": 2,
        "items": [
            {
                "full_name": "whitead/synspace",
                "name": "synspace",
                "owner": {"login": "whitead"},
                "html_url": "https://github.com/whitead/synspace",
                "description": "Synthetic accessibility enumeration in chemical space",
                "stargazers_count": 180,
                "forks_count": 20,
                "language": "Python",
                "license": {"spdx_id": "MIT"},
                "topics": ["chemistry"],
                "archived": False,
                "pushed_at": now,
                "open_issues_count": 2,
            },
            {
                "full_name": "someone/awesome-chemistry",
                "name": "awesome-chemistry",
                "owner": {"login": "someone"},
                "html_url": "https://github.com/someone/awesome-chemistry",
                "description": "Awesome list of chemistry links",
                "stargazers_count": 9000,
                "forks_count": 500,
                "language": "Python",
                "license": {"spdx_id": "MIT"},
                "topics": [],
                "archived": False,
                "pushed_at": now,
                "open_issues_count": 0,
            },
        ],
    }
    fake = _FakeClient(payload)
    result = asyncio.run(search_repos(
        "Estimate synthetic accessibility SA score for molecules",
        limit=5,
        client=fake,  # type: ignore[arg-type]
    ))
    urls = [c.url for c in result.candidates]
    assert "https://github.com/whitead/synspace" in urls
    assert all("awesome" not in c.repo_name.lower() for c in result.candidates)
    assert result.candidates[0].fit_score > 0
    assert result.search_queries



