"""GitHub discover → validate → rank for Experiment Module / Alembic.

Wired from ``build_experiment_context`` when ``route_alembic`` is on and MCP
inventory does not cover the ask — fills ``experiment_context.repo_candidates``.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import httpx
from dotenv import dotenv_values

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
DEFAULT_LANGUAGE = "python"
DEFAULT_LIMIT = 5
DEFAULT_SEARCH_PER_QUERY = 15
DEFAULT_TIMEOUT_S = 20.0
MAX_QUERIES = 2  # unauthenticated quota: 30 search/hr

_DENY_NAME_RE = re.compile(
    r"(?:awesome[\-_]|cheat[\-_]?sheet|\btutorials?\b|\bcourses?\b|examples?\-only|"
    r"\blearning[\-_]?resources?\b|\binterview[\-_]?prep\b|\broadmap\b|\bbookmarks?\b)",
    re.I,
)
_STOPWORDS = frozenset(
    """
    a an the and or for to of in on with without using use via from into by
    is are be been being this that these those it its as at than then so
    can could should would will may might must need needs needed please
    implement write create make build run execute compute calculate find
    get set show list check test evaluate assess compare analyze analyse
    generate suggest propose design develop code script python repo
    repository github tool tools library package module function method
    task experiment scientific research paper based basedon usingbased
    данные задача эксперимент напиши реализуй сделай проверь найди
    """.split()
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")
_REPO_URL_RE = re.compile(
    r"https?://(?:www\.)?github\.com/(?P<owner>[A-Za-z0-9_.\-]+)/(?P<repo>[A-Za-z0-9_.\-]+)",
    re.I,
)
_PERMISSIVE_LICENSE_IDS = frozenset({
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0", "mpl-2.0",
})
_COPYLEFT_LICENSE_IDS = frozenset({
    "gpl-2.0", "gpl-3.0", "agpl-3.0", "lgpl-2.1", "lgpl-3.0",
})


def ensure_github_token_env() -> str:
    """Return a GitHub token, loading ``CoScientist/.env`` if needed."""
    token = (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "").strip()
    if token:
        return token
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return ""
    try:
        for key, value in dotenv_values(env_path).items():
            if value is not None and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        logger.warning("could not read %s for GitHub token: %s", env_path, exc)
    return (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "").strip()


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(re.sub(r"[_\-]+", " ", text.lower()))


@dataclass
class RepoValidation:
    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reject_reason: str | None = None


@dataclass
class RepoCandidate:
    url: str
    host: str = "github.com"
    owner: str = ""
    repo_name: str = ""
    source: str = "github_search"
    provenance_ref: str = ""
    fit_score: float = 0.0
    fit_reason: str = ""
    validation: RepoValidation = field(default_factory=lambda: RepoValidation(ok=True))
    description: str = ""
    stars: int = 0
    forks: int = 0
    language: str | None = None
    license_id: str | None = None
    topics: list[str] = field(default_factory=list)
    archived: bool = False
    pushed_at: str | None = None
    open_issues: int = 0
    score_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo_name}".strip("/")

    def to_context_item(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "host": self.host,
            "owner": self.owner,
            "repo_name": self.repo_name,
            "kind": "git_repo",
            "source": self.source,
            "fit_score": f"{self.fit_score:.3f}",
            "fit_reason": self.fit_reason[:240],
            "description": (self.description or "")[:200],
            "stars": self.stars,
            "topics": list(self.topics or [])[:8],
        }


@dataclass
class RepoSearchResult:
    query: str
    search_queries: list[str]
    candidates: list[RepoCandidate]
    rejected: list[RepoCandidate] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total_raw: int = 0


def extract_keywords(text: str, *, limit: int = 12) -> list[str]:
    if not text:
        return []
    counts: dict[str, int] = {}
    for tok in _tokenize(text):
        if tok in _STOPWORDS or tok.isdigit():
            continue
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return [t for t, _ in ranked[:limit]]


def extract_explicit_github_urls(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _REPO_URL_RE.finditer(text or ""):
        owner, repo = match.group("owner"), match.group("repo").removesuffix(".git")
        key = f"{owner}/{repo}".lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((owner, repo))
    return out


def build_search_queries(
    ask: str,
    *,
    language: str = DEFAULT_LANGUAGE,
    max_queries: int = MAX_QUERIES,
) -> list[str]:
    """1–2 GitHub queries; first token alone so AND-bags do not miss niche packages."""
    ask = (ask or "").strip()
    if not ask:
        return []
    keywords = extract_keywords(ask, limit=8)
    pack = [k for k in keywords if re.search(r"py$", k) and len(k) >= 5]
    keywords = pack + [k for k in keywords if k not in pack]
    queries: list[str] = []
    seen: set[str] = set()

    def _add(body: str) -> None:
        parts = [body.strip()] if body.strip() else []
        if language:
            parts.append(f"language:{language}")
        parts.append("archived:false")
        query = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if not query or query in seen:
            return
        if not re.search(r"[A-Za-z0-9]", re.sub(r"\b(?:language|archived|topic):[^\s]+", "", query)):
            return
        seen.add(query)
        queries.append(query)

    if keywords:
        _add(keywords[0])
        if not pack and len(keywords) > 1:
            _add(" OR ".join(keywords[1:4]))
    if not queries and ask:
        _add(" ".join(extract_keywords(ask, limit=4)))
    return queries[:max_queries]


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _freshness_score(pushed_at: str | None) -> tuple[float, str]:
    dt = _parse_dt(pushed_at)
    if dt is None:
        return 5.0, "push date unknown"
    days = max(0, (datetime.now(timezone.utc) - dt).days)
    if days <= 90:
        return 25.0, f"pushed {days}d ago"
    if days <= 365:
        return 15.0, f"pushed {days}d ago"
    if days <= 730:
        return 8.0, f"pushed {days}d ago"
    return 2.0, f"stale ({days}d)"


def _keyword_overlap_score(ask_tokens: set[str], blob: str) -> tuple[float, int]:
    if not ask_tokens or not blob:
        return 0.0, 0
    hits = ask_tokens & set(_tokenize(blob))
    if not hits:
        return 0.0, 0
    return min(30.0, float(len(hits) * 5)), len(hits)


def alembic_served_repo_identities() -> set[str]:
    """Repo URLs whose Alembic MCP is already served (best-effort)."""
    try:
        from CoScientist.tools import alembic_tools as at

        at.reload_mcp_builds()
        return {
            at._repo_identity(rec["repo_url"])
            for rec in at._JOBS.values()
            if rec.get("status") == "done"
            and str(rec.get("mcp_url") or "").startswith("http")
        }
    except Exception:
        return set()


def score_candidate(
    candidate: RepoCandidate,
    ask: str,
    served_repos: set[str] | None = None,
) -> RepoCandidate:
    """Set fit_score / fit_reason / score_breakdown (0–1 / 0–100)."""
    ask_tokens = set(extract_keywords(ask, limit=20))
    stars = max(int(candidate.stars or 0), 0)
    forks = max(int(candidate.forks or 0), 0)
    stars_pts = min(30.0, math.log10(max(stars, 1)) * 10.0)
    forks_pts = min(10.0, math.log10(max(forks, 1)) * 5.0)
    fresh_pts, fresh_note = _freshness_score(candidate.pushed_at)
    lic = (candidate.license_id or "").lower()
    if lic in _PERMISSIVE_LICENSE_IDS:
        license_pts, lic_note = 15.0, f"permissive license {lic}"
    elif lic and lic not in {"noassertion", "other"}:
        license_pts, lic_note = 8.0, f"license {lic}"
    else:
        license_pts, lic_note = 0.0, "no clear license"
    blob = " ".join([
        candidate.repo_name or "",
        candidate.description or "",
        " ".join(candidate.topics or []),
        candidate.full_name,
    ])
    overlap_pts, n_hits = _keyword_overlap_score(ask_tokens, blob)
    name_bonus = 0.0
    name_l = re.sub(r"[\-_]+", "", (candidate.repo_name or "").lower())
    for tok in ask_tokens:
        needle = re.sub(r"[\s\-_]+", "", tok.lower())
        if len(needle) >= 4 and (needle == name_l or needle in name_l):
            name_bonus = 12.0
            break
    penalty = 0.0
    notes: list[str] = []
    if candidate.archived:
        penalty += 40.0
        notes.append("archived")
    if _DENY_NAME_RE.search(candidate.full_name) or _DENY_NAME_RE.search(candidate.description or ""):
        penalty += 20.0
        notes.append("tutorial/meta name")
    if lic in _COPYLEFT_LICENSE_IDS:
        penalty += 5.0
        notes.append(f"copyleft {lic}")
    served_pts = 0.0
    if served_repos:
        ident = re.sub(r"\.git$", "", (candidate.url or "").strip().rstrip("/")).lower()
        if ident in served_repos:
            served_pts = 24.0
    total = max(
        0.0,
        min(100.0, stars_pts + forks_pts + fresh_pts + license_pts + overlap_pts + name_bonus + served_pts - penalty),
    )
    reason_bits = [f"stars={stars}", fresh_note, lic_note, f"keyword_hits={n_hits}"]
    if name_bonus:
        reason_bits.append("name_boost")
    if served_pts:
        reason_bits.append("alembic_already_served")
    if notes:
        reason_bits.append("penalties=" + ",".join(notes))
    candidate.fit_score = round(total / 100.0, 4)
    candidate.fit_reason = "; ".join(reason_bits)
    candidate.score_breakdown = {
        "stars": round(stars_pts, 2),
        "forks": round(forks_pts, 2),
        "freshness": round(fresh_pts, 2),
        "license": round(license_pts, 2),
        "keyword_overlap": round(overlap_pts, 2),
        "name_bonus": round(name_bonus, 2),
        "served_bonus": round(served_pts, 2),
        "penalty": round(penalty, 2),
        "raw_0_100": round(total, 2),
    }
    return candidate


def validate_candidate(
    candidate: RepoCandidate,
    *,
    denylist: Sequence[str] = (),
    require_language: str | None = DEFAULT_LANGUAGE,
) -> RepoCandidate:
    url = (candidate.url or "").strip()
    parsed = urlparse(url if "://" in url else f"https://{url}")
    lang = (candidate.language or "").lower()
    lang_ok = (not require_language) or (not lang) or lang == require_language.lower()
    hay = f"{candidate.url} {candidate.full_name}".lower()
    denied = any(needle and needle in hay for entry in denylist or () if (needle := str(entry or "").strip().lower()))
    desc = (candidate.description or "").strip()
    checks = {
        "url_shape": bool(parsed.netloc and parsed.path.strip("/").count("/") >= 1),
        "not_archived": not bool(candidate.archived),
        "has_name": bool(candidate.owner and candidate.repo_name),
        "language_ok": lang_ok,
        "not_denylisted": not denied,
        "not_meta_repo": not bool(
            _DENY_NAME_RE.search(candidate.full_name) or _DENY_NAME_RE.search(candidate.description or "")
        ),
        "has_signal": bool(desc or candidate.topics or candidate.stars >= 5),
    }
    ok = all(checks.values())
    candidate.validation = RepoValidation(
        ok=ok, checks=checks, reject_reason=None if ok else ", ".join(k for k, v in checks.items() if not v),
    )
    return candidate


def _item_from_github(item: dict[str, Any], *, source: str, provenance: str) -> RepoCandidate:
    license_obj = item.get("license") or {}
    license_id = None
    if isinstance(license_obj, dict):
        license_id = license_obj.get("spdx_id") or license_obj.get("key")
    owner_obj = item.get("owner") or {}
    owner = owner_obj.get("login") if isinstance(owner_obj, dict) else ""
    name = item.get("name") or ""
    full = item.get("full_name") or f"{owner}/{name}"
    if "/" in full and not owner:
        owner, name = full.split("/", 1)
    html_url = (item.get("html_url") or f"https://github.com/{full}").rstrip("/")
    return RepoCandidate(
        url=html_url,
        host="github.com",
        owner=owner or "",
        repo_name=name or "",
        source=source,
        provenance_ref=provenance,
        description=(item.get("description") or "")[:500],
        stars=int(item.get("stargazers_count") or 0),
        forks=int(item.get("forks_count") or 0),
        language=item.get("language"),
        license_id=license_id,
        topics=list(item.get("topics") or []),
        archived=bool(item.get("archived")),
        pushed_at=item.get("pushed_at") or item.get("updated_at"),
        open_issues=int(item.get("open_issues_count") or 0),
    )


class GitHubRepoClient:
    def __init__(
        self,
        *,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.token = (token if token is not None else os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "").strip()
        self.timeout = timeout
        self._external = client
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "coscientist-repo-searcher/0.1",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def __aenter__(self) -> "GitHubRepoClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, headers=self._headers(), follow_redirects=True)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._external is None and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def search_repositories(self, query: str, *, per_page: int = DEFAULT_SEARCH_PER_QUERY) -> dict[str, Any]:
        assert self._client is not None
        response = await self._client.get(
            f"{GITHUB_API}/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": max(1, min(per_page, 30))},
        )
        if response.status_code == 403 and "rate limit" in response.text.lower():
            raise RuntimeError(f"GitHub rate limit: {response.text[:200]}")
        response.raise_for_status()
        return response.json()

    async def get_repository(self, owner: str, repo: str) -> dict[str, Any]:
        assert self._client is not None
        response = await self._client.get(f"{GITHUB_API}/repos/{owner}/{repo}")
        response.raise_for_status()
        return response.json()


async def search_repos(
    ask: str,
    *,
    limit: int = DEFAULT_LIMIT,
    language: str = DEFAULT_LANGUAGE,
    denylist: Sequence[str] = (),
    token: str | None = None,
    client: httpx.AsyncClient | None = None,
    include_explicit_urls: bool = True,
    min_fit_score: float = 0.0,
) -> RepoSearchResult:
    ask = (ask or "").strip()
    queries = build_search_queries(ask, language=language)
    result = RepoSearchResult(query=ask, search_queries=list(queries), candidates=[], rejected=[])
    if not ask:
        result.errors.append("empty ask")
        return result

    by_key: dict[str, RepoCandidate] = {}
    gh = GitHubRepoClient(token=token, client=client)
    manage = client is None
    if manage:
        await gh.__aenter__()
    try:
        if include_explicit_urls:
            for owner, repo in extract_explicit_github_urls(ask):
                try:
                    raw = await gh.get_repository(owner, repo)
                    cand = _item_from_github(raw, source="ask", provenance=f"explicit:{owner}/{repo}")
                    by_key[cand.full_name.lower()] = cand
                except Exception as exc:  # noqa: BLE001
                    result.errors.append(f"explicit {owner}/{repo}: {exc}")
        for query in queries:
            try:
                payload = await gh.search_repositories(query)
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"search {query!r}: {exc}")
                logger.warning("repo search failed query=%r err=%s", query, exc)
                continue
            items = payload.get("items") or []
            result.total_raw += len(items)
            for item in items:
                cand = _item_from_github(item, source="github_search", provenance=query)
                key = cand.full_name.lower()
                if key not in by_key:
                    by_key[key] = cand
                elif by_key[key].source != "ask" and cand.stars > by_key[key].stars:
                    by_key[key] = cand
    finally:
        if manage:
            await gh.__aexit__()

    accepted: list[RepoCandidate] = []
    rejected: list[RepoCandidate] = []
    served = alembic_served_repo_identities()
    for cand in by_key.values():
        validate_candidate(cand, denylist=denylist, require_language=language)
        score_candidate(cand, ask, served_repos=served)
        overlap = float(cand.score_breakdown.get("keyword_overlap") or 0)
        if cand.source != "ask" and overlap <= 0 and cand.stars < 50:
            cand.validation = RepoValidation(
                ok=False,
                checks={**cand.validation.checks, "keyword_overlap": False},
                reject_reason="no keyword overlap with ask",
            )
        if not cand.validation.ok or cand.fit_score < min_fit_score:
            if cand.validation.ok and cand.fit_score < min_fit_score:
                cand.validation = RepoValidation(
                    ok=False,
                    checks={**cand.validation.checks, "min_fit_score": False},
                    reject_reason=f"fit_score {cand.fit_score:.3f} < {min_fit_score}",
                )
            rejected.append(cand)
        else:
            accepted.append(cand)
    accepted.sort(key=lambda c: (c.source != "ask", -c.fit_score, -c.stars))
    rejected.sort(key=lambda c: -c.fit_score)
    result.candidates = accepted[: max(1, limit)] if accepted else []
    result.rejected = rejected[:10]
    return result


def search_repos_sync(ask: str, **kwargs: Any) -> RepoSearchResult:
    """Sync wrapper safe inside ADK's already-running asyncio loop."""
    if kwargs.get("token") is None:
        kwargs["token"] = ensure_github_token_env() or None

    def _run() -> RepoSearchResult:
        return asyncio.run(search_repos(ask, **kwargs))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result(timeout=120)


__all__ = [
    "RepoCandidate",
    "RepoSearchResult",
    "build_search_queries",
    "extract_keywords",
    "search_repos",
    "search_repos_sync",
    "validate_candidate",
]
