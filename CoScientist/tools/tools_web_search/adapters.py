import json
import re
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup


from CoScientist.tools.tools_web_search.models import FoundMCPServer


class _Source(ABC):
    """Interface every registry adapter implements."""

    name: str

    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    @abstractmethod
    async def search(self, query: str) -> list[FoundMCPServer]:
        ...


class _McpServersCom(_Source):
    """Adapter for mcpservers.com (Next.js RSC payload)."""

    name = "mcpservers.com"
    _LANGUAGES = {1: "Python", 2: "JavaScript", 3: "TypeScript", 4: "Go", 5: "Rust"}

    async def search(self, query: str) -> list[FoundMCPServer]:
        url = f"https://mcpservers.com/search?q={quote(query)}"
        r = await self._client.get(url)
        r.raise_for_status()
        return self._parse(r.text)

    def _parse(self, html: str) -> list[FoundMCPServer]:
        soup = BeautifulSoup(html, "html.parser")
        raw = None
        for script in soup.find_all("script"):
            text = script.string or ""
            if r'\"initialMcps\"' in text or '"initialMcps"' in text:
                raw = text
                break
        if not raw:
            return []

        push_match = re.search(
            r'self\.__next_f\.push\(\[\s*\d+\s*,\s*(".*?")\s*\]\)',
            raw, re.DOTALL,
        )
        if not push_match:
            return []
        decoded = json.loads(push_match.group(1))

        m = re.search(r'"initialMcps"\s*:\s*(\[.*?\])\s*,\s*"searchQuery"', decoded, re.DOTALL)
        if not m:
            return []
        raw_servers: list[dict[str, Any]] = json.loads(m.group(1))

        out: list[FoundMCPServer] = []
        for s in raw_servers:
            slug = s["slug"]
            out.append(FoundMCPServer(
                name=s.get("name") or slug,
                slug=slug,
                description=s.get("description") or "",
                source=self.name,
                page_url=f"https://mcpservers.com/servers/{slug}",
                repo_url=s.get("github_url") or None,
                category=(s.get("categories") or [None])[0],
                categories=list(s.get("categories") or []),
                author=s.get("author"),
                language=self._LANGUAGES.get(s.get("mcp_language")),
                stars=s.get("stars"),
                official=bool(s.get("official")),
                recommended=bool(s.get("recommended")),
                supports_sse=bool(s.get("is_sse")),
            ))
        return out


class _McpServersOrg(_Source):
    """Adapter for mcpservers.org (TanStack Router hydration payload)."""

    name = "mcpservers.org"

    _RECORD_RE = re.compile(
        r'\$R\[\d+\]\s*=\s*\{'
        r'id:(?P<id>\d+),'
        r'slug:"(?P<slug>[^"]+)",'
        r'name:"(?P<name>(?:[^"\\]|\\.)*)",'
        r'description:"(?P<description>(?:[^"\\]|\\.)*)",'
        r'content:[^,]+,'
        r'url:"(?P<url>(?:[^"\\]|\\.)*)",'
        r'category:"(?P<category>[^"]+)",'
        r'tags:\$R\[\d+\]=\[(?P<tags>[^\]]*)\],'
        r'featured:(?P<featured>!?[01]|true|false)'
        r'\}'
    )
    _PAGINATION_RE = re.compile(
        r'pagination:\$R\[\d+\]=\{'
        r'totalPages:(\d+),currentPage:(\d+),totalItems:(\d+),'
        r'itemsPerPage:(\d+),'
        r'hasNextPage:(!?[01]|true|false),'
        r'hasPrevPage:(!?[01]|true|false)'
    )

    async def search(self, query: str, max_pages: int = 5) -> list[FoundMCPServer]:
        servers: list[FoundMCPServer] = []
        page = 1
        while page <= max_pages:
            page_servers, has_next, total_pages = await self._fetch_page(query, page)
            servers.extend(page_servers)
            if not has_next or page >= total_pages:
                break
            page += 1
        return servers

    async def _fetch_page(self, query: str, page: int) -> tuple[list[FoundMCPServer], bool, int]:
        url = f"https://mcpservers.org/search?query={quote(query)}&page={page}"
        r = await self._client.get(url)
        r.raise_for_status()
        return self._parse(r.text)

    def _parse(self, html: str) -> tuple[list[FoundMCPServer], bool, int]:
        soup = BeautifulSoup(html, "html.parser")
        script = next(
            (s.string for s in soup.find_all("script")
             if s.string and "$_TSR.router" in s.string),
            None,
        )
        if not script:
            return self._parse_dom(soup), False, 1

        servers: list[FoundMCPServer] = []
        for m in self._RECORD_RE.finditer(script):
            d = m.groupdict()
            servers.append(FoundMCPServer(
                name=self._unescape(d["name"]),
                slug=d["slug"],
                description=self._unescape(d["description"]),
                source=self.name,
                page_url=f"https://mcpservers.org/servers/{d['slug']}",
                repo_url=self._unescape(d["url"]) or None,
                category=d["category"] or None,
                categories=self._parse_tags(d["tags"]),
                recommended=d["featured"] in ("!0", "true"),
            ))

        pm = self._PAGINATION_RE.search(script)
        if pm:
            total_pages = int(pm.group(1))
            has_next = pm.group(5) in ("!0", "true")
        else:
            total_pages, has_next = 1, False
        return servers, has_next, total_pages

    def _parse_dom(self, soup: BeautifulSoup) -> list[FoundMCPServer]:
        out: list[FoundMCPServer] = []
        for a in soup.select('a[href^="/servers/"]'):
            slug = a["href"].removeprefix("/servers/")
            name_el = a.select_one(".text-lg.font-semibold")
            desc_el = a.select_one(".text-sm.text-gray-600")
            out.append(FoundMCPServer(
                name=name_el.get_text(strip=True) if name_el else slug,
                slug=slug,
                description=desc_el.get_text(strip=True) if desc_el else "",
                source=self.name,
                page_url=f"https://mcpservers.org/servers/{slug}",
            ))
        return out

    @staticmethod
    def _unescape(s: str) -> str:
        try:
            return json.loads(f'"{s}"')
        except json.JSONDecodeError:
            return s

    @staticmethod
    def _parse_tags(raw: str) -> list[str]:
        if not raw.strip():
            return []
        return [t.strip().strip('"') for t in raw.split(",") if t.strip()]