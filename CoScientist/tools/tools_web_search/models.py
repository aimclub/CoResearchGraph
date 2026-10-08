from pydantic import BaseModel, Field, HttpUrl


#----WEB TOOL SEARCH----------------------------------------------------------------------------------
class FoundMCPServer(BaseModel):
    """A single MCP server entry normalized across sources."""

    name: str = Field(description="Display name of the server.")
    slug: str = Field(description="URL-safe identifier within the source registry.")
    description: str = Field(default="", description="Short human description of the server.")
    source: str = Field(description="Origin registry, e.g. 'mcpservers.com' or 'mcpservers.org'.")
    page_url: HttpUrl = Field(description="Canonical registry page for this server.")
    repo_url: HttpUrl | None = Field(default=None, description="Source repository URL if known.")

    category: str | None = Field(default=None, description="Primary category, if provided.")
    categories: list[str] = Field(default_factory=list, description="All categories/tags.")
    author: str | None = Field(default=None, description="Author or maintainer.")
    language: str | None = Field(default=None, description="Implementation language, if known.")
    stars: int | None = Field(default=None, description="GitHub stars, if reported by the source.")
    official: bool = Field(default=False, description="Marked official by the source.")
    recommended: bool = Field(default=False, description="Marked recommended/featured by the source.")
    supports_sse: bool = Field(default=False, description="Server-Sent Events transport supported.")


class SourceError(BaseModel):
    """Per-source failure record so the agent can see partial results clearly."""
    source: str
    error: str


class MCPSearchResult(BaseModel):
    """Aggregated, normalized search result returned to the agent."""

    query: str = Field(description="The query that produced these results.")
    servers: list[FoundMCPServer] = Field(default_factory=list, description="All matching servers.")
    total_found: int = Field(description="Total number of servers returned.")
    sources_used: list[str] = Field(description="Registries successfully queried.")
    errors: list[SourceError] = Field(
        default_factory=list,
        description="Sources that failed; remaining sources still produced results.",
    )

    def to_agent_text(self, limit: int | None = 20) -> str:
        """Compact text rendering suited for direct injection into an LLM tool result."""
        items = self.servers if limit is None else self.servers[:limit]
        lines = [f'Query: "{self.query}"  ({self.total_found} result(s) from {", ".join(self.sources_used) or "no sources"})']
        if self.errors:
            for e in self.errors:
                lines.append(f"  [warn] {e.source}: {e.error}")
        for i, s in enumerate(items, 1):
            flags = []
            if s.official: flags.append("official")
            if s.recommended: flags.append("recommended")
            if s.supports_sse: flags.append("sse")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            meta = []
            if s.language: meta.append(s.language)
            if s.stars is not None: meta.append(f"{s.stars}★")
            if s.category: meta.append(s.category)
            meta_str = f"  ({'; '.join(meta)})" if meta else ""
            lines.append(f"{i}. {s.name}{flag_str}{meta_str} — {s.source}")
            if s.description:
                lines.append(f"   {s.description}")
            lines.append(f"   page: {s.page_url}" + (f"   repo: {s.repo_url}" if s.repo_url else ""))
        if limit is not None and len(self.servers) > limit:
            lines.append(f"... and {len(self.servers) - limit} more.")
        return "\n".join(lines)
