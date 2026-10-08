import hashlib
import time
from typing import Iterable

import requests
import feedparser

from ..sources.base import ArticleSource
from ..domain.entities import Article


class ArxivSource(ArticleSource):

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(
        self,
        query: str,
        domain:  str = "",
        max_results: int = 50,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        polite_delay: float = 3.0,
    ):
        self.query = query
        self.domain = domain
        self.max_results = max_results
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.polite_delay = polite_delay
        
    def list_articles(self) -> Iterable[Article]:
        params = {
            "search_query": self.query,
            "start": 0,
            "max_results": self.max_results,
            "sortBy": self.sort_by,
            "sortOrder": self.sort_order,
        }

        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        feed = feedparser.parse(response.text)

        for entry in feed.entries:
            arxiv_id = entry.id.split("/abs/")[-1]

            article_id = hashlib.md5(arxiv_id.encode("utf-8")).hexdigest()

            yield Article(
                id=article_id,
                source_type="remote",
                source_ref=arxiv_id,
                name=entry.title.strip(),
                domain=self.domain,
                metadata={
                    "arxiv_id": arxiv_id,
                    "published": entry.published,
                    "updated": entry.updated,
                    "authors": [a.name for a in entry.authors],
                    # "summary": entry.summary,  # Disabled: summaries are not stored.
                    "categories": entry.tags if hasattr(entry, "tags") else [],
                },
            )
            
        time.sleep(self.polite_delay)

    def fetch(self, article: Article) -> bytes:
        arxiv_id = article.source_ref
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        return response.content
