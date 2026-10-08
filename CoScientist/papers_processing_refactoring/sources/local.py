import hashlib
from pathlib import Path

from ..sources.base import ArticleSource
from ..domain.entities import Article


class LocalSource(ArticleSource):
    
    def __init__(self, papers_dir: Path):
        self.papers_dir = papers_dir
    
    def list_articles(self):
        for pdf in self.papers_dir.glob("*.pdf"):
            article_id = hashlib.md5(pdf.read_bytes()).hexdigest()
            yield Article(
                id=article_id,
                source_type="local",
                source_ref=pdf,
                name=pdf.name,
                # source=self
            )
    
    def fetch(self, article: Article) -> bytes:
        with open(Path(article.source_ref), "rb") as f:
            return f.read()