import logging

from ..base import ETLStep
from ..context import ETLContext
from ...sources.base import ArticleSource

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FetchStep(ETLStep):
    
    name = "fetching"

    def __init__(self, source: ArticleSource):
        self.source = source

    def run(self, ctx: ETLContext) -> None:
        try:
            raw_bytes = self.source.fetch(ctx.article)
        except Exception as e:
            logger.error(f"Fetching PDF bytes failed for article {ctx.article.id}: {e}")
            raise e
            
        ctx.raw_data = raw_bytes
        ctx.artifact_store.put_file(ctx.article.id, self.name, "source.pdf", raw_bytes)
