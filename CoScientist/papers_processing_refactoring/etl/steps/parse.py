import logging
import os
import tempfile
from pathlib import Path

from ..base import ETLStep
from ..context import ETLContext
from ...utils.marker_client import MarkerClient, convert_pdf_with_splitting

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParseStep(ETLStep):

    name = "parsing"
        
    def __init__(self):
        self.marker_client = MarkerClient(base_url=os.getenv("MARKER_URL", "http://localhost:8080/convert"))

    def run(self, ctx: ETLContext) -> None:
        article_id = ctx.article.id

        pdf_data = ctx.artifact_store.get_file(article_id, "fetching", "source.pdf")

        if not pdf_data:
            raise RuntimeError(f"ParseStep: PDF data not found for {article_id}")

        pdf_path = Path(tempfile.gettempdir()) / "papers_ingest" / f"{article_id}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(pdf_data)
        
        try:
            res = convert_pdf_with_splitting(client=self.marker_client, pdf_uri=str(pdf_path))
        finally:
            pdf_path.unlink(missing_ok=True)

        ctx.artifact_store.put_html(article_id, self.name, res.text)
        ctx.artifact_store.put_images(article_id, self.name, res.images)
