import logging
import shutil
from pathlib import Path

from ..base import ETLStep
from ..context import ETLContext

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PublishStep(ETLStep):
    
    name = "publish"
    
    def run(self, ctx: ETLContext) -> None:
        article_id = ctx.article.id
        
        manifest_data = ctx.artifact_store.get_metadata(article_id, "metadata_extraction")
        if not manifest_data:
            raise RuntimeError(f"{self.name} step requires processing metadata")
        
        paper_metadata = manifest_data["paper_metadata"]
        
        chunks_to_upload = []
        vectors_to_upload = []
        
        if ctx.chunks and ctx.embeddings:
            for role, chunks in ctx.chunks.items():
                if role not in ctx.embeddings:
                    continue
                
                role_embeddings = ctx.embeddings[role]["vectors"]
                
                if len(chunks) == len(role_embeddings):
                    chunks_to_upload.extend(chunks)
                    vectors_to_upload.extend(role_embeddings)
                else:
                    raise RuntimeError(
                        "Number of input texts and number of embeddings do not match: "
                        f"{len(chunks)} != {len(role_embeddings)}"
                    )
        
        try:
            if chunks_to_upload:
                ctx.vector_store.delete_by_article_id(article_id)
                ctx.vector_store.upsert_chunks(chunks_to_upload, vectors_to_upload)
            
            pdf_data = ctx.artifact_store.get_file(article_id, "fetching", "source.pdf")
            
            if not pdf_data:
                raise RuntimeError(f"{self.name} step requires source PDF file")
            
            html = ctx.artifact_store.get_html(article_id, "metadata_extraction")
            if not html:
                raise RuntimeError(f"{self.name} step requires HTML")
            
            image_names = ctx.artifact_store.list_images(article_id, "image_captioning")
            images = {
                name: ctx.artifact_store.get_image(article_id, "image_captioning", name)
                for name in image_names
            }
            
            ctx.public_store.publish_article(
                domain=paper_metadata["domain"],
                article_id=article_id,
                html=html,
                images=images,
                metadata=manifest_data,
                pdf_data=pdf_data,
            )
            
            # Clean up artifacts and source file after successful publication
            ctx.artifact_store.delete_article(article_id)
            if ctx.article.source_type == "local":
                source_path = Path(ctx.article.source_ref)
                if source_path.exists():
                    new_name = f"{article_id}_{source_path.name}"
                    destination = ctx.processed_papers_path / new_name
                    shutil.move(str(source_path), str(destination))
                    logger.info(f"[{self.name}] Moved processed source file to {destination}")
        
        except Exception as e:
            logger.error(f"[{self.name}] Error publishing {article_id}. Rolling back vector and artifact stores...")
            ctx.vector_store.delete_by_article_id(article_id)
            ctx.public_store.delete_article(paper_metadata["domain"], article_id)
            raise e
