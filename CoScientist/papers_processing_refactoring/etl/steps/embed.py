import logging

from ..base import ETLStep
from ..context import ETLContext

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingStep(ETLStep):
    
    name = "embed"
    
    def run(self, ctx: ETLContext) -> None:
        if not ctx.chunks:
            raise RuntimeError("EmbeddingStep requires chunks")
        
        ctx.embeddings = {}
        
        for role, chunks in ctx.chunks.items():
            if not chunks:
                continue
            
            texts = [chunk.content for chunk in chunks]
            
            try:
                vectors = ctx.embedding_model.embed_documents(texts)
            except Exception as e:
                logger.error(f"Embedding calculation failed {e}")
                raise e
            
            ctx.embeddings[role] = {
                "chunk_ids": [chunk.id for chunk in chunks],
                "vectors": vectors,
            }
