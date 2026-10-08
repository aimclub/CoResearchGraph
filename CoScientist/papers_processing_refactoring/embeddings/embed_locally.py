import logging
from typing import List

from .base_embedder import BatchedEmbeddingModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalEmbeddingModel(BatchedEmbeddingModel):

    def __init__(self, model_name: str, batch_size: int = 32):
        super().__init__(batch_size=batch_size)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers must be installed for LocalEmbeddingModel"
            ) from e

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise e

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        return embeddings.tolist()