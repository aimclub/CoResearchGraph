from typing import List, Tuple

from .base_reranker import BatchedReranker


class LocalReranker(BatchedReranker):
    
    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-multilingual-reranker-base",
        max_length: int = 2048,
        batch_size: int = 32,
        trust_remote_code: bool = True,
    ):
        super().__init__(batch_size=batch_size)

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers must be installed for LocalReranker"
            ) from e

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
        )

        # warmup
        self.model.predict([["warmup", "query"]])

    def _score_batch(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:

        scores = self.model.predict(pairs)
        return scores.tolist()