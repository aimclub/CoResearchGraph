from abc import ABC, abstractmethod
from typing import List, Tuple


class Reranker(ABC):
    """
    Cross-encoder style reranker.
    Used only at retrieval stage.
    """

    @abstractmethod
    def score_pairs(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:
        """
        Score (query, document) pairs.
        """
        raise NotImplementedError

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int | None = None,
    ) -> List[tuple[str, float]]:
        """
        Convenience helper:
        - builds pairs
        - scores
        - sorts
        """

        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self.score_pairs(pairs)
        
        if len(scores) != len(pairs):
            raise RuntimeError(
                "Number of input pairs and number of scores do not match: "
                f"{len(scores)} != {len(pairs)}"
            )

        ranked = list(zip(documents, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            ranked = ranked[:top_k]

        return ranked


class BatchedReranker(Reranker):
    """
    Adds transparent batching.
    """

    def __init__(self, batch_size: int = 32):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive integer, got {batch_size}")
        self.batch_size = batch_size

    def score_pairs(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:

        if not pairs:
            return []

        all_scores = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            batch_scores = self._score_batch(batch)
            all_scores.extend(batch_scores)

        return all_scores

    @abstractmethod
    def _score_batch(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:
        raise NotImplementedError