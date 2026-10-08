import requests
from typing import List

from .base_embedder import BatchedEmbeddingModel


class APIEmbeddingModel(BatchedEmbeddingModel):
    
    def __init__(
        self,
        url: str,
        timeout: int = 120,
        batch_size: int = 16,
        headers: dict | None = None,
    ):
        super().__init__(batch_size=batch_size)
        self.url = url
        self.timeout = timeout
        self.headers = headers or {}

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = requests.post(
                self.url,
                json=texts,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding API request failed ({self.url}, {len(texts)} texts): {e}") from e

        if "embeddings" not in data:
            raise RuntimeError(f"Embedding API response missing 'embeddings' key: {data!r}")

        if len(data["embeddings"]) != len(texts):
            raise RuntimeError(
                "Number of input texts and number of embeddings do not match: "
                f"{len(data['embeddings'])} != {len(texts)}")
        # Expected format:
        # { "embeddings": [[...], [...], ...] }
        return data["embeddings"]