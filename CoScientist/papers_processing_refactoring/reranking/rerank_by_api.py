import requests
from typing import List, Tuple

from .base_reranker import BatchedReranker


class APIReranker(BatchedReranker):

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

    def _score_batch(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:
        try:
            response = requests.post(
                self.url,
                json=pairs,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Reranking API request failed ({self.url}, {len(pairs)} pairs): {e}") from e
        # Expected:
        # { "scores": [...] }
        return data["scores"]
