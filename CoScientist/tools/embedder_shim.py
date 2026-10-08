"""Shape-safe wrapper around ``rag_tools``' HTTP embedder.

``BaseEmbedder.embed`` normalises with a hard
``np.linalg.norm(embeddings, axis=1, keepdims=True)``, which assumes the backend
always answers with a 2-D batch. ``APIEmbedder._embed_impl`` breaks that
assumption in two ways:

* it catches **every** transport error and returns ``[]``, so a service outage
  arrives as ``np.array([])`` — 1-D — and raises ``AxisError: axis 1 is out of
  bounds for array of dimension 1``, hiding the real cause behind a shape
  complaint;
* an embedding service that answers a single-text request with one *flat*
  vector produces the same 1-D array.

Either way the tool being indexed registers with ``status=error`` and never
lands in the catalogue, so ``Retrieve_tools`` silently loses it: the system then
behaves as if the tool did not exist and re-implements the work instead.

:class:`SafeAPIEmbedder` overrides only ``_embed_impl`` — it returns one vector
per input text, whatever rank the backend answered with, and turns "no usable
vectors" into an explicit :class:`EmbeddingUnavailableError`. Everything else
(batching, normalisation, the ``str`` → 1-D / ``list`` → 2-D contract) stays
with the base class.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from rag_tools.retrieval import APIEmbedder


class EmbeddingUnavailableError(RuntimeError):
    """The embedding backend returned no usable vectors for the given texts."""


def _as_vectors(raw: Any, expected: int) -> List[np.ndarray]:
    """``raw`` as exactly ``expected`` 1-D float vectors, whatever its rank."""
    array = np.asarray(raw, dtype=np.float32)
    if array.size == 0:
        raise EmbeddingUnavailableError(
            f"embedding backend returned no vectors for {expected} text(s) — "
            "the embedding service is unreachable or erroring"
        )
    matrix = np.atleast_2d(array)
    if matrix.shape[0] != expected:
        raise EmbeddingUnavailableError(
            f"embedding backend returned {matrix.shape[0]} vector(s) for "
            f"{expected} text(s) — refusing to mis-align vectors against their "
            "payloads"
        )
    return list(matrix)


class SafeAPIEmbedder(APIEmbedder):
    """``APIEmbedder`` whose backend answer is always a well-formed batch."""

    async def _embed_impl(
        self, texts: List[str], batch_size: Optional[int]
    ) -> List[np.ndarray]:
        return _as_vectors(await super()._embed_impl(texts, batch_size), len(texts))
