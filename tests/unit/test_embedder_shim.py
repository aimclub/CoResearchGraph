"""SafeAPIEmbedder: whatever shape the embedding service answers with, a tool
still gets indexed — or the failure is explicit. No network; the HTTP call is
replaced.

The bug this guards: rag_tools' APIEmbedder swallows transport errors and
returns ``[]``, and some services answer a one-text request with a flat vector.
Both make ``BaseEmbedder.embed`` raise AxisError while normalising, so the tool
being indexed registers with status=error and silently disappears from the
catalogue — the system then behaves as if it never existed.
"""

import asyncio

import numpy as np
import pytest
from rag_tools.retrieval import APIEmbedder

from CoScientist.tools.embedder_shim import EmbeddingUnavailableError, SafeAPIEmbedder


def _embedder(monkeypatch, answer):
    """A SafeAPIEmbedder whose backend always answers with ``answer``.

    ``__new__`` skips ``APIEmbedder.__init__`` so the test needs no settings;
    the patched base ``_embed_impl`` is the HTTP call the wrapper delegates to.
    """

    async def _backend(self, texts, batch_size):
        return answer

    monkeypatch.setattr(APIEmbedder, "_embed_impl", _backend)
    emb = SafeAPIEmbedder.__new__(SafeAPIEmbedder)
    emb._initialized = True
    emb._embedding_dim = 3
    return emb


def test_a_flat_single_vector_is_still_a_batch_of_one(monkeypatch):
    """A service that answers one text with a bare vector must not blow up."""
    emb = _embedder(monkeypatch, [0.0, 3.0, 4.0])

    out = asyncio.run(emb.embed("one chunk"))

    assert out.shape == (3,)  # single text in, single vector out
    assert out == pytest.approx([0.0, 0.6, 0.8], abs=1e-6)  # L2-normalised


def test_a_normal_batch_is_unchanged(monkeypatch):
    emb = _embedder(
        monkeypatch, [np.array([3.0, 0.0, 0.0]), np.array([0.0, 0.0, 5.0])]
    )

    out = asyncio.run(emb.embed(["a", "b"]))

    assert out.shape == (2, 3)
    assert out[0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert out[1] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)


def test_an_unreachable_service_says_so_instead_of_raising_axiserror(monkeypatch):
    """APIEmbedder returns [] on any transport error. That must not surface as a
    shape complaint, or the real cause never reaches the caller."""
    emb = _embedder(monkeypatch, [])

    with pytest.raises(EmbeddingUnavailableError, match="unreachable"):
        asyncio.run(emb.embed(["a"]))


def test_a_short_answer_is_refused_rather_than_mis_aligned(monkeypatch):
    """Fewer vectors than texts would silently pair chunk 2 with vector 1."""
    emb = _embedder(monkeypatch, [np.array([1.0, 0.0, 0.0])])

    with pytest.raises(EmbeddingUnavailableError, match="1 vector"):
        asyncio.run(emb.embed(["a", "b"]))
