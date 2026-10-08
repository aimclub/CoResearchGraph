from typing import List, Optional, Dict, Any

from ..domain.entities import Chunk
from ..storage.vector.base import VectorStore
from ..embeddings.base_embedder import EmbeddingModel
from ..reranking.base_reranker import Reranker


class TwoStageRetriever:
    
    def __init__(
            self,
            vector_store: VectorStore,
            embedding_model: EmbeddingModel,
            reranker: Optional[Reranker] = None
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.reranker = reranker
    
    def retrieve(
            self,
            query: str,
            top_k: int = 20,
            rerank_k: int = 5,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        
        query_vector = self.embedding_model.embed_query(query)
        
        initial_chunks = self.vector_store.search(
            query_vector=query_vector,
            limit=top_k,
            filters=filters
        )
        
        if not initial_chunks or not self.reranker:
            return initial_chunks[:rerank_k]
        
        pairs = [(query, chunk.content) for chunk in initial_chunks]
        
        scores = self.reranker.score_pairs(pairs)
        
        if len(scores) != len(pairs):
            raise RuntimeError(
                "Number of input pairs and number of scores do not match: "
                f"{len(scores)} != {len(pairs)}"
            )
        
        scored_chunks = list(zip(initial_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        final_chunks = []
        for chunk, score in scored_chunks[:rerank_k]:
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["reranker_score"] = float(score)
            final_chunks.append(chunk)
        
        return final_chunks