import ast
from typing import Any, Dict, Optional

import chromadb

from .base import VectorStore
from ...domain.entities import Chunk


class ChromaVectorStore(VectorStore):
    def __init__(self, chroma_host: str, chroma_port: int, chroma_collection: str):
        self.client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=chromadb.Settings(allow_reset=False),
        )
        self.collection = self.client.get_or_create_collection(name=chroma_collection)
    
    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")
        
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            meta: Dict[str, Any] = {
                "article_id": chunk.article_id,
                "role": chunk.role,
                "domain": chunk.domain or "default",
                "field": chunk.field or "default",
                "modality": chunk.modality,
            }
            if chunk.metadata:
                meta.update({
                    k: v if isinstance(v, (int, float, bool, str)) else str(v) for k, v in chunk.metadata.items()
                })
            metadatas.append(meta)
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def search(self, query_vector: list[float] | Dict[str, Any], limit: int = 5, filters: Optional[dict] = None) -> list[Chunk]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=filters
        )
        
        chunks = []
        if not results["ids"] or not results["ids"][0]:
            return chunks
        
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for i in range(len(ids)):
            meta = metas[i] or {}
            imgs_in_chunk = meta.pop("imgs_in_chunk", None)
            if imgs_in_chunk:
                imgs_in_chunk = ast.literal_eval(imgs_in_chunk)
            meta["chroma_score"] = distances[i]
            chunks.append(
                Chunk(
                    id=ids[i],
                    article_id=meta.pop("article_id", "unknown"),
                    role=meta.pop("role", "body"),
                    modality=meta.pop("modality", "text"),
                    domain=meta.pop("domain", None),
                    field=meta.pop("field", None),
                    content=docs[i],
                    metadata=meta,
                    images_in_chunk=imgs_in_chunk
                )
            )
        return chunks
    
    def delete_by_article_id(self, article_id: str) -> None:
        self.collection.delete(where={"article_id": article_id})
        
    def show_collections(self):
        return self.client.list_collections()
    
    def delete_collection(self, name: str):
        self.client.delete_collection(name)
    