from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional

from ...domain.entities import Chunk


class VectorStore(ABC):
    
    @abstractmethod
    def upsert_chunks(
            self,
            chunks: List[Chunk],
            embeddings: List[Union[List[float], Dict[str, Any]]]
    ) -> None:
        """
        Загружает или обновляет чанки и их эмбеддинги.
        `embeddings` может быть:
          - List[float] (для простых dense моделей, как Chroma)
          - Dict[str, Any] (для мульти-векторов, как Qdrant, напр. {"dense": [...], "sparse": [...]})
        """
        pass
    
    @abstractmethod
    def search(
            self,
            query_vector: Union[List[float], Dict[str, Any]],
            limit: int = 5,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Метод для будущего Retriever'а.
        """
        pass
    
    @abstractmethod
    def delete_by_article_id(self, article_id: str) -> None:
        """
        Полезно для перезаписи или удаления статьи целиком.
        """
        pass