from abc import ABC, abstractmethod
from typing import Iterable

from ..domain.entities import Article


class ArticleSource(ABC):
    
    @abstractmethod
    def list_articles(self) -> Iterable[Article]:
        pass
        
    @abstractmethod
    def fetch(self, article: Article) -> bytes:
        pass