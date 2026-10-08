from pathlib import Path
from typing import Optional, Dict, Any, List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ConfigDict, computed_field

from ..domain.entities import Article, Chunk, ChunkRole
from ..embeddings import *
from ..storage.artifacts.domain_s3 import S3DomainArtifactStore
from ..storage.artifacts.etl_s3 import S3ETLArtifactStore
from ..storage.state import SQLiteStateManager, PostgreSQLStateManager
from ..storage.vector.base import VectorStore


class ETLContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    article: Article

    raw_data: Optional[bytes] = None
    parsed_representation: Optional[str] = None
    chunks: Dict[ChunkRole, List[Chunk]] = Field(default_factory=dict)
    embeddings: Dict[ChunkRole, Dict[str, List[List[float]] | List[str]]] = Field(default_factory=dict)
    
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    
    state_manager: SQLiteStateManager | PostgreSQLStateManager
    artifact_store: S3ETLArtifactStore
    public_store: S3DomainArtifactStore
    vector_store: VectorStore
    
    llm: ChatOpenAI
    embedding_model: EmbeddingModel
    
    @computed_field
    @property
    def processed_papers_path(self) -> Path:
        source = Path(self.article.source_ref)
        if source.is_file():
            articles_dir = source.parent
        else:
            articles_dir = source
        processed_dir = articles_dir.parent / "processed"
        processed_dir.mkdir(exist_ok=True)
        return processed_dir
