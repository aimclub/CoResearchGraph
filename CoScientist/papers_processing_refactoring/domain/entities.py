from enum import Enum
from pathlib import Path
from typing import Any, Literal, Dict, Optional, Union, Mapping

from pydantic import BaseModel, model_validator


class Article(BaseModel):
    id: str
    source_type: Literal["local", "remote"]
    source_ref: Union[str, Path]
    name: str
    domain: str = "default"
    metadata: Optional[Dict[str, Any]] = None
    
    @model_validator(mode="after")
    def validate_source_pair(self):
        if self.source_type == "local" and not isinstance(self.source_ref, Path):
            raise ValueError("local articles must use a Path for source_ref")
        if self.source_type == "remote" and isinstance(self.source_ref, Path):
            raise ValueError("remote articles must use a string URL for source_ref")
        return self


class ChunkRole(str, Enum):
    BODY = "body"
    SUMMARY = "summary"
    IMAGE_CAPTION = "image_caption"
    TABLE = "table"


class Chunk(BaseModel):
    id: str
    article_id: str
    domain: Optional[str] = None
    field: Optional[str] = None
    modality: Literal["text", "image"]
    content: str
    metadata: Optional[Mapping[str, Any]] = None
    role: str
    images_in_chunk: Optional[list[str]] = None
    

class KnowledgeDomain(BaseModel):
    name: str
    description: str
    

class ImageInfo(BaseModel):
    id: str
    file_name: str
    original_src: Any
    is_kept: bool = True
    caption: Optional[str] = None
    final_s3_url: Optional[str] = None
