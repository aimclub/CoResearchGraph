from typing import Dict, Any

from .base_embedder import EmbeddingModel
from .embed_by_api import APIEmbeddingModel
from .embed_locally import LocalEmbeddingModel


def create_embedding_model(config: Dict[str, Any]) -> EmbeddingModel:
    """
    Example config:

    {
        "type": "api",
        "batch_size": 32
    }

    or

    {
        "type": "local",
        "batch_size": 64
    }
    """

    model_type = config.get("type", "unknown")

    if model_type == "api":
        embedder_url = config.get("url", "")
        if not embedder_url:
            raise ValueError("Embedder URL must be specified in the config")
        return APIEmbeddingModel(
            url=embedder_url,
            timeout=config.get("timeout", 120),
            batch_size=config.get("batch_size", 16),
            headers=config.get("headers"),
        )

    if model_type == "local":
        return LocalEmbeddingModel(
            model_name=config.get("model_name", "BAAI/bge-m3"),
            batch_size=config.get("batch_size", 16),
        )

    raise ValueError(f"Unknown embedding model type: {model_type}")