from typing import Dict, Any

from .base_reranker import Reranker
from .rerank_locally import LocalReranker
from .rerank_by_api import APIReranker


def create_reranker(config: Dict[str, Any]) -> Reranker:
    """
    Example config:

    {
        "type": "local",
        "batch_size": 16
    }

    or

    {
        "type": "api",
        "batch_size": 32
    }
    """

    model_type = config.get("type")
    
    if model_type == "api":
        reranker_url = config.get("url", "")
        if not reranker_url:
            raise ValueError("Reranker URL must be specified in the config")
        return APIReranker(
            url=reranker_url,
            timeout=config.get("timeout", 120),
            batch_size=config.get("batch_size", 16),
            headers=config.get("headers"),
        )
        

    if model_type == "local":
        return LocalReranker(
            model_name=config.get(
                "model_name",
                "Alibaba-NLP/gte-multilingual-reranker-base",
            ),
            max_length=config.get("max_length", 2048),
            batch_size=config.get("batch_size", 16),
            trust_remote_code=config.get("trust_remote_code", True),
        )

    raise ValueError(f"Unknown reranker type: {model_type}")