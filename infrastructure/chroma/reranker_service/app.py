import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from sentence_transformers import CrossEncoder

model: CrossEncoder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes and loads the cross-encoder model for reranking search results, 
    performing a warm-up prediction to optimize performance.
    
    Args:
        app (FastAPI): The FastAPI application instance.
    
    Yields:
        None
    
    Returns:
        None
    
    Fields initialized:
        model: The CrossEncoder model for reranking.
    """
    global model
    model = CrossEncoder(
        "Alibaba-NLP/gte-multilingual-reranker-base",
        max_length=2048,
        trust_remote_code=True
    )
    model.predict([["warmup", "query"]])  # Прогрев модели
    logging.info("Reranker model loaded")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/rerank")
async def rerank(pairs: list[tuple[str, str]]):
    """
    Predicts the relevance score for pairs of strings using a pre-trained model.
    
    Args:
        pairs: A list of tuples, where each tuple contains two strings to be scored.
    
    Returns:
        dict: A dictionary containing a list of relevance scores, 
            where each score corresponds to the input pair.
    """
    scores = model.predict(pairs).tolist()
    return {"scores": scores}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
