import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

model: SentenceTransformer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
    Loads the embedding model and performs a warmup to ensure fast response times for subsequent queries.
    
    Args:
        app: The FastAPI application instance.
    
    Initializes the following class fields:
        model: The SentenceTransformer model used for converting text into numerical vector representations.
    
    Returns:
        None
    '''
    global model
    model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
    model.encode(["warmup"])
    logging.info("Embedding model loaded")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/embed")
async def embed(texts: list[str]):
    """
    Generates numerical representations of input texts.
    
    Args:
        texts (list[str]): A list of strings to be converted into embeddings.
    
    Returns:
        dict: A dictionary containing the generated embeddings. The dictionary
            has a single key, "embeddings", which maps to a list of lists,
            where each inner list represents the embedding for a corresponding
            input text.
    """
    embeddings = model.encode(texts).tolist()
    return {"embeddings": embeddings}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
