import requests
import chromadb
import pytest


CHROMADB_HOST = "localhost"
CHROMADB_PORT = 9941
EMBEDDING_SERVICE_HOST = "localhost"
EMBEDDING_SERVICE_PORT = 5000
RERANKER_SERVICE_HOST = "localhost"
RERANKER_SERVICE_PORT = 5001

def test_chroma():
    """
    Tests the Chroma database interaction for document storage and retrieval.
    
    This method verifies the end-to-end functionality of storing documents with associated metadata in ChromaDB 
    and then querying for those documents based on textual content. It ensures data can be reliably added, searched,
    and that the expected results are returned.
    
    Args:
        None
    
    Returns:
        None
    """
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    client.delete_collection(name="test_collection")
    collection = client.create_collection(name="test_collection")

    # Add some data to the collection
    collection.add(
        documents=["This is a test document.", "Another test document."],
        metadatas=[{"source": "test"}, {"source": "test"}],
        ids=["doc1", "doc2"]
    )

    # Query the collection
    results = collection.query(
        query_texts=["test"],
        n_results=2
    )

    # Assert that the query returned the expected results
    assert len(results["ids"]) == 2
    assert len(results["documents"]) == 2
    assert len(results["metadatas"]) == 2
    # Clean up the client
    client.delete_collection(name="test_collection")
    client.delete()

def test_embedding_service():
    """
    Tests the embedding service by sending a request with sample texts and validating the response.
    
    This method verifies the embedding service's functionality by sending a POST request containing a list of texts.
    It checks if the service returns a successful status code, contains the expected data structure ("embeddings"),
    and returns embeddings with the correct dimensions and data types. This ensures that the service is correctly 
    transforming text into numerical representations suitable for semantic search and analysis.
    
    Args:
        None
    
    Returns:
        None
    """
    response = requests.post(
        f"http://{EMBEDDING_SERVICE_HOST}:{EMBEDDING_SERVICE_PORT}/embed",
        json=[
            "This is a test text.",
            "This is another test text.",
            "Text 3",
            "Text 4"
        ]
    )
    assert response.status_code == 200
    data = response.json()
    assert list(data) == ["embeddings"]
    embeddings = data["embeddings"]

    # Validate shape
    assert isinstance(embeddings, list)
    assert len(embeddings) == 4
    assert all(isinstance(item, list) for item in embeddings)
    assert all(len(item) == 1024 for item in embeddings)
    assert all(isinstance(element, float) for item in embeddings for element in item)

def test_reranker_service():
    """
    Tests the reranker service by sending a request with sample text pairs and validating the response.
    
    This method verifies the service's ability to process text pairs and return relevance scores,
    ensuring the response is correctly formatted and contains expected data types. This helps
    confirm the reranker is functioning as intended before integration with the larger system.
    
    Args:
        None
    
    Returns:
        None
    """
    response = requests.post(
        f"http://{RERANKER_SERVICE_HOST}:{RERANKER_SERVICE_PORT}/rerank",
        json=[
            ["This is a test text.", "This is another test text."],
            ["Text 3", "Text 4"],
            ["CI/CD is good", "Roses are red, violets are blue"],
        ]
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert list(data) == ["scores"]
    # Validate shape
    assert isinstance(data["scores"], list)
    assert len(data["scores"]) == 3
    assert all(isinstance(item, float) for item in data["scores"])

