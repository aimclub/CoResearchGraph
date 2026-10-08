import pytest
import os
import uuid
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from CoScientist.chemical_utils.chemical_functions import extract_molecules_from_pdf, extract_reactions_from_pdf, remove_keys
from CoScientist.paper_analysis.chroma_db_operations import ChromaClient, ChromaDBPaperStore
from CoScientist.paper_parser.documents_processing import process_all_documents
from CoScientist.paper_analysis.question_processing import process_question, simple_query_llm
from CoScientist.paper_parser.s3_connection import s3_service
from CoScientist.paper_analysis.prompts import sys_prompt, explore_my_papers_prompt

load_dotenv(find_dotenv())

PAPERS_STORAGE_PATH = Path("tests/integration/data")
PARSE_RESULTS_PATH = Path("tests/integration/parse_results")
QUESTION_DB = "How are UV-curable monomeric or oligomeric urethane acrylates used in stereolithography?"
QUESTION_USER = "Who is the author of the article?"
VISION_LLM_URL = os.getenv("LLM__VISION_URL")

def _unique(name: str) -> str:
    """Generates a unique collection or bucket name by appending a random 8-character UUID suffix."""
    return f"{name}_{uuid.uuid4().hex[:8]}"

@pytest.fixture(scope="class")
def client() -> ChromaClient:
    """Fixture providing a ChromaDB client instance."""
    return ChromaClient()

@pytest.fixture(scope="class")
def s3_prefix() -> dict[str, str]:
    """Fixture providing unique s3 prefix."""
    return _unique("tests")

@pytest.fixture(scope="class")
def collection_names() -> dict[str, str]:
    """Fixture providing unique names for test ChromaDB collections."""
    return {
        "sum": _unique("tests_paper_summaries_img2txt"),
        "txt": _unique("tests_text_context_img2txt"),
        "img": _unique("tests_image_context"),
    }
    
@pytest.fixture(scope="class")
def paper_store(collection_names: dict[str, str]) -> ChromaDBPaperStore:
    """Fixture providing a ChromaDBPpaperStore instance."""
    return ChromaDBPaperStore(
            sum_collection_name=collection_names["sum"],
            txt_collection_name=collection_names["txt"],
            img_collection_name=collection_names["img"]
        )

@pytest.fixture(scope="class", autouse=True)    
def session_cleanup(client: ChromaClient,
                    collection_names: dict[str, str],
                    s3_prefix: str):
    """
    Fixture that cleans up all temporary files after tests, including:
    - created Chroma collections
    - created objects in S3 bucket
    - downloaded picture from S3
    - .last_activity file in the data folder
    """
    yield
    
    client.delete_collection(collection_names["sum"])
    client.delete_collection(collection_names["txt"])
    client.delete_collection(collection_names["img"])
    collections_list = [c.name for c in client.show_collections()]
    if any(name in collections_list for name in collection_names.values()):
        raise AssertionError("Some collections are still present in Chroma!")
    
    objects_to_delete = s3_service.list_objects(prefix=f"{s3_prefix}/test_paper")
    for obj in objects_to_delete:
        s3_service.delete_file_object(f"{s3_prefix}/test_paper", obj.split("/")[-1])
    if s3_service.list_objects(prefix=f"{s3_prefix}/test_paper") != []:
        raise AssertionError("Some test objects are still present in S3!")
    
    picture_path = f"{PAPERS_STORAGE_PATH}/_page_0_Figure_10.jpeg"
    if os.path.exists(picture_path):
        os.remove(picture_path)
        
    activity_path = "tests/integration/data/.last_activity"
    if os.path.exists(activity_path):
        os.remove(activity_path)


@pytest.mark.incremental
class TestPaperAnalysis:
    """
    Integration test suite for ChromaDB collections, PDF upload, question answering,
    and user article querying.
    """
    def test_01_create_collection(self,
                                  client: ChromaClient,
                                  paper_store: ChromaDBPaperStore, # ensures collections are created
                                  collection_names: dict[str, str]) -> None:
        """Test that ChromaDB collections can be created successfully."""
        collections_list = [c.name for c in client.show_collections()]
        assert collection_names["sum"] in collections_list
        assert collection_names["txt"] in collections_list
        assert collection_names["img"] in collections_list
    
    def test_02_upload_pdf(self,
                           paper_store: ChromaDBPaperStore,
                           s3_prefix: str,
                           client: ChromaClient,
                           collection_names: dict[str, str]
                           ) -> None:
        """Test that the PDF upload script runs without errors.""" 
        paper_store.run_marker_pdf(PAPERS_STORAGE_PATH, PARSE_RESULTS_PATH)
        assert os.path.isfile(os.path.join(PARSE_RESULTS_PATH, f"test_paper/test_paper.html"))
        process_all_documents(Path(PARSE_RESULTS_PATH), s3_service, s3_prefix, paper_store)
        assert f"{s3_prefix}/test_paper/test_paper_processed.html" in s3_service.list_objects(prefix=f"{s3_prefix}/test_paper")
        sum_collection = client.get_or_create_chroma_collection(collection_names["sum"])
        txt_collection = client.get_or_create_chroma_collection(collection_names["txt"])
        img_collection = client.get_or_create_chroma_collection(collection_names["img"])
        assert sum_collection.count() == 1
        assert txt_collection.count() == 10
        assert img_collection.count() == 11
        assert not os.listdir(PARSE_RESULTS_PATH)
    
    def test_03_query(self,
                      paper_store: ChromaDBPaperStore,
                      s3_prefix: str) -> None:
        """Test that querying the database with a predefined question returns a valid answer and metadata."""
        result = process_question(QUESTION_DB, sys_prompt, paper_store)
        assert "answer" in result
        assert "metadata" in result
        assert isinstance(result["answer"], str)
        assert result["answer"].strip() != ""
        meta = result["metadata"]
        assert isinstance(meta, dict)
        for field in ("text_context", "image_context"):
            assert field in meta
            assert meta[field] not in (None, "", [], {})
        assert len(meta["text_context"]) != 0 # chunks that LLM considered to be relevant
        assert len(meta["image_context"]) != 0 # images that LLM considered to be relevant
        s3_service.download_image_from_s3(f"{s3_prefix}/test_paper/_page_0_Figure_10.jpeg",
                                          f"{PAPERS_STORAGE_PATH}/_page_0_Figure_10.jpeg")
        assert "_page_0_Figure_10.jpeg" in os.listdir(PAPERS_STORAGE_PATH)
    
    def test_04_query_user_article(self) -> None:
        """Test that querying user-uploaded PDFs with the Vision LLM returns the right answer."""
        pdfs_dirs = [os.path.join(PAPERS_STORAGE_PATH, f)
                     for f in os.listdir(PAPERS_STORAGE_PATH)
                     if f.lower().endswith(".pdf")]
        
        img_descriptions = 'This is additional information about the reactions and molecules that are presented' \
                    'on the images in the paper. They are passed in the same order as papers themselves.' \
                    'Use them to answer the question.\n\n'
        for paper in pdfs_dirs:
            with open(paper, 'rb') as paper_bytes:
                detected_reactions = remove_keys(extract_reactions_from_pdf(paper_bytes))
                img_descriptions += f'Reactions: {str(detected_reactions)}\n'
            with open(paper, 'rb') as paper_bytes:
                detected_molecules = remove_keys(extract_molecules_from_pdf(paper_bytes))
                img_descriptions += f'Molecules: {str(detected_molecules)}\n'
            img_descriptions += '\n\n'

        result = simple_query_llm(VISION_LLM_URL, QUESTION_USER, explore_my_papers_prompt,pdfs_dirs, img_descriptions)
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert "Norbert Moszner" in result["answer"]