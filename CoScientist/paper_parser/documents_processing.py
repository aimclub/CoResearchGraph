from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os

from protollm.connectors import create_llm_connector

from CoScientist.config import settings as app_settings
from CoScientist.paper_analysis.chroma_db_operations import ChromaDBPaperStore, ExpandedSummary
from CoScientist.paper_analysis.settings import allowed_providers
from CoScientist.paper_parser.parse_and_split import (
    IMAGES_PATH,
    PAPERS_PATH,
    USE_S3,
    clean_up_html,
    clean_up_after_processing,
    html_chunking,
)
from CoScientist.paper_parser.s3_connection import S3BucketService, s3_service as default_s3_service

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
SUMMARY_LLM_URL = os.getenv("LLM__SUMMARY_URL")

process_local_store: ChromaDBPaperStore = None


def init_process(paper_store: ChromaDBPaperStore = None):
    """
    Initializes a process-local storage for papers.

    This method creates an isolated storage instance for each process,
    allowing concurrent access and modification of paper data without interference.
    This ensures data consistency and avoids race conditions when multiple processes
    are analyzing papers simultaneously.
    
    Returns:
        None
    """
    global process_local_store
    process_local_store = paper_store or ChromaDBPaperStore()
    

def clean_up_storages(embedding_storage: ChromaDBPaperStore, file_storage: S3BucketService, paper_name: str):
    try:
        embedding_storage.clean_up_collections(paper_name)
    except Exception as cleanup_error:
        print(f"Error during vector store cleanup for {paper_name}: {cleanup_error}")
    if USE_S3:
        try:
            file_storage.clean_up_by_prefix(paper_name)
        except Exception as s3_cleanup_error:
            print(f"Error during S3 cleanup for {paper_name}: {s3_cleanup_error}")


def process_single_document(folder_path: Path, s3_service: S3BucketService, s3_prefix: str = None):
    """
    Processes a single document (paper) from a given folder path.

    This method extracts text from an HTML representation of a scientific paper, cleans and structures the content,
    and then prepares it for efficient knowledge retrieval by storing it in a vector database (ChromaDB). This
    involves summarizing the paper, breaking it down into smaller chunks, and indexing associated images as text.

    Args:
        folder_path (Path): The path to the folder containing the paper's HTML and PDF files.

    Returns:
        None
    """
    paper_name = folder_path.name
    paper_name_to_load = Path(paper_name + ".pdf")
    parsed_file_path = Path(folder_path, paper_name + ".html")
    with open(parsed_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    try:
        print("Checking for documents in ChromaDB and files in S3...")
        clean_up_storages(process_local_store, s3_service, paper_name)
        print(f"Starting post-processing paper: {paper_name}")
        if USE_S3:
            s3_paper_name = f"{s3_prefix}/{paper_name}" if s3_prefix else paper_name
            parsed_paper, mapping = clean_up_html(folder_path, paper_name, text, s3_service, s3_paper_name)
        else:
            parsed_paper, mapping = clean_up_html(folder_path, paper_name, text)
        print(f"Finished post-processing paper: {paper_name}")
        
        llm = create_llm_connector(SUMMARY_LLM_URL, extra_body={"provider": {"only": allowed_providers}})
        struct_llm = llm.with_structured_output(schema=ExpandedSummary)
        paper_summary = process_local_store._generate_expanded_summary(parsed_paper, struct_llm)
        
        documents = html_chunking(parsed_paper, paper_name, paper_summary)
        
        print(f"Starting loading paper: {paper_name}")
        process_local_store.add_paper_summary_to_db(str(paper_name_to_load), parsed_paper, paper_summary)
        process_local_store.store_text_chunks_in_chromadb(documents)
        process_local_store.store_images_in_chromadb_txt_format(str(folder_path), str(paper_name_to_load), mapping)
        print(f"Finished loading paper: {paper_name}")
        if USE_S3:
            clean_up_after_processing(folder_path)
    except Exception as e:
        print(f"Error in {paper_name}: {str(e)}")
        print(f"Cleaning up data for {paper_name}...")
        clean_up_storages(process_local_store, s3_service, paper_name)
        print(f"Cleanup completed for {paper_name}")


def process_all_documents(base_dir: Path,
                          s3_service: S3BucketService | None = None,
                          s3_prefix: str = None,
                          paper_store: ChromaDBPaperStore | None = None):
    """
    Processes documents within subdirectories of a given base directory in parallel.

    This method identifies subdirectories within the provided base directory and processes each one concurrently
    using a thread pool. This allows for faster processing of large collections of documents.

    Args:
        base_dir: The base directory containing the subdirectories, each representing a document.

    Returns:
        None
    """
    paper_store = paper_store or ChromaDBPaperStore()
    s3_service = s3_service or default_s3_service
    folders = [d for d in base_dir.iterdir() if d.is_dir()]
    with ThreadPoolExecutor(max_workers=2, initializer=init_process(paper_store)) as pool:
        pool.map(lambda folder: process_single_document(folder, s3_service, s3_prefix), folders)


if __name__ == "__main__":

    p_path = PAPERS_PATH
    res_path = IMAGES_PATH
    
    p_store = ChromaDBPaperStore()
    p_store.run_marker_pdf(p_path, res_path)
    del p_store
    process_all_documents(Path(res_path))
    
    # p_store.clean_up_collections("paper-filename")  # pass filename without .pdf