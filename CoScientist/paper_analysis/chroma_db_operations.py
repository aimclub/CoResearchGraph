import base64
import logging
import os
import uuid
from pathlib import Path

import chromadb
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.models import Collection
from chromadb.utils.data_loaders import ImageLoader
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage
from protollm.connectors import create_llm_connector
from pydantic import BaseModel, Field, model_validator
import requests

from CoScientist.chemical_utils.chemical_functions import extract_molecules_from_figure, extract_reactions_from_figure
from CoScientist.paper_analysis.prompts import summarisation_prompt
from CoScientist.paper_analysis.research_taxonomy import (
    ResearchDomain,
    format_domain_subdomain_mapping_for_prompt,
    get_sub_domains_for_domain,
)
from CoScientist.paper_analysis.settings import allowed_providers
from CoScientist.paper_analysis.settings import settings as default_settings
from CoScientist.paper_parser.utils import load_image_as_binary

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DATA_LOADER = ImageLoader()
CHROMA_DB_PATH = os.getenv("STORAGE__CHROMA_STORAGE")

VISION_LLM_URL = os.getenv("LLM__VISION_URL")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOMAIN_SUBDOMAIN_MAPPING = format_domain_subdomain_mapping_for_prompt()


class ExpandedSummary(BaseModel):
    """
    Expanded version of paper's summary.
    """
    paper_summary: str = Field(description="Summary of the paper.")
    paper_title: str = Field(
        description="Title of the paper. If the title is not explicitly specified, use the default value - 'NO TITLE'"
    )
    publication_year: int = Field(
        description="Year of publication of the paper. If the publication year is not explicitly specified, use the default"
            " value - 9999."
    )
    paper_authors: str = Field(
        description="Authors of the paper. If the authors are not explicitly specified, use the default value - 'NO AUTHORS'"
    )
    publication_source: str = Field(
        description="Source where the paper was published. If the source is not explicitly specified, use the default value - 'NO SOURCE'"
    )
    research_domain: ResearchDomain = Field(
        description="Research domain of the paper. Must be one of the predefined values."
        " If the area has no match in the predefined list or is hard to determine, use the default value - 'Other'"
    )
    research_sub_domain: str = Field(
        description="Research sub-domain of the paper. Should be one of the predefined sub-domains corresponding to the research domain."
        " Must match the selected research_domain based on this mapping:\n"
        f"{DOMAIN_SUBDOMAIN_MAPPING}\n"
        " If the sub-domain has no match in the predefined list or is hard to determine, use the default value - 'Other'"
    )

    @model_validator(mode="after")
    def validate_domain_sub_domain_pair(self):
        allowed_sub_domains = get_sub_domains_for_domain(self.research_domain)
        if self.research_sub_domain not in allowed_sub_domains:
            raise ValueError(
                "research_sub_domain must belong to selected research_domain"
            )
        return self


class CustomEmbeddingFunction(EmbeddingFunction):
    """
    Creates embeddings from text using a custom function.

    This class provides a way to generate embeddings for text data using a
    user-defined function. It takes the embedding function as a constructor argument.

    Attributes:
        - embedding_function: The function used to generate embeddings.
    """

    def __call__(self, texts: Documents) -> Embeddings:
        """
        Retrieves embeddings for a list of documents using a ChromaDBPaperStore.
        
        This method transforms text into numerical vector representations (embeddings). These embeddings capture the
        semantic meaning of the documents, allowing for efficient comparison and retrieval of relevant information.

        Args:
            self: The instance of the class.
            texts: The documents to retrieve embeddings for (list of strings).

        Returns:
            Embeddings: The embeddings for the input documents (list of lists of floats).
        """
        embeddings = ChromaDBPaperStore.get_embeddings(texts)
        return embeddings


class ChromaClient:
    """
    A client for interacting with a Chroma database.

    This class provides methods to manage Chroma collections, including creating, querying, and deleting them. It
    abstracts the underlying ChromaDB client for easier use.

    Attributes:
        - client: The ChromaDB client instance.
        - collection_name: The name of the Chroma collection to use.
        - embedding_function: The embedding function used for vectorizing data.
    """

    def __init__(self):
        """
        Initializes the ChromaDB client.

        Connects to a ChromaDB instance to enable storage and retrieval of scientific paper data for question
        answering. Initializes the ChromaDB client using the host, port, and reset settings from the default settings.

        Returns:
            None
        """
        # self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.client = chromadb.HttpClient(
            host=default_settings.chroma_host,
            port=default_settings.chroma_port,
            settings=chromadb.Settings(allow_reset=default_settings.allow_reset),
        )

    def get_or_create_chroma_collection(
        self,
        collection: str,
        embedding_function: EmbeddingFunction[Documents] | None = None,
    ) -> Collection:
        """
        Gets or creates a Chroma collection.

        This method ensures a Chroma collection exists for storing and retrieving document data. It prioritizes
        retrieving an existing collection by name. If a collection with the given name doesn't exist, it creates a new
        one, configured with the specified embedding function and a default data loader. This enables efficient storage
        and search of scientific documents.

        Args:
            collection (str): The name of the Chroma collection to retrieve or create.
            embedding_function (EmbeddingFunction[Documents] | None, optional): An optional embedding function to use
                for the collection. If None, the default embedding function is used. Defaults to None.

        Returns:
            Collection: The Chroma collection.
        """
        return self.client.get_or_create_collection(
            name=collection,
            embedding_function=embedding_function,
            data_loader=DATA_LOADER,
        )

    @staticmethod
    def query_chromadb(
        collection: Collection,
        query_text: str,
        metadata_filter: dict = None,
        chunk_num: int = 3,
    ) -> dict:
        """
        Queries a ChromaDB collection to find relevant information based on a text query.

        Args:
            collection: The ChromaDB collection to query.
            query_text: The text query to perform.  This is the information the user is seeking.
            metadata_filter: Optional dictionary to filter results based on metadata. Defaults to None.
            chunk_num: The number of results to return. Determines how many of the most relevant documents will be
                returned. Defaults to 3.

        Returns:
            dict: A dictionary containing the query results, including:
                'documents' - the text of the retrieved documents;
                'metadatas' - the metadata associated with each document;
                'distances' - a measure of similarity between the query and each document.
        """
        return collection.query(
            query_texts=[query_text],
            n_results=chunk_num,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )
        
    @staticmethod
    def update_chroma_collection(collection: Collection, ids: list[str], metadata: dict):
        """
        Updates a Chroma collection with new metadata.

        Args:
            collection: The Chroma collection to update.
            ids: The IDs of the documents to update.
            metadata: The new metadata to add to the documents.
        """
        collection.update(ids=ids, metadatas=metadata)
        return collection.get(ids=ids)
    
    def delete_collection(self, name: str):
        """
        Deletes a collection and its associated data from the database.

        Args:
            name (str): The name of the collection to delete.

        Returns:
            None
        """
        self.client.delete_collection(name)
        
    def show_collections(self):
        """
        Retrieves the names of all collections stored within the connected Chroma database.

        Returns:
            list: A list of strings, where each string is the name of a collection.
        """
        return self.client.list_collections()


class ChromaDBPaperStore:
    """
    A class for storing and searching paper data using ChromaDB.

    This class provides functionality to store text chunks and images (converted to text) from research papers in a
    ChromaDB vector database and perform similarity searches to retrieve relevant context. It leverages embeddings
    for semantic search and optionally uses a reranker to improve search results.

    Attributes:
        - client: ChromaDB client instance.
        - collection_name: Name of the ChromaDB collection.
        - llm_name: Name of the Large Language Model.
        - embedding_function: Function for generating embeddings.
        - reranker: Reranker model for refining search results.
    """

    def __init__(self,
                 sum_collection_name: str | None = None,
                 txt_collection_name: str | None = None,
                 img_collection_name: str | None = None):
        """
        Initializes the multimodal retriever.

        This constructor establishes connections to the language model and Chroma database,
        and prepares collections for storing and retrieving information from summaries, texts,
        and images extracted from scientific papers. This setup enables efficient semantic
        search and question answering over the paper content.

        Initializes the following class fields:
            llm_url (str): The URL for the Large Language Model (LLM). Defaults to VISION_LLM_URL.
            client (ChromaClient): An instance of ChromaClient for interacting with the Chroma database.
            sum_collection_name (str): The name of the Chroma collection for summaries.
            txt_collection_name (str): The name of the Chroma collection for texts.
            img_collection_name (str): The name of the Chroma collection for images.
            sum_chunk_num (int): The number of chunks for summaries. Defaults to 15.
            final_sum_chunk_num (int): The number of chunks for final summaries. Defaults to 3.
            txt_chunk_num (int): The number of chunks for texts. Defaults to 15.
            img_chunk_num (int): The number of chunks for images. Defaults to 2.
            sum_collection (ChromaCollection): The Chroma collection for summaries.
            txt_collection (ChromaCollection): The Chroma collection for texts.
            img_collection (ChromaCollection): The Chroma collection for images.
            workers (int): The number of worker threads to use. Defaults to 2.

        Returns:
            None
        """
        self.llm_url = VISION_LLM_URL

        self.client = ChromaClient()

        self.sum_collection_name = sum_collection_name or os.getenv("COLLECTIONS__SUMMARIES")
        self.txt_collection_name = txt_collection_name or os.getenv("COLLECTIONS__TEXTS")
        self.img_collection_name = img_collection_name or os.getenv("COLLECTIONS__IMAGES")

        self.sum_chunk_num = 15
        self.final_sum_chunk_num = 3
        self.txt_chunk_num = 15
        self.img_chunk_num = 2

        self.sum_collection = self.client.get_or_create_chroma_collection(
            self.sum_collection_name, CustomEmbeddingFunction()
        )
        self.txt_collection = self.client.get_or_create_chroma_collection(
            self.txt_collection_name, CustomEmbeddingFunction()
        )
        self.img_collection = self.client.get_or_create_chroma_collection(
            self.img_collection_name, CustomEmbeddingFunction()
        )
        self.workers = 1

    @staticmethod
    def _image_to_base64(image_path: str) -> str:
        """
        Encodes an image file to a base64 string representation.

        This is needed to store images alongside paper data in a format suitable for vector database embedding and
        retrieval. Converting the image to a base64 string allows it to be included as text within the document
        representation.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The base64 encoded string representation of the image.
        """
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_string

    def _image_to_text(self, image_path: str) -> str:
        """
        Extracts a concise textual description of an image, focusing on its core content as it relates to scientific
        papers.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: A succinct text description of the image, geared towards understanding its role within a scientific
                context.
        """
        sys_prompt = (
            "This is an image from a scientific paper in chemistry. "
            "Write a short but succinct description of the image that reflects its essence."
            "Be as concise as possible. "
            "Only use data from image, do NOT make anything up."
        )
        model = create_llm_connector(
            self.llm_url, temperature=0.015, top_p=0.95, extra_body={"provider": {"only": allowed_providers}}
        )
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": sys_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._image_to_base64(image_path)}"
                        },
                    },
                ],
            )
        ]
        res = model.invoke(messages)
        return res.content

    def store_text_chunks_in_chromadb(self, content: list, window_size: int = 15) -> None:
        """
        Stores text chunks into ChromaDB for efficient retrieval.
            
        The method transforms the input text chunks into numerical embeddings and stores them, along with the
        original text and associated metadata, into a ChromaDB collection. This allows for semantic search and
        retrieval of relevant information based on the content of the scientific papers. Unique IDs are assigned
        to each chunk to ensure proper indexing within the database.

        Args:
            content: A list of text chunks to be stored.
            window_size (int): Window size for simultaneous texts processing. Defaults to 15.

        Returns:
            None
        """
        chunks_num = len(content)
        if chunks_num  > window_size:
            embeddings = []
            cuts = int(chunks_num / window_size)
            for i in range(cuts + 1):
                embeddings += self.get_embeddings(
                    [text_chunk.page_content for text_chunk in content[i * window_size:(i + 1) * window_size]]
                )
        else:
            embeddings = self.get_embeddings([text_chunk.page_content for text_chunk in content])

        self.txt_collection.add(
            ids=[str(uuid.uuid4()) for _ in range(len(content))],
            documents=[text_chunk.page_content for text_chunk in content],
            embeddings=embeddings,
            metadatas=[{"type": "text", **text_chunk.metadata} for text_chunk in content]
        )

    def store_images_in_chromadb_txt_format(
            self, image_dir: str, paper_name: str, url_mapping: dict, window_size: int = 15
    ) -> None:
        """
        Stores images from a directory in ChromaDB in text format.

        Reads images from the given directory, generates text descriptions for each image,
        and stores these descriptions along with associated metadata in a ChromaDB collection.
        This allows for semantic search and retrieval of images based on their content.

        Args:
            image_dir (str): The path to the directory containing the images.
            paper_name (str): The name of the paper associated with the images, used for metadata.
            url_mapping (dict): Mapping between local image paths and storage paths
            window_size (int): Window size for simultaneous texts processing. Defaults to 15.

        Returns:
            None
        """
        image_descriptions = []
        image_paths = []
        image_counter = 0
        valid_paths = list(url_mapping.keys())

        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = (Path(image_dir) / filename).as_posix()
                if img_path in valid_paths:
                    image_descriptions.append(self._image_to_text(img_path))
                    image_paths.append(url_mapping[img_path])
                    image_counter += 1

        if image_counter > window_size:
            embeddings = []
            cuts = int(image_counter / window_size)
            for i in range(cuts + 1):
                embeddings += self.get_embeddings(image_descriptions[i * window_size:(i + 1) * window_size])
        elif 0 < image_counter <= window_size:
            embeddings = self.get_embeddings(image_descriptions)
        else:
            return
        self.img_collection.add(
            ids=[str(uuid.uuid4()) for _ in range(image_counter)],
            documents=image_descriptions,
            embeddings=embeddings,
            metadatas=[
                {"type": "image", "source": paper_name, "image_path": img_path} for img_path in image_paths
            ]
        )


    def search_for_papers(self,
                          query: str,
                          chunks_num: int = None,
                          final_chunks_num: int = None,
                          meta_filter: dict = None) -> dict:
        """
        Retrieves relevant papers from the database based on a user's query.

        This method performs an initial search to identify potentially relevant papers
        and then refines the results using a reranker to improve accuracy and relevance.
        The final results are returned as a list of paper sources.

        Args:
            query (str): The search query string.
            chunks_num (int, optional): The number of chunks to retrieve initially.
                Defaults to self.sum_chunk_num if None.
            final_chunks_num (int, optional): The number of final chunks to return.
                Defaults to self.final_sum_chunk_num if None.
            meta_filter (dict, optional): Filters for more subtle vector storage search

        Returns:
            dict: A dictionary containing a list of paper sources under the key 'answer'.
        """
        chunks_num = chunks_num if chunks_num else self.sum_chunk_num
        final_chunks_num = final_chunks_num if final_chunks_num else self.final_sum_chunk_num
        raw_docs = self.client.query_chromadb(
            self.sum_collection, query, chunk_num=chunks_num, metadata_filter=meta_filter
        )
        docs = self.search_with_reranker(query, raw_docs, top_k=final_chunks_num)
        res = {doc[2]["source"]: {"title": doc[2]["paper_title"], "year": doc[2]["publication_year"]} for doc in docs}
        return res

    def retrieve_context(
            self, query: str, relevant_papers: dict = None, meta_filter: dict = None
    ) -> tuple[list, dict, dict]:
        """
        Retrieves relevant information from text and images associated with scientific papers based on a user query.

        This method aims to identify and extract the most pertinent data from a collection of scientific documents,
        facilitating quick access to key insights related to a specific research question. It first identifies relevant
        papers and then utilizes vector similarity search to find corresponding text and image chunks.

        Args:
            query (str): The search query used to identify relevant information.
            relevant_papers (list, optional): A list of pre-identified relevant papers. Defaults to None, in which case
                a search for relevant papers is initiated.
            meta_filter (dict, optional): A dictionary of metadata filters to apply during the search.

        Returns:
            tuple[list, dict]: A tuple containing the retrieved text and image context.
                - text_context (list): A list of text chunks deemed most relevant to the query.
                - image_context (dict): A dictionary containing image data associated with the query.
        """
        if not relevant_papers:
            relevant_papers = self.search_for_papers(query, meta_filter=meta_filter)

        raw_text_context = self.client.query_chromadb(
            self.txt_collection,
            query,
            {"source": {"$in": list(relevant_papers.keys())}},
            self.txt_chunk_num,
        )
        image_context = self.client.query_chromadb(
            self.img_collection,
            query,
            {"source": {"$in": list(relevant_papers.keys())}},
            self.img_chunk_num,
        )

        text_context = self.search_with_reranker(query, raw_text_context, top_k=5)

        # Add title and year to metadata
        for cont in text_context:
            cont[2]['title'] = relevant_papers[cont[2]['source']]['title']
            cont[2]['year'] = relevant_papers[cont[2]['source']]['year']
        for image in image_context['metadatas'][0]:
            image['title'] = relevant_papers[image['source']]['title']
            image['year'] = relevant_papers[image['source']]['year']

        return text_context, image_context

    @staticmethod
    def get_molecule_and_reactions_data(img: dict):
        image_bytes = load_image_as_binary(img["image_path"])
        molecules_res = extract_molecules_from_figure(image_bytes)
        img["domain_metadata.molecules"] = str(molecules_res.get("data"))
        reactions_res = extract_reactions_from_figure(image_bytes)
        img["domain_metadata.reactions"] = str(reactions_res.get("data"))

    def get_image_data_for_chem(self, file_path: str) -> dict:
        image_data = self.client.query_chromadb(
            self.img_collection,
            "",
            {"image_path": file_path}
        )
        img = image_data["metadatas"][0][0]
        img_id = image_data["ids"][0][0]

        if "domain_metadata.molecules" not in img or "domain_metadata.reactions" not in img:
            self.get_molecule_and_reactions_data(img)
            self.img_collection.update(ids=[img_id], metadatas=[img])
        return img
    
    def search_with_reranker(
            self, query: str, initial_results: dict, top_k: int = 1
    ) -> list[tuple[str, str, dict, float]]:
        """
        Refines initial search results by assessing the relevance of each document to the query.

        Args:
            query (str): The search query string.
            initial_results (dict): A dictionary containing the initial search results with keys 'documents',
                'metadatas', and 'ids'. Each key maps to a list of corresponding values.
            top_k: The number of top results to return after reranking. Defaults to 1.

        Returns:
            list[tuple[str, str, dict, float]]: A list of tuples, where each tuple contains the document ID,
                the document text, its metadata, and the reranking score. The list is sorted by the reranking score
                in descending order, and only the top_k results are included.
        """
        metadatas = initial_results['metadatas'][0]
        documents = initial_results["documents"][0]
        ids = initial_results["ids"][0]

        pairs = [[query, doc.replace("passage: ", "")] for doc in documents]
        
        rerank_scores = self.rerank(pairs)

        scored_docs = list(zip(ids, documents, metadatas, rerank_scores))
        scored_docs.sort(key=lambda x: x[3], reverse=True)

        return scored_docs[:top_k]

    def _generate_expanded_summary(self, parsed_paper: str, llm) -> ExpandedSummary:
        """
        Generates an expanded summary of a paper using a language model.

        This method takes parsed paper content and uses an LLM to extract and structure
        key information including summary, title, authors, publication year, and source.

        Args:
            parsed_paper (str): The text content of the parsed paper.
            llm: The language model used to generate the summary.

        Returns:
            ExpandedSummary: An object containing the paper's structured summary information.
        """
        messages = [HumanMessage(content=summarisation_prompt + parsed_paper)]

        for attempt in range(3):
            try:
                expanded_summary: ExpandedSummary = llm.invoke(messages)
                return expanded_summary
            except Exception as e:
                last_error = e
                messages.append(
                        HumanMessage(
                            content="Previous response was invalid JSON. Respond with ONLY valid JSON."
                        )
                    )
                continue
    
        raise RuntimeError(
            f"Failed to get valid structured response after 3 attempts. "
            f"Last error: {last_error}"
        ) from last_error


    def add_paper_summary_to_db(self, paper_name: str, parsed_paper: str, expanded_summary: ExpandedSummary) -> None:
        """
        Adds a paper summary to the document collection for efficient information retrieval.

        This method processes a paper's content by generating an expanded summary using a language model. It then
        stores this summary, along with relevant metadata, in a vector database to enable semantic search and
        question answering.

        Args:
            paper_name (str): The name of the scientific paper.
            parsed_paper (str): The text content of the parsed paper.
            llm: The language model used to generate the summary.

        Returns:
            None
        """
        doc = Document(
            page_content=expanded_summary.paper_summary,
            metadata={
                "source": paper_name,
                "paper_title": expanded_summary.paper_title,
                "publication_year": expanded_summary.publication_year,
                "paper_authors": expanded_summary.paper_authors,
                "publication_source": expanded_summary.publication_source,
                "research_domain": expanded_summary.research_domain,
                "research_sub_domain": expanded_summary.research_sub_domain,
            }
        )
        embedding = self.get_embeddings([doc.page_content])
        self.sum_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[doc.page_content],
            embeddings=embedding,
            metadatas=[{"type": "text", **doc.metadata}]
        )
        print(f"Summary loaded for: {paper_name}")

    def run_marker_pdf(self, path_to_docs, out_path) -> None:
        """
        Executes a shell script to extract marker data from a PDF file.

        This method utilizes a shell script to parse markers from a given PDF,
        converting the paper content into a structured format suitable for analysis
        and storage. The number of processing workers is configurable to manage
        performance based on system resources.

        Args:
            path_to_docs (str): The path to the input PDF file.
            out_path (str): The path to save the output file containing marker data.

        Returns:
            None
        """
        try:
            os.system(
                " ".join(
                    [
                        "sh",
                        str(Path("CoScientist/paper_analysis/marker_parsing.sh")),
                        str(path_to_docs),
                        str(out_path),
                        str(self.workers)
                    ]
                )
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    @staticmethod
    def get_embeddings(texts: list[str]) -> list[np.ndarray]:
        """
        Retrieves numerical representations of text for semantic understanding and comparison.

        Args:
            texts: A list of strings to be converted into embeddings.

        Returns:
            list[np.ndarray]: A list of embedding vectors, one corresponding to each input text.

        Raises:
            Exception: If the embedding service is unavailable or returns an error.
        """
        embedding_service_url = "http://" + default_settings.embedding_host + ":"\
                                + str(default_settings.embedding_port)\
                                + default_settings.embedding_endpoint
        try:
            response = requests.post(
                embedding_service_url,
                json=texts,
                timeout=1000
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            logger.error(f"Embedding service error: {str(e)}")
            raise
    
    @staticmethod
    def rerank(pairs: list[list[str]]) -> list[float]:
        """
        Reranks a list of document pairs to determine their relevance.

        This method sends the pairs to a dedicated reranker service and retrieves scores
        indicating the relevance of each pair. This helps to refine search results
        and prioritize the most pertinent documents.

        Args:
            pairs: A list of pairs to rerank. Each pair is a list of strings representing documents.

        Returns:
            list[float]: A list of reranking scores, where each score corresponds to the input pair.
                         Higher scores indicate greater relevance.

        Raises:
            Exception: If the reranker service is unavailable or returns an error.
        """
        reranker_service_url = "http://" + default_settings.reranker_host + ":" \
                                + str(default_settings.reranker_port) \
                                + default_settings.reranker_endpoint
        try:
            response = requests.post(
                reranker_service_url,
                json=pairs,
                timeout=1000
            )
            response.raise_for_status()
            return response.json()["scores"]
        except Exception as e:
            logger.error(f"Reranker service error: {str(e)}")
            raise
        
    def clean_up_collections(self, paper_name: str):
        """
        Deletes all documents related to one article from all collections.
        
        Args:
            paper_name: A name of the article file

        Returns:
            None
        """
        for collection in [self.sum_collection, self.txt_collection, self.img_collection]:
            ids_to_delete = self.client.query_chromadb(
                collection,
                "",
                {"source": {"$in": [f"{paper_name}.pdf"]}},
                10000
            )["ids"][0]
            if len(ids_to_delete) > 0:
                collection.delete(ids=ids_to_delete)
                print(f"Deleted {len(ids_to_delete)} documents from {collection.name} collection")
            else:
                print(f"No documents to delete for {paper_name}.pdf in {collection.name} collection")