# TODO: get rid of redundant functions and imports
import base64
import time
import pikepdf

from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_openai import ChatOpenAI
from protollm.connectors import create_llm_connector
from pydantic import BaseModel, Field, field_validator, model_validator
from pypdf import PdfReader, PdfWriter
from io import BytesIO

from CoScientist.paper_analysis.chroma_db_operations import ChromaDBPaperStore
from CoScientist.paper_analysis.prompts import sys_prompt, extract_query_filters_prompt
from CoScientist.paper_analysis.research_taxonomy import (
    ResearchDomain,
    get_sub_domains_for_domain,
)
from CoScientist.paper_analysis.settings import allowed_providers
from CoScientist.paper_parser.utils import convert_to_base64, prompt_func
from CoScientist.chemical_utils.chemical_functions import *
from CoScientist.paper_analysis.domain_metadata import format_domain_metadata, add_domain_metadata_to_img_info

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

VISION_LLM_URL = os.getenv("LLM__VISION_URL", "")


class ResearchDomainFilter(BaseModel):
    """A domain and the fields that must be matched within it."""

    domain: ResearchDomain = Field(description="One OpenAlex research domain")
    fields: list[str] = Field(
        description="Exactly five distinct fields that belong to this domain",
        min_length=5,
        max_length=5,
    )

    @field_validator("fields", mode="before")
    @classmethod
    def normalize_fields(cls, value):
        if value is None or isinstance(value, list):
            return value
        return [value]

    @model_validator(mode="after")
    def validate_fields(self):
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("fields must contain exactly five distinct values")

        allowed_fields = get_sub_domains_for_domain(self.domain)
        invalid_fields = [
            field for field in self.fields
            if field not in allowed_fields
        ]
        if invalid_fields:
            raise ValueError(
                "fields must belong to the selected domain: "
                f"{', '.join(invalid_fields)}"
            )
        return self


class QueryFilters(BaseModel):
    """Metadata filters extracted from a user question."""
    authors: list[str] | None = Field(
        description="Author names mentioned in the question",
        default=None
    )
    publication_year_min: int | None = Field(
        description="Minimum publication year for filtering",
        default=None
    )
    publication_year_max: int | None = Field(
        description="Maximum publication year for filtering",
        default=None
    )
    publication_year_exact: int | None = Field(
        description="Exact publication year when specified",
        default=None
    )
    source: str | None = Field(
        description="Journal or publication source name",
        default=None
    )
    domains: list[ResearchDomainFilter] = Field(
        description=(
            "Exactly two distinct research-domain selections, each with "
            "exactly five distinct fields paired with that domain."
        ),
        min_length=2,
        max_length=2,
    )

    @field_validator("authors", mode="before")
    def normalize_list_fields(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            return v
        return [v]

    @model_validator(mode="after")
    def validate_distinct_domains(self):
        domains = [selection.domain for selection in self.domains]
        if len(domains) != len(set(domains)):
            raise ValueError("domains must contain exactly two distinct values")
        return self


def extract_metadata_filters(question: str, llm_url: str, extraction_prompt: str) -> QueryFilters:
    """
    Uses LLM to extract metadata filters from user question.
    
    Args:
        question: The user's question string
        
    Returns:
        QueryFilters: Structured filters including authors, years, source, domain, and sub-domain
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            llm = create_llm_connector(
                llm_url,
                temperature=0.1
            )
            # base_url, model_name = llm_url.split(";")
            # llm = ChatOpenAI(
            #     model=model_name,
            #     base_url=base_url,
            #     api_key=os.getenv("LLM__SERVICE_KEY"),  # noqa
            #     temperature=0.1
            # )
            
            struct_llm = llm.with_structured_output(schema=QueryFilters)
            
            prompt = extraction_prompt + f"\n\nUSER QUESTION: {question}"
            
            filters: QueryFilters = struct_llm.invoke([HumanMessage(content=prompt)])  # noqa
            return filters
        except Exception as e:
            print(f"Error extracting metadata filters (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 1.5 ** attempt
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed to extract metadata filters for question: {question}")
                raise RuntimeError(
                    f"Failed to extract valid metadata filters after "
                    f"{max_retries} attempts: {e}"
                ) from e


def build_chroma_where_filter(filters: QueryFilters) -> dict | None:
    """
    Converts QueryFilters to ChromaDB where clause format.
    
    Args:
        filters: QueryFilters object with extracted metadata
        
    Returns:
        dict: ChromaDB where clause ready for collection.query(), or None if no filters
        
    Example output:
        {"paper_authors": {"$in": ["Smith"]}}
        {
            "$and": [
                {"paper_authors": {"$in": ["Smith"]}},
                {"publication_year": {"$gte": 2020}}
            ]
        }
    """
    conditions = []
    
    if filters.authors is not None:
        conditions.append({"authors": {"$in": filters.authors}})
    
    if filters.publication_year_exact is not None:
        conditions.append({"publication_year": {"$eq": filters.publication_year_exact}})
    elif filters.publication_year_min is not None or filters.publication_year_max is not None:
        year_condition = {}
        if filters.publication_year_min is not None:
            year_condition["$gte"] = filters.publication_year_min
        if filters.publication_year_max is not None:
            year_condition["$lte"] = filters.publication_year_max
        if year_condition:
            conditions.append({"publication_year": year_condition})
    
    if filters.source is not None:
        conditions.append({"source": {"$eq": filters.source}})
    
    if filters.domains:
        domain_conditions = []
        for selection in filters.domains:
            domain_condition = {"domain": {"$eq": selection.domain}}
            if selection.fields:
                domain_conditions.append({
                    "$and": [
                        domain_condition,
                        {"field": {"$in": selection.fields}},
                    ]
                })
            else:
                domain_conditions.append(domain_condition)

        conditions.append(
            domain_conditions[0]
            if len(domain_conditions) == 1
            else {"$or": domain_conditions}
        )
    
    if not conditions:
        return None
    
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def get_domain_metadata_type(filters: QueryFilters) -> str | None:
    """Return the domain-specific metadata handler relevant to the query."""
    if not filters.domains:
        return None

    if any(
        "Chemistry" in (selection.fields or [])
        for selection in filters.domains
    ):
        return "Chemistry"
    return None


def query_llm(
    model_url: str,
    question: str,
    system_prompt: str,
    txt_context: str,
    img_paths: list[str]
) -> tuple:
    """
    Queries a Large Language Model (LLM) to answer questions using provided context.

    This method constructs a query incorporating both textual and visual information, then sends it to the specified
    LLM. This allows the LLM to leverage diverse data sources for a more informed response.

    Args:
        model_url (str): The URL of the LLM model to use for querying.
        question (str): The question to be answered by the LLM.
        txt_context (str): Textual information to provide context for the question.
        img_paths (list[str]): A list of file paths to images to be used as context.

    Returns:
        tuple: A tuple containing the LLM's response content (str) and a dictionary of response metadata (dict).
    """
    llm = create_llm_connector(model_url, extra_body={"provider": {"only": allowed_providers}}, temperature=0.05)

    class ResScheme(BaseModel):
        answer: str = Field(description="The answer to the query", default="")
        explanation: str = Field(description="The logical reasoning for the answer", default="")
        chunk_explanation: str = Field(description="The explanation why the chosen chunk/chunks are relevant to the answer", default="")
        img_explanation: str = Field(description="The explanation why the chosen image/images are relevant to the answer", default="")
        relevant_text: list[int] = Field(description="A list of integers representing the relevant text chunk numbers, numeration of chunks starts with 1", default=[])
        relevant_images: list[int] = Field(description="A list of integers representing the relevant image numbers, numeration of images starts with 1", default=[])

    structured_llm = llm.with_structured_output(schema=ResScheme)

    img_context = list(map(convert_to_base64, img_paths))
    messages = [
        SystemMessage(content=system_prompt),
        prompt_func(
            {
                "text": f"USER QUESTION: {question}\n\nCONTEXT: {txt_context}",
                "image": img_context,
            }
        ),
    ]

    for attempt in range(3):
        try:
            res = structured_llm.invoke(messages)
            content = {
                'answer': res.answer,
                'explanation': res.explanation,
                'chunk_explanation': res.chunk_explanation,
                'img_explanation': res.img_explanation,
                'relevant_text': res.relevant_text,
                'relevant_images': res.relevant_images
            }
            return content
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



def simple_query_llm(
    model_url: str,
    question: str,
    system_prompt: str,
    pdfs: list,
    img_descriptions: str) -> dict:
    """
    Queries a language model with a question and a list of PDF documents to provide context for answering the question.

    Args:
        model_url (str): The URL of the language model to use for querying.
        question (str): The question to ask the language model.
        pdfs (list): A list of paths to PDF documents to provide as context.

    Returns:
        dict: A dictionary containing the answer from the language model. The dictionary has a single key, 'answer',
            which holds the answer string.
    """

    llm = create_llm_connector(model_url)

    content = []
    
    writer = PdfWriter()

    # Merge all PDFs
    for paper_pdf in pdfs:
        reader = PdfReader(paper_pdf)
        for page in reader.pages:
            writer.add_page(page)

    merged_buffer = BytesIO()
    writer.write(merged_buffer)
    merged_buffer.seek(0)
   
    # Linearize merged PDF
    clean_buffer = BytesIO()
    with pikepdf.open(merged_buffer) as pdf:
        pdf.save(clean_buffer, linearize=True)
    clean_buffer.seek(0)

    base64_pdf = base64.b64encode(clean_buffer.read()).decode("utf-8")
    paper_part = {
        "type": "file",
        "file": {
            "filename": "merged_papers.pdf",
            "file_data": f"data:application/pdf;base64,{base64_pdf}",
        },
    }
    content.append(paper_part)

    text_part = {"type": "text", "text": f"USER QUESTION: {question}\n\n{img_descriptions}"}
    content.append(text_part)
    from langchain_core.messages import HumanMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ]
    
    for attempt in range(3):
        try:
            res = llm.invoke(messages)
            return {'answer': res.content}
        except Exception as e:
            print(f"LLM query error: {str(e)}. Retrying ({attempt + 1}/3)")
            time.sleep(1.2 ** attempt)
            
    return {'answer': 'LLM invocation failed after 3 attempts.'}


def process_question(
    question: str,
    system_prompt: str,
    store: ChromaDBPaperStore) -> dict:
    """
    Processes a question by retrieving relevant text and image context from scientific papers and querying a Large Language Model (LLM) to generate an answer.

    Args:
        question (str): The input question string.

    Returns:
        dict: A dictionary containing the answer and associated metadata:
            'answer' - the answer generated by the LLM based on the provided context;
            'metadata' - a dictionary containing:
                'text_context' - the concatenated text from relevant paper chunks, including metadata;
                'image_context' - the set of image paths identified as relevant to the question;
                'metadata' - Additional metadata returned by the LLM query.
    """
    meta_filter = extract_metadata_filters(question, VISION_LLM_URL, extract_query_filters_prompt)
    meta_filter_chroma = build_chroma_where_filter(meta_filter)
    domain_metadata_type = get_domain_metadata_type(meta_filter)
    
    txt_data, img_data = store.retrieve_context(question, meta_filter=meta_filter_chroma)
    txt_context = ""
    relevant_txt_context = []
    img_paths = []

    # Combine text context
    for idx, chunk in enumerate(txt_data, start=1):
        txt_context += (
            f"{idx}. "
            + "\nChunk: "
            + chunk[1].replace("passage: ", "")
            + "\n\n"
        )
    
    # Combine images for context (from chunk text and fom DB)
    for chunk_meta in [chunk[2] for chunk in txt_data]:
        for img_path in eval(chunk_meta["imgs_in_chunk"]):

            img_info = {
                'path': img_path,
                'Source': chunk_meta['source'],
                'Paper': chunk_meta['title'],
                'Year': chunk_meta['year']
            }

            image_data = store.client.query_chromadb(
                    store.img_collection,
                    "",
                    {"image_path": img_path}
                )
            img_meta = image_data["metadatas"][0][0]
            img_info = add_domain_metadata_to_img_info(domain_metadata_type, img_meta, img_info)
            img_paths.append(img_info)
    
    for img_meta in img_data["metadatas"][0]:
        if img_meta['image_path'] not in [d['path'] for d in img_paths]:
            img_info = {
                'path': img_meta['image_path'],
                'Source': chunk_meta['source'],
                'Paper': chunk_meta['title'],
                'Year': chunk_meta['year']
            }
            image_data = store.client.query_chromadb(
                    store.img_collection,
                    "",
                    {"image_path": img_meta['image_path']}
                )
            img_meta = image_data["metadatas"][0][0]
            img_info = add_domain_metadata_to_img_info(domain_metadata_type, img_meta, img_info)
            img_paths.append(img_info)

    img_paths_list = set([d['path'] for d in img_paths])
    
    domain_metadata = format_domain_metadata(domain_metadata_type, img_paths)
    if domain_metadata != "":
        txt_context += f"Domain metadata\n{domain_metadata}\n\n"
    else:
        txt_context += "No domain metadata found for context."

    ans = query_llm(VISION_LLM_URL, question, system_prompt, txt_context, list(img_paths_list))

    # Separate relevant context
    relevant_txt_data = [txt_data[num - 1] for num in ans['relevant_text']]
    relevant_img_context = [img_paths[num - 1] for num in ans['relevant_images']]

    for idx, chunk in enumerate(relevant_txt_data, start=1):
        relevant_txt_context.append({
            'chunk': f"Chunk {idx}: \n"
                     + chunk[1].replace("passage: ", "")
                     + "\n\n",
            'Source': chunk[2]['source'],
            'Paper': chunk[2]['title'],
            'Year': chunk[2]['year'],
        })

    return {
        "chunk_metadata": txt_data,
        "img_metadata": img_data,
        "answer": ans['answer'],
        "explanation": ans['explanation'],
        "chunk_explanation": ans.get('chunk_explanation', ''),
        "img_explanation": ans.get('img_explanation', ''),
        "metadata": {
            "text_context": relevant_txt_context,
            "image_context": relevant_img_context,
        },
    }


def query_llm_with_context(
    model_url: str,
    question: str,
    system_prompt: str,
    txt_context: str,
    images_base64: list[str],
) -> dict:
    """
    Queries a vision-capable LLM with textual context and base64-encoded
    images.  Returns a structured dict with the answer, explanations and
    indices of relevant chunks / images.
    """
    llm = create_llm_connector(
        model_url,
        extra_body={"provider": {"only": allowed_providers}},
        temperature=0.05,
    )
    # base_url, model_name = model_url.split(";")
    # llm = ChatOpenAI(
    #     model=model_name,
    #     base_url=base_url,
    #     api_key=os.getenv("LLM__SERVICE_KEY"),  # noqa
    #     temperature=0.1
    # )

    class ResScheme(BaseModel):
        answer: str = Field(
            description="The answer to the query", default=""
        )
        explanation: str = Field(
            description="The logical reasoning for the answer", default=""
        )
        chunk_explanation: str = Field(
            description=(
                "The explanation why the chosen chunk/chunks "
                "are relevant to the answer"
            ),
            default="",
        )
        img_explanation: str = Field(
            description=(
                "The explanation why the chosen image/images "
                "are relevant to the answer"
            ),
            default="",
        )
        relevant_text: list[int] = Field(
            description=(
                "A list of integers representing the relevant text chunk "
                "numbers, numeration of chunks starts with 1"
            ),
            default=[],
        )
        relevant_images: list[int] = Field(
            description=(
                "A list of integers representing the relevant image "
                "numbers, numeration of images starts with 1"
            ),
            default=[],
        )

    structured_llm = llm.with_structured_output(schema=ResScheme)

    messages = [
        SystemMessage(content=system_prompt),
        prompt_func(
            {
                "text": f"USER QUESTION: {question}\n\nCONTEXT: {txt_context}",
                "image": images_base64,
            }
        ),
    ]

    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            res = structured_llm.invoke(messages)
            return {
                "answer": res.answer,
                "explanation": res.explanation,
                "chunk_explanation": res.chunk_explanation,
                "img_explanation": res.img_explanation,
                "relevant_text": res.relevant_text,
                "relevant_images": res.relevant_images,
            }
        except Exception as e:
            last_error = e
            messages.append(
                HumanMessage(
                    content=(
                        "Previous response was invalid JSON. "
                        "Respond with ONLY valid JSON."
                    )
                )
            )
    raise RuntimeError(
        f"Failed to get valid structured response after 3 attempts. "
        f"Last error: {last_error}"
    ) from last_error


logger = logging.getLogger(__name__)


def process_scientific_question(
    question: str,
    system_prompt: str,
    retriever,
    simple_retriever,
    s3_store,
    llm_url: str,
    top_k: int = 60,
    rerank_k: int = 20,
) -> dict:
    """
    End-to-end pipeline: metadata filters → direct body retrieval →
    image loading from S3 → LLM answer generation.
    """

    # ── 1. Extracting meta filters from a question ────────────────────
    meta_filter = extract_metadata_filters(
        question, llm_url, extract_query_filters_prompt
    )
    meta_filter_chroma = build_chroma_where_filter(meta_filter)
    domain_metadata_type = get_domain_metadata_type(meta_filter) or ""

    # ── 2. Retrieving and reranking body chunks ──────────────────────
    papers: list[dict] = []
    titles: dict[str, str] = {}
    article_ids: list[str] = []
    seen_articles: set[str] = set()

    body_filters: dict = {"role": {"$eq": "body"}}
    if meta_filter_chroma:
        body_filters = {
            "$and": [meta_filter_chroma, {"role": {"$eq": "body"}}]
        }
    try:
        body_chunks = retriever.retrieve(
            query=question, top_k=top_k, rerank_k=rerank_k,
            filters=body_filters,
        )
    except Exception as e:
        logger.error(f"Failed to retrieve body chunks: {e}")
        return {"answer": f"Failed to retrieve text chunks. Error: {e}"}
    for chunk in body_chunks:
        if chunk.article_id in seen_articles:
            continue
        seen_articles.add(chunk.article_id)
        article_ids.append(chunk.article_id)
        title = (chunk.metadata or {}).get("paper_title", "")
        titles[chunk.article_id] = title
        papers.append({
            "article_id": chunk.article_id,
            "title": title,
            "domain": chunk.domain,
            "field": chunk.field,
        })

    if not article_ids:
        return {"answer": "No relevant papers found in the database."}

    # ── 3. Collecting image names and retrieving captions ─────────────
    raw_image_names: set[str] = set()
    for chunk in body_chunks:
        raw_image_names.update(chunk.images_in_chunk or [])

    captions: dict[tuple[str, str], object] = {}
    if raw_image_names:
        image_ids = [n.split(".")[0] for n in raw_image_names]
        try:
            for c in simple_retriever.retrieve(
                query=question,
                top_k=1000,
                filters={
                    "$and": [
                        {"article_id": {"$in": article_ids}},
                        {"role": {"$eq": "image_caption"}},
                        {"image_id": {"$in": image_ids}},
                    ]
                },
            ):
                img_id = (c.metadata or {}).get("image_id", "")
                captions[(c.article_id, img_id + ".jpeg")] = c
        except Exception as e:
            logger.warning(f"Failed to retrieve image captions: {e}")

    # ── 4. Downloading images from S3 (base64) ─────────────────────────
    images_base64: list[str] = []
    images_meta: list[dict] = []
    seen_images: set[tuple[str, str]] = set()

    for chunk in body_chunks:
        for img_name in chunk.images_in_chunk or []:
            key = (chunk.article_id, img_name)
            if key in seen_images:
                continue
            seen_images.add(key)
            try:
                img_bytes = s3_store.get_image_bytes_from_s3(
                    chunk.domain, chunk.article_id, img_name
                )
                if isinstance(img_bytes, (bytes, bytearray)) and img_bytes:
                    images_base64.append(base64.b64encode(img_bytes).decode())
                    caption_obj = captions.get(key)
                    img_info = {
                        "image_name": img_name,
                        "caption": (
                            caption_obj.content if caption_obj else ""
                        ),
                        "article_id": chunk.article_id,
                        "title": titles.get(chunk.article_id, ""),
                    }
                    caption_meta = (caption_obj.metadata or {}) if caption_obj else {}
                    img_info = add_domain_metadata_to_img_info(
                        domain_metadata_type, caption_meta, img_info
                    )
                    images_meta.append(img_info)
            except Exception:
                logger.warning(
                    "Could not load image %s for article %s",
                    img_name,
                    chunk.article_id,
                )

    # ── 5. Forming the text context ────────────────────────────────────
    txt_parts: list[str] = []
    for idx, chunk in enumerate(body_chunks, start=1):
        txt_parts.append(
            f"{idx}. Paper: {titles.get(chunk.article_id, 'Unknown')}\n"
            f"Chunk: {chunk.content}"
        )
    txt_context = "\n\n".join(txt_parts)

    # Image captions are added to the text so that the LLM can refer to them by number
    if images_meta:
        img_lines = ["\nImages:"]
        for idx, img in enumerate(images_meta, start=1):
            img_lines.append(
                f"Image {idx}: {img['caption'] or 'No caption'} "
                f"(Paper: {img['title']})"
            )
        txt_context += "\n" + "\n".join(img_lines)

    # Domain meta-information (chemistry, etc.), if applicable
    domain_metadata = format_domain_metadata(domain_metadata_type, images_meta)
    if domain_metadata != "":
        txt_context += f"Domain metadata\n{domain_metadata}\n\n"
    else:
        txt_context += "No domain metadata found for context."

    # ── 6. Querying the LLM ────────────────────────────────────────────
    ans = query_llm_with_context(
        llm_url, question, system_prompt, txt_context, images_base64
    )

    # ── 7. Composing a response with relevant context ──────────────────
    relevant_txt_context: list[dict] = []
    for num in ans.get("relevant_text", []):
        if 1 <= num <= len(body_chunks):
            chunk = body_chunks[num - 1]
            relevant_txt_context.append(
                {
                    "chunk": f"Chunk {num}:\n{chunk.content}\n",
                    "Source": (chunk.metadata or {}).get("source", ""),
                    "Paper": titles.get(chunk.article_id, ""),
                    "Year": (chunk.metadata or {}).get("publication_year", ""),
                }
            )

    relevant_img_context: list[dict] = []
    for num in ans.get("relevant_images", []):
        if 1 <= num <= len(images_meta):
            relevant_img_context.append(images_meta[num - 1])

    return {
        "papers": papers,
        "answer": ans["answer"],
        "explanation": ans["explanation"],
        "chunk_explanation": ans.get("chunk_explanation", ""),
        "img_explanation": ans.get("img_explanation", ""),
        "metadata": {
            "text_context": relevant_txt_context,
            "image_context": relevant_img_context,
        },
    }


if __name__ == "__main__":
    # file_paths = []  # Enter list of paths to images here
    #
    # images = list(map(convert_to_base64, file_paths))
    #
    # llm = create_llm_connector(VISION_LLM_URL)
    #
    # # question = ("Какая реакция идет протекает на 6 стадии Total Synthesis of (−)-Glionitrin A/B? Какие реагенты"
    # #             " участвовали в реакции и какой продукт получили? Какой получился выход?")
    # question = ("I need all the compounds that were used in the experiments. Obligatorily I need all results to be in"
    #             " the form of a table of 2 columns where in the first column were the names by IUPAC numberclature and"
    #             " in the second column in SMILES notation. Don't add it to this list of reaction products for me. Can"
    #             " you do that?")
    # context = ""
    #
    # messages = [
    #     SystemMessage(content=sys_prompt),
    #     prompt_func({"text": f"USER QUESTION: {question}\n\nCONTEXT: {context}", "image": images})
    # ]
    # # messages = [
    # #     SystemMessage(content="You're a useful assistant. You only ever reply in the form of valid JSON."),
    # #     prompt_func(
    # #         {
    # #             "text": "For the provided images, generate a detailed clear description. If there is a table in the"
    # #                     " image, parse it and return it in HTML format. If you see chemical compounds in the figures,"
    # #                     " output the names of the compounds according to IUPAC nomenclature.\n"
    # #                     " As a response, return ONLY JSON of the following form: {‘figure_1’:"
    # #                     " ‘figure_1_description’, ‘figure_2’: ‘figure_2_description’, ‘table_1’:"
    # #                     " ‘table_1_description’...}",
    # #             "image": images
    # #         }
    # #     )
    # # ]
    #
    # res = llm.invoke(messages)
    # print(res.content)
    # print(res.response_metadata)

    #######################################################

    paper_store = ChromaDBPaperStore()
    # question = 'What aliphatic hydroxy acids are present in the papers published in 2022? Give me their SMILES.'
    question = 'What components are involved in the synthesis of BASHY dyes, and what are the uses of these dyes?'
    # question = 'What IC50 values do weakly active and highly active Bruton\'s tyrosine kinase inhibitors have?'
    # question = 'How does the synthesis of Glionitrin A/B happen?'

    # res = simple_query_llm(VISION_LLM_URL, question, [paper])
    result = process_question(question, sys_prompt, paper_store)
    from pprint import pprint
    pprint(result)
