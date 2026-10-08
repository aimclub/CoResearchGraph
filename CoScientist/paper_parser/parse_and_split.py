import logging
import os
from pathlib import Path
import re
import shutil

from bs4 import BeautifulSoup, Tag
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered, save_output
from protollm.connectors import create_llm_connector

from CoScientist.paper_parser.parser_prompts import cls_prompt, table_extraction_prompt
from CoScientist.paper_parser.utils import prompt_func, convert_to_base64
from CoScientist.paper_parser.s3_connection import S3BucketService
from CoScientist.config import get_settings
from CoScientist.paper_analysis.settings import allowed_providers

settings = get_settings()
_log = logging.getLogger(__name__)

ROOT_DIR = settings.storage.root_dir
PARSE_RESULTS_PATH = os.path.join(ROOT_DIR, settings.storage.parse_results)
PAPERS_PATH = os.path.join(ROOT_DIR, settings.storage.papers_storage)
IMAGES_PATH = os.path.join(ROOT_DIR, settings.storage.img_storage)
VISION_LLM_URL = settings.llm.vision_url
VISION_LLM_NAME = VISION_LLM_URL.split(';')[1]
LLM_SERVICE_CC_URL = settings.llm.service_cc_url
LLM_SERVICE_KEY = settings.llm.service_key
MARKER_LLM = settings.llm.marker_model
LLM_SERVICE_URL = settings.llm.service_url
IMAGE_RESOLUTION_SCALE = 2.0
USE_S3 = settings.s3.use_s3


def parse_with_marker(paper_name: str, use_llm: bool=False) -> (str, Path):
    """
    Parses a paper to extract structured information using a marker-based approach.
    
    This method converts a PDF paper to HTML, optionally enhances parsing with an LLM, and saves the results. It
    facilitates the extraction of text and images for further analysis and querying.
    
    Args:
        paper_name (str): The name of the paper file (PDF).
        use_llm (bool, optional): A boolean flag indicating whether to utilize the LLM service during parsing.
            Defaults to False.
    
    Returns:
        tuple: A tuple containing the stem of the paper name (without extension) and the path to the output directory 
               where the rendered HTML and associated images are saved.
    """
    config = {
        "output_format": "html",
        "use_llm": use_llm,
        "openai_api_key": LLM_SERVICE_KEY,
        "openai_model": MARKER_LLM,
        "openai_base_url": LLM_SERVICE_URL
    }
    config_parser = ConfigParser(config)

    file_name = Path(paper_name)

    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config=config_parser.generate_config_dict(),
        renderer=config_parser.get_renderer(),
        llm_service="marker.services.openai.OpenAIService"
    )
    rendered = converter(paper_name)
    text, _, images = text_from_rendered(rendered)

    output_dir = Path(PARSE_RESULTS_PATH, str(file_name.stem) + "_marker")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_output(rendered, output_dir=str(output_dir), fname_base=f"{file_name.stem}")
    return file_name.stem, output_dir


def clean_up_html(
        doc_dir: Path, file_name: str, html: str, s3_service: S3BucketService = None, paper_s3_prefix: str = None
) -> (str, dict):
    """
    Cleans up HTML content by removing irrelevant sections like acknowledgements and references, and processes images
    to either remove them or replace them with extracted tables.
    
    Args:
        doc_dir (Path): The directory containing the document.
        file_name (Path): The name of the HTML file.
        html (str): The HTML content as a string.
        s3_service (S3BucketService): The client for working with S3 service.
        paper_s3_prefix (str): The prefix for generating the object's key in S3. Defaults to None.
    
    Returns:
        str: The cleaned HTML content as a string, potentially with images replaced by tables.
    """
    soup = BeautifulSoup(html, "lxml")

    blacklist = [
        "author information", "associated content", "acknowledgment", "acknowledgement", "acknowledgments",
        "acknowledgements", "references", "data availability", "declaration of competing interest",
        "credit authorship contribution statement", "funding", "ethical statements", "supplementary materials",
        "conflict of interest", "conflicts of interest", "author contributions", "data availability statement",
        "ethics approval", "supplementary information"
    ]
    for header in soup.find_all(["h1", "h2", "h3"]):
        header_text = header.get_text(strip=True).lower()

        if any(exclude in header_text for exclude in blacklist):
            next_node = header.next_sibling

            elements_to_remove = []
            while next_node and next_node.name not in ["h1", "h2"]:
                elements_to_remove.append(next_node)
                next_node = next_node.next_sibling

            header.decompose()
            for element in elements_to_remove:
                if isinstance(element, Tag):
                    element.decompose()

    llm = create_llm_connector(VISION_LLM_URL, extra_body={"provider": {"only": allowed_providers}})

    image_url_mapping = {}

    for img in soup.find_all('img'):
        img_src = img.get("src")
        if not img_src:
            continue

        local_img_path = (Path(doc_dir) / img_src).as_posix()
        try:
            images = list(map(convert_to_base64, [local_img_path]))
        except OSError as e:
            if e.errno == 2:
                print(f"File not found: {e}")
                continue
            else:
                print(f"Error from OS: {e}")
                continue
        query = [prompt_func({"text": cls_prompt, "image": images})]
        res_1 = llm.invoke(query).content
        if res_1.strip() == "False":
            parent_p = img.find_parent('p')
            if parent_p:
                parent_p.decompose()
                os.remove(local_img_path)
        else:
            table_query = [prompt_func({"text": table_extraction_prompt, "image": images})]
            res_2 = llm.invoke(table_query).content
            if res_2.strip() != "No table":
                pattern = r'<table\b[^>]*>.*?</table>'
                match = re.search(pattern, res_2, re.DOTALL)
                if match:
                    html_table = match.group(0)
                    table_soup = BeautifulSoup(html_table, 'html.parser')
                    parent_p = img.find_parent('p')
                    if parent_p:
                        parent_p.replace_with(table_soup)
                        os.remove(local_img_path)
            elif s3_service and paper_s3_prefix:
                s3_key = f"{paper_s3_prefix}/{img_src}"
                s3_service.upload_file_object(paper_s3_prefix, img_src, local_img_path)
                s3_url = f"{s3_service.endpoint.rstrip('/')}/{s3_service.bucket_name}/{s3_key}"
                img['src'] = s3_url
                image_url_mapping[local_img_path] = s3_url
            else:
                image_url_mapping[local_img_path] = local_img_path

    new_file_name = f"{file_name}_processed.html"
    new_path = (Path(doc_dir, new_file_name)).as_posix()
    with open(new_path, "w", encoding='utf-8') as file:
        file.write(str(soup.prettify()))

    if s3_service and paper_s3_prefix:
        s3_service.upload_file_object(paper_s3_prefix, new_file_name, new_path)

    return soup.prettify(), image_url_mapping
    

def html_chunking(html_string: str, paper_name: str, paper_summary) -> list:
    """
    Chunks an HTML string into semantic passages for efficient information retrieval.
    
    This method splits an HTML string into smaller, meaningful chunks based on the structure of the document,
    preserving semantic elements like headers, lists, and tables. This allows for targeted analysis and answering of
    questions related to the content. Each chunk is enriched with metadata about images present and its source
    document.
    
    Args:
        html_string (str): The HTML string representing the paper content.
        paper_name (str): The name of the paper associated with the HTML string.
    
    Returns:
        list: A list of Document objects, where each object represents a chunk of the HTML. Each Document has a
        'page_content' (string) attribute holding the text of the chunk and a 'metadata' (dictionary)
        attribute containing 'imgs_in_chunk' (string) with image URLs found in the chunk, and 'source' (string)
        indicating the paper name with ".pdf" extension.
    """

    def custom_table_extractor(table_tag):
        return str(table_tag).replace("\n", "")

    headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]

    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=headers_to_split_on,
        max_chunk_size=2500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". "],
        elements_to_preserve=["ul", "table", "ol"],
        preserve_images=True,
        custom_handlers={"table": custom_table_extractor}
    )

    documents = splitter.split_text(html_string)
    for doc in documents:
        doc.page_content = "passage: " + doc.page_content  # Maybe delete "passage: " addition
        doc.metadata.update({
            "imgs_in_chunk": str(extract_img_url(doc.page_content, paper_name)),
            "source": f"{paper_name}.pdf",
            "paper_title": paper_summary.paper_title,
            "publication_year": paper_summary.publication_year,
            "paper_authors": paper_summary.paper_authors,
            "publication_source": paper_summary.publication_source,
            "research_area": paper_summary.research_domain,
        })

    return documents


def extract_img_url(doc_text: str, p_name: str) -> list[str]:
    """
    Extracts image URLs from a document text related to scientific papers.
    
    This method identifies image references within the text and constructs their full paths for access. It focuses on
    JPEG images specifically referenced using a specific markdown-like syntax.
    
    Args:
        doc_text: The text of the scientific document to analyze.
        p_name: The name of the project or paper, used to organize image paths.
    
    Returns:
        list: A list of strings, where each string is a full path to an image extracted from the document,
        constructed using the project path and the image filename.
    """
    pattern = r'!\[image:([^\]]+\.jpeg)\]\(([^)]+\.jpeg)\)'
    matches = re.findall(pattern, doc_text)
    if USE_S3:
        return [entry[0] for entry in matches]
    else:
        return [os.path.join(PARSE_RESULTS_PATH, p_name, entry[0]) for entry in matches]


def clean_up_after_processing(doc_dir: str | Path) -> None:
    """
    Deletes local parsing results.
    
    Args:
        doc_dir (str): The path to the local parsing results.

    Returns:
        None
    """
    if os.path.exists(doc_dir):
        try:
            shutil.rmtree(doc_dir)
            print(f"Directory '{doc_dir}' and its contents removed successfully.")
        except OSError as e:
            print(f"Error: {e}")
    else:
        print(f"Directory '{doc_dir}' does not exist.")


if __name__ == "__main__":
    # documents = loader(parse_and_clean(
    #     "path-to-paper-file"))
    # for document in documents:
    #     print(document)
    
    p_path = PAPERS_PATH
    paper = "paper-filename"
    paper_path = os.path.join(p_path, paper)
    f_name, dir_name = parse_with_marker(paper_name=paper_path)
    # parsed_paper = clean_up_html(paper_name=f_name, doc_dir=dir_name)
    # chunks = html_chunking(html_string=parsed_paper, paper_name=f_name)
    # print(chunks)
