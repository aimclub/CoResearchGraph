import base64
from io import BytesIO
from typing import Any, Callable, Dict, List
from urllib.parse import urlparse
import logging
import time

from langchain_core.messages import HumanMessage
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from .openalex import UNKNOWN_PUBLICATION_YEAR

logger = logging.getLogger(__name__)


class ExpandedSummary(BaseModel):
    """Expanded version of paper's summary."""
    paper_summary: str = Field(description="Summary of the paper.")
    paper_title: str = Field(
        description="Title of the paper. If the title is not explicitly specified, use the default value - 'NO TITLE'"
    )
    publication_year: int = Field(
        description=(
            f"Year of publication of the paper. If the publication year is not explicitly specified, use the default "
            f"value - {UNKNOWN_PUBLICATION_YEAR}."
        )
    )
    authors: str = Field(
        description=(
            "Authors of the paper: a string of comma separated first and last names or surnames and initials. "
            "If the authors are not explicitly specified, use the default value 'NO AUTHORS'."
        )
    )
    source: str = Field(
        description=(
            "Source where the paper was published. If the source is not explicitly specified, use the default "
            "value - 'UNDEFINED'"
        )
    )


class PaperMetadata(BaseModel):
    """Bibliographic metadata extracted from a paper."""

    paper_title: str = Field(
        description="Full paper title, or 'NO TITLE' when it is not explicitly specified."
    )
    publication_year: int = Field(
        description=f"Publication year, or {UNKNOWN_PUBLICATION_YEAR} when it is not explicitly specified."
    )
    authors: str = Field(
        description="Comma-separated authors, or 'NO AUTHORS' when they are not explicitly specified."
    )
    source: str = Field(
        description="Journal, conference, or publisher, or 'UNDEFINED' when it is not explicitly specified."
    )


def convert_to_base64(file_path, s3_store):
    """
    Convert an image file to a Base64 encoded string.

    This method reads an image from the specified file path, encodes it as a JPEG image in memory, then converts it
    into a Base64 string representation.

    Args:
        file_path (str): The path to the image file.
        s3_store: S3-like store with papers files

    Returns:
        str: A Base64 encoded string representing the JPEG image.
    """
    if file_path.startswith("http://"):
        s3_key, bucket_name = extract_s3_bucket_and_key(file_path)
        pil_image = Image.open(BytesIO(s3_store.get_image_bytes_from_s3(s3_key, bucket_name)))
    else:
        pil_image = Image.open(file_path)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def prompt_func(data):
    """
    Creates a structured message containing text and images for use in a conversational context.

    This method prepares the input data into a format suitable for presenting information in a multi-modal interface,
    by converting images to data URIs that can be directly embedded in a message and combining them with the provided
    text.

    Args:
        data (dict): A dictionary containing the message content:
            - "text" (str): The text content of the message;
            - "image" (list): A list of base64 encoded JPEG images to include in the message.

    Returns:
        HumanMessage: A HumanMessage object with a structured 'content' list.
            The 'content' list contains dictionaries representing each part of the message,
            with "type" keys indicating whether it's "text" or "image_url". Image URLs
            are formatted as data URIs.
    """
    text = data["text"]
    imgs = data["image"]
    content_parts = []
    
    for img in imgs:
        image_part = {
            "type": "image_url",
            # "image_url": f"data:image/jpeg;base64,{img}",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"},
        }
        content_parts.append(image_part)
    
    text_part = {"type": "text", "text": text}
    content_parts.append(text_part)
    
    return HumanMessage(content=content_parts)


def extract_s3_bucket_and_key(s3_url: str):
    """
    Extracts the file key in S3 storage and the bucket name from the full file path.

    Args:
        s3_url: The full path to the file in S3 storage.

    Returns:
        A tuple of S3 key and bucket name.
    """
    o = urlparse(s3_url)
    bucket, key = o.path.split('/', 2)[1:]
    return key, bucket


def pil_to_base64(image_object, img_format="JPEG"):
    """
    Converts a PIL Image object to a base64 encoded string.

    Args:
        image_object: The PIL Image object.
        img_format: The image format for saving (e.g., "JPEG", "PNG").

    Returns:
        A base64 encoded string.
    """
    buffered = BytesIO()
    image_object.save(buffered, format=img_format)
    img_bytes = buffered.getvalue()
    img_b64bytes = base64.b64encode(img_bytes)
    img_b64string = img_b64bytes.decode('utf-8')
    return img_b64string


OPENALEX_TAXONOMY: Dict[str, List[str]] = {
    "Life Sciences": [
        "Agricultural and Biological Sciences",
        "Biochemistry, Genetics and Molecular Biology",
        "Immunology and Microbiology",
        "Neuroscience",
        "Pharmacology, Toxicology and Pharmaceutics"
    ],
    "Social Sciences": [
        "Arts and Humanities",
        "Business, Management and Accounting",
        "Decision Sciences",
        "Economics, Econometrics and Finance",
        "Psychology",
        "Social Sciences"
    ],
    "Physical Sciences": [
        "Chemical Engineering",
        "Chemistry",
        "Computer Science",
        "Earth and Planetary Sciences",
        "Energy",
        "Engineering",
        "Environmental Science",
        "Materials Science",
        "Mathematics",
        "Physics and Astronomy"
    ],
    "Health Sciences": [
        "Dentistry",
        "Health Professions",
        "Medicine",
        "Nursing",
        "Veterinary"
    ]
}


class OpenAlexClassification(BaseModel):
    """Domain and field clarification via LLM based on """
    primary_domain: str = Field(description="The name of the primary domain in English (e.g., 'Physical Sciences').")
    primary_field: str = Field(description="The name of the primary field in English (e.g., 'Computer Science').")
    confidence_score: float | int = Field(
        description="Confidence score of the classification ranging from 0.00 to 1.00.")
    justification: str = Field(
        description="A brief justification of the choice in Russian (1-2 sentences), referencing specific terms from the abstract or title."
    )

    @model_validator(mode='after')
    def validate_domain_field_mapping(self) -> 'OpenAlexClassification':
        """Validates that the selected field strictly belongs to the selected domain."""
        domain = self.primary_domain.strip()
        field = self.primary_field.strip()

        if domain not in OPENALEX_TAXONOMY:
            raise ValueError(
                f"Invalid domain '{domain}'. Must be exactly one of: {list(OPENALEX_TAXONOMY.keys())}."
            )

        allowed_fields = OPENALEX_TAXONOMY[domain]
        if field not in allowed_fields:
            raise ValueError(
                f"Invalid field '{field}' for domain '{domain}'. "
                f"The field must strictly belong to the chosen domain. Allowed fields for '{domain}' are: {allowed_fields}."
            )

        self.primary_domain = domain
        self.primary_field = field
        
        return self


def invoke_llm_with_retry(
    llm: Any,
    input_data: Any,
    *,
    operation: str,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    response_validator: Callable[[Any], bool] | None = None,
) -> Any:
    """Invoke and validate an LLM response with exponential backoff."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.invoke(input_data)
            if response is None:
                raise ValueError("LLM returned None")
            if response_validator is not None and not response_validator(response):
                raise ValueError(
                    f"LLM returned an invalid response of type {type(response).__name__}"
                )
            return response
        except Exception as e:
            if attempt == max_attempts:
                logger.exception(
                    "LLM operation '%s' failed after %d attempts",
                    operation,
                    max_attempts,
                )
                raise RuntimeError(
                    f"Failed to invoke or validate LLM response after "
                    f"{max_attempts} attempts: {e}"
                ) from e

            delay = initial_delay * (2 ** (attempt - 1))
            logger.warning(
                "LLM operation '%s' failed on attempt %d/%d; retrying in %.1f seconds",
                operation,
                attempt,
                max_attempts,
                delay,
                exc_info=True,
            )
            time.sleep(delay)
