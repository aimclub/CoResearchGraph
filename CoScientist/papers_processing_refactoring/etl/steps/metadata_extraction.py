import logging

from langchain_core.messages import HumanMessage
from pydantic import Field, create_model

from ..base import ETLStep
from ..context import ETLContext
from ...utils.general_utils import OpenAlexClassification, PaperMetadata
from ...utils.general_utils import invoke_llm_with_retry
from ...utils.openalex import UNKNOWN_PUBLICATION_YEAR, find_doi_by_title, get_openalex_metadata
from ...utils.prompts import classification_from_content_prompt, metadata_extraction_prompt


logger = logging.getLogger(__name__)
GEMINI_3_5_FLASH_LITE_INPUT_LIMIT = 1_048_576
SAFE_INPUT_TOKEN_LIMIT = int(GEMINI_3_5_FLASH_LITE_INPUT_LIMIT * 0.9)
PAPER_METADATA_FIELDS = ("paper_title", "publication_year", "authors", "source")
LLM_UNDEFINED_VALUES = {
    "paper_title": "NO TITLE",
    "publication_year": UNKNOWN_PUBLICATION_YEAR,
    "authors": "NO AUTHORS",
    "source": "UNDEFINED",
}


def _paper_metadata_from_article(article_metadata: dict) -> dict:
    """Return valid bibliographic values supplied by the article source."""
    result = {}
    for field in ("paper_title", "authors", "source"):
        value = article_metadata.get(field)
        if isinstance(value, str) and value.strip():
            result[field] = value.strip()

    publication_year = article_metadata.get("publication_year")
    if (type(publication_year) is int and publication_year > 0
            and publication_year != UNKNOWN_PUBLICATION_YEAR):
        result["publication_year"] = publication_year
    return result


def _split_html_in_half(html: str) -> tuple[str, str]:
    midpoint = len(html) // 2
    boundaries = []
    for tag in ("</p>", "</div>", "</section>"):
        position = html.rfind(tag, 0, midpoint)
        if position >= 0:
            boundaries.append(position + len(tag))
    split_at = max(boundaries, default=midpoint)
    if split_at <= len(html) // 4:
        split_at = midpoint
    return html[:split_at], html[split_at:]


def _missing_metadata_model(missing_fields: tuple[str, ...]):
    """Build a structured-output model containing only the requested fields."""
    definitions = {}
    for field in missing_fields:
        model_field = PaperMetadata.model_fields[field]
        definitions[field] = (
            model_field.annotation,
            Field(description=model_field.description),
        )
    return create_model(
        "MissingPaperMetadata_" + "_".join(missing_fields),
        **definitions,
    )


def _merge_missing_metadata(first: dict, second: dict, missing_fields: tuple[str, ...]) -> dict:
    """Prefer values found in the first HTML part, then try the second part."""
    merged = {}
    for field in missing_fields:
        first_value = first[field]
        undefined = LLM_UNDEFINED_VALUES[field]
        if isinstance(undefined, str) and isinstance(first_value, str):
            is_undefined = first_value.casefold() == undefined.casefold()
        else:
            is_undefined = first_value == undefined
        merged[field] = second[field] if is_undefined else first_value
    return merged


def _extract_metadata_from_two_parts(
        metadata_llm, prompt: str, missing_fields: tuple[str, ...],
        html: str, article_id: str) -> dict:
    first_html, second_html = _split_html_in_half(html)
    part_metadata = []
    for part_number, html_part in enumerate((first_html, second_html), start=1):
        response = invoke_llm_with_retry(
            metadata_llm,
            [HumanMessage(content=prompt + html_part)],
            operation=f"extract metadata for article {article_id}, part {part_number}/2",
        )
        part_metadata.append(response.model_dump())
    return _merge_missing_metadata(
        part_metadata[0], part_metadata[1], missing_fields,
    )


class MetadataExtractionStep(ETLStep):
    """Extract paper metadata and classification without generating a summary."""

    name = "metadata_extraction"

    def run(self, ctx: ETLContext) -> None:
        article_id = ctx.article.id

        html = ctx.artifact_store.get_html(article_id, "image_captioning")
        if not html:
            raise RuntimeError(f"{self.name} step requires cleaned HTML")

        manifest_data = ctx.artifact_store.get_metadata(article_id, "image_captioning") or {}
        article_metadata = ctx.article.metadata or {}
        supplied_metadata = _paper_metadata_from_article(article_metadata)

        if len(supplied_metadata) == len(PAPER_METADATA_FIELDS):
            paper_metadata = PaperMetadata(**supplied_metadata)
        else:
            missing_fields = tuple(
                field for field in PAPER_METADATA_FIELDS
                if field not in supplied_metadata
            )
            response_model = _missing_metadata_model(missing_fields)
            metadata_llm = ctx.llm.with_structured_output(response_model)
            prompt = metadata_extraction_prompt({
                field: PaperMetadata.model_fields[field].description
                for field in missing_fields
            })
            metadata_input = prompt + html
            estimated_tokens = len(metadata_input.encode("utf-8"))
            if estimated_tokens > SAFE_INPUT_TOKEN_LIMIT:
                logger.info(
                    "Article %s metadata input is too large for one request "
                    "(%d estimated tokens); splitting HTML into two parts",
                    article_id,
                    estimated_tokens,
                )
                llm_metadata = _extract_metadata_from_two_parts(
                    metadata_llm, prompt, missing_fields, html, article_id,
                )
            else:
                response = invoke_llm_with_retry(
                    metadata_llm,
                    [HumanMessage(content=metadata_input)],
                    operation=f"extract metadata for article {article_id}",
                )
                llm_metadata = response.model_dump()
            paper_metadata = PaperMetadata(
                **supplied_metadata,
                **llm_metadata,
            )

        # The source record may not contain a title. From this point onward use
        # the title resolved above, including the LLM fallback when necessary.
        ctx.article.name = paper_metadata.paper_title

        is_sapphire = article_metadata.get("ingestion_source") == "sapphire"
        doi = article_metadata.get("doi") if is_sapphire else None
        if isinstance(doi, str) and doi.strip() and doi.strip().lower() != "unknown":
            doi = doi.strip()
        elif is_sapphire:
            # The OpenAlex work was already fetched by Sapphire. Trust the DOI
            # absence in that card instead of performing a second lookup.
            doi = None
        else:
            doi = find_doi_by_title(
                paper_metadata.paper_title,
                paper_metadata.publication_year,
            )
        domain = field = None
        source = paper_metadata.source
        if is_sapphire:
            # Sapphire already fetched the OpenAlex work before downloading the PDF.
            domain = article_metadata.get("domain")
            field = article_metadata.get("field")
            domain = domain.strip() if isinstance(domain, str) and domain.strip() else None
            field = field.strip() if isinstance(field, str) and field.strip() else None
        elif domain is None or field is None:
            openalex_metadata = get_openalex_metadata(
                doi=doi,
                title=paper_metadata.paper_title,
                publication_year=paper_metadata.publication_year,
            )
            if openalex_metadata is not None:
                domain, field, openalex_source = openalex_metadata
                if openalex_source is not None:
                    source = openalex_source

        if domain is None or field is None:
            classification_llm = ctx.llm.with_structured_output(OpenAlexClassification)
            llm_classification: OpenAlexClassification = invoke_llm_with_retry(
                classification_llm,
                [
                    HumanMessage(
                        content=classification_from_content_prompt.format(
                            TITLE=paper_metadata.paper_title,
                            ARTICLE_CONTENT=html,
                        )
                    )
                ],
                operation=f"classify article {article_id}",
            )
            # Keep each valid value from the already fetched OpenAlex card and
            # use the LLM only for the classification fields that are missing.
            domain = domain or llm_classification.primary_domain
            field = field or llm_classification.primary_field

        manifest_data["paper_metadata"] = {
            **paper_metadata.model_dump(),
            "doi": doi or "unknown",
            "domain": domain,
            "field": field,
            "source": source,
        }
        manifest_data["paper_of_file_name"] = ctx.article.name
        manifest_data["article_metadata"] = ctx.article.metadata

        ctx.artifact_store.put_html(article_id, self.name, html)
        ctx.artifact_store.put_metadata(article_id, self.name, manifest_data)
