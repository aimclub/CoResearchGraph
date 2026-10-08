import logging

from langchain_core.messages import HumanMessage

from ..base import ETLStep
from ..context import ETLContext
from ...utils.general_utils import OpenAlexClassification, PaperMetadata
from ...utils.general_utils import invoke_llm_with_retry
from ...utils.openalex import UNKNOWN_PUBLICATION_YEAR, find_doi_by_title, get_openalex_metadata
from ...utils.prompts import classification_from_content_prompt, metadata_extraction_prompt


logger = logging.getLogger(__name__)
GEMINI_3_5_FLASH_LITE_INPUT_LIMIT = 1_048_576
SAFE_INPUT_TOKEN_LIMIT = int(GEMINI_3_5_FLASH_LITE_INPUT_LIMIT * 0.9)


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


def _merge_metadata(first: PaperMetadata, second: PaperMetadata) -> PaperMetadata:
    def prefer_defined(first_value: str, second_value: str, undefined: str) -> str:
        return first_value if first_value.casefold() != undefined.casefold() else second_value

    publication_year = first.publication_year
    if publication_year == UNKNOWN_PUBLICATION_YEAR:
        publication_year = second.publication_year

    return PaperMetadata(
        paper_title=prefer_defined(first.paper_title, second.paper_title, "NO TITLE"),
        publication_year=publication_year,
        authors=prefer_defined(first.authors, second.authors, "NO AUTHORS"),
        source=prefer_defined(first.source, second.source, "UNDEFINED"),
    )


def _extract_metadata_from_two_parts(metadata_llm, html: str, article_id: str) -> PaperMetadata:
    first_html, second_html = _split_html_in_half(html)
    part_metadata = []
    for part_number, html_part in enumerate((first_html, second_html), start=1):
        part_input = metadata_extraction_prompt + html_part
        part_metadata.append(
            invoke_llm_with_retry(
                metadata_llm,
                [HumanMessage(content=part_input)],
                operation=f"extract metadata for article {article_id}, part {part_number}/2",
            )
        )
    return _merge_metadata(*part_metadata)


class MetadataExtractionStep(ETLStep):
    """Extract paper metadata and classification without generating a summary."""

    name = "metadata_extraction"

    def run(self, ctx: ETLContext) -> None:
        article_id = ctx.article.id

        html = ctx.artifact_store.get_html(article_id, "image_captioning")
        if not html:
            raise RuntimeError(f"{self.name} step requires cleaned HTML")

        manifest_data = ctx.artifact_store.get_metadata(article_id, "image_captioning") or {}
        metadata_llm = ctx.llm.with_structured_output(PaperMetadata)
        metadata_input = metadata_extraction_prompt + html
        estimated_tokens = len(metadata_input.encode("utf-8"))
        if estimated_tokens > SAFE_INPUT_TOKEN_LIMIT:
            logger.info(
                "Article %s metadata input is too large for one request "
                "(%d estimated tokens); splitting HTML into two parts",
                article_id,
                estimated_tokens,
            )
            paper_metadata = _extract_metadata_from_two_parts(metadata_llm, html, article_id)
        else:
            paper_metadata = invoke_llm_with_retry(
                metadata_llm,
                [HumanMessage(content=metadata_input)],
                operation=f"extract metadata for article {article_id}",
            )

        doi = find_doi_by_title(
            paper_metadata.paper_title,
            paper_metadata.publication_year,
        )
        openalex_metadata = get_openalex_metadata(
            doi=doi,
            title=paper_metadata.paper_title,
            publication_year=paper_metadata.publication_year,
        )
        domain = field = None
        source = paper_metadata.source
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
            domain = llm_classification.primary_domain
            field = llm_classification.primary_field

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
