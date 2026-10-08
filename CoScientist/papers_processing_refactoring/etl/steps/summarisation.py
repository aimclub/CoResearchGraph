from langchain_core.messages import HumanMessage

from ..base import ETLStep
from ..context import ETLContext
from ...utils.general_utils import ExpandedSummary, OpenAlexClassification
from ...utils.general_utils import invoke_llm_with_retry
from ...utils.prompts import summarisation_prompt, classification_prompt


class PaperSummarisatonStep(ETLStep):
    
    name = "paper_summarisation"
    
    def run(self, ctx: ETLContext) -> None:
        
        article_id = ctx.article.id
        
        html = ctx.artifact_store.get_html(article_id, "image_captioning")
        if not html:
            raise RuntimeError(f"{self.name} step requires cleaned HTML")
        
        manifest_data = ctx.artifact_store.get_metadata(article_id, "image_captioning") or dict()
        
        summary_llm = ctx.llm.with_structured_output(ExpandedSummary)
        expanded_summary: ExpandedSummary = invoke_llm_with_retry(
            summary_llm,
            [HumanMessage(content=summarisation_prompt + html)],
            operation=f"summarise article {article_id}",
        )
        
        prompt_content = classification_prompt.format(
            TITLE=expanded_summary.paper_title, PAPER_SUMMARY=expanded_summary.paper_summary,
        )
        classification_llm = ctx.llm.with_structured_output(OpenAlexClassification)
        classification: OpenAlexClassification = invoke_llm_with_retry(
            classification_llm,
            [HumanMessage(content=prompt_content)],
            operation=f"classify summarised article {article_id}",
        )
        
        manifest_data["summary"] = {
            "paper_summary": expanded_summary.paper_summary,
            "paper_title": expanded_summary.paper_title,
            "publication_year": expanded_summary.publication_year,
            "authors": expanded_summary.authors,
            "source": expanded_summary.source,
            "domain": classification.primary_domain,
            "field": classification.primary_field,
        }
        manifest_data["paper_of_file_name"] = ctx.article.name
        manifest_data["article_metadata"] = ctx.article.metadata
        
        ctx.artifact_store.put_html(article_id, self.name, html)
        if manifest_data:
            ctx.artifact_store.put_metadata(article_id, self.name, manifest_data)
