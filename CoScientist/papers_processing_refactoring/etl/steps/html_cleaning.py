from ..base import ETLStep
from ..context import ETLContext
from ...utils.html_processing.cleaning import clean_html


class HtmlCleaningStep(ETLStep):
    
    name = "html_cleaning"
    
    def run(self, ctx: ETLContext) -> None:
        article_id = ctx.article.id
        
        parsed_html = ctx.artifact_store.get_html(article_id, "parsing")
        if not parsed_html:
            raise RuntimeError(f"{self.name} step requires HTML")
        
        cleaned_html = clean_html(parsed_html)
        
        ctx.artifact_store.put_html(article_id, self.name, cleaned_html)
