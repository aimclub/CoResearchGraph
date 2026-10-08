import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from ..base import ETLStep
from ..context import ETLContext
from ...domain.entities import ImageInfo
from ...utils.html_processing.images import (
    pil_to_base64,
    check_image_relevance,
    try_extract_table
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageFilteringStep(ETLStep):
    
    name = "image_filtering"

    def run(self, ctx: ETLContext) -> None:
        
        article_id = ctx.article.id
        
        html = ctx.artifact_store.get_html(article_id, "html_cleaning")
        if not html:
            raise RuntimeError(f"{self.name} step requires cleaned HTML")
        
        soup = BeautifulSoup(html, "lxml")
        
        kept_images_manifest = []
        images_to_save = {}
        
        for img_tag in soup.find_all('img'):
            img_src: Any = img_tag.get("src")
            if not img_src:
                continue
            
            file_name = Path(urlparse(img_src).path).name
            
            pil_img = ctx.artifact_store.get_image(article_id, "parsing", file_name)
            if not pil_img:
                logger.warning(
                    f"[{self.name}] Image {file_name} from article [{article_id}] not found in store."
                )
                continue
            
            image_b64 = pil_to_base64(pil_img)
            parent_p = img_tag.find_parent('p')
            
            if not check_image_relevance(image_b64, ctx.llm):
                if parent_p:
                    parent_p.decompose()
                else:
                    img_tag.decompose()
                continue
            
            table_html = try_extract_table(image_b64, ctx.llm)
            if table_html:
                table_soup = BeautifulSoup(table_html, 'html.parser')
                if parent_p:
                    parent_p.replace_with(table_soup)
                else:
                    img_tag.replace_with(table_soup)
                continue
            
            img_id = file_name.rsplit(".", 1)[0]
            
            img_info = ImageInfo(
                id=img_id,
                file_name=file_name,
                original_src=img_src
            )
            
            kept_images_manifest.append(img_info)
            images_to_save[file_name] = pil_img
        
        ctx.artifact_store.put_html(article_id, self.name, str(soup.prettify()))
        
        if images_to_save:
            ctx.artifact_store.put_images(article_id, self.name, images_to_save)
        
        manifest_data = {"images": [img.model_dump() for img in kept_images_manifest]}
        ctx.artifact_store.put_metadata(article_id, self.name, manifest_data)
