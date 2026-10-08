from ..base import ETLStep
from ..context import ETLContext
from ...domain.entities import Chunk, ChunkRole, ImageInfo
from ...utils.html_processing.splitting import chunk_html_to_chunks, make_chunk_id


class ChunkingStep(ETLStep):
    
    name = "chunking"

    def run(self, ctx: ETLContext) -> None:
        
        article_id = ctx.article.id
        
        html = ctx.artifact_store.get_html(article_id, "metadata_extraction")
        if not html:
            raise RuntimeError(f"{self.name} step requires cleaned HTML")
        
        manifest_data = ctx.artifact_store.get_metadata(article_id, "metadata_extraction")
        if not manifest_data:
            raise RuntimeError(f"{self.name} step requires processing metadata")
        
        paper_metadata = manifest_data["paper_metadata"]
        images_data = [ImageInfo(**data) for data in manifest_data.get("images", [])]

        body_chunks = chunk_html_to_chunks(
            html, article_id, paper_metadata["domain"], paper_metadata["field"]
        )
        common_metadata = {
            "paper_title": paper_metadata["paper_title"],
            "publication_year": paper_metadata["publication_year"],
            "authors": paper_metadata["authors"],
            "source": paper_metadata["source"],
            "doi": paper_metadata["doi"],
        }
        for chunk in body_chunks:
            chunk.metadata = {**(chunk.metadata or {}), **common_metadata}

        image_caption_chunks = []
        for idx, img in enumerate(images_data):
            if not img.caption:
                continue

            text = img.caption.strip()
            if not text:
                continue

            chunk_id = make_chunk_id(
                article_id,
                ChunkRole.IMAGE_CAPTION,
                idx,
                text,
            )

            image_caption_chunks.append(
                Chunk(
                    id=chunk_id,
                    article_id=article_id,
                    domain=paper_metadata["domain"] or "default",
                    field=paper_metadata["field"] or "default",
                    role=ChunkRole.IMAGE_CAPTION,
                    modality="text",
                    content=text,
                    metadata={"image_id": img.id, **common_metadata},
                )
            )
        
        # Summary chunk creation is intentionally disabled.
        # summary_chunk = Chunk(
        #     id=make_chunk_id(article_id, ChunkRole.SUMMARY, 1, summary_data["paper_summary"]),
        #     article_id=article_id,
        #     domain=summary_data["domain"] or "default",
        #     field=summary_data["field"] or "default",
        #     role=ChunkRole.SUMMARY,
        #     modality="text",
        #     content=summary_data["paper_summary"],
        #     metadata={
        #         "paper_title": summary_data["paper_title"],
        #         "publication_year": summary_data["publication_year"],
        #         "authors": summary_data["authors"],
        #         "source": summary_data["source"],
        #     },
        # )

        ctx.chunks = {
            ChunkRole.BODY: body_chunks,
            ChunkRole.IMAGE_CAPTION: image_caption_chunks,
            # ChunkRole.SUMMARY: [summary_chunk],  # Disabled.
        }
