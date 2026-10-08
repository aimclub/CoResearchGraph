import hashlib
import re
from typing import List, Optional

from langchain_text_splitters import HTMLSemanticPreservingSplitter

from ...domain.entities import Chunk, ChunkRole


def _custom_table_extractor(table_tag):
    return str(table_tag).replace("\n", "")


def chunk_html_to_chunks(
    html_string: str,
    article_id: str,
    article_domain: Optional[str|None],
    article_field: Optional[str|None],
) -> List[Chunk]:

    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
    ]

    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=headers_to_split_on,
        max_chunk_size=2500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". "],
        elements_to_preserve=["ul", "table", "ol"],
        preserve_images=True,
        custom_handlers={"table": _custom_table_extractor},
    )

    documents = splitter.split_text(html_string)

    chunks: list[Chunk] = []

    for idx, doc in enumerate(documents):
        text = doc.page_content.strip()
        if not text:
            continue

        imgs_in_chunk = extract_img_url(text)

        chunk_id = make_chunk_id(
            article_id,
            ChunkRole.BODY,
            idx,
            text,
        )

        chunks.append(
            Chunk(
                id=chunk_id,
                article_id=article_id,
                domain=article_domain or "default",
                field=article_field or "default",
                role=ChunkRole.BODY,
                modality="text",
                content=text,
                metadata={
                    "imgs_in_chunk": imgs_in_chunk,
                },
            )
        )

    return chunks


def make_chunk_id(
    article_id: str,
    role: ChunkRole,
    idx: int,
    content: str,
) -> str:
    h = hashlib.md5(content.encode("utf-8")).hexdigest()
    return f"{article_id}:{role.value}:{idx}:{h}"


def extract_img_url(doc_text: str) -> list[str]:
    pattern = r'!\[image:([^\]]+\.jpeg)\]\(([^)]+\.jpeg)\)'
    matches = re.findall(pattern, doc_text)
    return [entry[0] for entry in matches]
