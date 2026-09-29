import os
import re
from io import BytesIO
import logging

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_request

from CoScientist.paper_parser.s3_connection import S3BucketService
from openalex_client import OpenAlexClient

def _s3_env(primary: str, fallback: str) -> str | None:
    """S3__* is the name every server and the main app use. The bare names stay
    as a fallback for one release, so an existing deployment keeps working."""
    return os.getenv(primary) or os.getenv(fallback)


s3_service = S3BucketService(
    endpoint=_s3_env("S3__ENDPOINT_URL", "ENDPOINT_URL"),
    access_key=_s3_env("S3__ACCESS_KEY", "ACCESS_KEY"),
    secret_key=_s3_env("S3__SECRET_KEY", "SECRET_KEY"),
    bucket_name=_s3_env("S3__BUCKET_NAME", "BUCKET_NAME"),
)

mcp = FastMCP("PapersSearch")


def _get_credentials(email: str | None = None, api_key: str | None = None) -> tuple[str | None, str | None]:
    """Return (email, api_key): tool args overlay headers, then environment."""
    header_email, header_key = None, None
    try:
        headers = get_http_request().headers
        header_email = headers.get("x-openalex-email")
        header_key = headers.get("x-openalex-api-key")
    except Exception:
        pass
    resolved_email = (
        email
        or header_email
        or os.getenv("OPENALEX_EMAIL")
        or os.getenv("SERVICES__OPENALEX_EMAIL")
    )
    resolved_key = (
        api_key
        or header_key
        or os.getenv("OPENALEX_API_KEY")
        or os.getenv("SERVICES__OPENALEX_API_KEY")
    )
    return resolved_email, resolved_key


def _sanitize_filename(name: str) -> str:
    if not name:
        return "unnamed_paper"
    return re.sub(r'[\\/*?:"<>|]', "", str(name))

@mcp.tool()
def search_entity(entity_type: str, entity_name: str, email: str = None, api_key: str = None) -> dict:
    """
    Search for an entity (author, source, institution) in OpenAlex and return its ID.
    This ID can be further used to search for papers using the search_papers or download_papers_from_search tools.

    Args:
        entity_type: Type of entity to search for ("author", "source", "institution")
        entity_name: Name of the entity to search for (e.g., author name, journal name, institution name)
        email: Optional OpenAlex mailto / polite-pool email (overrides headers/env).
        api_key: Optional OpenAlex API key (overrides headers/env).
    """
    email, api_key = _get_credentials(email, api_key)
    client = OpenAlexClient(email=email, api_key=api_key)
    result = client.search_entity(entity_type=entity_type, entity_name=entity_name)
    if result:
        return {'answer': f'Entity ID: {result["id"]}'}
    else:
        return {'answer': 'No matching entity found.'}


@mcp.tool()
def search_papers(
    keywords: str = None,
    author_id: str = None,
    institution_id: str = None,
    source_id: str = None,
    publication_year: str = None,
    open_access: bool = True,
    has_pdf: bool = True,
    limit: int = 10,
    sort: str = None,
    email: str = None,
    api_key: str = None,
) -> dict:
    """
    Search papers in OpenAlex using filters and return normalized metadata.

    Args:
        keywords: Search keywords (e.g., "crispr cas9")
        author_id: OpenAlex author ID (e.g., "A123...")
        institution_id: OpenAlex institution ID
        source_id: OpenAlex source ID
        publication_year: Year filter (e.g., "2025", ">2020")
        open_access: Include only open access works
        has_pdf: Include only works with PDF available
        limit: Max number of results
        sort: OpenAlex sort field (e.g., "cited_by_count:desc")
        email: Optional OpenAlex mailto / polite-pool email (overrides headers/env).
        api_key: Optional OpenAlex API key (overrides headers/env).
    """
    email, api_key = _get_credentials(email, api_key)
    client = OpenAlexClient(email=email, api_key=api_key)
    response = client.search_works(
        keywords=keywords,
        author_id=author_id,
        institution_id=institution_id,
        source_id=source_id,
        publication_year=publication_year,
        open_access=open_access,
        has_pdf=has_pdf,
        limit=limit,
        sort=sort,
    )

    works = response.get("results", [])
    if not works:
        return {"answer": "No papers found for the given filters.", "metadata": {"papers": []}}

    papers = []
    for work in works:
        location = work.get("primary_location") or {}
        papers.append(
            {
                "title": work.get("title"),
                "doi": work.get("doi"),
                "publication_year": work.get("publication_year"),
                "cited_by_count": work.get("cited_by_count"),
                "is_oa": work.get("is_oa"),
                "pdf_url": location.get("pdf_url")
            }
        )

    titles = [paper.get("title") for paper in papers if paper.get("title")]
    answer = "Found papers:\n" + "\n".join(titles)
    return {"answer": answer, "metadata": {"papers": papers}}


@mcp.tool()
def download_papers_from_search(
    keywords: str = None,
    author_id: str = None,
    institution_id: str = None,
    source_id: str = None,
    publication_year: str = None,
    open_access: bool = True,
    limit: int = 10,
    sort: str = None,
    session_id: str = "1",
    user_id: str = "1",
    email: str = None,
    api_key: str = None,
) -> dict:
    """Search papers in OpenAlex and upload found PDFs directly to S3.

    Optional ``email`` / ``api_key`` overlay headers and environment credentials.

    Each uploaded paper carries ``bucket``, ``s3_key`` and ``presigned_url``. The
    bucket and the key are the durable reference. The URL expires in one hour.
    """
    email, api_key = _get_credentials(email, api_key)
    client = OpenAlexClient(email=email, api_key=api_key)
    response = client.search_works(
        keywords=keywords,
        author_id=author_id,
        institution_id=institution_id,
        source_id=source_id,
        publication_year=publication_year,
        open_access=open_access,
        has_pdf=True,
        limit=limit,
        sort=sort,
    )

    works = response.get("results", [])
    if not works:
        return {"answer": "No papers found for the given filters.", "metadata": {"papers": []}}

    logging.info(f"Found {len(works)} papers with PDFs available for download.")
    # The ephemeral/ top segment is what the bucket lifecycle rule filters on.
    # Objects written at the old prefix matched no rule and never expired.
    s3_prefix = f"ephemeral/{user_id}/{session_id}/papers_search_results/"
    s3_client = s3_service.create_s3_client()
    uploaded = []

    for index, work in enumerate(works):
        content_urls = work.get("content_urls") or {}
        pdf_url = content_urls.get("pdf")
        if not pdf_url:
            continue

        title = work.get("title") or f"paper_{work.get('id', index)}"
        file_name = f"{_sanitize_filename(str(title))}.pdf"
        s3_key = f"{s3_prefix.rstrip('/')}/{file_name}"

        response = client.request_with_retry(endpoint=pdf_url, stream=True)
        destination_path = f"{s3_prefix.rstrip('/')}/{file_name}"
        s3_client.upload_fileobj(BytesIO(response.content), s3_service.bucket_name, destination_path)
        logging.info(f"Uploaded paper '{title}' to S3 at {destination_path}")

        uploaded.append(
            {
                "id": work.get("id"),
                "paper_title": title,
                "pdf_url": pdf_url,
                "bucket": s3_service.bucket_name,
                "s3_key": s3_key,
                "presigned_url": s3_service.generate_presigned_url(s3_key, expiration=3600),
                "doi": work.get("doi"),
                "publication_year": work.get("publication_year"),
            }
        )

    if not uploaded:
        return {"answer": "No files were uploaded.", "metadata": {"papers": []}}

    titles = [item["paper_title"] for item in uploaded]
    answer = "Papers were uploaded to S3:\n" + "\n".join(titles)
    return {"answer": answer, "metadata": {"papers": uploaded}}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7333"))
    mcp.run(transport="http", host="127.0.0.1", port=port, path="/mcp")
