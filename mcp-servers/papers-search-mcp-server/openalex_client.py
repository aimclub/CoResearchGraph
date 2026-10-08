from __future__ import annotations

import time
import logging
import os
from urllib.parse import urljoin
import requests
from pprint import pprint

DOWNLOADED_PAPERS_PATH = os.environ.get("DOWNLOADED_PAPERS_PATH")
OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL")


class OpenAlexClient:
    """Client for interacting with the OpenAlex API."""

    BASE_URL = "https://api.openalex.org"

    def __init__(
        self,
        email: str = None,
        s3_service: S3BucketService = None,
    ) -> None:
        self.email = email

    def request_with_retry(
        self,
        endpoint: str,
        params: dict = None,
        max_retries: int = 3,
        timeout: int = 30,
        stream: bool = False,
    ) -> requests.Response:
        """Make an HTTP GET request with retry logic for rate limits and server errors."""
        url = endpoint if endpoint.startswith("http") else urljoin(self.BASE_URL, endpoint)
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=timeout, stream=stream)
                if response.status_code == 200:
                    return response
                if response.status_code == 403 or response.status_code >= 500:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    response.raise_for_status()
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logging.info(f"Retrying... Attempt {attempt + 2}")
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise Exception(f"Failed after {max_retries} retries")

    def search_works(
        self,
        keywords: str = None,
        author_id: str = None,
        institution_id: str = None,
        source_id: str = None,
        publication_year: str = None,
        open_access: bool = True,
        has_pdf: bool = True,
        limit: int = 10,
        sort: str = None
    ):
        """
        Search for works in OpenAlex based on various filters.

        Args:
            keywords: Search keywords for title and abstract, e.g. 'machine learning drug discovery'
            author_id: Filter by OpenAlex author ID
            institution_id: Filter by OpenAlex institution ID
            source_id: Filter by OpenAlex source ID
            publication_year: Filter by publication year (e.g., ">2020", "2020-2022")
            open_access: Whether to filter for open access papers
            has_pdf: Whether to filter for papers with PDF available
            limit: Number of results to return (max 200)
            sort: Sorting criterion (e.g., "publication_date", "cited_by_count:desc")
        """
        filters = []

        if keywords:
            keywords = keywords.translate(str.maketrans(",:|!+><*?~", "          "))
            keywords = " ".join(keywords.split())
            filters.append(f"fulltext.search:{keywords.replace(' ', '+')}")
        if author_id:
            filters.append(f"author.id:{author_id}")
        if institution_id:
            filters.append(f"institution.id:{institution_id}")
        if source_id:
            filters.append(f"primary_location.source.id:{source_id}")
        if publication_year:
            filters.append(f"publication_year:{publication_year}")
        if open_access:
            filters.append("is_oa:true")
        if has_pdf:
            filters.append("has_content.pdf:true")

        
        params = {"filter": ",".join(filters)}
        
        if sort:
            params["sort"] = sort
        
        params["per-page"] = limit

        response = self.request_with_retry(endpoint="works", params=params).json()

        return response

    def search_entity(
        self,
        entity_type: str,
        entity_name: str
    ) -> dict:
        """
        Search for an entity ID (author, source, institution) by its name in OpenAlex.

        Args:
            entity_type: Type of entity to search for ("author", "source", "institution")
            entity_name: Name of the entity to search for, e.g., author name, journal name,
            or institution name
        Returns:
            Dictionary containing the most relevant search result for the specified entity
        """
        endpoint_map = {
            "author": "authors",
            "source": "sources",
            "institution": "institutions"
        }
        if entity_type not in endpoint_map:
            raise ValueError(f"Unsupported entity type: {entity_type}")
        
        params = {"search": entity_name, "per-page": 1}
        response = self.request_with_retry(
            endpoint=endpoint_map[entity_type],
            params=params
        ).json()
        return response.get("results", [])[0]


if __name__ == "__main__":
    client = OpenAlexClient(email=OPENALEX_EMAIL)
    # Example works search:
    result = client.search_works(
        institution_id="i173089394",  # Replace with a valid institution ID
        publication_year="2025",
        limit=1,
        sort="cited_by_count:desc"
    )
    # Example entity search:
    # result = client.search_entity(
    #     entity_type="institution",
    #     entity_name="ITMO University"
    # )
    pprint(result)