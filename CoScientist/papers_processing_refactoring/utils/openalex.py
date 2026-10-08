import logging
import re
import time
from typing import Any

import requests

from CoScientist.paper_analysis.research_taxonomy import DOMAIN_TO_SUBDOMAINS
from CoScientist.papers_processing_refactoring.app.settings import OpenAlexSettings

UNKNOWN_PUBLICATION_YEAR = 9999
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
logger = logging.getLogger(__name__)


def _normalize_title(title: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", title.casefold()).split())


def _normalize_doi(doi: str) -> str:
    return re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
        "",
        doi.strip(),
        flags=re.IGNORECASE,
    ).lower()


def _request_json(
    url: str,
    params: dict[str, str],
    *,
    max_retries: int = 3,
) -> dict[str, Any]:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

        if response.status_code == 200:
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(f"{url} returned an unexpected response")
            return payload

        is_retryable = response.status_code == 429 or response.status_code >= 500
        if is_retryable and attempt < max_retries - 1:
            time.sleep(2 ** attempt)
            continue

        response.raise_for_status()

    raise RuntimeError(f"{url} request failed after retries")


def _openalex_params(params: dict[str, str]) -> dict[str, str]:
    params = dict(params)
    settings = OpenAlexSettings()
    if settings.email:
        params["mailto"] = settings.email
    if settings.api_key:
        params["api_key"] = settings.api_key.get_secret_value()
    return params


def _exact_openalex_work(
    results: list[Any],
    title: str,
    publication_year: int,
) -> dict[str, Any] | None:
    normalized_title = _normalize_title(title)
    exact_matches = [
        work
        for work in results
        if isinstance(work, dict)
        and isinstance(work.get("title"), str)
        and _normalize_title(work["title"]) == normalized_title
    ]
    if publication_year != UNKNOWN_PUBLICATION_YEAR:
        year_matches = [work for work in exact_matches if work.get("publication_year") == publication_year]
        if year_matches:
            return year_matches[0]
    return exact_matches[0] if exact_matches else None


def _crossref_publication_year(work: dict[str, Any]) -> int | None:
    for key in ("published", "published-print", "published-online", "issued"):
        date = work.get(key)
        date_parts = date.get("date-parts") if isinstance(date, dict) else None
        if (
            isinstance(date_parts, list)
            and date_parts
            and isinstance(date_parts[0], list)
            and date_parts[0]
            and isinstance(date_parts[0][0], int)
        ):
            return date_parts[0][0]
    return None


def _find_doi_in_openalex(title: str, publication_year: int) -> str | None:
    payload = _request_json(
        OPENALEX_WORKS_URL,
        _openalex_params({"search": title, "per-page": "10"}),
    )
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("OpenAlex response has no works list")
    work = _exact_openalex_work(results, title, publication_year)
    doi = work.get("doi") if work else None
    return _normalize_doi(doi) if isinstance(doi, str) and doi.strip() else None


def _find_doi_in_crossref(title: str, publication_year: int) -> str | None:
    params = {"query.title": title, "rows": "10"}
    settings = OpenAlexSettings()
    if settings.email:
        params["mailto"] = settings.email
    if publication_year != UNKNOWN_PUBLICATION_YEAR:
        params["filter"] = (
            f"from-pub-date:{publication_year}-01-01,"
            f"until-pub-date:{publication_year}-12-31"
        )

    payload = _request_json(CROSSREF_WORKS_URL, params)
    message = payload.get("message")
    works = message.get("items") if isinstance(message, dict) else None
    if not isinstance(works, list):
        raise RuntimeError("Crossref response has no works list")

    normalized_title = _normalize_title(title)
    exact_matches = []
    for work in works:
        if not isinstance(work, dict):
            continue
        titles = work.get("title")
        candidate_title = titles[0] if isinstance(titles, list) and titles else None
        if not isinstance(candidate_title, str) or _normalize_title(candidate_title) != normalized_title:
            continue
        exact_matches.append(work)
        if publication_year != UNKNOWN_PUBLICATION_YEAR and _crossref_publication_year(work) == publication_year:
            doi = work.get("DOI")
            if isinstance(doi, str) and doi.strip():
                return _normalize_doi(doi)

    for work in exact_matches:
        doi = work.get("DOI")
        if isinstance(doi, str) and doi.strip():
            return _normalize_doi(doi)
    return None


def find_doi_by_title(title: str, publication_year: int) -> str | None:
    """Find a DOI by exact title in OpenAlex, using Crossref as fallback."""
    try:
        doi = _find_doi_in_openalex(title, publication_year)
    except (requests.exceptions.RequestException, RuntimeError, ValueError) as exc:
        logger.warning("OpenAlex DOI lookup failed for %r: %s", title, exc)
        doi = None
    if doi is None:
        try:
            doi = _find_doi_in_crossref(title, publication_year)
        except (requests.exceptions.RequestException, RuntimeError, ValueError) as exc:
            logger.warning("Crossref DOI lookup failed for %r: %s", title, exc)
    if doi is None:
        logger.error("DOI was not found for %r in OpenAlex or Crossref", title)
    return doi


def _find_openalex_work_by_doi(doi: str) -> dict[str, Any] | None:
    payload = _request_json(
        OPENALEX_WORKS_URL,
        _openalex_params({"filter": f"doi:{_normalize_doi(doi)}", "per-page": "1"}),
    )
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("OpenAlex response has no works list")
    return results[0] if results and isinstance(results[0], dict) else None


def _find_openalex_work_by_title(title: str, publication_year: int) -> dict[str, Any] | None:
    payload = _request_json(
        OPENALEX_WORKS_URL,
        _openalex_params({"search": title, "per-page": "10"}),
    )
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("OpenAlex response has no works list")
    return _exact_openalex_work(results, title, publication_year)


def _classification_from_work(work: dict[str, Any] | None) -> tuple[str, str] | None:
    if work is None:
        return None

    primary_topic = work.get("primary_topic")
    if not isinstance(primary_topic, dict):
        return None

    domain = primary_topic.get("domain")
    field = primary_topic.get("field")
    domain_name = domain.get("display_name") if isinstance(domain, dict) else None
    field_name = field.get("display_name") if isinstance(field, dict) else None
    if not isinstance(domain_name, str) or not isinstance(field_name, str):
        return None
    if field_name not in DOMAIN_TO_SUBDOMAINS.get(domain_name, []):
        logger.warning(
            "OpenAlex returned a domain/field pair outside the configured taxonomy: %s / %s",
            domain_name,
            field_name,
        )
        return None
    return domain_name, field_name


def _source_from_work(work: dict[str, Any]) -> str | None:
    locations = [work.get("primary_location"), work.get("best_oa_location")]
    raw_locations = work.get("locations")
    if isinstance(raw_locations, list):
        locations.extend(raw_locations)

    for location in locations:
        if not isinstance(location, dict):
            continue
        source = location.get("source")
        source_name = source.get("display_name") if isinstance(source, dict) else None
        if isinstance(source_name, str) and source_name.strip():
            return source_name.strip()
    return None


def _metadata_from_work(
    work: dict[str, Any] | None,
) -> tuple[str | None, str | None, str | None] | None:
    if work is None:
        return None

    classification = _classification_from_work(work)
    domain, field = classification if classification is not None else (None, None)
    return domain, field, _source_from_work(work)


def get_openalex_metadata(
    *,
    doi: str | None,
    title: str,
    publication_year: int,
) -> tuple[str | None, str | None, str | None] | None:
    """Find domain, field, and source by DOI first, then by exact title."""
    if doi:
        try:
            metadata = _metadata_from_work(_find_openalex_work_by_doi(doi))
            if metadata is not None:
                return metadata
        except (requests.exceptions.RequestException, RuntimeError, ValueError) as exc:
            logger.warning("OpenAlex metadata lookup by DOI failed for %s: %s", doi, exc)

    try:
        return _metadata_from_work(_find_openalex_work_by_title(title, publication_year))
    except (requests.exceptions.RequestException, RuntimeError, ValueError) as exc:
        logger.warning("OpenAlex metadata lookup by title failed for %r: %s", title, exc)
        return None
