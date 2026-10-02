from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter

import requests

from .client import normalize_doi, openalex_id
from .registry import FINAL_STATUSES

logger = logging.getLogger(__name__)


def allowed_domains(publication):
    """Accept positive AI/chembio scores only when no photonics score is positive.

    Missing or malformed domain entries do not establish domain membership.
    These Sapphire scores are separate from the OpenAlex domain/field taxonomy.
    """
    domains = publication.get('domains')
    if not isinstance(domains, list):
        return False
    positive = set()
    for domain in domains:
        if not isinstance(domain, dict):
            continue
        name, score = domain.get('name'), domain.get('score')
        if (isinstance(name, str) and type(score) in (int, float)
                and math.isfinite(score) and score > 0):
            positive.add(name.strip().lower())
    return bool(positive & {'ai', 'chembio'}) and 'photonics' not in positive


def classification(work):
    """Return available domain and field display names from a work primary topic.

    Missing values are returned as None so the ETL can classify the paper with
    its LLM.
    """
    topic = work.get('primary_topic') or {}
    domain = (topic.get('domain') or {}).get('display_name')
    field = (topic.get('field') or {}).get('display_name')
    domain = domain.strip() if isinstance(domain, str) and domain.strip() else None
    field = field.strip() if isinstance(field, str) and field.strip() else None
    return domain, field


def _source_name(container, name_field):
    """Return a venue name only when source type is journal or conference."""
    if not isinstance(container, dict):
        return None
    source_type = container.get('type')
    if not isinstance(source_type, str) or source_type.strip().lower() not in {'journal', 'conference'}:
        return None
    value = container.get(name_field)
    return value.strip() if isinstance(value, str) and value.strip() else None


def publication_bibliographic_metadata(publication):
    """Normalize bibliographic fields already present in `/publications`."""
    metadata = {}

    name = publication.get('name')
    if isinstance(name, str) and name.strip():
        metadata['paper_title'] = name.strip()

    publication_date = publication.get('publication_date')
    if isinstance(publication_date, str):
        year = publication_date.strip().split('-')[0]
        if year.isdigit() and int(year) > 0:
            metadata['publication_year'] = int(year)

    author_names = []
    authors = publication.get('authors')
    if isinstance(authors, str) and authors.strip():
        author_names.append(authors.strip())
    elif isinstance(authors, list):
        for author in authors:
            if isinstance(author, str) and author.strip():
                author_names.append(author.strip())
                continue
            if not isinstance(author, dict):
                continue
            name = author.get('name') or author.get('display_name')
            if not isinstance(name, str) or not name.strip():
                parts = [author.get(key) for key in ('first_name', 'middle_name', 'last_name')]
                name = ' '.join(part.strip() for part in parts if isinstance(part, str) and part.strip())
            if isinstance(name, str) and name.strip():
                author_names.append(name.strip())
    if author_names:
        metadata['authors'] = ', '.join(author_names)

    source = None
    for field in ('journal', 'conference'):
        venue = publication.get(field)
        name = venue.get('name') if isinstance(venue, dict) else None
        if isinstance(name, str) and name.strip():
            source = name.strip()
            break
    source = source or _source_name(publication.get('source'), 'name')
    if source:
        metadata['source'] = source
    return metadata


def bibliographic_metadata(work):
    """Return normalized bibliographic fields available in an OpenAlex work.

    Keep only journal and conference sources. Repository locations are useful
    for downloading a PDF, but they are not the publication venue stored in
    chunk metadata.
    """
    metadata = {}

    for title in (work.get('title'), work.get('display_name')):
        if isinstance(title, str) and title.strip():
            metadata['paper_title'] = title.strip()
            break

    publication_year = work.get('publication_year')
    if type(publication_year) is int and publication_year > 0:
        metadata['publication_year'] = publication_year

    authors = []
    authorships = work.get('authorships')
    for authorship in authorships if isinstance(authorships, list) else []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get('author')
        name = author.get('display_name') if isinstance(author, dict) else None
        if not isinstance(name, str) or not name.strip():
            name = authorship.get('raw_author_name')
        if isinstance(name, str) and name.strip():
            authors.append(name.strip())
    if authors:
        metadata['authors'] = ', '.join(authors)

    locations = [work.get('primary_location'), work.get('best_oa_location')]
    raw_locations = work.get('locations')
    if isinstance(raw_locations, list):
        locations.extend(raw_locations)
    for location in locations:
        if not isinstance(location, dict):
            continue
        source_name = _source_name(location.get('source'), 'display_name')
        if source_name:
            metadata['source'] = source_name
            break

    return metadata


def _doi(metadata):
    """Return a normalized DOI from a top-level value or an ids object."""
    doi = normalize_doi(metadata.get('doi'))
    ids = metadata.get('ids')
    return doi or (normalize_doi(ids.get('doi')) if isinstance(ids, dict) else None)


def pdf_urls(work):
    """Return deduplicated direct PDF URLs from an OpenAlex card."""
    candidates = []
    content = work.get('content_urls')
    if isinstance(content, dict):
        candidates.append(content.get('pdf'))
    for name in ('best_oa_location', 'primary_location'):
        location = work.get(name)
        if isinstance(location, dict) and location.get('is_oa') is not False:
            candidates.append(location.get('pdf_url'))
    locations = work.get('locations')
    for location in locations if isinstance(locations, list) else []:
        if isinstance(location, dict) and location.get('is_oa') is True:
            candidates.append(location.get('pdf_url'))
    return list(dict.fromkeys(url.strip() for url in candidates if isinstance(url, str) and url.strip()))


class SapphirePipeline:
    """Backend implements contains(metadata), upload(key, bytes), ingest(...)."""

    def __init__(self, client, backend, registry=None):
        """Bind the Sapphire client, RAG backend, and optional status registry."""
        self.client = client
        self.backend = backend
        self.registry = registry

    def process(self, publication):
        """Process one publication and return its ingestion or skip reason.

        Check domain scores, DOI, open access, and the work ID; enrich metadata; request a PDF from the crawler;
        check the PDF hash; then upload to S3 and run ETL. Return ingested,
        domain_filtered, already_in_rag, not_open_access, missing_openalex_id, or no_usable_pdf.
        Try the PDF crawler first, then direct PDF URLs. Preserve transient failures for retry.
        """
        if not allowed_domains(publication):
            return 'domain_filtered'
        identifier = openalex_id(publication.get('openalex_id'))
        publication_doi = normalize_doi(publication.get('doi'))
        metadata = {
            'openalex_id': identifier or '',
            'doi': publication_doi or '',
        }
        if self.backend.contains(metadata):
            return 'already_in_rag'
        if publication.get('is_open_access') is not True:
            return 'not_open_access'
        metadata.update(publication_bibliographic_metadata(publication))

        work = None
        domain = field = None

        def load_work():
            """Fetch and apply the OpenAlex card at most once, only when needed."""
            nonlocal work, domain, field
            if work is not None:
                return work
            if not identifier:
                return None
            work = self.client.work(identifier)
            domain, field = classification(work)
            for name, value in bibliographic_metadata(work).items():
                metadata.setdefault(name, value)
            if domain is not None:
                metadata['domain'] = domain
            if field is not None:
                metadata['field'] = field
            return work

        pdf = None
        retry_error = None
        source_url = f'https://doi.org/{publication_doi}' if publication_doi else None

        if source_url is None:
            work = load_work()
            if work is None:
                return 'missing_openalex_id'
            metadata['doi'] = _doi(work) or ''
            if metadata['doi'] and self.backend.contains(metadata):
                return 'already_in_rag'
            source_url = (
                f'https://doi.org/{metadata["doi"]}' if metadata['doi'] else None
            )

        if source_url is not None:
            try:
                logger.info('Requesting PDF from crawler for %s', identifier)
                pdf = self.client.download_pdf(source_url)
            except (requests.RequestException, TimeoutError, ValueError) as error:
                if not (isinstance(error, requests.HTTPError) and getattr(error, 'status_code', None) == 422):
                    retry_error = error
                logger.warning('PDF crawler failed for %s; trying direct PDF URLs (%s)',
                               identifier, type(error).__name__)
        if pdf is None:
            work = load_work()
            if work is None:
                if retry_error is not None:
                    raise retry_error
                return 'missing_openalex_id'
            for url in pdf_urls(work):
                try:
                    pdf = self.client.download_pdf_direct(url)
                except (requests.RequestException, TimeoutError, ValueError) as error:
                    if isinstance(error, requests.HTTPError):
                        status = getattr(error, 'status_code', None)
                        if status is None or status in (401, 403, 408, 429) or status >= 500:
                            retry_error = error
                    elif isinstance(error, (requests.RequestException, TimeoutError)):
                        retry_error = error
                    logger.warning('Direct PDF unavailable for %s (%s)', identifier, type(error).__name__)
                    continue
                logger.info('Downloaded PDF directly for %s from %s', identifier, url)
                source_url = url
                break
        if pdf is None:
            if retry_error is not None:
                raise retry_error
            return 'no_usable_pdf'

        # Even when DOI downloading succeeds, enrich `/publications` metadata
        # from the card before ETL. load_work() keeps publication values and
        # fetches the card at most once; ETL fills any remaining gaps with LLM.
        load_work()

        # Same content ID as LocalSource: catches previously ingested local PDFs.
        article_id = hashlib.md5(pdf).hexdigest()
        if self.backend.contains({'article_id': article_id}):
            return 'already_in_rag'
        key = f'articles/{domain or "Unclassified"}/{article_id}/paper.pdf'
        metadata.update(download_source_url=source_url, s3_key=key, ingestion_source='sapphire')
        self.backend.upload(key, pdf)
        self.backend.ingest(article_id, metadata)
        return 'ingested'

    def run(self, max_articles=None):
        """Process publications sequentially and return counters by outcome.

        Args:
            max_articles: Maximum number of records to process in this run,
                including newly determined skips and failures. Publications with
                a final registry status do not consume the limit. None processes
                records until pagination ends.

        Count per-publication exceptions as failed and continue. Page-fetch errors
        propagate. Raise ValueError when a supplied limit is not positive.
        """
        if max_articles is not None and max_articles < 1:
            raise ValueError('max_articles must be positive')
        counts = Counter()
        attempted = 0
        publications = iter(self.client.publications())
        while max_articles is None or attempted < max_articles:
            try:
                publication = next(publications)
            except StopIteration:
                break
            counts['seen'] += 1
            registry_key = self.registry.register(publication) if self.registry else None
            previous_status = self.registry.status(registry_key) if self.registry else None
            if previous_status in FINAL_STATUSES:
                counts['registry_skipped'] += 1
                counts[f'registry_{previous_status}'] += 1
                logger.info(
                    '[scan %s] %s: registry_skip (%s)',
                    counts['seen'], publication.get('openalex_id'), previous_status,
                )
                continue
            attempted += 1
            progress = f'{attempted}/{max_articles}' if max_articles is not None else str(attempted)
            logger.info(
                "[%s] Processing publication: %s | %s",
                progress, publication.get('openalex_id'), publication.get('name') or 'Untitled',
            )
            if self.registry:
                self.registry.start_attempt(registry_key)
            try:
                outcome = self.process(publication)
            except Exception as error:
                logger.exception('Publication failed: %s', publication.get('openalex_id'))
                outcome = 'failed'
                if self.registry:
                    self.registry.finish(registry_key, outcome, f'{type(error).__name__}: {error}')
            else:
                if self.registry:
                    self.registry.finish(registry_key, outcome)
            counts[outcome] += 1
            logger.info('[%s] %s: %s', progress, publication.get('openalex_id'), outcome)
        return dict(counts)
