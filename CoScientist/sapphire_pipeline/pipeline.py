from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from urllib.parse import urlsplit

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
    """Return domain and field display names from a work primary topic.

    Raise ValueError if either name is missing or the domain is unsafe for
    use as a component of the initial S3 object key.
    """
    topic = work.get('primary_topic') or {}
    domain = (topic.get('domain') or {}).get('display_name')
    field = (topic.get('field') or {}).get('display_name')
    if not all(isinstance(value, str) and value.strip() for value in (domain, field)):
        raise ValueError('Work has no primary_topic.domain/field.display_name')
    if any(char in domain for char in ('/', '\\', '\x00')) or domain in ('.', '..'):
        raise ValueError('Domain is not a safe S3 path component')
    return domain, field


def download_source(publication, doi):
    """Use a DOI URL, or an explicit DBLP URL when no DOI is available."""
    if doi:
        return f'https://doi.org/{doi}'
    url = publication.get('dblp_url')
    if isinstance(url, str):
        url = url.strip()
        parsed = urlsplit(url)
        if (parsed.scheme in ('http', 'https') and parsed.hostname in ('dblp.org', 'www.dblp.org')
                and not parsed.username and not parsed.password):
            return url
    return None


def pdf_urls(publication, work):
    """Return deduplicated fallback URLs in the original download order."""
    candidates = [publication.get('pdf_url')]
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
            'sapphire_publication_id': str(publication.get('id') or ''),
        }
        if self.backend.contains(metadata):
            return 'already_in_rag'
        if publication.get('is_open_access') is not True:
            return 'not_open_access'
        if not identifier:
            return 'missing_openalex_id'
        work = self.client.work(identifier)
        domain, field = classification(work)
        metadata.update(domain=domain, field=field)
        metadata['doi'] = metadata['doi'] or normalize_doi(work.get('doi')) or ''
        if metadata['doi'] and metadata['doi'] != publication_doi and self.backend.contains(metadata):
            return 'already_in_rag'
        source_url = download_source(publication, metadata['doi'])
        pdf = None
        retry_error = None
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
            for url in pdf_urls(publication, work):
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
                source_url = url
                metadata['pdf_url'] = url
                break
        if pdf is None:
            if retry_error is not None:
                raise retry_error
            return 'no_usable_pdf'
        # Same content ID as LocalSource: catches previously ingested local PDFs.
        article_id = hashlib.md5(pdf).hexdigest()
        if self.backend.contains({'article_id': article_id}):
            return 'already_in_rag'
        key = f'articles/{domain}/{article_id}/paper.pdf'
        metadata.update(download_source_url=source_url, s3_key=key, ingestion_source='sapphire')
        self.backend.upload(key, pdf)
        self.backend.ingest(article_id, publication.get('name') or work.get('display_name') or identifier, metadata)
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
