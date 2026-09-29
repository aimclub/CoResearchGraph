from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def openalex_id(value):
    """Return a canonical W-prefixed work ID from an OpenAlex ID or URL.

    Return None when the value is not a supported work identifier.
    """
    if not isinstance(value, str):
        return None
    value = value.strip().rstrip('/')
    if value.startswith(('http://openalex.org/', 'https://openalex.org/')):
        value = value.rsplit('/', 1)[-1]
    return value.upper() if re.fullmatch(r'W\d+', value, re.I) else None


def normalize_doi(value):
    """Strip known DOI prefixes and lowercase the identifier.

    Return None for non-string values or values without the expected DOI shape.
    """
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    for prefix in ('https://doi.org/', 'http://doi.org/', 'https://dx.doi.org/', 'http://dx.doi.org/', 'doi:'):
        if value.startswith(prefix):
            value = value[len(prefix):].strip()
            break
    return value if value.startswith('10.') and '/' in value else None


class SapphireClient:
    def __init__(self, url, page_size, timeout, max_pdf_bytes,
                 pdf_crawler_url, pdf_crawler_username, pdf_crawler_password, pdf_crawler_timeout,
                 openalex_email, openalex_api_key):
        """Create a reusable HTTP session with retries for transient GET failures.

        Args:
            url: Base Sapphire API URL.
            page_size: Number of publications requested per page.
            timeout: Connection and read timeout in seconds for each request.
            max_pdf_bytes: Maximum allowed size of a downloaded PDF in bytes.
            pdf_crawler_url: Base URL of the PDF Crawler Service.
            pdf_crawler_username: Optional HTTP Basic Auth username.
            pdf_crawler_password: Optional HTTP Basic Auth password.
            pdf_crawler_timeout: Timeout for synchronous PDF downloads in seconds.
            openalex_email: Contact email for direct OpenAlex downloads.
            openalex_api_key: API key for direct OpenAlex downloads.

        Raises:
            ValueError: If page_size, timeout, or max_pdf_bytes is not positive.
        """
        self.url = url.rstrip('/')
        self.page_size = page_size
        self.timeout = timeout
        self.max_pdf_bytes = max_pdf_bytes
        self.openalex_email = openalex_email
        self.openalex_api_key = openalex_api_key

        self.pdf_crawler_url = pdf_crawler_url.rstrip('/') + '/api/v1/download'
        self.pdf_crawler_auth = (pdf_crawler_username, pdf_crawler_password) if pdf_crawler_username else None
        self.pdf_crawler_timeout = pdf_crawler_timeout
        self.session = requests.Session()
        # A 429 may include Retry-After of several hours; never sleep inside a
        # single PDF attempt. The pipeline records rate limits for a later run.
        retry = Retry(
            total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504],
            allowed_methods=['GET'], respect_retry_after_header=False,
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retry))
        self.session.mount('https://', HTTPAdapter(max_retries=retry))
        # A synchronous download can be expensive; retry on a later pipeline run.
        self.session.mount(self.pdf_crawler_url, HTTPAdapter(max_retries=0))
        # Retry logs can expose URLs containing API keys.
        self.session.mount('https://api.openalex.org/', HTTPAdapter(max_retries=0))
        self.session.mount('https://content.openalex.org/', HTTPAdapter(max_retries=0))

    def close(self):
        """Close the HTTP session and release its pooled connections."""
        self.session.close()

    def get_json(self, path, **params):
        """GET an API path with query parameters and return its decoded JSON.

        HTTP, transport, and JSON decoding errors propagate to the caller.
        """
        with self.session.get(self.url + path, params=params, timeout=self.timeout) as response:
            response.raise_for_status()
            return response.json()

    def publications(self):
        """Yield positive-score AI and chembio publications from paginated API queries.

        Alternate pages between the two domains so a run limited by
        ``max_articles`` is not filled by one domain first. Remove publications
        returned by both queries. The Sapphire API has no open-access query
        parameter, so the pipeline validates that response field separately.
        """
        states = {
            domain: {'offset': 0, 'previous': None}
            for domain in ('ai', 'chembio')
        }
        seen = set()
        while states:
            for domain in tuple(states):
                state = states[domain]
                page = self.get_json(
                    '/publications',
                    domain=domain,
                    domain_score=1e-12,
                    limit=self.page_size,
                    offset=state['offset'],
                )
                if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
                    raise ValueError('/publications must return an array of objects')
                if not page:
                    del states[domain]
                    continue
                if page == state['previous']:
                    raise RuntimeError(
                        f'Sapphire returned the same page twice for {domain}; offset may be ignored'
                    )
                state['offset'] += len(page)
                state['previous'] = page
                for publication in page:
                    identity = self._publication_identity(publication)
                    if identity not in seen:
                        seen.add(identity)
                        yield publication

    @staticmethod
    def _publication_identity(publication):
        """Return the best available identity for cross-domain deduplication."""
        if publication.get('id') not in (None, ''):
            return 'id', str(publication['id'])
        identifier = openalex_id(publication.get('openalex_id'))
        if identifier:
            return 'openalex', identifier
        doi = normalize_doi(publication.get('doi'))
        if doi:
            return 'doi', doi
        return 'payload', json.dumps(
            publication, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str,
        )

    def work(self, identifier):
        """Fetch an OpenAlex work through Sapphire using its ID or OpenAlex URL.

        Raise ValueError for an invalid identifier, a non-object response, or a
        response whose supplied work ID differs from the requested ID.
        """
        identifier = openalex_id(identifier)
        if not identifier:
            raise ValueError('Invalid OpenAlex work ID')
        work = self.get_json(f'/crawler/openalex/works/{identifier}')
        if not isinstance(work, dict):
            raise ValueError('OpenAlex work response must be an object')
        if work.get('id') and openalex_id(work['id']) != identifier:
            raise ValueError('OpenAlex work ID does not match the requested publication')
        return work

    def download_pdf(self, url):
        """Request a PDF by DOI/DBLP URL from the PDF Crawler Service.

        HTTP 422 means the crawler could not download the article. Other HTTP
        errors and invalid responses propagate so the registry allows a retry.
        Validate the streamed body size and PDF signature before returning bytes.
        """
        parsed = urlsplit(url)
        if (parsed.scheme not in ('http', 'https') or parsed.username or parsed.password
                or parsed.hostname not in ('doi.org', 'dx.doi.org', 'dblp.org', 'www.dblp.org')):
            raise ValueError('PDF crawler requires a DOI or DBLP URL')
        try:
            with self.session.get(
                self.pdf_crawler_url, params={'url': url},
                auth=self.pdf_crawler_auth, timeout=self.pdf_crawler_timeout,
                stream=True, allow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    error = requests.HTTPError(f'{response.status_code} from PDF crawler')
                    error.status_code = response.status_code
                    raise error
                return self._read_pdf(response)
        except requests.HTTPError:
            raise
        except requests.RequestException as exc:
            raise type(exc)(f'PDF crawler request failed: {type(exc).__name__}') from None

    def _read_pdf(self, response):
        """Validate size and signature for either download route."""
        if int(response.headers.get('Content-Length', 0)) > self.max_pdf_bytes:
            raise ValueError('PDF exceeds size limit')
        data = bytearray()
        for chunk in response.iter_content(64 * 1024):
            data.extend(chunk)
            if len(data) > self.max_pdf_bytes:
                raise ValueError('PDF exceeds size limit')
            if len(data) >= 1024 and b'%PDF-' not in data[:1024]:
                raise ValueError('URL returned non-PDF content')
        if b'%PDF-' not in data[:1024]:
            raise ValueError('URL returned non-PDF content')
        return bytes(data)

    def download_pdf_direct(self, url):
        """Download a fallback PDF URL from Sapphire/OpenAlex metadata directly."""
        parsed = urlsplit(url)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or parsed.username or parsed.password):
            raise ValueError('PDF URL must be HTTP(S) without credentials')
        params = {}
        if parsed.scheme == 'https' and parsed.hostname in ('api.openalex.org', 'content.openalex.org'):
            if self.openalex_email:
                params['mailto'] = self.openalex_email
            if self.openalex_api_key:
                params['api_key'] = self.openalex_api_key
        try:
            with self.session.get(url, params=params, timeout=self.timeout, stream=True) as response:
                if response.status_code != 200:
                    error = requests.HTTPError(f'{response.status_code} downloading PDF from {parsed.hostname}')
                    error.status_code = response.status_code
                    raise error
                return self._read_pdf(response)
        except requests.HTTPError:
            raise
        except requests.RequestException as exc:
            raise type(exc)(f'PDF request to {parsed.hostname} failed: {type(exc).__name__}') from None
