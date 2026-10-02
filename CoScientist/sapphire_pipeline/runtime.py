"""Adapter to papers_processing_refactoring; imported only by the actual run command."""
from __future__ import annotations

import sys
import types
from pathlib import Path

# Follow the standalone ETL entry point: avoid initializing the agent application.
if 'CoScientist' not in sys.modules:
    package = types.ModuleType('CoScientist')
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    package.__package__ = 'CoScientist'
    sys.modules['CoScientist'] = package

from CoScientist.papers_processing_refactoring.app.main_process import (
    build_shared_services, build_state_store, settings,
)
from CoScientist.papers_processing_refactoring.domain.entities import Article
from CoScientist.papers_processing_refactoring.etl import (
    ETLContext, ETLPipeline, FetchStep, ParseStep, HtmlCleaningStep,
    ImageFilteringStep, ImageCaptioningStep, MetadataExtractionStep,
    ChunkingStep, EmbeddingStep, PublishStep,
)

from .client import normalize_doi


from .etl_adapters import S3Source, AddIngestionSource


def doi_query_values(value):
    """Return exact Chroma values for common canonical and legacy DOI forms."""
    doi = normalize_doi(value)
    if not doi:
        return []
    suffixes = (doi, doi.upper())
    prefixes = (
        '', 'https://doi.org/', 'http://doi.org/',
        'https://dx.doi.org/', 'http://dx.doi.org/', 'doi:',
    )
    return sorted({form + suffix
                   for prefix in prefixes
                   for form in (prefix, prefix.upper())
                   for suffix in suffixes})


class RAGBackend:
    def __init__(self):
        """Create ETL services without scanning the Chroma collection at startup."""
        self.settings = settings
        self.services = build_shared_services(settings)
        self.public = self.services['public_store']
        self.collection = self.services['vector_store'].collection
        self.article_ids = set()
        self.dois = set()

    def remember(self, metadata):
        """Add an article ID and normalized DOI to the in-memory duplicate index.

        Ignore missing identifiers and invalid DOI values.
        """
        if metadata.get('article_id'):
            self.article_ids.add(metadata['article_id'])
        doi = normalize_doi(metadata.get('doi'))
        if doi:
            self.dois.add(doi)

    def contains(self, metadata):
        """Check one article ID or DOI against Chroma and cache positive matches.

        OpenAlex IDs are not used for deduplication. DOI matching covers the
        canonical form and common legacy URL/case forms stored in Chroma.
        """
        article_id = metadata.get('article_id')
        if article_id:
            if article_id in self.article_ids:
                return True
            if self.collection.get(where={'article_id': article_id}, limit=1, include=[])['ids']:
                self.article_ids.add(article_id)
                return True

        doi = normalize_doi(metadata.get('doi'))
        if doi:
            if doi in self.dois:
                return True
            if self.collection.get(
                where={'doi': {'$in': doi_query_values(doi)}}, limit=1, include=[],
            )['ids']:
                self.dois.add(doi)
                return True
        return False

    def upload(self, key, pdf):
        """Write PDF bytes to the supplied key in the configured public S3 bucket."""
        self.public.client.put_object(
            Bucket=self.public.bucket, Key=key, Body=pdf, ContentType='application/pdf',
        )

    def ingest(self, article_id, metadata):
        """Run the original papers ETL for a PDF already uploaded to S3.

        Args:
            article_id: MD5 of the PDF bytes, used for state and Chroma records.
            metadata: Article metadata including domain and s3_key.

        Reuse intermediate states, or clear a previous completed publication state
        when reprocessing. After ETL, verify that Chroma contains article chunks
        and remember their identity. Service and processing errors propagate.
        """
        article = Article(
            id=article_id, name=metadata.get('paper_title') or '',
            domain=metadata.get('domain') or 'Unclassified', metadata=metadata,
            source_type='remote', source_ref=f"s3://{self.public.bucket}/{metadata['s3_key']}",
        )
        with build_state_store(self.settings) as state:
            # A completed ETL whose vectors were removed must be rebuilt.
            if state.get_status(article_id, 'publish') == 'done':
                state.clear_data(article_id)
            ctx = ETLContext(
                article=article, state_manager=state,
                artifact_store=self.services['artifact_store'], public_store=self.public,
                vector_store=self.services['vector_store'], llm=self.services['llm_model'],
                embedding_model=self.services['embedding_model'],
            )
            ETLPipeline([
                FetchStep(S3Source(self.public)), ParseStep(), HtmlCleaningStep(),
                ImageFilteringStep(), ImageCaptioningStep(), MetadataExtractionStep(),
                ChunkingStep(), AddIngestionSource(),
                EmbeddingStep(), PublishStep(),
            ]).run(ctx)
            if not self.collection.get(where={'article_id': article_id}, limit=1, include=[])['ids']:
                raise RuntimeError('ETL completed without publishing vectors')
        self.remember(dict(metadata, article_id=article_id))
