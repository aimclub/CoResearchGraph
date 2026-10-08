from __future__ import annotations

import logging
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def _bootstrap_coscientist_package() -> None:
    """Registers a lightweight wrapper for the CoScientist package to ensure that importing sub-packages does not
    execute `CoScientist/__init__.py` in the execution environment.

    It does not alter `sys.path` (there are no name conflicts with the standard library) and only takes effect when
    CoScientist has not yet been imported.
    """
    if "CoScientist" in sys.modules:
        return

    # main_process.py -> app -> papers_processing_refactoring -> CoScientist
    coscientist_dir = Path(__file__).resolve().parents[2]

    package = types.ModuleType("CoScientist")
    package.__path__ = [str(coscientist_dir)]
    package.__package__ = "CoScientist"
    package.__file__ = str(coscientist_dir / "__init__.py")
    sys.modules["CoScientist"] = package


_bootstrap_coscientist_package()

from CoScientist.papers_processing_refactoring.app.config_loader import get_settings
from CoScientist.papers_processing_refactoring.etl import *
from CoScientist.papers_processing_refactoring.embeddings import *
from CoScientist.papers_processing_refactoring.scheduling.scheduler import IngestionScheduler, Schedule
from CoScientist.papers_processing_refactoring.sources.local import LocalSource
from CoScientist.papers_processing_refactoring.storage.state import *
from CoScientist.papers_processing_refactoring.storage.artifacts import *
from CoScientist.papers_processing_refactoring.storage.vector import *
from CoScientist.papers_processing_refactoring.definitions import CONFIG_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_WORKERS = 3
SLEEP_TIME = 10
POOLING_TIME_MINUTES = 1

load_dotenv(CONFIG_PATH)
settings = get_settings()


def build_vector_store(etl_settings):
    if etl_settings.vectordb.backend == "chromadb":
        return ChromaVectorStore(
            etl_settings.vectordb.chroma.host,
            etl_settings.vectordb.chroma.port,
            etl_settings.vectordb.chroma.collection
        )
    else:
        raise ValueError("Vector store configuration must be provided")
    

def build_artifacts_stores(etl_settings):
    etl_art_store = S3ETLArtifactStore(
        endpoint=etl_settings.s3.endpoint,
        access_key=etl_settings.s3.access_key,
        secret_key=etl_settings.s3.secret_key.get_secret_value(),
        bucket=etl_settings.s3.etl_bucket
    )
    public_art_store = S3DomainArtifactStore(
        endpoint=etl_settings.s3.endpoint,
        access_key=etl_settings.s3.access_key,
        secret_key=etl_settings.s3.secret_key.get_secret_value(),
        bucket=etl_settings.s3.public_bucket
    )
    return etl_art_store, public_art_store


def build_state_store(etl_settings):
    if etl_settings.database.type == "sqlite":
        return SQLiteStateManager(etl_settings.database.sqlite_path)
    elif etl_settings.database.type == "postgres":
        return PostgreSQLStateManager(etl_settings.database.postgresql_dsn)
    else:
        raise ValueError("State store configuration must be provided")


def build_shared_services(etl_settings):
    """Creates clients without per-article state (vector database, S3, LLM, embeddings), which can be safely reused
    across all worker threads. The state manager is not included here — it has its own connection to the database,
    so it is recreated for each article in `process_single_article`.
    """
    embedding_model = create_embedding_model({
        "type": etl_settings.embeddings.type,
        "url": etl_settings.embeddings.api_url,
        "model_name": etl_settings.embeddings.model_name,
        "batch_size": etl_settings.embeddings.batch_size,
    })
    artifact_store, public_store = build_artifacts_stores(etl_settings)
    return {
        "vector_store": build_vector_store(etl_settings),
        "artifact_store": artifact_store,
        "public_store": public_store,
        "llm_model": ChatOpenAI(
            model=etl_settings.llm.llm_name,
            base_url=etl_settings.llm.llm_base_url,
            api_key=etl_settings.llm.llm_api_key.get_secret_value(),  # noqa
            temperature=0.1
        ),
        "embedding_model": embedding_model,
    }


def process_single_article(article, app_settings, services):
    logger.info(f"[{article.name}] Thread started...")

    with build_state_store(app_settings) as state_manager:
        if state_manager.get_status(article.id, "publish") == "done":
            return f"[{article.name}] Already processed. Skipped."

        if any(elem["status"] == "running" for elem in state_manager.list_states(article.id)):
            return f"[{article.name}] Processing is already running. Skipped."

        local_source = LocalSource(app_settings.files.directory)

        pipeline = ETLPipeline(
            steps=[
                FetchStep(source=local_source),
                ParseStep(),
                HtmlCleaningStep(),
                ImageFilteringStep(),
                ImageCaptioningStep(),
                # PaperSummarisatonStep(),  # Disabled: summaries are no longer generated.
                MetadataExtractionStep(),
                ChunkingStep(),
                EmbeddingStep(),
                PublishStep()
            ]
        )

        ctx = ETLContext(
            article=article,
            state_manager=state_manager,
            artifact_store=services["artifact_store"],
            public_store=services["public_store"],
            vector_store=services["vector_store"],
            llm=services["llm_model"],
            embedding_model=services["embedding_model"]
        )

        start = time.perf_counter()

        try:
            pipeline.run(ctx)
            end = time.perf_counter()
            return f"[{article.id}] Success in {end - start:.2f}s"
        except Exception as e:
            end = time.perf_counter()
            logger.error(f"[{article.id}] Failed: {str(e)}", exc_info=True)
            return f"[{article.id}] Failed in {end - start:.2f}s"


def handle_articles_batch(articles, app_settings, services):
    logger.info(f"Scheduler found {len(articles)} articles. Starting parallel processing...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_article = {
            executor.submit(process_single_article, art, app_settings, services): art
            for art in articles
        }

        for future in as_completed(future_to_article):
            article = future_to_article[future]
            try:
                result_msg = future.result()
                logger.info(result_msg)
            except Exception as e:
                logger.error(f"[{article.name}] Unhandled exception during processing: {e}", exc_info=True)


def main():
    logger.info("Starting Papers ETL Daemon...")

    with build_state_store(settings) as state_manager:
        logger.info("Cleaning up hanging tasks...")
        state_manager.reset_running_states()

    local_source = LocalSource(settings.files.directory)
    services = build_shared_services(settings)

    scheduler = IngestionScheduler(
        on_batch=lambda batch: handle_articles_batch(batch, settings, services)
    )
    scheduler.register(local_source, Schedule(timedelta(minutes=POOLING_TIME_MINUTES)))
    
    try:
        while True:
            scheduler.poll()
            time.sleep(SLEEP_TIME)
    except KeyboardInterrupt:
        logger.info("Shutting down daemon...")


if __name__ == "__main__":
    main()
