from __future__ import annotations

import argparse
import fcntl
import json
import logging
import sys
import types
from pathlib import Path

# Support direct execution without initializing the CoScientist agent application.
if not __package__:
    if 'CoScientist' not in sys.modules:
        package = types.ModuleType('CoScientist')
        package.__path__ = [str(Path(__file__).resolve().parents[1])]
        package.__package__ = 'CoScientist'
        sys.modules['CoScientist'] = package
    __package__ = 'CoScientist.sapphire_pipeline'

from .client import SapphireClient
from .pipeline import SapphirePipeline
from .registry import PublicationRegistry
from .settings import OpenAlexSettings, PDFCrawlerSettings, SapphireSettings
from CoScientist.papers_processing_refactoring.definitions import CONFIG_PATH


def main():
    """Run one ingestion pass using CLI options and a local exclusive lock.

    Print outcome counters and return 1 for article failures, otherwise 0.
    Argument and lock errors exit through argparse; other errors propagate.
    Close the HTTP client when the pass finishes or fails.
    """
    parser = argparse.ArgumentParser(description='Update RAG from Sapphire (one complete pass)')
    parser.add_argument('--page-size', type=int, default=100, help='Number of records to fetch per request')
    parser.add_argument('--max-articles', type=int, help='Maximum number of records to inspect, including skips and failures')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--max-pdf-mb', type=int, default=100)
    parser.add_argument('--lock-file', type=Path, default=Path('data/sapphire_pipeline.lock'))
    parser.add_argument(
        '--registry-file', type=Path, default=Path('data/sapphire_registry.sqlite3'),
        help='SQLite file used to persist publication statuses between runs',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sapphire = SapphireSettings(_env_file=CONFIG_PATH)
    crawler = PDFCrawlerSettings(_env_file=CONFIG_PATH)
    openalex = OpenAlexSettings(_env_file=CONFIG_PATH)
    client = SapphireClient(
        sapphire.url, args.page_size, args.timeout, args.max_pdf_mb * 1024 * 1024,
        pdf_crawler_url=crawler.url,
        pdf_crawler_username=crawler.username,
        pdf_crawler_password=crawler.password.get_secret_value() if crawler.password else None,
        pdf_crawler_timeout=crawler.timeout,
        openalex_email=openalex.email,
        openalex_api_key=openalex.api_key.get_secret_value() if openalex.api_key else None,
    )
    try:
        args.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with args.lock_file.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.exit(1, 'Sapphire pipeline is already running with this lock file\n')
            from .runtime import RAGBackend
            with PublicationRegistry(args.registry_file) as registry:
                counts = SapphirePipeline(client, RAGBackend(), registry).run(
                    max_articles=args.max_articles
                )
            print(json.dumps(counts, ensure_ascii=False, indent=2))
            return 1 if counts.get('failed') else 0
    finally:
        client.close()


if __name__ == '__main__':
    raise SystemExit(main())
