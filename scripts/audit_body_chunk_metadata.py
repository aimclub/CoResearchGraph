"""Audit required bibliographic metadata on all body chunks in Chroma."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, TypeVar

import chromadb
from dotenv import find_dotenv, load_dotenv


FIELDS = ("paper_title", "publication_year", "authors", "source")
PLACEHOLDERS = {
    "paper_title": {"no title"},
    "publication_year": {"9999"},
    "authors": {"no authors"},
    "source": {"undefined", "unknown"},
}
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.getenv("CHROMADB_HOST")
        or os.getenv("HOSTS_PORTS__CHROMA_HOST", "localhost"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(
            os.getenv("CHROMADB_PORT")
            or os.getenv("HOSTS_PORTS__CHROMA_PORT", "8000")
        ),
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("CHROMADB_COLLECTION", "coscientist_papers"),
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/chroma_audit/body_metadata_audit.json"),
    )
    return parser.parse_args()


def retry(operation: Callable[[], T], description: str) -> T:
    for attempt in range(1, 11):
        try:
            return operation()
        except Exception as exc:
            if attempt == 10:
                raise
            print(
                f"{description} failed ({attempt}/10): {type(exc).__name__}; retrying",
                flush=True,
            )
            time.sleep(min(attempt * 2, 10))
    raise AssertionError("unreachable")


def empty(value: Any) -> bool:
    return value is None or isinstance(value, str) and not value.strip()


def main() -> int:
    load_dotenv(find_dotenv())
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    collection = retry(
        lambda: chromadb.HttpClient(host=args.host, port=args.port).get_collection(
            args.collection
        ),
        "Connect to Chroma",
    )

    missing = Counter()
    empty_values = Counter()
    placeholders = Counter()
    metadata_key_presence = Counter()
    affected_articles: dict[str, set[str]] = defaultdict(set)
    affected_chunks: set[str] = set()
    placeholder_chunks: set[str] = set()
    examples: list[dict[str, Any]] = []
    body_articles: set[str] = set()
    seen_chunk_ids: set[str] = set()
    body_chunks = 0

    def load_batch(start: int) -> dict[str, Any]:
        return retry(
            lambda: collection.get(
                where={"role": "body"},
                limit=args.batch_size,
                offset=start,
                include=["metadatas"],
            ),
            f"Load body metadata batch at offset {start}",
        )

    offset = 0
    while True:
        result = load_batch(offset)
        for chunk_id, metadata in zip(result["ids"], result["metadatas"]):
            if chunk_id in seen_chunk_ids:
                raise RuntimeError(f"Duplicate chunk ID returned during pagination: {chunk_id}")
            seen_chunk_ids.add(chunk_id)
            metadata = metadata or {}
            metadata_key_presence.update(metadata.keys())
            article_id = str(metadata.get("article_id", ""))
            if article_id:
                body_articles.add(article_id)
            invalid: dict[str, str] = {}
            for field in FIELDS:
                if field not in metadata:
                    missing[field] += 1
                    invalid[field] = "missing"
                elif empty(metadata[field]):
                    empty_values[field] += 1
                    invalid[field] = "empty"
                elif str(metadata[field]).strip().casefold() in PLACEHOLDERS[field]:
                    placeholders[field] += 1
                    placeholder_chunks.add(chunk_id)
                if field in invalid and article_id:
                    affected_articles[field].add(article_id)
            if invalid:
                affected_chunks.add(chunk_id)
                if len(examples) < 20:
                    examples.append(
                        {
                            "chunk_id": chunk_id,
                            "article_id": article_id or None,
                            "invalid_fields": invalid,
                        }
                    )

        batch_count = len(result["ids"])
        body_chunks += batch_count
        print(f"Scanned {body_chunks} body chunks", flush=True)
        if batch_count < args.batch_size:
            break
        offset += batch_count

    report = {
        "collection": args.collection,
        "host": args.host,
        "port": args.port,
        "body_chunks": body_chunks,
        "unique_body_articles": len(body_articles),
        "body_chunks_with_all_required_fields": body_chunks - len(affected_chunks),
        "body_chunks_with_missing_or_empty_fields": len(affected_chunks),
        "body_chunks_with_placeholder_values": len(placeholder_chunks),
        "fields": {
            field: {
                "missing": missing[field],
                "empty": empty_values[field],
                "placeholder": placeholders[field],
                "affected_articles": len(affected_articles[field]),
            }
            for field in FIELDS
        },
        "metadata_key_presence": dict(sorted(metadata_key_presence.items())),
        "examples": examples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
