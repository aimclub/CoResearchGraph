"""Copy bibliographic metadata from legacy summary chunks to body chunks.

The script is a dry run by default. Pass ``--apply`` to update Chroma. Existing
body metadata is preserved; the four bibliographic fields are replaced with the
values from the summary chunk for the same article.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
from dotenv import find_dotenv, load_dotenv


FIELDS = ("paper_title", "publication_year", "authors", "source")
MAX_ATTEMPTS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes to Chroma")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--article-batch-size",
        type=int,
        default=100,
        help="Number of summary article IDs per filtered body query",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/chroma_audit/body_metadata_backfill.json"),
    )
    return parser.parse_args()


def is_missing(value: Any) -> bool:
    return value is None or isinstance(value, str) and not value.strip()


def normalized_author_names(value: str) -> list[str]:
    """Compare author lists while ignoring common academic degree suffixes."""
    tokens = re.findall(r"[a-z]+", value.casefold())
    return [token for token in tokens if token not in {"phd", "md"}]


def batches(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def with_retry(operation, description: str):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            print(
                f"{description} failed ({attempt}/{MAX_ATTEMPTS}): "
                f"{type(exc).__name__}; retrying",
                flush=True,
            )
            time.sleep(min(attempt * 2, 10))


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.article_batch_size < 1:
        raise ValueError("batch sizes must be positive")

    load_dotenv(find_dotenv())
    host = os.getenv("CHROMADB_HOST") or os.getenv("HOSTS_PORTS__CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMADB_PORT") or os.getenv("HOSTS_PORTS__CHROMA_PORT", "8000"))
    collection_name = os.getenv("CHROMADB_COLLECTION", "coscientist_papers")
    collection = with_retry(
        lambda: chromadb.HttpClient(host=host, port=port).get_collection(collection_name),
        "Connect to Chroma",
    )

    summaries = with_retry(
        lambda: collection.get(where={"role": "summary"}, include=["metadatas"]),
        "Load summary chunks",
    )
    by_article: dict[str, dict[str, Any]] = {}
    incomplete_summaries: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    resolved_conflicts: list[dict[str, Any]] = []

    for chunk_id, metadata in zip(summaries["ids"], summaries["metadatas"]):
        metadata = metadata or {}
        article_id = metadata.get("article_id")
        missing = [field for field in FIELDS if is_missing(metadata.get(field))]
        if is_missing(article_id) or missing:
            incomplete_summaries.append(
                {"chunk_id": chunk_id, "article_id": article_id, "missing_fields": missing}
            )
            continue
        values = {field: metadata[field] for field in FIELDS}
        previous = by_article.get(str(article_id))
        if previous is not None and previous != values:
            same_non_author_fields = all(
                previous[field] == values[field]
                for field in FIELDS
                if field != "authors"
            )
            equivalent_authors = normalized_author_names(str(previous["authors"])) == (
                normalized_author_names(str(values["authors"]))
            )
            if same_non_author_fields and equivalent_authors:
                selected = max((previous, values), key=lambda item: len(str(item["authors"])))
                by_article[str(article_id)] = selected
                resolved_conflicts.append(
                    {
                        "article_id": article_id,
                        "first": previous,
                        "second": values,
                        "selected": selected,
                        "reason": "Equivalent author names; selected the more detailed value",
                    }
                )
                continue
            conflicts.append(
                {"article_id": article_id, "first": previous, "second": values}
            )
            continue
        by_article[str(article_id)] = values

    if conflicts:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "collection": collection_name,
                    "applied": False,
                    "summary_chunks": len(summaries["ids"]),
                    "usable_summary_articles": len(by_article),
                    "incomplete_summaries": incomplete_summaries,
                    "conflicts": conflicts,
                    "error": "Conflicting summary metadata; no records were updated",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(f"Found {len(conflicts)} conflicting summary metadata records")

    update_ids: list[str] = []
    update_metadatas: list[dict[str, Any]] = []
    matched_articles: set[str] = set()
    changed_fields: Counter[str] = Counter()

    article_ids = sorted(by_article)
    article_id_batches = list(batches(article_ids, args.article_batch_size))
    for batch_number, article_id_batch in enumerate(article_id_batches, start=1):
        body = with_retry(
            lambda: collection.get(
                where={
                    "$and": [
                        {"role": "body"},
                        {"article_id": {"$in": article_id_batch}},
                    ]
                },
                include=["metadatas"],
            ),
            f"Load body chunks for article batch {batch_number}",
        )
        for chunk_id, metadata in zip(body["ids"], body["metadatas"]):
            metadata = dict(metadata or {})
            article_id = str(metadata.get("article_id", ""))
            summary_metadata = by_article.get(article_id)
            if summary_metadata is None:
                continue
            matched_articles.add(article_id)
            for field, value in summary_metadata.items():
                if metadata.get(field) != value:
                    changed_fields[field] += 1
            metadata.update(summary_metadata)
            update_ids.append(chunk_id)
            update_metadatas.append(metadata)
        print(
            f"Read article batch {batch_number}/{len(article_id_batches)}; "
            f"selected {len(update_ids)} body chunks",
            flush=True,
        )

    report: dict[str, Any] = {
        "collection": collection_name,
        "applied": args.apply,
        "summary_chunks": len(summaries["ids"]),
        "usable_summary_articles": len(by_article),
        "incomplete_summaries": incomplete_summaries,
        "conflicts": conflicts,
        "resolved_conflicts": resolved_conflicts,
        "matched_body_articles": len(matched_articles),
        "body_chunks_selected": len(update_ids),
        "fields_changed": dict(changed_fields),
    }

    if args.apply:
        for ids_batch, metadata_batch in zip(
            batches(update_ids, args.batch_size),
            batches(update_metadatas, args.batch_size),
        ):
            with_retry(
                lambda: collection.update(ids=ids_batch, metadatas=metadata_batch),
                "Update body metadata batch",
            )

        verification_missing = Counter()
        verified = 0
        for ids_batch in batches(update_ids, args.batch_size):
            result = with_retry(
                lambda: collection.get(ids=ids_batch, include=["metadatas"]),
                "Verify updated body metadata batch",
            )
            verified += len(result["ids"])
            for metadata in result["metadatas"]:
                metadata = metadata or {}
                for field in FIELDS:
                    if is_missing(metadata.get(field)):
                        verification_missing[field] += 1
        report["verified_body_chunks"] = verified
        report["verification_missing_fields"] = dict(verification_missing)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
