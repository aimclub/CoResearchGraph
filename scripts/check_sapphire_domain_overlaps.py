#!/usr/bin/env python3
"""Check whether Sapphire publications have several positive domain scores."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter

import requests


DEFAULT_URL = "http://fpin-projects.ru:12280"
REQUESTED_DOMAINS = ("ai", "chembio")


def publication_identity(publication: dict) -> tuple[str, str]:
    """Return a stable identity for deduplication across domain queries."""
    for field in ("id", "openalex_id", "doi"):
        value = publication.get(field)
        if value not in (None, ""):
            return field, str(value)
    return "payload", json.dumps(
        publication,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def domain_scores(publication: dict) -> dict[str, float]:
    """Return finite numeric scores keyed by normalized domain name."""
    result = {}
    domains = publication.get("domains")
    if not isinstance(domains, list):
        return result
    for domain in domains:
        if not isinstance(domain, dict):
            continue
        name = domain.get("name")
        score = domain.get("score")
        if (
            isinstance(name, str)
            and name.strip()
            and type(score) in (int, float)
            and math.isfinite(score)
        ):
            result[name.strip().lower()] = float(score)
    return result


def fetch_publications(
    session: requests.Session,
    base_url: str,
    page_size: int,
    timeout: float,
    domain_score: float,
    max_pages: int | None,
) -> tuple[dict[tuple[str, str], dict], Counter]:
    """Fetch AI and chembio result sets and deduplicate their publications."""
    publications = {}
    returned_rows = Counter()

    for requested_domain in REQUESTED_DOMAINS:
        offset = 0
        page_number = 0
        previous_page = None

        while max_pages is None or page_number < max_pages:
            response = session.get(
                f"{base_url}/publications",
                params={
                    "domain": requested_domain,
                    "domain_score": domain_score,
                    "limit": page_size,
                    "offset": offset,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
                raise ValueError("/publications returned a value other than an array of objects")
            if not page:
                break
            if page == previous_page:
                raise RuntimeError(
                    f"Sapphire repeated the same {requested_domain!r} page at offset {offset}"
                )

            page_number += 1
            returned_rows[requested_domain] += len(page)
            for publication in page:
                publications[publication_identity(publication)] = publication

            offset += len(page)
            previous_page = page
            print(
                f"[{requested_domain}] pages={page_number}, rows={returned_rows[requested_domain]}, "
                f"unique_total={len(publications)}",
                file=sys.stderr,
                flush=True,
            )

    return publications, returned_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check positive domain-score combinations in Sapphire publications "
            "returned by the ai and chembio filters."
        )
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Sapphire base URL")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--domain-score", type=float, default=1e-12)
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Maximum number of ai/chembio + photonics examples to print",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Optional page limit per requested domain for a quick partial check",
    )
    args = parser.parse_args()

    if args.page_size < 1 or args.timeout <= 0 or args.domain_score <= 0:
        parser.error("page-size, timeout, and domain-score must be positive")
    if args.examples < 0 or (args.max_pages is not None and args.max_pages < 1):
        parser.error("examples must be non-negative and max-pages must be positive")

    try:
        with requests.Session() as session:
            publications, returned_rows = fetch_publications(
                session=session,
                base_url=args.url.rstrip("/"),
                page_size=args.page_size,
                timeout=args.timeout,
                domain_score=args.domain_score,
                max_pages=args.max_pages,
            )
    except (requests.RequestException, ValueError, RuntimeError) as error:
        print(f"Sapphire check failed: {error}", file=sys.stderr)
        return 1

    combinations = Counter()
    overlap_examples = []
    overlap_count = 0
    multiple_positive_count = 0

    for publication in publications.values():
        scores = domain_scores(publication)
        positive = tuple(sorted(name for name, score in scores.items() if score > 0))
        combinations[positive] += 1
        if len(positive) > 1:
            multiple_positive_count += 1
        if "photonics" in positive and ({"ai", "chembio"} & set(positive)):
            overlap_count += 1
            if len(overlap_examples) < args.examples:
                overlap_examples.append(
                    {
                        "openalex_id": publication.get("openalex_id"),
                        "name": publication.get("name"),
                        "positive_domains": list(positive),
                        "scores": scores,
                    }
                )

    result = {
        "scan_is_partial": args.max_pages is not None,
        "returned_rows": dict(returned_rows),
        "unique_publications": len(publications),
        "positive_domain_combinations": {
            " + ".join(names) if names else "none": count
            for names, count in sorted(combinations.items())
        },
        "multiple_positive_domain_count": multiple_positive_count,
        "ai_or_chembio_with_photonics_count": overlap_count,
        "ai_or_chembio_with_photonics_exists": overlap_count > 0,
        "examples": overlap_examples,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
