"""Curate a subset of repos from repository_inventory.jsonl.

Two sources:
  inventory  — random N-per-domain sampler (deterministic, seeded).
               Replaces the old subset_by_domain.py.
  toolmaker  — resolve URLs for a fixed, hand-picked list of repo names
               that constitutes the ToolMaker evaluation set.

Usage:
    # Re-sample the toolrosella subset from the current inventory:
    python subset.py --source inventory --output toolrosella_subset.txt

    # Regenerate the ToolMaker subset (URLs pulled from inventory):
    python subset.py --source toolmaker --output toolmaker_subset.txt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


BASE      = Path(__file__).parent
INVENTORY = BASE / "repository_inventory.jsonl"

# Curated ToolMaker set — 14 repos spanning pathology / medical imaging /
# NLP / classical ML / structural & cell biology. Kept here (not in a
# separate text file) so subset.py is the single source of truth for
# what "toolmaker" means; the inventory just resolves the URLs.
TOOLMAKER_REPOS: tuple[str, ...] = (
    "CONCH", "MUSK", "MedSAM", "MedSSS", "ModernBERT", "PathFinderCRC",
    "RETFound_MAE", "STAMP", "TabPFN", "UNI", "cytopus", "esm",
    "flowmap", "nnUNet",
)


def load_inventory() -> list[dict]:
    return [
        json.loads(line)
        for line in INVENTORY.read_text().splitlines()
        if line.strip()
    ]


def sample_by_domain(rows: list[dict], n: int, seed: int) -> list[dict]:
    by_domain: dict[str, list] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    rng = random.Random(seed)
    subset: list[dict] = []
    for domain, candidates in sorted(by_domain.items()):
        subset.extend(rng.sample(candidates, min(n, len(candidates))))
    return subset


def resolve_curated(rows: list[dict], names: tuple[str, ...]) -> list[dict]:
    by_name = {r["repo_name"]: r for r in rows}
    resolved: list[dict] = []
    missing:  list[str]  = []
    for name in names:
        if name in by_name:
            resolved.append(by_name[name])
        else:
            missing.append(name)
    if missing:
        sys.exit(
            f"[subset] repos not found in inventory: {', '.join(missing)}\n"
            f"         add them to repository_inventory.jsonl first."
        )
    return resolved


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source", choices=("inventory", "toolmaker"),
                    default="inventory",
                    help="Sampling strategy (default: inventory).")
    ap.add_argument("--output", type=Path, default=None,
                    help="Where to write the URL list. "
                         "Defaults: toolrosella_subset.txt for inventory, "
                         "toolmaker_subset.txt for toolmaker.")
    ap.add_argument("--n", type=int, default=2,
                    help="Repos per domain when --source=inventory (default 2).")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed when --source=inventory (default 42).")
    return ap.parse_args()


def main() -> None:
    ns   = parse_args()
    rows = load_inventory()

    if ns.source == "inventory":
        subset      = sample_by_domain(rows, ns.n, ns.seed)
        default_out = BASE / "toolrosella_subset.txt"
    else:  # toolmaker
        subset      = resolve_curated(rows, TOOLMAKER_REPOS)
        default_out = BASE / "toolmaker_subset.txt"

    out = ns.output or default_out
    out.write_text("\n".join(r["github_url"] for r in subset) + "\n")

    print(f"[subset] wrote {len(subset)} URLs ({ns.source}) → {out.name}")

    if ns.source == "inventory":
        by_domain: dict[str, list] = defaultdict(list)
        for r in rows:
            by_domain[r["domain"]].append(r)
        for domain, candidates in sorted(by_domain.items()):
            picked = min(ns.n, len(candidates))
            print(f"  {picked:>2}  {domain}")


if __name__ == "__main__":
    main()
