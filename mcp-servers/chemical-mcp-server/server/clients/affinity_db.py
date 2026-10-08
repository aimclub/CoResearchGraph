import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import aiohttp

logger = logging.getLogger(__name__)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
VALID_AFFINITY_TYPES = {"Ki", "Kd", "IC50", "EC50"}


def _strip_bindingdb_smiles(raw: Any) -> str:
    if raw is None:
        return ""
    return re.sub(r"\s*\|.*\|$", "", str(raw))


async def fetch_uniprot_id(
    session: aiohttp.ClientSession,
    protein_name: str,
    organism_id: int = 9606,
    max_retries: int = 5,
    delay: float = 0.5,
) -> Optional[str]:
    """
    Asynchronously fetch UniProt ID for a given protein name.
    Retries up to `max_retries` times in case of network or transient API errors.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"{protein_name} AND organism_id:{organism_id}",
        "format": "json",
        "size": 1,
        "fields": "accession",
    }

    for attempt in range(max_retries):
        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    await asyncio.sleep(delay * (1 + attempt * 0.5))
                    continue
                data = await resp.json()
                results = data.get("results", [])
                if results:
                    return results[0].get("primaryAccession")
                return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("[UniProt] Attempt %s failed: %s", attempt + 1, e)
            await asyncio.sleep(delay * (1 + attempt * 0.5))
    return None


async def fetch_affinity_bindingdb(
    session: aiohttp.ClientSession,
    uniprot_id: str,
    affinity_type: str,
    cutoff: int,
    max_retries: int = 5,
    delay: float = 0.5,
) -> List[Dict]:
    """
    Asynchronously retrieve affinity values from BindingDB for a given UniProt ID.
    Retries on network errors or incomplete data.
    """
    url = (
        "http://bindingdb.org/rest/getLigandsByUniprot?"
        f"uniprot={uniprot_id};{cutoff}&response=application/json"
    )

    for attempt in range(max_retries):
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    logger.error(
                        "[BindingDB] HTTP %s for %s, retrying...",
                        resp.status,
                        uniprot_id,
                    )
                    await asyncio.sleep(delay * (1 + attempt * 0.5))
                    continue
                data = json.loads(await resp.text())
                affinities = (
                    data.get("getLindsByUniprotResponse", {}).get("bdb.affinities", [])
                    or data.get("bdb.affinities", [])
                    or []
                )

                result = [
                    {
                        "monomerid": a.get("bdb.monomerid"),
                        "smiles": _strip_bindingdb_smiles(a.get("bdb.smile")),
                        "affinity_type": a.get("bdb.affinity_type"),
                        "affinity": a.get("bdb.affinity"),
                    }
                    for a in affinities
                    if a.get("bdb.affinity_type") == affinity_type
                ]
                return result
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("[BindingDB] Attempt %s failed: %s", attempt + 1, e)
            await asyncio.sleep(delay * (1 + attempt * 0.5))
    return []


async def _aio_fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    timeout: int = 30,
    max_retries: int = 4,
    retry_delay: float = 0.5,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> dict:
    for attempt in range(max_retries):
        try:
            if semaphore:
                async with semaphore:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if isinstance(data, dict):
                                return data
            else:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, dict):
                            return data
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        await asyncio.sleep(retry_delay * (1 + 0.5 * attempt))
    return {}


async def _resolve_chembl_target_id(
    session: aiohttp.ClientSession,
    target_name: str,
    limit: int = 5,
    max_retries: int = 3,
) -> List[tuple[Optional[str], Optional[str]]] | str:
    """
    Search ChEMBL target endpoint and return list of (target_chembl_id, organism).
    Retries until valid data is obtained. Returns empty string if none found.
    """
    if not target_name:
        return ""

    for attempt in range(max_retries):
        search_url = (
            f"{CHEMBL_BASE}/target/search?q={quote(target_name)}"
            f"&format=json&limit={limit}"
        )
        data = await _aio_fetch_json(session, search_url)
        targets = data.get("targets", [])
        if targets and isinstance(targets, list):
            chembl_ids = [
                (target.get("target_chembl_id"), target.get("organism"))
                for target in targets
            ]
            if chembl_ids:
                return chembl_ids
        await asyncio.sleep(0.5 * (1 + attempt * 0.5))

    return ""


def _normalize_activities(
    activities: List[dict],
    target_id: str,
    affinity_type: str,
) -> List[Dict]:
    out: List[Dict] = []
    for act in activities:
        val = act.get("standard_value")
        out.append(
            {
                "smiles": act.get("canonical_smiles") or "",
                "affinity_type": affinity_type,
                "affinity_value": float(val) if val not in (None, "", "NA") else None,
                "affinity_units": act.get("standard_units") or "",
                "source": "ChEMBL",
                "target_id": target_id,
            }
        )
    return out


async def _fetch_chembl_activity_async(
    session: aiohttp.ClientSession,
    target_id: str,
    affinity_type: str = "Ki",
    limit_per_page: int = 1000,
    max_records: int = 100000,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> List[Dict]:
    """Fetch all ChEMBL activity pages concurrently for a given target."""
    if affinity_type not in VALID_AFFINITY_TYPES:
        return []

    base_url = (
        f"{CHEMBL_BASE}/activity.json?"
        f"target_chembl_id={quote(target_id)}&"
        f"standard_type={quote(affinity_type)}&"
        f"limit={limit_per_page}&offset=0&include=molecule"
    )
    first_data = await _aio_fetch_json(session, base_url, semaphore=semaphore)
    if not first_data:
        return []

    activities = first_data.get("activities", []) or []
    results = _normalize_activities(activities, target_id, affinity_type)

    page_meta = first_data.get("page_meta", {}) or {}
    try:
        total_count = int(page_meta.get("total_count", len(results)))
    except Exception:
        total_count = len(results)

    desired_total = min(total_count, max_records)
    already = len(results)
    remaining = max(0, desired_total - already)
    if remaining <= 0:
        return results

    offsets = list(range(limit_per_page, limit_per_page + remaining, limit_per_page))
    offsets = [o for o in offsets if o < desired_total]

    async def fetch_page(offset: int) -> List[Dict]:
        url = (
            f"{CHEMBL_BASE}/activity.json?"
            f"target_chembl_id={quote(target_id)}&"
            f"standard_type={quote(affinity_type)}&"
            f"limit={limit_per_page}&offset={offset}&include=molecule"
        )
        data = await _aio_fetch_json(session, url, semaphore=semaphore)
        acts = data.get("activities", []) if data else []
        return _normalize_activities(acts, target_id, affinity_type)

    tasks = [fetch_page(off) for off in offsets]
    page_results = await asyncio.gather(*tasks, return_exceptions=True)

    for pr in page_results:
        if isinstance(pr, list):
            results.extend(pr)

    if len(results) > desired_total:
        results = results[:desired_total]

    return results


async def _fetch_chembl_data_with_session(
    session: aiohttp.ClientSession,
    target_name: str,
    target_id: Optional[str] = None,
    affinity_type: str = "Ki",
    max_records: int = 10000,
    concurrency_limit: int = 10,
) -> List[Dict]:
    """
    ChEMBL fetch using an existing aiohttp session (caller owns lifecycle and connector limits).
    """
    semaphore = asyncio.Semaphore(concurrency_limit)
    results: List[Dict] = []

    if not target_id:
        chembl_targets = await _resolve_chembl_target_id(session, target_name)
        if not chembl_targets:
            return []
    else:
        chembl_targets = [(target_id, "unknown")]

    tasks = [
        _fetch_chembl_activity_async(
            session=session,
            target_id=tid,
            affinity_type=affinity_type,
            max_records=max_records,
            semaphore=semaphore,
        )
        for tid, _ in chembl_targets
    ]

    all_data = await asyncio.gather(*tasks, return_exceptions=True)

    for (tid, organism), data in zip(chembl_targets, all_data):
        if isinstance(data, Exception):
            continue
        for rec in data:
            rec["target_id"] = tid
            rec["organism"] = organism
        results.extend(data)

    return results


async def fetch_chembl_data(
    target_name: str,
    target_id: Optional[str] = None,
    affinity_type: str = "Ki",
    max_records: int = 10000,
    concurrency_limit: int = 10,
    *,
    session: aiohttp.ClientSession | None = None,
) -> List[Dict]:
    """
    High-performance concurrent ChEMBL data fetcher.
    Fetches multiple targets and multiple pages concurrently with controlled concurrency.

    Args:
        session: If provided, all HTTP calls reuse this session (same connection pool / timeouts
            as the rest of the caller). If omitted, a dedicated temporary session is created.
    """
    if session is not None:
        return await _fetch_chembl_data_with_session(
            session,
            target_name,
            target_id=target_id,
            affinity_type=affinity_type,
            max_records=max_records,
            concurrency_limit=concurrency_limit,
        )
    async with aiohttp.ClientSession() as owned_session:
        return await _fetch_chembl_data_with_session(
            owned_session,
            target_name,
            target_id=target_id,
            affinity_type=affinity_type,
            max_records=max_records,
            concurrency_limit=concurrency_limit,
        )
