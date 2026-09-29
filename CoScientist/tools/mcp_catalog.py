"""Deployment-wide catalogue of configured MCP tools for the Web UI.

The catalogue joins two views of the same system:

* the Tool RAG PostgreSQL registry (indexed, possibly stale metadata);
* a bounded MCP ``tools/list`` probe (the server's current public contract).

No business tool is called and no registry state is changed.  The public
snapshot deliberately excludes endpoints, headers and credentials.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

import yaml

from CoScientist.config import get_settings
from CoScientist.web.session_store import state_dir

logger = logging.getLogger(__name__)

_PRESENTATION_PATH = Path(__file__).parents[1] / "config" / "tool_catalog.yaml"
_SNAPSHOT_NAME = "tool_catalog.json"
_LEASE_NAME = "tool_catalog.refresh.lock"
_SCHEMA_LIMIT = 64_000
_DESCRIPTION_LIMIT = 12_000
_DEFAULT_TTL = 300
_DEFAULT_PROBE_TIMEOUT = 15.0
_DEFAULT_CONCURRENCY = 4
_LEASE_SECONDS = 120

_snapshot: Optional[Dict[str, Any]] = None
_snapshot_location: Optional[Path] = None
_refresh_task: Optional[asyncio.Task] = None
_state_lock = asyncio.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _snapshot_path() -> Path:
    return state_dir() / _SNAPSHOT_NAME


def _lease_path() -> Path:
    return state_dir() / _LEASE_NAME


def _empty_snapshot(*, refreshing: bool = False) -> Dict[str, Any]:
    return {
        "version": 1,
        "snapshot_id": None,
        "checked_at": None,
        "last_success_at": None,
        "refreshing": refreshing,
        "stale": True,
        "partial": True,
        "registry": {"status": "pending", "server_count": 0, "tool_count": 0},
        "summary": {
            "current_tools": 0,
            "indexed_tools": 0,
            "reachable_servers": 0,
            "configured_servers": 0,
            "categories": 0,
            "attention_servers": 0,
        },
        "servers": [],
        "tools": [],
    }


def _read_snapshot() -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(_snapshot_path().read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) and payload.get("version") == 1 else None
    except (OSError, json.JSONDecodeError):
        return None


def _write_snapshot(payload: Mapping[str, Any]) -> None:
    path = _snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temp, path)


def _is_stale(payload: Mapping[str, Any]) -> bool:
    checked = _parse_time(payload.get("checked_at"))
    ttl = max(15, int(os.getenv("MCP_CATALOG__REFRESH_SECONDS", str(_DEFAULT_TTL))))
    return checked is None or (datetime.now(timezone.utc) - checked).total_seconds() >= ttl


def _public_snapshot(payload: Mapping[str, Any], *, refreshing: bool) -> Dict[str, Any]:
    # A disk snapshot can outlive a presentation-only YAML edit. Reapply the
    # local labels on the response copy so reads become current immediately;
    # discovery status, schemas and the cached snapshot remain untouched.
    result = _reapply_presentation(payload)
    result["refreshing"] = refreshing
    result["stale"] = _is_stale(result)
    return result


def _load_presentation() -> Dict[str, Any]:
    try:
        payload = yaml.safe_load(_PRESENTATION_PATH.read_text(encoding="utf-8")) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:  # noqa: BLE001 - catalogue must work without overrides
        logger.warning("MCP catalogue presentation metadata is unavailable: %s", exc)
        return {}


def load_presentation() -> Dict[str, Any]:
    """Load the local, public-safe presentation metadata for catalogue items."""
    return _load_presentation()


def _plain_json(value: Any, *, limit: int = _SCHEMA_LIMIT) -> Any:
    try:
        raw = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return {}
    if len(raw) > limit:
        return {"truncated": True, "size": len(raw)}
    return json.loads(raw)


def _clean_description(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:_DESCRIPTION_LIMIT]


def _short_description(value: Any, limit: int = 360) -> str:
    text = _clean_description(value)
    if len(text) <= limit:
        return text
    sentence = re.search(r"^.{40,%d}?[.!?](?:\s|$)" % limit, text)
    return sentence.group(0).strip() if sentence else text[:limit].rstrip() + "…"


def _header_map(value: Any) -> Dict[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items() if item is not None}


_ABBREVIATIONS = {
    "ai": "AI", "api": "API", "csv": "CSV", "db": "DB", "fedot": "FEDOT",
    "gan": "GAN", "html": "HTML", "iupac": "IUPAC", "ld50": "LD50",
    "ml": "ML", "mcp": "MCP", "ocr": "OCR", "pdf": "PDF", "rdkit": "RDKit",
    "s3": "S3", "shap": "SHAP", "smiles": "SMILES", "url": "URL",
}


def _human_name(name: str) -> str:
    words = re.sub(r"[-_]+", " ", str(name or "tool")).split()
    rendered = [_ABBREVIATIONS.get(word.lower(), word.lower()) for word in words]
    result = " ".join(rendered)
    return result[:1].upper() + result[1:]


def _category(server: Mapping[str, Any], tool: Mapping[str, Any]) -> str:
    haystack = " ".join(
        str(value or "")
        for value in (server.get("name"), server.get("description"), tool.get("name"), tool.get("description"))
    ).lower()
    rules = (
        ("automl", r"automl|fedot|train_ml|predict_ml"),
        ("chemistry", r"molecul|chemical|tox|smiles|docking|reaction|protein|medchem"),
        ("epidemiology", r"epidem|influenza|seir|surrogate"),
        ("literature", r"paper|openalex|literature|article|scientific database"),
        ("data", r"dataset|database|similarity"),
        ("files", r"vault|artifact|upload|download"),
        ("web", r"tavily|web search|crawl|extract url"),
        ("reporting", r"normcontrol|report|gost"),
    )
    return next((key for key, pattern in rules if re.search(pattern, haystack)), "other")


def _role(tool_name: str) -> str:
    name = tool_name.lower()
    supporting = (
        "health", "status", "state", "get_mcp_logs", "list_", "get_upload_link",
        "get_download_link", "cleanup_session", "update_artifact_metadata",
    )
    return "supporting" if any(token in name for token in supporting) else "scientific"


def _metadata_hash(name: str, description: str, schema: Any) -> str:
    material = json.dumps([name, description, schema], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _safe_url_key(value: str) -> str:
    """Normalize for internal deduplication while retaining no public secret."""
    parsed = urlsplit(str(value).strip())
    host = (parsed.hostname or "").lower()
    port = f":{parsed.port}" if parsed.port else ""
    path = (parsed.path or "/").rstrip("/") or "/"
    # Query may carry a Tavily key; hash it instead of keeping it in the key.
    query_hash = hashlib.sha256((parsed.query or "").encode()).hexdigest()[:12] if parsed.query else ""
    return urlunsplit((parsed.scheme.lower(), f"{host}{port}", path, query_hash, ""))


def _endpoint_id(url_key: str) -> str:
    return "configured-" + hashlib.sha256(url_key.encode()).hexdigest()[:16]


async def _read_registry() -> Dict[str, Any]:
    connection = None
    try:
        import asyncpg
        from rag_tools.config.settings import get_settings as rag_settings

        cfg = rag_settings().postgres
        connection = await asyncpg.connect(
            host=cfg.host,
            port=cfg.port,
            user=cfg.user,
            password=cfg.password,
            database=cfg.database,
            timeout=8,
            command_timeout=10,
        )
        async with connection.transaction(readonly=True):
            server_rows = await connection.fetch(
                "SELECT server_id, name, protocol, url, headers, description, status, "
                "last_synced FROM servers ORDER BY name, server_id"
            )
            tool_rows = await connection.fetch(
                "SELECT tool_id, server_id, name, description, input_schema, output_schema, "
                "tags, status, updated_at FROM tools ORDER BY server_id, name"
            )
        servers = []
        for row in server_rows:
            item = dict(row)
            item["headers"] = _header_map(item.get("headers"))
            item["last_synced"] = str(item["last_synced"]) if item.get("last_synced") else None
            servers.append(item)
        tools = []
        for row in tool_rows:
            item = dict(row)
            item["input_schema"] = _plain_json(item.get("input_schema") or {})
            item["output_schema"] = _plain_json(item.get("output_schema")) if item.get("output_schema") else None
            item["tags"] = list(item.get("tags") or [])
            item["updated_at"] = str(item["updated_at"]) if item.get("updated_at") else None
            tools.append(item)
        return {"status": "ready", "servers": servers, "tools": tools}
    except Exception as exc:  # noqa: BLE001 - one failed source leaves configured MCPs
        logger.warning("MCP catalogue registry read failed: %s", type(exc).__name__)
        return {"status": "unavailable", "error_type": type(exc).__name__, "servers": [], "tools": []}
    finally:
        if connection is not None:
            try:
                await connection.close()
            except Exception:  # noqa: BLE001 - closing must not erase a valid read
                pass


def _configured_endpoints() -> List[Dict[str, Any]]:
    settings = get_settings()
    endpoints: List[Dict[str, Any]] = []

    def add(name: str, url: Optional[str], *, headers: Optional[Dict[str, str]] = None,
            tool_filter: Optional[Iterable[str]] = None) -> None:
        if url:
            endpoints.append({
                "name": name,
                "url": str(url),
                "headers": dict(headers or {}),
                "tool_filter": list(tool_filter) if tool_filter else None,
                "source": "settings",
            })

    add("paper-analysis", settings.mcp.paper_analysis_url)
    add(
        "papers-search",
        settings.mcp.papers_search_url,
        headers={
            key: value for key, value in {
                "x-openalex-email": settings.services.openalex_email,
                "x-openalex-api-key": settings.services.openalex_api_key,
            }.items() if value
        },
    )
    add("vault", settings.mcp.vault_url, tool_filter=("get_upload_link", "get_download_link"))
    add("normcontrol", settings.mcp.normcontrol_url)
    if settings.services.tavily_api_key:
        add("Tavily", f"https://mcp.tavily.com/mcp/?tavilyApiKey={settings.services.tavily_api_key}")
    return endpoints


def _merge_endpoints(registry: Mapping[str, Any]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for server in registry.get("servers") or []:
        if str(server.get("protocol")) != "http" or not server.get("url"):
            continue
        key = _safe_url_key(str(server["url"]))
        merged[key] = {
            "id": str(server["server_id"]),
            "registry_id": str(server["server_id"]),
            "name": str(server.get("name") or server["server_id"]),
            "url": str(server["url"]),
            "headers": _header_map(server.get("headers")),
            "tool_filter": None,
            "sources": ["registry"],
            "aliases": [str(server.get("name") or server["server_id"])],
            "registry": dict(server),
            "url_key": key,
        }
    for endpoint in _configured_endpoints():
        key = _safe_url_key(endpoint["url"])
        current = merged.get(key)
        if current is None:
            current = {
                "id": _endpoint_id(key),
                "registry_id": None,
                "name": endpoint["name"],
                "url": endpoint["url"],
                "headers": {},
                "tool_filter": endpoint.get("tool_filter"),
                "sources": [],
                "aliases": [],
                "registry": {},
                "url_key": key,
            }
            merged[key] = current
        current["headers"].update(endpoint.get("headers") or {})
        if endpoint.get("tool_filter"):
            current["tool_filter"] = list(endpoint["tool_filter"])
        if "settings" not in current["sources"]:
            current["sources"].append("settings")
        if endpoint["name"] not in current["aliases"]:
            current["aliases"].append(endpoint["name"])
    return list(merged.values())


async def _probe_endpoint(endpoint: Mapping[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    timeout = max(2.0, float(os.getenv("MCP_CATALOG__PROBE_TIMEOUT", str(_DEFAULT_PROBE_TIMEOUT))))
    async with semaphore:
        try:
            import anyio
            import httpx
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client

            with anyio.fail_after(timeout):
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout),
                    trust_env=False,
                    headers=dict(endpoint.get("headers") or {}),
                ) as http:
                    async with streamable_http_client(str(endpoint["url"]), http_client=http) as (read, write, _):
                        async with ClientSession(read, write) as session:
                            initialized = await session.initialize()
                            cursor: Optional[str] = None
                            seen: set[str] = set()
                            tools: List[Dict[str, Any]] = []
                            for _page in range(50):
                                result = await session.list_tools(cursor=cursor)
                                for tool in result.tools:
                                    dumped = tool.model_dump(by_alias=True) if hasattr(tool, "model_dump") else {}
                                    tools.append({
                                        "name": str(tool.name),
                                        "description": _clean_description(tool.description),
                                        "input_schema": _plain_json(dumped.get("inputSchema") or {}),
                                        "output_schema": _plain_json(dumped.get("outputSchema")) if dumped.get("outputSchema") else None,
                                    })
                                cursor = result.nextCursor
                                if not cursor:
                                    break
                                if cursor in seen:
                                    raise RuntimeError("repeated pagination cursor")
                                seen.add(cursor)
                            else:
                                raise RuntimeError("too many tools/list pages")
                            info = getattr(initialized, "serverInfo", None)
                            server_info = {
                                "name": str(getattr(info, "name", "") or ""),
                                "version": str(getattr(info, "version", "") or ""),
                            }
            return {"status": "reachable", "tools": tools, "server_info": server_info, "checked_at": _utc_now()}
        except BaseException as exc:  # cancellation/exception groups are normalized below
            if isinstance(exc, asyncio.CancelledError):
                raise
            while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
                exc = exc.exceptions[0]
            return {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "tools": [],
                "server_info": {},
                "checked_at": _utc_now(),
            }


def _localized(value: Any, fallback: str, *, ru_fallback: Optional[str] = None) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {"ru": ru_fallback or fallback, "en": fallback}
    return {
        "ru": str(value.get("ru") or ru_fallback or fallback),
        "en": str(value.get("en") or fallback),
    }


def resolve_tool_presentation(
    name: str,
    description: str = "",
    *,
    server_id: Optional[str] = None,
    registry_id: Optional[str] = None,
    registry_key: Optional[str] = None,
    presentation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, str]]:
    """Resolve bilingual display metadata for a built-in or MCP tool.

    Built-in registry entries use ``local_tools[registry_key:name]`` before
    ``tool_defaults``. MCP entries retain the catalogue's exact registry-id,
    endpoint-id, then default precedence. Russian fallbacks for built-ins are
    deliberately neutral so an English implementation description is never
    presented as Russian text.
    """
    payload = presentation if presentation is not None else _load_presentation()
    payload = payload if isinstance(payload, Mapping) else {}
    local_tools = payload.get("local_tools") or {}
    remote_tools = payload.get("tools") or {}
    tool_defaults = payload.get("tool_defaults") or {}
    override: Mapping[str, Any] = {}

    if registry_key:
        candidate = local_tools.get(f"{registry_key}:{name}")
        if not isinstance(candidate, Mapping):
            candidate = tool_defaults.get(name)
        if isinstance(candidate, Mapping):
            override = candidate
        english_title = _human_name(name)
        english_summary = _short_description(description) or english_title
        return {
            "display_name": _localized(
                override.get("display_name"),
                english_title,
                ru_fallback="Инструмент",
            ),
            "summary": _localized(
                override.get("description"),
                english_summary,
                ru_fallback="Описание на русском пока не добавлено.",
            ),
        }

    stable_key = f"{registry_id}:{name}" if registry_id else None
    candidate = remote_tools.get(stable_key) if stable_key else None
    if not isinstance(candidate, Mapping):
        candidate = remote_tools.get(f"{server_id}:{name}") if server_id else None
    if not isinstance(candidate, Mapping):
        candidate = tool_defaults.get(name)
    if isinstance(candidate, Mapping):
        override = candidate
    english_title = _human_name(name)
    english_summary = _short_description(description) or english_title
    title = _localized(
        override.get("display_name"),
        english_title,
        ru_fallback="Инструмент",
    )
    return {
        "display_name": title,
        "summary": _localized(
            override.get("description"),
            english_summary,
            ru_fallback="Описание на русском пока не добавлено.",
        ),
    }


def _reapply_presentation(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply current YAML presentation fields to a response copy of a snapshot."""
    result = dict(payload)
    presentation = load_presentation()
    server_meta = presentation.get("servers") or {}
    tool_meta = presentation.get("tools") or {}
    tool_defaults = presentation.get("tool_defaults") or {}

    servers = []
    for raw in payload.get("servers") or []:
        item = dict(raw)
        registry_id = item.get("registry_id")
        override = server_meta.get(str(registry_id)) or server_meta.get(str(item.get("id"))) or {}
        if isinstance(override, Mapping):
            current_title = item.get("display_name") or {}
            current_description = item.get("description") or {}
            if "display_name" in override:
                item["display_name"] = _localized(
                    override.get("display_name"),
                    str(current_title.get("en") or _human_name(str(item.get("name") or item.get("id")))),
                )
            if "description" in override:
                item["description"] = _localized(
                    override.get("description"),
                    str(current_description.get("en") or _human_name(str(item.get("name") or item.get("id")))),
                )
            for field in ("category", "role", "featured", "priority"):
                if field in override:
                    item[field] = override[field]
        servers.append(item)
    result["servers"] = servers

    server_registry_ids = {
        str(item.get("id")): item.get("registry_id")
        for item in servers
    }
    tools = []
    for raw in payload.get("tools") or []:
        item = dict(raw)
        name = str(item.get("name") or "")
        server_id = str(item.get("server_id") or "")
        registry_id = server_registry_ids.get(server_id)
        stable_key = f"{registry_id}:{name}" if registry_id else None
        override = (
            tool_meta.get(stable_key)
            or tool_meta.get(f"{server_id}:{name}")
            or tool_defaults.get(name)
            or {}
        )
        if isinstance(override, Mapping) and override:
            localized = resolve_tool_presentation(
                name,
                str(item.get("original_description") or ""),
                server_id=server_id,
                registry_id=registry_id,
                presentation=presentation,
            )
            if "display_name" in override:
                item["display_name"] = localized["display_name"]
            if "description" in override:
                item["summary"] = localized["summary"]
            for field in ("category", "role", "priority"):
                if field in override:
                    item[field] = override[field]
        else:
            # Old disk snapshots may still contain the previous machine-name
            # fallback. Keep their English text, but normalize Russian copy on
            # read without waiting for a discovery refresh.
            localized = resolve_tool_presentation(name, str(item.get("original_description") or ""), presentation={})
            item["display_name"] = {**localized["display_name"], **(item.get("display_name") or {}), "ru": localized["display_name"]["ru"]}
            item["summary"] = {**localized["summary"], **(item.get("summary") or {}), "ru": localized["summary"]["ru"]}
        tools.append(item)
    result["tools"] = tools
    return result


def _assemble_catalog(
    registry: Mapping[str, Any],
    endpoints: List[Mapping[str, Any]],
    probes: List[Mapping[str, Any]],
    *,
    previous: Optional[Mapping[str, Any]] = None,
    checked_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Pure merger used by refresh and focused tests."""
    presentation = load_presentation()
    server_meta = presentation.get("servers") or {}
    tool_meta = presentation.get("tools") or {}
    tool_defaults = presentation.get("tool_defaults") or {}
    checked_at = checked_at or _utc_now()
    registry_tools: Dict[str, List[Dict[str, Any]]] = {}
    for raw in registry.get("tools") or []:
        registry_tools.setdefault(str(raw.get("server_id")), []).append(dict(raw))
    previous_servers = {str(item.get("id")): item for item in (previous or {}).get("servers", [])}
    previous_tools: Dict[tuple[str, str], Dict[str, Any]] = {
        (str(item.get("server_id")), str(item.get("name"))): dict(item)
        for item in (previous or {}).get("tools", [])
    }

    public_servers: List[Dict[str, Any]] = []
    public_tools: List[Dict[str, Any]] = []
    any_success = False
    for endpoint, probe in zip(endpoints, probes):
        server_id = str(endpoint["id"])
        registry_id = endpoint.get("registry_id")
        indexed = registry_tools.get(str(registry_id), []) if registry_id else []
        live = {str(tool["name"]): dict(tool) for tool in probe.get("tools") or []}
        indexed_by_name = {str(tool["name"]): tool for tool in indexed}
        reachable = probe.get("status") == "reachable"
        any_success = any_success or reachable
        previous_server = previous_servers.get(server_id) or {}
        previous_names = {name for sid, name in previous_tools if sid == server_id}
        effective_names = set(live) if reachable else set(indexed_by_name) | previous_names

        registry_record = endpoint.get("registry") or {}
        override = server_meta.get(str(registry_id)) or server_meta.get(server_id) or {}
        fallback_server_name = _human_name(str(endpoint.get("name") or server_id))
        server_title = _localized(override.get("display_name"), fallback_server_name)
        server_description = _localized(
            override.get("description"),
            _clean_description(registry_record.get("description")) or fallback_server_name,
        )
        live_names, indexed_names = set(live), set(indexed_by_name)
        discrepancy = {
            "has_difference": reachable and live_names != indexed_names,
            "live_only": sorted(live_names - indexed_names),
            "indexed_only": sorted(indexed_names - live_names),
        }
        last_success_at = checked_at if reachable else previous_server.get("last_success_at")
        server_public = {
            "id": server_id,
            "registry_id": registry_id,
            "name": str(endpoint.get("name") or server_id),
            "display_name": server_title,
            "description": server_description,
            "aliases": list(endpoint.get("aliases") or []),
            "sources": list(endpoint.get("sources") or []),
            "protocol": str(registry_record.get("protocol") or "http"),
            "category": str(override.get("category") or "other"),
            "role": str(override.get("role") or "scientific"),
            "featured": bool(override.get("featured")),
            "priority": int(override.get("priority", 100)),
            "registry_status": registry_record.get("status") if registry_record else None,
            "registry_last_synced": registry_record.get("last_synced") if registry_record else None,
            "discovery_status": str(probe.get("status") or "unavailable"),
            "error_type": probe.get("error_type"),
            "checked_at": probe.get("checked_at") or checked_at,
            "last_success_at": last_success_at,
            "server_info": dict(probe.get("server_info") or {}),
            "tool_counts": {"current": len(effective_names), "live": len(live_names), "indexed": len(indexed_names)},
            "discrepancy": discrepancy,
        }
        public_servers.append(server_public)

        for name in effective_names:
            current = live.get(name) or indexed_by_name.get(name) or previous_tools.get((server_id, name), {})
            indexed_tool = indexed_by_name.get(name) or {}
            previous_tool = previous_tools.get((server_id, name)) or {}
            description = _clean_description(current.get("description") or previous_tool.get("original_description"))
            input_schema = _plain_json(current.get("input_schema") or {})
            output_schema = _plain_json(current.get("output_schema")) if current.get("output_schema") else None
            stable_registry_key = f"{registry_id}:{name}" if registry_id else None
            override_tool = (
                tool_meta.get(stable_registry_key)
                or tool_meta.get(f"{server_id}:{name}")
                or tool_defaults.get(name)
                or {}
            )
            category = str(override_tool.get("category") or _category(registry_record or endpoint, current))
            role = str(override_tool.get("role") or _role(name))
            localized = resolve_tool_presentation(
                name,
                description,
                server_id=server_id,
                registry_id=registry_id,
                presentation=presentation,
            )
            title = localized["display_name"]
            summary = localized["summary"]
            source = "live" if name in live else ("registry" if name in indexed_by_name else "snapshot")
            allowed = endpoint.get("tool_filter")
            public_tools.append({
                "id": f"{server_id}:{name}",
                "server_id": server_id,
                "registry_tool_id": indexed_tool.get("tool_id"),
                "name": name,
                "display_name": title,
                "summary": summary,
                "original_description": description,
                "input_schema": input_schema,
                "output_schema": output_schema,
                "category": category,
                "role": role,
                "priority": int(override_tool.get("priority", 100)),
                "status": "available" if reachable and name in live else "saved",
                "metadata_source": source,
                "metadata_hash": _metadata_hash(name, description, input_schema),
                "registry_status": indexed_tool.get("status"),
                "available_to_agents": allowed is None or name in allowed,
            })

    public_servers.sort(key=lambda item: (item["priority"], item["display_name"]["ru"].casefold()))
    server_order = {item["id"]: index for index, item in enumerate(public_servers)}
    public_tools.sort(key=lambda item: (
        server_order.get(item["server_id"], 9999), item["priority"], item["display_name"]["ru"].casefold()
    ))
    current_tools = sum(1 for tool in public_tools if tool["status"] == "available")
    attention = sum(
        1 for server in public_servers
        if server["discovery_status"] != "reachable" or server["discrepancy"]["has_difference"]
    )
    categories = {tool["category"] for tool in public_tools}
    partial = registry.get("status") != "ready" or attention > 0
    return {
        "version": 1,
        "snapshot_id": uuid.uuid4().hex,
        "checked_at": checked_at,
        "last_success_at": checked_at if any_success else (previous or {}).get("last_success_at"),
        "refreshing": False,
        "stale": False,
        "partial": partial,
        "registry": {
            "status": registry.get("status", "unavailable"),
            "error_type": registry.get("error_type"),
            "server_count": len(registry.get("servers") or []),
            "tool_count": len(registry.get("tools") or []),
        },
        "summary": {
            "current_tools": current_tools,
            "indexed_tools": len(registry.get("tools") or []),
            "reachable_servers": sum(1 for server in public_servers if server["discovery_status"] == "reachable"),
            "configured_servers": len(public_servers),
            "categories": len(categories),
            "attention_servers": attention,
        },
        "servers": public_servers,
        "tools": public_tools,
    }


def _try_acquire_lease() -> bool:
    path = _lease_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Disk snapshots are best-effort. The process-local task is still
        # single-flight, so refresh in memory when a read-only deployment gives
        # us nowhere to place the cross-process lease.
        return True
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            if time.time() - path.stat().st_mtime <= _LEASE_SECONDS:
                return False
            path.unlink(missing_ok=True)
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, OSError):
            return False
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"{os.getpid()} {time.time():.0f}\n")
    return True


def _lease_active() -> bool:
    try:
        return time.time() - _lease_path().stat().st_mtime <= _LEASE_SECONDS
    except OSError:
        return False


async def refresh_catalog() -> Dict[str, Any]:
    """Build and persist a new public snapshot; callers share one task."""
    global _snapshot, _snapshot_location
    location = _snapshot_path()
    if _snapshot_location is not None and _snapshot_location != location:
        _snapshot = None
    _snapshot_location = location
    if not _try_acquire_lease():
        latest = _read_snapshot() or _snapshot or _empty_snapshot(refreshing=True)
        return _public_snapshot(latest, refreshing=True)
    try:
        previous = _read_snapshot() or _snapshot
        registry = await _read_registry()
        endpoints = _merge_endpoints(registry)
        concurrency = max(1, int(os.getenv("MCP_CATALOG__CONCURRENCY", str(_DEFAULT_CONCURRENCY))))
        semaphore = asyncio.Semaphore(concurrency)
        probes = await asyncio.gather(*(_probe_endpoint(endpoint, semaphore) for endpoint in endpoints))
        result = _assemble_catalog(registry, endpoints, probes, previous=previous)
        try:
            _write_snapshot(result)
        except OSError as exc:
            logger.warning("MCP catalogue snapshot could not be saved: %s", type(exc).__name__)
        _snapshot = result
        _snapshot_location = location
        return result
    finally:
        try:
            _lease_path().unlink(missing_ok=True)
        except OSError:
            pass


async def request_refresh(*, force: bool = False) -> bool:
    """Start a single-flight refresh. Return True only for a newly started task."""
    global _refresh_task
    async with _state_lock:
        if _refresh_task is not None and not _refresh_task.done():
            return False
        if _lease_active():
            return False
        current = _read_snapshot() or _snapshot
        if not force and current is not None and not _is_stale(current):
            return False
        _refresh_task = asyncio.create_task(refresh_catalog(), name="mcp-tool-catalog-refresh")
        return True


async def get_catalog(*, refresh_if_stale: bool = True) -> Dict[str, Any]:
    """Return immediately from memory/disk and refresh stale data in background."""
    global _snapshot, _snapshot_location
    location = _snapshot_path()
    if _snapshot_location is not None and _snapshot_location != location:
        _snapshot = None
    _snapshot_location = location
    disk = _read_snapshot()
    if disk and (
        _snapshot is None
        or (_parse_time(disk.get("checked_at")) or datetime.min.replace(tzinfo=timezone.utc))
        > (_parse_time(_snapshot.get("checked_at")) or datetime.min.replace(tzinfo=timezone.utc))
    ):
        _snapshot = disk
    if refresh_if_stale:
        await request_refresh(force=False)
    refreshing = (_refresh_task is not None and not _refresh_task.done()) or _lease_active()
    return _public_snapshot(_snapshot or _empty_snapshot(refreshing=refreshing), refreshing=refreshing)


async def close_catalog() -> None:
    """Await an in-flight refresh during graceful application shutdown."""
    task = _refresh_task
    if task is not None and not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        except asyncio.CancelledError:
            task.cancel()


__all__ = [
    "get_catalog",
    "request_refresh",
    "refresh_catalog",
    "close_catalog",
    "load_presentation",
    "resolve_tool_presentation",
]
