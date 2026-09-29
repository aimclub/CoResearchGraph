"""Tools for fedotmas inference"""

import asyncio
import logging
from typing import List, Optional, Dict, Any

from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from CoScientist.storage import RetrievalToolResult
from CoScientist.experiments.capabilities.contracts import contract_metadata

from rag_tools import create_manager, MCPServer
from rag_tools.storage import PostgresClient
from rag_tools.config.settings import get_settings
from rag_tools.retrieval import APIReranker, BM25Reranker, HybridReranker
from rag_tools.storage.models import RetrievalResult

from CoScientist.tools.embedder_shim import SafeAPIEmbedder

settings = get_settings()
_logger = logging.getLogger(__name__)

# The full description + input_schema are returned INLINE to the calling agent
# (planner/orchestrator) in each retrieve_tools response. The session-state
# `accumulated_tools` is re-injected into the downstream rerankers' prompts on
# every turn, so descriptions are capped. Structured contracts remain in state;
# each downstream prompt owns its compact projection of those contracts.
_ACCUM_DESC_CAP = 600
# Parallel retrieve_tools calls race on read-modify-write of accumulated_tools
# when ADK clones state per parallel tool invocation. Serialize merges via a
# process-global session buffer so no retrieved (server_id, tool) pair is dropped.
_ACCUM_LOCK = asyncio.Lock()
_SESSION_ACCUMULATED: dict[str, list] = {}


def _retrieval_session_key(tool_context: Optional[ToolContext]) -> str:
    if tool_context is None:
        return "no-session"
    sid = getattr(tool_context, "session_id", None)
    if sid:
        return str(sid)
    inv = getattr(tool_context, "_invocation_context", None)
    session = getattr(inv, "session", None) if inv is not None else None
    sid = getattr(session, "id", None) if session is not None else None
    return str(sid or id(tool_context))


def clear_session_accumulated_tools(session_key: str | None = None) -> None:
    """Drop process-global retrieval buffer (called when rerank clears state)."""
    if session_key is None:
        _SESSION_ACCUMULATED.clear()
        return
    _SESSION_ACCUMULATED.pop(str(session_key), None)


def _merge_accumulated_tools(
    accumulated: list,
    results: List[RetrievalToolResult],
    *,
    query: str,
) -> list:
    """Merge retrieval hits by (server_id, tool); preserve existing tool_index."""
    by_key = {
        (str(t.get("server_id") or ""), str(t.get("tool") or "")): t
        for t in accumulated
        if isinstance(t, dict) and t.get("tool") and t.get("server_id")
    }
    last_idx = max((int(t.get("tool_index") or 0) for t in by_key.values()), default=0) + 1
    for tool_result in results:
        key = (str(tool_result.server_id or ""), str(tool_result.tool or ""))
        if not key[0] or not key[1]:
            continue
        existing = by_key.get(key)
        if existing is None:
            row = {
                "tool": tool_result.tool,
                "server_id": tool_result.server_id,
                "description": (tool_result.description or "")[:_ACCUM_DESC_CAP],
                "input_schema": tool_result.input_schema,
                "score": tool_result.score,
                "url": getattr(tool_result, "url", None),
                "tool_index": last_idx,
                "retrieval_query": query,
                **contract_metadata(tool_result.model_dump()),
            }
            by_key[key] = row
            last_idx += 1
        else:
            if not existing.get("input_schema") and tool_result.input_schema:
                existing["input_schema"] = tool_result.input_schema
            if not existing.get("url") and getattr(tool_result, "url", None):
                existing["url"] = tool_result.url
            existing.update(contract_metadata(tool_result.model_dump()))
    # Stable order by tool_index for reranker index alignment.
    return sorted(by_key.values(), key=lambda t: int(t.get("tool_index") or 0))


async def _fetch_full_tool_meta(server_ids) -> Dict[tuple, Dict[str, Any]]:
    """Map exact registry identities to descriptions and input/output contracts."""
    meta: Dict[tuple, Dict[str, Any]] = {}
    if not server_ids:
        return meta
    postgres = PostgresClient(settings.postgres)
    try:
        await postgres.initialize()
        server_urls: Dict[str, str] = {}
        for sid in server_ids:
            try:
                srv = await postgres.get_server(sid)
            except Exception as exc:
                _logger.warning("retrieve_tools: could not fetch server %r: %s", sid, exc)
                srv = None
            url = getattr(srv, "url", None) if srv is not None else None
            protocol = getattr(srv, "protocol", None) if srv is not None else None
            if protocol == "http" and isinstance(url, str) and url.startswith("http"):
                server_urls[sid] = url
            try:
                tools = await postgres.get_tools_by_server(sid)
            except Exception as exc:
                _logger.warning(
                    "retrieve_tools: could not fetch full metadata for server %r: %s",
                    sid, exc,
                )
                continue
            for t in tools:
                name = getattr(t, "name", None)
                if not name:
                    continue
                schema = getattr(t, "input_schema", None)
                if schema is not None and not isinstance(schema, dict):
                    dump = getattr(schema, "model_dump", None)
                    schema = dump() if callable(dump) else getattr(schema, "__dict__", None)
                output_schema = getattr(t, "output_schema", None)
                if output_schema is not None and not isinstance(output_schema, dict):
                    dump = getattr(output_schema, "model_dump", None)
                    output_schema = dump() if callable(dump) else getattr(output_schema, "__dict__", None)
                meta[(sid, name)] = {
                    "description": getattr(t, "description", None),
                    "input_schema": schema,
                    "url": server_urls.get(sid),
                    **contract_metadata({
                        "input_schema": schema,
                        "output_schema": output_schema,
                        "data_contract": getattr(t, "data_contract", None),
                    }),
                }
    finally:
        await postgres.close()
    return meta


class RetrievalToolSet(BaseToolset):
    """Toolset for rag tool usage"""
    def __init__(self, prefix: str = "rag_"):
        super().__init__()
        self.tool_name_prefix = prefix
        self.wrapper = None

    def get_tools(
        self,
        readonly_context: Optional[ReadonlyContext]
    ) -> List[BaseTool]:

        tools = [self.retrieve_tools, self.get_server_info]

        return tools
        
    async def close(self) -> None:
        await asyncio.sleep(0)  # Placeholder for async cleanup if needed


    async def retrieve_tools(self, query: str,
                                    tool_context: ToolContext = None
                                    ) -> Dict[str, Any]:
        """
        Tool for retrieving MCP tools from DB using RAG. 
        
        Args:
            query: query to use for tools lookup in database using RAG.
        
        Returns:
            List ot the most relevant tools in db which can be used to solve the task .
        """
        manager = None
        try:
            embedder = SafeAPIEmbedder(settings.api_embedding)
            api_reranker = APIReranker(settings.api_reranker)
            bm2_reranker = BM25Reranker(settings.bm_reranker)
            reranker = HybridReranker([api_reranker, bm2_reranker], settings.hybrid_reranker)
            manager = await create_manager(settings, embedder, reranker)

            retrieved_tools: List[RetrievalResult] = await manager.retrieve_tools(
                query=query,
                top_k=settings.rag.default_top_k,
                rerank=True,
                rerank_top_k=settings.rag.rerank_top_k,
                min_score=settings.rag.min_relevance_score)

            # The RAG layer returns only a truncated (~chunk_size) chunk of each
            # tool's description and drops its argument schema — so the calling
            # agent never sees what a tool RETURNS or which arguments it accepts.
            # Re-fetch the FULL description + input_schema from the registry.
            full_meta = await _fetch_full_tool_meta(
                {r.server_id for r in retrieved_tools}
            )

            results = [
                RetrievalToolResult(
                    tool=r.name,
                    server_id=r.server_id,
                    description=full_meta.get((r.server_id, r.name), {}).get("description") or r.description,
                    input_schema=full_meta.get((r.server_id, r.name), {}).get("input_schema"),
                    output_schema=full_meta.get((r.server_id, r.name), {}).get("output_schema"),
                    data_contract=full_meta.get((r.server_id, r.name), {}).get("data_contract"),
                    score=r.rerank_score,
                    url=full_meta.get((r.server_id, r.name), {}).get("url"),
                )
                for r in retrieved_tools
            ]
        except Exception as e:  # noqa: BLE001
            # The tool index / DB being unreachable (e.g. no VPN, timeout) must
            # NOT crash the whole run — return a graceful error so the agent can
            # proceed or abstain (e.g. NO_MATCHING_TOOL → CoderAgent).
            _logger.exception("retrieve_tools unavailable: %s", e)
            acc = tool_context.state.get('accumulated_tools', []) if tool_context is not None else []
            return {
                "status": "error",
                "result": [],
                "accumulated_count": len(acc),
                "message": f"Tool retrieval is unavailable right now (tool index/DB unreachable): {type(e).__name__}: {e}",
            }
        finally:
            # Always release the manager's DB/HTTP connections, even on error.
            if manager is not None:
                try:
                    await manager.close()
                except Exception:  # noqa: BLE001
                    pass

        if tool_context is None:
            return {
                "status": "success",
                "result": [r.model_dump() for r in results],
                "accumulated_count": len(results),
                "message": f"Retrieved {len(results)} tools (no session accumulation).",
            }

        # ACCUMULATE into process-global session buffer then state (locked:
        # parallel retrieve_tools must not clobber each other's merges when ADK
        # hands each call a forked state snapshot).
        async with _ACCUM_LOCK:
            session_key = _retrieval_session_key(tool_context)
            prior = list(
                _SESSION_ACCUMULATED.get(session_key)
                or tool_context.state.get("accumulated_tools")
                or []
            )
            accumulated = _merge_accumulated_tools(prior, results, query=query)
            _SESSION_ACCUMULATED[session_key] = accumulated
            tool_context.state["accumulated_tools"] = list(accumulated)
            tool_context.state["retrieval_queries"] = list(
                tool_context.state.get("retrieval_queries") or []
            ) + [query]
            # Durable experiment inventory — survives rerank clearing accumulated_tools.
            try:
                from CoScientist.experiments.context.builder import (
                    RETRIEVED_CAPABILITIES_KEY,
                    _merge_capabilities,
                    _normalize_capabilities,
                )

                tool_context.state[RETRIEVED_CAPABILITIES_KEY] = _merge_capabilities(
                    tool_context.state.get(RETRIEVED_CAPABILITIES_KEY),
                    _normalize_capabilities(accumulated),
                )
            except Exception:  # noqa: BLE001 — inventory is best-effort
                pass

        return {
            "status": "success",
            "result": [r.model_dump() for r in results],
            "accumulated_count": len(accumulated),
            "message": f"Retrieved {len(results)} tools. Total accumulated: {len(accumulated)}."
        }

    async def get_server_info(self, server_id: str) -> Dict[str, Any]:
        """
        Returns MCP server metadata. 
        
        Args:
            server_id: server id to look up for.
        
        Returns:
            Server metadata.
        """

        postgres = PostgresClient(settings.postgres)
        try:
            await postgres.initialize()
            server: MCPServer = await postgres.get_server(server_id)
        except Exception as e:  # noqa: BLE001 — DB unreachable must not crash the run
            _logger.warning("get_server_info unavailable: %r", e)
            return {"status": "error", "result": None,
                    "message": f"Server lookup unavailable (tool index/DB unreachable): {e}"}
        finally:
            # Always release the DB connection, even if the lookup raises.
            try:
                await postgres.close()
            except Exception:  # noqa: BLE001
                pass

        # get_server returns None when no server matches the id — don't call
        # model_dump on it (AttributeError), report a clean not-found instead.
        if server is None:
            return {
                "status": "error",
                "result": None,
                "message": f"No MCP server found for server_id={server_id!r}.",
            }

        return {
            "status": "success",
            # model_dump(mode="json") so datetime/enum fields become JSON-native
            # — a raw MCPServer (or its datetime fields) is not JSON serializable
            # and would crash any json.dumps consumer (web ws, Opik trace, ADK state).
            "result": server.model_dump(mode="json"),
        }



    
retrieval_toolset = RetrievalToolSet()
retrieval_toolset_instance = retrieval_toolset.get_tools(None)

