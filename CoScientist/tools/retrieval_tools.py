"""Tools for fedotmas inference"""

import asyncio
import logging
from typing import List, Optional, Dict, Any

from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from CoScientist.storage import RetrievalToolResult

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
# every turn, so there we keep a capped description and drop the schema to avoid
# unbounded context bloat (and the schema-validation pressure it puts on the
# structured-output rerankers).
_ACCUM_DESC_CAP = 600


async def _fetch_full_tool_meta(server_ids) -> Dict[tuple, Dict[str, Any]]:
    """Map ``(server_id, tool_name) -> {description, input_schema}`` from the registry.

    The RAG retrieval path returns a truncated description chunk and no schema;
    the full, authoritative tool metadata lives in the Postgres registry. We
    fetch it once per unique server. Best-effort: a server that fails to resolve
    is simply skipped (the caller falls back to the RAG chunk).
    """
    meta: Dict[tuple, Dict[str, Any]] = {}
    if not server_ids:
        return meta
    postgres = PostgresClient(settings.postgres)
    try:
        await postgres.initialize()
        for sid in server_ids:
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
                    # pydantic model / other -> plain dict for JSON serialisation
                    dump = getattr(schema, "model_dump", None)
                    schema = dump() if callable(dump) else getattr(schema, "__dict__", None)
                meta[(sid, name)] = {
                    "description": getattr(t, "description", None),
                    "input_schema": schema,
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
                    score=r.rerank_score,
                )
                for r in retrieved_tools
            ]
        except Exception as e:  # noqa: BLE001
            # The tool index / DB being unreachable (e.g. no VPN, timeout) must
            # NOT crash the whole run — return a graceful error so the agent can
            # proceed or abstain (e.g. NO_MATCHING_TOOL → CoderAgent).
            _logger.warning("retrieve_tools unavailable: %r", e)
            acc = tool_context.state.get('accumulated_tools', []) if tool_context is not None else []
            return {
                "status": "error",
                "result": [],
                "accumulated_count": len(acc),
                "message": f"Tool retrieval is unavailable right now (tool index/DB unreachable): {e}",
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

        # ACCUMULATE into state
        accumulated = tool_context.state.get('accumulated_tools', [])
        existing_tools = {t['tool'] for t in accumulated}
        last_idx = len(accumulated) + 1

        for tool_result in results:
            if tool_result.tool not in existing_tools:
                accumulated.append({
                    'tool': tool_result.tool,
                    'server_id': tool_result.server_id,
                    # Capped here (not in the inline response) — this dict is
                    # re-injected into the rerankers' prompts every turn.
                    'description': (tool_result.description or "")[:_ACCUM_DESC_CAP],
                    'score': tool_result.score,
                    'tool_index': last_idx,
                    'retrieval_query': query,  # Track which query found this
                })
                last_idx += 1
        
        tool_context.state['accumulated_tools'] = accumulated
        tool_context.state['retrieval_queries'] = tool_context.state.get('retrieval_queries', []) + [query]

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

