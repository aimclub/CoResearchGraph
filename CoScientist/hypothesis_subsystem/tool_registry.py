"""
HypothesisToolRegistry — dynamic registration and selection of hypothesis tools.

Starts with MooseChemTool pre-registered. Additional tools can be registered
at runtime via register() without changing any core agent code.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from CoScientist.hypothesis_subsystem.base_tool import BaseHypothesisTool
from CoScientist.hypothesis_subsystem.models import HypothesisQuery, ToolCatalog, ToolResult, ValidationToolInfo


# Static fallback tool catalog used when the FedotMAS RAG database is
# unreachable — lists the known tools from chemical-mcp-server so the
# generator can still produce tool-aware hypotheses.
_STATIC_FALLBACK_TOOLS: List[ValidationToolInfo] = [
    ValidationToolInfo(
        name="compute_property",
        description="Compute molecular properties (MW, LogP, QED, etc.) from SMILES.",
        limitations="Requires valid SMILES; ≤500 Da recommended.",
    ),
    ValidationToolInfo(
        name="run_docking",
        description="Run molecular docking simulation (AutoDock Vina).",
        limitations="Requires protein PDB ID and ligand SMILES; ≤500 Da.",
    ),
    ValidationToolInfo(
        name="predict_affinity",
        description="Predict binding affinity (IC50/Kd) from ligand structure.",
        limitations="Requires SMILES; model trained on kinase inhibitors.",
    ),
    ValidationToolInfo(
        name="estimate_synthesizability",
        description="Estimate synthetic accessibility (SA score) from SMILES.",
        limitations="Requires valid SMILES; heuristic score (1-10).",
    ),
    ValidationToolInfo(
        name="retrosynthesis_plan",
        description="Generate retrosynthesis routes for a target molecule.",
        limitations="Requires SMILES; commercial availability of precursors not guaranteed.",
    ),
]


class HypothesisToolRegistry:
    """
    Registry of hypothesis-generation strategy tools.

    Usage:
        registry = HypothesisToolRegistry()
        registry.register(MooseChemTool(model="gpt-4"))
        result = await registry.invoke_with_fallback(query, preferred="MooseChem")
    """

    def __init__(self, tools: Optional[List[BaseHypothesisTool]] = None):
        self._tools: Dict[str, BaseHypothesisTool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, tool: BaseHypothesisTool) -> None:
        """
        Register a hypothesis-generation tool.

        Args:
            tool: A BaseHypothesisTool subclass instance.

        Raises:
            ValueError: If a tool with the same strategy_type is already registered
                        (call unregister first to replace).
        """
        if tool.strategy_type in self._tools:
            raise ValueError(
                f"Tool '{tool.strategy_type}' is already registered. "
                f"Unregister it first."
            )
        self._tools[tool.strategy_type] = tool

    def unregister(self, strategy_type: str) -> None:
        """
        Remove a tool from the registry by strategy type.

        Args:
            strategy_type: The strategy identifier to remove.
        """
        self._tools.pop(strategy_type, None)

    def get(self, strategy_type: str) -> Optional[BaseHypothesisTool]:
        """
        Get a tool by strategy type.

        Args:
            strategy_type: The strategy identifier.

        Returns:
            The tool instance, or None if not registered.
        """
        return self._tools.get(strategy_type)

    def list_strategies(self) -> List[str]:
        """
        Return all registered strategy identifiers.
        """
        return list(self._tools.keys())

    def describe_all(self) -> Dict[str, str]:
        """
        Return a dict of strategy_type → description for all registered tools.
        """
        return {st: tool.describe() for st, tool in self._tools.items()}

    # ------------------------------------------------------------------
    # Validation tool discovery (external MCP RAG)
    # ------------------------------------------------------------------

    async def discover_validation_tools(self, research_question: str) -> ToolCatalog:
        """Query the FedotMAS MCP RAG database for validation tools.

        Returns a ToolCatalog with tools that can potentially test/validate
        hypotheses for the given research question. Falls back to a static
        list (chemical-mcp-server tools) when the RAG database is unreachable.
        """
        try:
            from rag_tools import create_manager
            from rag_tools.config.settings import get_settings as rag_get_settings
            from rag_tools.storage import PostgresClient
            from rag_tools.retrieval import (
                APIEmbedder,
                APIReranker,
                BM25Reranker,
                HybridReranker,
            )

            rag_settings = rag_get_settings()
            embedder = APIEmbedder(rag_settings.api_embedding)
            api_reranker = APIReranker(rag_settings.api_reranker)
            bm25_reranker = BM25Reranker(rag_settings.bm_reranker)
            reranker = HybridReranker([api_reranker, bm25_reranker], rag_settings.hybrid_reranker)
            manager = await create_manager(rag_settings, embedder, reranker)

            try:
                retrieved = await manager.retrieve_tools(
                    query=research_question,
                    top_k=rag_settings.rag.default_top_k,
                    rerank=True,
                    rerank_top_k=rag_settings.rag.rerank_top_k,
                    min_score=rag_settings.rag.min_relevance_score,
                )
                postgres = PostgresClient(rag_settings.postgres)
                full_meta: Dict[tuple, dict] = {}
                try:
                    await postgres.initialize()
                    for sid in {r.server_id for r in retrieved}:
                        try:
                            tools = await postgres.get_tools_by_server(sid)
                        except Exception:
                            continue
                        for t in tools:
                            name = getattr(t, "name", None)
                            if name:
                                schema = getattr(t, "input_schema", None)
                                if schema is not None and not isinstance(schema, dict):
                                    dump = getattr(schema, "model_dump", None)
                                    schema = dump() if callable(dump) else getattr(schema, "__dict__", None)
                                full_meta[(sid, name)] = {
                                    "description": getattr(t, "description", None),
                                    "input_schema": schema,
                                }
                finally:
                    await postgres.close()

                tools = [
                    ValidationToolInfo(
                        name=r.name,
                        description=full_meta.get((r.server_id, r.name), {}).get("description") or r.description,
                        server_id=r.server_id,
                        input_schema=full_meta.get((r.server_id, r.name), {}).get("input_schema"),
                        retrieval_score=r.rerank_score if hasattr(r, "rerank_score") else None,
                    )
                    for r in retrieved
                ]
                return ToolCatalog(
                    tools=tools,
                    retrieval_query=research_question,
                    source="rag",
                )
            finally:
                await manager.close()
        except Exception:
            return ToolCatalog(
                tools=list(_STATIC_FALLBACK_TOOLS),
                retrieval_query=research_question,
                source="static_fallback",
            )

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    async def invoke_with_fallback(
        self,
        query: HypothesisQuery,
        preferred: Optional[str] = None,
    ) -> ToolResult:
        """
        Invoke the preferred tool, falling back to first available if missing.

        Args:
            query: The hypothesis generation query.
            preferred: Preferred strategy type. If None or not found, uses the
                       first registered tool.

        Returns:
            ToolResult from the invoked strategy.

        Raises:
            RuntimeError: If no tools are registered.
        """
        if not self._tools:
            raise RuntimeError("No hypothesis tools registered in the registry.")

        tool = None
        if preferred:
            tool = self._tools.get(preferred)

        if tool is None:
            # Fall back to first available
            first_key = next(iter(self._tools))
            tool = self._tools[first_key]

        if not tool.validate_query(query):
            return ToolResult(
                strategy_type=tool.strategy_type,
                hypotheses=[],
                metadata={"reason": "Query validation failed"},
                success=False,
                error_message=f"Query is not valid for tool '{tool.strategy_type}'.",
            )

        return await tool.invoke(query)

    async def invoke_all(
        self,
        query: HypothesisQuery,
    ) -> Dict[str, ToolResult]:
        """
        Invoke ALL registered tools and return results keyed by strategy_type.

        Args:
            query: The hypothesis generation query.

        Returns:
            Dict mapping strategy_type to ToolResult.
        """
        results: Dict[str, ToolResult] = {}
        for st, tool in self._tools.items():
            if tool.validate_query(query):
                results[st] = await tool.invoke(query)
            else:
                results[st] = ToolResult(
                    strategy_type=st,
                    hypotheses=[],
                    metadata={"reason": "Query validation failed"},
                    success=False,
                    error_message=f"Query is not valid for tool '{st}'.",
                )
        return results
