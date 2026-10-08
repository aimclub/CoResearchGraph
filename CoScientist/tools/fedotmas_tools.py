"""Tools for fedotmas inference"""

import asyncio
from typing import List, Optional, Dict, Any

from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from fedotmas import MAS, HttpMCPServer
from fedotmas.plugins import LoggingPlugin, WebSearchLimitPlugin

try:  # Tracing is optional telemetry, and newer than the pinned-by-nothing dep.
    from fedotmas.plugins import LangfusePlugin
except ImportError:  # pragma: no cover - depends on the installed FEDOT.MAS
    # The dependency is tracked by git URL with no version, so an environment
    # can easily predate this plugin. Losing traces is a nuisance; refusing to
    # import is the whole application failing to start over telemetry.
    LangfusePlugin = None

from CoScientist.tools.fedot_artifact_plugin import ArtifactCapturePlugin
from CoScientist.logging.metrics import UsageMetricsPlugin
from rag_tools import MCPServer
from rag_tools.storage import PostgresClient
from rag_tools.config.settings import get_settings

settings = get_settings()


class FedotMASToolset(BaseToolset):
    """Toolset for fedotmas usage"""
    def __init__(self, prefix: str = "fedot_"):
        super().__init__()
        self.tool_name_prefix = prefix

    def get_tools(
        self,
        readonly_context: Optional[ReadonlyContext]
    ) -> List[BaseTool]:

        tools = [self.fedot_tool]
        return tools
        
    async def close(self) -> None:
        await asyncio.sleep(0)  # Placeholder for async cleanup if needed

    async def fedot_tool(self, task_description: str,  tool_context: ToolContext = None) -> Dict[str, Any]:
        """
        Tool for generating and executing multi-agent pipelines via FEDOT.MAS. Use it for experiments completion and calculations
        
        Args:
            task_description: Clear description of the task, including goals,
                            inputs, constraints, and expected outputs.
        
        Returns:
            Result of the executed MAS pipeline.
        """
        state = tool_context.state if tool_context is not None else {}
        candidates = state.get('filtered_tools') or state.get('accumulated_tools') or []

        servers = []
        postgres = PostgresClient(settings.postgres)
        try:
            await postgres.initialize()
            try:
                server_ids = {t['server_id'] for t in candidates if t.get('server_id')}
                servers = [await postgres.get_server(server_id) for server_id in server_ids]
            finally:
                # Always release the DB connection, even if a lookup raised.
                await postgres.close()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "fedot_tool: database lookup failed (%s); proceeding with empty servers list", exc
            )

        servers = [server for server in servers if (server is not None and server.protocol == 'http')]
        servers_payload = {server.name: HttpMCPServer(url=server.url, description=server.description)
                           for server in servers}

        # Deployed web MCPs are stored in state as dicts (see WebToolsDeployerAgent).
        web_servers = state.get('deployed_mcps', [])
        web_servers_payload = {
            s['name']: HttpMCPServer(url=s['url'], description=s.get('description', ''))
            for s in web_servers
        }
        servers_payload.update(web_servers_payload)

        # F010.A3/A4: an after_tool_callback plugin captures S3 artifact links
        # (results_presigned_url) at the tool-call boundary, BEFORE FEDOT.MAS sub-agents
        # paraphrase them away / hallucinate molecules.
        # NB: passing plugins= REPLACES MAS defaults, so re-include them.
        cap = ArtifactCapturePlugin()
        # NOTE (F015): do NOT impose a short FEDOT timeout — confirm the pipeline produces
        # CORRECT results first, optimize stage latency later. timeout=None = unbounded.
        # The except branch below still returns captured artifacts on a FEDOT error, so a
        # link produced before a failure is never lost.
        #
        # The ONE exception is the reranker-fallback path: nobody chose to run
        # FEDOT there, a malformed model reply did, so it gets a bounded budget.
        FEDOT_TIMEOUT_S = None
        if not state.get('filtered_tools') and candidates:
            try:
                from CoScientist.config import get_settings as _cs_settings
                FEDOT_TIMEOUT_S = _cs_settings().web.fedot_fallback_timeout_s or None
            except Exception:  # noqa: BLE001 — no budget is better than no run
                FEDOT_TIMEOUT_S = None
        result = None
        status, err = "success", None
        try:
            mas = MAS(
                mcp_servers=servers_payload,
                # UsageMetricsPlugin bills FEDOT.MAS sub-agents' own LLM traffic
                # against them, same as any other AgentTool sub-runner — without
                # it their model calls are invisible to the cost ledger.
                plugins=[
                    LoggingPlugin(),
                    WebSearchLimitPlugin(max_calls_per_agent=4),
                    *([LangfusePlugin(trace_name="coscientist:fedot")]
                      if LangfusePlugin is not None else []),
                    cap,
                    UsageMetricsPlugin(),
                ],
            )
            result = await mas.run(task_description, timeout=FEDOT_TIMEOUT_S)
        except (asyncio.TimeoutError, TimeoutError):
            status, err = "timeout", f"FEDOT.MAS exceeded {FEDOT_TIMEOUT_S}s"
        except Exception as e:
            status, err = "error", f"FEDOT.MAS run failed: {e}"

        # Fallback (F010.A4): scan the final MAS state for presigned URLs the plugin may
        # have missed (only when a result actually came back).
        if result is not None:
            import re as _re
            import json as _json
            try:
                _txt = _json.dumps(result, default=str, ensure_ascii=False)
            except Exception:
                _txt = str(result)
            _known = {a.get("url") for a in cap.captured}
            for _u in dict.fromkeys(_re.findall(r"https?://[^\s\"'<>)\\]+X-Amz-[^\s\"'<>)\\]+", _txt)):
                if _u not in _known:
                    cap.captured.append({"url": _u, "tool": "fedot_state_scan"})

        # Surface the REAL artifacts in the return value AND shared session state — so the
        # link survives even when FEDOT.MAS timed out AFTER generation (F015 Mode B fix).
        if cap.captured and tool_context is not None:
            tool_context.state["fedot_artifacts"] = cap.captured

        ret = {"status": status, "artifacts": cap.captured}
        if result is not None:
            ret["result"] = result
        if err:
            ret["error"] = err
        return ret

    
fedot_toolset = FedotMASToolset()
fedot_toolset_instance = fedot_toolset.get_tools(None)
