"""MCP toolsets of the microfluidics services.

Built with the core helper (CoScientist/tools/research_tools.py), so they share
its timeouts, retries and connection resilience. Unset URLs build nothing and
the agents keep their stubs (see bindings.py).
"""
from CoScientist.config import get_settings
from CoScientist.microfluidics.settings import get_microfluidics_settings
from CoScientist.tools.research_tools import _http_mcp_toolset
from CoScientist.utils.selective_proxy import create_mcp_proxy_httpx_factory

settings = get_settings()
mf = get_microfluidics_settings()

# ---------------------------------------------------------------------------
# Microfluidics MCP server — Bearer-token auth, optionally proxied
# (MCP__MICROFLUIDICS_URL / MCP__MICROFLUIDICS_API_KEY in .env).
# ---------------------------------------------------------------------------
_microfluidics_headers = (
    {"Authorization": f"Bearer {mf.microfluidics_api_key}"}
    if mf.microfluidics_api_key
    else {}
)

microfluidics_toolset_instance = _http_mcp_toolset(
    mf.microfluidics_url,
    headers=_microfluidics_headers,
    httpx_client_factory=(
        create_mcp_proxy_httpx_factory(
            settings.services.proxy_url,
            enabled_fn=settings.web.use_proxy,
        )
        if settings.services.proxy_url
        else None
    ),
)

# ---------------------------------------------------------------------------
# Microfluidics case services — economics (stage 5) and CFD (stage 9).
# Internal-network servers: no proxy. Only the CFD server takes a key
# (X-API-Key). Unset URLs leave the agents on their stubs.
# ---------------------------------------------------------------------------
_CFD_TIMEOUT = 60 * 15.0  # a flow simulation can run for minutes
# rank_routes_by_cost resolves every name through PubChem / CIR before pricing:
# a few routes take minutes. The read timeout bounds a whole tool call.
_ECONOMICS_TIMEOUT = 60 * 10.0

# The economics server's tools, as it names them (checked against the server in
# tests/unit/microfluidics/fixtures/economics_mcp/tools.json). The filter keeps the agent's surface
# exactly what bindings.py documents, even if the server grows new tools.
ECONOMICS_MCP_TOOLS = [
    "search_reagents_by_name",
    "get_price",
    "search_by_structure",
    "resolve_chemicals",
    "estimate_synthesis_cost",
    "rank_routes_by_cost",
]

microfluidic_economic_toolset_instance = _http_mcp_toolset(
    mf.microfluidic_economic_url,
    sse_read_timeout=_ECONOMICS_TIMEOUT,
    timeout=_ECONOMICS_TIMEOUT,
    tool_filter=ECONOMICS_MCP_TOOLS,
    # Lookups and cost estimates only: repeating a lost call is harmless.
    retry_calls=True,
)

# The CFD service's tools (tests/unit/microfluidics/fixtures/cfd_mcp/tools.json), filtered like the
# economics ones so the surface is exactly what bindings.py documents.
CFD_MCP_TOOLS = [
    "cfd_list_reactors",
    "cfd_run_reactor_experiment",
    "cfd_get_experiment_result",
    "cfd_list_artifacts",
    "cfd_cancel_run",
]

microfluidic_cfd_toolset_instance = _http_mcp_toolset(
    mf.microfluidic_cfd_url,
    # Above the run's own wait_seconds (600 by default): the call returns
    # "pending" on its own instead of the client timing out first.
    sse_read_timeout=_CFD_TIMEOUT,
    timeout=_CFD_TIMEOUT,
    tool_filter=CFD_MCP_TOOLS,
    headers=(
        {"X-API-Key": mf.microfluidic_cfd_api_key}
        if mf.microfluidic_cfd_api_key
        else {}
    ),
)
