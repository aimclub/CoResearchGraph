"""One process, the real path: build_mcp_server (reuse) -> what the agent gets."""
import asyncio, types
from dotenv import load_dotenv; load_dotenv()
import CoScientist.tools.alembic_tools as at
from CoScientist.tools.dynamic_tools import DynamicMCPToolset

REPO = "https://github.com/microsoft/autogen"

async def main():
    snap = await at.build_mcp_server(REPO, tool_context=None)
    print("build_mcp_server ->", {k: snap.get(k) for k in
          ("status", "job_id", "mcp_url", "container", "note")})
    print("_JOBS:", {k: (v.get("status"), v.get("mcp_url"), v.get("container"))
                     for k, v in at._JOBS.items()})
    print("live_build_servers ->", at.live_build_servers())

    ctx = types.SimpleNamespace(state={})   # session state deliberately empty
    ts = DynamicMCPToolset()
    tools = await ts.get_tools(ctx)
    print("tools handed to the experiment agent:", [t.name for t in tools])
    await ts.close()

asyncio.run(main())
