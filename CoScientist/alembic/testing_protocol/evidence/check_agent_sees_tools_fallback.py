import asyncio, types
from dotenv import load_dotenv; load_dotenv()
import CoScientist.tools.alembic_tools as at
from CoScientist.tools.dynamic_tools import DynamicMCPToolset

at._JOBS["x"] = {"job_id": "x", "repo_url": "https://github.com/microsoft/autogen",
                 "status": "done", "mcp_url": "http://localhost:22129/mcp",
                 "container": "alembic-serve-autogen-aeca09"}

async def main():
    ctx = types.SimpleNamespace(state={})          # session state carries nothing
    ts = DynamicMCPToolset()
    tools = await ts.get_tools(ctx)
    print("state:", ctx.state)
    print("tools from the process-local build registry:", [t.name for t in tools])
    await ts.close()
asyncio.run(main())
