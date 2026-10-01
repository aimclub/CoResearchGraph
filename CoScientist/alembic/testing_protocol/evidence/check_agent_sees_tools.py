#!/usr/bin/env python3
"""Does the experiment agent actually see a freshly served MCP server as tools?

`build_mcp_server` publishes the new server's address into the session state
under ``deployed_mcps``. The experiment agent's toolset (`DynamicMCPToolset`)
re-reads that state every turn and connects to the server, so its tools become
the agent's own. This script feeds the toolset the same state and prints the
tool names it hands to the agent.

Usage: python nirsii/evidence/check_agent_sees_tools.py <mcp_url> [name]
"""
import asyncio
import sys
import types

from dotenv import load_dotenv

load_dotenv()

from CoScientist.tools.dynamic_tools import DynamicMCPToolset      # noqa: E402

URL = sys.argv[1]
NAME = sys.argv[2] if len(sys.argv) > 2 else "autogen"


async def main() -> None:
    state = {"deployed_mcps": [{"url": URL, "name": NAME}]}
    ctx = types.SimpleNamespace(state=state)
    toolset = DynamicMCPToolset()
    tools = await toolset.get_tools(ctx)
    print("session state seen by the agent:")
    print(f"  deployed_mcps = {state['deployed_mcps']}")
    print(f"\ntools handed to the experiment agent: {len(tools)}")
    for t in tools:
        print(f"  - {t.name}")
    await toolset.close()


asyncio.run(main())
