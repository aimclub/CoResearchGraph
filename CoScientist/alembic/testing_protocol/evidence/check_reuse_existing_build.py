#!/usr/bin/env python3
"""Deterministic check of the "tool is already built" path.

Calls the same function the McpBuilderAgent calls. When a repository was built
before and its image is still on the host, the module does not rebuild: it
re-serves the existing image and reports the live MCP address.

Usage: python nirsii/evidence/check_reuse_existing_build.py <repo_url>
"""
import asyncio
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from CoScientist.tools.alembic_tools import (      # noqa: E402
    build_mcp_server, check_mcp_build,
)

REPO = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/microsoft/autogen"


async def main() -> None:
    out = await build_mcp_server(REPO)
    print("build_mcp_server(%r) ->" % REPO)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:1200])

    job = out.get("job_id")
    for _ in range(120):
        if out.get("status") == "done":
            break
        await asyncio.sleep(5)
        out = await check_mcp_build(job)
    print("\ncheck_mcp_build(%r) ->" % job)
    print(json.dumps({k: v for k, v in out.items() if k != "log_tail"},
                     ensure_ascii=False, indent=2)[:1200])


asyncio.run(main())
