#!/usr/bin/env python3
"""Query a served alembic MCP server over the MCP protocol: list its tools,
validate each tool's input JSON Schema, and (optionally) call one tool.

Usage: python check_mcp_server.py <mcp_url> [tool_name] [json_args]
"""
import asyncio, json, sys

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from jsonschema import Draft202012Validator


async def main() -> None:
    url = sys.argv[1]
    async with Client(StreamableHttpTransport(url)) as client:
        tools = await client.list_tools()
        print(f"MCP server: {url}")
        print(f"tools: {len(tools)}")
        valid = 0
        for t in tools:
            schema = getattr(t, "input_schema", None) or t.inputSchema
            try:
                Draft202012Validator.check_schema(schema)
                ok = "valid JSON Schema"
                valid += 1
            except Exception as exc:                          # noqa: BLE001
                ok = f"INVALID schema: {exc}"
            params = ", ".join((schema or {}).get("properties", {}))
            print(f"  - {t.name}({params}) — {ok}")
        print(f"tool schemas valid: {valid}/{len(tools)}")

        if len(sys.argv) > 2:
            name, args = sys.argv[2], json.loads(sys.argv[3] if len(sys.argv) > 3 else "{}")
            print(f"\ncall {name}({json.dumps(args, ensure_ascii=False)}):")
            res = await client.call_tool(name, args)
            text = res.content[0].text if res.content else ""
            print(text[:1200])


asyncio.run(main())
