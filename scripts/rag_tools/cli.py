import argparse
import asyncio
import json
import os
from dotenv import load_dotenv

from rag_tools import create_manager
from rag_tools.config.settings import get_settings
from rag_tools.retrieval import APIEmbedder, APIReranker, BM25Reranker, HybridReranker


# -----------------------
# INIT
# -----------------------
load_dotenv()


def _exc_detail(e: BaseException) -> str:
    """Readable detail of an exception, EXPANDING ExceptionGroup/TaskGroup and
    cause chains — so 'unhandled errors in a TaskGroup (1 sub-exception)' turns
    into the actual underlying error (connection refused / timeout / 401 / …)."""
    parts: list[str] = []

    def walk(exc: BaseException, depth: int) -> None:
        pad = "    " * depth
        parts.append(f"{pad}{type(exc).__name__}: {exc}")
        subs = getattr(exc, "exceptions", None)  # ExceptionGroup
        if subs:
            for s in subs:
                walk(s, depth + 1)
            return
        cause = exc.__cause__ or exc.__context__
        if cause is not None:
            parts.append(f"{pad}  ↳ caused by:")
            walk(cause, depth + 1)

    walk(e, 0)
    return "\n".join(parts)


async def init_manager():
    settings = get_settings()

    embedder = APIEmbedder(settings.api_embedding)
    api_reranker = APIReranker(settings.api_reranker)
    bm25_reranker = BM25Reranker(settings.bm_reranker)
    reranker = HybridReranker([api_reranker, bm25_reranker], settings.hybrid_reranker)

    manager = await create_manager(settings, embedder, reranker)
    return manager


# -----------------------
# COMMANDS
# -----------------------

async def cmd_add(args):
    manager = await init_manager()

    server = await manager.add_server(
        protocol="http",
        url=args.url,
        name=args.name,
        description=args.description or "",
        headers={"Authorization": f"Bearer {args.token}"} if args.token else None,
        sync_tools=True
    )

    print(f"✅ Added server: {server.name}")
    print(f"   ID: {server.server_id}")

    await manager.close()


async def cmd_remove(args):
    manager = await init_manager()

    await manager.remove_server(args.server_id)

    print(f"🗑 Removed server: {args.server_id}")

    await manager.close()


async def cmd_sync(args):
    manager = await init_manager()

    result = await manager.sync_server(args.server_id)

    print(f"🔄 Synced server: {args.server_id}")
    print(result)

    await manager.close()


async def cmd_sync_all(args):
    manager = await init_manager()

    results = await manager.sync_all_servers()

    ok = [r for r in results if r.success]
    bad = [r for r in results if not r.success]
    print(f"🔄 Sync results: {len(ok)} ok, {len(bad)} failed\n")
    for r in results:
        mark = "✅" if r.success else "❌"
        line = (f"{mark} {r.server_name} ({r.server_id}) "
                f"+{r.tools_added} ~{r.tools_updated} -{r.tools_removed}")
        if r.errors:
            line += f"   errors: {r.errors}"
        print(line)

    # The manager only kept str(ExceptionGroup) ('unhandled errors in a
    # TaskGroup …'), which hides the real cause. Re-run the sync STEP BY STEP
    # (connect → list_tools → postgres add → embed+qdrant index) and print the
    # EXPANDED exception, so we see WHICH step fails and why.
    if bad:
        from rag_tools.tools.sync_mcp import MCPSyncer
        from rag_tools.ingestion.parser import mcp_tool_info_to_model
        from mcp import ClientSession
        pg = getattr(manager, "postgres", None) or getattr(manager, "_postgres", None)
        indexer = getattr(manager, "_indexer", None) or getattr(manager, "indexer", None)

        async def _step(label, coro):
            try:
                res = await coro
                print(f"      ✓ {label}: OK")
                return res, None
            except BaseException as e:  # noqa: BLE001 — diagnostic
                print(f"      ✖ {label} FAILED:")
                print("        " + _exc_detail(e).replace("\n", "\n        "))
                return None, e

        print("\n🔎 Diagnosing failures (per step, ExceptionGroup expanded):")
        for r in bad:
            server = await pg.get_server(r.server_id)
            if not server:
                print(f"  ✖ {r.server_id}: not found in DB")
                continue
            print(f"  • {server.name}  url={server.url}")
            existing, err = await _step("postgres.get_tools_by_server", pg.get_tools_by_server(server.server_id))
            if err:
                continue
            existing_names = {t.name for t in (existing or [])}
            try:
                client = MCPSyncer.get_transport(server)
                async with client as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await _step("session.initialize", session.initialize())
                        resp, err = await _step("session.list_tools", session.list_tools())
                        if err:
                            continue
                        new_tools = [mcp_tool_info_to_model(t, server.server_id)
                                     for t in resp.tools if t.name not in existing_names]
                        print(f"      new tools to index: {len(new_tools)}")
                        if new_tools:
                            _, err = await _step("postgres.bulk_add_tools", pg.bulk_add_tools(new_tools))
                            if not err and indexer is not None:
                                await _step("indexer.index_tools_batch (embed+qdrant)",
                                            indexer.index_tools_batch(new_tools))
            except BaseException as e:  # noqa: BLE001 — connection/teardown level
                print("      ✖ connection/teardown:")
                print("        " + _exc_detail(e).replace("\n", "\n        "))

    await manager.close()


async def cmd_load(args):
    manager = await init_manager()

    with open(args.file, "r") as f:
        servers = json.load(f)

    for s in servers:
        try:
            server = await manager.add_server(
                protocol="http",
                url=s["url"],
                name=s["name"],
                description=s.get("description", ""),
                headers={"Authorization": f"Bearer {s['token']}"} if s.get("token") else None,
                sync_tools=True
            )

            print(f"✅ Added: {server.name}")

        except Exception as e:
            print(f"❌ Failed: {s.get('name')} -> {e}")

    await manager.close()


async def cmd_list(args):
    manager = await init_manager()

    servers = await manager.postgres.list_servers()

    print("📦 Registered servers:\n")
    for s in servers:
        print(f"- {s.name}")
        print(f"  ID: {s.server_id}")
        print(f"  URL: {s.url}")
        print(f"  Description: {s.description}")
        print(f"  Status: {s.status}")
        print()

    await manager.close()


async def cmd_export_csv(args):
    import csv

    def _v(x):
        return getattr(x, "value", x) if x is not None else ""

    manager = await init_manager()
    pg = getattr(manager, "postgres", None) or getattr(manager, "_postgres", None)

    servers = await pg.list_servers()
    out = args.out or "mcp_servers_tools.csv"
    n_rows = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "server_id", "server_name", "server_url", "server_protocol",
            "server_status", "server_description",
            "tool_name", "tool_description", "tool_tags", "tool_input_schema",
        ])
        for s in servers:
            base = [s.server_id, s.name, s.url, _v(getattr(s, "protocol", "")),
                    _v(getattr(s, "status", "")), s.description or ""]
            tools = await pg.get_tools_by_server(s.server_id)
            if not tools:
                w.writerow(base + ["", "", "", ""])
                n_rows += 1
            for t in tools:
                w.writerow(base + [
                    t.name,
                    (t.description or ""),
                    ",".join(getattr(t, "tags", None) or []),
                    json.dumps(getattr(t, "input_schema", None) or {}, ensure_ascii=False),
                ])
                n_rows += 1

    print(f"📄 Exported {len(servers)} servers, {n_rows} rows → {out}")
    await manager.close()


# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG Tools MCP Server CLI"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # add
    p_add = subparsers.add_parser("add", help="Add a server")
    p_add.add_argument("--url", required=True)
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--description", default="")
    p_add.add_argument("--token", default=None)
    p_add.set_defaults(func=cmd_add)

    # remove
    p_remove = subparsers.add_parser("remove", help="Remove a server")
    p_remove.add_argument("server_id")
    p_remove.set_defaults(func=cmd_remove)

    # sync
    p_sync = subparsers.add_parser("sync", help="Sync a server")
    p_sync.add_argument("server_id")
    p_sync.set_defaults(func=cmd_sync)

    # sync-all
    p_sync_all = subparsers.add_parser("sync-all", help="Sync all servers")
    p_sync_all.set_defaults(func=cmd_sync_all)

    # load
    p_load = subparsers.add_parser("load", help="Load servers from JSON")
    p_load.add_argument("file", help="Path to JSON file")
    p_load.set_defaults(func=cmd_load)

    # list
    p_list = subparsers.add_parser("list", help="List all servers")
    p_list.set_defaults(func=cmd_list)

    # export-csv
    p_csv = subparsers.add_parser("export-csv", help="Export all servers + tools (with descriptions) to CSV")
    p_csv.add_argument("--out", default="mcp_servers_tools.csv", help="output CSV path")
    p_csv.set_defaults(func=cmd_export_csv)

    args = parser.parse_args()

    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()