"""Health check for the rag_tools registry infra (Postgres + Qdrant).

Verifies the SAME datastore clients ``Retrieve_tools`` uses
(``CoScientist/tools/retrieval_tools.py``) can connect and query an EMPTY
registry:

  * ``PostgresClient.initialize()`` — opens the pool AND creates the schema
    (SQLAlchemy ``create_all``) — then ``list_servers()`` returns ``[]``.
  * ``QdrantClientWrapper.connect()`` + ``health_check()``.

Connection settings come from env (``POSTGRES__*`` / ``QDRANT__*`` — see
``rag_tools.config.settings``); a bare ``.env`` uses rag_tools' localhost
defaults, which match the compose bring-up. The embedding / reranker API
servers are deliberately NOT exercised — they are separate infra, and an empty
registry connects and queries without them.

Usage:
    python scripts/rag_tools/health_check.py [--retries N] [--delay S] [--json]

Exit code 0 = healthy; non-zero prints a readable reason (expanding
TaskGroup/cause chains, like the sibling cli.py).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import asdict, dataclass


def _exc_detail(e: BaseException) -> str:
    """Readable detail of an exception, expanding ExceptionGroup/TaskGroup and
    cause chains (so 'unhandled errors in a TaskGroup' becomes the real
    connection-refused / timeout / auth error underneath)."""
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


@dataclass
class RegistryHealth:
    """Outcome of one probe of the registry datastores."""

    postgres_ok: bool
    qdrant_ok: bool
    server_count: int
    qdrant_collections: int

    def ok(self) -> bool:
        return self.postgres_ok and self.qdrant_ok

    def summary(self) -> str:
        return (
            f"postgres: {'ok' if self.postgres_ok else 'FAIL'} "
            f"({self.server_count} servers registered) | "
            f"qdrant: {'ok' if self.qdrant_ok else 'FAIL'} "
            f"({self.qdrant_collections} collections)"
        )


async def probe_registry(postgres, qdrant) -> RegistryHealth:
    """Connect to both datastores and query the (empty) registry.

    Takes already-constructed clients so it can be unit-tested with fakes; the
    live wiring is in :func:`run_health_check`.
    """
    await postgres.initialize()
    try:
        servers = await postgres.list_servers()
    finally:
        await postgres.close()

    await qdrant.connect()
    try:
        health = await qdrant.health_check()
    finally:
        await qdrant.close()

    return RegistryHealth(
        postgres_ok=True,
        qdrant_ok=(health.get("status") == "ok"),
        server_count=len(servers),
        qdrant_collections=int(health.get("collections", 0)),
    )


def _build_clients(settings):
    """The exact clients Retrieve_tools connects through, from env settings."""
    from rag_tools.storage import PostgresClient
    from rag_tools.storage.qdrant_client import QdrantClientWrapper

    return PostgresClient(settings.postgres), QdrantClientWrapper(settings.qdrant)


async def run_health_check(
    settings, *, retries: int = 1, delay: float = 2.0, timeout: float = 10.0
) -> RegistryHealth:
    """Probe the registry, retrying transient connection failures (containers
    that are up but not yet accepting connections).

    Each attempt is bounded by ``timeout`` seconds: the rag_tools clients open
    no connect timeout of their own, so a wrong/unreachable host (e.g. a
    ``localhost`` that resolves to an IPv6 dead-end) would otherwise hang
    indefinitely. A timed-out attempt is treated as transient and retried."""
    last_exc: BaseException | None = None
    for attempt in range(1, retries + 1):
        postgres, qdrant = _build_clients(settings)
        try:
            return await asyncio.wait_for(
                probe_registry(postgres, qdrant), timeout=timeout
            )
        except Exception as exc:  # noqa: BLE001 — surface the readable reason
            last_exc = exc
            if attempt < retries:
                await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="rag_tools registry health check")
    p.add_argument(
        "--retries",
        type=int,
        default=1,
        help="connection attempts before giving up (default: 1)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="seconds between attempts (default: 2.0)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="per-attempt connect/query timeout in seconds (default: 10.0)",
    )
    p.add_argument("--json", action="store_true", help="print result as JSON")
    return p.parse_args(argv)


def main(argv=None) -> int:
    from dotenv import load_dotenv
    from rag_tools.config.settings import get_settings

    load_dotenv()
    args = _parse_args(argv)
    settings = get_settings()

    pg = settings.postgres
    print(
        f"[rag-health] postgres={pg.host}:{pg.port}/{pg.database} "
        f"qdrant={settings.qdrant.url}"
    )
    try:
        health = asyncio.run(
            run_health_check(
                settings,
                retries=args.retries,
                delay=args.delay,
                timeout=args.timeout,
            )
        )
    except BaseException as exc:  # noqa: BLE001
        print("[rag-health] UNHEALTHY — could not connect/query the registry:")
        print(_exc_detail(exc))
        return 1

    if args.json:
        import json

        print(json.dumps(asdict(health)))
    else:
        print(f"[rag-health] {health.summary()}")

    if not health.ok():
        print("[rag-health] UNHEALTHY")
        return 1
    print("[rag-health] OK — registry reachable and queryable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
