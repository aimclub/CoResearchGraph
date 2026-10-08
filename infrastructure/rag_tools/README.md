# rag_tools registry infra — Postgres + Qdrant

The persistent tool catalogue behind `Retrieve_tools`
(`CoScientist/tools/retrieval_tools.py`). **Postgres** holds the authoritative
MCP server/tool metadata; **Qdrant** holds the embedding index used for RAG
retrieval. Pre-converted Alembic tools live here so `Retrieve_tools` finds them
across runs and machines.

The embedding / reranker API servers used by the
*full* RAG query path are **separate** infra (`infrastructure/chroma/`) and are
not required here — a fresh, empty registry connects and is queryable without
them.

## One-command bring-up

```bash
bash scripts/rag_tools/rag_infra.sh up
```

Starts both services (detached) and blocks until the health check passes. Then:

```bash
bash scripts/rag_tools/rag_infra.sh health     # re-run the health check
bash scripts/rag_tools/rag_infra.sh down       # stop, KEEP the data volumes
bash scripts/rag_tools/rag_infra.sh destroy    # stop and DELETE the data volumes
```

Equivalent raw compose (the wrapper just adds `--env-file .env` + the health
wait):

```bash
docker compose -f infrastructure/rag_tools/docker-compose.yml up -d
docker compose -f infrastructure/rag_tools/docker-compose.yml down     # + -v to wipe data
```

## Health check

```bash
python scripts/rag_tools/health_check.py                       # exit 0 = healthy
python scripts/rag_tools/health_check.py --json                # machine-readable
python scripts/rag_tools/health_check.py --retries 30 --delay 2 # wait for start-up
```

Each attempt is bounded by `--timeout` seconds (default 10) so a wrong or
unreachable host fails fast and is retried instead of hanging (the rag_tools
clients open no connect timeout of their own).

It connects through the **same** `rag_tools` clients `Retrieve_tools` uses —
`PostgresClient(settings.postgres)` and `QdrantClientWrapper(settings.qdrant)`
— then `PostgresClient.initialize()` (which also creates the schema) +
`list_servers()` and `QdrantClientWrapper.health_check()`. On a fresh bring-up
`list_servers()` returns `[]` (an empty registry queried successfully).

## Connection settings (env)

Read from the process env / `.env` by `rag_tools.config.settings` (nested
delimiter `__`). Defaults match rag_tools' own defaults, so a bare `.env` works
against this local bring-up:

| env | default | used by |
|---|---|---|
| `POSTGRES__HOST` | `localhost` | app/health (not compose) |
| `POSTGRES__PORT` | `5432` | app/health **and** the host port compose publishes |
| `POSTGRES__USER` | `rag_tools` | app/health + Postgres container |
| `POSTGRES__PASSWORD` | `rag_tools_password` | app/health + Postgres container |
| `POSTGRES__DATABASE` | `rag_tools` | app/health + Postgres container |
| `QDRANT__URL` | `http://localhost:6333` | app/health |
| `QDRANT__API_KEY` | *(none)* | app/health |
| `RAG_QDRANT_HTTP_PORT` | `6333` | host port compose publishes (match `QDRANT__URL`) |
| `RAG_POSTGRES_IMAGE` / `RAG_QDRANT_IMAGE` | pinned in compose | image override |

> The health check honours the **same** env, so point it at *local* infra:
> keep `POSTGRES__HOST`/`QDRANT__URL` at their localhost defaults (a `.env` that
> aims them at a remote cluster will health-check the remote, not these
> containers).
>
> **`localhost` vs `127.0.0.1`:** if the host also runs its own Postgres, or
> `localhost` resolves to IPv6 (`::1`) first, a `localhost` connection can hit
> the wrong listener and time out. Set `POSTGRES__HOST=127.0.0.1` to force IPv4
> at the published container port.

## Data & teardown

State persists in the named volumes `rag_tools-registry_rag_postgres_data` and
`…_rag_qdrant_data`. `down` keeps them (restarting resumes the same registry);
`destroy` (`down -v`) deletes them for a clean slate.
