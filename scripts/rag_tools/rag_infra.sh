#!/usr/bin/env bash
# One-command bring-up / health / teardown for the rag_tools registry infra
# (Postgres + Qdrant). See infrastructure/rag_tools/README.md.
#
#   bash scripts/rag_tools/rag_infra.sh up        # start + wait until healthy
#   bash scripts/rag_tools/rag_infra.sh health    # re-run the health check
#   bash scripts/rag_tools/rag_infra.sh down      # stop (keep data volumes)
#   bash scripts/rag_tools/rag_infra.sh destroy   # stop + delete data volumes
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE="$ROOT/infrastructure/rag_tools/docker-compose.yml"

# Feed the app's .env to compose so credentials/ports interpolate identically
# to what the health check (and Retrieve_tools) read.
ENV_ARGS=()
[ -f "$ROOT/.env" ] && ENV_ARGS+=(--env-file "$ROOT/.env")

PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY="python3"

cmd="${1:-up}"
case "$cmd" in
  up)
    docker compose -f "$COMPOSE" "${ENV_ARGS[@]}" up -d
    echo "[rag-infra] waiting for Postgres + Qdrant to accept connections…"
    "$PY" "$ROOT/scripts/rag_tools/health_check.py" --retries 30 --delay 2
    ;;
  health)
    shift
    "$PY" "$ROOT/scripts/rag_tools/health_check.py" "$@"
    ;;
  down)
    docker compose -f "$COMPOSE" "${ENV_ARGS[@]}" down
    ;;
  destroy)
    docker compose -f "$COMPOSE" "${ENV_ARGS[@]}" down -v
    ;;
  *)
    echo "usage: $(basename "$0") {up|health|down|destroy}" >&2
    exit 2
    ;;
esac
