#!/usr/bin/env bash
# Build a serving MCP container from an ALREADY-BUILT alembic output and run it.
# Fully algorithmic: no LLM, no `docker commit`. Prints
#   SERVE_URL=http://127.0.0.1:<port>
# on success (the web dashboard parses that line).
#
#   docker/alembic/build_serve.sh <repo_url>
#
# Env:
#   MCP_HOST_PORT  host port to publish (default 8000)
set -euo pipefail

REPO_URL="${1:?usage: build_serve.sh <repo_url>}"
NAME="$(basename "${REPO_URL%.git}")"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"     # outer repo root (holds docker/ and .alembic/)
OUT="$ROOT/.alembic/$NAME/output"

[ -d "$OUT" ]            || { echo "no output dir at $OUT — run the pipeline first"; exit 1; }
[ -f "$OUT/server.py" ] || { echo "no server.py — the wrapper stage did not complete"; exit 1; }
[ -f "$OUT/setup.sh" ]  || { echo "no setup.sh — cannot rebuild the environment"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "docker not found on host"; exit 1; }

IMAGE="alembic-serve-$NAME"
CONTAINER="alembic-serve-$NAME"
PORT="${MCP_HOST_PORT:-8000}"

echo "[build_serve] building $IMAGE from $OUT (rebuilding venvs from setup.sh)…"
docker build -f "$ROOT/docker/alembic/serve.Dockerfile" \
    --build-arg REPO_NAME="$NAME" \
    -t "$IMAGE" "$ROOT"

echo "[build_serve] (re)starting container $CONTAINER on :$PORT…"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
# S3 pass-through for served tool file I/O (helpers/s3_transfer.py) — unset =
# no-op. Forwarded via -e so docker only sets what is actually in this shell's
# env; keep this list in sync with start_chain.py's SERVE_ONLY_ENV. The
# S3__* names are the vault contract's own (read first by _env()); the bare
# names on the line above are a DEPRECATED legacy fallback kept only so an
# old shell env keeps working — both are still forwarded here regardless.
docker run -d --name "$CONTAINER" -p "$PORT:8000" -e MCP_PORT=8000 \
    -e ENDPOINT_URL -e ACCESS_KEY -e SECRET_KEY -e BUCKET_NAME \
    -e S3_REGION -e S3_PRESIGN_EXPIRATION -e S3_HTTP_TIMEOUT -e S3_HTTP_MAX_BYTES \
    -e S3__ENDPOINT_URL -e S3__ACCESS_KEY -e S3__SECRET_KEY -e S3__BUCKET_NAME \
    -e S3__EXTERNAL_ENDPOINT_URL -e S3_UPLOAD_MAX_BYTES \
    "$IMAGE" serve "$REPO_URL" >/dev/null

# Wait until the container is up (streamable-http server has no plain "/" route,
# so we treat "still running after a moment" as healthy).
for _ in $(seq 1 30); do
    state="$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)"
    [ "$state" = "true" ] && break
    sleep 1
done
if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)" != "true" ]; then
    echo "[build_serve] container exited — logs:"
    docker logs --tail 40 "$CONTAINER" 2>&1 || true
    exit 1
fi

echo "[build_serve] MCP server container is up."
echo "SERVE_URL=http://127.0.0.1:$PORT"
