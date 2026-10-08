# S3 MCP Vault Server

An MCP (Model Context Protocol) server for S3-backed file management. It gives AI
agents a persistent data vault (MinIO) with scoped keys, retention tiers, and one
return contract.

The server came from the standalone `s3mcp` project. It lives here now because
every CoScientist pipeline hands files over through it. The full design document
stays in that repository.

## Key layout and retention

Every object lives under one pattern:

```
<retention>/<user_id>/<session_id>/<feature>/<filename>
```

- `ephemeral/` — scratch files. A lifecycle rule deletes them after
  `EPHEMERAL_TTL_DAYS` (default 7).
- `permanent/` — deliverables. Public-read. No expiry.

The server builds every key. Callers never compose keys.

## Link TTL policy

- Ephemeral objects return presigned URLs that expire with the object. The link TTL
  equals the remaining lifetime of the object.
- Permanent objects return plain unsigned URLs. They never expire.
- The policy is fixed. No env var configures it.

## Tool surfaces

The env var `VAULT_SURFACE` selects the registered tools: `worker`, `framework`, or
`all` (default).

CoScientist runs one instance with `all`. The worker toolset names the two tools
it wants, so an agent sees the worker surface only, and framework code reaches
`promote_artifact` on the same port. Two instances also work, for a deployment
that must keep the split on the server.

- Worker surface: `get_upload_link`, `get_download_link`.
- Framework surface: `promote_artifact`, `cleanup_session`, `update_artifact_metadata`,
  `get_session_manifest`, `list_artifacts`.

Every tool returns JSON with `bucket`, `s3_key`, and a URL field. The pair
`bucket` + `s3_key` is the durable reference. Store it. Mint URLs on demand.

## Installation and setup

1. Copy `.env.example` to `.env` and fill in the values. Against an external S3
   or MinIO, set `S3__ENDPOINT_URL` and `S3__EXTERNAL_ENDPOINT_URL` to the same
   routable address.

2. Copy `mcp-servers/.env.example` to `mcp-servers/.env`. Docker Compose reads
   that file for the build proxy. Without it, the image build runs `pip` with no
   proxy, which fails on a host with restricted egress.

3. Start the server from `mcp-servers/`:
   ```bash
   docker compose up -d --build vault-mcp-server
   ```
   It listens on host port 7338. Inside the network it listens on 7331, the same
   port as every sibling MCP server.

4. To get a local MinIO with it, add the `local-s3` profile:
   ```bash
   docker compose --profile local-s3 up -d --build
   ```
   The profile starts MinIO (ports 9000 and 9001) and `minio-setup`. The setup
   container creates the bucket and the access user, adds the lifecycle rule on
   `ephemeral/`, and sets anonymous download on `permanent/`.

   The profile is off by default. The other servers point at an S3 endpoint the
   compose file does not define, so a MinIO that starts by itself would change
   where they write.

5. Against an external MinIO, run the last two setup commands once by hand:
   ```bash
   mc ilm rule add --expire-days 7 --prefix ephemeral/ <alias>/<bucket>
   mc anonymous set download <alias>/<bucket>/permanent
   ```
   Skip them and nothing under `ephemeral/` expires, and the plain `permanent/`
   URLs do not resolve.

6. Point an agent at the HTTP endpoint: `http://localhost:7338/mcp`

## Health and verification

Run the demonstration client. It exercises the full lifecycle: upload, download,
metadata, promotion, manifest, and cleanup.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 mcp_client.py
```

Set `VAULT_MCP_URL` to reach a server on another host or port.

**Success criteria:**
- `[+] Upload complete.`
- `[+] Permanent download OK: ...`
- `[+] Manifest holds ... artifacts.`

### Manual component checks

1. **MinIO API**: `curl http://localhost:9000/minio/health/live`
2. **MinIO console**: Visit `http://localhost:9001` (admin/password123).
3. **MCP server**: `curl -v http://localhost:7338/mcp`

## Configuration

- `S3__ENDPOINT_URL`: Where the server reaches S3. For a MinIO in this Compose
  project, the service name (`http://minio:9000`).
- `S3__EXTERNAL_ENDPOINT_URL`: How the caller sees S3 (`http://localhost:9000`).
  Against an external S3 or MinIO, both keys hold the same routable address.
- `S3__ACCESS_KEY`, `S3__SECRET_KEY`, `S3__BUCKET_NAME`: Credentials and bucket.
- `EPHEMERAL_TTL_DAYS`: Lifetime of ephemeral objects. Default 7. The lifecycle
  rule in `docker-compose.yml` reads the same name from the compose `.env` file.
  Set it in both places, or a link outlives its object.
- `VAULT_SURFACE`: `worker`, `framework`, or `all`. Default `all`. See
  "Tool surfaces" above.
- `MCP_PORT`: Port inside the container. Default 8000. The compose file sets
  7331 to match the sibling servers.
- `MINIO_*` names work as fallback aliases for one release.
