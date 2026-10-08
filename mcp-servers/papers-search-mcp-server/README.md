# papers-search-mcp-server

## Environment

Create a `.env` file in this directory based on `.env.example`.

## Server Tools

The server exposes three MCP tools:

### `search_entity`

Finds an OpenAlex entity ID by name.

Main arguments:

- `entity_type`: one of `author`, `source`, `institution`
- `entity_name`: entity name to search for

Use this tool when a paper query depends on an author, journal, or institution filter.

### `search_papers`

Searches OpenAlex works and returns paper metadata.

Main arguments:

- `keywords`: free-text topic or query string
- `author_id`: OpenAlex author ID
- `institution_id`: OpenAlex institution ID
- `source_id`: OpenAlex source ID
- `publication_year`: year or year filter such as `2025` or `>2020`
- `open_access`: whether to keep only open-access papers
- `has_pdf`: whether to keep only papers with an available PDF
- `limit`: maximum number of results
- `sort`: OpenAlex sort field such as `publication_year:desc` or `cited_by_count:desc`

This tool returns a human-readable summary and normalized metadata for the matching papers.

### `download_papers_from_search`

Searches OpenAlex works and uploads matching PDFs directly to S3.

Main arguments:

- All search arguments from `search_papers`
- `session_id`: session identifier used in the S3 prefix
- `user_id`: user identifier used in the S3 prefix

Uploaded files are stored under the S3 prefix `user_id/session_id/web_search_res/`.

## Run With uv

From `mcp-servers/papers-search-mcp-server`:

```bash
set -a
source .env
set +a
uv sync --frozen --no-install-project
uv run --no-project python papers_search_server.py
```

## Run With Docker

Build from `mcp-servers/papers-search-mcp-server`, but use the repository root as the Docker build context:

```bash
docker build -f Dockerfile -t papers-search-mcp-server ../..
```

Run the container with the environment file and port mapping:

```bash
docker run --rm -i -p 7331:7331 --env-file .env papers-search-mcp-server
```

The server will be available at `http://localhost:7331/mcp`.