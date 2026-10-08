# papers-search-mcp-server

## Environment

Create a `.env` file in this directory based on `.env.example`.

## Server Tools

This server exposes two MCP tools:

### 1. `explore_chemistry_database`

- Purpose: Answer general chemistry questions using the indexed chemistry papers database (Chroma).
- Input:
	- `task` (string): user question.
- Output:
	- JSON with `answer` and supporting context/metadata.
- Use when:
	- you need database-backed chemistry answers not tied to a specific user-uploaded paper set.

### 2. `explore_my_papers`

- Purpose: Answer questions about user-provided PDF papers, including figures/reactions/molecules and paper metadata.
- Input:
	- `task` (string): user question about uploaded papers.
	- `config` (RunnableConfig): used for MCP session context (includes `session_id`).
- Output:
	- JSON with `answer` and metadata.
	- Returns `{"answer": "No papers provided for search."}` if no PDFs are found.
- Use when:
	- the question is about the current uploaded paper set, specific document details, or cross-paper comparison.

## Run With uv

From `mcp-servers/paper-analysis-mcp-server`:

```bash
set -a
source .env
set +a
uv sync --frozen --no-install-project
uv run --no-project python papers_search_server.py
```

## Run With Docker

Build from `mcp-servers/paper-analysis-mcp-server`, but use the repository root as the Docker build context:

```bash
docker build -f Dockerfile -t paper-analysis-mcp-server ../..
```

Run the container with the environment file and port mapping:

```bash
docker run --rm -i -p 7331:7331 --env-file .env paper-analysis-mcp-server
```

The server will be available at `http://localhost:7331/mcp`.