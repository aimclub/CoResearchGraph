# dataset-collection-mcp-server

## Environment

Create a `.env` file in this directory based on `.env.example`.

## Server Tools

The server exposes one MCP tool:

### `extract_mols_prop_dataset`

Builds a molecular dataset from one or more PDF papers.

What it does:

- Converts PDF pages to images
- Detects molecules (SMILES + identifiers) from figures via chemical service
- Extracts requested molecular properties with an LLM
- Merges molecules and properties into a tab-separated dataset
- Uploads final dataset to S3
- Returns metadata with dataset S3 URI and S3 paths to annotated images

Main arguments:

- `model_url`: LLM endpoint URL
- `question`: extraction instruction (for example, collect MIC values)
- `pdfs`: list of local PDF paths
- `session_id`: session identifier used in result paths
- `user_id`: user identifier used in S3 prefix

Dataset is uploaded under the S3 key pattern:

- `user_id/session_id/dataset_collection/final_dataset_<run_id>.csv`

## Run With uv

From `mcp-servers/dataset-collection-mcp-server`:

```bash
set -a
source .env
set +a
uv sync --frozen --no-install-project
uv run --no-project python server/dataset_collection_server.py
```

## Run With Docker

Build from `mcp-servers/dataset-collection-mcp-server`, but use the repository root as the Docker build context:

```bash
docker build -f Dockerfile -t dataset-collection-mcp-server ../..
```

Run the container with the environment file:

```bash
docker run --rm -i -p 7331:7331 --env-file .env dataset-collection-mcp-server
```
