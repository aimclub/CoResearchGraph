# chemical-mcp-server

Standalone MCP (Model Context Protocol) server for chemistry tools: SMILES, molecular properties, docking, molecule/reaction OCR, visualization, retrosynthesis, and forward synthesis prediction.

Runs independently with **uv**, **Docker** or **docker-compose**.

## Requirements

- Python 3.11+
- Optional: [uv](https://docs.astral.sh/uv/) for install/run
- Optional: Docker & docker-compose for containerized run

## Setup (local with uv)

```bash
cd mcp-servers/chemical-mcp-server
cp .env.example .env
# Edit .env if needed (CHEM_SERVICES_HOST, CHEM_SERVICES_PORT)

uv sync
```

## Run (local)

```bash
# With uv
uv run python -m server.chemical_server

# Or after uv sync
chemical-mcp-server
```

Server listens on `http://0.0.0.0:7331/mcp`.

## Run with Docker

```bash
cp .env.example .env
docker compose up --build
```

## Run with Docker (one-off)

```bash
docker build -t chemical-mcp-server .
docker run -p 7331:7331 --env-file .env chemical-mcp-server
```

## Environment (.env)

| Variable | Description | Default |
|----------|-------------|--------|
| `CHEM_SERVICES_HOST` | Host of the chemistry API (OpenChemIE/docking) | `localhost` |
| `CHEM_SERVICES_PORT` | Port of the chemistry API | `8005` |
| `CHEM_SERVICES_TIMEOUT` | Request timeout for chemistry API (seconds) | `60` |
| `RETROSYNTHESIS_SERVICES_HOST` | Host of the retrosynthesis/ASKCOS API | `localhost` |
| `RETROSYNTHESIS_SERVICES_PORT` | Port of the retrosynthesis/ASKCOS API | `8001` |
| `RETROSYNTHESIS_REQUEST_TIMEOUT` | Request timeout for retrosynthesis API (seconds) | `60` |
| `S3__ENDPOINT_URL` | S3-compatible storage endpoint URL | — |
| `S3__BUCKET_NAME` | S3 bucket for storing images and visualizations | — |
| `S3__ACCESS_KEY` | S3 access key | — |
| `S3__SECRET_KEY` | S3 secret key | — |
| `CHEM_MCP_HOST` | MCP server bind address | `0.0.0.0` |
| `CHEM_MCP_PORT` | MCP server port | `7331` |
| `CHEM_MCP_PATH` | MCP server HTTP path | `/mcp` |

Copy `.env.example` to `.env` and adjust as needed.

## Tools exposed via MCP

### Molecule utilities
- `name2smiles` — convert a molecule name to SMILES via PubChem
- `smiles2name` — convert SMILES to IUPAC name via PubChem
- `smiles2prop` — calculate RDKit molecular descriptors from SMILES
- `visualize_molecule` — render interactive 3-D HTML viewer (uploaded to S3, presigned URL returned)

### Activity data
- `fetch_activity_data` — fetch protein–ligand activity data from BindingDB or ChEMBL; saves to CSV

### OCR / image analysis
- `extract_molecules` — detect molecular structures in images (URLs); returns SMILES + annotated image
- `extract_reactions` — detect chemical reactions in images (URLs); returns reaction dicts + annotated image

### Docking
- `calculate_docking` — compute docking score for a SMILES against a PDB receptor; visualization uploaded to S3

### Retrosynthesis & synthesis prediction
- `retrosynthesis_tree_search` — plan retrosynthetic routes for a target SMILES using ASKCOS tree search.
  Accepts `mode` (`fast` / `balanced` / `deep`). Returns ranked routes with steps, reactants, scores,
  and per-route reaction-strip images (uploaded to S3, presigned URLs in `metadata.route_images`).

- `classify_reaction` — classify reaction SMILES (`A.B>>C`) into named reaction classes using ASKCOS.
  Returns ranked hits with class/superclass names and confidence scores.

- `forward_predict` — predict reaction products from reactant SMILES using ASKCOS forward models
  (`wldn5`, `graph2smiles`, `augmented_transformer`). Returns a ranked list of predicted products with
  scores, plus two images uploaded to S3:
  - `metadata.predictions_image` — grid of all predicted product structures
  - `metadata.top_reactions_image` — reaction drawings for the top 3 predictions (reactants → product)

### S3 storage
Result images (route visualizations, product grids, reaction drawings, docking HTML, molecule viewers)
are stored in S3 under the following prefixes:

| Prefix | Content |
|--------|---------|
| `chemical_mcp/molecule_visualizations/` | 3-D molecule HTML |
| `chemical_mcp/annotated_images/` | OCR-annotated images |
| `chemical_mcp/docking_results/` | Docking HTML viewers |
| `chemical_mcp/retrosynthesis/` | Retrosynthesis route images |
| `chemical_mcp/forward_prediction/` | Forward-prediction product grids and reaction images |

All presigned URLs expire after **1 hour**.
