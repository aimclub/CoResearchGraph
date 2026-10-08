# moosechem-mcp-server

MCP server exposing [MOOSE-Chem](https://github.com/ZonglinY/MOOSE-Chem) as a
tool for the CoScientist multi-agent system: automated literature corpus
building (PubMed + OpenAlex) and iterative chemical hypothesis generation
via the original MOOSE-Chem evolutionary pipeline.

## Tools

- **`build_corpus`** — generates broad search queries via LLM from a research
  question + background survey (avoiding terms that leak the answer), searches
  **PubMed** (NCBI E-utilities) and **OpenAlex** (free, no API key) in parallel,
  deduplicates by title, saves a `[[title, abstract], ...]` corpus compatible
  with MOOSE-Chem's `custom_inspiration_corpus_path`. Returns a `corpus_job_id`
  immediately — corpus building runs in the background.
- **`check_corpus_status`** — polls a background corpus build started by
  `build_corpus`.
- **`run_moosechem`** — starts the MOOSE-Chem pipeline (inspiration screening →
  hypothesis generation → evaluation) as a background job. Returns a `job_id`
  immediately since a full run takes 25-60+ minutes depending on parameters
  (see "Known gotchas" below).
- **`check_moosechem_status`** — polls a background run started by `run_moosechem`.
- **`get_hypotheses`** — returns top-N hypotheses by MOOSE-Chem score from a
  finished run, enriched with LLM-extracted `tools` and `variables`
  (independent/dependent/covariates), plus the inspiration paper's title and
  abstract (looked up in the corpus).
- **`get_inspirations`** — shows the screening funnel: which papers from the
  corpus MOOSE-Chem selected as final inspiration sources.

## Environment variables

See `.env.example`. Required: `OPENROUTER_API_KEY`.

## Docker (recommended — via docker-compose)

From `mcp-servers/`:

```bash
cp moosechem-mcp-server/.env.example moosechem-mcp-server/.env
# edit .env and set your OPENROUTER_API_KEY
docker-compose up moosechem-mcp-server
```

The server is reachable on the host at `http://localhost:7335/mcp` (mapped
from container port `7331`; see `docker-compose.yml`).

## Docker (manual)

Build from the **repository root** (matches the convention of other
mcp-servers in this repo):

```bash
docker build -f mcp-servers/moosechem-mcp-server/Dockerfile -t moosechem-mcp-server .
docker run -p 7335:7331 --env-file mcp-servers/moosechem-mcp-server/.env moosechem-mcp-server
```

## Local development (without Docker)

```bash
uv sync
uv run python moosechem_server.py
```

Server listens on `http://0.0.0.0:7331/mcp`.

## Integration with CoScientist

The `MooseChemMCPTool` client (in
`CoScientist/hypothesis_subsystem/moosechem_mcp_tool.py`) connects to this
server and implements the `BaseHypothesisTool` contract. Its
`strategy_type` **must** be exactly `"MooseChem"` — the generator agent looks
up the tool in the registry by this exact key
(`registry.get("MooseChem")`); a mismatched key silently falls back to the
native `MooseChemTool` instead of this MCP-backed one.

## Known gotchas

- **Caching by research question.** MOOSE-Chem writes results into a
  directory slugged from the research question. Re-running the same question
  reuses the existing result instead of recomputing — including its
  inspiration list, which may go stale relative to a freshly rebuilt corpus
  (abstract lookups can silently miss). For a clean end-to-end run, remove the
  job's result directory and restart the container (clears `/app/jobs` and
  `MOOSECHEM_PATH/<slug>_mcp/`) before calling `build_corpus` again.
- **`get_hypotheses` timeout.** For each returned hypothesis, this tool makes
  two additional LLM calls server-side (tools + variables extraction). At 5
  hypotheses with untruncated abstracts this can take 60-90+ seconds — make
  sure the MCP client's HTTP timeout for this specific call is generous
  (180s+), not the same short timeout used for the fast `build_corpus` /
  `run_moosechem` trigger calls.
- **MOOSE-Chem run time is parameter-sensitive.** `main.sh` is patched at
  runtime (via `_run_moosechem_job`) with `--num_self_explore_steps_each_line`
  and `--num_itr_self_refine`. Original MOOSE-Chem defaults (3/3) give the
  intended EA depth but take up to an hour; lower values speed things up at
  the cost of exploration depth. `--num_screening_window_size` should stay at
  15 — reducing it breaks the inspiration screening funnel.
- **Job status files** are written to `MOOSECHEM_JOBS_DIR` (default
  `/app/jobs`). Mount this as a volume if you need job history to survive
  container restarts.
- **Result JSON files** (corpus, hypotheses) live inside the MOOSE-Chem
  directory tree on the container's local disk. If results need to persist
  beyond the container lifecycle, mount `MOOSECHEM_PATH/Data` and the
  checkpoint output directory as volumes (see `docker-compose.yml` at the
  `mcp-servers/` root for the pattern used by sibling servers).
