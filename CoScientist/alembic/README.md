# Alembic

## Overview

Alembic is a multi-agent pipeline that automatically generates a deployable [FastMCP](https://github.com/jlowin/fastmcp) server from any scientific GitHub repository. Given a repo URL, it clones the code, sets up a reproducible Python environment, writes tool functions with pytest tests, validates every tool against the repo's own code, renders a FastMCP server, and packages the result as a Docker image — all without human intervention.

The LLM proposes; **code disposes.** Each stage ends at a deterministic *gate* — a plan check, an env check, an artefact check, a validation loop — that the model cannot talk its way past. A failed gate rolls the stage's files back to a checkpoint and reruns it with a note.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Live Dashboard (Web UI)](#live-dashboard-web-ui)
- [Running a Benchmark](#running-a-benchmark)
- [Project Structure](#project-structure)
- [Configuration](#configuration)

---

## How It Works

The pipeline runs five stages inside a Docker container, each followed by a code-enforced gate:

| Stage | Driver | What it does | Gate |
|---|---|---|---|
| 1 | **Explorer** (agent) | Clones the repo, reads the code, writes a `plan.json` of tools to build | plan is well-formed & non-empty |
| 2 | **Environment** (agent) | Builds the main `.venv` (repo + deps), writes `reports/environment.md` | repo imports cleanly under `.venv` |
| 3 | **Coder** (agent) | Writes one `output/tools/<tool>.py` per tool plus `test_smoke_*` / `test_invoc_*` tests | files exist & compile |
| 4 | **Validator** (code, not an agent) | Runs pytest + a live invocation per tool through the tools venv; calls a **Debugger** agent on failures | every tool green |
| 5 | **Wrapper** (code + fallback agent) | Renders `output/server.py` deterministically and builds the separate `.venv-server` (fastmcp) | server compiles & imports |

**Two venvs, never mixed:** `.venv` holds the repo and its dependencies (where tools and tests run); `.venv-server` holds only fastmcp. Every tool shells through `helpers/run_function.py` in the tools venv, printing a `<<<ALEMBIC_RESULT>>>` sentinel + JSON — so `server.py` is a subprocess router, not an in-process shim, and a dependency clash between the repo and fastmcp is impossible.

**Validation is evidence-based.** Tests split by name: `test_smoke_*` (does it run?) vs `test_invoc_*` (is the output correct?). A tool is `perfect` only if it passes **and** has ≥1 green invocation test. A `test_invoc_*` that mocks or patches the repo instead of calling it is automatically reclassified as a smoke test — so a tool can't reach `perfect` on hollow, self-referential validation.

On success the build container is committed to `alembic-tool:<repo-name>` and launched with the MCP server listening on a random host port.

---

## Getting Started

**Prerequisites:** Docker, Python 3.11+, and a `.env` file at the project root with at least one LLM API key.

```env
OPENROUTER_API_KEY=sk-or-...
MODEL=openrouter/qwen/qwen3-235b-a22b-2507   # optional override
```

**Build and serve a repo in one command:**

```bash
python CoScientist/alembic/start_chain.py https://github.com/Roestlab/massformer
```

The script:
1. Builds `alembic-base:latest` once (skipped on subsequent runs).
2. Runs the pipeline inside a container.
3. Commits the result and starts the MCP server.

```
[start-chain] MCP server up.
  url       : http://localhost:24371/mcp
  logs      : docker logs -f alembic-serve-massformer-a1b2c3
  stop      : docker stop alembic-serve-massformer-a1b2c3 && docker rm alembic-serve-massformer-a1b2c3
```

**Useful flags:**

```bash
# Resume from a specific stage (workdir is preserved)
python CoScientist/alembic/start_chain.py <repo_url> --resume coder

# Build image without starting the server
python CoScientist/alembic/start_chain.py <repo_url> --no-serve

# Force rebuild of the base image
python CoScientist/alembic/start_chain.py <repo_url> --rebuild-base

# GPU access inside the container
python CoScientist/alembic/start_chain.py <repo_url> --gpus all
```

### Guiding what gets built

By default the Explorer decides autonomously which tools to expose. You can steer
it with one of two mechanisms, at opposite ends of a strictness spectrum:

- **Required tasks (hard, enforced).** Pin exact tool(s) — name, argument names,
  return shape — that *must* appear, verified against real repo code by the plan
  gate (the run fails if one can't be grounded). Used for TM-Bench-style
  evaluation. Set `ALEMBIC_TASKS` (env) or pass `--tasks` to a JSON/YAML task
  object, a path, or comma-separated paths (each `{name, description, arguments,
  returns, example}`); several tasks run against one repo. `ALEMBIC_TARGET_TASK`
  is the old single-task spelling.

- **Soft hint (steer, not enforced).** Free-text describing the *idea* of a tool
  you'd like to see mined **among the others**, with no forced name/signature and
  no gate — applied to the Explorer stage only, dropped silently if the repo has
  no real code for it. Set `ALEMBIC_HINTS` (env) or pass `--hints`:

  ```bash
  python CoScientist/alembic/start_chain.py <repo_url> \
    --hints "a train + a predict entry point for the survival model"
  ```

  The two compose: required tools stay pinned while the hint steers the rest.

---

## Live Dashboard (Web UI)

A local, browser-based dashboard that runs the pipeline **without Docker** and
streams every stage and agent action in real time:

- **top** — the five-stage rail (explorer → environment → coder → validator →
  wrapper), lit as each stage runs / passes / fails;
- **left** — an accumulating column: the exploration report, the generated
  output files, how to run them (the recorded `setup.sh`), and per-tool
  invocation examples;
- **right** — the generated tools as cards with live pass/fail validation
  badges (hover a failure for the error) and a **Call** form that invokes each
  tool-function on demand — the same `invoke_tool_function` path the validator
  uses, so you can exercise a tool the instant the Coder writes it (wrapping to
  a real MCP server is only the final stage);
- **bottom** — a live activity feed of every agent tool call.

**Prerequisites:** the same `.env` as [Getting Started](#getting-started), plus
`fastapi` and `uvicorn` (already in the project `requirements.txt`).

**Start it (from the repo root):**

```bash
python CoScientist/alembic/web/server.py
```

Then open **http://127.0.0.1:8100**, paste a repo URL (e.g.
`https://github.com/whitead/synspace`) and press **Run**.

Notes:
- Run it from the project root — the pipeline writes its workdir to `./.alembic/`,
  the same location as the CLI.
- **Stop** invalidates the run; the pipeline unwinds at its next stage/tool
  boundary (a subprocess already in flight finishes first).
- The pipeline itself is unchanged for the CLI/benchmark: it emits events
  through the optional `alembic.events` bus, a no-op unless the dashboard
  installs a sink. All UI enrichment (reading `plan.json` / `validation.json` /
  the output files off disk) lives in `web/app.py`, not in the pipeline.

---

## Running a Benchmark

`benchmarks/alembic/run_benchmark.py` processes multiple repos in parallel
(with a `git ls-remote` reachability pre-check) and writes a Markdown
summary.

```bash
# Explicit list, 4 parallel workers (default)
python benchmarks/alembic/run_benchmark.py \
    --repos https://github.com/Roestlab/massformer \
            https://github.com/whitead/synspace \
            https://github.com/CrystalEye42/OpenChemIE

# From a file (one URL per line, '#' = comment), 8 workers
python benchmarks/alembic/run_benchmark.py \
    --repos-file repos.txt \
    --parallel 8
```

Results default to `benchmarks/alembic/runs/<timestamp>/` (`summary.md`,
`summary.json`, `logs/*.log`) — pass `--output`/`--json-output`/`--log-dir`
to override.

---

## Project Structure

```
CoScientist/alembic/
├── agents.py          # The 5 LLM agents (explorer, environment, coder, debugger, wrapper)
├── main.py            # Pipeline orchestrator + gates + the deterministic validator loop
├── contract.py        # ToolReport / passed / perfect — what counts as a validated tool
├── config.py          # Stages, timeouts, model string (all env-overridable)
├── start_chain.py     # CLI: build base image → run pipeline → commit → serve
├── common.py          # Shared Docker helpers (ensure_base_image, get_repo_name)
├── events.py          # Optional live-event bus (no-op for CLI; feeds the web UI)
├── instructions/      # System prompts for each agent
├── tools/             # Agent tools (fs, shell, venv, invoke, codegen) + scripts/run_function.py
└── web/               # Live dashboard: FastAPI + WebSocket (app.py, server.py, templates/)

docker/alembic/
├── Dockerfile         # Base image (python:3.11 + build deps + alembic code)
├── entrypoint.py      # Container entrypoint (build / serve / help)
├── serve.py           # FastMCP server launcher inside container
└── requirements.txt   # Pipeline dependencies

benchmarks/alembic/
└── run_benchmark.py   # Parallel multi-repo runner (git ls-remote pre-check → summary.md)
```

---

## Configuration

All settings are passed through environment variables (`.env` or shell):

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `openrouter/qwen/qwen3-235b-a22b-2507` | LiteLLM model string for all agents |
| `OPENROUTER_API_KEY` | — | Required when using OpenRouter |
| `OPENAI_API_KEY` | — | Required when using OpenAI models |
| `TAVILY_API_KEY` | — | Optional; enables web search inside the Explorer |
| `MCP_PORT` | `8000` | Port the FastMCP server listens on inside the container |
| `ALEMBIC_WORKDIR` | `/work/.alembic` | In-container working directory for repos and reports |
| `S3__ENDPOINT_URL` | — | S3-compatible endpoint; set together with the three below to enable S3 file pass-through |
| `S3__ACCESS_KEY` | — | S3 access key |
| `S3__SECRET_KEY` | — | S3 secret key |
| `S3__BUCKET_NAME` | — | Default S3 bucket for uploaded output files |
| `S3_REGION` | `us-east-1` | Region passed to boto3; botocore requires one even against a fully custom endpoint |
| `S3_PRESIGN_EXPIRATION` | `3600` (1 hour) | Seconds a presigned URL for an uploaded output file stays valid (clamped to 1–604800) |
| `S3_HTTP_TIMEOUT` | `300` | Seconds before an `http(s)://` input download times out |
| `S3_HTTP_MAX_BYTES` | `1073741824` (1 GiB) | Size cap for an `http(s)://` input download; exceeding it aborts the call |
| `S3_UPLOAD_MAX_BYTES` | `1073741824` (1 GiB) | Size cap for an uploaded output file; a larger file stays local and the call still succeeds |
| `S3__EXTERNAL_ENDPOINT_URL` | `S3__ENDPOINT_URL` | Endpoint presigned links are built for, when the caller reaches S3 at another address than the serve container |

### S3 file pass-through

`S3__ENDPOINT_URL`/`S3__ACCESS_KEY`/`S3__SECRET_KEY`/`S3__BUCKET_NAME` are
all-or-nothing: with all four set, the generated `server.py` (via
`helpers/s3_transfer.py`) handles files at the served MCP boundary instead of
requiring local paths that only exist inside the build container. With any of
the four unset, `server.py` behaves exactly as it did before this existed.
These are the vault contract's own names — the main app's nested-settings
spelling (see `config/settings.py:S3Settings`) — and are read first. Each is
also accepted under its bare legacy spelling (`ENDPOINT_URL`, `ACCESS_KEY`,
`SECRET_KEY`, `BUCKET_NAME`) as a **deprecated** fallback, kept only so an old
`.env` keeps working until those names are retired; a `.env` written today
should use the `S3__*` names. A missing or broken `helpers/s3_transfer.py`
degrades the same way (S3 off) rather than breaking the server's import.

- **Convention.** Any tool parameter or result field named `*_path` or
  `*_file` (case-insensitive) is treated as a file reference.
- **Input.** A `*_path`/`*_file` argument given as `s3://bucket/key` or
  `http(s)://...` (scheme matched case-insensitively) is downloaded to its own
  scratch subdirectory before the tool runs, and the tool sees a local path —
  same as any other input. A plain local path is never intercepted. Two
  different input URIs that happen to share a basename never collide — each
  download gets an isolated subdirectory.
- **Output.** A `*_path`/`*_file` result field that is an existing local file
  outside the mounted `/mount/data` and the per-call scratch dir, and no
  larger than `S3_UPLOAD_MAX_BYTES`, is uploaded and presigned after the tool
  returns. Inside the cloned repo only a file written during the call counts:
  generated tools often resolve a relative out path against the repo, while a
  source or data file the tool merely returns stays out. The original local
  field is kept, and one nested
  `<field>_s3` entry — `{"bucket", "s3_key", "presigned_url", "expires_in"}` —
  is added alongside it. Not flat `<field>_s3_key` / `<field>_presigned_url`
  siblings: the framework's artifact walker
  (`CoScientist/utils/s3_refs.py:_walk`) only recognises a durable reference
  from a dict that carries both `bucket` and `s3_key` together, same as every
  MCP server's own return contract (see
  `mcp-servers/chemical-mcp-server/server/utils/vault.py:contract`). Two
  output fields sharing a basename still get distinct S3 keys — the field
  name is part of the key (see Key layout). A tool that echoes a downloaded
  input path back in its result gets no `_s3` entry for it: that file is the
  caller's own input, and its local copy is deleted right after the call.
- **Error asymmetry.** A failed *input* download raises and fails the whole
  tool call — a tool must never silently run on the wrong or missing data. A
  failed *output* upload only logs
  `[s3] upload failed for <path>: <TypeName>: <message[:200]>` to stderr
  (visible in `docker logs`) and is otherwise silent — an otherwise-successful
  tool call is never turned into a failure just because publishing its result
  to S3 didn't work.
- **Key layout.**
  `ephemeral/<user>/<session>/alembic_<repo>/<tool>/<call-id>/<field>/<file>`
  for an uploaded output file — the vault contract's own layout (see
  `mcp-servers/vault-mcp-server/vault_server.py`): every object lands under
  `ephemeral/` (what the bucket's lifecycle rule filters on to reclaim it),
  and promotion to `permanent/` is the framework's job, not this server's.
  `<user>`/`<session>` come from the `user_id`/`session_id` scope params
  every generated tool declares when both are set — filled in automatically
  at the ADK tool-call boundary by `SessionScopePlugin`
  (`CoScientist/tools/session_scope_plugin.py`) — falling back to the
  `X-Coscientist-User` / `X-Coscientist-Session` request headers (a dormant
  fallback: nothing in this codebase sends those headers today) when the
  params are empty, and to `local`/`default` when neither source has
  anything — a shared namespace, not a per-caller one. A tool whose own
  signature already names a `user_id`/`session_id` parameter of its own
  (domain data, e.g. a database row id) does not get that value used for
  scope either — it still reaches the tool as a normal argument, but the S3
  prefix falls back to headers/`local`/`default` for that slot, so a tool's
  own id can never leak into the bucket layout. That "reaches the tool as a
  normal argument" only holds for a *non-empty* value: `SessionScopePlugin`
  fills in ANY declared param left blank at the ADK boundary, scope-named or
  not, so a caller that leaves the tool's own `user_id`/`session_id` empty
  gets it silently populated from the graph's scope, same as the dedicated
  scope params.
- **Not a security boundary.** Neither the param- nor the header-derived
  scoping is a security boundary, just a namespacing convenience; a presigned
  URL grants access to anyone who holds it, and nothing here authenticates
  the caller. An `http(s)://` input is fetched from inside the serve
  container, which holds the S3 credentials, and any host that container can
  reach is allowed.
- **Reaching S3 from the serve container.** The endpoint is opened from inside
  the container, where `localhost` and `0.0.0.0` are the container itself.
  `start_chain.py` handles a loopback endpoint on the local daemon: the
  container gets it as `host.docker.internal` (`--add-host
  host.docker.internal:host-gateway`), and `S3__EXTERNAL_ENDPOINT_URL` keeps
  the configured address for presigned links. MinIO then has to listen on the
  docker bridge as well, a `-p 127.0.0.1:9000:9000` publish is not enough. A
  remote endpoint is passed as is. S3 calls are bounded like the rest of the
  project (5 s to connect, 30 s per read, 3 attempts).
- **`build_serve.sh`** forwards all S3 variables (`ENDPOINT_URL`/
  `ACCESS_KEY`/`SECRET_KEY`/`BUCKET_NAME`/`S3_REGION`/`S3_PRESIGN_EXPIRATION`/
  `S3_HTTP_TIMEOUT`/`S3_HTTP_MAX_BYTES`/`S3_UPLOAD_MAX_BYTES`, plus the
  `S3__*` names) to the serve container straight from the calling shell's
  environment — it does **not** read `.env` (that's `start_chain.py`'s job)
  and does not rewrite a loopback endpoint; export them yourself before
  invoking it.
- **Serve-only, by design.** `start_chain.py` never forwards these into the
  *build* container (it runs arbitrary repository code) — only into *serve*.
  A consequence: a tool whose recorded `sample_args` references an `s3://`
  URI will fail the validator's live-invocation check during the build
  (S3 isn't configured there) even though the same call succeeds once served.
