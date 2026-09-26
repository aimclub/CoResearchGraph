# Synapse Contracts v1 — adapter layer for CoScientist checkpoints

**Date:** 2026-07-27
**Status:** Design approved, ready for implementation plan
**Branch:** `feat/synapse-v1-adapter` (off `feat/checkpoints` @ `7f48e2d`)

## Context

Synapse published a trial "core contract v1" for how an external MAS plugs into the
platform (`SynapseNmas-contracts-v1`, simplified from the earlier `SynapseNmas.md`).
This session verified that CoScientist's existing checkpoint subsystem already
realizes almost the whole v1 checkpoint model: snapshots save on module boundaries,
the snapshot stays on the MAS side (local zip store), and restore = a new run
rehydrated from the bundle over A2A (proven end-to-end: kill → fresh process →
`POST /restore` → continue via `message/send`).

This spec closes the gaps between what exists and v1: five plumbing items plus the
structural "one adapter endpoint" requirement. There is **no live Synapse platform**
to integrate against, so we build the adapter side of the contract **and** a small
local mock platform that exercises it end-to-end (the same way we drove the restore
demo this session). Later the mock is swapped for the real Synapse.

## Goals (the 6 items)

1. **Platform-issued `run_id`** (and incoming `traceparent`) instead of a self-generated id.
2. **Adapter → platform "snapshot ready"** outbound notification with a snapshot reference.
3. **Stable `snapshot_ref`** the platform can store and hand back (snapshot stays with the MAS).
4. **Platform-initiated restore** ("continue from `point_id`") driving a new run.
5. **OTel minimal:** accept `traceparent`, hang our steps (`invoke_agent`/`execute_tool`/`chat`
   + core attributes) under the platform's trace, export via OTLP.
6. **One adapter endpoint / one run / one set of points** for the whole MAS.

## Non-goals

- Deterministic `replay` (strict/fresh) — not in v1; needs the platform journal.
- Full GenAI semantic-convention coverage (all attributes, token/cost accounting) — minimal only.
- A dedicated `/fork` endpoint or a branch-tree UI — v1 doesn't ask for it; restore already branches.
- Real Synapse integration — out of scope; we target a documented interface + a local mock.
- Changing the science, the agent tree, or the sub-agents' internal behaviour.

## Background — what already works (verified this session)

- Checkpoint = zip bundle (`manifest.json` + `blobs/sha256-*`), stored by `LocalZipStore`
  under `checkpoints_data/<run_id>/` on the MAS side.
- Manifest already carries `checkpoint_id`, `run_id`, `created_at`, `label`, `resume_from`,
  session blobs, module stores, and reproducibility pins; secrets are redacted.
- `run_id = f"{session.app_name}__{session.id}"` (`capture.py:209`) — **self-generated**.
- REST API (`checkpoints/api.py`): `GET list`, `GET /{id}`, `GET /{id}/bundle`,
  `POST /{id}/restore`. No outbound notification anywhere.
- Restore creates a new run (new `contextId`), rehydrates from the bundle, continues over
  plain A2A. Process-wide busy gate returns `409` during an active invocation.
- In `run_all` each of the 7 A2A servers checkpoints its own session into the shared dir →
  multiple `run_id`s (`orchestrator__`, `research__`, `hypotheses__`, `coder__`).

## Architecture

New, small, and reuses the working checkpoint/restore code.

### Components

- **`CoScientist/checkpoints/synapse.py`** — the v1 bridge (the only substantial new module):
  - reads platform `run_id` + `traceparent` from the incoming A2A `message.metadata`;
  - stashes them in session state (`synapse:run_id`, `synapse:traceparent`);
  - after a checkpoint is saved, fires the best-effort "snapshot ready" callback;
  - builds the `snapshot_ref`.
- **`scripts/mock_synapse.py`** — the platform stand-in (test/dev harness):
  - issues a `run_id` + a root `traceparent`, sends the task to the adapter via A2A
    `message/send` with those in `metadata`;
  - exposes `POST /points` to receive "snapshot ready" callbacks and record points;
  - triggers restore: picks a recorded point, issues a **new** `run_id`, calls the
    adapter's restore with the `point_id`, then continues;
  - collects OTLP spans to show the stitched trace.
- **Hooks into existing files** (small diffs):
  - `plugin.py` — after `store.save(...)`, call the synapse callback;
  - `capture.py` — take `run_id` from `state["synapse:run_id"]` when present (fallback = current);
  - `api.py` — the platform-facing checkpoint list is filtered to the current orchestrator run;
  - `a2a/serve.py` / `a2a/server.py` — read `message.metadata`, set the OTel context from
    `traceparent`, advertise the checkpoint extension in the Agent Card.

### The 6 changes — concrete

| # | v1 item | Change |
|---|---|---|
| endpoint | one MAS → one A2A endpoint, one set of points | Adapter = **orchestrator only** (`a2a serve orchestrator`, remote mode). Its checkpoint API returns only the **current orchestrator run's** points (filter `store.list`). Sub-agent servers still checkpoint internally but are invisible to the platform. |
| 1 | platform-issued `run_id` (+`traceparent`) | Read from `message.metadata` (`synapse.run_id`, `traceparent`) at entry → session state. `capture.py` stamps `run_id` from there, fallback = `app_name__id`. |
| 2 | adapter → platform "snapshot ready" | After `store.save`, best-effort `POST {SYNAPSE__CALLBACK_URL}/points` with `{point_id, run_id, time, label, snapshot_ref}`. No separate event. Failures logged, never break the run. |
| 3 | stable `snapshot_ref` | `snapshot_ref` = URL of the existing `GET /api/checkpoints/{id}/bundle` (snapshot stays with us; platform keeps the link). Added to the callback and the manifest. |
| 4 | platform-initiated restore | Reuse `POST /restore`; the **mock platform** is the caller. The restored run takes the platform's **new** `run_id` (ties to #1). Busy gate (409) already enforces "prior instance stopped". |
| 5 | OTel minimal | At entry, build an OTel context from `traceparent` so ADK spans (`invoke_agent`/`execute_tool`/`chat` + attrs: agent/tool/model, status, `run_id`) hang under the platform trace; export OTLP to the mock's collector. |

### Data flow (end-to-end through the mock)

```
mock: issue run-1a2b + traceparent → A2A message/send (metadata) → adapter (orchestrator)
adapter: run_id/traceparent into state; ADK spans hang under the platform trace
module boundary: store.save(bundle) → POST /points {point_id, snapshot_ref=…/bundle} → mock records point
mock: "continue from point_id" → issues a new run_id → POST /restore → adapter rehydrates from bundle → new run
```

### Interfaces

- **Incoming A2A `message.metadata`:** `synapse.run_id: str`, `traceparent: str` (W3C).
- **Callback body (`POST {SYNAPSE__CALLBACK_URL}/points`):**
  `{ point_id, run_id, time (ISO-8601), label, snapshot_ref (URL) }`.
- **`snapshot_ref`:** `http://<adapter-host>:<port>/api/checkpoints/{checkpoint_id}/bundle`.
- **Config (settings, `SYNAPSE__*`):** `enabled: bool`, `callback_url: str|None`,
  `bundle_base_url: str|None` (host:port used to build `snapshot_ref`),
  `otlp_endpoint: str|None`. All default off / None → v1 bridge is a no-op, existing
  behaviour unchanged.

### Testing

- New e2e (patterned on `tests/e2e_checkpoint_a2a.py`) driven **through the mock**, using the
  scripted deterministic agent (no LLM): assert
  (a) point `run_id` == the platform-issued id;
  (b) the "snapshot ready" callback arrived with a resolvable `snapshot_ref`;
  (c) restore-by-`point_id` reconstitutes state and the new run continues;
  (d) the exported trace has one platform root with our steps as children.
- `tests/e2e_checkpoint_a2a.py` (bare restore) stays green — the bridge is off by default.
- Unit tests for `synapse.py`: metadata parsing, callback body shape, `snapshot_ref` build,
  `run_id` precedence (platform value wins, fallback otherwise).

## File layout / branch

- Branch: `feat/synapse-v1-adapter` (off `feat/checkpoints`).
- New: `CoScientist/checkpoints/synapse.py`, `scripts/mock_synapse.py`,
  `tests/e2e_synapse_v1.py`, this spec.
- Edited: `checkpoints/plugin.py`, `checkpoints/capture.py`, `checkpoints/api.py`,
  `a2a/serve.py`, `a2a/server.py`, `config/settings.py` (+`SynapseSettings`).

## Risks / open questions

- **Metadata plumbing depth:** confirm ADK's A2A path surfaces `message.metadata` to a
  callback/entry hook; if not, fall back to a light `/runs` service call for run_id/traceparent.
- **OTLP export in `run_all`:** the adapter (orchestrator) is the trace parent; sub-agent
  servers' spans need the same trace context propagated over A2A — for v1-minimal it is enough
  to show the orchestrator subtree stitched under the platform root.
- **Single-endpoint filtering:** "current orchestrator run" must be well-defined when several
  restores coexist; filter by the active/most-recent orchestrator `run_id`, not all of them.
- **Adjacent (not in scope but noted):** v1 §4 requires `input-required` to reach the external
  endpoint; in A2A mode the HITL handler is not swapped today — a separate known gap.
