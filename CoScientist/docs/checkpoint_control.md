# Checkpoint control access and notification delivery

## Administrative trust boundary

All five operations under `/api/checkpoints` (including the web app's
router) require `Authorization: Bearer <credential>`.
Set `CHECKPOINTS__API_TOKEN` to a randomly generated secret of at least
32 characters, supplied through deployment secret configuration.
The value is a Pydantic `SecretStr`; never put it in a URL, agent prompt,
frontend bundle, browser storage, or an AgentCard.

This credential grants **instance-administrator access to every Run and
checkpoint in the dedicated CoScientist instance**. It belongs only to the
trusted Synapse backend. Synapse checks end-user tenant/project/Run/point
permissions before invoking this API. The adapter validates the administrative
credential on every operation; it does not validate per-user Run claims.
Ordinary A2A clients must not receive this credential.

This is not multi-tenant isolation inside CoScientist: bundles include shared
process stores and cross-run knowledge memory. A manifest's `run_id` alone
cannot safely authorize a different tenant to download the ZIP. Different
trust domains need separate instances/storage or a separate isolation design.
Registration and restore are also privileged because they mutate shared state.

Responses:

- Missing configuration: **503**, all control operations remain closed.
  Local checkpoint capture and ordinary A2A execution still work.
- Missing, wrong, or non-Bearer credential: **401**, without reading checkpoint
  storage or changing session/registry/store state.
- Valid administrator credential: existing request/response shapes and busy
  restore protection are unchanged.

The unauthenticated web panel explains that checkpoint management is available
through Synapse. It does not embed the administrator credential.

## Synapse configuration

Configure the CoScientist A2A server connection in Synapse with its existing
authentication setting:

```json
{"auth": {"type": "bearer", "token_env": "COSCIENTIST_CHECKPOINT_TOKEN"}}
```

The Synapse backend's `COSCIENTIST_CHECKPOINT_TOKEN` must contain the same secret
as CoScientist's `CHECKPOINTS__API_TOKEN`. Its existing CheckpointClient forwards
this operational authentication to registration and restore. Server-to-server
bundle download must also supply it. A `snapshot_ref` is an identifier/URL, not
a public signed download link. Keep TLS or a trusted private transport between
the two services; this setting does not secure the separate A2A data API.

Existing deployments enabling checkpoints must configure both sides before
upgrading. An unconfigured deployment will receive a closed control API, not an
anonymous compatibility fallback. This change does not automatically modify
any deployment secrets.

## Restore during execution

Restore returns **409** while any checkpoint-enabled Runner in this process
is executing, even on a different A2A port. There is no maximum busy age:
an invocation running for hours is still active. Restore imports shared
process stores, so a timeout cannot safely establish that they are idle.

Both the A2A factory and the CLI/web manager use `CheckpointRunner`. Custom
hosts installing `CheckpointPlugin` must also use this Runner. Each invocation
holds its own guard until the underlying ADK iterator has closed, including
cleanup after completion, failure, cancellation or early stream closure.
This also covers ADK error paths that skip `after_run_callback`. Finishing one
invocation does not release another invocation's guard. Stream consumers must
close an unfinished iterator (for example with `contextlib.aclosing`). A stuck
invocation remains busy until it actually terminates; elapsed time is not a
recovery mechanism.

## Notification delivery

After a bundle is saved locally, capture enqueues the existing snapshot-ready
payload and returns. One process-wide worker thread sends requests; the event
loop and other A2A servers do not wait for callback HTTP.

The queue holds at most **64 waiting notifications plus one in flight**.
Each HTTP attempt uses a **5-second HTTPX timeout**. There are no automatic
retries. Overflow, non-success HTTP status, transport failure, and incomplete
shutdown produce `[SNAPSHOT_NOTIFY]` warnings identifying the point, without
logging callback URLs or payloads. Failures leave the bundle on disk.
Normal process exit allows up to five seconds for draining; forced termination
and prolonged outages can lose notifications. The queue is not a durable outbox
and does not promise eventual delivery. Operators can inspect/list local
checkpoints through the authenticated control API.

OTel export remains independently handled by `BatchSpanProcessor`.

## Verification without LLM calls

```sh
LLM__MAIN_MODEL=test-model uv run --frozen pytest \
  tests/test_checkpoint_auth.py tests/test_checkpoint_busy.py \
  tests/test_snapshot_notifications.py \
  tests/test_synapse_bridge.py tests/test_synapse_otel.py \
  tests/test_synapse_trace_context.py tests/test_synapse_remote_context.py \
  tests/test_synapse_a2a_boundary.py tests/test_synapse_native_a2a.py \
  tests/test_synapse_remote_a2a.py -q
LLM__MAIN_MODEL=test-model uv run --frozen python tests/e2e_synapse_v1.py
LLM__MAIN_MODEL=test-model uv run --frozen python tests/e2e_checkpoint_control.py
LLM__MAIN_MODEL=test-model uv run --frozen python tests/e2e_checkpoint_busy.py
```

The HTTP scenario uses scripted ADK agents, not a model. Test credentials are
local fixtures, not deployment credentials.
