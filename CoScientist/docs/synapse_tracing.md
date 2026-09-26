# Synapse trace-context boundary

Status: accepted for the Synapse v1 bridge repair.

## Context

Synapse registers a canonical `run_id`, A2A `context_id`, and W3C
`traceparent` before the first platform request. The A2A SDK and ADK create
their native handler and `invocation` spans before ADK plugin callbacks run.
Starting a bridge span in `before_run_callback` therefore cannot parent those
native spans. Remote ADK agents also create a new context ID in the receiving
process, where the process-local registration is unavailable.

## Decision

The application installs a typed JSON-RPC handler owned by CoScientist. It
enters the trace scope before delegating to the SDK `JSONRPCHandler` methods.
Streaming methods keep the scope for the complete async iteration and reset it
in `finally`, including close and cancellation. Task operations recover their
context ID through the configured `TaskStore`; protocol payload parsing and
validation remain SDK responsibilities.

For a registered context, its canonical `run_id` always wins. A valid inbound
parent is used only when it belongs to the registered trace; otherwise the
registered root is used. For an unregistered remote context, a valid W3C
`traceparent` together with `run_id` baggage continues the distributed trace.
With neither source, execution keeps its previous standalone behavior.
Registration with a missing or invalid root remains authoritative and suppresses
foreign inbound or ambient Run binding instead of falling through to it.

The scope uses OpenTelemetry context and baggage. A small synchronous
`SpanProcessor.on_start` copies the active canonical `run_id` to native spans
when their trace matches the active parent. Export remains exclusively on the
standard `BatchSpanProcessor`; request paths never flush or wait for export.

`RemoteA2aAgent` receives a supported ADK request interceptor. It injects the
current W3C trace context and baggage into the SDK `ClientCallContext` HTTP
headers. It does not replace remote A2A context IDs, ADK session IDs, checkpoint
registry entries, or manifest identity.
The interceptor copies `ClientCallContext.state`, `http_kwargs`, and `headers`
before injection, preserving the scientific session state and existing
authorization or custom headers across concurrent calls.

## Failure behavior

Invalid or incomplete tracing input is ignored. Trace setup and lookup failures
are logged and degrade tracing only. Business awaits remain outside tracing
fallback handlers, so failures are neither swallowed nor retried. Context
tokens are always reset in the task that attached them.
