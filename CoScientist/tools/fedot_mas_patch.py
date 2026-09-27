"""Minimal CoScientist workarounds for fragile FEDOT.MAS ``transfer_to_agent`` routing.

Kept separate from ``fedotmas_tools.py`` so the tool wiring stays readable and
patches survive review / can be dropped when upstream fixes land.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from google.adk.agents.base_agent import BaseAgent

from google.adk.plugins import BasePlugin

from fedotmas import MAS, MAW
from fedotmas.common.llm import _ProxyClient
from fedotmas.mas.builder import _build_routing_agent, build_routing_system
from fedotmas.mas.models import MASAgentConfig, MASConfig
from fedotmas.maw.models import MAWAgentConfig, MAWConfig

from CoScientist.agents.callbacks.json_output import _unwrap_completion_state

_log = logging.getLogger(__name__)

# LiteLLM-style provider prefixes. FEDOT's ``_ProxyClient`` forwards model names
# as-is to OPENAI_BASE_URL; OpenRouter expects ``vendor/model`` (e.g.
# ``z-ai/glm-5.2``), so we strip the LiteLLM router segment (``openrouter/``).
_LITELLM_PROVIDER_PREFIXES = ("openai/", "openrouter/", "azure/", "anthropic/", "google/")


def strip_litellm_provider_prefix(model: str) -> str:
    """Drop a leading LiteLLM provider segment for OpenAI-compatible proxies."""
    if not model or "/" not in model:
        return model
    lower = model.lower()
    for prefix in _LITELLM_PROVIDER_PREFIXES:
        if lower.startswith(prefix):
            return model[len(prefix):]
    return model


def ensure_openai_provider_prefix(model: str) -> str:
    """FEDOT ``MASConfig`` requires a provider prefix; bare ids get ``openai/``.

    Keep ``openai/<id>`` / ``openrouter/<vendor>/<id>`` in validated configs;
    ``_install_proxy_model_strip`` removes the LiteLLM segment on the wire.
    """
    if not model or not isinstance(model, str) or "/" in model:
        return model
    return f"openai/{model}"


def _prefix_bare_models_in_obj(obj: Any) -> Any:
    """Recursively rewrite bare ``model`` string fields to ``openai/...``."""
    if isinstance(obj, dict):
        return {
            k: ensure_openai_provider_prefix(v) if k == "model" and isinstance(v, str)
            else _prefix_bare_models_in_obj(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_prefix_bare_models_in_obj(item) for item in obj]
    return obj


def _autonomous_kwargs(func: Any, autonomous: bool) -> dict[str, Any]:
    """``{"autonomous": ...}`` when *func* accepts it, else ``{}``.

    FEDOT.MAS gained a keyword-only ``autonomous`` on ``build`` and the routing
    builders after the commit uv.lock pins (8c74cad -> 7d6ee16). Threading it
    unconditionally breaks the pinned version; refusing it breaks the new one,
    where BaseMAS.build_app calls ``self.build(config, autonomous=...)`` and our
    override raised TypeError on every single FEDOT run.
    """
    try:
        import inspect
        if "autonomous" in inspect.signature(func).parameters:
            return {"autonomous": autonomous}
    except (TypeError, ValueError):  # builtins / C callables
        pass
    return {}


def _install_proxy_model_strip() -> None:
    """Monkeypatch ``_ProxyClient.acompletion`` once so proxy calls strip providers."""
    if getattr(_ProxyClient.acompletion, "_coscientist_strips_provider", False):
        return
    original = _ProxyClient.acompletion

    async def acompletion(self, model, messages, tools, **kwargs):  # noqa: ANN001
        from CoScientist.execution_control import before_model_attempt
        # OpenAI's default transport retries would otherwise bypass the shared
        # quota. This proxy sends one HTTP attempt per reservation.
        if hasattr(self, "_client"):
            self._client.max_retries = 0
        await before_model_attempt("fedot_proxy", metadata={"model": model})
        stripped = strip_litellm_provider_prefix(model)
        if stripped != model:
            _log.debug("PatchedMAS: proxy model %r → %r", model, stripped)
        return await original(self, stripped, messages, tools, **kwargs)

    acompletion._coscientist_strips_provider = True  # type: ignore[attr-defined]
    _ProxyClient.acompletion = acompletion  # type: ignore[method-assign]


def _install_parse_llm_output_unwrap() -> None:
    from fedotmas.meta._helpers import parse_llm_output as _orig

    if getattr(_orig, "_coscientist_unwraps_completion_state", False):
        return

    def parse_llm_output(raw: Any, schema: Any) -> Any:  # noqa: ANN001
        unwrapped = _unwrap_completion_state(raw)
        # Routing meta sometimes emits bare ids (``glm-4.7``); MASConfig
        # validation requires a provider prefix. Prefix before pydantic parse.
        if isinstance(unwrapped, (dict, list)):
            unwrapped = _prefix_bare_models_in_obj(unwrapped)
        elif isinstance(unwrapped, str):
            try:
                parsed = json.loads(unwrapped)
            except Exception:
                parsed = None
            if isinstance(parsed, (dict, list)):
                unwrapped = json.dumps(_prefix_bare_models_in_obj(parsed), ensure_ascii=False)
        return _orig(unwrapped, schema)

    parse_llm_output._coscientist_unwraps_completion_state = True  # type: ignore[attr-defined]
    import fedotmas.meta._helpers as _helpers

    _helpers.parse_llm_output = parse_llm_output  # type: ignore[method-assign]


def ensure_fedot_openai_proxy_compat() -> None:
    """Install wire-strip + MASConfig model-prefix fixes (idempotent).

    Safe to call from vanilla ``MAS`` paths (clean arm) as well as ``PatchedMAS``.
    """
    _install_proxy_model_strip()
    _install_fedot_budget_client()
    _install_parse_llm_output_unwrap()


def _install_fedot_budget_client() -> None:
    """Cover FEDOT's non-proxy LiteLLM clients without counting proxy twice."""
    import importlib
    import fedotmas.common.llm as llm_module

    original = llm_module.make_llm
    if getattr(original, "_coscientist_budgeted", False):
        return

    class BudgetedClient:
        def __init__(self, client):
            self.client = client

        def __getattr__(self, name):
            return getattr(self.client, name)

        async def acompletion(self, *args, **kwargs):
            from CoScientist.execution_control import before_model_attempt
            await before_model_attempt("fedot_litellm", metadata={"model": kwargs.get("model")})
            kwargs["num_retries"] = 0
            return await self.client.acompletion(*args, **kwargs)

    def make_llm(cfg):
        model = original(cfg)
        if not isinstance(model.llm_client, _ProxyClient):
            model.llm_client = BudgetedClient(model.llm_client)
        return model

    make_llm._coscientist_budgeted = True
    llm_module.make_llm = make_llm
    # These modules import the factory by name; changing the source module
    # alone leaves their existing aliases outside the budget boundary.
    for module_name in ("fedotmas.control._controller", "fedotmas.maw.builder", "fedotmas.meta._adk_runner"):
        module = importlib.import_module(module_name)
        if getattr(module, "make_llm", None) is original:
            module.make_llm = make_llm


# Configurable guard against injecting an unbounded task into every worker.
_MAX_TASK_CHARS = int(os.getenv("COSCIENTIST_FEDOT_TASK_MAX_CHARS", "6000"))
_IDENTIFIER_RE = re.compile(r"(?<!\w)[a-z][a-z0-9]+(?:_[a-z0-9]+)+(?!\w)")

_NO_GREET_SUFFIX = (
    " The user task is already provided below. Do NOT greet or ask clarifying"
    " questions. Immediately use your tools to complete your part and return"
    " concrete results (SMILES, scores, URLs) — never an empty acknowledgment."
)

_COORDINATOR_TRANSFER_SUFFIX = (
    " When calling transfer_to_agent, assume the worker may NOT see prior chat."
    " Restate the USER TASK and pass any intermediate artifacts (SMILES lists,"
    " docking scores, S3 URLs) explicitly in your message. Never greet the user."
)


class PatchedMAS(MAS):
    """MAS with small CoScientist patches around ADK AutoFlow routing.

    1. Drop tool names that are not real MCP servers (meta sometimes puts
       ``transfer_to_agent`` in ``tools`` → ``Unknown MCP server``).
    2. Single worker → run it directly (skip coordinator / transfer).
    3. Multi-worker → inject the full user task into every agent instruction
       so a ``transfer_to_agent`` with ``result: null`` does not lose the task.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Install proxy/model compat explicitly (idempotent) rather than as an
        # import-time side effect, so importing this module is inert.
        ensure_fedot_openai_proxy_compat()
        super().__init__(*args, **kwargs)
        self._pending_task: str | None = None

    async def generate_config(self, task: str) -> MASConfig:
        self._pending_task = task
        return await super().generate_config(task)

    async def build_and_run(
        self,
        config: MASConfig,
        user_query: str,
        *,
        initial_state: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self._pending_task = user_query
        return await super().build_and_run(config, user_query, initial_state=initial_state, timeout=timeout)

    def build(self, config: MASConfig, *, autonomous: bool = True) -> BaseAgent:
        registry = self._mcp_registry
        worker_models = self._worker_map()

        # Collapse to one worker BEFORE sanitize: sanitize injects ## USER TASK
        # into every worker (and may attach the same servers to empty-tool
        # workers), which would otherwise make all workers look equally relevant.
        picked = self._pick_single_worker(config)
        if picked is not None:
            config = config.model_copy(update={"workers": [picked]})
        config = self._sanitize_config(config)

        if len(config.workers) == 1:
            _log.info(
                "PatchedMAS: single worker %r — skipping coordinator/transfer",
                config.workers[0].name,
            )
            return _build_routing_agent(
                config.workers[0], registry, worker_models,
                **_autonomous_kwargs(_build_routing_agent, autonomous),
            )

        # Sanitize can equalize toolsets (empty-tool workers get all servers);
        # retry the collapse once before falling back to full transfer routing.
        picked = self._pick_single_worker(config)
        if picked is not None:
            return _build_routing_agent(
                picked, registry, worker_models,
                **_autonomous_kwargs(_build_routing_agent, autonomous),
            )

        _log.info(
            "PatchedMAS: building routing system (%d workers) with task injection",
            len(config.workers),
        )
        return build_routing_system(
            config, mcp_registry=registry, worker_models=worker_models,
            **_autonomous_kwargs(build_routing_system, autonomous),
        )

    def _pick_single_worker(self, config: MASConfig) -> MASAgentConfig | None:
        """Reduce many workers to one when transfer routing adds no capability.

        Tried in order: (1) an unambiguous runtime-identifier match from the
        task, (2) the sole registered MCP server, (3) an identical toolset across
        every worker. Returns None when workers genuinely differ, so the full
        coordinator/transfer routing system is built. No repository-owned
        function or server allowlist is used.
        """
        workers = config.workers
        if len(workers) <= 1:
            return workers[0] if workers else None

        # (1) Runtime-identifier narrowing — only on an unambiguous winner, so a
        # named function is never attached to the wrong MCP server on a tie.
        identifiers = set(_IDENTIFIER_RE.findall((self._pending_task or "").lower()))
        if identifiers:
            scored = sorted(
                (
                    (
                        sum(
                            1
                            for ident in identifiers
                            if re.search(
                                rf"(?<!\w){re.escape(ident)}(?!\w)",
                                f"{w.name} {w.description} {w.instruction} {' '.join(w.tools)}".lower(),
                            )
                        ),
                        w,
                    )
                    for w in workers
                ),
                key=lambda sw: sw[0],
                reverse=True,
            )
            if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                _log.info(
                    "PatchedMAS: narrowing %d workers → %r from runtime identifiers %s",
                    len(workers), scored[0][1].name, sorted(identifiers),
                )
                return scored[0][1]

        # (2) A single registered MCP server ⇒ prefer the worker already listing it.
        known = list(getattr(self, "_mcp_registry", None) or {})
        if len(known) == 1:
            server = known[0]
            pick = next((w for w in workers if server in (w.tools or [])), workers[0])
            _log.info(
                "PatchedMAS: single MCP server %r — collapsing %d workers → %r",
                server, len(workers), pick.name,
            )
            return pick

        # (3) Identical toolset across all workers ⇒ transfers add no capability.
        signatures = {tuple(sorted(w.tools or [])) for w in workers}
        if len(signatures) == 1:
            _log.info(
                "PatchedMAS: all workers share tools %s — collapsing to %r",
                sorted(next(iter(signatures))), workers[0].name,
            )
            return workers[0]
        return None

    def _sanitize_config(self, config: MASConfig) -> MASConfig:
        # Registry keys are MCP *server names* (e.g. GenerativeMoleculeModels).
        # Meta may emit MCP *function* names where registry *server* names are
        # required; those must not leave the worker with an empty toolset.
        known = sorted(self._mcp_registry or {})
        known_set = set(known)
        task_block = self._task_block()

        def clean(agent: MASAgentConfig, *, is_coordinator: bool) -> MASAgentConfig:
            original = list(agent.tools)
            tools = [t for t in original if t in known_set]
            bad = [t for t in original if t not in known_set]
            if bad:
                _log.warning(
                    "PatchedMAS: dropping unknown tools %s from agent %r (known=%s)",
                    bad, agent.name, known,
                )
            # Workers with no valid server tools cannot call MCP — attach every
            # server we were given (CoScientist already filters to task-relevant
            # servers before constructing PatchedMAS).
            if not tools and known and not is_coordinator:
                tools = list(known)
                _log.warning(
                    "PatchedMAS: agent %r had no registry tools (had %s) — attaching %s so MCP can run",
                    agent.name, original, tools,
                )
            instruction = agent.instruction.rstrip()
            if tools and "Do NOT greet" not in instruction:
                instruction += _NO_GREET_SUFFIX
            if is_coordinator and "Restate the USER TASK" not in instruction:
                instruction += _COORDINATOR_TRANSFER_SUFFIX
            if task_block and "## USER TASK" not in instruction:
                instruction += task_block
            return agent.model_copy(update={"tools": tools, "instruction": instruction})

        return config.model_copy(
            update={
                "coordinator": clean(config.coordinator, is_coordinator=True),
                "workers": [clean(w, is_coordinator=False) for w in config.workers],
            }
        )

    def _task_block(self) -> str:
        task = (self._pending_task or "").strip()
        if not task:
            return ""
        if len(task) > _MAX_TASK_CHARS:
            task = task[:_MAX_TASK_CHARS] + "\n…[truncated]"
        return f"\n\n## USER TASK (authoritative — execute this; do not greet)\n{task}\n"


class PatchedMAW(MAW):
    """MAW with the one CoScientist patch that still applies.

    MAW is a *fixed pipeline* (Sequential/Parallel/Loop over an agent pool), so
    two of the three PatchedMAS workarounds are moot here: there is no
    coordinator and no ``transfer_to_agent``, so nothing can lose the task
    mid-hop and there is no single-worker collapse to make. What survives is
    tool sanitising — the meta-agent still emits MCP *function* names where
    registry *server* names belong, which would leave an agent unable to call
    MCP at all.

    Chosen over MAS because MAW generates its config in two smaller LLM calls
    (agent pool, then pipeline tree) instead of one large MASConfig. On this
    stand the single-shot MASConfig failed to parse in roughly half of all runs
    (measured 2026-09-01: 66 retry failures, 82 "did not produce
    agent_system_config" across 46 runs, on two different models).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ensure_fedot_openai_proxy_compat()
        super().__init__(*args, **kwargs)
        self._pending_task: str | None = None

    async def generate_config(self, task: str) -> MAWConfig:
        self._pending_task = task
        return await super().generate_config(task)

    async def build_and_run(
        self,
        config: MAWConfig,
        user_query: str,
        *,
        initial_state: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self._pending_task = user_query
        return await super().build_and_run(
            config, user_query, initial_state=initial_state, timeout=timeout
        )

    def build(self, config: MAWConfig, *, autonomous: bool = True) -> BaseAgent:
        return super().build(
            self._sanitize_config(config),
            **_autonomous_kwargs(super().build, autonomous),
        )

    def _sanitize_config(self, config: MAWConfig) -> MAWConfig:
        """Keep only tool names the MCP registry actually knows.

        Registry keys are MCP *server* names. An agent left with an empty
        toolset cannot reach MCP at all, so — as in the MAS path — it gets every
        server we were handed; CoScientist has already narrowed those to the
        task before constructing the engine.
        """
        known = sorted(self._mcp_registry or {})
        known_set = set(known)
        task_block = self._task_block()

        def clean(agent: MAWAgentConfig) -> MAWAgentConfig:
            original = list(agent.tools)
            tools = [t for t in original if t in known_set]
            if bad := [t for t in original if t not in known_set]:
                _log.warning(
                    "PatchedMAW: dropping unknown tools %s from agent %r (known=%s)",
                    bad, agent.name, known,
                )
            if not tools and known:
                tools = list(known)
                _log.warning(
                    "PatchedMAW: agent %r had no registry tools (had %s) — attaching %s so MCP can run",
                    agent.name, original, tools,
                )
            instruction = agent.instruction.rstrip()
            if tools and "Do NOT greet" not in instruction:
                instruction += _NO_GREET_SUFFIX
            # No transfer_to_agent in a fixed pipeline, but a step still only
            # sees what upstream steps wrote to state, so restate the task.
            if task_block and "## USER TASK" not in instruction:
                instruction += task_block
            return agent.model_copy(update={"tools": tools, "instruction": instruction})

        return config.model_copy(update={"agents": [clean(a) for a in config.agents]})

    def _task_block(self) -> str:
        task = (self._pending_task or "").strip()
        if not task:
            return ""
        if len(task) > _MAX_TASK_CHARS:
            task = task[:_MAX_TASK_CHARS] + "\n…[truncated]"
        return f"\n\n## USER TASK (authoritative — execute this; do not greet)\n{task}\n"


class MetaJsonRecoveryPlugin(BasePlugin):
    """Recover a meta-agent config that the model emitted as prose or as a thought.

    ``_execute_meta_call`` reads the generated config from ``state[output_key]``
    and raises ``"<agent> did not produce '<key>' in session state"`` when it is
    empty. That message reads like the model returned nothing, but the dominant
    cause is narrower: the model DID answer, and ADK could not store the answer
    because it was not clean JSON in a plain text part — GLM in particular parks
    structured output in a ``thought`` part, and models routinely wrap the object
    in a fence or a sentence of preamble.

    This is by far the biggest single failure mode of the FEDOT route on this
    stand: 82 of 88 config-generation failures measured on 2026-09-01 were this
    message, against 6 genuine ValidationErrors, and it reproduces on both
    engines (MAS ``routing_meta_agent``, MAW ``pool_generator`` /
    ``pipeline_generator``) and on two different models.

    CoScientist already solves exactly this for its own agents with
    ``sanitize_json_output``; ``_execute_meta_call`` forwards ``plugins=`` to the
    ADK Runner, so the same repair can be delivered into FEDOT's meta agents
    without patching the library. Nothing is invented here: if no JSON can be
    recovered the response passes through untouched and the original error
    stands.
    """

    def __init__(self) -> None:
        super().__init__(name="coscientist_meta_json_recovery")
        self.repaired: list[str] = []

    async def after_model_callback(
        self, *, callback_context: Any, llm_response: Any
    ) -> Any | None:
        from CoScientist.agents.callbacks.json_output import _extract_json

        # Scope: only agents that declare an output_schema, i.e. the ones whose
        # answer ADK tries to store as structured output and whose failure to
        # store it produces "did not produce '<key>' in session state".
        #
        # This matters because plugins passed to the engine reach the PIPELINE
        # agents too, not just the meta agents. Observed 2026-09-02: an earlier
        # version rewrote a worker's answer ("rebuilt experiment_executor config
        # from the text part"), which replaces a prose report containing a JSON
        # blob with the blob alone — silently discarding the narrative the caller
        # actually reads. A worker has no output_schema, so this check excludes it.
        if not _expects_structured_output(callback_context):
            return None

        content = getattr(llm_response, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            return None
        # A tool call is not a config answer.
        if any(getattr(p, "function_call", None) for p in parts):
            return None

        plain = "".join(
            p.text for p in parts
            if getattr(p, "text", None) and not getattr(p, "thought", False)
        )
        thought = "".join(
            p.text for p in parts
            if getattr(p, "text", None) and getattr(p, "thought", False)
        )
        if plain.strip() and _extract_json(plain) is not None and plain.strip().startswith("{"):
            return None  # already clean — do not touch a good answer

        payload = _extract_json(plain) if plain.strip() else None
        source = "text"
        if payload is None and thought.strip():
            payload = _extract_json(thought)
            source = "thought"
        if not isinstance(payload, dict):
            # Nothing recoverable — the original error will report it. Say WHY
            # when the cause is knowable, because the alternative is what
            # happened on 2026-09-17: routing_meta_agent was cut off at the
            # output limit on both fedot_tool calls of EXP-3, the module fell
            # back to react_tools, and logs/app.log carried not one mention of
            # it. The reason survived only inside the TaskResult summary, where
            # nobody watching the run would look.
            if _finished_truncated(llm_response):
                _log.warning(
                    "MetaJsonRecovery: %s hit the output limit "
                    "(finish_reason=MAX_TOKENS) with no complete JSON left to "
                    "recover. Raise FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS — the "
                    "library default is 2048, and a reasoning model spends that "
                    "same budget on its thinking.",
                    _agent_name_of(callback_context),
                )
            return None

        agent = _agent_name_of(callback_context)
        self.repaired.append(agent)
        cut_off = (
            " — the answer was also flagged MAX_TOKENS, so the config arrived "
            "complete but only just; consider raising "
            "FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS"
            if _finished_truncated(llm_response) else ""
        )
        _log.warning(
            "MetaJsonRecovery: rebuilt %s config from the %s part "
            "(model returned it outside a clean JSON text part)%s",
            agent, source, cut_off,
        )
        return _rewritten_json_response(llm_response, payload)


def _finished_truncated(llm_response: Any) -> bool:
    """Did the model stop because it ran out of output budget?

    Read by name rather than by identity: the enum lives in google.genai.types
    and a provider that reports the reason as a bare string should read the
    same. Anything unrecognised is treated as "not truncated" — this only ever
    decides whether to add a line to the log.
    """
    reason = getattr(llm_response, "finish_reason", None)
    if reason is None:
        return False
    return "MAX_TOKENS" in str(getattr(reason, "name", reason)).upper()


def _agent_name_of(callback_context: Any) -> str:
    for attr in ("agent_name", "_agent_name"):
        if (value := getattr(callback_context, attr, None)):
            return str(value)
    ctx = getattr(callback_context, "_invocation_context", None)
    agent = getattr(ctx, "agent", None)
    return str(getattr(agent, "name", "meta_agent"))


def _rewritten_json_response(llm_response: Any, payload: dict) -> Any:
    """Same response, with one plain text part holding exactly the JSON object.

    The error flag is cleared with it, and that is the point. LiteLLM sets
    ``error_code`` for ANY non-STOP finish reason, content or not
    (``google/adk/models/lite_llm.py``), and the meta runner raises on any
    ``error_code`` at all (``fedotmas/meta/_adk_runner.py``) — with no carve-out,
    unlike its own sibling ``fedotmas/core/runner.py``, which continues on
    MAX_TOKENS whenever the step still produced content.

    So a config recovered in full still killed the run on the flag alone: the
    repair was real and the error stood anyway. Observed on 2026-09-17, when
    recovery fired inside two of the three failing fedot_tool windows of EXP-3
    and the calls failed regardless. If a complete object came back, the answer
    is usable whatever made the provider stop.
    """
    import copy as _copy
    import json as _json

    from google.genai import types as _types

    out = _copy.deepcopy(llm_response)
    out.content = _types.Content(
        role=getattr(getattr(llm_response, "content", None), "role", "model") or "model",
        parts=[_types.Part(text=_json.dumps(payload, ensure_ascii=False))],
    )
    for field in ("error_code", "error_message"):
        if getattr(out, field, None) is not None:
            try:
                setattr(out, field, None)
            except Exception:  # noqa: BLE001 - a frozen response is still repaired
                pass
    return out


def _expects_structured_output(callback_context: Any) -> bool:
    """True only for an agent whose answer ADK stores against an output_schema.

    Kept deliberately narrow and fail-closed: if the agent cannot be reached the
    answer is left alone, because rewriting a free-text answer loses information
    while declining to rewrite only leaves the original error in place.
    """
    agent = None
    getter = getattr(callback_context, "get_invocation_context", None)
    if callable(getter):
        try:
            agent = getattr(getter(), "agent", None)
        except Exception:  # noqa: BLE001 — a diagnostic path must never break the run
            agent = None
    if agent is None:
        ctx = getattr(callback_context, "_invocation_context", None)
        agent = getattr(ctx, "agent", None)
    return getattr(agent, "output_schema", None) is not None
