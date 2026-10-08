"""What a run cost: tokens and money per agent, plus sandbox compute.

Three things feed one per-session ledger:

* :class:`UsageMetricsPlugin` — an ADK plugin that sees every model call. Like
  the tool-activity plugin it also fires inside ``AgentTool`` sub-runners
  (plugins are inherited by the nested Runner), so a subordinate's own LLM
  traffic is counted against the subordinate, not swallowed by the delegation.
* :func:`record_completion` — for the handful of LLM calls made straight
  through ``litellm`` (critic, semantic layer, research validator) rather than
  through an ADK agent. They cost money too.
* :func:`record_sandbox_run` — the OpenHands sandbox bills its own agent, GPU
  and electricity; ``sandbox_tools`` forwards each finished run's digest here
  so one number covers the whole system.

**Metrics never reach a model.** They are host metadata: an agent that can see
its own bill optimises the bill instead of the task, and the numbers would cost
context tokens on every tool result to say so. Nothing here is written into
agent-visible state, and the sandbox client keeps them out of its tool result
for the same reason. The ways out are :func:`snapshot` (HTTP/console) and the
sink registered with :func:`set_metrics_sink` (the Web UI).

Pricing comes dynamically from OpenRouter's live API (cached in memory for
1 hour) and litellm's model table. Models that are not found are counted in
tokens and reported as ``unpriced_calls`` rather than silently priced at zero.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import urllib.request
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from google.adk.plugins.base_plugin import BasePlugin

from CoScientist.graph.session_scope import SessionKey, session_key

logger = logging.getLogger("CoScientist.logging.metrics")

#: Live snapshots are pushed at most this often per session — an agent run makes
#: hundreds of model calls and a browser needs none of that resolution.
PUSH_INTERVAL = float(os.getenv("METRICS_PUSH_INTERVAL", "2.0"))

#: The 12 sandbox numbers, in the vocabulary the sandbox server itself uses.
SANDBOX_FIELDS = (
    "runs", "wall_seconds", "agent_seconds", "queue_seconds",
    "cpu_core_seconds", "gpu_seconds", "energy_wh",
    "llm_calls", "total_tokens",
    "api_cost_usd", "energy_cost_usd", "total_cost_usd",
)

MetricsSink = Callable[[SessionKey, dict], Awaitable[None]]


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

_price_lock = threading.Lock()
_overrides: Optional[Dict[str, Dict[str, float]]] = None


def _price_overrides() -> Dict[str, Dict[str, float]]:
    """Parse ``LLM_PRICE_OVERRIDES`` once: JSON text or a path to a JSON file."""
    global _overrides
    if _overrides is not None:
        return _overrides
    with _price_lock:
        if _overrides is not None:
            return _overrides
        raw = (os.getenv("LLM_PRICE_OVERRIDES") or "").strip()
        parsed: Dict[str, Dict[str, float]] = {}
        if raw:
            try:
                if not raw.startswith("{"):
                    raw = open(raw, encoding="utf-8").read()
                loaded = json.loads(raw)
                parsed = {
                    str(model): {k: float(v) for k, v in prices.items()}
                    for model, prices in loaded.items()
                    if isinstance(prices, dict)
                }
            except Exception as exc:  # noqa: BLE001 - bad config must not break a run
                logger.warning("Ignoring LLM_PRICE_OVERRIDES: %s", exc)
        _overrides = parsed
    return _overrides


_openrouter_cache: Dict[str, tuple[float, Optional[Dict[str, float]]]] = {}
_openrouter_lock = threading.Lock()


def _fetch_openrouter_price_for_model(model: str) -> Optional[Dict[str, float]]:
    """Fetch live pricing for a specific requested model from OpenRouter API."""
    url = "https://openrouter.ai/api/v1/models"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CoScientist/1.0", "Accept": "application/json"},
    )
    target_slug = model.removeprefix("openrouter/").lstrip("~")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            for item in models:
                mid = str(item.get("id") or "")
                clean_mid = mid.removeprefix("openrouter/").lstrip("~")
                if clean_mid == target_slug or mid == model:
                    pr = item.get("pricing", {})
                    p_in = float(pr.get("prompt", 0) or 0) * 1_000_000
                    p_out = float(pr.get("completion", 0) or 0) * 1_000_000
                    p_cache = float(pr.get("input_cache_read", 0) or pr.get("prompt", 0) or 0) * 1_000_000
                    return {"input": p_in, "output": p_out, "cache_read": p_cache}
            return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to fetch OpenRouter pricing for %s: %s", model, exc)
        return None


def _get_openrouter_api_price(model: str) -> Optional[Dict[str, float]]:
    """Get model pricing from OpenRouter API for the requested model, cached for 1 hour."""
    now = time.monotonic()
    cached = _openrouter_cache.get(model)
    if cached is not None and (now - cached[0]) < 3600:
        return cached[1]

    with _openrouter_lock:
        cached = _openrouter_cache.get(model)
        if cached is not None and (now - cached[0]) < 3600:
            return cached[1]
        price = _fetch_openrouter_price_for_model(model)
        _openrouter_cache[model] = (now, price)
        return price


def price_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> tuple[float, str]:
    """Price one model call.

    ``prompt_tokens`` counts the whole input including the cached part, which
    is what every provider reports and what litellm expects — ``cached_tokens``
    only says how much of it was served from cache (and is therefore cheaper).

    Returns ``(usd, source)`` where source is ``override``, ``litellm``,
    ``openrouter-api`` or ``unavailable``. Zero cost with source ``unavailable``
    means "we do not know what this model costs", not "it was free".
    """
    fresh = max(prompt_tokens - cached_tokens, 0)

    override = _price_overrides().get(model)
    if override:
        cache_rate = override.get("cache_read", override.get("input", 0.0))
        usd = (
            fresh * override.get("input", 0.0)
            + cached_tokens * cache_rate
            + completion_tokens * override.get("output", 0.0)
        ) / 1_000_000
        return usd, "override"

    try:
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_input_tokens=cached_tokens,
        )
        return float(prompt_cost) + float(completion_cost), "litellm"
    except Exception:  # noqa: BLE001 - unmapped model, offline table, bad slug
        logger.debug("No price for model %s in litellm", model, exc_info=True)

    if model.startswith("openrouter/") or model.startswith("~"):
        or_price = _get_openrouter_api_price(model)
        if or_price:
            cache_rate = or_price.get("cache_read", or_price.get("input", 0.0))
            usd = (
                fresh * or_price.get("input", 0.0)
                + cached_tokens * cache_rate
                + completion_tokens * or_price.get("output", 0.0)
            ) / 1_000_000
            return usd, "openrouter-api"

    return 0.0, "unavailable"


def price_first(
    candidates: Iterable[str],
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> tuple[str, float, str]:
    """Price under the first name that has a price; return which one that was.

    A call is known by two names and they are not interchangeable. The request
    carries the routed slug (``openrouter/qwen/qwen3-235b-a22b-2507``) — the one
    litellm prices and the one an override key must match. The response echoes
    whatever the provider calls itself (``qwen/qwen3-235b-a22b-2507``), prefix
    stripped, which prices as unknown. Trying both keeps a renamed or aliased
    model from being reported as free.

    A price found under a later name is marked ``-alias``: it is the underlying
    provider's own list price, not the router's, so it may sit slightly below
    what was actually billed. An override on the configured slug is tried first
    and wins, which is how to pin the exact number.

    With no price anywhere, the FIRST candidate is returned as the label: that
    is the name to put in ``LLM_PRICE_OVERRIDES``, so the report names something
    actionable rather than the provider's echo.
    """
    names = [name for name in dict.fromkeys(candidates) if name]
    if not names:
        return "unknown", 0.0, "unavailable"

    for index, name in enumerate(names):
        cost, source = price_call(
            name, prompt_tokens, completion_tokens, cached_tokens,
        )
        if source != "unavailable":
            return name, cost, source if index == 0 else f"{source}-alias"
    return names[0], 0.0, "unavailable"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class ModelUsage:
    """Everything one agent spent on one model."""

    model: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    seconds: float = 0.0
    cost_source: str = "unavailable"

    def add(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        reasoning_tokens: int,
        total_tokens: int,
        cost_usd: float,
        cost_source: str,
        seconds: float,
    ) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cached_tokens += cached_tokens
        self.reasoning_tokens += reasoning_tokens
        self.total_tokens += total_tokens
        self.cost_usd += cost_usd
        self.seconds += seconds
        if cost_source == "unavailable":
            self.unpriced_calls += 1
        else:
            self.cost_source = cost_source

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "unpriced_calls": self.unpriced_calls,
            "seconds": round(self.seconds, 2),
            "cost_source": self.cost_source,
        }


@dataclass
class AgentUsage:
    """One agent's model traffic plus any sandbox it drove."""

    agent: str
    models: Dict[str, ModelUsage] = field(default_factory=dict)
    #: sandbox id -> digest. Keyed by sandbox because the server reports
    #: cumulative figures per container: a follow-up REPLACES its entry.
    sandboxes: Dict[str, dict] = field(default_factory=dict)

    def model(self, name: str) -> ModelUsage:
        usage = self.models.get(name)
        if usage is None:
            usage = ModelUsage(model=name)
            self.models[name] = usage
        return usage

    def llm_totals(self) -> dict:
        return {
            "calls": sum(m.calls for m in self.models.values()),
            "prompt_tokens": sum(m.prompt_tokens for m in self.models.values()),
            "completion_tokens": sum(m.completion_tokens for m in self.models.values()),
            "cached_tokens": sum(m.cached_tokens for m in self.models.values()),
            "reasoning_tokens": sum(m.reasoning_tokens for m in self.models.values()),
            "total_tokens": sum(m.total_tokens for m in self.models.values()),
            "cost_usd": round(sum(m.cost_usd for m in self.models.values()), 6),
            "unpriced_calls": sum(m.unpriced_calls for m in self.models.values()),
            "seconds": round(sum(m.seconds for m in self.models.values()), 2),
        }

    def sandbox_totals(self) -> dict:
        return sum_sandbox(self.sandboxes.values())

    def as_dict(self) -> dict:
        llm = self.llm_totals()
        sandbox = self.sandbox_totals()
        return {
            "agent": self.agent,
            "llm": llm,
            "sandbox": sandbox if sandbox["runs"] else None,
            "cost_usd": round(llm["cost_usd"] + sandbox["total_cost_usd"], 6),
            "models": [
                usage.as_dict() for usage in
                sorted(self.models.values(), key=lambda m: -m.cost_usd)
            ],
        }


def sum_sandbox(digests: Iterable[dict]) -> dict:
    """Add up sandbox digests into one set of totals."""
    totals = {field_name: 0.0 for field_name in SANDBOX_FIELDS}
    for digest in digests:
        for field_name in SANDBOX_FIELDS:
            value = digest.get(field_name)
            if isinstance(value, (int, float)):
                totals[field_name] += value
    for field_name in ("runs", "llm_calls", "total_tokens"):
        totals[field_name] = int(totals[field_name])
    for field_name in ("api_cost_usd", "energy_cost_usd", "total_cost_usd"):
        totals[field_name] = round(totals[field_name], 6)
    for field_name in ("wall_seconds", "agent_seconds", "queue_seconds",
                       "cpu_core_seconds", "gpu_seconds", "energy_wh"):
        totals[field_name] = round(totals[field_name], 2)
    return totals


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class UsageLedger:
    """Per-session, per-agent usage. Process-local and thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._agents: Dict[SessionKey, Dict[str, AgentUsage]] = {}
        self._started: Dict[SessionKey, float] = {}
        self._pushed: Dict[SessionKey, float] = {}

    def _agent(self, key: SessionKey, agent: str) -> AgentUsage:
        agents = self._agents.setdefault(key, {})
        self._started.setdefault(key, time.time())
        usage = agents.get(agent)
        if usage is None:
            usage = AgentUsage(agent=agent)
            agents[agent] = usage
        return usage

    def record_llm(
        self,
        key: SessionKey,
        agent: str,
        model: str,
        *,
        alt_models: Iterable[str] = (),
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        total_tokens: Optional[int] = None,
        seconds: float = 0.0,
        cost_usd: Optional[float] = None,
    ) -> None:
        """Record one call. ``model`` is the preferred name, ``alt_models`` the
        other names the same call is known by — see :func:`price_first`."""
        if cost_usd is not None and cost_usd >= 0:
            name, cost, source = model, float(cost_usd), "openrouter-response"
        else:
            name, cost, source = price_first(
                (model, *alt_models), prompt_tokens, completion_tokens, cached_tokens,
            )
        with self._lock:
            self._agent(key, agent).model(name).add(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                total_tokens=(
                    total_tokens if total_tokens is not None
                    else prompt_tokens + completion_tokens
                ),
                cost_usd=cost,
                cost_source=source,
                seconds=seconds,
            )

    def record_sandbox(
        self, key: SessionKey, agent: str, sandbox_id: str, digest: dict,
    ) -> None:
        with self._lock:
            self._agent(key, agent).sandboxes[sandbox_id] = dict(digest)

    def snapshot(self, key: SessionKey) -> dict:
        """A JSON-safe view of everything spent in this session so far."""
        with self._lock:
            agents = [usage.as_dict() for usage in self._agents.get(key, {}).values()]
            started = self._started.get(key)

        agents.sort(key=lambda a: -a["cost_usd"])
        llm = {
            name: sum(agent["llm"][name] for agent in agents)
            for name in (
                "calls", "prompt_tokens", "completion_tokens", "cached_tokens",
                "reasoning_tokens", "total_tokens", "unpriced_calls",
            )
        }
        llm["cost_usd"] = round(sum(a["llm"]["cost_usd"] for a in agents), 6)
        llm["seconds"] = round(sum(a["llm"]["seconds"] for a in agents), 2)
        # Name the models nobody could price, so the reader knows exactly what
        # to put in LLM_PRICE_OVERRIDES instead of hunting for it.
        llm["unpriced_models"] = sorted({
            model["model"]
            for agent in agents for model in agent["models"]
            if model["unpriced_calls"]
        })
        sandbox = sum_sandbox(
            agent["sandbox"] for agent in agents if agent["sandbox"]
        )
        return {
            "session": {"user_id": key[0], "session_id": key[1]},
            "updated_at": datetime.now().isoformat(),
            "since": datetime.fromtimestamp(started).isoformat() if started else None,
            "llm": llm,
            "sandbox": sandbox,
            "totals": {
                "cost_usd": round(llm["cost_usd"] + sandbox["total_cost_usd"], 6),
                "api_cost_usd": round(llm["cost_usd"] + sandbox["api_cost_usd"], 6),
                "energy_cost_usd": sandbox["energy_cost_usd"],
                "llm_calls": llm["calls"] + sandbox["llm_calls"],
                "total_tokens": llm["total_tokens"] + sandbox["total_tokens"],
                "energy_wh": sandbox["energy_wh"],
                "cpu_core_seconds": sandbox["cpu_core_seconds"],
                "gpu_seconds": sandbox["gpu_seconds"],
                # Honesty flag: some calls ran on a model with no known price,
                # so the money above is a floor, not the bill.
                "complete": llm["unpriced_calls"] == 0,
            },
            "agents": agents,
        }

    def reset(self, key: SessionKey) -> dict:
        """Forget one session, handing back its final snapshot."""
        final = self.snapshot(key)
        with self._lock:
            self._agents.pop(key, None)
            self._started.pop(key, None)
            self._pushed.pop(key, None)
        return final

    def sessions(self) -> List[SessionKey]:
        with self._lock:
            return list(self._agents)

    def due_for_push(self, key: SessionKey, force: bool) -> bool:
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._pushed.get(key, 0.0)) < PUSH_INTERVAL:
                return False
            self._pushed[key] = now
        return True


LEDGER = UsageLedger()


# ---------------------------------------------------------------------------
# Live channel
# ---------------------------------------------------------------------------

_sink: Optional[MetricsSink] = None


def set_metrics_sink(sink: Optional[MetricsSink]) -> None:
    """Register (or clear, with ``None``) the observer for live snapshots."""
    global _sink
    _sink = sink


def publish(key: SessionKey, *, force: bool = False) -> None:
    """Offer the current snapshot to the sink, rate-limited unless forced.

    Fire-and-forget by design: this is called from the model callback on the
    hot path, and delivering a number must never slow down or fail a run. With
    no event loop running (CLI, a worker thread) there is nothing to schedule
    on and the snapshot is simply skipped — :func:`snapshot` still has it.
    """
    sink = _sink
    if sink is None or not LEDGER.due_for_push(key, force):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def deliver() -> None:
        try:
            await sink(key, LEDGER.snapshot(key))
        except Exception as exc:  # noqa: BLE001 - an observer never fails a run
            logger.warning("Metrics sink failed: %s", exc)

    loop.create_task(deliver())


# ---------------------------------------------------------------------------
# Recording entry points
# ---------------------------------------------------------------------------

#: The session the current invocation belongs to, for code too deep to be
#: handed one. ``asyncio.create_task`` copies the current context, so a
#: background job spawned mid-run (semantic extraction, the research validator)
#: still bills the session that spawned it.
_CURRENT: ContextVar[Optional[SessionKey]] = ContextVar("metrics_session", default=None)


def bind_session(key: SessionKey) -> None:
    """Mark this task (and anything it spawns) as belonging to ``key``."""
    _CURRENT.set(key)


def resolve_key(
    context: Any = None, key: Optional[SessionKey] = None,
) -> SessionKey:
    """Best available session identity: explicit, then context, then ambient."""
    if key is not None:
        return key
    if context is not None:
        return session_key(context)
    return _CURRENT.get() or session_key(None)


def snapshot(context: Any = None, *, key: Optional[SessionKey] = None) -> dict:
    """Everything spent in one session, by agent. Pass a context or a key."""
    return LEDGER.snapshot(resolve_key(context, key))


def reset_session(context: Any = None, *, key: Optional[SessionKey] = None) -> dict:
    """Drop one session's ledger, returning its final snapshot."""
    return LEDGER.reset(resolve_key(context, key))


def record_sandbox_run(
    key: SessionKey, agent: str, sandbox_id: str, digest: dict,
) -> None:
    """File one finished sandbox run against the agent that started it."""
    LEDGER.record_sandbox(key, agent, sandbox_id, digest)
    publish(key, force=True)


def record_completion(
    response: Any,
    *,
    model: str,
    agent: str,
    context: Any = None,
    key: Optional[SessionKey] = None,
    seconds: float = 0.0,
) -> None:
    """Count a raw ``litellm`` completion — the calls no ADK plugin can see.

    With neither ``context`` nor ``key``, the session is taken from the ambient
    binding (:func:`bind_session`), which is what background jobs spawned inside
    a run have.

    Never raises: a metrics bug must not take down the caller it is measuring.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(completion_details, "reasoning_tokens", 0) or 0
        resolved = resolve_key(context, key)

        raw_cost = getattr(usage, "cost", None)
        if raw_cost is None and isinstance(usage, dict):
            raw_cost = usage.get("cost")
        cost_val = float(raw_cost) if raw_cost is not None else None

        LEDGER.record_llm(
            resolved,
            agent,
            model,
            alt_models=(str(getattr(response, "model", "") or ""),),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cached_tokens=int(cached),
            reasoning_tokens=int(reasoning),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0) or None,
            seconds=seconds,
            cost_usd=cost_val,
        )
        publish(resolved)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not record completion for %s: %s", agent, exc)


# ---------------------------------------------------------------------------
# ADK plugin
# ---------------------------------------------------------------------------

class UsageMetricsPlugin(BasePlugin):
    """Count tokens and price every model call, per agent, per session."""

    def __init__(self, name: str = "usage_metrics") -> None:
        super().__init__(name=name)
        # (invocation id, agent name) -> (started, requested model). Agent names
        # are unique in a tree and one agent's calls are sequential, so this
        # pairs a response with its own request even under parallel sub-agents.
        self._pending: Dict[tuple, tuple] = {}

    @staticmethod
    def _call_key(callback_context: Any) -> tuple:
        return (
            getattr(callback_context, "invocation_id", "") or "",
            getattr(callback_context, "agent_name", "") or "system",
        )

    async def before_run_callback(self, *, invocation_context) -> None:
        # Bind the session for the whole invocation task, so the LLM calls made
        # outside the agent tree (critic, semantic extraction, validator) know
        # whose bill they are on without every layer having to pass a key.
        try:
            bind_session(session_key(invocation_context))
        except Exception:  # noqa: BLE001 - context shapes vary across ADK paths
            pass
        return None

    async def before_model_callback(self, *, callback_context, llm_request) -> None:
        self._pending[self._call_key(callback_context)] = (
            time.monotonic(), getattr(llm_request, "model", None) or "",
        )
        return None

    async def after_model_callback(self, *, callback_context, llm_response) -> None:
        usage = getattr(llm_response, "usage_metadata", None)
        # Streamed partials carry no usage; only the aggregated response does,
        # and litellm attaches it exactly once — so nothing is counted twice.
        if usage is None or getattr(llm_response, "partial", False):
            return None

        call_key = self._call_key(callback_context)
        started, requested = self._pending.pop(call_key, (None, ""))
        # The request's name comes first: it is the routed slug the agent was
        # configured with, which is what litellm prices. The response echoes the
        # provider's own name for the model — prefix stripped, and unpriceable.
        echoed = getattr(llm_response, "model_version", None) or ""
        try:
            key = session_key(callback_context)
        except Exception:  # noqa: BLE001 - context shapes vary across ADK paths
            return None

        prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion = int(getattr(usage, "candidates_token_count", 0) or 0)

        raw_cost = getattr(usage, "cost", None) or getattr(llm_response, "cost", None)
        cost_val = float(raw_cost) if raw_cost is not None else None

        LEDGER.record_llm(
            key,
            call_key[1],
            str(requested or echoed or "unknown"),
            alt_models=(str(echoed),),
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=int(getattr(usage, "cached_content_token_count", 0) or 0),
            reasoning_tokens=int(getattr(usage, "thoughts_token_count", 0) or 0),
            total_tokens=int(getattr(usage, "total_token_count", 0) or 0) or None,
            seconds=(time.monotonic() - started) if started else 0.0,
            cost_usd=cost_val,
        )
        publish(key)
        return None

    async def on_model_error_callback(
        self, *, callback_context, llm_request, error,
    ) -> None:
        # A failed call never reaches after_model_callback; drop its start
        # record so the table cannot grow without bound over a long run.
        self._pending.pop(self._call_key(callback_context), None)
        return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _money(value: float) -> str:
    return f"${value:,.4f}" if value else "$0.0000"


def format_report(data: dict, *, title: str = "Usage & cost") -> str:
    """Render a snapshot as a console block."""
    llm, sandbox, totals = data["llm"], data["sandbox"], data["totals"]
    width = 78
    lines = [
        "=" * width,
        f"{title} — session {data['session']['session_id']}",
        "=" * width,
        f"{'agent':<26}{'calls':>7}{'tokens':>12}{'cost':>12}  models",
        "-" * width,
    ]
    for agent in data["agents"]:
        models = ", ".join(m["model"].split("/")[-1] for m in agent["models"]) or "—"
        lines.append(
            f"{agent['agent'][:26]:<26}{agent['llm']['calls']:>7}"
            f"{agent['llm']['total_tokens']:>12,}"
            f"{_money(agent['cost_usd']):>12}  {models}"
        )
        if agent["sandbox"]:
            box = agent["sandbox"]
            lines.append(
                f"{'  └ sandbox':<26}{box['runs']:>7}{box['total_tokens']:>12,}"
                f"{_money(box['total_cost_usd']):>12}  "
                f"{box['agent_seconds']:.0f}s agent, {box['gpu_seconds']:.0f}s GPU, "
                f"{box['energy_wh']:.1f} Wh"
            )
    lines.append("-" * width)
    lines.append(
        f"{'TOTAL':<26}{totals['llm_calls']:>7}{totals['total_tokens']:>12,}"
        f"{_money(totals['cost_usd']):>12}"
    )
    if sandbox["runs"]:
        lines.append(
            f"  sandbox: {sandbox['runs']} run(s), "
            f"{sandbox['cpu_core_seconds']:.0f} CPU-s, "
            f"{sandbox['gpu_seconds']:.0f} GPU-s, {sandbox['energy_wh']:.1f} Wh "
            f"(API {_money(sandbox['api_cost_usd'])} + "
            f"energy {_money(sandbox['energy_cost_usd'])})"
        )
    if not totals["complete"]:
        lines.append(
            f"  note: {llm['unpriced_calls']} call(s) ran on a model with no known "
            "price — the total is a floor. Price it with LLM_PRICE_OVERRIDES: "
            + ", ".join(llm["unpriced_models"])
        )
    lines.append("=" * width)
    return "\n".join(lines)


def log_report(key: SessionKey, *, title: str = "Usage & cost") -> dict:
    """Log the session's report at INFO and return the snapshot behind it."""
    data = LEDGER.snapshot(key)
    if data["agents"]:
        logger.info("\n%s", format_report(data, title=title))
    return data


__all__ = [
    "LEDGER",
    "SANDBOX_FIELDS",
    "AgentUsage",
    "ModelUsage",
    "UsageLedger",
    "UsageMetricsPlugin",
    "bind_session",
    "format_report",
    "log_report",
    "price_call",
    "price_first",
    "publish",
    "record_completion",
    "record_sandbox_run",
    "reset_session",
    "resolve_key",
    "set_metrics_sink",
    "snapshot",
    "sum_sandbox",
]
