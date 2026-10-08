"""Drive one ADK agent turn robustly.

Bundles the cross-cutting concerns the pipeline shouldn't care about: the
loguru sinks, the ADK/LiteLLM compatibility patches (applied at import), the
failure-taxonomy classifier (F12), and the guarded single-turn runner
(``run_agent``) with guard-retries, loop breakers, a soft deadline, and
transient-provider-fault retries.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import traceback
import litellm
from loguru import logger
from google.adk.runners import Runner
from google.genai import types

from alembic import config
from alembic.events import emit, safe
from alembic.tools.fs import enable_read_dedup

# ── LLM/provider fault classification (shared with agents.ResilientLiteLlm) ────
# Request-shape errors a retry can never fix — everything else (5xx, rate limits,
# timeouts, connection drops, OpenRouter's transient 403 "Access denied by
# security policy") is a retryable provider fault.
_PERMANENT_LLM_ERRORS = tuple(
    c for c in (getattr(litellm, n, None) for n in (
        "BadRequestError", "ContextWindowExceededError",
        "ContentPolicyViolationError", "NotFoundError", "UnsupportedParamsError"))
    if isinstance(c, type))

_TRANSIENT_MARKERS = (
    "security policy", "access denied", "rate limit", "overloaded", "429",
    "temporarily unavailable", "service unavailable", "timed out", "timeout",
    "connection", "openrouterexception", "bad gateway", "gateway time",
    "internal server error", "502", "503", "504",
)


def _retryable_llm_error(e: Exception) -> bool:
    if _PERMANENT_LLM_ERRORS and isinstance(e, _PERMANENT_LLM_ERRORS):
        return False
    mod = type(e).__module__ or ""
    if mod.split(".", 1)[0] in ("litellm", "openai", "httpx", "httpcore"):
        return True
    return any(m in str(e).lower() for m in _TRANSIENT_MARKERS)


def _short_err(e: Exception) -> str:
    first = str(e).strip().splitlines()[0] if str(e).strip() else ""
    return f"{type(e).__name__}: {first[:200]}"

# ── loguru terminal sink ──────────────────────────────────────────────────────
# backtrace/diagnose OFF: a provider fault otherwise printed a screen-high
# annotated traceback (`└ <Response [403 Forbidden]>` …). Real bugs still get a
# plain one-frame traceback, just not the variable-value wall.
logger.remove()
logger.add(sys.stderr,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
           level="DEBUG", colorize=True, backtrace=False, diagnose=False)

# ── Patch ADK tool lookup: hallucinated tool name → error stub, not a crash (F19) ─
import google.adk.flows.llm_flows.functions as _adk_fns

_original_get_tool = _adk_fns._get_tool


class _UnknownToolStub:
    """Feeds the error back to the LLM instead of killing the run. Mirrors
    BaseTool's full public attribute surface so no ADK code path AttributeErrors."""
    def __init__(self, called_name: str, available: list):
        self.name = called_name
        self.description = f"Unknown tool stub for '{called_name}'."
        self.is_long_running = False
        self.custom_metadata = None
        self._msg = (f"Tool '{called_name}' does not exist. Use one of these exact "
                     f"names: {', '.join(sorted(available))}. Retry with the correct name.")

    async def run_async(self, *, args=None, tool_context=None, **_):
        return {"error": self._msg}


def _safe_get_tool(function_call, tools_dict):
    try:
        return _original_get_tool(function_call, tools_dict)
    except (ValueError, KeyError):
        return _UnknownToolStub(function_call.name, list(tools_dict.keys()))


_adk_fns._get_tool = _safe_get_tool

# ── Detect silent LiteLLM/provider faults (F22) ───────────────────────────────
# OpenRouter can surface an internal fault as finish_reason "error", which
# LiteLLM maps to "stop" with only a warning — no exception. Hook its logger
# and flag the context so the stage loop retries the turn.
_transient_provider_fault: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "transient_provider_fault", default=False)


class _LiteLLMFaultDetector(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        if "Unmapped finish_reason 'error'" in msg:
            _transient_provider_fault.set(True)


logging.getLogger("LiteLLM").addHandler(_LiteLLMFaultDetector())

# ── Failure taxonomy (F12) ────────────────────────────────────────────────────
# Classify a tool-failure's error text into a fixed bucket. Generated server.py
# wraps the real exception in a RuntimeError, whose name appears BEFORE the
# wrapped traceback's real exception — so "last needle wins" recovers the root.
_ERROR_TAXONOMY = [
    ("ModuleNotFoundError", "ModuleNotFound"), ("ImportError", "Import"),
    ("FileNotFoundError", "FileNotFound"), ("AttributeError", "AttributeError"),
    ("KeyError", "KeyError"), ("IndexError", "IndexError"), ("NameError", "NameError"),
    ("TypeError", "TypeError"), ("ValueError", "ValueError"),
    ("UnicodeDecodeError", "Encoding"), ("IndentationError", "Syntax"),
    ("SyntaxError", "Syntax"), ("TimeoutExpired", "Timeout"), ("timed out", "Timeout"),
    ("No matching distribution", "Environment"), ("Could not find a version", "Environment"),
    ("error: subprocess-exited-with-error", "Environment"), ("Failed building wheel", "Environment"),
    ("CalledProcessError", "Runtime"), ("non-zero exit status", "Runtime"),
]


def classify_error(text: str) -> str:
    if not text:
        return "Unknown"
    best_label, best_pos = None, -1
    for needle, label in _ERROR_TAXONOMY:
        pos = text.rfind(needle)
        if pos > best_pos:
            best_pos, best_label = pos, label
    return best_label or "Other"


def _trunc(text: str, n: int = 2000) -> str:
    text = str(text).replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def _price_usage(usage) -> float:
    """$ for one model call, or 0.0 if litellm has no price for config.MODEL.

    Self-contained on purpose: alembic ships into its own Docker image without
    the rest of the CoScientist package (see config.py's import note), so this
    cannot reuse CoScientist.logging.metrics — it prices straight off litellm,
    already a pipeline dependency.
    """
    try:
        prompt = getattr(usage, "prompt_token_count", 0) or 0
        completion = getattr(usage, "candidates_token_count", 0) or 0
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=config.MODEL, prompt_tokens=prompt, completion_tokens=completion,
        )
        return float(prompt_cost) + float(completion_cost)
    except Exception:  # noqa: BLE001 - unmapped model must not break a run
        return 0.0


def _log_event(agent_name: str, event) -> None:
    if not event.content or not event.content.parts:
        return
    for part in event.content.parts:
        if part.text:
            snippet = _trunc(part.text.strip())
            if event.is_final_response():
                logger.info(f"[{agent_name}] FINAL: {snippet}")
            else:
                logger.debug(f"[{agent_name}] text:  {snippet}")
        elif getattr(part, "function_call", None):
            logger.debug(f"[{agent_name}] CALL  {part.function_call.name}({_trunc(str(part.function_call.args))})")
        elif getattr(part, "function_response", None):
            logger.debug(f"[{agent_name}] RESP  {part.function_response.name} → {_trunc(str(part.function_response.response))}")


_VALIDATION_TOOLS = {"invoke_tool_function", "run_tool_tests"}


def _tool_outcome(name: str, response) -> tuple[bool | None, str]:
    if not isinstance(response, dict):
        return None, ""
    if name == "invoke_tool_function":
        return response.get("ok"), f"{response.get('error','')}\n{response.get('traceback','')}"
    if name == "run_tool_tests":
        return not response.get("failures"), str(response.get("failures") or response.get("error") or "")
    return response.get("passed"), str(response.get("error") or response.get("output") or "")


async def _run_agent_once(agent, runner, session_id, message, required_report,
                          deadline=None, progress=None):
    """Run one invocation. Returns (final, wrote_report, steps, tokens, cost_usd,
    transient_fault, tool_calls, failures_by_class, abort_reason)."""
    _transient_provider_fault.set(False)
    enable_read_dedup(agent.name == "explorer")
    content = types.Content(role="user", parts=[types.Part(text=message)])
    final, wrote_report, step, total_tokens, total_cost = "Agent did not produce a final response.", False, 0, 0, 0.0
    last_call, tool_repeats = None, 0
    tool_calls: dict[str, int] = {}
    call_key_counts: dict[tuple, int] = {}
    failures_by_class: dict[str, int] = {}
    if progress is not None:
        progress["tool_calls"] = tool_calls

    def _fault():
        return _transient_provider_fault.get()

    try:
        async for event in runner.run_async(user_id=config.USER_ID, session_id=session_id, new_message=content):
            step += 1
            _log_event(agent.name, event)
            usage = getattr(event, "usage_metadata", None)
            if usage:
                total_tokens += getattr(usage, "total_token_count", 0) or 0
                total_cost += _price_usage(usage)

            if event.content:
                for part in event.content.parts:
                    if getattr(part, "function_call", None):
                        fc = part.function_call
                        await emit({"type": "tool_call", "stage": agent.name,
                                    "name": fc.name,
                                    "args": safe(dict(fc.args) if fc.args else {})})
                        tool_calls[fc.name] = tool_calls.get(fc.name, 0) + 1
                        call_key = (fc.name, str(fc.args))
                        tool_repeats = tool_repeats + 1 if call_key == last_call else 1
                        last_call = call_key
                        if tool_repeats >= config.MAX_TOOL_REPEATS:
                            logger.warning(f"[{agent.name}] ABORT: {fc.name} called {tool_repeats}x identical — break.")
                            return (final, wrote_report, step, total_tokens, total_cost, _fault(), tool_calls, failures_by_class, "tool_repeat")
                        if fc.name not in config.TOOL_CYCLE_EXEMPT:
                            call_key_counts[call_key] = call_key_counts.get(call_key, 0) + 1
                            if call_key_counts[call_key] >= config.MAX_TOOL_CYCLE:
                                logger.warning(f"[{agent.name}] ABORT: {fc.name} called {call_key_counts[call_key]}x (cycle) — break.")
                                return (final, wrote_report, step, total_tokens, total_cost, _fault(), tool_calls, failures_by_class, "tool_cycle")

                    fr = getattr(part, "function_response", None)
                    if fr:
                        await emit({"type": "tool_result", "stage": agent.name,
                                    "name": fr.name, "response": safe(fr.response)})
                    if fr and fr.name == "write_report" and required_report:
                        if required_report in str((fr.response or {}).get("report_path", "")):
                            wrote_report = True
                    elif fr and fr.name in _VALIDATION_TOOLS:
                        ok, err_text = _tool_outcome(fr.name, fr.response)
                        if ok is False:
                            failures_by_class[classify_error(err_text)] = failures_by_class.get(classify_error(err_text), 0) + 1
                            if progress is not None:
                                progress["last_failure"] = f"{fr.name}: {_trunc(err_text)}"

            if step >= config.MAX_STEPS:
                logger.warning(f"[{agent.name}] ABORT: reached {config.MAX_STEPS} steps.")
                return (final, wrote_report, step, total_tokens, total_cost, _fault(), tool_calls, failures_by_class, "max_steps")

            if deadline is not None and time.monotonic() >= deadline:
                logger.warning(f"[{agent.name}] ABORT: soft deadline — break for a final write_report nudge.")
                return (final, wrote_report, step, total_tokens, total_cost, _fault(), tool_calls, failures_by_class, "soft_deadline")

            if event.is_final_response():
                if event.content and event.content.parts:
                    final = event.content.parts[0].text or final
                elif event.actions and event.actions.escalate:
                    final = f"Agent escalated: {event.error_message or 'No message.'}"
                break

    except json.JSONDecodeError as e:
        logger.warning(f"[{agent.name}] invalid JSON in tool call (char {e.pos}): {e.msg}")
    except Exception as e:
        # ResilientLiteLlm retries transient provider faults; anything that still
        # reaches here (a non-retryable API error, or a tool call that raised deep
        # in ADK) is logged as ONE line — exception + originating frame — never the
        # multi-frame async wall. Stage resets/debugger rounds handle recovery.
        tb = traceback.extract_tb(e.__traceback__)
        where = f"{tb[-1].filename.split('/')[-1]}:{tb[-1].lineno}" if tb else "?"
        logger.warning(f"[{agent.name}] event-loop error: {_short_err(e)}  (raised at {where})")

    return final, wrote_report, step, total_tokens, total_cost, _fault(), tool_calls, failures_by_class, None


async def run_agent(agent, session_service, session_id, message,
                    required_report=None, deadline=None, progress=None):
    """Run an agent, retrying if the write_report guard isn't satisfied.

    Returns (final_text, total_steps, total_tokens, stage_metrics) where
    stage_metrics = {tool_calls, failures_by_class, guard_retries,
    transient_fault_retries, abort_reason, cost_usd}."""
    runner = Runner(agent=agent, app_name=config.APP_NAME, session_service=session_service)
    final = "Agent did not produce a final response."
    richest_final = ""   # longest real response across attempts — when the agent
    # narrates its report instead of calling write_report, the first (pre-nag)
    # answer holds the full report; later nag-replies are terse. Salvage wants
    # the richest, not the last. (Used only for logging + report salvage.)
    total_steps = total_tokens = guard_retries = transient_fault_retries = 0
    total_cost = 0.0
    tool_calls: dict[str, int] = {}
    failures_by_class: dict[str, int] = {}
    abort_reason = None
    current_message, current_deadline = message, deadline

    for attempt in range(config.MAX_GUARD_RETRIES + 1):
        fault_retries = 0
        while True:
            (final, wrote_report, steps, tokens, cost, transient_fault,
             call_counts, fail_counts, this_abort) = await _run_agent_once(
                agent, runner, session_id, current_message, required_report,
                deadline=current_deadline, progress=progress)
            total_steps += steps
            total_tokens += tokens
            total_cost += cost
            for k, v in call_counts.items():
                tool_calls[k] = tool_calls.get(k, 0) + v
            for k, v in fail_counts.items():
                failures_by_class[k] = failures_by_class.get(k, 0) + v
            abort_reason = this_abort
            if final and final != "Agent did not produce a final response." \
                    and len(final) > len(richest_final):
                richest_final = final
            if not transient_fault or fault_retries >= config.MAX_TRANSIENT_FAULT_RETRIES:
                break
            fault_retries += 1
            transient_fault_retries += 1
            logger.warning(f"[{agent.name}] transient provider fault — retry {fault_retries}/{config.MAX_TRANSIENT_FAULT_RETRIES} (off-budget).")

        nudges = []
        if required_report and not wrote_report:
            if this_abort == "soft_deadline":
                nudges.append(f"You are almost out of time. Call write_report with "
                              f"report_name='{required_report}' IMMEDIATELY — a partial or FAILED "
                              f"report is acceptable. Do not call any other tool first.")
                current_deadline = None
            else:
                nudges.append(f"You have not called write_report with report_name='{required_report}'. "
                              f"Call write_report now to save your findings.")

        if not nudges:
            break
        if attempt >= config.MAX_GUARD_RETRIES:
            logger.warning(f"[guard] Max retries ({config.MAX_GUARD_RETRIES}) reached — giving up.")
            abort_reason = "guard_exhausted"
            break
        guard_retries += 1
        current_message = "IMPORTANT: " + " ".join(nudges)
        logger.warning(f"[guard] Retry {attempt + 1}/{config.MAX_GUARD_RETRIES}: {current_message[:120]}")

    return (richest_final or final), total_steps, total_tokens, {
        "tool_calls": tool_calls, "failures_by_class": failures_by_class,
        "guard_retries": guard_retries, "transient_fault_retries": transient_fault_retries,
        "abort_reason": abort_reason, "cost_usd": total_cost,
    }
