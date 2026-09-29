"""A few lines on what one agent did, written on request by a small model.

The execution log shows an agent's task, its final report, its tool calls and
its artifacts — everything, which is more than a reader wants when the
question is only "what happened here". The panel offers a button; this module
turns the agent's trace into a short prompt, asks a cheap model, and remembers
the answer so a second click (or another reader) costs nothing.
"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TRACE_LIMIT = 60_000          # characters of trace handed to the model
_ARGS, _RESULT, _REPORT = 800, 1_500, 6_000
_MAX_TOKENS = 1_400
_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_CACHE_SIZE = 256

_LANGUAGE = {"ru": "Russian", "en": "English"}

_SYSTEM = (
    "You write, for a domain specialist, an account of what one agent did "
    "during its run inside a multi-agent research system. You are given the "
    "agent's task, the tool calls it made in order (with arguments and results, "
    "cut for length), the files and links it produced, and its final report.\n"
    "Write in {language} as a markdown document. Be as concrete and detailed as "
    "the trace allows — up to about 350 words: name the methods, models, "
    "parameters, datasets, queries, commands, metrics and numbers that appear "
    "in the trace, and say why each step was taken when the trace shows it. "
    "If the agent did little, say so in a few lines instead of padding.\n"
    "Structure:\n"
    "- first line: one bold sentence with the outcome of this run;\n"
    "- '### {h_task}': the task in the domain's own terms;\n"
    "- '### {h_done}': the steps, in order, with their parameters and the "
    "reasoning behind them;\n"
    "- '### {h_tools}': one bullet per tool used, what it was used for and "
    "what it returned;\n"
    "- '### {h_result}': the results — numbers, metrics, tables, and each file "
    "or link produced with what it contains;\n"
    "- '### {h_issues}': failures, retries, limits and anything left undone "
    "(omit the section if there were none);\n"
    "- '### {h_check}': two or three points a specialist should verify.\n"
    "State only what the trace shows; never invent numbers or files. Use "
    "inline code for tool names, file names, commands and parameters; keep "
    "file and link URLs exactly as they appear. No preamble, no closing line."
)

_HEADINGS = {
    "ru": {"h_task": "Задача", "h_done": "Что сделано", "h_tools": "Инструменты",
           "h_result": "Результаты", "h_issues": "Проблемы и ограничения",
           "h_check": "На что обратить внимание"},
    "en": {"h_task": "Task", "h_done": "What was done", "h_tools": "Tools",
           "h_result": "Results", "h_issues": "Issues and limits",
           "h_check": "What to check"},
}


def _cut(value: Any, limit: int) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text if len(text) <= limit else text[:limit] + "…"


def _secs(node: Dict[str, Any]) -> str:
    start, end = node.get("t_start"), node.get("t_end")
    if start is None or end is None:
        return ""
    s = end - start
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m {int(s % 60)}s"


def trace_of(node: Dict[str, Any]) -> str:
    """The agent's run as plain text a small model can read in one go."""
    name = node.get("executor_agent") or node.get("label") or "agent"
    lines: List[str] = [f"AGENT: {name}"]
    if node.get("runs", 1) > 1:
        lines.append(f"RUN: {node.get('run')} of {node.get('runs')} in this request")
    if node.get("status"):
        lines.append(f"STATUS: {node['status']}" + (f" ({_secs(node)})" if _secs(node) else ""))
    if node.get("input"):
        lines.append("TASK: " + _cut(node["input"], _REPORT))
    calls = node.get("calls") or []
    if calls:
        lines.append(f"TOOL CALLS ({len(calls)}):")
        for i, c in enumerate(calls, 1):
            head = f"{i}. {c.get('tool') or 'call'} [{c.get('status') or '?'}"
            if c.get("duration") is not None:
                head += f", {c['duration']:.1f}s"
            head += "]"
            lines.append(head)
            if c.get("input"):
                lines.append("   args: " + _cut(c["input"], _ARGS))
            if c.get("output"):
                lines.append("   result: " + _cut(c["output"], _RESULT))
            files = list(c.get("output_files") or [])
            if files:
                lines.append("   files: " + ", ".join(files[:6]))
    else:
        lines.append("TOOL CALLS: none")
    artifacts = [a.get("uri") for a in (node.get("artifacts") or []) if a.get("uri")]
    if artifacts:
        lines.append("ARTIFACTS: " + ", ".join(artifacts[:12]))
    if node.get("output"):
        lines.append("FINAL REPORT: " + _cut(node["output"], _REPORT))
    text = "\n".join(lines)
    if len(text) > _TRACE_LIMIT:
        text = text[:_TRACE_LIMIT] + "\n… [trace cut for length]"
    return text


def model_name() -> Tuple[Optional[str], Optional[str]]:
    """(model, api_base): the dedicated small model, else the one named in
    ``summary_url`` ("base;model"), else the main model."""
    from CoScientist.config import get_settings
    s = get_settings().llm
    if s.agent_summary_model:
        return s.agent_summary_model, s.main_url
    if s.summary_url and ";" in s.summary_url:
        base, model = s.summary_url.split(";", 1)
        return model.strip() or s.main_model, base.strip() or s.main_url
    return s.main_model, s.main_url


def _routable(model: str, base: Optional[str]) -> str:
    """The model name as litellm needs it.

    ``summary_url`` names its model the way the endpoint does
    (``google/gemini-2.0-flash-lite-001``), without the provider prefix
    litellm routes by, so litellm refuses it. When litellm cannot tell the
    provider from the name and there is a base URL, the endpoint is an
    OpenAI-compatible one and the call goes through as ``openai/<model>``.
    """
    import litellm
    try:
        litellm.get_llm_provider(model)
        return model
    except Exception:  # noqa: BLE001 — "provider not provided" is the case we handle
        return f"openai/{model}" if base else model


async def _complete(system: str, user: str) -> Tuple[str, str]:
    import litellm
    from CoScientist.config import get_settings
    s = get_settings().llm
    model, base = model_name()
    if not model:
        raise RuntimeError("no model configured: set LLM__AGENT_SUMMARY_MODEL or LLM__MAIN_MODEL")
    # The small model is a preference, not a requirement: a name the endpoint
    # no longer serves (models get retired under a config that still names
    # them) falls back to the main model rather than leaving the panel empty.
    attempts = [(model, base)]
    if s.main_model and s.main_model != model:
        attempts.append((s.main_model, s.main_url))
    resp, used = None, model
    for i, (candidate, candidate_base) in enumerate(attempts):
        try:
            from CoScientist.execution_control import before_model_attempt
            await before_model_attempt(
                "agent_summary",
                metadata={"model": candidate, "fallback": i > 0},
            )
            resp = await litellm.acompletion(
                model=_routable(candidate, candidate_base), api_base=candidate_base,
                api_key=s.openai_api_key,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0, timeout=s.request_timeout, max_tokens=_MAX_TOKENS,
                num_retries=0,
            )
            used = candidate
            break
        except Exception as exc:  # noqa: BLE001
            unknown = isinstance(exc, (litellm.NotFoundError, litellm.BadRequestError))
            if not unknown or i == len(attempts) - 1:
                raise
    model = used
    try:
        from CoScientist.logging.metrics import record_completion
        record_completion(resp, model=model, agent="AgentSummary")
    except Exception:  # noqa: BLE001 — accounting must not fail the answer
        pass
    return (resp.choices[0].message.content or "").strip(), model


def _key(scope: str, trace: str, lang: str) -> str:
    return hashlib.sha1(f"{scope}|{lang}|{trace}".encode("utf-8")).hexdigest()


def _stamp(trace: str) -> str:
    """A short digest of the trace an account was written from.

    Deliberately not `_key`: that one is namespaced by scope, which is right for
    an in-process cache shared by every session and wrong for a record that has
    to survive an export into a different session id.
    """
    from CoScientist.graph.summary_store import stamp

    return stamp(trace)


async def summarize(node: Dict[str, Any], *, lang: str = "ru",
                    scope: str = "", force: bool = False,
                    store: Any = None) -> Dict[str, Any]:
    """``{"summary", "model", "cached", "stamp"}`` for one agent node.

    Cached on the trace itself, so a node that has not changed is never sent
    twice and a running agent gets a fresh summary once it moved on; ``force``
    asks the model again regardless (the panel's "regenerate").

    Two layers of memory. The in-process LRU is the hot one. ``store`` — when a
    caller passes one — is the durable one: without it every restart threw away
    every account a study had paid a model to write, and an imported session
    bundle arrived with none. It is injected rather than reached for, so this
    module keeps no knowledge of where a session's files live.

    ``stamp`` is returned so a caller can tell a stored account apart from a
    current one instead of presenting a stale summary as fresh.
    """
    lang = lang if lang in _LANGUAGE else "ru"
    trace = trace_of(node)
    key = _key(scope, trace, lang)
    node_id = str(node.get("id") or "")
    trace_stamp = _stamp(trace)

    hit = None if force else _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return dict(hit, cached=True)

    if store is not None and not force:
        try:
            kept = store.get(node_id, lang, trace_stamp)
        except Exception:  # noqa: BLE001 — a saving must not fail the answer
            kept = None
        if kept and kept.get("summary"):
            entry = {"summary": kept["summary"], "model": kept.get("model", ""),
                     "cached": True, "stamp": trace_stamp}
            _CACHE[key] = dict(entry, cached=False)
            return entry

    system = _SYSTEM.format(language=_LANGUAGE[lang], **_HEADINGS[lang])
    summary, model = await _complete(system, trace)
    entry = {"summary": summary, "model": model, "cached": False,
             "stamp": trace_stamp}
    _CACHE[key] = entry
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    if store is not None:
        try:
            store.put(node_id, lang, trace_stamp, summary, model)
        except Exception:  # noqa: BLE001
            logger.debug("could not store the summary of %s", node_id, exc_info=True)
    return entry
