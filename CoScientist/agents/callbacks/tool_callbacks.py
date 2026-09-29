import json
import os
import re
from difflib import SequenceMatcher, get_close_matches

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import logging
logger = logging.getLogger(__name__)

# ── Executor tool-match thresholds (the Coder↔Executor redirect mechanism) ───
# A retrieved tool counts as a real match only at/above _KEEP. When NOTHING
# clears _KEEP we look at the single best score:
#   * best >= _ABSTAIN  -> marginal salvage: take top-2 and proceed (cautious).
#   * best <  _ABSTAIN  -> ABSTAIN: leave the tool set empty and flag a no-match,
#                          so ExperimentAgent redirects to CoderAgent instead of
#                          "solving" the task with an unrelated tool (e.g. running
#                          a GAN trainer for a "train a transformer" task).


# State key carrying the executor's tool-match verdict for the redirect guard.
TOOL_MATCH_STATE_KEY = "executor_tool_match"

RERANK_SCORED = "scored"                  # the model ranked the candidates
RERANK_RECOVERED = "recovered_local"      # ranked by the local cross-encoder instead
RERANK_PARSE_FAILED = "parse_failed"      # output_key payload unreadable
RERANK_EMPTY_RANKING = "empty_ranking"    # readable, but not one usable {index, score}
RERANK_NO_CANDIDATES = "no_candidates"    # retrieval accumulated nothing to rank

UNJUDGED_REASONS = frozenset({RERANK_PARSE_FAILED, RERANK_EMPTY_RANKING})


class RerankParseError(ValueError):
    """A reranker's ``output_key`` payload could not be read as a JSON object.

    Deliberately distinct from an empty ranking: see UNJUDGED_REASONS above.
    """


def _output_key_json(value: Any) -> Dict[str, Any]:
    """Coerce a reranker's ``output_key`` payload to a dict, or raise.

    With ``output_schema`` set, ADK validates the model's text and stores a dict.
    We deliberately drop the schema on the rerankers so ADK never sends OpenRouter
    a *strict* json_schema ``response_format`` (some providers stall on the
    grammar-constrained decode it implies), which leaves ``output_key`` holding
    raw text — already reduced to bare JSON by ``sanitize_json_output``.

    Raises:
        RerankParseError: the payload is missing, empty, not JSON, or not an object.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError as exc:
            raise RerankParseError(f"not valid JSON: {value[:300]!r}") from exc
        if isinstance(parsed, dict):
            return parsed
        raise RerankParseError(
            f"got a JSON {type(parsed).__name__}, expected an object"
        )
    raise RerankParseError(f"empty or missing payload ({type(value).__name__})")


def _score_map(entries: Any, *, cast: Callable[[Any], Any]) -> Dict[int, Any]:
    """Build ``{index: score}``, skipping anything malformed.

    Without ``output_schema`` nothing validates the model's shape any more, so a
    missing/!int ``index`` or a non-numeric ``score`` must degrade to "this tool
    went unscored" rather than raise inside an after_agent callback.
    """
    out: Dict[int, Any] = {}
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        try:
            out[index] = cast(entry.get("score"))
        except (TypeError, ValueError):
            continue
    return out

# ── ToolReranker shortlist (local cross-encoder pre-pass) ────────────────────
# Every entry in `accumulated_tools` carries the score of the retrieval query
# that FOUND it, and those queries differ per tool — so the scores share no
# scale. (Observed: a tool whose query happened to contain its own name scored
# 0.74 while the one the task actually needed scored 0.38 and sat 12th.)
# Truncating on them drops the right tool, so we re-score every candidate
# against ONE query — the task — with the local reranker service, and shortlist
# on that. Putting candidates on a common scale is exactly the job the LLM
# reranker does next; this just narrows what it has to read.
_RERANK_SHORTLIST_KEY = "reranker_candidates"
_RERANK_SHORTLIST_SIZE = int(os.getenv("RERANK_SHORTLIST_SIZE", "8"))
_SHORTLIST_SCORES_KEY = "shortlist_scores"
_EXPLICIT_TARGET_KEY = "explicit_tool_target"


def _first_text(content) -> str:
    """First text part of a Content, or '' if there is none."""
    if content is None or not getattr(content, "parts", None):
        return ""
    for part in content.parts:
        if getattr(part, "text", None):
            return part.text
    return ""


async def shortlist_reranker_tools(callback_context: CallbackContext) -> None:
    """Narrow `accumulated_tools` to the top-K by local cross-encoder score.

    Also records every candidate's score in ``state['shortlist_scores']`` — the
    LLM reranker that runs next can come back unreadable, and those scores are
    then the cheapest way to still rank the candidates (see
    ``after_tool_reranker_agent``).

    Best-effort by construction: every early return leaves the full tool list in
    place, so an unreachable reranker service costs context, never correctness.
    """
    state = callback_context.state
    acc: List[Dict[str, Any]] = state.get('accumulated_tools') or []
    # Set first: the prompt reads this key, so it must be populated on every path.
    state[_RERANK_SHORTLIST_KEY] = acc
    state[_SHORTLIST_SCORES_KEY] = {}
    task = _first_text(getattr(callback_context, "user_content", None)).strip()
    target = re.search(r"\bTarget tool:\s*([A-Za-z_][A-Za-z_0-9]*)\b", task)
    state[_EXPLICIT_TARGET_KEY] = target.group(1) if target else None

    # Scored even when the list is short enough to need no truncation: the scores
    # themselves are the point now, not just the ordering. Costs one extra call
    # to the local cross-encoder on the common path, and buys a deterministic
    # recovery for every reranker failure below the shortlist threshold.
    if not acc:
        return None

    if not task:
        logger.info(
            "shortlist: no task text on this invocation — passing all %d tools", len(acc)
        )
        return None

    documents = [
        f"{tool.get('tool', '')}: {tool.get('description', '')}" for tool in acc
    ]
    try:
        from rag_tools.retrieval import APIReranker

        ranked = await APIReranker().rerank_with_scores(
            task, documents, top_k=len(documents)
        )
    except Exception as exc:  # noqa: BLE001 — never fail the run over a shortlist
        logger.warning(
            "shortlist: reranker unavailable (%r) — passing all %d tools", exc, len(acc)
        )
        return None

    # APIReranker swallows its own HTTP errors and answers with all-zero scores.
    # That ordering is just the input order, so shortlisting on it would silently
    # cut arbitrary tools — treat it as "service down" instead.
    if not ranked or max(score for _, score in ranked) <= 0.0:
        logger.warning(
            "shortlist: no usable scores from the reranker — passing all %d tools", len(acc)
        )
        return None

    state[_SHORTLIST_SCORES_KEY] = {
        acc[i]["tool_index"]: float(score)
        for i, score in ranked
        if 0 <= i < len(acc) and isinstance(acc[i].get("tool_index"), int)
    }

    if len(acc) <= _RERANK_SHORTLIST_SIZE:
        return None

    shortlist = [acc[i] for i, _ in ranked[:_RERANK_SHORTLIST_SIZE] if 0 <= i < len(acc)]
    if target:
        shortlist.extend(
            item for item in acc
            if item.get("tool") == target.group(1) and item not in shortlist
        )
    if not shortlist:
        return None

    logger.info(
        "shortlist: %d -> %d tools for the reranker (best=%s)",
        len(acc), len(shortlist), round(ranked[0][1], 3),
    )
    state[_RERANK_SHORTLIST_KEY] = shortlist
    return None


# Rendered into FedotAgent's prompt via `{fedot_candidates?}`.
_FEDOT_CANDIDATES_KEY = "fedot_candidates"


def inject_fedot_candidates(callback_context: CallbackContext) -> None:
    """before_agent for FedotAgent: show it the tools `fedot_tool` will get.

    Same expression the tool itself uses — the reranker's pick when there is one,
    the unfiltered candidate pool when the reranker's answer was unreadable — so
    the prompt and the actual MCP server set can never disagree about what is on
    the table.
    """
    state = callback_context.state
    state[_FEDOT_CANDIDATES_KEY] = (
        state.get('filtered_tools') or state.get('accumulated_tools') or []
    )
    return None


def announce_attached_tools(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Tell the executor which tools it holds, right before it decides.

    The catalogue search earlier in the conversation does not see locally served
    servers, and the executor echoed its "nothing matched" as ``NO_MATCHING_TOOL``.
    """
    names = sorted(n for n in (llm_request.tools_dict or {}))
    if not names:
        return
    note = (
        "TOOLS ATTACHED TO YOU RIGHT NOW: " + ", ".join(names) + ".\n"
        "This list is the ground truth for the NO_MATCHING_TOOL decision. Any "
        "earlier statement in this conversation that a tool 'does not exist' or "
        "'is not registered' came from a catalogue search, which does not see "
        "servers deployed locally for this task. If a tool the task names is in "
        "the list above, CALL IT instead of answering NO_MATCHING_TOOL."
    )
    # As a trailing user turn: a system-instruction edit does not survive every
    # model wrapper, and this decision must not be made on stale information.
    llm_request.contents.append(
        types.Content(role="user", parts=[types.Part(text=note)])
    )
    logger.debug("ATTACHED_TOOLS %s", names)


def before_tool_reranker_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Drop ToolRetriever/ToolReranker dumps from the next LLM request.

    ADK often prefixes sibling output as ``For context:[Agent] …`` (one part),
    so an exact ``== 'For context:'`` match never fired and the planner saw the
    full retrieve_tools novels — then invented tools absent from inventory.
    """
    kept: List[Any] = []
    for content in llm_request.contents or []:
        parts = list(getattr(content, "parts", None) or [])
        blob = "\n".join(str(getattr(p, "text", None) or "") for p in parts).lstrip()
        if blob.startswith("For context:"):
            continue
        if "[ToolRetrieverAgent]" in blob or "[ToolReranker]" in blob:
            continue
        kept.append(content)
    llm_request.contents = kept


# Set when ToolReranker scores were applied from after_model (skip after_agent).
_TOOL_RERANK_APPLIED_KEY = "_tool_rerank_applied"


def _score_items_from_reranked_state(raw: Any) -> List[Dict[str, Any]]:
    """Normalize ``reranked_tools`` state (dict / model / list) to score dicts."""
    if raw is None:
        return []
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if isinstance(raw, dict):
        tools = raw.get("tools") or []
    elif isinstance(raw, list):
        tools = raw
    else:
        return []
    out: List[Dict[str, Any]] = []
    for t in tools:
        if hasattr(t, "model_dump"):
            t = t.model_dump()
        if not isinstance(t, dict):
            continue
        try:
            out.append({"index": int(t["index"]), "score": float(t["score"])})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _llm_response_text(llm_response: LlmResponse, *, include_thoughts: bool) -> str:
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return ""
    chunks: List[str] = []
    for p in parts:
        text = getattr(p, "text", None)
        if not text:
            continue
        if not include_thoughts and getattr(p, "thought", False):
            continue
        chunks.append(text)
    return "".join(chunks)


def _score_items_from_llm_response(llm_response: LlmResponse) -> Optional[List[Dict[str, Any]]]:
    """Parse ToolRanking scores from the model response (not from output_key state).

    Prefer non-thought text (post-sanitize path); fall back to thoughts — GLM often
    parks the JSON ranking in a thought part while the logger shows the plain text empty.
    """
    from CoScientist.agents.callbacks.json_output import _extract_json, _normalize_ranking_payload

    for include_thoughts in (False, True):
        text = _llm_response_text(llm_response, include_thoughts=include_thoughts)
        if not text.strip():
            continue
        extracted = _extract_json(text)
        if extracted is None:
            continue
        items = _score_items_from_reranked_state(_normalize_ranking_payload(extracted))
        if items:
            return items
    return None


def _tool_rank_key(tool: Dict[str, Any]) -> int:
    # Prefer explicit tool_index; fall back to 1-based list position.
    raw = tool.get("tool_index", tool.get("index"))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def apply_tool_rerank_scores(
    state: Any,
    score_items: List[Dict[str, Any]],
    *,
    reason: str = RERANK_SCORED,
) -> None:
    """Filter ``accumulated_tools`` by rerank scores; set match verdict + filtered_tools.

    Records WHY the tool set came out as it did (see the RERANK_* constants), so
    the guards downstream can tell "judged, nothing relevant" (abstain to
    CoderAgent) apart from "we could not read the answer" (recover locally, or
    hand the unfiltered candidates to the FEDOT.MAS fallback).
    """
    from CoScientist.config import get_settings

    web_settings = get_settings().web
    keep_score = web_settings.executor_tool_keep_score
    abstain_score = web_settings.executor_tool_abstain_score
    rerank_map: Dict[int, float] = {int(t["index"]): float(t["score"]) for t in score_items}
    acc_tools: List[Dict[str, Any]] = list(state.get("accumulated_tools") or [])

    filtered_tools: List[Dict[str, Any]] = [
        tool for tool in acc_tools
        if rerank_map.get(_tool_rank_key(tool), 0) >= keep_score
    ]
    # Some models emit 0-based indices while tool_index is 1-based (or vice versa).
    if not filtered_tools and rerank_map and acc_tools:
        shifted = {
            tool for tool in acc_tools
            if rerank_map.get(_tool_rank_key(tool) - 1, 0) >= keep_score
            or rerank_map.get(_tool_rank_key(tool) + 1, 0) >= keep_score
        }
        if shifted:
            filtered_tools = list(shifted)

    best_score = max(rerank_map.values(), default=0.0)
    matched = bool(filtered_tools)

    if not filtered_tools and best_score >= abstain_score:
        # Marginal salvage: nothing cleared _KEEP but the best is not hopeless —
        # take top-2 and proceed cautiously (preserves the old behaviour here).
        top_ids = {
            idx for idx, _ in sorted(
                rerank_map.items(), key=lambda x: x[1], reverse=True
            )[:2]
        }
        filtered_tools = [t for t in acc_tools if _tool_rank_key(t) in top_ids]
        if not filtered_tools:
            filtered_tools = [
                t for t in acc_tools
                if (_tool_rank_key(t) - 1) in top_ids or (_tool_rank_key(t) + 1) in top_ids
            ]
        matched = bool(filtered_tools)
    target = state.get(_EXPLICIT_TARGET_KEY)
    if target:
        pinned = next((item for item in acc_tools if item.get("tool") == target), None)
        if pinned is None:
            raise RuntimeError(f"Explicit target tool {target} was not retrieved")
        if pinned not in filtered_tools:
            filtered_tools.append(pinned)
        matched = True
    # else (best < _ABSTAIN): ABSTAIN — leave filtered_tools empty so the
    # redirect guard on ExperimentAgent sends the task to CoderAgent instead of
    # running an unrelated tool. Only for a reason that actually judged them.

    # Record the verdict for the redirect guard / the orchestrator's critic.
    state[TOOL_MATCH_STATE_KEY] = {
        "matched": matched,
        "best_score": round(best_score, 3),
        "kept": len(filtered_tools),
        "reason": reason,
        "candidates": len(acc_tools),
    }
    state["filtered_tools"] = filtered_tools

    if reason in UNJUDGED_REASONS:
        # Keep the candidate pool: FedotAgent builds its MCP server set out of
        # `accumulated_tools`, and clearing it here is what would make an
        # unreadable reranker answer unrecoverable.
        logger.warning(
            "reranker verdict=%s — keeping %d candidate(s) for the FEDOT.MAS fallback",
            reason, len(acc_tools),
        )
        return

    state["accumulated_tools"] = []
    state[_RERANK_SHORTLIST_KEY] = []
    state[_SHORTLIST_SCORES_KEY] = {}
    state["retrieval_queries"] = []
    state[_TOOL_RERANK_APPLIED_KEY] = True
    # Drop process-global buffer so the next discovery pass starts clean.
    try:
        from CoScientist.tools.retrieval_tools import clear_session_accumulated_tools

        clear_session_accumulated_tools()
    except Exception:  # noqa: BLE001
        pass


def after_tool_reranker_model(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """after_model: apply ToolReranker scores from the response body.

    ``output_key`` is often invisible in ``after_agent`` (ADK state-delta timing),
    which produced false ``best_score=0.0`` / empty ``filtered_tools``. Reading the
    ranking JSON here (after ``sanitize_json_output``) avoids that race.
    """
    if any(
        getattr(p, "function_call", None)
        for p in (getattr(getattr(llm_response, "content", None), "parts", None) or [])
    ):
        return None
    items = _score_items_from_llm_response(llm_response)
    if not items:
        logger.warning(
            "[%s] after_model tool rerank: no parseable scores in response",
            _agent_name(callback_context),
        )
        return None
    apply_tool_rerank_scores(callback_context.state, items, reason=RERANK_SCORED)
    return None


def after_tool_reranker_agent(
    callback_context: CallbackContext
) -> None:
    """Turn ToolReranker's output into ``filtered_tools`` + a reasoned verdict.

    Preferred path: ``sanitize_json_output`` / ``after_tool_reranker_model``
    apply ToolRanking scores when the JSON is parsed — avoids ADK output_key
    timing races. This after_agent hook is the upstream-compatible fallback
    and also recovers from an unreadable ranking via the local cross-encoder.
    """
    current_state = callback_context.state
    if current_state.get(_TOOL_RERANK_APPLIED_KEY):
        return None

    acc_tools: List[Dict[str, Any]] = current_state.get("accumulated_tools") or []
    rerank_map: Dict[int, float] = {}
    reason = RERANK_SCORED
    try:
        payload = _output_key_json(current_state.get("reranked_tools"))
    except RerankParseError as exc:
        logger.warning("reranker output unusable — %s", exc)
        reason = RERANK_PARSE_FAILED
    else:
        rerank_map = _score_map(payload.get("tools"), cast=float)
        if not rerank_map:
            # HEAD's looser parser (model_dump / list payloads).
            score_items = _score_items_from_reranked_state(current_state.get("reranked_tools"))
            if score_items:
                rerank_map = {int(t["index"]): float(t["score"]) for t in score_items}
            elif acc_tools:
                logger.warning(
                    "reranker returned no usable {index, score} pairs for %d candidate(s)",
                    len(acc_tools),
                )
                reason = RERANK_EMPTY_RANKING

    if not acc_tools:
        reason = RERANK_NO_CANDIDATES

    if reason in UNJUDGED_REASONS:
        recovered = {
            idx: score
            for idx, score in (current_state.get(_SHORTLIST_SCORES_KEY) or {}).items()
            if isinstance(idx, int) and isinstance(score, (int, float))
        }
        if recovered:
            rerank_map = {int(k): float(v) for k, v in recovered.items()}
            reason = RERANK_RECOVERED
            logger.info(
                "reranker unusable — ranked %d candidate(s) by local cross-encoder score",
                len(rerank_map),
            )

    score_items = [{"index": idx, "score": score} for idx, score in rerank_map.items()]
    apply_tool_rerank_scores(current_state, score_items, reason=reason)
    return None


def after_fullset_reranker_agent(
    callback_context: CallbackContext
) -> None:
    """Turn FullSetToolReranker's output into ``filtered_mcps``.

    Parsed the same defensive way as ToolReranker's: this used to lean on
    ``output_schema`` for its shape and index straight into the payload, so a
    string, a missing ``index`` or a non-list ``mcp_scores`` raised
    AttributeError/KeyError/TypeError out of an after_agent callback and killed
    the whole run. Deploying no web MCP is a fine outcome; crashing is not.
    """
    current_state = callback_context.state

    try:
        payload = _output_key_json(current_state.get('reranked_web_servers'))
        scores = payload.get('mcp_scores')
    except RerankParseError as exc:
        logger.warning("web-MCP reranker output unusable — %s; deploying none", exc)
        scores = None

    # Binary deploy score per MCP index — truthiness selects deploy. A model that
    # answers with the 0.0-1.0 floats the sibling reranker asks for still works:
    # bool() of a non-zero float is True.
    rerank_map: Dict[int, bool] = {
        idx: bool(score) for idx, score in _score_map(scores, cast=bool).items()
    }
    acc_mcps: List[Dict[str, Any]] = current_state.get('accumulated_web_mcps') or []

    filtered_mcps: List[Dict[str, Any]] = [
        mcp for mcp in acc_mcps
        if rerank_map.get(mcp.get('index', -1), False)
    ]

    callback_context.state['filtered_mcps'] = filtered_mcps
    callback_context.state['accumulated_web_mcps'] = []
    callback_context.state['retrieval_queries_mcp'] = []
    return

def before_get_task(callback_context: CallbackContext):
    """Ensure session has a task list and sanitize active_tasks for the target agent before it runs."""
    master = callback_context.state.get("_master_active_tasks")
    active = callback_context.state.get("active_tasks")

    if master is None and active is None:
        callback_context.state["active_tasks"] = []
        callback_context.state["_master_active_tasks"] = []
        return None

    if master is None:
        master = list(active) if isinstance(active, list) else []
        callback_context.state["_master_active_tasks"] = master

    current_agent = getattr(callback_context, "agent_name", None)
    from CoScientist.tools.task_tracker import clean_tasks_for_agent
    callback_context.state["active_tasks"] = clean_tasks_for_agent(master, current_agent)
    return None


def inject_graph_root(callback_context: CallbackContext):
    """Give the agent the session graph root: every agent, its capabilities and
    this session's trace, rendered via the {graph_root?} placeholder.

    Best-effort — the graph must never break a run. Yields nothing when the
    execution graph is switched off, so the placeholder stays empty instead of
    describing a feature the agent no longer has tools for.

    What earlier sessions established is no longer injected here. It is served by
    the research graph's cross-run index, which a run consults deliberately
    rather than receiving as ambient context.
    """
    try:
        from CoScientist.config import get_settings
        if not get_settings().web.knowledge_graph_enabled:
            callback_context.state['graph_root'] = ""
            return None
    except Exception:  # noqa: BLE001
        pass

    summary = ""
    try:
        from CoScientist.graph.memory import get_knowledge_graph
        summary = get_knowledge_graph(callback_context).root_summary()
    except Exception:  # noqa: BLE001
        pass
    callback_context.state['graph_root'] = summary
    return None


# ── Dataset archive attached by the user in the web UI ────────────────────────
# The link is set on the session (web/app.py) and surfaces in the agent's
# instructions, so the agent KNOWS about the archive and passes it as
# `dataset_url` when the work it is doing actually needs that data. Nothing
# fills the argument in for it — sending the data is the agent's own decision.
DATASET_URL_STATE_KEY = "dataset_url"
DATASET_CONTEXT_STATE_KEY = "dataset_context"


def inject_dataset_context(callback_context: CallbackContext):
    """before_agent: render state['dataset_url'] into the prompt's dataset block.

    The instruction carries ``{dataset_context?}`` rather than the raw URL, so a
    session with no attached archive gets nothing at all instead of a heading
    describing data that does not exist.
    """
    url = str(callback_context.state.get(DATASET_URL_STATE_KEY) or "").strip()
    callback_context.state[DATASET_CONTEXT_STATE_KEY] = (
        "## Dataset attached to this session\n"
        f"The user attached a dataset archive (.zip): {url}\n"
        "When a step needs that data, send the link along as the `dataset_url`\n"
        "argument of the tool that fetches it (e.g. `run_sandbox_task`) — the\n"
        "sandbox is a separate machine and this is how the archive gets there.\n"
        "Judge for yourself whether a given call needs it, and never substitute\n"
        "a different dataset for the one the user attached.\n"
    ) if url else ""
    return None


# Recognisable token the orchestrator prompt / post-critic key off to re-route.
NO_MATCHING_TOOL_TOKEN = "NO_MATCHING_TOOL"

_TOOL_FAILURE_STATUSES = frozenset({
    "blocked", "error", "failed", "failure", "refused", "rejected",
})
_TOOL_PENDING_STATUSES = frozenset({
    "accepted", "in_progress", "pending", "queued", "running", "submitted",
})
_TOOL_SUCCESS_STATUSES = frozenset({"complete", "completed", "done", "ok", "success"})


def _tool_content_payload(content: Any) -> tuple[Any, str]:
    """Extract MCP ``content[].text`` without assuming one SDK representation."""
    if not isinstance(content, list):
        return None, ""
    texts: List[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
        else:
            text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    joined = "\n".join(texts)
    if len(texts) == 1:
        try:
            return json.loads(texts[0]), joined
        except (TypeError, ValueError):
            pass
    return None, joined


def normalize_tool_observation(response: Any) -> Dict[str, Any]:
    """Normalize the MCP/ADK result shapes used by experiment routing.

    This is an observation, not a scientific verdict.  It identifies explicit
    transport/tool failures, pending jobs and the deterministic
    ``NO_MATCHING_TOOL`` signal while preserving structured payloads.  Plain
    prose remains ``unknown`` so a callback never invents success from tone.
    """
    if hasattr(response, "model_dump"):
        try:
            response = response.model_dump(by_alias=True)
        except Exception:  # noqa: BLE001 - opaque SDK result, inspect attributes below
            pass
    if not isinstance(response, dict) and any(
        hasattr(response, name) for name in ("structuredContent", "content", "isError")
    ):
        response = {
            "structuredContent": getattr(response, "structuredContent", None),
            "content": getattr(response, "content", None),
            "isError": getattr(response, "isError", False),
        }

    if not isinstance(response, dict):
        message = str(response or "").strip()
        no_match = NO_MATCHING_TOOL_TOKEN in message.upper()
        return {
            "status": "failure" if no_match else "unknown",
            "is_error": no_match,
            "error_code": "no_matching_tool" if no_match else None,
            "no_matching_tool": no_match,
            "pending": False,
            "job_id": None,
            "message": message[:4000],
            "data": None,
        }

    structured = response.get("structuredContent")
    if structured is None:
        structured = response.get("structured_content")
    parsed_content, content_text = _tool_content_payload(response.get("content"))
    data = structured if structured is not None else parsed_content
    if data is None:
        # An unwrapped dict is itself useful structured output.  Wrapper keys
        # are removed so callers do not materialise protocol metadata as data.
        wrapper_keys = {
            "content", "structuredContent", "structured_content", "isError", "is_error",
            "status", "message", "error", "error_code", "error_message", "ok",
        }
        remainder = {k: v for k, v in response.items() if k not in wrapper_keys}
        data = remainder or None

    raw_status = str(response.get("status") or "").strip().lower().replace("-", "_")
    error_value = response.get("error")
    explicit_error = bool(response.get("isError") or response.get("is_error"))
    if error_value not in (None, False, "", [], {}):
        explicit_error = True
    if raw_status in _TOOL_FAILURE_STATUSES:
        explicit_error = True

    message_parts: List[str] = []
    for value in (
        response.get("message"), response.get("error_message"), error_value, content_text,
    ):
        if value not in (None, ""):
            message_parts.append(str(value))
    message = "\n".join(message_parts).strip()
    search_blob = " ".join((message, str(data or ""), raw_status)).upper()
    no_match = NO_MATCHING_TOOL_TOKEN in search_blob
    if no_match:
        explicit_error = True

    pending = raw_status in _TOOL_PENDING_STATUSES
    if explicit_error:
        status = "failure"
        pending = False
    elif pending:
        status = "pending"
    elif raw_status in _TOOL_SUCCESS_STATUSES or response.get("ok") is True:
        status = "success"
    else:
        status = "unknown"

    job_id = response.get("job_id") or response.get("jobId")
    if job_id is None and isinstance(data, dict):
        job_id = data.get("job_id") or data.get("jobId")
    error_code = response.get("error_code")
    if no_match:
        error_code = "no_matching_tool"
    elif explicit_error and not error_code:
        error_code = raw_status if raw_status in _TOOL_FAILURE_STATUSES else "tool_error"
    return {
        "status": status,
        "raw_status": raw_status or None,
        "is_error": explicit_error,
        "error_code": error_code,
        "no_matching_tool": no_match,
        "pending": pending,
        "job_id": str(job_id) if job_id not in (None, "") else None,
        "message": message[:4000],
        "data": data,
    }


def record_experiment_tool_observation(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Any,
) -> None:
    """Keep a bounded, machine-readable trace of route MCP outcomes."""
    state = getattr(tool_context, "state", None)
    if not state or not state.get("experiment_runtime"):
        return None
    observation = normalize_tool_observation(tool_response)
    row = {
        key: value for key, value in observation.items()
        if key != "data" and value not in (None, "")
    }
    row["tool"] = str(getattr(tool, "name", "") or "")
    row["argument_names"] = sorted(str(key) for key in (args or {}))
    history = list(state.get("experiment_tool_observations") or [])
    history.append(row)
    state["experiment_tool_observations"] = history[-50:]
    try:
        from CoScientist.experiments.runtime.state_machine import active_attempt

        _, _, attempt = active_attempt(state)
        if attempt.get("status") in {None, "running"}:
            attempt["family_tool_called"] = True
            attempt["last_tool_observation"] = row
    except Exception:  # noqa: BLE001 - observation must not alter tool delivery
        pass
    return None


def rerank_fallback_active(state: Any) -> bool:
    """True when the executor's tool set must go to the FEDOT.MAS fallback.

    Three conditions, all required:
      * the reranker never JUDGED the candidates (see UNJUDGED_REASONS) — note
        this is already past the local cross-encoder recovery in
        ``after_tool_reranker_agent``, so it means both rankers came up empty;
      * retrieval did accumulate candidates, so there is something to hand over;
      * the fallback is switched on.

    Read by ``ExecutorSwitchAgent`` (which child runs) and by
    ``redirect_when_no_tools`` (abstain, or stand aside for the fallback), so the
    two can never disagree about what happens next.
    """
    verdict = (state.get(TOOL_MATCH_STATE_KEY) if state else None) or {}
    if verdict.get("reason") not in UNJUDGED_REASONS:
        return False
    if not (state.get("accumulated_tools") or []):
        return False
    try:
        from CoScientist.config import get_settings
        return bool(get_settings().web.fedot_fallback_enabled)
    except Exception:  # noqa: BLE001 — a settings failure must not switch it ON
        logger.warning("fedot fallback: settings unreadable — treating as disabled")
        return False


def redirect_when_no_tools(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """before_agent_callback for ExperimentAgent: abstain → redirect to CoderAgent.

    By the time ExperimentAgent runs, the tool-prep pipeline has set
    ``executor_tool_match``. If no retrieved tool matched the task (and no web
    MCP was deployed), running FEDOT would just pick the nearest-but-wrong tool
    (the "train a GAN for a transformer task" failure). Instead we short-circuit
    the agent and return a structured redirect: the message is the tool
    pipeline's final answer, so TaskExecutorAgent (the router that called it)
    re-issues the step without it reaching the orchestrator: to McpBuilderAgent,
    which builds an MCP tool server from a scientific repository, or to
    CoderAgent for a one-off run.
    """
    state = callback_context.state
    verdict = state.get(TOOL_MATCH_STATE_KEY) or {}
    has_local = bool(state.get("filtered_tools"))
    has_web = bool(state.get("filtered_mcps"))
    # Servers McpBuilderAgent built or reused for this session are tools too: from
    # state, or from its build registry when the nested invocation's state is lost.
    has_deployed = bool(state.get("deployed_mcps"))
    if not has_deployed:
        try:
            from CoScientist.graph.session_scope import session_key
            from CoScientist.tools.alembic_tools import live_build_servers

            has_deployed = bool(live_build_servers(session_key(callback_context)))
        except Exception:  # noqa: BLE001 - never block the run on this check
            has_deployed = False

    # Only abstain on an explicit no-match verdict with nothing usable.
    if verdict.get("matched") or has_local or has_web or has_deployed:
        return None

    if rerank_fallback_active(state):
        logger.info(
            "[%s] no tools, but the reranker never judged them (%s) — the FEDOT.MAS "
            "fallback owns this task, not CoderAgent",
            _agent_name(callback_context), verdict.get("reason"),
        )
        return None

    best = verdict.get("best_score", 0.0)
    message = (
        f"{NO_MATCHING_TOOL_TOKEN}: No ready-made MCP tool matches this task "
        f"(best tool relevance was {best}, below the bar). This looks like custom "
        "engineering — a specific architecture, a named repository/example code, "
        "or writing and running code — which no existing tool covers. Do NOT "
        "treat a tool that shares only the verb (e.g. 'train a GAN' for a 'train a "
        "transformer' request) as a match. Re-issue this step to ONE of:\n"
        "  - McpBuilderAgent: when the capability exists as CODE in a scientific "
        "repository (the task or its paper names one, or one is easy to find) and "
        "the tool would be reused: it builds and serves a validated MCP tool "
        "server from that repository, closing this gap for every later task.\n"
        "  - CoderAgent: when the work is a one-off computation or there is no "
        "repository to build from.\n"
        "Do not run this tool pipeline again for this step."
    )
    logger.info(
        "TOOL_GAP_DETECTED best=%s inventory=0 → McpBuilderAgent (build from repo) "
        "| CoderAgent (one-off)", best,
    )
    state["fedot_results"] = message
    return types.Content(role="model", parts=[types.Part(text=message)])


# ResearchAgent as AgentTool dies if we replace a function_call with model text:
# that text is the sub-agent's final answer. Rewrite known aliases in-place so
# the real tool runs and the agent gets another turn.
_SHELL_PROGRAMS = frozenset({
    "awk", "cat", "cd", "chmod", "cp", "echo", "find", "git", "grep", "head",
    "ls", "mkdir", "mv", "pwd", "rm", "sed", "tail", "touch", "wc",
})
_TOOL_ALIASES: Dict[str, Sequence[str]] = {
    "download_papers": ("download_papers_from_search",),
    "explore_literature": (
        "search_papers", "explore_scientific_database", "tavily_search",
    ),
    "explore_papers": ("explore_my_papers", "search_papers"),
    "explore_scientific_database": (
        "search_papers", "explore_scientific_database", "tavily_search",
    ),
    "pubmed_search": ("search_papers", "tavily_search"),
    "search_literature": ("search_papers", "tavily_search"),
    "search_scientific_database": ("search_papers", "tavily_search"),
    "search_scientific_papers": ("search_papers",),
}
_QUERY_KEYS = ("query", "question", "task", "request", "q")
_QUERY_TARGETS = frozenset({
    "download_papers_from_search", "search_papers", "tavily_search",
})
_QUESTION_TARGETS = frozenset({
    "explore_scientific_database", "explore_my_papers",
})


def _function_call_args(fc: Any) -> Dict[str, Any]:
    raw = getattr(fc, "args", None) or {}
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    return dict(raw) if isinstance(raw, dict) else {}


def _remap_hallucinated_args(target: str, args: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(args)
    if target in _QUERY_TARGETS and "query" not in out:
        for key in _QUERY_KEYS:
            if key in out and key != "query":
                out["query"] = out.pop(key)
                break
    if target in _QUESTION_TARGETS and "question" not in out:
        for key in _QUERY_KEYS:
            if key in out and key != "question":
                out["question"] = out.pop(key)
                break
    return out


def resolve_hallucinated_tool(name: str, valid: Iterable[str]) -> Optional[str]:
    """Map a hallucinated tool name onto exactly one real tool, or None."""
    valid_set = {item for item in valid if item}
    if not name or name in valid_set:
        return None
    for candidate in _TOOL_ALIASES.get(name, ()):
        if candidate in valid_set:
            return candidate
    if name.lower() in _SHELL_PROGRAMS:
        return None
    close = get_close_matches(name, valid_set, n=2, cutoff=0.78)
    if len(close) == 1:
        return close[0]
    if len(close) >= 2:
        first = SequenceMatcher(None, name, close[0]).ratio()
        second = SequenceMatcher(None, name, close[1]).ratio()
        if first - second >= 0.08:
            return close[0]
    return None


def make_unknown_tool_guard(valid_names: Iterable[str]) -> Callable:
    """Build an after_model_callback that intercepts hallucinated tool calls.

    When the LLM emits a function call whose name is NOT a real tool of the
    agent, ADK raises and kills the whole run before any tool/agent callback can
    react (e.g. CoderAgent calling `find` directly instead of
    `execute_bash("find ...")`).

    Prefer rewriting a known alias (or a uniquely close name) into a real
    function_call so AgentTool sub-agents keep a turn. Only unmatched names
    become a corrective text reply.
    """
    valid = {name for name in valid_names if name}

    def guard(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        content = getattr(llm_response, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            return None
        rewritten: List[Any] = []
        unresolved: List[str] = []
        changed = False
        for part in parts:
            fc = getattr(part, "function_call", None)
            name = getattr(fc, "name", None) if fc is not None else None
            if not name or name in valid:
                rewritten.append(part)
                continue
            alias = resolve_hallucinated_tool(name, valid)
            if alias:
                logger.warning(
                    "[%s] rewriting hallucinated tool %s → %s",
                    _agent_name(callback_context),
                    name,
                    alias,
                )
                rewritten.append(
                    types.Part.from_function_call(
                        name=alias,
                        args=_remap_hallucinated_args(alias, _function_call_args(fc)),
                    )
                )
                changed = True
            else:
                unresolved.append(name)
                rewritten.append(part)
        if unresolved:
            bad = ", ".join(sorted(set(unresolved)))
            allowed = ", ".join(sorted(valid))
            logger.warning(
                "[%s] hallucinated tool call(s): %s",
                _agent_name(callback_context),
                bad,
            )
            msg = (
                f"The tool(s) `{bad}` do not exist — they are not in your tool list. "
                f"Your only tools are: {allowed}. Shell programs (find, grep, ls, cat, "
                "wc, git, sed, awk, …) are NOT tools — run them INSIDE execute_bash, "
                "e.g. execute_bash(command=\"find . -name '*.py' | wc -l\"). "
                "Re-issue your request calling ONLY a tool from the list above."
            )
            return LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text=msg)])
            )
        if changed:
            return LlmResponse(
                content=types.Content(role="model", parts=rewritten)
            )
        return None

    return guard


def make_plan_registration_guard() -> Callable:
    """Build an after_model_callback that ends the planner's turn once the plan
    is registered, instead of letting it re-register forever.

    `create_plan` NORMALISES what it is given: it renumbers ids, drops
    OrchestratorAgent tasks and MERGES consecutive tasks with the same executor
    assignee. The planner prompt tells the model to check the returned plan — so
    when the plan it gets back is not the one it sent, the model registers again
    to "fix" it, gets the same normalisation, and loops. It cannot win: the
    difference it is chasing is the tracker's own doing.

    That loop is reachable on its own, but the plan critic makes it likely: ask
    for "a separate analysis step" next to an existing executor step and the
    tracker merges the two back together on every attempt.

    A registered plan is exactly ``state['active_tasks']`` being non-empty —
    SessionAgent clears it before each planner run, so the flag is per-run and
    a retry after a REJECTED create_plan (which registers nothing) still works.
    """

    def guard(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        tasks = callback_context.state.get("active_tasks")
        if not tasks:
            return None  # nothing registered yet — the first call must go through
        content = getattr(llm_response, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            return None
        if not any(
            getattr(getattr(p, "function_call", None), "name", None) == "create_plan"
            for p in parts
        ):
            return None

        logger.warning(
            "[%s] create_plan called again after %d task(s) were registered — "
            "ending the turn instead of re-registering",
            _agent_name(callback_context), len(tasks),
        )
        roster = "\n".join(
            f"{i}. {t.get('title')} → {t.get('assignee')}"
            for i, t in enumerate(tasks, 1)
        )
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(
                text=f"The plan is registered and stands as follows:\n\n{roster}"
            )])
        )

    return guard


def _note_step_participants(graph: Any, tasks: List[Dict[str, Any]],
                            matched: Dict[int, str],
                            by_ref: Dict[str, str]) -> None:
    """Carry the tracker's answer about a plan step onto the step's node.

    Two different claims, and the difference is the point. `assignee` is who the
    plan NAMED — an intention, and the only thing recorded until now. `executors`
    is who the tracker watched move the step, which `set_task_status` began
    keeping once it stopped discarding the agent it is handed.

    Both are stored with their basis so the panel can say which is which. A run
    where the plan named one agent and another did the work is ordinary; a
    panel that shows only the first is how a reader ends up sure of the wrong
    thing.
    """
    try:
        rows = []
        for i, task in enumerate(tasks):
            nid = matched.get(i) or by_ref.get(f"ps_{i}")
            if not nid:
                continue
            if assignee := str(task.get("assignee") or "").strip():
                rows.append({"node_id": nid, "agent": assignee,
                             "basis": "assignee"})
            for worker in (task.get("executors") or [])[:8]:
                if str(worker or "").strip():
                    rows.append({"node_id": nid, "agent": str(worker).strip(),
                                 "basis": "work_order"})
        if rows:
            graph.add_contributors(rows, source=_PLAN_SOURCE)
    except Exception:  # noqa: BLE001 — bookkeeping never breaks the mirror
        logger.debug("plan mirror: could not record participants", exc_info=True)


def _agent_name(callback_context: CallbackContext) -> str:
    return getattr(callback_context, "agent_name", None) or "agent"


def print_research_agent_tool_call(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Any,
) -> None:
    """Print tool calls and persist downloaded S3 keys to session state."""
    try:
        logger.info(f"\n[ResearchAgent tool called] {tool.name}")
        logger.info(f"[ResearchAgent tool args] {args}")
    except Exception as e:
        logger.error(f"Error in print_research_agent_tool_call: {e}")

    if tool.name != "download_papers_from_search":
        return

    try:
        papers = (tool_response or {}).get("metadata", {}).get("papers", [])
        new_keys = [p["s3_key"] for p in papers if p.get("s3_key")]
        if not new_keys:
            return
        existing: List[str] = tool_context.state.get("downloaded_paper_s3_keys", [])
        merged_keys: List[str] = existing + [k for k in new_keys if k not in existing]
        tool_context.state["downloaded_paper_s3_keys"] = merged_keys
        logger.info(
            "Registered %d downloaded paper S3 key(s) in session state.",
            len(merged_keys),
        )
    except Exception as e:
        logger.error("Failed to persist downloaded paper S3 keys: %s", e)

async def capture_mcp_artifacts(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Any,
) -> None:
    """after_tool: mirror the artifacts a tool returned, and record where.

    Many MCP tools (e.g. the tox-antitargets suite) render a plot server-side and
    return a presigned URL to it (commonly ``metadata.figure.artifact``). That link
    only lives in the tool result; with the aggregator running ``include_contents:
    none`` it never reaches the report unless captured here — at the AGENT's own
    tool boundary, which fires for sub-agent (AgentTool) MCP calls where an
    App-level plugin does not.

    The mirroring itself lives in ``reporting.mirror`` and is shared with
    ``McpArtifactCapturePlugin``. It used to be duplicated, and the two copies
    drifted: the plugin wrote the durable on-disk index and this one did not, so
    a sub-agent's figures were lost on restart. One body now, two thin callers.
    """
    record_experiment_tool_observation(tool, args, tool_context, tool_response)
    try:
        from CoScientist.reporting.collect import find_artifact_urls
        urls = find_artifact_urls(tool_response)
    except Exception:  # noqa: BLE001 — capture must never break a tool call
        return
    if not urls:
        return

    name = getattr(tool, "name", None)
    mirrored = []
    try:
        import asyncio

        from CoScientist.graph.session_scope import session_key
        from CoScientist.reporting.mirror import mirror_tool_result

        # Resolved on the loop — see the plugin's twin: `session_key` mutates
        # ADK state, and the download runs in a thread.
        scope = session_key(tool_context)
        mirrored = await asyncio.to_thread(
            mirror_tool_result, tool, tool_context, tool_response, scope
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("capture_mcp_artifacts: mirroring failed: %s", e)

    try:
        from CoScientist.reporting.artifact_index import record

        by_url = {
            m["source_url"]: m for m in mirrored
            if isinstance(m, dict) and m.get("source_url")
        }
        # The same filter as the durable index: this list is read by the
        # report collector through its `*_artifacts` state sweep, so an icon
        # left here reaches the reader by a second door.
        from CoScientist.reporting.collect import _is_page_chrome

        urls = [u for u in urls if not _is_page_chrome(u)]
        existing = list(tool_context.state.get("mcp_artifacts") or [])
        seen = {a.get("url") for a in existing if isinstance(a, dict)}
        entries = []
        for u in urls:
            mirror_record = by_url.get(u) or {}
            entries.append({
                "bucket": mirror_record.get("bucket"),
                "s3_key": mirror_record.get("s3_key"),
                "artifact_id": mirror_record.get("artifact_id"),
                "tool": name,
                "label": mirror_record.get("label") or "artifact",
                "url": u,
            })
            if u in seen:
                continue
            seen.add(u)
            existing.append({
                "url": u, "tool": name,
                "artifact_id": mirror_record.get("artifact_id"),
            })
        tool_context.state["mcp_artifacts"] = existing
        # The half this callback never did. State lives in an in-memory session
        # service; the file on disk is what survives a restart.
        # Page furniture never enters the durable index. Written once, it is
        # read by the report collector for the rest of the session — and there
        # it carries no `source_kind`, so nothing downstream can tell an icon
        # from a figure. 70 of the 219 rows recorded across real sessions are a
        # publisher's letterhead.
        from CoScientist.reporting.collect import _is_page_chrome

        entries = [e for e in entries
                   if not _is_page_chrome(str(e.get("url") or ""))]
        record(entries, tool_context)
        logger.info(
            "capture_mcp_artifacts: %s → +%d artifact URL(s), %d mirrored (%d total)",
            name, len(urls),
            sum(1 for m in mirrored if m.get("state") == "stored"), len(existing),
        )
    except Exception as e:  # noqa: BLE001
        logger.error("capture_mcp_artifacts failed: %s", e)


# A dropped MCP session is the transport failing, not the agent using up an
# attempt: ADK turns it into {"error": "... MCP session connection lost: ..."}.
_TRANSIENT_TOOL_ERROR_MARKERS = ("mcp session connection lost",)
_TOOL_CALL_CHARGES_KEY = "_tool_call_charges"


def is_transient_tool_error(tool_response: Any) -> bool:
    if isinstance(tool_response, dict):
        tool_response = tool_response.get("error")
    if not isinstance(tool_response, str):
        return False
    text = tool_response.lower()
    return any(marker in text for marker in _TRANSIENT_TOOL_ERROR_MARKERS)


def charge_tool_call(
    tool_context: ToolContext, state_key: str, subkey: Optional[str] = None
) -> None:
    """Remember which counter a before_tool limiter bumped for this call, so
    ``refund_transient_tool_error`` can undo it if the call never reached the
    server. ``subkey`` addresses a counter inside a dict-valued state entry."""
    call_id = getattr(tool_context, "function_call_id", None)
    if not call_id:
        return
    charges = dict(tool_context.state.get(_TOOL_CALL_CHARGES_KEY) or {})
    charges[call_id] = list(charges.get(call_id, [])) + [[state_key, subkey]]
    tool_context.state[_TOOL_CALL_CHARGES_KEY] = charges


def refund_transient_tool_error(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Any,
) -> None:
    """after_tool: return the attempt to every limiter that charged this call
    when the MCP session dropped mid-call."""
    del args
    call_id = getattr(tool_context, "function_call_id", None)
    charges = tool_context.state.get(_TOOL_CALL_CHARGES_KEY) or {}
    if not call_id or call_id not in charges:
        return None
    charges = dict(charges)
    entries = charges.pop(call_id)
    tool_context.state[_TOOL_CALL_CHARGES_KEY] = charges
    if not is_transient_tool_error(tool_response):
        return None
    for state_key, subkey in entries:
        if subkey is None:
            count = int(tool_context.state.get(state_key, 0) or 0)
            tool_context.state[state_key] = max(0, count - 1)
        else:
            counts = dict(tool_context.state.get(state_key) or {})
            counts[subkey] = max(0, int(counts.get(subkey, 0) or 0) - 1)
            tool_context.state[state_key] = counts
    logger.info(
        "[%s] %s: MCP session dropped, attempt not counted",
        getattr(tool_context, "agent_name", "?"), getattr(tool, "name", "?"),
    )
    return None


class SearchLimiter:

    _STATE_KEY = "_search_limiter_count"

    def __init__(self, max_searches: int = 5):
        self.max_searches = max_searches

    def reset_search_budget(self, callback_context: CallbackContext) -> None:
        """before_agent: cap is per ResearchAgent invocation, not per session."""
        try:
            callback_context.state[self._STATE_KEY] = 0
        except Exception:
            return None
        return None

    @staticmethod
    def is_counted_search(name: str) -> bool:
        """Count OpenAlex / web search only — not downloads or research_*.

        Token ``search`` used to match ``download_papers_from_search`` and eat
        the whole ResearchAgent budget after one 429, so Tavily never ran.
        """
        tokens = [tok for tok in re.split(r"[^a-z]+", (name or "").lower()) if tok]
        if "search" not in tokens:
            return False
        if "download" in tokens:
            return False
        if tokens[:1] == ["research"]:
            return False
        return True

    @staticmethod
    def is_failed_search_response(tool_response: Any) -> bool:
        if tool_response is None or is_transient_tool_error(tool_response):
            return True
        blob = tool_response
        if isinstance(tool_response, dict):
            if tool_response.get("isError") is True:
                return True
            blob = tool_response
        text = str(blob).lower()
        markers = (
            "error calling tool",
            "too many requests",
            "429",
            "sslerror",
            "max retries exceeded",
        )
        return any(marker in text for marker in markers)

    def limit_searches(self, tool, args: dict, tool_context: ToolContext) -> Optional[dict]:
        if not self.is_counted_search(getattr(tool, "name", "")):
            return None
        count = int(tool_context.state.get(self._STATE_KEY, 0) or 0)
        if count >= self.max_searches:
            return {
                "result": (
                    f"Search limit reached ({self.max_searches} searches allowed). "
                    "You MUST now synthesize your answer from the results you already have. "
                    "Do NOT attempt any more searches."
                )
            }
        return None

    def record_search_result(
        self,
        tool,
        args: dict,
        tool_context: ToolContext,
        tool_response: Any,
    ) -> None:
        if not self.is_counted_search(getattr(tool, "name", "")):
            return None
        if self.is_failed_search_response(tool_response):
            logger.info(
                "search failed, not counting toward limiter: %s",
                getattr(tool, "name", ""),
            )
            return None
        count = int(tool_context.state.get(self._STATE_KEY, 0) or 0) + 1
        tool_context.state[self._STATE_KEY] = count
        return None


class TavilySearchLimiter:
    """Give each agent an independent budget for Tavily web searches only.

    Economics agents also use MCP tools whose names include ``search``; those
    are supplier-catalogue operations rather than web searches and must not
    consume this fallback budget.
    """

    _STATE_KEY = "_tavily_search_limiter_counts"

    def __init__(self, max_searches: int = 5):
        self.max_searches = max_searches

    def limit_searches(self, tool, args: dict, tool_context: ToolContext) -> Optional[dict]:
        if getattr(tool, "name", "") != "tavily_search":
            return None

        agent = getattr(tool_context, "agent_name", None) or "unknown"
        counts = tool_context.state.get(self._STATE_KEY, {})
        counts = dict(counts) if isinstance(counts, dict) else {}
        count = int(counts.get(agent, 0)) + 1
        counts[agent] = count
        tool_context.state[self._STATE_KEY] = counts
        charge_tool_call(tool_context, self._STATE_KEY, agent)
        if count > self.max_searches:
            return {
                "result": (
                    f"Web-search limit reached for {agent} ({self.max_searches} Tavily searches allowed). "
                    "Use the evidence already found or request the missing information from the human."
                )
            }
        return None


class PerToolCallLimiter:
    """Limit each tool independently within one agent execution branch.

    Parallel ``AgentTool`` calls share the session state and invocation id, but
    ADK gives every delegated agent run its own branch.  Including that branch
    in the counter key keeps two concurrent ResearchAgent runs from consuming
    each other's quota.
    """

    _STATE_KEY_PREFIX = "_per_tool_call_limiter"

    def __init__(self, max_calls: int = 2, per_tool: Optional[Dict[str, int]] = None):
        limits = [max_calls, *(per_tool or {}).values()]
        if min(limits) < 1:
            raise ValueError("max_calls must be at least 1")
        self.max_calls = max_calls
        self.per_tool = dict(per_tool or {})

    def limit_tool_calls(
        self, tool: BaseTool, args: dict, tool_context: ToolContext
    ) -> Optional[dict]:
        del args  # Every call counts, regardless of whether its arguments differ.
        tool_name = str(getattr(tool, "name", "") or "unknown_tool")
        agent_name = str(getattr(tool_context, "agent_name", "") or "agent")
        invocation_id = str(getattr(tool_context, "invocation_id", "") or "invocation")
        branch = str(getattr(tool_context, "branch", "") or "root")
        state_key = (
            f"{self._STATE_KEY_PREFIX}:{invocation_id}:{branch}:{agent_name}:{tool_name}"
        )

        limit = self.per_tool.get(tool_name, self.max_calls)
        count = int(tool_context.state.get(state_key, 0)) + 1
        tool_context.state[state_key] = count
        if count <= limit:
            charge_tool_call(tool_context, state_key)
            return None

        return {
            "status": "blocked",
            "blocked_by": "per_tool_call_limiter",
            "tool": tool_name,
            "limit": limit,
            "message": (
                f"Tool call limit reached: `{tool_name}` may be used at most "
                f"{limit} times in this research task. Synthesize the "
                "answer from existing results or use a different tool."
            ),
        }


class PaperSearchGuard:
    """Clamp paper-search MCP result sets before they reach OpenAlex."""

    metadata_limit = 5
    download_limit = 3

    def guard_paper_search(self, tool, args: dict, tool_context: ToolContext) -> None:
        del tool_context  # callback API parity; this guard needs no session state
        if tool.name == "search_papers":
            args["limit"] = min(self._positive_int(args.get("limit"), self.metadata_limit),
                                self.metadata_limit)
        elif tool.name == "download_papers_from_search":
            args["limit"] = min(self._positive_int(args.get("limit"), self.download_limit),
                                self.download_limit)

    @staticmethod
    def _positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

def inject_original_query(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Replace the last message in llm_request.contents with the original query."""

    original = getattr(callback_context, "user_content", None)
    if original is None or not original.parts:
        return

    # Extract original text
    original_text = None
    for part in original.parts:
        if part.text:
            original_text = part.text
            break
    if not original_text:
        return

    # Replace the last user-role content in llm_request.contents
    for i, content in enumerate(llm_request.contents):
        if getattr(content, "role", "user") == "user" and getattr(content, "parts", None):
            llm_request.contents[i] = types.Content(
                role="user",
                parts=[types.Part(text=original_text)],
            )
            logger.info(
                "[OrchestratorAgent] Replaced planner messages with original user query"
            )
            return

# ── the plan, mirrored into the research graph ───────────────────────────────
# The middle layer of the story — what each hypothesis is actually checked BY —
# existed only if a model remembered to commit it. In a run whose agents all
# skipped research_commit the graph drew a question with nothing underneath, and
# no amount of work on the viewer can draw a method that was never recorded. A
# registered plan is already a deterministic, ordered list of steps: deriving
# one planned VerificationMethod per step costs no LLM call and cannot be
# forgotten. Written as "plan-mirror" rather than as an agent, so a reader can
# see at a glance that no model chose these.

#: normalized task title -> the PlanStep id created for it, under the study
#: generation it was written against.
_VM_BY_TASK_KEY = "_research_vm_by_task"
_PLAN_SOURCE = "plan-mirror"


def _task_key(task: Dict[str, Any]) -> str:
    return " ".join(str(task.get("title") or "").split()).lower()[:120]


def _live_plan_steps(graph: Any) -> Dict[str, Dict[str, Any]]:
    """What the graph already holds as plan steps, in the order it holds them."""
    out: Dict[str, Dict[str, Any]] = {}
    try:
        nodes = graph.full().get("nodes") or []
    except Exception:  # noqa: BLE001 — a mirror must never break its caller
        return out
    for n in nodes:
        if n.get("type") != "PlanStep":
            continue
        attrs = n.get("attrs") or {}
        out[str(n.get("id"))] = {
            "status": n.get("status"),
            "key": " ".join(str(attrs.get("title") or "").split()).lower()[:120],
            "plan_task_id": str(attrs.get("plan_task_id") or "").strip(),
            "assignee": str(attrs.get("assignee") or "").strip(),
            "words": _step_words(attrs.get("title"), attrs.get("description")),
            "attrs": attrs,
        }
    return out


#: Words too common in a plan to tell two steps apart.
_STOP = frozenset((
    "и", "или", "для", "на", "по", "с", "со", "в", "во", "из", "не", "от", "до",
    "при", "как", "что", "это", "все", "the", "and", "for", "with", "of", "to",
    "a", "an", "in", "on", "шаг", "этап", "задача", "провести", "выполнить",
    "сделать", "получить", "оценить",
))


def _step_words(*parts: Any) -> set:
    said = " ".join(str(p or "") for p in parts).lower()
    return {w for w in re.findall(r"[\w\-]{4,}", said) if w not in _STOP}


def _overlap(a: set, b: set) -> float:
    """Jaccard, because neither side is the reference — a reworded step may be
    longer or shorter than the one it replaces."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


#: How much of the wording a reworded step has to keep to still be the same
#: step. Low on purpose: the planner may rewrite a title wholesale and keep
#: only the instrument or the measurement in it, and the plan's own id and the
#: assignee have already had to agree before this is consulted at all.
_REWORD_FLOOR = 0.15


def _match_steps(tasks: List[Dict[str, Any]], live: Dict[str, Dict[str, Any]],
                 memo: Dict[str, str]) -> Dict[int, str]:
    """Which PlanStep node each task in the CURRENT plan belongs to.

    Three passes, most certain first. Every pass claims a step exclusively, so
    two tasks can never be mirrored onto one node.

    1. what this session already mirrored, by title (the memo);
    2. a step in the graph whose title is word-for-word the task's;
    3. a step standing in the plan's own slot — same `plan_task_id`, same
       assignee — whose wording the task still partly keeps.

    Pass 3 is the one the live run needed. Ids are positional (``create_plan``
    hands out ``TASK-1…n`` after ordering), which is why they are not trusted
    alone: the assignee and the surviving words have to agree as well, and if
    they do not the task falls through to being created, which is what happened
    before. The cost of a wrong match is a retitled card; the cost of no match
    is a duplicate step and an orphan beside it.
    """
    claimed: set = set()
    matched: Dict[int, str] = {}

    for i, task in enumerate(tasks):
        step = memo.get(_task_key(task))
        if step and step not in claimed:
            matched[i] = step
            claimed.add(step)

    by_key: Dict[str, str] = {}
    for sid, data in live.items():
        if data["key"]:
            by_key.setdefault(data["key"], sid)
    for i, task in enumerate(tasks):
        if i in matched:
            continue
        step = by_key.get(_task_key(task))
        if step and step not in claimed:
            matched[i] = step
            claimed.add(step)

    for i, task in enumerate(tasks):
        if i in matched:
            continue
        tid = str(task.get("id") or "").strip()
        if not tid:
            continue
        words = _step_words(task.get("title"), task.get("description"))
        who = str(task.get("assignee") or "").strip()
        for sid, data in live.items():
            if sid in claimed or data["plan_task_id"] != tid:
                continue
            if who and data["assignee"] and who != data["assignee"]:
                continue
            # Nothing to compare (a step recorded without a title) leaves the
            # id and the assignee as the whole of the evidence.
            if words and data["words"] and _overlap(words, data["words"]) < _REWORD_FLOOR:
                continue
            matched[i] = sid
            claimed.add(sid)
            break
    return matched


def _step_attrs_differ(live: Dict[str, Dict[str, Any]], step_id: str,
                       task: Dict[str, Any]) -> bool:
    """Whether the card would read differently now than it does in the graph.

    Over `_card_attrs`, so a description the planner DELETED counts as a
    difference — compared over `_step_attrs`, which drops empty values, an
    emptied field was invisible and stayed on the card.

    `plan_task_id` is not compared and not rewritten. It is positional —
    `create_plan` hands out TASK-1…n afresh after ordering — and the methods
    that realise a step carry the id the step had when they were written.
    Following the plan's renumbering would leave the step holding an id that
    belongs, on those methods, to a different step, and `_realises_edges` would
    then draw the link onto the wrong card.
    """
    data = live.get(step_id)
    if data is None:
        return False
    stored = data["attrs"]
    return any(str(stored.get(k) or "") != str(v or "")
               for k, v in _card_attrs(task).items())


#: Node types that can be what a plan step turned into.
_REALISING_TYPES = ("VerificationMethod", "Hypothesis", "Evidence", "Conclusion")

#: The tracker's status words, lowercased into the PlanStep vocabulary. Anything
#: unrecognised is a step nobody has started.
_STEP_STATUS = {"todo": "todo", "in_progress": "in_progress", "done": "done",
                "blocked": "blocked", "cancelled": "blocked", "failed": "blocked"}


def _step_status(task: Dict[str, Any]) -> str:
    return _STEP_STATUS.get(str(task.get("status") or "").strip().lower(), "todo")


def _may_move(current: Any, want: str) -> bool:
    """Whether the graph would accept this step moving there.

    A commit is all-or-nothing, so one impossible status change costs the
    retitles, the new steps and the retirements sent with it. `create_plan`
    re-issues EVERY task as TODO when a plan is revised, which asks a finished
    step to go back to «не начат»; PlanStep has no such transition, and the
    whole mirror fell silent for the rest of the run.
    """
    cur = str(current or "").strip()
    if not cur or cur == want:
        return False
    try:
        from CoScientist.graph.research import schema
        allowed = schema.STATUS_TRANSITIONS.get("PlanStep") or ()
    except Exception:  # noqa: BLE001 — a mirror must never break its caller
        return True
    return (cur, want) in {tuple(p) for p in allowed}


def _card_attrs(task: Dict[str, Any]) -> Dict[str, Any]:
    """Every field of the card, including the ones the plan has emptied.

    `_step_attrs` drops empty values, which is right when a step is created and
    wrong when it is rewritten: a description the planner deleted would stay on
    the card forever, and `_step_attrs_differ` would not even see the deletion.
    `plan_task_id` is deliberately NOT here — see `_step_attrs_differ`.
    """
    full = {k: task.get(k, "") or "" for k in
            ("title", "description", "assignee", "notes")}
    full["tools"] = ", ".join(str(t) for t in (task.get("tools") or []) if t)
    return full


def _step_attrs(task: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in {
        "title": task.get("title", ""),
        "description": task.get("description", ""),
        "plan_task_id": str(task.get("id") or ""),
        "assignee": task.get("assignee", ""),
        "notes": task.get("notes", ""),
        # What the plan already knows about the HOW. The planner can read the
        # tool list, so this is the one part of a method it can answer in
        # advance, and whoever writes the method starts from it instead of
        # guessing.
        "tools": ", ".join(str(t) for t in (task.get("tools") or []) if t),
    }.items() if v}


def _study_generation(graph: Any) -> str:
    """Identity of the LIVE study, so a memo cannot outlive its graph.

    `research_init` archives the study and replaces the graph with an empty one,
    and `_next_id` scans only the live graph — so ids restart at PS1. The memo
    of "which step did I mirror for which task" lives in ADK session state,
    which survives that reset, so after a switch the old ids matched the new
    study's fresh ones and the mirror both skipped tasks it had never mirrored
    HERE and wrote an edge to an id that now names somebody else's node.
    Checking that an id merely resolves cannot catch this; only identity can.
    """
    try:
        full = graph.full()
        # `research_id`, not the timestamp. `init_research` mints it with a
        # uuid, so two studies started in quick succession differ — whereas
        # `created_at` is a wall clock that ticks every ~15 ms on Windows, so a
        # fast re-init produced the SAME key, the memo was kept, and the mirror
        # decided the new study's steps had already been drawn. It surfaced as
        # an order-dependent test failure; in a run it would have silently left
        # the new study with no plan column at all.
        return f"{full.get('research_id')}:{full.get('root_id')}"
    except Exception:  # noqa: BLE001
        return ""


def _realises_edges(graph: Any) -> List[Dict[str, Any]]:
    """`realises` edges for work that names the plan step it carried out.

    A method, hypothesis or conclusion whose `plan_task_id` matches a step is
    what became of that step, and this edge is the one thing that lets a reader
    cross between the intention and the record. Derived from an attribute its
    writer set — not guessed from wording — and only ever added, because a step
    can be realised by several things and none of them stops being true.
    """
    try:
        full = graph.full()
        nodes = full.get("nodes") or []
        already = {(e.get("from"), e.get("to")) for e in (full.get("edges") or [])
                   if e.get("type") == "realises"}
    except Exception:  # noqa: BLE001
        return []
    # BOTH sides come from the graph, not from the task list. The work that
    # carries out a step is written turns after the plan was registered, and by
    # then the task list is empty — reading the mapping from it made the
    # linking unreachable exactly when it was needed.
    step_by_task = {str((n.get("attrs") or {}).get("plan_task_id") or ""): n.get("id")
                    for n in nodes if n.get("type") == "PlanStep"}
    step_by_task.pop("", None)
    if not step_by_task:
        return []
    out = []
    for n in nodes:
        if n.get("type") not in _REALISING_TYPES:
            continue
        task_id = str((n.get("attrs") or {}).get("plan_task_id") or "")
        step = step_by_task.get(task_id)
        if not step or (n.get("id"), step) in already:
            continue
        out.append({"type": "realises", "from": n.get("id"), "to": step})
    return out


def sync_plan_to_research_graph(tasks: Iterable[Dict[str, Any]], graph: Any,
                                state: Any, question: str = "") -> Optional[Any]:
    """Mirror the registered plan into the graph as one PlanStep per step.

    It used to write the plan as `VerificationMethod` nodes, and that was the
    most misleading thing in the graph a scientist reads. A step is an
    INTENTION — what to do, in what order, by whom. A method answers a
    different question: by what MEANS was this established, and against which
    bar. Conflated, the planner's task list appeared as "methods" hanging off
    the research question, carrying no instrument, indistinguishable from a
    method an agent had designed for a hypothesis — and the reader could not
    tell the plan from the record.

    They are separate nodes now, joined by `realises`, so the graph shows both
    the intention and what was made of it. Methods are written by the agents
    that own them; when a hypothesis has none, the graph says so through
    `queries.hypotheses_without_methods` instead of filling the gap with the
    plan.

    Idempotent: a task already mirrored is remembered in session state, keyed to
    the study generation, so a re-plan adds only what is new; a step whose
    tracker status has moved is advanced in place.
    """
    tasks = [t for t in (tasks or []) if isinstance(t, dict)]
    root = graph.root_id()
    if not root:
        seed = question or (tasks[0].get("title", "") if tasks else "")
        if not seed:
            return None
        root = (graph.ensure_root(seed) or {}).get("root_id")
    if not root:
        return None
    # Deliberately NOT `if not tasks: return None`. The `realises` links are a
    # fact about the GRAPH, not about the plan: the work that carries out a step
    # is written after the plan is registered, and by then the task list this is
    # called with can be empty. Gating on it made the linking unreachable.
    gen = _study_generation(graph)
    try:
        memo = dict(state.get(_VM_BY_TASK_KEY) or {})
    except Exception:  # noqa: BLE001 — a stateless caller still gets the mirror
        memo = {}
    seen = dict(memo.get("ids") or {}) if memo.get("gen") == gen else {}
    live = _live_statuses(graph)
    # What the experiment tasks under each step say. The module records its
    # results through the store, not through the tracker, so without this a
    # step whose every task is done still reads "not started".
    from_tasks = _status_from_tasks(graph)

    live_steps = _live_plan_steps(graph)
    matched = _match_steps(tasks, live_steps, seen)
    # Rebuilt, not added to: a memo that keeps the title a retitle superseded
    # goes on asserting a wording the plan no longer uses, and the next plan to
    # contain that wording takes the step away from the task whose slot it is.
    seen = {}

    creates, keys, updates, retitles = [], [], [], []
    for i, task in enumerate(tasks):
        key = _task_key(task)
        if not key:
            continue
        step = matched.get(i)
        if step:
            seen[key] = step
            # A step the planner reworded is the same step: its history, the
            # work already hung under it and the links into it all belong to
            # the work, not to the sentence describing it. Rewriting the card
            # is what the operator asked for; a second card beside the first is
            # what they got, because the mirror recognised a step only by its
            # title. On session_d3ce3a45bdb24272b28efd3f976ec16b two reworded
            # steps made a six-step plan into an eight-step column.
            if _step_attrs_differ(live_steps, step, task):
                retitles.append({"id": step, "attrs": _card_attrs(task)})
            tracked = _step_status(task)
            want = tracked
            derived = from_tasks.get(step)
            if tracked != "blocked":
                # A failed child is a terminal outcome, not work that is still
                # running.  `blocked` is deliberately outside `_furthest`, so
                # handle that verdict before comparing ordinary progress.
                want = ("blocked" if derived == "blocked"
                        else _furthest(tracked, derived) or tracked)
            if live.get(step) not in (None, want) and _may_move(live.get(step), want):
                reason = ("план перевёл шаг в состояние «" + _RU_STEP.get(want, want) + "»"
                          if want == tracked else
                          "задачи эксперимента под этим шагом " + _RU_STEP.get(want, want))
                updates.append({"id": step, "status": want, "reason": reason})
            if want != tracked:
                # The roadmap the operator reads is the tracker; leaving it
                # behind would make the two views of one step disagree.
                _mark_step(state, task, want)
            continue
        keys.append((f"ps_{i}", key))
        creates.append({"type": "PlanStep", "ref": f"ps_{i}",
                        "status": _step_status(task), "attrs": _step_attrs(task)})

    # A step the revised plan no longer contains, and that nobody ever started,
    # is not part of the study any more. Left at `todo` it reads as work still
    # ahead. Only `todo`: a step that ran, or finished, happened — the plan
    # changing afterwards does not unhappen it.
    #
    # And only against a plan there is. This runs on every orchestrator turn,
    # where the task list can be empty for reasons that have nothing to do with
    # the plan — a restarted process reading an existing graph, a mirror called
    # before the tracker is populated — and "no tasks" would then retire the
    # whole column. An empty list is no news about the plan; the `realises`
    # links below are what that call is for.
    for sid, data in (live_steps.items() if tasks else ()):
        if sid in matched.values() or data["status"] != "todo":
            continue
        if not _may_move(data["status"], "blocked"):
            continue
        updates.append({"id": sid, "status": "blocked",
                        "reason": "шаг убран при пересмотре плана"})

    result = None
    if creates or updates or retitles:
        result = graph.commit(source=_PLAN_SOURCE, nodes=creates + retitles,
                              status_updates=updates, partial_edges=True)
        if not result.ok:
            logger.warning("plan -> research graph refused: %s", result.errors[:3])
            return result
        # By ref, not by position: the same commit now carries the retitles as
        # attrs-merges, and the store may answer a create with a node it
        # already held, so the echo list is no longer one entry per new step in
        # the order they were sent.
        by_ref = {e.get("ref"): e.get("id")
                  for e in result.committed.get("nodes", []) if e.get("ref")}
        for ref, key in keys:
            if nid := by_ref.get(ref):
                seen[key] = nid
        _note_step_participants(graph, tasks, matched, by_ref)
        try:
            state[_VM_BY_TASK_KEY] = {"gen": gen, "ids": seen}
        except Exception:  # noqa: BLE001
            pass

    # Read after the commit above, so a step created just now is linkable in
    # this same call rather than a turn later.
    edges = _realises_edges(graph)
    if edges:
        linked = graph.commit(source=_PLAN_SOURCE, edges=edges, partial_edges=True)
        if not linked.ok:
            logger.warning("plan links refused: %s", linked.errors[:3])
        result = linked if result is None else result
    return result


#: How far along a step is. `blocked` is not on this scale — it is a verdict,
#: not a distance — so it never loses to a derived status.
_STEP_PROGRESS = {"todo": 0, "in_progress": 1, "done": 2}
#: The same states in the words the card shows, for the «почему» line.
_RU_STEP = {"todo": "не начат", "in_progress": "выполняются", "done": "выполнены",
            "blocked": "заблокирован"}


def _furthest(*statuses: Optional[str]) -> Optional[str]:
    """The most advanced of the statuses on the progress scale."""
    ranked = [(s, _STEP_PROGRESS[s]) for s in statuses
              if s in _STEP_PROGRESS]
    if not ranked:
        return None
    return max(ranked, key=lambda pair: pair[1])[0]


def _status_from_tasks(graph: Any) -> Dict[str, str]:
    """PlanStep id -> what the experiment tasks under it say, if anything.

    A skipped optional task does not hold back siblings that completed; when
    every child was skipped, however, the step ended without a result and must
    not keep pulsing. A task that only exists as a plan (`planned`) has not
    started it either. Everything else has: the step is at least under way,
    and when all of its tasks are done, so is it.
    """
    try:
        full = graph.full() or {}
    except Exception:  # noqa: BLE001 — a status that cannot be read is not a fault
        return {}
    kinds = {n.get("id"): n.get("type") for n in (full.get("nodes") or [])
             if isinstance(n, dict)}
    states = {n.get("id"): str(n.get("status") or "") for n in (full.get("nodes") or [])
              if isinstance(n, dict)}
    children: Dict[str, list] = {}
    for edge in (full.get("edges") or []):
        if not isinstance(edge, dict) or edge.get("type") != "elaborates":
            continue
        src, dst = edge.get("from"), edge.get("to")
        if kinds.get(src) == "ExperimentTask" and kinds.get(dst) == "PlanStep":
            children.setdefault(str(dst), []).append(states.get(src, ""))

    out: Dict[str, str] = {}
    for step, statuses in children.items():
        counted = [s for s in statuses if s != "skipped"]
        if not counted:
            if statuses and all(s == "skipped" for s in statuses):
                out[step] = "blocked"
            continue
        if all(s == "done" for s in counted):
            out[step] = "done"
        elif any(s == "running" for s in counted):
            out[step] = "in_progress"
        elif any(s == "failed" for s in counted):
            # No task is executing and at least one finished with an error.
            # Calling that state `in_progress` kept the parent card pulsing for
            # the rest of the session and concealed the failure itself.
            out[step] = "blocked"
        elif any(s == "done" for s in counted):
            # Some work is complete and another task is still merely planned.
            out[step] = "in_progress"
    return out


def _mark_step(state: Any, task: Dict[str, Any], status: str) -> None:
    """Carry a derived step status back into the tracker.

    Through `set_task_status`, which is the tracker's one writer, so the two
    state keys it keeps cannot drift apart.
    """
    task_id = str(task.get("id") or "").strip()
    if not task_id:
        return
    tracker = {"in_progress": "IN_PROGRESS", "done": "DONE",
               "blocked": "FAILED"}.get(status)
    if not tracker:
        return
    try:
        from CoScientist.tools.task_tracker import set_task_status
        set_task_status(state, task_id, tracker,
                        notes="задачи эксперимента под этим шагом "
                              + _RU_STEP.get(status, status))
    except Exception as exc:  # noqa: BLE001 — the graph is already right
        logger.warning("could not move step %s to %s: %s", task_id, status, exc)


def _live_statuses(graph: Any) -> Dict[str, str]:
    """id -> status for the steps already in the graph."""
    try:
        return {n.get("id"): n.get("status") for n in (graph.full().get("nodes") or [])
                if n.get("type") == "PlanStep"}
    except Exception:  # noqa: BLE001
        return {}


#: State key holding what the plan looked like at the last mirror.
_PLAN_FINGERPRINT_KEY = "_plan_mirror_fingerprint"


def _plan_fingerprint(tasks: Iterable[Dict[str, Any]]) -> str:
    """Id and status of every step, in order. Everything the graph copies that
    can change after the plan is registered."""
    return "|".join(
        f"{t.get('id')}:{str(t.get('status') or '').strip().lower()}"
        for t in tasks if isinstance(t, dict))


def mirror_plan_after_create(tool: BaseTool, args: Dict[str, Any],
                             tool_context: ToolContext,
                             tool_response: Any) -> None:
    """after_tool: keep the plan column level with the plan.

    Named for `create_plan` because that is where it started, and kept under
    that name because three YAML files register it — but it no longer fires
    only there. It used to, and that was the defect: the roadmap reached the
    graph once, at registration, when every step was still "todo", and a study
    that ran to completion still showed five steps nobody had started.

    Two triggers, cheapest first. A plan whose fingerprint has not moved costs
    a join and a dictionary lookup, which is what the other ~230 tool calls of
    a run get. `create_plan` is handled on its own because its own response
    carries the new list before state is read back.
    """
    try:
        name = getattr(tool, "name", "")
        state = tool_context.state
        if name == "create_plan" and isinstance(tool_response, dict):
            tasks = tool_response.get("plan") or []
        else:
            tasks = state.get("_master_active_tasks") or []
            if not tasks:
                return
            fingerprint = _plan_fingerprint(tasks)
            if state.get(_PLAN_FINGERPRINT_KEY) == fingerprint:
                return
            state[_PLAN_FINGERPRINT_KEY] = fingerprint
    except Exception as exc:  # noqa: BLE001 — mirroring must never break a tool
        logger.warning("plan mirror could not read the plan: %s", exc)
        return
    try:
        from CoScientist.graph.research.store import get_research_graph
        sync_plan_to_research_graph(
            tasks, get_research_graph(tool_context), state,
            str((state or {}).get("user_query", "")),
        )
    except Exception as exc:  # noqa: BLE001 — mirroring must never break a tool
        logger.warning("plan mirror failed: %s", exc)


def mirror_plan_before_agent(callback_context: CallbackContext):
    """before_agent: mirror the plan even when `create_plan` never fired.

    `create_plan` belongs to PlannerAgent, which ships disabled, and an operator
    can register a roadmap straight into state from the web UI — so the after_tool
    hook alone would be dead code in the default configuration. This reads the
    same list the executors read (`_master_active_tasks`), which every one of
    those paths writes. Idempotent, so both hooks may fire for one plan.
    """
    try:
        from CoScientist.graph.research.store import get_research_graph
        sync_plan_to_research_graph(
            callback_context.state.get("_master_active_tasks") or [],
            get_research_graph(callback_context), callback_context.state,
            str(callback_context.state.get("user_query", "")),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("plan mirror failed: %s", exc)
    return None


class ForbidExploreMyPapersGuard:
    """Blocks an agent from calling `explore_my_papers`.

    Used in microfluidics to prevent ResearchAgent from repeatedly reading
    user-uploaded papers, reserving `explore_my_papers` for PaperRetriever.
    """

    def guard_tool(
        self, tool: BaseTool, args: dict, tool_context: ToolContext
    ) -> Optional[dict]:
        del args
        tool_name = str(getattr(tool, "name", "") or "")
        if tool_name == "explore_my_papers":
            agent_name = str(getattr(tool_context, "agent_name", "") or "ResearchAgent")
            logger.warning(
                "[ForbidExploreMyPapersGuard] Blocked explore_my_papers call by %s",
                agent_name,
            )
            return {
                "status": "blocked",
                "blocked_by": "ForbidExploreMyPapersGuard",
                "tool": "explore_my_papers",
                "message": (
                    f"Call to `explore_my_papers` is strictly forbidden for {agent_name}. "
                    "Analysis of user-uploaded papers is performed exclusively by PaperRetriever. "
                    "Use explore_scientific_database, search_papers, or tavily_search instead."
                ),
            }
        return None
