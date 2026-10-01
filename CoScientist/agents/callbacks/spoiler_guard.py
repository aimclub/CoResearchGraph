"""Blind-review guard: a validation run must not read its own answer.

A claim that the literature has already refuted is handed to the system as if
it were open. The system may search the web and the paper index, so the
refutation is one query away. This guard sits on the search-capable agents
(``agents/blind.yaml``) and works in three layers:

1. a date cutoff: OpenAlex queries get ``publication_year`` forced to the
   review window, and any result that carries a later date is dropped;
2. a blocklist: a query naming a listed term is refused with a message the
   agent can act on, and a result block naming one is dropped (structured
   results) or redacted sentence by sentence (plain text);
3. an LLM judge: the blocks that passed the first two layers are read once
   more by a model asked one question, "does this block reveal the outcome of
   checking the claim", and flagged blocks are dropped.

Every intervention is appended to ``state["_spoiler_guard_log"]`` and logged
under ``[SpoilerGuard]`` so the trace shows what the run was kept from seeing.

The after_tool hook edits the tool response in place and returns None: ADK
stops the callback chain at the first non-None return, and the counters and
the link registry behind this guard must still run.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_STATE_KEY = "_spoiler_guard_log"
REDACTED = "[removed by the blind-review guard]"
_GUARDED_TOOL = re.compile(r"(search|extract|crawl|paper|explore|download|fetch)", re.I)
_DATE_KEYS = ("published_date", "publication_date", "publication_year", "year", "date", "published")
_LIST_KEYS = ("results", "papers", "items", "works", "data", "documents")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-Я0-9\"'(\[])")


def _flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(f"{k}\n{_flatten(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return "\n".join(_flatten(v) for v in value)
    if hasattr(value, "model_dump"):
        try:
            return _flatten(value.model_dump())
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _year_of(item: Dict[str, Any]) -> Optional[int]:
    for key in _DATE_KEYS:
        raw = item.get(key)
        if raw is None:
            continue
        m = re.search(r"(19|20)\d{2}", str(raw))
        if m:
            return int(m.group(0))
    return None


def guard_log(state: Any) -> List[Dict[str, Any]]:
    try:
        raw = state.get(LOG_STATE_KEY)
    except Exception:  # noqa: BLE001
        return []
    return list(raw) if isinstance(raw, list) else []


class SpoilerGuard:
    def __init__(
        self,
        *,
        blocklist: List[str],
        cutoff_year: Optional[int] = None,
        claim: str = "",
        judge: bool = True,
        judge_model: Optional[str] = None,
        judge_timeout: float = 45.0,
        judge_max_items: int = 25,
        judge_max_chars: int = 1500,
    ) -> None:
        self.terms = [t.strip() for t in blocklist if t and t.strip()]
        self._patterns = [re.compile(re.escape(t), re.I) for t in self.terms]
        self.cutoff_year = cutoff_year
        self.claim = claim
        self.judge_enabled = judge
        self.judge_model = judge_model
        self.judge_timeout = judge_timeout
        self.judge_max_items = judge_max_items
        self.judge_max_chars = judge_max_chars

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def applies(tool_name: str) -> bool:
        name = tool_name or ""
        if name.startswith("research_") or name in ("retrieve_tools", "get_server_info"):
            return False
        return bool(_GUARDED_TOOL.search(name))

    def hit(self, text: str) -> Optional[str]:
        for term, pat in zip(self.terms, self._patterns):
            if pat.search(text or ""):
                return term
        return None

    def _too_late(self, item: Dict[str, Any]) -> Optional[int]:
        if self.cutoff_year is None:
            return None
        year = _year_of(item)
        if year is not None and year > self.cutoff_year:
            return year
        return None

    @staticmethod
    def _log(tool_context: Any, entry: Dict[str, Any]) -> None:
        entry = {"t": round(time.time(), 3), **entry}
        logger.warning("[SpoilerGuard] %s", json.dumps(entry, ensure_ascii=False)[:600])
        try:
            state = tool_context.state
            log = list(state.get(LOG_STATE_KEY) or [])
            log.append(entry)
            state[LOG_STATE_KEY] = log
        except Exception:  # noqa: BLE001
            pass

    # ── before_tool ──────────────────────────────────────────────────────────
    def guard_query(self, tool: Any, args: Dict[str, Any], tool_context: Any) -> Optional[Dict[str, Any]]:
        name = getattr(tool, "name", "") or ""
        if not self.applies(name):
            return None
        term = self.hit(_flatten(args))
        if term:
            self._log(tool_context, {"kind": "query_blocked", "tool": name, "term": term,
                                     "args": _flatten(args)[:300]})
            return {
                "status": "blocked",
                "error": (
                    "blind-review guard: the query names material outside the review "
                    f"window ({term!r}). Rephrase it without that reference; this review "
                    "has to reach its own verdict from the original claim, its code and data."
                ),
            }
        if self.cutoff_year is not None and name == "search_papers":
            wanted = f"<{self.cutoff_year + 1}"
            if args.get("publication_year") != wanted:
                args["publication_year"] = wanted
                self._log(tool_context, {"kind": "date_filter", "tool": name, "publication_year": wanted})
        return None

    # ── after_tool ───────────────────────────────────────────────────────────
    async def guard_result(self, tool: Any, args: Dict[str, Any], tool_context: Any, tool_response: Any) -> None:
        name = getattr(tool, "name", "") or ""
        if not self.applies(name):
            return None
        try:
            await self._filter_in_place(tool_response, name, tool_context)
        except Exception as exc:  # noqa: BLE001
            self._log(tool_context, {"kind": "guard_error", "tool": name, "error": repr(exc)[:200]})
        return None

    async def _filter_in_place(self, node: Any, tool: str, tool_context: Any) -> None:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if isinstance(value, str):
                    node[key] = await self._filter_text(value, tool, tool_context)
                else:
                    await self._filter_in_place(value, tool, tool_context)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                if isinstance(value, str):
                    node[i] = await self._filter_text(value, tool, tool_context)
                else:
                    await self._filter_in_place(value, tool, tool_context)

    async def _filter_text(self, text: str, tool: str, tool_context: Any) -> str:
        stripped = text.lstrip()
        if stripped.startswith(("{", "[")):
            try:
                obj = json.loads(text)
            except ValueError:
                obj = None
            if isinstance(obj, (dict, list)):
                changed = await self._filter_structured(obj, tool, tool_context)
                return json.dumps(obj, ensure_ascii=False) if changed else text
        return await self._filter_plain(text, tool, tool_context)

    async def _filter_structured(self, obj: Any, tool: str, tool_context: Any) -> bool:
        """Drop result items; returns True when something changed."""
        items: Optional[List[Any]] = None
        holder: Any = None
        key: Optional[str] = None
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict):
            for k in _LIST_KEYS:
                if isinstance(obj.get(k), list):
                    items, holder, key = obj[k], obj, k
                    break
        if items is None or not any(isinstance(it, dict) for it in items):
            # No list of result records: treat the string leaves as plain text.
            return await self._filter_leaves(obj, tool, tool_context)
        kept: List[Any] = []
        to_judge: List[Tuple[int, str]] = []
        changed = False
        for it in items:
            if not isinstance(it, dict):
                kept.append(it)
                continue
            blob = _flatten(it)
            term = self.hit(blob)
            if term:
                changed = True
                self._log(tool_context, {"kind": "result_dropped", "tool": tool, "term": term,
                                         "title": str(it.get("title") or it.get("name") or "")[:160],
                                         "url": str(it.get("url") or it.get("doi") or "")[:200]})
                continue
            late = self._too_late(it)
            if late:
                changed = True
                self._log(tool_context, {"kind": "date_dropped", "tool": tool, "year": late,
                                         "title": str(it.get("title") or it.get("name") or "")[:160],
                                         "url": str(it.get("url") or it.get("doi") or "")[:200]})
                continue
            to_judge.append((len(kept), blob[: self.judge_max_chars]))
            kept.append(it)
        flagged = await self._judge([t for _, t in to_judge]) if to_judge else set()
        if flagged:
            changed = True
            drop = {to_judge[i][0] for i in flagged}
            for idx in sorted(drop, reverse=True):
                it = kept[idx]
                self._log(tool_context, {"kind": "judge_dropped", "tool": tool,
                                         "title": str(it.get("title") or it.get("name") or "")[:160] if isinstance(it, dict) else "",
                                         "url": str(it.get("url") or it.get("doi") or "")[:200] if isinstance(it, dict) else ""})
                del kept[idx]
        if changed:
            if holder is not None and key is not None:
                holder[key] = kept
                for k in ("count", "total", "n_results", "num_results"):
                    if isinstance(holder.get(k), int):
                        holder[k] = len(kept)
            else:
                obj[:] = kept
        return changed

    async def _filter_leaves(self, obj: Any, tool: str, tool_context: Any) -> bool:
        changed = False
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if isinstance(v, str):
                    new = await self._filter_plain(v, tool, tool_context)
                    if new != v:
                        obj[k] = new
                        changed = True
                elif isinstance(v, (dict, list)):
                    changed = await self._filter_leaves(v, tool, tool_context) or changed
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str):
                    new = await self._filter_plain(v, tool, tool_context)
                    if new != v:
                        obj[i] = new
                        changed = True
                elif isinstance(v, (dict, list)):
                    changed = await self._filter_leaves(v, tool, tool_context) or changed
        return changed

    async def _filter_plain(self, text: str, tool: str, tool_context: Any) -> str:
        if not text or len(text) < 3:
            return text
        if self.hit(text):
            parts = _SENTENCE.split(text)
            out = []
            n = 0
            for sent in parts:
                term = self.hit(sent)
                if term:
                    out.append(REDACTED)
                    n += 1
                else:
                    out.append(sent)
            self._log(tool_context, {"kind": "redacted", "tool": tool, "sentences": n})
            text = " ".join(out)
        if self.judge_enabled and len(text) > 200:
            flagged = await self._judge([text[: self.judge_max_chars * 4]])
            if flagged:
                self._log(tool_context, {"kind": "judge_dropped", "tool": tool, "chars": len(text)})
                return REDACTED + " (the block described the outcome of this review)"
        return text

    # ── judge ────────────────────────────────────────────────────────────────
    async def _judge(self, blocks: List[str]) -> set:
        if not self.judge_enabled or not blocks:
            return set()
        blocks = blocks[: self.judge_max_items]
        try:
            import litellm
            from CoScientist.config import get_settings

            model = self.judge_model or get_settings().llm.main_model
            numbered = "\n\n".join(f"[{i}] {b}" for i, b in enumerate(blocks))
            system = (
                "You are the blind-review guard of an automated science system. The system "
                f"is independently re-checking this claim: {self.claim or '(claim not named)'}. "
                "It must not learn from search results whether later work confirmed or refuted "
                "the claim, which later methods or papers did so, or how the claim fared in "
                "reproductions. You receive numbered text blocks. Return JSON of the form "
                '{"spoilers": [indices]} listing every block that reveals such an outcome: a later '
                "paper or method reported to beat, refute, reproduce or question the claim, a "
                "comparison of the claimed method with simpler alternatives, or discussion of the "
                "claim's reproducibility or leakage. Blocks that describe the original method, its "
                "data, its code, or unrelated topics are not spoilers. Return only the JSON."
            )
            async with asyncio.timeout(self.judge_timeout):
                resp = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": numbered}],
                    temperature=0.0,
                    max_tokens=400,
                    timeout=self.judge_timeout,
                    num_retries=0,
                )
            raw = resp["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}
            idx = data.get("spoilers") or []
            return {int(i) for i in idx if isinstance(i, (int, float, str)) and str(i).isdigit() and int(i) < len(blocks)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SpoilerGuard] judge unavailable, blocks pass unjudged: %r", exc)
            return set()


_GUARD: Optional[SpoilerGuard] = None


def guard_from_settings() -> SpoilerGuard:
    """One guard per process, built from ``settings.blind``."""
    global _GUARD
    if _GUARD is None:
        from CoScientist.config import get_settings

        cfg = get_settings().blind
        _GUARD = SpoilerGuard(
            blocklist=list(cfg.blocklist),
            cutoff_year=cfg.cutoff_year,
            claim=cfg.claim,
            judge=cfg.judge,
            judge_model=cfg.judge_model,
            judge_timeout=cfg.judge_timeout,
            judge_max_items=cfg.judge_max_items,
            judge_max_chars=cfg.judge_max_chars,
        )
        if not cfg.enabled:
            logger.warning("[SpoilerGuard] BLIND__ENABLED is false: the guard is wired but passes everything")
            _GUARD.terms, _GUARD._patterns, _GUARD.cutoff_year, _GUARD.judge_enabled = [], [], None, False
    return _GUARD


def reset_guard() -> None:
    global _GUARD
    _GUARD = None


__all__ = ["SpoilerGuard", "guard_from_settings", "guard_log", "reset_guard", "LOG_STATE_KEY", "REDACTED"]
