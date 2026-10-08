# -*- coding: utf-8 -*-
"""Slide-style render of a research graph (1280-wide dark deck slide).

Reads a research graph (the `research_active.json` snapshot or a live store) and
draws it the way a research summary is actually presented: a context band on top
(question / framing / target metric), one ROW PER HYPOTHESIS reading left to
right as its story — hypothesis → verification method (+ tools, + acceptance
criterion) → evidence → conclusion — status counters above, and an overall
outcome plus the tool inventory at the bottom.

Where the graph has no data for a panel the panel is still drawn, marked as
"не задано" — a visible gap is what prompts the operator (HITL) to fill it in,
which is better than silently rendering an empty deck.

Usage:
    python -m CoScientist.graph.research.slide_render <research_active.json> [out.svg]
"""
from __future__ import annotations

import collections
import html
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# ── design tokens ────────────────────────────────────────────────────────────
W = 1280
BG, SLIDE = "#05070b", "#0e1117"
INK, MUTED, DIM = "#dfe7f0", "#9aa7b8", "#5b6676"
NEUTRAL_FILL, NEUTRAL_STROKE = "#161b26", "#2b3341"
BLUE, BLUE_FILL = "#58a6ff", "#12233a"
AMBER, AMBER_FILL = "#d29922", "#2a2110"
ORANGE, ORANGE_FILL = "#d98829", "#2a1e10"
RED, RED_FILL = "#f0554d", "#2a1517"
GREEN, GREEN_FILL = "#3fb950", "#0f2619"
YELLOW, YELLOW_FILL = "#d4a017", "#2a2210"
FONT = "Arial, Helvetica, sans-serif"

# hypothesis status -> (stroke, fill, glyph, human label)
H_STATUS = {
    "confirmed": (GREEN, GREEN_FILL, "✓", "подтверждена"),
    "refuted": (RED, RED_FILL, "✗", "опровергнута"),
    "under_verification": (YELLOW, YELLOW_FILL, "⧗", "на проверке"),
    "formulated": (NEUTRAL_STROKE, NEUTRAL_FILL, "○", "сформулирована"),
    "postponed": (NEUTRAL_STROKE, NEUTRAL_FILL, "⏸", "отложена"),
}
# Labels come in two languages: the live view is read in Russian, a paper figure
# has to be in English. Selected with SLIDE_LANG=en (or lang= on render_slide).
_L = {
    "ru": {"confirmed": "подтверждено", "refuted": "опровергнуто",
           "under_verification": "на проверке", "formulated": "сформулировано",
           "question": "Вопрос", "framing": "Постановка", "metric": "Целевая метрика",
           "hypothesis": "Гипотеза", "method": "Метод", "evidence": "Свидетельство",
           "conclusion": "Вывод", "outcome": "Итог", "tools": "Инструменты",
           "tested_by": "проверяется", "yields": "даёт", "based_on": "основание",
           "stats": "{h} гипотез · {e} свидетельств · {c} выводов · {n} узлов",
           "backlog": "Отложенный бэклог ({n})",
           "completion": "завершение: {v}",
           "no_method": "метод не определён — гипотеза отложена",
           "no_ev": {"planned": "метод ещё не запускался — свидетельств нет",
                     "running": "метод выполняется — свидетельств пока нет",
                     "failed": "метод завершился ошибкой — свидетельств нет",
                     "done": "метод отработал, но свидетельство не записано",
                     "": "свидетельств пока нет"},
           "no_cl": "свидетельств нет — выводить не из чего",
           "no_cl2": "свидетельства есть, вывод ещё не сформулирован",
           "hitl_yes": "уточнена с пользователем (HITL)",
           "hitl_no": "не подтверждена пользователем",
           "no_tools": "инструменты не объявлены",
           "form": "Форма", "domain": "Область", "gap": "Пробел", "trl": "УГТ"},
    "en": {"confirmed": "confirmed", "refuted": "refuted",
           "under_verification": "under verification", "formulated": "formulated",
           "question": "Research question", "framing": "Framing", "metric": "Target criteria",
           "hypothesis": "Hypothesis", "method": "Method", "evidence": "Evidence",
           "conclusion": "Conclusion", "outcome": "Outcome", "tools": "Tools",
           "tested_by": "tested by", "yields": "yields", "based_on": "based on",
           "stats": "{h} hypotheses · {e} evidence · {c} conclusions · {n} nodes",
           "backlog": "Postponed backlog ({n})",
           "completion": "completion: {v}", "no_method": "no method — hypothesis postponed",
           "no_ev": {"planned": "method not started — no evidence",
                     "running": "method running — no evidence yet",
                     "failed": "method failed — no evidence",
                     "done": "method finished without recording evidence",
                     "": "no evidence yet"},
           "no_cl": "no evidence — nothing to conclude from",
           "no_cl2": "evidence on file, conclusion not written yet",
           "hitl_yes": "confirmed with the operator (HITL)",
           "hitl_no": "not confirmed by the operator",
           "no_tools": "no tools declared",
           "form": "Form", "domain": "Domain", "gap": "Gap", "trl": "TRL"},
}
_LANG = os.getenv("SLIDE_LANG", "ru")


def T(key: str):
    return _L.get(_LANG, _L["ru"]).get(key, _L["ru"].get(key, key))


COUNTERS = [("confirmed", GREEN), ("refuted", RED),
            ("under_verification", YELLOW), ("formulated", NEUTRAL_STROKE)]

E = lambda s: html.escape(str(s), quote=False)

# ── text metrics (width-aware wrap; the deck is dense, so this must be tight) ─
_NAR = set("iljtfI.,:;'|!()[]{} -·")
_WID = set("mwMWШЩЮ—@%#")
# Cyrillic runs materially wider than Latin at the same size (measured on
# DejaVu Sans: lowercase 0.664 em vs 0.562, uppercase 0.765 vs 0.673). The
# original table used the Latin figure for everything, so Russian text was
# under-measured by ~24% and overflowed its cards. These are set a little below
# the DejaVu measurements because the deck renders in Arial/Helvetica, whose
# Cyrillic is narrower.
_CYR_WID = set("щшжфюым")


def _is_cyr(c: str) -> bool:
    return "Ѐ" <= c <= "ӿ"


def _cw(c: str, fs: float) -> float:
    if c in _NAR:
        return fs * 0.30
    if c in _WID or c in _CYR_WID:
        return fs * 0.86
    if _is_cyr(c):
        return fs * (0.72 if c.isupper() else 0.60)
    if c.isupper():
        return fs * 0.66
    return fs * 0.535


#: The per-class table was fitted against Arial, but a figure exported to PDF is
#: rasterised with the fallback (DejaVu Sans), which is ~9% wider — enough for a
#: line that "fits" to run past its card in print. Calibrated against DejaVu so
#: the exported figure is right; Arial in the browser then has slack, which is
#: the safe direction to be wrong in.
_WIDTH_CALIBRATION = 1.10


def _tw(s: str, fs: float) -> float:
    return sum(_cw(c, fs) for c in s) * _WIDTH_CALIBRATION


def wrap(text: str, budget: float, fs: float, max_lines: int) -> List[str]:
    lines: List[str] = []
    cur = ""
    for word in (text or "").split():
        while _tw(word, fs) > budget:
            k = 1
            while k < len(word) and _tw(word[: k + 1], fs) <= budget:
                k += 1
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:k])
            word = word[k:]
        cand = word if not cur else cur + " " + word
        if _tw(cand, fs) <= budget:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and _tw(lines[-1] + "…", fs) > budget:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "…"
    return lines


# ── graph reading ────────────────────────────────────────────────────────────
class Graph:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {n["id"]: n for n in data.get("nodes", [])}
        self.edges: List[Dict[str, Any]] = data.get("edges", [])
        self.root: Optional[str] = data.get("root_id")

    def by_type(self, *types: str) -> List[Dict[str, Any]]:
        t = {x.lower() for x in types}
        out = [n for n in self.nodes.values() if (n.get("type") or "").lower() in t]
        return sorted(out, key=lambda n: (len(re.sub(r"\d", "", n["id"])), n["id"]))

    def linked(self, node_id: str, etype: str, *, incoming: bool = False,
               types: Tuple[str, ...] = ()) -> List[Dict[str, Any]]:
        want = {t.lower() for t in types}
        out = []
        for e in self.edges:
            if e.get("type") != etype:
                continue
            src, dst = e.get("from"), e.get("to")
            other = src if (incoming and dst == node_id) else (dst if (not incoming and src == node_id) else None)
            if other and other in self.nodes:
                n = self.nodes[other]
                if not want or (n.get("type") or "").lower() in want:
                    out.append(n)
        return out

    def any_linked(self, node_id: str, etypes: Tuple[str, ...], *, incoming: bool = False,
                   types: Tuple[str, ...] = ()) -> List[Dict[str, Any]]:
        seen, out = set(), []
        for et in etypes:
            for n in self.linked(node_id, et, incoming=incoming, types=types):
                if n["id"] not in seen:
                    seen.add(n["id"])
                    out.append(n)
        return out


def a(node: Optional[Dict[str, Any]], *keys: str) -> str:
    """First non-empty attr among `keys`."""
    if not node:
        return ""
    attrs = node.get("attrs") or {}
    for k in keys:
        v = attrs.get(k)
        if v not in (None, "", [], {}):
            return str(v)
    return ""


def human_touched(node: Optional[Dict[str, Any]]) -> bool:
    """Did a human contribute this node? (HITL co-building marks it.)"""
    if not node:
        return False
    attrs = node.get("attrs") or {}
    if str(node.get("source", "")).lower() in ("human", "user", "operator"):
        return True
    for key in ("hitl", "human_refined", "refined_by_human", "confirmed_by_human"):
        if attrs.get(key):
            return True
    return False


# ── svg primitives ───────────────────────────────────────────────────────────
def card(x: float, y: float, w: float, h: float, fill: str, stroke: str) -> str:
    return (f'<g transform="translate({x:.0f},{y:.0f})">'
            f'<rect width="{w:.0f}" height="{h:.0f}" rx="10" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.7"/>')


#: Emoji need a colour-emoji font. The browser has one; the rasteriser that
#: exports a figure to PDF does not, and every icon comes out as a tofu box.
#: SLIDE_NO_EMOJI=1 drops them and reclaims the space for the title.
_NO_EMOJI = os.getenv("SLIDE_NO_EMOJI", "") not in ("", "0", "false")


def head(glyph: str, gcolor: str, emoji: str, title: str) -> str:
    out = (f'<text x="13" y="21" font-size="10.5" fill="{gcolor}" '
           f'font-family="{FONT}">{E(glyph)}</text>')
    if not _NO_EMOJI:
        out += f'<text x="28" y="21" font-size="10" font-family="{FONT}">{E(emoji)}</text>'
    x = 28 if _NO_EMOJI else 45
    return out + (f'<text x="{x}" y="21" font-size="11" font-weight="700" fill="{INK}" '
                  f'font-family="{FONT}">{E(title)}</text>')


def body(lines: List[str], y0: float = 38.0, fs: float = 9.6, lh: float = 12.4,
         color: str = INK) -> str:
    return "".join(
        f'<text x="13" y="{y0 + i * lh:.1f}" font-size="{fs}" fill="{color}" '
        f'font-family="{FONT}">{E(ln)}</text>' for i, ln in enumerate(lines))


def note(text: str, y: float, color: str = DIM, fs: float = 9.0) -> str:
    return (f'<text x="13" y="{y:.1f}" font-size="{fs}" fill="{color}" '
            f'font-family="{FONT}">{E(text)}</text>')


def chip(x: float, y: float, label: str, w: float = 0, fs: float = 8.6,
         max_w: float = 0) -> str:
    # Tool names are free text ("RDKit Ertl sascorer (эталонный SA)") and used to
    # run straight out of the method card and over the arrow label beside it.
    if max_w and _tw(label, fs) + 16 > max_w:
        while label and _tw(label + "…", fs) + 16 > max_w:
            label = label[:-1]
        label = (label + "…") if label else ""
    w = w or _tw(label, fs) + 16
    return (f'<g transform="translate({x:.0f},{y:.0f})">'
            f'<rect width="{w:.0f}" height="17" rx="8" fill="{NEUTRAL_FILL}" '
            f'stroke="{NEUTRAL_STROKE}" stroke-width="1"/>'
            f'<text x="8" y="12" font-size="{fs}" fill="{MUTED}" font-family="{FONT}">{E(label)}</text></g>'), w


def arrow(x1: float, y: float, x2: float, label: str, marker: str = "d",
          color: str = "#4a5462") -> str:
    mid = (x1 + x2) / 2
    return (f'<line x1="{x1:.0f}" y1="{y:.0f}" x2="{x2:.0f}" y2="{y:.0f}" stroke="{color}" '
            f'stroke-width="1.4" marker-end="url(#{marker})"/>'
            f'<text x="{mid:.0f}" y="{y - 6:.0f}" font-size="8.6" fill="{DIM}" '
            f'text-anchor="middle" font-family="{FONT}">{E(label)}</text>')


NOT_SET = "не задано — уточнить у пользователя (HITL)"


def _row_layout(g: "Graph", h: Dict[str, Any], i: int) -> Dict[str, Any]:
    """Content + measured heights for one hypothesis row (hypothesis → method →
    evidence → conclusion). Measuring before drawing lets the slide grow to fit."""
    stroke, fill, glyph, _word = H_STATUS.get(h.get("status") or "", H_STATUS["formulated"])
    htext = wrap(a(h, "formulation", "statement") or NOT_SET, 266, 9.6, 5)
    hh = max(100, 38 + len(htext) * 12.4 + 16)

    vms = g.linked(h["id"], "tested_by", types=("VerificationMethod",))
    vm = vms[0] if vms else None
    # A postponed hypothesis has no method by design; NOT_SET's "ask the operator"
    # reads as a missing input rather than a deliberate state.
    _vm_missing = (T("no_method")
                   if (h.get("status") or "") == "postponed" else NOT_SET)
    vtext = wrap(a(vm, "procedure", "method", "description", "method_type") or _vm_missing,
                 220, 9.4, 4)
    cc = g.linked(h["id"], "formulated_for", incoming=True, types=("ConfirmationCriteria",))
    ctext = a(cc[0], "success_metric", "threshold", "criterion") if cc else ""
    tools = g.any_linked(vm["id"], ("uses", "requires"), types=("Tool",)) if vm else []
    vh = max(96, 38 + len(vtext) * 12.0 + (14 if ctext else 0) + (20 if tools else 6) + 10)

    evs = g.any_linked(h["id"], ("supports", "refutes", "relates_to"), incoming=True,
                       types=("Evidence",))
    etext: List[str] = []
    # Three, not two: a hypothesis is often decided by the LAST piece of evidence
    # filed against it (an audit, a refutation), and cropping to two hid exactly
    # the one that changed the verdict.
    for ev in evs[:3]:
        # Evidence is written by the agents as metric/value/description, which the
        # earlier key list missed entirely -- real measurements rendered as
        # "not set". Lead with the number, since that is what a reader looks for.
        txt = a(ev, "content", "finding", "summary", "description", "observation")
        metric, value = a(ev, "metric"), a(ev, "value")
        if metric and value:
            txt = f"{metric} = {value}" + (f" — {txt}" if txt else "")
        elif value:
            txt = f"{value}" + (f" — {txt}" if txt else "")
        etext += wrap(txt, 206, 9.4, 3)
    if not etext:
        # NOT_SET ("уточнить у пользователя") is right for context the operator can
        # supply, but wrong here: evidence is produced by running the method, not
        # obtained by asking a human — inviting the operator to fill it in invites
        # fabrication. Say what the method is actually doing instead.
        _no = T("no_ev")
        etext = wrap(_no.get((vm or {}).get("status") or "", _no[""]), 206, 9.4, 2)
    eh = max(70, 38 + len(etext) * 12.0 + 14)

    cls: List[Dict[str, Any]] = []
    for ev in evs:
        cls += g.linked(ev["id"], "based_on", incoming=True, types=("Conclusion",))
    if not cls and cc:
        cls = g.linked(cc[0]["id"], "determines_sufficiency", types=("Conclusion",))
    # A hypothesis can be judged more than once — evidence arrives late, or a
    # first pass was made on an incomplete record. Showing the FIRST conclusion
    # then displays a superseded verdict beside the evidence that overturned it.
    # Approved beats draft, and newer beats older.
    cls = {c["id"]: c for c in cls}.values()
    cl = max(cls, key=lambda c: ((c.get("status") == "approved"),
                                 c.get("created_at") or 0, c["id"]), default=None)
    cltext = wrap(a(cl, "synthesis", "conclusion", "statement")
                  or (T("no_cl") if not evs
                      else T("no_cl2")),
                  274, 9.4, 4)
    ch = max(70, 38 + len(cltext) * 12.0 + 14)

    return dict(stroke=stroke, fill=fill, glyph=glyph, htext=htext, hh=hh,
                rationale=a(h, "rationale"), vm=vm, vtext=vtext, ctext=ctext, tools=tools,
                vh=vh, evs=evs, etext=etext, eh=eh, cl=cl, cltext=cltext, ch=ch,
                status=h.get("status") or "")


# ── the slide ────────────────────────────────────────────────────────────────
def render_slide(data: Dict[str, Any], compact: Optional[bool] = None) -> str:
    g = Graph(data)
    q = g.nodes.get(g.root) or (g.by_type("ResearchQuestion") or [None])[0]
    hyps = g.by_type("Hypothesis")

    # Compact (SLIDE_COMPACT=1, or compact=True): a postponed branch that never
    # produced evidence gets a one-line mention instead of a full empty row. In a
    # study that keeps three alternatives in the backlog those rows are most of
    # the picture and carry none of the result — unusable as a paper figure.
    if compact is None:
        compact = os.getenv("SLIDE_COMPACT", "") not in ("", "0", "false")
    backlog: List[Dict[str, Any]] = []
    if compact:
        def _idle(h):
            return (h.get("status") == "postponed"
                    and not g.any_linked(h["id"], ("supports", "refutes", "relates_to"),
                                         incoming=True, types=("Evidence",)))
        backlog = [h for h in hyps if _idle(h)]
        kept = [h for h in hyps if not _idle(h)]
        if kept:                      # never collapse to an empty slide
            hyps = kept

    # Pre-pass: lay out every row's four cards, so the footer is placed below the
    # TALLEST card actually drawn (a long method/evidence card is taller than the
    # nominal row pitch and would otherwise sit under the footer).
    rows = [_row_layout(g, h, i) for i, h in enumerate(hyps)]
    row_y = [240 + i * 140 for i in range(len(hyps))]
    last_bottom = 240
    for y, r in zip(row_y, rows):
        last_bottom = max(last_bottom, y + r["hh"], y + 26 + r["vh"] - 26,
                          y + r["vh"], y + 26 + r["eh"], y + 26 + r["ch"])
    backlog_y = int(last_bottom + 16) if backlog else 0
    foot_y = int((backlog_y + 34 if backlog else last_bottom) + 22)
    H = int(foot_y + 63 + 26)

    out: List[str] = [
        f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{W}" height="{H}" fill="{SLIDE}"/>', "<defs>",
    ]
    for mid, col in (("d", "#4a5462"), ("dr", RED), ("dy", YELLOW), ("db", BLUE), ("do", ORANGE)):
        out.append(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" '
                   f'markerHeight="6.5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
    out.append("</defs>")

    # ── status counters ──
    counts = collections.Counter((n.get("status") or "") for n in hyps)
    for i, (key, col) in enumerate(COUNTERS):
        x = 24 + i * 148
        out.append(f'<g transform="translate({x},30)">'
                   f'<rect width="140" height="46" rx="9" fill="{NEUTRAL_FILL}" '
                   f'stroke="{NEUTRAL_STROKE}" stroke-width="1.4"/>'
                   f'<text x="13" y="23" font-size="16" font-weight="700" fill="{col}" '
                   f'font-family="{FONT}">{counts.get(key, 0)}</text>'
                   f'<text x="13" y="38" font-size="9" fill="{MUTED}" font-family="{FONT}">{E(T(key))}</text></g>')
    meta = T("stats").format(h=len(hyps), e=len(g.by_type("Evidence")),
                             c=len(g.by_type("Conclusion")), n=len(g.nodes))
    out.append(f'<text x="620" y="59" font-size="9.4" fill="{DIM}" font-family="{FONT}">{E(meta)}</text>')

    # ── context band: question / framing / target metric ──
    qtext = a(q, "formulation") or NOT_SET
    out.append(card(24, 100, 470, 85, BLUE_FILL, BLUE) +
               head("○", BLUE, "❓", T("question")) +
               body(wrap(qtext, 444, 9.6, 4)) + "</g>")

    frame_bits = [(k, a(q, k)) for k in ("target_setting", "research_form", "domain", "gap", "trl")]
    frame_bits = [(k, v) for k, v in frame_bits if v]
    label_ru = {"target_setting": T("framing"), "research_form": T("form"), "domain": T("domain"),
                "gap": T("gap"), "trl": T("trl")}
    lines: List[str] = []
    for k, v in frame_bits:
        lines += wrap(f"{label_ru.get(k, k)}: {v}", 494, 9.4, 3)
    if not lines:
        lines = wrap(NOT_SET, 494, 9.4, 2)
    lines = lines[:9]
    out.append(card(514, 100, 520, 140, AMBER_FILL, AMBER) +
               head("○", AMBER, "🧩", T("framing")) +
               body(lines, fs=9.4, lh=11.4) +
               note(("" if _NO_EMOJI else "🧑‍🔬 ") + T("hitl_yes") if human_touched(q)
                    else ("" if _NO_EMOJI else "🧑‍🔬 ") + T("hitl_no"), 131) + "</g>")

    # A card labelled "target metric" must show a measurable target. It used to
    # show the question's `completion_criteria`, which is the STOPPING RULE
    # ("прагматичный" / "исчерпывающий") — read as a metric it is nonsense. The
    # real targets are the ConfirmationCriteria thresholds; the stopping rule
    # goes to a footnote, labelled for what it is.
    thresholds: List[str] = []
    for c in g.by_type("ConfirmationCriteria"):
        t = a(c, "success_metric", "threshold", "criterion")
        if t and t not in thresholds:
            thresholds.append(t)
    metric = "; ".join(thresholds[:2])
    rest = len(thresholds) - 2
    if rest > 0:
        metric += f" (+{rest} ещё)"
    completion = a(q, "completion_criteria")
    out.append(card(1054, 100, 202, 108, AMBER_FILL, AMBER) +
               head("○", AMBER, "🎯", T("metric")) +
               body(wrap(metric or NOT_SET, 176, 9.4, 4 if completion else 5),
                    fs=9.4, lh=11.6) +
               (note(wrap(T("completion").format(v=completion), 176, 8.8, 1)[0], 99, DIM, 8.8)
                if completion else "") + "</g>")

    # ── one row per hypothesis ──
    for i, (h, r) in enumerate(zip(hyps, rows)):
        y = row_y[i]
        stroke, fill, glyph = r["stroke"], r["fill"], r["glyph"]

        out.append(card(24, y, 292, r["hh"], fill, stroke) +
                   head(glyph, stroke, "💡", f'{T("hypothesis")} {i + 1}') + body(r["htext"]) +
                   (note(wrap(r["rationale"], 266, 8.8, 1)[0], r["hh"] - 9)
                    if r["rationale"] else "") + "</g>")

        seg = [card(376, y, 246, r["vh"], BLUE_FILL, BLUE),
               head("◉" if r["vm"] else "○", BLUE, "⚙", f'{T("method")} {i + 1}'),
               body(r["vtext"], fs=9.4, lh=12.0)]
        cy = 38 + len(r["vtext"]) * 12.0 + 6
        if r["ctext"]:
            seg.append(note(("" if _NO_EMOJI else "🎯 ") + wrap(r["ctext"], 210, 8.8, 1)[0], cy, MUTED, 8.8))
            cy += 14
        tx = 13
        for t in r["tools"][:3]:
            avail = 246 - 13 - tx          # keep the chip row inside the 246px card
            if avail < 46:                 # no room left for a legible chip
                break
            markup, cw = chip(tx, cy - 9, a(t, "name") or t["id"], max_w=avail)
            seg.append(markup)
            tx += cw + 6
        out.append("".join(seg) + "</g>")

        ecol, efill = (ORANGE, ORANGE_FILL) if r["evs"] else (NEUTRAL_STROKE, NEUTRAL_FILL)
        out.append(card(666, y + 26, 232, r["eh"], efill, ecol) +
                   head("●" if r["evs"] else "○", ecol, "🔬", f'{T("evidence")} {i + 1}') +
                   body(r["etext"], fs=9.4, lh=12.0) + "</g>")

        cl = r["cl"]
        ccol, cfill = ((GREEN, GREEN_FILL) if (cl and r["status"] == "confirmed")
                       else (RED, RED_FILL) if (cl and r["status"] == "refuted")
                       else (ORANGE, ORANGE_FILL) if cl else (NEUTRAL_STROKE, NEUTRAL_FILL))
        out.append(card(956, y + 26, 300, r["ch"], cfill, ccol) +
                   head(glyph if cl else "○", ccol, "📝", f'{T("conclusion")} {i + 1}') +
                   body(r["cltext"], fs=9.4, lh=12.0) + "</g>")

        out.append(arrow(320, y + r["hh"] / 2, 372, T("tested_by")))
        out.append(arrow(626, y + 26 + r["eh"] / 2, 662, T("yields")))
        out.append(arrow(902, y + 26 + r["ch"] / 2, 952, T("based_on")))

    if backlog:
        names = "; ".join(
            (a(h, "formulation", "statement") or h["id"])[:70].rstrip() + "…"
            for h in backlog[:3])
        out.append(card(24, backlog_y, 1232, 30, NEUTRAL_FILL, NEUTRAL_STROKE) +
                   f'<text x="13" y="19" font-size="9.6" fill="{MUTED}" '
                   f'font-family="{FONT}">{"" if _NO_EMOJI else "⏸ "}'
                   f'{E(T("backlog").format(n=len(backlog)))}: {E(names)}</text></g>')

    # ── outcome + tools ──
    conf = counts.get("confirmed", 0)
    refu = counts.get("refuted", 0)
    prog = counts.get("under_verification", 0)
    # The outcome is the STANDING verdict, not the oldest one on file: a study
    # judged twice (evidence arriving after a first pass) would otherwise show a
    # superseded conclusion as its headline.
    concls = sorted(g.by_type("Conclusion"),
                    key=lambda c: ((c.get("status") == "approved"),
                                   c.get("created_at") or 0, c["id"]))
    summary = (a(concls[-1], "synthesis") if concls else "") or NOT_SET
    out.append(card(24, foot_y, 900, 61, "#131a27", BLUE) +
               head("○", BLUE, "▣", T("outcome")) +
               body(wrap(summary, 858, 9.4, 2), y0=38, fs=9.4, lh=12.0) + "</g>")
    out.append(f'<text x="{24 + 900 - 6:.0f}" y="{foot_y + 21:.0f}" font-size="9" fill="{DIM}" '
               f'text-anchor="end" font-family="{FONT}">'
               f'{E(f"✓{conf} · ✗{refu} · ⧗{prog}")}</text>')

    tools_all = sorted({a(t, "name") or t["id"] for t in g.by_type("Tool")})
    ttext = " · ".join(tools_all) if tools_all else T("no_tools")
    out.append(card(940, foot_y, 316, 63, ORANGE_FILL, ORANGE) +
               head("○", ORANGE, "🔧", T("tools") + " · MCP") +
               body(wrap(ttext, 290, 9.2, 2), fs=9.2, lh=11.6) + "</g>")

    out.append("</svg>")
    return "".join(out)


def render_html(svg: str, title: str = "research graph") -> str:
    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    w, h = (m.group(1), m.group(2)) if m else (W, 720)
    return (f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>{E(title)}</title>'
            "<style>*{margin:0;padding:0;box-sizing:border-box}"
            f"html,body{{background:{BG}}}body{{display:flex;justify-content:center;padding:20px}}"
            f".slide{{width:{w}px;max-width:100%;aspect-ratio:{w}/{h};background:{SLIDE};"
            "border-radius:12px;overflow:hidden;box-shadow:0 12px 40px -18px #000}"
            "svg{display:block;width:100%;height:100%}</style></head>"
            f'<body><div class="slide">{svg}</div></body></html>')


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    src = argv[0]
    out_svg = argv[1] if len(argv) > 1 else "research_slide.svg"
    data = json.load(open(src, encoding="utf-8"))
    svg = render_slide(data)
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)
    out_html = re.sub(r"\.svg$", "", out_svg) + ".html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(render_html(svg, title=out_svg))
    print(f"wrote {out_svg} + {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
