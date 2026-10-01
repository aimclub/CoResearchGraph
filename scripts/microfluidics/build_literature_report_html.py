#!/usr/bin/env python3
"""Build an HTML view of the microfluidics literature-search run.

Reads the reproducible run artifacts and renders a single self-contained HTML
document in the visual style of ``tz_documents/TZ_svod_...html`` (the input
package), so the results can be read side by side with the requests that
produced them.

Inputs:
  - evaluation/microfluidics/report.md            (run metadata header)
  - evaluation/microfluidics/literature_results.jsonl  (80 MCP responses)
  - tz_documents/TZ_svod_20260710_141503.html     (14 original client requests)

Output:
  - evaluation/microfluidics/literature_results.html
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "evaluation/microfluidics/report.md"
JSONL = ROOT / "evaluation/microfluidics/literature_results.jsonl"
SVOD = ROOT / "tz_documents/TZ_svod_20260710_141503.html"
OUT = ROOT / "evaluation/microfluidics/literature_results.html"

SUB_LABELS = ("Keywords/terms:", "Tables:", "Images:", "Main substances:")


def esc(s: object) -> str:
    return html.escape(str(s if s is not None else ""))


def parse_report_meta(text: str) -> dict:
    meta = {}
    for key, pat in (
        ("date", r"Дата прогона:\s*(.+)"),
        ("source", r"Источник:\s*`?([^`\n]+)`?"),
        ("profile", r"Профиль:\s*`?([^`\n]+)`?"),
        ("mcp", r"MCP:\s*(.+)"),
    ):
        m = re.search(pat, text)
        if m:
            meta[key] = m.group(1).strip().replace("`", "")
    return meta


def load_request_titles() -> dict:
    """Original client-request blockquotes, one per section id=req-N."""
    doc = SVOD.read_text(encoding="utf-8")
    reqs = {}
    parts = re.split(r'<section id="req-(\d+)">', doc)
    for i in range(1, len(parts), 2):
        num = int(parts[i])
        m = re.search(r"<blockquote>(.*?)</blockquote>", parts[i + 1], re.S)
        if m:
            reqs[num] = html.unescape(re.sub(r"<[^>]+>", "", m.group(1)).strip())
    return reqs


def parsed_response(rec: dict):
    txt = rec["response"]["result"]["content"][0]["text"]
    return json.loads(txt) if isinstance(txt, str) else txt


def split_response(rec: dict) -> tuple[list, list]:
    """Return (article_dicts, note_messages) for one MCP response.

    Normal responses parse to a list of article dicts. A no-result / in-band
    error parses to a dict like ``{"answer": "..."}``; occasional stray
    elements inside a list are collected as notes too.
    """
    obj = parsed_response(rec)
    real, notes = [], []
    if isinstance(obj, dict):
        notes.append(obj.get("answer") or json.dumps(obj, ensure_ascii=False))
        return real, notes
    for a in obj if isinstance(obj, list) else []:
        if isinstance(a, dict) and "title" in a:
            real.append(a)
        elif isinstance(a, dict):
            notes.append(a.get("answer") or json.dumps(a, ensure_ascii=False))
        else:
            notes.append(str(a))
    return real, notes


def count_articles(rec: dict) -> int:
    return len(split_response(rec)[0])


def short(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def render_summary(part: str) -> str:
    """Render an article summary: abstract paragraph + labeled sub-sections."""
    chunks = [c.strip() for c in part.split("\n\n") if c.strip()]
    out = []
    for ch in chunks:
        label = next((l for l in SUB_LABELS if ch.startswith(l)), None)
        if label:
            body = ch[len(label):].strip()
            out.append(
                f'<div class="sub"><span class="sub-l">{esc(label)}</span> {esc(body)}</div>'
            )
        else:
            out.append(f'<p class="abs">{esc(ch)}</p>')
    return "\n".join(out)


def score_pct(x: float) -> str:
    try:
        return f"{float(x) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def render_article(a: dict, idx: int) -> str:
    title = esc(a.get("title") or "Без названия")
    field = a.get("field")
    domain = a.get("domain")
    rr = a.get("rerank_score")
    ini = a.get("initial_score")
    badges = []
    if field:
        badges.append(f'<span class="badge s-field">{esc(field)}</span>')
    if domain:
        badges.append(f'<span class="badge s-domain">{esc(domain)}</span>')
    if rr is not None:
        badges.append(
            f'<span class="badge s-rerank" title="rerank score">rerank {score_pct(rr)}</span>'
        )
    if ini is not None:
        badges.append(
            f'<span class="badge s-initial" title="initial retrieval score">retrieval {score_pct(ini)}</span>'
        )
    aid = esc((a.get("article_id") or "")[:12])
    summary = render_summary(a.get("summary") or "")
    return f"""<article class="card">
<div class="art-head"><span class="art-n">{idx}</span><span class="art-title">{title}</span></div>
<div class="badges">{''.join(badges)}</div>
{summary}
<div class="art-id">article_id: <code>{aid}…</code></div>
</article>"""


def render_note(notes: list) -> str:
    body = "<br>".join(esc(m) for m in notes)
    return (
        '<div class="note">В базе не найдено релевантных статей по этому запросу. '
        f"Ответ сервиса: {body}</div>"
    )


def render_lit(rec: dict) -> tuple[str, int]:
    real, notes = split_response(rec)
    lit_id = esc(rec["id"])
    task = rec.get("task") or ""
    query = rec.get("query_en") or ""
    extract = rec.get("extract") or ""
    count = len(real)
    count_lbl = f"{count} статей" if count else "нет статей"
    meta_tbl = f"""<table class="lit-meta"><tbody>
<tr><th>Задача</th><td>{esc(task)}</td></tr>
<tr><th>Поисковый запрос (EN)</th><td><code>{esc(query)}</code></td></tr>
<tr><th>Что извлечь</th><td>{esc(extract)}</td></tr>
</tbody></table>"""
    if real:
        cards = "\n".join(render_article(a, i + 1) for i, a in enumerate(real))
    else:
        cards = ""
    note = render_note(notes) if notes else ""
    body = meta_tbl + note + cards
    open_attr = "" if count else " open"
    summary_line = (
        f'<span class="lit-id">{lit_id}</span>'
        f'<span class="lit-task">{esc(short(task, 96))}</span>'
        f'<span class="lit-count">{count_lbl}</span>'
    )
    block = f"""<details class="lit"{open_attr}>
<summary>{summary_line}</summary>
{body}
</details>"""
    return block, count


def build() -> None:
    report_text = REPORT.read_text(encoding="utf-8")
    meta = parse_report_meta(report_text)
    titles = load_request_titles()
    recs = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

    by_req: "OrderedDict[int, list]" = OrderedDict()
    for r in recs:
        by_req.setdefault(r["request"], []).append(r)
    for k in by_req:
        by_req[k].sort(key=lambda r: r["id"])

    # aggregate stats
    fields = Counter()
    domains = Counter()
    total_articles = 0
    for r in recs:
        real, _ = split_response(r)
        for a in real:
            total_articles += 1
            fields[a.get("field")] += 1
            domains[a.get("domain")] += 1

    # ---- head / css (tz_svod DNA + results extensions) ----
    css = """:root{color-scheme:light dark}
*{box-sizing:border-box}
body{font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 margin:0 auto;max-width:940px;padding:2rem 1.2rem;color:#1c1c1c;background:#fff}
h1{font-size:1.55rem;margin:0 0 .2rem}
h2{font-size:1.2rem;margin:2.2rem 0 .5rem;border-bottom:1px solid #e4e4e4;padding-bottom:.3rem}
h3{font-size:1rem;margin:1.4rem 0 .3rem}
.meta{color:#6a6a6a;font-size:.85rem;margin-bottom:1.4rem}
p{margin:.5rem 0}
blockquote{margin:.4rem 0;padding:.5rem .9rem;border-left:3px solid #b7b7b7;background:#f6f6f6}
table{border-collapse:collapse;width:100%;margin:.3rem 0 1rem;font-size:.9rem}
th,td{border:1px solid #e2e2e2;padding:.4rem .6rem;text-align:left;vertical-align:top}
th{background:#f2f2f2;font-weight:600}
td.r,th.r{text-align:right}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em;
 background:#f0f0f0;padding:.05rem .3rem;border-radius:4px;overflow-wrap:anywhere}
.badge{display:inline-block;padding:.08rem .5rem;border-radius:999px;font-size:.74rem;
 white-space:nowrap;border:1px solid transparent}
.s-field{background:#e6f0fb;color:#1c5fa8;border-color:#c3ddf5}
.s-domain{background:#eae6f7;color:#5b3fb0;border-color:#d3c9ef}
.s-rerank{background:#e6f4ea;color:#1e7e34;border-color:#bfe3ca;font-weight:600}
.s-initial{background:#f0f0f0;color:#777;border-color:#ddd}
section{margin-top:2.5rem;padding-top:1rem;border-top:2px solid #ccc}
nav ol{line-height:1.7}
.legend{font-size:.82rem;color:#6a6a6a;margin:.2rem 0 1rem}
.legend .badge{margin-right:.3rem}
details.lit{border:1px solid #e2e2e2;border-radius:8px;margin:.6rem 0;background:#fbfbfb}
details.lit[open]{background:#fff}
details.lit>summary{cursor:pointer;padding:.55rem .8rem;list-style:none;
 display:flex;gap:.6rem;align-items:baseline;flex-wrap:wrap}
details.lit>summary::-webkit-details-marker{display:none}
details.lit>summary::before{content:"▸";color:#999;font-size:.8rem}
details.lit[open]>summary::before{content:"▾"}
.lit-id{font-family:ui-monospace,Menlo,monospace;font-size:.82rem;color:#1c5fa8;font-weight:600}
.lit-task{flex:1;min-width:12rem;color:#333}
.lit-count{font-size:.78rem;color:#6a6a6a;white-space:nowrap}
.lit>*:not(summary){margin-left:.8rem;margin-right:.8rem}
.lit table.lit-meta{margin:.5rem .8rem;width:calc(100% - 1.6rem)}
.lit table.lit-meta th{width:11rem;white-space:nowrap;vertical-align:top}
.card{border-left:3px solid #cfe3f7;background:#fafcff;border-radius:0 6px 6px 0;
 padding:.5rem .8rem;margin:.6rem .8rem}
.art-head{display:flex;gap:.5rem;align-items:baseline}
.art-n{font-family:ui-monospace,Menlo,monospace;font-size:.78rem;color:#8aa;min-width:1.4rem}
.art-title{font-weight:600;font-size:.95rem}
.badges{display:flex;flex-wrap:wrap;gap:.3rem;margin:.35rem 0}
.abs{margin:.35rem 0;font-size:.9rem}
.sub{font-size:.82rem;color:#555;margin:.2rem 0}
.sub-l{font-weight:600;color:#444}
.art-id{font-size:.72rem;color:#999;margin-top:.3rem}
.note{background:#fff4e0;color:#a9690a;border:1px solid #f2dcae;border-radius:6px;
 padding:.5rem .8rem;margin:.6rem .8rem;font-size:.88rem}
@media (prefers-color-scheme:dark){
 body{background:#171717;color:#e8e8e8}
 h2{border-color:#333} h3{color:#f0f0f0}
 .meta,.usage,.legend,.lit-count{color:#9a9a9a}
 blockquote{background:#222;border-color:#555}
 th{background:#242424} th,td{border-color:#333} code{background:#2a2a2a}
 .s-field{background:#132840;color:#8bb9e8;border-color:#274a70}
 .s-domain{background:#241d3d;color:#b7a4ef;border-color:#40356a}
 .s-rerank{background:#16351f;color:#7fd39a;border-color:#2c5e3a}
 .s-initial{background:#242424;color:#999;border-color:#3a3a3a}
 section{border-color:#444}
 details.lit{border-color:#333;background:#1c1c1c}
 details.lit[open]{background:#161616}
 .lit-id{color:#8bb9e8} .lit-task{color:#ccc}
 .card{border-left-color:#274a70;background:#141a22}
 .art-title{color:#f0f0f0} .abs{color:#dcdcdc} .sub{color:#a8a8a8} .sub-l{color:#c4c4c4}
 .note{background:#3a2c12;color:#e6b566;border-color:#5e4a24}
}
@media print{body{max-width:none;padding:0}details.lit{break-inside:avoid}}"""

    parts = []
    A = parts.append
    A("<!doctype html>")
    A('<html lang="ru"><head><meta charset="utf-8">')
    A('<meta name="viewport" content="width=device-width, initial-scale=1">')
    A("<title>Результаты литературного поиска · микрофлюидика</title>")
    A(f"<style>{css}</style>")
    A("</head><body>")

    A("<h1>Результаты литературного поиска по пакету «микрофлюидика»</h1>")
    A(
        f'<div class="meta">Дата прогона {esc(meta.get("date","—"))} · '
        f'источник <code>{esc(meta.get("source","—"))}</code> · '
        f'профиль «{esc(meta.get("profile","—"))}» · '
        f'MCP {esc(meta.get("mcp","—"))} · запросов в пакете: {len(by_req)}</div>'
    )
    A(
        "<p>Все целевые запросы <code>LIT-*</code> из сводного пакета ТЗ последовательно "
        "отправлены в сервис поиска литературы (RAG). Ниже — сгруппированные по исходным "
        "запросам заказчика ответы сервиса: для каждой задачи показаны найденные статьи с "
        "оценками релевантности (retrieval и rerank), областью и рефератом.</p>"
    )
    A(
        '<div class="legend">Оценки статьи: '
        '<span class="badge s-rerank">rerank</span>переранжирование под задачу · '
        '<span class="badge s-initial">retrieval</span>первичная близость поиска · '
        '<span class="badge s-field">поле</span> · '
        '<span class="badge s-domain">домен</span>. '
        "Статьи упорядочены по rerank (наиболее релевантные сверху).</div>"
    )

    # ---- summary ----
    n_lit = len(recs)
    n_ok = sum(1 for r in recs if r.get("status") == "ok")
    A('<section id="summary">')
    A("<h2>Сводка прогона</h2>")
    A("<table><tbody>")
    A(f'<tr><th>Исходных ТЗ</th><td class="r">{len(by_req)}</td></tr>')
    A(f'<tr><th>Литературных задач (LIT-*)</th><td class="r">{n_lit}</td></tr>')
    A(f'<tr><th>Успешных ответов MCP</th><td class="r">{n_ok}</td></tr>')
    A(f'<tr><th>Всего найдено статей</th><td class="r">{total_articles}</td></tr>')
    A("</tbody></table>")

    A("<h3>Разбивка по исходным запросам</h3>")
    A('<table><thead><tr><th class="r">Запрос</th><th class="r">LIT-задач</th>'
      '<th class="r">Найдено статей</th></tr></thead><tbody>')
    for req in sorted(by_req):
        recs_r = by_req[req]
        arts = sum(count_articles(r) for r in recs_r)
        A(f'<tr><td class="r"><a href="#req-{req}">{req}</a></td>'
          f'<td class="r">{len(recs_r)}</td><td class="r">{arts}</td></tr>')
    A("</tbody></table>")

    A("<h3>Распределение найденных статей по областям</h3>")
    A('<table><thead><tr><th>Поле (field)</th><th class="r">Статей</th></tr></thead><tbody>')
    for name, cnt in fields.most_common():
        A(f"<tr><td>{esc(name)}</td><td class=\"r\">{cnt}</td></tr>")
    A("</tbody></table>")

    A(
        "<p>Методика: для каждой строки <code>LIT-*</code> в сервис передавались русская "
        "формулировка задачи, англоязычный поисковый запрос <code>query_en</code> и перечень "
        "данных для извлечения; ответ записывался сразу после вызова. Сервис вернул по 10 статей "
        "на запрос; один запрос (см. Запрос 14) не дал совпадений в базе. LLM-as-judge и "
        "сравнительная оценка качества в этот прогон не включались. Артефакты: "
        "<code>evaluation/microfluidics/literature_results.jsonl</code>, "
        "<code>scripts/run_microfluidics_literature.py</code>.</p>"
    )
    A("</section>")

    # ---- TOC ----
    A('<nav><h2>Содержание</h2><ol>')
    for req in sorted(by_req):
        title = short(titles.get(req, f"Запрос {req}"), 88)
        A(f'<li><a href="#req-{req}">Запрос {req} — {esc(title)}</a></li>')
    A("</ol></nav>")

    # ---- per request ----
    for req in sorted(by_req):
        recs_r = by_req[req]
        arts = sum(count_articles(r) for r in recs_r)
        A(f'<section id="req-{req}">')
        A(f"<h2>Запрос {req}. Результаты литературного поиска</h2>")
        if req in titles:
            A(f"<blockquote>{esc(titles[req])}</blockquote>")
        A(f'<div class="meta">{len(recs_r)} задач LIT-* · найдено статей: {arts}</div>')
        for rec in recs_r:
            block, _ = render_lit(rec)
            A(block)
        A("</section>")

    A("</body></html>")

    OUT.write_text("\n".join(parts), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size_kb:.0f} KB) · {len(by_req)} requests · "
          f"{len(recs)} LIT tasks · {total_articles} articles")


if __name__ == "__main__":
    build()
