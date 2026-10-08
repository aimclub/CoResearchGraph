"""Deterministic renderer: StructuredTZ -> the human-readable ТЗ document.

Reproduces the structure of the reference document
``Пример_уточненного_структурированного_ТЗ_для_агентов.md``:

  1. заголовок (версия, дата, статус документа)
  2. правила интерпретации полей (легенда статусов)
  3. исходный запрос заказчика в свободной форме
  4. единая структура собранных данных (обзорная таблица по блокам)
  5. таблицы блоков |Поле|Значение|Статус|
  6. поля, которые остаются свободными или незаполненными
  7. проверка соответствия исходному опроснику

The renderer is pure formatting: everything it prints comes from the
validated StructuredTZ (or is computed from it), so the document can never
disagree with the machine-readable ТЗ the rest of the pipeline consumes.
"""
from __future__ import annotations

import html
import json
from datetime import date
from typing import Any, Union

from CoScientist.microfluidics.models import (
    CANONICAL_BLOCKS,
    OPEN_STATUSES,
    StructuredTZ,
)

_LEGEND = """\
| Статус поля            | Как интерпретировать                                                                                                        |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Задано заказчиком      | Конкретное значение получено от заказчика и может использоваться агентами как входное ограничение или критерий               |
| Уточнено оператором    | Значение добавлено оператором/агентом постановки ТЗ из контекста и может использоваться как рабочее ограничение              |
| Не задано              | Значение отсутствует; агент не должен его придумывать, но может запросить уточнение или выполнить поиск вариантов            |
| Свободный комментарий  | Неформализованная информация; не используется как жёсткий критерий ранжирования, пока не переведена в конкретный параметр    |
| Рассчитывается агентом | Значение должно быть определено на следующем этапе работы системы                                                            |"""

_OPEN_GUIDANCE = {
    "не задано": "Не подменять предположениями; запросить уточнение или выполнить поиск вариантов",
    "рассчитывается агентом": "Определяется на следующем этапе работы системы",
}


def _cap(status: str) -> str:
    return status[:1].upper() + status[1:] if status else status


def _cell(text: str) -> str:
    """Make arbitrary text safe inside a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _coerce(tz: Union[StructuredTZ, dict, str, Any]) -> StructuredTZ:
    if isinstance(tz, StructuredTZ):
        return tz
    if isinstance(tz, str):
        tz = json.loads(tz)
    return StructuredTZ.model_validate(tz)


def render_tz_document(tz: Union[StructuredTZ, dict, str], version: str = "v1") -> str:
    """Render the full ТЗ Markdown document from a (validated) StructuredTZ."""
    tz = _coerce(tz)
    out: list[str] = []

    # 1 ── header
    out.append("# Уточнённое структурированное ТЗ для системы ИИ-агентов")
    out.append("")
    out.append(f"Версия: {version}  ")
    out.append(f"Дата: {date.today().strftime('%d.%m.%Y')}  ")
    out.append(
        "Статус документа: сформирован агентом постановки ТЗ (CoScientist, "
        "кейс «микрофлюидика»)"
    )
    out.append("\n---\n")

    # 2 ── interpretation rules
    out.append("## Правила интерпретации полей\n")
    out.append(
        "Документ является единым структурированным входом для последующих "
        "ИИ-агентов. Для каждого поля указывается статус:\n"
    )
    out.append(_LEGEND)
    out.append(
        "\nНеконкретные формулировки («доступное сырьё», «устойчивые поставки», "
        "«приемлемая цена») не используются как жёсткие критерии: для "
        "автоматической обработки они переведены в измеримые поля или помечены "
        "как свободный комментарий."
    )
    out.append("\n---\n")

    # 3 ── original request
    out.append("## Исходный запрос заказчика в свободной форме\n")
    request = (tz.original_request or "").strip() or "Не задан"
    out.append("\n".join(f"> {line}" for line in request.splitlines() if line.strip()))
    out.append("\n---\n")

    # 4 ── overview table
    out.append("## Единая структура собранных данных\n")
    out.append("| Блок данных | Собрано в ТЗ | Заполненность | Использование далее |")
    out.append("| --- | --- | --- | --- |")
    for block in tz.blocks:
        total = len(block.fields)
        filled = sum(1 for f in block.fields if f.is_set())
        if total == 0 or filled == 0:
            collected = "Нет"
        elif filled == total:
            collected = "Да"
        else:
            collected = "Частично"
        usage = block.usage or "—"
        out.append(
            f"| {_cell(block.title)} | {collected} | {filled} из {total} полей "
            f"| {_cell(usage)} |"
        )
    out.append("\n---\n")

    # 5 ── per-block tables
    for block in tz.blocks:
        out.append(f"## {block.title.strip()}\n")
        out.append("| Поле | Значение | Статус |")
        out.append("| --- | --- | --- |")
        for f in block.fields:
            out.append(f"| {_cell(f.name)} | {_cell(f.value)} | {_cap(f.status)} |")
        out.append("\n---\n")

    # 6 ── open fields
    open_rows = [
        (block.title, f)
        for block in tz.blocks
        for f in block.fields
        if f.status in OPEN_STATUSES
    ]
    out.append("## Поля, которые остаются свободными или незаполненными\n")
    if open_rows:
        out.append(
            "Эти поля не должны подменяться предположениями агентов без "
            "отдельной пометки.\n"
        )
        out.append("| Блок | Поле | Текущее значение | Как использовать дальше |")
        out.append("| --- | --- | --- | --- |")
        for title, f in open_rows:
            out.append(
                f"| {_cell(title)} | {_cell(f.name)} | {_cell(f.value)} "
                f"| {_OPEN_GUIDANCE[f.status]} |"
            )
    else:
        out.append("Все поля ТЗ заполнены.")
    out.append("\n---\n")

    # 7 ── questionnaire coverage check
    out.append("## Проверка соответствия исходному опроснику\n")
    out.append("| Элемент опросника | Наличие в ТЗ | Комментарий |")
    out.append("| --- | --- | --- |")
    for title in CANONICAL_BLOCKS:
        block = tz.block(title)
        if block is None or not block.fields:
            out.append(f"| {title} | Отсутствует | Блок не сформирован |")
            continue
        filled = sum(1 for f in block.fields if f.is_set())
        out.append(
            f"| {title} | Есть | Заполнено {filled} из {len(block.fields)} полей |"
        )
    extra = [b.title for b in tz.blocks if b.title not in CANONICAL_BLOCKS]
    if extra:
        for title in extra:
            out.append(f"| {_cell(title)} | Дополнительный блок | Вне опросника |")

    return "\n".join(out).rstrip() + "\n"


def _query_dicts(queries: Any) -> list[dict]:
    """Normalise the ``tz_literature_queries`` state value to plain dicts.

    Accepts a ``LiteratureQueries`` model, a ``{"queries": [...]}`` dict, a bare
    list of queries, or the raw JSON string produced before the state commit.
    """
    if queries is None:
        return []
    if isinstance(queries, str):
        try:
            queries = json.loads(queries)
        except json.JSONDecodeError:
            return []
    if hasattr(queries, "model_dump"):
        queries = queries.model_dump()
    if isinstance(queries, dict):
        queries = queries.get("queries", [])
    rows: list[dict] = []
    for q in queries or []:
        if hasattr(q, "model_dump"):
            q = q.model_dump()
        if not isinstance(q, dict):
            continue
        rows.append(
            {
                "id": str(q.get("id") or q.get("query_id") or ""),
                "task": str(q.get("task") or ""),
                "query_en": str(q.get("query_en") or q.get("query") or ""),
                "extract": [str(x) for x in (q.get("extract") or [])],
            }
        )
    return rows


def render_queries_markdown(queries: Any) -> str:
    """Render the literature-agent target queries as a Markdown section."""
    rows = _query_dicts(queries)
    out = ["## Целевые запросы к литературному агенту", ""]
    if not rows:
        out.append("_Запросы к литературному агенту не сформированы._")
        return "\n".join(out) + "\n"
    out.append(
        "Задачи, выведенные из ТЗ: для каждой — англоязычный поисковый запрос и "
        "перечень данных, которые литературный агент должен извлечь."
    )
    out.append("")
    out.append("| ID | Задача | Поисковый запрос (EN) | Что извлечь |")
    out.append("| --- | --- | --- | --- |")
    for q in rows:
        extract = ", ".join(q["extract"]) if q["extract"] else "—"
        out.append(
            f"| {_cell(q['id'])} | {_cell(q['task'])} "
            f"| {_cell(q['query_en'])} | {_cell(extract)} |"
        )
    return "\n".join(out).rstrip() + "\n"


def render_tz_and_queries_markdown(
    tz: Union[StructuredTZ, dict, str], queries: Any, version: str = "v1"
) -> str:
    """Full hand-off document (Markdown): the ТЗ followed by its literature queries."""
    document = render_tz_document(tz, version=version).rstrip()
    return document + "\n\n" + render_queries_markdown(queries)


_HTML_STYLE = """\
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 margin:0 auto;max-width:940px;padding:2rem 1.2rem;color:#1c1c1c;background:#fff}
h1{font-size:1.55rem;margin:0 0 .2rem}
h2{font-size:1.2rem;margin:2.2rem 0 .5rem;border-bottom:1px solid #e4e4e4;padding-bottom:.3rem}
h3{font-size:1rem;margin:1.4rem 0 .3rem}
.meta{color:#6a6a6a;font-size:.85rem;margin-bottom:1.4rem}
.usage{color:#6a6a6a;font-size:.85rem;margin:.1rem 0 .4rem}
blockquote{margin:.4rem 0;padding:.5rem .9rem;border-left:3px solid #b7b7b7;background:#f6f6f6}
table{border-collapse:collapse;width:100%;margin:.3rem 0 1rem;font-size:.9rem}
th,td{border:1px solid #e2e2e2;padding:.4rem .6rem;text-align:left;vertical-align:top}
th{background:#f2f2f2;font-weight:600}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em;
 background:#f0f0f0;padding:.05rem .3rem;border-radius:4px}
.badge{display:inline-block;padding:.08rem .5rem;border-radius:999px;font-size:.76rem;
 white-space:nowrap;border:1px solid transparent}
.s-given{background:#e6f4ea;color:#1e7e34;border-color:#bfe3ca}
.s-op{background:#fff4e0;color:#a9690a;border-color:#f2dcae}
.s-calc{background:#eae6f7;color:#5b3fb0;border-color:#d3c9ef}
.s-comment{background:#e6f0fb;color:#1c5fa8;border-color:#c3ddf5}
.s-open{background:#f0f0f0;color:#777;border-color:#ddd}
@media (prefers-color-scheme:dark){
 body{background:#171717;color:#e8e8e8}
 h2{border-color:#333} h3{color:#f0f0f0}
 .meta,.usage{color:#9a9a9a}
 blockquote{background:#222;border-color:#555}
 th{background:#242424} th,td{border-color:#333} code{background:#2a2a2a}
 .s-given{background:#16351f;color:#7fd39a;border-color:#2c5e3a}
 .s-op{background:#3a2c12;color:#e6b566;border-color:#5e4a24}
 .s-calc{background:#241d3d;color:#b7a4ef;border-color:#40356a}
 .s-comment{background:#132840;color:#8bb9e8;border-color:#274a70}
 .s-open{background:#242424;color:#999;border-color:#3a3a3a}
}
@media print{body{max-width:none;padding:0}}
"""

_STATUS_HTML_CLASS = {
    "задано заказчиком": "s-given",
    "уточнено оператором": "s-op",
    "рассчитывается агентом": "s-calc",
    "свободный комментарий": "s-comment",
}


def _h(text: Any) -> str:
    return html.escape(str(text), quote=True)


def render_tz_and_queries_html(
    tz: Union[StructuredTZ, dict, str], queries: Any, version: str = "v1"
) -> str:
    """Full hand-off document as self-contained, theme-aware HTML."""
    tz = _coerce(tz)
    rows = _query_dicts(queries)
    request = (tz.original_request or "").strip() or "не задан"
    parts: list[str] = [
        "<!doctype html>",
        '<html lang="ru"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Структурированное ТЗ и запросы к лит-агенту</title>",
        f"<style>{_HTML_STYLE}</style></head><body>",
        "<h1>Структурированное ТЗ и целевые запросы к литературному агенту</h1>",
        f'<div class="meta">Версия {_h(version)} · {date.today():%d.%m.%Y} · '
        "CoScientist · профиль «микрофлюидика»</div>",
        "<h2>Исходный запрос заказчика</h2>",
        f"<blockquote>{_h(request).replace(chr(10), '<br>')}</blockquote>",
        "<h2>Структурированное ТЗ</h2>",
    ]
    for block in tz.blocks:
        parts.append(f"<h3>{_h(block.title.strip())}</h3>")
        if getattr(block, "usage", ""):
            parts.append(f'<div class="usage">{_h(block.usage)}</div>')
        parts.append(
            "<table><thead><tr><th>Поле</th><th>Значение</th>"
            "<th>Статус</th></tr></thead><tbody>"
        )
        for f in block.fields:
            value = (f.value or "").strip() or "—"
            cls = _STATUS_HTML_CLASS.get(f.status, "s-open")
            parts.append(
                f"<tr><td>{_h(f.name)}</td>"
                f"<td>{_h(value).replace(chr(10), '<br>')}</td>"
                f'<td><span class="badge {cls}">{_h(f.status)}</span></td></tr>'
            )
        parts.append("</tbody></table>")
    parts.append("<h2>Целевые запросы к литературному агенту</h2>")
    if not rows:
        parts.append("<p>Запросы к литературному агенту не сформированы.</p>")
    else:
        parts.append(
            "<table><thead><tr><th>ID</th><th>Задача</th>"
            "<th>Поисковый запрос (EN)</th><th>Что извлечь</th></tr></thead><tbody>"
        )
        for q in rows:
            extract = ", ".join(q["extract"]) if q["extract"] else "—"
            parts.append(
                f"<tr><td>{_h(q['id'])}</td><td>{_h(q['task'])}</td>"
                f"<td><code>{_h(q['query_en'])}</code></td>"
                f"<td>{_h(extract)}</td></tr>"
            )
        parts.append("</tbody></table>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


__all__ = [
    "render_tz_document",
    "render_queries_markdown",
    "render_tz_and_queries_markdown",
    "render_tz_and_queries_html",
]
