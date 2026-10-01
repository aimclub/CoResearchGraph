"""The literature agent's summary and tables, rendered for the final report.

``literature_analysis`` (the verified structured result of module A) is JSON —
good for the design stage, unreadable to a customer. This renders it into a
Markdown section: a short summary, then tables of sources, analogues, routes
(with per-step conditions and verification status), facts and gaps. The
section is stored under ``state["literature_markdown"]`` for the report
prompt to insert VERBATIM, and posted to the chat as the literature stage's
own deliverable.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from google.adk.agents.callback_context import CallbackContext

logger = logging.getLogger(__name__)

MARKDOWN_KEY = "literature_markdown"


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _cell(value: Any, limit: int = 160) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(v) for v in value if str(v).strip())
    text = " ".join(str(value).split()).replace("|", "\\|")
    if not text:
        return "—"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        return "_нет данных_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
    return "\n".join(out) + "\n"


def _named(values: Any) -> str:
    parts = []
    for v in values or []:
        if isinstance(v, dict) and v.get("name"):
            cond = f" ({v.get('conditions')})" if v.get("conditions") else ""
            parts.append(f"{v['name']} = {v.get('value', '')}{cond}")
    return "; ".join(parts)


def _verification(item: Dict[str, Any]) -> str:
    refs = [e for e in (item.get("evidence") or []) if isinstance(e, dict)]
    if not refs:
        return "не проверено"
    statuses = [str(e.get("verification_status") or "unverified") for e in refs]
    verified = statuses.count("verified")
    if verified == len(statuses):
        return f"подтверждено ({verified})"
    if "conflicting" in statuses:
        return f"расхождения ({statuses.count('conflicting')})"
    return f"частично ({verified}/{len(statuses)})"


def _sources(item: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> str:
    seen: List[str] = []
    for e in item.get("evidence") or []:
        if isinstance(e, dict):
            rec = records.get(str(e.get("source_id") or ""))
            if rec:
                seen.append(rec.get("url") or rec.get("external_id")
                            or rec.get("title") or rec["source_id"])
    seen.extend(str(s) for s in (item.get("sources") or []))
    return "; ".join(dict.fromkeys(s for s in seen if s))


def render_literature_markdown(analysis: Any, literature_report: Any = None,
                               queries: Any = None) -> str:
    """Markdown section «Литература: саммари и таблицы» from literature_analysis."""
    a = _as_dict(analysis)
    records = {str(r.get("source_id")): r for r in (a.get("source_records") or [])
               if isinstance(r, dict) and r.get("source_id")}
    analogues = [x for x in (a.get("analogues") or []) if isinstance(x, dict)]
    routes = [x for x in (a.get("synthesis_routes") or []) if isinstance(x, dict)]
    facts = [x for x in (a.get("facts") or []) if isinstance(x, dict)]
    gaps = [str(g) for g in (a.get("gaps") or []) if str(g).strip()]
    target = a.get("target_molecule") if isinstance(a.get("target_molecule"), dict) else {}
    verified_sources = sum(1 for r in records.values() if r.get("verified_by"))

    lines: List[str] = ["## Литература: саммари и таблицы", ""]

    # ── summary ──
    lines.append("### Саммари")
    if target.get("name") or target.get("smiles"):
        fixed = "задана заказчиком" if target.get("fixed") else "выбрана по литературе"
        lines.append(
            f"- **Целевая молекула:** {target.get('name') or '—'}"
            f"{' (`' + target['smiles'] + '`)' if target.get('smiles') else ''} — {fixed}"
            f"{'; CAS ' + target['cas'] if target.get('cas') else ''}."
        )
    q_list = _as_dict(queries).get("queries") if queries else None
    if q_list:
        lines.append(f"- **Поисковых задач:** {len(q_list)} ("
                     + ", ".join(str(q.get('id')) for q in q_list if isinstance(q, dict)) + ").")
    lines.append(
        f"- **Источников:** {len(records)}, из них с проверкой полного текста: {verified_sources}."
    )
    lines.append(
        f"- **Найдено:** аналогов — {len(analogues)}, маршрутов синтеза — {len(routes)}, "
        f"фактов — {len(facts)}, пробелов — {len(gaps)}."
    )
    if routes:
        best = routes[0]
        lines.append(
            f"- **Ключевой маршрут из литературы:** {best.get('product') or '—'} — "
            f"{' → '.join(str(s.get('operation') or '') for s in best.get('steps') or [] if isinstance(s, dict)) or '—'}"
            f" (проверка: {_verification(best)})."
        )
    if literature_report:
        lines.append("")
        lines.append("**Сводка литературного оркестратора по задачам:**")
        lines.append("")
        lines.append(str(literature_report).strip())
    lines.append("")

    # ── sources ──
    lines.append("### Таблица 1. Источники")
    lines.append(_table(
        ["ID", "Тип", "Название", "URL / №", "Полный текст", "Проверен"],
        [[sid, r.get("source_type"), r.get("title"),
          r.get("url") or r.get("external_id"),
          "да" if r.get("full_text_available") else "нет",
          r.get("verified_by") or "—"] for sid, r in records.items()],
    ))

    # ── analogues ──
    lines.append("### Таблица 2. Аналоги целевого продукта")
    lines.append(_table(
        ["Вещество", "SMILES", "Класс", "Свойства", "Чем полезен для ТЗ", "Источники"],
        [[x.get("name"), x.get("smiles"), x.get("compound_class"), _named(x.get("properties")),
          x.get("relevance"), _sources(x, records)] for x in analogues],
    ))

    # ── routes ──
    lines.append("### Таблица 3. Маршруты синтеза из литературы")
    route_rows: List[List[Any]] = []
    step_rows: List[List[Any]] = []
    for i, r in enumerate(routes, 1):
        steps = [s for s in (r.get("steps") or []) if isinstance(s, dict)]
        route_rows.append([
            f"L{i}", r.get("product"), len(steps),
            " → ".join(str(s.get("operation") or "") for s in steps),
            r.get("flow_suitability"), _verification(r), _sources(r, records),
        ])
        for j, s in enumerate(steps, 1):
            step_rows.append([
                f"L{i}.{j}", s.get("operation"), s.get("reagents"), s.get("products"),
                _named(s.get("conditions")), s.get("yield_value"), _verification(s),
            ])
    lines.append(_table(
        ["№", "Продукт", "Стадий", "Операции", "Пригодность для потока", "Проверка", "Источники"],
        route_rows,
    ))
    if step_rows:
        lines.append("#### Таблица 3а. Стадии маршрутов: реагенты, продукты, условия, выход")
        lines.append(_table(
            ["Стадия", "Операция", "Реагенты", "Продукты", "Условия", "Выход", "Проверка"],
            step_rows,
        ))

    # ── facts ──
    lines.append("### Таблица 4. Факты по поисковым задачам")
    lines.append(_table(
        ["Задача", "Факт", "Проверка", "Источники"],
        [[f.get("query_id") or "—", f.get("statement"), _verification(f), _sources(f, records)]
         for f in facts],
    ))

    # ── gaps ──
    lines.append("### Пробелы (не найдено в литературе)")
    lines.extend(f"- {g}" for g in gaps) if gaps else lines.append("_нет_")
    lines.append("")
    return "\n".join(lines)


async def publish_literature_summary(callback_context: CallbackContext) -> None:
    """After EvidenceVerifierAgent: render the literature section, keep it in
    state for the report and post it to the chat.

    Posted through ``report_output`` (the agent-output sink), NOT returned as
    after_agent Content: that agent carries ``output_schema``, and ADK saves
    an after_agent Content event to ``output_key`` through the schema — the
    Markdown would then be validated as LiteratureAnalysis and fail the run.
    """
    state = callback_context.state
    analysis = state.get("literature_analysis")
    if not analysis:
        return None
    try:
        markdown = render_literature_markdown(
            analysis, state.get("literature_report"), state.get("tz_literature_queries"),
        )
    except Exception as exc:  # noqa: BLE001 — never break the run over a render
        logger.warning("literature_report: render failed: %s", exc)
        return None
    state[MARKDOWN_KEY] = markdown
    try:
        from CoScientist.logging.agent_output import report_output
        await report_output(callback_context, {
            "agent": getattr(callback_context, "agent_name", None) or "EvidenceVerifierAgent",
            "caller": "ModuleA_TZLiterature",
            "call_id": f"literature_summary-{getattr(callback_context, 'invocation_id', '')}",
            "content": markdown,
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("literature_report: could not post the summary: %s", exc)
    return None


__all__ = ["MARKDOWN_KEY", "publish_literature_summary", "render_literature_markdown"]
