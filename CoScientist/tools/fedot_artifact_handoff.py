"""Schema-driven handoff of tabular MCP artifacts between sequential executor steps.

An MCP tool uploads its result to S3 and returns a presigned URL (in practice a
CSV; a JSON array of records is also handled). The next step's tool may need
columns from that artifact as arguments. We fetch + parse it into
``{columns, rows}`` and, where a column name equals a tool argument name
(case-insensitive), bind those authoritative values into the task text so the
next step cannot invent placeholders. Matching is equality only — no alias map,
no domain allowlist.
"""

from __future__ import annotations

import csv
import html
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

_log = logging.getLogger(__name__)

_MAX_ROWS = 10
_MAX_BYTES = 200_000
# Presigned viz/binaries we never try to parse as a handoff table.
_SKIP_EXTENSIONS = (".html", ".htm", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".svg")


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value).strip()


def _is_html(text: str) -> bool:
    return (text or "").lstrip()[:64].lower().startswith(("<!doctype", "<html", "<head", "<?xml"))


def _table_from_records(records: Sequence[Any], *, max_rows: int = _MAX_ROWS) -> Optional[Dict[str, Any]]:
    """Build ``{columns, rows}`` from a list of record dicts (JSON shape)."""
    records = [r for r in records if isinstance(r, Mapping)][:max_rows]
    if not records:
        return None
    columns: List[str] = []
    for row in records:
        for key in row:
            if str(key) not in columns:
                columns.append(str(key))
    if not columns:
        return None
    rows = [{c: _cell(r.get(c)) for c in columns} for r in records]
    return {"columns": columns, "rows": rows, "format": "json"}


def normalize_records_to_table(payload: Any, *, max_rows: int = _MAX_ROWS) -> Optional[Dict[str, Any]]:
    """Normalize a parsed JSON payload into ``{columns, rows, format}``."""
    if isinstance(payload, list):
        return _table_from_records(payload, max_rows=max_rows) if payload else None
    if not isinstance(payload, Mapping):
        return None
    cols, rows = payload.get("columns"), payload.get("rows")
    if isinstance(cols, list) and isinstance(rows, list):
        cols = [str(c) for c in cols if str(c)]
        rows = [{c: _cell(r.get(c)) for c in cols} for r in rows[:max_rows] if isinstance(r, Mapping)]
        if cols and rows:
            return {"columns": cols, "rows": rows, "format": payload.get("format") or "json"}
    for key in ("rows", "data", "results", "items", "records"):
        nested = payload.get(key)
        if isinstance(nested, list) and nested:
            return _table_from_records(nested, max_rows=max_rows)
    return None


def parse_json_table(text: str, *, max_rows: int = _MAX_ROWS) -> Optional[Dict[str, Any]]:
    """Parse JSON text (object/array) into a handoff table, or None."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return normalize_records_to_table(payload, max_rows=max_rows)


def parse_csv_table(text: str, *, max_rows: int = _MAX_ROWS) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw or _is_html(raw) or "," not in raw.splitlines()[0]:
        return None
    try:
        reader = csv.DictReader(io.StringIO(raw, newline=""))
        columns = [c for c in (reader.fieldnames or []) if c]
        rows = [
            {c: (row.get(c) or "").strip() for c in columns}
            for i, row in enumerate(reader)
            if i < max_rows
        ]
    except csv.Error as exc:
        _log.info("artifact handoff: csv parse failed (%s)", exc)
        return None
    return {"columns": columns, "rows": rows, "format": "csv"} if columns else None


def parse_artifact_table(
    text: str, *, max_rows: int = _MAX_ROWS, path_hint: str = ""
) -> Optional[Dict[str, Any]]:
    """Best-effort parse of fetched artifact bytes into a handoff table (CSV or JSON)."""
    raw = text or ""
    if not raw.strip() or _is_html(raw):
        return None
    json_first = raw.lstrip()[:1] in "{[" or ".json" in (path_hint or "").lower()
    parsers = (
        lambda: parse_json_table(raw, max_rows=max_rows),
        lambda: parse_csv_table(raw, max_rows=max_rows),
    )
    if not json_first:
        parsers = parsers[::-1]
    for parser in parsers:
        table = parser()
        if table:
            return table
    return None


def fetch_artifact_table(
    url: str, *, max_rows: int = _MAX_ROWS, max_bytes: int = _MAX_BYTES
) -> Optional[Dict[str, Any]]:
    """Download + parse an artifact URL. Returns None on any failure.

    """
    if not url:
        return None
    try:
        import requests

        # Same as reporting.collect._download: MCP servers often HTML-escape the
        # presigned URL; downloading the literal string corrupts SigV4 params.
        resp = requests.get(html.unescape(url), timeout=20)
        resp.raise_for_status()
        raw = resp.content[:max_bytes].decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        _log.info("artifact handoff: fetch failed for %s (%s)", url[:120], exc)
        return None
    table = parse_artifact_table(raw, max_rows=max_rows, path_hint=url.split("?", 1)[0])
    if table is not None:
        table["url"] = url
    return table


def arg_names_from_input_schema(schema: Mapping[str, Any] | None) -> List[str]:
    """Tool arg names that may consume upstream columns (prefer ``required``)."""
    props = (schema or {}).get("properties") if schema else None
    if not isinstance(props, Mapping):
        return []
    required = schema.get("required") if isinstance(schema.get("required"), list) else None
    return [str(r) for r in required if str(r) in props] if required else list(props)


def _projection_arg_names(
    tables: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
) -> List[str]:
    """Schema arg names ∪ table headers (casefold-unique; headers keep their case)."""
    names: List[str] = []
    seen: set[str] = set()
    for cand in (
        *(name for tool in tools or [] for name in arg_names_from_input_schema(tool.get("input_schema"))),
        *(col for table in tables or [] for col in table.get("columns") or []),
    ):
        if str(cand).casefold() not in seen:
            seen.add(str(cand).casefold())
            names.append(str(cand))
    return names


def project_tables_to_schema_args(
    tables: Sequence[Mapping[str, Any]],
    arg_names: Sequence[str],
    *,
    max_values_per_arg: int = _MAX_ROWS,
) -> Dict[str, List[str]]:
    """Map table columns → arg names via casefold equality only."""
    if not tables or not arg_names:
        return {}
    want = {a.casefold(): a for a in arg_names}
    out: Dict[str, List[str]] = {}
    for table in tables:
        col_map = {str(c).casefold(): str(c) for c in (table.get("columns") or [])}
        rows = table.get("rows") or []
        for cf, arg in want.items():
            col = col_map.get(cf)
            if col is None:
                continue
            bucket = out.setdefault(arg, [])
            for row in rows:
                if len(bucket) >= max_values_per_arg:
                    break
                val = str(row.get(col) or "").strip() if isinstance(row, Mapping) else ""
                if val and val not in bucket:
                    bucket.append(val)
    return {k: v for k, v in out.items() if v}


def format_upstream_inputs(projected: Mapping[str, Sequence[str]]) -> str:
    if not projected:
        return ""
    lines = [
        "Upstream artifact inputs (from previous MCP structured results).",
        "For each key below, tool arguments with the same name MUST use ONLY these values.",
        "Do not invent replacements for these keys.",
    ]
    for key, values in projected.items():
        lines.append(f"{key}:")
        lines.extend(f"  - {v}" for v in values)
    return "\n".join(lines)


def bind_upstream_inputs_to_task(
    task_description: str,
    tables: Sequence[Mapping[str, Any]],
    filtered_tools: Sequence[Mapping[str, Any]],
) -> str:
    """Append authoritative upstream values when schema↔column projection is non-empty."""
    projected = project_tables_to_schema_args(tables, _projection_arg_names(tables, filtered_tools))
    block = format_upstream_inputs(projected)
    if not block or block in (task_description or ""):
        return task_description
    return f"{(task_description or '').rstrip()}\n\n## UPSTREAM ARTIFACT INPUTS (authoritative)\n{block}\n"


def _artifact_table(art: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve one artifact to ``{columns, rows, format}``: embedded → inline → fetch."""
    if not isinstance(art, Mapping):
        return None
    embedded = art.get("table")
    if isinstance(embedded, Mapping) and embedded.get("columns"):
        return {
            "columns": list(embedded.get("columns") or []),
            "rows": list(embedded.get("rows") or []),
            "format": embedded.get("format"),
        }
    url = art.get("url")
    if not url:
        for key in ("records", "rows", "results", "items", "data"):
            inline = art.get(key)
            if isinstance(inline, list) and inline:
                return normalize_records_to_table(inline)
        return None
    hint = f"{art.get('s3_key') or ''} {str(url).split('?', 1)[0]}".lower()
    if any(ext in hint for ext in _SKIP_EXTENSIONS):
        return None
    return fetch_artifact_table(str(url))


def materialize_tables_from_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
    existing: Sequence[Mapping[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Fetch/parse tables for artifacts not already materialized.

    Also embeds a compact ``table`` payload onto each artifact dict (in-place)
    so the handoff survives even if the parallel ``fedot_artifact_tables``
    state key is dropped between AgentTool invocations.
    """
    tables: List[Dict[str, Any]] = [dict(t) for t in (existing or [])]
    seen_urls = {t.get("url") for t in tables if t.get("url")}
    for art in artifacts or []:
        url = art.get("url") if isinstance(art, Mapping) else None
        if url and url in seen_urls:
            continue
        table = _artifact_table(art)
        if table is None:
            continue
        compact = {
            "columns": list(table.get("columns") or []),
            "rows": list(table.get("rows") or []),
            "format": table.get("format"),
        }
        if isinstance(art, MutableMapping):
            art["table"] = compact
        tables.append({"url": url, **compact} if url else dict(compact))
        if url:
            seen_urls.add(url)
    return tables


def tables_from_state(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Load tables from the dedicated key or embedded ``fedot_artifacts[].table``."""
    tables = [dict(t) for t in (state.get("fedot_artifact_tables") or []) if isinstance(t, Mapping)]
    return tables or materialize_tables_from_artifacts(state.get("fedot_artifacts") or [], [])


def tables_from_resolved_inputs(
    resolved_inputs: Sequence[Mapping[str, Any]] | None,
    *,
    max_rows: int = _MAX_ROWS,
    max_bytes: int = _MAX_BYTES,
) -> List[Dict[str, Any]]:
    """Materialize EM ``resolved_inputs`` URLs/workspace paths into handoff tables.

    Prefer Experiment Module lineage over ambient session artifacts so tox /
    property steps consume generated molecules, not placeholders.
    """
    tables: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in resolved_inputs or []:
        if not isinstance(item, Mapping):
            continue
        url = item.get("resolved_url")
        if isinstance(url, str) and url.strip():
            key = url.strip()
            if key in seen:
                continue
            table = fetch_artifact_table(key, max_rows=max_rows, max_bytes=max_bytes)
            if table is None:
                continue
            seen.add(key)
            tables.append(table)
            continue
        path = item.get("resolved_workspace_path")
        if not isinstance(path, str) or not path.strip():
            continue
        key = path.strip()
        if key in seen:
            continue
        try:
            raw = Path(key).read_bytes()[:max_bytes].decode("utf-8", "replace")
        except OSError as exc:
            _log.info("artifact handoff: workspace read failed for %s (%s)", key, exc)
            continue
        table = parse_artifact_table(raw, max_rows=max_rows, path_hint=key)
        if table is None:
            continue
        table["workspace_path"] = key
        seen.add(key)
        tables.append(table)
    return tables


def seed_upstream_from_resolved_inputs(
    state: MutableMapping[str, Any],
    resolved_inputs: Sequence[Mapping[str, Any]] | None,
    filtered_tools: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, List[str]]:
    """Write EM resolved tables into state and return projected schema bindings.

    When lineage tables exist they replace ambient ``fedot_artifact_tables`` for
    projection so Fedot/React cannot invent placeholders from stale session CSVs.
    """
    em_tables = tables_from_resolved_inputs(resolved_inputs)
    if not em_tables:
        return {}
    state["fedot_artifact_tables"] = em_tables
    tools = list(filtered_tools if filtered_tools is not None else state.get("filtered_tools") or [])
    projected = project_tables_to_schema_args(em_tables, _projection_arg_names(em_tables, tools))
    block = format_upstream_inputs(projected)
    state["upstream_artifact_inputs"] = block
    if projected:
        _log.info(
            "artifact handoff: seeded from resolved_inputs args=%s",
            {k: len(v) for k, v in projected.items()},
        )
    return projected


def inject_upstream_artifacts(callback_context) -> None:
    """before_agent: expose projected upstream inputs in session state for prompts."""
    state: MutableMapping[str, Any] = callback_context.state
    tables = tables_from_state(state)
    if tables and not state.get("fedot_artifact_tables"):
        state["fedot_artifact_tables"] = tables
    tools = list(state.get("filtered_tools") or [])
    projected = project_tables_to_schema_args(tables, _projection_arg_names(tables, tools))
    state["upstream_artifact_inputs"] = format_upstream_inputs(projected)
    if projected:
        _log.info("artifact handoff: projected args=%s", {k: len(v) for k, v in projected.items()})
