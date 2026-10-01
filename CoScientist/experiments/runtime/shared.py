"""Shared Experiment Module runtime constants and soft-match helpers."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

# Set by enforce_experiment_module_first when it rewrites a first-shot Research
# call into the module (structural reroute, not an explicit orchestrator pick).
# assess_experiment_inventory_feasibility consumes this once: only an ask that
# arrived via that structural reroute is eligible for an early NO_MATCHING_TOOL;
# an orchestrator call to the module made on purpose is always trusted.
GATE_ROUTED_STATE_KEY = "_em_entered_via_gate"

# Admissions that a "success" was built on fake/proxy evidence → force partial.
FABRICATION_MARKERS = re.compile(
    r"(?i)\b("
    r"simulated|hardcoded|hard-coded|fabricated|fabrication|"
    r"placeholder|mock(?:ed)?|synthetic\s+dataset|fake\s+citation|"
    r"invented\s+(?:findings?|citations?|data)|as\s+if\s+(?:the\s+)?image|"
    r"no\s+real\s+(?:pubmed|search|api)|literature\s+data\s+was\s+simulated|"
    r"simulate_p(?:ubmed)?"
    r")\b"
)

_NAME_KEY = re.compile(r"[^a-z0-9]+")


def schema_offers_s3_upload(schema: Any) -> bool:
    """True when a tool input_schema accepts upload_results_to_s3."""
    if not isinstance(schema, dict):
        return False
    if "upload_results_to_s3" in schema:
        return True
    props = schema.get("properties")
    return isinstance(props, dict) and "upload_results_to_s3" in props


def artifact_name_key(name: str) -> str:
    """Normalize artifact basename stem for soft matching (csv ≈ candidates.csv)."""
    return _NAME_KEY.sub("", Path(str(name)).stem.lower())


def parse_fenced_json(text: str, *, prefer_list: bool = False) -> Any:
    """Parse a JSON document from LLM text, tolerating ``` fences and leading prose.

    Raises ``json.JSONDecodeError`` when no document can be decoded.
    ``prefer_list=True`` makes the leading-prose fallback look for ``[`` first.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:-1] if lines and lines[-1].strip().startswith("```") else lines[1:]
        text = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    try:
        return decoder.raw_decode(text)[0]
    except json.JSONDecodeError:
        start = text.find("[") if prefer_list else -1
        if start < 0:
            start = text.find("{")
        if start < 0:
            raise
        return decoder.raw_decode(text[start:])[0]


def session_inventory_rows(state: Any, *, scoped: bool = True) -> list[dict[str, Any]]:
    """Flatten capability rows from session state.

    ``scoped=False`` limits to this-run retrieve/discover keys (leftover
    isolation); ``scoped=True`` also includes filtered/accumulated tools.
    """
    from CoScientist.experiments.context.builder import (
        DISCOVERED_CAPABILITIES_KEY,
        RETRIEVED_CAPABILITIES_KEY,
    )

    keys: tuple[str, ...] = (RETRIEVED_CAPABILITIES_KEY, DISCOVERED_CAPABILITIES_KEY)
    if scoped:
        keys += ("filtered_tools", "accumulated_tools")
    rows: list[dict[str, Any]] = []
    getter = getattr(state, "get", None)
    if not callable(getter):
        return rows
    for key in keys:
        blob = getter(key)
        if isinstance(blob, list):
            rows.extend(item for item in blob if isinstance(item, dict))
    return rows


def audit(
    log: logging.Logger,
    message: str,
    *,
    stdout: str | None = None,
    level: int = logging.INFO,
) -> None:
    """Emit an Experiment Module audit marker.

    Writes ``message`` to the caller's logger (preserving per-module logger
    names) and mirrors it to stdout when COSCIENTIST_EXPERIMENT_AUDIT_STDOUT=1.
    ``stdout`` overrides the console line for markers whose smoke-script format
    intentionally differs from the log line (e.g. adds ``ids=[...]``).
    """
    log.log(level, message)
    if os.getenv("COSCIENTIST_EXPERIMENT_AUDIT_STDOUT") == "1":
        print(message if stdout is None else stdout, flush=True)


__all__ = [
    "FABRICATION_MARKERS",
    "GATE_ROUTED_STATE_KEY",
    "artifact_name_key",
    "audit",
    "parse_fenced_json",
    "schema_offers_s3_upload",
    "session_inventory_rows",
]
