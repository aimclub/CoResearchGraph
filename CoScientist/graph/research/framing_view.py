"""Build the reader-facing technical-specification sidebar.

The graph stores research facts, while the confirmed ``ResearchFrame`` keeps
the original per-field values and provenance.  This adapter joins the two into
one deterministic view.  It also recovers a conservative view from old graph
snapshots that predate embedded frame snapshots.
"""
from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from CoScientist.context_init.models import ResearchFrame
from CoScientist.context_init.presentation import SECTIONS, STATUS_I18N


VIEW_SCHEMA_VERSION = 1
_HUMAN_STATUSES = {"задано заказчиком", "уточнено оператором"}
_OPEN_STATUSES = {"не задано", "рассчитывается агентом"}
_EMPTY_VALUE_PARTS = {"", "-", "—", "не задано", "not specified", "none", "null"}

_CONSTRAINT_BLOCKS = {
    "profile": "Профиль исследования",
    "methodological_norms": "Методологические нормы",
    "theoretical_framework": "Теоретические рамки",
    "domain_standards": "Доменные стандарты",
    "ethics": "Этика и регуляторика",
    "expert_knowledge": "Экспертное знание",
    "roles": "Роли участников",
}
_RESOURCE_FIELD = {"gpu_hours", "tokens", "money", "time", "expert_hours"}
_TOOL_FIELD = {"computational", "laboratory", "analytical", "informational"}
_BASE_FIELD = {"dataset": "datasets", "corpus": "corpora",
               "knowledge_base": "trusted_kb"}


def _source_for_status(status: str) -> str:
    if status in _HUMAN_STATUSES:
        return "human"
    if status in _OPEN_STATUSES or status == "не требуется":
        return ""
    return "agent"


def _status_label(status: str) -> Dict[str, str]:
    return STATUS_I18N.get(status, {"ru": status, "en": status})


def _frame_from_snapshot(snapshot: Any) -> Optional[ResearchFrame]:
    raw = snapshot
    if isinstance(raw, dict) and isinstance(raw.get("frame"), dict):
        raw = raw["frame"]
    if not isinstance(raw, dict):
        return None
    try:
        return ResearchFrame.model_validate(raw).normalized()
    except Exception:  # an old or partial snapshot falls back to graph recovery
        return None


def _field(frame: ResearchFrame, block: str, name: str):
    item = frame.block(block)
    if item is None:
        return None
    return next((field for field in item.fields if field.name == name), None)


def _set(frame: ResearchFrame, block: str, name: str, value: Any,
         source: str, origins: Dict[str, Dict[str, str]]) -> bool:
    if value in (None, "", [], {}):
        return False
    field = _field(frame, block, name)
    if field is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    if field.status not in _OPEN_STATUSES and field.value not in ("", "Не задано"):
        field.value = f"{field.value}\n{text}" if text not in field.value else field.value
    else:
        field.value = text
    field.status = "задано заказчиком" if source in {"human", "user", "operator"} \
        else "предложено агентом"
    origins[f"{block}:{name}"] = {
        "status": "восстановлено из графа",
        "source": source or "",
    }
    return True


def _known_content(content: Any, names: Iterable[str]) -> Tuple[Dict[str, str], str]:
    """Read ``key: value; key: value`` without splitting semicolons in values.

    Only a semicolon followed by another known key is a boundary.  If an old
    single-field block contains free prose, all of it belongs to that field.
    Ambiguous multi-field prose is returned untouched as additional context.
    """
    text = str(content or "").strip()
    known = tuple(names)
    if not text:
        return {}, ""
    key_pattern = "|".join(re.escape(name) for name in known)
    matches = list(re.finditer(rf"(?:^|;\s*)({key_pattern})\s*:\s*", text))
    if matches:
        out: Dict[str, str] = {}
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            value = text[match.end():end].rstrip(" ;\n")
            if value:
                out[match.group(1)] = value
        return out, ""
    if len(known) == 1:
        return {known[0]: text}, ""
    return {}, text


def _nodes(graph_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [node for node in (graph_data.get("nodes") or []) if isinstance(node, dict)]


def _recover_frame(graph_data: Dict[str, Any]
                   ) -> Tuple[ResearchFrame, Dict[str, Dict[str, str]], List[Dict[str, Any]]]:
    frame = ResearchFrame.blank()
    origins: Dict[str, Dict[str, str]] = {}
    additional: List[Dict[str, Any]] = []
    nodes = _nodes(graph_data)
    root_id = graph_data.get("root_id")
    root = next((n for n in nodes if n.get("id") == root_id), None)
    if root is None:
        root = next((n for n in nodes if n.get("type") == "ResearchQuestion"), None)
    if root:
        attrs = dict(root.get("attrs") or {})
        source = str(root.get("source") or "")
        for section in SECTIONS:
            for shown in section.fields:
                if shown.block.startswith("$"):
                    continue
                if shown.name in attrs:
                    _set(frame, shown.block, shown.name, attrs[shown.name], source, origins)

    for node in nodes:
        kind = node.get("type")
        attrs = dict(node.get("attrs") or {})
        source = str(node.get("source") or "")
        if kind == "Constraint":
            block = _CONSTRAINT_BLOCKS.get(str(attrs.get("subtype") or ""))
            target = frame.block(block) if block else None
            if target:
                parsed, ambiguous = _known_content(
                    attrs.get("content"), (field.name for field in target.fields))
                for name, value in parsed.items():
                    _set(frame, block, name, value, source, origins)
                if ambiguous:
                    additional.append({
                        "id": str(node.get("id") or ""),
                        "label": {"ru": block, "en": block},
                        "value": ambiguous,
                        "note": {
                            "ru": "Сохранённое описание старой сессии; границы полей неизвестны.",
                            "en": "Saved description from an older session; field boundaries are unknown.",
                        },
                    })
        elif kind == "Resource":
            name = str(attrs.get("resource_type") or "")
            if name in _RESOURCE_FIELD:
                if attrs.get("remaining") is not None and attrs.get("limit") is not None:
                    value = f"{attrs['remaining']} / {attrs['limit']}"
                else:
                    value = attrs.get("note")
                _set(frame, "Ресурсы и бюджеты", name, value, source, origins)
        elif kind == "Tool":
            name = str(attrs.get("tool_type") or "")
            if name in _TOOL_FIELD:
                _set(frame, "Инструменты", name,
                     attrs.get("name") or attrs.get("description"), source, origins)
        elif kind == "EmpiricalBase":
            name = _BASE_FIELD.get(str(attrs.get("base_type") or ""))
            if name:
                _set(frame, "Эмпирическая база", name,
                     attrs.get("source_ref") or attrs.get("name") or attrs.get("description"),
                     source, origins)
        elif kind == "ConfirmationCriteria":
            for name in ("threshold", "confirmations_needed", "reproducibility"):
                _set(frame, "Условия подтверждения", name, attrs.get(name), source, origins)
        elif kind == "CostModel":
            for name in ("cost_rule", "stop_rule", "expected_effect"):
                _set(frame, "Модель стоимости", name, attrs.get(name), source, origins)

    return frame, origins, additional


def _value(field: Any) -> str:
    if field is None or field.status in _OPEN_STATUSES:
        return ""
    value = str(field.value or "").strip()
    # Older auto-approved frames contain composite placeholders such as
    # "Не задано / Не задано" with a closed status.  They are still absence of
    # data, not content worth showing in a technical specification.
    parts = [part.strip().casefold() for part in re.split(r"\s*[/|]\s*", value)]
    return "" if parts and all(part in _EMPTY_VALUE_PARTS for part in parts) else value


def _field_view(shown: Any, value: str, status: str, source: str,
                *, field_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": field_id or shown.id,
        "field_key": shown.name,
        "block_key": shown.block,
        "label": shown.label,
        "help": shown.help,
        "value": value,
        "value_format": "markdown" if ("\n" in value or len(value) > 120) else "text",
        "status": status,
        "status_label": _status_label(status),
        "source": source,
        "empty": not bool(value),
    }


def build_framing_details(graph_data: Dict[str, Any], snapshot: Any = None,
                          documents: Optional[List[Dict[str, Any]]] = None,
                          study_id: str = "active") -> Dict[str, Any]:
    """Return the complete, localisable sidebar contract for one study."""
    frame = _frame_from_snapshot(snapshot)
    recovered = frame is None
    origins: Dict[str, Dict[str, str]] = {}
    additional: List[Dict[str, Any]] = []
    if frame is None:
        frame, origins, additional = _recover_frame(graph_data)

    sections: List[Dict[str, Any]] = []
    filled = 0
    total = 0
    for section in SECTIONS:
        fields: List[Dict[str, Any]] = []
        for shown in section.fields:
            if shown.block == "$operations":
                if frame.operations:
                    for index, operation in enumerate(frame.operations, 1):
                        label = {
                            "ru": f"Задача {index} · {operation.operation_id}",
                            "en": f"Task {index} · {operation.operation_id}",
                        }
                        dynamic = SimpleNamespace(
                            id=f"$operations:{operation.operation_id}",
                            name=operation.operation_id,
                            block="$operations",
                            label=label,
                            help=shown.help,
                        )
                        fields.append(_field_view(
                            dynamic, operation.statement, "задано заказчиком", "frame",
                        ))
                else:
                    fields.append(_field_view(shown, "", "не задано", ""))
                continue
            if shown.block == "$meta":
                value = str(frame.original_request or "").strip()
                status = "задано заказчиком" if value else "не задано"
                fields.append(_field_view(shown, value, status, "request" if value else ""))
                continue
            item = _field(frame, shown.block, shown.name)
            origin = origins.get(shown.id, {})
            status = str(origin.get("status") or (item.status if item else "не задано"))
            source = str(origin.get("source") or _source_for_status(status))
            fields.append(_field_view(shown, _value(item), status, source))
        total += len(fields)
        visible_fields = [field for field in fields if not field["empty"]]
        filled += len(visible_fields)
        section_documents = list(documents or []) if section.documents else []
        if not visible_fields and not section_documents:
            continue
        sections.append({
            "id": section.id,
            "title": section.title,
            "open_by_default": section.open_by_default,
            "fields": visible_fields,
            "documents": section_documents,
        })

    topic = _value(_field(frame, "Основание и приёмка", "topic_name"))
    question = _value(_field(frame, "Вопрос исследования", "formulation"))
    payload: Dict[str, Any] = {
        "schema_version": VIEW_SCHEMA_VERSION,
        "study_id": study_id,
        "research_id": str(graph_data.get("research_id") or ""),
        "root_id": str(graph_data.get("root_id") or ""),
        "title": topic or question or "Техническое задание",
        "source_mode": "graph_recovery" if recovered else "research_frame",
        "source_note": ({
            "ru": "Часть структуры восстановлена из старого снимка графа.",
            "en": "Part of the structure was recovered from an older graph snapshot.",
        } if recovered else {
            "ru": "Составлено из параметров исследования.",
            "en": "Built from the research parameters.",
        }),
        "completion": {"filled": filled, "total": total, "missing": total - filled},
        "sections": sections,
        "additional_context": additional,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    payload["revision"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return payload


__all__ = ["VIEW_SCHEMA_VERSION", "build_framing_details"]
