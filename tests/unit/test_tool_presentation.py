from __future__ import annotations

from CoScientist.tools import mcp_catalog
from CoScientist.tools.mcp_catalog import resolve_tool_presentation


def test_builtin_exact_local_override_precedes_defaults_and_keeps_bilingual_metadata():
    presentation = {
        "local_tools": {
            "demo:sample_tool": {
                "display_name": {"ru": "Локальный инструмент", "en": "Local tool"},
                "description": {"ru": "Локальное описание", "en": "Local description"},
            },
        },
        "tool_defaults": {
            "sample_tool": {
                "display_name": {"ru": "Значение по умолчанию", "en": "Default"},
                "description": {"ru": "Описание по умолчанию", "en": "Default description"},
            },
        },
    }

    result = resolve_tool_presentation(
        "sample_tool",
        "English implementation description",
        registry_key="demo",
        presentation=presentation,
    )

    assert result == {
        "display_name": {"ru": "Локальный инструмент", "en": "Local tool"},
        "summary": {"ru": "Локальное описание", "en": "Local description"},
    }


def test_remote_registry_id_override_precedes_endpoint_and_defaults():
    presentation = {
        "tools": {
            "registry:sample_tool": {
                "display_name": {"ru": "Реестр", "en": "Registry"},
                "description": {"ru": "Описание реестра", "en": "Registry description"},
            },
            "endpoint:sample_tool": {
                "display_name": {"ru": "Эндпоинт", "en": "Endpoint"},
                "description": {"ru": "Описание эндпоинта", "en": "Endpoint description"},
            },
        },
        "tool_defaults": {
            "sample_tool": {
                "display_name": {"ru": "По умолчанию", "en": "Default"},
                "description": {"ru": "Описание по умолчанию", "en": "Default description"},
            },
        },
    }

    result = resolve_tool_presentation(
        "sample_tool",
        "Live description",
        server_id="endpoint",
        registry_id="registry",
        presentation=presentation,
    )

    assert result["display_name"]["ru"] == "Реестр"
    assert result["summary"]["en"] == "Registry description"


def test_builtin_fallback_is_neutral_in_russian_and_uses_english_source_in_english():
    result = resolve_tool_presentation(
        "sample_tool",
        "English implementation description",
        registry_key="missing",
        presentation={},
    )

    assert result["display_name"] == {"ru": "Инструмент", "en": "Sample tool"}
    assert result["summary"] == {
        "ru": "Описание на русском пока не добавлено.",
        "en": "English implementation description",
    }


def test_remote_fallback_is_neutral_in_russian():
    result = resolve_tool_presentation(
        "machine_name",
        "English implementation description",
        server_id="endpoint",
        registry_id="registry",
        presentation={},
    )

    assert result["display_name"] == {"ru": "Инструмент", "en": "Machine name"}
    assert result["summary"] == {
        "ru": "Описание на русском пока не добавлено.",
        "en": "English implementation description",
    }


def test_public_snapshot_reapplies_yaml_labels_without_mutating_cached_snapshot(monkeypatch):
    presentation = {
        "servers": {
            "registry": {
                "display_name": {"ru": "Новый сервер", "en": "New server"},
                "description": {"ru": "Новое описание", "en": "New description"},
                "category": "science",
            },
        },
        "tools": {
            "registry:sample_tool": {
                "display_name": {"ru": "Новый инструмент", "en": "New tool"},
                "description": {"ru": "Новое описание инструмента", "en": "New tool description"},
                "priority": 2,
            },
        },
    }
    monkeypatch.setattr(mcp_catalog, "load_presentation", lambda: presentation)
    cached = {
        "version": 1,
        "checked_at": "2026-09-26T12:00:00+00:00",
        "servers": [{
            "id": "endpoint",
            "registry_id": "registry",
            "name": "Old",
            "display_name": {"ru": "Старый сервер", "en": "Old server"},
            "description": {"ru": "Старое описание", "en": "Old description"},
            "category": "other",
        }],
        "tools": [{
            "id": "endpoint:sample_tool",
            "server_id": "endpoint",
            "name": "sample_tool",
            "display_name": {"ru": "Старый инструмент", "en": "Old tool"},
            "summary": {"ru": "Старое описание инструмента", "en": "Old tool description"},
            "original_description": "Live description",
            "priority": 100,
        }],
    }

    public = mcp_catalog._public_snapshot(cached, refreshing=False)

    assert public["servers"][0]["display_name"]["ru"] == "Новый сервер"
    assert public["tools"][0]["summary"]["ru"] == "Новое описание инструмента"
    assert public["tools"][0]["priority"] == 2
    assert cached["servers"][0]["display_name"]["ru"] == "Старый сервер"
    assert cached["tools"][0]["summary"]["ru"] == "Старое описание инструмента"


def test_assemble_and_public_snapshot_share_unknown_tool_fallback():
    registry = {
        "status": "ready",
        "servers": [{"server_id": "registry", "name": "Server"}],
        "tools": [],
    }
    endpoint = {
        "id": "endpoint",
        "registry_id": "registry",
        "name": "Server",
        "registry": {"server_id": "registry", "name": "Server"},
        "tool_filter": None,
    }
    probe = {
        "status": "reachable",
        "tools": [{
            "name": "machine_name_xyz",
            "description": "English implementation description",
            "input_schema": {},
        }],
    }

    assembled = mcp_catalog._assemble_catalog(registry, [endpoint], [probe])
    public = mcp_catalog._public_snapshot(assembled, refreshing=False)
    assembled_tool = assembled["tools"][0]
    public_tool = public["tools"][0]

    assert assembled_tool["display_name"] == public_tool["display_name"]
    assert assembled_tool["summary"] == public_tool["summary"]
    assert assembled_tool["display_name"]["ru"] == "Инструмент"
    assert assembled_tool["summary"]["ru"] == "Описание на русском пока не добавлено."


def test_old_snapshot_unknown_fallback_is_normalized_without_discovery(monkeypatch):
    monkeypatch.setattr(mcp_catalog, "load_presentation", lambda: {})
    cached = {"servers": [], "tools": [{
        "id": "server:unknown", "server_id": "server", "name": "unknown",
        "status": "saved", "display_name": {"ru": "Инструмент «unknown»", "en": "Old English title"},
        "summary": {"ru": "Old English description", "en": "Preserved English description"},
    }]}
    tool = mcp_catalog._public_snapshot(cached, refreshing=False)["tools"][0]
    assert tool["display_name"] == {"ru": "Инструмент", "en": "Old English title"}
    assert tool["summary"] == {"ru": "Описание на русском пока не добавлено.", "en": "Preserved English description"}
    assert tool["status"] == "saved"
    assert cached["tools"][0]["summary"]["ru"] == "Old English description"
