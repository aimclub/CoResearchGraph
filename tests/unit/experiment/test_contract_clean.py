"""Unit tests for clean architecture contract, grouping, suppression, and URL requirements."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.capabilities.inventory import get_grouped_mcp_inventory
from CoScientist.experiments.critique.validator import validate_and_critique_plan
from CoScientist.experiments.schemas.models import MCPServerRef, ExperimentPlan
from .helpers import _task, _plan


def test_grouped_mcp_inventory_groups_multiple_tools_under_same_url():
    """Two tools with the same server_id and url group into one server entry."""
    rows = [
        {
            "server_id": "srv-chem",
            "name": "srv-chem",
            "url": "http://127.0.0.1:8000/mcp",
            "tool": "calculate_docking",
            "description": "Docking tool",
            "input_schema": {"type": "object"},
            "family": "mcp",
        },
        {
            "server_id": "srv-chem",
            "name": "srv-chem",
            "url": "http://127.0.0.1:8000/mcp",
            "tool": "generate_case_mols",
            "description": "Generation tool",
            "input_schema": {"type": "object"},
            "family": "mcp",
        },
    ]
    grouped = get_grouped_mcp_inventory(rows)
    assert len(grouped) == 1
    assert grouped[0]["server_id"] == "srv-chem"
    assert grouped[0]["url"] == "http://127.0.0.1:8000/mcp"
    tool_names = [t["name"] for t in grouped[0]["tools"]]
    assert "calculate_docking" in tool_names
    assert "generate_case_mols" in tool_names


def test_mcpserverref_registry_without_url_raises_validation_error():
    """MCPServerRef with source='registry' and no url fails validation."""
    with pytest.raises(ValidationError, match="requires url"):
        MCPServerRef.model_validate({
            "source": "registry",
            "server_id": "srv-chem",
            "name": "srv-chem",
            "tools": [{"name": "generate_mols"}],
        })


def test_mcpserverref_disallows_composite_slash_in_server_id():
    """MCPServerRef disallows composite server_id like 'srv/tool'."""
    with pytest.raises(ValidationError, match="composite"):
        MCPServerRef.model_validate({
            "source": "registry",
            "server_id": "srv-chem/generate_mols",
            "name": "srv-chem",
            "url": "http://127.0.0.1:8000/mcp",
            "tools": [{"name": "generate_mols"}],
        })


def test_fill_server_urls_is_removed_from_routing():
    """Verify fill_server_urls does not exist in routing or state_machine."""
    import CoScientist.experiments.runtime.routing as routing
    assert not hasattr(routing, "fill_server_urls")
    assert not hasattr(routing, "_lookup_http_urls")


def test_validate_and_critique_plan_no_magic_repair_pure_pydantic():
    """validate_and_critique_plan validates payload directly with Pydantic."""
    raw_invalid_payload = {
        "tasks": [
            {
                "id": "EXP-1",
                "route": "fedot_mas",
                "mcp_servers": [
                    {
                        "source": "registry",
                        "server_id": "srv-chem",
                        # Missing url
                        "tools": [{"name": "generate_mols"}],
                    }
                ],
            }
        ]
    }
    from CoScientist.experiments.critique.validator import PlanValidationError
    with pytest.raises(PlanValidationError):
        validate_and_critique_plan(
            raw_invalid_payload,
            settings=ExperimentsSettings(),
        )
