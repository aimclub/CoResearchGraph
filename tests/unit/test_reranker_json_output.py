import json
from types import SimpleNamespace

from google.adk.models import LlmResponse
from google.genai import types

from CoScientist.agents.callbacks.json_output import sanitize_json_output
from CoScientist.assembly import load_config
from CoScientist.storage.models import MCPRanking


def test_mcp_reranker_sanitizes_fenced_json_before_schema_validation():
    payload = {
        "mcp_scores": [{"index": 1, "score": False}],
        "reasoning": "Local tools already satisfy the requirements.",
    }
    fenced = f"```json\n{json.dumps(payload)}\n```"
    response = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=fenced)])
    )

    cleaned = sanitize_json_output(
        SimpleNamespace(agent_name="FullSetToolReranker"), response
    )

    assert cleaned is not None
    parsed = MCPRanking.model_validate_json(cleaned.content.parts[0].text)
    assert parsed.mcp_scores[0].index == 1
    assert parsed.mcp_scores[0].score is False


def test_all_schema_constrained_rerankers_wire_json_sanitizer():
    config = load_config()

    for name in ("ToolReranker", "FullSetToolReranker"):
        assert "sanitize_json_output" in config.agent(name).callbacks.after_model
