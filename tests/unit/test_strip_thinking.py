from __future__ import annotations

import pytest
from google.genai import types

from CoScientist.main import CoScientistManager
from CoScientist.utils.text import strip_thinking


def test_strip_thinking_tags():
    assert strip_thinking("<think>Reasoning tokens here</think>Actual message") == "Actual message"
    assert strip_thinking("<thought>Another tag</thought>\nAnswer") == "Answer"
    assert strip_thinking("<think>Unclosed thinking block...") == ""
    assert strip_thinking("Clean message without thinking") == "Clean message without thinking"
    assert strip_thinking("") == ""
    assert strip_thinking(None) == ""


def test_final_text_skips_thought_parts_and_tags():
    # Event with thought part and answer part
    event = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="Thinking about this deeply...", thought=True),
                        types.Part(text="<think>additional thought</think>Here is the final report."),
                    ],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )

    # Wrap in an object simulating ADK Event with is_final_response=True
    class MockEvent:
        def __init__(self, content):
            self.content = content
            self.actions = None

        def is_final_response(self):
            return True

    mock_event = MockEvent(event.candidates[0].content)
    final_text = CoScientistManager._final_text(mock_event)
    assert final_text == "Here is the final report."
