"""Text sanitization and formatting helpers."""
from __future__ import annotations

import re

# Matches inline thinking tags like <think>...</think> or <thought>...</thought>,
# including unclosed tags at the end of a turn/stream.
_THINKING_TAGS_RE = re.compile(
    r"<(?:think|thought)>.*?(?:</(?:think|thought)>|$)",
    re.DOTALL | re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> and <thought>...</thought> blocks from text."""
    if not text or not isinstance(text, str):
        return ""
    return _THINKING_TAGS_RE.sub("", text).strip()
