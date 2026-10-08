"""Composable prompt construction.

Agent prompts are assembled from named, optionally-conditional parts instead of
one big string literal. This makes it easy to toggle a section on/off (e.g.
include the planner instructions only when the planner is enabled) without
duplicating the whole prompt.

Example:
    prompt = (
        PromptBuilder()
        .add(HEADER)
        .add(PLANNER_BLOCK, when=use_planner)
        .add(ROUTING)
        .build()
    )

Sections that are empty, blank, or added with ``when=False`` are skipped.
Remaining sections are joined with a blank line (configurable).
"""

from typing import Iterable, List


def render_template(template: str, **values: str) -> str:
    """Fill ``<<NAME>>`` placeholders in a template with the given values.

    Uses ``<<NAME>>`` sentinels (not str.format) so that literal ``{ }`` in a
    prompt — e.g. JSON output examples — never need escaping. Unknown
    placeholders are left untouched; missing values simply aren't substituted.
    """
    out = template
    for key, value in values.items():
        out = out.replace(f"<<{key}>>", value)
    return out


class PromptBuilder:
    """Accumulate ordered prompt sections and join them into one string."""

    def __init__(self, sep: str = "\n\n"):
        self._parts: List[str] = []
        self._sep = sep

    def add(self, body: str, *, when: bool = True) -> "PromptBuilder":
        """Append one section. Skipped if ``when`` is False or the body is blank."""
        if when and body and body.strip():
            self._parts.append(body.strip("\n"))
        return self

    def add_all(self, bodies: Iterable[str], *, when: bool = True) -> "PromptBuilder":
        """Append several sections under a single condition."""
        for body in bodies:
            self.add(body, when=when)
        return self

    def build(self) -> str:
        """Return the assembled prompt."""
        return self._sep.join(self._parts)

    def __str__(self) -> str:  # convenience
        return self.build()
