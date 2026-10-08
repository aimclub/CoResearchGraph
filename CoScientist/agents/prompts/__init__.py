"""Prompt templates for the agents.

Templates live in :mod:`CoScientist.agents.prompts.templates` and register
themselves in the assembly registry under the names ``system.yaml`` references
(``prompt: <name>``). They are rendered by the assembler with a PromptContext,
which fills the unified placeholders (<<TOOLS>>, <<AGENTS>>, <<ROUTING>>,
<<HITL>>) from the same config that wires the agents.
"""
from CoScientist.agents.prompts.builder import PromptBuilder, render_template

__all__ = ["PromptBuilder", "render_template"]
