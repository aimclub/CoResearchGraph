"""Name -> object registries backing the YAML agent config.

Every name that appears in ``system.yaml`` (a tool, a callback, a prompt, an
agent class, an output schema, a planner) must be registered here by
:mod:`CoScientist.assembly.bindings` (tools/callbacks/classes) or
:mod:`CoScientist.agents.prompts.templates` (prompts). The assembler only ever
looks things up — it never imports concrete tool modules itself.

Tools carry their own prompt documentation (:class:`ToolDoc`). The assembler
renders an agent's "available tools" prompt section from the docs of the tools
that are *actually attached*, so the prompt can never advertise a tool the
agent does not have (and vice versa).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

# Callback kinds exactly as the LlmAgent constructor names them (minus `_callback`).
CALLBACK_KINDS = (
    "before_model",
    "after_model",
    "before_tool",
    "after_tool",
    "before_agent",
    "after_agent",
)


@dataclass(frozen=True)
class ToolDoc:
    """Prompt-facing documentation for ONE tool (one bullet in the prompt).

    `name` must be the exact tool name the model calls. `signature` is what is
    shown in the prompt (name + key args). `usage` lines become sub-bullets.
    """

    name: str
    signature: str
    purpose: str
    usage: Sequence[str] = ()

    def render(self) -> str:
        lines = [f"* `{self.signature}` — {self.purpose}"]
        lines.extend(f"    - {u}" for u in self.usage)
        return "\n".join(lines)


@dataclass(frozen=True)
class ToolEntry:
    """A named, attachable tool/toolset.

    factory: returns the tool object(s) to attach (a tool, a list of tools, or a
        toolset), or None when the tool is not available in this deployment
        (e.g. an optional MCP URL is unset). The factory is called once per
        system build.
    docs: prompt documentation for every tool the entry contributes. May be a
        zero-argument callable instead, for entries whose guidance depends on
        what else is configured (e.g. the sandbox docs cross-reference the local
        coder tools only while those are switched on). Read it via
        ``resolved_docs()``, never off the attribute.
    optional: when True, a None from the factory silently drops the entry;
        otherwise None is a configuration error.
    runtime_resolved: True for MCP toolsets whose real tool surface is fetched
        from the remote server at runtime — for those the attached-vs-documented
        name check is skipped (docs are trusted).
    """

    key: str
    factory: Callable[[], Any]
    docs: Union[Sequence[ToolDoc], Callable[[], Sequence[ToolDoc]]] = ()
    optional: bool = False
    runtime_resolved: bool = False

    def resolved_docs(self) -> Sequence[ToolDoc]:
        """The entry's docs, calling the factory form when there is one."""
        return self.docs() if callable(self.docs) else self.docs


@dataclass(frozen=True)
class CallbackEntry:
    """A named agent callback.

    Either a plain `func`, or (when `needs_context=True`) a `factory` that takes
    the agent's PromptContext and returns the callback — used for callbacks whose
    behaviour depends on the assembled configuration (e.g. the critics, whose
    LLM prompts embed the orchestrator's current roster).
    """

    key: str
    kind: str  # one of CALLBACK_KINDS
    func: Optional[Callable] = None
    factory: Optional[Callable[[Any], Callable]] = None

    def __post_init__(self) -> None:
        if self.kind not in CALLBACK_KINDS:
            raise ValueError(f"Unknown callback kind {self.kind!r} for {self.key!r}")
        if (self.func is None) == (self.factory is None):
            raise ValueError(f"Callback {self.key!r} needs exactly one of func/factory")

    def resolve(self, ctx: Any) -> Callable:
        return self.func if self.func is not None else self.factory(ctx)


@dataclass
class Registry:
    """All named extension points the YAML can reference."""

    tools: Dict[str, ToolEntry] = field(default_factory=dict)
    callbacks: Dict[str, CallbackEntry] = field(default_factory=dict)
    prompts: Dict[str, Callable[[Any], str]] = field(default_factory=dict)
    agent_classes: Dict[str, type] = field(default_factory=dict)
    output_schemas: Dict[str, type] = field(default_factory=dict)
    planners: Dict[str, Callable[[], Any]] = field(default_factory=dict)

    # ── registration ─────────────────────────────────────────────────────────
    def _put(self, table: Dict[str, Any], key: str, value: Any, what: str) -> None:
        if key in table:
            raise ValueError(f"Duplicate {what} registration: {key!r}")
        table[key] = value

    def register_tool(self, entry: ToolEntry) -> None:
        self._put(self.tools, entry.key, entry, "tool")

    def register_callback(self, entry: CallbackEntry) -> None:
        self._put(self.callbacks, entry.key, entry, "callback")

    def register_prompt(self, key: str, builder: Callable[[Any], str]) -> None:
        self._put(self.prompts, key, builder, "prompt")

    def register_agent_class(self, key: str, cls: type) -> None:
        self._put(self.agent_classes, key, cls, "agent class")

    def register_output_schema(self, key: str, schema: type) -> None:
        self._put(self.output_schemas, key, schema, "output schema")

    def register_planner(self, key: str, factory: Callable[[], Any]) -> None:
        self._put(self.planners, key, factory, "planner")

    # ── lookup ───────────────────────────────────────────────────────────────
    def _get(self, table: Dict[str, Any], key: str, what: str) -> Any:
        if key not in table:
            known = ", ".join(sorted(table)) or "(none registered)"
            raise KeyError(f"Unknown {what} {key!r}. Known: {known}")
        return table[key]

    def tool(self, key: str) -> ToolEntry:
        return self._get(self.tools, key, "tool")

    def callback(self, key: str) -> CallbackEntry:
        return self._get(self.callbacks, key, "callback")

    def prompt(self, key: str) -> Callable[[Any], str]:
        return self._get(self.prompts, key, "prompt")

    def agent_class(self, key: str) -> type:
        return self._get(self.agent_classes, key, "agent class")

    def output_schema(self, key: str) -> type:
        return self._get(self.output_schemas, key, "output schema")

    def planner(self, key: str) -> Callable[[], Any]:
        return self._get(self.planners, key, "planner")


# The process-wide registry instance. bindings.py / templates.py populate it.
REGISTRY = Registry()


def render_tool_docs(docs: Sequence[ToolDoc]) -> str:
    return "\n".join(d.render() for d in docs)


__all__ = [
    "CALLBACK_KINDS",
    "CallbackEntry",
    "REGISTRY",
    "Registry",
    "ToolDoc",
    "ToolEntry",
    "render_tool_docs",
]
