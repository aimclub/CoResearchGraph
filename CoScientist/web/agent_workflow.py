"""Presentation-only workflow; execution still belongs to the YAML assembler.

Blocks retain completion boundaries when internal agents are hidden. Instances
identify call sites, not historical activations, so shared agents can be drawn
without forcing unrelated callers into the same layout level.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from CoScientist.assembly.schema import COMPOSITE_CLASSES, SystemConfig


def project_workflow(config: SystemConfig, *, nir_enabled: bool) -> dict[str, Any]:
    instances: list[dict[str, Any]] = []

    def visit(name: str, path: str, ancestors: tuple[str, ...] = ()) -> dict | None:
        agent = config.agent(name)
        if name == "NirReportAgent" and not nir_enabled:
            return None
        block_id = f"{path}/{name}"
        visible = not agent.internal
        repeated = name in ancestors
        if visible:
            instances.append({"id": block_id, "agentId": name, "recursive": repeated})
        if repeated:
            return {"id": block_id, "type": "agent", "agentId": name} if visible else None

        def children(names: list[str], relation: str, filter_enabled: bool = True) -> list[dict]:
            result = []
            for child in names:
                if filter_enabled and not config.agent(child).is_enabled():
                    continue
                value = visit(child, f"{block_id}/{relation}", (*ancestors, name))
                if value:
                    result.append(value)
            return result

        # Match assembler semantics: enabled composites attach declared children;
        # custom agents filter their children. A custom child list alone does not
        # establish sequence (ExecutorSwitch, for example, selects a route).
        nested = children(
            agent.children if agent.cls not in COMPOSITE_CLASSES or agent.is_enabled() else [],
            "child", agent.cls not in COMPOSITE_CLASSES,
        )
        calls = children(agent.subordinates, "call")
        kind = agent.cls if agent.cls in ("sequential", "parallel", "loop") else (
            "choice" if agent.cls == "custom:executor_switch" else "delegates"
        )
        kind = "sequence" if kind == "sequential" else kind
        body = {"id": f"{block_id}/body", "type": kind, "children": nested} if nested else None
        if calls:
            # Even pipeline.linear only describes status presentation; an LLM
            # root still chooses its calls. Never turn that into guaranteed flow.
            calls_block = {"id": f"{block_id}/calls", "type": "delegates", "children": calls,
                           "relation": "delegate"}
            if config.pipeline.linear and agent.root:
                calls_block["presentationOrder"] = "declared"
            body = {"id": f"{block_id}/body", "type": "sequence", "children": [body, calls_block]} if body else calls_block
        if visible:
            block: dict[str, Any] = {"id": block_id, "type": "agent", "agentId": name}
            if body:
                block["body"] = body
                block["composite"] = agent.cls in COMPOSITE_CLASSES
            return block
        # Hide the internal header, never the ordering information it contains.
        return body

    stages = [name for name in config.pipeline.pre if config.agent(name).is_enabled()]
    stages.append(config.root.name)
    stages.extend(name for name in config.pipeline.post if config.agent(name).is_enabled())
    blocks = [value for name in stages if (value := visit(name, "workflow"))]
    counts = Counter(item["agentId"] for item in instances)
    for item in instances:
        item["reused"] = counts[item["agentId"]] > 1
    coordinator = next((item for item in instances if item["agentId"] in ("OrchestratorAgent", "RootOrchestrator")),
                       instances[0] if instances else None)
    return {
        "version": 1,
        "root": {"id": "workflow", "type": "sequence", "children": blocks},
        "instances": instances,
        "configuration": {"agentId": "__mas_session__", "ownerInstanceId": coordinator["id"] if coordinator else None},
    }
