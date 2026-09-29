# Literature boundary in the experiments profile

`ResearchAgent` is a direct tool of `OrchestratorAgent`, not an executor inside
`ExperimentModuleAgent`. The parent owns literature collection and the complete
research roadmap. The experiment module owns computational work and consumes
the collected evidence and data references.

## Enforcement

- `agents/experiments.yaml` removes ResearchAgent from the experiment executor.
  Planner/executor prompts and `hitl/pipeline_scope.py` describe the same boundary.
- `experiments/scope.py` projects explicit literature operations out of experiment
  coverage, preserving original operation IDs. Ambiguous mixed operations stay in
  scope so calculations are not silently dropped. This text classification is a
  conservative heuristic, not a general semantic classifier.
- `context/builder.py` exposes `operations` for computation and
  `external_literature_operations` for parent-owned work. The full root operations,
  source request, discovery cache, prior evidence and data references remain intact.
  Literature tools are excluded from the module's capability projection.
- `critique/validator.py` does not demand coverage of external operations or tools.
  Literature tasks cannot be assigned to Coder, Medical, Alembic or an MCP as a
  replacement for ResearchAgent.
- `runtime/state_machine.py` checks the same boundary before starting an attempt;
  `runtime/guards.py` rejects direct legacy executor calls to ResearchAgent.
- The module's `skip_literature_only_experiment` callback returns a misplaced
  literature-only request to the orchestrator before tool discovery or planning.
  It does not claim completion of the research requirement.

Already supplied papers/data may still be analyzed, including clinical PICO
extraction. Code/API documentation access is not literature collection. Missing
required literature inputs must be reported to the parent, not fabricated or
silently replaced by another search inside the module.

## Compatibility and rollout

Legacy `route=research` and `prepare_via=research` values remain readable in saved
plans/results. They are not executable routes in the updated module. This change
does not migrate or automatically resume previously saved research sessions.
Restart the application/rebuild the agent tree to load the updated configuration.

Regression tests: `tests/unit/experiment/test_literature_scope.py`, plus profile,
context, critique and runtime tests in the same directory. They check the built
agent tree, mixed operation IDs, input preservation, legacy refusal and tool
coverage without live model calls.
