"""Alternative graph presentations keep workflow identities and safe routes."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from CoScientist.config import get_settings
from CoScientist.config.settings import AgentOverride
from CoScientist.web.agent_tree import project_agent_tree


NODE = shutil.which("node")
LAYOUT = Path(__file__).resolve().parents[2] / "CoScientist/web/static/js/agent_tree_variants.js"
CHECK = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
global.window = {};
require(process.argv[1]);
const workflow = JSON.parse(fs.readFileSync(0, 'utf8'));
const api = window.AgentTreeVariants;
for (const design of ['atlas', 'workshop']) for (const width of [500, 850, 1400]) {
  const g = api.layout(workflow, design, {width});
  assert.deepEqual(g, api.layout(workflow, design, {width}), 'layout must be deterministic');
  const ids = new Map(g.nodes.map(n => [n.id, n]));
  const expected = workflow.instances.map(n => n.id);
  if (workflow.configuration?.ownerInstanceId) expected.push(workflow.configuration.ownerInstanceId + '/configuration');
  assert.deepEqual([...ids.keys()].sort(), expected.sort(), 'all visible occurrences retained');
  const border = (p, n) => n.shape === 'circle'
    ? Math.abs(Math.hypot(p.x - n.x - n.w / 2, p.y - n.y - n.h / 2) - n.w / 2) < .01
    : n.shape === 'hexagon' ? Math.abs(Math.max(Math.abs(p.y - n.y - n.h / 2) / (n.h / 2),
      Math.abs(p.x - n.x - n.w / 2) / (n.w / 2) + Math.abs(p.y - n.y - n.h / 2) / n.h) - 1) < .01
    : p.x >= n.x - .01 && p.x <= n.x + n.w + .01 && p.y >= n.y - .01 && p.y <= n.y + n.h + .01 &&
      Math.min(Math.abs(p.x - n.x), Math.abs(p.x - n.x - n.w), Math.abs(p.y - n.y), Math.abs(p.y - n.y - n.h)) < .01;
  const inside = (p, n) => n.shape === 'circle'
    ? Math.hypot(p.x - n.x - n.w / 2, p.y - n.y - n.h / 2) < n.w / 2 - 1
    : n.shape === 'hexagon' ? Math.abs(p.y - n.y - n.h / 2) < n.h / 2 - 1 &&
      Math.abs(p.x - n.x - n.w / 2) + Math.abs(p.y - n.y - n.h / 2) * n.w / (2 * n.h) < n.w / 2 - 1
    : p.x > n.x + 1 && p.x < n.x + n.w - 1 && p.y > n.y + 1 && p.y < n.y + n.h - 1;
  const properIntersection = (a, b, c, d) => {
    const cross = (p, q, r) => (q.x-p.x)*(r.y-p.y)-(q.y-p.y)*(r.x-p.x);
    const abC = cross(a,b,c), abD = cross(a,b,d), cdA = cross(c,d,a), cdB = cross(c,d,b);
    return abC * abD < -0.01 && cdA * cdB < -0.01;
  };
  g.nodes.forEach((n, i) => {
    assert([n.x, n.y, n.w, n.h].every(Number.isFinite));
    g.nodes.slice(i + 1).forEach(m => assert(n.x + n.w <= m.x || m.x + m.w <= n.x || n.y + n.h <= m.y || m.y + m.h <= n.y, 'overlapping nodes'));
  });
  for (const edge of g.edges) {
    const message = design + ': ' + edge.from + ' -> ' + edge.to;
    assert(border(edge.points[0], ids.get(edge.from)), 'detached start: ' + message);
    assert(border(edge.points.at(-1), ids.get(edge.to)), 'detached end: ' + message);
    edge.points.slice(1).forEach((b, i) => {
      const a = edge.points[i], steps = Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / 3);
      for (let step = 1; step < steps; step++) {
        const p = {x: a.x + (b.x - a.x) * step / steps, y: a.y + (b.y - a.y) * step / steps};
        assert(!g.nodes.some(n => inside(p, n)), 'edge crosses a node: ' + message);
      }
    });
  }
  if (design === 'atlas') g.edges.forEach((edge, index) => {
    g.edges.slice(index + 1).forEach(other => {
      if (edge.from === other.from || edge.from === other.to || edge.to === other.from || edge.to === other.to) return;
      edge.points.slice(1).forEach((b, segment) => other.points.slice(1).forEach((d, otherSegment) => {
        assert(!properIntersection(edge.points[segment], b, other.points[otherSegment], d),
          `edge crossing: ${edge.from} -> ${edge.to} ${JSON.stringify(edge.points)} with ${other.from} -> ${other.to} ${JSON.stringify(other.points)}`);
      }));
    });
  });
  const owner = ids.get(workflow.configuration?.ownerInstanceId);
  if (owner) {
    const config = ids.get(owner.id + '/configuration');
    assert.equal(config.y, owner.y); assert(config.x > owner.x);
  }
  const experiment = ['ExperimentPlannerAgent', 'ExperimentExecutorAgent', 'ExperimentResultReviewAgent']
    .map(id => g.nodes.find(n => n.agentId === id));
  if (experiment.every(Boolean)) {
    assert(experiment.every(n => n.y === experiment[0].y), 'experiment steps share a horizontal row');
    assert(experiment[0].x < experiment[1].x && experiment[1].x < experiment[2].x, 'plan precedes execution and review');
  }
}
"""


@pytest.mark.skipif(not NODE, reason="Node.js required for graph geometry")
@pytest.mark.parametrize("profile", ["system", "experiments", "hypotheses", "microfluidics"])
@pytest.mark.parametrize("mode", ["planner", "init", "orchestrator", "orchestrator_planner", "orchestrator_plan"])
@pytest.mark.parametrize("disable_optional", [False, True])
def test_design_geometry_across_profiles(monkeypatch, profile, mode, disable_optional):
    monkeypatch.setenv("COSCIENTIST_CONFIG", profile)
    snapshot = get_settings().model_copy(deep=True)
    snapshot.web.start_mode = mode
    snapshot.context_init.enabled = True
    snapshot.nir.enabled = True
    snapshot.mcp.normcontrol_url = "http://normcontrol.invalid/mcp"
    if disable_optional:
        for name in ("ResearchAgent", "HypothesesAgent", "CoderAgent"):
            snapshot.agents.overrides[name] = AgentOverride(enabled=False)
        snapshot.web.medical_agent_enabled = False
    graph = project_agent_tree(snapshot, desired_revision=1, active_revision=1, running=False)
    result = subprocess.run(
        [NODE, "-e", CHECK, str(LAYOUT)], input=json.dumps(graph["workflow"]),
        encoding="utf-8", capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
