/* Alternative presentations of the same workflow. No execution settings live here. */
(() => {
  'use strict';
  const designs = {
    atlas: { title: 'Атлас', description: 'Светлая схема исследования', w: 232, h: 104, gap: 32, step: 58 },
    workshop: { title: 'Мастерская', description: 'Соберите свою систему исследования', w: 208, h: 88, gap: 36, step: 58 },
  };
  const point = (x, y) => ({ x, y });
  const center = n => point(n.x + n.w / 2, n.y + n.h / 2);
  const distance = (a, b) => Math.hypot(b.x - a.x, b.y - a.y);

  // Test the open interior, so a route may touch an expanded obstacle corner.
  function intersects(a, b, box, padding = 0) {
    const min = [box.x - padding + .01, box.y - padding + .01];
    const max = [box.x + box.w + padding - .01, box.y + box.h + padding - .01];
    let lo = 0, hi = 1;
    for (let axis = 0; axis < 2; axis++) {
      const start = axis ? a.y : a.x, delta = axis ? b.y - a.y : b.x - a.x;
      if (Math.abs(delta) < .001) { if (start < min[axis] || start > max[axis]) return false; }
      else {
        const t1 = (min[axis] - start) / delta, t2 = (max[axis] - start) / delta;
        lo = Math.max(lo, Math.min(t1, t2)); hi = Math.min(hi, Math.max(t1, t2));
        if (lo > hi) return false;
      }
    }
    return hi >= 0 && lo <= 1;
  }
  function border(node, toward) {
    const c = center(node), dx = toward.x - c.x, dy = toward.y - c.y;
    const scale = node.shape === 'circle' ? node.w / 2 / Math.hypot(dx, dy)
      : node.shape === 'hexagon' ? 1 / Math.max(Math.abs(dy) / (node.h / 2), Math.abs(dx) / (node.w / 2) + Math.abs(dy) / node.h)
      : 1 / Math.max(Math.abs(dx) / (node.w / 2), Math.abs(dy) / (node.h / 2));
    return point(c.x + dx * scale, c.y + dy * scale);
  }
  function route(source, target, nodes) {
    const start = center(source), end = center(target);
    const obstacles = nodes.filter(n => n !== source && n !== target);
    const clear = (a, b, padding = 12) => !obstacles.some(n => intersects(a, b, n, padding));
    // Connect across the gap between rows, rather than casting rays from card
    // centers through neighbouring cards in the source row.
    let ports;
    if (target.y >= source.y + source.h) ports = [point(start.x, source.y + source.h), point(end.x, target.y)];
    else if (source.y >= target.y + target.h) ports = [point(start.x, source.y), point(end.x, target.y + target.h)];
    else if (target.x >= source.x + source.w) ports = [point(source.x + source.w, start.y), point(target.x, end.y)];
    else ports = [point(source.x, start.y), point(target.x + target.w, end.y)];
    if (clear(ports[0], ports[1], 2)) return ports;
    if (clear(start, end)) return [border(source, end), border(target, start)];
    // Only obstructed links need the visibility graph. A small cost per
    // segment favours a few diagonal lines over a long orthogonal staircase.
    const vertices = [start, end];
    obstacles.forEach(n => {
      [n.x - 12, n.x + n.w + 12].forEach(x =>
        [n.y - 12, n.y + n.h + 12].forEach(y => vertices.push(point(x, y))));
    });
    const cost = vertices.map(() => Infinity), previous = [], seen = new Set();
    cost[0] = 0;
    for (let count = 0; count < vertices.length; count++) {
      let current = -1;
      for (let i = 0; i < vertices.length; i++) if (!seen.has(i) && (current < 0 || cost[i] < cost[current])) current = i;
      if (current < 0 || !Number.isFinite(cost[current])) break;
      if (current === 1) break;
      seen.add(current);
      vertices.forEach((v, i) => {
        if (seen.has(i) || !clear(vertices[current], v)) return;
        const next = cost[current] + distance(vertices[current], v) + 12;
        if (next < cost[i]) { cost[i] = next; previous[i] = current; }
      });
    }
    if (!Number.isFinite(cost[1])) throw new Error('No unobstructed agent connection');
    const indices = [1];
    while (indices[0] !== 0) indices.unshift(previous[indices[0]]);
    const points = indices.map(i => vertices[i]);
    points[0] = border(source, points[1]);
    points[points.length - 1] = border(target, points[points.length - 2]);
    return points;
  }

  function layout(workflow, design = 'workshop', { width = 1200 } = {}) {
    const spec = designs[design], nodes = [], edges = [], byId = new Map(), roots = [];
    const instances = new Map(workflow.instances.map(item => [item.id, item]));
    const ownerId = workflow.configuration?.ownerInstanceId;
    function dimensions(id, agentId) {
      if (design !== 'workshop') return { w: spec.w, h: spec.h, shape: spec.shape };
      if (id === ownerId) return { w: 170, h: 170, shape: 'circle' };
      if (agentId === '__mas_session__') return { w: 190, h: 170, shape: 'hexagon' };
      if (['ResearchAgent', 'HypothesesAgent', 'MedicalAgent', 'ExperimentModuleAgent', 'LiteratureOrchestrator'].includes(agentId)) return { w: 146, h: 146, shape: 'circle' };
      if (['ExperimentAgent', 'CoderAgent', 'FedotAgent', 'McpBuilderAgent'].includes(agentId)) return { w: 166, h: 138, shape: 'hexagon' };
      return { w: spec.w, h: spec.h, shape: 'rect' };
    }
    const connect = (from, to, relation) => {
      if (from !== to) edges.push({ from, to, relation, arrow: true });
    };
    function visit(block, parent = null) {
      if (block.type === 'agent') {
        const node = { ...instances.get(block.id), id: block.id, agentId: block.agentId,
          parent, children: [], bodyKind: block.body?.type, ...dimensions(block.id, block.agentId) };
        nodes.push(node); byId.set(node.id, node);
        if (parent) byId.get(parent).children.push(node.id); else roots.push(node.id);
        if (block.body) {
          const body = visit(block.body, node.id);
          const relation = { delegates: 'delegate', parallel: 'parallel', choice: 'choice' }[block.body.type] || 'flow';
          body.heads.forEach(id => connect(node.id, id, relation));
        }
        // Completion of an agent includes its nested calls, just as in the
        // original workflow projection; its caller continues from this agent.
        return { heads: [node.id], tails: [node.id] };
      }
      const parts = (block.children || []).map(child => visit(child, parent));
      if (!parts.length) return { heads: [], tails: [] };
      if (block.type === 'sequence' || block.type === 'loop') {
        parts.slice(1).forEach((part, index) => parts[index].tails.forEach(from => part.heads.forEach(to => connect(from, to, 'flow'))));
        if (block.type === 'loop') parts[parts.length - 1].tails.forEach(from => parts[0].heads.forEach(to => connect(from, to, 'loop')));
        return { heads: parts[0].heads, tails: parts[parts.length - 1].tails };
      }
      return { heads: parts.flatMap(p => p.heads), tails: parts.flatMap(p => p.tails) };
    }
    visit(workflow.root);
    const owner = byId.get(ownerId), ownerRoot = roots.findIndex(id => id === ownerId);
    function assign(id, level) {
      const node = byId.get(id); node.level = level;
      node.children.forEach(child => assign(child, level + 1));
    }
    roots.forEach((id, i) => assign(id, ownerRoot > 0 && i < ownerRoot ? 0 : 1));
    if (owner) {
      const config = { id: `${owner.id}/configuration`, agentId: '__mas_session__',
        level: owner.level, parent: null, children: [], ...dimensions('', '__mas_session__') };
      nodes.push(config); byId.set(config.id, config); connect(owner.id, config.id, 'settings');
    }
    const levels = [...new Set(nodes.map(n => n.level))].sort((a, b) => a - b);
    const rows = levels.map(level => nodes.filter(n => n.level === level));
    const rowWidth = row => row.reduce((total, n) => total + n.w, 0) + Math.max(0, row.length - 1) * spec.gap;
    const available = Math.max(360, width - 64), packedRows = [];
    // Keep reports adjacent on the left and settings on the right. This avoids
    // routing a report link across the settings card or the entire experiment.
    rows.forEach(row => {
      row.sort((a, b) => (byId.get(a.parent)?.x || 0) - (byId.get(b.parent)?.x || 0));
      if (owner && row.includes(owner)) {
        const reports = roots.slice(ownerRoot + 1).map(id => byId.get(id)).filter(n => row.includes(n)).reverse();
        const config = byId.get(`${owner.id}/configuration`);
        const rest = row.filter(n => n !== owner && n !== config && !reports.includes(n));
        row.splice(0, row.length, ...rest, ...reports, owner, config);
      } else {
        // In unordered calls, put the agent with nested work near the middle.
        // Sequential siblings keep their YAML order from left to right.
        const parents = [...new Set(row.map(n => n.parent))];
        parents.forEach(parent => {
          if (byId.get(parent)?.bodyKind !== 'delegates') return;
          const children = row.filter(n => n.parent === parent);
          const major = children.reduce((best, n) => n.children.length > best.children.length ? n : best, children[0]);
          if (!major?.children.length) return;
          const rest = children.filter(n => n !== major);
          rest.splice(Math.floor(rest.length / 2), 0, major);
          const begin = row.findIndex(n => n.parent === parent); row.splice(begin, children.length, ...rest);
        });
      }
      // Keep sequential children together. Unordered calls may wrap; the
      // frame explains membership, not a chronology between visual rows.
      const units = [];
      row.forEach(node => {
        const last = units.at(-1), parent = byId.get(node.parent);
        if (last && ((parent && ['sequence', 'loop'].includes(parent.bodyKind) && last[0].parent === node.parent)
          || (node.agentId === '__mas_session__' && last.at(-1) === owner))) last.push(node);
        else units.push([node]);
      });
      let packed = [];
      units.forEach(unit => {
        if (packed.length && rowWidth([...packed, ...unit]) > available) { packedRows.push(packed); packed = []; }
        packed.push(...unit);
      });
      if (packed.length) packedRows.push(packed);
    });
    const w = Math.max(available, spec.w, ...packedRows.map(rowWidth)) + 64;
    let rowY = 28;
    packedRows.forEach((row, rowIndex) => {
      const gap = row.length > 1 ? Math.min(100, spec.gap + (w - 64 - rowWidth(row)) / (row.length - 1)) : 0;
      const span = row.reduce((sum, n) => sum + n.w, 0) + gap * (row.length - 1);
      let left = (w - span) / 2;
      const rowHeight = Math.max(...row.map(n => n.h));
      row.forEach(node => { node.x = left; node.y = rowY + (rowHeight - node.h) / 2; node.row = rowIndex; left += node.w + gap; });
      rowY += rowHeight + spec.step;
    });
    const groups = [];
    nodes.forEach(parent => {
      const children = parent.children.map(id => byId.get(id));
      if (children.length < 2) return;
      [...new Set(children.map(n => n.row))].forEach(row => {
      const siblings = children.filter(n => n.row === row);
      const left = Math.min(...siblings.map(n => n.x)), right = Math.max(...siblings.map(n => n.x + n.w));
      const top = Math.min(...siblings.map(n => n.y)), bottom = Math.max(...siblings.map(n => n.y + n.h));
      groups.push({ id: `${parent.id}/lane/${row}`, kind: 'lane', labelAgentId: parent.agentId,
        x: left - 14, y: top - 28, w: right - left + 28, h: bottom - top + 42 });
      });
    });
    const unique = [...new Map(edges.map(e => [`${e.from}|${e.to}|${e.relation}`, e])).values()];
    unique.forEach(edge => { edge.points = route(byId.get(edge.from), byId.get(edge.to), nodes); edge.straight = true; });
    return { w, h: Math.max(...nodes.map(n => n.y + n.h), 0) + 32, nodes, groups, edges: unique, variant: design, fallback: false };
  }
  const path = points => points.map((p, i) => `${i ? 'L' : 'M'}${p.x},${p.y}`).join(' ');
  window.AgentTreeVariants = { designs, layout, path };
})();
