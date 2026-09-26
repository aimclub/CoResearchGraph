/* Workflow geometry. SVG rendering/camera follow the embedded FEDOT viewer;
 * ordered blocks are placed by the repository's bundled ELK engine. */
(() => {
  'use strict';
  const W = 224, H = 88, GAP = 28, STEP = 40, PAD = 16;
  const point = (x, y) => ({ x, y });
  const empty = () => ({ w: 0, h: 0, nodes: [], groups: [], edges: [],
    entry: point(0, 0), exit: point(0, 0), top: point(0, 0), bottom: point(0, 0), left: point(0, 0), right: point(0, 0) });
  function merge(target, source, x, y) {
    source.nodes.forEach(n => target.nodes.push({ ...n, x: n.x + x, y: n.y + y }));
    source.groups.forEach(g => target.groups.push({ ...g, x: g.x + x, y: g.y + y }));
    source.edges.forEach(e => target.edges.push({ ...e, points: e.points.map(p => point(p.x + x, p.y + y)) }));
  }
  const shifted = (p, x, y) => point(p.x + x, p.y + y);
  const link = (from, to, a, b, relation = 'flow', arrow = true, direction = 'DOWN') => ({
    from, to, relation, arrow,
    points: direction === 'RIGHT'
      ? [a, point((a.x + b.x) / 2, a.y), point((a.x + b.x) / 2, b.y), b]
      : [a, point(a.x, (a.y + b.y) / 2), point(b.x, (a.y + b.y) / 2), b],
  });
  // A block exposes routes between its boundary and actual cards. Keeping the
  // whole route prevents nested sequences from attaching to empty box centers.
  function portRoute(block, direction, output = false, x = 0, y = 0) {
    const side = direction === 'RIGHT' ? (output ? 'right' : 'left') : (output ? 'bottom' : 'top');
    return (block[output ? 'outputs' : 'inputs']?.[direction] || [block[side]])
      .map(p => shifted(p, x, y));
  }
  function connect(from, to, source, target, relation = 'flow', arrow = true, direction = 'DOWN') {
    const edge = link(from, to, source[source.length - 1], target[0], relation, arrow, direction);
    edge.points = [...source, ...edge.points.slice(1, -1), ...target];
    return edge;
  }

  async function layout(workflow, { width = 1100, engine = null } = {}) {
    let fallback = !engine;
    // Keep the overview deliberately wide. At desktop sizes four peers fit in
    // one logical row and the camera, rather than the graph, owns navigation.
    const columns = width >= 900 ? 4 : 3;
    const occurrence = new Map(workflow.instances.map(item => [item.id, item]));
    const owner = workflow.configuration?.ownerInstanceId;

    async function sequence(children, direction = 'DOWN') {
      if (!children.length) return empty();
      if (children.length === 1) return children[0];
      let places;
      if (!fallback) {
        try {
          const result = await engine.layout({
            id: 'sequence',
            layoutOptions: { 'elk.algorithm': 'layered', 'elk.direction': direction,
              'elk.edgeRouting': 'ORTHOGONAL', 'elk.padding': '[top=0,left=0,bottom=0,right=0]',
              'elk.layered.spacing.nodeNodeBetweenLayers': String(STEP),
              'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES' },
            children: children.map((c, i) => ({ id: String(i), width: c.w, height: c.h })),
            edges: children.slice(1).map((_, i) => ({ id: `e${i}`, sources: [String(i)], targets: [String(i + 1)] })),
          });
          const byId = new Map(result.children.map(c => [c.id, c]));
          places = children.map((_, i) => byId.get(String(i)));
          const axis = direction === 'RIGHT' ? 'x' : 'y', size = direction === 'RIGHT' ? 'w' : 'h';
          if (places.some((p, i) => !p || !Number.isFinite(p[axis]) || (i && p[axis] < places[i - 1][axis] + children[i - 1][size]))) {
            throw new Error('Invalid ordered block layout');
          }
        } catch (error) {
          fallback = true;
          console.warn('Workflow layout: using ordered fallback', error);
        }
      }
      const out = empty();
      const horizontal = direction === 'RIGHT';
      if (horizontal) out.h = Math.max(...children.map(c => c.h));
      else out.w = Math.max(...children.map(c => c.w));
      let cursor = 0;
      const positions = children.map((c, i) => {
        const offset = places ? places[i][horizontal ? 'x' : 'y'] - places[0][horizontal ? 'x' : 'y'] : cursor;
        cursor = offset + (horizontal ? c.w : c.h) + STEP;
        return horizontal ? { x: offset, y: (out.h - c.h) / 2 } : { x: (out.w - c.w) / 2, y: offset };
      });
      children.forEach((child, i) => {
        const p = positions[i];
        merge(out, child, p.x, p.y);
        if (i) {
          const before = children[i - 1], prev = positions[i - 1];
          out.edges.push(connect(before.lastId, child.firstId,
            portRoute(before, direction, true, prev.x, prev.y),
            portRoute(child, direction, false, p.x, p.y), 'flow', true, direction));
        }
      });
      const last = children.length - 1;
      if (horizontal) out.w = positions[last].x + children[last].w;
      else out.h = positions[last].y + children[last].h;
      const inputDown = portRoute(children[0], 'DOWN', false, positions[0].x, positions[0].y);
      const inputRight = portRoute(children[0], 'RIGHT', false, positions[0].x, positions[0].y);
      const outputDown = portRoute(children[last], 'DOWN', true, positions[last].x, positions[last].y);
      const outputRight = portRoute(children[last], 'RIGHT', true, positions[last].x, positions[last].y);
      out.top = point(inputDown[0].x, 0); out.bottom = point(outputDown[outputDown.length - 1].x, out.h);
      out.left = point(0, inputRight[0].y); out.right = point(out.w, outputRight[outputRight.length - 1].y);
      out.inputs = { DOWN: [out.top, ...inputDown], RIGHT: [out.left, ...inputRight] };
      out.outputs = { DOWN: [...outputDown, out.bottom], RIGHT: [...outputRight, out.right] };
      out.entry = horizontal ? out.left : out.top; out.exit = horizontal ? out.right : out.bottom;
      out.firstId = children[0].firstId; out.lastId = children[last].lastId;
      return out;
    }

    async function pipeline(children) {
      const ownerIndex = children.findIndex(child => child.nodes.some(node => node.id === owner));
      if (ownerIndex < 0) return sequence(children);
      const preparation = await sequence(children.slice(0, ownerIndex), 'RIGHT');
      // The final report stays to the right of the coordinator scope: the arrow
      // still conveys completion order without adding another tall screen.
      const main = await sequence(children.slice(ownerIndex), 'RIGHT');
      return sequence([preparation, main].filter(block => block.nodes.length), 'DOWN');
    }

    function branches(children, block) {
      if (!children.length) return empty();
      if (children.length === 1) return children[0];
      const out = empty(), rows = [], joins = [], routedRows = [];
      // Unordered leaf alternatives go together. Large modules get a full row,
      // avoiding a tall, mostly empty row next to a multi-step experiment.
      const small = children.filter(c => c.w <= W + 2 * PAD);
      const large = children.filter(c => c.w > W + 2 * PAD);
      if (block.presentationOrder === 'declared') children.forEach(c => rows.push([c]));
      else {
        for (let i = 0; i < small.length; i += columns) rows.push(small.slice(i, i + columns));
        large.forEach(c => rows.push([c]));
      }
      const rowWidth = row => row.reduce((total, c) => total + c.w, 0) + GAP * (row.length - 1);
      out.w = Math.max(...rows.map(rowWidth)) + 48;
      let y = 54;
      out.entry = point(out.w / 2, 0);
      rows.forEach(row => {
        let x = (out.w - rowWidth(row)) / 2;
        const railY = y - 18, ports = [];
        row.forEach(child => {
          merge(out, child, x, y);
          const input = portRoute(child, 'DOWN', false, x, y), end = input[0];
          const relation = block.type === 'delegates' ? 'delegate' : block.type;
          ports.push(end.x);
          out.edges.push({ from: null, to: child.firstId, relation, arrow: true,
            points: [point(end.x, railY), ...input] });
          if (block.type === 'parallel' || block.type === 'choice') {
            const output = portRoute(child, 'DOWN', true, x, y), start = output[output.length - 1];
            const join = { from: child.lastId, to: null, relation, arrow: false,
              points: [...output, point(start.x, y + child.h + 12), point(out.w - 10, y + child.h + 12)] };
            out.edges.push(join); joins.push(join);
          }
          x += child.w + GAP;
        });
        routedRows.push({ railY, ports });
        y += Math.max(...row.map(c => c.h)) + STEP;
      });
      out.h = y - STEP + 24;
      out.exit = point(out.w / 2, out.h);
      const relation = block.type === 'delegates' ? 'delegate' : block.type;
      const spineX = 12, firstRail = routedRows[0].railY, lastRail = routedRows[routedRows.length - 1].railY;
      // One visible trunk replaces a stack of identical edges from the caller.
      out.edges.unshift({ from: null, to: null, relation, arrow: false,
        points: [out.entry, point(out.entry.x, firstRail - 14), point(spineX, firstRail - 14), point(spineX, lastRail)] });
      routedRows.forEach(row => out.edges.unshift({ from: null, to: null, relation, arrow: false,
        points: [point(spineX, row.railY), point(Math.max(spineX, ...row.ports), row.railY)] }));
      if (block.type === 'parallel' || block.type === 'choice') {
        // Join every parallel branch, or the selected choice branch, before
        // the following sequential block. Delegated calls have no such join.
        joins.forEach(e => {
          e.points.push(point(out.w - 10, out.h), out.exit);
        });
      }
      out.top = out.entry; out.bottom = out.exit;
      out.left = point(0, out.h / 2); out.right = point(out.w, out.h / 2);
      out.groups.unshift({ id: block.id, kind: block.type, x: 0, y: 0, w: out.w, h: out.h });
      out.firstId = children[0].firstId; out.lastId = children[children.length - 1].lastId;
      return out;
    }

    async function visit(block) {
      if (block.type !== 'agent') {
        const children = [];
        for (const child of block.children || []) children.push(await visit(child));
        if (['choice', 'delegates', 'parallel'].includes(block.type)) return branches(children, block);
        const wide = block.id.includes('/ExperimentModuleAgent/body');
        const content = block.id === 'workflow' ? await pipeline(children) : await sequence(children, wide ? 'RIGHT' : 'DOWN');
        if (block.type !== 'loop') return content;
        const out = empty();
        out.w = content.w + 32; out.h = content.h + 56;
        merge(out, content, 16, 36);
        out.entry = shifted(content.entry, 16, 36); out.exit = shifted(content.exit, 16, 36);
        const inputDown = portRoute(content, 'DOWN', false, 16, 36), inputRight = portRoute(content, 'RIGHT', false, 16, 36);
        const outputDown = portRoute(content, 'DOWN', true, 16, 36), outputRight = portRoute(content, 'RIGHT', true, 16, 36);
        out.top = point(inputDown[0].x, 0); out.bottom = point(outputDown[outputDown.length - 1].x, out.h);
        out.left = point(0, inputRight[0].y); out.right = point(out.w, outputRight[outputRight.length - 1].y);
        out.inputs = { DOWN: [out.top, ...inputDown], RIGHT: [out.left, ...inputRight] };
        out.outputs = { DOWN: [...outputDown, out.bottom], RIGHT: [...outputRight, out.right] };
        out.firstId = content.firstId; out.lastId = content.lastId;
        out.groups.unshift({ id: block.id, kind: 'loop', x: 0, y: 0, w: out.w, h: out.h });
        out.edges.push({ from: content.lastId, to: content.firstId, relation: 'loop', arrow: true,
          points: [...outputDown, point(out.bottom.x, out.h - 8), point(8, out.h - 8), point(8, 28), point(out.top.x, 28), ...inputDown] });
        return out;
      }
      const body = block.body ? await visit(block.body) : null;
      const out = empty(), isOwner = block.id === owner;
      out.w = Math.max(body ? body.w + 2 * PAD : W, isOwner ? W * 3 + 80 + 2 * PAD : W);
      const x = (out.w - W) / 2, y = body ? PAD : 0;
      out.h = body ? PAD + H + STEP + body.h + PAD : H + (isOwner ? STEP / 2 : 0);
      out.nodes.push({ id: block.id, agentId: block.agentId, x, y, w: W, h: H, ...occurrence.get(block.id) });
      out.top = point(x + W / 2, y); out.bottom = point(out.w / 2, out.h);
      out.left = point(0, y + H / 2);
      // The coordinator's settings card occupies its right side. Exit below
      // that card through the gap above delegated branches, then go to reports.
      const outputStart = isOwner ? point(x + W - 24, y + H) : point(x + W, y + H / 2);
      const exitY = isOwner ? y + H + STEP / 2 : y + H / 2;
      out.right = point(out.w, exitY);
      out.inputs = { DOWN: [out.top], RIGHT: [out.left, point(x, y + H / 2)] };
      out.outputs = {
        RIGHT: [outputStart, point(outputStart.x, exitY), out.right],
        DOWN: body ? [outputStart, point(outputStart.x, exitY), point(out.w - PAD / 2, exitY),
          point(out.w - PAD / 2, out.h - PAD / 2), point(out.bottom.x, out.h - PAD / 2), out.bottom]
          : [point(x + W / 2, y + H), out.bottom],
      };
      out.entry = out.top; out.exit = out.bottom;
      out.firstId = block.id; out.lastId = block.id;
      if (body) {
        const bx = (out.w - body.w) / 2, by = PAD + H + STEP;
        merge(out, body, bx, by);
        const delegated = block.body.relation === 'delegate' || block.body.type === 'delegates';
        const branching = ['choice', 'delegates', 'parallel'].includes(block.body.type) && block.body.children.length > 1;
        out.edges.push(connect(block.id, body.firstId, [point(x + W / 2, y + H)],
          portRoute(body, 'DOWN', false, bx, by), delegated ? 'delegate' : 'flow', !branching));
        out.edges.forEach(e => { if (!e.from) e.from = block.id; if (!e.to) e.to = block.id; });
        out.groups.unshift({ id: block.id, agentId: block.agentId, kind: block.composite ? 'module' : 'scope', x: 0, y: 0, w: out.w, h: out.h });
      }
      if (isOwner) {
        const configId = `${block.id}/configuration`, configX = x + W + 40;
        out.nodes.push({ id: configId, agentId: '__mas_session__', x: configX, y, w: W, h: H });
        out.edges.push({ from: block.id, to: configId, relation: 'settings', arrow: true,
          points: [point(x + W, y + H / 2), point(configX, y + H / 2)] });
      }
      return out;
    }
    const result = await visit(workflow.root);
    result.fallback = fallback;
    return result;
  }

  // Rounded orthogonal paths keep edges outside card rectangles.
  function path(points) {
    const clean = points.filter((p, i) => !i || p.x !== points[i - 1].x || p.y !== points[i - 1].y);
    if (!clean.length) return '';
    let d = `M${clean[0].x},${clean[0].y}`;
    for (let i = 1; i < clean.length - 1; i++) {
      const a = clean[i - 1], b = clean[i], c = clean[i + 1];
      const ab = Math.hypot(b.x - a.x, b.y - a.y), bc = Math.hypot(c.x - b.x, c.y - b.y);
      const r = Math.min(8, ab / 2, bc / 2);
      const pre = point(b.x + (a.x - b.x) * r / ab, b.y + (a.y - b.y) * r / ab);
      const post = point(b.x + (c.x - b.x) * r / bc, b.y + (c.y - b.y) * r / bc);
      d += ` L${pre.x},${pre.y} Q${b.x},${b.y} ${post.x},${post.y}`;
    }
    const last = clean[clean.length - 1];
    return d + ` L${last.x},${last.y}`;
  }
  window.AgentTreeLayout = { layout, path, W, H };
})();
