// =========================================================================
// Call Graph — live map of the agents a run went through, in the side nav
//
// The entities are the agents themselves. Modules — sequential, parallel and
// loop agents, and routers — do no work of their own, so they are never drawn:
// the agents inside them take their place, one after another or side by side.
// The orchestrator is drawn once; every agent then hangs off the one that ran
// before it, so a run reads as a chain rather than a star. Agents running at
// the same time share a predecessor and sit side by side. An agent that calls
// agents of its own starts a chain below it (entered by a dashed line), and
// the run carries on from the end of that chain.
//
// Three sources, best first: agent runs (`agent_start`/`agent_end` on the
// `tool_activity` stream); delegation calls alone, from records older than
// those; and the orchestrator's function calls on `agent_event`, for sessions
// recorded before tool activity reached the disk. The last two know a module
// only by name, so it is unfolded into its agents from the profile config.
// =========================================================================
    const CallGraph = (() => {
      // Metrics at the default rail width; render() scales them with the rail.
      const BASE_AVAIL = 220, NODE_H = 26, LAYER_GAP = 22, COL_GAP = 8, PAD = 10, MAX_W = 260, MIN_W = 72;
      const FONT = 10.5, CHAR_W = 6.1, SCALE_MIN = 0.9, SCALE_MAX = 1.6;
      const SVG_NS = 'http://www.w3.org/2000/svg';

      // Agent metadata from /api/agents. Until it arrives, a name is judged by
      // its shape alone; its arrival replays the log.
      let agentNames = new Set();
      let titles = {};
      let composites = {};
      let roots = new Set(['OrchestratorAgent', 'RootOrchestrator']);
      const COMPOSITE_CLASSES = new Set(['SequentialAgent', 'ParallelAgent', 'LoopAgent']);

      let log = [];
      let runsModel = createModel();
      let callsModel = createModel();
      let legacyModel = createModel();
      let renderQueued = false;

      function newCtx(ownerId) {
        return { owner: ownerId, tail: ownerId == null ? [] : [ownerId], preds: [], batch: [], open: new Set() };
      }

      function createModel() {
        const m = {
          nodes: [], byId: new Map(), byCall: new Map(), bySpawn: new Map(), runs: [],
          root: null, rootCtx: newCtx(null), seq: 0,
        };
        return m;
      }

      function isComposite(name, agentClass) {
        return Object.prototype.hasOwnProperty.call(composites, name)
          || COMPOSITE_CLASSES.has(agentClass)
          || /^Module[A-Z0-9]*_/.test(name) || name === 'ResearchPipeline';
      }

      function isRoot(name) {
        return roots.has(name);
      }

      function addNode(m, ctx, name, preds, status) {
        const node = {
          id: m.seq++, name, preds, status, parentCtx: ctx,
          nested: ctx !== m.rootCtx && preds.length === 1 && preds[0] === ctx.owner,
        };
        node.ctx = newCtx(node.id);
        m.nodes.push(node);
        m.byId.set(node.id, node);
        return node;
      }

      // The orchestrator: drawn once, after whatever ran before it.
      function ensureRoot(m, name, ctx = m.rootCtx) {
        if (m.root) return m.root;
        const root = addNode(m, ctx, name || 'OrchestratorAgent', ctx.tail.slice(), 'root');
        root.nested = false;
        ctx.tail = [root.id];
        m.root = root;
        return root;
      }

      // A new entry in `ctx`. Entries opened while others are still open run
      // alongside them and share their predecessors.
      function beginBatch(ctx) {
        if (ctx.open.size === 0) {
          ctx.preds = ctx.tail.slice();
          ctx.batch = [];
        }
      }

      function openNode(m, ctx, name, key) {
        beginBatch(ctx);
        const node = addNode(m, ctx, name, ctx.preds, 'running');
        ctx.batch.push(node.id);
        node.callKey = key || `anon-${node.id}`;
        ctx.open.add(node.callKey);
        return node;
      }

      // The run carries on from where each finished entry's own work ended:
      // the last of the agents it called, or the entry itself if none.
      function settle(m, ctx) {
        if (ctx.open.size === 0) ctx.tail = ctx.batch.flatMap(id => m.byId.get(id).ctx.tail);
      }

      function closeNode(m, node, failed) {
        if (!node || node.status !== 'running') return;
        node.status = failed ? 'error' : 'done';
        node.parentCtx.open.delete(node.callKey);
        settle(m, node.parentCtx);
      }

      // A module known only by name: lay its agents out as its config runs
      // them. Returns the ids the run continues from.
      function unfold(m, ctx, name, preds, depth = 0) {
        const comp = composites[name];
        if (!comp || depth > 8) {
          if (isComposite(name) || isInternalAgent(name) || isRoot(name)) return preds;
          return [addNode(m, ctx, name, preds, 'done').id];
        }
        if (comp.kind === 'parallel') {
          const exits = comp.children.flatMap(child => unfold(m, ctx, child, preds, depth + 1));
          return exits.length ? exits : preds;
        }
        if (comp.kind === 'switch') return preds;  // which child ran is not recorded
        return comp.children.reduce((p, child) => unfold(m, ctx, child, p, depth + 1), preds);
      }

      // Put a finished module's agents into `ctx` as one entry of its batch.
      function insertModule(m, ctx, name) {
        beginBatch(ctx);
        const exits = unfold(m, ctx, name, ctx.preds);
        exits.forEach(id => { if (!ctx.batch.includes(id)) ctx.batch.push(id); });
        settle(m, ctx);
      }

      // ── Source 1: agent runs ─────────────────────────────────────────────
      function findRun(m, name, instance, oldest) {
        const match = r => r.name === name && (!instance || !r.instance || r.instance === instance);
        return oldest ? m.runs.find(match) : m.runs.slice().reverse().find(match);
      }

      function isSequential(name, agentClass) {
        const comp = composites[name];
        return agentClass === 'SequentialAgent' || agentClass === 'LoopAgent'
          || (comp && (comp.kind === 'sequential' || comp.kind === 'loop'));
      }

      // The run a start belongs to. Most name their parent. The inner
      // sessions of a custom agent do not, and a router names itself though it
      // never reports a run: those belong to the agent whose name they extend
      // (TZSpecAgent_task → TZSpecAgent), else to the innermost running agent
      // that was itself placed by name.
      function parentRun(m, d) {
        const named = d.parent && findRun(m, d.parent, d.parent_instance, false);
        if (named) return named;
        const active = m.runs.slice().reverse();
        const sibling = active.find(r => r.name === d.author && r.named);
        return active.find(r => d.author.startsWith(r.name + '_'))
          || (sibling && sibling.parent)
          || active.find(r => r.node && r.named)
          || null;
      }

      // An agent cannot outlive the one that ran it: ending a run ends every
      // run still open below it, innermost first.
      function endRun(m, run, failed) {
        m.runs.filter(r => r.parent === run).forEach(r => endRun(m, r, false));
        const i = m.runs.indexOf(run);
        if (i >= 0) m.runs.splice(i, 1);
        if (run.node) closeNode(m, run.node, failed);
      }

      function runStart(d) {
        const m = runsModel;
        const name = d.author;
        if (!name || name === 'system') return;
        const parent = parentRun(m, d);
        // A sequential module starts its next agent only after the last one
        // finished, whether or not that one reported its end.
        if (parent && parent.sequential) {
          m.runs.filter(r => r.parent === parent).forEach(r => endRun(m, r, false));
        }
        const ctx = parent ? parent.childCtx : m.rootCtx;
        const run = {
          name, instance: d.agent_instance || null, parent, node: null, childCtx: ctx,
          named: !!d.parent, sequential: isSequential(name, d.agent_class),
        };
        if (isRoot(name) && ctx === m.rootCtx) {
          ensureRoot(m, name, ctx);   // its calls continue the main chain
        } else if (!isComposite(name, d.agent_class) && !isInternalAgent(name) && !(m.root && name === m.root.name)) {
          run.node = openNode(m, ctx, name, null);
          run.childCtx = run.node.ctx;
          if (d.spawn_call_id) m.bySpawn.set(d.spawn_call_id, run.node);
        }
        m.runs.push(run);
      }

      function runEnd(d) {
        const m = runsModel;
        const run = findRun(m, d.author, d.agent_instance, true);
        if (run) endRun(m, run, !!d.failed);
      }

      function runError(d) {
        const m = runsModel;
        const target = d.target_agent || d.tool;
        const node = (d.call_id && m.bySpawn.get(d.call_id))
          || m.nodes.slice().reverse().find(n => n.name === target && n !== m.root);
        if (!node) return;
        if (node.status === 'running') closeNode(m, node, true);
        else node.status = 'error';
      }

      // ── Source 2: delegation calls (records without agent runs) ──────────
      function callerCtx(m, author) {
        const node = m.nodes.slice().reverse().find(n => n !== m.root && n.status === 'running' && n.name === author);
        if (node) return node.ctx;
        if (isRoot(author) || !m.root) ensureRoot(m, isRoot(author) ? author : 'OrchestratorAgent');
        return m.rootCtx;
      }

      function openCall(m, author, target, callId) {
        if (isRoot(target) || (m.root && target === m.root.name)) return;
        const ctx = callerCtx(m, author);
        if (isComposite(target)) { insertModule(m, ctx, target); return; }
        const node = openNode(m, ctx, target, callId);
        m.byCall.set(node.callKey, node);
      }

      function closeCall(m, target, callId, failed) {
        const node = (callId && m.byCall.get(callId))
          || m.nodes.find(n => n !== m.root && n.status === 'running' && n.name === target);
        closeNode(m, node, failed);
      }

      // A call hands work to an agent when it names one. Older records say so
      // only through the name: a configured agent, an ...Agent, or an
      // AgentTool's lone `request` argument.
      function delegationTarget(name, args, declared) {
        if (!name) return null;
        if (name === 'transfer_to_agent') {
          const target = args && (args.agent_name || args.agentName);
          return target ? String(target) : null;
        }
        const n = String(name);
        if (declared || agentNames.has(n) || /Agent$/.test(n) || isComposite(n)) return n;
        if (args && typeof args === 'object' && !Array.isArray(args)) {
          const keys = Object.keys(args);
          if (keys.length === 1 && keys[0] === 'request') return n;
        }
        return null;
      }

      function applyActivity(d) {
        if (d.phase === 'agent_start') { runStart(d); return; }
        if (d.phase === 'agent_end') { runEnd(d); return; }
        if (d.phase === 'error' && d.is_delegation) runError(d);

        const m = callsModel;
        if (d.phase === 'call') {
          // Current records mark delegations (and always carry `parent`);
          // only older ones, without either, are judged by the name.
          if (!d.is_delegation && 'parent' in d) return;
          const target = delegationTarget(d.target_agent || d.tool, d.args, d.is_delegation);
          if (!target || isInternalAgent(target)) return;
          openCall(m, d.author || 'system', target, d.call_id);
        } else if (d.phase === 'result' || d.phase === 'error') {
          if (!d.is_delegation && !(d.call_id && m.byCall.has(d.call_id))) return;
          closeCall(m, String(d.target_agent || d.tool), d.call_id, d.phase === 'error');
        }
      }

      // ── Source 3: the orchestrator's own function calls ──────────────────
      function applyAgentEvent(d) {
        const m = legacyModel;
        const author = d.author;
        if (!author || isInternalAgent(author)) return;
        const targets = (d.tool_calls || [])
          .map(tc => ({ tc, target: delegationTarget(tc && tc.name, tc && tc.args, false) }))
          .filter(({ target }) => target && !isInternalAgent(target));
        if (targets.length) {
          // An agent takes its next turn only once every call of its last one
          // has answered, so a call still open here was cut off (a stop, a
          // restored checkpoint) rather than running alongside.
          const ctx = callerCtx(m, author);
          m.nodes.filter(n => n.parentCtx === ctx && n.status === 'running').forEach(n => closeNode(m, n, false));
        }
        targets.forEach(({ tc, target }) => openCall(m, author, target, tc.id || null));
        (d.tool_responses || []).forEach(tr => {
          if (tr && tr.name) closeCall(m, String(tr.name), tr.id || null, false);
        });
      }

      function applyIdle() {
        // Innermost first, so a caller closes onto its callees' final tail.
        [runsModel, callsModel, legacyModel].forEach(m => {
          m.runs = [];
          m.nodes.slice().reverse().forEach(n => { if (n.status === 'running') closeNode(m, n, false); });
        });
      }

      function apply(entry) {
        if (entry.kind === 'activity') applyActivity(entry.data);
        else if (entry.kind === 'agent_event') applyAgentEvent(entry.data);
        else applyIdle();
      }

      function push(entry) {
        log.push(entry);
        apply(entry);
        schedule();
      }

      const ACTIVITY_PHASES = ['agent_start', 'agent_end', 'call', 'result', 'error'];

      function feed(data) {
        if (!data || !ACTIVITY_PHASES.includes(data.phase)) return;
        push({ kind: 'activity', data });
      }

      function feedAgentEvent(data) {
        if (!data || !(data.tool_calls || data.tool_responses)) return;
        push({ kind: 'agent_event', data: { author: data.author, tool_calls: data.tool_calls, tool_responses: data.tool_responses } });
      }

      // The run is over: whatever is still marked running has finished.
      function markIdle() {
        push({ kind: 'idle' });
      }

      function rebuild() {
        runsModel = createModel();
        callsModel = createModel();
        legacyModel = createModel();
        log.forEach(apply);
        schedule();
      }

      function setAgentMeta(meta) {
        agentNames = new Set((meta.names || []).map(String));
        titles = meta.titles || {};
        composites = meta.composites || {};
        roots = new Set(['OrchestratorAgent', 'RootOrchestrator', ...(meta.roots || [])]);
        if (log.length) rebuild();
        else schedule();
      }

      // Sessions whose transcript predates agent-run records: their runs
      // come from the execution graph instead, once, after the replay.
      let generation = 0;
      async function loadHistory(userId, sessionId) {
        if (!userId || !sessionId) return;
        const agents = m => m.nodes.length - (m.root ? 1 : 0);
        if (agents(runsModel) > 0) return;
        const mine = generation;
        try {
          const resp = await fetch(`/api/users/${encodeURIComponent(userId)}/sessions/${encodeURIComponent(sessionId)}/agent-runs`);
          if (!resp.ok) return;
          const data = await resp.json();
          // Another session opened meanwhile, or live runs arrived first.
          if (mine !== generation || agents(runsModel) > 0) return;
          (data.events || []).forEach(event => log.push({ kind: 'activity', data: event }));
          log.push({ kind: 'idle' });
          rebuild();
        } catch (_) { /* the transcript's own sources remain */ }
      }

      function reset() {
        generation++;
        camera = null;
        log = [];
        runsModel = createModel();
        callsModel = createModel();
        legacyModel = createModel();
        render();
      }

      // The richest source that saw a single agent wins.
      function shown() {
        const agents = m => m.nodes.length - (m.root ? 1 : 0);
        return [runsModel, callsModel, legacyModel].find(m => agents(m) > 0) || runsModel;
      }

      function schedule() {
        if (renderQueued) return;
        renderQueued = true;
        requestAnimationFrame(() => { renderQueued = false; render(); });
      }

      // The Russian title from the agent config, else the bare name.
      function displayName(name) {
        const title = titles[name];
        if (title) return title;
        const s = String(name).replace(/Agent$/, '');
        return s || String(name);
      }

      function svgEl(tag, attrs, text) {
        const el = document.createElementNS(SVG_NS, tag);
        Object.entries(attrs || {}).forEach(([k, v]) => el.setAttribute(k, String(v)));
        if (text !== undefined) el.textContent = text;
        return el;
      }

      function fitLabel(text, width, k) {
        const max = Math.max(3, Math.floor((width - 26 * k) / (CHAR_W * k)));
        return text.length > max ? text.slice(0, max - 1) + '…' : text;
      }

      function render() {
        const scroller = document.getElementById('call-graph-scroll');
        const svg = document.getElementById('call-graph-svg');
        const empty = document.getElementById('call-graph-empty');
        const count = document.getElementById('call-graph-count');
        if (!scroller || !svg) return;
        const { nodes, root } = shown();
        const calls = nodes.filter(n => n !== root);
        if (count) count.textContent = calls.length ? String(calls.length) : '';
        empty.classList.toggle('hidden', nodes.length > 0);
        svg.classList.toggle('hidden', nodes.length === 0);
        svg.replaceChildren();
        if (!nodes.length) return;

        const byId = new Map(nodes.map(n => [n.id, n]));
        const layers = [];
        nodes.forEach(n => {
          n.layer = n.preds.length ? 1 + Math.max(...n.preds.map(p => byId.get(p).layer)) : 0;
          (layers[n.layer] = layers[n.layer] || []).push(n);
        });

        const avail = Math.max(120, scroller.clientWidth - PAD * 2);
        // A wider rail draws a bigger graph, not just a wider one — as long as
        // it still fits the panel's height; past that it scrolls at base size.
        const unitH = layers.length * NODE_H + (layers.length - 1) * LAYER_GAP;
        const fitH = (scroller.clientHeight - PAD * 2) / unitH;
        const k = Math.min(SCALE_MAX, Math.max(SCALE_MIN, Math.min(avail / BASE_AVAIL, fitH)));
        const nodeH = NODE_H * k, layerGap = LAYER_GAP * k, colGap = COL_GAP * k;
        // Each row sizes its own boxes: a lone call gets a wide box for its
        // (long, Russian) title; a parallel row shares the width.
        const rowWidth = count => Math.max(MIN_W * k, Math.min(MAX_W * k, (avail - colGap * (count - 1)) / count));
        const contentW = Math.max(avail, ...layers.map(l => l.length * rowWidth(l.length) + (l.length - 1) * colGap));
        const pos = new Map();
        layers.forEach((layer, li) => {
          const w = rowWidth(layer.length);
          const rowW = layer.length * w + (layer.length - 1) * colGap;
          const x0 = PAD + (contentW - rowW) / 2;
          layer.forEach((n, i) => pos.set(n.id, { x: x0 + i * (w + colGap), y: PAD + li * (nodeH + layerGap), w }));
        });
        const totalW = contentW + PAD * 2;
        const totalH = PAD * 2 + layers.length * nodeH + (layers.length - 1) * layerGap;
        const world = svgEl('g', { class: 'cg-world' });
        svg.appendChild(world);

        const edges = svgEl('g', { class: 'cg-edges' });
        nodes.forEach(n => n.preds.forEach(pid => {
          const a = pos.get(pid), b = pos.get(n.id);
          const x1 = a.x + a.w / 2, y1 = a.y + nodeH, x2 = b.x + b.w / 2, y2 = b.y;
          const my = (y1 + y2) / 2;
          edges.appendChild(svgEl('path', {
            d: `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`,
            class: `cg-edge${n.nested ? ' cg-edge-nested' : ''}${n.status === 'running' ? ' cg-edge-live' : ''}`,
          }));
        }));
        world.appendChild(edges);

        nodes.forEach(n => {
          const p = pos.get(n.id);
          const g = svgEl('g', { class: `cg-node cg-${n.status}`, transform: `translate(${p.x},${p.y})` });
          const label = displayName(n.name);
          g.appendChild(svgEl('title', {}, label === n.name ? n.name : `${label} (${n.name})`));
          g.appendChild(svgEl('rect', { width: p.w, height: nodeH, rx: 6 * k, class: 'cg-box' }));
          // Dot and title centred together in the box.
          const gap = 8 * k;
          const text = svgEl('text', {
            x: p.w / 2 + gap / 2, y: nodeH / 2 + 4 * k, 'font-size': FONT * k,
            'text-anchor': 'middle', class: 'cg-label',
          }, fitLabel(label, p.w, k));
          g.appendChild(text);
          world.appendChild(g);
          const tw = text.getComputedTextLength() || text.textContent.length * CHAR_W * k;
          g.insertBefore(svgEl('circle', {
            cx: p.w / 2 + gap / 2 - tw / 2 - gap, cy: nodeH / 2, r: 3 * k, class: 'cg-dot',
          }), text);
        });

        // Unless the reader has moved the view, the whole graph is fitted into
        // the middle of the panel — shrunk down to FIT_MIN at most; a graph
        // taller than that keeps its newest calls in sight.
        const W = scroller.clientWidth, H = scroller.clientHeight;
        const fit = Math.max(FIT_MIN, Math.min(1, W / totalW, H / totalH));
        const w = totalW * fit, h = totalH * fit;
        autoView = {
          tx: (W - w) / 2,
          ty: h <= H ? (H - h) / 2 : H - h,
          s: fit,
        };
        applyCamera();
      }

      // ── View: drag to pan, wheel to zoom, double-click to fit ────────────
      let camera = null;      // null: follow autoView
      let autoView = { tx: 0, ty: 0, s: 1 };
      const ZOOM_MIN = 0.25, ZOOM_MAX = 4, FIT_MIN = 0.85;

      function applyCamera() {
        const world = document.querySelector('#call-graph-svg .cg-world');
        const v = camera || autoView;
        if (world) world.setAttribute('transform', `translate(${v.tx},${v.ty}) scale(${v.s})`);
      }

      function initView(scroller) {
        let drag = null;
        scroller.addEventListener('pointerdown', event => {
          if (event.button !== 0) return;
          camera = camera || { ...autoView };
          drag = { x: event.clientX, y: event.clientY, tx: camera.tx, ty: camera.ty, id: event.pointerId };
          scroller.setPointerCapture(event.pointerId);
          scroller.classList.add('dragging');
        });
        scroller.addEventListener('pointermove', event => {
          if (!drag || event.pointerId !== drag.id) return;
          camera.tx = drag.tx + event.clientX - drag.x;
          camera.ty = drag.ty + event.clientY - drag.y;
          applyCamera();
        });
        const stop = event => {
          if (!drag || event.pointerId !== drag.id) return;
          drag = null;
          scroller.classList.remove('dragging');
        };
        scroller.addEventListener('pointerup', stop);
        scroller.addEventListener('pointercancel', stop);
        scroller.addEventListener('wheel', event => {
          event.preventDefault();
          camera = camera || { ...autoView };
          const rect = scroller.getBoundingClientRect();
          const px = event.clientX - rect.left, py = event.clientY - rect.top;
          const scale = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, camera.s * Math.exp(-event.deltaY * 0.0015)));
          // Zoom about the pointer: the point under it stays put.
          camera.tx = px - (px - camera.tx) * (scale / camera.s);
          camera.ty = py - (py - camera.ty) * (scale / camera.s);
          camera.s = scale;
          applyCamera();
        }, { passive: false });
        scroller.addEventListener('dblclick', () => { camera = null; applyCamera(); });
      }

      function init() {
        const scroller = document.getElementById('call-graph-scroll');
        if (!scroller) return;
        initView(scroller);
        if (window.ResizeObserver) new ResizeObserver(() => schedule()).observe(scroller);
        render();
      }

      return { init, reset, feed, feedAgentEvent, markIdle, setAgentMeta, loadHistory };
    })();

    document.addEventListener('DOMContentLoaded', () => CallGraph.init());
