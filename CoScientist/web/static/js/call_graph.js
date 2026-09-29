// =========================================================================
// Call Graph — the agents of a session and which of them ran, in the side nav
//
// The skeleton comes from the session's config (/call-graph): the pipeline
// stages one after another, the coordinator once, and every agent it can call
// fanning out from it as a branch of its own. Modules are never drawn: a
// sequential one becomes a chain of its agents, a parallel one or a router
// puts them side by side, and an agent's own calls hang off it. Until an
// agent runs, its place is drawn dashed.
//
// Runs are laid over it: each call of an agent is its own node in that
// agent's place, lit by its state (running, done, failed). Places no call
// reached keep their dashed placeholder, under the latest call only.
//
// Runs come from the richest record the session has: agent runs
// (`agent_start`/`agent_end` on `tool_activity`, or the execution graph of an
// older session); else delegation calls; else the orchestrator's function
// calls on `agent_event`. The model is rebuilt from the log on every frame
// it changed in, so late metadata or history never leaves it half-applied.
// =========================================================================
    const CallGraph = (() => {
      const NODE_W = 150, NODE_H = 26, H_GAP = 10, V_GAP = 26, PAD = 12;
      const FONT = 10.5, CHAR_W = 6.1;
      const SVG_NS = 'http://www.w3.org/2000/svg';
      const COMPOSITE_CLASSES = new Set(['SequentialAgent', 'ParallelAgent', 'LoopAgent']);

      // Agent metadata from /api/agents (all profiles) and the session's own
      // skeleton; either arriving re-renders from the log.
      let agentNames = new Set();
      let titles = {};
      let composites = {};
      let roots = new Set(['OrchestratorAgent', 'RootOrchestrator']);
      let skeleton = null;

      let log = [];
      let renderQueued = false;

      // ── Metadata ─────────────────────────────────────────────────────────
      function isComposite(name, agentClass) {
        return Object.prototype.hasOwnProperty.call(composites, name)
          || (skeleton && skeleton.composites.has(name))
          || COMPOSITE_CLASSES.has(agentClass)
          || /^Module[A-Z0-9]*_/.test(name) || name === 'ResearchPipeline';
      }

      function isSequential(name, agentClass) {
        const comp = composites[name];
        return agentClass === 'SequentialAgent' || agentClass === 'LoopAgent'
          || (comp && (comp.kind === 'sequential' || comp.kind === 'loop'));
      }

      // A call hands work to an agent when it names one. Older records say so
      // only through the name: a configured agent, an ...Agent, a module, or
      // an AgentTool's lone `request` argument.
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

      // ── Log → normalised run events ──────────────────────────────────────
      // { t: 'start', name, parent, parentInstance, instance, agentClass, spawn }
      // { t: 'end', name, instance, failed } · { t: 'error', target, callId } · { t: 'idle' }
      function runEvents() {
        const hasRuns = log.some(e => e.kind === 'activity' && e.data.phase === 'agent_start');
        if (hasRuns) return log.flatMap(fromRunRecord);
        const calls = synthFromCalls();
        return calls.some(e => e.t === 'start') ? calls : synthFromAgentEvents();
      }

      function fromRunRecord(entry) {
        if (entry.kind === 'idle') return [{ t: 'idle' }];
        if (entry.kind !== 'activity') return [];
        const d = entry.data;
        if (d.phase === 'agent_start') {
          return [{
            t: 'start', name: d.author, parent: d.parent, parentInstance: d.parent_instance,
            instance: d.agent_instance, agentClass: d.agent_class, spawn: d.spawn_call_id,
          }];
        }
        if (d.phase === 'agent_end') return [{ t: 'end', name: d.author, instance: d.agent_instance, failed: !!d.failed }];
        if (d.phase === 'error' && d.is_delegation) return [{ t: 'error', target: d.target_agent || d.tool, callId: d.call_id }];
        return [];
      }

      // A module known only by name ran its agents as its config says.
      function unfoldModule(name, instance, depth = 0) {
        const comp = composites[name];
        if (!comp || comp.kind === 'switch' || depth > 8) return [];
        const out = [];
        const child = (c, i) => {
          const id = `${instance}/${i}`;
          const inner = [{ t: 'start', name: c, parent: name, parentInstance: instance, instance: id },
            ...unfoldModule(c, id, depth + 1)];
          return { inner, end: { t: 'end', name: c, instance: id } };
        };
        const parts = comp.children.map(child);
        if (comp.kind === 'parallel') {
          parts.forEach(p => out.push(...p.inner));
          parts.forEach(p => out.push(p.end));
        } else {
          parts.forEach(p => out.push(...p.inner, p.end));
        }
        return out;
      }

      // Records of calls never say the coordinator itself ran: it did, as
      // soon as it called anyone.
      function synthStart(name, parent, instance, started) {
        const out = [];
        if (parent && roots.has(parent) && !started.has(parent)) {
          started.add(parent);
          out.push({ t: 'start', name: parent, instance: `root:${parent}` });
        }
        return [...out, { t: 'start', name, parent, instance }, ...unfoldModule(name, instance)];
      }

      function synthFromCalls() {
        const out = [];
        const started = new Set();
        const open = new Map();   // call id -> { name, instance }
        log.forEach((entry, i) => {
          if (entry.kind === 'idle') { out.push({ t: 'idle' }); started.clear(); return; }
          if (entry.kind !== 'activity') return;
          const d = entry.data;
          if (d.phase === 'call') {
            // Current records mark delegations (and always carry `parent`);
            // only older ones, without either, are judged by the name.
            if (!d.is_delegation && 'parent' in d) return;
            const target = delegationTarget(d.target_agent || d.tool, d.args, d.is_delegation);
            if (!target) return;
            const instance = `call:${d.call_id || i}`;
            open.set(d.call_id || `#${i}`, { name: target, instance });
            out.push(...synthStart(target, d.author, instance, started));
          } else if ((d.phase === 'result' || d.phase === 'error') && d.call_id && open.has(d.call_id)) {
            const run = open.get(d.call_id);
            open.delete(d.call_id);
            out.push({ t: 'end', name: run.name, instance: run.instance, failed: d.phase === 'error' });
          }
        });
        return out;
      }

      function synthFromAgentEvents() {
        const out = [];
        const started = new Set();
        const open = new Map();   // author -> [{ name, instance }]
        log.forEach((entry, i) => {
          if (entry.kind === 'idle') { out.push({ t: 'idle' }); started.clear(); return; }
          if (entry.kind !== 'agent_event') return;
          const d = entry.data;
          const author = d.author;
          if (!author) return;
          const mine = open.get(author) || [];
          const targets = (d.tool_calls || [])
            .map(tc => delegationTarget(tc && tc.name, tc && tc.args, false))
            .filter(Boolean);
          if (targets.length) {
            // An agent takes its next turn only once every call of its last
            // one answered: a call still open here was cut off.
            mine.splice(0).forEach(r => out.push({ t: 'end', name: r.name, instance: r.instance }));
            targets.forEach((target, j) => {
              const instance = `ae:${i}:${j}`;
              mine.push({ name: target, instance });
              out.push(...synthStart(target, author, instance, started));
            });
          }
          (d.tool_responses || []).forEach(tr => {
            const k = mine.findIndex(r => tr && r.name === tr.name);
            if (k >= 0) {
              const [r] = mine.splice(k, 1);
              out.push({ t: 'end', name: r.name, instance: r.instance });
            }
          });
          open.set(author, mine);
        });
        return out;
      }

      // ── Run events → instances laid over the skeleton ────────────────────
      function buildModel() {
        const slots = skeleton ? skeleton.slots : [];
        const slotById = new Map(slots.map(s => [s.id, s]));
        const dynamic = new Map();   // owner|name -> slot not in the config
        const top = { slot: null, name: null, children: [], order: -1 };
        const runs = [];
        const bySpawn = new Map();
        const instances = [];
        let rootInst = null;
        let order = 0;
        let tick = 0;   // position in the run stream: tells overlapping runs apart

        const isRootSlot = slot => skeleton ? slot && slot.id === skeleton.rootSlot : false;
        const slotIdOf = inst => (inst.slot ? inst.slot.id : null);

        function findRun(name, instance, oldest) {
          const match = r => r.name === name && (!instance || !r.instance || r.instance === instance);
          return oldest ? runs.find(match) : runs.slice().reverse().find(match);
        }

        // The run a start belongs to. Most name their parent. The inner
        // sessions of a custom agent do not, and a router names itself though
        // it never reports a run: those belong to the agent whose name they
        // extend (TZSpecAgent_task → TZSpecAgent), to the parent of a running
        // namesake, else to the innermost running agent placed by name.
        function parentRun(e) {
          const named = e.parent && findRun(e.parent, e.parentInstance, false);
          if (named) return named;
          const active = runs.slice().reverse();
          const sibling = active.find(r => r.name === e.name && r.named);
          return active.find(r => e.name.startsWith(r.name + '_'))
            || (sibling && sibling.parent)
            || active.find(r => r.inst && r.named)
            || null;
        }

        function findSlot(owner, name) {
          const ownerId = slotIdOf(owner);
          const slot = slots.find(s => s.agent === name && s.owner === ownerId)
            // Recorded as called by the step it follows (the planner "calling"
            // the coordinator after it): the next step of that chain.
            || slots.find(s => s.agent === name && ownerId !== null && s.parent === ownerId);
          if (slot) return slot;
          const key = `${ownerId}|${name}`;
          if (!dynamic.has(key)) {
            dynamic.set(key, { id: `d${dynamic.size}`, agent: name, parent: ownerId, owner: ownerId, kind: 'call', dynamic: true });
          }
          return dynamic.get(key);
        }

        // Where an instance hangs: under its owner, or — for a step of a
        // chain — under the latest run of the step before it.
        function hostFor(owner, slot) {
          const ownerId = slotIdOf(owner);
          let target = slot.parent;
          while (target != null && target !== ownerId) {
            let best = null;
            const walk = inst => inst.children.forEach(c => {
              if (c.slot && c.slot.id === target && (!best || c.order > best.order)) best = c;
              walk(c);
            });
            walk(owner);
            if (best) return best;
            const parentSlot = slotById.get(target);
            target = parentSlot ? parentSlot.parent : null;
          }
          return owner;
        }

        function endRun(run, failed) {
          runs.filter(r => r.parent === run).forEach(r => endRun(r, false));
          const i = runs.indexOf(run);
          if (i >= 0) runs.splice(i, 1);
          if (run.inst && run.inst !== rootInst && run.inst.status === 'running') {
            run.inst.status = failed ? 'error' : 'done';
            run.inst.end = tick;
          }
        }

        function start(e) {
          if (!e.name || e.name === 'system') return;
          const parent = parentRun(e);
          // A sequential module starts its next agent only after the last one
          // finished, whether or not that one reported its end.
          if (parent && parent.sequential) runs.filter(r => r.parent === parent).forEach(r => endRun(r, false));
          const owner = parent ? (parent.inst || parent.owner) : top;
          const run = {
            name: e.name, instance: e.instance || null, parent, owner, inst: null,
            named: !!e.parent, sequential: isSequential(e.name, e.agentClass),
          };
          runs.push(run);
          if (isComposite(e.name, e.agentClass)) return;
          const slot = findSlot(owner, e.name);
          const isRoot = isRootSlot(slot) || (!skeleton && roots.has(e.name) && owner === top);
          if (isRoot && rootInst) { run.inst = rootInst; return; }   // drawn once
          const inst = { slot, name: e.name, status: isRoot ? 'root' : 'running', children: [], order: order++, start: tick, end: Infinity };
          hostFor(owner, slot).children.push(inst);
          instances.push(inst);
          run.inst = inst;
          if (isRoot) rootInst = inst;
          if (e.spawn) bySpawn.set(e.spawn, inst);
        }

        runEvents().forEach(e => {
          tick++;
          if (e.t === 'start') start(e);
          else if (e.t === 'end') {
            const run = findRun(e.name, e.instance, true);
            if (run) endRun(run, e.failed);
          } else if (e.t === 'error') {
            const inst = (e.callId && bySpawn.get(e.callId))
              || instances.slice().reverse().find(i => i.name === e.target && i !== rootInst);
            if (inst && inst !== rootInst) inst.status = 'error';
          } else if (e.t === 'idle') {
            runs.slice().forEach(r => endRun(r, false));
          }
        });

        // Places with no run yet: a skeleton without the coordinator having
        // run still shows it, at the top.
        const children = new Map();
        [...slots, ...dynamic.values()].forEach(s => {
          const list = children.get(s.parent) || [];
          list.push(s);
          children.set(s.parent, list);
        });
        [...dynamic.values()].forEach(d => slotById.set(d.id, d));
        return { top, children, slotById, calls: instances.filter(i => i !== rootInst).length, latest: instances[instances.length - 1] || null };
      }

      // ── Instances → the drawn tree ───────────────────────────────────────
      // Groups of parallel calls a click unfolded, each with its fold-back timer.
      const expanded = new Set();
      const foldTimers = new Map();

      function visualTree(model) {
        const hidden = name => isInternalAgent(name);
        const flatten = list => list.flatMap(v => (hidden(v.name) ? v.children : [v]));

        // A run that skipped a step (a stage that did not run) hangs above
        // it: the skipped step's placeholder, which holds that run's own
        // place again, is not drawn.
        const within = (slot, ancestor) => {
          for (let s = slot; s; s = model.slotById.get(s.parent)) if (s === ancestor) return true;
          return false;
        };

        // All calls of one agent in one place are one stack until a click
        // unfolds it. How they ran shows on it: side by side if they all
        // overlapped, one after another if none did, mixed otherwise.
        function fromRuns(runs, withPlaceholders) {
          if (runs.length === 1) return [fromInst(runs[0], withPlaceholders)];
          const clusters = [];
          runs.forEach(r => {
            const last = clusters[clusters.length - 1];
            if (last && r.start < last.end) { last.size++; last.end = Math.max(last.end, r.end); }
            else clusters.push({ size: 1, end: r.end });
          });
          const mode = clusters.length === 1 ? 'parallel' : clusters.every(c => c.size === 1) ? 'sequential' : 'mixed';
          const key = `${runs[0].slot.id}:${runs[0].order}`;
          if (expanded.has(key)) {
            return runs.map((r, i) => ({ ...fromInst(r, withPlaceholders && i === runs.length - 1), memberOf: key }));
          }
          const statuses = runs.map(r => r.status);
          return [{
            name: runs[0].name, called: true, kind: runs[0].slot.kind,
            status: statuses.includes('running') ? 'running' : statuses.includes('error') ? 'error' : 'done',
            key: `g${key}`, group: { key, count: runs.length, mode, first: `i${runs[0].order}` }, inst: runs[runs.length - 1],
            children: runs.flatMap((r, i) => kidsOf(r, r.slot.id, withPlaceholders && i === runs.length - 1)),
          }];
        }

        function kidsOf(inst, slotId, withPlaceholders) {
          const out = [];
          const used = new Set();
          (model.children.get(slotId) || []).forEach(slot => {
            const runs = inst ? inst.children.filter(c => c.slot === slot) : [];
            runs.forEach(r => used.add(r));
            const bypassed = inst && inst.children.some(c => c.slot !== slot && within(c.slot, slot));
            if (runs.length && slot.kind === 'critic') out.push(fromCritic(runs));
            else if (runs.length) out.push(...fromRuns(runs, true));
            else if (withPlaceholders && !slot.dynamic && !bypassed) out.push(fromSlot(slot));
          });
          if (inst) {
            const rest = inst.children.filter(c => !used.has(c));
            [...new Set(rest.map(c => c.slot))].forEach(slot => out.push(...fromRuns(rest.filter(c => c.slot === slot), true)));
          }
          return flatten(out);
        }
        // A critic is drawn once beside its agent, however many rounds it had.
        function fromCritic(runs) {
          const last = runs[runs.length - 1];
          return { ...fromInst(last, false), children: [], critic: true, rounds: runs.length };
        }
        // Only the latest call of a place shows where it could still go.
        function fromInst(inst, latest) {
          return { key: `i${inst.order}`, name: inst.name, status: inst.status, called: true, inst, kind: inst.slot.kind, children: kidsOf(inst, inst.slot.id, latest) };
        }
        function fromSlot(slot) {
          return { key: `p${slot.id}`, name: slot.agent, status: 'idle', called: false, kind: slot.kind, children: kidsOf(null, slot.id, true) };
        }
        return kidsOf(model.top, null, true);
      }

      function foldAfter() {
        const seconds = (typeof appSettings !== 'undefined' && Number(appSettings.general.callGraphCollapseSeconds)) || 10;
        return seconds * 1000;
      }

      function unfold(key) {
        expanded.add(key);
        clearTimeout(foldTimers.get(key));
        foldTimers.set(key, setTimeout(() => fold(key), foldAfter()));
        schedule();
      }

      function fold(key) {
        clearTimeout(foldTimers.get(key));
        foldTimers.delete(key);
        expanded.delete(key);
        schedule();
      }

      // ── Feeding ──────────────────────────────────────────────────────────
      function push(entry) {
        log.push(entry);
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

      function setAgentMeta(meta) {
        agentNames = new Set((meta.names || []).map(String));
        titles = meta.titles || {};
        composites = meta.composites || {};
        roots = new Set(['OrchestratorAgent', 'RootOrchestrator', ...(meta.roots || [])]);
        schedule();
      }

      // Loads for the open session; a switch meanwhile drops the answer.
      let generation = 0;
      async function fetchSession(path, userId, sessionId) {
        const mine = generation;
        const resp = await fetch(`/api/users/${encodeURIComponent(userId)}/sessions/${encodeURIComponent(sessionId)}/${path}`);
        if (!resp.ok) return null;
        const data = await resp.json();
        return mine === generation ? data : null;
      }

      async function loadSkeleton(userId, sessionId) {
        if (!userId || !sessionId) return;
        try {
          const data = await fetchSession('call-graph', userId, sessionId);
          if (!data) return;
          skeleton = { slots: data.slots || [], rootSlot: data.rootSlot, composites: new Set(data.composites || []) };
          schedule();
        } catch (_) { /* runs still draw, without placeholders */ }
      }

      // Sessions whose transcript predates agent-run records: their runs
      // come from the execution graph instead, once, after the replay.
      async function loadHistory(userId, sessionId) {
        if (!userId || !sessionId) return;
        const hasRuns = () => log.some(e => e.kind === 'activity' && e.data.phase === 'agent_start');
        if (hasRuns()) return;
        try {
          const data = await fetchSession('agent-runs', userId, sessionId);
          if (!data || hasRuns() || !(data.events || []).length) return;
          data.events.forEach(event => log.push({ kind: 'activity', data: event }));
          log.push({ kind: 'idle' });
          schedule();
        } catch (_) { /* the transcript's own sources remain */ }
      }

      function loadSession(userId, sessionId) {
        loadSkeleton(userId, sessionId);
        loadHistory(userId, sessionId);
      }

      function reset() {
        generation++;
        camera = null;
        shownView = null;
        lastPos = new Map();
        lastEls = new Map();
        [...expanded].forEach(key => clearTimeout(foldTimers.get(key)));
        expanded.clear();
        foldTimers.clear();
        log = [];
        skeleton = null;
        render();
      }

      function schedule() {
        if (renderQueued) return;
        renderQueued = true;
        requestAnimationFrame(() => { renderQueued = false; render(); });
      }

      // ── Drawing ──────────────────────────────────────────────────────────
      // The short Russian name from the call-graph dictionary, else the bare name.
      function displayName(name) {
        return titles[name] || String(name);
      }

      function svgEl(tag, attrs, text) {
        const el = document.createElementNS(SVG_NS, tag);
        Object.entries(attrs || {}).forEach(([k, v]) => el.setAttribute(k, String(v)));
        if (text !== undefined) el.textContent = text;
        return el;
      }

      function fitLabel(text, width) {
        const max = Math.max(3, Math.floor((width - 26) / CHAR_W));
        return text.length > max ? text.slice(0, max - 1) + '…' : text;
      }

      // The framing of the task opens the run and the report closes it: they
      // are lifted out of the tree and set above and below it.
      const SOURCE_AGENTS = new Set(['ContextInitAgent', 'InitAgent', 'TZSpecAgent']);
      const SINK_AGENTS = new Set(['ReportAgent', 'ResultAggregatorAgent', 'NirReportAgent']);
      const ZIGZAG_MIN = 3, ZIG_GAP = 24, CRITIC_GAP = 28;

      // Lays one band — a forest — out as a tidy tree at the origin, made
      // narrower: a run of three or more siblings without branches of their
      // own is set in two staggered rows, every second one a level lower, in
      // the gap between the two above it.
      function layoutBand(roots) {
        // A critic sits beside the agent it reviews, on the left, joined to it
        // by one horizontal line; that agent goes first among its siblings,
        // so the critic is at the edge of the row and crosses nothing.
        const isCritic = v => v.critic || v.kind === 'critic' || /CriticAgent$/.test(v.name);
        const withCritic = list => {
          list.forEach(v => {
            const critics = v.children.filter(isCritic);
            v.criticNode = critics[critics.length - 1] || null;
            if (v.criticNode) { v.criticNode.children = []; v.criticNode.isCriticNode = true; }
            v.children = withCritic(v.children.filter(c => !isCritic(c)));
          });
          return [...list.filter(v => v.criticNode), ...list.filter(v => !v.criticNode)];
        };
        roots = withCritic(roots.filter(v => !isCritic(v)));
        const isLeaf = v => !v.children.length && !v.criticNode;
        const unitsOf = v => {
          const units = [];
          let run = [];
          const flush = () => {
            if (run.length >= ZIGZAG_MIN) units.push({ zigzag: run });
            else run.forEach(c => units.push({ node: c }));
            run = [];
          };
          v.children.forEach(c => { if (isLeaf(c)) run.push(c); else { flush(); units.push({ node: c }); } });
          flush();
          return units;
        };
        const step = NODE_W + ZIG_GAP;
        const unitWidth = u => (u.node ? u.node.width
          : Math.max(Math.ceil(u.zigzag.length / 2) * step - ZIG_GAP, Math.floor(u.zigzag.length / 2) * step + step / 2 - ZIG_GAP));
        const rowWidth = units => units.reduce((sum, u, i) => sum + unitWidth(u) + (i ? H_GAP : 0), 0);
        // Room a critic needs left of its agent beyond the free space the
        // agent already has there, centred over a wider row of children.
        const extra = v => {
          if (!v.criticNode) return 0;
          const own = Math.max(NODE_W, rowWidth(v.units));
          return Math.max(0, NODE_W + CRITIC_GAP - (own - NODE_W) / 2);
        };
        const measure = v => {
          v.children.forEach(measure);
          v.units = unitsOf(v);
          v.width = extra(v) + Math.max(NODE_W, rowWidth(v.units));
        };
        const rowY = depth => depth * (NODE_H + V_GAP);
        const nodes = [];
        const edges = [];
        let maxDepth = 0;
        const place = (v, left, depth) => {
          left += extra(v);
          const own = v.width - extra(v);
          v.x = left + own / 2 - NODE_W / 2;
          v.y = rowY(depth);
          if (v.criticNode) {
            const c = v.criticNode;
            c.x = v.x - NODE_W - CRITIC_GAP;
            c.y = v.y;
            nodes.push(c);
            edges.push({ from: c, to: v, side: true });
          }
          nodes.push(v);
          maxDepth = Math.max(maxDepth, depth);
          let x = left + (own - rowWidth(v.units)) / 2;
          v.units.forEach(u => {
            if (u.node) {
              place(u.node, x, depth + 1);
              edges.push({ from: v, to: u.node });
            } else {
              u.zigzag.forEach((c, i) => {
                const lower = i % 2 === 1;
                c.width = NODE_W;
                c.units = [];
                place(c, x + Math.floor(i / 2) * step + (lower ? step / 2 : 0), depth + (lower ? 2 : 1));
                edges.push({ from: v, to: c, through: lower ? rowY(depth + 1) : null });
              });
            }
            x += unitWidth(u) + H_GAP;
          });
        };
        let x = 0;
        roots.forEach(v => { measure(v); place(v, x, 0); x += v.width + H_GAP; });
        return { roots, nodes, edges, width: x - H_GAP, height: rowY(maxDepth) + NODE_H };
      }

      // Bands top to bottom: each framing agent, the tree, the reports. Every
      // leaf of a band is joined to every root of the band below it.
      function layout(forest) {
        const sources = [];
        const sinks = [];
        const lift = (list, inSink) => list.flatMap(v => {
          if (!inSink && SOURCE_AGENTS.has(v.name)) {
            // Its own calls stay with it; the step after it takes its place.
            sources.push(v);
            const next = v.children.filter(c => c.kind === 'step');
            v.children = lift(v.children.filter(c => c.kind !== 'step'), inSink);
            return lift(next, inSink);
          }
          if (!inSink && SINK_AGENTS.has(v.name)) {
            sinks.push(v);
            v.children = lift(v.children, true);
            return [];
          }
          v.children = lift(v.children, inSink);
          return [v];
        });
        const main = lift(forest, false);
        const bands = [...sources.map(v => [v]), main, sinks].filter(b => b.length).map(layoutBand);
        const width = Math.max(...bands.map(b => b.width));
        const all = [];
        const edges = [];
        let y = PAD;
        bands.forEach((band, i) => {
          const dx = PAD + (width - band.width) / 2;
          band.nodes.forEach(v => { v.x += dx; v.y += y; });
          band.edges.forEach(e => { if (e.through != null) e.through += y; });
          all.push(...band.nodes);
          edges.push(...band.edges);
          const prev = bands[i - 1];
          if (prev) {
            prev.nodes.filter(v => !v.children.length && !v.isCriticNode).forEach(leaf =>
              band.roots.forEach(root => edges.push({ from: leaf, to: root, link: true })));
          }
          y += band.height + V_GAP;
        });
        return { all, edges, totalW: width + PAD * 2, totalH: y - V_GAP + PAD };
      }

      function edgePath(e) {
        const x1 = e.from.x + NODE_W / 2, y1 = e.from.y + NODE_H;
        const x2 = e.to.x + NODE_W / 2, y2 = e.to.y;
        if (e.side) {
          // Critic → its agent: straight across.
          const y = e.from.y + NODE_H / 2;
          return `M${e.from.x + NODE_W},${y} L${e.to.x},${y}`;
        }
        if (e.link) {
          // Straight down the leaf's own column, then into the band below.
          const yb = Math.max(y1, y2 - V_GAP), my = (yb + y2) / 2;
          return `M${x1},${y1} L${x1},${yb} C${x1},${my} ${x2},${my} ${x2},${y2}`;
        }
        if (e.through != null) {
          // To the lower staggered row: down through the gap in the upper one.
          const yu = e.through - 4, my = (y1 + yu) / 2;
          return `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${yu} L${x2},${y2}`;
        }
        const my = (y1 + y2) / 2;
        return `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
      }

      function render() {
        const scroller = document.getElementById('call-graph-scroll');
        const svg = document.getElementById('call-graph-svg');
        const empty = document.getElementById('call-graph-empty');
        const count = document.getElementById('call-graph-count');
        if (!scroller || !svg) return;
        const model = buildModel();
        const forest = visualTree(model);
        if (count) count.textContent = model.calls ? String(model.calls) : '';
        empty.classList.toggle('hidden', forest.length > 0);
        svg.classList.toggle('hidden', forest.length === 0);
        const oldEls = lastEls;
        lastEls = new Map();
        svg.replaceChildren();
        if (!forest.length) return;

        const { all, edges: links, totalW, totalH } = layout(forest);
        const world = svgEl('g', { class: 'cg-world' });
        svg.appendChild(world);
        // Start from where the last drawing left the view; it glides on below.
        if (shownView) world.style.transform = viewTransform(shownView);

        // Every node moves from where it was drawn last: the calls of a stack
        // fan out of it on a click and gather back into it on folding, and
        // whatever the change pushed aside slides over. New nodes fade in.
        const before = lastPos;
        const animate = before.size > 0 && !REDUCED_MOTION.matches;
        const moves = [];
        lastPos = new Map();

        const edges = svgEl('g', { class: 'cg-edges' });
        links.forEach(e => edges.appendChild(svgEl('path', {
          d: edgePath(e),
          class: `cg-edge${e.to.called && e.from.called ? '' : ' cg-edge-idle'}${e.to.status === 'running' ? ' cg-edge-live' : ''}`,
        })));
        world.appendChild(edges);

        let focus = null;
        all.forEach(v => {
          const g = svgEl('g', {
            class: `cg-node cg-${v.status}${v.group ? ' cg-group' : ''}${v.memberOf ? ' cg-member' : ''}`,
          });
          const at = `translate(${v.x}px, ${v.y}px)`;
          const from = animate && (before.get(v.key)
            || (v.memberOf && before.get(`g${v.memberOf}`))
            || (v.group && before.get(v.group.first)));
          if (from && (from.x !== v.x || from.y !== v.y)) {
            g.style.transform = `translate(${from.x}px, ${from.y}px)`;
            moves.push([g, at]);
          } else {
            g.style.transform = at;
            if (animate && !from) { g.style.opacity = '0'; moves.push([g, at]); }
          }
          lastPos.set(v.key, { x: v.x, y: v.y });
          lastEls.set(v.key, { el: g, memberOf: v.memberOf });
          if (v.group) g.setAttribute('data-group', v.group.key);
          if (v.memberOf) g.setAttribute('data-member', v.memberOf);
          const title = displayName(v.name);
          const label = title;
          const how = v.group && { parallel: 'параллельно', sequential: 'по очереди', mixed: 'частью параллельно' }[v.group.mode];
          const state = !v.called ? ' — не вызывался'
            : v.group ? ` — ${v.group.count} вызовов, ${how}; щелчок раскрывает`
              : v.memberOf ? ' — щелчок сворачивает' : '';
          g.appendChild(svgEl('title', {}, (title === v.name ? v.name : `${title} (${v.name})`) + state));
          // A stack: cards peeking out from behind — to the side for calls
          // that ran together, below for calls that ran one after another.
          if (v.group) {
            const side = v.group.mode === 'parallel';
            [2, 1].forEach(k => g.appendChild(svgEl('rect', {
              x: side ? 3 * k : 0, y: side ? 0 : 4 * k, width: NODE_W, height: NODE_H, rx: 6, class: 'cg-box cg-stack',
            })));
          }
          g.appendChild(svgEl('rect', { width: NODE_W, height: NODE_H, rx: 6, class: 'cg-box' }));
          // Dot and title centred together in the box; a stack keeps room on
          // the right for its count.
          // Only a state worth telling gets a dot: running, failed, not run
          // yet, the coordinator. A finished call is plain.
          const dotted = v.status !== 'done';
          const gap = dotted ? 8 : 0;
          const count = v.group ? v.group.count : v.rounds > 1 ? v.rounds : 0;
          const badge = count ? 22 : 0;
          const text = svgEl('text', {
            x: (NODE_W - badge) / 2 + gap / 2, y: NODE_H / 2 + 4, 'font-size': FONT,
            'text-anchor': 'middle', class: 'cg-label',
          }, fitLabel(label, NODE_W - badge));
          if (count) {
            g.appendChild(svgEl('text', {
              x: NODE_W - 7, y: NODE_H / 2 + 4, 'font-size': FONT, 'text-anchor': 'end', class: 'cg-count',
            }, v.group && v.group.mode !== 'parallel' ? `↓${count}` : `×${count}`));
          }
          g.appendChild(text);
          world.appendChild(g);
          const tw = text.getComputedTextLength() || text.textContent.length * CHAR_W;
          if (dotted) {
            g.insertBefore(svgEl('circle', {
              cx: (NODE_W - badge) / 2 + gap / 2 - tw / 2 - gap, cy: NODE_H / 2, r: 3, class: 'cg-dot',
            }), text);
          }
          if (v.status === 'running' && (!focus || v.inst.order > focus.inst.order)) focus = v;
        });

        // Unless the reader has moved the view: while agents run, the latest
        // of them is centred at a readable size; otherwise the whole graph is
        // fitted into the middle of the panel as an overview.
        const W = scroller.clientWidth, H = scroller.clientHeight;
        // Full screen has the room to draw it bigger than life size.
        const grow = document.fullscreenElement === document.getElementById('side-nav-panel') ? FULL_MAX : 1;
        const whole = Math.min(grow, W / totalW, H / totalH);
        const fit = focus ? Math.max(FIT_MIN, whole) : Math.max(OVERVIEW_MIN, whole);
        const w = totalW * fit, h = totalH * fit;
        const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));
        autoView = {
          tx: w <= W || !focus ? (W - w) / 2 : clamp(W / 2 - (focus.x + NODE_W / 2) * fit, W - w, 0),
          ty: h <= H || !focus ? Math.max(0, (H - h) / 2) : clamp(H / 2 - (focus.y + NODE_H / 2) * fit, H - h, 0),
          s: fit,
        };
        // Calls of a stack that just folded: their old cards glide into it
        // and fade, drawn under it.
        const ghosts = [];
        if (animate) {
          const firstNode = world.querySelector('.cg-node');
          oldEls.forEach(({ el, memberOf }, key) => {
            const into = memberOf && !lastPos.has(key) && lastPos.get(`g${memberOf}`);
            if (!into) return;
            el.style.opacity = '1';
            world.insertBefore(el, firstNode);
            ghosts.push([el, `translate(${into.x}px, ${into.y}px)`]);
          });
        }
        if (!moves.length && !ghosts.length) { applyCamera(); return; }
        // Transitions on, the old places committed, then the new ones set.
        world.classList.add('cg-anim');
        edges.style.opacity = '0';
        svg.getBoundingClientRect();
        requestAnimationFrame(() => {
          moves.forEach(([g, at]) => { g.style.transform = at; g.style.opacity = ''; });
          ghosts.forEach(([el, at]) => { el.style.transform = at; el.style.opacity = '0'; });
          setTimeout(() => ghosts.forEach(([el]) => el.remove()), ANIM_MS + 50);
          edges.style.opacity = '';
          applyCamera();
          clearTimeout(animEnd);
          animEnd = setTimeout(() => world.classList.remove('cg-anim'), ANIM_MS + 250);
        });
      }

      // ── View: drag to pan, wheel to zoom, double-click to fit ────────────
      let camera = null;      // null: follow autoView
      let autoView = { tx: 0, ty: 0, s: 1 };
      let shownView = null;   // the view last put on screen
      let lastPos = new Map(); // node key -> where it was drawn last
      let lastEls = new Map(); // node key -> its element, for folding calls away
      let animEnd = null;
      const ANIM_MS = 380;
      const REDUCED_MOTION = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: true };
      const ZOOM_MIN = 0.1, ZOOM_MAX = 4, FIT_MIN = 0.85, OVERVIEW_MIN = 0.12, FULL_MAX = 2;

      const viewTransform = v => `translate(${v.tx}px, ${v.ty}px) scale(${v.s})`;

      function applyCamera() {
        const world = document.querySelector('#call-graph-svg .cg-world');
        const v = camera || autoView;
        shownView = { ...v };
        if (world) world.style.transform = viewTransform(v);
      }

      function stopAnimation() {
        const world = document.querySelector('#call-graph-svg .cg-world');
        if (world) world.classList.remove('cg-anim');
      }

      function initView(scroller) {
        let drag = null;
        let dragged = false;
        scroller.addEventListener('pointerdown', event => {
          if (event.button !== 0) return;
          drag = { x: event.clientX, y: event.clientY, id: event.pointerId, moving: false };
          stopAnimation();
          dragged = false;
          scroller.setPointerCapture(event.pointerId);
        });
        scroller.addEventListener('pointermove', event => {
          if (!drag || event.pointerId !== drag.id) return;
          const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
          // A click that wobbles a pixel is still a click.
          if (!drag.moving) {
            if (Math.hypot(dx, dy) < 4) return;
            camera = camera || { ...autoView };
            drag.moving = true;
            drag.tx = camera.tx;
            drag.ty = camera.ty;
            scroller.classList.add('dragging');
          }
          camera.tx = drag.tx + dx;
          camera.ty = drag.ty + dy;
          applyCamera();
        });
        const stop = event => {
          if (!drag || event.pointerId !== drag.id) return;
          dragged = drag.moving;
          drag = null;
          scroller.classList.remove('dragging');
        };
        scroller.addEventListener('pointerup', stop);
        scroller.addEventListener('pointercancel', stop);
        // A stack of parallel calls unfolds on a click and folds back on its
        // own; a click on one of its unfolded calls folds it at once. The
        // second click of a double-click is the fit gesture, not a toggle.
        scroller.addEventListener('click', event => {
          if (dragged || event.detail > 1) return;
          const hit = document.elementFromPoint(event.clientX, event.clientY);
          const node = hit && hit.closest && hit.closest('.cg-node');
          if (!node) return;
          const group = node.getAttribute('data-group');
          const member = node.getAttribute('data-member');
          if (group) unfold(group);
          else if (member) fold(member);
        });
        scroller.addEventListener('wheel', event => {
          event.preventDefault();
          stopAnimation();
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

      // ── Full screen: the panel alone fills the screen; Esc or the same
      // button brings it back. The resize observer refits the graph.
      function toggleFullscreen() {
        const panel = document.getElementById('side-nav-panel');
        if (!panel) return;
        if (document.fullscreenElement === panel) document.exitFullscreen();
        else if (panel.requestFullscreen) panel.requestFullscreen().catch(() => {});
      }

      function syncFullscreen() {
        const panel = document.getElementById('side-nav-panel');
        const full = !!panel && document.fullscreenElement === panel;
        const icon = document.getElementById('call-graph-full-icon');
        const button = document.getElementById('call-graph-full');
        if (icon) icon.textContent = full ? 'close_fullscreen' : 'open_in_full';
        if (button) {
          const label = t(full ? 'nav.callGraphExitFull' : 'nav.callGraphFull');
          button.title = label;
          button.setAttribute('aria-label', label);
        }
        // A new size is a new view: fit it, unless the reader moved it.
        schedule();
      }

      function init() {
        const scroller = document.getElementById('call-graph-scroll');
        if (!scroller) return;
        document.addEventListener('fullscreenchange', syncFullscreen);
        const button = document.getElementById('call-graph-full');
        if (button && !document.fullscreenEnabled) button.classList.add('hidden');
        initView(scroller);
        if (window.ResizeObserver) new ResizeObserver(() => schedule()).observe(scroller);
        render();
      }

      return { init, reset, feed, feedAgentEvent, markIdle, setAgentMeta, loadSession, loadSkeleton, toggleFullscreen };
    })();

    document.addEventListener('DOMContentLoaded', () => CallGraph.init());
