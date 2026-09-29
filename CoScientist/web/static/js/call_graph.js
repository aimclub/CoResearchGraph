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
      // Card size of the view being laid out: agent cards inside a block,
      // block cards on the map (set by each renderer before layout).
      let NODE_W = 132, NODE_H = 62;
      const H_GAP = 12, V_GAP = 30, PAD = 16;
      const FONT = 10.5, CHAR_W = 6.1;
      const SVG_NS = 'http://www.w3.org/2000/svg';
      const COMPOSITE_CLASSES = new Set(['SequentialAgent', 'ParallelAgent', 'LoopAgent']);
      const PREP_AGENTS = new Set(['ContextInitAgent', 'InitAgent', 'TZSpecAgent']);

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
            // Not in the config: it belongs to the block of the agent that ran it.
            // Agents framing the task belong to the preparation wherever they ran.
            const ownerSlot = ownerId && slotById.get(ownerId);
            const prep = skeleton && (skeleton.blocks || []).find(b => b.kind === 'prep');
            // One the coordinator ran is a branch of its own on the map.
            const branch = skeleton && ownerId && ownerId === skeleton.rootSlot;
            dynamic.set(key, {
              id: `d${dynamic.size}`, agent: name, parent: ownerId, owner: ownerId, kind: 'call', dynamic: true,
              block: PREP_AGENTS.has(name) && prep ? prep.id : branch ? `dyn:${name}` : ownerSlot ? ownerSlot.block : null,
            });
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
        // A run with a run still going below it waits; one without works.
        const below = inst => {
          inst.runningBelow = false;
          inst.children.forEach(c => { if (below(c) || c.status === 'running') inst.runningBelow = true; });
          return inst.runningBelow;
        };
        below(top);
        return {
          top, children, slotById, instances, rootInst, active: runs.length > 0,
          calls: instances.filter(i => i !== rootInst).length, latest: instances[instances.length - 1] || null,
        };
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
            key: `g${key}`, group: { key, count: runs.length, mode, first: `i${runs[0].order}` }, inst: runs[runs.length - 1], members: runs,
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
          return { key: `p${slot.id}`, name: slot.agent, status: 'idle', called: false, kind: slot.kind, slotBlock: slot.block || null, children: kidsOf(null, slot.id, true) };
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
          skeleton = {
            slots: data.slots || [], rootSlot: data.rootSlot, blocks: data.blocks || [], blockTitles: data.blockTitles || {},
            composites: new Set(data.composites || []),
          };
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

      // FEDOT.MAS runs: each builds a system of agents of its own, named by
      // FEDOT, recorded in the session's own FEDOT journal. Read on opening
      // the session and polled while a run is going.
      let fedotRuns = [];
      let fedotScope = null;
      let fedotTimer = null;
      let fedotLoading = false;
      async function loadFedot(userId, sessionId) {
        if (!userId || !sessionId) return;
        fedotScope = [userId, sessionId];
        clearTimeout(fedotTimer);
        fedotTimer = null;
        fedotLoading = true;
        try {
          const list = await fetchSession('fedot/runs', userId, sessionId);
          if (!list) return;
          const metas = (list.runs || []).slice().sort((a, b) => a.started_at - b.started_at);
          const known = new Map(fedotRuns.map(r => [r.meta.run_id, r]));
          const runs = [];
          for (const meta of metas) {
            const old = known.get(meta.run_id);
            if (old && old.meta.status !== 'running' && meta.status === old.meta.status) { runs.push(old); continue; }
            const data = await fetchSession(`fedot/runs/${meta.run_id}`, userId, sessionId);
            if (!data) return;
            runs.push(parseFedotRun(data.run || meta, data.events || []));
          }
          fedotRuns = runs;
          schedule();
        } catch (_) { /* the map still draws, without systems */ }
        finally { fedotLoading = false; }
        if (fedotRuns.some(r => r.running) || fedotWanted) {
          fedotTimer = setTimeout(() => loadFedot(...fedotScope), 2500);
        }
      }
      // A FEDOT-building agent is running: its system may appear any moment.
      let fedotWanted = false;

      function parseFedotRun(meta, events) {
        const agents = new Map();
        const touch = name => {
          if (!agents.has(name)) agents.set(name, { name, starts: 0, done: 0, failed: false });
          return agents.get(name);
        };
        let task = meta.task || '';
        let ended = meta.status !== 'running';
        let failed = meta.status === 'error' || meta.status === 'timeout';
        events.forEach(e => {
          if (e.type === 'run_start' && e.task) task = e.task;
          if (e.type === 'config' && e.config) {
            const c = e.config;
            const listed = c.agents || [c.coordinator, ...(c.workers || [])];
            listed.filter(Boolean).forEach(a => touch(String(a.name || a)));
          }
          if (e.type === 'agent_start' && e.agent) touch(e.agent).starts++;
          if (e.type === 'agent_done' && e.agent) touch(e.agent).done++;
          if (e.type === 'model_error' && e.agent) touch(e.agent).failed = true;
          if (e.type === 'run_end') { ended = true; failed = failed || (e.status && e.status !== 'ok' && e.status !== 'success'); }
        });
        const list = [...agents.values()].map(a => ({
          ...a,
          status: a.starts === 0 ? 'idle'
            : a.starts > a.done && !ended ? 'work'
              : a.failed || (failed && a.starts > a.done) ? 'error' : 'done',
        }));
        return { meta, task, agents: list, running: !ended, failed };
      }

      function loadSession(userId, sessionId) {
        loadSkeleton(userId, sessionId);
        loadHistory(userId, sessionId);
        loadFedot(userId, sessionId);
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
        fedotRuns = [];
        clearTimeout(fedotTimer);
        path = [];
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

      // ── Views ─────────────────────────────────────────────────────────────
      // The map of blocks, or — a click away — the inside of one block, or
      // of one FEDOT.MAS system. `path` is the way down, for the breadcrumbs.
      let path = [];

      const STATUS_TEXT = { idle: 'Не запущен', done: 'Завершён', wait: 'Ожидает', work: 'В работе', error: 'Ошибка' };
      const GROUP_TEXT = { idle: 'Не запущен', done: 'Готово', wait: 'Ожидает', work: 'В работе', error: 'Ошибка' };
      const ICONS = { idle: '–', done: '✓', wait: '◷', work: '●', error: '!' };

      function plural(n, one, few, many) {
        const m10 = n % 10, m100 = n % 100;
        if (m10 === 1 && m100 !== 11) return one;
        if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
        return many;
      }

      // A call's state on a card: a call still running with another running
      // below it waits for it; one without works itself.
      function instState(inst, model) {
        if (!inst) return 'idle';
        if (inst === model.rootInst) return inst.runningBelow ? 'wait' : model.active ? 'work' : 'done';
        if (inst.status === 'running') {
          if (inst.runningBelow) return 'wait';
          if (FEDOT_AGENTS.test(inst.name) && fedotRuns.some(r => r.running)) return 'wait';
          return 'work';
        }
        return inst.status === 'error' ? 'error' : 'done';
      }

      function mergeStates(states) {
        if (!states.length) return 'idle';
        if (states.includes('work')) return 'work';
        if (states.includes('wait')) return 'wait';
        if (states.includes('error')) return 'error';
        return states.every(s => s === 'idle') ? 'idle' : 'done';
      }

      const FEDOT_AGENTS = /^FedotAgent$/;

      function blockOf(inst) {
        return (inst && inst.slot && inst.slot.block) || '__other__';
      }

      // The blocks of the map with what ran in them.
      function mapBlocks(model) {
        const blocks = (skeleton && skeleton.blocks || []).map(b => ({ ...b }));
        // Branches the config does not have, found in what ran.
        [...new Set(model.instances.map(blockOf).filter(id => id.startsWith('dyn:')))].forEach(id => {
          const name = id.slice(4);
          const title = (skeleton.blockTitles || {})[name]
            || displayName(name).replace(/^Агент\s+/, '').replace(/^./, ch => ch.toUpperCase());
          blocks.push({ id, kind: 'branch', title, parent: null, head: null, agent: name });
        });
        if (model.instances.some(i => blockOf(i) === '__other__')) {
          blocks.push({ id: '__other__', kind: 'other', title: 'Другие агенты', parent: null });
        }
        blocks.forEach(b => {
          b.insts = model.instances.filter(i => blockOf(i) === b.id);
          b.state = mergeStates(b.insts.map(i => instState(i, model)));
          const head = b.insts.filter(i => i.slot && (i.slot.id === b.head || (!b.head && i.name === b.agent)));
          b.count = head.length || (b.insts.length ? 1 : 0);
          // The agents of a group: every one it can run (critic included)
          // and every one that did run in it — what opening it shows.
          const agents = new Set(b.insts.map(i => i.name));
          model.slotById.forEach(slot => { if (slot.block === b.id) agents.add(slot.agent); });
          if (b.kind === 'prep') b.size = agents.size;
          b.agents = agents.size;
          b.group = !!b.size && (b.kind === 'prep' || !!b.module);
        });
        return blocks;
      }

      function navigate(to) {
        path = to;
        camera = null;
        lastPos = new Map();
        lastEls = new Map();
        entering = true;
        schedule();
      }
      let entering = false;

      function render() {
        const scroller = document.getElementById('call-graph-scroll');
        const svg = document.getElementById('call-graph-svg');
        const empty = document.getElementById('call-graph-empty');
        if (!scroller || !svg) return;
        const model = buildModel();
        fedotWanted = model.instances.some(i => FEDOT_AGENTS.test(i.name) && i.status === 'running');
        if (fedotWanted && fedotScope && !fedotTimer && !fedotLoading) loadFedot(...fedotScope);
        const blocks = mapBlocks(model);
        const useMap = !!(skeleton && skeleton.blocks && skeleton.blocks.length);
        // A view whose block is gone (another session's path) falls back.
        const at = path[path.length - 1];
        if (at && at.type === 'block' && !blocks.some(b => b.id === at.id)) path = [];
        if (at && at.type === 'mas' && !fedotRuns.some(r => r.meta.run_id === at.id)) path = [];

        let scene;
        const view = path[path.length - 1];
        if (!view && useMap) scene = mapScene(model, blocks);
        else if (view && view.type === 'mas') scene = masScene(view.id, model, blocks);
        else scene = treeScene(model, view ? view.id : null, blocks);

        renderChrome(model, blocks, scene);
        empty.classList.toggle('hidden', scene.cards.length > 0);
        svg.classList.toggle('hidden', scene.cards.length === 0);
        const oldEls = lastEls;
        lastEls = new Map();
        svg.replaceChildren();
        if (!scene.cards.length) return;
        drawScene(svg, scroller, scene, oldEls);
      }

      // ── The map: blocks in the design of the research map ────────────────
      const BLOCK_W = 116, BLOCK_H = 104, BLOCK_GAP = 28, FRAME_PAD = 20;

      function mapScene(model, blocks) {
        NODE_W = BLOCK_W;
        NODE_H = BLOCK_H;
        const cards = [];
        const links = [];
        const frames = [];
        const card = (b, x, y) => {
          const c = {
            key: `B${b.id}`, x, y, w: BLOCK_W, h: BLOCK_H, title: b.title, state: b.state, count: b.count,
            label: b.group ? 'ГРУППА' : null,
            subtitle: b.group
              ? `${b.size} ${b.kind === 'prep' ? plural(b.size, 'агент', 'агента', 'агентов') : plural(b.size, 'этап', 'этапа', 'этапов')} · ${GROUP_TEXT[b.state]}`
              : STATUS_TEXT[b.state],
            tip: `${b.title}${b.group ? ` — группа: ${b.kind === 'prep' ? `${b.size} ${plural(b.size, 'агент', 'агента', 'агентов')}` : `${b.size} ${plural(b.size, 'этап', 'этапа', 'этапов')}, ${b.agents} ${plural(b.agents, 'агент', 'агента', 'агентов')}`}` : ''}: ${STATUS_TEXT[b.state].toLowerCase()}, запусков ${b.count}. Щелчок — открыть.`,
            attrs: { 'data-block': b.id }, big: true,
          };
          cards.push(c);
          return c;
        };
        const prep = blocks.find(b => b.kind === 'prep');
        const root = blocks.find(b => b.kind === 'root');
        const row = [
          ...blocks.filter(b => b.kind === 'branch' && !b.parent),
          ...blocks.filter(b => b.kind === 'other'),
          ...blocks.filter(b => b.kind === 'post'),
        ];
        const nestedOf = id => blocks.filter(b => b.parent === id);

        // The branches, in a frame: a column each, nested blocks below.
        const frameTop = PAD + BLOCK_H + 44;
        let x = PAD + FRAME_PAD;
        let bottom = frameTop + FRAME_PAD + BLOCK_H;
        const placed = new Map();
        row.forEach(b => {
          const c = card(b, x, frameTop + FRAME_PAD);
          placed.set(b.id, c);
          let y = c.y;
          let parent = c;
          const walk = id => nestedOf(id).forEach(n => {
            y += BLOCK_H + 34;
            const nc = card(n, x, y);
            placed.set(n.id, nc);
            links.push({ from: parent, to: nc, kind: 'down' });
            parent = nc;
            walk(n.id);
          });
          walk(b.id);
          bottom = Math.max(bottom, y + BLOCK_H);
          x += BLOCK_W + BLOCK_GAP;
        });
        const frameW = Math.max(BLOCK_W, x - BLOCK_GAP - PAD - FRAME_PAD) + FRAME_PAD * 2;
        if (row.length) frames.push({ x: PAD, y: frameTop, w: frameW, h: bottom + FRAME_PAD - frameTop });

        // The coordinator above the middle of the frame, the preparation
        // to its left.
        let rootCard = null;
        if (root) {
          const rx = PAD + frameW / 2 - BLOCK_W / 2;
          rootCard = card(root, rx, PAD);
          placed.set(root.id, rootCard);
          if (prep) {
            const pc = card(prep, rx - BLOCK_W - 56, PAD);
            placed.set(prep.id, pc);
            links.push({ from: pc, to: rootCard, kind: 'side' });
          }
          row.forEach(b => links.push({ from: rootCard, to: placed.get(b.id), kind: 'bus', busY: frameTop - 14 }));
        } else if (prep) {
          placed.set(prep.id, card(prep, PAD, PAD));
        }

        // FEDOT.MAS systems: a frame each below, its agents in a row, fed by
        // the block that builds them.
        const topCards = cards.length, topFrames = frames.length;
        const builder = blocks.find(b => b.kind === 'mas_builder' && placed.has(b.id))
          || (root && placed.has(root.id) ? root : null);
        let masY = bottom + FRAME_PAD + 56;
        let masX = PAD;
        const masW = BLOCK_W, masGap = 36, masHead = 44;
        // Side by side, so no line to one system crosses another; a long
        // queue of systems wraps.
        const rowLimit = Math.max(frameW, 3 * (BLOCK_W * 3 + masGap * 2 + FRAME_PAD * 2 + 30));
        let rowH = 0;
        fedotRuns.forEach((run, i) => {
          const n = Math.max(1, run.agents.length);
          const w = FRAME_PAD * 2 + n * masW + (n - 1) * masGap;
          const h = masHead + BLOCK_H + FRAME_PAD;
          if (masX > PAD && masX + w > PAD + rowLimit) { masX = PAD; masY += rowH + 30; rowH = 0; }
          const label = `FEDOT.MAS / ${String(i + 1).padStart(2, '0')}`;
          const task = (run.task || '').replace(/\s+/g, ' ').trim();
          const newAgents = `${run.agents.length} ${plural(run.agents.length, 'новый агент', 'новых агента', 'новых агентов')}`;
          frames.push({
            x: masX, y: masY, w, h, label, subtitle: `${task.length > 48 ? task.slice(0, 47) + '…' : task}${task ? ' · ' : ''}${newAgents}`,
            mas: run.meta.run_id, state: run.running ? 'work' : run.failed ? 'error' : 'done',
          });
          let prev = null;
          run.agents.forEach((a, j) => {
            const c = {
              key: `M${run.meta.run_id}:${a.name}`, x: masX + FRAME_PAD + j * (masW + masGap), y: masY + masHead,
              w: masW, h: BLOCK_H, title: humanize(a.name), state: a.status, count: a.starts,
              subtitle: STATUS_TEXT[a.status], tip: `${a.name} — агент FEDOT.MAS: ${STATUS_TEXT[a.status].toLowerCase()}, запусков ${a.starts}.`,
              attrs: { 'data-mas': run.meta.run_id }, big: true,
            };
            cards.push(c);
            if (prev) links.push({ from: prev, to: c, kind: 'side' });
            prev = c;
          });
          const first = cards.find(c => c.key === `M${run.meta.run_id}:${(run.agents[0] || {}).name}`);
          const from = builder && placed.get(builder.id);
          if (from && first) links.push({ from, to: first, kind: 'mas', busY: bottom + FRAME_PAD + 22, entryX: masX - 14 });
          masX += w + 30;
          rowH = Math.max(rowH, h);
        });

        // Systems wider than the frame: centre the top of the map over them.
        const masRight = Math.max(0, ...frames.slice(topFrames).map(f => f.x + f.w));
        const topRight = PAD + frameW;
        if (masRight > topRight) {
          const dx = (masRight - topRight) / 2;
          cards.slice(0, topCards).forEach(c => { c.x += dx; });
          frames.slice(0, topFrames).forEach(f => { f.x += dx; });
        }
        // Shift everything right if the preparation sticks out on the left.
        const minX = Math.min(...cards.map(c => c.x), ...frames.map(f => f.x));
        if (minX < PAD) {
          const dx = PAD - minX;
          cards.forEach(c => { c.x += dx; });
          frames.forEach(f => { f.x += dx; });
        }
        const maxX = Math.max(...cards.map(c => c.x + c.w), ...frames.map(f => f.x + f.w));
        const maxY = Math.max(...cards.map(c => c.y + c.h), ...frames.map(f => f.y + f.h));
        const notes = [];
        const groups = blocks.filter(b => b.group);
        if (groups.length) {
          notes.push(groups.map(b => `${b.title} ${b.kind === 'prep' ? 'объединяет' : 'включает'} ${b.size} ${b.kind === 'prep' ? plural(b.size, 'агента', 'агентов', 'агентов') : plural(b.size, 'этап', 'этапа', 'этапов')}`).join(' · '));
        }
        return { cards, links, frames, totalW: maxX + PAD, totalH: maxY + PAD, notes, map: true };
      }

      function humanize(name) {
        const s = String(name).replace(/[_-]+/g, ' ').replace(/([a-zа-я])([A-ZА-Я])/g, '$1 $2').trim();
        return s.charAt(0).toUpperCase() + s.slice(1);
      }

      // ── Inside a block: its agents, as the detailed tree ─────────────────
      function treeScene(model, blockId, blocks) {
        NODE_W = 132;
        NODE_H = 62;
        let forest = visualTree(model);
        const titleOf = id => (blocks.find(b => b.id === id) || {}).title || 'Блок';
        const stateOf = id => (blocks.find(b => b.id === id) || {}).state || 'idle';
        if (blockId) {
          // The block's own agents; a call into another block is a link to it.
          const vblock = v => (v.inst ? blockOf(v.inst) : v.slotBlock || '__other__');
          const roots = [];
          const find = (list, parentBlock) => list.forEach(v => {
            const b = vblock(v);
            if (b === blockId && parentBlock !== blockId) roots.push(v);
            find(v.children, b);
          });
          find(forest, null);
          const prune = v => {
            const links = new Map();
            v.children = v.children.flatMap(c => {
              const b = vblock(c);
              if (b === blockId) return [prune(c)];
              if (!links.has(b)) {
                links.set(b, { key: `L${b}`, name: titleOf(b), linkTo: b, status: 'link', linkState: stateOf(b), called: stateOf(b) !== 'idle', children: [] });
                return [links.get(b)];
              }
              return [];
            });
            return v;
          };
          forest = roots.map(prune);
        }
        const { all, edges, totalW, totalH } = layout(forest);
        const cards = all.map(v => agentCard(v, model));
        const byNode = new Map(all.map((v, i) => [v, cards[i]]));
        const links = edges.map(e => ({ ...e, from: byNode.get(e.from), to: byNode.get(e.to), tree: e }));
        return { cards, links, frames: [], totalW, totalH, notes: [], map: false };
      }

      function agentCard(v, model) {
        if (v.linkTo) {
          return {
            key: v.key, x: v.x, y: v.y, w: NODE_W, h: NODE_H, title: v.name, state: v.linkState, link: true,
            subtitle: `Блок · ${STATUS_TEXT[v.linkState]}`, tip: `Блок «${v.name}» — щелчок открывает его.`,
            attrs: { 'data-link': v.linkTo }, count: null,
          };
        }
        const state = v.group
          ? mergeStates(v.members.map(i => instState(i, model)))
          : v.called ? instState(v.inst, model) : 'idle';
        const title = displayName(v.name);
        const how = v.group && { parallel: 'параллельно', sequential: 'по очереди', mixed: 'частью параллельно' }[v.group.mode];
        const tip = (title === v.name ? v.name : `${title} (${v.name})`)
          + (!v.called ? ' — не вызывался'
            : v.group ? ` — ${v.group.count} вызовов, ${how}; щелчок раскрывает`
              : v.memberOf ? ' — щелчок сворачивает' : `: ${STATUS_TEXT[state].toLowerCase()}`);
        const attrs = {};
        if (v.group) attrs['data-group'] = v.group.key;
        if (v.memberOf) attrs['data-member'] = v.memberOf;
        return {
          key: v.key, x: v.x, y: v.y, w: NODE_W, h: NODE_H, title, state,
          count: v.group ? v.group.count : v.rounds > 1 ? v.rounds : v.called ? 1 : 0,
          countMode: v.group && v.group.mode !== 'parallel' ? '↓' : '×',
          stack: v.group ? v.group.mode : null, subtitle: STATUS_TEXT[state], tip, attrs,
          memberOf: v.memberOf, groupKey: v.group && v.group.key, first: v.group && v.group.first,
        };
      }

      // ── Inside a FEDOT.MAS system ─────────────────────────────────────────
      function masScene(runId, model, blocks) {
        NODE_W = 132;
        NODE_H = 62;
        const run = fedotRuns.find(r => r.meta.run_id === runId);
        const cards = [];
        const links = [];
        let prev = null;
        (run ? run.agents : []).forEach((a, j) => {
          const c = {
            key: `A${a.name}`, x: PAD + j * (NODE_W + 40), y: PAD, w: NODE_W, h: NODE_H,
            title: humanize(a.name), state: a.status, count: a.starts, subtitle: STATUS_TEXT[a.status],
            tip: `${a.name}: ${STATUS_TEXT[a.status].toLowerCase()}, запусков ${a.starts}.`, attrs: {},
          };
          cards.push(c);
          if (prev) links.push({ from: prev, to: c, kind: 'side' });
          prev = c;
        });
        const n = cards.length;
        const notes = run && run.task ? [run.task] : [];
        return { cards, links, frames: [], totalW: PAD * 2 + n * NODE_W + Math.max(0, n - 1) * 40, totalH: PAD * 2 + NODE_H, notes, map: false };
      }

      // ── Drawing a scene ──────────────────────────────────────────────────
      function linkPath(l) {
        const a = l.from, b = l.to;
        if (l.tree) return edgePath({ ...l.tree, from: { x: a.x, y: a.y }, to: { x: b.x, y: b.y } });
        if (l.kind === 'side') {
          const y = a.y + a.h / 2;
          return `M${a.x + a.w},${y} L${b.x - 3},${b.y + b.h / 2}`;
        }
        const x1 = a.x + a.w / 2, y1 = a.y + a.h, x2 = b.x + b.w / 2, y2 = b.y - 3;
        if (l.kind === 'mas') {
          // Into a system from its left side: down to the bus, across to
          // beside the frame, down, and in to the first agent.
          const ym = b.y + b.h / 2;
          return `M${x1},${y1} L${x1},${l.busY} L${l.entryX},${l.busY} L${l.entryX},${ym} L${b.x - 3},${ym}`;
        }
        if (l.kind === 'down' || Math.abs(x1 - x2) < 1) return `M${x1},${y1} L${x2},${y2}`;
        // Right angles: down to the bus, across, down into the card.
        const yb = l.busY != null ? l.busY : (y1 + y2) / 2;
        return `M${x1},${y1} L${x1},${yb} L${x2},${yb} L${x2},${y2}`;
      }

      function wrapTitle(text, width, size) {
        const perLine = Math.max(4, Math.floor(width / (size * 0.58)));
        const words = String(text).split(/\s+/);
        const lines = [''];
        words.forEach(w => {
          const cur = lines[lines.length - 1];
          if (!cur) lines[lines.length - 1] = w;
          else if ((cur + ' ' + w).length <= perLine) lines[lines.length - 1] = cur + ' ' + w;
          else lines.push(w);
        });
        const out = lines.slice(0, 2);
        if (lines.length > 2 || out.some(l => l.length > perLine)) {
          out[out.length - 1] = out[out.length - 1].slice(0, perLine - 1) + '…';
        }
        return out;
      }

      function drawCard(parent, c, before, animate, moves) {
        const g = svgEl('g', {
          class: `cg-card cg-st-${c.state}${c.stack ? ' cg-stackcard' : ''}${c.link ? ' cg-link' : ''}`
            + `${c.attrs['data-block'] || c.attrs['data-group'] || c.attrs['data-member'] || c.attrs['data-link'] || c.attrs['data-mas'] ? ' cg-clickable' : ''}`,
        });
        Object.entries(c.attrs).forEach(([k, v]) => g.setAttribute(k, v));
        g.appendChild(svgEl('title', {}, c.tip));
        if (c.stack) {
          const side = c.stack === 'parallel';
          [2, 1].forEach(k => g.appendChild(svgEl('rect', {
            x: side ? 4 * k : 0, y: side ? 0 : 4 * k, width: c.w, height: c.h, rx: 8, class: 'cg-card-box cg-card-stack',
          })));
        }
        // A group: more layers behind it, like a deck of the agents it holds.
        if (c.label) {
          [2, 1].forEach(k => g.appendChild(svgEl('rect', {
            x: 5 * k, y: 5 * k, width: c.w, height: c.h, rx: 8, class: `cg-card-box cg-card-layer cg-card-layer-${k}`,
          })));
        }
        g.appendChild(svgEl('rect', { width: c.w, height: c.h, rx: 8, class: 'cg-card-box' }));
        const big = !!c.big;
        const small = big ? 9 : 8.5;
        // Top row: the group mark or the state icon, and the count.
        g.appendChild(svgEl('text', { x: 10, y: big ? 18 : 15, 'font-size': c.label ? 7.5 : small + 1.5, class: `cg-card-mark${c.label ? ' cg-card-group' : ''}` },
          c.label || (c.link ? '↗' : ICONS[c.state])));
        if (c.count != null) {
          g.appendChild(svgEl('text', { x: c.w - 10, y: big ? 18 : 15, 'font-size': small, 'text-anchor': 'end', class: 'cg-card-count' },
            `${c.countMode || '×'}${c.count}`));
        }
        const size = big ? 12.5 : 10.5;
        const lines = wrapTitle(c.title, c.w - 16, size);
        const mid = big ? c.h / 2 + 2 : c.h / 2 + 3;
        lines.forEach((line, i) => g.appendChild(svgEl('text', {
          x: c.w / 2, y: mid + (i - (lines.length - 1) / 2) * (size + 3), 'font-size': size, 'text-anchor': 'middle', class: 'cg-card-title',
        }, line)));
        if (c.subtitle) {
          g.appendChild(svgEl('text', { x: c.w / 2, y: c.h - (big ? 11 : 8), 'font-size': small, 'text-anchor': 'middle', class: 'cg-card-sub' },
            c.subtitle.length > (big ? 26 : 24) ? c.subtitle.slice(0, big ? 25 : 23) + '…' : c.subtitle));
        }
        const at = `translate(${c.x}px, ${c.y}px)`;
        const from = animate && (before.get(c.key)
          || (c.memberOf && before.get(`g${c.memberOf}`))
          || (c.first && before.get(c.first)));
        if (from && (from.x !== c.x || from.y !== c.y)) {
          g.style.transform = `translate(${from.x}px, ${from.y}px)`;
          moves.push([g, at]);
        } else {
          g.style.transform = at;
          if (animate && !from) { g.style.opacity = '0'; moves.push([g, at]); }
        }
        lastPos.set(c.key, { x: c.x, y: c.y });
        lastEls.set(c.key, { el: g, memberOf: c.memberOf });
        parent.appendChild(g);
        return g;
      }

      function drawScene(svg, scroller, scene, oldEls) {
        const defs = svgEl('defs');
        ['', '-idle', '-live'].forEach(kind => {
          const m = svgEl('marker', { id: `cg-arrow${kind}`, viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' });
          m.appendChild(svgEl('path', { d: 'M0,0 L8,4 L0,8 z', class: `cg-arrowhead${kind}` }));
          defs.appendChild(m);
        });
        svg.appendChild(defs);
        const world = svgEl('g', { class: 'cg-world' });
        svg.appendChild(world);
        if (shownView) world.style.transform = viewTransform(shownView);

        scene.frames.forEach(f => {
          const fg = svgEl('g', { class: `cg-frame${f.mas ? ' cg-frame-mas cg-clickable' : ''}` });
          if (f.mas) fg.setAttribute('data-mas', f.mas);
          fg.appendChild(svgEl('rect', { x: f.x, y: f.y, width: f.w, height: f.h, rx: 10, class: 'cg-frame-box' }));
          if (f.label) {
            fg.appendChild(svgEl('text', { x: f.x + 16, y: f.y + 20, 'font-size': 11, class: 'cg-frame-label' }, f.label));
            fg.appendChild(svgEl('text', { x: f.x + 16, y: f.y + 34, 'font-size': 9, class: 'cg-frame-sub' }, f.subtitle || ''));
            fg.appendChild(svgEl('title', {}, `${f.label} — система агентов FEDOT.MAS. Щелчок — открыть.`));
          }
          world.appendChild(fg);
        });

        const edges = svgEl('g', { class: 'cg-edges' });
        scene.links.forEach(l => {
          if (!l.from || !l.to) return;
          const idle = l.to.state === 'idle' || l.from.state === 'idle';
          const live = l.to.state === 'work' || l.to.state === 'wait';
          const kind = idle ? '-idle' : live ? '-live' : '';
          edges.appendChild(svgEl('path', {
            d: linkPath(l), class: `cg-edge${idle ? ' cg-edge-idle' : ''}${live ? ' cg-edge-live' : ''}`,
            'marker-end': `url(#cg-arrow${kind})`,
          }));
        });
        world.appendChild(edges);

        const before = lastPos;
        const animate = before.size > 0 && !REDUCED_MOTION.matches;
        const moves = [];
        lastPos = new Map();
        const nodes = svgEl('g', { class: 'cg-cards' });
        world.appendChild(nodes);
        scene.cards.forEach(c => drawCard(nodes, c, before, animate, moves));

        // Unless the reader moved the view: while something runs, the busiest
        // card is centred at a readable size; otherwise the whole scene fits.
        const W = scroller.clientWidth, H = scroller.clientHeight;
        const grow = document.fullscreenElement === document.getElementById('side-nav-panel') ? FULL_MAX : 1;
        const whole = Math.min(grow, W / scene.totalW, H / scene.totalH);
        // The map is the overview: always whole. Inside a block the work in
        // progress is what the reader came for.
        const focus = scene.map ? null : [...scene.cards].reverse().find(c => c.state === 'work');
        const fit = focus && whole < FIT_MIN ? FIT_MIN : Math.max(OVERVIEW_MIN, whole);
        const w = scene.totalW * fit, h = scene.totalH * fit;
        const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));
        autoView = {
          tx: w <= W || !focus ? (W - w) / 2 : clamp(W / 2 - (focus.x + focus.w / 2) * fit, W - w, 0),
          ty: h <= H || !focus ? Math.max(0, (H - h) / 2) : clamp(H / 2 - (focus.y + focus.h / 2) * fit, H - h, 0),
          s: fit,
        };

        // A stack that just folded: its old cards glide into it and fade.
        const ghosts = [];
        if (animate) {
          oldEls.forEach(({ el, memberOf }, key) => {
            const into = memberOf && !lastPos.has(key) && lastPos.get(`g${memberOf}`);
            if (!into) return;
            el.style.opacity = '1';
            nodes.insertBefore(el, nodes.firstChild);
            ghosts.push([el, `translate(${into.x}px, ${into.y}px)`]);
          });
        }
        if (entering && !REDUCED_MOTION.matches) {
          // Arriving in another view: it settles in from slightly closer.
          entering = false;
          const v = camera || autoView;
          world.style.opacity = '0';
          world.style.transform = viewTransform({ ...v, s: v.s * 1.06, tx: v.tx - scene.totalW * v.s * 0.03, ty: v.ty - scene.totalH * v.s * 0.03 });
          world.classList.add('cg-anim');
          svg.getBoundingClientRect();
          requestAnimationFrame(() => {
            world.style.opacity = '';
            applyCamera();
            clearTimeout(animEnd);
            animEnd = setTimeout(() => world.classList.remove('cg-anim'), ANIM_MS + 250);
          });
          return;
        }
        entering = false;
        if (!moves.length && !ghosts.length) { applyCamera(); return; }
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

      // ── The panel around the graph: path, legend, counters, activity ────
      const OPTIONS_KEY = 'coscientist.call_graph_options';
      let options = { legend: true, stats: true, activity: true };
      try { options = { ...options, ...JSON.parse(localStorage.getItem(OPTIONS_KEY) || '{}') }; } catch (_) { /* defaults */ }

      function setOption(name, value) {
        options[name] = !!value;
        try { localStorage.setItem(OPTIONS_KEY, JSON.stringify(options)); } catch (_) { /* this tab only */ }
        schedule();
      }

      function toggleOptions(force) {
        const menu = document.getElementById('call-graph-opts-menu');
        if (!menu) return;
        const open = force != null ? force : menu.classList.contains('hidden');
        menu.classList.toggle('hidden', !open);
        document.getElementById('call-graph-opts')?.setAttribute('aria-expanded', String(open));
        if (open) {
          menu.querySelectorAll('input[data-opt]').forEach(input => { input.checked = !!options[input.dataset.opt]; });
        }
      }

      function renderChrome(model, blocks, scene) {
        const esc = v => String(v).replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
        // Breadcrumbs: the map, then each level down.
        const crumbs = document.getElementById('call-graph-crumbs');
        if (crumbs) {
          const names = path.map(p => (p.type === 'mas'
            ? `FEDOT.MAS / ${String(fedotRuns.findIndex(r => r.meta.run_id === p.id) + 1).padStart(2, '0')}`
            : (blocks.find(b => b.id === p.id) || {}).title || 'Блок'));
          crumbs.classList.toggle('hidden', !path.length);
          crumbs.innerHTML = [`<button type="button" data-depth="0" class="cg-crumb">Карта</button>`,
            ...names.map((n, i) => `<span class="cg-crumb-sep" aria-hidden="true">›</span>`
              + (i === names.length - 1 ? `<span class="cg-crumb-here">${esc(n)}</span>`
                : `<button type="button" data-depth="${i + 1}" class="cg-crumb">${esc(n)}</button>`))].join('');
        }
        const legend = document.getElementById('call-graph-legend');
        if (legend) legend.classList.toggle('hidden', !options.legend);

        // Counters: who works now, how much of the map the run reached, how
        // many systems FEDOT built.
        const working = model.instances.filter(i => i !== model.rootInst && instState(i, model) === 'work');
        const fedotWorking = fedotRuns.flatMap((r, k) => r.agents.filter(a => a.status === 'work').map(a => ({ run: k, a })));
        const mapCards = scene.map ? scene.cards : [];
        const engaged = mapCards.filter(c => c.state !== 'idle').length;
        const stats = document.getElementById('call-graph-stats');
        if (stats) {
          stats.classList.toggle('hidden', !options.stats);
          stats.innerHTML = [
            [working.length + fedotWorking.length, 'работают сейчас'],
            [scene.map ? `${engaged}<small>/${mapCards.length}</small>` : `${model.calls}`, scene.map ? 'узлов задействовано' : 'вызовов'],
            [fedotRuns.length, 'создано МАС'],
          ].map(([n, l]) => `<div class="cg-stat"><b>${n}</b><span>${l}</span></div>`).join('');
        }
        const count = document.getElementById('call-graph-count');
        if (count) count.textContent = options.stats || !model.calls ? '' : String(model.calls);

        // "Working now": in full screen only, beside the graph.
        const panel = document.getElementById('call-graph-activity');
        if (panel) {
          const full = document.fullscreenElement === document.getElementById('side-nav-panel');
          panel.classList.toggle('hidden', !(full && options.activity));
          const waiting = blocks.filter(b => b.state === 'wait').map(b => b.title);
          const items = [
            ...working.map(i => displayName(i.name)),
            ...fedotWorking.map(({ run, a }) => `${humanize(a.name)} · МАС ${String(run + 1).padStart(2, '0')}`),
          ];
          panel.innerHTML = `<h4>В работе сейчас</h4>`
            + (items.length ? `<ul>${items.map(t => `<li>${esc(t)}</li>`).join('')}</ul>` : `<p class="cg-muted">Сейчас никто не работает</p>`)
            + (waiting.length ? `<p class="cg-muted">Ожидают результат: ${esc(waiting.join(', '))}</p>` : '')
            + (scene.notes.length ? `<h4>На карте</h4><p class="cg-muted">${esc(scene.notes.join(' · '))}</p>` : '');
        }
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
          const node = hit && hit.closest && hit.closest('.cg-card, .cg-frame-mas');
          if (!node) return;
          const attr = name => node.getAttribute(name);
          if (attr('data-group')) unfold(attr('data-group'));
          else if (attr('data-member')) fold(attr('data-member'));
          // Falling into a block, a linked block, or a FEDOT.MAS system.
          else if (attr('data-block')) navigate([...path, { type: 'block', id: attr('data-block') }]);
          else if (attr('data-link')) navigate([{ type: 'block', id: attr('data-link') }]);
          else if (attr('data-mas')) navigate([...path.filter(p => p.type !== 'mas'), { type: 'mas', id: attr('data-mas') }]);
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
        document.getElementById('call-graph-crumbs')?.addEventListener('click', event => {
          const depth = event.target.closest && event.target.closest('[data-depth]');
          if (depth) navigate(path.slice(0, Number(depth.dataset.depth)));
        });
        document.getElementById('call-graph-opts-menu')?.addEventListener('change', event => {
          if (event.target.dataset && event.target.dataset.opt) setOption(event.target.dataset.opt, event.target.checked);
        });
        document.addEventListener('click', event => {
          const menu = document.getElementById('call-graph-opts-menu');
          if (menu && !menu.classList.contains('hidden') && !event.target.closest('#call-graph-opts, #call-graph-opts-menu')) toggleOptions(false);
        });
        const button = document.getElementById('call-graph-full');
        if (button && !document.fullscreenEnabled) button.classList.add('hidden');
        initView(scroller);
        if (window.ResizeObserver) new ResizeObserver(() => schedule()).observe(scroller);
        render();
      }

      return { init, reset, feed, feedAgentEvent, markIdle, setAgentMeta, loadSession, loadSkeleton, toggleFullscreen, toggleOptions };
    })();

    document.addEventListener('DOMContentLoaded', () => CallGraph.init());
