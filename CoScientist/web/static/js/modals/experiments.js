// =========================================================================
// Experiment Viewer — agent call tree
// =========================================================================
    // =========================================================================
    // Experiment Viewer — agent call tree
    // =========================================================================
    // A call and its result reach the browser as two separate events, and they
    // interleave freely: agents fire tools in parallel, so a call's answer can
    // land after three other tools have come and gone. Every event is folded
    // into a single record addressed by `call_id` (the id ADK stamps on the
    // function call — never the tool name, which repeats), and the feed
    // groups records by the agent that made them rather than by arrival time:
    // each agent gets its own branch, so two agents running at once render as
    // two independent branches instead of one interleaved, hard-to-scan list.
    //
    // Delegation (`transfer_to_agent`, or an AgentTool call whose tool name IS
    // the subordinate's own agent name — see `activityRecordCall` above) is
    // the edge between branches: ADK always fires the delegating call's
    // `before_tool_callback` before the nested Runner it spawns produces any
    // event of its own, so the parent branch is guaranteed to exist before
    // the child's first record arrives — no reordering to guard against.

    let toolCallRecords = [];          // every call of the run, oldest first
    let toolCallsById = new Map();     // uid -> record
    let agentNodes = new Map();        // agent name -> { name, calls: [], firstSeenAt }
    let agentOrder = [];               // agent names, in first-seen order
    let agentParent = new Map();       // child agent name -> parent agent name
    let agentSpawnCall = new Map();    // child agent name -> uid of its delegation call
    const collapsedAgents = new Set(); // agent names whose branch is folded
    let toolSeq = 0;
    const tvOpenCards = new Set();     // uids whose bodies are unfolded
    const tvExpandedBlocks = new Set();
    let tvExpandAll = false;

    function resetExperimentViewer() {
      toolCallRecords = [];
      toolCallsById = new Map();
      agentNodes = new Map();
      agentOrder = [];
      agentParent = new Map();
      agentSpawnCall = new Map();
      collapsedAgents.clear();
      tvOpenCards.clear();
      tvExpandedBlocks.clear();
      tvExpandAll = false;
    }

    function openToolsViewer() {
      const modal = document.getElementById('experiment-modal');
      if (!modal) return;
      modal.classList.remove('hidden');
      loadAgentHierarchy();
      renderExperimentFeed();
    }

    function closeExperimentViewer() {
      const modal = document.getElementById('experiment-modal');
      if (modal) modal.classList.add('hidden');
    }

    function clearExperimentViewer() {
      resetExperimentViewer();
      renderExperimentFeed();
    }

    function safeTimeStr(val) {
      if (!val) return '';
      const d = (val instanceof Date) ? val : new Date(val);
      return isNaN(d.getTime()) ? '' : d.toLocaleTimeString('en-GB', { hour12: false });
    }

    // The branch a call belongs to — created on first use, kept for the rest
    // of the run so every later call from (or delegation into) this agent
    // lands in the same place.
    function agentNode(name) {
      let node = agentNodes.get(name);
      if (!node) {
        node = { name, calls: [], firstSeenAt: new Date() };
        agentNodes.set(name, node);
        agentOrder.push(name);
      }
      return node;
    }

    // KNOWN_AGENTS is declared in activity_rail.js; ensure fallback on window if loaded standalone
    if (typeof KNOWN_AGENTS === 'undefined') {
      window.KNOWN_AGENTS = new Set();
    }
    [
      'OrchestratorAgent', 'PlannerAgent', 'PlanningPipelineAgent',
      'HypothesesAgent', 'ResearchAgent', 'TaskExecutorAgent',
      'ToolPipelineAgent', 'ToolPreparerAgent', 'ParallelToolSearcherAgent',
      'LocalToolsExtractorAgent', 'ToolRetrieverAgent', 'ToolWebSearcherAgent',
      'ToolReranker', 'FullSetToolReranker', 'WebToolsDeployerAgent',
      'ExperimentAgent', 'CoderAgent', 'DatasetCollectorAgent',
      'MedicalAgent', 'McpBuilderAgent', 'ContextInitAgent',
      'ContextInitSessionAgent', 'ResultAggregatorAgent', 'FedotAgent'
    ].forEach(name => KNOWN_AGENTS.add(name));

    const STATIC_PARENT_MAP = new Map([
      ['PlannerAgent', 'OrchestratorAgent'],
      ['HypothesesAgent', 'OrchestratorAgent'],
      ['ResearchAgent', 'OrchestratorAgent'],
      ['TaskExecutorAgent', 'OrchestratorAgent'],
      ['MedicalAgent', 'OrchestratorAgent'],
      ['McpBuilderAgent', 'OrchestratorAgent'],
      ['ToolPipelineAgent', 'TaskExecutorAgent'],
      ['CoderAgent', 'TaskExecutorAgent'],
      ['DatasetCollectorAgent', 'CoderAgent'],
      ['ToolPreparerAgent', 'ToolPipelineAgent'],
      ['ExperimentAgent', 'ToolPipelineAgent'],
      ['ParallelToolSearcherAgent', 'ToolPreparerAgent'],
      ['FullSetToolReranker', 'ToolPreparerAgent'],
      ['WebToolsDeployerAgent', 'ToolPreparerAgent'],
      ['LocalToolsExtractorAgent', 'ParallelToolSearcherAgent'],
      ['ToolWebSearcherAgent', 'ParallelToolSearcherAgent'],
      ['ToolRetrieverAgent', 'LocalToolsExtractorAgent'],
      ['ToolReranker', 'LocalToolsExtractorAgent'],
      ['ContextInitAgent', 'OrchestratorAgent'],
      ['ContextInitSessionAgent', 'OrchestratorAgent'],
      ['ResultAggregatorAgent', 'OrchestratorAgent'],
    ]);

    async function loadAgentHierarchy() {
      try {
        const resp = await fetch('/api/agents');
        if (!resp.ok) return;
        const data = await resp.json();
        if (data && data.hierarchy && data.hierarchy.parents) {
          for (const [child, parent] of Object.entries(data.hierarchy.parents)) {
            STATIC_PARENT_MAP.set(child, parent);
          }
        }
        if (data && Array.isArray(data.agents)) {
          data.agents.forEach(a => KNOWN_AGENTS.add(a.name));
        }
      } catch {
        // retain static defaults
      }
    }
    loadAgentHierarchy();

    function resolveAndLinkParent(child, parentHint = null, spawnUid = null) {
      if (!child) return;
      agentNode(child);
      if (spawnUid && !agentSpawnCall.has(child)) {
        agentSpawnCall.set(child, spawnUid);
      }
      if (agentParent.has(child)) return;

      const parent = parentHint || STATIC_PARENT_MAP.get(child);
      if (parent && parent !== child) {
        agentParent.set(child, parent);
        agentNode(parent);
        resolveAndLinkParent(parent);
      }
    }

    // Mirrors the detection in `activityRecordCall` above: `transfer_to_agent`
    // (built-in routing) and an AgentTool call, whose tool name IS the
    // subordinate's agent name, are both a hand-off — the edge between two
    // branches of the tree, not a leaf tool call.
    function delegationTarget(tc) {
      if (tc.target_agent) return tc.target_agent;
      if (tc.is_delegation && tc.name) return tc.name;
      if (tc.name === 'transfer_to_agent') {
        return (tc.args && (tc.args.agent_name || tc.args.agentName)) || null;
      }
      if (KNOWN_AGENTS.has(tc.name)) return tc.name;
      return /Agent$/.test(tc.name) ? tc.name : null;
    }

    function addExperimentAgentEvent(author, data) {
      const node = agentNode(author);
      if (data.agent_class) node.agentClass = data.agent_class;
      resolveAndLinkParent(author, data.parent);
      node.status = (data.phase === 'agent_start') ? 'running' : 'idle';
      if (data.timestamp) {
        node.lastActive = new Date(data.timestamp);
      }
      renderExperimentFeed();
    }

    function addExperimentToolCall(author, tc) {
      const at = tc.timestamp ? new Date(tc.timestamp) : new Date();
      const target = delegationTarget(tc);

      resolveAndLinkParent(author, tc.parent);

      const rec = {
        uid: 'call-' + (++toolSeq),
        callId: tc.callId || null,
        author: author,
        name: tc.name,
        args: tc.args,
        argsTruncated: !!tc.truncated,
        argsUnknown: false,
        isDelegation: !!target,
        targetAgent: target,
        result: null,
        resultTruncated: false,
        status: 'running',
        startedAt: at,
        endedAt: null,
      };

      if (target) {
        resolveAndLinkParent(target, author, rec.uid);
      }

      agentNode(author).calls.push(rec);
      toolCallRecords.push(rec);
      toolCallsById.set(rec.uid, rec);
      if (tvExpandAll) tvOpenCards.add(rec.uid);
      trimExperimentLog();
      renderExperimentFeed();
    }

    // A result names its own call through `call_id`, which is what makes the
    // pairing safe when an agent runs the same tool twice at once. Only when
    // that id is missing does the newest still-running call of the same agent
    // and tool stand in for it.
    function matchToolCall(author, tr) {
      for (let i = toolCallRecords.length - 1; i >= 0; i--) {
        const rec = toolCallRecords[i];
        if (tr.callId && rec.callId === tr.callId) return rec;
      }
      if (tr.callId) return null;
      for (let i = toolCallRecords.length - 1; i >= 0; i--) {
        const rec = toolCallRecords[i];
        if (rec.status === 'running' && rec.author === author && rec.name === tr.name) return rec;
      }
      return null;
    }

    function addExperimentToolResponse(author, tr) {
      let rec = matchToolCall(author, tr);
      if (!rec) {
        // A result whose call this tab never saw (feed cleared mid-run, or the
        // server trimmed the older half of its replay history): keep it
        // visible as a call whose arguments are simply unknown.
        addExperimentToolCall(author, {
          name: tr.name, args: null, callId: tr.callId, timestamp: tr.timestamp,
        });
        rec = toolCallRecords[toolCallRecords.length - 1];
        rec.argsUnknown = true;
      }
      rec.status = tr.failed ? 'error' : 'success';
      rec.result = tr.response;
      rec.resultTruncated = !!tr.truncated;
      rec.endedAt = tr.timestamp ? new Date(tr.timestamp) : new Date();
      trimExperimentLog();
      renderExperimentFeed();
    }

    function truncateStr(s, max) {
      if (typeof s !== 'string') s = JSON.stringify(s, null, 2) || '';
      return s.length > max ? s.slice(0, max) + '…' : s;
    }

    // --- Human-readable rendering of tool args/results ------------------------
    // Raw JSON (`{"a": {"b": [1, 2]}}`) is hard to scan in a feed. These turn
    // any JSON-safe value into an indented key/value tree with no literal
    // braces or brackets — objects and arrays both recurse, so a list of
    // objects or a dict of lists still comes out readable.
    function tvIsPlainObject(v) {
      return v !== null && typeof v === 'object' && !Array.isArray(v);
    }

    function tvScalarHtml(v) {
      if (v === null || v === undefined) return '<span class="tv-null">null</span>';
      if (typeof v === 'boolean' || typeof v === 'number') {
        return `<span class="tv-scalar">${v}</span>`;
      }
      return `<span class="tv-string">${escHtml(String(v))}</span>`;
    }

    // A string value that is itself JSON text (a tool that returns, say, a
    // dict field pre-serialized to a string) parses the same way a top-level
    // result does — see `tvRenderAny` — so it renders as a tree instead of
    // an escaped one-line blob of braces and quotes.
    function tvTryParseJsonString(str) {
      const trimmed = str.endsWith(' …') ? str.slice(0, -2).trim() : str.trim();
      if (trimmed.length < 2 || (trimmed[0] !== '{' && trimmed[0] !== '[')) return null;
      try {
        return { value: JSON.parse(trimmed), truncated: false };
      } catch {
        const repaired = tvRepairTruncatedJson(trimmed);
        return repaired !== null ? { value: repaired, truncated: true } : null;
      }
    }

    function tvRenderNested(parsed, depth) {
      const note = parsed.truncated
        ? `<div style="padding-left:${depth * 16}px" class="tv-empty">… truncated (server preview cap)</div>`
        : '';
      return tvRender(parsed.value, depth) + note;
    }

    function tvRender(value, depth) {
      depth = depth || 0;
      const pad = depth ? ` style="padding-left:${depth * 16}px"` : '';
      if (Array.isArray(value)) {
        if (value.length === 0) return `<div${pad} class="tv-empty">(empty list)</div>`;
        return value.map(item => {
          const parsed = typeof item === 'string' ? tvTryParseJsonString(item) : null;
          if (parsed) {
            return `<div${pad} class="tv-row"><span class="tv-bullet">–</span>${tvRenderNested(parsed, depth + 1)}</div>`;
          }
          const nested = tvIsPlainObject(item) || Array.isArray(item);
          return `<div${pad} class="tv-row"><span class="tv-bullet">–</span>${nested ? tvRender(item, depth + 1) : tvScalarHtml(item)}</div>`;
        }).join('');
      }
      if (tvIsPlainObject(value)) {
        const keys = Object.keys(value);
        if (keys.length === 0) return `<div${pad} class="tv-empty">(empty)</div>`;
        return keys.map(k => {
          const v = value[k];
          const parsed = typeof v === 'string' ? tvTryParseJsonString(v) : null;
          if (parsed) {
            return `<div${pad} class="tv-row"><span class="tv-key">${escHtml(k)}</span>:</div>${tvRenderNested(parsed, depth + 1)}`;
          }
          const nested = (tvIsPlainObject(v) && Object.keys(v).length > 0) || (Array.isArray(v) && v.length > 0);
          if (nested) {
            return `<div${pad} class="tv-row"><span class="tv-key">${escHtml(k)}</span>:</div>${tvRender(v, depth + 1)}`;
          }
          return `<div${pad} class="tv-row"><span class="tv-key">${escHtml(k)}</span>: ${tvScalarHtml(v)}</div>`;
        }).join('');
      }
      return `<div${pad}>${tvScalarHtml(value)}</div>`;
    }

    // A dict/list the server had to flatten because its JSON text blew past
    // the sink's preview cap (see `_PREVIEW_LIMIT` in tool_activity.py) is cut
    // off mid-value and ends in " …" — no longer valid JSON. `retrieve_tools`
    // results (a list of tool descriptions) hit this constantly. Rather than
    // give up and show the raw fragment, close whatever string/brackets were
    // left open and, if that's still unparsable, drop the last (also-cut)
    // element and retry — walking back to the last point that *was* complete.
    function tvRepairTruncatedJson(text) {
      const closers = { '{': '}', '[': ']' };
      function closeOpen(str) {
        const stack = [];
        let inString = false;
        let escape = false;
        for (let i = 0; i < str.length; i++) {
          const c = str[i];
          if (inString) {
            if (escape) escape = false;
            else if (c === '\\') escape = true;
            else if (c === '"') inString = false;
            continue;
          }
          if (c === '"') { inString = true; continue; }
          if (c === '{' || c === '[') stack.push(c);
          else if (c === '}' || c === ']') stack.pop();
        }
        let candidate = inString ? str + '"' : str;
        candidate = candidate.replace(/[,:\s]+$/, '');
        return candidate + stack.slice().reverse().map(c => closers[c]).join('');
      }

      for (let cut = text.length; cut > 0;) {
        try {
          return JSON.parse(closeOpen(text.slice(0, cut)));
        } catch {
          const prevComma = text.lastIndexOf(',', cut - 1);
          if (prevComma < 0) return null;
          cut = prevComma;
        }
      }
      return null;
    }

    // `value` may already be a plain string — either a genuine string result,
    // or a JSON-ish dict/list truncated by the server's preview cap.
    function tvRenderAny(value) {
      if (typeof value === 'string') {
        const parsed = tvTryParseJsonString(value);
        if (parsed) return tvRenderNested(parsed, 0);
        const trimmed = value.trim();
        return trimmed ? `<div class="tv-text">${escHtml(value)}</div>` : '<div class="tv-empty">(empty)</div>';
      }
      if (value === null || value === undefined || (tvIsPlainObject(value) && Object.keys(value).length === 0)) {
        return '<div class="tv-empty">(none)</div>';
      }
      return tvRender(value, 0);
    }

    // Collapsed height must match the `.tv-collapsible` max-height in <style>
    // (12rem @ 16px base) — used to decide whether a block needs its toggle.
    const TV_COLLAPSED_MAX_PX = 192;

    // Card bodies are rendered only while unfolded, so a run with hundreds of
    // calls still keeps the feed's DOM small.
    function toggleToolCard(uid) {
      if (tvOpenCards.has(uid)) tvOpenCards.delete(uid);
      else tvOpenCards.add(uid);
      renderExperimentFeed();
    }

    function toggleAgentNode(name) {
      if (collapsedAgents.has(name)) collapsedAgents.delete(name);
      else collapsedAgents.add(name);
      renderExperimentFeed();
    }

    // Sticky: a later call opens unfolded too while "Expand all" is on.
    function toggleExperimentExpandAll() {
      tvExpandAll = !tvExpandAll;
      tvOpenCards.clear();
      if (tvExpandAll) toolCallRecords.forEach(rec => tvOpenCards.add(rec.uid));
      renderExperimentFeed();
    }

    // One "Show more" button covers two different jobs, picked per click:
    //  - a block that only *looks* cut off (tall preview, nothing hidden from
    //    the server) just expands its CSS max-height — no network involved;
    //  - a block whose value the server actually truncated (see
    //    `_PREVIEW_LIMIT` in tool_activity.py) fetches the untruncated value
    //    exactly once first — the record keeps it, so collapsing and expanding
    //    again never re-fetches.
    async function toggleTvBlock(btn) {
      const blockId = btn.dataset.tvToggle;
      const el = document.getElementById(blockId);
      if (!el) return;
      const rec = toolCallsById.get(btn.dataset.uid);
      const field = btn.dataset.field;
      const truncated = rec && (field === 'args' ? rec.argsTruncated : rec.resultTruncated);

      if (rec && rec.callId && truncated && !el.classList.contains('tv-expanded')) {
        btn.disabled = true;
        btn.textContent = 'Loading…';
        try {
          const data = await apiJson(sessionApi('/tool-activity/' + encodeURIComponent(rec.callId)));
          const full = Object.prototype.hasOwnProperty.call(data, field) ? data[field] : null;
          if (field === 'args') {
            rec.args = full;
            rec.argsTruncated = false;
          } else {
            rec.result = field === 'error' ? { error: full } : full;
            rec.resultTruncated = false;
          }
          el.innerHTML = tvRenderAny(field === 'args' ? rec.args : rec.result);
        } catch (err) {
          btn.disabled = false;
          btn.textContent = 'Retry — ' + (err.message || 'failed to load full result');
          return;
        }
        btn.disabled = false;
      }

      const expanded = el.classList.toggle('tv-expanded');
      if (expanded) tvExpandedBlocks.add(blockId); else tvExpandedBlocks.delete(blockId);
      btn.textContent = expanded ? 'Show less' : 'Show more';
    }

    // Whether a block's toggle button is needed depends on its rendered
    // height, which is only known once the HTML is actually in the DOM — a
    // block still awaiting its server fetch always gets one regardless.
    function initTvToggles(root) {
      root.querySelectorAll('.tv-collapsible').forEach(el => {
        const btn = root.querySelector(`[data-tv-toggle="${el.id}"]`);
        if (!btn) return;
        if (el.classList.contains('tv-expanded')) btn.textContent = 'Show less';
        if (btn.dataset.needsFetch) return;
        btn.classList.toggle('hidden', el.scrollHeight <= TV_COLLAPSED_MAX_PX + 4);
      });
    }

    // Every tool call in the run lands here now, so keep the log bounded and
    // rebuild the DOM only while the viewer is actually open.
    const EXPERIMENT_LOG_LIMIT = 800;

    function trimExperimentLog() {
      if (toolCallRecords.length <= EXPERIMENT_LOG_LIMIT) return;
      const dropped = toolCallRecords.splice(0, toolCallRecords.length - EXPERIMENT_LOG_LIMIT);
      const touchedAgents = new Set();
      const droppedUids = new Set(dropped.map(rec => rec.uid));
      agentSpawnCall.forEach((uid, child) => {
        if (droppedUids.has(uid)) agentSpawnCall.delete(child);
      });
      dropped.forEach(rec => {
        toolCallsById.delete(rec.uid);
        tvOpenCards.delete(rec.uid);
        ['args', 'result', 'error'].forEach(field => tvExpandedBlocks.delete(`tv-${rec.uid}-${field}`));
        const node = agentNodes.get(rec.author);
        if (node) {
          node.calls = node.calls.filter(call => call !== rec);
          touchedAgents.add(rec.author);
        }
      });
      // A branch with nothing left of its own and no child branch to carry
      // is dead weight — drop it. One still holding a child stays, as a bare
      // header, so that child keeps its place in the tree.
      touchedAgents.forEach(name => {
        const node = agentNodes.get(name);
        if (!node || node.calls.length) return;
        const hasChildren = agentOrder.some(n => agentParent.get(n) === name);
        if (hasChildren) return;
        agentNodes.delete(name);
        agentOrder = agentOrder.filter(n => n !== name);
        agentParent.delete(name);
        agentSpawnCall.delete(name);
        collapsedAgents.delete(name);
      });
    }

    const TV_STATUS = {
      running: {
        icon: 'progress_activity', tone: 'text-tertiary',
        border: 'border-tertiary/25', bg: 'bg-tertiary/5',
      },
      success: {
        icon: 'check_circle', tone: 'text-secondary',
        border: 'border-secondary/25', bg: 'bg-secondary/5',
      },
      error: {
        icon: 'error', tone: 'text-error',
        border: 'border-error/30', bg: 'bg-error/5',
      },
    };

    function tvDuration(rec) {
      const ms = Math.max(0, (rec.endedAt || new Date()) - rec.startedAt);
      if (ms < 1000) return ms + 'ms';
      if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
      return Math.floor(ms / 60000) + 'm' + String(Math.round((ms % 60000) / 1000)).padStart(2, '0') + 's';
    }

    // The one-line gist of a call's arguments, so a folded card still says
    // *what* was asked — `query=SMILES of ibuprofen` rather than a bare name.
    function tvArgsSummary(args) {
      if (args === null || args === undefined) return '';
      if (typeof args === 'string') return truncateStr(args.replace(/\s+/g, ' ').trim(), 110);
      if (typeof args !== 'object') return String(args);
      const parts = (Array.isArray(args) ? args.map((v, i) => [String(i), v]) : Object.entries(args))
        .map(([key, value]) => {
          const text = (value === null || typeof value !== 'object')
            ? String(value)
            : JSON.stringify(value);
          return key + '=' + truncateStr(String(text).replace(/\s+/g, ' ').trim(), 46);
        });
      return truncateStr(parts.join('  '), 110);
    }

    function tvHasValue(value) {
      if (value === null || value === undefined || value === '') return false;
      if (Array.isArray(value)) return value.length > 0;
      if (typeof value === 'object') return Object.keys(value).length > 0;
      return true;
    }

    // One labelled args/output block inside an unfolded card. `field` is also
    // the key the server stores the untruncated value under (args/result/error).
    function tvValueBlock(rec, field, label, value, emptyText) {
      const blockId = `tv-${rec.uid}-${field}`;
      const truncated = field === 'args' ? rec.argsTruncated : rec.resultTruncated;
      const needsFetch = truncated && !!rec.callId;
      const expandedCls = tvExpandedBlocks.has(blockId) ? ' tv-expanded' : '';
      return `
        <div>
          <div class="text-[9px] font-bold text-outline-variant uppercase tracking-widest mb-1">${label}</div>
          <div id="${blockId}" class="tv-collapsible${expandedCls} bg-surface-container-lowest/80 border border-outline-variant/10 rounded-md p-2.5 text-[11px] text-on-surface-variant font-mono leading-relaxed">
            ${tvHasValue(value) ? tvRenderAny(value) : `<div class="tv-empty">${escHtml(emptyText)}</div>`}
          </div>
          <div class="flex justify-end">
            <button data-tv-toggle="${blockId}" data-uid="${rec.uid}" data-field="${field}"
              data-needs-fetch="${needsFetch ? '1' : ''}" onclick="toggleTvBlock(this)"
              class="${needsFetch ? '' : 'hidden'} mt-1 text-[9px] font-bold text-outline-variant hover:text-primary uppercase tracking-wider">Show more</button>
          </div>
        </div>`;
    }

    function renderToolCardBody(rec) {
      const args = tvValueBlock(
        rec, 'args', 'Arguments', rec.args,
        rec.argsUnknown ? '(the call itself was not recorded)' : '(no arguments)',
      );
      const output = rec.status === 'running'
        ? `<div>
             <div class="text-[9px] font-bold text-outline-variant uppercase tracking-widest mb-1">Output</div>
             <div class="bg-surface-container-lowest/80 border border-outline-variant/10 rounded-md p-2.5 text-[11px]">
               <span class="tv-empty">waiting for the result…</span>
             </div>
           </div>`
        : tvValueBlock(
          rec,
          rec.status === 'error' ? 'error' : 'result',
          rec.status === 'error' ? 'Error' : 'Output',
          rec.result,
          '(empty)',
        );
      return `<div class="px-2.5 pb-2.5 pt-2 space-y-2 border-t border-outline-variant/10">${args}${output}</div>`;
    }

    // One tool call: status, name, argument gist and duration on a single
    // row, with both bodies folded away underneath. The agent that made the
    // call is never repeated here — it's already the branch header above.
    function renderToolCard(rec) {
      const st = TV_STATUS[rec.status] || TV_STATUS.running;
      const open = tvOpenCards.has(rec.uid);
      const running = rec.status === 'running';
      const summary = tvArgsSummary(rec.args);
      const icon = rec.isDelegation ? 'alt_route' : st.icon;
      const meta = safeTimeStr(rec.startedAt);
      return `
        <div class="rounded-md border ${st.border} ${st.bg} overflow-hidden">
          <button type="button" onclick="toggleToolCard('${rec.uid}')"
            class="w-full flex items-center gap-2 px-2.5 py-1.5 text-left hover:bg-surface-variant/20 transition-colors">
            <span class="material-symbols-outlined text-[14px] text-outline-variant shrink-0">${open ? 'expand_more' : 'chevron_right'}</span>
            <span class="material-symbols-outlined text-[14px] ${st.tone} shrink-0${running ? ' tv-spin' : ''}">${icon}</span>
            <span class="text-[11px] font-bold font-mono text-on-surface shrink-0">${escHtml(rec.name)}</span>
            ${rec.isDelegation ? '<span class="shrink-0 text-[8px] font-bold uppercase tracking-wider text-primary/70">delegates</span>' : ''}
            <span class="flex-1 min-w-0 truncate text-[10px] font-mono text-outline-variant/70">${escHtml(summary)}</span>
            <span class="shrink-0 text-[9px] font-mono ${st.tone}">${running ? 'running…' : tvDuration(rec)}</span>
            <span class="shrink-0 text-[9px] font-mono text-outline-variant/70">${meta}</span>
          </button>
          ${open ? renderToolCardBody(rec) : ''}
        </div>`;
    }

    // One branch of the call tree: the agent's own tool calls, each child
    // branch nested and indented directly under the `delegates` card that
    // spawned it. Two agents delegated to in parallel therefore stay next to
    // their own hand-off rows instead of both piling up after the parent's
    // last call, where neither could be told apart.
    function renderAgentNode(name, visited = new Set()) {
      if (!name || visited.has(name)) return '';
      visited.add(name);

      const node = agentNodes.get(name);
      if (!node) return '';
      const calls = node.calls;
      const children = agentOrder.filter(n => agentParent.get(n) === name && !visited.has(n));
      // A child whose delegation card is gone — trimmed out of the log, or
      // never seen because the feed joined the run late — still belongs to
      // this branch: it goes at the tail rather than disappearing.
      const ownUids = new Set(calls.map(rec => rec.uid));
      const childrenByCall = new Map();
      const tailChildren = [];
      children.forEach(child => {
        const uid = agentSpawnCall.get(child);
        if (uid && ownUids.has(uid)) {
          if (!childrenByCall.has(uid)) childrenByCall.set(uid, []);
          childrenByCall.get(uid).push(child);
        } else {
          tailChildren.push(child);
        }
      });
      const running = calls.filter(rec => rec.status === 'running').length;
      const failed = calls.filter(rec => rec.status === 'error').length;
      const done = calls.length - running;
      const isAgentRunning = node.status === 'running' || running > 0;
      const pills = [
        running ? `<span class="text-tertiary">${running} running</span>` : (node.status === 'running' ? `<span class="text-tertiary">running</span>` : ''),
        failed ? `<span class="text-error">${failed} failed</span>` : '',
        (!running && calls.length) ? `<span class="text-secondary">${done - failed}/${calls.length} ok</span>` : '',
      ].filter(Boolean).join('<span class="text-outline-variant/30">·</span>');
      const collapsed = collapsedAgents.has(name);
      const lastActive = calls.length
        ? (calls[calls.length - 1].endedAt || calls[calls.length - 1].startedAt)
        : (node.lastActive || node.firstSeenAt);

      const nestBranches = names => names.length ? `
            <div class="ml-3 pl-3 border-l-2 border-outline-variant/15 space-y-1.5">
              ${names.map(childName => renderAgentNode(childName, new Set(visited))).join('')}
            </div>` : '';

      const body = collapsed ? '' : `
        <div class="px-2 pb-2 space-y-1.5">
          ${calls.map(rec => renderToolCard(rec) + nestBranches(childrenByCall.get(rec.uid) || [])).join('')}
          ${nestBranches(tailChildren)}
        </div>`;

      return `
        <div class="rounded-lg border border-outline-variant/15 bg-surface-container-low/40">
          <button type="button" onclick="toggleAgentNode('${escHtml(name)}')"
            class="w-full flex items-center gap-2 px-2.5 py-2 text-left hover:bg-surface-variant/20 transition-colors">
            <span class="material-symbols-outlined text-[14px] text-outline-variant shrink-0">${collapsed ? 'chevron_right' : 'expand_more'}</span>
            <span class="material-symbols-outlined text-[14px] text-primary shrink-0">${agentIcon(name)}</span>
            <span class="text-[11px] font-bold uppercase tracking-wider text-on-surface shrink-0">${escHtml(name)}</span>
            ${calls.length ? `<span class="text-[9px] font-mono px-1.5 py-0.5 rounded bg-primary/10 text-primary shrink-0">${calls.length} call${calls.length === 1 ? '' : 's'}</span>` : (node.agentClass ? `<span class="text-[8px] font-mono uppercase px-1.5 py-0.5 rounded bg-outline-variant/10 text-outline-variant shrink-0">${escHtml(node.agentClass.replace('Agent', ''))}</span>` : '')}
            <span class="flex-1"></span>
            <span class="flex items-center gap-1.5 text-[9px] font-mono shrink-0">${pills}</span>
            <span class="shrink-0 text-[9px] font-mono text-outline-variant/60">${safeTimeStr(lastActive)}</span>
          </button>
          ${body}
        </div>`;
    }

    function renderExperimentFeed() {
      const modal = document.getElementById('experiment-modal');
      if (modal && modal.classList.contains('hidden')) return;
      const feed = document.getElementById('experiment-feed');
      if (!feed) return;

      const counter = document.getElementById('experiment-event-count');
      if (counter) {
        const running = toolCallRecords.filter(rec => rec.status === 'running').length;
        const failed = toolCallRecords.filter(rec => rec.status === 'error').length;
        counter.textContent = toolCallRecords.length + ' calls'
          + (running ? ' · ' + running + ' running' : '')
          + (failed ? ' · ' + failed + ' failed' : '');
      }
      const expandBtn = document.getElementById('experiment-expand-all');
      if (expandBtn) expandBtn.textContent = tvExpandAll ? 'Collapse all' : 'Expand all';

      if (toolCallRecords.length === 0 && agentNodes.size === 0) {
        feed.innerHTML = `
          <div class="flex flex-col items-center justify-center h-full opacity-40 py-16">
            <span class="material-symbols-outlined text-4xl text-primary/30 mb-3">science</span>
            <p class="text-sm text-outline-variant font-medium">No tool activity yet</p>
            <p class="text-[10px] text-outline-variant/50 mt-1">Tool calls and results from agents will appear here in real time</p>
          </div>`;
        return;
      }

      // Follow the tail only while the user is already there — re-rendering on
      // every event must not yank the feed away from a card being read.
      const atBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 80;
      const keepTop = feed.scrollTop;
      const roots = agentOrder.filter(name => !agentParent.has(name));
      feed.innerHTML = roots.map(name => renderAgentNode(name)).join('');
      initTvToggles(feed);
      feed.scrollTop = atBottom ? feed.scrollHeight : keepTop;
    }

    // Explicitly bind to window for global access across templates and scripts
    window.openToolsViewer = openToolsViewer;
    window.closeExperimentViewer = closeExperimentViewer;
    window.clearExperimentViewer = clearExperimentViewer;
    window.resetExperimentViewer = resetExperimentViewer;
    window.toggleToolCard = toggleToolCard;
    window.toggleAgentNode = toggleAgentNode;
    window.toggleExperimentExpandAll = toggleExperimentExpandAll;
    window.toggleTvBlock = toggleTvBlock;
    window.addExperimentAgentEvent = addExperimentAgentEvent;
    window.addExperimentToolCall = addExperimentToolCall;
    window.addExperimentToolResponse = addExperimentToolResponse;
