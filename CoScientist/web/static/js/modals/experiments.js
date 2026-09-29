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
    let agentNodes = new Map();        // runtime key -> { name, calls: [], firstSeenAt }
    let agentOrder = [];               // runtime keys, in first-seen order
    let agentParent = new Map();       // child runtime key -> parent runtime key
    let agentSpawnCall = new Map();    // child runtime key -> uid of its delegation call
    const collapsedAgents = new Set(); // runtime keys whose branch is folded
    let toolSeq = 0;
    const tvOpenCards = new Set();     // uids whose bodies are unfolded
    const tvExpandedBlocks = new Set();
    let tvExpandAll = false;

    // How args/results are shown: the readable key/value tree ('yaml') or
    // pretty-printed JSON ('json'). Per-browser preference, not per-run.
    const TV_FORMAT_KEY = 'coscientist.toolsViewer.format';
    let tvFormat = 'yaml';
    try {
      if (localStorage.getItem(TV_FORMAT_KEY) === 'json') tvFormat = 'json';
    } catch { /* storage unavailable: keep the default */ }

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
    // The configured name is a label, not an identity: two parallel AgentTool
    // calls can both be `CoderAgent`.  Child ADK session ids distinguish those
    // runs while retaining the friendly name in the branch header.
    function agentKey(name, instance = null) {
      return instance ? `${name}\u0000${instance}` : name;
    }

    function agentNode(name, instance = null) {
      const key = agentKey(name, instance);
      let node = agentNodes.get(key);
      if (!node) {
        node = { key, name, calls: [], firstSeenAt: new Date() };
        agentNodes.set(key, node);
        agentOrder.push(key);
      }
      return node;
    }

    // KNOWN_AGENTS is declared in activity_rail.js; ensure fallback on window if loaded standalone
    if (typeof KNOWN_AGENTS === 'undefined') {
      window.KNOWN_AGENTS = new Set();
    }
    [
      'OrchestratorAgent', 'PlannerAgent', 'PlanCriticAgent', 'PlanningPipelineAgent',
      'HypothesesAgent', 'ResearchAgent', 'TaskExecutorAgent',
      'ToolPipelineAgent', 'ToolPreparerAgent', 'ParallelToolSearcherAgent',
      'LocalToolsExtractorAgent', 'ToolRetrieverAgent', 'ToolWebSearcherAgent',
      'ToolReranker', 'FullSetToolReranker', 'WebToolsDeployerAgent',
      'ExperimentAgent', 'CoderAgent', 'DatasetCollectorAgent',
      'MedicalAgent', 'McpBuilderAgent', 'ContextInitAgent',
      'ContextInitSessionAgent', 'ResultAggregatorAgent', 'FedotAgent',
      // Experiment Module stages (experiments profile).
      'ExperimentModuleAgent', 'ExperimentPlannerAgent',
      'ExperimentExecutorAgent', 'ExperimentResultReviewAgent'
    ].forEach(name => KNOWN_AGENTS.add(name));

    // INTERNAL_AGENTS and isInternalAgent() come from activity_rail.js, which
    // loads first. Redeclaring them here is a SyntaxError for the const, and
    // a global function that shadows window.isInternalAgent and recurses.

    const STATIC_PARENT_MAP = new Map([
      ['PlannerAgent', 'OrchestratorAgent'],
      ['Planner', 'OrchestratorAgent'],
      ['HypothesesAgent', 'OrchestratorAgent'],
      ['ResearchAgent', 'OrchestratorAgent'],
      ['TaskExecutorAgent', 'OrchestratorAgent'],
      ['MedicalAgent', 'OrchestratorAgent'],
      ['McpBuilderAgent', 'OrchestratorAgent'],
      ['ToolPipelineAgent', 'TaskExecutorAgent'],
      ['CoderAgent', 'TaskExecutorAgent'],
      ['DatasetCollectorAgent', 'CoderAgent'],
      ['ToolPreparerAgent', 'ToolPipelineAgent'],
      ['ExperimentAgent', 'TaskExecutorAgent'],
      ['ParallelToolSearcherAgent', 'ToolPreparerAgent'],
      ['FullSetToolReranker', 'ToolPreparerAgent'],
      ['WebToolsDeployerAgent', 'ToolPreparerAgent'],
      ['LocalToolsExtractorAgent', 'ParallelToolSearcherAgent'],
      ['ToolWebSearcherAgent', 'ParallelToolSearcherAgent'],
      ['ToolRetrieverAgent', 'LocalToolsExtractorAgent'],
      ['ToolReranker', 'LocalToolsExtractorAgent'],
      ['ContextInitAgent', 'OrchestratorAgent'],
      // Not a YAML agent (SessionAgent's plan critic), so /api/agents never lists it.
      ['PlanCriticAgent', 'PlannerAgent'],
      ['ContextInitSessionAgent', 'OrchestratorAgent'],
      ['ResultAggregatorAgent', 'OrchestratorAgent'],
      ['ExecutorSwitchAgent', 'TaskExecutorAgent'],
      // Experiment Module: on the experiments profile it stands where
      // TaskExecutorAgent stands here, with its four stages beneath it. This
      // map is only the fallback — /api/agents carries the live hierarchy.
      ['ExperimentModuleAgent', 'OrchestratorAgent'],
      ['ExperimentPlannerAgent', 'ExperimentModuleAgent'],
      ['ExperimentExecutorAgent', 'ExperimentModuleAgent'],
      ['ExperimentResultReviewAgent', 'ExperimentModuleAgent'],
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
        if (data && Array.isArray(data.internal_agents)) {
          data.internal_agents.forEach(name => INTERNAL_AGENTS.add(name));
        }
        if (data && Array.isArray(data.agents)) {
          data.agents.forEach(a => {
            if (!a.is_internal && !isInternalAgent(a.name)) {
              KNOWN_AGENTS.add(a.name);
            }
          });
        }
      } catch {
        // retain static defaults
      }
    }
    loadAgentHierarchy();

    function resolveNonInternalParent(name) {
      let curr = name;
      const visited = new Set();
      while (curr && isInternalAgent(curr) && !visited.has(curr)) {
        visited.add(curr);
        curr = STATIC_PARENT_MAP.get(curr) || null;
      }
      return (curr && !isInternalAgent(curr)) ? curr : null;
    }

    // Would linking `node` under `parent` close a loop? A loop leaves the tree
    // without a root, and the viewer renders nothing at all.
    function isAncestor(node, parent) {
      const seen = new Set();
      for (let curr = parent; curr && !seen.has(curr); curr = agentParent.get(curr)) {
        if (curr === node) return true;
        seen.add(curr);
      }
      return false;
    }

    function resolveAndLinkParent(childKey, parentHint = null, spawnUid = null, parentInstance = null) {
      const childNode = agentNodes.get(childKey);
      const child = childNode ? childNode.name : childKey;
      if (!child || isInternalAgent(child)) return;
      if (!childNode) childKey = agentNode(child).key;
      if (spawnUid && !agentSpawnCall.has(childKey)) {
        agentSpawnCall.set(childKey, spawnUid);
      }
      if (agentParent.has(childKey)) return;

      const parent = resolveNonInternalParent(parentHint) || resolveNonInternalParent(STATIC_PARENT_MAP.get(child));
      // The runtime identity belongs to the hinted parent itself; an ancestor
      // found by skipping internal agents may run in another (outer) session.
      const parentKey = parent ? agentNode(parent, parent === parentHint ? parentInstance : null).key : null;
      if (parent && parent !== child && !isInternalAgent(parent) && !isAncestor(childKey, parentKey)) {
        agentParent.set(childKey, parentKey);
        resolveAndLinkParent(parentKey);
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
      if (isInternalAgent(author)) return;
      const node = agentNode(author, data.agent_instance);
      if (data.agent_class) node.agentClass = data.agent_class;
      const spawnUid = data.phase === 'agent_start'
        ? findDelegationSpawn(author, data.parent, data.parent_instance, data.spawn_call_id)
        : null;
      resolveAndLinkParent(node.key, data.parent, spawnUid, data.parent_instance);
      node.status = (data.phase === 'agent_start') ? 'running' : 'idle';
      if (data.timestamp) {
        node.lastActive = new Date(data.timestamp);
      }
      renderExperimentFeed();
    }

    function addExperimentToolCall(author, tc) {
      if (isInternalAgent(author)) return;
      const at = tc.timestamp ? new Date(tc.timestamp) : new Date();
      const target = delegationTarget(tc);
      if (target && isInternalAgent(target)) {
        return;
      }

      const node = agentNode(author, tc.agentInstance);
      resolveAndLinkParent(node.key, tc.parent, null, tc.parentInstance);

      const rec = {
        uid: 'call-' + (++toolSeq),
        callId: tc.callId || null,
        author: author,
        authorKey: node.key,
        name: tc.name,
        args: tc.args,
        argsTruncated: !!tc.truncated,
        argsUnknown: false,
        isDelegation: !!target,
        targetAgent: target,
        result: null,
        resultTruncated: false,
        // Extra input blocks known once the call closes, keyed by the field
        // the server stores their full value under: {value, truncated}.
        inputs: {},
        status: 'running',
        startedAt: at,
        endedAt: null,
      };

      // The nested run publishes its own `agent_start` with a distinct child
      // session id.  Linking it here by configured name would pre-create one
      // shared target node and merge concurrent identical delegations.

      node.calls.push(rec);
      toolCallRecords.push(rec);
      toolCallsById.set(rec.uid, rec);
      if (tvExpandAll) tvOpenCards.add(rec.uid);
      trimExperimentLog();
      renderExperimentFeed();
    }

    // A child starts after its AgentTool call was announced.  Pair it with an
    // as-yet-unclaimed delegation from the same runtime parent, so concurrent
    // launches of the same named agent remain next to their own hand-off row.
    // The server names the spawning call (`spawn_call_id`) when it knows it;
    // the name-based search below is the fallback for events without it.
    function findDelegationSpawn(target, parent, parentInstance, spawnCallId = null) {
      if (!parent) return null;
      const parentKey = agentKey(parent, parentInstance);
      if (spawnCallId) {
        const rec = toolCallRecords.find(r => r.callId === spawnCallId && r.authorKey === parentKey);
        if (rec) return rec.uid;
      }
      const claimed = new Set(agentSpawnCall.values());
      for (let i = toolCallRecords.length - 1; i >= 0; i--) {
        const rec = toolCallRecords[i];
        if (rec.isDelegation && rec.targetAgent === target && rec.authorKey === parentKey && !claimed.has(rec.uid)) {
          return rec.uid;
        }
      }
      return null;
    }

    // A result names its own call through `call_id`, which is what makes the
    // pairing safe when an agent runs the same tool twice at once. Only when
    // that id is missing does the newest still-running call of the same agent
    // and tool stand in for it.
    function matchToolCall(author, tr) {
      for (let i = toolCallRecords.length - 1; i >= 0; i--) {
        const rec = toolCallRecords[i];
        if (tr.callId && rec.callId === tr.callId && rec.authorKey === agentKey(author, tr.agentInstance)) return rec;
      }
      if (tr.callId) return null;
      for (let i = toolCallRecords.length - 1; i >= 0; i--) {
        const rec = toolCallRecords[i];
        if (rec.status === 'running' && rec.authorKey === agentKey(author, tr.agentInstance) && rec.name === tr.name) return rec;
      }
      return null;
    }

    function addExperimentToolResponse(author, tr) {
      if (isInternalAgent(author)) return;
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
      rec.inputs = {};
      if (tvHasValue(tr.effectiveArgs)) {
        rec.inputs.effective_args = { value: tr.effectiveArgs, truncated: !!tr.effectiveArgsTruncated };
      }
      if (tvHasValue(tr.stateInputs)) {
        rec.inputs.state_inputs = { value: tr.stateInputs, truncated: !!tr.stateInputsTruncated };
      }
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
        ? `<div style="padding-left:${depth * 16}px" class="tv-empty">${t('experiments.truncated')}</div>`
        : '';
      return tvRender(parsed.value, depth) + note;
    }

    function tvRender(value, depth) {
      depth = depth || 0;
      const pad = depth ? ` style="padding-left:${depth * 16}px"` : '';
      if (Array.isArray(value)) {
        if (value.length === 0) return `<div${pad} class="tv-empty">${t('experiments.emptyList')}</div>`;
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
        if (keys.length === 0) return `<div${pad} class="tv-empty">${t('experiments.emptyValue')}</div>`;
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

    // Pretty-printed JSON with the same colours as the tree view. Built by
    // walking the value rather than regex-highlighting JSON.stringify output,
    // so escaping stays correct whatever the strings contain.
    function tvJsonHtml(value, indent) {
      indent = indent || '';
      const inner = indent + '  ';
      if (value === null || value === undefined) return '<span class="tv-null">null</span>';
      if (typeof value === 'boolean' || typeof value === 'number') {
        return `<span class="tv-scalar">${value}</span>`;
      }
      if (typeof value === 'string') {
        return `<span class="tv-json-string">${escHtml(JSON.stringify(value))}</span>`;
      }
      if (Array.isArray(value)) {
        if (value.length === 0) return '[]';
        return '[\n' + value.map(v => inner + tvJsonHtml(v, inner)).join(',\n') + '\n' + indent + ']';
      }
      const keys = Object.keys(value);
      if (keys.length === 0) return '{}';
      return '{\n' + keys.map(k =>
        `${inner}<span class="tv-key">${escHtml(JSON.stringify(k))}</span>: ${tvJsonHtml(value[k], inner)}`,
      ).join(',\n') + '\n' + indent + '}';
    }

    function tvRenderJson(value, truncated) {
      const note = truncated ? `<div class="tv-empty">${t('experiments.truncated')}</div>` : '';
      return `<div class="tv-json">${tvJsonHtml(value)}</div>${note}`;
    }

    // `value` may already be a plain string — either a genuine string result,
    // or a JSON-ish dict/list truncated by the server's preview cap.
    function tvRenderAny(value) {
      if (typeof value === 'string') {
        const parsed = tvTryParseJsonString(value);
        if (parsed) {
          return tvFormat === 'json' ? tvRenderJson(parsed.value, parsed.truncated) : tvRenderNested(parsed, 0);
        }
        const trimmed = value.trim();
        return trimmed ? `<div class="tv-text">${escHtml(value)}</div>` : `<div class="tv-empty">${t('experiments.emptyValue')}</div>`;
      }
      if (value === null || value === undefined || (tvIsPlainObject(value) && Object.keys(value).length === 0)) {
        return `<div class="tv-empty">${t('experiments.noValue')}</div>`;
      }
      return tvFormat === 'json' ? tvRenderJson(value, false) : tvRender(value, 0);
    }

    // Collapsed height must match the `.tv-collapsible` max-height in <style>
    // (12rem @ 16px base) — used to decide whether a block needs its toggle.
    const TV_COLLAPSED_MAX_PX = 192;

    // Card bodies are rendered only while unfolded, so a run with hundreds of
    // calls still keeps the feed's DOM small.
    function toggleToolCard(uid, ev) {
      // Keyboard activation mirrors a native button (Enter / Space).
      if (ev && ev.type === 'keydown') {
        if (ev.key !== 'Enter' && ev.key !== ' ') return;
        ev.preventDefault();
      }
      // Finishing a mouse selection of the tool name must not fold the card.
      const sel = window.getSelection && window.getSelection();
      if (ev && ev.type === 'click' && sel && !sel.isCollapsed
          && ev.currentTarget && ev.currentTarget.contains(sel.anchorNode)) return;
      if (tvOpenCards.has(uid)) tvOpenCards.delete(uid);
      else tvOpenCards.add(uid);
      renderExperimentFeed();
    }

    // The key travels URI-encoded in `data-agent-key`: it may hold a NUL
    // (name/instance separator), which an HTML attribute cannot carry.
    function toggleAgentNode(el, ev) {
      if (ev && ev.type === 'keydown') {
        if (ev.key !== 'Enter' && ev.key !== ' ') return;
        ev.preventDefault();
      }
      // Finishing a mouse selection of the agent name must not fold the branch.
      const sel = window.getSelection && window.getSelection();
      if (ev && ev.type === 'click' && sel && !sel.isCollapsed
          && el.contains(sel.anchorNode)) return;
      const key = decodeURIComponent(el.dataset.agentKey || '');
      if (collapsedAgents.has(key)) collapsedAgents.delete(key);
      else collapsedAgents.add(key);
      renderExperimentFeed();
    }

    // Sticky: a later call opens unfolded too while "Expand all" is on.
    function toggleExperimentExpandAll() {
      tvExpandAll = !tvExpandAll;
      tvOpenCards.clear();
      if (tvExpandAll) toolCallRecords.forEach(rec => tvOpenCards.add(rec.uid));
      renderExperimentFeed();
    }

    function setToolsViewerFormat(format) {
      tvFormat = format === 'json' ? 'json' : 'yaml';
      try { localStorage.setItem(TV_FORMAT_KEY, tvFormat); } catch { /* ignore */ }
      renderExperimentFeed();
    }

    // A block's value and truncation flag by its server field name:
    // args, result/error, or one of the extra `rec.inputs` blocks.
    function tvFieldValue(rec, field) {
      if (field === 'args') return rec.args;
      if (field === 'result' || field === 'error') return rec.result;
      return rec.inputs[field] ? rec.inputs[field].value : null;
    }

    function tvFieldTruncated(rec, field) {
      if (field === 'args') return rec.argsTruncated;
      if (field === 'result' || field === 'error') return rec.resultTruncated;
      return !!(rec.inputs[field] && rec.inputs[field].truncated);
    }

    // Replaces a server-truncated value on the record with the full one.
    async function tvFetchFullValue(rec, field) {
      const data = await apiJson(sessionApi('/tool-activity/' + encodeURIComponent(rec.callId)));
      const full = Object.prototype.hasOwnProperty.call(data, field) ? data[field] : null;
      if (field === 'args') {
        rec.args = full;
        rec.argsTruncated = false;
      } else if (field === 'result' || field === 'error') {
        rec.result = field === 'error' ? { error: full } : full;
        rec.resultTruncated = false;
      } else {
        rec.inputs[field] = { value: full, truncated: false };
      }
    }

    // YAML view expands JSON-text strings into trees (see `tvRender`); the
    // copied YAML does the same so it matches what is on screen.
    function tvExpandJsonStrings(value) {
      if (typeof value === 'string') {
        const parsed = tvTryParseJsonString(value);
        return parsed ? tvExpandJsonStrings(parsed.value) : value;
      }
      if (Array.isArray(value)) return value.map(tvExpandJsonStrings);
      if (tvIsPlainObject(value)) {
        const out = {};
        Object.keys(value).forEach(k => { out[k] = tvExpandJsonStrings(value[k]); });
        return out;
      }
      return value;
    }

    // Plain-text form of a value in the current format — what the copy
    // button puts on the clipboard, instead of the tree's rendered DOM text.
    function tvValueText(value) {
      if (value === null || value === undefined) return '';
      if (typeof value === 'string') {
        const parsed = tvTryParseJsonString(value);
        if (!parsed) return value;
        value = parsed.value;
      }
      if (tvFormat === 'yaml' && typeof jsyaml !== 'undefined') {
        return jsyaml.dump(tvExpandJsonStrings(value), { lineWidth: -1, noRefs: true }).replace(/\n$/, '');
      }
      return JSON.stringify(value, null, 2);
    }

    async function tvWriteClipboard(text) {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return;
      }
      // Plain-http deployments have no async clipboard API.
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try {
        if (!document.execCommand('copy')) throw new Error('copy failed');
      } finally {
        ta.remove();
      }
    }

    async function copyTvToolName(btn) {
      const rec = toolCallsById.get(btn.dataset.uid);
      if (!rec) return;
      const icon = btn.querySelector('.material-symbols-outlined');
      try {
        await tvWriteClipboard(rec.name);
        if (icon) icon.textContent = 'check';
      } catch {
        if (icon) icon.textContent = 'error';
      }
      setTimeout(() => { if (icon) icon.textContent = 'content_copy'; }, 1200);
    }

    async function copyTvAgentName(btn) {
      const node = agentNodes.get(decodeURIComponent(btn.dataset.agentKey || ''));
      if (!node) return;
      const icon = btn.querySelector('.material-symbols-outlined');
      try {
        await tvWriteClipboard(tvAgentName(node.name));
        if (icon) icon.textContent = 'check';
      } catch {
        if (icon) icon.textContent = 'error';
      }
      setTimeout(() => { if (icon) icon.textContent = 'content_copy'; }, 1200);
    }

    async function copyTvBlock(btn) {
      const rec = toolCallsById.get(btn.dataset.uid);
      if (!rec) return;
      const field = btn.dataset.field;
      const icon = btn.querySelector('.material-symbols-outlined');
      const flash = (name) => {
        if (!icon) return;
        icon.textContent = name;
        setTimeout(() => { icon.textContent = 'content_copy'; }, 1200);
      };
      btn.disabled = true;
      try {
        if (tvFieldTruncated(rec, field) && rec.callId) {
          await tvFetchFullValue(rec, field);
          const el = document.getElementById(`tv-${rec.uid}-${field}`);
          if (el) el.innerHTML = tvRenderAny(tvFieldValue(rec, field));
        }
        await tvWriteClipboard(tvValueText(tvFieldValue(rec, field)));
        flash('check');
      } catch {
        flash('error');
      } finally {
        btn.disabled = false;
      }
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
      const truncated = rec && tvFieldTruncated(rec, field);

      if (rec && rec.callId && truncated && !el.classList.contains('tv-expanded')) {
        btn.disabled = true;
        btn.textContent = t('common.loading');
        try {
          await tvFetchFullValue(rec, field);
          el.innerHTML = tvRenderAny(tvFieldValue(rec, field));
        } catch (err) {
          btn.disabled = false;
          btn.textContent = t('experiments.retry', { error: (err.message || t('experiments.loadFailed')) });
          return;
        }
        btn.disabled = false;
      }

      const expanded = el.classList.toggle('tv-expanded');
      if (expanded) tvExpandedBlocks.add(blockId); else tvExpandedBlocks.delete(blockId);
      btn.textContent = expanded ? t('common.showLess') : t('common.showMore');
    }

    // Whether a block's toggle button is needed depends on its rendered
    // height, which is only known once the HTML is actually in the DOM — a
    // block still awaiting its server fetch always gets one regardless.
    function initTvToggles(root) {
      root.querySelectorAll('.tv-collapsible').forEach(el => {
        const btn = root.querySelector(`[data-tv-toggle="${el.id}"]`);
        if (!btn) return;
        if (el.classList.contains('tv-expanded')) btn.textContent = t('common.showLess');
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
        ['args', 'result', 'error', 'effective_args', 'state_inputs'].forEach(field => tvExpandedBlocks.delete(`tv-${rec.uid}-${field}`));
        const node = agentNodes.get(rec.authorKey);
        if (node) {
          node.calls = node.calls.filter(call => call !== rec);
          touchedAgents.add(rec.authorKey);
        }
      });
      // A branch with nothing left of its own and no child branch to carry
      // is dead weight — drop it. One still holding a child stays, as a bare
      // header, so that child keeps its place in the tree.
      touchedAgents.forEach(key => {
        const node = agentNodes.get(key);
        if (!node || node.calls.length) return;
        const hasChildren = agentOrder.some(n => agentParent.get(n) === key);
        if (hasChildren) return;
        agentNodes.delete(key);
        agentOrder = agentOrder.filter(n => n !== key);
        agentParent.delete(key);
        agentSpawnCall.delete(key);
        collapsedAgents.delete(key);
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
    // the key the server stores the untruncated value under (args, result,
    // error, effective_args, state_inputs).
    function tvValueBlock(rec, field, label, value, emptyText) {
      const blockId = `tv-${rec.uid}-${field}`;
      const truncated = tvFieldTruncated(rec, field);
      const needsFetch = truncated && !!rec.callId;
      const expandedCls = tvExpandedBlocks.has(blockId) ? ' tv-expanded' : '';
      return `
        <div>
          <div class="flex items-center justify-between mb-1">
            <div class="text-[9px] font-bold text-outline-variant uppercase tracking-widest">${label}</div>
            ${tvHasValue(value) ? `<button type="button" data-uid="${rec.uid}" data-field="${field}" onclick="copyTvBlock(this)"
              title="${t('common.copy')}" class="flex items-center text-outline-variant hover:text-primary transition-colors">
              <span class="material-symbols-outlined text-[14px]">content_copy</span>
            </button>` : ''}
          </div>
          <div id="${blockId}" class="tv-collapsible${expandedCls} bg-surface-container-lowest/80 border border-outline-variant/10 rounded-md p-2.5 text-[11px] text-on-surface-variant font-mono leading-relaxed">
            ${tvHasValue(value) ? tvRenderAny(value) : `<div class="tv-empty">${escHtml(emptyText)}</div>`}
          </div>
          <div class="flex justify-end">
            <button data-tv-toggle="${blockId}" data-uid="${rec.uid}" data-field="${field}"
              data-needs-fetch="${needsFetch ? '1' : ''}" onclick="toggleTvBlock(this)"
              class="${needsFetch ? '' : 'hidden'} mt-1 text-[9px] font-bold text-outline-variant hover:text-primary uppercase tracking-wider">${t('common.showMore')}</button>
          </div>
        </div>`;
    }

    function renderToolCardBody(rec) {
      const args = tvValueBlock(
        rec, 'args', t('experiments.args'), rec.args,
        rec.argsUnknown ? t('experiments.callNotRecorded') : t('experiments.noArgs'),
      );
      // What the tool ran on beyond the model's own arguments.
      const extra = [
        ['effective_args', t('experiments.effectiveArgs')],
        ['state_inputs', t('experiments.stateInputs')],
      ].filter(([field]) => rec.inputs[field])
        .map(([field, label]) => tvValueBlock(rec, field, label, rec.inputs[field].value, t('experiments.noValue')))
        .join('');
      const output = rec.status === 'running'
        ? `<div>
             <div class="text-[9px] font-bold text-outline-variant uppercase tracking-widest mb-1">${t('experiments.output')}</div>
             <div class="bg-surface-container-lowest/80 border border-outline-variant/10 rounded-md p-2.5 text-[11px]">
               <span class="tv-empty">${t('experiments.waiting')}</span>
             </div>
           </div>`
        : tvValueBlock(
          rec,
          rec.status === 'error' ? 'error' : 'result',
          rec.status === 'error' ? t('experiments.error') : t('experiments.output'),
          rec.result,
          t('experiments.emptyValue'),
        );
      return `<div class="px-2.5 pb-2.5 pt-2 space-y-2 border-t border-outline-variant/10">${args}${extra}${output}</div>`;
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
          <div role="button" tabindex="0" aria-expanded="${open}"
            onclick="toggleToolCard('${rec.uid}', event)" onkeydown="toggleToolCard('${rec.uid}', event)"
            class="group w-full flex items-center gap-2 px-2.5 py-1.5 text-left cursor-pointer hover:bg-surface-variant/20 transition-colors">
            <span class="material-symbols-outlined text-[14px] text-outline-variant shrink-0">${open ? 'expand_more' : 'chevron_right'}</span>
            <span class="material-symbols-outlined text-[14px] ${st.tone} shrink-0${running ? ' tv-spin' : ''}">${icon}</span>
            ${rec.isDelegation && rec.targetAgent
              ? `<span class="text-[11px] font-bold text-on-surface shrink-0 select-text cursor-text" title="${escHtml(rec.name)}" translate="no">${escHtml(tvAgentName(rec.targetAgent))}</span>`
              : `<span class="text-[11px] font-bold font-mono text-on-surface shrink-0 select-text cursor-text">${escHtml(rec.name)}</span>`}
            <button type="button" data-uid="${rec.uid}" onclick="event.stopPropagation(); copyTvToolName(this)"
              onkeydown="event.stopPropagation()" title="${t('common.copy')}"
              class="shrink-0 -ml-1 flex items-center text-outline-variant opacity-0 group-hover:opacity-100 focus:opacity-100 hover:text-primary transition-opacity">
              <span class="material-symbols-outlined text-[12px]">content_copy</span>
            </button>
            ${rec.isDelegation ? '<span class="shrink-0 text-[8px] font-bold uppercase tracking-wider text-primary/70">delegates</span>' : ''}
            ${rec.inputs.state_inputs ? `<span class="shrink-0 text-[8px] font-bold uppercase tracking-wider text-tertiary/80" title="${t('experiments.stateBadgeTitle')}">${t('experiments.stateBadge')}</span>` : ''}
            <span class="flex-1 min-w-0 truncate text-[10px] font-mono text-outline-variant/70">${escHtml(summary)}</span>
            <span class="shrink-0 text-[9px] font-mono ${st.tone}">${running ? 'running…' : tvDuration(rec)}</span>
            <span class="shrink-0 text-[9px] font-mono text-outline-variant/70">${meta}</span>
          </div>
          ${open ? renderToolCardBody(rec) : ''}
        </div>`;
    }

    // Agents are shown by their Russian role name — the same table the rail,
    // the chat and the graph use (status_indicator.js); the runtime id stays
    // in the tooltip and in what the copy button copies.
    function tvAgentName(name) {
      return (window.StatusIndicator && StatusIndicator.agentName)
        ? (StatusIndicator.agentName(name) || String(name || ''))
        : String(name || '').replace(/Agent$/, '');
    }

    // A delegated agent's hand-off, shown inside its own block: what it was
    // asked and what it answered, folded like a tool card body.
    function renderSpawnRow(rec) {
      const open = tvOpenCards.has(rec.uid);
      const st = TV_STATUS[rec.status] || TV_STATUS.running;
      return `
        <div class="rounded-md border ${st.border} ${st.bg} overflow-hidden">
          <div role="button" tabindex="0" aria-expanded="${open}"
            onclick="toggleToolCard('${rec.uid}', event)" onkeydown="toggleToolCard('${rec.uid}', event)"
            class="w-full flex items-center gap-2 px-2.5 py-1.5 text-left cursor-pointer hover:bg-surface-variant/20 transition-colors">
            <span class="material-symbols-outlined text-[14px] text-outline-variant shrink-0">${open ? 'expand_more' : 'chevron_right'}</span>
            <span class="material-symbols-outlined text-[14px] text-primary/70 shrink-0">assignment</span>
            <span class="text-[11px] font-bold text-on-surface shrink-0">${t('experiments.delegatedTask')}</span>
            <span class="flex-1 min-w-0 truncate text-[10px] font-mono text-outline-variant/70">${escHtml(tvArgsSummary(rec.args))}</span>
            <span class="shrink-0 text-[9px] font-mono ${st.tone}">${rec.status === 'running' ? 'running…' : tvDuration(rec)}</span>
          </div>
          ${open ? renderToolCardBody(rec) : ''}
        </div>`;
    }

    // One branch of the call tree: the agent's own tool calls, in order. A
    // delegation whose agent is known is shown as that agent's own block, in
    // the delegation's place, with the hand-off (`spawnRec`) inside it — two
    // agents delegated to in parallel therefore stay where they were launched.
    // A delegation whose agent never reported stays a `delegates` card.
    function renderAgentNode(key, visited = new Set(), spawnRec = null) {
      const node = agentNodes.get(key);
      if (!node || isInternalAgent(node.name) || visited.has(key)) return '';
      visited.add(key);

      const calls = node.calls;
      const children = agentOrder.filter(n => {
        const child = agentNodes.get(n);
        return agentParent.get(n) === key && !visited.has(n) && child && !isInternalAgent(child.name);
      });
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
      const failed = calls.filter(rec => rec.status === 'error').length
        + (spawnRec && spawnRec.status === 'error' ? 1 : 0);
      const done = calls.length - running;
      const isAgentRunning = node.status === 'running' || running > 0;
      const pills = [
        running ? `<span class="text-tertiary">${running} running</span>` : (node.status === 'running' ? `<span class="text-tertiary">running</span>` : ''),
        failed ? `<span class="text-error">${failed} failed</span>` : '',
        (!running && calls.length) ? `<span class="text-secondary">${done - failed}/${calls.length} ok</span>` : '',
      ].filter(Boolean).join('<span class="text-outline-variant/30">·</span>');
      // Nothing to unfold — no calls of its own and no visible child — means
      // no chevron and no toggle: the header is just a label.
      const expandable = calls.length > 0 || children.length > 0 || !!spawnRec;
      const collapsed = expandable && collapsedAgents.has(key);
      const lastActive = calls.length
        ? (calls[calls.length - 1].endedAt || calls[calls.length - 1].startedAt)
        : (node.lastActive || node.firstSeenAt);

      const nestBranches = names => names.length ? `
            <div class="ml-3 pl-3 border-l-2 border-outline-variant/15 space-y-1.5">
              ${names.map(childName => renderAgentNode(childName, new Set(visited))).join('')}
            </div>` : '';

      const body = collapsed ? '' : `
        <div class="px-2 pb-2 space-y-1.5">
          ${spawnRec ? renderSpawnRow(spawnRec) : ''}
          ${calls.map(rec => {
            const spawned = childrenByCall.get(rec.uid) || [];
            if (!rec.isDelegation || !spawned.length) return renderToolCard(rec) + nestBranches(spawned);
            return spawned.map((child, i) => renderAgentNode(child, new Set(visited), i === 0 ? rec : null)).join('');
          }).join('')}
          ${nestBranches(tailChildren)}
        </div>`;

      const cleanName = escHtml(tvAgentName(node.name));
      const encodedKey = escHtml(encodeURIComponent(key));

      return `
        <div class="rounded-lg border border-outline-variant/15 bg-surface-container-low/40">
          <div ${expandable ? `role="button" tabindex="0" aria-expanded="${!collapsed}" data-agent-key="${encodedKey}"
            onclick="toggleAgentNode(this, event)" onkeydown="toggleAgentNode(this, event)"` : ''}
            class="group w-full flex items-center gap-2 px-2.5 py-2 text-left ${expandable ? 'cursor-pointer hover:bg-surface-variant/20' : ''} transition-colors">
            ${expandable
              ? `<span class="material-symbols-outlined text-[14px] text-outline-variant shrink-0">${collapsed ? 'chevron_right' : 'expand_more'}</span>`
              : '<span class="w-[14px] shrink-0" aria-hidden="true"></span>'}
            <span class="material-symbols-outlined text-[14px] text-primary shrink-0">${agentIcon(node.name)}</span>
            <span class="text-[11px] font-bold uppercase tracking-wider text-on-surface shrink-0 select-text cursor-text" title="${escHtml(node.name)}" translate="no">${cleanName}</span>
            <button type="button" data-agent-key="${encodedKey}" onclick="event.stopPropagation(); copyTvAgentName(this)"
              onkeydown="event.stopPropagation()" title="${t('common.copy')}"
              class="shrink-0 -ml-1 flex items-center text-outline-variant opacity-0 group-hover:opacity-100 focus:opacity-100 hover:text-primary transition-opacity">
              <span class="material-symbols-outlined text-[12px]">content_copy</span>
            </button>
            ${calls.length ? `<span class="text-[9px] font-mono px-1.5 py-0.5 rounded bg-primary/10 text-primary shrink-0">${calls.length} call${calls.length === 1 ? '' : 's'}</span>` : ''}
            <span class="flex-1"></span>
            <span class="flex items-center gap-1.5 text-[9px] font-mono shrink-0">${pills}</span>
            <span class="shrink-0 text-[9px] font-mono text-outline-variant/60">${safeTimeStr(lastActive)}</span>
          </div>
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
      document.querySelectorAll('[data-tv-format]').forEach(btn => {
        const active = btn.dataset.tvFormat === tvFormat;
        btn.classList.toggle('bg-primary/15', active);
        btn.classList.toggle('text-primary', active);
        btn.classList.toggle('text-outline-variant', !active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
      });

      const roots = agentOrder.filter(key => {
        const node = agentNodes.get(key);
        return !agentParent.has(key) && node && !isInternalAgent(node.name);
      });

      if (toolCallRecords.length === 0 && (roots.length === 0 || agentNodes.size === 0)) {
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
      feed.innerHTML = roots.map(key => renderAgentNode(key)).join('');
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
    window.setToolsViewerFormat = setToolsViewerFormat;
    window.toggleTvBlock = toggleTvBlock;
    window.copyTvBlock = copyTvBlock;
    window.copyTvToolName = copyTvToolName;
    window.copyTvAgentName = copyTvAgentName;
    window.addExperimentAgentEvent = addExperimentAgentEvent;
    window.addExperimentToolCall = addExperimentToolCall;
    window.addExperimentToolResponse = addExperimentToolResponse;
