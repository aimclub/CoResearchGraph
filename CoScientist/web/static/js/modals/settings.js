// =========================================================================
// Settings Modal
// =========================================================================
// The form is rendered from SETTINGS_SECTIONS: every field names the
// appSettings path it edits, when a change takes effect and what it depends
// on. The modal edits a draft copy; nothing reaches the server until Save.

    const SETTINGS_SCOPES = {
      instant: 'bolt',
      session: 'restart_alt',
      browser: 'devices',
      reload: 'refresh',
    };

    const OPENROUTER_POPULAR_PROVIDERS = [
      'DeepInfra', 'Together', 'Fireworks', 'Groq', 'Lepton', 'Novita',
      'Chutes', 'Cerebras', 'SambaNova', 'Nebius', 'Mistral',
    ];

    // The `reasoning:` vocabulary of system.yaml, in the order the modal offers it.
    const REASONING_LEVELS = ['off', 'minimal', 'low', 'medium', 'high'];

    // In "orchestrator plans" mode the PlannerAgent is not built at all, so its
    // own switches have nothing to act on.
    function plannerInactive(d) {
      return getSettingPath(d, 'general.startMode') === 'orchestrator_planner'
        ? { key: 'settings.inactive.noPlanner' } : null;
    }

    const SETTINGS_SECTIONS = [
      {
        id: 'research', icon: 'science',
        groups: [
          {
            fields: [
              {
                id: 'startMode', path: 'general.startMode', type: 'cards', scope: 'session', env: 'START_MODE',
                options: [
                  { value: 'planner', icon: 'account_tree' },
                  { value: 'orchestrator', icon: 'bolt' },
                  { value: 'orchestrator_planner', icon: 'alt_route' },
                ],
              },
              { id: 'contextInit', path: 'general.contextInitEnabled', type: 'toggle', scope: 'session', env: 'RESEARCH_FRAME', envAliases: ['CONTEXT_INIT__ENABLED'] },
              // Reasoning of the agents this switch attaches (the frame and the
              // technical specification), kept in agents.overrides like any
              // other agent's. These agents are not listed under Agents.
              { id: 'contextInitReasoning', type: 'agentReasoning', agentsOf: 'general.contextInitEnabled', scope: 'session', parent: 'contextInit' },
              { id: 'maxHypotheses', path: 'hypothesesAgent.maxActiveHypotheses', type: 'number', min: 1, max: 5, scope: 'session', env: 'HYPOTHESES__MAX_ACTIVE' },
            ],
          },
          {
            heading: 'planning',
            fields: [
              { id: 'critic', path: 'plannerAgent.criticEnabled', type: 'toggle', scope: 'session', env: 'PLANNER__CRITIC_ENABLED', inactive: plannerInactive },
              { id: 'criticRounds', path: 'plannerAgent.criticRounds', type: 'number', min: 1, max: 5, scope: 'session', parent: 'critic', env: 'PLANNER__CRITIC_ROUNDS', inactive: plannerInactive },
              { id: 'mergeTasks', path: 'plannerAgent.mergeTasksEnabled', type: 'toggle', scope: 'instant', env: 'PLANNER__MERGE_TASKS' },
              { id: 'plannerRetrieval', path: 'plannerAgent.retrievalEnabled', type: 'toggle', scope: 'session', env: 'PLANNER__RETRIEVAL_ENABLED', inactive: plannerInactive },
              {
                id: 'plannerGraph', path: 'plannerAgent.graphEnabled', type: 'toggle', scope: 'session', env: 'PLANNER__GRAPH_ENABLED',
                inactive: d => plannerInactive(d) || (getSettingPath(d, 'general.knowledgeGraphEnabled')
                  ? null : { key: 'settings.inactive.knowledgeGraph', section: 'graphs' }),
              },
            ],
          },
          {
            heading: 'modules',
            fields: [
              // A runtime gate: NirReportAgent is attached whenever a normcontrol
              // server is configured, and this decides whether the operator is
              // offered the GOST report at the end of a run. Greyed out where
              // there is no server to submit the document to.
              {
                id: 'nirReport', path: 'nirReport.enabled', type: 'toggle', scope: 'instant', env: 'NIR__ENABLED',
                inactive: d => (getSettingPath(d, 'nirReport.available') ? null : { key: 'settings.inactive.nirUnavailable' }),
              },
              { id: 'nirReportReasoning', type: 'agentReasoning', agentsOf: 'nirReport.enabled', scope: 'session', parent: 'nirReport' },
            ],
          },
        ],
      },
      {
        id: 'approvals', icon: 'verified_user',
        groups: [
          {
            fields: [
              { id: 'hitl', path: 'general.hitlEnabled', type: 'toggle', scope: 'session', env: 'HITL__ENABLED' },
              // The one knob that decides whether this run asks a human and for
              // how long: auto never asks, basic waits ten minutes, debug waits
              // for you. Silence approves in none of them — a run that must
              // proceed unattended says `auto` out loud. Deliberately NOT
              // parented to `hitl`: `auto` is what makes an unattended run
              // possible, and hiding it behind the switch that turns the asking
              // off would hide the only honest way to do that.
              { id: 'hitlMode', path: 'general.hitlMode', type: 'segmented', options: ['auto', 'basic', 'debug'], scope: 'instant', env: 'HITL__MODE',
                inactive: d => getSettingPath(d, 'general.hitlModePinnedByEnv')
                  ? { key: 'settings.inactive.envPinned', vars: { name: 'HITL__MODE' } } : null },
              // Legacy, and superseded by the mode above: it is still read to
              // DERIVE a mode for a stand configured before the mode existed
              // (a non-positive value means "wait for me", i.e. debug).
              { id: 'hitlTimeout', path: 'general.hitlAutoApproveTimeout', type: 'timeout', fallback: 300, scope: 'instant', parent: 'hitl', advanced: true, env: 'HITL_AUTO_APPROVE_TIMEOUT', envAliases: ['HITL__AUTO_APPROVE_TIMEOUT', 'HITL_TIMEOUT_SECONDS'],
                inactive: () => ({ key: 'settings.inactive.supersededByMode' }) },
              { id: 'workOrder', path: 'general.workOrderEnabled', type: 'toggle', scope: 'session', parent: 'hitl', env: 'WORK_ORDER__ENABLED' },
              { id: 'workOrderVeto', path: 'general.workOrderVetoSeconds', type: 'timeout', fallback: 60, scope: 'instant', parent: 'workOrder', env: 'WORK_ORDER__VETO_SECONDS',
                inactive: () => ({ key: 'settings.inactive.supersededByMode' }) },
            ],
          },
          {
            // Deliberately NOT parented to `hitl`: the experiment module asks
            // for these two even when the switch above is off, and a window
            // that runs out pauses the run instead of approving it. Greying
            // them out with the global switch would hide the only way through.
            heading: 'experimentReview',
            fields: [
              { id: 'experimentPlanAuto', path: 'experimentModule.planAutoApprove', type: 'toggle', scope: 'instant', env: 'EXPERIMENTS__PLAN_AUTO_APPROVE', inactive: () => ({ key: 'settings.inactive.supersededByMode' }) },
              {
                id: 'experimentPlanTimeout', path: 'experimentModule.planReviewTimeoutS', type: 'number', min: 30, max: 86400, scope: 'instant', env: 'EXPERIMENTS__PLAN_REVIEW_TIMEOUT_S',
                inactive: () => ({ key: 'settings.inactive.supersededByMode' }),
              },
              { id: 'experimentResultAuto', path: 'experimentModule.resultAutoApprove', type: 'toggle', scope: 'instant', env: 'EXPERIMENTS__RESULT_AUTO_APPROVE', inactive: () => ({ key: 'settings.inactive.supersededByMode' }) },
              {
                id: 'experimentResultTimeout', path: 'experimentModule.resultReviewTimeoutS', type: 'number', min: 30, max: 86400, scope: 'instant', env: 'EXPERIMENTS__RESULT_REVIEW_TIMEOUT_S',
                inactive: () => ({ key: 'settings.inactive.supersededByMode' }),
              },
              { id: 'experimentRouteAlembic', path: 'experimentModule.routeAlembic', type: 'toggle', scope: 'instant', env: 'EXPERIMENTS__ROUTE_ALEMBIC' },
              { id: 'alembicAgentBuild', path: 'alembicHub.agentBuildEnabled', type: 'toggle', scope: 'instant', parent: 'experimentRouteAlembic', env: 'ALEMBIC__AGENT_BUILD_ENABLED' },
              { id: 'alembicHubSearch', path: 'alembicHub.searchEnabled', type: 'toggle', scope: 'instant', parent: 'experimentRouteAlembic', env: 'ALEMBIC_HUB__SEARCH_ENABLED' },
              { id: 'alembicHubAutoUpload', path: 'alembicHub.autoUpload', type: 'toggle', scope: 'instant', parent: 'experimentRouteAlembic', env: 'ALEMBIC_HUB__AUTO_UPLOAD' },
            ],
          },
        ],
      },
      {
        // Per-agent switches over system.yaml; the list is drawn from
        // /api/agents/catalog, the values live in appSettings.agents.
        id: 'agents', icon: 'smart_toy',
        groups: [
          {
            fields: [
              {
                id: 'defaultReasoning', path: 'agents.defaultReasoning', type: 'select', scope: 'session',
                options: ['', ...REASONING_LEVELS], env: 'AGENTS__DEFAULT_REASONING',
              },
            ],
          },
          {
            heading: 'agentList',
            fields: [
              { id: 'agentOverrides', path: 'agents.overrides', type: 'agents', scope: 'session', env: 'AGENTS__OVERRIDES' },
            ],
          },
          {
            // The settings some agents' `enabled` refers to in system.yaml
            // (catalog `enabledSetting`). They have no rows of their own: the
            // agent's switch in the list above edits them, because the runtime
            // reads them too and an override would leave it behind. Listed here
            // for Save, reset, search markers and the .env file.
            hidden: true,
            fields: [
              // Off also takes the medical route out of running experiments.
              { id: 'medicalAgent', path: 'medicalAgent.enabled', type: 'toggle', scope: 'session', scopeHintKey: 'settings.f.medicalAgent.scopeHint', env: 'MEDICAL__ENABLED' },
              // FedotAgent: the reranker fallback in the main profile, the
              // FEDOT.MAS route in the experiments profile. Off also reaches a
              // running session, hence the scope hints.
              { id: 'fedotFallback', path: 'taskExecutorAgent.fedotFallback', type: 'toggle', scope: 'session', scopeHintKey: 'settings.f.fedotFallback.scopeHint', env: 'EXECUTOR__FEDOT_FALLBACK' },
              { id: 'experimentRouteFedot', path: 'experimentModule.routeFedot', type: 'toggle', scope: 'session', scopeHintKey: 'settings.f.experimentRouteFedot.scopeHint', env: 'EXPERIMENTS__ROUTE_FEDOT' },
            ],
          },
        ],
      },
      {
        id: 'models', icon: 'memory',
        groups: [{
          fields: [
            { id: 'providerSort', path: 'general.openrouterProviderSort', type: 'segmented', options: ['default', 'price', 'latency', 'throughput'], scope: 'session', env: 'OPENROUTER_PROVIDER_SORT', envAliases: ['LLM__OPENROUTER_PROVIDER_SORT'] },
            { id: 'providerOrder', path: 'general.openrouterProviderOrder', type: 'chips', suggestions: OPENROUTER_POPULAR_PROVIDERS, scope: 'session', env: 'OPENROUTER_PROVIDER_ORDER', envAliases: ['LLM__OPENROUTER_PROVIDER_ORDER'] },
            { id: 'maxRetries', path: 'general.maxRetries', type: 'number', min: 0, max: 10, scope: 'instant', advanced: true, env: 'LLM_MAX_RETRIES' },
          ],
        }],
      },
      {
        id: 'tools', icon: 'construction',
        groups: [
          {
            heading: 'search',
            fields: [
              { id: 'maxSearches', path: 'researchAgent.maxSearches', type: 'number', min: 0, max: 20, scope: 'session', env: 'RESEARCH_AGENT_SEARCHES' },
            ],
          },
          {
            heading: 'code',
            fields: [
              { id: 'coderMode', path: 'coderAgent.mode', type: 'segmented', options: ['local', 'openhands'], scope: 'session', env: 'CODER__MODE' },
              {
                id: 'sandboxUrl', path: 'coderAgent.sandboxUrl', type: 'text', mono: true, placeholder: 'http://localhost:8884', scope: 'instant', env: 'SANDBOX_URL',
                validate: d => {
                  const url = String(getSettingPath(d, 'coderAgent.sandboxUrl') || '').trim();
                  if (url && !/^https?:\/\/\S+$/i.test(url)) return { key: 'settings.err.url' };
                  if (!url && getSettingPath(d, 'coderAgent.mode') === 'openhands') return { key: 'settings.err.sandboxRequired' };
                  return null;
                },
              },
              { id: 'workspaceId', path: 'coderAgent.workspaceId', type: 'text', mono: true, placeholderKey: 'settings.f.workspaceId.placeholder', scope: 'instant', advanced: true, env: 'CODER_WORKSPACE_ID' },
            ],
          },
          {
            heading: 'toolSelection',
            fields: [
              { id: 'keepScore', path: 'taskExecutorAgent.keepScore', type: 'number', float: true, min: 0, max: 1, step: 0.05, scope: 'instant', advanced: true, env: 'EXECUTOR_TOOL_KEEP_SCORE' },
              {
                id: 'abstainScore', path: 'taskExecutorAgent.abstainScore', type: 'number', float: true, min: 0, max: 1, step: 0.05, scope: 'instant', advanced: true, env: 'EXECUTOR_TOOL_ABSTAIN_SCORE',
                validate: d => {
                  const keep = getSettingPath(d, 'taskExecutorAgent.keepScore');
                  const abstain = getSettingPath(d, 'taskExecutorAgent.abstainScore');
                  if (typeof keep === 'number' && typeof abstain === 'number' && abstain > keep) {
                    return { key: 'settings.err.abstainAboveKeep', vars: { keep } };
                  }
                  return null;
                },
              },
            ],
          },
        ],
      },
      {
        id: 'graphs', icon: 'hub',
        groups: [
          {
            fields: [
              { id: 'knowledgeGraph', path: 'general.knowledgeGraphEnabled', type: 'toggle', scope: 'session', env: 'GRAPH__ENABLED' },
              { id: 'researchGraph', path: 'general.researchGraphEnabled', type: 'toggle', scope: 'session', env: 'RESEARCH_GRAPH__ENABLED' },
            ],
          },
          {
            heading: 'envOnly',
            fields: [
              { id: 'autoClearGraph', path: 'general.autoClearGraphEnabled', type: 'env', env: 'GRAPH__AUTO_CLEAR' },
            ],
          },
          { heading: 'danger', fields: [{ id: 'dangerZone', type: 'danger' }] },
        ],
      },
      {
        id: 'interface', icon: 'palette',
        groups: [{
          fields: [
            { id: 'language', type: 'language', scope: 'browser' },
            { id: 'theme', type: 'theme', scope: 'browser' },
            {
              id: 'lightDim', type: 'lightDim', scope: 'browser',
              inactive: () => currentTheme === 'light' ? null : { key: 'settings.inactive.lightOnly' },
            },
            { id: 'accent', type: 'accent', scope: 'browser' },
            { id: 'font', type: 'font', scope: 'browser' },
            { id: 'showInternal', type: 'browserToggle', scope: 'browser', env: 'SHOW_INTERNAL__ENABLED' },
          ],
        }],
      },
      {
        id: 'system', icon: 'dns',
        groups: [
          {
            fields: [
              { id: 'defaultUsername', path: 'general.coscientistUsername', type: 'text', placeholderKey: 'settings.f.defaultUsername.placeholder', scope: 'reload', env: 'COSCIENTIST_USERNAME', envAliases: ['DEFAULT_USERNAME'] },
              { id: 'opik', path: 'general.opikEnabled', type: 'toggle', scope: 'session', env: 'OPIK__ENABLED' },
              { id: 'autoNaming', path: 'general.autoNamingEnabled', type: 'toggle', scope: 'instant', env: 'AUTO_NAMING__ENABLED' },
            ],
          },
          {
            heading: 'transfer',
            fields: [{ id: 'envTransfer', type: 'envTransfer' }],
          },
          {
            heading: 'envOnly',
            fields: [
              { id: 'useProxy', path: 'general.useProxy', type: 'env', env: 'USE_PROXY' },
            ],
          },
        ],
      },
    ];

    const SETTINGS_FIELDS = SETTINGS_SECTIONS.flatMap(section =>
      section.groups.flatMap(group => group.fields.map(field => ({ ...field, section: section.id, hidden: !!group.hidden }))));
    const SETTINGS_FIELD_BY_ID = Object.fromEntries(SETTINGS_FIELDS.map(f => [f.id, f]));
    // Fields whose value is sent on Save (read-only .env values are not).
    const EDITABLE_TYPES = new Set(['toggle', 'number', 'text', 'cards', 'segmented', 'chips', 'timeout', 'select', 'agents']);
    const TIMEOUT_MAX_SECONDS = 86400;
    // Seconds last typed into a timeout field, restored when "wait" is switched back to auto.
    const settingsLastTimeout = {};
    const EDITABLE_FIELDS = SETTINGS_FIELDS.filter(f => f.path && EDITABLE_TYPES.has(f.type));

    let settingsSaved = null;      // last values confirmed by the server
    let settingsDraft = null;      // what the form currently shows
    let settingsDefaults = null;   // values the server was launched with
    let settingsSection = 'research';
    let settingsQuery = '';
    let settingsLoadFailed = false;
    let settingsStatus = null;     // { key, vars, kind } shown in the footer
    const settingsAdvancedOpen = new Set();
    // pending: null | 'session' | 'memory' (inline confirm) | 'memory-final' (second dialog)
    const settingsDanger = { pending: null, busy: false, message: '', kind: '' };

    // ── helpers ─────────────────────────────────────────────────────────────
    function getSettingPath(obj, path) {
      return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
    }

    function setSettingPath(obj, path, value) {
      const keys = path.split('.');
      const last = keys.pop();
      const target = keys.reduce((o, k) => (o[k] = o[k] || {}), obj);
      target[last] = value;
    }

    function cloneSettings(obj) {
      return JSON.parse(JSON.stringify(obj));
    }

    function tf(key, vars) {
      let text = t(key);
      Object.entries(vars || {}).forEach(([name, value]) => {
        text = text.split(`{${name}}`).join(String(value));
      });
      return text;
    }

    function sameValue(a, b) {
      return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
    }

    function settingsModalOpen() {
      const modal = document.getElementById('settings-modal');
      return !!modal && !modal.classList.contains('hidden');
    }

    function formatSettingValue(field, value) {
      if (field.type === 'toggle' || field.type === 'env') return t(value ? 'settings.value.on' : 'settings.value.off');
      if (field.type === 'timeout') {
        return typeof value === 'number' && value > 0 ? tf('settings.timeout.after', { n: value }) : t('settings.timeout.wait');
      }
      if (field.type === 'select') return t(`settings.reasoning.${value || 'inherit'}`);
      if (field.type === 'agents') return tf('settings.agents.changed', { n: Object.keys(value || {}).length });
      if (value === '' || value == null) return t('settings.value.empty');
      if (field.type === 'cards' || field.type === 'segmented') {
        return t(`settings.f.${field.id}.opt.${value}`, String(value));
      }
      return String(value);
    }

    function fieldError(field, draft) {
      const value = getSettingPath(draft, field.path);
      if (field.type === 'number') {
        const bad = typeof value !== 'number' || Number.isNaN(value)
          || value < field.min || value > field.max
          || (!field.float && !Number.isInteger(value));
        if (bad) {
          return { key: field.float ? 'settings.err.rangeFloat' : 'settings.err.rangeInt', vars: { min: field.min, max: field.max } };
        }
      }
      if (field.type === 'timeout' && value !== -1) {
        if (typeof value !== 'number' || !Number.isInteger(value) || value < 1 || value > TIMEOUT_MAX_SECONDS) {
          return { key: 'settings.err.rangeInt', vars: { min: 1, max: TIMEOUT_MAX_SECONDS } };
        }
      }
      return field.validate ? field.validate(draft) : null;
    }

    // Children are only meaningful while their parent switch (and its parents) is on.
    function parentOff(field, draft) {
      if (!field.parent) return false;
      const parent = SETTINGS_FIELD_BY_ID[field.parent];
      return !getSettingPath(draft, parent.path) || parentOff(parent, draft);
    }

    // The switch that hides this field: the outermost parent that is off.
    function blockingParent(field, draft) {
      let blocker = null;
      for (let f = field; f.parent; ) {
        f = SETTINGS_FIELD_BY_ID[f.parent];
        if (!getSettingPath(draft, f.path)) blocker = f;
      }
      return blocker;
    }

    function dirtyFields() {
      if (!settingsDraft || !settingsSaved) return [];
      return EDITABLE_FIELDS.filter(f => !sameValue(getSettingPath(settingsDraft, f.path), getSettingPath(settingsSaved, f.path)));
    }

    function erroredFields() {
      if (!settingsDraft) return [];
      return EDITABLE_FIELDS.filter(f => !parentOff(f, settingsDraft) && fieldError(f, settingsDraft));
    }

    function fieldMatchesQuery(field, query) {
      const haystack = [
        t(`settings.f.${field.id}.label`), t(`settings.f.${field.id}.desc`), field.env || '',
      ].join(' ').toLowerCase();
      return haystack.includes(query);
    }

    // ── server sync ─────────────────────────────────────────────────────────
    function mergeServerSettings(data) {
      agentsWithCustomModel.clear();
      Object.keys(appSettings).forEach(group => {
        if (!data[group] || typeof data[group] !== 'object') return;
        Object.keys(appSettings[group]).forEach(key => {
          if (data[group][key] !== undefined && data[group][key] !== null) appSettings[group][key] = data[group][key];
        });
      });
      if (data.defaults) settingsDefaults = data.defaults;
      refreshPlanGate();
    }

    async function loadSettings() {
      try {
        const url = (activeUser && activeSession) ? sessionApi('/settings') : '/api/settings';
        const resp = await fetch(url);
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        mergeServerSettings(await resp.json());
        settingsLoadFailed = false;
        if (showInternalStored === null) setShowInternal(appSettings.general.showInternal, { remember: false });
      } catch (e) {
        settingsLoadFailed = true;
        console.warn('Failed to load settings from server:', e);
      }
      if (!activeSandboxWatchUrl) {
        updateCoderSandboxButton(null);
      }
      return !settingsLoadFailed;
    }

    // ── open / close ────────────────────────────────────────────────────────
    async function openSettings(sectionId) {
      const modal = document.getElementById('settings-modal');
      if (sectionId && SETTINGS_SECTIONS.some(s => s.id === sectionId)) settingsSection = sectionId;
      settingsQuery = '';
      document.getElementById('settings-search').value = '';
      settingsDanger.pending = null;
      settingsDanger.message = '';
      settingsStatus = { key: 'settings.status.loading', kind: 'muted' };
      settingsDraft = null;
      modal.classList.remove('hidden');
      applyLanguage();
      renderSettings();

      loadAgentsCatalog();
      await loadSettings();
      settingsSaved = cloneSettings(appSettings);
      settingsDraft = cloneSettings(appSettings);
      settingsStatus = settingsLoadFailed ? { key: 'settings.status.loadFailed', kind: 'error' } : null;
      renderSettings();
    }

    function closeSettings(force = false) {
      if (!force && dirtyFields().length && !confirm(t('settings.confirmDiscard'))) return;
      document.getElementById('settings-modal').classList.add('hidden');
      settingsDraft = null;
    }

    function discardSettingsChanges() {
      if (!settingsSaved) return;
      agentsWithCustomModel.clear();
      settingsDraft = cloneSettings(settingsSaved);
      settingsStatus = null;
      renderSettings();
    }

    function resetSettingsSection(sectionId) {
      // The appearance lives in this browser, not in the server defaults.
      if (sectionId === 'interface') { resetAppearance(); renderSettings(); return; }
      if (!settingsDefaults || !settingsDraft) return;
      if (sectionId === 'agents') agentsWithCustomModel.clear();
      EDITABLE_FIELDS.filter(f => f.section === sectionId).forEach(f => {
        const value = getSettingPath(settingsDefaults, f.path);
        if (value !== undefined) setSettingPath(settingsDraft, f.path, cloneSettings(value));
      });
      const map = cloneSettings(agentOverrides());
      const launched = getSettingPath(settingsDefaults, 'agents.overrides') || {};
      SETTINGS_FIELDS.filter(f => f.section === sectionId && f.type === 'agentReasoning')
        .forEach(f => reasoningAgents(f).forEach(agent => {
          map[agent.name] = { ...(map[agent.name] || {}), reasoning: (launched[agent.name] || {}).reasoning };
        }));
      setSettingPath(settingsDraft, 'agents.overrides', normalizeOverrides(map));
      settingsStatus = null;
      renderSettings();
    }

    function selectSettingsSection(sectionId) {
      settingsSection = sectionId;
      settingsQuery = '';
      document.getElementById('settings-search').value = '';
      renderSettings();
      document.getElementById('settings-body').scrollTop = 0;
    }

    function onSettingsSearch(value) {
      settingsQuery = value.trim().toLowerCase();
      renderSettings();
    }

    document.addEventListener('keydown', (e) => {
      if (e.key !== 'Escape' || !settingsModalOpen()) return;
      if (settingsDanger.pending === 'memory-final') {
        if (!settingsDanger.busy) { settingsDanger.pending = null; renderSettings(); }
        return;
      }
      closeSettings();
    });

    // ── rendering ───────────────────────────────────────────────────────────
    function renderSettings() {
      if (!settingsModalOpen()) return;
      renderSettingsNav();
      const body = document.getElementById('settings-body');
      body.innerHTML = settingsDraft
        ? (settingsQuery ? renderSettingsSearch() : renderSettingsSection(SETTINGS_SECTIONS.find(s => s.id === settingsSection)))
        : `<p class="text-xs text-outline-variant">${escHtml(t('settings.status.loading'))}</p>`;
      decorateSettings();
      renderSettingsDialog();
    }

    function renderSettingsNav() {
      const nav = document.getElementById('settings-nav');
      nav.innerHTML = SETTINGS_SECTIONS.map(section => {
        const active = !settingsQuery && section.id === settingsSection;
        return `
          <button type="button" data-action="section" data-section="${section.id}"
            class="w-full flex items-center gap-2.5 px-3 py-2 rounded-md text-left text-xs transition-colors
              ${active ? 'bg-primary/10 text-primary font-semibold' : 'text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface'}">
            <span class="material-symbols-outlined text-base">${section.icon}</span>
            <span class="flex-1 truncate">${escHtml(t(`settings.section.${section.id}`))}</span>
            <span data-nav-marker="${section.id}" class="w-1.5 h-1.5 rounded-full hidden"></span>
          </button>`;
      }).join('');
    }

    function renderSettingsSection(section) {
      const hasEditable = EDITABLE_FIELDS.some(f => f.section === section.id);
      // A section of browser-only fields: nothing goes through Save, and its
      // reset puts back the appearance at once.
      const browserOnly = section.groups.every(g => g.fields.every(f => f.scope === 'browser'));
      const resetHint = t(browserOnly ? 'settings.resetHint.browser' : 'settings.resetHint');
      const resetBtn = (hasEditable && settingsDefaults) || browserOnly ? `
        <button type="button" data-action="reset-section" data-section="${section.id}" title="${escHtml(resetHint)}"
          class="shrink-0 flex items-center gap-1 px-2.5 py-1.5 rounded-md text-[11px] text-on-surface-variant border border-outline-variant/20 hover:text-on-surface hover:bg-surface-container-high transition-colors">
          <span class="material-symbols-outlined text-sm">settings_backup_restore</span>${escHtml(t('settings.reset'))}
        </button>` : '';
      return `
        <div class="flex items-start justify-between gap-4 mb-1">
          <div>
            <h4 class="font-headline text-base font-bold text-on-surface">${escHtml(t(`settings.section.${section.id}`))}</h4>
            <p class="text-xs text-on-surface-variant/80 mt-0.5">${escHtml(t(`settings.section.${section.id}.desc`))}</p>
          </div>
          ${resetBtn}
        </div>
        <p class="flex items-center gap-1.5 text-[11px] text-outline-variant/80 mb-5">
          <span class="material-symbols-outlined text-sm">info</span>${escHtml(t(browserOnly ? 'settings.banner.browser' : 'settings.banner'))}
        </p>
        <div class="space-y-6">
          ${section.groups.map((group, i) => (group.hidden ? '' : renderSettingsGroup(group, `${section.id}:${i}`))).join('')}
        </div>`;
    }

    function renderSettingsGroup(group, groupKey) {
      const visible = group.fields.filter(f => !parentOff(f, settingsDraft));
      const basic = visible.filter(f => !f.advanced);
      const advanced = visible.filter(f => f.advanced);
      // Keep the advanced block open while one of its fields needs fixing.
      const open = settingsAdvancedOpen.has(groupKey) || advanced.some(f => fieldError(f, settingsDraft));
      const heading = group.heading ? `
        <div class="mb-2">
          <h5 class="text-[11px] font-bold uppercase tracking-widest font-headline ${group.heading === 'danger' ? 'text-error' : 'text-outline-variant'}">
            ${escHtml(t(`settings.group.${group.heading}`))}</h5>
          ${i18n[`settings.group.${group.heading}.desc`] ? `<p class="text-[11px] text-outline-variant/80 mt-0.5">${escHtml(t(`settings.group.${group.heading}.desc`))}</p>` : ''}
        </div>` : '';
      const advancedBlock = advanced.length ? `
        <button type="button" data-action="toggle-advanced" data-group="${groupKey}"
          class="flex items-center gap-1 px-4 py-2.5 w-full text-left text-[11px] font-semibold text-outline-variant hover:text-on-surface transition-colors ${basic.length ? 'border-t border-outline-variant/10' : ''}">
          <span class="material-symbols-outlined text-sm">${open ? 'expand_less' : 'expand_more'}</span>
          ${escHtml(tf('settings.advanced', { n: advanced.length }))}
        </button>
        ${open ? advanced.map(f => renderSettingRow(f)).join('') : ''}` : '';
      if (group.heading === 'danger') {
        return `<section>${heading}${renderDangerZone()}</section>`;
      }
      return `
        <section>
          ${heading}
          <div class="bg-surface-container-lowest rounded-lg border border-outline-variant/10 divide-y divide-outline-variant/10">
            ${basic.map(f => renderSettingRow(f)).join('')}
            ${advancedBlock}
          </div>
        </section>`;
    }

    function renderSettingsSearch() {
      const blocks = SETTINGS_SECTIONS.map(section => {
        // The agent list is searched agent by agent: an agent's switch is its
        // row, whatever setting stands behind it.
        const matches = SETTINGS_FIELDS.filter(f => f.section === section.id && !f.hidden
          && f.type !== 'danger' && f.type !== 'agents' && fieldMatchesQuery(f, settingsQuery));
        const agents = section.id === 'agents' && agentsCatalog
          ? listedAgents().filter(a => (!a.internal || agentsShowInternal) && agentMatchesQuery(a, settingsQuery)) : [];
        if (!matches.length && !agents.length) return '';
        return `
          <section>
            <button type="button" data-action="section" data-section="${section.id}"
              class="mb-2 flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-widest font-headline text-outline-variant hover:text-primary">
              <span class="material-symbols-outlined text-sm">${section.icon}</span>${escHtml(t(`settings.section.${section.id}`))}
            </button>
            <div class="bg-surface-container-lowest rounded-lg border border-outline-variant/10 divide-y divide-outline-variant/10">
              ${matches.map(f => renderSettingRow(f, { search: true })).join('')}
              ${agents.map(renderAgentRow).join('')}
            </div>
          </section>`;
      }).join('');
      return blocks
        ? `<div class="space-y-6">${blocks}</div>`
        : `<p class="text-xs text-outline-variant py-8 text-center">${escHtml(t('settings.noResults'))}</p>`;
    }

    function renderSettingRow(field, opts = {}) {
      let inactive = field.inactive ? field.inactive(settingsDraft) : null;
      if (!inactive && opts.search && parentOff(field, settingsDraft)) {
        const parent = blockingParent(field, settingsDraft);
        inactive = { key: 'settings.inactive.parentOff', vars: { parent: t(`settings.f.${parent.id}.label`) }, section: parent.section };
      }
      const disabled = !!inactive;
      const label = escHtml(t(`settings.f.${field.id}.label`));
      const desc = i18n[`settings.f.${field.id}.desc`] ? t(`settings.f.${field.id}.desc`) : '';
      const wide = ['cards', 'chips', 'agents', 'envTransfer'].includes(field.type);
      // scopeHintKey: a field whose effect does not fit its scope's stock hint.
      const scopeHint = field.scope ? `${t(`settings.scope.${field.scope}`)} — ${t(field.scopeHintKey || `settings.scope.${field.scope}.hint`)}` : '';
      const scope = field.scope ? `
        <span class="material-symbols-outlined text-[14px] text-outline-variant/70 cursor-help"
          role="img" aria-label="${escHtml(scopeHint)}" title="${escHtml(scopeHint)}">${SETTINGS_SCOPES[field.scope]}</span>` : '';
      let envHint = field.env ? tf('settings.envVar', { name: field.env }) : '';
      if (envHint && i18n[`settings.f.${field.id}.envValues`]) {
        envHint += '\n' + t('settings.envValues') + '\n' + t(`settings.f.${field.id}.envValues`);
      }
      const envInfo = field.env && field.type !== 'env' ? `
        <span class="material-symbols-outlined text-[14px] text-outline-variant/60 cursor-help"
          role="img" aria-label="${escHtml(envHint)}" title="${escHtml(envHint)}">info</span>` : '';
      const inactiveNote = inactive ? `
        <p class="text-[11px] text-tertiary/90 mt-1.5 flex items-center gap-1 flex-wrap">
          <span class="material-symbols-outlined text-sm">block</span>${escHtml(tf(inactive.key, inactive.vars))}
          ${inactive.section ? `<button type="button" data-action="section" data-section="${inactive.section}" class="underline hover:text-on-surface">${escHtml(t('settings.goto'))}</button>` : ''}
        </p>` : '';
      const control = renderSettingControl(field, disabled);
      return `
        <div data-row="${field.id}" class="relative px-4 py-3.5 border-l-2 border-l-transparent ${field.parent && !opts.search ? 'pl-8' : ''}">
          <div class="${wide ? 'space-y-3' : 'flex items-start justify-between gap-6'}">
            <div class="min-w-0 ${disabled ? 'opacity-60' : ''}">
              <div class="flex items-center gap-2 flex-wrap">
                <label for="sf-${field.id}" class="text-[13px] font-semibold text-on-surface">${label}</label>
                <span data-default-marker class="hidden w-1.5 h-1.5 rounded-full bg-primary/70"></span>
                ${scope}${envInfo}
              </div>
              ${desc ? `<p class="text-[12px] text-on-surface-variant/75 mt-1 leading-relaxed">${desc}</p>` : ''}
              ${inactiveNote}
              <p data-error class="hidden text-[11px] text-error mt-1.5 flex items-center gap-1"></p>
            </div>
            <div class="${wide ? '' : 'shrink-0 pt-0.5'} ${disabled ? 'opacity-50 pointer-events-none' : ''}">${control}</div>
          </div>
        </div>`;
    }

    // On/off switch; `attrs` are the checkbox's own attributes (id, data-*).
    function renderSwitch(attrs, checked, disabled, extraCls = '') {
      return `
        <label class="relative inline-flex items-center ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'} ${extraCls}">
          <input type="checkbox" ${attrs} class="sr-only peer" ${checked ? 'checked' : ''} ${disabled ? 'disabled' : ''} />
          <span class="w-10 h-6 rounded-full bg-surface-variant border border-outline-variant/30 peer-checked:bg-primary peer-checked:border-primary transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-primary/50"></span>
          <span class="absolute left-1 top-1 w-4 h-4 rounded-full bg-on-surface-variant peer-checked:bg-on-primary peer-checked:translate-x-4 transition-transform"></span>
        </label>`;
    }

    function renderSettingControl(field, disabled) {
      const value = field.path ? getSettingPath(settingsDraft, field.path) : undefined;
      const dis = disabled ? 'disabled' : '';
      const inputCls = 'bg-surface-container-high border border-outline-variant/20 text-on-surface text-xs rounded-md px-3 py-2 focus:ring-1 focus:ring-primary/40 focus:border-primary/40 disabled:cursor-not-allowed';
      switch (field.type) {
        case 'toggle':
          return renderSwitch(`id="sf-${field.id}" data-field="${field.id}"`, value, disabled);
        case 'timeout': {
          const auto = value !== -1;
          return `
            <div class="flex flex-col items-end gap-2">
              <div id="sf-${field.id}" role="radiogroup" class="inline-flex gap-0.5 p-0.5 rounded-md bg-surface-container-high border border-outline-variant/20">
                ${[['wait', !auto], ['auto', auto]].map(([mode, selected]) => `
                  <button type="button" role="radio" aria-checked="${selected}" data-action="timeout-mode" data-field="${field.id}" data-mode="${mode}" ${dis}
                    class="px-3 py-1.5 rounded text-[11px] font-semibold transition-colors ${selected ? 'bg-primary text-on-primary' : 'text-on-surface-variant hover:text-on-surface'}">
                    ${escHtml(t(`settings.timeout.mode.${mode}`))}
                  </button>`).join('')}
              </div>
              ${auto ? `
                <label class="flex items-center gap-2 text-[11px] text-on-surface-variant">
                  ${escHtml(t('settings.timeout.after.label'))}
                  <input type="number" inputmode="numeric" data-field="${field.id}" data-timeout-seconds min="1" max="${TIMEOUT_MAX_SECONDS}" step="1"
                    value="${escHtml(value ?? '')}" ${dis} aria-label="${escHtml(t('settings.timeout.after.label'))}"
                    class="${inputCls} font-mono w-24 text-center" />
                  ${escHtml(t('settings.timeout.seconds'))}
                </label>` : ''}
            </div>`;
        }
        case 'number':
          return `<input id="sf-${field.id}" type="number" inputmode="${field.float ? 'decimal' : 'numeric'}" data-field="${field.id}"
            min="${field.min}" max="${field.max}" step="${field.step || 1}" value="${escHtml(value ?? '')}" ${dis}
            class="${inputCls} font-mono w-24 text-center" />`;
        case 'text':
          return `<input id="sf-${field.id}" type="text" data-field="${field.id}" value="${escHtml(value ?? '')}" ${dis}
            placeholder="${escHtml(field.placeholderKey ? t(field.placeholderKey) : (field.placeholder || ''))}"
            class="${inputCls} ${field.mono ? 'font-mono' : ''} w-64" />`;
        case 'segmented':
          return `
            <div id="sf-${field.id}" role="radiogroup" class="inline-flex flex-wrap gap-0.5 p-0.5 rounded-md bg-surface-container-high border border-outline-variant/20">
              ${field.options.map(opt => `
                <button type="button" role="radio" aria-checked="${value === opt}" data-action="choose" data-field="${field.id}" data-value="${opt}" ${dis}
                  class="px-3 py-1.5 rounded text-[11px] font-semibold transition-colors ${value === opt ? 'bg-primary text-on-primary' : 'text-on-surface-variant hover:text-on-surface'}">
                  ${escHtml(t(`settings.f.${field.id}.opt.${opt}`, opt))}
                </button>`).join('')}
            </div>`;
        case 'cards':
          return `
            <div id="sf-${field.id}" role="radiogroup" class="grid grid-cols-1 sm:grid-cols-3 gap-2">
              ${field.options.map(opt => {
                const selected = value === opt.value;
                return `
                  <button type="button" role="radio" aria-checked="${selected}" data-action="choose" data-field="${field.id}" data-value="${opt.value}" ${dis}
                    class="text-left p-3 rounded-lg border transition-colors ${selected ? 'border-primary bg-primary/10' : 'border-outline-variant/20 hover:border-outline-variant/50 hover:bg-surface-container-high'}">
                    <span class="flex items-center gap-2">
                      <span class="material-symbols-outlined text-base ${selected ? 'text-primary' : 'text-outline-variant'}">${opt.icon}</span>
                      <span class="text-xs font-semibold text-on-surface">${escHtml(t(`settings.f.${field.id}.opt.${opt.value}`))}</span>
                    </span>
                    <span class="block text-[11px] text-on-surface-variant/75 mt-1.5 leading-snug">${escHtml(t(`settings.f.${field.id}.opt.${opt.value}.desc`))}</span>
                    <span class="block text-[10px] font-mono text-outline-variant mt-2">${escHtml(t(`settings.f.${field.id}.opt.${opt.value}.flow`))}</span>
                  </button>`;
              }).join('')}
            </div>`;
        case 'chips': {
          const items = String(value || '').split(',').map(s => s.trim()).filter(Boolean);
          const listId = `sf-${field.id}-suggestions`;
          return `
            <div class="flex flex-wrap items-center gap-1.5 p-1.5 rounded-md bg-surface-container-high border border-outline-variant/20 focus-within:border-primary/40">
              ${items.map((item, i) => `
                <span class="inline-flex items-center gap-1 pl-2 pr-1 py-0.5 rounded bg-primary/10 border border-primary/20 text-[11px] text-on-surface">
                  <span class="text-[10px] text-outline-variant font-mono">${i + 1}</span>${escHtml(item)}
                  <button type="button" data-action="chip-remove" data-field="${field.id}" data-index="${i}" aria-label="remove"
                    class="material-symbols-outlined text-sm text-outline-variant hover:text-error">close</button>
                </span>`).join('')}
              <input id="sf-${field.id}" type="text" data-chips="${field.id}" list="${listId}" ${dis}
                placeholder="${escHtml(t('settings.f.providerOrder.placeholder'))}"
                class="flex-1 min-w-[10rem] bg-transparent border-0 text-xs text-on-surface px-1 py-1 focus:ring-0" />
              <datalist id="${listId}">
                ${field.suggestions.filter(s => !items.includes(s)).map(s => `<option value="${escHtml(s)}"></option>`).join('')}
              </datalist>
            </div>`;
        }
        case 'browserToggle':
          return renderSwitch(`id="sf-${field.id}" data-browser-toggle="${field.id}"`, showInternal, false);
        case 'language': {
          // The one language control sets the interface and the report. The
          // server rejects a change while a run is active; lock the radio too.
          const locked = typeof runActive !== 'undefined' && runActive;
          return `
            <div role="radiogroup" class="inline-flex gap-0.5 p-0.5 rounded-md bg-surface-container-high border border-outline-variant/20 ${locked ? 'opacity-40' : ''}">
              ${['ru', 'en'].map(lang => `
                <button type="button" role="radio" aria-checked="${currentLang === lang}" data-action="language" data-lang="${lang}"
                  ${locked ? 'disabled' : ''}
                  class="px-3 py-1.5 rounded text-[11px] font-semibold transition-colors ${currentLang === lang ? 'bg-primary text-on-primary' : 'text-on-surface-variant hover:text-on-surface'} disabled:cursor-not-allowed">
                  ${lang === 'ru' ? 'Русский' : 'English'}
                </button>`).join('')}
            </div>`;
        }
        case 'theme':
          return `
            <div role="radiogroup" class="inline-flex gap-0.5 p-0.5 rounded-md bg-surface-container-high border border-outline-variant/20">
              ${['dark', 'light'].map(theme => `
                <button type="button" role="radio" aria-checked="${currentTheme === theme}" data-action="theme" data-theme="${theme}"
                  class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-[11px] font-semibold transition-colors ${currentTheme === theme ? 'bg-primary text-on-primary' : 'text-on-surface-variant hover:text-on-surface'}">
                  <span class="material-symbols-outlined text-sm">${theme === 'dark' ? 'dark_mode' : 'light_mode'}</span>${escHtml(t(`settings.f.theme.opt.${theme}`))}
                </button>`).join('')}
            </div>`;
        case 'lightDim':
          // 0 = brightest, 100 = most muted; shown as brightness, so reversed.
          return `
            <div class="flex items-center gap-3">
              <span class="material-symbols-outlined text-sm text-outline-variant">brightness_low</span>
              <input id="sf-${field.id}" type="range" min="0" max="100" step="5" value="${100 - currentLightDim}" data-appearance="lightDim" ${dis}
                class="w-36 accent-primary cursor-pointer disabled:cursor-not-allowed" />
              <span class="material-symbols-outlined text-sm text-outline-variant">brightness_high</span>
              <span data-light-dim-value class="w-9 text-right text-[11px] font-mono text-on-surface-variant">${100 - currentLightDim}%</span>
            </div>`;
        case 'accent': {
          // "Theme" swatch = no override: each theme keeps its own cyan.
          const swatch = (value, color, label) => `
            <button type="button" role="radio" aria-checked="${currentAccent === value}" aria-label="${escHtml(label)}" title="${escHtml(label)}"
              data-action="accent" data-accent="${value || ''}"
              class="w-6 h-6 rounded-full border border-outline-variant/30 transition-shadow ${currentAccent === value ? 'ring-2 ring-offset-2 ring-offset-surface-container-lowest ring-on-surface' : 'hover:ring-2 hover:ring-outline-variant/40'}"
              style="background:${color}"></button>`;
          const custom = currentAccent && !ACCENT_PRESETS.includes(currentAccent);
          return `
            <div role="radiogroup" class="flex items-center gap-2 flex-wrap justify-end">
              ${swatch(null, `rgb(${currentTheme === 'light' ? '0 200 212' : '0 218 243'})`, t('settings.f.accent.default'))}
              ${ACCENT_PRESETS.map(hex => swatch(hex, hex, hex)).join('')}
              <label title="${escHtml(t('settings.f.accent.custom'))}"
                class="relative w-6 h-6 rounded-full cursor-pointer border border-outline-variant/30 flex items-center justify-center overflow-hidden ${custom ? 'ring-2 ring-offset-2 ring-offset-surface-container-lowest ring-on-surface' : ''}"
                style="background:${custom ? currentAccent : 'conic-gradient(#f43f5e, #f59e0b, #10b981, #06b6d4, #3b82f6, #8b5cf6, #f43f5e)'}">
                <input id="sf-${field.id}" type="color" data-appearance="accent" value="${currentAccent || '#00D9E5'}"
                  aria-label="${escHtml(t('settings.f.accent.custom'))}" class="absolute inset-0 opacity-0 cursor-pointer" />
              </label>
            </div>`;
        }
        case 'font':
          return `
            <select id="sf-${field.id}" data-appearance="font" class="${inputCls} min-w-[11rem]">
              ${Object.entries(UI_FONTS).map(([id, font]) => `
                <option value="${id}" ${currentFont === id ? 'selected' : ''} style="font-family:${escHtml(font.stack)}">
                  ${escHtml(id === 'system' ? t('settings.f.font.system') : font.name)}${id === DEFAULT_FONT ? ` · ${escHtml(t('settings.f.font.default'))}` : ''}
                </option>`).join('')}
            </select>`;
        case 'select':
          return `
            <select id="sf-${field.id}" data-field="${field.id}" ${dis} class="${inputCls} min-w-[11rem]">
              ${field.options.map(opt => `
                <option value="${escHtml(opt)}" ${value === opt ? 'selected' : ''}>${escHtml(opt
                  ? t(`settings.reasoning.${opt}`)
                  : tf('settings.agents.inheritWith', { value: t(`settings.reasoning.${(agentsCatalog && agentsCatalog.defaults.reasoning) || 'inherit'}`) }))}</option>`).join('')}
            </select>`;
        case 'agents':
          return renderAgentsControl();
        case 'agentReasoning':
          return renderAgentReasoningControl(field, disabled);
        case 'envTransfer':
          return renderEnvTransfer();
        case 'env':
          return `
            <div class="text-right">
              <span class="inline-block px-2 py-1 rounded text-[11px] font-semibold ${value ? 'bg-secondary/10 text-secondary' : 'bg-surface-container-high text-outline-variant'}">
                ${escHtml(formatSettingValue(field, value))}</span>
              <p class="text-[10px] font-mono text-outline-variant mt-1">${escHtml(field.env)}</p>
            </div>`;
        default:
          return '';
      }
    }

    // Markers that change while typing — updated in place so inputs keep focus.
    function decorateSettings() {
      const body = document.getElementById('settings-body');
      if (settingsDraft) {
        body.querySelectorAll('[data-row]').forEach(row => {
          const field = SETTINGS_FIELD_BY_ID[row.dataset.row];
          if (field.type === 'agentReasoning') {
            const dirty = agentReasoningDirty(field);
            row.classList.toggle('border-l-primary', dirty);
            row.classList.toggle('bg-primary/[0.03]', dirty);
            return;
          }
          if (!field.path || !EDITABLE_TYPES.has(field.type)) return;
          const draftValue = getSettingPath(settingsDraft, field.path);
          const dirty = !sameValue(draftValue, getSettingPath(settingsSaved, field.path));
          row.classList.toggle('border-l-primary', dirty);
          row.classList.toggle('bg-primary/[0.03]', dirty);

          const marker = row.querySelector('[data-default-marker]');
          const defaultValue = settingsDefaults ? getSettingPath(settingsDefaults, field.path) : undefined;
          const differs = defaultValue !== undefined && !sameValue(draftValue, defaultValue);
          marker.classList.toggle('hidden', !differs);
          marker.title = differs ? tf('settings.differsDefault', { value: formatSettingValue(field, defaultValue) }) : '';

          const errorEl = row.querySelector('[data-error]');
          const error = fieldError(field, settingsDraft);
          errorEl.classList.toggle('hidden', !error);
          errorEl.innerHTML = error ? `<span class="material-symbols-outlined text-sm">error</span>${escHtml(tf(error.key, error.vars))}` : '';
        });
      }

      const dirty = dirtyFields();
      const errors = erroredFields();
      SETTINGS_SECTIONS.forEach(section => {
        const marker = document.querySelector(`[data-nav-marker="${section.id}"]`);
        if (!marker) return;
        const hasError = errors.some(f => f.section === section.id);
        const hasDirty = dirty.some(f => f.section === section.id && !(f.type === 'agents' && agentsCatalog))
          || (section.id === 'agents' && agentsCatalog && listedAgentsDirty())
          || SETTINGS_FIELDS.some(f => f.section === section.id && f.type === 'agentReasoning' && agentReasoningDirty(f));
        marker.classList.toggle('hidden', !hasError && !hasDirty);
        marker.classList.toggle('bg-error', hasError);
        marker.classList.toggle('bg-primary', !hasError && hasDirty);
      });

      const status = document.getElementById('settings-status');
      let message;
      let kind;
      // A failed load or save outranks the counters: the draft is still dirty
      // after a failed save, and the count alone would hide why.
      if (settingsStatus && settingsStatus.kind === 'error') {
        message = tf(settingsStatus.key, settingsStatus.vars);
        kind = 'error';
      } else if (errors.length) {
        message = tf('settings.status.errors', { n: errors.length });
        kind = 'error';
      } else if (dirty.length) {
        message = tf('settings.status.dirty', { n: dirty.length });
        kind = 'primary';
      } else if (settingsStatus) {
        message = tf(settingsStatus.key, settingsStatus.vars);
        kind = settingsStatus.kind;
      } else {
        message = t('settings.status.clean');
        kind = 'muted';
      }
      status.textContent = message;
      status.className = 'text-[11px] flex-1 min-w-0 truncate ' + ({
        error: 'text-error', primary: 'text-primary', success: 'text-secondary', muted: 'text-outline-variant',
      }[kind] || 'text-outline-variant');

      document.getElementById('settings-save-btn').disabled = !settingsDraft || !dirty.length || errors.length > 0;
      document.getElementById('settings-discard-btn').disabled = !dirty.length;
    }

    // ── editing ─────────────────────────────────────────────────────────────
    function updateSettingDraft(field, value, rerender) {
      setSettingPath(settingsDraft, field.path, value);
      settingsStatus = null;
      if (rerender) renderSettings();
      else decorateSettings();
    }

    function parseSettingNumber(raw) {
      if (String(raw).trim() === '') return '';
      const num = Number(raw);
      return Number.isNaN(num) ? String(raw) : num;
    }

    function addProviderChip(field, input) {
      const name = input.value.replace(/,/g, ' ').trim();
      if (!name) return;
      const items = String(getSettingPath(settingsDraft, field.path) || '').split(',').map(s => s.trim()).filter(Boolean);
      if (!items.some(item => item.toLowerCase() === name.toLowerCase())) items.push(name);
      updateSettingDraft(field, items.join(', '), true);
      document.getElementById(`sf-${field.id}`)?.focus();
    }

    function initSettingsEvents() {
      const modal = document.getElementById('settings-modal');
      if (!modal || modal.dataset.bound) return;
      modal.dataset.bound = '1';
      const body = document.getElementById('settings-body');

      modal.addEventListener('click', (e) => {
        if (e.target === modal) { closeSettings(); return; }
        const btn = e.target.closest('[data-action]');
        if (!btn || btn.disabled) return;
        const field = SETTINGS_FIELD_BY_ID[btn.dataset.field];
        switch (btn.dataset.action) {
          case 'section': selectSettingsSection(btn.dataset.section); break;
          case 'reset-section': resetSettingsSection(btn.dataset.section); break;
          case 'toggle-advanced': {
            const key = btn.dataset.group;
            if (settingsAdvancedOpen.has(key)) settingsAdvancedOpen.delete(key); else settingsAdvancedOpen.add(key);
            renderSettings();
            break;
          }
          case 'choose': updateSettingDraft(field, btn.dataset.value, true); break;
          case 'timeout-mode': {
            const current = getSettingPath(settingsDraft, field.path);
            if (btn.dataset.mode === 'wait') {
              if (current !== -1) settingsLastTimeout[field.id] = current;
              updateSettingDraft(field, -1, true);
            } else if (current === -1) {
              updateSettingDraft(field, settingsLastTimeout[field.id] ?? field.fallback, true);
              body.querySelector(`[data-field="${field.id}"][data-timeout-seconds]`)?.focus();
            }
            break;
          }
          case 'chip-remove': {
            const items = String(getSettingPath(settingsDraft, field.path) || '').split(',').map(s => s.trim()).filter(Boolean);
            items.splice(Number(btn.dataset.index), 1);
            updateSettingDraft(field, items.join(', '), true);
            break;
          }
          case 'language': applyLanguage(btn.dataset.lang); break;
          case 'theme': setTheme(btn.dataset.theme); renderSettings(); break;
          case 'accent': setAccent(btn.dataset.accent || null); renderSettings(); break;
          case 'danger-ask': settingsDanger.pending = btn.dataset.target; settingsDanger.message = ''; renderSettings(); break;
          case 'danger-cancel': settingsDanger.pending = null; renderSettings(); break;
          case 'danger-confirm':
            if (settingsDanger.pending === 'memory') { settingsDanger.pending = 'memory-final'; renderSettings(); }
            else deleteGraphData(settingsDanger.pending);
            break;
          case 'danger-final': deleteGraphData('memory'); break;
          case 'agent-reset': resetAgentOverride(btn.dataset.agent); break;
          case 'env-export': exportSettingsEnv(); break;
          case 'env-import': body.querySelector('[data-env-import]')?.click(); break;
          case 'env-report-close': envImportReport = null; renderSettings(); break;
        }
      });

      body.addEventListener('change', (e) => {
        // Browser-only switches apply at once, like the language: no draft, no Save.
        if (e.target.dataset.browserToggle === 'showInternal') {
          setShowInternal(e.target.checked);
          return;
        }
        // The colour picker previews on `input`; redraw once it is closed.
        if (e.target.dataset.appearance === 'accent') { setAccent(e.target.value); renderSettings(); return; }
        if (e.target.dataset.appearance === 'font') { setFont(e.target.value); return; }
        const el = e.target;
        if (el.dataset.agentEnabled) { setAgentOverride(el.dataset.agentEnabled, 'enabled', el.checked); return; }
        if (el.dataset.agentReasoning) { setAgentOverride(el.dataset.agentReasoning, 'reasoning', el.value); return; }
        if (el.dataset.agentModelChoice) {
          const name = el.dataset.agentModelChoice;
          if (el.value === '__custom__') {
            agentsWithCustomModel.add(name);
            refreshAgentRows();
          } else {
            agentsWithCustomModel.delete(name);
            setAgentOverride(name, 'model', el.value);
          }
          return;
        }
        if (el.dataset.agentModel) { setAgentOverride(el.dataset.agentModel, 'model', el.value.trim()); return; }
        if (el.dataset.agentLimit) {
          const limit = (catalogAgent(el.dataset.agentLimit) || {}).toolLimit;
          const raw = el.value.trim();
          const value = raw === '' || !limit ? '' : Math.min(limit.max, Math.max(limit.min, Math.round(Number(raw))));
          setAgentOverride(el.dataset.agentLimit, 'limit', Number.isFinite(value) ? value : '');
          return;
        }
        if (el.hasAttribute('data-agents-internal')) { agentsShowInternal = el.checked; renderAgentsList(); return; }
        if (el.hasAttribute('data-env-import')) { importSettingsEnv(el.files && el.files[0]); el.value = ''; return; }
        const field = SETTINGS_FIELD_BY_ID[el.dataset.field];
        if (field && field.type === 'toggle') updateSettingDraft(field, el.checked, true);
        // Re-rendered: the agent rows name the inherited level next to their own.
        if (field && field.type === 'select') updateSettingDraft(field, el.value, true);
      });

      body.addEventListener('input', (e) => {
        const el = e.target;
        if (el.dataset.appearance === 'accent') { setAccent(el.value); return; }
        if (el.dataset.appearance === 'lightDim') {
          setLightDim(100 - Number(el.value));
          el.parentElement.querySelector('[data-light-dim-value]').textContent = `${el.value}%`;
          return;
        }
        if (el.hasAttribute('data-agents-filter')) { agentsFilter = el.value; renderAgentsList(); return; }
        const field = SETTINGS_FIELD_BY_ID[el.dataset.field];
        if (!field) return;
        if (field.type === 'number' || field.type === 'timeout') updateSettingDraft(field, parseSettingNumber(el.value), false);
        else if (field.type === 'text') updateSettingDraft(field, el.value, false);
      });

      body.addEventListener('keydown', (e) => {
        const field = SETTINGS_FIELD_BY_ID[e.target.dataset.chips];
        if (!field) return;
        if (e.key === 'Enter' || e.key === ',') {
          e.preventDefault();
          addProviderChip(field, e.target);
        } else if (e.key === 'Backspace' && !e.target.value) {
          const items = String(getSettingPath(settingsDraft, field.path) || '').split(',').map(s => s.trim()).filter(Boolean);
          if (!items.length) return;
          items.pop();
          updateSettingDraft(field, items.join(', '), true);
          document.getElementById(`sf-${field.id}`)?.focus();
        }
      });

      // Picking a suggestion from the datalist fires no keydown — add it on blur.
      body.addEventListener('focusout', (e) => {
        const field = SETTINGS_FIELD_BY_ID[e.target.dataset.chips];
        if (field && e.target.value.trim()) addProviderChip(field, e.target);
      });
    }

    // ── agents ──────────────────────────────────────────────────────────────
    // The catalog is what system.yaml declares (GET /api/agents/catalog); the
    // draft holds only what the operator changed on top of it. A value equal
    // to the declared one is dropped, so the stored map stays the diff.
    let agentsCatalog = null;
    let agentsCatalogError = '';
    let agentsFilter = '';
    let agentsShowInternal = false;
    const agentsWithCustomModel = new Set();

    async function loadAgentsCatalog() {
      try {
        const url = (activeUser && activeSession) ? sessionApi('/agents/catalog') : '/api/agents/catalog';
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(await fetchErrorMessage(resp));
        agentsCatalog = await resp.json();
        agentsCatalogError = '';
      } catch (e) {
        agentsCatalogError = e.message || String(e);
      }
      renderSettings();
    }

    function agentOverrides() {
      return (settingsDraft && settingsDraft.agents && settingsDraft.agents.overrides) || {};
    }

    function catalogAgent(name) {
      return agentsCatalog ? agentsCatalog.agents.find(a => a.name === name) : null;
    }

    // Sorted keys and no empty entries, so "changed" compares by content.
    function normalizeOverrides(map) {
      const out = {};
      Object.keys(map).sort().forEach(name => {
        const entry = {};
        ['enabled', 'reasoning', 'model', 'limit'].forEach(key => {
          const value = map[name] ? map[name][key] : undefined;
          if (value !== undefined && value !== null && value !== '') entry[key] = value;
        });
        if (Object.keys(entry).length) out[name] = entry;
      });
      return out;
    }

    // The rows live in the Agents section or in search results; redraw
    // whichever is on screen.
    function refreshAgentRows() {
      if (!settingsQuery && document.getElementById('settings-agents-list')) renderAgentsList();
      else renderSettings();
    }

    // Agents configured in another section (an agentReasoning field names the
    // setting that switches them): not repeated in the Agents list.
    const AGENTS_SHOWN_ELSEWHERE = new Set(
      SETTINGS_FIELDS.filter(f => f.type === 'agentReasoning').map(f => f.agentsOf));

    function listedAgents() {
      return agentsCatalog
        ? agentsCatalog.agents.filter(a => !AGENTS_SHOWN_ELSEWHERE.has(a.enabledSetting)) : [];
    }

    // The agents with a model of their own that an agentReasoning field tunes.
    function reasoningAgents(field) {
      return agentsCatalog
        ? agentsCatalog.agents.filter(a => a.enabledSetting === field.agentsOf && a.hasModel) : [];
    }

    // agents.overrides is one map; the Agents marker counts only its own rows.
    function listedAgentsDirty() {
      const saved = (settingsSaved && getSettingPath(settingsSaved, 'agents.overrides')) || {};
      return listedAgents().some(agent => !sameValue(saved[agent.name], agentOverrides()[agent.name]));
    }

    function agentReasoningDirty(field) {
      const saved = (settingsSaved && getSettingPath(settingsSaved, 'agents.overrides')) || {};
      return reasoningAgents(field).some(agent =>
        ((saved[agent.name] || {}).reasoning || '') !== ((agentOverrides()[agent.name] || {}).reasoning || ''));
    }

    // The reasoning <select> of one agent: "as in the profile (level)" first.
    function renderReasoningSelect(agent, cls, disabled) {
      const value = (agentOverrides()[agent.name] || {}).reasoning || '';
      const inherited = agent.reasoning
        || getSettingPath(settingsDraft, 'agents.defaultReasoning') || agentsCatalog.defaults.reasoning;
      return `
        <select data-agent-reasoning="${escHtml(agent.name)}" ${disabled ? 'disabled' : ''}
          aria-label="${escHtml(`${t('settings.agents.reasoning')}: ${agentTitle(agent.name)}`)}" class="${cls}">
          <option value="">${escHtml(tf('settings.agents.inheritWith', { value: t(`settings.reasoning.${inherited || 'inherit'}`) }))}</option>
          ${REASONING_LEVELS.map(level => `<option value="${level}" ${value === level ? 'selected' : ''}>${escHtml(t(`settings.reasoning.${level}`))}</option>`).join('')}
        </select>`;
    }

    function renderAgentReasoningControl(field, disabled) {
      if (!agentsCatalog) {
        return `<span class="text-[11px] ${agentsCatalogError ? 'text-error' : 'text-outline-variant'}">${escHtml(agentsCatalogError
          ? tf('settings.agents.loadFailed', { error: agentsCatalogError })
          : t('settings.agents.loading'))}</span>`;
      }
      const agents = reasoningAgents(field);
      const cls = 'bg-surface-container-high border border-outline-variant/20 text-on-surface text-xs rounded-md pl-2.5 pr-8 py-1.5 min-w-[11rem] focus:ring-1 focus:ring-primary/40 disabled:cursor-not-allowed';
      // Named per agent only when there is more than one to tell apart.
      return `
        <div class="flex flex-col items-end gap-2">
          ${agents.map(agent => `
            <label class="flex items-center gap-2 text-[11px] text-on-surface-variant">
              ${agents.length > 1 ? escHtml(agentTitle(agent.name)) : ''}
              ${renderReasoningSelect(agent, cls, disabled)}
            </label>`).join('')}
        </div>`;
    }

    // The form field an agent's switch edits when system.yaml ties its
    // `enabled` to a setting (catalog `enabledSetting`), else null.
    function agentSettingField(agent) {
      return agent && agent.enabledSetting
        ? SETTINGS_FIELDS.find(f => f.path === agent.enabledSetting) || null : null;
    }

    // The override keys that count for this agent: `enabled` is dead for an
    // agent the backend will not override (locked, or switched by a setting).
    function liveOverride(agent) {
      const override = { ...(agentOverrides()[agent.name] || {}) };
      if (agent.lock || agentSettingField(agent)) delete override.enabled;
      return override;
    }

    function setAgentOverride(name, key, value) {
      const agent = catalogAgent(name);
      const map = cloneSettings(agentOverrides());
      map[name] = { ...(map[name] || {}) };
      const field = key === 'enabled' ? agentSettingField(agent) : null;
      if (field) {
        delete map[name].enabled;
        setSettingPath(settingsDraft, field.path, value);
      } else {
        // Equal to the declared value is not an override: drop it.
        const declared = !agent ? undefined : key === 'limit' ? agentLimitDefault(agent) : agent[key];
        if (value === '' || value == null || value === declared) delete map[name][key];
        else map[name][key] = value;
      }
      updateSettingDraft(SETTINGS_FIELD_BY_ID.agentOverrides, normalizeOverrides(map), false);
      refreshAgentRows();
    }

    function resetAgentOverride(name) {
      agentsWithCustomModel.delete(name);
      const map = cloneSettings(agentOverrides());
      delete map[name];
      const field = agentSettingField(catalogAgent(name));
      const launched = field && settingsDefaults ? getSettingPath(settingsDefaults, field.path) : undefined;
      if (launched !== undefined) setSettingPath(settingsDraft, field.path, launched);
      updateSettingDraft(SETTINGS_FIELD_BY_ID.agentOverrides, normalizeOverrides(map), false);
      refreshAgentRows();
    }

    // The budget an agent's limiter has without an override: the profile's,
    // or the draft value of the setting it follows (the global search cap).
    function agentLimitDefault(agent) {
      const limit = agent && agent.toolLimit;
      if (!limit) return undefined;
      const fromSetting = limit.setting ? getSettingPath(settingsDraft, limit.setting) : undefined;
      return typeof fromSetting === 'number' ? fromSetting : limit.default;
    }

    function effectiveEnabled(agent) {
      if (agent.lock === 'startMode') {
        const mode = getSettingPath(settingsDraft, 'general.startMode');
        if (agent.name === 'PlannerAgent') {
          return mode !== 'orchestrator_planner' && mode !== 'orchestrator_plan';
        }
        if (agent.name === 'PlanningPipelineAgent' || agent.name === 'InitAgent') {
          return mode === 'planner' || mode === 'init';
        }
      }
      if (agent.lock) return agent.enabled;
      const field = agentSettingField(agent);
      if (field) return !!getSettingPath(settingsDraft, field.path);
      const override = agentOverrides()[agent.name];
      return override && typeof override.enabled === 'boolean' ? override.enabled : agent.enabled;
    }

    // Changed against the profile: an override, or a setting-backed switch
    // away from the value the server was launched with.
    function agentChanged(agent) {
      if (Object.keys(liveOverride(agent)).length) return true;
      const field = agentSettingField(agent);
      if (!field || agent.lock || !settingsDefaults) return false;
      const launched = getSettingPath(settingsDefaults, field.path);
      return launched !== undefined && !sameValue(getSettingPath(settingsDraft, field.path), launched);
    }

    function agentMatchesQuery(agent, query) {
      const field = agentSettingField(agent);
      return !query || [
        agent.name, agentTitle(agent.name), agent.description,
        field ? t(`settings.f.${field.id}.label`) : '', field ? field.env : '',
      ].join(' ').toLowerCase().includes(query);
    }

    // Agents that stop being called when `name` is off: its subordinates (and
    // theirs) that no other enabled agent reaches.
    function agentsCutOffBy(name) {
      if (!agentsCatalog) return [];
      const byName = Object.fromEntries(agentsCatalog.agents.map(a => [a.name, a]));
      const reachable = new Set();
      const walk = n => {
        if (reachable.has(n) || n === name || !byName[n] || !effectiveEnabled(byName[n])) return;
        reachable.add(n);
        byName[n].subordinates.forEach(walk);
      };
      agentsCatalog.agents.filter(a => a.root || a.stage).forEach(a => walk(a.name));
      const lost = [];
      const collect = n => (byName[n] ? byName[n].subordinates : []).forEach(child => {
        if (reachable.has(child) || lost.includes(child) || !byName[child] || byName[child].internal) return;
        lost.push(child);
        collect(child);
      });
      collect(name);
      return lost;
    }

    function renderAgentsControl() {
      if (!agentsCatalog) {
        return `<p class="text-[12px] ${agentsCatalogError ? 'text-error' : 'text-outline-variant'}">${escHtml(agentsCatalogError
          ? tf('settings.agents.loadFailed', { error: agentsCatalogError })
          : t('settings.agents.loading'))}</p>`;
      }
      const internalCount = listedAgents().filter(a => a.internal).length;
      return `
        <div class="space-y-3">
          <div class="flex flex-wrap items-center gap-3">
            <input type="search" data-agents-filter value="${escHtml(agentsFilter)}" autocomplete="off" spellcheck="false"
              placeholder="${escHtml(t('settings.agents.filter'))}" aria-label="${escHtml(t('settings.agents.filter'))}"
              class="bg-surface-container-high border border-outline-variant/20 text-on-surface text-xs rounded-md px-3 py-2 w-64 focus:ring-1 focus:ring-primary/40 focus:border-primary/40" />
            <label class="flex items-center gap-2 text-[11px] text-on-surface-variant cursor-pointer select-none">
              <input type="checkbox" data-agents-internal ${agentsShowInternal ? 'checked' : ''}
                class="rounded border-outline-variant/40 bg-surface-container-high text-primary focus:ring-primary/40" />
              ${escHtml(tf('settings.agents.showInternal', { n: internalCount }))}
            </label>
            <span class="flex-1"></span>
            <span data-agents-changed class="text-[11px] text-outline-variant tabular-nums">${escHtml(agentsChangedText())}</span>
          </div>
          <div id="settings-agents-list" class="rounded-lg border border-outline-variant/15 divide-y divide-outline-variant/10">${renderAgentRows()}</div>
        </div>`;
    }

    function agentsChangedText() {
      const n = agentsCatalog ? listedAgents().filter(agentChanged).length : Object.keys(agentOverrides()).length;
      return n ? tf('settings.agents.changed', { n }) : '';
    }

    // Redraws the rows only, so the filter box keeps focus while typing.
    function renderAgentsList() {
      const list = document.getElementById('settings-agents-list');
      if (list) list.innerHTML = renderAgentRows();
      const changed = document.querySelector('[data-agents-changed]');
      if (changed) changed.textContent = agentsChangedText();
    }

    function renderAgentRows() {
      const query = agentsFilter.trim().toLowerCase();
      const rows = listedAgents().filter(agent => (!agent.internal || agentsShowInternal) && agentMatchesQuery(agent, query));
      if (!rows.length) {
        return `<p class="px-4 py-6 text-center text-[12px] text-outline-variant">${escHtml(t('settings.agents.none'))}</p>`;
      }
      return rows.map(renderAgentRow).join('');
    }

    // Agents are named by their Russian role (the StatusIndicator table); the
    // runtime id stays next to it, since overrides are keyed by it.
    function agentTitle(name) {
      return (window.StatusIndicator && StatusIndicator.agentName)
        ? (StatusIndicator.agentName(name) || name) : name;
    }

    function renderAgentRow(agent) {
      const override = liveOverride(agent);
      const changed = agentChanged(agent);
      const enabled = effectiveEnabled(agent);
      const settingField = agentSettingField(agent);
      const locked = !!agent.lock;
      const id = `sf-agent-${agent.name}`;
      const badge = (text, tone) =>
        `<span class="px-1.5 rounded border text-[10px] ${tone || 'text-outline-variant border-outline-variant/25'}">${escHtml(text)}</span>`;
      const badges = [
        agent.root ? badge(t('settings.agents.badge.root'), 'text-primary border-primary/30') : '',
        agent.stage ? badge(t(`settings.agents.badge.${agent.stage}`)) : '',
        agent.internal ? badge(t('settings.agents.badge.internal')) : '',
      ].join('');
      const calledBy = agent.parents.length
        ? `<span class="text-[10px] text-outline-variant">${escHtml(tf('settings.agents.calledBy', { names: agent.parents.map(agentTitle).join(', ') }))}</span>` : '';

      const notes = [];
      if (locked) {
        notes.push(`<span class="material-symbols-outlined text-sm" aria-hidden="true">lock</span>${escHtml(agent.controlReason || t(`settings.agents.lock.${agent.lock}`))}`
          + (agent.lock === 'startMode'
            ? ` <button type="button" data-action="section" data-section="research" class="underline hover:text-on-surface">${escHtml(t('settings.goto'))}</button>`
            : ''));
      } else if (settingField) {
        // One setting may switch several agents (the research frame: the frame
        // and the technical specification); say so, and when it takes effect.
        const peers = agentsCatalog.agents.filter(a => a !== agent && a.enabledSetting === agent.enabledSetting);
        notes.push(escHtml(tf('settings.agents.enabledSetting', { env: settingField.env })));
        if (peers.length) notes.push(escHtml(tf('settings.agents.sharedWith', { names: peers.map(a => agentTitle(a.name)).join(', ') })));
        if (settingField.scopeHintKey) notes.push(escHtml(t(settingField.scopeHintKey)));
      }
      if (locked && agent.enabledRef && agent.lock === 'setting') {
        notes.push(escHtml(tf('settings.agents.enabledRef', { ref: agent.enabledRef })));
      }
      if (!enabled && !locked) {
        const lost = agentsCutOffBy(agent.name);
        if (lost.length) notes.push(escHtml(tf('settings.agents.cascade', { names: lost.map(agentTitle).join(', ') })));
      }

      const modelValue = override.model || '';
      const aliasSelected = Object.prototype.hasOwnProperty.call(agentsCatalog.models, modelValue);
      const customModel = (!!modelValue && !aliasSelected) || agentsWithCustomModel.has(agent.name);
      const modelChoice = customModel ? '__custom__' : modelValue;
      const inheritedModel = agentsCatalog.models[agent.model] || agent.model;
      const resolved = modelValue
        ? (agentsCatalog.models[modelValue] || modelValue)
        : inheritedModel;
      const modelSelect = `
        <select data-agent-model-choice="${escHtml(agent.name)}"
          aria-label="${escHtml(`${t('settings.agents.model')}: ${agentTitle(agent.name)}`)}"
          class="w-full bg-surface-container-high border border-outline-variant/20 text-on-surface text-xs rounded-md pl-2.5 pr-8 py-1.5 focus:ring-1 focus:ring-primary/40">
          <option value="" ${modelChoice === '' ? 'selected' : ''}>${escHtml(tf('settings.agents.inheritWith', { value: `${agent.model} — ${inheritedModel}` }))}</option>
          ${Object.entries(agentsCatalog.models).map(([alias, model]) =>
            `<option value="${escHtml(alias)}" ${modelChoice === alias ? 'selected' : ''}>${escHtml(`${alias} — ${model || ''}`)}</option>`).join('')}
          <option value="__custom__" ${customModel ? 'selected' : ''}>${escHtml(t('settings.agents.modelCustom'))}</option>
        </select>`;
      const customModelInput = !customModel ? '' : `
        <input type="text" data-agent-model="${escHtml(agent.name)}" value="${escHtml(modelValue)}"
          autocomplete="off" spellcheck="false" placeholder="provider/model"
          aria-label="${escHtml(t('settings.agents.modelCustom'))}"
          class="w-full bg-surface-container-high border border-outline-variant/20 text-on-surface text-xs font-mono rounded-md px-2.5 py-1.5 focus:ring-1 focus:ring-primary/40" />`;
      // The budget of the agent's limiter callback: search calls, or calls of
      // each tool. Empty = the profile's (or the global search cap).
      const limit = agent.toolLimit;
      const limitControl = !limit ? '' : `
          <label class="flex flex-col gap-1 min-w-0 text-[10px] text-outline-variant" title="${escHtml(t(`settings.agents.limit.${limit.kind}.hint`))}">
            ${escHtml(t(`settings.agents.limit.${limit.kind}`))}
            <input type="number" inputmode="numeric" min="${limit.min}" max="${limit.max}" step="1"
              data-agent-limit="${escHtml(agent.name)}" value="${override.limit ?? ''}"
              placeholder="${escHtml(tf(limit.setting ? 'settings.agents.limitPlaceholderSetting' : 'settings.agents.limitPlaceholder', { n: agentLimitDefault(agent) }))}"
              class="w-full bg-surface-container-high border border-outline-variant/20 text-on-surface text-xs rounded-md px-2.5 py-1.5 tabular-nums focus:ring-1 focus:ring-primary/40" />
          </label>`;
      // Label over control: reasoning, then the model with the litellm string
      // an alias resolves to under it, then the tool budget where there is one.
      const columns = (agent.hasModel ? 2 : 0) + (limit ? 1 : 0);
      const modelControls = !columns ? '' : `
        <div class="mt-2.5 pl-[52px] grid grid-cols-1 ${columns === 3 ? 'sm:grid-cols-3' : 'sm:grid-cols-2'} gap-3 ${enabled ? '' : 'opacity-50'}">
          ${!agent.hasModel ? limitControl : `
          <label class="flex flex-col gap-1 min-w-0 text-[10px] text-outline-variant">
            ${escHtml(t('settings.agents.reasoning'))}
            ${renderReasoningSelect(agent, 'w-full bg-surface-container-high border border-outline-variant/20 text-on-surface text-xs rounded-md pl-2.5 pr-8 py-1.5 focus:ring-1 focus:ring-primary/40', false)}
          </label>
          <label class="flex flex-col gap-1 min-w-0 text-[10px] text-outline-variant">
            ${escHtml(t('settings.agents.model'))}
            ${modelSelect}
            ${customModelInput}
            ${resolved ? `<span class="font-mono truncate" title="${escHtml(resolved)}" translate="no">→ ${escHtml(resolved)}</span>` : ''}
            <span>${escHtml(t('settings.agents.modelScope'))}</span>
          </label>
          ${limitControl}`}
        </div>`;

      return `
        <div data-agent-row="${escHtml(agent.name)}" class="px-4 py-3 border-l-2 ${changed ? 'border-l-primary bg-primary/[0.03]' : 'border-l-transparent'}">
          <div class="flex items-start gap-3">
            ${renderSwitch(`id="${escHtml(id)}" data-agent-enabled="${escHtml(agent.name)}"
              aria-label="${escHtml(tf('settings.agents.toggle', { name: agentTitle(agent.name) }))}"`, enabled, locked, `shrink-0 mt-0.5 ${locked ? 'opacity-50' : ''}`)}
            <div class="min-w-0 flex-1 ${enabled ? '' : 'opacity-70'}">
              <div class="flex items-center gap-2 flex-wrap">
                <label for="${escHtml(id)}" class="text-[12px] font-medium text-on-surface">${escHtml(agentTitle(agent.name))}</label>
                <span class="font-mono text-[10px] text-outline-variant" translate="no">${escHtml(agent.name)}</span>
                ${badges}${calledBy}
              </div>
              ${agent.description ? `<p class="text-[11px] text-on-surface-variant/80 mt-0.5 leading-snug line-clamp-2" title="${escHtml(agent.description)}">${escHtml(agent.description)}</p>` : ''}
              ${notes.map(note => `<p class="text-[11px] text-tertiary/90 mt-1 flex items-center gap-1 flex-wrap">${note}</p>`).join('')}
            </div>
            ${changed ? `
              <button type="button" data-action="agent-reset" data-agent="${escHtml(agent.name)}" title="${escHtml(t('settings.agents.resetHint'))}"
                class="shrink-0 flex items-center gap-1 px-2 py-1 rounded-md text-[11px] text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high transition-colors">
                <span class="material-symbols-outlined text-sm" aria-hidden="true">undo</span>${escHtml(t('settings.agents.reset'))}
              </button>` : ''}
          </div>
          ${modelControls}
        </div>`;
    }

    // ── export / import (.env) ──────────────────────────────────────────────
    // The fields that carry an `env` name are the file's vocabulary. Export
    // writes what the form shows (unsaved changes included); import fills the
    // form and stops there, so every change is visible before Save.
    let envImportReport = null;

    function transferFields() {
      return SETTINGS_FIELDS.filter(f => f.path && f.env);
    }

    function envFieldIndex() {
      const index = {};
      transferFields().forEach(f => [f.env, ...(f.envAliases || [])].forEach(name => { index[name] = f; }));
      return index;
    }

    function envQuote(value) {
      const text = String(value);
      if (/^[A-Za-z0-9_.:\/@+,=-]*$/.test(text)) return text;
      return `"${text.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n')}"`;
    }

    function envSerialize(field, value) {
      // Single quotes: python-dotenv takes the JSON literally.
      if (field.type === 'agents') return `'${JSON.stringify(normalizeOverrides(value || {}))}'`;
      if (typeof value === 'boolean') return value ? 'true' : 'false';
      if (value == null) return '';
      return envQuote(value);
    }

    function buildEnvExport() {
      const lines = [
        t('settings.transfer.header1'),
        tf('settings.transfer.header2', { date: new Date().toLocaleString(currentLang === 'ru' ? 'ru-RU' : 'en-GB') }),
        t('settings.transfer.header3'),
      ];
      SETTINGS_SECTIONS.forEach(section => {
        const fields = transferFields().filter(f => f.section === section.id);
        if (!fields.length) return;
        lines.push('', `# ${t(`settings.section.${section.id}`)}`);
        fields.forEach(f => {
          lines.push(`# ${t(`settings.f.${f.id}.label`)}`);
          lines.push(`${f.env}=${envSerialize(f, getSettingPath(settingsDraft, f.path))}`);
        });
      });
      return lines.join('\n') + '\n';
    }

    function exportSettingsEnv() {
      if (!settingsDraft) return;
      const stamp = new Date().toISOString().slice(0, 16).replace(/[-:]/g, '').replace('T', '-');
      const name = `coscientist-settings-${stamp}.env`;
      const link = document.createElement('a');
      link.href = URL.createObjectURL(new Blob([buildEnvExport()], { type: 'text/plain;charset=utf-8' }));
      link.download = name;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
      settingsStatus = { key: 'settings.transfer.exported', vars: { name }, kind: 'success' };
      decorateSettings();
    }

    // KEY=value lines the way python-dotenv reads them: an optional `export `,
    // single quotes literal, double quotes with backslash escapes, `#` comments.
    function parseEnvText(text) {
      const entries = [];
      String(text).split(/\r?\n/).forEach(line => {
        const m = /^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/.exec(line);
        if (!m) return;
        const raw = m[2];
        let value;
        if (raw.startsWith("'")) {
          const end = raw.indexOf("'", 1);
          value = end > 0 ? raw.slice(1, end) : raw.slice(1);
        } else if (raw.startsWith('"')) {
          value = '';
          for (let i = 1; i < raw.length; i++) {
            const ch = raw[i];
            if (ch === '\\' && i + 1 < raw.length) {
              const next = raw[++i];
              value += next === 'n' ? '\n' : next;
            } else if (ch === '"') {
              break;
            } else {
              value += ch;
            }
          }
        } else {
          value = raw.replace(/\s+#.*$/, '').trim();
        }
        entries.push([m[1], value]);
      });
      return entries;
    }

    function envValueFor(field, raw) {
      const text = String(raw).trim();
      switch (field.type) {
        case 'toggle':
          if (/^(true|1|yes|on)$/i.test(text)) return { value: true };
          if (/^(false|0|no|off)$/i.test(text)) return { value: false };
          return { error: t('settings.transfer.err.bool') };
        case 'number':
        case 'timeout': {
          const num = Number(text);
          if (text === '' || Number.isNaN(num)) return { error: t('settings.transfer.err.number') };
          return { value: field.type === 'timeout' ? Math.trunc(num) : num };
        }
        case 'segmented':
        case 'cards': {
          const options = field.options.map(o => (typeof o === 'object' ? o.value : o));
          const match = options.find(o => o.toLowerCase() === text.toLowerCase());
          return match ? { value: match } : { error: tf('settings.transfer.err.option', { options: options.join(', ') }) };
        }
        case 'select': {
          const match = field.options.find(o => o === text.toLowerCase());
          return match !== undefined
            ? { value: match }
            : { error: tf('settings.transfer.err.option', { options: field.options.filter(Boolean).join(', ') }) };
        }
        case 'agents':
          return envAgentsValue(text);
        default:
          return { value: text };
      }
    }

    // AGENTS__OVERRIDES: a JSON object; unknown agents and levels are reported,
    // not stored, so a file from another profile cannot slip them through.
    function envAgentsValue(text) {
      let parsed = null;
      try { parsed = JSON.parse(text || '{}'); } catch (_) { /* reported below */ }
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return { error: t('settings.transfer.err.json') };
      const clean = {};
      const warnings = [];
      Object.entries(parsed).forEach(([name, entry]) => {
        if (agentsCatalog && !catalogAgent(name)) {
          warnings.push(tf('settings.transfer.err.unknownAgent', { name }));
          return;
        }
        if (!entry || typeof entry !== 'object') return;
        const out = {};
        if (typeof entry.enabled === 'boolean') out.enabled = entry.enabled;
        if (entry.reasoning != null && entry.reasoning !== '') {
          const level = String(entry.reasoning).toLowerCase();
          if (REASONING_LEVELS.includes(level)) out.reasoning = level;
          else warnings.push(`${name}: ${tf('settings.transfer.err.option', { options: REASONING_LEVELS.join(', ') })}`);
        }
        if (entry.model) out.model = String(entry.model).trim();
        clean[name] = out;
      });
      return { value: normalizeOverrides(clean), warnings };
    }

    async function importSettingsEnv(file) {
      if (!file || !settingsDraft) return;
      let text;
      try {
        text = await file.text();
      } catch (e) {
        envImportReport = { readError: e.message || String(e) };
        renderSettings();
        return;
      }
      const index = envFieldIndex();
      const report = { file: file.name, applied: [], skipped: [], unknown: [], errors: [] };
      parseEnvText(text).forEach(([name, raw]) => {
        // The one per-agent knob that predates the Agents section.
        if (name === 'HYPOTHESES__REASONING') {
          const level = String(raw).trim().toLowerCase();
          if (!REASONING_LEVELS.includes(level)) {
            report.errors.push({ name, message: tf('settings.transfer.err.option', { options: REASONING_LEVELS.join(', ') }) });
            return;
          }
          const map = cloneSettings(agentOverrides());
          map.HypothesesAgent = { ...(map.HypothesesAgent || {}), reasoning: level };
          setSettingPath(settingsDraft, 'agents.overrides', normalizeOverrides(map));
          report.applied.push(name);
          return;
        }
        const field = index[name];
        if (!field) { report.unknown.push(name); return; }
        if (!EDITABLE_TYPES.has(field.type)) { report.skipped.push(name); return; }
        const result = envValueFor(field, raw);
        if (result.error) { report.errors.push({ name, message: result.error }); return; }
        (result.warnings || []).forEach(message => report.errors.push({ name, message }));
        setSettingPath(settingsDraft, field.path, result.value);
        report.applied.push(name);
      });
      envImportReport = report;
      settingsStatus = report.applied.length
        ? { key: 'settings.transfer.imported', vars: { n: report.applied.length }, kind: 'primary' }
        : null;
      renderSettings();
    }

    function renderEnvReport(report) {
      const lines = [];
      if (report.readError) {
        lines.push(`<p class="text-error">${escHtml(tf('settings.transfer.report.readFailed', { error: report.readError }))}</p>`);
      } else {
        if (!report.applied.length && !report.errors.length) {
          lines.push(`<p>${escHtml(t('settings.transfer.report.nothing'))}</p>`);
        }
        if (report.applied.length) {
          lines.push(`<p class="text-on-surface">${escHtml(tf('settings.transfer.report.applied', { n: report.applied.length }))}</p>`);
        }
        if (report.errors.length) {
          lines.push(`<p class="text-error">${escHtml(t('settings.transfer.report.errors'))}</p>`
            + `<ul class="list-disc pl-5 text-error">${report.errors.map(e =>
              `<li><code class="font-mono" translate="no">${escHtml(e.name)}</code>: ${escHtml(e.message)}</li>`).join('')}</ul>`);
        }
        if (report.skipped.length) {
          lines.push(`<p>${escHtml(tf('settings.transfer.report.skipped', { names: report.skipped.join(', ') }))}</p>`);
        }
        if (report.unknown.length) {
          lines.push(`<p>${escHtml(tf('settings.transfer.report.unknown', { names: report.unknown.join(', ') }))}</p>`);
        }
      }
      return `
        <div class="mt-3 p-3 rounded-md border border-outline-variant/20 bg-surface-container-high/40 text-[11px] text-on-surface-variant space-y-1.5" role="status" aria-live="polite">
          <div class="flex items-start justify-between gap-3">
            <p class="font-semibold text-on-surface">${escHtml(report.file ? tf('settings.transfer.report.title', { file: report.file }) : t('settings.transfer.import'))}</p>
            <button type="button" data-action="env-report-close" aria-label="${escHtml(t('settings.close'))}"
              class="material-symbols-outlined text-base text-outline-variant hover:text-on-surface">close</button>
          </div>
          ${lines.join('')}
        </div>`;
    }

    function renderEnvTransfer() {
      const btn = 'inline-flex items-center gap-1.5 px-3 py-2 rounded-md text-[11px] font-semibold border border-outline-variant/25 text-on-surface hover:bg-surface-container-high transition-colors';
      return `
        <div>
          <div class="flex flex-wrap gap-2">
            <button type="button" data-action="env-export" class="${btn}">
              <span class="material-symbols-outlined text-sm" aria-hidden="true">download</span>${escHtml(t('settings.transfer.export'))}
            </button>
            <button type="button" data-action="env-import" class="${btn}">
              <span class="material-symbols-outlined text-sm" aria-hidden="true">upload</span>${escHtml(t('settings.transfer.import'))}
            </button>
            <input type="file" data-env-import accept=".env,.txt,text/plain" class="hidden" />
          </div>
          ${envImportReport ? renderEnvReport(envImportReport) : ''}
        </div>`;
    }

    // ── saving ──────────────────────────────────────────────────────────────
    async function saveSettings() {
      if (!settingsDraft || erroredFields().length) return;
      const changed = dirtyFields();
      if (!changed.length) return;

      const payload = {};
      EDITABLE_FIELDS.forEach(f => {
        let value = getSettingPath(settingsDraft, f.path);
        if (typeof value === 'string') value = value.trim();
        setSettingPath(payload, f.path, value);
      });

      const saveBtn = document.getElementById('settings-save-btn');
      saveBtn.disabled = true;
      const status = document.getElementById('settings-status');
      status.textContent = t('settings.status.saving');
      status.className = 'text-[11px] flex-1 min-w-0 truncate text-primary/70 animate-pulse';

      try {
        const url = (activeUser && activeSession) ? sessionApi('/settings') : '/api/settings';
        const resp = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!resp.ok) throw new Error(await fetchErrorMessage(resp));
        mergeServerSettings(await resp.json());
        settingsSaved = cloneSettings(appSettings);
        settingsDraft = cloneSettings(appSettings);
        addTelemetry('SETTINGS :: saved ' + changed.map(f => f.id).join(', '));
        if (!activeSandboxWatchUrl) updateCoderSandboxButton(null);

        let key = 'settings.status.saved';
        if (changed.some(f => f.scope === 'session')) key = 'settings.status.savedSession';
        else if (changed.some(f => f.scope === 'reload')) key = 'settings.status.savedReload';
        settingsStatus = { key, kind: 'success' };
      } catch (e) {
        settingsStatus = { key: 'settings.status.saveFailed', vars: { error: e.message || e }, kind: 'error' };
      }
      renderSettings();
    }

    // ── danger zone ─────────────────────────────────────────────────────────
    function renderDangerZone() {
      const session = activeSession ? (activeSession.title || activeSession.id) : '';
      const noSession = !activeUser || !activeSession;
      const btnCls = 'shrink-0 px-3 py-2 rounded-md text-[11px] font-bold uppercase tracking-wider border border-error/30 text-error hover:bg-error/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-1.5';
      const row = (target, icon) => {
        const pending = settingsDanger.pending === target || (target === 'memory' && settingsDanger.pending === 'memory-final');
        let confirmBlock = '';
        if (pending) {
          confirmBlock = `
            <div class="mt-3 p-3 rounded-md bg-error-container/20 border border-error/20 space-y-2">
              <p class="text-[12px] text-on-surface">${escHtml(tf(`settings.danger.${target}.confirm`, { name: session }))}</p>
              <div class="flex flex-wrap items-center gap-2">
                <button type="button" id="settings-danger-confirm" data-action="danger-confirm" ${settingsDanger.busy ? 'disabled' : ''}
                  class="px-3 py-1.5 rounded-md text-[11px] font-bold bg-error text-on-error hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed">
                  ${escHtml(t('settings.danger.confirmBtn'))}</button>
                <button type="button" data-action="danger-cancel" ${settingsDanger.busy ? 'disabled' : ''}
                  class="px-3 py-1.5 rounded-md text-[11px] text-on-surface-variant hover:text-on-surface">${escHtml(t('settings.danger.cancel'))}</button>
              </div>
            </div>`;
        }
        return `
          <div class="px-4 py-3.5">
            <div class="flex items-start justify-between gap-6">
              <div class="min-w-0">
                <p class="text-[13px] font-semibold text-on-surface">${escHtml(t(`settings.danger.${target}.label`))}</p>
                <p class="text-[12px] text-on-surface-variant/75 mt-1 leading-relaxed">${escHtml(t(`settings.danger.${target}.desc`))}</p>
                ${noSession ? `<p class="text-[11px] text-tertiary/90 mt-1.5">${escHtml(t('settings.danger.noSession'))}</p>` : ''}
              </div>
              <button type="button" data-action="danger-ask" data-target="${target}" ${noSession || pending || settingsDanger.busy ? 'disabled' : ''} class="${btnCls}">
                <span class="material-symbols-outlined text-sm">${icon}</span>${escHtml(t(`settings.danger.${target}.btn`))}
              </button>
            </div>
            ${confirmBlock}
          </div>`;
      };
      const message = settingsDanger.message ? `
        <p class="px-4 py-2.5 text-[11px] font-mono ${settingsDanger.kind === 'error' ? 'text-error' : settingsDanger.kind === 'busy' ? 'text-primary/70 animate-pulse' : 'text-secondary'}">
          ${escHtml(settingsDanger.message)}</p>` : '';
      return `
        <div class="rounded-lg border border-error/25 divide-y divide-error/15 bg-surface-container-lowest">
          ${row('session', 'layers_clear')}
          ${row('memory', 'delete_forever')}
          ${message}
        </div>`;
    }

    // The global memory takes a second, separate "are you sure" window: it is
    // gone for every session at once, so one misclick must not be enough.
    function renderSettingsDialog() {
      const dialog = document.getElementById('settings-dialog');
      const open = settingsModalOpen() && settingsDanger.pending === 'memory-final';
      dialog.classList.toggle('hidden', !open);
      if (!open) { dialog.innerHTML = ''; return; }
      dialog.innerHTML = `
        <div role="alertdialog" aria-modal="true" aria-labelledby="settings-dialog-title" aria-describedby="settings-dialog-text"
          class="w-full max-w-md mx-4 rounded-xl border border-error/30 bg-surface-container shadow-2xl p-5">
          <div class="flex items-start gap-3">
            <span class="material-symbols-outlined text-error text-2xl">warning</span>
            <div class="min-w-0">
              <h4 id="settings-dialog-title" class="font-headline text-sm font-bold text-on-surface">${escHtml(t('settings.danger.memory.finalTitle'))}</h4>
              <p id="settings-dialog-text" class="text-xs text-on-surface-variant mt-2 leading-relaxed">${escHtml(t('settings.danger.memory.finalText'))}</p>
            </div>
          </div>
          <div class="flex justify-end gap-2 mt-5">
            <button type="button" data-action="danger-cancel" ${settingsDanger.busy ? 'disabled' : ''}
              class="px-4 py-2 rounded-md text-[11px] font-bold bg-surface-container-high border border-outline-variant/20 text-on-surface hover:bg-surface-variant/40">
              ${escHtml(t('settings.danger.cancel'))}</button>
            <button type="button" id="settings-dialog-confirm" data-action="danger-final" ${settingsDanger.busy ? 'disabled' : ''}
              class="px-4 py-2 rounded-md text-[11px] font-bold bg-error text-on-error hover:brightness-110 disabled:opacity-40">
              ${escHtml(t('settings.danger.memory.finalBtn'))}</button>
          </div>
        </div>`;
      // Safe default: Enter/Space on the focused button cancels.
      dialog.querySelector('[data-action="danger-cancel"]').focus();
    }

    function describeGraphDeletion(deleted) {
      return Object.entries(deleted || {}).map(([name, info]) => {
        if (info && info.error) return `${name} failed (${info.error})`;
        if (name === 'memory') {
          return `${name} (${info.entities || 0} entities, ${info.relations || 0} relations)`;
        }
        return name;
      }).join(', ');
    }

    // "session" wipes both per-session graphs; "memory" is the installation-wide
    // knowledge memory. Each view is its own DELETE so a failure names which one.
    async function deleteGraphData(target) {
      const views = target === 'memory' ? ['memory'] : ['execution', 'research'];
      settingsDanger.busy = true;
      settingsDanger.message = t('settings.danger.deleting');
      settingsDanger.kind = 'busy';
      renderSettings();
      const done = [];
      try {
        for (const view of views) {
          const response = await fetch(sessionApi(`/graph?view=${encodeURIComponent(view)}`), { method: 'DELETE' });
          let data = {};
          try { data = await response.json(); } catch (_) { /* empty response */ }
          if (!response.ok) {
            throw new Error(data.detail || describeGraphDeletion(data.deleted) || `HTTP ${response.status}`);
          }
          const described = describeGraphDeletion(data.deleted);
          if (described) done.push(described);
        }
        settingsDanger.message = tf('settings.danger.done', { what: done.join(', ') || t('settings.danger.nothing') });
        settingsDanger.kind = 'success';
        addTelemetry(`GRAPHS :: deleted ${views.join(', ')}`);
      } catch (error) {
        const prefix = done.length ? tf('settings.danger.done', { what: done.join(', ') }) + ' ' : '';
        settingsDanger.message = prefix + tf('settings.danger.failed', { error: error.message || error });
        settingsDanger.kind = 'error';
      } finally {
        settingsDanger.busy = false;
        settingsDanger.pending = null;
        renderSettings();
      }
    }

    initSettingsEvents();
