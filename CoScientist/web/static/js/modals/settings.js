// =========================================================================
// Settings Modal
// =========================================================================
    // =========================================================================
    // Settings Modal
    // =========================================================================
    async function loadSettings() {
      try {
        const resp = await fetch('/api/settings');
        if (resp.ok) {
          const data = await resp.json();
          if (data.general) {
            appSettings.general.startMode = data.general.startMode || 'planner';
            appSettings.general.maxRetries = data.general.maxRetries ?? 3;
            appSettings.general.hitlEnabled = data.general.hitlEnabled ?? false;
            appSettings.general.hitlAutoApproveTimeout = data.general.hitlAutoApproveTimeout ?? 300;
            appSettings.general.usePlanner = data.general.usePlanner ?? true;
            appSettings.general.useProxy = data.general.useProxy ?? appSettings.general.useProxy;
            appSettings.general.opikEnabled = data.general.opikEnabled ?? false;
            appSettings.general.autoNamingEnabled = data.general.autoNamingEnabled ?? true;
            appSettings.general.coscientistUsername = data.general.coscientistUsername || '';
            appSettings.general.contextInitEnabled = data.general.contextInitEnabled ?? true;
            appSettings.general.knowledgeGraphEnabled = data.general.knowledgeGraphEnabled ?? true;
            appSettings.general.autoClearGraphEnabled = data.general.autoClearGraphEnabled ?? false;
            appSettings.general.researchGraphEnabled = data.general.researchGraphEnabled ?? true;
          }
          if (data.plannerAgent) {
            appSettings.plannerAgent.retrievalEnabled = data.plannerAgent.retrievalEnabled ?? true;
            appSettings.plannerAgent.graphEnabled = data.plannerAgent.graphEnabled ?? true;
            appSettings.plannerAgent.criticEnabled = data.plannerAgent.criticEnabled ?? false;
            appSettings.plannerAgent.criticRounds = data.plannerAgent.criticRounds ?? 1;
            appSettings.plannerAgent.mergeTasksEnabled = data.plannerAgent.mergeTasksEnabled ?? true;
          }
          if (data.hypothesesAgent) {
            appSettings.hypothesesAgent.maxActiveHypotheses = data.hypothesesAgent.maxActiveHypotheses ?? 1;
          }
          if (data.researchAgent) {
            appSettings.researchAgent.maxSearches = data.researchAgent.maxSearches ?? 2;
          }
          if (data.taskExecutorAgent) {
            appSettings.taskExecutorAgent.keepScore = data.taskExecutorAgent.keepScore ?? 0.3;
            appSettings.taskExecutorAgent.abstainScore = data.taskExecutorAgent.abstainScore ?? 0.2;
          }
          if (data.coderAgent) {
            appSettings.coderAgent.sandboxUrl = data.coderAgent.sandboxUrl || 'http://localhost:8884';
            appSettings.coderAgent.workspaceId = data.coderAgent.workspaceId || '';
            appSettings.coderAgent.mode = data.coderAgent.mode || 'local';
          }
          if (!activeSandboxWatchUrl) {
            updateCoderSandboxButton(null);
          }
        }
      } catch (e) {
        console.warn('Failed to load settings from server:', e);
      }
    }

    async function openSettings() {
      const modal = document.getElementById('settings-modal');
      modal.classList.remove('hidden');
      // Apply current language to the modal (highlight toggle + translate)
      applyLanguage();
      // Load latest from server
      await loadSettings();
      // Sync UI with current state
      document.getElementById('start-mode-select').value = appSettings.general.startMode;
      document.getElementById('max-retries-input').value = appSettings.general.maxRetries;
      document.getElementById('hitl-enabled-checkbox').checked = appSettings.general.hitlEnabled;
      document.getElementById('hitl-timeout-input').value = appSettings.general.hitlAutoApproveTimeout ?? 300;
      document.getElementById('use-planner-checkbox').checked = appSettings.general.usePlanner;
      document.getElementById('use-proxy-checkbox').checked = appSettings.general.useProxy;
      document.getElementById('opik-enabled-checkbox').checked = appSettings.general.opikEnabled;
      document.getElementById('auto-naming-enabled-checkbox').checked = appSettings.general.autoNamingEnabled;
      document.getElementById('default-username-input').value = appSettings.general.coscientistUsername || '';
      document.getElementById('context-init-enabled-checkbox').checked = appSettings.general.contextInitEnabled ?? true;
      document.getElementById('knowledge-graph-checkbox').checked = appSettings.general.knowledgeGraphEnabled;
      document.getElementById('auto-clear-graph-checkbox').checked = appSettings.general.autoClearGraphEnabled;
      document.getElementById('research-graph-checkbox').checked = appSettings.general.researchGraphEnabled;
      document.getElementById('planner-retrieval-checkbox').checked = appSettings.plannerAgent.retrievalEnabled;
      document.getElementById('planner-graph-checkbox').checked = appSettings.plannerAgent.graphEnabled;
      document.getElementById('planner-critic-checkbox').checked = appSettings.plannerAgent.criticEnabled;
      document.getElementById('planner-critic-rounds').value = appSettings.plannerAgent.criticRounds;
      document.getElementById('planner-merge-tasks-checkbox').checked = appSettings.plannerAgent.mergeTasksEnabled;
      document.getElementById('hypotheses-max-active').value = appSettings.hypothesesAgent.maxActiveHypotheses;
      document.getElementById('research-max-searches').value = appSettings.researchAgent.maxSearches;
      document.getElementById('task-exec-keep-score').value = appSettings.taskExecutorAgent.keepScore;
      document.getElementById('task-exec-abstain-score').value = appSettings.taskExecutorAgent.abstainScore;
      document.getElementById('coder-sandbox-url').value = appSettings.coderAgent.sandboxUrl;
      document.getElementById('coder-workspace-id').value = appSettings.coderAgent.workspaceId;
      document.getElementById('coder-mode-select').value = appSettings.coderAgent.mode || 'local';
      // A deletion result from an earlier visit says nothing about now.
      const graphDeleteStatus = document.getElementById('graph-delete-status');
      graphDeleteStatus.textContent = '';
      graphDeleteStatus.classList.add('hidden');
      updateStartModeUI();
      updateGraphToggleUI();
      updatePlanCriticUI();
      updateCoderToolsUI();
    }

    function updateStartModeUI() {
      const startMode = document.getElementById('start-mode-select').value;
      const plannerCheckbox = document.getElementById('use-planner-checkbox');
      const plannerNote = document.getElementById('use-planner-note');

      if (startMode === 'planner' || startMode === 'orchestrator_planner') {
        plannerCheckbox.disabled = true;
        plannerCheckbox.checked = false;
        if (plannerNote) plannerNote.classList.remove('hidden');
      } else {
        plannerCheckbox.disabled = false;
        plannerCheckbox.checked = appSettings.general.usePlanner ?? true;
        if (plannerNote) plannerNote.classList.add('hidden');
      }
    }

    // The planner's Graph Tools switch only picks WHETHER the planner reads the
    // knowledge graph — with the graph itself off there is nothing to read, so
    // grey it out instead of letting it claim a tool the agent will not get.
    function updateGraphToggleUI() {
      const graphOn = document.getElementById('knowledge-graph-checkbox').checked;
      const plannerGraph = document.getElementById('planner-graph-checkbox');
      const note = document.getElementById('planner-graph-note');

      plannerGraph.disabled = !graphOn;
      plannerGraph.checked = graphOn && (appSettings.plannerAgent.graphEnabled ?? true);
      if (note) note.classList.toggle('hidden', graphOn);
    }

    // The round budget only means anything while the critic runs at all, so it
    // follows the Plan Critic switch instead of sitting there claiming a review
    // nobody will perform.
    function updatePlanCriticUI() {
      const criticOn = document.getElementById('planner-critic-checkbox').checked;
      const rounds = document.getElementById('planner-critic-rounds');
      const row = document.getElementById('planner-critic-rounds-row');

      rounds.disabled = !criticOn;
      row.classList.toggle('opacity-40', !criticOn);
    }

    // Deleting a graph rewrites files on the server, so the confirmation names
    // exactly what goes — in particular that the knowledge memory is shared by
    // every session, not just the one currently open.
    const GRAPH_DELETE_LABELS = {
      execution: "this session's execution graph",
      research: "this session's research graph",
      memory: 'the GLOBAL knowledge memory (shared by every session)',
      all: "this session's execution and research graphs AND the GLOBAL knowledge memory",
    };

    function describeGraphDeletion(deleted) {
      return Object.entries(deleted || {}).map(([name, info]) => {
        if (info && info.error) return `${name} failed (${info.error})`;
        if (name === 'memory') {
          return `${name} (${info.entities || 0} entities, ${info.relations || 0} relations)`;
        }
        return name;
      }).join(', ');
    }

    async function deleteGraphData() {
      const target = document.getElementById('graph-delete-target').value;
      const button = document.getElementById('graph-delete-btn');
      const status = document.getElementById('graph-delete-status');
      if (!confirm(`Delete ${GRAPH_DELETE_LABELS[target]}?\n\nThis cannot be undone from the UI.`)) return;

      status.classList.remove('hidden');
      status.textContent = 'Deleting...';
      status.className = 'text-[10px] font-mono mt-1.5 text-primary/70 animate-pulse';
      button.disabled = true;
      try {
        const url = sessionApi(`/graph?view=${encodeURIComponent(target)}`);
        const response = await fetch(url, { method: 'DELETE' });
        let data = {};
        try { data = await response.json(); } catch (_) { /* empty response */ }
        if (!response.ok) {
          throw new Error(data.detail || describeGraphDeletion(data.deleted) || `HTTP ${response.status}`);
        }
        status.textContent = `Deleted ${describeGraphDeletion(data.deleted) || 'nothing'}.`;
        status.className = 'text-[10px] font-mono mt-1.5 text-secondary';
        addTelemetry(`GRAPHS :: deleted ${target}`);
      } catch (error) {
        status.textContent = 'Error deleting graphs: ' + (error.message || error);
        status.className = 'text-[10px] font-mono mt-1.5 text-error';
      } finally {
        button.disabled = false;
      }
    }

    // Warn when the coder ends up with no way to run anything: openhands mode
    // and no sandbox URL leaves CoderAgent unable to execute at all.
    function updateCoderToolsUI() {
      const mode = document.getElementById('coder-mode-select').value;
      const localOn = mode === 'local';
      const sandboxUrl = document.getElementById('coder-sandbox-url').value.trim();
      const note = document.getElementById('coder-local-tools-note');
      if (note) note.classList.toggle('hidden', localOn || !!sandboxUrl);
    }

    function closeSettings() {
      document.getElementById('settings-modal').classList.add('hidden');
    }

    async function saveSettings() {
      // Read from UI
      appSettings.general.startMode = document.getElementById('start-mode-select').value;
      appSettings.general.maxRetries = parseInt(document.getElementById('max-retries-input').value, 10) || 3;
      appSettings.general.hitlEnabled = document.getElementById('hitl-enabled-checkbox').checked;
      const hitlTimeoutVal = parseInt(document.getElementById('hitl-timeout-input').value, 10);
      appSettings.general.hitlAutoApproveTimeout = !isNaN(hitlTimeoutVal) ? hitlTimeoutVal : 300;
      if (appSettings.general.startMode === 'orchestrator') {
        appSettings.general.usePlanner = document.getElementById('use-planner-checkbox').checked;
      }
      appSettings.general.useProxy = document.getElementById('use-proxy-checkbox').checked;
      appSettings.general.opikEnabled = document.getElementById('opik-enabled-checkbox').checked;
      appSettings.general.autoNamingEnabled = document.getElementById('auto-naming-enabled-checkbox').checked;
      appSettings.general.coscientistUsername = document.getElementById('default-username-input').value.trim();
      appSettings.general.contextInitEnabled = document.getElementById('context-init-enabled-checkbox').checked;
      appSettings.general.knowledgeGraphEnabled = document.getElementById('knowledge-graph-checkbox').checked;
      appSettings.general.researchGraphEnabled = document.getElementById('research-graph-checkbox').checked;

      appSettings.plannerAgent.retrievalEnabled = document.getElementById('planner-retrieval-checkbox').checked;
      // Disabled while the knowledge graph is off — keep the stored preference
      // so it comes back as it was when the graph is switched on again.
      if (appSettings.general.knowledgeGraphEnabled) {
        appSettings.plannerAgent.graphEnabled = document.getElementById('planner-graph-checkbox').checked;
      }
      appSettings.plannerAgent.criticEnabled = document.getElementById('planner-critic-checkbox').checked;
      const criticRounds = parseInt(document.getElementById('planner-critic-rounds').value, 10);
      if (!isNaN(criticRounds) && criticRounds >= 1) {
        appSettings.plannerAgent.criticRounds = criticRounds;
      }

      appSettings.plannerAgent.mergeTasksEnabled = document.getElementById('planner-merge-tasks-checkbox').checked;

      const maxActive = parseInt(document.getElementById('hypotheses-max-active').value, 10);
      if (!isNaN(maxActive) && maxActive >= 1 && maxActive <= 5) {
        appSettings.hypothesesAgent.maxActiveHypotheses = maxActive;
      }

      const maxSearches = parseInt(document.getElementById('research-max-searches').value, 10);
      if (!isNaN(maxSearches) && maxSearches >= 0) {
        appSettings.researchAgent.maxSearches = maxSearches;
      }

      appSettings.taskExecutorAgent.keepScore = parseFloat(document.getElementById('task-exec-keep-score').value) || 0.3;
      appSettings.taskExecutorAgent.abstainScore = parseFloat(document.getElementById('task-exec-abstain-score').value) || 0.2;

      appSettings.coderAgent.sandboxUrl = document.getElementById('coder-sandbox-url').value.trim() || 'http://localhost:8884';
      appSettings.coderAgent.workspaceId = document.getElementById('coder-workspace-id').value.trim();
      appSettings.coderAgent.mode = document.getElementById('coder-mode-select').value;
      if (!activeSandboxWatchUrl) {
        updateCoderSandboxButton(null);
      }

      const status = document.getElementById('settings-status');
      status.textContent = 'Saving...';
      status.className = 'text-[10px] font-mono text-primary/70 animate-pulse';
      status.classList.remove('hidden');

      try {
        const resp = await fetch('/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(appSettings),
        });
        if (resp.ok) {
          addTelemetry('SETTINGS :: saved dynamically');
          status.textContent = 'Settings saved.';
          status.className = 'text-[10px] font-mono text-secondary';
          setTimeout(() => { status.classList.add('hidden'); closeSettings(); }, 800);
        } else {
          throw new Error('HTTP ' + resp.status);
        }
      } catch (e) {
        status.textContent = 'Error saving settings: ' + e.message;
        status.className = 'text-[10px] font-mono text-error';
      }
    }

