// =========================================================================
// Plan Tracker — the right-hand panel: one row per plan step, with its state
// and, folded under it, what actually ran inside that step.
//
// It holds no plan of its own: the roadmap modal already merges every source
// (session snapshot, create_plan / update_task_status results, manual edits)
// into one task list, and calls render() whenever that list changes.
//
// Sub-steps are a different matter. The planner registers a FLAT task list —
// `task_tracker.py` mints {id, title, description, assignee, status,
// parent_id, notes, tools} and nothing else, and `parent_id` there is a
// prerequisite edge, not a parent-child one. So the sub-steps shown here are
// not read from the plan: they are what the activity stream says happened
// under each step, one line per agent that worked on it, named by what that
// agent does. That is live data, so a step from a session whose server has
// since restarted shows no sub-steps rather than invented ones.
//
// A LINEAR pipeline (config `pipeline.linear`, e.g. microfluidics) is a plan
// in itself: its stages are the lines, and the status indicator — which
// tracks the stage the run is at — calls render() when a stage changes. The
// roadmap's tasks then nest under the stage of their assignee (the LIT-xx
// tasks under «Анализ литературы», whose members include ResearchAgent).
// =========================================================================
(function () {
  'use strict';

  const STATUS_VIEW = {
    done: { icon: 'check_circle', iconClass: 'text-secondary', textClass: 'text-on-surface-variant', label: 'plan.status.done' },
    in_progress: { icon: 'radio_button_checked', iconClass: 'text-primary', textClass: 'text-on-surface font-medium', label: 'plan.status.in_progress' },
    error: { icon: 'error', iconClass: 'text-error', textClass: 'text-error', label: 'plan.status.error' },
    todo: { icon: 'radio_button_unchecked', iconClass: 'text-outline-variant/60', textClass: 'text-outline-variant', label: 'plan.status.todo' },
  };

  // The row last scrolled to: the list follows the work only when it moves
  // on, so a user reading further down is not yanked back on every update.
  let followedKey = null;

  // taskId -> Map(agent -> {agent, status, tools}) in arrival order.
  const subStepsByTask = new Map();
  // Who delegated to whom, so an agent three levels down can still be traced
  // back to the step whose assignee started the chain.
  const delegatedBy = new Map();
  // What the reader decided by hand, either way. The running step opens
  // itself, so only an explicit close can keep it shut; and a step opened by
  // hand stays open when the work moves on — an accordion that closes what you
  // opened to show you something else is worse than one that never opens.
  const openTaskIds = new Set();
  const closedTaskIds = new Set();
  // Finished steps above the running one fold into a single line once there
  // are more than a few: on step 9 of 11 the eight ticks say less than "8
  // done" does, and they push the step that matters out of view.
  const DONE_FOLD_MIN = 4;
  let showAllDone = false;

  // Agents that are the machinery around a step rather than work inside it.
  // The orchestrator delegates every step, so listing it under all of them
  // says nothing; the planner's work IS the plan, not a step within it.
  const NOT_A_SUBSTEP = new Set([
    'OrchestratorAgent', 'PlannerAgent', 'PlanningPipelineAgent',
    'PlanCriticAgent', 'InitAgent', 'system',
  ]);

  function currentTasks() {
    return (window.RoadmapModal && window.RoadmapModal.getTasks()) || [];
  }

  function normalize(status) {
    return window.RoadmapModal.normalizeStatus(status);
  }

  // ── Sub-steps: reading them off the activity stream ─────────────────────

  function sameAgent(a, b) {
    if (!a || !b) return false;
    return String(a).trim().toLowerCase() === String(b).trim().toLowerCase();
  }

  // What to call the work, rather than what to call the agent: "Генерация
  // гипотез" reads as a step, "агент генерации гипотез" reads as a job title.
  // An agent with no entry falls back to the role table in status_indicator.
  /** The agent's Russian role name, as the rail and the chat show it. */
  function agentRole(agent) {
    return (window.StatusIndicator && StatusIndicator.agentName)
      ? (StatusIndicator.agentName(agent) || String(agent || '')) : String(agent || '');
  }

  function substepLabel(agent) {
    const key = 'substep.' + agent;
    if (typeof i18n !== 'undefined' && i18n[key]) return t(key);
    if (window.StatusIndicator && StatusIndicator.agentLabel) {
      const role = StatusIndicator.agentLabel(agent, true);
      if (role) return role;
    }
    return String(agent || '').replace(/Agent$/, '');
  }

  // The step an agent's work belongs to. Nothing on the activity stream
  // carries a task id, so the join is made on the assignee: this agent, or
  // whoever delegated to it, or whoever delegated to them.
  //
  // Assignee first and "whatever is running" only as a fallback, deliberately.
  // A replayed snapshot arrives with every task already at its FINAL status,
  // so an attribution that leaned on "which step was running" would put a
  // whole finished session's work under one step, or under none. The assignee
  // is the same on replay as it was live.
  function taskIdFor(agent) {
    const tasks = currentTasks();
    if (!tasks.length) return null;

    const chain = [];
    for (let name = agent, hops = 0; name && hops < 8; name = delegatedBy.get(name), hops++) {
      chain.push(name);
    }

    for (const name of chain) {
      const owned = tasks.filter(task => sameAgent(task.assignee, name));
      if (owned.length === 1) return owned[0].id || null;
      if (owned.length > 1) {
        // One agent assigned to several steps: the running one if there is
        // one, else the earliest that has not finished.
        const running = owned.find(task => normalize(task.status) === 'in_progress');
        if (running) return running.id || null;
        const pending = owned.find(task => normalize(task.status) !== 'done');
        return (pending || owned[owned.length - 1]).id || null;
      }
    }

    // No step claims this agent. Live, that is work under whatever is running;
    // on a replay of a finished run nothing is, and guessing would be worse
    // than showing no sub-steps.
    const running = tasks.find(task => normalize(task.status) === 'in_progress');
    return running ? (running.id || null) : null;
  }

  function substepsOf(taskId) {
    if (!subStepsByTask.has(taskId)) subStepsByTask.set(taskId, new Map());
    return subStepsByTask.get(taskId);
  }

  function noteAgent(agent, status) {
    if (!agent || NOT_A_SUBSTEP.has(agent)) return false;
    if (typeof isInternalAgent === 'function' && isInternalAgent(agent)) return false;
    const taskId = taskIdFor(agent);
    if (!taskId) return false;
    const steps = substepsOf(taskId);
    const existing = steps.get(agent);
    if (existing) {
      // A step an agent comes back to is still one step; only its state moves,
      // and never backwards from done to running on a late-arriving event.
      if (status === 'done' || existing.status !== 'done') existing.status = status;
      return true;
    }
    steps.set(agent, { agent: agent, status: status, tools: 0 });
    return true;
  }

  function noteTool(agent) {
    if (!agent || NOT_A_SUBSTEP.has(agent)) return false;
    const taskId = taskIdFor(agent);
    if (!taskId) return false;
    const step = substepsOf(taskId).get(agent);
    if (!step) return false;
    step.tools += 1;
    return true;
  }

  // Called for every `tool_activity` event, live (ws.js) and replayed from a
  // session snapshot (sessions.js). Returns nothing; it redraws when it has
  // something new to say.
  function feed(event) {
    if (!event) return;
    if (event.type === 'session_snapshot') {
      subStepsByTask.clear();
      delegatedBy.clear();
      openTaskIds.clear();
      closedTaskIds.clear();
      return;
    }
    if (event.type !== 'tool_activity') return;

    let changed = false;
    if (event.phase === 'agent_start') {
      if (event.parent) delegatedBy.set(event.author, event.parent);
      changed = noteAgent(event.author, 'in_progress');
    } else if (event.phase === 'agent_end') {
      changed = noteAgent(event.author, 'done');
    } else if (event.phase === 'call') {
      if (event.is_delegation && event.target_agent) {
        delegatedBy.set(event.target_agent, event.author);
        changed = noteAgent(event.target_agent, 'in_progress');
      } else {
        changed = noteTool(event.author);
      }
    } else if (event.phase === 'result' || event.phase === 'error') {
      if (event.is_delegation && event.target_agent) {
        changed = noteAgent(event.target_agent, 'done');
      }
    }
    if (changed) scheduleRender();
  }

  let renderQueued = false;

  function scheduleRender() {
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(() => { renderQueued = false; render(); });
  }

  // ── Rendering ───────────────────────────────────────────────────────────

  function substepRow(step, number) {
    const view = STATUS_VIEW[step.status] || STATUS_VIEW.todo;
    const tools = step.tools
      ? `<span class="text-[10px] font-mono text-outline-variant/80 shrink-0"
           title="${escHtml(t('plan.substep.toolCount', { n: step.tools }))}">${step.tools}⚙</span>`
      : '';
    // The activity and the agent that did it, in that order — the activity is
    // what the reader is following, the agent is what they can match against
    // the rail and the execution graph. `flex-wrap` lets the agent drop to its
    // own line in the 320px column instead of squeezing the activity name.
    return `
          <li class="flex flex-wrap items-baseline gap-x-1.5 py-0.5">
            <span class="material-symbols-outlined text-[14px] shrink-0 ${view.iconClass}">${view.icon}</span>
            <span class="text-[10px] font-mono text-outline-variant tabular-nums shrink-0">${escHtml(number)}</span>
            <span class="text-[11px] leading-snug min-w-0 flex-1 break-words ${view.textClass}"
              title="${escHtml(step.agent)}">${escHtml(substepLabel(step.agent))}</span>
            ${tools}
            ${agentRole(step.agent).toLowerCase() === substepLabel(step.agent).toLowerCase() ? '' : `
            <span class="w-full pl-[38px] text-[10px] text-outline-variant/70 truncate">${escHtml(agentRole(step.agent))}</span>`}
          </li>`;

  }

  // ── Linear pipeline: its stages are the plan ──────────────────────────

  function currentStages() {
    return (window.StatusIndicator && window.StatusIndicator.stages()) || [];
  }

  /** Rows of a plain roadmap: one per task. */
  function taskRows(tasks, normalize) {
    return tasks.map((task, idx) => ({
      key: 'task:' + (task.id || idx),
      id: task.id || idx + 1,
      number: idx + 1,
      title: task.title || t('plan.untitled', 'Untitled task'),
      status: normalize(task.status),
      nested: false,
    }));
  }

  /** Rows of a linear pipeline: its stages, each followed by the roadmap
   *  tasks its members execute. A task whose assignee is in no stage goes
   *  with the others; if none is, the tasks follow the last stage. */
  function stageRows(stages, tasks, normalize) {
    const stageOf = name => (name ? window.StatusIndicator.stageOf(name) : -1);
    const hosts = tasks.map(task => stageOf(task.assignee));
    const fallback = hosts.find(host => host >= 0);
    const host = i => (hosts[i] >= 0 ? hosts[i] : (fallback === undefined ? stages.length - 1 : fallback));
    const subRows = taskRows(tasks, normalize).map(row => Object.assign(row, { nested: true }));

    return stages.flatMap((stage, i) => [{
      key: 'stage:' + stage.agent,
      id: i + 1,
      number: i + 1,
      title: stage.title,
      status: stage.status,
      nested: false,
    }].concat(subRows.filter((_, j) => host(j) === i)));
  }

  function render(tasks = currentTasks()) {
    const panel = document.getElementById('plan-tracker');
    const list = document.getElementById('plan-tracker-list');
    if (!panel || !list) return;
    const stages = currentStages();

    // No plan yet: stay out of the layout entirely rather than reserving an
    // empty card, so the sidebar doesn't show a box with nothing in it.
    if (!tasks.length && !stages.length) {
      panel.classList.add('hidden');
      followedKey = null;
      list.innerHTML = '';
      return;
    }
    panel.classList.remove('hidden');

    if (stages.length) {
      renderStages(list, stages, tasks);
      return;
    }

    const done = tasks.filter(task => normalize(task.status) === 'done').length;
    const percent = Math.round((done / tasks.length) * 100);
    document.getElementById('plan-tracker-count').textContent = t('plan.progress', { done: done, total: tasks.length });
    document.getElementById('plan-tracker-bar').style.width = percent + '%';

    const active = tasks.find(task => normalize(task.status) === 'in_progress');
    renderTopbarStep(tasks, active);

    // The leading run of finished steps, minus the last one (kept visible so
    // the running step has its predecessor above it).
    let leadingDone = 0;
    while (leadingDone < tasks.length && normalize(tasks[leadingDone].status) === 'done') leadingDone++;
    const foldCount = !showAllDone && leadingDone >= DONE_FOLD_MIN ? leadingDone - 1 : 0;
    const foldRow = foldCount ? `
        <li>
          <button type="button" onclick="togglePlanDone()" aria-expanded="false"
            class="w-full flex items-center gap-1.5 pl-1 pr-1 py-1.5 rounded-md text-left text-[11px] text-outline-variant hover:text-on-surface hover:bg-surface-container transition-colors">
            <span class="w-[15px] shrink-0"></span>
            <span class="material-symbols-outlined text-[15px] shrink-0 text-secondary" aria-hidden="true">done_all</span>
            <span>${escHtml(t('plan.doneFolded', { n: foldCount }))}</span>
          </button>
        </li>` : '';

    list.innerHTML = foldRow + tasks.map((task, idx) => {
      if (idx < foldCount) return '';
      const state = normalize(task.status);
      const view = STATUS_VIEW[state] || STATUS_VIEW.todo;
      const title = task.title || t('plan.untitled', 'Untitled task');
      const taskId = task.id || '';
      const steps = [...(subStepsByTask.get(taskId) || new Map()).values()];
      // The running step opens itself so the reader sees what is happening
      // without asking; a step opened by hand stays open regardless.
      const open = steps.length > 0 && !closedTaskIds.has(taskId)
        && (openTaskIds.has(taskId) || (active && active.id === taskId));
      const tooltip = `${taskId || idx + 1} · ${t(view.label)}\n${title}`;
      const rowClass = state === 'in_progress' ? 'bg-surface-container-high' : '';
      const stepsDone = steps.filter(step => step.status === 'done').length;

      const disclosure = steps.length
        ? `<span class="material-symbols-outlined text-[15px] shrink-0 mt-px text-outline-variant transition-transform ${open ? 'rotate-90' : ''}">chevron_right</span>`
        : '<span class="w-[15px] shrink-0"></span>';
      const counter = steps.length
        ? `<span class="text-[10px] font-mono text-outline-variant/80 shrink-0 mt-px">${stepsDone}/${steps.length}</span>`
        : '';

      return `
        <li data-task-id="${escHtml(taskId)}" class="rounded-md ${rowClass}"${state === 'in_progress' ? ' aria-current="step"' : ''}>
          <div ${steps.length ? `role="button" tabindex="0" aria-expanded="${open}" onclick="togglePlanStep('${escJs(taskId)}')"
                 onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();togglePlanStep('${escJs(taskId)}')}"
                 class="cursor-pointer rounded-md hover:bg-surface-container transition-colors"` : 'class=""'}
            title="${escHtml(tooltip)}">
            <div class="flex items-start gap-1.5 pl-1 pr-1 py-1.5">
              ${disclosure}
              <span class="material-symbols-outlined text-[15px] shrink-0 mt-px ${view.iconClass}">${view.icon}</span>
              <span class="text-[10px] text-outline-variant tabular-nums shrink-0 mt-0.5 w-4 text-right">${idx + 1}</span>
              <span class="text-[12px] leading-snug break-words min-w-0 flex-1 ${view.textClass}">${mdInline(title)}</span>
              ${counter}
            </div>
          </div>
          ${open ? `<ol class="ml-[22px] mr-1 mb-1.5 pl-2 border-l border-outline-variant/20">${
        steps.map((step, i) => substepRow(step, `${idx + 1}.${i + 1}`)).join('')}</ol>` : ''}
        </li>`;
    }).join('');

    if (active && active.id !== followedKey) {
      followedKey = active.id;
      const row = list.querySelector(`[data-task-id="${CSS.escape(active.id || '')}"]`);
      if (row) list.scrollTop = Math.max(0, row.offsetTop - list.clientHeight / 3);
    }
  }

  function renderStages(list, stages, tasks) {
    // The count and the bar measure the stages: the roadmap is one stage's
    // business.
    const rows = stageRows(stages, tasks, normalize);
    const top = rows.filter(row => !row.nested);
    const done = top.filter(row => row.status === 'done').length;
    const percent = Math.round((done / top.length) * 100);
    document.getElementById('plan-tracker-count').textContent = t('plan.progress', { done: done, total: top.length });
    document.getElementById('plan-tracker-bar').style.width = percent + '%';

    list.innerHTML = rows.map(row => {
      const view = STATUS_VIEW[row.status] || STATUS_VIEW.todo;
      const tooltip = `${row.id} · ${t(view.label)}\n${row.title}`;
      const rowClass = view === STATUS_VIEW.in_progress ? 'bg-primary/5 border-primary/60' : 'border-transparent';
      const indent = row.nested ? 'ml-5' : '';
      return `
        <li data-row-key="${escHtml(row.key)}" title="${escHtml(tooltip)}"
          class="flex items-center gap-2 pl-1.5 pr-1 py-1 rounded-r border-l-2 ${indent} ${rowClass}">
          <span class="material-symbols-outlined text-[14px] shrink-0 ${view.iconClass}">${view.icon}</span>
          <span class="text-[9px] font-mono text-outline-variant/70 w-4 text-right shrink-0">${row.number}</span>
          <span class="text-[11px] leading-snug truncate ${view.textClass}">${escHtml(row.title)}</span>
        </li>`;
    }).join('');

    // Follow the innermost work: a running task rather than its stage.
    const active = rows.filter(row => row.status === 'in_progress').sort((a, b) => b.nested - a.nested)[0];
    if (active && active.key !== followedKey) {
      followedKey = active.key;
      const row = list.querySelector(`[data-row-key="${CSS.escape(active.key)}"]`);
      if (row) list.scrollTop = Math.max(0, row.offsetTop - list.clientHeight / 3);
    }
  }

  // "Stage 9 of 11" in the top bar: the running step, or the next one to run.
  function renderTopbarStep(tasks, active) {
    const el = document.getElementById('topbar-step');
    if (!el) return;
    if (!tasks.length) {
      el.classList.add('hidden');
      return;
    }
    const doneCount = tasks.filter(task => normalize(task.status) === 'done').length;
    const position = active ? tasks.indexOf(active) + 1 : Math.min(doneCount + 1, tasks.length);
    el.textContent = t('plan.stageOf', { n: position, total: tasks.length });
    el.title = active ? (active.title || '') : '';
    el.classList.remove('hidden');
  }

  function togglePlanDone() {
    showAllDone = !showAllDone;
    render();
  }

  function togglePlanStep(taskId) {
    const row = document.querySelector(`#plan-tracker-list [data-task-id="${CSS.escape(taskId)}"] [aria-expanded]`);
    const isOpen = !!row && row.getAttribute('aria-expanded') === 'true';
    openTaskIds.delete(taskId);
    closedTaskIds.delete(taskId);
    (isOpen ? closedTaskIds : openTaskIds).add(taskId);
    render();
  }

  window.PlanTracker = { render: render, feed: feed };
  window.togglePlanStep = togglePlanStep;
  window.togglePlanDone = togglePlanDone;
  render();
})();
