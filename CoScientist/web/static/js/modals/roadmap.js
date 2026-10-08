// =========================================================================
// Roadmap & Execution Plan Modal (Interactive Visual Plan Editor)
// =========================================================================
(function () {
  'use strict';

  let currentTasks = [];
  let editingTaskId = null;
  let isNewTaskDraft = false;
  let activeFilter = 'all';    // 'all' | 'active' | 'done' | 'todo'
  let searchQuery = '';
  let isSaving = false;
  const expandedTaskIds = new Set();

  const KNOWN_AGENTS = [
    'ResearchAgent',
    'PlannerAgent',
    'PlanningPipelineAgent',
    'CoderAgent',
    'TaskExecutorAgent',
    'DatasetCollectorAgent',
    'HypothesesAgent',
    'MedicalAgent',
    'McpBuilderAgent',
    'ToolPipelineAgent',
    'ToolPreparerAgent',
    'ResultAggregatorAgent',
    'FedotAgent',
    'OrchestratorAgent',
    'system',
  ];

  const AGENT_ICONS = {
    OrchestratorAgent: 'hub',
    PlannerAgent: 'map',
    PlanningPipelineAgent: 'map',
    HypothesesAgent: 'lightbulb',
    ResearchAgent: 'travel_explore',
    TaskExecutorAgent: 'alt_route',
    CoderAgent: 'terminal',
    DatasetCollectorAgent: 'dataset',
    MedicalAgent: 'medical_services',
    McpBuilderAgent: 'construction',
    ToolPipelineAgent: 'checklist',
    ToolPreparerAgent: 'precision_manufacturing',
    ResultAggregatorAgent: 'summarize',
    FedotAgent: 'auto_graph',
    system: 'settings_suggest',
  };

  const STATUS_CONFIG = {
    done: {
      label: 'Completed',
      icon: 'check_circle',
      badgeClass: 'text-secondary bg-secondary/10 border-secondary/30',
      barClass: 'bg-secondary',
      spin: false,
    },
    in_progress: {
      label: 'In Progress',
      icon: 'autorenew',
      badgeClass: 'text-primary bg-primary/10 border-primary/40 animate-pulse',
      barClass: 'bg-primary shadow-[0_0_8px_rgba(0,218,243,0.5)]',
      spin: true,
    },
    error: {
      label: 'Failed',
      icon: 'error',
      badgeClass: 'text-error bg-error/10 border-error/30',
      barClass: 'bg-error',
      spin: false,
    },
    todo: {
      label: 'Pending',
      icon: 'schedule',
      badgeClass: 'text-outline-variant bg-surface-container-high border-outline-variant/20',
      barClass: 'bg-outline-variant/40',
      spin: false,
    },
  };

  function normalizeStatus(status) {
    const s = String(status || '').toLowerCase().trim();
    if (/done|complete|finish|success/.test(s)) return 'done';
    if (/progress|running|active|doing/.test(s)) return 'in_progress';
    if (/fail|error|cancel/.test(s)) return 'error';
    return 'todo';
  }

  function getAgentIcon(assignee) {
    if (!assignee) return 'smart_toy';
    return AGENT_ICONS[assignee] || 'smart_toy';
  }

  function agentShortName(name) {
    if (!name) return 'Agent';
    return String(name).replace(/Agent$/, '');
  }

  function escHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ── Open / Close Modal ───────────────────────────────────────────────────
  async function openRoadmapEditor() {
    const modal = document.getElementById('roadmap-modal');
    const status = document.getElementById('roadmap-status');
    const saveBtn = document.getElementById('save-roadmap-btn');

    if (!modal) return;
    modal.classList.remove('hidden');
    updateRoadmapModalButtons();

    // If we already have live tasks, render them immediately so user sees the plan right away
    if (currentTasks.length > 0) {
      renderRoadmapView();
    }

    if (status) {
      status.classList.remove('hidden');
      status.textContent = 'Syncing roadmap…';
      status.className = 'text-[10px] font-mono text-primary/70 animate-pulse';
    }
    if (saveBtn) saveBtn.disabled = true;

    try {
      const response = await fetch(roadmapUrl());
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      let fetched = [];
      if (Array.isArray(data.tasks)) {
        fetched = data.tasks;
      } else if (data.content && data.content.trim()) {
        try {
          const parsed = JSON.parse(data.content);
          fetched = Array.isArray(parsed) ? parsed : (parsed.tasks || []);
        } catch (_) {
          fetched = [];
        }
      }

      if (fetched.length > 0 || currentTasks.length === 0) {
        currentTasks = fetched;
      }

      if (status) {
        status.textContent = 'Roadmap synced.';
        status.className = 'text-[10px] font-mono text-secondary';
        setTimeout(() => { if (status) status.classList.add('hidden'); }, 1800);
      }
    } catch (error) {
      console.error('Failed to load roadmap:', error);
      if (status) {
        status.textContent = 'Error syncing roadmap: ' + error.message;
        status.className = 'text-[10px] font-mono text-error';
      }
    } finally {
      if (saveBtn) saveBtn.disabled = false;
      renderRoadmapView();
    }
  }

  function closeRoadmapEditor() {
    const modal = document.getElementById('roadmap-modal');
    if (modal) modal.classList.add('hidden');
    // If a draft task was being added with no title, clean it up
    if (isNewTaskDraft && editingTaskId) {
      cancelEditTask(editingTaskId);
    }
  }

  // ── Render Master ────────────────────────────────────────────────────────
  function renderRoadmapView() {
    updateProgressAndStats();
    renderRoadmapVisual();
  }

  function updateProgressAndStats() {
    const total = currentTasks.length;
    let done = 0;
    let active = 0;
    let todo = 0;

    currentTasks.forEach(t => {
      const norm = normalizeStatus(t.status);
      if (norm === 'done') done++;
      else if (norm === 'in_progress') active++;
      else todo++;
    });

    const percent = total > 0 ? Math.round((done / total) * 100) : 0;

    const pBar = document.getElementById('roadmap-progress-bar');
    const pText = document.getElementById('roadmap-progress-text');
    const cBadge = document.getElementById('roadmap-counter-badge');

    if (pBar) pBar.style.width = percent + '%';
    if (pText) pText.textContent = `${done} of ${total} completed (${percent}%)`;
    if (cBadge) cBadge.textContent = `${total} task${total === 1 ? '' : 's'}`;

    // Filter counts
    const fAll = document.getElementById('rf-all-count');
    const fActive = document.getElementById('rf-active-count');
    const fDone = document.getElementById('rf-done-count');
    const fTodo = document.getElementById('rf-todo-count');
    if (fAll) fAll.textContent = total;
    if (fActive) fActive.textContent = active;
    if (fDone) fDone.textContent = done;
    if (fTodo) fTodo.textContent = todo;

    // Footer stats
    const sTotal = document.getElementById('roadmap-stat-total');
    const sActive = document.getElementById('roadmap-stat-active');
    const sDone = document.getElementById('roadmap-stat-done');
    if (sTotal) sTotal.textContent = `Total: ${total}`;
    if (sActive) sActive.textContent = `In progress: ${active}`;
    if (sDone) sDone.textContent = `Done: ${done}`;
  }

  // ── Visual View Rendering ────────────────────────────────────────────────
  function renderRoadmapVisual() {
    const listEl = document.getElementById('roadmap-tasks-list');
    const emptyEl = document.getElementById('roadmap-empty-state');
    if (!listEl) return;

    let filtered = currentTasks.filter(t => {
      // Always show currently edited task regardless of filter/search
      if (t.id && t.id === editingTaskId) return true;

      const norm = normalizeStatus(t.status);
      if (activeFilter === 'active' && norm !== 'in_progress') return false;
      if (activeFilter === 'done' && norm !== 'done') return false;
      if (activeFilter === 'todo' && (norm === 'done' || norm === 'in_progress')) return false;

      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        const str = `${t.id || ''} ${t.title || ''} ${t.description || ''} ${t.assignee || ''} ${t.notes || ''}`.toLowerCase();
        if (!str.includes(q)) return false;
      }
      return true;
    });

    if (!currentTasks.length || !filtered.length) {
      listEl.innerHTML = '';
      if (emptyEl) {
        emptyEl.classList.remove('hidden');
        const emptyMsg = emptyEl.querySelector('p.text-sm');
        if (emptyMsg) {
          emptyMsg.textContent = currentTasks.length === 0
            ? 'No tasks found in roadmap'
            : 'No tasks match current filter/search';
        }
      }
      return;
    }

    if (emptyEl) emptyEl.classList.add('hidden');

    listEl.innerHTML = filtered.map((task, idx) => {
      const norm = normalizeStatus(task.status);
      const cfg = STATUS_CONFIG[norm] || STATUS_CONFIG.todo;
      const agentIcon = getAgentIcon(task.assignee);
      const agentName = agentShortName(task.assignee);
      const taskId = task.id || `TASK-${idx + 1}`;
      const desc = task.description || '';
      const isLongDesc = desc.length > 200;
      const isExpanded = expandedTaskIds.has(taskId);
      const isEditing = (taskId === editingTaskId);

      if (isEditing) {
        // --- IN-CARD EDIT MODE ---
        const allAgents = Array.from(new Set([task.assignee, ...KNOWN_AGENTS].filter(Boolean)));
        const otherTasks = currentTasks.filter(t => (t.id || '') !== taskId);

        return `
          <div id="roadmap-card-${escHtml(taskId)}" class="roadmap-card editing flex items-stretch gap-0 rounded-xl border-2 border-primary/50 bg-surface-container/95 overflow-hidden shadow-xl transition-all">
            <!-- Left status accent bar -->
            <span class="w-1.5 shrink-0 ${cfg.barClass}"></span>

            <div class="flex-1 p-4 min-w-0 flex flex-col gap-3">
              <!-- Top Row: ID, Status Select, Assignee Select, Parent Select, and Delete -->
              <div class="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-outline-variant/15">
                <div class="flex flex-wrap items-center gap-2">
                  <!-- Task ID -->
                  <span class="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-surface-container-high text-primary border border-primary/30">
                    ${escHtml(taskId)}
                  </span>

                  <!-- Status Selector -->
                  <div class="relative">
                    <select id="edit-status-${escHtml(taskId)}"
                      class="text-[10px] font-semibold rounded-full bg-surface-container-highest border border-outline-variant/30 text-on-surface px-2.5 py-1 pr-6 focus:border-primary focus:ring-1 focus:ring-primary outline-none cursor-pointer">
                      <option value="TODO" ${norm === 'todo' ? 'selected' : ''}>Pending</option>
                      <option value="IN_PROGRESS" ${norm === 'in_progress' ? 'selected' : ''}>In Progress</option>
                      <option value="DONE" ${norm === 'done' ? 'selected' : ''}>Completed</option>
                      <option value="ERROR" ${norm === 'error' ? 'selected' : ''}>Failed</option>
                    </select>
                  </div>

                  <!-- Assignee Selector -->
                  <div class="relative">
                    <select id="edit-assignee-${escHtml(taskId)}"
                      class="text-[10px] font-medium rounded-full bg-surface-container-highest border border-outline-variant/30 text-on-surface px-2.5 py-1 pr-6 focus:border-primary focus:ring-1 focus:ring-primary outline-none cursor-pointer">
                      ${allAgents.map(a => `
                        <option value="${escHtml(a)}" ${task.assignee === a ? 'selected' : ''}>${escHtml(a)}</option>
                      `).join('')}
                    </select>
                  </div>

                  <!-- Prerequisite Selector -->
                  <div class="relative">
                    <select id="edit-parent-${escHtml(taskId)}"
                      class="text-[10px] font-mono rounded-full bg-surface-container-highest border border-outline-variant/30 text-on-surface px-2.5 py-1 pr-6 focus:border-primary focus:ring-1 focus:ring-primary outline-none cursor-pointer">
                      <option value="">No prerequisite</option>
                      ${otherTasks.map(ot => `
                        <option value="${escHtml(ot.id || '')}" ${task.parent_id === ot.id ? 'selected' : ''}>Depends on: ${escHtml(ot.id || '')}</option>
                      `).join('')}
                    </select>
                  </div>
                </div>

                <div class="flex items-center gap-1">
                  <button type="button" onclick="deleteRoadmapTask('${escHtml(taskId)}')"
                    class="p-1 rounded text-outline-variant hover:text-error hover:bg-error/10 transition-colors"
                    title="Delete task">
                    <span class="material-symbols-outlined text-sm">delete</span>
                  </button>
                </div>
              </div>

              <!-- Title Input -->
              <div class="flex flex-col gap-1">
                <label class="text-[9px] font-bold font-mono text-outline-variant uppercase tracking-wider">Task Title</label>
                <input type="text" id="edit-title-${escHtml(taskId)}" value="${escHtml(task.title || '')}"
                  placeholder="Enter task title…"
                  onkeydown="if(event.key==='Enter') saveEditTask('${escHtml(taskId)}')"
                  class="w-full text-xs font-semibold bg-surface-container-lowest border border-outline-variant/25 rounded-md px-3 py-1.5 text-on-surface placeholder:text-outline-variant/40 focus:border-primary/60 focus:ring-1 focus:ring-primary/40 outline-none transition-all" />
              </div>

              <!-- Description Textarea -->
              <div class="flex flex-col gap-1">
                <label class="text-[9px] font-bold font-mono text-outline-variant uppercase tracking-wider">Description</label>
                <textarea id="edit-desc-${escHtml(taskId)}" rows="3"
                  placeholder="Specific instructions or details for this task…"
                  class="w-full text-xs font-sans bg-surface-container-lowest border border-outline-variant/25 rounded-md px-3 py-1.5 text-on-surface placeholder:text-outline-variant/40 focus:border-primary/60 focus:ring-1 focus:ring-primary/40 outline-none resize-y leading-relaxed transition-all">${escHtml(task.description || '')}</textarea>
              </div>

              <!-- Notes Input -->
              <div class="flex flex-col gap-1">
                <label class="text-[9px] font-bold font-mono text-outline-variant uppercase tracking-wider">Notes / Parameters (Optional)</label>
                <input type="text" id="edit-notes-${escHtml(taskId)}" value="${escHtml(task.notes || '')}"
                  placeholder="e.g. Max iterations: 3, temperature: 0.2"
                  onkeydown="if(event.key==='Enter') saveEditTask('${escHtml(taskId)}')"
                  class="w-full text-[11px] font-sans bg-surface-container-lowest border border-outline-variant/25 rounded-md px-3 py-1 text-on-surface placeholder:text-outline-variant/40 focus:border-primary/60 focus:ring-1 focus:ring-primary/40 outline-none transition-all" />
              </div>

              <!-- Action Controls -->
              <div class="flex items-center justify-end gap-2 pt-1 border-t border-outline-variant/10">
                <button type="button" onclick="cancelEditTask('${escHtml(taskId)}')"
                  class="px-3 py-1 rounded-md text-[11px] font-medium text-outline-variant hover:text-on-surface hover:bg-surface-container-high transition-colors">
                  Cancel
                </button>
                <button type="button" onclick="saveEditTask('${escHtml(taskId)}')"
                  class="flex items-center gap-1 px-3.5 py-1 rounded-md text-[11px] font-semibold bg-primary text-on-primary shadow-sm hover:brightness-110 active:scale-95 transition-all">
                  <span class="material-symbols-outlined text-xs">check</span>
                  <span>Done</span>
                </button>
              </div>
            </div>
          </div>
        `;
      }

      // --- DEFAULT READ VIEW (PRESERVED AESTHETIC) ---
      return `
        <div id="roadmap-card-${escHtml(taskId)}" class="roadmap-card flex items-stretch gap-0 rounded-xl border border-outline-variant/15 bg-surface-container-lowest/80 overflow-hidden group">
          <!-- Left Status Indicator Bar -->
          <span class="w-1.5 shrink-0 ${cfg.barClass}"></span>

          <!-- Card Content -->
          <div class="flex-1 p-4 min-w-0 flex flex-col gap-2.5">
            <!-- Header row: ID, Status, Assignee, Parent Dep, and Actions -->
            <div class="flex flex-wrap items-center justify-between gap-2">
              <div class="flex flex-wrap items-center gap-2">
                <!-- Task ID -->
                <span class="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-surface-container-high text-on-surface border border-outline-variant/20">
                  ${escHtml(taskId)}
                </span>

                <!-- Status Badge (Clickable cycle) -->
                <button type="button" onclick="cycleTaskStatus('${escHtml(taskId)}')"
                  title="Click to toggle status"
                  class="flex items-center gap-1.5 text-[10px] font-semibold px-2 py-0.5 rounded-full border transition-all hover:scale-105 ${cfg.badgeClass}">
                  <span class="material-symbols-outlined text-[13px] ${cfg.spin ? 'animate-spin' : ''}">${cfg.icon}</span>
                  <span>${cfg.label}</span>
                </button>

                <!-- Assignee Agent Chip -->
                <span class="flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded-full bg-surface-container-high/80 border border-outline-variant/15 text-on-surface">
                  <span class="material-symbols-outlined text-xs text-primary">${agentIcon}</span>
                  <span>${escHtml(agentName)}</span>
                </span>

                <!-- Parent Dependency -->
                ${task.parent_id ? `
                  <button type="button" onclick="highlightParentTask('${escHtml(task.parent_id)}')"
                    title="Jump to prerequisite task"
                    class="flex items-center gap-1 text-[9px] font-mono px-2 py-0.5 rounded-full bg-primary/5 text-primary border border-primary/20 hover:bg-primary/10 transition-colors">
                    <span class="material-symbols-outlined text-[11px]">arrow_upward</span>
                    <span>Depends on: ${escHtml(task.parent_id)}</span>
                  </button>
                ` : ''}
              </div>

              <!-- Action Menu: Move Up, Move Down, Edit, Delete -->
              <div class="flex items-center gap-0.5 opacity-40 group-hover:opacity-100 transition-opacity">
                ${idx > 0 ? `
                  <button type="button" onclick="moveRoadmapTask('${escHtml(taskId)}', 'up')"
                    class="p-1 rounded text-outline-variant hover:text-primary hover:bg-surface-container-high transition-colors"
                    title="Move up">
                    <span class="material-symbols-outlined text-xs">arrow_upward</span>
                  </button>
                ` : ''}
                ${idx < currentTasks.length - 1 ? `
                  <button type="button" onclick="moveRoadmapTask('${escHtml(taskId)}', 'down')"
                    class="p-1 rounded text-outline-variant hover:text-primary hover:bg-surface-container-high transition-colors"
                    title="Move down">
                    <span class="material-symbols-outlined text-xs">arrow_downward</span>
                  </button>
                ` : ''}
                <button type="button" onclick="startEditTask('${escHtml(taskId)}')"
                  class="p-1 rounded text-outline-variant hover:text-primary hover:bg-surface-container-high transition-colors"
                  title="Edit task">
                  <span class="material-symbols-outlined text-sm">edit</span>
                </button>
                <button type="button" onclick="deleteRoadmapTask('${escHtml(taskId)}')"
                  class="p-1 rounded text-outline-variant hover:text-error hover:bg-error/10 transition-colors"
                  title="Delete task">
                  <span class="material-symbols-outlined text-sm">delete</span>
                </button>
              </div>
            </div>

            <!-- Task Title -->
            <div class="text-sm font-semibold text-on-surface leading-snug tracking-tight">
              ${escHtml(task.title || 'Untitled task')}
            </div>

            <!-- Task Description -->
            ${desc ? `
              <div class="relative">
                <div id="desc-${escHtml(taskId)}" class="text-xs text-on-surface-variant leading-relaxed font-sans whitespace-pre-line break-words roadmap-desc-clamp ${isExpanded || !isLongDesc ? 'expanded' : ''}">
                  ${escHtml(desc)}
                </div>
                ${isLongDesc ? `
                  <button type="button" onclick="toggleDescExpand('${escHtml(taskId)}', this)"
                    class="mt-1 text-[10px] font-semibold text-primary hover:underline flex items-center gap-0.5">
                    <span>${isExpanded ? 'Show less' : 'Show more'}</span>
                    <span class="material-symbols-outlined text-xs">${isExpanded ? 'expand_less' : 'expand_more'}</span>
                  </button>
                ` : ''}
              </div>
            ` : ''}

            <!-- Task Notes / Details -->
            ${task.notes ? `
              <div class="flex items-start gap-1.5 text-[10px] text-outline-variant/80 bg-surface-container-high/40 border border-outline-variant/10 rounded-md px-2.5 py-1.5">
                <span class="material-symbols-outlined text-xs text-primary/70 shrink-0 mt-0.5">info</span>
                <span class="italic leading-tight whitespace-pre-line break-words">${escHtml(task.notes)}</span>
              </div>
            ` : ''}
          </div>
        </div>
      `;
    }).join('');
  }

  // ── Task Actions & In-Place Editing ──────────────────────────────────────
  function addNewRoadmapTask() {
    // If another task is in edit mode, commit it first
    if (editingTaskId) {
      const ok = saveEditTask(editingTaskId);
      if (!ok) return;
    }

    // Determine next unique task id
    let maxNum = 0;
    currentTasks.forEach(t => {
      const match = String(t.id || '').match(/TASK-(\d+)/i);
      if (match) {
        const num = parseInt(match[1], 10);
        if (num > maxNum) maxNum = num;
      }
    });
    const nextId = `TASK-${maxNum + 1}`;
    const prevTaskId = currentTasks.length > 0 ? (currentTasks[currentTasks.length - 1].id || null) : null;

    const newTask = {
      id: nextId,
      title: '',
      assignee: 'ResearchAgent',
      status: 'TODO',
      parent_id: prevTaskId,
      description: '',
      notes: '',
    };

    currentTasks.push(newTask);
    editingTaskId = nextId;
    isNewTaskDraft = true;
    expandedTaskIds.add(nextId);
    activeFilter = 'all'; // ensure newly added task is visible

    updateProgressAndStats();
    renderRoadmapVisual();

    setTimeout(() => {
      const card = document.getElementById(`roadmap-card-${nextId}`);
      if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'center' });
        const input = document.getElementById(`edit-title-${nextId}`);
        if (input) input.focus();
      }
    }, 60);
  }

  function startEditTask(taskId) {
    if (editingTaskId && editingTaskId !== taskId) {
      const ok = saveEditTask(editingTaskId);
      if (!ok) return;
    }
    editingTaskId = taskId;
    isNewTaskDraft = false;
    expandedTaskIds.add(taskId);
    renderRoadmapVisual();

    setTimeout(() => {
      const input = document.getElementById(`edit-title-${taskId}`);
      if (input) input.focus();
    }, 60);
  }

  function cancelEditTask(taskId) {
    if (isNewTaskDraft && editingTaskId === taskId) {
      const task = currentTasks.find(t => (t.id || '') === taskId);
      if (!task || !task.title || !task.title.trim()) {
        currentTasks = currentTasks.filter(t => (t.id || '') !== taskId);
        expandedTaskIds.delete(taskId);
      }
    }
    editingTaskId = null;
    isNewTaskDraft = false;
    updateProgressAndStats();
    renderRoadmapVisual();
  }

  function saveEditTask(taskId) {
    const task = currentTasks.find(t => (t.id || '') === taskId);
    if (!task) return true;

    const titleEl = document.getElementById(`edit-title-${taskId}`);
    const descEl = document.getElementById(`edit-desc-${taskId}`);
    const assigneeEl = document.getElementById(`edit-assignee-${taskId}`);
    const statusEl = document.getElementById(`edit-status-${taskId}`);
    const parentEl = document.getElementById(`edit-parent-${taskId}`);
    const notesEl = document.getElementById(`edit-notes-${taskId}`);

    const title = titleEl ? titleEl.value.trim() : (task.title || '');
    if (!title) {
      if (titleEl) {
        titleEl.focus();
        titleEl.classList.add('border-error');
      }
      alert('Task title cannot be empty.');
      return false;
    }

    task.title = title;
    if (descEl) task.description = descEl.value.trim();
    if (assigneeEl) task.assignee = assigneeEl.value.trim() || 'ResearchAgent';
    if (statusEl) task.status = statusEl.value;
    if (parentEl) task.parent_id = parentEl.value ? parentEl.value : null;
    if (notesEl) {
      const val = notesEl.value.trim();
      if (val) task.notes = val;
      else delete task.notes;
    }

    editingTaskId = null;
    isNewTaskDraft = false;
    updateProgressAndStats();
    renderRoadmapVisual();

    if (window.StatusIndicator) {
      StatusIndicator.feed({ type: 'tasks_updated', tasks: currentTasks });
    }
    return true;
  }

  function moveRoadmapTask(taskId, direction) {
    const idx = currentTasks.findIndex(t => (t.id || '') === taskId);
    if (idx < 0) return;

    if (direction === 'up' && idx > 0) {
      const temp = currentTasks[idx];
      currentTasks[idx] = currentTasks[idx - 1];
      currentTasks[idx - 1] = temp;
    } else if (direction === 'down' && idx < currentTasks.length - 1) {
      const temp = currentTasks[idx];
      currentTasks[idx] = currentTasks[idx + 1];
      currentTasks[idx + 1] = temp;
    } else {
      return;
    }

    renderRoadmapVisual();
    if (window.StatusIndicator) {
      StatusIndicator.feed({ type: 'tasks_updated', tasks: currentTasks });
    }
  }

  function cycleTaskStatus(taskId) {
    const task = currentTasks.find(t => (t.id || '') === taskId);
    if (!task) return;
    const norm = normalizeStatus(task.status);
    const order = ['todo', 'in_progress', 'done'];
    const next = order[(order.indexOf(norm) + 1) % order.length];

    task.status = next === 'done' ? 'DONE' : next === 'in_progress' ? 'IN_PROGRESS' : 'TODO';
    updateProgressAndStats();
    renderRoadmapVisual();
    // Sync with status indicator
    if (window.StatusIndicator) {
      StatusIndicator.feed({ type: 'tasks_updated', tasks: currentTasks });
    }
  }

  function toggleDescExpand(taskId, btn) {
    if (expandedTaskIds.has(taskId)) {
      expandedTaskIds.delete(taskId);
    } else {
      expandedTaskIds.add(taskId);
    }
    const isExp = expandedTaskIds.has(taskId);
    const descEl = document.getElementById(`desc-${taskId}`);
    if (descEl) {
      if (isExp) descEl.classList.add('expanded');
      else descEl.classList.remove('expanded');
    }
    const button = btn || (descEl && descEl.parentElement ? descEl.parentElement.querySelector('button') : null);
    if (button) {
      button.innerHTML = isExp
        ? `<span>Show less</span><span class="material-symbols-outlined text-xs">expand_less</span>`
        : `<span>Show more</span><span class="material-symbols-outlined text-xs">expand_more</span>`;
    }
  }

  function toggleAllDescExpand(expandAll) {
    currentTasks.forEach(t => {
      const id = t.id || '';
      if (id) {
        if (expandAll) expandedTaskIds.add(id);
        else expandedTaskIds.delete(id);
      }
    });
    renderRoadmapVisual();
  }

  function highlightParentTask(parentId) {
    const targetCard = document.getElementById(`roadmap-card-${parentId}`);
    if (targetCard) {
      targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
      targetCard.classList.remove('highlighted');
      void targetCard.offsetWidth; // trigger reflow
      targetCard.classList.add('highlighted');
    } else {
      alert(`Prerequisite task '${parentId}' not found in the plan.`);
    }
  }

  function deleteRoadmapTask(taskId) {
    if (!confirm(`Delete task ${taskId}?`)) return;
    if (editingTaskId === taskId) {
      editingTaskId = null;
      isNewTaskDraft = false;
    }
    expandedTaskIds.delete(taskId);
    currentTasks = currentTasks.filter(t => (t.id || '') !== taskId);
    updateProgressAndStats();
    renderRoadmapVisual();
    if (window.StatusIndicator) {
      StatusIndicator.feed({ type: 'tasks_updated', tasks: currentTasks });
    }
  }

  function filterRoadmapTasks(filter) {
    activeFilter = filter;
    document.querySelectorAll('#roadmap-filter-tabs .roadmap-tab-btn').forEach(btn => {
      const f = btn.dataset.filter;
      if (f === filter) {
        btn.className = 'roadmap-tab-btn active px-2.5 py-1 rounded-md bg-surface-container-highest text-primary font-semibold border border-primary/20 transition-all';
      } else {
        btn.className = 'roadmap-tab-btn px-2.5 py-1 rounded-md text-outline-variant hover:text-on-surface transition-all';
      }
    });
    renderRoadmapVisual();
  }

  function searchRoadmapTasks(query) {
    searchQuery = (query || '').trim();
    renderRoadmapVisual();
  }

  // ── Save & Confirm Operations ────────────────────────────────────────────
  async function prepareTasksForSave() {
    if (editingTaskId) {
      saveEditTask(editingTaskId);
    }
    return currentTasks;
  }

  async function saveRoadmap() {
    if (isSaving) return;
    const status = document.getElementById('roadmap-status');
    const saveBtn = document.getElementById('save-roadmap-btn');

    try {
      const tasks = await prepareTasksForSave();
      isSaving = true;
      if (saveBtn) saveBtn.disabled = true;

      if (status) {
        status.classList.remove('hidden');
        status.textContent = 'Saving changes…';
        status.className = 'text-[10px] font-mono text-primary/70 animate-pulse';
      }

      const response = await fetch(roadmapUrl(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: JSON.stringify(tasks, null, 2) }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      if (data.status === 'success') {
        if (status) {
          status.textContent = 'Roadmap saved successfully.';
          status.className = 'text-[10px] font-mono text-secondary';
        }
        addTelemetry('ROADMAP :: saved successfully');

        // Instantly notify StatusIndicator
        if (window.StatusIndicator) {
          StatusIndicator.feed({ type: 'tasks_updated', tasks: tasks });
        }

        setTimeout(closeRoadmapEditor, 800);
      } else {
        throw new Error(data.error || 'Failed to save roadmap');
      }
    } catch (error) {
      console.error('Failed to save roadmap:', error);
      if (status) {
        status.textContent = 'Error saving roadmap: ' + error.message;
        status.className = 'text-[10px] font-mono text-error';
      }
    } finally {
      isSaving = false;
      if (saveBtn) saveBtn.disabled = false;
    }
  }

  function updateRoadmapModalButtons() {
    const confirmBtn = document.getElementById('confirm-roadmap-btn');
    const reviseBtn = document.getElementById('revise-roadmap-btn');
    const feedbackContainer = document.getElementById('roadmap-feedback-container');
    if (!confirmBtn) return;
    if (window.currentPlannerHitlRequest) {
      confirmBtn.classList.remove('hidden');
      if (reviseBtn) reviseBtn.classList.remove('hidden');
      if (feedbackContainer) feedbackContainer.classList.remove('hidden');
    } else {
      confirmBtn.classList.add('hidden');
      if (reviseBtn) reviseBtn.classList.add('hidden');
      if (feedbackContainer) feedbackContainer.classList.add('hidden');
    }
  }

  async function reviseRoadmap() {
    if (!window.currentPlannerHitlRequest) {
      alert('No pending roadmap confirmation request found.');
      return;
    }

    const feedbackInput = document.getElementById('roadmap-feedback-input');
    const status = document.getElementById('roadmap-status');
    const feedback = feedbackInput ? feedbackInput.value.trim() : '';

    if (status) {
      status.classList.remove('hidden');
      status.textContent = 'Sending revision request…';
      status.className = 'text-[10px] font-mono text-primary/70 animate-pulse';
    }

    try {
      const tasks = await prepareTasksForSave();
      const response = await fetch(roadmapUrl(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: JSON.stringify(tasks, null, 2) }),
      });

      if (response.ok) {
        const reqId = window.currentPlannerHitlRequest.request_id;
        if (window.ws && window.ws.readyState === 1) {
          window.ws.send(JSON.stringify({
            type: 'hitl_response',
            request_id: reqId,
            action: 'edit',
            approved: false,
            instructions: feedback || null,
            free_input: feedback || null,
          }));
        }
        if (window.StatusIndicator) {
          StatusIndicator.feed({ type: 'hitl_response', request_id: reqId });
          StatusIndicator.feed({ type: 'tasks_updated', tasks: tasks });
        }

        const hitlPanel = document.getElementById('hitl-panel');
        if (hitlPanel) hitlPanel.classList.add('hidden');
        addSystemMsg('✗ HITL Sent for Revision' + (feedback ? ': ' + feedback : ''));

        window.currentPlannerHitlRequest = null;
        updateRoadmapModalButtons();
        if (status) {
          status.textContent = 'Revision requested.';
          status.className = 'text-[10px] font-mono text-error';
        }
        setTimeout(closeRoadmapEditor, 900);
      }
    } catch (error) {
      console.error('Error revising roadmap:', error);
      if (status) {
        status.textContent = 'Error: ' + error.message;
        status.className = 'text-[10px] font-mono text-error';
      }
    }
  }

  async function saveAndConfirmRoadmap() {
    if (!window.currentPlannerHitlRequest) {
      alert('No pending roadmap confirmation request found.');
      return;
    }

    const status = document.getElementById('roadmap-status');
    const saveBtn = document.getElementById('save-roadmap-btn');
    const confirmBtn = document.getElementById('confirm-roadmap-btn');

    if (status) {
      status.classList.remove('hidden');
      status.textContent = 'Saving and confirming plan…';
      status.className = 'text-[10px] font-mono text-primary/70 animate-pulse';
    }
    if (saveBtn) saveBtn.disabled = true;
    if (confirmBtn) confirmBtn.disabled = true;

    try {
      const tasks = await prepareTasksForSave();
      const response = await fetch(roadmapUrl(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: JSON.stringify(tasks, null, 2) }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      if (data.status === 'success') {
        if (status) {
          status.textContent = 'Roadmap saved and confirmed.';
          status.className = 'text-[10px] font-mono text-secondary';
        }
        addTelemetry('ROADMAP :: saved successfully');

        const reqId = window.currentPlannerHitlRequest.request_id;
        respondHITL(reqId, true);

        if (window.StatusIndicator) {
          StatusIndicator.feed({ type: 'tasks_updated', tasks: tasks });
        }

        setTimeout(closeRoadmapEditor, 900);
      } else {
        throw new Error(data.error || 'Unknown error');
      }
    } catch (error) {
      console.error('Failed to save and confirm roadmap:', error);
      if (status) {
        status.textContent = 'Error: ' + error.message;
        status.className = 'text-[10px] font-mono text-error';
      }
    } finally {
      if (saveBtn) saveBtn.disabled = false;
      if (confirmBtn) confirmBtn.disabled = false;
    }
  }

  // ── Real-Time WebSocket & State Synchronisation ───────────────────────────
  function updateTasks(newTasks, notifyIndicator = false) {
    if (!Array.isArray(newTasks)) return;

    const prevMap = new Map((currentTasks || []).map(t => [t.id, t]));

    if (editingTaskId) {
      // If user is currently editing a task, avoid clobbering active edits
      const updatedMap = new Map(newTasks.map(t => [t.id, t]));
      currentTasks = currentTasks.map(t => {
        if (t.id === editingTaskId) return t; // preserve edited card
        const updated = updatedMap.get(t.id);
        if (!updated) return t;
        return {
          ...t,
          ...updated,
          description: (updated.description !== undefined && updated.description !== null && updated.description !== '')
            ? updated.description
            : (t.description || ''),
          notes: (updated.notes !== undefined && updated.notes !== null && updated.notes !== '')
            ? updated.notes
            : (t.notes || ''),
        };
      });
      newTasks.forEach(t => {
        if (!currentTasks.some(ct => ct.id === t.id)) {
          currentTasks.push(t);
        }
      });
    } else {
      currentTasks = newTasks.map(t => {
        const prev = prevMap.get(t.id) || {};
        return {
          ...prev,
          ...t,
          description: (t.description !== undefined && t.description !== null && t.description !== '')
            ? t.description
            : (prev.description || ''),
          notes: (t.notes !== undefined && t.notes !== null && t.notes !== '')
            ? t.notes
            : (prev.notes || ''),
        };
      });
    }

    updateProgressAndStats();
    renderRoadmapVisual();

    if (notifyIndicator && window.StatusIndicator) {
      StatusIndicator.feed({ type: 'tasks_updated', tasks: currentTasks });
    }
  }

  function handleSingleTaskUpdate(updatedTask, notifyIndicator = false) {
    if (!updatedTask || !updatedTask.id) return;
    // Do not overwrite task if currently being edited
    if (editingTaskId && editingTaskId === updatedTask.id) return;

    const idx = currentTasks.findIndex(t => (t.id || '') === (updatedTask.id || ''));
    if (idx >= 0) {
      const prev = currentTasks[idx];
      currentTasks[idx] = {
        ...prev,
        ...updatedTask,
        description: (updatedTask.description !== undefined && updatedTask.description !== null && updatedTask.description !== '')
          ? updatedTask.description
          : (prev.description || ''),
        notes: (updatedTask.notes !== undefined && updatedTask.notes !== null && updatedTask.notes !== '')
          ? updatedTask.notes
          : (prev.notes || ''),
      };
    } else {
      currentTasks.push(updatedTask);
    }
    updateProgressAndStats();
    renderRoadmapVisual();

    if (notifyIndicator && window.StatusIndicator) {
      StatusIndicator.feed({ type: 'tasks_updated', tasks: currentTasks });
    }
  }

  function feed(data) {
    if (!data) return;
    if (data.type === 'tool_activity' && data.phase === 'result') {
      const tool = String(data.tool || '');
      if (/create_plan/i.test(tool)) {
        let plan = data.result && (data.result.plan || data.result.tasks);
        if (!plan && typeof data.result === 'string') {
          try {
            const parsed = JSON.parse(data.result);
            plan = parsed.plan || parsed.tasks;
          } catch (_) {}
        }
        if (Array.isArray(plan)) {
          updateTasks(plan, false);
        }
      } else if (/task_status|update_task|add_task/i.test(tool)) {
        let task = data.result && data.result.task;
        if (!task && typeof data.result === 'string') {
          try {
            const parsed = JSON.parse(data.result);
            task = parsed.task;
          } catch (_) {}
        }
        if (task && task.id) {
          handleSingleTaskUpdate(task, false);
        }
      }
    } else if (data.type === 'session_snapshot') {
      if (Array.isArray(data.active_tasks)) {
        updateTasks(data.active_tasks, false);
      }
    } else if (data.type === 'tasks_updated') {
      if (Array.isArray(data.tasks)) {
        updateTasks(data.tasks, false);
      }
    }
  }

  // ── Global Window Exports ────────────────────────────────────────────────
  window.RoadmapModal = {
    feed: feed,
    updateTasks: updateTasks,
    handleSingleTaskUpdate: handleSingleTaskUpdate,
    getTasks: () => currentTasks,
  };
  window.openRoadmapEditor = openRoadmapEditor;
  window.closeRoadmapEditor = closeRoadmapEditor;
  window.saveRoadmap = saveRoadmap;
  window.reviseRoadmap = reviseRoadmap;
  window.saveAndConfirmRoadmap = saveAndConfirmRoadmap;
  window.updateRoadmapModalButtons = updateRoadmapModalButtons;
  window.cycleTaskStatus = cycleTaskStatus;
  window.toggleDescExpand = toggleDescExpand;
  window.toggleAllDescExpand = toggleAllDescExpand;
  window.highlightParentTask = highlightParentTask;
  window.deleteRoadmapTask = deleteRoadmapTask;
  window.filterRoadmapTasks = filterRoadmapTasks;
  window.searchRoadmapTasks = searchRoadmapTasks;
  window.addNewRoadmapTask = addNewRoadmapTask;
  window.startEditTask = startEditTask;
  window.cancelEditTask = cancelEditTask;
  window.saveEditTask = saveEditTask;
  window.moveRoadmapTask = moveRoadmapTask;
  // Deprecated stubs to avoid reference errors
  window.switchRoadmapView = () => renderRoadmapVisual();
  window.formatRoadmapYaml = () => {};
  window.copyRoadmapYaml = () => {};
  window.handleRoadmapYamlInput = () => {};
})();
