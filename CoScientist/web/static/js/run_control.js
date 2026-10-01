// Durable pause/resume and explicit budget renewal. Never auto-approve a quota.
(() => {
  let current = null;
  let busy = false;
  let attentionKey = '';
  const label = (ru, en) => typeof currentLang !== 'undefined' && currentLang === 'en' ? en : ru;
  const rootUrl = () => `/api/users/${encodeURIComponent(activeUser.id)}/sessions/${encodeURIComponent(activeSession.id)}`;

  function render() {
    const panel = document.getElementById('run-control-panel');
    const pause = document.getElementById('pause-run-btn');
    const resume = document.getElementById('resume-run-btn');
    if (!panel) return;
    panel.classList.toggle('hidden', !current);
    if (!current) {
      if (pause) pause.classList.add('hidden');
      if (resume) resume.classList.add('hidden');
      return;
    }
    const causes = current.pause_causes || [];
    const paused = causes.length > 0 || ['paused', 'pause_requested'].includes(current.state);
    const terminal = ['completed', 'stopped'].includes(current.state);
    const needsBudget = causes.includes('budget_exhausted');
    const stalled = current.pending_decisions?.execution_error?.kind === 'semantic_control_loop_guard';
    const decisions = current.pending_decisions || {};
    const planRecovery = Object.values(decisions).find(item => item.kind === 'experiment_plan_recovery');
    const capacity = Object.values(decisions).find(item => item.kind === 'plan_capacity_conflict');
    const roadmap = Object.values(decisions).find(item => item.kind === 'scientific_outcome'
      && ['root_roadmap_incomplete', 'root_roadmap_not_successful'].includes(item.reason));
    const mayResume = paused && causes.every(cause => ['manual', 'server_restart', 'execution_error'].includes(cause)
      || decisions[cause]?.kind === 'experiment_plan_recovery');
    if (pause) {
      pause.classList.toggle('hidden', paused || terminal);
      pause.disabled = busy;
      pause.textContent = label('Пауза', 'Pause');
    }
    if (resume) {
      resume.classList.toggle('hidden', !mayResume || terminal);
      resume.disabled = busy;
      resume.textContent = planRecovery ? label('Согласовать сохранённый план', 'Review saved plan') : label('Продолжить', 'Resume');
    }
    const budget = current.budget || {};
    const title = needsBudget ? label('Бюджет обращений исчерпан', 'Model-call quota exhausted')
      : paused ? label('Исследование на паузе', 'Research paused')
      : current.state === 'completed' && current.disposition?.kind === 'completed_limited' ? label('Завершено с ограничениями', 'Completed with limitations')
      : current.state === 'completed' && current.disposition?.kind === 'completed_negative' ? label('Завершено: критерий не достигнут', 'Completed: target not met')
      : current.state === 'completed' ? label('Цикл завершён', 'Run finished')
      : current.state === 'stopped' ? label('Остановлено пользователем', 'Stopped')
      : label('Исследование выполняется', 'Research in progress');
    let html = `<div class="flex flex-col items-start gap-1"><strong>${escHtml(title)}</strong>`
      + `<span class="tabular-nums">${escHtml(label('Обращения к моделям', 'Model calls'))}: ${Number(budget.used || 0)} / ${Number(budget.limit || 100)}</span></div>`
      + `<p class="mt-1 text-outline-variant">${escHtml(label('Внутренние обращения удалённых агентов в этот лимит не входят.', 'Internal calls of remote agents are outside this quota.'))}</p>`;
    if (paused) html += `<p class="mt-1">${escHtml(label('Прогресс сохранён. Уже запущенные внешние задания могут продолжать работу.', 'Progress is saved. Already dispatched external jobs may still be running.'))}</p>`;
    if (planRecovery) html += `<p class="mt-2">${escHtml(label('Автоматические правки плана исчерпаны. Можно повторно открыть согласование сохранённого выполнимого варианта. План не генерируется заново, счётчики не сбрасываются.', 'Automatic plan revisions are exhausted. Reopen review of the saved executable candidate without regenerating the plan or resetting counters.'))}</p>`;
    if (capacity) {
      html += `<p class="mt-2">${escHtml(label('Обязательные операции не помещаются в лимит задач плана.', 'Required operations exceed the plan task limit.'))}</p>`;
      if (capacity.can_raise_limit) html += `<p class="mt-1">${escHtml(label('Текущий лимит:', 'Current limit:'))} ${Number(capacity.max_plan_tasks || capacity.current_max_plan_tasks || 0)}; ${escHtml(label('запрошено:', 'requested:'))} ${Number(capacity.requested_max_plan_tasks || 0)}.</p>
        <button type="button" data-approve-capacity class="mt-2 px-3 py-1.5 rounded border border-primary/40">${escHtml(label('Разрешить лимит задач для этого плана', 'Approve this plan’s task limit'))}</button>
        <p class="mt-1 text-outline-variant">${escHtml(label('Общий бюджет обращений к моделям не увеличится.', 'The model-call budget will not increase.'))}</p>`;
      else html += `<p class="mt-1">${escHtml(label('Превышен максимум 20 задач. Остановите текущий запуск и согласуйте разделение исследования на этапы или изменение объёма. Операции не удалены.', 'The hard 20-task limit is exceeded. Stop this run and agree on splitting the research into stages or revising scope. No operations were discarded.'))}</p>`;
    }
    if (roadmap) {
      html += `<p class="mt-2">${escHtml(label('В исследовании остались незавершённые пункты:', 'Outstanding research items:'))} ${escHtml((roadmap.unfinished_task_ids || []).join(', '))}</p>
        <div class="flex flex-wrap gap-2 mt-2"><button type="button" data-outcome-continue class="px-3 py-1.5 rounded border border-primary/40">${escHtml(label('Продолжить оставшиеся задачи', 'Continue outstanding tasks'))}</button>
        <button type="button" data-outcome-limited class="px-3 py-1.5 rounded border border-outline-variant/40">${escHtml(label('Принять неполный результат', 'Accept limited outcome'))}</button></div>
        <p class="mt-1 text-outline-variant">${escHtml(label('Принятие неполного результата не пометит эти задачи выполненными. В отчёте останутся ограничения.', 'Accepting a limited outcome does not mark these tasks done. The report will retain the limitations.'))}</p>`;
    }
    if (stalled) html += `<p class="mt-2">${escHtml(label('Управляющие действия перестали продвигать исследование. Можно завершить невыполненные задачи с указанием ограничений и перейти к оценке результатов. Успешные результаты сохранятся; повтор задач выбирается отдельно на этапе оценки.', 'Control actions stopped making progress. You can close unfinished tasks with documented limitations and review the results. Successful results are retained; any redo is selected separately during review.'))}</p>
      <button type="button" data-finish-stalled class="mt-2 px-3 py-1.5 rounded border border-primary/40">${escHtml(label('Завершить с ограничениями и оценить результаты', 'Finish with limitations and review results'))}</button>`;
    if (needsBudget) {
      html += `<div class="flex flex-wrap gap-2 mt-2"><button type="button" data-budget-continue class="px-3 py-1.5 rounded border border-primary/40 text-primary">${escHtml(label('Продолжить: ещё 100 обращений', 'Continue: 100 more calls'))}</button>`
        + `<button type="button" data-budget-stop class="px-3 py-1.5 rounded border border-error/40 text-error">${escHtml(label('Остановить', 'Stop'))}</button></div>`;
    } else if (causes.some(cause => cause.startsWith('hitl:'))) {
      html += `<p class="mt-2">${escHtml(label('Ответьте на сохранённый запрос согласования в чате.', 'Answer the pending approval request in the chat.'))}</p>`;
    } else if (causes.includes('unknown_completion')) {
      html += `<p class="mt-2 text-error">${escHtml(label('Исход внешнего действия неизвестен. Автоматический повтор заблокирован; требуется проверка задания.', 'An external action has an unknown outcome. Automatic replay is blocked; inspect the job first.'))}</p>`;
      const actions = current.pending_decisions?.unknown_completion?.actions || [];
      html += actions.map(action => `<div class="mt-2 border border-outline-variant/30 rounded p-2">
        <p>${escHtml(action.tool || action.action_id)}</p>
        <p class="text-outline-variant">${escHtml(label('Сначала проверьте внешнее задание. Повтор может продублировать действие; работающий процесс повторять нельзя.', 'Inspect the remote job first. A retry can duplicate work; do not retry a job that is still running.'))}</p>
        <textarea data-inspection-notes="${escHtml(action.action_id)}" rows="2" aria-label="${escHtml(label('Результат проверки', 'Inspection notes'))}" class="w-full mt-1 p-2 bg-surface-container-lowest rounded"></textarea>
        <button type="button" data-authorize-retry="${escHtml(action.action_id)}" class="mt-1 px-3 py-1.5 rounded border border-error/40">${escHtml(label('Проверено: разрешить повтор', 'Inspected: authorize retry'))}</button>
      </div>`).join('');
    }
    panel.innerHTML = html;
    panel.querySelector('[data-budget-continue]')?.addEventListener('click', () => command('budget/decision', {decision: 'continue', decision_id: crypto.randomUUID()}));
    panel.querySelector('[data-budget-stop]')?.addEventListener('click', () => command('budget/decision', {decision: 'stop', decision_id: crypto.randomUUID()}));
    panel.querySelector('[data-finish-stalled]')?.addEventListener('click', () => command('recovery/finish-with-limits', {confirmed: true}));
    const decideOutcome = (decision, extra = {}) => command('outcome/decision', {
      decision, decision_id: crypto.randomUUID(), confirmed: true, ...extra,
    });
    panel.querySelector('[data-approve-capacity]')?.addEventListener('click', () => decideOutcome('approve_capacity'));
    panel.querySelector('[data-outcome-continue]')?.addEventListener('click', () => decideOutcome('continue'));
    panel.querySelector('[data-outcome-limited]')?.addEventListener('click', () => decideOutcome('accept_limited', {
      accepted_item_ids: roadmap?.unfinished_task_ids || [],
    }));
    panel.querySelectorAll('[data-authorize-retry]').forEach(button => button.addEventListener('click', () => {
      const actionId = button.dataset.authorizeRetry;
      const notes = [...panel.querySelectorAll('[data-inspection-notes]')]
        .find(field => field.dataset.inspectionNotes === actionId)?.value.trim();
      if (!notes) { addSystemMsg(label('Опишите результат проверки внешнего задания.', 'Describe your inspection of the remote job.')); return; }
      command('recovery/authorize-retry', {action_id: actionId, notes, confirmed: true});
    }));
    panel.querySelectorAll('button').forEach(button => { button.disabled = busy; });
  }

  function apply(payload, {snapshot = false} = {}) {
    if (!snapshot && current && payload && payload.run_id === current.run_id
        && Number(payload.revision) < Number(current.revision)) return;
    const previous = current;
    current = payload || null;
    render();
    // Ordinary counter updates respect the folded spend panel. A new budget
    // or recovery decision must expose its controls, even when it was closed.
    const attentionCauses = (current?.pause_causes || []).filter(cause =>
      ['budget_exhausted', 'unknown_completion', 'planning_review', 'result_review', 'scientific_work_incomplete'].includes(cause)
      || (cause === 'execution_error'
        && current.pending_decisions?.execution_error?.kind === 'semantic_control_loop_guard'));
    const nextAttentionKey = attentionCauses.length
      ? `${current.run_id}:${current.budget?.limit}:${attentionCauses.join(',')}` : '';
    if (nextAttentionKey && nextAttentionKey !== attentionKey) {
      const usage = document.getElementById('usage-panel');
      if (usage) usage.open = true;
    }
    attentionKey = nextAttentionKey;
    if (!current) return;
    if (window.StatusIndicator) StatusIndicator.feed({...current, type: 'run_control'});
    const paused = (current.pause_causes || []).length > 0;
    const status = paused ? 'paused' : current.state === 'running' ? 'processing' : 'idle';
    const previousStatus = !previous ? null : (previous.pause_causes || []).length
      ? 'paused' : previous.state === 'running' ? 'processing' : 'idle';
    if (typeof applyRunStatus === 'function'
        && (snapshot || previous?.run_id !== current.run_id || status !== previousStatus)) {
      applyRunStatus(status);
    }
    if (typeof RunTimer !== 'undefined' && current.started_at && (paused || status === 'processing')) {
      RunTimer.start(current.started_at);
    }
    if (paused) {
      document.body.dataset.runPhase = 'waiting';
      if (typeof renderStatusBadge === 'function') renderStatusBadge();
    }
  }

  async function refresh() {
    if (!activeUser || !activeSession) return;
    const userId = activeUser.id, sessionId = activeSession.id;
    const response = await fetch(`${rootUrl()}/run-control`);
    if (!response.ok) return;
    const data = await response.json();
    if (activeUser.id === userId && activeSession.id === sessionId) apply(data.run, {snapshot: true});
  }

  async function command(action, extra = {}) {
    if (!current || busy) return;
    const userId = activeUser.id, sessionId = activeSession.id;
    const runId = current.run_id;
    busy = true;
    render();
    try {
      const response = await fetch(`${rootUrl()}/runs/${encodeURIComponent(current.run_id)}/${action}`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({expected_revision: current.revision, ...extra}),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || response.statusText);
      if (activeUser?.id === userId && activeSession?.id === sessionId && current?.run_id === runId) apply(body.run);
    } catch (error) {
      addSystemMsg(error.message);
      await refresh();
    } finally {
      busy = false;
      render();
    }
  }

  window.RunControl = {
    isActive: () => !!current && ['running', 'pause_requested', 'paused'].includes(current.state),
    feed: data => data.type === 'session_snapshot' ? apply(data.run_control, {snapshot: true})
      : data.type === 'run_control' ? apply(data) : undefined,
    pause: () => command('pause'), resume: () => command('resume'), refresh,
  };
})();
