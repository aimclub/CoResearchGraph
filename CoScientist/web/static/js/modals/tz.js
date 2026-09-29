// =========================================================================
// ТЗ panel (microfluidics profile) — the structured ТЗ, section by section.
//
// Fed by `tz_snapshot` events while TZSpecAgent fills the ТЗ, and by the
// agent's HITL form request (`form.kind === 'tz'`) when the operator's turn
// comes. Sections with empty fields are listed FIRST ("ожидают заполнения
// человеком"); a section moves down to the filled ones — kept in ascending
// order — as soon as its empty fields are filled and the focus leaves it.
// Fields the operator leaves empty go to the agent; the fields it filled come
// back highlighted in the next round. The ⊘ button right of a field keeps it
// empty on purpose: it becomes «не требуется» — deliberately unconstrained,
// never handed to the agent (also the way to clear an agent value that is
// not needed); ↺ in its place takes that back.
// =========================================================================
(function () {
  'use strict';

  let snapshot = null;   // latest tz_snapshot
  let request = null;    // pending HITL request carrying the ТЗ form
  let draft = {};        // {section: {field: value}} — what the operator typed
  let stash = {};        // what a field held before it was marked «не задавать»
  let sessionId = null;

  // The value that marks a field «не требуется» — the server's, when it sent one.
  const NOT_REQUIRED_DEFAULT = 'Не требуется — без ограничения';

  const STATUS_CHIP = {
    'задано заказчиком': 'text-secondary border-secondary/30 bg-secondary/10',
    'автоподбор': 'text-on-surface-variant border-primary/20 bg-primary/5',
    'уточнено оператором': 'text-primary border-primary/30 bg-primary/10',
    'не задано': 'text-error border-error/30 bg-error/10',
    'не требуется': 'text-outline-variant border-outline-variant/30 bg-transparent italic',
    'свободный комментарий': 'text-on-surface-variant border-outline-variant/30 bg-surface-container-high',
    'рассчитывается агентом': 'text-outline-variant border-outline-variant/30 bg-surface-container-high',
    'заполнено агентом': 'text-tertiary border-tertiary/40 bg-tertiary/10',
  };

  const PHASE_TEXT = {
    filling: 'Агент заполняет ТЗ по разделам…',
    review: 'Проверка оператором',
    agent_filling: 'Агент заполняет поля, оставленные оператором…',
    final: 'ТЗ согласовано',
    stored: 'ТЗ из состояния сессии',
  };

  function esc(value) {
    if (value === null || value === undefined) return '';
    return String(value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function formMode() { return !!request; }

  // The data the panel shows: the pending form wins over the live snapshot.
  function view() {
    if (request && request.form) return request.form;
    return snapshot || null;
  }

  function editable(field) { return !field.deferred; }

  function notRequiredValue() {
    const v = view();
    return (v && v.not_required_value) || NOT_REQUIRED_DEFAULT;
  }

  function norm(value) { return String(value || '').replace(/\s+/g, ' ').trim().toLowerCase(); }

  function isNotRequired(value) {
    const text = norm(value);
    return text === norm(notRequiredValue()) || text === 'не требуется';
  }

  function draftValue(section, field) {
    const s = draft[section.title];
    return s && Object.prototype.hasOwnProperty.call(s, field.name) ? s[field.name] : '';
  }

  // In the form: a section waits for the human while any editable field is
  // empty. Outside the form: while it has «не задано» fields.
  function isAwaiting(section) {
    if (formMode()) {
      return (section.fields || []).some(f => editable(f) && !String(draftValue(section, f)).trim());
    }
    return !!section.awaiting;
  }

  function resetDraft() {
    draft = {};
    stash = {};
    const v = view();
    ((v && v.sections) || []).forEach(s => {
      draft[s.title] = {};
      (s.fields || []).forEach(f => {
        if (editable(f)) draft[s.title][f.name] = f.awaiting ? '' : String(f.value || '');
      });
    });
  }

  function emptyCount() {
    const v = view();
    let n = 0;
    ((v && v.sections) || []).forEach(s => (s.fields || []).forEach(f => {
      if (formMode() && editable(f) && !String(draftValue(s, f)).trim()) n++;
    }));
    return n;
  }

  function notRequiredCount() {
    const v = view();
    let n = 0;
    ((v && v.sections) || []).forEach(s => (s.fields || []).forEach(f => {
      if (formMode() ? (editable(f) && isNotRequired(draftValue(s, f))) : f.not_required) n++;
    }));
    return n;
  }

  // ── rendering ──────────────────────────────────────────────────────────
  function fieldHtml(section, field, fi) {
    const agent = field.agent_filled;
    const pending = field.agent_pending && !formMode();
    const form = formMode() && editable(field);
    const value = form ? draftValue(section, field) : '';
    const unset = form && isNotRequired(value);
    // In the form the chip follows the operator's choice right away.
    const status = unset ? 'не требуется' : field.status;
    const chip = STATUS_CHIP[status] || STATUS_CHIP['свободный комментарий'];
    const rowClass = unset
      ? 'border-l-2 border-outline-variant/30'
      : agent
        ? 'border-l-2 border-tertiary bg-tertiary/5'
        : (field.awaiting ? 'border-l-2 border-error/60' : 'border-l-2 border-transparent');
    let valueHtml;
    if (form) {
      const placeholder = field.awaiting
        ? 'Не задано — впишите значение; пустое заполнит агент, ⊘ справа — оставить пустым'
        : 'Пусто — заполнит агент';
      // Values already there read as text until hovered/focused; the ones
      // waiting for the operator and the agent's ones stand out.
      const look = unset
        ? 'bg-transparent border-dashed border-outline-variant/30 text-outline-variant italic'
        : agent
          ? 'bg-tertiary/5 border-tertiary/40'
          : field.awaiting
            ? 'bg-error/5 border-error/25'
            : 'bg-transparent border-transparent hover:border-outline-variant/30';
      const hint = unset
        ? 'Вернуть поле: вписать значение или отдать агенту'
        : 'Оставить поле пустым — без ограничения, агент не будет его заполнять';
      // The «leave empty» button sits right of the value, one click per field.
      valueHtml = `
        <div class="flex items-start gap-1.5">
          <textarea rows="1" data-section="${esc(section.title)}" data-field="${esc(field.name)}"
            id="tz-f-${section.num}-${fi}" placeholder="${esc(placeholder)}" ${unset ? 'readonly' : ''}
            class="tz-input flex-1 min-w-0 resize-none border ${look} rounded-md px-2 py-1 text-[11px] text-on-surface placeholder:text-outline-variant/50 focus:bg-surface-container-lowest focus:border-primary/50 focus:ring-1 focus:ring-primary/30 outline-none">${esc(value)}</textarea>
          <button type="button" data-section="${esc(section.title)}" data-field="${esc(field.name)}"
            title="${esc(hint)}" aria-label="${esc(hint)}" aria-pressed="${unset ? 'true' : 'false'}"
            class="tz-toggle-nr shrink-0 w-7 h-7 flex items-center justify-center rounded-md border transition-colors ${unset
              ? 'text-primary border-primary/40 bg-primary/10 hover:bg-primary/20'
              : 'text-outline-variant border-outline-variant/25 hover:text-error hover:border-error/40 hover:bg-error/5'}">
            <span class="material-symbols-outlined text-[16px] pointer-events-none">${unset ? 'undo' : 'block'}</span>
          </button>
        </div>`;
    } else {
      const shown = field.awaiting ? '—' : field.value;
      const muted = field.awaiting || field.not_required;
      valueHtml = `<div class="text-[11px] text-on-surface whitespace-pre-line break-words ${muted ? 'text-outline-variant italic' : ''}">${esc(shown)}</div>`;
    }
    return `
      <div class="tz-field flex flex-col gap-1 pl-2 py-1 rounded-sm ${rowClass}">
        <div class="flex flex-wrap items-center gap-1.5">
          <span class="text-[11px] font-medium text-on-surface-variant">${esc(field.name)}</span>
          <span class="text-[9px] px-1.5 py-px rounded-full border ${chip}">${esc(status)}</span>
          ${agent && !unset ? '<span class="text-[9px] font-semibold text-tertiary">● проверьте значение агента</span>' : ''}
          ${pending ? '<span class="text-[9px] font-semibold text-tertiary animate-pulse">агент заполняет…</span>' : ''}
        </div>
        ${valueHtml}
      </div>`;
  }

  function sectionHtml(section) {
    const awaiting = isAwaiting(section);
    const agentCount = (section.fields || []).filter(f => f.agent_filled).length;
    const badge = awaiting
      ? '<span class="text-[9px] font-semibold px-2 py-0.5 rounded-full text-error bg-error/10 border border-error/30">ожидает заполнения</span>'
      : '<span class="text-[9px] font-semibold px-2 py-0.5 rounded-full text-secondary bg-secondary/10 border border-secondary/30">заполнен</span>';
    const question = awaiting && section.question ? `
      <p class="text-[11px] text-on-surface leading-snug">${esc(section.question)}</p>
      ${section.hint ? `<p class="text-[10px] text-outline-variant italic">${esc(section.hint)}</p>` : ''}` : '';
    return `
      <div class="tz-section rounded-xl border ${awaiting ? 'border-error/25' : 'border-outline-variant/15'} bg-surface-container-lowest/80 p-3 flex flex-col gap-2 transition-all"
        data-section="${esc(section.title)}" data-num="${section.num}" id="tz-sec-${section.num}">
        <div class="flex flex-wrap items-center gap-2">
          <span class="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-surface-container-high text-primary border border-primary/20">${section.num}</span>
          <span class="text-xs font-semibold text-on-surface">${esc(section.title)}</span>
          ${badge}
          ${agentCount ? `<span class="text-[9px] font-semibold px-2 py-0.5 rounded-full text-tertiary bg-tertiary/10 border border-tertiary/30">заполнено агентом: ${agentCount}</span>` : ''}
        </div>
        ${section.usage ? `<p class="text-[10px] text-outline-variant">${esc(section.usage)}</p>` : ''}
        ${question}
        <div class="flex flex-col gap-1.5">${(section.fields || []).map((f, i) => fieldHtml(section, f, i)).join('')}</div>
      </div>`;
  }

  function sortedSections() {
    const v = view();
    return ((v && v.sections) || []).slice().sort((a, b) => a.num - b.num);
  }

  function render() {
    const modal = document.getElementById('tz-modal');
    if (!modal) return;
    const v = view();
    const sections = sortedSections();
    const awaiting = sections.filter(isAwaiting);
    const filled = sections.filter(s => !isAwaiting(s));

    document.getElementById('tz-group-awaiting').innerHTML = awaiting.map(sectionHtml).join('');
    document.getElementById('tz-group-filled').innerHTML = filled.map(sectionHtml).join('');
    document.getElementById('tz-empty-state').classList.toggle('hidden', sections.length > 0);
    document.getElementById('tz-awaiting-wrap').classList.toggle('hidden', !awaiting.length);
    document.getElementById('tz-filled-wrap').classList.toggle('hidden', !filled.length);

    renderHeader(v);
    updateCounters();
    modal.querySelectorAll('.tz-input').forEach(autosize);
  }

  function renderHeader(v) {
    const phase = formMode() ? 'review' : ((v && v.phase) || '');
    const badge = document.getElementById('tz-phase-badge');
    badge.textContent = PHASE_TEXT[phase] || (v ? 'ТЗ' : 'нет данных');
    badge.className = 'text-[9px] font-mono font-semibold px-2 py-0.5 rounded-full border '
      + (formMode() ? 'text-error border-error/30 bg-error/10 animate-pulse'
        : phase === 'final' ? 'text-secondary border-secondary/30 bg-secondary/10'
          : 'text-primary border-primary/20 bg-surface-container-highest');

    const bar = document.getElementById('tz-progress-bar');
    const text = document.getElementById('tz-progress-text');
    let done = 0; let total = 0; let label = '';
    if (v && v.agent_fill && !formMode()) {
      done = v.agent_fill.done; total = v.agent_fill.total;
      label = `агент дозаполнил ${done} из ${total} разделов`;
    } else if (v && v.progress) {
      done = v.progress.filled; total = v.progress.total;
      label = `заполнено разделов: ${done} из ${total}`;
    } else if (v && v.sections) {
      done = v.sections.length; total = v.sections.length;
      label = `разделов: ${total}`;
    }
    bar.style.width = total ? Math.round(100 * done / total) + '%' : '0%';
    text.textContent = label;

    const intro = document.getElementById('tz-intro');
    if (formMode()) {
      intro.innerHTML = `
        <span class="material-symbols-outlined text-error text-sm">ads_click</span>
        <span>${esc(request.form.intro || '')}</span>`;
      intro.classList.remove('hidden');
    } else {
      intro.classList.add('hidden');
    }
  }

  function updateCounters() {
    const v = view();
    const sections = (v && v.sections) || [];
    const awaiting = sections.filter(isAwaiting).length;
    const agent = sections.reduce((n, s) => n + (s.fields || []).filter(f => f.agent_filled).length, 0);
    const empty = emptyCount();
    const unset = notRequiredCount();
    document.getElementById('tz-awaiting-count').textContent = awaiting;
    document.getElementById('tz-filled-count').textContent = sections.length - awaiting;
    document.getElementById('tz-stats').textContent =
      `Разделов: ${sections.length} · ожидают: ${awaiting}`
      + (agent ? ` · заполнено агентом: ${agent}` : '')
      + (unset ? ` · не требуется: ${unset}` : '');

    const submit = document.getElementById('tz-submit-btn');
    submit.classList.toggle('hidden', !formMode());
    const leaveEmpty = document.getElementById('tz-leave-empty-btn');
    if (leaveEmpty) {
      leaveEmpty.classList.toggle('hidden', !formMode() || !empty);
      document.getElementById('tz-leave-empty-label').textContent = `Не заполнять пустые (${empty})`;
    }
    document.getElementById('tz-submit-label').textContent = empty
      ? `Отправить · пустые поля (${empty}) заполнит агент`
      : 'Подтвердить ТЗ';
  }

  function autosize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight + 2, 160) + 'px';
  }

  // Put a section card into the group it belongs to now, keeping the group
  // in ascending section order. Only this card moves, so the focus the
  // operator just moved elsewhere stays where it is.
  function placeSection(card) {
    const title = card.getAttribute('data-section');
    const section = sortedSections().find(s => s.title === title);
    if (!section) return;
    const target = document.getElementById(isAwaiting(section) ? 'tz-group-awaiting' : 'tz-group-filled');
    if (card.parentElement === target) return;
    const num = section.num;
    const before = Array.from(target.children).find(c => Number(c.getAttribute('data-num')) > num);
    const fresh = document.createElement('div');
    fresh.innerHTML = sectionHtml(section).trim();
    const replacement = fresh.firstElementChild;
    target.insertBefore(replacement, before || null);
    card.remove();
    replacement.querySelectorAll('.tz-input').forEach(autosize);
    replacement.classList.add('ring-1', 'ring-primary/40');
    setTimeout(() => replacement.classList.remove('ring-1', 'ring-primary/40'), 900);
    document.getElementById('tz-awaiting-wrap').classList.toggle(
      'hidden', !document.getElementById('tz-group-awaiting').children.length);
    document.getElementById('tz-filled-wrap').classList.toggle(
      'hidden', !document.getElementById('tz-group-filled').children.length);
    updateCounters();
  }

  function onInput(e) {
    const el = e.target;
    if (!el.classList || !el.classList.contains('tz-input')) return;
    const section = el.getAttribute('data-section');
    (draft[section] = draft[section] || {})[el.getAttribute('data-field')] = el.value;
    autosize(el);
    updateCounters();
  }

  function setNotRequired(section, field, on) {
    const values = (draft[section] = draft[section] || {});
    const key = section + '\u0000' + field;
    if (on) {
      if (!isNotRequired(values[field])) stash[key] = values[field] || '';
      values[field] = notRequiredValue();
    } else {
      values[field] = stash[key] || '';
      delete stash[key];
    }
  }

  // ⊘ (leave empty) / ↺ (take it back) on one field.
  function onClick(e) {
    const button = e.target.closest && e.target.closest('.tz-toggle-nr');
    if (!button || !formMode()) return;
    const section = button.getAttribute('data-section');
    const field = button.getAttribute('data-field');
    setNotRequired(section, field, !isNotRequired((draft[section] || {})[field]));
    render();
  }

  // Every field still empty → «не задавать». Marks only: the operator sees
  // the result and submits it with the usual button.
  function leaveEmptyUnset() {
    if (!formMode()) return;
    const v = view();
    ((v && v.sections) || []).forEach(s => (s.fields || []).forEach(f => {
      if (editable(f) && !String(draftValue(s, f)).trim()) setNotRequired(s.title, f.name, true);
    }));
    render();
  }

  function onFocusOut(e) {
    const card = e.target.closest && e.target.closest('.tz-section');
    if (!card || (e.relatedTarget && card.contains(e.relatedTarget))) return;
    placeSection(card);
  }

  // ── actions ────────────────────────────────────────────────────────────
  async function openTzPanel() {
    const modal = document.getElementById('tz-modal');
    if (!modal) return;
    modal.classList.remove('hidden');
    if (!snapshot && !request && typeof sessionApi === 'function') {
      try {
        const response = await fetch(sessionApi('/tz'));
        if (response.ok) {
          const data = await response.json();
          if (data && Array.isArray(data.sections) && data.sections.length) snapshot = data;
        }
      } catch (err) {
        console.warn('ТЗ panel: fetch failed', err);
      }
    }
    render();
  }

  function closeTzPanel() {
    const modal = document.getElementById('tz-modal');
    if (modal) modal.classList.add('hidden');
  }

  function submitTzForm() {
    if (!request) return;
    const requestId = request.request_id;
    const empty = emptyCount();
    const unset = notRequiredCount();
    const formValues = {};
    Object.keys(draft).forEach(section => {
      formValues[section] = {};
      Object.keys(draft[section]).forEach(field => {
        formValues[section][field] = String(draft[section][field] || '').trim();
      });
    });
    if (!sendHitlResponse({
      type: 'hitl_response',
      request_id: requestId,
      action: 'approve',
      approved: true,
      form_values: formValues,
    })) return;
    const panel = document.getElementById('hitl-panel');
    if (panel) panel.classList.add('hidden');
    markCard(requestId, empty
      ? `Отправлено — пустые поля (${empty}) заполняет агент, затем ТЗ вернётся на проверку.`
      : 'ТЗ подтверждено.');
    addSystemMsg((empty
      ? `✓ ТЗ отправлено: пустых полей передано агенту — ${empty}`
      : '✓ ТЗ подтверждено оператором')
      + (unset ? ` · без ограничения («не требуется»): ${unset}` : ''));
    request = null;
    draft = {};
    stash = {};
    render();
  }

  function markCard(requestId, text) {
    const status = document.getElementById('tz-hitl-status-' + requestId);
    if (status) status.textContent = text;
  }

  // ── inbound ────────────────────────────────────────────────────────────
  function onHitlRequest(data, { history = false } = {}) {
    if (history) {
      // Transcript replay is informational. Keep no editable form state;
      // an unresolved request is supplied again as a live redelivery.
      request = null;
      draft = {};
      stash = {};
    } else {
      request = data;
      resetDraft();
    }
    const form = data.form || {};
    const counts = form.counts || {};

    const panel = document.getElementById('hitl-panel');
    // A historical request is not actionable. Do not leave the old
    // "Оставить пустым" controls visible after the run has continued.
    if (panel && history) panel.classList.add('hidden');
    if (panel && !history) {
      panel.classList.remove('hidden');
      panel.innerHTML = `
        <div class="relative bg-surface-container-lowest p-4 rounded-xl border border-primary/30 shadow-2xl flex flex-col gap-2">
          <h3 class="font-headline font-bold text-on-surface text-sm uppercase tracking-tight">Проверка ТЗ</h3>
          <p class="text-[11px] text-on-surface-variant">${esc(form.intro || '')}</p>
          <button onclick="openTzPanel()" class="flex items-center justify-center gap-2 bg-primary text-on-primary py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] hover:brightness-110 transition-all">
            <span class="material-symbols-outlined text-sm">assignment</span> Открыть ТЗ
          </button>
        </div>`;
    }
    const cardHtml = `
      <div class="my-6 relative msg-enter" id="tz-hitl-card-${esc(data.request_id)}" data-hitl-card="${esc(data.request_id)}">
        <div class="relative bg-surface-container-lowest p-6 rounded-xl border border-primary/30 shadow-2xl">
          <div class="flex items-center gap-3 mb-2">
            <div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center shadow-[0_0_15px_rgba(0,218,243,0.4)]">
              <span class="material-symbols-outlined text-on-primary text-sm">assignment</span>
            </div>
            <h3 class="font-headline font-bold text-on-surface uppercase tracking-tight">Проверка ТЗ — раунд ${esc(form.round || 1)}</h3>
          </div>
          <p class="text-sm text-on-surface-variant leading-relaxed pl-11">${esc(form.intro || data.message || '')}</p>
          <div class="flex flex-wrap gap-2 mt-3 pl-11 text-[10px] font-mono">
            <span class="px-2 py-0.5 rounded-full border border-error/30 text-error">пустых полей: ${esc(counts.awaiting_fields || 0)}</span>
            <span class="px-2 py-0.5 rounded-full border border-tertiary/30 text-tertiary">заполнено агентом: ${esc(counts.agent_filled || 0)}</span>
            <span class="px-2 py-0.5 rounded-full border border-outline-variant/30 text-outline-variant">разделов: ${esc(counts.sections || 0)}</span>
          </div>
          <div class="mt-4 pl-11 flex items-center gap-3">
            <button ${history ? 'disabled' : 'onclick="openTzPanel()"'} class="flex items-center gap-2 ${history ? 'bg-surface-container-high text-outline-variant cursor-not-allowed' : 'bg-primary text-on-primary shadow-lg shadow-primary/20 hover:brightness-110 active:scale-95'} px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] transition-all">
              <span class="material-symbols-outlined text-base">assignment</span> Открыть ТЗ
            </button>
            <span id="tz-hitl-status-${esc(data.request_id)}" class="text-[10px] text-outline-variant font-mono"></span>
          </div>
        </div>
      </div>`;
    const previousCard = document.getElementById(`tz-hitl-card-${CSS.escape(data.request_id || '')}`);
    if (previousCard) previousCard.outerHTML = cardHtml;
    else appendMsgToFeed(cardHtml);
    // Requests replayed from the session transcript are historical entries.
    // They must not reopen the modal on every page load. A live request is
    // redelivered separately after the snapshot and is the only event that
    // should interrupt the operator with the ТЗ form.
    if (!history) openTzPanel();
  }

  function clearRequest(requestId, note) {
    if (!request || (requestId && request.request_id !== requestId)) return;
    markCard(request.request_id, note || '');
    request = null;
    draft = {};
    render();
  }

  function feed(data) {
    if (!data || data.type !== 'tz_snapshot') return;
    snapshot = data;
    // While the operator is filling the form, keep their typing on screen.
    if (!formMode()) render();
  }

  function restore(tz, forSession) {
    if (forSession !== sessionId) {
      sessionId = forSession;
      request = null;
      draft = {};
    }
    snapshot = tz && Array.isArray(tz.sections) ? tz : null;
    render();
  }

  document.addEventListener('DOMContentLoaded', () => {
    const body = document.getElementById('tz-modal-body');
    if (!body) return;
    body.addEventListener('input', onInput);
    body.addEventListener('focusout', onFocusOut);
    body.addEventListener('click', onClick);
  });

  window.TZPanel = { feed, onHitlRequest, clearRequest, restore, isFormOpen: formMode };
  window.openTzPanel = openTzPanel;
  window.closeTzPanel = closeTzPanel;
  window.submitTzForm = submitTzForm;
  window.tzLeaveEmptyUnset = leaveEmptyUnset;
})();
