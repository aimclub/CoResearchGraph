// =========================================================================
// HITL UI
// =========================================================================
// =========================================================================
// HITL UI
// =========================================================================
// The card header is built on the client from agent_name / invoked_via /
// trigger, so it follows the UI language. Older requests (and replays of
// them) carry the trigger only as an English "[CALLBACK: X]" message prefix;
// internal-loop ones as a fixed English message.
const INTERNAL_LOOP_RE = /^(?:\[INTERNAL_LOOP:[^\]]*\]\s*)?Agent '([^']+)' proposes its result\. Please review\.?\s*$/;
const LEGACY_PREFIX_RE = /^\[(CALLBACK|INTERNAL_LOOP):\s*([A-Z_]*)[^\]]*\]\s*/;
const TRIGGER_KEYS = {
  before_tool: 'beforeTool',
  after_agent: 'afterAgent',
  before_agent: 'beforeAgent',
  bash_command: 'bashCommand',
  work_order: 'workOrder',
  work_order_amendment: 'workOrderAmendment',
  work_report: 'workReport',
};
const hitlCards = new Map();  // request_id -> payload, re-rendered on language switch
const workOrderStripState = new Map(); // request_id -> manually chosen open state
// Lifecycle and operator input must outlive the DOM node. Localisation redraws
// the structured cards with outerHTML, so DOM-only `data-answered` state would
// make an already answered card actionable again.
const hitlCardState = new Map(); // request_id -> { closed, history, feedback, assumptions, findings }

function hitlState(requestId) {
  const rid = String(requestId || '');
  let state = hitlCardState.get(rid);
  if (!state) {
    state = { closed: false, history: false };
    hitlCardState.set(rid, state);
  }
  return state;
}

function captureHitlCardState(requestId) {
  const rid = String(requestId || '');
  const state = hitlState(rid);
  const card = document.querySelector(`[data-hitl-card="${CSS.escape(rid)}"]`);
  const feedback = document.getElementById('hitl-feedback-' + rid);
  if (feedback) state.feedback = feedback.value;
  if (!card) return state;

  state.assumptions = Object.fromEntries(
    [...card.querySelectorAll('input[data-wo-assumption]')]
      .map(el => [el.dataset.woAssumption, el.checked])
  );
  state.findings = Object.fromEntries(
    [...card.querySelectorAll('input[data-wr-finding]')]
      .map(el => [el.dataset.wrFinding, el.checked])
  );
  return state;
}

function restoreHitlCardState(requestId) {
  const rid = String(requestId || '');
  const state = hitlState(rid);
  const card = document.querySelector(`[data-hitl-card="${CSS.escape(rid)}"]`);
  const feedback = document.getElementById('hitl-feedback-' + rid);
  if (feedback && Object.prototype.hasOwnProperty.call(state, 'feedback')) {
    feedback.value = state.feedback;
  }
  if (card) {
    card.querySelectorAll('input[data-wo-assumption]').forEach(el => {
      if (Object.prototype.hasOwnProperty.call(state.assumptions || {}, el.dataset.woAssumption)) {
        el.checked = state.assumptions[el.dataset.woAssumption];
      }
    });
    card.querySelectorAll('input[data-wr-finding]').forEach(el => {
      if (Object.prototype.hasOwnProperty.call(state.findings || {}, el.dataset.wrFinding)) {
        el.checked = state.findings[el.dataset.wrFinding];
      }
    });
    woRecount(card);
  }
  if (state.closed || state.history) {
    disableHitlControls(rid, { remember: false });
  }
}

function resetHitlUiState() {
  hitlCards.clear();
  hitlCardState.clear();
  workOrderStripState.clear();
  if (typeof planByRequest !== 'undefined') planByRequest.clear();
  if (typeof planOpenTasks !== 'undefined') planOpenTasks.clear();
}
window.resetHitlUiState = resetHitlUiState;

function fillHitl(key, params) {
  return t(key).replace(/\{(\w+)\}/g, (m, k) => (params[k] != null ? params[k] : m));
}

function hitlTrigger(data) {
  if (data.trigger) return data.trigger;
  const m = LEGACY_PREFIX_RE.exec(data.message || '');
  return m && m[1] === 'CALLBACK' ? m[2].toLowerCase() : null;
}

// The agent's display name (e.g. «Анализ литературы»), not its class name.
function hitlAgentName(name) {
  if (!name) return '';
  return (window.StatusIndicator && StatusIndicator.agentName)
    ? (StatusIndicator.agentName(name) || name) : name;
}

function hitlParams(data) {
  const ctx = data.context || {};
  return {
    agent: hitlAgentName(data.agent_name) || '?',
    tool: ctx.tool || data.trigger || '?',
    rule: ctx.matched_rule || '?',
  };
}

function localizeHitlMessage(data) {
  const trigger = hitlTrigger(data);
  const key = TRIGGER_KEYS[trigger];
  // A legacy before_agent message embeds the user query, which the template would drop.
  if (key && !(trigger === 'before_agent' && !data.trigger)) {
    return fillHitl('hitl.msg.' + key, hitlParams(data));
  }
  const message = data.message || '';
  const m = INTERNAL_LOOP_RE.exec(message);
  if (m) return t('hitl.internalLoop').replace('{agent}', hitlAgentName(m[1]));
  return message.replace(LEGACY_PREFIX_RE, '');
}

function describeHitlVia(data) {
  const params = hitlParams(data);
  const key = TRIGGER_KEYS[hitlTrigger(data)];
  if (key) return fillHitl('hitl.via.' + key, params);
  switch (data.invoked_via) {
    case 'tool': return fillHitl('hitl.via.tool', params);
    case 'callback': return t('hitl.via.callback');
    case 'internal_loop': return t('hitl.via.internalLoop');
    case 'request_input': return t('hitl.via.requestInput');
    case 'sandbox_download_cancel': return t('hitl.via.downloadCancel');
  }
  if (INTERNAL_LOOP_RE.test(data.message || '')) return t('hitl.via.internalLoop');
  return data.invoked_via && data.invoked_via !== 'unspecified' ? data.invoked_via : t('hitl.via.unknown');
}

// The single place the request details (tool arguments, output, command) are shown.
function hitlDetailBlock(data) {
  const ctx = data.context || {};
  if (ctx.output) {
    // A tool's arguments are code; an agent's proposed result is a plan, a
    // summary, a page of prose. `code` decides which of the two treatments
    // the block gets — the same <pre> used to set both in IBM Plex Mono.
    const isToolCall = hitlTrigger(data) === 'before_tool';
    return {
      labelKey: isToolCall ? 'hitl.block.toolCall' : 'hitl.block.output',
      // A structured output would print as "[object Object]".
      text: typeof ctx.output === 'object' ? JSON.stringify(ctx.output, null, 2) : String(ctx.output),
      code: isToolCall,
    };
  }
  if (ctx.command) return { labelKey: 'hitl.block.command', text: String(ctx.command), code: true };
  if (ctx.user_query) return { labelKey: 'hitl.block.userQuery', text: String(ctx.user_query), code: false };
  return null;
}

// The agent's message is markdown; the "via" line is ours and stays text.
function hitlDynamicHtml(part, text) {
  return part === 'message' ? mdInline(text) : escHtml(text);
}

function hitlDynamic(data, part, text) {
  return `<span data-hitl-part="${part}" data-hitl-rid="${escHtml(data.request_id || '')}">${hitlDynamicHtml(part, text)}</span>`;
}

function hitlLabel(key) {
  return `<span data-i18n="${key}">${escHtml(t(key))}</span>`;
}

function relocalizeHitlCards() {
  document.querySelectorAll('[data-hitl-part]').forEach(el => {
    const data = hitlCards.get(el.dataset.hitlRid);
    if (!data) return;
    const part = el.dataset.hitlPart;
    el.innerHTML = hitlDynamicHtml(part, part === 'via' ? describeHitlVia(data) : localizeHitlMessage(data));
  });
  redrawPlanCards();
  redrawWorkOrderCards();
}

function redrawWorkOrderCards() {
  hitlCards.forEach((data, rid) => {
    const trigger = hitlTrigger(data);
    if (trigger !== 'work_order' && trigger !== 'work_order_amendment'
        && trigger !== 'work_step' && trigger !== 'work_report') return;
    const card = document.querySelector(`[data-hitl-card="${CSS.escape(rid)}"]`);
    if (!card) return;
    const state = captureHitlCardState(rid);
    const live = !state.closed && !state.history;
    if (trigger === 'work_report') renderWorkReportCard(live, data);
    else if (trigger === 'work_step') renderWorkStepCard(live, data);
    else renderWorkOrderCard(live, data);
    restoreHitlCardState(rid);
    const remembered = workOrderStripState.get(rid);
    if (remembered == null) return;
    const fresh = document.querySelector(`[data-hitl-card="${CSS.escape(rid)}"] details[data-wo-strip]`);
    if (fresh) {
      fresh.open = remembered;
      fresh.removeAttribute('data-auto-open');
    }
  });
}

// The plan card builds every label in JS — counts interpolated into "{n} tasks",
// the design-matrix headers, the per-task field names — so there is no
// data-i18n span for applyTranslations to swap. It is redrawn from the request
// it was drawn from instead; fold state lives in planOpenTasks, so the same
// tasks stay open across the redraw.
function redrawPlanCards() {
  if (typeof planByRequest === 'undefined') return;
  planByRequest.forEach((_, rid) => {
    const data = hitlCards.get(rid);
    // Only cards actually on screen: placeHitlCard appends when it finds none,
    // so a stale entry would resurrect a card the session has already cleared.
    if (!data || !document.querySelector(`[data-hitl-card="${CSS.escape(rid)}"]`)) return;
    const state = captureHitlCardState(rid);
    renderExperimentPlanReview(!state.closed && !state.history, data);
    restoreHitlCardState(rid);
  });
}
window.relocalizeHitlCards = relocalizeHitlCards;

// A card is keyed by its request id: a request the server redelivers (a
// reconnect while it is still open) replaces its disabled history copy in
// place instead of appearing twice.
function placeHitlCard(rid, html) {
  const existing = rid ? document.querySelector(`[data-hitl-card="${CSS.escape(rid)}"]`) : null;
  if (existing) {
    existing.outerHTML = html;
  } else {
    appendMsgToFeed(html);
  }
}

// history=true renders a card from the session transcript (reload, import):
// no countdown, controls disabled until the server redelivers the request as
// still open.
//
// There used to be a second copy of every card in the right sidebar. It said
// the same thing as the one in the feed and could not be answered, so it was
// two places to read and one place to act. `live` is the only thing the
// renderers ever took from it — whether this request is still open — so that
// is what they are given now.
function showHITL(data, { history = false } = {}) {
  const live = !history;
  const rid = data.request_id || '';
  hitlCards.set(rid, data);
  // A live redelivery is authoritative: the server only redelivers requests
  // that are still pending. A transcript copy stays locked until either its
  // recorded outcome is replayed or that live redelivery arrives.
  Object.assign(hitlState(rid), { closed: false, history });

  // The microfluidics ТЗ has its own panel, which owns both the form and the
  // running document; hand the request over and let it draw the card.
  if (data.form && data.form.kind === 'tz' && window.TZPanel) {
    TZPanel.onHitlRequest(data, { history });
    scrollChat();
    return;
  }

  // Structured intake (e.g. the research frame): render a per-field form
  // instead of the free-text review, then stop — the other HITL points keep
  // the free-text / option path below.
  if (data.form && Array.isArray(data.form.blocks)) {
    renderHitlForm(live, data);
  } else if (((data.context || {}).experiment_plan || {}).tasks) {
    // An experiment plan is not a paragraph to skim: it is a design matrix and
    // a task list, and it is what the human is being asked to approve. The
    // backend ships it structured next to the rendered Markdown, so it gets a
    // view of its own rather than a <pre> of pipe-separated rows.
    renderExperimentPlanReview(live, data);
  } else if (data.trigger === 'work_step') {
    // One finished step of a Work Order: what was sent, expected and found,
    // with the calls the system recorded — accept, redo, or stop the order.
    renderWorkStepCard(live, data);
  } else if (data.trigger === 'work_report') {
    // Work Report (what the agent found, against its order): accept, send back
    // for rework with findings marked wrong, or reject.
    renderWorkReportCard(live, data);
  } else if (String(data.trigger || '').startsWith('work_order')) {
    // Work Order (the agent's contract before it acts): its own card with
    // assumptions to uncheck, a veto countdown and a Pause button.
    renderWorkOrderCard(live, data);
  } else {
    renderHitlCard(live, data);
  }
  restoreHitlCardState(rid);
  scrollChat();
}

// Buttons of a review card. Primary is the expected answer, secondary the
// alternative, and the destructive one is quiet until it is asked for twice.
const HITL_BTN = 'inline-flex items-center justify-center gap-1.5 h-9 px-4 rounded-md text-[12px] font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed';
const HITL_BTN_PRIMARY = `${HITL_BTN} bg-primary text-on-primary hover:brightness-110`;
const HITL_BTN_SECONDARY = `${HITL_BTN} border border-outline-variant/30 text-on-surface hover:bg-surface-container-high`;
const HITL_BTN_DANGER = `${HITL_BTN} text-error hover:bg-error/10`;
const HITL_TEXTAREA = 'w-full bg-surface-container-lowest border border-outline-variant/25 rounded-md px-3 py-2 text-[12px] text-on-surface focus:outline-none focus:border-primary/60 transition-colors';

function hitlFeedbackField(rid, labelKey, placeholderKey, optional) {
  return `
          <div>
            <label for="hitl-feedback-${rid}" class="block text-[11px] text-on-surface-variant mb-1.5">${hitlLabel(labelKey)}${optional
      ? ` <span class="text-outline-variant">(${hitlLabel('hitl.fb.optional')})</span>` : ''}</label>
            <textarea id="hitl-feedback-${rid}" name="hitl-feedback" rows="2" autocomplete="off"
              data-i18n-placeholder="${placeholderKey}" placeholder="${escHtml(t(placeholderKey))}"
              oninput="syncHitlButtons('${escJs(rid)}')" onkeydown="hitlFeedbackKeydown(event, '${escJs(rid)}')"
              class="${HITL_TEXTAREA}"></textarea>
          </div>`;
}

function hitlSubmitHint() {
  return `<span class="hidden sm:flex items-center gap-1 ml-auto text-[10px] text-outline-variant select-none">`
    + `<kbd>Ctrl</kbd><kbd>Enter</kbd> ${hitlLabel('hitl.hintSubmit')}</span>`;
}

// With corrections typed, the answer is "revise"; without, it is "accept".
// One of the two is live at a time, so neither button quietly means the other.
function syncHitlButtons(rid) {
  const field = document.getElementById('hitl-feedback-' + rid);
  const hasText = !!(field && field.value.trim());
  const accept = document.getElementById('hitl-accept-' + rid);
  const revise = document.getElementById('hitl-revise-' + rid);
  if (accept && !accept.closest('[data-answered]')) accept.disabled = hasText;
  if (revise && !revise.closest('[data-answered]')) revise.disabled = !hasText;
}

function hitlFeedbackKeydown(event, rid) {
  if (event.key !== 'Enter' || !(event.ctrlKey || event.metaKey) || event.isComposing) return;
  event.preventDefault();
  const card = document.querySelector(`[data-hitl-card="${CSS.escape(rid)}"]`);
  const data = hitlCards.get(rid) || {};
  const field = document.getElementById('hitl-feedback-' + rid);
  const hasText = !!(field && field.value.trim());
  if (!card || card.querySelector('[data-answered]')) return;
  if (data.action_type === 'provide_input') respondHITLInput(rid);
  else if (hasText) respondHITLEdit(rid);
  else if (!(data.options && data.options.length)) respondHITL(rid, true);
}

// Rejecting is not undoable, so the first click only arms the button.
const hitlRejectTimers = new Map();

function confirmHitlReject(rid) {
  const button = document.getElementById('hitl-reject-' + rid);
  if (!button) return;
  if (button.dataset.armed === '1') {
    clearTimeout(hitlRejectTimers.get(rid));
    hitlRejectTimers.delete(rid);
    respondHITL(rid, false);
    return;
  }
  button.dataset.armed = '1';
  button.className = `${HITL_BTN} bg-error text-on-error hover:brightness-110`;
  button.innerHTML = hitlLabel('hitl.btn.rejectConfirm');
  hitlRejectTimers.set(rid, setTimeout(() => {
    hitlRejectTimers.delete(rid);
    if (button.disabled) return;
    delete button.dataset.armed;
    button.className = HITL_BTN_DANGER;
    button.innerHTML = hitlLabel('hitl.btn.rejectAsk');
  }, 4000));
}

window.syncHitlButtons = syncHitlButtons;
window.hitlFeedbackKeydown = hitlFeedbackKeydown;
window.confirmHitlReject = confirmHitlReject;

function renderHitlCard(live, data) {
  const rid = data.request_id;
  const ridJs = escJs(rid || '');
  const messageHtml = hitlDynamic(data, 'message', localizeHitlMessage(data));
  const viaHtml = hitlDynamic(data, 'via', describeHitlVia(data));
  const displayAgent = (window.StatusIndicator && StatusIndicator.agentName)
    ? StatusIndicator.agentName(data.agent_name)
    : data.agent_name;
  const agentHtml = data.agent_name
    ? `<span translate="no" class="font-bold text-on-surface-variant">${escHtml(displayAgent)}</span> · ` : '';

  const openRoadmapChatBtn = data.agent_name !== 'PlannerAgent' ? '' : `
        <div>
          <button type="button" onclick="openRoadmapEditor()" class="${HITL_BTN_SECONDARY}">
            <span class="material-symbols-outlined text-[18px]" aria-hidden="true">map</span> ${hitlLabel('hitl.btn.openRoadmap')}
          </button>
        </div>`;

  const isProvideInput = data.action_type === 'provide_input';
  const hasOptions = !!(data.options && data.options.length);

  // The request details: tool arguments are code, an agent's proposed
  // result is prose and is read as such.
  const detail = hitlDetailBlock(data);
  const detailBody = !detail ? '' : detail.code
    ? `<pre class="font-mono text-[13px] leading-relaxed text-on-surface whitespace-pre-wrap `
      + `bg-surface-container-lowest px-3 py-2.5 rounded-lg border border-outline-variant/15">${escHtml(detail.text)}</pre>`
    : `<div class="md-body hitl-prose text-on-surface">${renderMarkdown(detail.text)}</div>`;
  const asDocument = documentBlock(data);
  const outputBlock = !detail && !asDocument ? '' : `
        <div>
          <p class="text-[10px] font-medium text-outline-variant mb-1.5">${hitlLabel(
      detail ? detail.labelKey : 'hitl.block.output')}</p>
          ${asDocument || foldable(detailBody, detail.text, { bg: 'rgb(var(--c-surface-container-low))' })}
        </div>`;

  let controls;
  if (isProvideInput) {
    controls = `
          ${hitlFeedbackField(rid, 'hitl.fb.inputLabel', 'hitl.ph.input', false)}
          <div class="flex flex-wrap items-center gap-2">
            <button type="button" onclick="respondHITLInput('${ridJs}')" class="${HITL_BTN_PRIMARY}">
              <span class="material-symbols-outlined text-[18px]" aria-hidden="true">send</span> ${hitlLabel('hitl.btn.send')}
            </button>
            ${hitlSubmitHint()}
          </div>`;
  } else if (hasOptions) {
    controls = `
          <div class="flex flex-wrap gap-2">
            ${data.options.map(o => `
            <button type="button" onclick="respondHITLOption('${ridJs}', '${escJs(o)}')" class="${HITL_BTN_SECONDARY}">${escHtml(o)}</button>`).join('')}
          </div>
          ${hitlFeedbackField(rid, 'hitl.fb.replyLabel', 'hitl.ph.reply', false)}
          <div class="flex flex-wrap items-center gap-2">
            <button type="button" id="hitl-revise-${rid}" onclick="respondHITLEdit('${ridJs}')" disabled class="${HITL_BTN_PRIMARY}">
              <span class="material-symbols-outlined text-[18px]" aria-hidden="true">reply</span> ${hitlLabel('hitl.btn.reply')}
            </button>
            ${hitlSubmitHint()}
          </div>`;
  } else {
    controls = `
          ${hitlFeedbackField(rid, 'hitl.fb.label', 'hitl.ph.revise', true)}
          <div class="flex flex-wrap items-center gap-2">
            <button type="button" id="hitl-accept-${rid}" onclick="respondHITL('${ridJs}', true)" class="${HITL_BTN_PRIMARY}">
              <span class="material-symbols-outlined text-[18px]" aria-hidden="true">check</span> ${hitlLabel('hitl.btn.acceptResult')}
            </button>
            <button type="button" id="hitl-revise-${rid}" onclick="respondHITLEdit('${ridJs}')" disabled class="${HITL_BTN_SECONDARY}">
              ${hitlLabel('hitl.btn.sendRevise')}
            </button>
            <button type="button" id="hitl-reject-${rid}" onclick="confirmHitlReject('${ridJs}')" class="${HITL_BTN_DANGER}">
              ${hitlLabel('hitl.btn.rejectAsk')}
            </button>
            ${hitlSubmitHint()}
          </div>`;
  }

  placeHitlCard(rid, `
    <section class="my-2 msg-enter" data-hitl-card="${escHtml(rid || '')}" aria-labelledby="hitl-title-${escHtml(rid || '')}">
      <div class="bg-surface-container-low rounded-xl border border-outline-variant/25">
        <div class="flex items-start gap-3 px-5 pt-4">
          <div class="w-8 h-8 rounded-lg bg-tertiary/10 text-tertiary flex items-center justify-center shrink-0" aria-hidden="true">
            <span class="material-symbols-outlined text-[18px]">front_hand</span>
          </div>
          <div class="min-w-0">
            <h3 id="hitl-title-${escHtml(rid || '')}" class="text-[14px] font-semibold text-on-surface leading-snug break-words">${messageHtml}</h3>
            <p class="text-[10px] text-outline-variant mt-0.5 break-words">${agentHtml}${viaHtml}</p>
          </div>
        </div>
        <div class="px-5 pb-4 pt-3 sm:pl-16 flex flex-col gap-4">
          ${outputBlock}
          <div id="hitl-controls-${rid}" class="flex flex-col gap-3">${controls}
          </div>
          ${openRoadmapChatBtn}
        </div>
      </div>
    </section>`);
}

function disableHitlControls(requestId, { remember = true } = {}) {
  const state = captureHitlCardState(requestId);
  if (remember) {
    state.closed = true;
    state.history = false;
  }
  stopWorkOrderCountdown(requestId);
  // Answered, timed out or cancelled: the question is closed, so the panel it
  // opened closes with it. One the reader opened by hand stays.
  if (remember && window.closeDocumentForRequest) closeDocumentForRequest(requestId);
  // Nothing left to act on: fold whatever this card was showing open.
  const card = document.querySelector(`[data-hitl-card="${CSS.escape(String(requestId || ''))}"]`);
  if (remember && card) card.querySelectorAll('.fold-open').forEach(collapseFold);
  // And close the strip we opened because an answer was needed. Only that one:
  // a strip the reader opened by hand is being read, and a timeout firing under
  // it is no reason to snap it shut.
  //
  // The marks inside are locked too. A card that timed out or was cancelled
  // never went through respondWorkOrder, so its assumption and finding boxes
  // stayed clickable on a question nobody is asking any more.
  if (card) {
    if (remember) {
      card.querySelectorAll('details[data-wo-strip][data-auto-open]').forEach(el => {
        el.open = false;
        el.removeAttribute('data-auto-open');
      });
    }
    card.querySelectorAll('input[data-wo-assumption], input[data-wr-finding]')
      .forEach(el => { el.disabled = true; });
  }
  const box = document.getElementById('hitl-controls-' + requestId);
  if (!box) return;
  box.dataset.answered = '1';
  box.querySelectorAll('button, textarea').forEach(el => {
    el.disabled = true;
    el.classList.add('opacity-40', 'pointer-events-none');
  });
}

let currentPlannerHitlRequest = null;

// The chat line recording an answer. One place for both the live click and
// the transcript replay, so a reloaded session reads the same as the live one.
function hitlResponseSummary(response) {
  const request = hitlCards.get(response.request_id || '') || {};
  const feedback = String(response.instructions || response.free_input || '').trim();
  const action = response.action;
  const source = response.decision_source
    || (response.auto === 'mode=auto' ? 'mode_auto' : 'human');
  const withSource = text => source === 'human' ? text
    : `${text} — ${t('hitl.source.' + source, source)}`;
  if (request.form && Array.isArray(request.form.blocks)) {
    const values = response.form_values;
    const n = values ? Object.values(values).reduce((s, o) => s + Object.keys(o || {}).length, 0) : 0;
    return withSource(values ? t('hitl.form.saved').replace('{n}', n) : t('hitl.form.skipped'));
  }
  if (String(request.trigger || '').startsWith('work_order') && action === 'approve') {
    const rejected = ((response.form_values || {}).rejected_assumption_ids || []).length;
    return withSource(t('workOrder.approved')
      + (rejected ? ' — ' + t('workOrder.rejectedAssumptions').replace('{n}', rejected) : '')
      + (feedback ? ': ' + feedback : ''));
  }
  if (request.trigger === 'work_step') {
    const key = action === 'approve' ? 'workStep.accepted' : action === 'edit' ? 'workStep.sentBack' : 'workStep.stopped';
    const stepId = ((request.context || {}).step || {}).id || '';
    return withSource(t(key).replace('{step}', stepId) + (feedback ? ': ' + feedback : ''));
  }
  if (request.trigger === 'work_report') {
    const disputed = ((response.form_values || {}).disputed_finding_ids || []).length;
    const key = action === 'approve' ? 'workReport.accepted' : action === 'edit' ? 'workReport.sentBack' : 'workReport.rejected';
    return withSource(t(key)
      + (disputed ? ' — ' + t('workReport.disputedCount').replace('{n}', disputed) : '')
      + (feedback ? ': ' + feedback : ''));
  }
  // The receipts a run leaves in the feed. The keys were written when the rest
  // of this function was localised; these three lines kept their literals, so a
  // Russian session recorded every verdict as "✓ HITL Approved".
  if (action === 'provide_input')
    return withSource(t('hitl.inputSent', { feedback: feedback || t('hitl.empty') }));
  if (action === 'select') return withSource('☑ ' + (response.selected_option || feedback));
  if (action === 'edit') return withSource(t('hitl.revisionRequested', { feedback: feedback }));
  // The feedback used to hang off the rejected branch alone: `a ? b : c + d`
  // groups as `a ? b : (c + d)`, so an approval with a comment dropped it.
  return withSource(t(response.approved ? 'hitl.approved' : 'hitl.rejected')
    + (feedback ? ': ' + feedback : ''));
}

function hitlTimeoutSummary(data) {
  return t('hitl.timeoutMsg', { seconds: (data.timeout_seconds || 300), agent: hitlAgentName(data.agent_name) });
}
window.hitlTimeoutSummary = hitlTimeoutSummary;

// Replay of a recorded outcome onto its card: what the operator typed or
// unchecked is put back, then the card is locked.
function applyHitlOutcome(event) {
  const rid = event.request_id || '';
  if (event.type === 'hitl_response') {
    const card = rid ? document.querySelector(`[data-hitl-card="${CSS.escape(rid)}"]`) : null;
    const feedbackEl = document.getElementById('hitl-feedback-' + rid);
    const feedback = event.instructions || event.free_input;
    if (feedbackEl && feedback && event.action !== 'select') feedbackEl.value = feedback;
    const values = event.form_values || {};
    if (card) {
      card.querySelectorAll('textarea[data-field]').forEach(el => {
        const block = values[el.getAttribute('data-block')];
        const value = block && block[el.getAttribute('data-field')];
        if (value != null) el.value = value;
      });
      const rejected = values.rejected_assumption_ids || [];
      card.querySelectorAll('input[data-wo-assumption]').forEach(el => {
        el.checked = !rejected.includes(el.dataset.woAssumption);
        el.disabled = true;
      });
      // The findings the operator marked wrong, put back the same way. Without
      // this a reloaded report drew every finding unmarked, whatever was sent.
      const disputed = values.disputed_finding_ids || [];
      card.querySelectorAll('input[data-wr-finding]').forEach(el => {
        el.checked = disputed.includes(el.dataset.wrFinding);
        el.disabled = true;
      });
      woRecount(card);
    }
    disableHitlControls(rid);
    addSystemMsg(hitlResponseSummary(event), event.timestamp);
  } else if (event.type === 'hitl_timeout') {
    disableHitlControls(rid);
    addSystemMsg(hitlTimeoutSummary(event), event.timestamp);
  } else if (event.type === 'hitl_cancelled') {
    disableHitlControls(rid);
  }
}
window.applyHitlOutcome = applyHitlOutcome;

function sendHitlResponse(payload) {
  const rid = String(payload.request_id || '');
  const state = hitlState(rid);
  const controls = document.getElementById('hitl-controls-' + rid);
  if (state.closed || state.history || (controls && controls.dataset.answered === '1')) {
    return false;
  }
  if (!ws || ws.readyState !== 1) {
    addSystemMsg(t('hitl.sendUnavailable'));
    return false;
  }
  try {
    ws.send(JSON.stringify(payload));
  } catch (_) {
    addSystemMsg(t('hitl.sendUnavailable'));
    return false;
  }
  // Close centrally and synchronously. Every HITL surface, including the TZ
  // panel, now gets the same duplicate-click protection.
  disableHitlControls(rid);
  const request = hitlCards.get(rid);
  if (payload.action === 'approve' && request && request.agent_name === 'PlannerAgent') {
    releasePlanGate();
  }
  if (window.StatusIndicator) {
    StatusIndicator.feed({ type: 'hitl_response', request_id: payload.request_id });
  }
  addSystemMsg(hitlResponseSummary(payload));
  return true;
}

function respondHITLInput(requestId) {
  const feedbackEl = document.getElementById('hitl-feedback-' + requestId);
  const feedback = feedbackEl ? feedbackEl.value.trim() : '';
  if (!sendHitlResponse({
    type: 'hitl_response',
    request_id: requestId,
    action: 'provide_input',
    approved: true,
    instructions: feedback,
    free_input: feedback,
  })) return;

  if (currentPlannerHitlRequest && currentPlannerHitlRequest.request_id === requestId) {
    currentPlannerHitlRequest = null;
    updateRoadmapModalButtons();
  }
}

// A positive answer APPROVES, notes and all. For cards that carry their own
// «Доработать» button the shorthand below — approve + notes means revise — was
// pure harm: the operator pressed Утвердить, wrote a clarification, and the
// plan they had just approved was thrown away and written again from scratch.
// The note travels with the approval as the operator's own instruction.
function respondHITLApprove(requestId) {
  const feedbackEl = document.getElementById('hitl-feedback-' + requestId);
  const feedback = feedbackEl ? feedbackEl.value.trim() : '';
  if (!sendHitlResponse({
    type: 'hitl_response',
    request_id: requestId,
    action: 'approve',
    approved: true,
    instructions: feedback || null,
    free_input: null,
  })) return;

  if (currentPlannerHitlRequest && currentPlannerHitlRequest.request_id === requestId) {
    currentPlannerHitlRequest = null;
    updateRoadmapModalButtons();
  }
}
window.respondHITLApprove = respondHITLApprove;

function respondHITL(requestId, approved) {
  const feedbackEl = document.getElementById('hitl-feedback-' + requestId);
  const feedback = feedbackEl ? feedbackEl.value.trim() : '';
  // Approving with feedback means the operator wants the output revised.
  const action = approved && feedback ? 'edit' : (approved ? 'approve' : 'reject');
  if (!sendHitlResponse({
    type: 'hitl_response',
    request_id: requestId,
    action,
    approved: action === 'approve',
    instructions: feedback || null,
    free_input: feedback || null,
  })) return;

  if (currentPlannerHitlRequest && currentPlannerHitlRequest.request_id === requestId) {
    currentPlannerHitlRequest = null;
    updateRoadmapModalButtons();
  }
}

function respondHITLOption(requestId, option) {
  // A question-window option button: a complete answer by itself.
  if (!sendHitlResponse({
    type: 'hitl_response',
    request_id: requestId,
    action: 'select',
    approved: true,
    selected_option: option,
    instructions: option,
    free_input: option,
  })) return;
}

function respondHITLEdit(requestId) {
  // Send the operator's corrections: the agent rewrites its output with them.
  const feedbackEl = document.getElementById('hitl-feedback-' + requestId);
  const feedback = feedbackEl ? feedbackEl.value.trim() : '';
  if (!feedback) {
    addSystemMsg(t('hitl.reviseEmpty'));
    if (feedbackEl) feedbackEl.focus();
    return;
  }
  if (!sendHitlResponse({
    type: 'hitl_response',
    request_id: requestId,
    action: 'edit',
    approved: false,
    instructions: feedback,
    free_input: feedback,
  })) return;

  if (currentPlannerHitlRequest && currentPlannerHitlRequest.request_id === requestId) {
    currentPlannerHitlRequest = null;
    updateRoadmapModalButtons();
  }
}

// ── Structured frame form (research frame intake) ────────────────────────
function renderHitlForm(live, data) {
  const form = data.form;
  const rid = data.request_id;
  // The backend sends {en, ru} dicts for the display strings. The canonical
  // Russian `title` / `name` stay the round-trip keys in data-block / data-field.
  const loc = (v, fallback) => {
    if (v && typeof v === 'object') return v[currentLang] || v.ru || v.en || fallback;
    return v || fallback;
  };
  const blocksHtml = form.blocks.map((b, bi) => {
    const fieldsHtml = (b.fields || []).map((f, fi) => {
      const openTag = f.open
        ? `<span class="text-[11px] text-error uppercase tracking-wider">${escHtml(t('hitl.form.notSet'))}</span>`
        : `<span class="text-[11px] text-outline-variant uppercase tracking-wider">${escHtml(f.status || '')}</span>`;
      const val = f.open ? '' : String(f.value || '');
      const label = loc(f.label, f.name);
      const placeholder = loc(f.placeholder, t('hitl.form.placeholderFallback'));
      return `
            <div class="flex flex-col gap-1">
              <div class="flex items-center justify-between">
                <label class="text-[15px] font-semibold text-on-surface">${escHtml(label)}</label>
                ${openTag}
              </div>
              <textarea id="frm-${rid}-${bi}-${fi}" data-block="${escJs(b.title)}" data-field="${escJs(f.name)}"
                rows="2" placeholder="${escHtml(placeholder)}"
                class="w-full bg-surface-container-high border border-outline-variant/25 rounded-md px-3 py-2 text-[15px] text-on-surface placeholder:text-outline-variant focus:outline-none focus:border-primary/50">${escHtml(val)}</textarea>
            </div>`;
    }).join('');
    const blockTitle = loc(b.title_i18n, b.title);
    const blockUsage = loc(b.usage_i18n, b.usage);
    return `
          <div class="mt-4 border border-outline-variant/15 rounded-lg p-4 bg-surface-container-high/30">
            <p class="text-[19px] font-bold text-on-surface leading-tight">${escHtml(blockTitle)}</p>
            ${blockUsage ? `<div class="text-[13px] text-on-surface-variant/80 mb-2.5 mt-0.5">${mdBlock(blockUsage)}</div>` : '<div class="mb-2"></div>'}
            <div class="flex flex-col gap-2">${fieldsHtml}</div>
          </div>`;
  }).join('');


  // Open while the request is live — a form cannot be filled in folded —
  // and folded once it has been answered or replayed from the transcript.
  const blocksBlock = foldable(blocksHtml, blocksHtml, { bg: 'rgb(var(--c-surface-container-low))', open: live });

  const formTitle = loc(form.title_i18n, form.title || t('hitl.form.title'));
  const formIntro = loc(form.intro_i18n, form.intro || data.message || '');
  // insertAdjacentHTML, not `innerHTML +=`: re-parsing the whole feed would
  // wipe whatever the operator is typing into the earlier cards.
  placeHitlCard(rid, `
        <div id="hitl-controls-${rid}" data-hitl-card="${escHtml(rid || '')}" class="my-6 relative msg-enter max-w-4xl">
          <div class="relative bg-surface-container-low p-5 rounded-xl border border-primary/40 shadow-2xl">
            <div class="flex items-center gap-3 mb-2">
              <div class="w-8 h-8 rounded-lg bg-tertiary/10 flex items-center justify-center" aria-hidden="true">
                <span class="material-symbols-outlined text-tertiary text-[18px]">fact_check</span>
              </div>
              <h3 class="font-headline font-bold text-base text-on-surface uppercase tracking-tight">${escHtml(formTitle)}</h3>
            </div>
            <div class="text-sm text-on-surface-variant leading-relaxed">${mdBlock(formIntro)}</div>
            ${blocksBlock}
            <div class="flex flex-wrap gap-3 mt-4">
              <button onclick="respondHITLForm('${rid}', true)" class="flex items-center justify-center gap-2 bg-primary text-on-primary px-4 py-2 rounded-md font-bold text-[12px] uppercase tracking-[0.08em] shadow-lg shadow-primary/20 hover:brightness-110 active:scale-95 transition-all">
                <span class="material-symbols-outlined text-base">check_circle</span> ${escHtml(t('hitl.form.save'))}
              </button>
              ${form.skippable === false ? '' : `<button onclick="respondHITLForm('${rid}', false)" class="flex items-center justify-center gap-2 bg-surface-container-high border border-outline-variant/20 text-on-surface px-4 py-2 rounded-md font-bold text-[12px] uppercase tracking-[0.08em] hover:bg-surface-container-highest transition-all">
                <span class="material-symbols-outlined text-base">skip_next</span> ${escHtml(t('hitl.form.skip'))}
              </button>`}
            </div>
          </div>
        </div>`);
}

function respondHITLForm(requestId, collect) {
  // collect=true: gather every non-empty field into form_values so the agent
  // marks them operator-set; collect=false: submit nothing (soft gate — the
  // run proceeds with the agent's drafted values).
  let formValues = null;
  if (collect) {
    formValues = {};
    document.querySelectorAll(`#hitl-controls-${requestId} textarea[data-field]`).forEach(el => {
      const v = el.value.trim();
      if (!v) return;
      const block = el.getAttribute('data-block');
      const field = el.getAttribute('data-field');
      (formValues[block] = formValues[block] || {})[field] = v;
    });
  }
  if (!sendHitlResponse({
    type: 'hitl_response',
    request_id: requestId,
    action: 'approve',
    approved: true,
    form_values: formValues,
  })) return;
}

// ── Work Order cards ─────────────────────────────────────────────────────
const WO_TIER_STYLE = {
  read: 'text-primary border-primary/30 bg-primary/10',
  compute: 'text-tertiary border-tertiary/30 bg-tertiary/10',
  side_effect: 'text-error border-error/30 bg-error/10',
};
const WO_STEP_MARK = { pending: '○', in_progress: '◐', done: '✓', skipped: '–' };
const woTimers = new Map();  // request_id -> interval id

function woChip(text, cls = 'text-on-surface-variant border-outline-variant/30 bg-surface-container-high') {
  return `<span class="inline-block text-[11px] font-mono px-2 py-0.5 rounded border ${cls}">${escHtml(text)}</span>`;
}

// Internal tools the agent named: always rendered, shown only while the viewer
// has "Show internal agents and tools" on (CSS on .wo-internal).
const WO_INTERNAL_CHIP = 'wo-internal text-outline-variant border-dashed border-outline-variant/30 bg-transparent';

function woToolChips(tools, internal) {
  return [
    ...(tools || []).map(tn => woChip(tn)),
    ...(internal || []).map(tn => woChip(tn, WO_INTERNAL_CHIP)),
  ].join(' ');
}

function woSection(labelKey, inner) {
  if (!inner) return '';
  return `
        <div class="mt-3 pt-3 border-t border-outline-variant/15 first:mt-0 first:pt-0 first:border-0">
          <p class="text-[11px] font-bold text-on-surface-variant uppercase tracking-wider mb-1.5">${hitlLabel(labelKey)}</p>
          ${inner}
        </div>`;
}

// ── The strip ────────────────────────────────────────────────────────────
// Findings, assumptions, steps, the goal and its criteria: once a run has done
// a few rounds they read as pages, and once a card is answered nobody reads
// them again. So every card carries them behind ONE line that says what is
// inside and how far it got — counts, step progress, the verdict, and the goal
// or summary cut to the width that is left — and the lists open on a click.
//
// <details>, deliberately, and not a fold. A closed <details> keeps its
// children in the DOM: respondWorkOrder still reads every checkbox, the replay
// restores them, and a progress notice still ticks a step inside it and the
// tally on the line itself. The `.fold-body` clip made exactly those
// unreachable, which is why the lists used to sit outside every fold.
//
// Callers pass an EMPTY body when there is nothing to list — a strip that opens
// onto an empty box is worse than no strip.
function woStrip(counts, lead, body, open) {
  if (!body || !String(body).trim()) return '';
  const parts = (counts || []).filter(Boolean)
    .join('<span class="text-outline-variant/50">·</span>');
  const text = String(lead || '').trim();
  // The counters may wrap onto a second line in a narrow column, but never push
  // past the card: a strip that makes the whole feed scroll sideways is the
  // opposite of what it is for. The lead text is what gives way first.
  return `
        <details data-wo-strip ${open ? 'open data-auto-open="1"' : ''}
          class="group mt-2 rounded-md border border-outline-variant/20 bg-surface-container/40">
          <summary class="flex items-center gap-2 min-w-0 px-2 py-1 cursor-pointer list-none select-none
                          text-[11px] leading-5 text-on-surface-variant hover:text-on-surface">
            <span class="material-symbols-outlined shrink-0 text-[15px] text-outline-variant transition-transform group-open:rotate-90">chevron_right</span>
            <span class="flex flex-wrap items-center gap-x-2 min-w-0">${parts || hitlLabel('woStrip.details')}</span>
            ${text ? `<span class="truncate min-w-0 flex-1 basis-24 text-outline-variant group-open:hidden" title="${escHtml(text)}">${escHtml(text)}</span>` : ''}
          </summary>
          <div class="px-3 pb-3 pt-2 border-t border-outline-variant/15">${body}</div>
        </details>`;
}

// One counter on the strip's line: a label and a value. `attrs` lets a live
// counter carry the hook its patcher looks for (data-wo-progress).
function woCount(labelKey, value, cls = '', attrs = '') {
  if (value === '' || value == null) return '';
  return `<span class="inline-flex items-baseline gap-1">${hitlLabel(labelKey)}`
    + `<span ${attrs} class="font-mono ${cls}">${escHtml(String(value))}</span></span>`;
}

// "closed/total" for a list of steps. `skipped` closes a step as `done` does —
// the server counts it that way too (work_order.report_warnings).
function woStepTally(steps) {
  const list = steps || [];
  if (!list.length) return '';
  const closed = list.filter(s => !WO_STEP_OPEN.has(s.status || 'pending')).length;
  return `${closed}/${list.length}`;
}

// A strip opens for exactly one reader: the one who has to answer this card
// now. That is every LIVE card — silence approves nothing (hitl/mode.py: in
// `basic` it refuses after ten minutes, in `debug` it waits, and in `auto` no
// card is drawn at all), so a countdown is not "it will decide itself" and must
// not hide the assumptions to reject or the findings to dispute behind a click.
// History, notices and answered cards stay one line; disableHitlControls folds
// the strip once the answer is in.
function woStripOpen(live) {
  return !!live;
}

// Re-count what the operator marked, onto the strip's line. The line is built
// from the server payload, which knows nothing of ticks made in this tab — and
// once the card is answered and its strip folds, that line is all anyone sees.
function woRecount(card) {
  if (!card) return;
  const kept = [...card.querySelectorAll('input[data-wo-assumption]')];
  const off = kept.filter(el => !el.checked).length;
  card.querySelectorAll('[data-wo-assumption-count]').forEach(el => {
    el.textContent = off ? `${kept.length - off}/${kept.length}` : String(kept.length);
  });
  const found = [...card.querySelectorAll('input[data-wr-finding]')];
  const wrong = found.filter(el => el.checked).length;
  card.querySelectorAll('[data-wr-finding-count]').forEach(el => {
    el.textContent = wrong ? `${found.length} ✗${wrong}` : String(found.length);
  });
}

// A card with a document shows its steps as one row of marks: the progress
// notices still replace a chip in place, so the run stays visible, and what each
// step is *for* is read in the panel. `woStepRow` keeps the full shape for a
// card that has no document to send the reader to.
function woStepChip(step) {
  const status = step.status || 'pending';
  const tip = [step.id, step.title, step.note, step.expected_outcome]
    .filter(Boolean).join(' — ');
  const tone = status === 'done' ? 'text-secondary border-secondary/40 bg-secondary/10'
    : status === 'in_progress' ? 'text-primary border-primary/50 bg-primary/10'
      : status === 'skipped' ? 'text-outline-variant/70 border-outline-variant/20'
        : 'text-outline-variant border-outline-variant/30';
  return `
          <li data-wo-step="${escHtml(step.id)}" data-wo-status="${escHtml(status)}"
            title="${escHtml(tip)}"
            class="inline-flex items-baseline gap-1 px-1.5 py-0.5 rounded border font-mono text-[11px] ${tone}">
            <span data-wo-mark>${WO_STEP_MARK[status] || '○'}</span>${escHtml(step.id)}
          </li>`;
}

function woStepRow(step, compact) {
  if (compact) return woStepChip(step);
  const status = step.status || 'pending';
  const tools = woToolChips(step.tools, step.internal_tools);
  return `
        <li data-wo-step="${escHtml(step.id)}" data-wo-status="${escHtml(status)}" class="flex flex-col gap-0.5">
          <div class="flex items-baseline gap-2">
            <span data-wo-mark class="font-mono text-primary w-4 text-center">${WO_STEP_MARK[status] || '○'}</span>
            <span class="font-mono text-[11px] text-outline-variant shrink-0">${escHtml(step.id)}</span>
            <span class="text-on-surface">${mdInline(step.title || '')}</span>
            <span class="flex flex-wrap gap-1">${tools}</span>
          </div>
          ${step.inputs ? `<p class="pl-6 text-[12px] text-on-surface-variant">↑ ${escHtml(t('workStep.sends'))}: ${mdInline(step.inputs)}</p>` : ''}
          ${step.expected_outcome ? `<p class="pl-6 text-[12px] text-on-surface-variant">→ ${mdInline(step.expected_outcome)}</p>` : ''}
          ${step.result ? `<p class="pl-6 text-[12px] text-on-surface">✓ ${escHtml(t('workStep.found'))}: ${mdInline(step.result)}</p>` : ''}
          <p data-wo-note class="pl-6 text-[12px] text-on-surface-variant italic">${mdInline(step.note || '')}</p>
          ${woStepReviewBadge(step.review)}
        </li>`;
}

const WS_REVIEW_STYLE = {
  accepted: 'text-primary border-primary/30 bg-primary/10',
  revise: 'text-tertiary border-tertiary/30 bg-tertiary/10',
  rejected: 'text-error border-error/30 bg-error/10',
};

function woStepReviewBadge(review) {
  if (!review || !review.status || review.status === 'pending') return '';
  const notes = review.notes ? ` <span class="text-[11px] text-outline-variant">${mdInline(review.notes)}</span>` : '';
  return `<p class="pl-6">${woChip(t('workStep.review.' + review.status, review.status), WS_REVIEW_STYLE[review.status])}${notes}</p>`;
}

// The contract itself. interactive=true renders assumptions as checkboxes.
//
// `opts.open` opens the strip (see woStripOpen); `opts.prefix` is anything that
// belongs inside it ahead of the contract — an amendment's reason and diff —
// and `opts.changes` counts that diff for the strip's line.
function workOrderBody(order, rid, interactive, compact, opts = {}) {
  const assumptions = (order.assumptions || []).map(a => {
    if (interactive) {
      return `
            <label class="flex items-start gap-2 cursor-pointer">
              <input type="checkbox" checked data-wo-assumption="${escHtml(a.id)}" class="mt-0.5 accent-primary" />
              <span><span class="font-mono text-[10px] text-outline-variant">${escHtml(a.id)}</span> ${mdInline(a.text)}</span>
            </label>`;
    }
    const struck = a.rejected ? 'line-through text-outline-variant' : '';
    return `<p class="${struck}"><span class="font-mono text-[10px] text-outline-variant">${escHtml(a.id)}</span> ${mdInline(a.text)}</p>`;
  }).join('');
  const effects = (order.side_effects || []).map(e =>
    woChip((e.kind || e) + (e.detail ? ': ' + e.detail : ''), WO_TIER_STYLE.side_effect)).join(' ');
  const tools = woToolChips(order.planned_tools, order.internal_tools);
  // Only internal tools: the whole section hides with them.
  const toolsOnlyInternal = !(order.planned_tools || []).length;
  const assumptionsBlock = woSection('workOrder.assumptions', assumptions
    ? (interactive ? `<p class="text-[12px] text-on-surface-variant mb-1.5">${hitlLabel('workOrder.assumptionsHint')}</p>` : '')
    + `<div class="flex flex-col gap-1">${assumptions}</div>`
    : '');
  const stepsBlock = woSection('workOrder.steps', (order.steps || []).length
    ? `<ol data-wo-steps data-compact="${compact ? '1' : '0'}" class="flex flex-col gap-1.5">`
      + order.steps.map(step => woStepRow(step, compact)).join('') + '</ol>'
    : '');

  // What the card is answered with, and what a progress notice patches. With a
  // document it is chips, because the words are in the file; without one there
  // is nowhere else for them to be, so it is the real thing.
  const lists = compact ? chipsRow(order, interactive) + stepsRow(order)
    : assumptionsBlock + stepsBlock;
  const controls = lists.trim() ? `
        <div class="text-xs text-on-surface-variant leading-relaxed flex flex-col gap-1.5">
          ${lists}
        </div>` : '';

  // Everything goes behind the strip, prose and controls alike — a closed
  // <details> clips nothing, so the checkboxes and step marks inside it keep
  // working (see woStrip). With a document the prose stays out: it is in the
  // file this card opens.
  const prose = compact ? '' : `
        <div class="text-xs text-on-surface-variant leading-relaxed mb-3">
          ${woSection('workOrder.goal', `<div class="text-on-surface">${mdBlock(order.goal || '')}</div>`)}
          ${woSection('workOrder.done', order.done_criteria ? mdBlock(order.done_criteria) : '')}
          ${tools ? `<div class="${toolsOnlyInternal ? 'wo-internal' : ''}">${woSection('workOrder.tools', `<div class="flex flex-wrap gap-1">${tools}</div>`)}</div>` : ''}
          ${woSection('workOrder.sideEffects', effects ? `<div class="flex flex-wrap gap-1">${effects}</div>` : '')}
          ${woSection('workOrder.expected', order.expected_outcome ? mdBlock(order.expected_outcome) : '')}
          ${woSection('workOrder.fallback', order.fallback ? mdBlock(order.fallback) : '')}
        </div>`;
  const all = order.assumptions || [];
  const dropped = all.filter(a => a.rejected).length;
  const counts = [
    opts.changes ? woCount('woStrip.changes', '+' + opts.changes, 'text-secondary') : '',
    woCount('woStrip.steps', woStepTally(order.steps), 'text-on-surface', 'data-wo-progress'),
    all.length ? woCount('woStrip.assumptions', dropped ? `${all.length - dropped}/${all.length}` : all.length,
      '', 'data-wo-assumption-count') : '',
  ];
  // Deviations are warnings: they stay outside the strip, where nobody has to
  // open anything to see that the agent stepped off its contract.
  return woStrip(counts, order.goal, (opts.prefix || '') + prose + controls, opts.open)
    + `<div data-wo-deviations class="flex flex-col gap-1 mt-1"></div>`;
}

// ── The two rows the card is answered with ────────────────────────────────
function chipsRow(order, interactive) {
  const chips = (order.assumptions || []).map(a => `
            <label title="${escHtml(a.id + ' — ' + (a.text || ''))}"
              class="inline-flex items-baseline gap-1 px-1.5 py-0.5 rounded border cursor-pointer
                     font-mono text-[11px] border-outline-variant/30 hover:border-primary/50
                     transition-colors ${a.rejected ? 'line-through-dim' : 'text-on-surface-variant'}">
              <input type="checkbox" data-wo-assumption="${escHtml(a.id)}"
                class="accent-primary align-middle"
                ${a.rejected ? '' : 'checked'} ${interactive ? '' : 'disabled'} />
              ${escHtml(a.id)}
            </label>`).join('');
  if (!chips) return '';
  return `<div class="flex items-baseline flex-wrap gap-1.5">
            <span class="text-[12px] text-outline-variant shrink-0">${hitlLabel('workOrder.assumptions')}</span>
            ${chips}
          </div>`;
}

// `done` is not the only way a step closes: the server counts `skipped` as
// closed too (work_order.report_warnings reads open_steps the same way), so a
// run that skipped one would otherwise read 3/4 for ever.
const WO_STEP_OPEN = new Set(['pending', 'in_progress']);

// The "2/4" beside the chips lives on the strip's line now (woCount with
// data-wo-progress), where it can be read without opening anything.
function stepsRow(order) {
  const steps = order.steps || [];
  if (!steps.length) return '';
  return `<div class="flex items-baseline flex-wrap gap-1.5">
            <span class="text-[12px] text-outline-variant shrink-0">${hitlLabel('workOrder.steps')}</span>
            <ol data-wo-steps data-compact="1" class="inline-flex flex-wrap items-baseline gap-1">${
    steps.map(woStepChip).join('')}</ol>
          </div>`;
}

// How many things an amendment adds — the number on the strip's line.
function workOrderDiffCount(ctx) {
  const diff = (ctx && ctx.diff) || {};
  return (diff.added_tools || []).length + (diff.added_side_effects || []).length
    + (diff.added_steps || []).length;
}

function workOrderDiff(ctx) {
  const diff = ctx.diff || {};
  const rows = [];
  (diff.added_tools || []).forEach(tn => rows.push('+ ' + tn));
  (diff.added_side_effects || []).forEach(e => rows.push('+ ' + e.kind + (e.detail ? ': ' + e.detail : '')));
  (diff.added_steps || []).forEach(st => rows.push('+ ' + st.id + '. ' + st.title));
  const reason = ctx.reason ? woSection('workOrder.reason', `<div class="text-on-surface">${mdBlock(ctx.reason)}</div>`) : '';
  const changes = rows.length ? woSection('workOrder.added',
    `<pre class="font-mono text-[12px] text-secondary whitespace-pre-wrap bg-surface-container-high px-2.5 py-2 rounded border border-outline-variant/20">${escHtml(rows.join('\n'))}</pre>`) : '';
  return reason + changes;
}

function woHeader(icon, titleKey, tier, agent, revision) {
  const tierCls = WO_TIER_STYLE[tier] || WO_TIER_STYLE.compute;
  const displayAgent = (window.StatusIndicator && StatusIndicator.agentName)
    ? StatusIndicator.agentName(agent)
    : (agent || '—');
  return `
        <div class="flex items-center gap-3 flex-wrap">
          <div class="w-8 h-8 rounded-lg bg-tertiary/10 flex items-center justify-center" aria-hidden="true">
            <span class="material-symbols-outlined text-tertiary text-[18px]">${icon}</span>
          </div>
          <h3 class="font-headline font-bold text-base text-on-surface uppercase tracking-tight">${hitlLabel(titleKey)}</h3>
          ${tier ? `<span class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border ${tierCls}">${hitlLabel('workOrder.tier.' + tier)}</span>` : ''}
          <span class="text-[12px] font-bold text-on-surface">${escHtml(displayAgent)}</span>
          <span class="font-mono text-[10px] text-outline-variant">rev ${escHtml(String(revision || 1))}</span>
        </div>`;
}

function renderWorkOrderCard(live, data) {
  const rid = data.request_id;
  const ctx = data.context || {};
  const order = ctx.work_order || {};
  const tier = ctx.tier || order.tier || 'compute';
  const isAmendment = data.trigger === 'work_order_amendment';
  // A history card never counts down: its window has closed.
  const timeout = live ? (Number(data.timeout_seconds) || 0) : 0;
  const messageHtml = hitlDynamic(data, 'message', localizeHitlMessage(data));


  const countdown = woCountdown(live, data, timeout, 'workOrder.countdown');

  placeHitlCard(rid, `
        <div class="my-6 relative msg-enter max-w-4xl" data-hitl-card="${escHtml(rid || '')}" data-wo-agent="${escHtml(data.agent_name || '')}" data-wo-rev="${escHtml(String(order.revision || 1))}">
          <div class="relative bg-surface-container-low p-5 rounded-xl border border-primary/40 shadow-2xl">
            ${woHeader(isAmendment ? 'edit_note' : 'assignment', isAmendment ? 'workOrder.amendTitle' : 'workOrder.title',
    tier, data.agent_name, order.revision)}
            <p class="text-sm text-on-surface-variant leading-relaxed mt-2">${messageHtml}</p>
            ${documentBlock(data)}
            ${workOrderBody(order, rid, !isAmendment, !!(data.document && data.document.artifact_id), {
    open: woStripOpen(live),
    prefix: isAmendment ? `<div class="text-xs text-on-surface-variant leading-relaxed mb-3">${workOrderDiff(ctx)}</div>` : '',
    changes: isAmendment ? workOrderDiffCount(ctx) : 0,
  })}
            ${countdown}
            <div id="hitl-controls-${rid}" class="mt-4 flex flex-col gap-2">
              <textarea id="hitl-feedback-${rid}" rows="2" data-i18n-placeholder="workOrder.ph.notes" placeholder="${escHtml(t('workOrder.ph.notes'))}"
                class="w-full bg-surface-container-high border border-outline-variant/25 rounded-md px-2.5 py-2 text-[13px] text-on-surface placeholder:text-outline-variant focus:outline-none focus:border-primary/50"></textarea>
              <div class="flex flex-wrap gap-3">
                <button onclick="respondWorkOrder('${rid}', 'approve')" class="flex items-center justify-center gap-2 bg-primary text-on-primary px-4 py-2 rounded-md font-bold text-[12px] uppercase tracking-[0.08em] shadow-lg shadow-primary/20 hover:brightness-110 active:scale-95 transition-all">
                  <span class="material-symbols-outlined text-base">check_circle</span> ${hitlLabel('hitl.btn.accept')}
                </button>
                ${timeout > 0 && !data.held ? `
                <button id="wo-pause-${rid}" onclick="holdWorkOrder('${rid}')" class="flex items-center justify-center gap-2 bg-surface-container-high border border-tertiary/30 text-tertiary px-4 py-2 rounded-md font-bold text-[12px] uppercase tracking-[0.08em] hover:bg-tertiary/10 transition-all">
                  <span class="material-symbols-outlined text-base">pause</span> ${hitlLabel('workOrder.btn.pause')}
                </button>` : ''}
                <button onclick="respondWorkOrder('${rid}', 'reject')" class="flex items-center justify-center gap-2 bg-surface-container-high border border-outline-variant/20 text-error px-4 py-2 rounded-md font-bold text-[12px] uppercase tracking-[0.08em] hover:bg-error/10 transition-all">
                  <span class="material-symbols-outlined text-base">close</span> ${hitlLabel('hitl.btn.reject')}
                </button>
              </div>
            </div>
          </div>
        </div>`);

  if (timeout > 0 && !data.held) startWorkOrderCountdown(rid, timeout);
}

// The veto countdown (or "waiting for your decision") under a live card.
function woCountdown(live, data, timeout, textKey) {
  const rid = data.request_id;
  if (!live) return '';
  if (timeout > 0 && !data.held) {
    return `
        <div id="wo-countdown-${rid}" data-wo-countdown-key="${escHtml(textKey)}" class="mt-4 flex flex-col gap-1">
          <p class="text-[11px] text-tertiary" data-wo-countdown-text>${escHtml(t(textKey).replace('{s}', Math.ceil(timeout)))}</p>
          <div class="h-1 w-full bg-surface-container-high rounded overflow-hidden">
            <div data-wo-countdown-bar class="h-full bg-tertiary transition-[width] duration-1000 ease-linear" style="width:100%"></div>
          </div>
        </div>`;
  }
  return `
        <p id="wo-countdown-${rid}" class="mt-4 text-[11px] text-outline-variant">${hitlLabel(data.held ? 'workOrder.paused' : 'workOrder.blocking')}</p>`;
}

function startWorkOrderCountdown(rid, timeout) {
  stopWorkOrderCountdown(rid);
  const deadline = Date.now() + timeout * 1000;
  const tick = () => {
    const box = document.getElementById('wo-countdown-' + rid);
    if (!box) return stopWorkOrderCountdown(rid);
    const left = Math.max(0, (deadline - Date.now()) / 1000);
    const text = box.querySelector('[data-wo-countdown-text]');
    const bar = box.querySelector('[data-wo-countdown-bar]');
    const key = box.dataset.woCountdownKey || 'workOrder.countdown';
    if (text) text.textContent = t(key).replace('{s}', Math.ceil(left));
    if (bar) bar.style.width = (100 * left / timeout) + '%';
    if (left <= 0) stopWorkOrderCountdown(rid);
  };
  tick();
  woTimers.set(rid, setInterval(tick, 1000));
}

function stopWorkOrderCountdown(rid) {
  const timer = woTimers.get(rid);
  if (timer) clearInterval(timer);
  woTimers.delete(rid);
}

// Local effect of a hold (this tab pressed Pause, or another tab did).
function applyWorkOrderHold(rid) {
  stopWorkOrderCountdown(rid);
  const box = document.getElementById('wo-countdown-' + rid);
  if (box) {
    box.className = 'mt-4 text-[11px] text-outline-variant';
    box.innerHTML = hitlLabel('workOrder.paused');
  }
  const pause = document.getElementById('wo-pause-' + rid);
  if (pause) pause.remove();
}
window.applyWorkOrderHold = applyWorkOrderHold;

function holdWorkOrder(rid) {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: 'hitl_hold', request_id: rid }));
  }
  applyWorkOrderHold(rid);
}

function respondWorkOrder(rid, action) {
  const feedbackEl = document.getElementById('hitl-feedback-' + rid);
  const feedback = feedbackEl ? feedbackEl.value.trim() : '';
  // The single approval button doubles as "revise" when notes were entered.
  if (action === 'approve' && feedback) action = 'edit';
  if (action === 'edit' && !feedback) {
    addSystemMsg(t('hitl.reviseEmpty'));
    if (feedbackEl) feedbackEl.focus();
    return;
  }
  const card = feedbackEl ? feedbackEl.closest('[data-hitl-card]') : null;
  const rejectedIds = card
    ? [...card.querySelectorAll('input[data-wo-assumption]')].filter(el => !el.checked).map(el => el.dataset.woAssumption)
    : [];
  const disputedIds = card
    ? [...card.querySelectorAll('input[data-wr-finding]')].filter(el => el.checked).map(el => el.dataset.wrFinding)
    : [];
  // The strip is about to fold; its line has to say what was just decided.
  woRecount(card);
  // What the operator marked travels with EVERY action, not with one of them.
  //
  // These two lists used to be attached per action — rejections only on
  // `approve`, disputes only on `edit` — while the single button silently turns
  // `approve` into `edit` the moment a note is typed (above). So the ordinary
  // move of unticking an assumption AND saying why dropped the untick on the
  // floor: the agent was asked to revise and never told which assumption the
  // operator had rejected. Marking something and being ignored is worse than
  // having no checkbox at all.
  const formValues = {};
  if (rejectedIds.length) formValues.rejected_assumption_ids = rejectedIds;
  if (disputedIds.length) formValues.disputed_finding_ids = disputedIds;
  if (!sendHitlResponse({
    type: 'hitl_response',
    request_id: rid,
    action: action,
    approved: action === 'approve',
    instructions: feedback || null,
    free_input: feedback || null,
    form_values: Object.keys(formValues).length ? formValues : null,
  })) return;
  const box = document.getElementById('wo-countdown-' + rid);
  if (box) box.remove();
}

// ── Work Report cards ────────────────────────────────────────────────────
const WR_VERDICT_STYLE = {
  met: 'text-primary border-primary/30 bg-primary/10',
  partial: 'text-tertiary border-tertiary/30 bg-tertiary/10',
  not_met: 'text-error border-error/30 bg-error/10',
};
const WR_CONFIDENCE_STYLE = {
  high: 'text-primary border-primary/30 bg-primary/10',
  medium: 'text-on-surface-variant border-outline-variant/30 bg-surface-container-high',
  low: 'text-tertiary border-tertiary/30 bg-tertiary/10',
};

function wrWarning(w) {
  const params = {
    steps: (w.steps || []).join(', '),
    findings: (w.findings || []).join(', '),
    count: w.count != null ? w.count : '',
    verdict: t('workReport.verdict.' + w.verdict, w.verdict || ''),
  };
  return `<span class="text-[11px] text-tertiary">⚠ ${escHtml(fillHitl('workReport.warn.' + w.code, params))}</span>`;
}

function wrFindingRow(f, interactive) {
  const head = `
            <span class="font-mono text-[10px] text-outline-variant">${escHtml(f.id)}</span>
            <span class="text-on-surface">${mdInline(f.text || '')}</span>
            ${woChip(t('workReport.confidence.' + f.confidence, f.confidence || ''), WR_CONFIDENCE_STYLE[f.confidence] || WR_CONFIDENCE_STYLE.medium)}
            ${f.step_id ? `<span class="font-mono text-[10px] text-outline-variant">${escHtml(f.step_id)}</span>` : ''}`;
  const evidence = f.evidence
    ? `<p class="pl-6 text-[11px] font-mono text-secondary break-all whitespace-pre-wrap">${escHtml(f.evidence)}</p>`
    : `<p class="pl-6 text-[11px] text-tertiary italic">${hitlLabel('workReport.noEvidence')}</p>`;
  if (interactive) {
    return `
          <li class="flex flex-col gap-0.5">
            <label class="flex items-baseline gap-2 flex-wrap cursor-pointer">
              <input type="checkbox" data-wr-finding="${escHtml(f.id)}" class="accent-error" title="${escHtml(t('workReport.markWrong'))}" />
              ${head}
            </label>
            ${evidence}
          </li>`;
  }
  const struck = f.disputed ? 'line-through' : '';
  return `
          <li class="flex flex-col gap-0.5">
            <div class="flex items-baseline gap-2 flex-wrap pl-6 ${struck}">${head}</div>
            ${evidence}
          </li>`;
}

// The report set against the order. interactive=true lets the human mark findings wrong.
// `open` opens the strip the lists sit behind (see woStripOpen).
function workReportBody(order, report, extra, interactive, compact, open) {
  const warnings = (extra.warnings || []).map(wrWarning).join('');
  const journal = extra.journal || {
    tool_calls: order.tool_calls || {},
    side_effects: [],
    amendments: order.amendments || [],
    deviations: order.deviations || [],
  };
  const disputed = new Set(report.disputed_finding_ids || []);
  const findings = (report.findings || [])
    .map(f => wrFindingRow({ ...f, disputed: disputed.has(f.id) }, interactive)).join('');
  // No verdict, no chip. A replayed report can carry an empty `done_verdict`
  // (replay.py records the call even when the tool refused it), and
  // t('workReport.verdict.') would print its own key — now on the strip's line,
  // the first thing anyone reads.
  const verdict = report.fallback || !report.done_verdict ? '' : woChip(
    t('workReport.verdict.' + report.done_verdict, report.done_verdict),
    WR_VERDICT_STYLE[report.done_verdict] || WR_VERDICT_STYLE.partial);
  const done = order.done_criteria || verdict ? `
            <p>${mdInline(order.done_criteria || '')} ${verdict}</p>
            ${report.done_evidence ? `<div class="text-[11px] text-outline-variant">${mdBlock(report.done_evidence)}</div>` : ''}` : '';
  const outcome = order.expected_outcome || report.actual_outcome ? `
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <div><p class="text-[10px] text-outline-variant">${hitlLabel('workOrder.expected')}</p>${mdBlock(order.expected_outcome || '—')}</div>
              <div><p class="text-[10px] text-outline-variant">${hitlLabel('workReport.actual')}</p><div class="text-on-surface">${mdBlock(report.actual_outcome || '—')}</div></div>
            </div>` : '';
  const artifacts = (report.artifacts || []).map(a => `
            <p class="flex items-baseline gap-2 flex-wrap">${woChip(t('workReport.kind.' + a.kind, a.kind || 'other'))}
              <span class="font-mono text-[11px] text-on-surface break-all">${escHtml(a.ref || '')}</span>
              ${a.description ? `<span class="text-[11px] text-outline-variant">${mdInline(a.description)}</span>` : ''}</p>`).join('');
  const calls = Object.entries(journal.tool_calls || {}).map(([tn, n]) => woChip(`${tn} ×${n}`)).join(' ');
  const effects = (journal.side_effects || []).map(k => woChip(k, WO_TIER_STYLE.side_effect)).join(' ');
  const amendments = (journal.amendments || []).map(a =>
    `<p class="text-[11px]"><span class="font-mono text-outline-variant">rev ${escHtml(String(a.revision || '?'))}</span> ${mdInline(a.reason || '')}</p>`).join('');
  const deviations = (journal.deviations || []).map(d =>
    `<p class="text-[11px] text-tertiary font-mono">⚠ ${escHtml(d.tool || '?')} — ${escHtml(t('workOrder.reason.' + d.reason, d.reason || ''))}</p>`).join('');
  const journalHtml = calls || effects || amendments || deviations ? `
            <div class="flex flex-col gap-1.5">
              ${calls ? `<div class="flex flex-wrap gap-1">${calls}</div>` : ''}
              ${effects ? `<div class="flex flex-wrap gap-1 items-center"><span class="text-[10px] text-outline-variant">${hitlLabel('workReport.sideEffectsDone')}</span> ${effects}</div>` : ''}
              ${amendments ? `<div><p class="text-[10px] text-outline-variant">${hitlLabel('workReport.amendments')}</p>${amendments}</div>` : ''}
              ${deviations ? `<div><p class="text-[10px] text-outline-variant">${hitlLabel('workReport.deviations')}</p>${deviations}</div>` : ''}
            </div>` : '';
  const summary = report.fallback
    ? `<p class="text-[10px] text-outline-variant mb-1">${hitlLabel('workReport.finalAnswer')}</p>
       ${foldable(`<div class="md-body hitl-prose text-on-surface border-l-2 border-primary/40 pl-4 pr-1">`
         + `${renderMarkdown(report.summary || '—')}</div>`, report.summary || '', { bg: 'rgb(var(--c-surface-container-low))' })}`
    : `<div class="text-on-surface">${mdBlock(report.summary || '')}</div>`;
  const findingsBlock = woSection('workReport.findings', findings
    ? (interactive ? `<p class="text-[12px] text-on-surface-variant mb-1.5">${hitlLabel('workReport.findingsHint')}</p>` : '')
      + `<ol class="flex flex-col gap-1.5">${findings}</ol>`
    : '');
  // A warning is the one thing that must not wait for someone to open a panel,
  // so it stays outside the strip — as one wrapped line, not a boxed list.
  const warningsBlock = warnings
    ? `<div class="mt-2 flex flex-wrap gap-x-4 gap-y-0.5">${warnings}</div>`
    : '';

  const all = report.findings || [];
  const nDisputed = all.filter(f => disputed.has(f.id)).length;
  const counts = [
    verdict,
    all.length ? woCount('woStrip.findings', nDisputed ? `${all.length} ✗${nDisputed}` : all.length,
      '', 'data-wr-finding-count') : '',
    woCount('woStrip.steps', woStepTally(order.steps), 'text-on-surface'),
    (report.artifacts || []).length ? woCount('woStrip.artifacts', report.artifacts.length) : '',
  ];
  const lead = report.fallback ? '' : (report.summary || report.actual_outcome || order.goal);

  // With a document the findings are all the card keeps: they carry the dispute
  // checkboxes, and the rest is in the file this card opens.
  // No findings in a compact card means nothing to list: an empty body, so no
  // strip at all rather than one that opens onto an empty box.
  const body = compact ? (findingsBlock ? `
        <div class="text-xs text-on-surface-variant leading-relaxed">${findingsBlock}</div>` : '') : `
        <div class="text-xs text-on-surface-variant leading-relaxed">
          ${woSection('workOrder.goal', mdBlock(order.goal || ''))}
          ${woSection('workReport.summary', summary)}
          ${findingsBlock}
          ${woSection('workOrder.done', done)}
          ${woSection('workReport.outcome', outcome)}
          ${woSection('workReport.steps', (order.steps || []).length
      ? `<ol class="flex flex-col gap-1.5">${order.steps.map(s => woStepRow(s, false)).join('')}</ol>` : '')}
          ${woSection('workReport.artifacts', artifacts ? `<div class="flex flex-col gap-1">${artifacts}</div>` : '')}
          ${woSection('workReport.journal', journalHtml)}
        </div>`;
  return warningsBlock + woStrip(counts, lead, body, open);
}

// Counts react at the moment a mark changes, not only after Submit/replay.
document.addEventListener('change', event => {
  const target = event.target;
  if (!target || !target.matches
      || !target.matches('input[data-wo-assumption], input[data-wr-finding]')) return;
  woRecount(target.closest('[data-hitl-card]'));
});

// Once the operator opens/closes a strip themselves it is no longer the strip
// auto-opened for an unanswered card. Timeouts and language redraws preserve
// that explicit choice.
document.addEventListener('toggle', event => {
  const details = event.target;
  if (!event.isTrusted || !details || !details.matches
      || !details.matches('details[data-wo-strip]')) return;
  details.removeAttribute('data-auto-open');
  const card = details.closest('[data-hitl-card]');
  if (card && card.dataset.hitlCard) {
    workOrderStripState.set(card.dataset.hitlCard, details.open);
  }
}, true);

function renderWorkReportCard(live, data) {
  const rid = data.request_id;
  const ctx = data.context || {};
  const order = ctx.work_order || {};
  const report = ctx.work_report || {};
  const tier = ctx.tier || order.tier || 'compute';
  const timeout = live ? (Number(data.timeout_seconds) || 0) : 0;
  const messageHtml = hitlDynamic(data, 'message', localizeHitlMessage(data));

  placeHitlCard(rid, `
        <div class="my-6 relative msg-enter max-w-4xl" data-hitl-card="${escHtml(rid || '')}" data-wr-agent="${escHtml(data.agent_name || '')}">
          <div class="relative bg-surface-container-low p-5 rounded-xl border border-primary/40 shadow-2xl">
            ${woHeader('fact_check', 'workReport.title', tier, data.agent_name, order.revision)}
            <p class="font-mono text-[10px] text-outline-variant mt-1">${escHtml(t('workReport.round').replace('{n}', report.round || 1))}</p>
            <p class="text-sm text-on-surface-variant leading-relaxed mt-2">${messageHtml}</p>
            ${documentBlock(data)}
            ${workReportBody(order, report, ctx, !report.fallback, !!(data.document && data.document.artifact_id),
    woStripOpen(live))}
            ${woCountdown(live, data, timeout, 'workReport.countdown')}
            <div id="hitl-controls-${rid}" class="mt-4 flex flex-col gap-2">
              <textarea id="hitl-feedback-${rid}" rows="2" data-i18n-placeholder="workReport.ph.notes" placeholder="${escHtml(t('workReport.ph.notes'))}"
                class="w-full bg-surface-container-high border border-outline-variant/25 rounded-md px-2.5 py-2 text-[13px] text-on-surface placeholder:text-outline-variant focus:outline-none focus:border-primary/50"></textarea>
              <div class="flex flex-wrap gap-3">
                <button onclick="respondWorkOrder('${rid}', 'approve')" class="flex items-center justify-center gap-2 bg-primary text-on-primary px-4 py-2 rounded-md font-bold text-[12px] uppercase tracking-[0.08em] shadow-lg shadow-primary/20 hover:brightness-110 active:scale-95 transition-all">
                  <span class="material-symbols-outlined text-base">check_circle</span> ${hitlLabel('hitl.btn.accept')}
                </button>
                ${timeout > 0 && !data.held ? `
                <button id="wo-pause-${rid}" onclick="holdWorkOrder('${rid}')" class="flex items-center justify-center gap-2 bg-surface-container-high border border-tertiary/30 text-tertiary px-4 py-2 rounded-md font-bold text-[12px] uppercase tracking-[0.08em] hover:bg-tertiary/10 transition-all">
                  <span class="material-symbols-outlined text-base">pause</span> ${hitlLabel('workOrder.btn.pause')}
                </button>` : ''}
                <button onclick="respondWorkOrder('${rid}', 'reject')" class="flex items-center justify-center gap-2 bg-surface-container-high border border-outline-variant/20 text-error px-4 py-2 rounded-md font-bold text-[12px] uppercase tracking-[0.08em] hover:bg-error/10 transition-all">
                  <span class="material-symbols-outlined text-base">close</span> ${hitlLabel('hitl.btn.reject')}
                </button>
              </div>
            </div>
          </div>
        </div>`);

  if (timeout > 0 && !data.held) startWorkOrderCountdown(rid, timeout);
}

// ── Work Step cards (step review) ────────────────────────────────────────
function wsCallRow(call) {
  const err = call.is_error;
  return `
          <li class="flex flex-col gap-0.5">
            <div class="flex items-baseline gap-2 flex-wrap">
              ${woChip(call.tool || '?', err ? WS_REVIEW_STYLE.rejected : '')}
              ${err ? `<span class="text-[10px] text-error font-bold uppercase">${escHtml(t('workStep.callError'))}</span>` : ''}
            </div>
            ${call.args ? `<pre class="pl-2 text-[11px] font-mono text-secondary whitespace-pre-wrap break-all bg-surface-container-high p-2 rounded border border-outline-variant/10 max-h-40 overflow-auto">${escHtml(call.args)}</pre>` : ''}
            ${call.result_excerpt ? `<pre class="pl-2 text-[11px] font-mono ${err ? 'text-error' : 'text-on-surface-variant'} whitespace-pre-wrap break-all bg-surface-container-high p-2 rounded border border-outline-variant/10 max-h-56 overflow-auto">${escHtml(call.result_excerpt)}</pre>` : ''}
          </li>`;
}

function workStepBody(order, step, calls) {
  const review = step.review || {};
  const history = (review.history || []).map(h => `
            <p class="text-[11px] text-outline-variant"><span class="font-mono">${escHtml(t('workReport.round').replace('{n}', h.round || '?'))}</span>
              ${mdInline(h.result || '—')}${h.notes ? ` — <span class="text-tertiary">${mdInline(h.notes)}</span>` : ''}</p>`).join('');
  const callRows = (calls || []).map(wsCallRow).join('');
  return `
        <div class="text-xs text-on-surface-variant leading-relaxed">
          ${woSection('workOrder.goal', mdBlock(order.goal || ''))}
          ${woSection('workStep.step', `<p class="text-on-surface"><span class="font-mono text-[10px] text-outline-variant">${escHtml(step.id || '')}</span> ${mdInline(step.title || '')}</p>
            <div class="flex flex-wrap gap-1 mt-1">${woToolChips(step.tools, step.internal_tools)}</div>`)}
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-3">
            <div><p class="text-[10px] font-bold text-outline-variant uppercase tracking-wider mb-1">${hitlLabel('workStep.sent')}</p>${mdBlock(step.inputs || '—')}</div>
            <div><p class="text-[10px] font-bold text-outline-variant uppercase tracking-wider mb-1">${hitlLabel('workStep.expected')}</p>${mdBlock(step.expected_outcome || '—')}</div>
            <div><p class="text-[10px] font-bold text-outline-variant uppercase tracking-wider mb-1">${hitlLabel('workStep.foundTitle')}</p><div class="text-on-surface">${mdBlock(step.result || '—')}</div>
              ${step.note ? `<div class="text-[11px] text-outline-variant italic mt-1">${mdBlock(step.note)}</div>` : ''}</div>
          </div>
          ${woSection('workStep.calls', callRows
    ? `<ol class="flex flex-col gap-2">${callRows}</ol>`
    : `<p class="text-[11px] text-tertiary">⚠ ${escHtml(t('workStep.noCalls'))}</p>`)}
          ${woSection('workStep.history', history)}
        </div>`;
}

function renderWorkStepCard(live, data) {
  const rid = data.request_id;
  const ctx = data.context || {};
  const order = ctx.work_order || {};
  const step = ctx.step || {};
  const tier = ctx.tier || order.tier || 'compute';
  const timeout = live ? (Number(data.timeout_seconds) || 0) : 0;
  const messageHtml = hitlDynamic(data, 'message', localizeHitlMessage(data));
  const round = (step.review || {}).round || 1;


  placeHitlCard(rid, `
        <div class="my-6 relative msg-enter" data-hitl-card="${escHtml(rid || '')}" data-ws-agent="${escHtml(data.agent_name || '')}">
          <div class="relative bg-surface-container-lowest p-6 rounded-xl border border-primary/30 shadow-2xl">
            ${woHeader('checklist', 'workStep.title', tier, data.agent_name, order.revision)}
            <p class="font-mono text-[10px] text-outline-variant mt-1">${escHtml(t('workReport.round').replace('{n}', round))}</p>
            <p class="text-sm text-on-surface-variant leading-relaxed mt-2">${messageHtml}</p>
            ${workStepBody(order, step, ctx.step_calls || step.calls)}
            ${woCountdown(live, data, timeout, 'workStep.countdown')}
            <div id="hitl-controls-${rid}" class="mt-4 flex flex-col gap-2">
              <textarea id="hitl-feedback-${rid}" rows="2" data-i18n-placeholder="workStep.ph.notes" placeholder="${escHtml(t('workStep.ph.notes'))}"
                class="w-full bg-surface-container-high border border-outline-variant/20 rounded-md p-2 font-mono text-[11px] text-on-surface placeholder:text-outline-variant focus:outline-none focus:border-primary/50"></textarea>
              <div class="flex flex-wrap gap-3">
                <button onclick="respondWorkOrder('${rid}', 'approve')" class="flex items-center justify-center gap-2 bg-primary text-on-primary px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] shadow-lg shadow-primary/20 hover:brightness-110 active:scale-95 transition-all">
                  <span class="material-symbols-outlined text-base">check_circle</span> ${hitlLabel('hitl.btn.accept')}
                </button>
                ${timeout > 0 && !data.held ? `
                <button id="wo-pause-${rid}" onclick="holdWorkOrder('${rid}')" class="flex items-center justify-center gap-2 bg-surface-container-high border border-tertiary/30 text-tertiary px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] hover:bg-tertiary/10 transition-all">
                  <span class="material-symbols-outlined text-base">pause</span> ${hitlLabel('workOrder.btn.pause')}
                </button>` : ''}
                <button onclick="respondWorkOrder('${rid}', 'edit')" class="flex items-center justify-center gap-2 bg-surface-container-high border border-outline-variant/20 text-on-surface px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] hover:bg-surface-container-highest transition-all">
                  <span class="material-symbols-outlined text-base">replay</span> ${hitlLabel('workStep.btn.redo')}
                </button>
                <button onclick="respondWorkOrder('${rid}', 'reject')" class="flex items-center justify-center gap-2 bg-surface-container-high border border-outline-variant/20 text-error px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] hover:bg-error/10 transition-all">
                  <span class="material-symbols-outlined text-base">stop_circle</span> ${hitlLabel('workStep.btn.stop')}
                </button>
              </div>
            </div>
          </div>
        </div>`);

  if (timeout > 0 && !data.held) startWorkOrderCountdown(rid, timeout);
}

function latestWorkOrderCard(agent) {
  const cards = document.querySelectorAll(`[data-wo-agent="${CSS.escape(agent || '')}"]`);
  return cards.length ? cards[cards.length - 1] : null;
}

// Non-blocking notices: a read-tier contract, step progress, a deviation, and
// (from a replayed transcript) a submitted Work Report.
function renderWorkOrderNotice(data) {
  const kind = data.kind;
  if (kind === 'report') {
    const order = data.work_order || {};
    const report = data.work_report || {};
    appendMsgToFeed(`
          <div class="my-3 relative msg-enter max-w-4xl" data-wr-agent="${escHtml(data.agent_name || '')}">
            <div class="relative bg-surface-container-low px-4 py-3 rounded-xl border border-outline-variant/25">
              ${woHeader('fact_check', 'workReport.title', data.tier || order.tier, data.agent_name, order.revision)}
              ${workReportBody(order, report, data, false)}
            </div>
          </div>`);
    scrollChat();
    return;
  }
  if (kind === 'declared' || kind === 'amended') {
    const order = data.work_order || {};
    const amended = kind === 'amended';
    appendMsgToFeed(`
          <div class="my-3 relative msg-enter max-w-4xl" data-wo-agent="${escHtml(data.agent_name || '')}" data-wo-rev="${escHtml(String(order.revision || 1))}">
            <div class="relative bg-surface-container-low px-4 py-3 rounded-xl border border-outline-variant/25">
              ${woHeader(amended ? 'edit_note' : 'assignment', amended ? 'workOrder.amendTitle' : 'workOrder.noticeTitle',
      data.tier || order.tier, data.agent_name, order.revision)}
              ${workOrderBody(order, '', false, false, {
      prefix: amended ? `<div class="text-xs text-on-surface-variant leading-relaxed mb-3">${workOrderDiff(data)}</div>` : '',
      changes: amended ? workOrderDiffCount(data) : 0,
    })}
            </div>
          </div>`);
    scrollChat();
    return;
  }
  const card = latestWorkOrderCard(data.agent_name);
  if (kind === 'progress' && data.step) {
    if (!card) return;
    let row = card.querySelector(`[data-wo-step="${CSS.escape(data.step.id)}"]`);
    const list = card.querySelector('[data-wo-steps]');
    // A card whose detail moved to a document lists its steps compactly; a
    // replacement row has to match, or one tick would bring the detail back.
    const html = woStepRow(data.step, !!list && list.dataset.compact === '1');
    if (row) {
      row.outerHTML = html;
    } else if (list) {
      list.insertAdjacentHTML('beforeend', html);
    }
    // The strip's line carries a "2/4"; without this it would still read 0/4
    // while the steps inside went green one by one — and with the strip closed,
    // that number is the only progress anyone sees.
    const tallies = card.querySelectorAll('[data-wo-progress]');
    if (tallies.length && list) {
      const steps = [...list.querySelectorAll('[data-wo-step]')];
      const closed = steps.filter(el => !WO_STEP_OPEN.has(el.dataset.woStatus || 'pending')).length;
      tallies.forEach(el => { el.textContent = `${closed}/${steps.length}`; });
    }
    return;
  }
  if (kind === 'deviation' && data.deviation) {
    const dev = data.deviation;
    const line = `⚠ ${t('workOrder.deviation')}: ${dev.tool} — ${t('workOrder.reason.' + dev.reason, dev.reason)}`;
    const box = card ? card.querySelector('[data-wo-deviations]') : null;
    if (box) {
      box.insertAdjacentHTML('beforeend', `<p class="text-[11px] text-tertiary font-mono">${escHtml(line)}</p>`);
    }
    addTelemetry('WORK ORDER :: ' + (data.agent_name || '?') + ' ' + line);
  }
}


// ── Experiment plan review ───────────────────────────────────────────────
// The plan the Experiment Module builds during a run is the one thing the
// human is asked to approve, and it used to arrive as Markdown in a <pre>:
// a design matrix written as pipe-separated rows, ten task sections under
// it, the whole thing in a 24rem scroll box. Nobody approves that; they
// approve whatever they can see in the first screen of it.
//
// The backend now ships the plan structured (context.experiment_plan, see
// CoScientist/experiments/plan_view.py), so it is drawn as what it is: a
// header with the goal and the totals, the design matrix as a real table,
// and one foldable card per task carrying that task's whole design. The
// answer is unchanged — Accept / Revise / Reject through the same handlers
// as any other review — so nothing downstream has to know about this view.

const planOpenTasks = new Set();   // "<request_id>:<task_id>" of unfolded cards
// The plan a card was drawn from, so folding a task re-renders from data
// rather than from the DOM it is about to replace.
const planByRequest = new Map();

const PLAN_ROUTE_TONE = {
  fedot_mas: 'text-primary border-primary/30 bg-primary/5',
  react_tools: 'text-primary border-primary/30 bg-primary/5',
  coder: 'text-tertiary border-tertiary/30 bg-tertiary/5',
  alembic_build: 'text-secondary border-secondary/30 bg-secondary/5',
  research: 'text-outline-variant border-outline-variant/30 bg-outline-variant/5',
  medical: 'text-outline-variant border-outline-variant/30 bg-outline-variant/5',
};
const PLAN_CHIP_TONE = 'text-on-surface-variant border-outline-variant/30 bg-surface-container-high';

// A slot the planner left empty is shown as empty. Printing its placeholder
// text ("unspecified", "n/a") made an unfilled design look filled in.
function planDash() { return '<span class="text-outline-variant/60">—</span>'; }

// Planner prose (a question, a description, a rationale) is markdown.
function planText(value) {
  const text = (value === 0 || value) ? String(value).trim() : '';
  return text ? mdInline(text) : planDash();
}

// A list reads as a list, not as one comma-glued line: a task's metrics,
// baselines and tools are each several short names and run together badly.
function planItems(items, empty) {
  const rows = (items || []).filter(x => x !== null && x !== undefined && String(x).trim());
  if (!rows.length) return empty === undefined ? planDash() : escHtml(empty);
  return rows.map(x => `<span class="inline-block bg-surface-container-high border border-outline-variant/25 rounded px-1.5 py-0.5 mr-1 mb-1 text-[11px] font-mono">${escHtml(String(x))}</span>`).join('');
}

function planChip(label, value, tone) {
  const name = label ? `<span class="opacity-80 uppercase tracking-wider">${escHtml(label)}</span>` : '';
  return `<span class="inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px] font-mono ${tone || PLAN_CHIP_TONE}">${name}${escHtml(String(value))}</span>`;
}

function planRouteChip(route) {
  return `<span class="inline-flex items-center rounded-md border px-2 py-0.5 text-[11px] font-mono ${PLAN_ROUTE_TONE[route] || PLAN_CHIP_TONE}">${escHtml(route || '')}</span>`;
}

function planField(label, valueHtml) {
  return `<div class="grid grid-cols-[minmax(104px,max-content)_1fr] gap-x-3 py-1.5 border-b border-outline-variant/15 last:border-0">
    <span class="text-[11px] uppercase tracking-wider text-outline-variant pt-0.5">${escHtml(label)}</span>
    <span class="text-[12px] text-on-surface-variant leading-relaxed break-words min-w-0">${valueHtml}</span>
  </div>`;
}

function planSection(title, bodyHtml) {
  if (!bodyHtml) return '';
  return `<div class="mt-3 pt-3 border-t border-outline-variant/15 first:mt-0 first:pt-0 first:border-0">
    <p class="text-[11px] font-bold text-on-surface-variant uppercase tracking-wider mb-1.5">${escHtml(title)}</p>
    ${bodyHtml}</div>`;
}

const PLAN_MATRIX_COLUMNS = [
  'plan.col.task', 'plan.col.hypothesis', 'plan.col.question', 'plan.col.dataset',
  'plan.col.baselines', 'plan.col.metrics', 'plan.col.tools', 'plan.col.artifacts',
  'plan.col.route',
];

function planMatrix(plan) {
  if (!Array.isArray(plan.matrix) || !plan.matrix.length) return '';
  const head = PLAN_MATRIX_COLUMNS
    .map(key => `<th class="text-left font-bold uppercase tracking-wider text-[11px] text-on-surface-variant px-2 py-2 whitespace-nowrap">${escHtml(t(key))}</th>`)
    .join('');
  const rows = plan.matrix.map(r => `
    <tr class="border-t border-outline-variant/10 align-top">
      <td class="px-2 py-1.5 font-mono text-[11px] text-primary whitespace-nowrap">${escHtml(r.task_id || '')}</td>
      <td class="px-2 py-1.5 font-mono text-[10px] whitespace-nowrap">${planText(r.hypothesis)}</td>
      <td class="px-2 py-1.5 text-[11px] min-w-[220px]">${planText(r.question)}</td>
      <td class="px-2 py-1.5 text-[11px]">${planText(r.dataset)}</td>
      <td class="px-2 py-1.5">${planItems(r.baselines)}</td>
      <td class="px-2 py-1.5">${planItems(r.metrics)}</td>
      <td class="px-2 py-1.5">${planItems(r.tools)}</td>
      <td class="px-2 py-1.5">${planItems(r.artifacts)}</td>
      <td class="px-2 py-1.5 whitespace-nowrap">${planRouteChip(r.route)}</td>
    </tr>`).join('');
  return planSection(t('plan.matrix'), `
    <div class="overflow-x-auto rounded-lg border border-outline-variant/10 bg-surface-container-high/30">
      <table class="w-full border-collapse text-on-surface-variant"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>
    </div>`);
}

function planTools(task) {
  const servers = task.mcp_servers || [];
  if (!servers.length) return planItems([], t('plan.noTools'));
  return servers.map(s => {
    const where = s.url ? ` <span class="text-outline-variant/70 break-all">${escHtml(s.url)}</span>` : '';
    const names = (s.tools || []).map(x => x.name + (x.required ? '' : ' (' + t('plan.optionalTool') + ')'));
    return `<div class="mb-1"><span class="font-mono text-[11px] text-on-surface">${escHtml(s.name || '')}</span>${where}
      <div class="mt-0.5">${planItems(names)}</div></div>`;
  }).join('');
}

function planCriteria(task) {
  const rows = task.success_criteria || [];
  if (!rows.length) return planDash();
  return rows.map(c => `<div class="mb-1.5">
    <span class="font-mono text-[11px] text-primary">${escHtml(c.criterion_id || '')}</span>
    <span class="text-[11px] uppercase tracking-wider text-outline-variant ml-1">${escHtml(c.kind || '')}</span>
    ${c.threshold ? `<span class="ml-1 font-mono text-[10px] text-tertiary">${escHtml(c.threshold)}</span>` : ''}
    <div class="text-[11px]">${planText(c.description)}</div>
    ${c.verification ? `<div class="text-[10px] text-outline-variant">${mdInline(c.verification)}</div>` : ''}
  </div>`).join('');
}

function planInputs(task) {
  const rows = task.input_data || [];
  if (!rows.length) return `<span class="text-outline-variant/60">${escHtml(t('plan.noInputs'))}</span>`;
  return rows.map(d => `<div class="mb-1">
    <span class="font-mono text-[10px] text-on-surface">${escHtml(d.data_id || '')}</span>
    <span class="text-[11px] uppercase tracking-wider text-outline-variant ml-1">${escHtml(d.kind || '')}</span>
    ${d.location ? `<div class="font-mono text-[10px] text-outline-variant break-all">${escHtml(d.location)}</div>` : ''}
    ${d.description ? `<div class="text-[11px]">${mdInline(d.description)}</div>` : ''}
  </div>`).join('');
}

function planDatasetCell(dataset) {
  if (!dataset || !dataset.name) return planDash();
  const ref = dataset.ref ? ` <span class="font-mono text-[10px] text-outline-variant break-all">${escHtml(dataset.ref)}</span>` : '';
  const notes = dataset.notes ? `<div class="text-[10px] text-outline-variant">${mdInline(dataset.notes)}</div>` : '';
  return escHtml(dataset.name) + ref + notes;
}

function planTaskCard(rid, task, index) {
  const open = planOpenTasks.has(rid + ':' + task.id);
  const design = task.design || {};
  const params = Object.entries(task.launch_params || {});
  const codeAssessment = task.code_assessment || {};
  const body = !open ? '' : `
    <div class="px-3 pb-3">
      ${planField(t('plan.task.question'), planText(design.question))}
      ${planField(t('plan.task.dataset'), planDatasetCell(design.dataset))}
      ${planField(t('plan.task.baselines'), planItems((design.baselines || []).map(b => b.name + ' (' + b.kind + ')')))}
      ${planField(t('plan.task.metrics'), planItems((design.metrics || []).map(m =>
        m.name + ' ' + m.direction + (m.threshold ? ' ' + m.threshold : '') + (m.test ? ' [' + m.test + ']' : ''))))}
      ${planField(t('plan.task.analysis'), planItems((design.analysis_artifacts || []).map(a => a.name + ' [' + a.role + '/' + a.prepare_via + ']')))}
      ${planField(t('plan.task.description'), planText(task.description))}
      ${task.rationale ? planField(t('plan.task.rationale'), planText(task.rationale)) : ''}
      ${planField(t('plan.task.tools'), planTools(task))}
      ${task.repo_url ? planField(t('plan.task.repo'), `<span class="font-mono text-[10px] break-all">${escHtml(task.repo_url)}</span>`) : ''}
      ${(codeAssessment.requirement && codeAssessment.requirement !== 'unknown') ? planField(
        t('plan.task.codeAssessment'),
        `<span class="font-mono text-[10px] text-primary">${escHtml(codeAssessment.requirement)}</span>` +
        (codeAssessment.evidence ? `<div class="text-[11px] mt-0.5">${mdInline(codeAssessment.evidence)}</div>` : '') +
        ((codeAssessment.entrypoints || []).length ? `<div class="font-mono text-[10px] text-outline-variant mt-0.5">${escHtml(codeAssessment.entrypoints.join(' · '))}</div>` : '')
      ) : ''}
      ${params.length ? planField(t('plan.task.params'), planItems(params.map(p => p[0] + '=' + p[1]))) : ''}
      ${planField(t('plan.task.inputs'), planInputs(task))}
      ${planField(t('plan.task.criteria'), planCriteria(task))}
      ${planField(t('plan.task.expected'), planItems((task.expected_artifacts || []).map(a => a.name + ' [' + a.role + ']')))}
      ${(task.warnings || []).length ? planField(t('plan.task.warnings'),
        `<span class="text-tertiary">${escHtml(task.warnings.join(' · '))}</span>`) : ''}
    </div>`;
  return `<div class="rounded-lg border border-outline-variant/10 bg-surface-container-high/30 mb-2">
    <button type="button" onclick="togglePlanTask('${escJs(rid)}','${escJs(task.id)}')"
      class="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-surface-container-high/60 rounded-lg transition-colors">
      <span class="material-symbols-outlined text-sm text-outline-variant">${open ? 'expand_more' : 'chevron_right'}</span>
      <span class="font-mono text-[11px] text-primary">${escHtml(task.id || ('#' + (index + 1)))}</span>
      <span class="text-[13px] font-medium text-on-surface truncate flex-1">${escHtml(task.name || '')}</span>
      ${task.optional ? planChip('', t('plan.task.optional')) : ''}
      ${(task.depends_on || []).length ? planChip(t('plan.task.after'), task.depends_on.join(', ')) : ''}
      ${planChip('', (task.est_duration_min || 0) + ' ' + t('plan.min'))}
      ${planRouteChip(task.route)}
    </button>
    ${body}</div>`;
}

function repaintPlanTasks(rid) {
  const box = document.getElementById('plan-tasks-' + rid);
  const plan = planByRequest.get(rid);
  if (!box || !plan) return;
  box.innerHTML = plan.tasks.map((task, i) => planTaskCard(rid, task, i)).join('');
  const toggle = document.getElementById('plan-toggle-all-' + rid);
  if (toggle) {
    const allOpen = plan.tasks.every(task => planOpenTasks.has(rid + ':' + task.id));
    toggle.textContent = allOpen ? t('plan.collapseAll') : t('plan.expandAll');
  }
}

function togglePlanTask(rid, taskId) {
  const key = rid + ':' + taskId;
  if (planOpenTasks.has(key)) planOpenTasks.delete(key); else planOpenTasks.add(key);
  repaintPlanTasks(rid);
}

function togglePlanAllTasks(rid) {
  const plan = planByRequest.get(rid);
  if (!plan) return;
  const allOpen = plan.tasks.every(task => planOpenTasks.has(rid + ':' + task.id));
  plan.tasks.forEach(task => {
    if (allOpen) planOpenTasks.delete(rid + ':' + task.id);
    else planOpenTasks.add(rid + ':' + task.id);
  });
  repaintPlanTasks(rid);
}

function planCritiqueBlock(plan) {
  const critique = plan.critique;
  if (!critique) return '';
  const approved = critique.verdict === 'approve';
  const issues = (critique.issues || []).map(i => `<div class="mb-1">
    <span class="text-[9px] uppercase tracking-wider ${i.severity === 'blocker' ? 'text-error' : 'text-tertiary'}">${escHtml(i.severity || '')}</span>
    <span class="text-[11px] uppercase tracking-wider text-outline-variant ml-1">${escHtml(i.category || '')}</span>
    ${i.task_id ? `<span class="font-mono text-[11px] text-primary ml-1">${escHtml(i.task_id)}</span>` : ''}
    <div class="text-[11px]">${planText(i.message)}</div>
    ${i.suggestion ? `<div class="text-[10px] text-outline-variant">${mdInline(i.suggestion)}</div>` : ''}
  </div>`).join('');
  return planSection(t('plan.critique'),
    `<p class="text-[11px] ${approved ? 'text-secondary' : 'text-tertiary'} mb-1">${escHtml(approved ? t('plan.critique.approve') : t('plan.critique.revise'))}</p>${issues}`);
}

function planBullets(list) {
  return (list || []).length
    ? `<ul class="list-disc list-inside text-[11px] text-on-surface-variant space-y-0.5">${list.map(x => `<li>${mdInline(x)}</li>`).join('')}</ul>`
    : '';
}

function renderExperimentPlanReview(live, data) {
  const plan = (data.context || {}).experiment_plan;
  const rid = data.request_id;
  const exhausted = !!plan.review_exhausted;
  planByRequest.set(rid, plan);


  const hypotheses = (plan.hypotheses || []).map(h =>
    `<div class="mb-1"><span class="font-mono text-[11px] text-primary">${escHtml(h.id || '')}</span>
      <span class="text-[11px] ml-1">${planText(h.statement)}</span></div>`).join('');

  // Everything that is read rather than glanced at. The goal stays above the
  // fold with the chips: between them they say what this plan is, which is
  // what someone scrolling past needs.
  const detail = `
    ${plan.hypothesis ? planField(t('plan.hypothesis'), planText(plan.hypothesis)) : ''}
    ${planField(t('plan.methods'), planItems(plan.methods))}
    ${hypotheses ? planSection(t('plan.hypotheses'), hypotheses) : ''}
    ${planCritiqueBlock(plan)}
    ${planMatrix(plan)}
    <div class="mt-3 flex items-center justify-between">
      <p class="text-[13px] font-bold text-on-surface-variant uppercase tracking-wider">${escHtml(t('plan.tasksTitle'))}</p>
      <button type="button" id="plan-toggle-all-${escHtml(rid)}" onclick="togglePlanAllTasks('${escJs(rid)}')"
        class="text-[13px] uppercase tracking-wider text-primary hover:underline">${escHtml(t('plan.expandAll'))}</button>
    </div>
    <div id="plan-tasks-${escHtml(rid)}" class="mt-1">${plan.tasks.map((task, i) => planTaskCard(rid, task, i)).join('')}</div>
    ${planSection(t('plan.risks'), planBullets(plan.risks))}
    ${planSection(t('plan.assumptions'), planBullets(plan.assumptions))}`;

  placeHitlCard(rid, `
<div data-hitl-card="${escHtml(rid || '')}" class="my-6 relative msg-enter max-w-4xl">
  <div class="absolute -inset-2 bg-gradient-to-r from-primary/10 via-transparent to-primary/10 blur-2xl opacity-40"></div>
  <div class="relative bg-surface-container-low p-5 rounded-xl border border-primary/40 shadow-2xl">
    <div class="flex items-center gap-3 mb-2">
      <div class="w-8 h-8 rounded-lg bg-tertiary/10 flex items-center justify-center" aria-hidden="true">
        <span class="material-symbols-outlined text-tertiary text-[18px]">science</span>
      </div>
      <h3 class="font-headline font-bold text-base text-on-surface uppercase tracking-tight">${escHtml(t('plan.title'))}</h3>
    </div>
    <p class="text-[10px] text-outline-variant font-mono mb-2">CTX: ${escHtml(String(rid).slice(0, 8))} · ${escHtml(data.agent_name || '')}</p>
    ${exhausted ? `<div class="mb-3 rounded-lg border border-tertiary/35 bg-tertiary/10 px-3 py-2">
      <p class="text-[12px] font-semibold text-tertiary">${escHtml(t('plan.exhausted.title'))}</p>
      <p class="text-[11px] text-on-surface-variant mt-0.5">${escHtml(t(
        plan.recovered_previous_candidate ? 'plan.exhausted.previousBody' : 'plan.exhausted.body'
      ))}</p>
    </div>` : ''}
    <div class="flex flex-wrap gap-1.5 mb-3">
      ${planChip('', t('plan.revision').replace('{n}', plan.revision), 'text-primary border-primary/30 bg-primary/5')}
      ${planChip('', t('plan.tasks').replace('{n}', plan.task_count))}
      ${planChip('', (plan.total_est_duration_min || 0) + ' ' + t('plan.min'))}
      ${(plan.routes || []).map(planRouteChip).join('')}
      ${plan.plan_id ? planChip('id', plan.plan_id) : ''}
    </div>
    ${planField(t('plan.goal'), planText(plan.goal))}
    ${documentBlock(data) || foldable(detail, JSON.stringify(plan), { bg: 'rgb(var(--c-surface-container-low))' })}
    <!-- Only the answer is disabled once this review is over (timeout, or
         the operator has answered): the plan stays readable and its task
         cards stay foldable, which is the whole point of drawing it. -->
    <div id="hitl-controls-${escHtml(rid)}" class="mt-4 flex flex-col gap-2">
      <textarea id="hitl-feedback-${escHtml(rid)}" rows="2" placeholder="${escHtml(t(exhausted ? 'plan.approvalNotePlaceholder' : 'plan.feedbackPlaceholder'))}"
        class="w-full bg-surface-container-high border border-outline-variant/25 rounded-md px-2.5 py-2 text-[13px] text-on-surface placeholder:text-outline-variant focus:outline-none focus:border-primary/50"></textarea>
      <div class="flex flex-wrap gap-3">
        <button onclick="respondHITLApprove('${escJs(rid)}')" class="flex items-center justify-center gap-2 bg-primary text-on-primary px-4 py-2 rounded-md font-bold text-[12px] uppercase tracking-[0.08em] shadow-lg shadow-primary/20 hover:brightness-110 active:scale-95 transition-all">
          <span class="material-symbols-outlined text-base">check_circle</span> ${escHtml(t(exhausted ? 'plan.acceptWithIssues' : 'plan.accept'))}
        </button>
        ${exhausted ? '' : `<button onclick="respondHITLEdit('${escJs(rid)}')" class="flex items-center justify-center gap-2 bg-surface-container-high border border-outline-variant/20 text-on-surface px-4 py-2 rounded-md font-bold text-[12px] uppercase tracking-[0.08em] hover:bg-surface-container-highest transition-all">
          <span class="material-symbols-outlined text-base">edit_note</span> ${escHtml(t('plan.revise'))}
        </button>`}
        <button onclick="respondHITL('${escJs(rid)}', false)" class="flex items-center justify-center gap-2 bg-surface-container-high border border-outline-variant/20 text-error px-4 py-2 rounded-md font-bold text-[12px] uppercase tracking-[0.08em] hover:bg-error/10 transition-all">
          <span class="material-symbols-outlined text-base">close</span> ${escHtml(t('plan.reject'))}
        </button>
      </div>
    </div>
  </div>
</div>`);
}
