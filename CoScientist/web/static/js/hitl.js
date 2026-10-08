// =========================================================================
// HITL UI
// =========================================================================
    // =========================================================================
    // HITL UI
    // =========================================================================
    // Internal-loop HITL requests arrive with a fixed English message
    // ("Agent 'X' proposes its result. Please review.", legacy ones carry an
    // "[INTERNAL_LOOP: ...]" prefix). Show a localized version instead.
    const INTERNAL_LOOP_RE = /^(?:\[INTERNAL_LOOP:[^\]]*\]\s*)?Agent '([^']+)' proposes its result\. Please review\.?\s*$/;
    function localizeHitlMessage(message) {
      const m = INTERNAL_LOOP_RE.exec(message || '');
      if (!m) return message;
      return t('hitl.internalLoop').replace('{agent}', m[1]);
    }

    function showHITL(data) {
      const panel = document.getElementById('hitl-panel');
      const feed = document.getElementById('chat-feed');

      // Structured intake (e.g. the research frame): render a per-field form
      // instead of the free-text review, then stop — the other HITL points keep
      // the free-text / option path below.
      if (data.form && Array.isArray(data.form.blocks)) {
        renderHitlForm(panel, feed, data);
        scrollChat();
        return;
      }

      let openRoadmapSidebarBtn = '';
      let openRoadmapChatBtn = '';
      if (data.agent_name === 'PlannerAgent') {
        openRoadmapSidebarBtn = `
      <button onclick="openRoadmapEditor()" class="w-full mt-2 flex items-center justify-center gap-2 bg-surface-variant border border-outline-variant/20 text-on-surface py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] hover:bg-surface-container-high transition-all">
        <span class="material-symbols-outlined text-sm">map</span> Open Roadmap
      </button>
    `;
        openRoadmapChatBtn = `
      <div class="mt-4 pl-11">
        <button onclick="openRoadmapEditor()" class="flex items-center justify-center gap-2 bg-surface-variant border border-outline-variant/20 text-on-surface px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-wider hover:bg-surface-container-high transition-all">
          <span class="material-symbols-outlined text-sm">map</span> Open Roadmap
        </button>
      </div>
    `;
      }

      const isProvideInput = data.action_type === 'provide_input';
      const hasOptions = !!(data.options && data.options.length);

      // Show in sidebar. For question windows (options present) or input requests the sidebar is
      // informational only — answer directly in the chat card.
      const sidebarButtons = (hasOptions || isProvideInput) ? `
        <p class="text-[10px] text-outline-variant leading-relaxed">Ответьте в карточке в чате.</p>` : `
        <div class="flex gap-3">
          <button onclick="respondHITL('${data.request_id}', true)" class="flex-1 flex items-center justify-center gap-2 bg-primary text-on-primary py-3 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] shadow-lg shadow-primary/20 hover:brightness-110 active:scale-95 transition-all">
            <span class="material-symbols-outlined text-base">check_circle</span> Accept
          </button>
          <button onclick="respondHITL('${data.request_id}', false)" class="flex-1 flex items-center justify-center gap-2 bg-surface-container-high border border-outline-variant/20 text-error py-3 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] hover:bg-error/10 transition-all">
            <span class="material-symbols-outlined text-base">close</span> Reject
          </button>
        </div>`;
      panel.classList.remove('hidden');
      panel.innerHTML = `
    <div class="relative">
      <div class="absolute -inset-2 bg-gradient-to-r from-primary/10 via-transparent to-primary/10 blur-2xl opacity-40"></div>
      <div class="relative bg-surface-container-lowest p-6 rounded-xl border border-primary/30 shadow-2xl flex flex-col gap-4">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center shadow-[0_0_15px_rgba(0,218,243,0.4)]">
            <span class="material-symbols-outlined text-on-primary text-sm">ads_click</span>
          </div>
          <h3 class="font-headline font-bold text-on-surface text-sm uppercase tracking-tight">HITL Required</h3>
        </div>
        <p class="text-xs text-on-surface-variant leading-relaxed">${escHtml(localizeHitlMessage(data.message))}</p>
        ${sidebarButtons}
        ${openRoadmapSidebarBtn}
      </div>
    </div>`;

      // Also show in chat: the proposed output itself + Accept / Revise controls.
      const proposedOutput = (data.context && data.context.output) ? String(data.context.output) : '';
      const outputBlock = proposedOutput ? `
        <div class="mt-3 pl-11">
          <p class="text-[10px] font-bold text-outline-variant uppercase tracking-wider mb-1">Proposed output</p>
          <pre class="font-mono text-[11px] leading-relaxed text-on-surface-variant whitespace-pre-wrap bg-surface-container-high p-3 rounded-lg border border-outline-variant/10 max-h-96 overflow-auto">${escHtml(proposedOutput)}</pre>
        </div>` : '';
      appendMsgToFeed(`
    <div class="my-6 relative msg-enter">
      <div class="absolute -inset-2 bg-gradient-to-r from-primary/10 via-transparent to-primary/10 blur-2xl opacity-40"></div>
      <div class="relative bg-surface-container-lowest p-6 rounded-xl border border-primary/30 shadow-2xl">
        <div class="flex items-center gap-3 mb-3">
          <div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center shadow-[0_0_15px_rgba(0,218,243,0.4)]">
            <span class="material-symbols-outlined text-on-primary text-sm">ads_click</span>
          </div>
          <h3 class="font-headline font-bold text-on-surface uppercase tracking-tight">Human-In-The-Loop Required</h3>
        </div>
        <p class="text-sm text-on-surface-variant leading-relaxed pl-11">${escHtml(localizeHitlMessage(data.message))}</p>
        <p class="text-[10px] text-outline-variant font-mono mt-2 pl-11">CTX: ${data.request_id.slice(0, 8)} · ${escHtml(data.agent_name || '')}</p>
        ${outputBlock}
        ${isProvideInput ? `
        <div id="hitl-controls-${data.request_id}" class="mt-4 pl-11 flex flex-col gap-2">
          <textarea id="hitl-feedback-${data.request_id}" rows="2" placeholder="Введите инструкции для агента..."
            class="w-full bg-surface-container-high border border-outline-variant/20 rounded-md p-2 font-mono text-[11px] text-on-surface placeholder:text-outline-variant focus:outline-none focus:border-primary/50"></textarea>
          <div class="flex">
            <button onclick="respondHITLInput('${data.request_id}')" class="flex items-center justify-center gap-2 bg-primary text-on-primary px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] shadow-lg shadow-primary/20 hover:brightness-110 active:scale-95 transition-all">
              <span class="material-symbols-outlined text-base">send</span> Отправить
            </button>
          </div>
        </div>` : hasOptions ? `
        <div id="hitl-controls-${data.request_id}" class="mt-4 pl-11 flex flex-col gap-2">
          <textarea id="hitl-feedback-${data.request_id}" rows="2" placeholder="Ваш ответ на вопрос — затем «Ответить»"
            class="w-full bg-surface-container-high border border-outline-variant/20 rounded-md p-2 font-mono text-[11px] text-on-surface placeholder:text-outline-variant focus:outline-none focus:border-primary/50"></textarea>
          <div class="flex flex-wrap gap-2">
            <button onclick="respondHITLEdit('${data.request_id}')" class="flex items-center justify-center gap-2 bg-primary text-on-primary px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] shadow-lg shadow-primary/20 hover:brightness-110 active:scale-95 transition-all">
              <span class="material-symbols-outlined text-base">reply</span> Ответить
            </button>
            ${data.options.map(o => `
            <button onclick="respondHITLOption('${data.request_id}', '${escJs(o)}')" class="flex items-center justify-center gap-2 bg-surface-container-high border border-outline-variant/20 text-on-surface px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] hover:bg-surface-container-highest transition-all">${escHtml(o)}</button>`).join('')}
          </div>
        </div>` : `
        <div id="hitl-controls-${data.request_id}" class="mt-4 pl-11 flex flex-col gap-2">
          <textarea id="hitl-feedback-${data.request_id}" rows="2" placeholder="Правки для агента — затем Revise"
            class="w-full bg-surface-container-high border border-outline-variant/20 rounded-md p-2 font-mono text-[11px] text-on-surface placeholder:text-outline-variant focus:outline-none focus:border-primary/50"></textarea>
          <div class="flex gap-3">
            <button onclick="respondHITL('${data.request_id}', true)" class="flex items-center justify-center gap-2 bg-primary text-on-primary px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] shadow-lg shadow-primary/20 hover:brightness-110 active:scale-95 transition-all">
              <span class="material-symbols-outlined text-base">check_circle</span> Accept
            </button>
            <button onclick="respondHITLEdit('${data.request_id}')" class="flex items-center justify-center gap-2 bg-surface-container-high border border-outline-variant/20 text-on-surface px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] hover:bg-surface-container-highest transition-all">
              <span class="material-symbols-outlined text-base">edit_note</span> Revise
            </button>
            <button onclick="respondHITL('${data.request_id}', false)" class="flex items-center justify-center gap-2 bg-surface-container-high border border-outline-variant/20 text-error px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] hover:bg-error/10 transition-all">
              <span class="material-symbols-outlined text-base">close</span> Reject
            </button>
          </div>
        </div>`}
        ${openRoadmapChatBtn}
      </div>
    </div>`);
    }

    function disableHitlControls(requestId) {
      const box = document.getElementById('hitl-controls-' + requestId);
      if (!box) return;
      box.querySelectorAll('button, textarea').forEach(el => {
        el.disabled = true;
        el.classList.add('opacity-40', 'pointer-events-none');
      });
    }

    let currentPlannerHitlRequest = null;

    function sendHitlResponse(payload) {
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify(payload));
      }
      if (window.StatusIndicator) {
        StatusIndicator.feed({ type: 'hitl_response', request_id: payload.request_id });
      }
    }

    function respondHITLInput(requestId) {
      const feedbackEl = document.getElementById('hitl-feedback-' + requestId);
      const feedback = feedbackEl ? feedbackEl.value.trim() : '';
      sendHitlResponse({
        type: 'hitl_response',
        request_id: requestId,
        action: 'provide_input',
        approved: true,
        instructions: feedback,
        free_input: feedback,
      });
      document.getElementById('hitl-panel').classList.add('hidden');
      disableHitlControls(requestId);
      addSystemMsg('💬 HITL Input: ' + (feedback || '(empty)'));

      if (currentPlannerHitlRequest && currentPlannerHitlRequest.request_id === requestId) {
        currentPlannerHitlRequest = null;
        updateRoadmapModalButtons();
      }
    }

    function respondHITL(requestId, approved) {
      const feedbackEl = document.getElementById('hitl-feedback-' + requestId);
      const feedback = feedbackEl ? feedbackEl.value.trim() : '';
      sendHitlResponse({
        type: 'hitl_response',
        request_id: requestId,
        action: approved ? 'approve' : 'reject',
        approved: approved,
        instructions: feedback || null,
        free_input: feedback || null,
      });
      document.getElementById('hitl-panel').classList.add('hidden');
      disableHitlControls(requestId);
      addSystemMsg(approved ? '✓ HITL Approved' : '✗ HITL Rejected' + (feedback ? ': ' + feedback : ''));

      if (currentPlannerHitlRequest && currentPlannerHitlRequest.request_id === requestId) {
        currentPlannerHitlRequest = null;
        updateRoadmapModalButtons();
      }
    }

    function respondHITLOption(requestId, option) {
      // A question-window option button: a complete answer by itself.
      sendHitlResponse({
        type: 'hitl_response',
        request_id: requestId,
        action: 'select',
        approved: true,
        selected_option: option,
        instructions: option,
        free_input: option,
      });
      document.getElementById('hitl-panel').classList.add('hidden');
      disableHitlControls(requestId);
      addSystemMsg('☑ ' + option);
    }

    function respondHITLEdit(requestId) {
      // Send the operator's corrections: the agent rewrites its output with them.
      const feedbackEl = document.getElementById('hitl-feedback-' + requestId);
      const feedback = feedbackEl ? feedbackEl.value.trim() : '';
      if (!feedback) {
        addSystemMsg('Введите правки в поле выше, затем нажмите Revise.');
        if (feedbackEl) feedbackEl.focus();
        return;
      }
      sendHitlResponse({
        type: 'hitl_response',
        request_id: requestId,
        action: 'edit',
        approved: false,
        instructions: feedback,
        free_input: feedback,
      });
      document.getElementById('hitl-panel').classList.add('hidden');
      disableHitlControls(requestId);
      addSystemMsg('✎ HITL Revision requested: ' + feedback);

      if (currentPlannerHitlRequest && currentPlannerHitlRequest.request_id === requestId) {
        currentPlannerHitlRequest = null;
        updateRoadmapModalButtons();
      }
    }

    // ── Structured frame form (research frame intake) ────────────────────────
    function renderHitlForm(panel, feed, data) {
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
            ? `<span class="text-[9px] text-error uppercase tracking-wider">${escHtml(t('hitl.form.notSet'))}</span>`
            : `<span class="text-[9px] text-outline-variant uppercase tracking-wider">${escHtml(f.status || '')}</span>`;
          const val = f.open ? '' : String(f.value || '');
          const label = loc(f.label, f.name);
          const placeholder = loc(f.placeholder, t('hitl.form.placeholderFallback'));
          return `
            <div class="flex flex-col gap-1">
              <div class="flex items-center justify-between">
                <label class="text-[11px] font-mono text-on-surface-variant">${escHtml(label)}</label>
                ${openTag}
              </div>
              <textarea id="frm-${rid}-${bi}-${fi}" data-block="${escJs(b.title)}" data-field="${escJs(f.name)}"
                rows="2" placeholder="${escHtml(placeholder)}"
                class="w-full bg-surface-container-high border border-outline-variant/20 rounded-md p-2 font-mono text-[11px] text-on-surface placeholder:text-outline-variant focus:outline-none focus:border-primary/50">${escHtml(val)}</textarea>
            </div>`;
        }).join('');
        const blockTitle = loc(b.title_i18n, b.title);
        const blockUsage = loc(b.usage_i18n, b.usage);
        return `
          <div class="mt-3 border border-outline-variant/10 rounded-lg p-3 bg-surface-container-high/40">
            <p class="text-[11px] font-bold text-on-surface uppercase tracking-wider">${escHtml(blockTitle)}</p>
            ${blockUsage ? `<p class="text-[10px] text-outline-variant mb-2">${escHtml(blockUsage)}</p>` : '<div class="mb-2"></div>'}
            <div class="flex flex-col gap-2">${fieldsHtml}</div>
          </div>`;
      }).join('');

      panel.classList.remove('hidden');
      panel.innerHTML = `
        <div class="relative bg-surface-container-lowest p-4 rounded-xl border border-primary/30 shadow-2xl flex flex-col gap-2">
          <h3 class="font-headline font-bold text-on-surface text-sm uppercase tracking-tight">${escHtml(t('hitl.form.sidebarTitle'))}</h3>
          <p class="text-[11px] text-on-surface-variant">${escHtml(t('hitl.form.sidebarHint'))}</p>
        </div>`;

      const formTitle = loc(form.title_i18n, form.title || t('hitl.form.title'));
      const formIntro = loc(form.intro_i18n, form.intro || data.message || '');
      feed.innerHTML += `
        <div id="hitl-controls-${rid}" class="my-6 relative msg-enter">
          <div class="relative bg-surface-container-lowest p-6 rounded-xl border border-primary/30 shadow-2xl">
            <div class="flex items-center gap-3 mb-2">
              <div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center shadow-[0_0_15px_rgba(0,218,243,0.4)]">
                <span class="material-symbols-outlined text-on-primary text-sm">fact_check</span>
              </div>
              <h3 class="font-headline font-bold text-on-surface uppercase tracking-tight">${escHtml(formTitle)}</h3>
            </div>
            <p class="text-xs text-on-surface-variant leading-relaxed">${escHtml(formIntro)}</p>
            ${blocksHtml}
            <div class="flex flex-wrap gap-3 mt-4">
              <button onclick="respondHITLForm('${rid}', true)" class="flex items-center justify-center gap-2 bg-primary text-on-primary px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] shadow-lg shadow-primary/20 hover:brightness-110 active:scale-95 transition-all">
                <span class="material-symbols-outlined text-base">check_circle</span> ${escHtml(t('hitl.form.save'))}
              </button>
              <button onclick="respondHITLForm('${rid}', false)" class="flex items-center justify-center gap-2 bg-surface-container-high border border-outline-variant/20 text-on-surface px-4 py-2 rounded-md font-bold text-[10px] uppercase tracking-[0.15em] hover:bg-surface-container-highest transition-all">
                <span class="material-symbols-outlined text-base">skip_next</span> ${escHtml(t('hitl.form.skip'))}
              </button>
            </div>
          </div>
        </div>`;
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
      sendHitlResponse({
        type: 'hitl_response',
        request_id: requestId,
        action: 'approve',
        approved: true,
        form_values: formValues,
      });
      document.getElementById('hitl-panel').classList.add('hidden');
      disableHitlControls(requestId);
      const n = formValues ? Object.values(formValues).reduce((s, o) => s + Object.keys(o).length, 0) : 0;
      addSystemMsg(collect ? t('hitl.form.saved').replace('{n}', n) : t('hitl.form.skipped'));
    }

