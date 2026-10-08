// =========================================================================
// Chat rendering & Input handling
// =========================================================================
    function scrollChat() {
      const feed = document.getElementById('chat-feed');
      feed.scrollTop = feed.scrollHeight;
    }


    function appendMsgToFeed(html) {
      const feed = document.getElementById('chat-feed');
      const placeholder = feed.querySelector('.text-outline-variant.font-medium');
      if (placeholder) {
        feed.innerHTML = '';
      }
      feed.insertAdjacentHTML('beforeend', html);
      scrollChat();
    }

    function addSystemMsg(text, timestamp = null) {
      appendMsgToFeed(`
    <div class="flex flex-col gap-2 max-w-2xl msg-enter">
      <div class="flex items-center gap-2">
        <span class="text-[10px] font-mono text-primary/70 bg-primary/5 px-2 py-0.5 rounded uppercase">System</span>
        <span class="text-[10px] text-outline-variant font-mono">${ts(timestamp)}</span>
      </div>
      <div class="bg-surface-container-lowest p-3 rounded-lg border border-outline-variant/10">
        <p class="font-mono text-[11px] leading-relaxed text-on-surface-variant whitespace-pre-wrap">${escHtml(text)}</p>
      </div>
    </div>`);

    }

    marked.setOptions({ breaks: true, gfm: true });

    const THINKING_RE = /<(think|thought)>([\s\S]*?)(?:<\/\1>|$)/gi;

    function stripThinking(text) {
      if (!text || typeof text !== 'string') return '';
      return text.replace(THINKING_RE, '').trim();
    }

    // The reasoning an agent shows on the way to an answer is worth reading —
    // it is how you tell a considered answer from a lucky one — but it is not
    // the answer, and printed inline it buries it. So it is kept, marked as
    // reasoning, and folded when long.
    function takeThinking(text) {
      if (!text || typeof text !== 'string') return '';
      const parts = [];
      String(text).replace(THINKING_RE, (_, _tag, body) => {
        const t = String(body || '').trim();
        if (t) parts.push(t);
        return '';
      });
      return parts.join('\n\n');
    }

    // Reasoning folds shorter than a deliverable: it is context, not the point.
    const THINKING_FOLD = 420;

    function thinkingBlock(text) {
      const thinking = takeThinking(text);
      if (!thinking) return '';
      const folded = thinking.length > THINKING_FOLD;
      const toggle = folded
        ? `<button onclick="toggleAgentOutput(this)"
             class="self-start text-[10px] font-mono uppercase tracking-widest text-outline-variant hover:text-on-surface transition-colors">
             Show full reasoning</button>`
        : '';
      return `
        <div class="flex flex-col gap-1.5 mb-2 pb-2 border-b border-outline-variant/15">
          <span class="self-start text-[9px] font-mono uppercase tracking-widest text-outline-variant bg-outline-variant/10 border border-outline-variant/20 px-1.5 py-0.5 rounded">Thinking</span>
          <div class="${folded ? 'max-h-24 overflow-hidden' : ''}">
            <div class="text-xs text-outline-variant leading-relaxed md-body">${renderMarkdown(thinking)}</div>
          </div>
          ${toggle}
        </div>`;
    }

    function renderMarkdown(text) {
      // Agents write their messages in markdown; render it to sanitized HTML
      // instead of showing the raw '**'/'`'/'#' syntax as plain text.
      const raw = typeof text === 'string' ? stripThinking(text) : String(text == null ? '' : text);
      const withLocalLinks = raw
        .replace(/(^|\s)(\/api\/tz-document[^\s)]*)/g, '$1[$2]($2)')
        // Live MCP build page: /alembic/builds/<job_id> (agent surfaces it as progress_page).
        .replace(/(^|\s)(\/alembic\/builds\/[A-Za-z0-9._-]+)/g, '$1[$2]($2)');
      const html = marked.parse(withLocalLinks);
      const clean = DOMPurify.sanitize(html, { ADD_ATTR: ['target'] });
      // A link whose URL path (before any ?query) ends in an image extension
      // gets an inline <img> preview under the link — markdown images render
      // on their own, but agents also emit image URLs as plain links. The & in
      // presigned S3 query strings shows up here as &amp; — the pattern allows
      // it.
      const withPreviews = clean.replace(
        /<a href="(https?:\/\/[^"]+|\/(?!\/)[^"]*)">([^<]*)<\/a>/g,
        (m, url, label) => {
          if (!/\.(png|jpe?g|gif|webp|svg|bmp|avif)$/i.test(url.split(/[?#]/)[0])) return m;
          return m + '<br><a href="' + url + '"><img src="' + url + '" alt="' + label + '"'
            + ' class="mt-2 max-w-full rounded-lg border border-outline-variant/20 cursor-zoom-in" loading="lazy"></a>';
        });
      return withPreviews.replace(/<a /g, '<a target="_blank" rel="noopener noreferrer" class="text-primary underline" ');
    }

    function getBaseSandboxUrl() {
      const customUrl = (typeof appSettings !== 'undefined' && appSettings.coderAgent && appSettings.coderAgent.sandboxUrl)
        ? appSettings.coderAgent.sandboxUrl.trim() : '';
      if (customUrl) {
        return customUrl.endsWith('/') ? customUrl : customUrl + '/';
      }
      return 'http://localhost:8884/';
    }

    function extractSandboxUrlFromText(text) {
      if (!text || typeof text !== 'string') return null;
      const matches = text.match(/(https?:\/\/[^\s\n"']+)/g);
      if (!matches) return null;
      for (const url of matches) {
        if (url.includes('task_id=') || text.includes('Sandbox is up:')) {
          return url;
        }
      }
      return null;
    }

    function updateCoderSandboxButton(fullUrl = null) {
      const linkEl = document.getElementById('coder-sandbox-link');
      const dotEl = document.getElementById('sandbox-status-dot');
      if (!linkEl) return;

      if (fullUrl) {
        activeSandboxWatchUrl = fullUrl;
        linkEl.href = fullUrl;
        linkEl.title = 'Open Active CoderSandbox: ' + fullUrl;
        if (dotEl) {
          dotEl.className = 'w-2 h-2 rounded-full bg-secondary animate-pulse';
          dotEl.title = 'Sandbox active: ' + fullUrl;
        }
      } else {
        activeSandboxWatchUrl = null;
        const baseUrl = getBaseSandboxUrl();
        linkEl.href = baseUrl;
        linkEl.title = 'Open CoderSandbox: ' + baseUrl;
        if (dotEl) {
          dotEl.className = 'w-2 h-2 rounded-full bg-outline-variant/40';
          dotEl.title = 'Sandbox standby: ' + baseUrl;
        }
      }
    }

    function checkAndDisplaySandboxLinks(author, tr, timestamp = null) {
      if (!tr || !tr.response || typeof tr.response !== 'object') return;
      const watchUrl = tr.response.watch_url || tr.response.watchUrl;
      const vscodeUrl = tr.response.vscode_url || tr.response.vscodeUrl;
      if (watchUrl) {
        updateCoderSandboxButton(watchUrl);
      }
      if (watchUrl || vscodeUrl) {
        const lines = [];
        if (watchUrl) lines.push(watchUrl);
        if (vscodeUrl) lines.push(vscodeUrl);
        const text = lines.join('\n');
        const feed = document.getElementById('chat-feed');
        if (feed && ((watchUrl && feed.innerText.includes(watchUrl)) || (vscodeUrl && feed.innerText.includes(vscodeUrl)))) {
          return;
        }
        addAgentMsg(author || 'system', text, timestamp);
      }
    }

    function hasText(value) {
      return typeof value === 'string' && stripThinking(value).length > 0;
    }

    function addAgentMsg(author, text, timestamp = null) {
      // Blank-but-present text produces an empty bubble; such events carry a
      // function call, not an utterance. (Also guards history recorded before
      // the backend started filtering them.)
      const cleanText = stripThinking(text);
      if (!hasText(cleanText)) return;
      appendMsgToFeed(`
    <div class="flex items-start gap-4 max-w-3xl msg-enter">
      <div class="w-8 h-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
        <span class="material-symbols-outlined text-sm text-primary">smart_toy</span>
      </div>
      <div class="flex flex-col gap-1">
        <div class="flex items-center gap-2">
          <span class="text-xs font-bold text-on-surface font-headline uppercase tracking-tight">${escHtml(author)}</span>
          <span class="text-[10px] text-outline-variant">${ts(timestamp)}</span>
        </div>
        <div class="bg-surface-container-high p-4 rounded-xl rounded-tl-none border border-outline-variant/5">
          ${thinkingBlock(text)}
          <div class="text-sm text-on-surface leading-relaxed md-body">${renderMarkdown(cleanText)}</div>
        </div>
      </div>
    </div>`);
    }

    // Above this many characters a final answer is folded, so one long report
    // (a full hypothesis set) cannot bury the rest of the conversation.
    const AGENT_OUTPUT_FOLD = 900;

    // A key agent's final answer (`agent_output`): the deliverable itself —
    // the hypotheses, the research summary, the execution report. It reaches
    // its caller as an AgentTool result and is never spoken in the top-level
    // stream, so it is rendered here as that agent's own, distinct message.
    function addAgentOutputMsg(agent, text, timestamp = null, caller = null) {
      const cleanText = stripThinking(text);
      if (!hasText(cleanText)) return;
      const body = cleanText;
      const folded = body.length > AGENT_OUTPUT_FOLD;
      const toggle = folded
        ? `<button onclick="toggleAgentOutput(this)"
             class="self-start text-[10px] font-mono uppercase tracking-widest text-primary/80 hover:text-primary transition-colors">
             Show full output</button>`
        : '';
      appendMsgToFeed(`
    <div class="flex items-start gap-4 max-w-3xl msg-enter">
      <div class="w-8 h-8 rounded-lg bg-secondary/10 border border-secondary/30 flex items-center justify-center shrink-0">
        <span class="material-symbols-outlined text-sm text-secondary">lightbulb</span>
      </div>
      <div class="flex flex-col gap-1 min-w-0">
        <div class="flex items-center gap-2 flex-wrap">
          <span class="text-xs font-bold text-on-surface font-headline uppercase tracking-tight">${escHtml(agent || 'agent')}</span>
          <span class="text-[9px] font-mono uppercase tracking-widest text-secondary bg-secondary/10 border border-secondary/20 px-1.5 py-0.5 rounded">Result</span>
          ${caller ? `<span class="text-[9px] font-mono text-outline-variant">→ ${escHtml(caller)}</span>` : ''}
          <span class="text-[10px] text-outline-variant">${ts(timestamp)}</span>
        </div>
        <div class="flex flex-col gap-2 bg-surface-container-high p-4 rounded-xl rounded-tl-none border border-secondary/20">
          ${thinkingBlock(text)}
          <div class="${folded ? 'max-h-64 overflow-hidden' : ''}">
            <div class="text-sm text-on-surface leading-relaxed break-words md-body">${renderMarkdown(body)}</div>
          </div>
          ${toggle}
        </div>
      </div>
    </div>`);
    }

    function toggleAgentOutput(button) {
      const box = button.previousElementSibling;
      if (!box) return;
      // Reasoning folds shorter than a deliverable, so the height class differs
      // per block; remember the button's own label for the same reason.
      const height = [...box.classList].find(c => c.startsWith('max-h-')) || button.dataset.fold;
      if (height) button.dataset.fold = height;
      if (!button.dataset.label) button.dataset.label = button.textContent.trim();
      const collapsed = box.classList.toggle(button.dataset.fold);
      box.classList.toggle('overflow-hidden', collapsed);
      button.textContent = collapsed ? button.dataset.label : 'Collapse';
      if (collapsed) box.scrollIntoView({ block: 'nearest' }); else scrollChat();
    }

    function addUserMsg(text, timestamp = null) {
      appendMsgToFeed(`
    <div class="flex items-start gap-4 max-w-2xl ml-auto flex-row-reverse msg-enter">
      <div class="w-8 h-8 rounded-lg bg-surface-container-highest flex items-center justify-center shrink-0">
        <span class="material-symbols-outlined text-sm text-on-surface">person</span>
      </div>
      <div class="flex flex-col gap-1 items-end">
        <div class="flex items-center gap-2">
          <span class="text-[10px] text-outline-variant">${ts(timestamp)}</span>
          <span class="text-xs font-bold text-on-surface font-headline uppercase tracking-tight">You</span>
        </div>
        <div class="bg-primary/5 p-4 rounded-xl rounded-tr-none border border-primary/20">
          <p class="text-sm text-on-surface leading-relaxed whitespace-pre-wrap break-words">${escHtml(text)}</p>
        </div>
      </div>
    </div>`);
    }

    // The three bouncing dots that used to live in the feed are gone: the
    // status indicator above the composer says the same thing and much more,
    // without pushing the conversation around. Both functions stay as shims so
    // every existing call site keeps working — the indicator derives its state
    // from the event stream itself, so only "a run is live" is worth relaying.
    function showTyping() {
      StatusIndicator.feed({ type: '__typing__' });
    }

    function hideTyping() {
      // No-op by design: `final_response` / `error` / `status` already close
      // the run in the indicator, and a bare hideTyping() must not erase a
      // state (e.g. "waiting for your answer") that is still true.
    }


    // =========================================================================
    // Form
    // =========================================================================
    // The composer is a textarea: it wraps long queries and grows with them,
    // so Enter has to stay "send" while Shift+Enter inserts a newline.
    function autoGrowChatInput() {
      const input = document.getElementById('chat-input');
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 240) + 'px';
    }

    function resetChatInput() {
      const input = document.getElementById('chat-input');
      input.value = '';
      input.style.height = '';
    }

    document.getElementById('chat-input').addEventListener('input', autoGrowChatInput);

    document.getElementById('chat-input').addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' || e.shiftKey || e.ctrlKey || e.altKey || e.metaKey || e.isComposing) return;
      e.preventDefault();
      document.getElementById('chat-form').requestSubmit();
    });

    document.getElementById('chat-form').addEventListener('submit', (e) => {
      e.preventDefault();
      const input = document.getElementById('chat-input');
      const msg = input.value.trim();
      if (!msg || !ws || ws.readyState !== 1) return;
      maybeAutoNameSession(msg);
      ws.send(JSON.stringify({ type: 'chat_message', message: msg }));
      addTelemetry('SEND :: user query');
    });

    function stopChat() {
      if (!ws || ws.readyState !== 1) return;
      ws.send(JSON.stringify({ type: 'stop_chat' }));
      addTelemetry('STOP :: user requested stop');
      StatusIndicator.markStopped();
    }


    function applyReportLanguage(lang) {
      reportLanguage = String(lang || '');
      const select = document.getElementById('report-lang-select');
      // With no server-side choice, show the interface language as the default.
      if (select) select.value = reportLanguage || currentLang;
    }

    function sendReportLanguage(lang) {
      if (!ws || ws.readyState !== 1) return false;
      ws.send(JSON.stringify({ type: 'set_report_language', report_language: lang }));
      return true;
    }

    function onReportLanguageChange() {
      // Send only; the mirror is set when the server echoes, as with the dataset.
      const select = document.getElementById('report-lang-select');
      if (sendReportLanguage(select.value)) return;
      // The socket is down, so the server never heard the pick. Put the select
      // back to the language that is still in effect, or the report comes out
      // in a language the UI no longer shows.
      applyReportLanguage(reportLanguage);
      addSystemMsg('Not connected — reconnect and try again.');
      addTelemetry('REPORT LANG :: not sent, socket is down');
    }

    function toggleAttachMenu(show) {
      const menu = document.getElementById('attach-menu');
      const open = show === undefined ? menu.classList.contains('hidden') : show;
      menu.classList.toggle('hidden', !open);
    }

    document.addEventListener('click', (event) => {
      // The button's own handler runs first and this must not undo it.
      if (event.target.closest('#attach-btn') || event.target.closest('#attach-menu')) return;
      toggleAttachMenu(false);
    });

    function applyDatasetUrl(url) {
      datasetUrl = String(url || '');
      renderAttachments();
    }

    function renderAttachments() {
      const row = document.getElementById('attachment-chips');
      if (!row) return;
      if (!datasetUrl) {
        row.innerHTML = '';
        row.classList.add('hidden');
        return;
      }
      row.innerHTML = `
    <span class="flex items-center gap-1.5 max-w-full bg-surface-container-high border border-primary/20 rounded-md pl-2 pr-1 py-1">
      <span class="material-symbols-outlined text-primary text-sm">folder_zip</span>
      <a href="${escHtml(datasetUrl)}" target="_blank" title="${escHtml(datasetUrl)}"
        class="font-mono text-[10px] text-on-surface-variant truncate max-w-[24rem] hover:text-primary">${escHtml(datasetUrl)}</a>
      <button type="button" onclick="clearDatasetLink()" title="Detach dataset"
        class="p-0.5 text-outline-variant hover:text-error transition-colors flex items-center">
        <span class="material-symbols-outlined text-sm">close</span>
      </button>
    </span>`;
      row.classList.remove('hidden');
    }

    function clearChat() {
      activityReset();
      StatusIndicator.reset();
      const feed = document.getElementById('chat-feed');
      if (feed) {
        feed.innerHTML = `
    <div class="flex flex-col items-center justify-center h-full opacity-50">
      <span class="material-symbols-outlined text-4xl text-primary/30 mb-3">hub</span>
      <p data-i18n="chat.sendQuery" class="text-sm text-outline-variant font-medium">Send a query to begin orchestration</p>
    </div>`;
      }
    }

