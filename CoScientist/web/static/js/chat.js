// =========================================================================
// Chat rendering & Input handling
// =========================================================================
    function scrollChat() {
      const feed = document.getElementById('chat-feed');
      feed.scrollTop = feed.scrollHeight;
    }


    function appendMsgToFeed(html) {
      const feed = document.getElementById('chat-feed');
      // The empty-feed hint is marked by an attribute, not found by its
      // classes: any card with the same utility classes used to match it and
      // wipe the feed on the next message.
      const placeholder = feed.querySelector('[data-feed-empty]');
      if (placeholder) {
        feed.innerHTML = '';
      }
      feed.insertAdjacentHTML('beforeend', html);
      scrollChat();
    }

    // A system notice is an event in the run, not a message to read: one
    // line with its time, in the muted colour, folded when it runs long.
    function addSystemMsg(text, timestamp = null) {
      appendMsgToFeed(`
    <div class="flex items-baseline gap-3 msg-enter">
      <span class="sr-only">${escHtml(t('chat.system'))}</span>
      <span class="text-[10px] text-outline-variant tabular-nums shrink-0 w-16">${ts(timestamp)}</span>
      <div class="min-w-0 flex-1">
        ${foldable(`<p class="text-[12px] leading-relaxed text-on-surface-variant whitespace-pre-wrap break-words">${escHtml(text)}</p>`, text, { bg: 'rgb(var(--c-background))' })}
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

    // Everything long in the feed folds the same way, and everything foldable
    // arrives folded. One agent report, one approved research frame or one
    // tool dump otherwise buries the whole conversation; the reader opens what
    // they want to read. 280 characters is about four lines at the body size —
    // below that there is nothing worth hiding.
    const FOLD_CHARS = 280;

    // `text` is measured rather than `bodyHtml`, so markup never counts toward
    // the threshold: a short sentence full of links is not a long message.
    // `open` is for the one case a fold must start expanded: a form the reader
    // has to fill in right now. It still gets the wrapper, so answering it can
    // fold it away — see collapseFold, called from disableHitlControls.
    function foldable(bodyHtml, text, { chars = FOLD_CHARS, bg = '', open = false } = {}) {
      if (String(text == null ? '' : text).length <= chars) return bodyHtml;
      const style = bg ? ` style="--fold-bg:${bg}"` : '';
      return `<div class="fold${open ? ' fold-open' : ''}"${style}>
          <div class="fold-body">${bodyHtml}</div>
          <button type="button" onclick="toggleFold(this)" aria-expanded="${open}"
            class="fold-btn">${escHtml(open ? t('chat.collapse') : t('chat.showFull'))}</button>
        </div>`;
    }

    // The character count above only guesses at height; what is actually cut
    // off depends on the width, the markup and the font. A fold that would
    // hide a line or two costs the reader a click to save them nothing, so
    // once a fold is on the page its real height is measured, and a block
    // that overshoots the clip by less than half of it is simply shown whole.
    // Re-measured on resize, since a narrower feed can push it back over.
    const FOLD_SLACK = 1.5;

    function settleFold(wrap) {
      const body = wrap.querySelector(':scope > .fold-body');
      if (!body) return;
      wrap.classList.remove('fold-fits');
      const clip = parseFloat(getComputedStyle(body).maxHeight);
      if (!clip) return; // open, so max-height is `none` — measure is moot
      wrap.classList.toggle('fold-fits', body.scrollHeight <= clip * FOLD_SLACK);
    }

    const foldResize = new ResizeObserver(entries => {
      for (const entry of entries) settleFold(entry.target.parentElement);
    });

    function watchFolds(root) {
      const folds = root.matches && root.matches('.fold') ? [root] : [];
      if (root.querySelectorAll) folds.push(...root.querySelectorAll('.fold'));
      for (const wrap of folds) {
        const body = wrap.querySelector(':scope > .fold-body');
        if (body) foldResize.observe(body);
        settleFold(wrap);
      }
    }

    new MutationObserver(records => {
      for (const record of records) record.addedNodes.forEach(watchFolds);
    }).observe(document.documentElement, { childList: true, subtree: true });
    if (document.fonts) document.fonts.ready.then(() => watchFolds(document));

    // A card that has been answered has nothing left to act on, so it tidies
    // itself away instead of staying open for the rest of the conversation.
    function collapseFold(wrap) {
      if (!wrap || !wrap.classList.contains('fold-open')) return;
      wrap.classList.remove('fold-open');
      const button = wrap.querySelector('.fold-btn');
      if (button) {
        button.setAttribute('aria-expanded', 'false');
        button.textContent = t('chat.showFull');
      }
    }

    function toggleFold(button) {
      const wrap = button.closest('.fold');
      if (!wrap) return;
      const open = wrap.classList.toggle('fold-open');
      button.setAttribute('aria-expanded', String(open));
      button.textContent = open ? t('chat.collapse') : t('chat.showFull');
      // Closing a block that ran past the viewport would otherwise leave the
      // reader somewhere below where the message now ends.
      if (!open) wrap.scrollIntoView({ block: 'nearest' });
    }

    // A message whose body was written to a document shows the summary and a
    // way in, not the body. `foldable` stays for everything with no document —
    // a short answer is still just an answer.
    function documentBlock(event) {
      const doc = (event && event.document) || null;
      if (!doc || !doc.artifact_id) return '';
      const summary = String((event && event.summary) || doc.title || '').trim();
      const title = doc.title || '';
      return `
        <div class="flex flex-col gap-2.5">
          ${summary ? `<div class="doc-summary break-words">${mdBlock(summary)}</div>` : ''}
          <button type="button" class="doc-btn self-start" aria-pressed="false"
            data-doc-id="${escHtml(doc.artifact_id)}"
            onclick="openDocument('${escJs(doc.artifact_id)}', '${escJs(title)}')">
            <span class="material-symbols-outlined text-base">description</span>${escHtml(t('doc.open'))}
          </button>
        </div>`;
    }

    function thinkingBlock(text) {
      const thinking = takeThinking(text);
      if (!thinking) return '';
      return `
        <div class="flex flex-col gap-1.5 mb-3 pb-3 border-b border-outline-variant/20">
          <span class="self-start text-[10px] font-mono uppercase tracking-widest text-outline-variant bg-outline-variant/10 border border-outline-variant/20 px-1.5 py-0.5 rounded">Thinking</span>
          ${foldable(`<div class="text-on-surface-variant md-body md-body-quiet">${renderMarkdown(thinking)}</div>`,
            thinking, { bg: 'rgb(var(--c-surface-container))' })}
        </div>`;
    }

    // Both escapers exist because `renderMarkdown` edits the HTML string
    // AFTER DOMPurify has run: nothing it adds there is sanitized again.
    // Text content needs the three markup characters escaped.
    function asText(value) {
      return String(value == null ? '' : value)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // An attribute value taken from already-escaped HTML needs only its
    // delimiter: escaping & again would print `&amp;amp;` in the alt text.
    function asAttr(value) {
      return String(value == null ? '' : value).replace(/"/g, '&quot;');
    }

    function renderMarkdown(text) {
      // Agents write their messages in markdown; render it to sanitized HTML
      // instead of showing the raw '**'/'`'/'#' syntax as plain text.
      const raw = typeof text === 'string' ? stripThinking(text) : String(text == null ? '' : text);
      const withLocalLinks = raw
        .replace(/(^|\s)(\/api\/tz-document[^\s)]*)/g, '$1[$2]($2)')
        // Reminted artifact paths: GFM only auto-links full URLs, so a bare
        // root-relative path would render as unclickable text.
        .replace(/(^|\s)(\/api\/artifact\/[^\s)]*)/g, '$1[$2]($2)')
        // The session's own copy of a file, resolved server-side from a
        // cos-artifact: reference. Same reason as the line above.
        .replace(/(^|\s)(\/api\/users\/[^\s)]*\/artifacts\/[^\s)]*)/g, '$1[$2]($2)')
        // Live MCP build page: /alembic/builds/<job_id> (agent surfaces it as progress_page).
        .replace(/(^|\s)(\/alembic\/builds\/[A-Za-z0-9._-]+)/g, '$1[$2]($2)');
      const html = marked.parse(withLocalLinks);
      const clean = DOMPurify.sanitize(html, { ADD_ATTR: ['target'] });
      // A bare URL becomes a link whose text is the URL itself. A presigned
      // URL runs to hundreds of characters, so show the file name instead.
      const withShortText = clean.replace(
        /<a href="([^"]+)">([^<]*)<\/a>/g,
        (m, url, text) => {
          const rawUrl = url.replace(/&amp;/g, '&');
          if (text.replace(/&amp;/g, '&') !== rawUrl) return m;
          const segment = rawUrl.split(/[?#]/)[0].split('/').filter(Boolean).pop() || '';
          let name = segment;
          try {
            // A URL is not required to be valid percent-encoding: one stray
            // '%' in a bare link threw URIError out of the whole render.
            name = decodeURIComponent(segment);
          } catch (err) {
            name = segment;
          }
          const short = name || rawUrl.split('/')[2] || 'link';
          return '<a href="' + url + '">' + asText(short) + '</a>';
        });
      // A link whose URL path (before any ?query) ends in an image extension
      // gets an inline <img> preview under the link — markdown images render
      // on their own, but agents also emit image URLs as plain links. The & in
      // presigned S3 query strings shows up here as &amp; — the pattern allows
      // it.
      const withPreviews = withShortText.replace(
        /<a href="(https?:\/\/[^"]+|\/(?!\/)[^"]*)">([^<]*)<\/a>/g,
        (m, url, label) => {
          if (!/\.(png|jpe?g|gif|webp|svg|bmp|avif)$/i.test(url.split(/[?#]/)[0])) return m;
          return m + '<br><a href="' + url + '"><img src="' + url + '" alt="' + asAttr(label) + '"'
            + ' class="mt-2 max-w-full rounded-lg border border-outline-variant/20 cursor-zoom-in" loading="lazy"></a>';
        });
      return withPreviews.replace(/<a /g, '<a target="_blank" rel="noopener noreferrer" class="text-primary underline" ');
    }

    // Agent-written fields inside cards (a goal, a step, a finding) are
    // markdown too, but they sit in the card's own type size: `md-fit` drops
    // the message-body size `.md-body` sets. A block field may hold lists
    // and paragraphs; an inline one sits in a <span>/<p>/<h3> and must not
    // open block elements there.
    function mdBlock(text) {
      if (text == null || text === '') return '';
      return `<div class="md-body md-fit">${renderMarkdown(String(text))}</div>`;
    }

    function mdInline(text) {
      if (text == null || text === '') return '';
      const html = DOMPurify.sanitize(marked.parseInline(String(text)), { ADD_ATTR: ['target'] })
        .replace(/<a /g, '<a target="_blank" rel="noopener noreferrer" class="text-primary underline" ');
      return `<span class="md-body md-fit">${html}</span>`;
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
        linkEl.title = t('chat.sandboxOpenActive', { url: fullUrl });
        if (dotEl) {
          dotEl.className = 'w-2 h-2 rounded-full bg-secondary ml-auto shrink-0';
          dotEl.title = t('chat.sandboxActive', { url: fullUrl });
        }
      } else {
        activeSandboxWatchUrl = null;
        const baseUrl = getBaseSandboxUrl();
        linkEl.href = baseUrl;
        linkEl.title = t('chat.sandboxOpen', { url: baseUrl });
        if (dotEl) {
          dotEl.className = 'w-2 h-2 rounded-full bg-outline-variant/40 ml-auto shrink-0';
          dotEl.title = t('chat.sandboxStandby', { url: baseUrl });
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

    // Tools that put a question to the operator: the text written next to such
    // a call is the context of the decision, so it stays in the chat.
    const OPERATOR_QUESTION_TOOLS = ['request_approval', 'request_selection', 'request_input', 'adk_request_input'];

    // Text that belongs outside the chat, in the telemetry panel:
    //  * a non-final `agent_event` written before the model's own tool call
    //    («Проверю граф…», «Запускаю ResearchAgent…») — an aside to the model,
    //    not a message to the reader. Server notices (the sandbox links) are
    //    non-final too but carry no call, so they stay;
    //  * anything the planner and its critic say: the plan has its own button
    //    and view, and the planner's approval card still reaches the chat as a
    //    `hitl_request`.
    const PLAN_AGENTS = ['PlannerAgent', 'PlanCriticAgent'];

    // The framing agent used to speak its whole frame summary into the chat
    // immediately after the review card had shown the reader the same text.
    // The backend stopped emitting it (e33f523 turned that event's content
    // into a log line), but reopening an older session replays the old
    // transcript — so the echo is dropped on the way in rather than printed
    // under the card that already carries it.
    function isFrameEcho(event) {
      return String(event.author || '').startsWith('ContextInit')
        && /(^|\n)##\s*Рамка исследования/.test(String(event.content || ''));
    }

    function isChatNoise(event) {
      if (!event) return false;
      if (PLAN_AGENTS.includes(event.author)) return true;
      if (isFrameEcho(event)) return true;
      if (event.is_final) return false;
      const calls = event.tool_calls || [];
      return calls.length > 0
        && !calls.some(call => OPERATOR_QUESTION_TOOLS.includes(call.name));
    }

    function addAgentMsg(author, text, timestamp = null, event = null) {
      // Blank-but-present text produces an empty bubble; such events carry a
      // function call, not an utterance. (Also guards history recorded before
      // the backend started filtering them.)
      const cleanText = stripThinking(text);
      if (!hasText(cleanText)) return;
      appendMsgToFeed(`
    <div class="flex items-start gap-4 max-w-4xl msg-enter">
      <div class="w-8 h-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
        <span class="material-symbols-outlined text-sm text-primary">smart_toy</span>
      </div>
      <div class="flex flex-col gap-1 flex-1 min-w-0">
        <div class="flex items-center gap-2">
          <span class="text-xs font-bold text-on-surface font-headline uppercase tracking-tight">${escHtml((window.StatusIndicator && StatusIndicator.agentName) ? StatusIndicator.agentName(author) : author)}</span>
          <span class="text-[10px] text-outline-variant">${ts(timestamp)}</span>
        </div>
        <div class="bg-surface-container p-4 rounded-xl rounded-tl-none border border-outline-variant/15">
          ${thinkingBlock(text)}
          ${documentBlock(event)
        || foldable(`<div class="text-on-surface md-body">${renderMarkdown(cleanText)}</div>`,
          cleanText, { bg: 'rgb(var(--c-surface-container))' })}
        </div>
      </div>
    </div>`);
    }

    // A key agent's final answer (`agent_output`): the deliverable itself —
    // the hypotheses, the research summary, the execution report. It reaches
    // its caller as an AgentTool result and is never spoken in the top-level
    // stream, so it is rendered here as that agent's own, distinct message.
    function addAgentOutputMsg(agent, text, timestamp = null, caller = null, event = null) {
      const cleanText = stripThinking(text);
      if (!hasText(cleanText)) return;
      const body = cleanText;
      appendMsgToFeed(`
    <div class="flex items-start gap-4 max-w-4xl msg-enter">
      <div class="w-8 h-8 rounded-lg bg-secondary/10 border border-secondary/30 flex items-center justify-center shrink-0">
        <span class="material-symbols-outlined text-sm text-secondary">lightbulb</span>
      </div>
      <div class="flex flex-col gap-1 flex-1 min-w-0">
        <div class="flex items-center gap-2 flex-wrap">
          <span class="text-xs font-bold text-on-surface font-headline uppercase tracking-tight">${escHtml((window.StatusIndicator && StatusIndicator.agentName) ? StatusIndicator.agentName(agent) : (agent || 'agent'))}</span>
          <span class="text-[9px] font-mono uppercase tracking-widest text-secondary bg-secondary/10 border border-secondary/20 px-1.5 py-0.5 rounded">${t('chat.result')}</span>
          ${caller ? `<span class="text-[9px] font-mono text-outline-variant">→ ${escHtml((window.StatusIndicator && StatusIndicator.agentName) ? StatusIndicator.agentName(caller) : caller)}</span>` : ''}
          <span class="text-[10px] text-outline-variant">${ts(timestamp)}</span>
        </div>
        <div class="flex flex-col gap-2 bg-surface-container p-4 rounded-xl rounded-tl-none border border-secondary/25">
          ${thinkingBlock(text)}
          ${documentBlock(event)
        || foldable(`<div class="text-on-surface break-words md-body">${renderMarkdown(body)}</div>`,
          body, { bg: 'rgb(var(--c-surface-container))' })}
        </div>
      </div>
    </div>`);
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
          <span class="text-xs font-bold text-on-surface font-headline uppercase tracking-tight">${t('chat.you')}</span>
        </div>
        <div class="bg-primary/5 p-4 rounded-xl rounded-tr-none border border-primary/20">
          <div class="text-sm text-on-surface leading-relaxed break-words">${mdBlock(text)}</div>
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
      if (typeof RunTimer !== 'undefined') RunTimer.start();
    });

    function stopChat() {
      if (!ws || ws.readyState !== 1) return;
      ws.send(JSON.stringify({ type: 'stop_chat' }));
      addTelemetry('STOP :: user requested stop');
      StatusIndicator.markStopped();
      if (typeof RunTimer !== 'undefined') RunTimer.finish();
    }


    function applyReportLanguage(lang) {
      // Records the server-known value. The settings toggle is the visible
      // control; the rejection handler reads this to put the UI back in sync.
      reportLanguage = String(lang || '');
    }

    function sendReportLanguage(lang) {
      if (!ws || ws.readyState !== 1) return false;
      ws.send(JSON.stringify({ type: 'set_report_language', report_language: lang }));
      return true;
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
        class="font-mono text-[11px] text-on-surface-variant truncate max-w-[28rem] hover:text-primary">${escHtml(datasetUrl)}</a>
      <button type="button" onclick="clearDatasetLink()" title="${t('chat.detachDataset')}"
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
    <div data-feed-empty class="flex flex-col items-center justify-center h-full">
      <span class="material-symbols-outlined text-4xl text-outline-variant/50 mb-3" aria-hidden="true">forum</span>
      <p data-i18n="chat.sendQuery" class="text-sm text-on-surface-variant">${escHtml(t('chat.sendQuery'))}</p>
    </div>`;
      }
    }
