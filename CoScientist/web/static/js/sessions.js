// =========================================================================
// Local users and sessions (Management & Bundles Export/Import)
// =========================================================================
    function populateUserSelectors() {
      const options = knownUsers.map(user =>
        `<option value="${escHtml(user.id)}" ${activeUser && activeUser.id === user.id ? 'selected' : ''}>${escHtml(user.nickname)}</option>`
      ).join('');
      document.getElementById('identity-user-select').innerHTML = options;
      document.getElementById('existing-user-block').classList.toggle('hidden', knownUsers.length === 0);
    }

    function showHiddenSessions() {
      try { return localStorage.getItem(SHOW_HIDDEN_SESSIONS_KEY) === '1'; } catch (_) { return false; }
    }

    // Sessions the picker offers: hidden ones only when asked for, and the
    // active one always, so the picker never lies about what is on screen.
    function visibleSessions(sessions = knownSessions) {
      if (showHiddenSessions()) return sessions;
      return sessions.filter(item => !item.hidden || (activeSession && activeSession.id === item.id));
    }

    function populateSessionSelector() {
      renderSessionTitle();
      document.getElementById('session-select').innerHTML = visibleSessions().map(session => {
        const label = session.hidden ? `${session.title} ${t('sessions.hiddenMark')}` : session.title;
        return `<option value="${escHtml(session.id)}" ${activeSession && activeSession.id === session.id ? 'selected' : ''}>${escHtml(label)}</option>`;
      }).join('') || `<option value="">${t('identity.noSessions')}</option>`;
      renderHiddenSessionControls();
    }

    function renderHiddenSessionControls() {
      const count = knownSessions.filter(item => item.hidden).length;
      const showBtn = document.getElementById('session-show-hidden-btn');
      if (showBtn) {
        const on = showHiddenSessions();
        showBtn.setAttribute('aria-checked', String(on));
        showBtn.querySelector('.material-symbols-outlined').textContent = on ? 'check_box' : 'check_box_outline_blank';
        document.getElementById('session-hidden-count').textContent = count ? String(count) : '';
      }
      const unhideBtn = document.getElementById('session-unhide-all-btn');
      if (unhideBtn) unhideBtn.classList.toggle('hidden', count === 0);
    }

    function toggleShowHiddenSessions() {
      try { localStorage.setItem(SHOW_HIDDEN_SESSIONS_KEY, showHiddenSessions() ? '0' : '1'); } catch (_) { }
      populateSessionSelector();
    }

    // Hides every session in one go so only sessions started from now on
    // are listed. Nothing is deleted. The open session stays in the picker
    // while it is open (visibleSessions), and a still-empty one is kept
    // visible, since it is the "new" one. The server never hides a running
    // session.
    async function hideOldSessions() {
      if (!activeUser) return openIdentityModal();
      try {
        // A fresh list: `empty` on the cached copy goes stale after the first message.
        const fresh = activeSession ? (await loadSessions(activeUser)).find(item => item.id === activeSession.id) : null;
        const keep = fresh && fresh.empty ? [fresh.id] : [];
        const data = await apiJson(`/api/users/${encodeURIComponent(activeUser.id)}/sessions/hide-old`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ keep })
        });
        try { localStorage.setItem(SHOW_HIDDEN_SESSIONS_KEY, '0'); } catch (_) { }
        await loadSessions(activeUser);
        if (activeSession) activeSession = knownSessions.find(item => item.id === activeSession.id) || activeSession;
        populateSessionSelector();
        addSystemMsg(data.hidden
          ? t('sessions.hiddenOld', { count: data.hidden })
          : t('sessions.nothingToHide'));
      } catch (error) { addSystemMsg(t('sessions.hideError', { error: error.message })); }
    }

    async function unhideAllSessions() {
      if (!activeUser) return;
      try {
        await apiJson(`/api/users/${encodeURIComponent(activeUser.id)}/sessions/unhide-all`, { method: 'POST' });
        await loadSessions(activeUser);
        if (activeSession) activeSession = knownSessions.find(item => item.id === activeSession.id) || activeSession;
        populateSessionSelector();
      } catch (error) { addSystemMsg(t('sessions.hideError', { error: error.message })); }
    }

    // The session's name in the top bar, after "Orchestrator /": which run
    // this page is showing, readable without opening the picker.
    function renderSessionTitle() {
      const el = document.getElementById('session-title');
      if (!el) return;
      const title = activeSession && activeSession.title ? activeSession.title : '';
      el.textContent = title;
      el.title = title;
      el.parentElement.classList.toggle('hidden', !title);
    }

    // The session menu beside the picker: new, rename, save, restore,
    // export, import. A menu, not six unlabelled icons in a row.
    function setSessionMenuOpen(open) {
      const menu = document.getElementById('session-menu');
      const button = document.getElementById('session-menu-btn');
      if (!menu || !button) return;
      menu.classList.toggle('hidden', !open);
      button.setAttribute('aria-expanded', String(open));
      if (open) {
        const first = menu.querySelector('[role^="menuitem"]');
        if (first) first.focus();
      }
    }

    function toggleSessionMenu() {
      const menu = document.getElementById('session-menu');
      setSessionMenuOpen(!!menu && menu.classList.contains('hidden'));
    }

    function runSessionMenu(action) {
      setSessionMenuOpen(false);
      action();
    }

    document.addEventListener('click', event => {
      if (!event.target.closest('#session-menu, #session-menu-btn')) setSessionMenuOpen(false);
    });

    document.addEventListener('keydown', event => {
      const menu = document.getElementById('session-menu');
      if (!menu || menu.classList.contains('hidden')) return;
      const items = [...menu.querySelectorAll('[role^="menuitem"]')];
      const index = items.indexOf(document.activeElement);
      if (event.key === 'Escape') {
        setSessionMenuOpen(false);
        document.getElementById('session-menu-btn').focus();
      } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        const step = event.key === 'ArrowDown' ? 1 : -1;
        items[(index + step + items.length) % items.length].focus();
      }
    });

    // The avatar is the nickname's first letter, kept in step with the name
    // wherever the name is written (sign-in, switch, language change).
    (function watchNickname() {
      const name = document.getElementById('active-nickname');
      const avatar = document.getElementById('active-avatar');
      if (!name || !avatar) return;
      const paint = () => {
        const known = typeof activeUser !== 'undefined' && activeUser && activeUser.nickname;
        avatar.textContent = known ? activeUser.nickname.trim().charAt(0).toUpperCase() || '?' : '?';
      };
      new MutationObserver(paint).observe(name, { childList: true, characterData: true, subtree: true });
      paint();
    })();

    function openIdentityModal() {
      populateUserSelectors();
      document.getElementById('identity-error').classList.add('hidden');
      document.getElementById('identity-close-btn').classList.toggle('hidden', !activeUser);
      document.getElementById('identity-modal').classList.remove('hidden');
      setTimeout(() => document.getElementById('nickname-input').focus(), 0);
    }

    function closeIdentityModal() {
      if (activeUser) document.getElementById('identity-modal').classList.add('hidden');
    }

    function showIdentityError(error) {
      const element = document.getElementById('identity-error');
      element.textContent = error.message || String(error);
      element.classList.remove('hidden');
    }

    async function loadSessions(user) {
      const data = await apiJson(`/api/users/${encodeURIComponent(user.id)}/sessions`);
      knownSessions = data.sessions || [];
      populateSessionSelector();
      return knownSessions;
    }

    async function createBlankSession(user) {
      const created = await apiJson(`/api/users/${encodeURIComponent(user.id)}/sessions`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New session' })
      });
      knownSessions.unshift(created.session);
      return created.session;
    }

    // startFresh creates a new session (or reuses the newest one if it has no messages yet).
    async function ensureUserSession(user, preferredSessionId = null, { startFresh = false } = {}) {
      activeUser = user;
      const sessions = await loadSessions(user);
      let selected = preferredSessionId ? sessions.find(item => item.id === preferredSessionId) : null;
      // Fallback picks come from what the picker shows, not from hidden sessions.
      const shown = showHiddenSessions() ? sessions : sessions.filter(item => !item.hidden);
      if (!selected && startFresh) {
        selected = shown.find(item => item.status === 'processing')
          || (shown.length > 0 && shown[0].empty ? shown[0] : null)
          || await createBlankSession(user);
      }
      if (!selected) {
        selected = shown.find(item => item.id === user.last_session_id)
          || shown[0]
          || await createBlankSession(user);
      }
      await activateSession(user, selected);
      closeIdentityModal();
    }

    async function registerLocalUser() {
      const nicknameInput = document.getElementById('nickname-input');
      const nickname = nicknameInput.value.trim();
      if (!nickname) return showIdentityError(new Error(t('identity.enterNick')));
      try {
        const data = await apiJson('/api/users', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ nickname })
        });
        knownUsers.push(data.user);
        nicknameInput.value = '';
        populateUserSelectors();
        await ensureUserSession(data.user, null, { startFresh: true });
        closeIdentityModal();
      } catch (error) { showIdentityError(error); }
    }

    async function continueExistingUser() {
      const userId = document.getElementById('identity-user-select').value;
      const user = knownUsers.find(item => item.id === userId);
      if (!user) return;
      try {
        await ensureUserSession(user, null, { startFresh: true });
        closeIdentityModal();
      } catch (error) { showIdentityError(error); }
    }

    async function onUserSelected(userId) {
      const user = knownUsers.find(item => item.id === userId);
      if (user) await ensureUserSession(user, null, { startFresh: true });
    }

    async function onSessionSelected(sessionId) {
      const session = knownSessions.find(item => item.id === sessionId);
      if (session && activeUser) await activateSession(activeUser, session);
    }

    async function createNewSession() {
      if (!activeUser) return openIdentityModal();
      let title = 'New session';
      if (!appSettings.general.autoNamingEnabled) {
        const inputTitle = prompt(t('sessions.titlePrompt'), 'New session');
        if (inputTitle === null) return;
        title = inputTitle.trim() || 'New session';
      }
      try {
        const data = await apiJson(`/api/users/${encodeURIComponent(activeUser.id)}/sessions`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title })
        });
        knownSessions.unshift(data.session);
        await activateSession(activeUser, data.session);
      } catch (error) { addSystemMsg(t('sessions.createError', { error: error.message })); }
    }

    async function renameCurrentSession() {
      if (!activeUser || !activeSession) return;
      const title = prompt(t('sessions.renamePrompt'), activeSession.title);
      if (!title || title.trim() === activeSession.title) return;
      try {
        const data = await apiJson(sessionApi(), {
          method: 'PATCH', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title })
        });
        activeSession = data.session;
        knownSessions = knownSessions.map(item => item.id === activeSession.id ? activeSession : item);
        populateSessionSelector();
      } catch (error) { addSystemMsg(t('sessions.renameError', { error: error.message })); }
    }

    /**
     * Generate session title based on prompt length (fixed max length: 40 chars).
     * Easily replaceable by an agent/LLM generator in the future.
     */
    function generateSessionTitle(messageText, maxLength = 40) {
      if (!messageText) return 'New session';
      let clean = messageText.trim().replace(/[\r\n]+/g, ' ').replace(/\s+/g, ' ');
      if (clean.length > maxLength) {
        return clean.slice(0, maxLength).trim() + '...';
      }
      return clean || 'New session';
    }

    async function maybeAutoNameSession(promptText) {
      if (!appSettings.general.autoNamingEnabled || !activeSession || !activeUser) return;
      const title = activeSession.title ? activeSession.title.trim() : '';
      const isDefaultTitle = !title || title === 'New session' || title.startsWith('New session');
      if (!isDefaultTitle) return;

      const newTitle = generateSessionTitle(promptText, 40);
      if (newTitle && newTitle !== activeSession.title) {
        try {
          const data = await apiJson(sessionApi(), {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newTitle })
          });
          activeSession = data.session;
          knownSessions = knownSessions.map(item => item.id === activeSession.id ? activeSession : item);
          populateSessionSelector();
        } catch (err) {
          console.warn('Auto-rename session failed:', err);
        }
      }
    }

    function disconnectCurrentSocket() {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
      if (ws) {
        intentionalDisconnect = true;
        ws.onclose = null;
        ws.close();
        ws = null;
      }
    }

    async function activateSession(user, session) {
      const changingSession = !activeSession || activeSession.id !== session.id;
      disconnectCurrentSocket();
      activeUser = user;
      activeSession = session;
      if (changingSession) runStatusVersion = -1;
      activeUser.last_session_id = session.id;
      knownUsers = knownUsers.map(item => item.id === user.id ? activeUser : item);
      localStorage.setItem(USER_STORAGE_KEY, user.id);
      localStorage.setItem(SESSION_STORAGE_KEY, session.id);
      if (serverBootId) localStorage.setItem(BOOT_STORAGE_KEY, serverBootId);
      localStorage.setItem(NICK_STORAGE_KEY, user.nickname);
      document.getElementById('active-nickname').textContent = user.nickname;
      document.getElementById('graph-link').href =
        `/graph?user_id=${encodeURIComponent(user.id)}&session_id=${encodeURIComponent(session.id)}`;
      const agentTreeLink = document.getElementById('agent-tree-link');
      if (agentTreeLink) agentTreeLink.href =
        `/agent-tree?user_id=${encodeURIComponent(user.id)}&session_id=${encodeURIComponent(session.id)}`;
      populateUserSelectors();
      populateSessionSelector();
      clearChat();
      // Drop the previous session's attachment; the snapshot brings the new one.
      applyDatasetUrl('');
      applyReportLanguage('');
      if (typeof loadSettings === 'function') await loadSettings();
      connect();
    }

    async function bootstrap() {
      updateCoderSandboxButton(null);
      try {
        const data = await apiJson('/api/users');
        serverBootId = data.serverBootId || null;
        knownUsers = data.users || [];
        populateUserSelectors();
        const savedUserId = localStorage.getItem(USER_STORAGE_KEY);
        let savedUser = knownUsers.find(item => item.id === savedUserId);
        if (data.defaultUsername) {
          const envNick = data.defaultUsername.trim().toLowerCase();
          const envUser = knownUsers.find(item => item.nickname && item.nickname.trim().toLowerCase() === envNick);
          if (envUser) {
            savedUser = envUser;
          }
        }
        if (!savedUser) {
          activeUser = null;
          activeSession = null;
          knownSessions = [];
          localStorage.removeItem(USER_STORAGE_KEY);
          localStorage.removeItem(SESSION_STORAGE_KEY);
          const noUserEntry = i18n['nav.noUser'];
          document.getElementById('active-nickname').textContent = (noUserEntry && noUserEntry[currentLang]) || 'No user selected';
          populateSessionSelector();
          openIdentityModal();
          return;
        }
        const urlParams = new URLSearchParams(window.location.search);
        const urlSessionId = urlParams.get('session_id');

        const navEntry = (typeof performance !== 'undefined' && performance.getEntriesByType)
          ? performance.getEntriesByType('navigation')[0]
          : null;
        const isReload = navEntry
          ? navEntry.type === 'reload'
          : (typeof performance !== 'undefined' && performance.navigation && performance.navigation.type === 1);

        const preferredSessionId = urlSessionId || (isReload ? localStorage.getItem(SESSION_STORAGE_KEY) : null);
        const startFresh = !preferredSessionId;

        await ensureUserSession(
          savedUser,
          preferredSessionId,
          { startFresh },
        );
      } catch (error) {
        addSystemMsg(t('sessions.initError', { error: error.message }));
        openIdentityModal();
      }
    }

    // The pill in the top bar. The run flag says whether anything is running;
    // the status line's phase (body[data-run-phase], set by status_indicator)
    // says whether it is running or stopped on a question for the user, which
    // is the one state worth a colour of its own.
    const STATUS_BADGE_TONES = {
      run: 'bg-primary/10 text-primary',
      wait: 'bg-tertiary/10 text-tertiary',
      fail: 'bg-error/10 text-error',
      idle: 'bg-surface-container-high text-on-surface-variant',
    };

    function renderStatusBadge() {
      const el = document.getElementById('status-badge');
      if (!el) return;
      const phase = document.body.dataset.runPhase || 'idle';
      let tone = runActive ? 'run' : 'idle';
      let key = runActive ? 'topbar.processing' : 'topbar.idle';
      if (phase === 'waiting' || phase === 'waiting_frame') { tone = 'wait'; key = 'topbar.waiting'; }
      else if (phase === 'error') { tone = 'fail'; key = 'topbar.failed'; }
      else if (phase === 'offline') { tone = 'idle'; key = 'topbar.offline'; }
      el.className = 'status-pill ' + STATUS_BADGE_TONES[tone];
      el.innerHTML = `<span class="status-pill-dot" aria-hidden="true"></span>${escHtml(t(key))}`;
    }
    window.renderStatusBadge = renderStatusBadge;

    function applyRunStatus(status, version = null) {
      if (version !== null && version !== undefined) {
        const parsedVersion = Number(version);
        if (Number.isFinite(parsedVersion) && parsedVersion < runStatusVersion) return;
        if (Number.isFinite(parsedVersion)) runStatusVersion = parsedVersion;
      }
      const processing = status === 'processing';
      const paused = status === 'paused';
      runActive = processing || paused;
      renderStatusBadge();
      document.getElementById('send-btn').disabled = processing || paused;
      // The language also drives the report, and the server rejects a mid-run
      // change. Re-render the settings panel so its language radio locks.
      if (typeof renderSettings === 'function'
          && !document.getElementById('settings-modal').classList.contains('hidden')) {
        renderSettings();
      }
      document.getElementById('stop-btn').classList.toggle('hidden', !processing && !paused);
      if (processing) {
        showTyping();
        if (typeof RunTimer !== 'undefined') RunTimer.start();
      } else if (paused) {
        // A pause keeps pending approvals, active tasks and the same run timer.
        hideTyping();
      } else {
        hideTyping();
        resetAgents();
        activityMarkIdle();
        currentPlannerHitlRequest = null;
        updateRoadmapModalButtons();
        if (typeof RunTimer !== 'undefined') RunTimer.finish();
      }
    }

    function renderSessionSnapshot(snapshot) {
      const previousSessionId = activeSession && activeSession.id;
      activeUser = snapshot.user || activeUser;
      activeSession = snapshot.session || activeSession;
      if (activeSession && activeSession.id !== previousSessionId) {
        runStatusVersion = -1;
      }
      activeUser.last_session_id = activeSession.id;
      knownUsers = knownUsers.map(item => item.id === activeUser.id ? activeUser : item);
      knownSessions = knownSessions.map(item => item.id === activeSession.id ? activeSession : item);
      document.getElementById('active-nickname').textContent = activeUser.nickname;
      const feed = document.getElementById('chat-feed');
      const messages = snapshot.messages || [];
      resetExperimentViewer();
      activityReset();
      StatusIndicator.reset();
      eventCount = 0;
      renderEventCount();
      feed.innerHTML = '';
      if (window.resetHitlUiState) resetHitlUiState();

      // Cost is cumulative per session, so the snapshot carries the current
      // figure directly — clear first, or a session switch would show the
      // previous session's spend until the next push.
      resetMetrics();
      renderMetrics(snapshot.metrics);
      if (window.CheckpointsModal) CheckpointsModal.onSnapshot(snapshot.checkpoints || []);

      // The attachment belongs to the session the snapshot describes.
      applyDatasetUrl(snapshot.dataset_url);
      // A session that already has a language keeps it. A fresh one adopts the
      // interface language AND records it, so flipping the interface toggle
      // later does not silently rewrite a report language already in use.
      if (snapshot.report_language) {
        applyReportLanguage(snapshot.report_language);
      } else {
        applyReportLanguage(currentLang);
        sendReportLanguage(currentLang);
      }
      updateCoderSandboxButton(null);

      if (snapshot.active_tasks && Array.isArray(snapshot.active_tasks)) {
        StatusIndicator.feed({ type: 'session_snapshot', active_tasks: snapshot.active_tasks }, true);
      }
      // The ТЗ panel shows the session's latest ТЗ; a pending ТЗ form is
      // redelivered right after this snapshot.
      if (window.TZPanel) TZPanel.restore(snapshot.tz || null, activeSession && activeSession.id);

      for (const message of messages) {
        // Quiet replay: the indicator recomputes its state from the history so
        // a reconnect lands on the truth, without re-announcing every step.
        StatusIndicator.feed(message, true);
        if (message.type === 'user_message') {
          addUserMsg(message.message, message.timestamp);
        } else if (message.type === 'agent_event') {
          activityTouchAgent(message.author, message.timestamp);
          CallGraph.feedAgentEvent(message);
          if (hasText(message.content) && !isChatNoise(message)) {
            addAgentMsg(message.author || 'system', message.content, message.timestamp, message);
            const foundUrl = extractSandboxUrlFromText(message.content);
            if (foundUrl) updateCoderSandboxButton(foundUrl);
          }
          (message.tool_calls || [])
            .filter(call => call.name === 'transfer_to_agent')
            .forEach(call => activityRecordCall(message.author, call, message.timestamp));
          (message.tool_responses || []).forEach(response => {
            checkAndDisplaySandboxLinks(message.author, response, message.timestamp);
          });
        } else if (message.type === 'agent_output') {
          activityTouchAgent(message.agent, message.timestamp);
          if (!PLAN_AGENTS.includes(message.agent)) {
            addAgentOutputMsg(message.agent, message.content, message.timestamp, message.caller, message);
          }
        } else if (message.type === 'tool_activity') {
          applyToolActivity(message, true);
          if (window.PlanTracker) PlanTracker.feed(message);
        } else if (message.type === 'hitl_request') {
          // Drawn locked; a request that is still open is redelivered by the
          // server right after the snapshot and unlocks its card in place.
          showHITL(message, { history: true });
        } else if (['hitl_response', 'hitl_timeout', 'hitl_cancelled'].includes(message.type)) {
          applyHitlOutcome(message);
        } else if (message.type === 'work_order_notice') {
          renderWorkOrderNotice(message);
        } else if (message.type === 'error') {
          addSystemMsg(t('common.errorPrefix', { error: message.message }), message.timestamp);
        }
      }
      // A different session is a different store: its documents have different
      // ids, and anything cached from the last one is now about nothing.
      if (window.resetDocuments) resetDocuments();
      if (window.refreshSessionDocuments) refreshSessionDocuments();

      if (!messages.length) clearChat();
      resetPlanGate(messages);
      eventCount = messages.length;
      renderEventCount();

      applyRunStatus(snapshot.status, snapshot.run_status_version);
      CallGraph.loadHistory(activeUser && activeUser.id, activeSession && activeSession.id);
      if (typeof RunTimer !== 'undefined') {
        RunTimer.restoreFromSnapshot(snapshot);
      }
      StatusIndicator.feed({ type: 'status', status: snapshot.status });
      if (window.RunControl) RunControl.feed(snapshot);
      populateUserSelectors();
      populateSessionSelector();
    }

    // === Session Export / Import / Save / Restore ===

    async function exportCurrentSession() {
      if (!activeUser || !activeSession) return addSystemMsg(t('sessions.noActiveExport'));
      try {
        addSystemMsg(t('sessions.exporting'));
        const resp = await fetch(`/api/users/${encodeURIComponent(activeUser.id)}/sessions/${encodeURIComponent(activeSession.id)}/export`, { method: 'POST' });
        if (!resp.ok) {
          const errMsg = await fetchErrorMessage(resp);
          throw new Error(errMsg);
        }
        const blob = await resp.blob();
        const disposition = resp.headers.get('content-disposition') || '';
        let filename = `session_${activeSession.id}.cossession.zip`;
        const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
        if (utf8Match) {
          try { filename = decodeURIComponent(utf8Match[1]); } catch (_) { }
        } else {
          const match = disposition.match(/filename="([^"]+)"/);
          if (match) filename = match[1];
        }
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        a.click();
        URL.revokeObjectURL(a.href);
        addSystemMsg(t('sessions.exported', { filename: filename }));
      } catch (err) { addSystemMsg(t('sessions.exportFailed', { error: err.message })); }
    }

    async function saveCurrentSession() {
      if (!activeUser || !activeSession) return addSystemMsg(t('sessions.noActiveSave'));
      try {
        addSystemMsg(t('sessions.saving'));
        const data = await apiJson(`/api/users/${encodeURIComponent(activeUser.id)}/sessions/${encodeURIComponent(activeSession.id)}/save`, { method: 'POST' });
        addSystemMsg(t('sessions.saved', { filename: data.filename }));
      } catch (err) { addSystemMsg(t('sessions.saveFailed', { error: err.message })); }
    }

    function triggerImportSession() {
      if (!activeUser) return openIdentityModal();
      document.getElementById('import-session-input').click();
    }

    async function handleImportFile(input) {
      const file = input.files[0];
      input.value = '';
      if (!file) return;
      if (!activeUser) return openIdentityModal();
      try {
        addSystemMsg(t('sessions.importing', { filename: file.name }));
        // Step 1: preview the bundle for MCP builds
        const previewForm = new FormData();
        previewForm.append('file', file);
        const previewResp = await fetch('/api/import-session/preview', {
          method: 'POST', body: previewForm
        });
        let rebuildMcp = false;
        if (previewResp.ok) {
          const preview = await previewResp.json();
          if (preview.has_mcp_builds) {
            rebuildMcp = await showMcpRebuildModal(preview.mcp_builds);
          }
        }
        // Step 2: actual import
        const formData = new FormData();
        formData.append('file', file);
        formData.append('rebuild_mcp', rebuildMcp ? 'true' : 'false');
        const resp = await fetch(`/api/users/${encodeURIComponent(activeUser.id)}/import-session`, {
          method: 'POST',
          body: formData
        });
        if (!resp.ok) {
          const errMsg = await fetchErrorMessage(resp);
          throw new Error(errMsg);
        }
        const result = await resp.json();
        const importedUser = result.user;
        const importedSession = result.session;
        if (importedUser.id !== activeUser.id) {
          if (!knownUsers.find(u => u.id === importedUser.id)) {
            knownUsers.push(importedUser);
          }
          activeUser = importedUser;
          populateUserSelectors();
        }
        knownSessions.unshift(importedSession);
        await activateSession(activeUser, importedSession);
        let msg = t('sessions.imported', { title: importedSession.title });
        if (rebuildMcp) msg += t('sessions.mcpRebuilds');
        addSystemMsg(msg);
      } catch (err) { addSystemMsg(t('sessions.importFailed', { error: err.message })); }
    }

    function openSavedSessionsModal() {
      document.getElementById('saved-sessions-modal').classList.remove('hidden');
      loadSavedSessionsList();
    }

    function closeSavedSessionsModal() {
      document.getElementById('saved-sessions-modal').classList.add('hidden');
    }

    // --- MCP Rebuild Modal ---
    let _mcpRebuildResolve = null;

    function showMcpRebuildModal(builds) {
      return new Promise(resolve => {
        _mcpRebuildResolve = resolve;
        const list = document.getElementById('mcp-rebuild-list');
        const repos = [...new Set(builds.filter(b => b.repo_url).map(b => b.repo_url))];
        list.innerHTML = repos.map(url => {
          const name = url.split('/').pop().replace(/\.git$/, '');
          const job = builds.find(b => b.repo_url === url);
          const status = job ? job.status : 'unknown';
          const badge = status === 'done'
            ? '<span class="text-[9px] px-1.5 py-0.5 rounded bg-primary/15 text-primary font-bold uppercase">done</span>'
            : `<span class="text-[9px] px-1.5 py-0.5 rounded bg-outline-variant/15 text-outline-variant font-bold uppercase">${escHtml(status)}</span>`;
          return `<div class="flex items-center justify-between bg-surface-container-high rounded-lg px-3 py-2 border border-outline-variant/10">
            <div class="flex items-center gap-2 min-w-0">
              <span class="material-symbols-outlined text-primary text-sm">dns</span>
              <span class="text-[10px] text-on-surface font-mono truncate">${escHtml(name)}</span>
            </div>
            ${badge}
          </div>`;
        }).join('');
        document.getElementById('mcp-rebuild-skip-btn').onclick = () => { closeMcpRebuildModal(); resolve(false); };
        document.getElementById('mcp-rebuild-confirm-btn').onclick = () => { closeMcpRebuildModal(); resolve(true); };
        document.getElementById('mcp-rebuild-modal').classList.remove('hidden');
      });
    }

    function closeMcpRebuildModal() {
      document.getElementById('mcp-rebuild-modal').classList.add('hidden');
      _mcpRebuildResolve = null;
    }

    async function loadSavedSessionsList() {
      const container = document.getElementById('saved-sessions-list');
      container.innerHTML = `<p class="text-[10px] text-outline-variant italic">${t('common.loading')}</p>`;
      try {
        const data = await apiJson('/api/saved-sessions');
        const sessions = data.sessions || [];
        if (!sessions.length) {
          container.innerHTML = `<p class="text-[10px] text-outline-variant italic">${t('saved.empty')}</p>`;
          return;
        }
        container.innerHTML = sessions.map(s => {
          const sizeKb = (s.size_bytes / 1024).toFixed(1);
          const date = s.exported_at ? new Date(s.exported_at).toLocaleString() : '—';
          return `<div class="flex items-center justify-between bg-surface-container-high rounded-lg px-4 py-3 border border-outline-variant/10">
            <div class="flex-1 min-w-0">
              <p class="text-xs font-bold text-on-surface truncate">${escHtml(s.title)}</p>
              <p class="text-[10px] text-outline-variant">${date} · ${sizeKb} KB${s.original_nickname ? t('sessions.savedBy', { nick: escHtml(s.original_nickname) }) : ''}</p>
            </div>
            <div class="flex gap-2 ml-3 shrink-0">
              <button onclick="restoreSavedSession('${escHtml(s.filename)}')" title="${t('saved.restore')}"
                class="px-3 py-1.5 bg-primary text-on-primary rounded text-[10px] font-bold uppercase hover:brightness-110">
                ${t('saved.restore')}</button>
              <button onclick="downloadSavedSession('${escHtml(s.filename)}')" title="${t('saved.download')}"
                class="px-3 py-1.5 bg-surface-container border border-outline-variant/20 text-on-surface rounded text-[10px] font-bold uppercase hover:bg-surface-variant/40">
                <span class="material-symbols-outlined text-sm">download</span></button>
              <button onclick="deleteSavedSession('${escHtml(s.filename)}')" title="${t('saved.delete')}"
                class="px-3 py-1.5 bg-surface-container border border-error/30 text-error rounded text-[10px] font-bold uppercase hover:bg-error/10">
                <span class="material-symbols-outlined text-sm">delete</span></button>
            </div>
          </div>`;
        }).join('');
      } catch (err) {
        container.innerHTML = `<p class="text-[10px] text-error">${escHtml(t('sessions.loadFailed', { error: err.message }))}</p>`;
      }
    }

    async function restoreSavedSession(filename) {
      try {
        // Step 1: preview via reading the saved bundle
        let rebuildMcp = false;
        try {
          const previewResp = await fetch(`/api/saved-sessions/${encodeURIComponent(filename)}/download`);
          if (previewResp.ok) {
            const blob = await previewResp.blob();
            const previewForm = new FormData();
            previewForm.append('file', blob, filename);
            const pResp = await fetch('/api/import-session/preview', { method: 'POST', body: previewForm });
            if (pResp.ok) {
              const preview = await pResp.json();
              if (preview.has_mcp_builds) {
                rebuildMcp = await showMcpRebuildModal(preview.mcp_builds);
              }
            }
          }
        } catch (_) { /* preview is best-effort */ }
        // Step 2: actual restore
        addSystemMsg(t('sessions.restoring', { filename: filename }));
        const resp = await fetch('/api/restore-session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename, rebuild_mcp: rebuildMcp })
        });
        if (!resp.ok) {
          const errMsg = await fetchErrorMessage(resp);
          throw new Error(errMsg);
        }
        const result = await resp.json();
        const restoredUser = result.user;
        const restoredSession = result.session;
        if (!knownUsers.find(u => u.id === restoredUser.id)) {
          knownUsers.push(restoredUser);
        }
        activeUser = restoredUser;
        populateUserSelectors();
        knownSessions.unshift(restoredSession);
        await activateSession(activeUser, restoredSession);
        closeSavedSessionsModal();
        let msg = t('sessions.restored', { title: restoredSession.title });
        if (rebuildMcp) msg += t('sessions.mcpRebuilds');
        addSystemMsg(msg);
      } catch (err) { addSystemMsg(t('sessions.restoreFailed', { error: err.message })); }
    }

    async function downloadSavedSession(filename) {
      try {
        const resp = await fetch(`/api/saved-sessions/${encodeURIComponent(filename)}/download`);
        if (!resp.ok) {
          const errMsg = await fetchErrorMessage(resp);
          throw new Error(errMsg);
        }
        const blob = await resp.blob();
        const disposition = resp.headers.get('content-disposition') || '';
        let dlFilename = filename;
        const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
        if (utf8Match) {
          try { dlFilename = decodeURIComponent(utf8Match[1]); } catch (_) { }
        } else {
          const match = disposition.match(/filename="([^"]+)"/);
          if (match) dlFilename = match[1];
        }
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = dlFilename;
        a.click();
        URL.revokeObjectURL(a.href);
      } catch (err) { addSystemMsg(t('sessions.downloadFailed', { error: err.message })); }
    }

    async function deleteSavedSession(filename) {
      if (!confirm(t('sessions.deleteConfirm', { filename: filename }))) return;
      try {
        const resp = await fetch(`/api/saved-sessions/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        if (!resp.ok) {
          const errMsg = await fetchErrorMessage(resp);
          throw new Error(errMsg);
        }
        loadSavedSessionsList();
        addSystemMsg(t('sessions.deleted', { filename: filename }));
      } catch (err) { addSystemMsg(t('sessions.deleteFailed', { error: err.message })); }
    }
