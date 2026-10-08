// =========================================================================
// Local users and sessions (Management & Bundles Export/Import)
// =========================================================================
    function populateUserSelectors() {
      const options = knownUsers.map(user =>
        `<option value="${escHtml(user.id)}" ${activeUser && activeUser.id === user.id ? 'selected' : ''}>${escHtml(user.nickname)}</option>`
      ).join('');
      document.getElementById('user-select').innerHTML = options || '<option value="">No local users</option>';
      document.getElementById('identity-user-select').innerHTML = options;
      document.getElementById('existing-user-block').classList.toggle('hidden', knownUsers.length === 0);
    }

    function populateSessionSelector() {
      document.getElementById('session-select').innerHTML = knownSessions.map(session =>
        `<option value="${escHtml(session.id)}" ${activeSession && activeSession.id === session.id ? 'selected' : ''}>${escHtml(session.title)}</option>`
      ).join('') || '<option value="">No sessions</option>';
    }

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

    async function ensureUserSession(user, preferredSessionId = null) {
      activeUser = user;
      let sessions = await loadSessions(user);
      if (!sessions.length) {
        const created = await apiJson(`/api/users/${encodeURIComponent(user.id)}/sessions`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: 'New session' })
        });
        sessions = [created.session];
        knownSessions = sessions;
      }
      const selected = sessions.find(item => item.id === preferredSessionId)
        || sessions.find(item => item.id === user.last_session_id)
        || sessions[0];
      await activateSession(user, selected);
      closeIdentityModal();
    }

    async function registerLocalUser() {
      const nicknameInput = document.getElementById('nickname-input');
      const nickname = nicknameInput.value.trim();
      if (!nickname) return showIdentityError(new Error('Enter a Nick.'));
      try {
        const data = await apiJson('/api/users', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ nickname })
        });
        knownUsers.push(data.user);
        nicknameInput.value = '';
        populateUserSelectors();
        await ensureUserSession(data.user);
        closeIdentityModal();
      } catch (error) { showIdentityError(error); }
    }

    async function continueExistingUser() {
      const userId = document.getElementById('identity-user-select').value;
      const user = knownUsers.find(item => item.id === userId);
      if (!user) return;
      try {
        await ensureUserSession(user, localStorage.getItem(SESSION_STORAGE_KEY));
        closeIdentityModal();
      } catch (error) { showIdentityError(error); }
    }

    async function onUserSelected(userId) {
      const user = knownUsers.find(item => item.id === userId);
      if (user) await ensureUserSession(user);
    }

    async function onSessionSelected(sessionId) {
      const session = knownSessions.find(item => item.id === sessionId);
      if (session && activeUser) await activateSession(activeUser, session);
    }

    async function createNewSession() {
      if (!activeUser) return openIdentityModal();
      let title = 'New session';
      if (!appSettings.general.autoNamingEnabled) {
        const inputTitle = prompt('Session title:', 'New session');
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
      } catch (error) { addSystemMsg('Could not create session: ' + error.message); }
    }

    async function renameCurrentSession() {
      if (!activeUser || !activeSession) return;
      const title = prompt('New session title:', activeSession.title);
      if (!title || title.trim() === activeSession.title) return;
      try {
        const data = await apiJson(sessionApi(), {
          method: 'PATCH', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title })
        });
        activeSession = data.session;
        knownSessions = knownSessions.map(item => item.id === activeSession.id ? activeSession : item);
        populateSessionSelector();
      } catch (error) { addSystemMsg('Could not rename session: ' + error.message); }
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
      localStorage.setItem(NICK_STORAGE_KEY, user.nickname);
      document.getElementById('active-nickname').textContent = user.nickname;
      document.getElementById('graph-link').href =
        `/graph?user_id=${encodeURIComponent(user.id)}&session_id=${encodeURIComponent(session.id)}`;
      populateUserSelectors();
      populateSessionSelector();
      clearChat();
      // Drop the previous session's attachment; the snapshot brings the new one.
      applyDatasetUrl('');
      applyReportLanguage('');
      connect();
    }

    async function bootstrap() {
      updateCoderSandboxButton(null);
      try {
        const data = await apiJson('/api/users');
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
        await ensureUserSession(savedUser, localStorage.getItem(SESSION_STORAGE_KEY));
      } catch (error) {
        addSystemMsg('Failed to initialize local sessions: ' + error.message);
        openIdentityModal();
      }
    }

    function applyRunStatus(status, version = null) {
      if (version !== null && version !== undefined) {
        const parsedVersion = Number(version);
        if (Number.isFinite(parsedVersion) && parsedVersion < runStatusVersion) return;
        if (Number.isFinite(parsedVersion)) runStatusVersion = parsedVersion;
      }
      const processing = status === 'processing';
      document.getElementById('status-badge').textContent =
        'Status: ' + (processing ? 'Processing' : 'Idle');
      document.getElementById('send-btn').disabled = processing;
      document.getElementById('stop-btn').classList.toggle('hidden', !processing);
      if (processing) {
        showTyping();
      } else {
        hideTyping();
        resetAgents();
        activityMarkIdle();
        document.getElementById('hitl-panel').classList.add('hidden');
        currentPlannerHitlRequest = null;
        updateRoadmapModalButtons();
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
      document.getElementById('event-count').textContent = 'Events: 0';
      feed.innerHTML = '';

      // Cost is cumulative per session, so the snapshot carries the current
      // figure directly — clear first, or a session switch would show the
      // previous session's spend until the next push.
      resetMetrics();
      renderMetrics(snapshot.metrics);

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

      for (const message of messages) {
        // Quiet replay: the indicator recomputes its state from the history so
        // a reconnect lands on the truth, without re-announcing every step.
        StatusIndicator.feed(message, true);
        if (message.type === 'user_message') {
          addUserMsg(message.message, message.timestamp);
        } else if (message.type === 'agent_event') {
          activityTouchAgent(message.author, message.timestamp);
          if (hasText(message.content)) {
            addAgentMsg(message.author || 'system', message.content, message.timestamp);
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
          addAgentOutputMsg(message.agent, message.content, message.timestamp, message.caller);
        } else if (message.type === 'tool_activity') {
          applyToolActivity(message, true);
        } else if (message.type === 'error') {
          addSystemMsg('Error: ' + message.message, message.timestamp);
        }
      }
      if (!messages.length) clearChat();
      eventCount = messages.length;
      document.getElementById('event-count').textContent = 'Events: ' + eventCount;

      applyRunStatus(snapshot.status, snapshot.run_status_version);
      StatusIndicator.feed({ type: 'status', status: snapshot.status });
      populateUserSelectors();
      populateSessionSelector();
    }

    // === Session Export / Import / Save / Restore ===

    async function exportCurrentSession() {
      if (!activeUser || !activeSession) return addSystemMsg('No active session to export.');
      try {
        addSystemMsg('📥 Exporting session…');
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
        addSystemMsg('✅ Session exported: ' + filename);
      } catch (err) { addSystemMsg('❌ Export failed: ' + err.message); }
    }

    async function saveCurrentSession() {
      if (!activeUser || !activeSession) return addSystemMsg('No active session to save.');
      try {
        addSystemMsg('💾 Saving session to disk…');
        const data = await apiJson(`/api/users/${encodeURIComponent(activeUser.id)}/sessions/${encodeURIComponent(activeSession.id)}/save`, { method: 'POST' });
        addSystemMsg('✅ Session saved: ' + data.filename);
      } catch (err) { addSystemMsg('❌ Save failed: ' + err.message); }
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
        addSystemMsg('📤 Importing session from ' + file.name + '…');
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
        let msg = '✅ Session imported: ' + importedSession.title;
        if (rebuildMcp) msg += ' (MCP rebuilds launched)';
        addSystemMsg(msg);
      } catch (err) { addSystemMsg('❌ Import failed: ' + err.message); }
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
      container.innerHTML = '<p class="text-[10px] text-outline-variant italic">Loading…</p>';
      try {
        const data = await apiJson('/api/saved-sessions');
        const sessions = data.sessions || [];
        if (!sessions.length) {
          container.innerHTML = '<p class="text-[10px] text-outline-variant italic">No saved sessions found.</p>';
          return;
        }
        container.innerHTML = sessions.map(s => {
          const sizeKb = (s.size_bytes / 1024).toFixed(1);
          const date = s.exported_at ? new Date(s.exported_at).toLocaleString() : '—';
          return `<div class="flex items-center justify-between bg-surface-container-high rounded-lg px-4 py-3 border border-outline-variant/10">
            <div class="flex-1 min-w-0">
              <p class="text-xs font-bold text-on-surface truncate">${escHtml(s.title)}</p>
              <p class="text-[10px] text-outline-variant">${date} · ${sizeKb} KB${s.original_nickname ? ' · by ' + escHtml(s.original_nickname) : ''}</p>
            </div>
            <div class="flex gap-2 ml-3 shrink-0">
              <button onclick="restoreSavedSession('${escHtml(s.filename)}')" title="Restore"
                class="px-3 py-1.5 bg-primary text-on-primary rounded text-[10px] font-bold uppercase hover:brightness-110">
                Restore</button>
              <button onclick="downloadSavedSession('${escHtml(s.filename)}')" title="Download"
                class="px-3 py-1.5 bg-surface-container border border-outline-variant/20 text-on-surface rounded text-[10px] font-bold uppercase hover:bg-surface-variant/40">
                <span class="material-symbols-outlined text-sm">download</span></button>
              <button onclick="deleteSavedSession('${escHtml(s.filename)}')" title="Delete"
                class="px-3 py-1.5 bg-surface-container border border-error/30 text-error rounded text-[10px] font-bold uppercase hover:bg-error/10">
                <span class="material-symbols-outlined text-sm">delete</span></button>
            </div>
          </div>`;
        }).join('');
      } catch (err) {
        container.innerHTML = `<p class="text-[10px] text-error">Failed to load saved sessions: ${escHtml(err.message)}</p>`;
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
        addSystemMsg('📂 Restoring session from ' + filename + '…');
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
        let msg = '✅ Session restored: ' + restoredSession.title;
        if (rebuildMcp) msg += ' (MCP rebuilds launched)';
        addSystemMsg(msg);
      } catch (err) { addSystemMsg('❌ Restore failed: ' + err.message); }
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
      } catch (err) { addSystemMsg('❌ Download failed: ' + err.message); }
    }

    async function deleteSavedSession(filename) {
      if (!confirm('Delete saved session "' + filename + '"?')) return;
      try {
        const resp = await fetch(`/api/saved-sessions/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        if (!resp.ok) {
          const errMsg = await fetchErrorMessage(resp);
          throw new Error(errMsg);
        }
        loadSavedSessionsList();
        addSystemMsg('🗑 Saved session deleted: ' + filename);
      } catch (err) { addSystemMsg('❌ Delete failed: ' + err.message); }
    }
