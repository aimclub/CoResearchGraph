// =========================================================================
// WebSocket Connection & Event Dispatcher
// =========================================================================
    // =========================================================================
    // WebSocket
    // =========================================================================
    function connect() {
      if (!activeUser || !activeSession) return;
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const userId = activeUser.id;
      const sessionId = activeSession.id;
      intentionalDisconnect = false;
      const socket = new WebSocket(
        `${proto}://${location.host}/ws?user_id=${encodeURIComponent(userId)}&session_id=${encodeURIComponent(sessionId)}`
      );
      ws = socket;

      socket.onopen = () => {
        const connEntry = i18n['nav.connected'];
        document.getElementById('conn-status').textContent = (connEntry && connEntry[currentLang]) || 'Connected';
        document.getElementById('conn-status').className = 'text-[8px] text-secondary uppercase font-bold tracking-widest';
        document.getElementById('live-dot').className = 'w-2 h-2 bg-secondary rounded-full animate-pulse';
        const badgeEntry = i18n['chat.online'];
        document.getElementById('active-badge').textContent = (badgeEntry && badgeEntry[currentLang]) || 'Online';
        document.getElementById('active-badge').className = 'text-[10px] bg-surface-container-highest px-3 py-1 rounded text-primary border border-primary/20 uppercase font-bold tracking-widest';
        document.getElementById('telemetry-live').innerHTML = '<span class="w-1 h-1 bg-secondary rounded-full"></span> ' + t('telemetry.live');
        document.getElementById('telemetry-live').className = 'text-[8px] font-bold text-secondary animate-pulse font-mono tracking-tighter uppercase flex items-center gap-1';
        addTelemetry('CONNECTED to backend');
        StatusIndicator.setConnected(true);
      };

      socket.onclose = (event) => {
        if (ws === socket) ws = null;
        StatusIndicator.setConnected(false);
        const connEntry = i18n['nav.disconnected'];
        document.getElementById('conn-status').textContent = (connEntry && connEntry[currentLang]) || 'Disconnected';
        document.getElementById('conn-status').className = 'text-[8px] text-error uppercase font-bold tracking-widest';
        document.getElementById('live-dot').className = 'w-2 h-2 bg-outline-variant/60 rounded-full';
        const badgeEntry = i18n['chat.offline'];
        document.getElementById('active-badge').textContent = (badgeEntry && badgeEntry[currentLang]) || 'Offline';
        document.getElementById('active-badge').className = 'text-[10px] bg-surface-container-highest px-3 py-1 rounded text-outline-variant border border-outline-variant/20 uppercase font-bold tracking-widest';
        if (intentionalDisconnect || !activeUser || !activeSession
          || activeUser.id !== userId || activeSession.id !== sessionId) return;
        // The auth gate refused the handshake. Measured in Chrome against the
        // shipped uvicorn: a refusal before the accept comes back as HTTP 403
        // on the upgrade, and the browser reports close code 1006 — never
        // 1008, never 4403. So neither branch below runs in the deployment.
        // They are kept for an ASGI server that closes cleanly, and for the
        // test client, which does report the code. The live path is the 1006
        // fall-through at the end, which probes with a fetch.
        if (event.code === 1008) {
          window.location.href = '/login';
          return;
        }
        // Refused for its Origin, not for want of a session. The user is
        // logged in, so /login would send them straight back here — a loop no
        // correct password can escape. Unreachable under uvicorn, as above:
        // in that deployment an origin refusal arrives as 1006 and falls
        // through to the retry. The server names the refused origin on stderr,
        // so journalctl carries the diagnosis — see deploy/README.md note 2.
        if (event.code === 4403) {
          addTelemetry('REFUSED :: origin not allowed — set AUTH__ALLOWED_ORIGINS');
          return;
        }
        if (event.code === 4404) {
          localStorage.removeItem(USER_STORAGE_KEY);
          localStorage.removeItem(SESSION_STORAGE_KEY);
          bootstrap();
          return;
        }
        addTelemetry('DISCONNECTED — retrying in 3s');
        // The live refusal path: every close the deployment produces lands
        // here as 1006. One cheap authenticated call separates "session
        // expired" from "server restarting": the fetch wrapper in state.js
        // redirects to /login on 401. If the server is simply down the call
        // fails and the retry below carries on as before. A call that
        // succeeds while the socket keeps failing means neither — the usual
        // cause is the origin check, which the server log names.
        fetch('/api/users').catch(() => { });
        reconnectTimer = setTimeout(connect, 3000);
      };

      socket.onerror = () => addTelemetry('ERROR :: WebSocket error');

      socket.onmessage = (e) => {
        const data = JSON.parse(e.data);
        // One entry point for the status indicator: it reduces the whole
        // stream itself, so no case below has to know it exists.
        StatusIndicator.feed(data);
        if (window.RoadmapModal && typeof window.RoadmapModal.feed === 'function') {
          window.RoadmapModal.feed(data);
        }
        switch (data.type) {
          case 'connected':
            addTelemetry('INIT :: ' + data.message);
            break;
          case 'session_snapshot':
            renderSessionSnapshot(data);
            break;
          case 'status':
            applyRunStatus(data.status, data.run_status_version);
            if (typeof RunTimer !== 'undefined') {
              if (data.status === 'processing') RunTimer.start(data.started_at);
              else RunTimer.finish(data.finished_at);
            }
            addTelemetry('STATUS :: ' + data.message);
            break;
          case 'user_message':
            addUserMsg(data.message, data.timestamp);
            eventCount++;
            renderEventCount();
            break;
          case 'agent_event':
            activityTouchAgent(data.author, data.timestamp);
            if (hasText(data.content)) {
              hideTyping();
              highlightAgent(data.author);
              addAgentMsg(data.author, data.content, data.timestamp);
              const foundUrl = extractSandboxUrlFromText(data.content);
              if (foundUrl) updateCoderSandboxButton(foundUrl);
              addTelemetry('EVENT :: ' + data.author + (data.is_final ? ' [FINAL]' : ''));
              if (!data.is_final) showTyping();
            } else {
              addTelemetry('EVENT :: ' + data.author + ' (no content)');
            }
            // Tool call/result rendering is driven by the `tool_activity`
            // stream instead: it also covers tools used inside AgentTool
            // sub-agents, which never appear in these top-level parts. Only
            // agent hand-offs and sandbox links are read from here.
            if (data.tool_calls) {
              data.tool_calls
                .filter(tc => tc.name === 'transfer_to_agent')
                .forEach(tc => activityRecordCall(data.author, tc, data.timestamp));
            }
            if (data.tool_responses) {
              data.tool_responses.forEach(tr => {
                checkAndDisplaySandboxLinks(data.author, tr, data.timestamp);
              });
            }
            break;
          case 'agent_output':
            // The run continues after a subordinate answers, so the typing
            // indicator stays up — only the deliverable is posted here.
            activityTouchAgent(data.agent, data.timestamp);
            highlightAgent(data.agent);
            addAgentOutputMsg(data.agent, data.content, data.timestamp, data.caller);
            addTelemetry('OUTPUT :: ' + data.agent + ' → ' + (data.caller || 'system'));
            break;
          case 'tool_activity':
            applyToolActivity(data);
            break;
          case 'metrics':
            renderMetrics(data);
            break;
          case 'final_response':
            hideTyping();
            resetAgents();
            activityMarkIdle();
            currentPlannerHitlRequest = null;
            updateRoadmapModalButtons();
            if (typeof RunTimer !== 'undefined') RunTimer.finish();
            addTelemetry('COMPLETE :: Final response received');
            break;
          case 'hitl_request':
            hideTyping();
            showHITL(data);
            if (data.agent_name === 'PlannerAgent') {
              currentPlannerHitlRequest = data;
              updateRoadmapModalButtons();
            }
            addTelemetry('HITL :: ' + data.agent_name + ' requests ' + data.action_type);
            break;
          case 'hitl_timeout':
            disableHitlControls(data.request_id);
            document.getElementById('hitl-panel').classList.add('hidden');
            currentPlannerHitlRequest = null;
            updateRoadmapModalButtons();
            addSystemMsg(hitlTimeoutSummary(data));
            addTelemetry('HITL :: auto-approve on timeout (' + (data.agent_name || '?') + ')');
            break;
          case 'hitl_hold':
            applyWorkOrderHold(data.request_id);
            addTelemetry('HITL :: countdown paused');
            break;
          case 'work_order_notice':
            renderWorkOrderNotice(data);
            addTelemetry('WORK ORDER :: ' + (data.agent_name || '?') + ' ' + (data.kind || ''));
            break;
          case 'hitl_cancelled':
            disableHitlControls(data.request_id);
            document.getElementById('hitl-panel').classList.add('hidden');
            currentPlannerHitlRequest = null;
            updateRoadmapModalButtons();
            addTelemetry('HITL :: cancelled with its run');
            break;
          case 'chat_rejected':
            addSystemMsg(data.message);
            addTelemetry('SEND :: rejected while session is processing');
            break;
          case 'dataset_url':
            applyDatasetUrl(data.dataset_url);
            addTelemetry('DATASET :: ' + (data.dataset_url ? 'attached' : 'detached'));
            addSystemMsg(data.dataset_url
              ? t('ws.datasetAttached', { url: data.dataset_url })
              : t('ws.datasetDetached'));
            break;
          case 'dataset_url_rejected':
            showDatasetError(data.message);
            addSystemMsg(t('ws.datasetRejected', { message: data.message }));
            addTelemetry('DATASET :: rejected');
            break;
          case 'report_language':
            applyReportLanguage(data.report_language);
            addTelemetry('REPORT LANG :: ' + data.report_language);
            break;
          case 'report_language_rejected':
            addSystemMsg(t('ws.reportLangRejected', { message: data.message }));
            addTelemetry('REPORT LANG :: rejected' + (data.reason ? ' (' + data.reason + ')' : ''));
            // The server kept the old language. Put the interface back on it;
            // applyLanguage() without an argument re-renders and does not resend.
            currentLang = reportLanguage || currentLang;
            applyLanguage();
            break;
          case 'chat_accepted': {
            const input = document.getElementById('chat-input');
            if (input.value.trim() === String(data.message_text || '').trim()) {
              resetChatInput();
            }
            break;
          }
          case 'error':
            hideTyping();
            addSystemMsg(t('common.errorPrefix', { error: data.message }));
            addTelemetry('ERROR :: ' + data.message);
            currentPlannerHitlRequest = null;
            updateRoadmapModalButtons();
            if (typeof RunTimer !== 'undefined') RunTimer.finish();
            break;
          case 'pong':
            break;
        }
      };
    }

