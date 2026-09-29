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
        document.getElementById('conn-dot').className = 'w-1.5 h-1.5 rounded-full bg-secondary shrink-0';
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
        document.getElementById('conn-dot').className = 'w-1.5 h-1.5 rounded-full bg-error shrink-0';
        if (intentionalDisconnect || !activeUser || !activeSession
          || activeUser.id !== userId || activeSession.id !== sessionId) return;
        if (event.code === 4404) {
          localStorage.removeItem(USER_STORAGE_KEY);
          localStorage.removeItem(SESSION_STORAGE_KEY);
          bootstrap();
          return;
        }
        addTelemetry('DISCONNECTED — retrying in 3s');
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
        // After the roadmap: the tracker reads the task list the line above
        // has just refreshed, so the two must not be swapped.
        if (window.PlanTracker && typeof window.PlanTracker.feed === 'function') {
          window.PlanTracker.feed(data);
        }
        if (window.TZPanel) window.TZPanel.feed(data);
        switch (data.type) {
          case 'run_control':
            if (window.RunControl) RunControl.feed(data);
            break;
          case 'connected':
            addTelemetry('INIT :: ' + data.message);
            break;
          case 'session_snapshot':
            renderSessionSnapshot(data);
            break;
          case 'agent_configuration':
            if (typeof loadSettings === 'function') loadSettings();
            // The set of agents changed: so did the places they can run in.
            if (activeUser && activeSession) CallGraph.loadSkeleton(activeUser.id, activeSession.id);
            addTelemetry('AGENTS :: configuration revision ' + data.desiredRevision);
            break;
          case 'status':
            applyRunStatus(data.status, data.run_status_version);
            if (typeof RunTimer !== 'undefined') {
              if (data.status === 'processing') RunTimer.start(data.started_at);
              else if (data.status !== 'paused') RunTimer.finish(data.finished_at);
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
            CallGraph.feedAgentEvent(data);
            if (isPostPlanAgent(data.author)) releasePlanGate();
            if (hasText(data.content) && isChatNoise(data)) {
              addTelemetry('NOTE :: ' + data.author + ' :: ' + stripThinking(data.content).slice(0, 200));
            } else if (hasText(data.content)) {
              hideTyping();
              highlightAgent(data.author);
              addAgentMsg(data.author, data.content, data.timestamp, data);
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
            if (PLAN_AGENTS.includes(data.agent)) {
              addTelemetry('OUTPUT :: ' + data.agent + ' (plan view only)');
              break;
            }
            highlightAgent(data.agent);
            addAgentOutputMsg(data.agent, data.content, data.timestamp, data.caller, data);
            if (window.refreshSessionDocuments) refreshSessionDocuments();
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
            if (data.document && window.openDocument) {
              openDocument(data.document.artifact_id, data.document.title);
            }
            if (window.TZPanel) TZPanel.clearRequest(null);
            resetAgents();
            activityMarkIdle();
            currentPlannerHitlRequest = null;
            updateRoadmapModalButtons();
            if (typeof RunTimer !== 'undefined' && !window.RunControl?.isActive?.()) RunTimer.finish();
            addTelemetry('COMPLETE :: Final response received');
            break;
          case 'hitl_request':
            hideTyping();
            showHITL(data);
            // Being asked to approve something is the moment to read it.
            if (data.document && window.openDocumentForRequest) {
              openDocumentForRequest(data.request_id, data.document);
            }
            if (window.refreshSessionDocuments) refreshSessionDocuments();
            if (data.agent_name === 'PlannerAgent') {
              currentPlannerHitlRequest = data;
              updateRoadmapModalButtons();
            }
            addTelemetry('HITL :: ' + ((window.StatusIndicator && StatusIndicator.agentName) ? StatusIndicator.agentName(data.agent_name) : data.agent_name) + ' requests ' + data.action_type);
            break;
          case 'hitl_timeout':
            if (data.paused) {
              // The server is still waiting on the same durable request.
              // Timeout is not rejection/approval and must not disable input.
              addSystemMsg(hitlTimeoutSummary(data));
              addTelemetry('HITL :: paused, awaiting explicit response');
              break;
            }
            if (data.agent_name === 'PlannerAgent' && !data.paused) releasePlanGate();
            disableHitlControls(data.request_id);
            if (window.TZPanel) TZPanel.clearRequest(data.request_id, 'Нет ответа — ТЗ принято как есть.');
            currentPlannerHitlRequest = null;
            updateRoadmapModalButtons();
            addSystemMsg(hitlTimeoutSummary(data));
            addTelemetry('HITL :: auto-approve on timeout (' + ((window.StatusIndicator && StatusIndicator.agentName) ? StatusIndicator.agentName(data.agent_name) : (data.agent_name || '?')) + ')');
            break;
          case 'hitl_hold':
            applyWorkOrderHold(data.request_id);
            addTelemetry('HITL :: countdown paused');
            break;
          case 'work_order_notice':
            renderWorkOrderNotice(data);
            addTelemetry('WORK ORDER :: ' + ((window.StatusIndicator && StatusIndicator.agentName) ? StatusIndicator.agentName(data.agent_name) : (data.agent_name || '?')) + ' ' + (data.kind || ''));
            break;
          case 'hitl_cancelled':
            disableHitlControls(data.request_id);
            if (window.TZPanel) TZPanel.clearRequest(data.request_id, 'Запрос отменён вместе с запуском.');
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
          case 'checkpoint_created':
            if (window.CheckpointsModal) CheckpointsModal.onCreated(data);
            addTelemetry('CHECKPOINT :: ' + (data.title || data.agent || '?'));
            break;
          case 'checkpoint_restored':
            addSystemMsg(
              `Состояние восстановлено перед стадией ${Number(data.stage_index) + 1}`
              + ` из ${data.stage_count || '?'}: ${data.title || data.agent || 'stage'}.`
              + (data.continue ? '\nПродолжение запущено автоматически.' : '')
            , data.timestamp);
            applyDatasetUrl(data.dataset_url || '');
            applyReportLanguage(data.report_language || '');
            addTelemetry('CHECKPOINT RESTORE :: ' + (data.title || data.agent || '?'));
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
            if (typeof RunTimer !== 'undefined' && !window.RunControl?.isActive?.()) RunTimer.finish();
            break;
          case 'pong':
            break;
        }
      };
    }
