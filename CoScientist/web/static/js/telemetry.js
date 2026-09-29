// =========================================================================
// Telemetry & Usage Metrics
// =========================================================================
    // =========================================================================
    // Telemetry
    // =========================================================================
    function renderEventCount() {
      const el = document.getElementById('event-count');
      if (el) el.textContent = t('topbar.events', { count: eventCount });
    }

    function addTelemetry(text) {
      const log = document.getElementById('telemetry-log');
      const t = new Date().toLocaleTimeString('en-GB', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
      log.innerHTML += `<div class="flex gap-2"><span class="text-outline-variant">[${t}]</span> ${escHtml(text)}</div>`;
      log.scrollTop = log.scrollHeight;
      eventCount++;
      renderEventCount();
    }


    // =========================================================================
    // Usage & Cost
    //
    // Cumulative per session and pushed from the server's ledger, so a
    // reconnecting tab gets the whole picture in one snapshot instead of having
    // to add up a stream. Rows are agents; a sandbox run shows as the child of
    // the agent that started it, because that is who spent the money.
    //
    // The panel is folded by default — the spend is checked now and then, not
    // watched — and the two numbers worth watching (total and elapsed) stay on
    // the summary row either way. Whether a reader keeps it open is a per-
    // browser habit, so it is remembered here rather than reset every reload.
    // =========================================================================
    const USAGE_PANEL_KEY = 'coscientist.usage_open';

    (function restoreUsagePanel() {
      const panel = document.getElementById('usage-panel');
      if (!panel) return;
      try {
        panel.open = localStorage.getItem(USAGE_PANEL_KEY) === 'true';
      } catch (_) { /* private mode: the default (folded) stands */ }
      panel.addEventListener('toggle', () => {
        try { localStorage.setItem(USAGE_PANEL_KEY, String(panel.open)); } catch (_) { }
      });
    })();

    // The agent's human name where the status indicator knows one.
    function agentName(name) {
      return (window.StatusIndicator && StatusIndicator.agentName) ? StatusIndicator.agentName(name) : name;
    }

    function fmtUsd(value) {
      const n = Number(value) || 0;
      if (n && n < 0.0001) return '<$0.0001';
      return '$' + n.toFixed(4);
    }

    function fmtTokens(value) {
      const n = Number(value) || 0;
      if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
      if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
      return String(n);
    }

    const USAGE_TOP_N = 5;
    let usageShowAll = false;
    let lastMetrics = null;

    function toggleUsageAll() {
      usageShowAll = !usageShowAll;
      if (lastMetrics) renderMetrics(lastMetrics);
    }

    function renderMetrics(data) {
      if (!data) return;
      const totals = data.totals || {};
      const llm = data.llm || {};
      const sandbox = data.sandbox || {};

      document.getElementById('metrics-total').textContent = fmtUsd(totals.cost_usd);

      const summaryEl = document.getElementById('metrics-summary');
      delete summaryEl.dataset.empty;
      const summary = [`${llm.calls || 0} ${t('metrics.calls')}`, `${fmtTokens(totals.total_tokens)} ${t('metrics.tokens')}`];
      if (sandbox.runs) {
        summary.push(`${sandbox.runs} sandbox`, `${Math.round(sandbox.gpu_seconds || 0)}s GPU`);
        if (sandbox.energy_wh) summary.push(`${(sandbox.energy_wh).toFixed(1)} Wh`);
      }
      summaryEl.textContent = summary.join(' · ');

      // Most expensive first, a bar for each relative to the top one, and the
      // long tail folded into one line: the question is "who spent it", which
      // the first few rows answer.
      const agents = (data.agents || []).slice()
        .sort((a, b) => (Number(b.cost_usd) || 0) - (Number(a.cost_usd) || 0));
      const top = Number(agents.length && agents[0].cost_usd) || 0;
      const shown = usageShowAll ? agents : agents.slice(0, USAGE_TOP_N);
      const rows = shown.map(agent => {
        const cost = Number(agent.cost_usd) || 0;
        const width = top ? Math.max(2, Math.round((cost / top) * 100)) : 0;
        const box = agent.sandbox;
        const child = box ? `
            <div class="flex justify-between gap-2 pl-3 text-outline-variant">
              <span class="truncate">sandbox · ${Math.round(box.agent_seconds || 0)}s</span>
              <span class="tabular-nums">${fmtUsd(box.total_cost_usd)}</span>
            </div>` : '';
        return `
          <div>
            <div class="flex items-baseline gap-2">
              <span class="flex-1 min-w-0 truncate text-on-surface-variant" title="${escHtml(agent.agent)}" translate="no">${escHtml(agentName(agent.agent))}</span>
              <span class="text-outline-variant tabular-nums whitespace-nowrap">${fmtTokens(agent.llm.total_tokens)}</span>
              <span class="w-16 text-right text-on-surface tabular-nums whitespace-nowrap">${fmtUsd(agent.cost_usd)}</span>
            </div>
            <div class="mt-1 h-[3px] rounded-full bg-surface-container-high overflow-hidden" aria-hidden="true">
              <div class="h-full rounded-full bg-outline-variant/50" style="width:${width}%"></div>
            </div>${child}
          </div>`;
      });
      const rest = agents.slice(USAGE_TOP_N);
      if (rest.length) {
        const restCost = rest.reduce((sum, agent) => sum + (Number(agent.cost_usd) || 0), 0);
        rows.push(`
          <button type="button" onclick="toggleUsageAll()" class="text-primary hover:underline">${escHtml(usageShowAll
            ? t('usage.showFewer')
            : t('usage.showRest', { n: rest.length, cost: fmtUsd(restCost) }))}</button>`);
      }
      lastMetrics = data;
      document.getElementById('metrics-agents').innerHTML =
        rows.join('') || '<div class="text-outline-variant">—</div>';

      // A model litellm has no price for is counted in tokens but not in money:
      // say so, rather than let the total read as the whole bill.
      const note = document.getElementById('metrics-note');
      if (totals.complete === false) {
        const models = (llm.unpriced_models || []).join(', ');
        note.textContent =
          t('metrics.unpriced', { count: llm.unpriced_calls }) +
          (models ? `: ${models}` : '.');
        note.classList.remove('hidden');
      } else {
        note.classList.add('hidden');
      }
    }

    function resetMetrics() {
      const summaryEl = document.getElementById('metrics-summary');
      document.getElementById('metrics-total').textContent = '$0.0000';
      summaryEl.textContent = t('metrics.noCalls');
      summaryEl.dataset.empty = '1';
      document.getElementById('metrics-agents').innerHTML = '';
      document.getElementById('metrics-note').classList.add('hidden');
      RunTimer.reset();
    }

    // =========================================================================
    // Run Execution Timer (Start to Finish)
    // =========================================================================
    function fmtDuration(ms) {
      const total = Math.max(0, Math.floor(ms / 1000));
      const h = Math.floor(total / 3600);
      const m = Math.floor((total % 3600) / 60);
      const s = total % 60;
      const pad = (n) => String(n).padStart(2, '0');
      if (h > 0) {
        return `${h}:${pad(m)}:${pad(s)}`;
      }
      return `${pad(m)}:${pad(s)}`;
    }

    function fmtDurationHuman(ms, isRu) {
      const total = Math.max(0, Math.round(ms / 1000));
      const h = Math.floor(total / 3600);
      const m = Math.floor((total % 3600) / 60);
      const s = total % 60;
      if (isRu) {
        const parts = [];
        if (h > 0) parts.push(`${h} ч`);
        if (m > 0 || h > 0) parts.push(`${m} мин`);
        parts.push(`${s} с`);
        return parts.join(' ');
      } else {
        const parts = [];
        if (h > 0) parts.push(`${h}h`);
        if (m > 0 || h > 0) parts.push(`${m}m`);
        parts.push(`${s}s`);
        return parts.join(' ');
      }
    }

    const RunTimer = {
      startedAt: null,
      finishedAt: null,
      lastElapsedMs: 0,
      timerId: null,
      isRunning: false,

      start(timestamp) {
        const parsed = timestamp != null ? new Date(timestamp).getTime() : NaN;
        if (this.isRunning && this.startedAt !== null) {
          // Status/usage notifications are not new runs. A server timestamp
          // may correct the optimistic client start, but an untimed update
          // must never reset the origin or replace the ticking interval.
          if (Number.isFinite(parsed)) this.startedAt = parsed;
          this.tick();
          return;
        }
        this.startedAt = Number.isFinite(parsed) ? parsed : Date.now();
        this.finishedAt = null;
        this.isRunning = true;
        this.stopInterval();

        const dot = document.getElementById('metrics-duration-dot');
        if (dot) dot.classList.remove('hidden');

        const durEl = document.getElementById('metrics-duration');
        if (durEl) durEl.classList.remove('text-outline-variant');

        this.tick();
        this.timerId = setInterval(() => this.tick(), 1000);
      },

      finish(timestamp) {
        // In particular, preserve a duration restored from session history.
        if (!this.isRunning) return;
        this.stopInterval();
        const parsed = timestamp ? new Date(timestamp).getTime() : Date.now();
        this.finishedAt = Number.isFinite(parsed) ? parsed : Date.now();
        this.isRunning = false;

        const dot = document.getElementById('metrics-duration-dot');
        if (dot) dot.classList.add('hidden');

        const durEl = document.getElementById('metrics-duration');
        if (durEl) {
          durEl.classList.remove('text-primary');
        }

        const elapsedMs = this.startedAt !== null ? Math.max(0, this.finishedAt - this.startedAt) : this.lastElapsedMs;
        this.lastElapsedMs = elapsedMs;
        this.render(elapsedMs, false);
      },

      setDuration(elapsedMs) {
        this.stopInterval();
        this.isRunning = false;
        this.startedAt = null;
        this.finishedAt = null;
        this.lastElapsedMs = Math.max(0, Number(elapsedMs) || 0);

        const dot = document.getElementById('metrics-duration-dot');
        if (dot) dot.classList.add('hidden');

        const durEl = document.getElementById('metrics-duration');
        if (durEl) {
          durEl.classList.remove('text-primary');
        }

        this.render(this.lastElapsedMs, false);
      },

      reset() {
        this.stopInterval();
        this.startedAt = null;
        this.finishedAt = null;
        this.lastElapsedMs = 0;
        this.isRunning = false;

        const dot = document.getElementById('metrics-duration-dot');
        if (dot) dot.classList.add('hidden');

        const durEl = document.getElementById('metrics-duration');
        if (durEl) {
          durEl.textContent = '00:00';
          durEl.classList.remove('text-primary');
          durEl.classList.add('text-outline-variant');
        }

        const wrap = document.getElementById('metrics-duration-wrap');
        if (wrap) {
          const isRu = typeof currentLang !== 'undefined' && currentLang === 'ru';
          wrap.title = isRu ? 'Время выполнения' : 'Run duration';
        }
        const topEl = document.getElementById('topbar-elapsed');
        if (topEl) topEl.classList.add('hidden');
      },

      stopInterval() {
        if (this.timerId) {
          clearInterval(this.timerId);
          this.timerId = null;
        }
      },

      tick() {
        if (this.startedAt === null) return;
        const elapsedMs = Math.max(0, Date.now() - this.startedAt);
        this.lastElapsedMs = elapsedMs;
        this.render(elapsedMs, true);
      },

      render(ms, running) {
        const durEl = document.getElementById('metrics-duration');
        if (durEl) {
          durEl.textContent = fmtDuration(ms);
        }
        const topEl = document.getElementById('topbar-elapsed');
        if (topEl) {
          topEl.textContent = `${t('topbar.elapsed')} ${fmtDuration(ms)}`;
          topEl.classList.toggle('hidden', !running);
        }
        const wrap = document.getElementById('metrics-duration-wrap');
        if (wrap) {
          const isRu = typeof currentLang !== 'undefined' && currentLang === 'ru';
          const human = fmtDurationHuman(ms, isRu);
          wrap.title = running
            ? (isRu ? `Выполняется: ${human}` : `Running: ${human}`)
            : (isRu ? `Время работы: ${human} (со старта до финиша)` : `Run time: ${human} (start to finish)`);
        }
      },

      reapplyLanguage() {
        this.render(this.lastElapsedMs, this.isRunning);
      },

      restoreFromSnapshot(snapshot) {
        if (!snapshot) return;
        const control = snapshot.run_control;
        const ongoing = control
          ? ['running', 'pause_requested', 'paused'].includes(control.state)
          : ['processing', 'paused'].includes(snapshot.status);
        if (ongoing) {
          // Wall time includes operator/budget pauses, just as it does before
          // reconnecting. The durable origin also survives server restarts.
          const start = control?.started_at
            || (snapshot.run_times && snapshot.run_times.started_at)
            || this._findLastUserMessageTimestamp(snapshot.messages)
            || Date.now();
          this.start(start);
        } else {
          if (snapshot.run_times && snapshot.run_times.started_at && snapshot.run_times.finished_at) {
            const start = new Date(snapshot.run_times.started_at).getTime();
            const finish = new Date(snapshot.run_times.finished_at).getTime();
            if (Number.isFinite(start) && Number.isFinite(finish) && finish >= start) {
              this.setDuration(finish - start);
              return;
            }
          }
          // Fallback to message history
          const duration = this._calcDurationFromMessages(snapshot.messages);
          if (duration !== null) {
            this.setDuration(duration);
          } else {
            this.reset();
          }
        }
      },

      _findLastUserMessageTimestamp(messages) {
        if (!Array.isArray(messages)) return null;
        for (let i = messages.length - 1; i >= 0; i--) {
          if (messages[i].type === 'user_message' && messages[i].timestamp) {
            return messages[i].timestamp;
          }
        }
        return null;
      },

      _calcDurationFromMessages(messages) {
        if (!Array.isArray(messages) || !messages.length) return null;
        let lastUserIndex = -1;
        for (let i = messages.length - 1; i >= 0; i--) {
          if (messages[i].type === 'user_message') {
            lastUserIndex = i;
            break;
          }
        }
        if (lastUserIndex === -1) return null;
        const userMsg = messages[lastUserIndex];
        const start = new Date(userMsg.timestamp).getTime();
        if (!Number.isFinite(start)) return null;

        let end = start;
        for (let i = messages.length - 1; i > lastUserIndex; i--) {
          if (messages[i].timestamp) {
            const t = new Date(messages[i].timestamp).getTime();
            if (Number.isFinite(t) && t >= start) {
              end = t;
              break;
            }
          }
        }
        return Math.max(0, end - start);
      }
    };
    window.RunTimer = RunTimer;
