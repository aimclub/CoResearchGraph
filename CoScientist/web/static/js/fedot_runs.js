/* Shared session/run picker for the trace and the vendored FEDOT graph. */
(() => {
  'use strict';
  window.FedotRuns = function ({host, onReset, onEvent, onStatus = () => {}}) {
    const query = new URLSearchParams(location.search);
    const user = query.get('user_id'), session = query.get('session_id');
    let selected = query.get('run_id') || '', stream = null, seq = 0, timer = null;
    let loading = false, runs = [], stopped = false;
    host.style.cssText = 'display:flex;flex-wrap:wrap;align-items:center;gap:12px;padding:12px 24px;border-bottom:1px solid #555';
    const label = document.createElement('label');
    label.textContent = 'Запуски FEDOT в этой сессии: ';
    const picker = document.createElement('select');
    picker.setAttribute('aria-label', 'Запуск FEDOT');
    picker.style.maxWidth = '65vw';
    const note = document.createElement('span');
    note.setAttribute('role', 'status');
    const link = document.createElement('a');
    link.textContent = location.pathname.startsWith('/fedot-demo') ? 'Трейсы этого запуска' : 'Граф этого запуска';
    link.hidden = true;
    const followLabel = document.createElement('label');
    const follow = document.createElement('input');
    follow.type = 'checkbox'; follow.checked = !selected;
    followLabel.append(follow, document.createTextNode(' Следить за новыми запусками'));
    label.append(picker); host.append(label, followLabel, link, note);
    if (!user || !session) {
      picker.disabled = true;
      note.textContent = 'Откройте FEDOT из нужной сессии CoScientist. Общая история всех сессий недоступна.';
      onReset(null);
      return {refresh: async () => {}};
    }
    const base = `/api/users/${encodeURIComponent(user)}/sessions/${encodeURIComponent(session)}/fedot/runs`;
    function select(runId) {
      stream?.close(); stream = null;
      selected = runId; seq = 0;
      const meta = runs.find(r => r.run_id === runId);
      onReset(meta || null);
      if (!meta) return;
      picker.value = runId;
      query.set('run_id', runId);
      history.replaceState(null, '', location.pathname + '?' + query);
      link.href = (location.pathname.startsWith('/fedot-demo') ? '/fedot-trace' : '/fedot-demo/') + '?' + query;
      link.hidden = false;
      note.textContent = `${runs.length} · ${meta.status}${meta.imported ? ' · импорт' : ''}`;
      const ownStream = new EventSource('/api/fedot-live-stream?' + query);
      stream = ownStream;
      ownStream.onmessage = event => {
        if (stream !== ownStream) return;
        let data;
        try { data = JSON.parse(event.data); } catch { return; }
        if (data.run_id !== selected || data.user_id !== user || data.session_id !== session || data.seq <= seq) return;
        seq = data.seq;
        onEvent(data);
        if (data.type === 'run_end') { ownStream.close(); onStatus(data); }
      };
      ownStream.addEventListener('complete', () => {
        ownStream.close();
        if (stream === ownStream) onStatus(meta);
      });
      ownStream.onerror = () => {
        if (stream === ownStream) note.textContent = 'Связь прервана; восстановление журнала…';
      };
    }
    picker.addEventListener('change', () => { follow.checked = false; select(picker.value); });
    follow.addEventListener('change', () => { if (follow.checked && runs.length) select(runs[0].run_id); });
    async function refresh() {
      if (loading || stopped) return;
      loading = true;
      try {
        const response = await fetch(base, {cache: 'no-store'});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (stopped) return;
        runs = data.runs;
        picker.replaceChildren();
        for (const run of runs) {
          const option = document.createElement('option');
          option.value = run.run_id;
          option.textContent = `${new Date(run.started_at * 1000).toLocaleString()} · ${run.status} · ${run.run_id.slice(0, 8)} · ${(run.task || '').slice(0, 90)}`;
          picker.append(option);
        }
        picker.disabled = !runs.length;
        if (!runs.length) {
          stream?.close(); stream = null; link.hidden = true;
          note.textContent = 'В этой сессии ещё нет сохранённых запусков FEDOT.';
          onReset(null);
        } else if (!selected || (follow.checked && selected !== runs[0].run_id)) {
          select(runs[0].run_id);
        } else if (!runs.some(r => r.run_id === selected)) {
          // A foreign/invalid deep link must never silently show a different run.
          stream?.close(); stream = null; link.hidden = true;
          picker.value = ''; onReset(null);
          note.textContent = 'Указанного запуска нет в этой сессии. Выберите запуск из списка.';
        } else if (!stream) {
          select(selected);
        } else {
          picker.value = selected;
          const meta = runs.find(r => r.run_id === selected);
          note.textContent = `${runs.length} запусков · ${meta.status}${meta.imported ? ' · импорт' : ''}`;
        }
      } catch (error) {
        note.textContent = `Не удалось загрузить историю: ${error.message}`;
      } finally { loading = false; }
    }
    refresh();
    timer = setInterval(refresh, 3000);
    window.addEventListener('pagehide', () => { stopped = true; clearInterval(timer); stream?.close(); });
    return {refresh};
  };
})();
