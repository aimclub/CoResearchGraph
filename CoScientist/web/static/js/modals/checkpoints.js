// =========================================================================
// Automatic pipeline checkpoints: list, restore, and continue.
// =========================================================================
(function () {
  'use strict';

  let known = [];
  let loading = false;

  function modal() { return document.getElementById('checkpoints-modal'); }
  function listEl() { return document.getElementById('checkpoints-list'); }
  function statusEl() { return document.getElementById('checkpoints-status'); }

  function setCount(value) {
    const badge = document.getElementById('checkpoints-count');
    if (!badge) return;
    const count = Math.max(0, Number(value) || 0);
    badge.textContent = String(count);
    badge.classList.toggle('hidden', count === 0);
    badge.classList.toggle('flex', count > 0);
  }

  function setStatus(message, kind) {
    const el = statusEl();
    if (!el) return;
    if (!message) {
      el.classList.add('hidden');
      return;
    }
    el.textContent = message;
    el.className = 'mx-5 mt-4 rounded-md border px-3 py-2 text-[11px] ' + (
      kind === 'error'
        ? 'border-error/30 bg-error/10 text-error'
        : 'border-primary/30 bg-primary/10 text-on-surface'
    );
  }

  function when(value) {
    const date = new Date(value || '');
    if (!Number.isFinite(date.getTime())) return '';
    return date.toLocaleString(currentLang === 'ru' ? 'ru-RU' : 'en-GB');
  }

  function render() {
    const el = listEl();
    if (!el) return;
    setCount(known.length);
    if (!known.length) {
      el.innerHTML = `
        <div class="h-36 flex flex-col items-center justify-center gap-2 text-outline-variant">
          <span class="material-symbols-outlined text-3xl opacity-50">history_toggle_off</span>
          <p class="text-xs">Контрольных точек пока нет.</p>
          <p class="text-[10px] opacity-70">Они появятся автоматически перед началом каждой стадии.</p>
        </div>`;
      return;
    }
    el.innerHTML = known.map(cp => {
      const position = Number(cp.stage_index) + 1;
      const total = Number(cp.stage_count) || '?';
      return `
        <div class="rounded-lg border border-outline-variant/15 bg-surface-container-low p-4 flex items-center gap-4">
          <div class="w-10 h-10 shrink-0 rounded-full bg-primary/10 border border-primary/25 flex items-center justify-center text-primary font-mono text-xs font-bold">
            ${position}/${total}
          </div>
          <div class="min-w-0 flex-1">
            <div class="text-xs font-bold text-on-surface truncate">${escHtml(cp.title || cp.agent || 'Stage')}</div>
            <div class="mt-1 text-[9px] font-mono text-outline-variant truncate">${escHtml(cp.agent || '')}</div>
            <div class="mt-1 text-[9px] text-outline-variant/70">${escHtml(when(cp.created_at))} · run ${escHtml(cp.run_version)}</div>
          </div>
          <button type="button" data-checkpoint-id="${escHtml(cp.id)}"
            class="checkpoint-restore shrink-0 px-3 py-2 rounded-md border border-primary/30 bg-primary/10 text-primary text-[10px] font-bold uppercase tracking-wide hover:bg-primary/20 transition-colors">
            Откатить и продолжить
          </button>
        </div>`;
    }).join('');
    el.querySelectorAll('.checkpoint-restore').forEach(button => {
      button.addEventListener('click', () => restore(button.dataset.checkpointId, button));
    });
  }

  async function refresh() {
    if (loading || !activeUser || !activeSession) return;
    loading = true;
    setStatus('');
    try {
      const data = await apiJson(sessionApi('/checkpoints'));
      known = Array.isArray(data.checkpoints) ? data.checkpoints : [];
      render();
    } catch (error) {
      setStatus(error.message || String(error), 'error');
    } finally {
      loading = false;
    }
  }

  async function restore(id, button) {
    const checkpoint = known.find(item => item.id === id);
    if (!checkpoint) return;
    const title = checkpoint.title || checkpoint.agent || id;
    if (!confirm(`Откатить состояние к стадии «${title}» и продолжить выполнение с неё?`)) return;
    button.disabled = true;
    button.textContent = 'Восстановление…';
    setStatus('Восстанавливаю состояние и запускаю продолжение…');
    try {
      await apiJson(sessionApi(`/checkpoints/${encodeURIComponent(id)}/restore`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ continue: true }),
      });
      setStatus(`Состояние восстановлено: ${title}. Продолжение запущено.`);
      setTimeout(closeCheckpointsModal, 700);
    } catch (error) {
      setStatus(error.message || String(error), 'error');
      button.disabled = false;
      button.textContent = 'Откатить и продолжить';
    }
  }

  function openCheckpointsModal() {
    if (!activeUser || !activeSession) {
      addSystemMsg('Сначала выберите сессию.');
      return;
    }
    modal().classList.remove('hidden');
    listEl().innerHTML = '<div class="text-xs text-outline-variant">Загрузка…</div>';
    refresh();
  }

  function closeCheckpointsModal() {
    const el = modal();
    if (el) el.classList.add('hidden');
  }

  function onCreated(checkpoint) {
    if (!checkpoint || !checkpoint.id) return;
    known = [checkpoint].concat(known.filter(item => item.id !== checkpoint.id));
    setCount(known.length);
    if (modal() && !modal().classList.contains('hidden')) render();
  }

  function onSnapshot(checkpoints) {
    known = Array.isArray(checkpoints) ? checkpoints : [];
    setCount(known.length);
    if (modal() && !modal().classList.contains('hidden')) render();
  }

  window.openCheckpointsModal = openCheckpointsModal;
  window.closeCheckpointsModal = closeCheckpointsModal;
  window.CheckpointsModal = { refresh, onCreated, onSnapshot };
})();
