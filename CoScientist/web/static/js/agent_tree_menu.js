/* Session switches shared by every occurrence of an agent in the drawing. */
(() => {
  'use strict';
  window.AgentTreeMenu = function ({ api, onSaving, onSaved, onChanged = () => {} }) {
    const $ = id => document.getElementById(id), menu = $('agent-menu');
    let catalog = [], graph = null, key = '', loadedKey = '', generation = 0, busy = false, disposed = false, workshopOpened = false, failureMessage = '';
    const locks = { root: 'Обязательный агент', startMode: 'Задаётся режимом запуска', internal: 'Системный агент',
      setting: 'Задаётся профилем', unavailable: 'Недоступен в этой конфигурации' };
    function status(message, failed = false) {
      failureMessage = failed ? message : '';
      $('agent-menu-status').textContent = message;
      $('agent-menu-status').classList.toggle('failed', failed);
      $('agent-menu-retry').classList.toggle('hidden', !failed);
      onChanged();
    }
    const allowed = (agent, enabled) => agent && (enabled ? agent.canEnable : agent.canDisable) === true;
    function render() {
      const connected = new Set((graph?.nodes || []).filter(n => n.selected !== false).map(n => n.id));
      const agents = catalog.filter(a => !a.internal && a.availableInProfile !== false), filter = $('agent-menu-search').value.trim().toLocaleLowerCase('ru');
      $('agent-menu-count').textContent = agents.length ? `${agents.filter(a => connected.has(a.name)).length}` : '';
      const focus = document.activeElement?.dataset.agentToggle, scroll = $('agent-menu-list').scrollTop;
      const rows = agents.filter(a => (a.title?.ru || 'Агент исследования').toLocaleLowerCase('ru').includes(filter)).map(agent => {
        const row = document.createElement('label'); row.className = 'agent-menu-row';
        const copy = document.createElement('span'), title = document.createElement('span'), note = document.createElement('small');
        title.className = 'agent-menu-name'; title.textContent = agent.title?.ru || 'Агент исследования';
        note.textContent = agent.controlReason || locks[agent.lock] || (!agent.effectiveEnabled ? 'Отключён' : connected.has(agent.name) ? 'В схеме' : 'Включён · ветка отключена');
        copy.append(title, note);
        const input = document.createElement('input'); input.type = 'checkbox'; input.role = 'switch';
        input.checked = !!agent.effectiveEnabled; input.disabled = busy || !allowed(agent, !input.checked);
        input.dataset.agentToggle = agent.name; input.setAttribute('aria-label', title.textContent);
        input.addEventListener('change', () => save(agent, input.checked));
        row.append(copy, input); return row;
      });
      if (!rows.length) {
        const empty = document.createElement('p'); empty.className = 'agent-menu-hint';
        empty.textContent = catalog.length ? 'Агенты не найдены.' : 'Загрузка агентов…'; rows.push(empty);
      }
      $('agent-menu-list').replaceChildren(...rows); $('agent-menu-list').scrollTop = scroll;
      if (focus) [...$('agent-menu-list').querySelectorAll('input')].find(n => n.dataset.agentToggle === focus)?.focus({ preventScroll: true });
      menu.setAttribute('aria-busy', String(busy));
      onChanged();
    }
    async function load() {
      if (!api || busy || disposed) return;
      const request = ++generation, requestedKey = key;
      status('Загрузка состава сессии…');
      try {
        const response = await fetch(`${api}/agents/catalog`, { cache: 'no-store' });
        if (!response.ok) throw new Error();
        const data = await response.json();
        if (disposed || request !== generation) return;
        catalog = data.agents; loadedKey = requestedKey; render(); status('Переключатели меняют состав сессии.');
      } catch (_) {
        if (!disposed && request === generation) status('Не удалось загрузить состав. Попробуйте ещё раз.', true);
      }
    }
    async function save(agentOrName, enabled) {
      const agent = typeof agentOrName === 'string' ? catalog.find(a => a.name === agentOrName) : agentOrName;
      if (busy || !allowed(agent, enabled) || disposed) return;
      busy = true; generation++; onSaving(); render(); status('Сохраняем состав сессии…');
      let revision = null;
      try {
        const response = await fetch(`${api}/agents/${encodeURIComponent(agent.name)}/enabled`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled }) });
        if (!response.ok) {
          const failure = await response.json().catch(() => ({}));
          throw new Error(typeof failure.detail === 'string' ? failure.detail : 'Не удалось сохранить изменение.');
        }
        const data = await response.json();
        if (disposed) return;
        catalog = data.agents; revision = data.agentConfiguration.desiredRevision;
        loadedKey = `${graph?.profile}:${revision}`;
        status('Сохранено. Новый состав — со следующего запроса.');
      } catch (failure) {
        const message = /[А-Яа-яЁё]/.test(failure.message) ? failure.message : 'Не удалось сохранить изменение.';
        if (!disposed) status(`${message} Состав не изменён.`, true);
      } finally {
        busy = false;
        if (!disposed) { await onSaved(revision); render(); }
      }
    }
    function setOpen(open) {
      $('agent-menu-toggle').setAttribute('aria-expanded', String(open));
      $('agent-menu-content').classList.toggle('hidden', !open);
      if (open && !catalog.length) load();
    }
    $('agent-menu-toggle').onclick = () => setOpen($('agent-menu-toggle').getAttribute('aria-expanded') !== 'true');
    menu.addEventListener('keydown', event => {
      if (event.key === 'Escape' && !$('agent-menu-content').classList.contains('hidden')) {
        event.stopPropagation(); $('agent-menu-toggle').click(); $('agent-menu-toggle').focus();
      }
    });
    $('agent-menu-search').addEventListener('input', render);
    $('agent-menu-retry').onclick = load;
    return {
      update(data, design) {
        if (disposed) return;
        graph = data; key = `${data.profile}:${data.desiredRevision}`;
        menu.classList.remove('hidden');
        $('agent-menu-title').textContent = 'Состав системы';
        if (!workshopOpened) { workshopOpened = true; if (innerWidth >= 1100) setOpen(true); }
        if (loadedKey !== key && !busy) { loadedKey = key; load(); }
        else if (!busy) render();
      },
      dispose() { disposed = true; generation++; },
      resume() { disposed = false; loadedKey = ''; },
      getAgent(name) { return catalog.find(a => a.name === name); },
      get busy() { return busy; },
      get failureMessage() { return failureMessage; },
      save,
      open() { setOpen(true); },
      close() { setOpen(false); },
    };
  };
})();
