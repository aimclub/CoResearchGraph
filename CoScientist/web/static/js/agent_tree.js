(() => {
  'use strict';
  const params = new URLSearchParams(location.search);
  const userId = params.get('user_id'), sessionId = params.get('session_id');
  const api = userId && sessionId ? `/api/users/${encodeURIComponent(userId)}/sessions/${encodeURIComponent(sessionId)}` : '';
  const $ = id => document.getElementById(id), svg = $('network'), panel = $('details');
  const localized = v => v && typeof v === 'object' ? v.ru || 'Описание пока не добавлено.' : String(v || '');
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  function el(tag, attrs = {}, content) {
    const result = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs).forEach(([k, v]) => result.setAttribute(k, String(v)));
    if (content !== undefined) result.textContent = content;
    return result;
  }
  const icons = {
    agent: ['M5 19v-2a7 7 0 0114 0v2', 'M12 11a4 4 0 100-8 4 4 0 000 8'],
    coord: ['M12 3v6M5 15V9h14v6', 'M2 15h6v6H2zM9 1h6v4H9zM16 15h6v6h-6z'],
    plan: ['M5 3h14v18H5zM8 8h8M8 12h8M8 16h5'],
    context: ['M4 4h16v12H9l-5 4zM8 8h8M8 12h5'],
    document: ['M5 2h10l4 4v16H5zM14 2v6h5M8 12h8M8 16h8'],
    report: ['M5 2h10l4 4v16H5zM8 17v-3M12 17V9M16 17v-6'],
    search: ['M16 16l5 5', 'M10 17a7 7 0 110-14 7 7 0 010 14'],
    idea: ['M9 18h6M9 21h6M8 14a7 7 0 118 0l-1 2H9z'],
    execute: ['M7 3l14 9-14 9z'],
    experiment: ['M9 2h6M10 2v7L4 19q-1 3 3 3h10q4 0 3-3L14 9V2M7 16h10'],
    review: ['M4 3h16v18H4zM8 12l3 3 6-7'],
    tools: ['M14 3a6 6 0 00-6 8L2 17l5 5 6-6a6 6 0 008-7l-5 3-4-4z'],
    ml: ['M4 4h6v6H4zM14 14h6v6h-6zM7 10v7h7M10 7h7v7'],
    code: ['M8 5l-6 7 6 7M16 5l6 7-6 7M14 2l-4 20'],
    data: ['M4 5c0-4 16-4 16 0s-16 4-16 0v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0'],
    medical: ['M8 2h8v6h6v8h-6v6H8v-6H2V8h6z'],
    economics: ['M3 21h18M6 17v-5M12 17V7M18 17V2'],
    settings: ['M3 6h18M3 12h18M3 18h18M8 3v6M16 9v6M9 15v6'],
  };
  let current = null, geometry = null, nodes = new Map(), chosen = null, panelSignature = '';
  let layoutGeneration = 0, fetchGeneration = 0, signature = '', pendingSignature = '';
  let toolGeneration = 0, toolsAbort = null, modalFocus = null, timer = null, destroyed = false;
  let engine = null;
  const camera = { x: 0, y: -24, scale: 1 };
  const designs = AgentTreeVariants.designs, designCameras = new Map();
  let design = 'workshop', renderedDesign = 'classic';
  let minimumRevision = 0;
  const agentMenu = AgentTreeMenu({ api,
    onSaving() { clearTimeout(timer); fetchGeneration++; },
    async onSaved(revision) { if (revision !== null) minimumRevision = Math.max(minimumRevision, revision); await refresh(); },
  });
  try { design = params.get('design') || localStorage.getItem('mas-graph-design') || 'workshop'; } catch (_) { /* Private browsing may disable storage. */ }
  if (!Object.hasOwn(designs, design)) design = 'workshop';
  function designControls() {
    document.querySelectorAll('.design-options button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.design === design)));
    $('design-description').textContent = designs[design].description;
  }
  designControls();
  try { if (typeof window.ELK === 'function') engine = new window.ELK(); } catch (_) { /* ordered fallback */ }
  function error(message) { $('error').textContent = message; $('error').classList.toggle('hidden', !message); }
  function header(data) {
    const modes = { planner: 'Предварительное планирование', init: 'Предварительное планирование', orchestrator: 'Планирование по задаче', orchestrator_planner: 'Планирует координатор' };
    const profiles = { hypotheses: 'Исследование гипотез', experiments: 'Вычислительные эксперименты', system: 'Базовая система', microfluidics: 'Микрофлюидный синтез' };
    $('context').textContent = `${profiles[data.profile] || 'Исследование'} · ${modes[data.startMode] || 'Настройки профиля'}`;
    $('revision').textContent = `Настройки ${data.desiredRevision} · Запуск ${data.activeRevision ?? 'ещё не начат'}`;
    $('state').textContent = data.pending ? 'Изменения — со следующего запроса' : data.running ? 'Запрос выполняется' : 'Конфигурация актуальна';
    $('state').className = `state ${data.pending ? 'pending' : 'ready'}`;
  }
  function applyCamera() {
    const r = svg.getBoundingClientRect();
    if (!r.width || !r.height) return;
    svg.setAttribute('viewBox', `${camera.x} ${camera.y} ${r.width / camera.scale} ${r.height / camera.scale}`);
    svg.dataset.scale = String(camera.scale); $('zoom-value').textContent = `${Math.round(camera.scale * 100)}%`;
  }
  function resetCamera() {
    camera.scale = 1; camera.x = geometry ? (geometry.w - svg.clientWidth) / 2 : 0; camera.y = -24; applyCamera();
  }
  function zoom(factor, x = svg.clientWidth / 2, y = svg.clientHeight / 2) {
    const old = camera.scale;
    camera.scale = Math.max(.18, Math.min(2.4, old * factor));
    camera.x += x / old - x / camera.scale; camera.y += y / old - y / camera.scale; applyCamera();
  }
  function fit() {
    if (!geometry) return;
    camera.scale = Math.max(.18, Math.min(1.6, (svg.clientWidth - 48) / geometry.w, (svg.clientHeight - 72) / geometry.h));
    camera.x = (geometry.w - svg.clientWidth / camera.scale) / 2;
    camera.y = (geometry.h - svg.clientHeight / camera.scale) / 2; applyCamera();
  }
  const measure = document.createElement('canvas').getContext('2d');
  function titleLines(title, width = 157, fontSize = 16, family = '"Source Sans 3", "Segoe UI", sans-serif') {
    measure.font = `600 ${fontSize}px ${family}`;
    const lines = [''];
    for (const word of title.split(/\s+/)) {
      const i = lines.length - 1, text = lines[i] ? `${lines[i]} ${word}` : word;
      if (measure.measureText(text).width <= width || !lines[i]) lines[i] = text; else lines.push(word);
    }
    if (lines.length > 3) lines.splice(2, lines.length - 2, lines.slice(2).join(' '));
    return lines.map(line => {
      if (measure.measureText(line).width <= width) return line;
      while (line.length && measure.measureText(line + '…').width > width) line = line.slice(0, -1);
      return line + '…';
    });
  }
  function drawCard(card) {
    const node = nodes.get(card.agentId), title = localized(node?.title) || 'Агент исследования';
    const g = el('g', { class: `agent-node role-${node?.icon || 'agent'}`, transform: `translate(${card.x},${card.y})`,
      'data-agent': card.agentId, 'data-view-id': card.id, 'data-x': card.x, 'data-y': card.y,
      tabindex: 0, role: 'button', 'aria-label': title, 'aria-pressed': 'false' });
    if (renderedDesign !== 'classic') return drawAlternativeCard(g, card, node, title);
    g.append(el('title', {}, title), el('rect', { class: 'node-card', width: card.w, height: card.h, rx: 12 }),
      el('rect', { class: 'node-icon-bg', x: 12, y: 14, width: 32, height: 32, rx: 9 }));
    const icon = el('g', { class: 'node-icon', transform: 'translate(17,19) scale(.9)', 'aria-hidden': true });
    (icons[node?.icon] || icons.agent).forEach(d => icon.append(el('path', { d }))); g.append(icon);
    const lines = titleLines(title), y = lines.length === 3 ? 21 : lines.length === 2 ? 28 : 37;
    lines.forEach((line, i) => g.append(el('text', { class: 'node-title', x: 54, y: y + i * 18 }, line)));
    const badge = card.agentId === '__mas_session__' ? 'Настройки исследования' : card.agentId === 'NirReportAgent' ? 'По запросу' : 'Подключён';
    g.append(el('text', { class: 'node-badge', x: 54, y: 77 }, badge)); return g;
  }
  const roleNames = { coord: 'Координация', plan: 'Планирование', context: 'Подготовка', document: 'Документы',
    report: 'Отчёт', search: 'Поиск знаний', idea: 'Гипотезы', execute: 'Выполнение', experiment: 'Эксперимент',
    review: 'Проверка', tools: 'Инструменты', ml: 'Машинное обучение', code: 'Разработка', data: 'Данные',
    medical: 'Экспертиза', economics: 'Экономика', settings: 'Настройки' };
  function drawAlternativeCard(g, card, node, title) {
    const role = roleNames[node?.icon] || 'Агент', mode = renderedDesign;
    const circle = mode === 'constellation', blueprint = mode === 'blueprint', atlas = mode === 'atlas';
    g.dataset.shape = circle ? 'circle' : 'rect'; g.dataset.width = card.w; g.dataset.height = card.h;
    g.append(el('title', {}, title));
    if (mode === 'workshop') return AgentTreeWorkshop.draw({ el, g, card, node, title, icons, titleLines });
    if (mode === 'architecture') return drawArchitectureCard(g, card, node, title, role);
    if (circle) {
      const r = card.w / 2;
      g.append(el('circle', { class: 'node-card', cx: r, cy: r, r }),
        el('path', { class: 'node-orbit', d: `M${r * .4},${r * .28} A${r - 7},${r - 7} 0 0 1 ${r * 1.72},${r * .6}` }));
    } else {
      g.append(el('rect', { class: 'node-card', width: card.w, height: card.h, rx: blueprint ? 3 : atlas ? 22 : 14 }));
      if (blueprint) g.append(el('path', { class: 'node-rule', d: `M0,28 H${card.w}` }));
      else if (!atlas) g.append(el('rect', { class: 'node-accent', x: 0, y: 16, width: 4, height: card.h - 32, rx: 2 }),
        el('path', { class: 'node-rule', d: `M14,${card.h - 28} H${card.w - 14}` }));
    }
    const iconX = circle ? card.w / 2 - 12 : atlas ? 22 : blueprint ? 12 : 18;
    const iconY = circle ? 23 : atlas ? 33 : blueprint ? 39 : 20;
    if (!blueprint) g.append(el('rect', { class: 'node-icon-bg', x: iconX - 6, y: iconY - 6, width: 36, height: 36, rx: atlas || circle ? 18 : 11 }));
    const icon = el('g', { class: 'node-icon', transform: `translate(${iconX},${iconY})`, 'aria-hidden': true });
    (icons[node?.icon] || icons.agent).forEach(d => icon.append(el('path', { d }))); g.append(icon);
    const size = circle ? 14 : blueprint ? 15 : 16;
    const textX = circle ? card.w / 2 : atlas ? 72 : blueprint ? 46 : 62;
    const lines = titleLines(title, circle ? card.w - 26 : card.w - textX - 14, size);
    const y = circle ? 84 - (lines.length - 1) * 8 : blueprint ? 42 : atlas ? 46 - (lines.length - 1) * 9 : 30;
    lines.forEach((line, i) => g.append(el('text', { class: 'node-title', x: textX, y: y + i * (circle || blueprint ? 16 : 18),
      'text-anchor': circle ? 'middle' : 'start', style: `font-size:${size}px` }, line)));
    const labelX = circle ? card.w / 2 : atlas ? 72 : 14;
    const labelY = circle ? 128 : blueprint ? 19 : card.h - 11;
    g.append(el('text', { class: 'node-role', x: labelX, y: labelY, 'text-anchor': circle ? 'middle' : 'start' }, role));
    const badge = card.agentId === '__mas_session__' ? 'Для сессии' : card.agentId === 'NirReportAgent' ? 'По запросу' : 'Подключён';
    if (circle) g.append(el('circle', { class: 'node-status-dot', cx: card.w / 2, cy: 144, r: 3 }));
    else if (blueprint) g.append(el('text', { class: 'node-badge', x: 46, y: card.h - 9 }, badge));
    else g.append(el('circle', { class: 'node-status-dot', cx: card.w - 16, cy: atlas ? 17 : card.h - 15, r: 3.5 }));
    return g;
  }
  function drawArchitectureCard(g, card, node, title, role) {
    const w = card.w, h = card.h;
    g.append(el('rect', { class: 'node-card', width: w, height: h, rx: 0 }),
      el('path', { class: 'plan-wall', d: `M6,24 V6 H${w - 6} V${h - 6} H6 V${h - 30} M0,28 H${w} M52,28 V${h} M0,${h - 24} H52` }),
      el('path', { class: 'plan-register', d: `M-5,0 H9 M0,-5 V9 M${w - 9},${h} H${w + 5} M${w},${h - 9} V${h + 5}` }));
    const icon = el('g', { class: 'node-icon', transform: 'translate(14,44)', 'aria-hidden': true });
    (icons[node?.icon] || icons.agent).forEach(d => icon.append(el('path', { d }))); g.append(icon);
    g.append(el('text', { class: 'node-role', x: 13, y: 19 }, role),
      el('text', { class: 'plan-number', x: 26, y: h - 9, 'text-anchor': 'middle' },
        String(geometry.nodes.findIndex(n => n.id === card.id) + 1).padStart(2, '0')));
    const lines = titleLines(title, w - 76, 17), y = 62 - (lines.length - 1) * 9;
    lines.forEach((line, i) => g.append(el('text', { class: 'node-title', x: 65, y: y + i * 19, style: 'font-size:17px' }, line)));
    g.append(el('text', { class: 'node-badge', x: 65, y: h - 12 },
      card.agentId === '__mas_session__' ? 'СОСТАВ СЕССИИ' : card.agentId === 'NirReportAgent' ? 'ПО ЗАПРОСУ' : 'АГЕНТ ПОДКЛЮЧЁН'));
    return g;
  }
  function renderScene() {
    const defs = el('defs');
    if (renderedDesign === 'workshop') defs.append(...AgentTreeWorkshop.definitions(el));
    ['flow', 'delegate', 'parallel', 'choice', 'loop', 'settings'].forEach(kind => {
      const marker = el('marker', { id: `arrow-${kind}`, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto' });
      marker.append(el('path', { d: 'M0 0L10 5L0 10z', class: `arrow-${kind}` })); defs.append(marker);
    });
    const groups = el('g', { class: 'groups' }), edges = el('g', { class: 'edges' }), cards = el('g', { class: 'cards' });
    geometry.groups.forEach(group => {
      const g = el('g', { class: `workflow-group ${group.kind}`, 'data-block-id': group.id });
      g.append(el('rect', { x: group.x, y: group.y, width: group.w, height: group.h, rx: renderedDesign === 'architecture' ? 0 : 16 }));
      const label = group.labelAgentId ? localized(nodes.get(group.labelAgentId)?.title)
        : { delegates: 'Вызываются по задаче', choice: 'Один из вариантов исполнения', parallel: 'Параллельная работа', loop: 'Повторение последовательности' }[group.kind];
      if (label) g.append(el('text', { x: group.x + 22, y: group.y + 20 }, label)); groups.append(g);
    });
    geometry.edges.forEach(edge => {
      const p = el('path', { d: (edge.straight ? AgentTreeVariants.path : AgentTreeLayout.path)(edge.points), class: `edge ${edge.relation}`, 'data-from': edge.from || '', 'data-to': edge.to || '' });
      if (edge.arrow) p.setAttribute('marker-end', `url(#arrow-${edge.relation})`);
      p.append(el('title', {}, ({ settings: 'Настройки сессии', delegate: 'Может вызвать', choice: 'Вариант исполнения', loop: 'Повторение', parallel: 'Параллельная работа' })[edge.relation] || 'Порядок выполнения')); edges.append(p);
      if (edge.relation === 'settings' && renderedDesign === 'classic') {
        const a = edge.points[0], b = edge.points[edge.points.length - 1];
        edges.append(el('text', { class: 'edge-label', x: (a.x + b.x) / 2, y: a.y - 52, 'text-anchor': 'middle' }, 'Настройки сессии'));
      }
    });
    geometry.nodes.forEach(card => cards.append(drawCard(card)));
    if (renderedDesign !== 'classic') {
      const ports = new Map();
      geometry.edges.forEach(edge => {
        [[edge.from, edge.points[0]], [edge.to, edge.points[edge.points.length - 1]]].forEach(([id, p]) => {
          ports.set(`${id}:${p.x.toFixed(2)},${p.y.toFixed(2)}`, p);
        });
      });
      ports.forEach(p => cards.append(renderedDesign === 'workshop' ? AgentTreeWorkshop.port(el, p)
        : el('circle', { class: 'connection-port', cx: p.x, cy: p.y, r: 3.2, 'aria-hidden': true })));
    }
    svg.replaceChildren(defs, groups, edges, cards); svg.dataset.design = renderedDesign;
    svg.dataset.layout = geometry.variant ? 'compact' : geometry.fallback ? 'fallback' : 'elk'; highlight();
  }
  function highlight() {
    svg.querySelectorAll('.agent-node').forEach(n => {
      const selected = n.dataset.viewId === chosen;
      n.classList.toggle('selected', selected); n.setAttribute('aria-pressed', String(selected));
    });
    svg.querySelectorAll('.edge').forEach(e => e.classList.toggle('selected', !!chosen && (e.dataset.from === chosen || e.dataset.to === chosen)));
  }
  const selectedCard = () => geometry?.nodes.find(card => card.id === chosen);
  function closeDetails() { panel.classList.remove('open'); document.querySelector('main').classList.add('panel-closed'); }
  function details() {
    const card = selectedCard(), node = card && nodes.get(card.agentId);
    if (!node) return;
    const configuration = card.agentId === '__mas_session__';
    const related = configuration ? [] : [...new Set(current.edges.filter(e => e.from === node.id).map(e => e.to))]
      .map(id => nodes.get(id)).filter(n => n && n.selected !== false && n.kind !== 'system');
    const next = JSON.stringify([card.id, node, related, configuration ? [current.desiredRevision, current.activeRevision] : null]);
    if (panelSignature === next) return;
    panelSignature = next;
    const location = configuration ? 'Параметры выбранной сессии' : node.stage === 'pre' ? 'Подготовка исследования' : node.stage === 'post' ? 'Завершение исследования' : card.id.endsWith('PlanningPipelineAgent/child/PlannerAgent') ? 'Начальное планирование' : card.id.includes('ExperimentModuleAgent') ? 'Вычислительный эксперимент' : 'Работа по задаче';
    panel.innerHTML = `<button id="details-close" class="icon panel-close" aria-label="Закрыть описание">×</button>
      <p class="eyebrow">${esc(location)}</p><h2>${esc(localized(node.title))}</h2>
      <p class="description">${esc(localized(node.descriptionLocalized))}</p>
      <div class="connection-state">${configuration ? 'Применение со следующего запроса' : 'Подключён к сессии'}</div>
      ${related.length ? `<h3>Связанные агенты</h3><ul class="related">${related.map(n => `<li>${esc(localized(n.title))}</li>`).join('')}</ul>` : ''}
      ${configuration ? `<dl class="meta"><dt>Настройки</dt><dd>Версия ${current.desiredRevision}</dd><dt>Текущий запуск</dt><dd>${current.activeRevision ?? 'Ещё не начат'}</dd></dl>` : '<button class="tool-button" id="open-tools">Инструменты</button>'}
      <details class="technical"><summary>Технические сведения</summary><dl class="meta"><dt>Идентификатор</dt><dd>${esc(configuration ? sessionId : node.name)}</dd>
      ${configuration ? '' : `<dt>Класс</dt><dd>${esc(node.class)}</dd><dt>Наборы инструментов</dt><dd>${esc(node.toolKeys.join(', ') || 'Нет')}</dd>`}</dl></details>`;
    panel.classList.remove('empty'); $('details-close').onclick = closeDetails;
    if (!configuration) $('open-tools').onclick = () => openTools(node);
  }
  function select(id) {
    if (chosen !== id) closeTools(false);
    chosen = id; document.querySelector('main').classList.remove('panel-closed'); panel.classList.add('open'); details(); highlight();
  }
  function closeTools(restore = true) {
    toolGeneration++; toolsAbort?.abort(); toolsAbort = null;
    const wasOpen = !$('tools-modal').classList.contains('hidden'); $('tools-modal').classList.add('hidden');
    if (wasOpen && restore && modalFocus?.isConnected) modalFocus.focus();
  }
  async function openTools(node) {
    closeTools(false); const request = ++toolGeneration; toolsAbort = new AbortController(); modalFocus = document.activeElement;
    $('tools-modal').classList.remove('hidden'); $('tools-close').focus(); $('tools-title').textContent = localized(node.title);
    $('tools-note').textContent = 'Загрузка инструментов…'; $('tools-list').replaceChildren();
    try {
      const response = await fetch(`${api}/agents/${encodeURIComponent(node.name)}/tools`, { cache: 'no-store', signal: toolsAbort.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (request !== toolGeneration || selectedCard()?.agentId !== node.id) return;
      $('tools-note').textContent = data.dynamic && !data.selectionReady ? 'MCP-инструменты будут определены при подготовке задачи. FEDOT AutoML закреплён первым.'
        : `Инструментов: ${data.tools.length}${data.dynamic ? ' · Выбор MCP относится к последней подготовленной задаче' : ''}${data.catalog?.stale ? ' · Сведения о доступности требуют обновления' : ''}`;
      $('tools-list').innerHTML = data.tools.length ? data.tools.map(tool => {
        const status = { available: 'Доступен', saved: 'Доступность не подтверждена', unavailable: 'Недоступен' }[tool.status] || 'Доступность не подтверждена';
        return `<article class="tool"><h3>${esc(localized(tool.display_name))}</h3><div class="badges">
          ${tool.pinned ? '<span class="badge pinned">FEDOT AutoML · первый</span>' : ''}<span class="badge ${tool.status === 'available' ? '' : 'off'}">${status}</span>
          <span class="badge neutral">${tool.selected ? (tool.kind === 'local' ? 'Подключён' : 'Выбран для задачи') : 'Не выбран для задачи'}</span></div>
          <p>${esc(localized(tool.summary))}</p><details class="technical"><summary>Техническое имя</summary><code>${esc(tool.name)}</code></details></article>`;
      }).join('') : '<p class="muted">У агента нет отображаемых инструментов.</p>';
    } catch (err) {
      if (request !== toolGeneration || err.name === 'AbortError') return;
      $('tools-note').textContent = 'Не удалось загрузить инструменты. Закройте окно и повторите попытку.';
    }
  }
  async function draw(data) {
    const width = svg.clientWidth || 1100, requestedDesign = design, bucket = design === 'classic' ? (width >= 900 ? 4 : 3) : 0;
    const next = JSON.stringify([data.nodes, data.workflow, bucket, requestedDesign]);
    if (next === signature) {
      // A resize/configuration may return to the committed state while another
      // layout is still running. That older result must not replace this scene.
      if (pendingSignature && pendingSignature !== next) { layoutGeneration++; pendingSignature = ''; svg.setAttribute('aria-busy', 'false'); }
      return;
    }
    if (next === pendingSignature) return;
    const generation = ++layoutGeneration; pendingSignature = next; svg.setAttribute('aria-busy', 'true');
    try {
      const result = requestedDesign === 'classic' ? await AgentTreeLayout.layout(data.workflow, { width, engine })
        : AgentTreeVariants.layout(data.workflow, requestedDesign);
      if (generation !== layoutGeneration || destroyed) return;
      const initial = !geometry, switched = renderedDesign !== requestedDesign, previous = selectedCard(); geometry = result; signature = next;
      renderedDesign = requestedDesign; document.body.dataset.design = renderedDesign;
      nodes = new Map(data.nodes.map(node => [node.id, node]));
      if (chosen && !geometry.nodes.some(n => n.id === chosen)) {
        chosen = geometry.nodes.find(n => n.agentId === previous?.agentId)?.id || null; closeTools(false);
        if (!chosen) { panelSignature = ''; panel.innerHTML = '<div class="empty-copy">Агент больше не подключён. Выберите другого участника на схеме.</div>'; closeDetails(); }
      }
      renderScene();
      agentMenu.update(data, renderedDesign);
      if (switched) {
        const saved = designCameras.get(renderedDesign);
        if (saved) { Object.assign(camera, saved); applyCamera(); } else fit();
      } else if (initial) {
        // A desktop opens with the whole wide workflow visible; compact screens
        // retain readable 100% cards and use pan navigation.
        if (svg.clientWidth >= 1000) fit(); else resetCamera();
      } else applyCamera();
      if (chosen) details();
      $('layout-note').textContent = result.fallback ? 'Упрощённое размещение · порядок работы сохранён' : '';
    } finally { if (generation === layoutGeneration) { pendingSignature = ''; svg.setAttribute('aria-busy', 'false'); } }
  }
  async function refresh() {
    clearTimeout(timer);
    if (!api) { error('Откройте конфигурацию МАС из выбранной сессии.'); return; }
    const request = ++fetchGeneration;
    try {
      const response = await fetch(`${api}/agent-tree`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json(); if (request !== fetchGeneration || destroyed) return;
      if (data.desiredRevision < minimumRevision) return;
      if (!data.workflow) throw new Error('Обновите сервер для отображения схемы процесса.');
      current = data; header(data); await draw(data);
      if (request === fetchGeneration) { error(''); agentMenu.update(data, renderedDesign); if (chosen) details(); }
    } catch (err) {
      if (request !== fetchGeneration || destroyed) return;
      error(`Не удалось обновить схему. ${geometry ? 'Показана последняя загруженная конфигурация. ' : ''}${err.message}`);
      $('state').textContent = 'Ошибка обновления'; $('state').className = 'state pending';
    } finally { if (!destroyed && request === fetchGeneration) timer = setTimeout(refresh, 3000); }
  }
  let drag = null, moved = false;
  svg.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    moved = false; drag = { x: e.clientX, y: e.clientY, cx: camera.x, cy: camera.y };
  });
  svg.addEventListener('pointermove', e => {
    if (!drag) return;
    if (!moved && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > 4) { moved = true; svg.setPointerCapture(e.pointerId); }
    if (moved) { camera.x = drag.cx - (e.clientX - drag.x) / camera.scale; camera.y = drag.cy - (e.clientY - drag.y) / camera.scale; applyCamera(); }
  });
  svg.addEventListener('pointerup', e => { drag = null; if (svg.hasPointerCapture(e.pointerId)) svg.releasePointerCapture(e.pointerId); });
  svg.addEventListener('pointercancel', () => { drag = null; moved = false; });
  svg.addEventListener('click', e => { if (moved) { moved = false; return; } const card = e.target.closest('.agent-node'); if (card) select(card.dataset.viewId); });
  svg.addEventListener('keydown', e => {
    const card = e.target.closest('.agent-node');
    if (card && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); select(card.dataset.viewId); }
    else if (['ArrowDown', 'ArrowUp', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
      e.preventDefault();
      if (e.key === 'ArrowDown') camera.y += 80 / camera.scale; if (e.key === 'ArrowUp') camera.y -= 80 / camera.scale;
      if (e.key === 'ArrowLeft') camera.x -= 80 / camera.scale; if (e.key === 'ArrowRight') camera.x += 80 / camera.scale; applyCamera();
    }
  });
  svg.addEventListener('wheel', e => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) { const r = svg.getBoundingClientRect(); zoom(Math.exp(-e.deltaY * .002), e.clientX - r.left, e.clientY - r.top); }
    else { camera.x += (e.shiftKey ? e.deltaY : e.deltaX) / camera.scale; camera.y += (e.shiftKey ? 0 : e.deltaY) / camera.scale; applyCamera(); }
  }, { passive: false });
  $('zoom-in').onclick = () => zoom(1.2); $('zoom-out').onclick = () => zoom(1 / 1.2);
  $('zoom-reset').onclick = resetCamera; $('fit').onclick = fit; $('tools-close').onclick = () => closeTools();
  document.querySelectorAll('.design-options button').forEach(button => {
    button.onclick = async () => {
      if (design === button.dataset.design) return;
      if (geometry) designCameras.set(renderedDesign, { ...camera });
      design = button.dataset.design; designControls();
      try {
        if (current) await draw(current);
        if (renderedDesign !== design) return;
        try { localStorage.setItem('mas-graph-design', design); } catch (_) { /* The switch also works without storage. */ }
        const url = new URL(location.href); url.searchParams.set('design', design); history.replaceState(null, '', url);
        error('');
      } catch (_) { error('Не удалось переключить оформление. Показан предыдущий вариант.'); }
    };
  });
  $('tools-modal').onclick = e => { if (e.target === $('tools-modal')) closeTools(); };
  document.addEventListener('keydown', e => {
    const modal = $('tools-modal');
    if (e.key === 'Escape') { if (!modal.classList.contains('hidden')) closeTools(); else closeDetails(); }
    if (e.key === 'Tab' && !modal.classList.contains('hidden')) {
      const focusable = [...modal.querySelectorAll('button, summary, a[href], [tabindex="0"]')];
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  });
  const observer = new ResizeObserver(() => {
    if (!svg.clientWidth || !svg.clientHeight) return;
    applyCamera(); if (current) draw(current).catch(() => error('Не удалось изменить размещение схемы.'));
  });
  observer.observe(document.querySelector('.canvas-wrap'));
  observer.observe(svg);
  window.addEventListener('pagehide', () => { destroyed = true; clearTimeout(timer); observer.disconnect(); toolsAbort?.abort(); engine?.terminateWorker?.(); agentMenu.dispose(); });
  window.addEventListener('pageshow', e => {
    if (!e.persisted) return;
    destroyed = false; pendingSignature = ''; layoutGeneration++; agentMenu.resume();
    try { engine = typeof window.ELK === 'function' ? new window.ELK() : null; } catch (_) { engine = null; }
    observer.observe(document.querySelector('.canvas-wrap')); observer.observe(svg); refresh();
  });
  (document.fonts?.ready || Promise.resolve()).then(refresh);
})();
