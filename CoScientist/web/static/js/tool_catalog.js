(() => {
  'use strict';

  const LANG_KEY = 'coscientist.lang';
  const POLL = { refreshing: 3000, normal: 30000, hidden: 90000 };
  const CATEGORY_ORDER = ['automl', 'chemistry', 'epidemiology', 'literature', 'data', 'files', 'web', 'reporting', 'other'];
  const COPY = {
    ru: {
      title: 'Каталог инструментов',
      lede: 'Подключённые научные инструменты CoScientist: какие задачи они решают, откуда доступны и когда их список был обновлён.',
      loading: 'Загрузка', ready: 'Актуально', refreshing: 'Обновляется', partial: 'Есть замечания', unavailable: 'Нет данных',
      refresh: 'Обновить сейчас', app: 'К приложению', checked: 'Последняя проверка: {time}', never: 'Проверка ещё не завершалась',
      tools: 'актуальных инструментов', servers: 'серверов отвечают', categories: 'предметных категорий', attention: 'требуют внимания',
      allConnections: 'Все подключения', instruments: 'Инструменты', search: 'Поиск', searchPlaceholder: 'Название, задача или сервер',
      category: 'Категория', serverLabel: 'Сервер', availability: 'Доступность', role: 'Назначение', allCategories: 'Все категории', allServers: 'Все серверы', allAvailability: 'Все инструменты', allRoles: 'Все назначения',
      available: 'Доступные', needsAttention: 'Требуют внимания', scientific: 'Научные', supporting: 'Служебные',
      shown: 'Показано {shown} из {total}', empty: 'По заданным фильтрам инструменты не найдены.', details: 'Технические сведения',
      machineName: 'Машинное имя', server: 'Сервер', source: 'Источник описания', parameters: 'Параметры', unknownSchema: 'Параметры не описаны',
      original: 'Исходное описание', agentLimited: 'Ограниченный доступ', fedot: 'FEDOT', more: 'Ещё операции ({count})',
      registryUnavailable: 'Реестр Tool RAG сейчас недоступен. Показаны сведения подключённых серверов и последний сохранённый снимок.',
      attentionNotice: '{count} подключений требуют внимания: сервер не ответил или его текущий список расходится с индексом.',
      firstLoad: 'Каталог строится. Страница обновится автоматически по мере получения ответов серверов.',
      refreshFailed: 'Не удалось запустить обновление каталога.',
      categoriesMap: { automl: 'AutoML', chemistry: 'Химия', epidemiology: 'Эпидемиология', literature: 'Научная литература', data: 'Данные', files: 'Файлы', web: 'Веб-поиск', reporting: 'Отчёты', other: 'Другое' },
      sourceMap: { live: 'MCP tools/list', registry: 'Tool RAG', snapshot: 'Сохранённый снимок' },
    },
    en: {
      title: 'Tool catalogue',
      lede: 'Scientific tools connected to CoScientist: what they do, where they are available and when their metadata was refreshed.',
      loading: 'Loading', ready: 'Current', refreshing: 'Refreshing', partial: 'Needs attention', unavailable: 'No data',
      refresh: 'Refresh now', app: 'Back to app', checked: 'Last checked: {time}', never: 'The first check has not finished yet',
      tools: 'current tools', servers: 'servers responding', categories: 'subject categories', attention: 'need attention',
      allConnections: 'All connections', instruments: 'Tools', search: 'Search', searchPlaceholder: 'Name, task or server',
      category: 'Category', serverLabel: 'Server', availability: 'Availability', role: 'Purpose', allCategories: 'All categories', allServers: 'All servers', allAvailability: 'All tools', allRoles: 'All purposes',
      available: 'Available', needsAttention: 'Needs attention', scientific: 'Scientific', supporting: 'Supporting',
      shown: 'Showing {shown} of {total}', empty: 'No tools match the selected filters.', details: 'Technical details',
      machineName: 'Machine name', server: 'Server', source: 'Metadata source', parameters: 'Parameters', unknownSchema: 'Parameters are not described',
      original: 'Original description', agentLimited: 'Restricted access', fedot: 'FEDOT', more: 'More operations ({count})',
      registryUnavailable: 'The Tool RAG registry is unavailable. Connected server metadata and the last saved snapshot are shown.',
      attentionNotice: '{count} connections need attention: a server did not respond or its current tools differ from the index.',
      firstLoad: 'The catalogue is being built. This page will update as servers respond.',
      refreshFailed: 'Could not start catalogue refresh.',
      categoriesMap: { automl: 'AutoML', chemistry: 'Chemistry', epidemiology: 'Epidemiology', literature: 'Scientific literature', data: 'Data', files: 'Files', web: 'Web search', reporting: 'Reports', other: 'Other' },
      sourceMap: { live: 'MCP tools/list', registry: 'Tool RAG', snapshot: 'Saved snapshot' },
    },
  };

  const $ = id => document.getElementById(id);
  let lang = localStorage.getItem(LANG_KEY) === 'en' ? 'en' : 'ru';
  let snapshot = null;
  let timer = null;
  let loading = false;

  function text(key, vars = {}) {
    let value = COPY[lang][key] ?? key;
    Object.entries(vars).forEach(([name, replacement]) => { value = value.replace(`{${name}}`, String(replacement)); });
    return value;
  }

  function localized(value, fallback = '') {
    if (!value || typeof value !== 'object') return String(value || fallback);
    return String(value[lang] || value.ru || value.en || fallback);
  }

  function node(tag, className, content) {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (content !== undefined) result.textContent = content;
    return result;
  }

  function badge(label, className = '') {
    return node('span', `badge ${className}`.trim(), label);
  }

  function formatDate(value) {
    if (!value) return text('never');
    const parsed = new Date(value);
    return Number.isNaN(parsed.valueOf()) ? text('never') : text('checked', {
      time: new Intl.DateTimeFormat(lang === 'ru' ? 'ru-RU' : 'en-GB', { dateStyle: 'medium', timeStyle: 'short' }).format(parsed),
    });
  }

  function categoryLabel(category) {
    return COPY[lang].categoriesMap[category] || COPY[lang].categoriesMap.other;
  }

  function serverFor(tool) {
    return (snapshot?.servers || []).find(server => server.id === tool.server_id) || {};
  }

  function setStaticCopy() {
    document.documentElement.lang = lang;
    document.title = `CoScientist — ${text('title')}`;
    $('page-title').textContent = text('title');
    $('page-lede').textContent = text('lede');
    $('refresh-button').textContent = text('refresh');
    document.querySelector('.top-actions a').textContent = text('app');
    $('language-button').textContent = lang === 'ru' ? 'EN' : 'RU';
    $('metric-tools-label').textContent = text('tools');
    $('metric-servers-label').textContent = text('servers');
    $('metric-categories-label').textContent = text('categories');
    $('metric-attention-label').textContent = text('attention');
    $('catalog-eyebrow').textContent = text('allConnections');
    $('catalog-heading').textContent = text('instruments');
    $('search-label').textContent = text('search');
    $('search-input').placeholder = text('searchPlaceholder');
    $('category-label').textContent = text('category');
    $('server-label').textContent = text('serverLabel');
    $('availability-label').textContent = text('availability');
    $('role-label').textContent = text('role');
    $('availability-filter').options[0].text = text('allAvailability');
    $('availability-filter').options[1].text = text('available');
    $('availability-filter').options[2].text = text('needsAttention');
    $('role-filter').options[0].text = text('allRoles');
    $('role-filter').options[1].text = text('scientific');
    $('role-filter').options[2].text = text('supporting');
    $('empty-result').textContent = text('empty');
  }

  function updateState() {
    const el = $('catalog-state');
    el.className = 'state-pill';
    if (!snapshot || !snapshot.snapshot_id) {
      el.textContent = snapshot?.refreshing ? text('refreshing') : text('loading');
      el.classList.add(snapshot?.refreshing ? 'refreshing' : 'error');
    } else if (snapshot.refreshing) {
      el.textContent = text('refreshing');
      el.classList.add('refreshing');
    } else if (snapshot.partial || snapshot.stale) {
      el.textContent = text('partial');
      el.classList.add('partial');
    } else {
      el.textContent = text('ready');
      el.classList.add('ready');
    }
    $('refresh-button').disabled = !!snapshot?.refreshing;
  }

  function renderNotice() {
    const target = $('notice');
    target.replaceChildren();
    if (!snapshot?.snapshot_id) {
      target.append(node('div', 'notice', text('firstLoad')));
      return;
    }
    if (snapshot.registry?.status !== 'ready') {
      target.append(node('div', 'notice error', text('registryUnavailable')));
      return;
    }
    const count = snapshot.summary?.attention_servers || 0;
    if (count) target.append(node('div', 'notice', text('attentionNotice', { count })));
  }

  function renderMetrics() {
    const summary = snapshot?.summary || {};
    $('metric-tools').textContent = summary.current_tools ?? '—';
    $('metric-servers').textContent = summary.configured_servers
      ? `${summary.reachable_servers || 0}/${summary.configured_servers}` : '—';
    $('metric-categories').textContent = summary.categories ?? '—';
    $('metric-attention').textContent = summary.attention_servers ?? '—';
    $('checked-at').textContent = formatDate(snapshot?.checked_at);
  }

  function appendTechnical(details, tool, server) {
    const summary = node('summary', '', text('details'));
    const box = node('div', 'technical');
    const dl = node('dl');
    const pairs = [
      [text('machineName'), tool.name, true],
      [text('server'), localized(server.display_name, server.name || tool.server_id), false],
      [text('source'), COPY[lang].sourceMap[tool.metadata_source] || tool.metadata_source, false],
    ];
    pairs.forEach(([label, value, code]) => {
      dl.append(node('dt', '', label));
      const dd = node('dd');
      dd.append(code ? node('code', '', value) : document.createTextNode(value));
      dl.append(dd);
    });
    box.append(dl);
    // The registry contract is commonly authored in English. Keep it available
    // in the English view, but do not mix it into the Russian catalogue.
    if (lang === 'en' && tool.original_description && tool.original_description !== localized(tool.summary)) {
      box.append(node('div', '', `${text('original')}: ${tool.original_description}`));
    }
    const schemaTitle = node('div', '', text('parameters'));
    const schema = tool.input_schema && Object.keys(tool.input_schema).length
      ? JSON.stringify(tool.input_schema, null, 2) : text('unknownSchema');
    box.append(schemaTitle, node('pre', '', schema));
    details.append(summary, box);
  }

  function toolCard(tool) {
    const server = serverFor(tool);
    const card = node('article', 'tool-card');
    card.dataset.toolId = tool.id;
    const meta = node('div', 'card-meta');
    meta.append(badge(categoryLabel(tool.category), 'category'));
    if (tool.role === 'supporting') meta.append(badge(text('supporting'), 'supporting'));
    if (!tool.available_to_agents) meta.append(badge(text('agentLimited'), 'supporting'));
    card.append(meta);
    card.append(node('h3', '', localized(tool.display_name, tool.name)));
    card.append(node('p', '', localized(tool.summary, tool.original_description)));
    card.append(node('div', 'server-line', localized(server.display_name, server.name || tool.server_id)));
    const details = node('details');
    details.dataset.detailsId = tool.id;
    appendTechnical(details, tool, server);
    card.append(details);
    return card;
  }

  function needsAttention(tool) {
    const server = serverFor(tool);
    return tool.status !== 'available'
      || server.discovery_status !== 'reachable'
      || Boolean(server.discrepancy?.has_difference);
  }

  function matches(tool) {
    const query = $('search-input').value.trim().toLocaleLowerCase(lang === 'ru' ? 'ru' : 'en');
    const category = $('category-filter').value;
    const serverId = $('server-filter').value;
    const availability = $('availability-filter').value;
    const role = $('role-filter').value;
    const server = serverFor(tool);
    const haystack = [tool.name, localized(tool.display_name), localized(tool.summary), tool.original_description,
      server.name, localized(server.display_name)].join(' ').toLocaleLowerCase(lang === 'ru' ? 'ru' : 'en');
    const attention = needsAttention(tool);
    const matchesAvailability = !availability
      || (availability === 'available' && !attention)
      || (availability === 'attention' && attention);
    return (!query || haystack.includes(query)) && (!category || tool.category === category)
      && (!serverId || tool.server_id === serverId)
      && matchesAvailability && (!role || tool.role === role);
  }

  function openDetails() {
    return new Set([...document.querySelectorAll('details[open][data-details-id]')].map(item => item.dataset.detailsId));
  }

  function restoreDetails(ids) {
    document.querySelectorAll('details[data-details-id]').forEach(item => { item.open = ids.has(item.dataset.detailsId); });
  }

  function renderFeatured(filteredTools) {
    const section = $('featured-section');
    const host = $('featured-card');
    host.replaceChildren();
    const featured = (snapshot?.servers || []).find(server => server.featured);
    if (!featured) { section.hidden = true; return new Set(); }
    const tools = filteredTools.filter(tool => tool.server_id === featured.id);
    if (!tools.length) { section.hidden = true; return new Set(); }
    section.hidden = false;
    const card = node('article', 'featured');
    const header = node('div', 'featured-header');
    const copy = node('div');
    const kicker = node('div', 'featured-kicker');
    kicker.append(badge(text('fedot'), 'available'));
    copy.append(kicker, node('h2', '', localized(featured.display_name, featured.name)), node('p', 'featured-description', localized(featured.description)));
    header.append(copy);
    card.append(header);
    const list = node('div', 'featured-tools');
    const primary = tools.slice(0, 2);
    primary.forEach(tool => {
      const item = node('div', 'featured-tool');
      item.append(node('h3', '', localized(tool.display_name, tool.name)), node('p', '', localized(tool.summary, tool.original_description)));
      const technical = node('details');
      technical.dataset.detailsId = tool.id;
      appendTechnical(technical, tool, featured);
      item.append(technical);
      list.append(item);
    });
    card.append(list);
    if (tools.length > primary.length) {
      const details = node('details');
      details.dataset.detailsId = `featured:${featured.id}`;
      details.append(node('summary', '', text('more', { count: tools.length - primary.length })));
      const more = node('div', 'featured-tools');
      tools.slice(primary.length).forEach(tool => {
        const item = node('div', 'featured-tool');
        item.append(node('h3', '', localized(tool.display_name, tool.name)), node('p', '', localized(tool.summary, tool.original_description)));
        const technical = node('details');
        technical.dataset.detailsId = tool.id;
        appendTechnical(technical, tool, featured);
        item.append(technical);
        more.append(item);
      });
      details.append(more);
      card.append(details);
    }
    host.append(card);
    return new Set(tools.map(tool => tool.id));
  }

  function renderFilters() {
    const select = $('category-filter');
    const value = select.value;
    const present = new Set((snapshot?.tools || []).map(tool => tool.category));
    select.replaceChildren(new Option(text('allCategories'), ''));
    CATEGORY_ORDER.filter(category => present.has(category)).forEach(category => select.add(new Option(categoryLabel(category), category)));
    if ([...select.options].some(option => option.value === value)) select.value = value;

    const serverSelect = $('server-filter');
    const serverValue = serverSelect.value;
    serverSelect.replaceChildren(new Option(text('allServers'), ''));
    [...(snapshot?.servers || [])]
      .sort((left, right) => localized(left.display_name, left.name).localeCompare(localized(right.display_name, right.name), lang))
      .forEach(server => serverSelect.add(new Option(localized(server.display_name, server.name), server.id)));
    if ([...serverSelect.options].some(option => option.value === serverValue)) serverSelect.value = serverValue;
  }

  function renderTools() {
    const opened = openDetails();
    const all = snapshot?.tools || [];
    const filtered = all.filter(matches);
    const featuredIds = renderFeatured(filtered);
    const ordinary = filtered.filter(tool => !featuredIds.has(tool.id));
    const grid = $('tools-grid');
    grid.replaceChildren(...ordinary.map(toolCard));
    $('empty-result').hidden = filtered.length > 0;
    $('visible-count').textContent = text('shown', { shown: filtered.length, total: all.length });
    restoreDetails(opened);
  }

  function render() {
    setStaticCopy();
    updateState();
    renderNotice();
    renderMetrics();
    renderFilters();
    renderTools();
  }

  function schedule() {
    clearTimeout(timer);
    const delay = document.hidden ? POLL.hidden : (snapshot?.refreshing ? POLL.refreshing : POLL.normal);
    timer = setTimeout(load, delay);
  }

  async function load() {
    if (loading) return;
    loading = true;
    try {
      const response = await fetch('/api/mcp-tools', { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      snapshot = await response.json();
      render();
    } catch (error) {
      if (!snapshot) snapshot = { refreshing: false, partial: true, summary: {}, servers: [], tools: [] };
      updateState();
      const notice = node('div', 'notice error', `${text('unavailable')}: ${error.message}`);
      $('notice').replaceChildren(notice);
    } finally {
      loading = false;
      schedule();
    }
  }

  async function refreshNow() {
    $('refresh-button').disabled = true;
    try {
      const response = await fetch('/api/mcp-tools/refresh', { method: 'POST', cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      snapshot = await response.json();
      render();
      clearTimeout(timer);
      timer = setTimeout(load, POLL.refreshing);
    } catch (error) {
      $('notice').replaceChildren(node('div', 'notice error', `${text('refreshFailed')} ${error.message}`));
      $('refresh-button').disabled = false;
    }
  }

  $('refresh-button').addEventListener('click', refreshNow);
  $('language-button').addEventListener('click', () => {
    lang = lang === 'ru' ? 'en' : 'ru';
    localStorage.setItem(LANG_KEY, lang);
    render();
  });
  ['search-input', 'category-filter', 'server-filter', 'availability-filter', 'role-filter'].forEach(id => {
    $(id).addEventListener(id === 'search-input' ? 'input' : 'change', renderTools);
  });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) load();
    else schedule();
  });

  setStaticCopy();
  load();
})();
