// =========================================================================
// i18n – lightweight translation system
// =========================================================================
// Текущий язык: 'ru' по умолчанию (или из localStorage)
// state.js is absent on the standalone pages (graph/builds), so every
// cross-module reference below is guarded — this file must load alone.
const I18N_LANG_KEY = (typeof LANG_STORAGE_KEY !== 'undefined') ? LANG_STORAGE_KEY : 'coscientist.lang';
let currentLang = localStorage.getItem(I18N_LANG_KEY) || 'ru';

const i18n = {
  // ── Navigation & Sidebar ──
  'nav.agents': { en: 'Agents', ru: 'Агенты' },
  'nav.user': { en: 'User', ru: 'Пользователь' },
  'nav.session': { en: 'Session', ru: 'Сессия' },
  'nav.noUser': { en: 'No user selected', ru: 'Пользователь не выбран' },
  'nav.connected': { en: 'Connected', ru: 'Подключено' },
  'nav.disconnected': { en: 'Disconnected', ru: 'Отключено' },
  'nav.callGraph': { en: 'Call graph', ru: 'Граф вызовов' },
  'nav.callGraphHint': { en: 'Drag to move · wheel to zoom · double-click to fit', ru: 'Перетаскивание — перемещение · колесо — масштаб · двойной клик — вписать' },
  'nav.callGraphFull': { en: 'Full screen', ru: 'На весь экран' },
  'nav.callGraphExitFull': { en: 'Exit full screen (Esc)', ru: 'Свернуть (Esc)' },
  'nav.callGraphEmpty': { en: 'Agents appear here as they are called', ru: 'Агенты появятся здесь по мере вызова' },
  'nav.orchestrator': { en: 'Orchestrator', ru: 'Оркестратор' },
  'rail.resize': {
    en: 'Drag to resize · double-click to reset',
    ru: 'Потяните, чтобы изменить ширину · двойной клик — сбросить',
  },
  'rail.show': { en: 'Show the plan column', ru: 'Показать колонку плана' },
  'rail.hide': { en: 'Hide the plan column', ru: 'Скрыть колонку плана' },

  // ── Composer: report language (NOT the interface language) ──
  'composer.reportLang.ru': { en: 'Report: RU', ru: 'Отчёт: RU' },
  'composer.reportLang.en': { en: 'Report: EN', ru: 'Отчёт: EN' },

  // Agent descriptions in side nav
  'agent.OrchestratorAgent.desc': { en: 'Master Orchestrator', ru: 'Главный оркестратор' },
  'agent.PlannerAgent.desc': { en: 'Roadmap Planner', ru: 'Планировщик задач' },
  'agent.TZSpecAgent.desc': { en: 'Technical Spec', ru: 'Техническое задание' },
  'agent.ToolsViewer.desc': { en: 'Tools Viewer', ru: 'Вызовы инструментов' },
  'agent.ToolCatalogue.desc': { en: 'MCP Tool Catalogue', ru: 'Каталог MCP-инструментов' },
  'agent.AgentTopology.desc': { en: 'MAS Configuration', ru: 'Конфигурация МАС' },
  'agent.KnowledgeGraph.desc': { en: 'Research Graph', ru: 'Граф исследований' },
  'agent.SessionTrace.desc': { en: 'Session Trace', ru: 'Трассировка сессии' },
  'agent.MCPBuilder.desc': { en: 'MCP Builder', ru: 'Сборщик MCP' },
  'agent.FedotTrace.desc': { en: 'FEDOT.MAS trace', ru: 'Трассировка FEDOT.MAS' },
  'agent.FedotDemo.desc': { en: 'FEDOT.MAS agent graph', ru: 'Граф агентов FEDOT.MAS' },
  'agent.PaperStatistics.desc': { en: 'Paper Statistics', ru: 'Статистика статей' },
  'agent.CoderSandbox.desc': { en: 'CoderSandbox', ru: 'Песочница кода' },
  'agent.SandboxArtifacts.desc': { en: 'Sandbox artifacts', ru: 'Артефакты песочницы' },
  'agent.__settings__.desc': { en: 'Settings', ru: 'Настройки' },

  // Chat controls & Header
  'chat.missionControl': { en: 'Mission Control', ru: 'Центр управления' },
  'chat.online': { en: 'Online', ru: 'Онлайн' },
  'chat.offline': { en: 'Offline', ru: 'Офлайн' },
  'chat.sendQuery': { en: 'Send a query to begin orchestration', ru: 'Отправьте запрос для начала работы' },
  'chat.placeholder': { en: 'Message the orchestrator…', ru: 'Напишите оркестратору…' },
  'chat.inputLabel': { en: 'Message to the orchestrator', ru: 'Сообщение оркестратору' },
  'chat.hintSend': { en: 'send', ru: 'отправить' },
  'chat.hintNewline': { en: 'new line', ru: 'новая строка' },
  'chat.stopShort': { en: 'Stop', ru: 'Остановить' },
  'chat.stopTitle': { en: 'Stop the agents', ru: 'Остановить агентов' },
  'chat.send': { en: 'Send', ru: 'Отправить' },
  'nav.group.work': { en: 'Work', ru: 'Работа' },
  'nav.group.observe': { en: 'Observe', ru: 'Наблюдение' },
  'nav.group.tools': { en: 'Tools', ru: 'Инструменты' },
  'nav.settings': { en: 'Settings', ru: 'Настройки' },
  'nav.switchUser': { en: 'Switch or add a user', ru: 'Сменить или добавить пользователя' },
  'nav.sessionPicker': { en: 'Session', ru: 'Сессия' },
  'nav.sessionMenu': { en: 'Session actions', ru: 'Действия с сессией' },
  'nav.sessionNew': { en: 'New session', ru: 'Новая сессия' },
  'nav.sessionRename': { en: 'Rename…', ru: 'Переименовать…' },
  'nav.sessionHideOld': { en: 'Hide old sessions', ru: 'Скрыть старые сессии' },
  'nav.sessionUnhideAll': { en: 'Unhide all', ru: 'Вернуть все скрытые' },
  'nav.sessionShowHidden': { en: 'Show hidden sessions', ru: 'Показывать скрытые' },
  'nav.sessionSave': { en: 'Save to disk', ru: 'Сохранить на диск' },
  'nav.sessionExport': { en: 'Export (.zip)', ru: 'Экспорт (.zip)' },
  'nav.sessionImport': { en: 'Import…', ru: 'Импорт…' },
  'nav.sessionRestore': { en: 'Restore saved…', ru: 'Восстановить сохранённую…' },
  'nav.toggleSidebar': { en: 'Show or hide the sidebar', ru: 'Показать или скрыть боковую панель' },
  'topbar.clear': { en: 'Clear the view (history is kept)', ru: 'Очистить ленту (история сохранится)' },
  'topbar.checkpoints': { en: 'Checkpoints', ru: 'Контрольные точки' },
  'telemetry.header': { en: 'Telemetry Output', ru: 'Лог телеметрии' },
  'usage.header': { en: 'Session spend', ru: 'Расходы сессии' },
  'usage.showRest': { en: '{n} more agents · {cost}', ru: 'Ещё агентов: {n} · {cost}' },
  'usage.showFewer': { en: 'Show top 5', ru: 'Показать первые 5' },
  'usage.duration': { en: 'Run duration', ru: 'Время выполнения' },
  'usage.durationRunning': { en: 'Running', ru: 'Выполняется' },
  'usage.durationFinished': { en: 'Completed in', ru: 'Выполнено за' },

  // ── Plan tracker (right sidebar) ──
  // ── Document panel ──
  // A long result is written to a file and opened here; the feed keeps a
  // summary and a button.
  'doc.open': { en: 'Open', ru: 'Открыть' },
  'doc.close': { en: 'Close document', ru: 'Закрыть документ' },
  'doc.untitled': { en: 'Document', ru: 'Документ' },
  'doc.loading': { en: 'Loading…', ru: 'Загрузка…' },
  'doc.failed': { en: 'Could not open this document.', ru: 'Не удалось открыть документ.' },
  'doc.sessionDocs': { en: 'Session documents', ru: 'Документы сессии' },
  'doc.empty': { en: 'No documents yet.', ru: 'Документов пока нет.' },

  'plan.header': { en: 'Plan', ru: 'План' },
  'plan.progress': { en: '{done} of {total}', ru: '{done} из {total}' },
  'plan.stageOf': { en: 'Stage {n} of {total}', ru: 'Этап {n} из {total}' },
  'plan.doneFolded': { en: '{n} more steps done', ru: 'Выполнено ещё этапов: {n}' },
  'plan.open': { en: 'Open roadmap', ru: 'Открыть план' },
  'plan.untitled': { en: 'Untitled task', ru: 'Задача без названия' },
  'plan.status.todo': { en: 'Pending', ru: 'Ожидает' },
  'plan.status.in_progress': { en: 'In progress', ru: 'В работе' },
  'plan.status.done': { en: 'Completed', ru: 'Выполнена' },
  'plan.status.error': { en: 'Failed', ru: 'Ошибка' },

  // ── Plan sub-steps ──
  // What actually ran under a plan step, one line per agent that worked on it.
  // The plan itself has no sub-steps — the planner registers a flat task list —
  // so these are read off the live activity stream and named here by the agent
  // that produced them. A name missing from this table falls back to the agent
  // role from status_indicator.js, so a new agent still reads as something.
  'plan.substeps': { en: 'Sub-steps', ru: 'Подшаги' },
  'plan.substep.toolCount': { en: '{n} tool call(s)', ru: 'вызовов инструментов: {n}' },
  'substep.OrchestratorAgent': { en: 'Coordination', ru: 'Координация работ' },
  'substep.ContextInitAgent': { en: 'Research frame', ru: 'Рамка исследования' },
  'substep.ContextInitSessionAgent': { en: 'Research frame', ru: 'Рамка исследования' },
  'substep.PlannerAgent': { en: 'Planning', ru: 'Построение плана' },
  'substep.PlanningPipelineAgent': { en: 'Planning', ru: 'Построение плана' },
  'substep.PlanCriticAgent': { en: 'Plan review', ru: 'Проверка плана' },
  'substep.HypothesesAgent': { en: 'Hypothesis generation', ru: 'Генерация гипотез' },
  'substep.ResearchAgent': { en: 'Literature search', ru: 'Поиск и разбор литературы' },
  'substep.TaskExecutorAgent': { en: 'Task execution', ru: 'Выполнение задачи' },
  'substep.ToolPipelineAgent': { en: 'Tool selection', ru: 'Подбор инструментов' },
  'substep.ToolPreparerAgent': { en: 'Tool preparation', ru: 'Подготовка инструментов' },
  'substep.McpBuilderAgent': { en: 'Tool build', ru: 'Сборка инструмента' },
  'substep.WebToolsDeployerAgent': { en: 'Tool deployment', ru: 'Подключение инструментов' },
  'substep.CoderAgent': { en: 'Code run', ru: 'Запуск кода в песочнице' },
  'substep.DatasetCollectorAgent': { en: 'Data collection', ru: 'Сбор данных' },
  'substep.MedicalAgent': { en: 'Medical analysis', ru: 'Медицинский анализ' },
  'substep.ExperimentAgent': { en: 'Experiment', ru: 'Эксперимент' },
  'substep.ExperimentModuleAgent': { en: 'Experiment module', ru: 'Модуль экспериментов' },
  'substep.ExperimentPlannerAgent': { en: 'Experiment planning', ru: 'Планирование эксперимента' },
  'substep.ExperimentExecutorAgent': { en: 'Experiment run', ru: 'Проведение эксперимента' },
  'substep.ExperimentResultReviewAgent': { en: 'Result review', ru: 'Приёмка результатов' },
  'substep.FedotAgent': { en: 'AutoML modelling', ru: 'Подбор модели AutoML' },
  'substep.ResultAggregatorAgent': { en: 'Report assembly', ru: 'Сборка отчёта' },
  'substep.NirReportAgent': { en: 'R&D report', ru: 'Оформление отчёта НИР' },

  // ── Activity Rail HUD ──
  'rail.agents': { en: 'Agents', ru: 'Агенты' },
  'rail.tools': { en: 'Tools', ru: 'Инструменты' },
  'rail.standby': { en: 'No tool calls yet', ru: 'Инструменты ещё не вызывались' },
  'rail.allCalls': { en: 'All tool calls', ru: 'Все вызовы инструментов' },
  'rail.agentOne': { en: 'agent', ru: 'агент' },
  'rail.agentFew': { en: 'agents', ru: 'агента' },
  'rail.agentMany': { en: 'agents', ru: 'агентов' },
  'rail.nowWorking': { en: 'Working now:', ru: 'Сейчас работает' },
  'rail.noTools': { en: 'No tool calls yet', ru: 'Инструменты ещё не вызывались' },
  'rail.toggle': { en: 'Show/hide agent activity', ru: 'Показать/скрыть активность агентов' },

  // ── Settings modal (rendered by modals/settings.js) ───────────────────────
  'settings.title': { en: 'Settings', ru: 'Настройки' },
  'settings.search': { en: 'Search settings', ru: 'Поиск настроек' },
  'settings.noResults': { en: 'Nothing found', ru: 'Ничего не найдено' },
  'settings.banner': {
    en: 'Shared by every user of this server and reset when it restarts. Permanent values live in .env.',
    ru: 'Настройки общие для всех пользователей сервера и сбрасываются при его перезапуске. Постоянные значения задаются в .env.'
  },
  'settings.banner.browser': {
    en: 'Stored in this browser only and applied at once — no need to press Save.',
    ru: 'Хранится только в этом браузере и применяется сразу — «Сохранить» нажимать не нужно.'
  },
  'settings.reset': { en: 'Reset to defaults', ru: 'По умолчанию' },
  'settings.resetHint': {
    en: 'Restore the values the server was started with (.env). Saved only after you press Save.',
    ru: 'Вернуть значения, с которыми запущен сервер (.env). Сохраняются только после нажатия «Сохранить».'
  },
  'settings.resetHint.browser': {
    en: 'Restore the default accent colour, font and brightness. Applies at once.',
    ru: 'Вернуть стандартные акцентный цвет, шрифт и яркость. Применяется сразу.'
  },
  'settings.advanced': { en: 'Advanced ({n})', ru: 'Расширенные ({n})' },
  'settings.differsDefault': { en: 'Differs from default: {value}', ru: 'Отличается от значения по умолчанию: {value}' },
  'settings.envVar': { en: 'Default comes from .env: {name}', ru: 'Значение по умолчанию задаётся в .env: {name}' },
  'settings.envValues': { en: 'Allowed values:', ru: 'Допустимые значения:' },
  'settings.timeout.mode.wait': { en: 'Wait for me', ru: 'Ждать решения' },
  'settings.timeout.mode.auto': { en: 'Approve after', ru: 'Одобрить через' },
  'settings.timeout.after.label': { en: 'after', ru: 'через' },
  'settings.timeout.seconds': { en: 's', ru: 'с' },
  'settings.timeout.wait': { en: 'wait for a human', ru: 'ждать решения' },
  'settings.timeout.after': { en: 'after {n} s', ru: 'через {n} с' },
  'settings.goto': { en: 'Open section', ru: 'Перейти' },
  'settings.value.on': { en: 'On', ru: 'Вкл' },
  'settings.value.off': { en: 'Off', ru: 'Выкл' },
  'settings.value.empty': { en: 'empty', ru: 'пусто' },
  'settings.discard': { en: 'Discard changes', ru: 'Отменить изменения' },
  'settings.close': { en: 'Close', ru: 'Закрыть' },
  'settings.save': { en: 'Save', ru: 'Сохранить' },
  'settings.confirmDiscard': {
    en: 'You have unsaved changes. Close without saving?',
    ru: 'Есть несохранённые изменения. Закрыть без сохранения?'
  },

  // When a change takes effect
  'settings.scope.instant': { en: 'Immediately', ru: 'Сразу' },
  'settings.scope.instant.hint': {
    en: 'Takes effect right after saving, including sessions that are already running.',
    ru: 'Действует сразу после сохранения, в том числе в уже запущенных сессиях.'
  },
  'settings.scope.session': { en: 'Next request', ru: 'Следующий запрос' },
  'settings.scope.session.hint': {
    en: 'The current run finishes on its existing agent tree; the next request rebuilds it without losing session history.',
    ru: 'Текущий запуск завершится на прежнем дереве агентов; перед следующим запросом оно пересоберётся без потери истории сессии.'
  },
  'settings.scope.browser': { en: 'This browser', ru: 'Этот браузер' },
  'settings.scope.browser.hint': {
    en: 'Stored in this browser and applied at once; not saved on the server.',
    ru: 'Хранится в этом браузере и применяется сразу; на сервер не сохраняется.'
  },
  'settings.scope.reload': { en: 'On page reload', ru: 'После перезагрузки' },
  'settings.scope.reload.hint': {
    en: 'Takes effect the next time the page is opened.',
    ru: 'Действует при следующем открытии страницы.'
  },

  // Why a field is greyed out
  'settings.inactive.noPlanner': {
    en: 'Not used: in "Orchestrator plans" mode there is no separate planner.',
    ru: 'Не действует: в режиме «Оркестратор планирует сам» отдельного планировщика нет.'
  },
  'settings.inactive.knowledgeGraph': {
    en: 'Not used: the research graph is off.',
    ru: 'Не действует: граф исследований выключен.'
  },
  'settings.inactive.autoApproved': {
    en: 'Not used: nobody is asked, so there is nothing to wait for.',
    ru: 'Не действует: подтверждение не запрашивается, ждать нечего.'
  },
  'settings.inactive.supersededByMode': {
    en: 'Superseded by the confirmation mode above; still read to derive a mode for a stand configured before it existed.',
    ru: 'Заменено режимом подтверждений выше; читается только чтобы вывести режим для стенда, настроенного до его появления.'
  },
  'settings.inactive.envPinned': {
    en: 'Locked by environment variable {name}; change it in the process environment.',
    ru: 'Заблокировано переменной окружения {name}; измените её в окружении процесса.'
  },
  'settings.inactive.nirUnavailable': {
    en: 'Not available: no normcontrol server is configured (MCP__NORMCONTROL_URL).',
    ru: 'Недоступно: не настроен сервер нормоконтроля (MCP__NORMCONTROL_URL).'
  },
  'settings.inactive.lightOnly': { en: 'Applies to the light theme only.', ru: 'Действует только в светлой теме.' },
  'settings.inactive.parentOff': {
    en: 'Works only while "{parent}" is on.',
    ru: 'Действует, только когда включено «{parent}».'
  },

  // Validation
  'settings.err.rangeInt': { en: 'Enter a whole number from {min} to {max}', ru: 'Введите целое число от {min} до {max}' },
  'settings.err.rangeFloat': { en: 'Enter a number from {min} to {max}', ru: 'Введите число от {min} до {max}' },
  'settings.err.abstainAboveKeep': {
    en: 'Must not exceed the keep threshold ({keep})',
    ru: 'Не может быть больше порога отбора ({keep})'
  },
  'settings.err.sandboxRequired': {
    en: 'Required: without a URL the OpenHands mode cannot run code',
    ru: 'Укажите адрес: без него режим OpenHands не сможет выполнять код'
  },
  'settings.err.url': { en: 'The URL must start with http:// or https://', ru: 'Адрес должен начинаться с http:// или https://' },

  // Footer status
  'settings.status.loading': { en: 'Loading…', ru: 'Загрузка…' },
  'settings.status.loadFailed': {
    en: 'Could not load settings from the server — showing the last known values.',
    ru: 'Не удалось загрузить настройки с сервера — показаны последние известные значения.'
  },
  'settings.status.clean': { en: 'No unsaved changes', ru: 'Нет несохранённых изменений' },
  'settings.status.dirty': { en: 'Unsaved changes: {n}', ru: 'Несохранённых изменений: {n}' },
  'settings.status.errors': { en: 'Fix errors to save: {n}', ru: 'Исправьте ошибки, чтобы сохранить: {n}' },
  'settings.status.saving': { en: 'Saving…', ru: 'Сохранение…' },
  'settings.status.saved': { en: 'Saved.', ru: 'Сохранено.' },
  'settings.status.savedSession': {
    en: 'Saved. Tree-level changes take effect on the next request.',
    ru: 'Сохранено. Изменения дерева агентов вступят в силу со следующего запроса.'
  },
  'settings.status.savedReload': {
    en: 'Saved. The change takes effect after a page reload.',
    ru: 'Сохранено. Изменение вступит в силу после перезагрузки страницы.'
  },
  'settings.status.saveFailed': { en: 'Could not save: {error}', ru: 'Не удалось сохранить: {error}' },

  // Settings → Agents
  'settings.section.agents': { en: 'Agents', ru: 'Агенты' },
  'settings.section.agents.desc': {
    en: 'Which agents take part, how deeply each one reasons and on which model. Changes apply on the next request.',
    ru: 'Какие агенты участвуют в работе, насколько глубоко каждый рассуждает и на какой модели. Изменения применяются со следующего запроса.'
  },
  'settings.f.defaultReasoning.label': { en: 'Default reasoning', ru: 'Ризонинг по умолчанию' },
  'settings.f.defaultReasoning.desc': {
    en: 'How much the model thinks before answering, for agents whose profile sets no level of their own. Higher is more careful, slower and more expensive.',
    ru: 'Сколько модель думает перед ответом у агентов, для которых в профиле не задан свой уровень. Чем выше, тем тщательнее, но медленнее и дороже.'
  },
  'settings.group.agentList': { en: 'Agents of the profile', ru: 'Агенты профиля' },
  'settings.group.agentList.desc': {
    en: 'A disabled agent is not built and disappears from the prompts of the agents that call it.',
    ru: 'Выключенный агент не собирается и пропадает из промптов тех агентов, которые его вызывают.'
  },
  'settings.f.agentOverrides.label': { en: 'Agent settings', ru: 'Настройки агентов' },
  'settings.f.agentOverrides.desc': {
    en: 'The switch turns an agent on or off. Reasoning and model replace the values from system.yaml; empty means as declared there.',
    ru: 'Переключатель включает или выключает агента. Ризонинг и модель заменяют значения из system.yaml; пустое поле означает «как там указано».'
  },
  'settings.reasoning.inherit': { en: 'As in the profile', ru: 'Как в профиле' },
  'settings.reasoning.off': { en: 'Off', ru: 'Выключен' },
  'settings.reasoning.minimal': { en: 'Minimal', ru: 'Минимальный' },
  'settings.reasoning.low': { en: 'Low', ru: 'Низкий' },
  'settings.reasoning.medium': { en: 'Medium', ru: 'Средний' },
  'settings.reasoning.high': { en: 'High', ru: 'Высокий' },
  'settings.agents.inheritWith': { en: 'As in the profile ({value})', ru: 'Как в профиле ({value})' },
  'settings.agents.filter': { en: 'Find an agent…', ru: 'Найти агента…' },
  'settings.agents.showInternal': { en: 'Internal agents ({n})', ru: 'Служебные агенты ({n})' },
  'settings.agents.changed': { en: 'Changed: {n}', ru: 'Изменено агентов: {n}' },
  'settings.agents.none': { en: 'No agent matches the filter.', ru: 'Ни один агент не подходит под фильтр.' },
  'settings.agents.loading': { en: 'Loading agents…', ru: 'Загрузка списка агентов…' },
  'settings.agents.loadFailed': { en: 'Could not load the agents: {error}', ru: 'Не удалось загрузить список агентов: {error}' },
  'settings.agents.toggle': { en: 'Agent {name} is on', ru: 'Агент {name} включён' },
  'settings.agents.badge.root': { en: 'root', ru: 'корневой' },
  'settings.agents.badge.pre': { en: 'before the orchestrator', ru: 'до оркестратора' },
  'settings.agents.badge.post': { en: 'after the orchestrator', ru: 'после оркестратора' },
  'settings.agents.badge.internal': { en: 'internal', ru: 'служебный' },
  'settings.agents.calledBy': { en: 'called by {names}', ru: 'вызывается из {names}' },
  'settings.agents.lock.root': {
    en: 'The root agent: a run has no other entry point.',
    ru: 'Корневой агент: без него запуск невозможен.'
  },
  'settings.agents.lock.internal': {
    en: 'An internal stage: it runs as part of its parent and is not switched separately.',
    ru: 'Служебный этап: работает в составе родителя и отдельно не отключается.'
  },
  'settings.agents.lock.startMode': {
    en: 'Whether this agent runs is decided by the start mode.',
    ru: 'Участие этого агента определяет режим запуска.'
  },
  'settings.agents.lock.unavailable': {
    en: 'Not available: no normcontrol server is configured (MCP__NORMCONTROL_URL).',
    ru: 'Недоступен: не настроен сервер нормоконтроля (MCP__NORMCONTROL_URL).'
  },
  'settings.agents.lock.setting': {
    en: 'Switched by a server setting that the interface does not change.',
    ru: 'Включается настройкой сервера, которую интерфейс не меняет.'
  },
  'settings.agents.enabledRef': {
    en: 'Setting: {ref}.',
    ru: 'Настройка: {ref}.'
  },
  'settings.agents.enabledSetting': {
    en: 'The switch is the {env} setting: the runtime reads it too.',
    ru: 'Переключатель — это настройка {env}: её же читает среда выполнения.'
  },
  'settings.agents.sharedWith': {
    en: 'Switched together with: {names}.',
    ru: 'Включается и выключается вместе с: {names}.'
  },
  'settings.agents.cascade': {
    en: 'These stop being called as well: {names}.',
    ru: 'Вместе с ним перестанут вызываться: {names}.'
  },
  'settings.agents.reasoning': { en: 'Reasoning', ru: 'Ризонинг' },
  'settings.agents.model': { en: 'Model', ru: 'Модель' },
  'settings.agents.modelCustom': { en: 'Custom model ID…', ru: 'Свой ID модели…' },
  'settings.agents.modelScope': {
    en: 'Changes only the LLM; the agent keeps its own prompt, tools and role.',
    ru: 'Меняется только LLM; промпт, инструменты и роль агента сохраняются.',
  },
  'settings.agents.modelPlaceholder': { en: 'as in the profile: {model}', ru: 'как в профиле: {model}' },
  'settings.agents.limit.searches': { en: 'Search limit', ru: 'Лимит поисков' },
  'settings.agents.limit.searches.hint': {
    en: 'Search calls allowed in one run of the agent. Empty: the common limit from Tools.',
    ru: 'Сколько поисковых вызовов разрешено за один запуск агента. Пусто — общий лимит из раздела «Инструменты».',
  },
  'settings.agents.limit.perTool': { en: 'Calls per tool', ru: 'Вызовов каждого инструмента' },
  'settings.agents.limit.perTool.hint': {
    en: 'How many times the agent may call each of its tools in one task; counted per tool.',
    ru: 'Сколько раз агент может вызвать каждый свой инструмент за одну задачу; считается отдельно для каждого инструмента.',
  },
  'settings.agents.limitPlaceholder': { en: 'as in the profile: {n}', ru: 'как в профиле: {n}' },
  'settings.agents.limitPlaceholderSetting': { en: 'common limit: {n}', ru: 'общий лимит: {n}' },
  'settings.agents.reset': { en: 'Reset', ru: 'Сбросить' },
  'settings.agents.resetHint': {
    en: 'Back to the values from system.yaml. Saved only after you press Save.',
    ru: 'Вернуть значения из system.yaml. Сохранится только после нажатия «Сохранить».'
  },

  // Settings → System → export / import
  'settings.group.transfer': { en: 'Move settings', ru: 'Перенос настроек' },
  'settings.group.transfer.desc': {
    en: 'The interface settings as .env lines: put them in the server’s .env or load them into another instance.',
    ru: 'Настройки интерфейса в виде строк .env: их можно положить в .env сервера или загрузить в другой экземпляр.'
  },
  'settings.f.envTransfer.label': { en: 'Export and import', ru: 'Экспорт и импорт' },
  'settings.f.envTransfer.desc': {
    en: 'The file holds the values shown in the form, unsaved changes included. API keys and passwords are never exported. Import fills the form: check the changes, then press Save.',
    ru: 'В файл попадают значения формы, включая несохранённые. Ключи API и пароли не экспортируются. Импорт заполняет форму: проверьте изменения и нажмите «Сохранить».'
  },
  'settings.transfer.export': { en: 'Export .env', ru: 'Экспорт .env' },
  'settings.transfer.import': { en: 'Import .env…', ru: 'Импорт .env…' },
  'settings.transfer.header1': { en: '# CoScientist web interface settings', ru: '# Настройки веб-интерфейса CoScientist' },
  'settings.transfer.header2': { en: '# Exported: {date}', ru: '# Экспорт: {date}' },
  'settings.transfer.header3': { en: '# API keys and passwords are not included.', ru: '# Ключи API и пароли сюда не входят.' },
  'settings.transfer.exported': { en: 'Saved {name}.', ru: 'Файл {name} сохранён.' },
  'settings.transfer.imported': {
    en: 'Imported {n} value(s) into the form. Check them and press Save.',
    ru: 'В форму загружено значений: {n}. Проверьте и нажмите «Сохранить».'
  },
  'settings.transfer.report.title': { en: 'Import from {file}', ru: 'Импорт из {file}' },
  'settings.transfer.report.applied': { en: 'Filled in the form: {n}.', ru: 'Заполнено в форме: {n}.' },
  'settings.transfer.report.errors': { en: 'Not applied:', ru: 'Не применено:' },
  'settings.transfer.report.skipped': {
    en: 'Set in the server’s .env only, skipped: {names}.',
    ru: 'Задаются только в .env сервера, пропущены: {names}.'
  },
  'settings.transfer.report.unknown': {
    en: 'Not interface settings, skipped: {names}.',
    ru: 'Не относятся к настройкам интерфейса, пропущены: {names}.'
  },
  'settings.transfer.report.nothing': {
    en: 'The file has no values the interface can change.',
    ru: 'В файле нет значений, которые меняет интерфейс.'
  },
  'settings.transfer.report.readFailed': { en: 'Could not read the file: {error}', ru: 'Не удалось прочитать файл: {error}' },
  'settings.transfer.err.bool': { en: 'expected true or false', ru: 'ожидается true или false' },
  'settings.transfer.err.number': { en: 'expected a number', ru: 'ожидается число' },
  'settings.transfer.err.option': { en: 'allowed values: {options}', ru: 'допустимые значения: {options}' },
  'settings.transfer.err.json': { en: 'expected a JSON object', ru: 'ожидается JSON-объект' },
  'settings.transfer.err.unknownAgent': { en: 'no agent {name} in this profile', ru: 'в этом профиле нет агента {name}' },

  // Sections
  'settings.section.interface': { en: 'Interface', ru: 'Интерфейс' },
  'settings.section.interface.desc': { en: 'How the web interface looks and behaves.', ru: 'Как выглядит и ведёт себя веб-интерфейс.' },
  'settings.section.research': { en: 'Research flow', ru: 'Ход исследования' },
  'settings.section.research.desc': {
    en: 'How a session starts and how the work gets planned.',
    ru: 'С чего начинается сессия и как планируется работа.'
  },
  'settings.section.approvals': { en: 'Approvals', ru: 'Подтверждения' },
  'settings.section.approvals.desc': {
    en: 'When the system stops and waits for your decision.',
    ru: 'Когда система останавливается и ждёт вашего решения.'
  },
  'settings.section.tools': { en: 'Tools & code', ru: 'Инструменты и код' },
  'settings.section.tools.desc': {
    en: 'Web search, code execution and MCP tool selection.',
    ru: 'Поиск в сети, выполнение кода и подбор MCP-инструментов.'
  },
  'settings.section.models': { en: 'Models', ru: 'Модели' },
  'settings.section.models.desc': {
    en: 'How requests to language models are routed through OpenRouter.',
    ru: 'Как запросы к языковым моделям распределяются через OpenRouter.'
  },
  'settings.section.graphs': { en: 'Graphs & memory', ru: 'Графы и память' },
  'settings.section.graphs.desc': {
    en: 'What the system records about runs and research.',
    ru: 'Что система запоминает о запусках и исследовании.'
  },
  'settings.section.system': { en: 'System', ru: 'Система' },
  'settings.section.system.desc': {
    en: 'Server-level options and values that are set only in .env.',
    ru: 'Параметры сервера и значения, которые задаются только в .env.'
  },

  // Groups inside sections
  'settings.group.planning': { en: 'Planning', ru: 'Планирование' },
  'settings.group.search': { en: 'Web search', ru: 'Поиск в сети' },
  'settings.group.code': { en: 'Code execution', ru: 'Выполнение кода' },
  'settings.group.modules': { en: 'Optional modules', ru: 'Дополнительные модули' },
  'settings.group.toolSelection': { en: 'Tool selection', ru: 'Подбор инструментов' },
  'settings.group.toolSelection.desc': {
    en: 'How TaskExecutorAgent decides which of the found MCP tools to use.',
    ru: 'Как TaskExecutorAgent решает, какие из найденных MCP-инструментов использовать.'
  },
  'settings.group.experimentReview': { en: 'Experiments', ru: 'Эксперименты' },
  'settings.group.experimentReview.desc': {
    en: 'These two are asked even when the switch above is off, and a window that runs out pauses the run rather than approving it.',
    ru: 'Эти два подтверждения спрашиваются даже при выключенном переключателе выше, а по истечении времени прогон встаёт на паузу, а не одобряется.'
  },
  'settings.group.danger': { en: 'Delete data', ru: 'Удаление данных' },
  'settings.group.danger.desc': {
    en: 'These actions run immediately and are not affected by Save or Discard.',
    ru: 'Эти действия выполняются сразу и не зависят от кнопок «Сохранить» и «Отменить».'
  },
  'settings.group.envOnly': { en: 'Set only in .env', ru: 'Задаётся только в .env' },
  'settings.group.envOnly.desc': {
    en: 'To change these, edit .env and restart the server.',
    ru: 'Чтобы изменить, отредактируйте .env и перезапустите сервер.'
  },

  // Allowed .env values, shown in the ⓘ tooltip where the format is not obvious
  'settings.f.startMode.envValues': {
    en: 'planner — plan first\norchestrator — straight to work\norchestrator_planner — orchestrator plans',
    ru: 'planner — сначала план\norchestrator — сразу к делу\norchestrator_planner — оркестратор планирует сам'
  },
  'settings.f.maxSearches.envValues': { en: '0–20 (integer). 0 — web search is off.', ru: '0–20 (целое). 0 — поиск запрещён.' },
  'settings.f.coderMode.envValues': {
    en: 'local — locally\nopenhands — OpenHands sandbox',
    ru: 'local — локально\nopenhands — песочница OpenHands'
  },
  'settings.f.sandboxUrl.envValues': {
    en: 'http(s)://host:port, e.g. http://localhost:8884. Empty — not set.',
    ru: 'http(s)://хост:порт, например http://localhost:8884. Пусто — не задан.'
  },
  'settings.f.workspaceId.envValues': {
    en: 'Folder name, e.g. workspace_1. Empty — created automatically.',
    ru: 'Имя папки, например workspace_1. Пусто — создаётся автоматически.'
  },
  'settings.f.keepScore.envValues': { en: 'Number from 0 to 1, e.g. 0.3', ru: 'Число от 0 до 1, например 0.3' },
  'settings.f.abstainScore.envValues': {
    en: 'Number from 0 to 1, not above the keep threshold, e.g. 0.2',
    ru: 'Число от 0 до 1, не больше порога отбора, например 0.2'
  },
  'settings.f.providerSort.envValues': {
    en: 'default — balanced\nprice — price\nlatency — latency\nthroughput — throughput',
    ru: 'default — баланс\nprice — цена\nlatency — отклик\nthroughput — скорость'
  },
  'settings.f.providerOrder.envValues': {
    en: 'Comma-separated provider names in priority order, e.g. Together, DeepInfra. Empty — any.',
    ru: 'Названия провайдеров через запятую в порядке приоритета, например Together, DeepInfra. Пусто — любые.'
  },
  'settings.f.maxRetries.envValues': { en: '0–10 (integer). 0 — do not retry.', ru: '0–10 (целое). 0 — не повторять.' },
  'settings.f.defaultUsername.envValues': {
    en: 'User nickname, e.g. alice. Also read from DEFAULT_USERNAME. Empty — not set.',
    ru: 'Никнейм пользователя, например alice. Читается также из DEFAULT_USERNAME. Пусто — не задан.'
  },

  // Fields — Interface
  'settings.f.language.label': { en: 'Language (interface and report)', ru: 'Язык (интерфейс и отчёт)' },
  'settings.f.language.desc': {
    en: 'One choice for the interface and the report. Locked while a run is active: switch before you start the session.',
    ru: 'Один выбор для интерфейса и отчёта. Во время выполнения запуска заблокировано: переключите язык до начала сессии.'
  },
  'settings.f.theme.label': { en: 'Theme', ru: 'Тема' },
  'settings.f.theme.desc': {
    en: 'Colour scheme of the web interface. Remembered in this browser only.',
    ru: 'Цветовая схема веб-интерфейса. Запоминается только в этом браузере.'
  },
  'settings.f.theme.opt.dark': { en: 'Dark', ru: 'Тёмная' },
  'settings.f.theme.opt.light': { en: 'Light', ru: 'Светлая' },
  'settings.f.lightDim.label': { en: 'Light theme brightness', ru: 'Яркость светлой темы' },
  'settings.f.lightDim.desc': {
    en: 'Lower it if the light theme feels glaring: backgrounds turn a muted grey and text darkens with them, so contrast is kept.',
    ru: 'Уменьшите, если светлая тема слепит: фон становится приглушённо-серым, текст темнеет вместе с ним, контраст сохраняется.'
  },
  'settings.f.accent.label': { en: 'Accent colour', ru: 'Акцентный цвет' },
  'settings.f.accent.desc': {
    en: 'Buttons, switches, links and highlights. Text shades are adjusted per theme to stay readable.',
    ru: 'Кнопки, переключатели, ссылки и выделения. Оттенок для текста подбирается под каждую тему, чтобы оставаться читаемым.'
  },
  'settings.f.accent.default': { en: 'Theme default', ru: 'Как в теме' },
  'settings.f.accent.custom': { en: 'Custom colour…', ru: 'Свой цвет…' },
  'settings.f.font.label': { en: 'Font', ru: 'Шрифт' },
  'settings.f.font.desc': {
    en: 'Interface text; code and numbers stay monospaced. Fonts other than the default load from Google Fonts.',
    ru: 'Текст интерфейса; код и числа остаются моноширинными. Шрифты, кроме стандартного, загружаются из Google Fonts.'
  },
  'settings.f.font.default': { en: 'default', ru: 'по умолчанию' },
  'settings.f.font.system': { en: 'System font', ru: 'Системный шрифт' },
  'settings.f.callGraphCollapse.label': { en: 'Fold parallel calls back after, s', ru: 'Сворачивать параллельные вызовы через, с' },
  'settings.f.callGraphCollapse.desc': {
    en: 'In the call graph, identical agents run in parallel are drawn as one stack. A click unfolds it; after this many seconds it folds back.',
    ru: 'На графе вызовов одинаковые агенты, работавшие параллельно, собраны в стопку. Щелчок раскрывает её, через столько секунд она сворачивается обратно.'
  },
  'settings.f.callGraphCollapse.envValues': { en: '1–600 (integer seconds).', ru: '1–600 (целое, секунды).' },
  'settings.f.showInternal.label': { en: 'Show internal agents and tools', ru: 'Показывать служебных агентов и инструменты' },
  'settings.f.showInternal.desc': {
    en: 'Pipeline stages, wrappers and system tools (marked internal in the system config) appear in the activity rail, the trace tree and Work Order cards. Useful for debugging.',
    ru: 'Этапы пайплайна, обёртки и системные инструменты (помечены internal в конфиге системы) появляются в панели активности, дереве вызовов и карточках плана работы. Полезно для отладки.'
  },

  // Fields — Research flow
  'settings.f.startMode.label': { en: 'How a session starts', ru: 'С чего начинается сессия' },
  'settings.f.startMode.desc': { en: 'Who picks up your request first.', ru: 'Кто первым берётся за ваш запрос.' },
  'settings.f.startMode.opt.planner': { en: 'Plan first', ru: 'Сначала план' },
  'settings.f.startMode.opt.planner.desc': {
    en: 'The planner writes a roadmap, then the orchestrator carries it out step by step.',
    ru: 'Планировщик составляет план, затем оркестратор выполняет его по шагам.'
  },
  'settings.f.startMode.opt.planner.flow': { en: 'Planner → Orchestrator', ru: 'Планировщик → Оркестратор' },
  'settings.f.startMode.opt.orchestrator': { en: 'Straight to work', ru: 'Сразу к делу' },
  'settings.f.startMode.opt.orchestrator.desc': {
    en: 'The orchestrator starts right away, without an upfront roadmap.',
    ru: 'Оркестратор начинает работу сразу, без предварительного плана.'
  },
  'settings.f.startMode.opt.orchestrator.flow': { en: 'Orchestrator', ru: 'Оркестратор' },
  'settings.f.startMode.opt.orchestrator_planner': { en: 'Orchestrator plans', ru: 'Оркестратор планирует сам' },
  'settings.f.startMode.opt.orchestrator_planner.desc': {
    en: 'The orchestrator writes the roadmap itself; the separate planner is switched off.',
    ru: 'Оркестратор сам составляет план; отдельный планировщик отключён.'
  },
  'settings.f.startMode.opt.orchestrator_planner.flow': { en: 'Orchestrator + plan', ru: 'Оркестратор + план' },
  'settings.f.contextInit.label': { en: 'Research frame', ru: 'Рамка исследования' },
  'settings.f.contextInit.desc': {
    en: 'Before the run, an agent drafts the question, constraints and success criteria and records them in the research graph. With approvals on, you can edit the frame in a form.',
    ru: 'Перед стартом агент формулирует вопрос, ограничения и критерии успеха и заносит их в граф исследования. При включённых подтверждениях рамку можно поправить в форме.'
  },
  'settings.f.contextInitReasoning.label': { en: 'Research frame reasoning', ru: 'Ризонинг рамки исследования' },
  'settings.f.contextInitReasoning.desc': {
    en: 'How much the agents that draft the frame and the technical specification think before answering. Higher is more careful, slower and more expensive.',
    ru: 'Сколько думают агенты, которые составляют рамку и техническое задание. Чем выше, тем тщательнее, но медленнее и дороже.'
  },
  'settings.f.maxHypotheses.label': { en: 'Hypotheses queued for testing', ru: 'Гипотез в очереди на проверку' },
  'settings.f.maxHypotheses.desc': {
    en: 'How many of the hypotheses proposed in one batch go straight into the testing queue; the rest are set aside, and the orchestrator can bring them back later. Testing itself still goes one hypothesis at a time: the next one starts after the current one has a verdict.',
    ru: 'Сколько гипотез из одной порции, предложенной генератором, сразу попадают в очередь на проверку; остальные откладываются, и оркестратор может вернуть их позже. Сама проверка всё равно идёт по одной: следующая гипотеза начинается, когда у текущей есть вердикт.'
  },
  'settings.f.maxHypotheses.envValues': {
    en: '1–5 (integer). 1 — only the best hypothesis is queued.',
    ru: '1–5 (целое). 1 — в очередь попадает только лучшая гипотеза.'
  },
  'settings.f.medicalAgent.label': { en: 'Medical agent', ru: 'Медицинский агент' },
  'settings.f.medicalAgent.desc': {
    en: 'MedicalAgent: PubMed search, PICO extraction, study taxonomy and DICOM image analysis. Off, the orchestrator is not offered it and experiment plans get no medical tasks — clinical literature goes to the research agent. Leave it off for studies with no clinical side. Agents served as separate A2A services read only the environment variable.',
    ru: 'MedicalAgent: поиск в PubMed, извлечение PICO, классификация дизайна исследований и анализ DICOM-снимков. Выключен — оркестратору он не предлагается, а в планах экспериментов нет медицинских задач: клиническую литературу ищет исследовательский агент. Выключайте для исследований без клинической части. Агенты, запущенные отдельными A2A-сервисами, читают только переменную окружения.'
  },
  'settings.f.medicalAgent.scopeHint': {
    en: 'The agent is added or removed for sessions that first run after saving; turning it off also takes the medical route out of experiments already running.',
    ru: 'Агент добавляется или убирается для сессий, впервые запущенных после сохранения; выключение также убирает медицинский маршрут из уже идущих экспериментов.'
  },
  'settings.f.nirReport.label': { en: 'R&D report (GOST 7.32-2017)', ru: 'Отчёт о НИР (ГОСТ 7.32-2017)' },
  'settings.f.nirReport.desc': {
    en: 'At the end of a run, offer to produce a normative DOCX report alongside the short Markdown one, built through the "Автонормоконтроль" service. You are asked first and fill in the title-page details; declining or ignoring the question changes nothing. Needs MCP__NORMCONTROL_URL — the switch is inactive without it. Costs a strong model and dozens of pages of generation.',
    ru: 'В конце прогона предлагать собрать нормативный документ DOCX в дополнение к краткому отчёту в Markdown — через сервис «Автонормоконтроль». Сначала спросят и попросят реквизиты титульного листа; отказ или игнорирование вопроса ничего не меняет. Требуется MCP__NORMCONTROL_URL — без него переключатель неактивен. Стоит сильной модели и десятков страниц генерации.'
  },
  'settings.f.nirReportReasoning.label': { en: 'R&D report reasoning', ru: 'Ризонинг отчёта о НИР' },
  'settings.f.nirReportReasoning.desc': {
    en: 'How much the agent that writes the report thinks. The profile sets high: the report has to stay faithful to the measured numbers.',
    ru: 'Сколько думает агент, который пишет отчёт. В профиле задан высокий: отчёт должен точно следовать измеренным значениям.'
  },
  'settings.f.critic.label': { en: 'Review the plan with a critic', ru: 'Проверять план критиком' },
  'settings.f.critic.desc': {
    en: 'Before execution a separate model checks the roadmap (assignees, coverage, dependencies) and sends it back if it objects. One extra model call per review.',
    ru: 'Перед выполнением отдельная модель проверяет план (исполнители, полнота, зависимости) и при замечаниях возвращает его на доработку. +1 вызов модели на проверку.'
  },
  'settings.f.criticRounds.label': { en: 'Times the plan can be sent back', ru: 'Сколько раз можно вернуть план' },
  'settings.f.criticRounds.desc': {
    en: '1 — a single review, after which the revised plan stands. Every extra round is a full replan.',
    ru: '1 — одна проверка, после доработки план принимается. Каждый дополнительный раунд — полное перепланирование.'
  },
  'settings.f.criticRounds.envValues': { en: '1–5 (integer)', ru: '1–5 (целое)' },
  'settings.f.mergeTasks.label': { en: 'Merge adjacent tasks', ru: 'Объединять соседние задачи' },
  'settings.f.mergeTasks.desc': {
    en: 'Consecutive tasks for the same executor run as one. Turn off to run every planned task separately.',
    ru: 'Идущие подряд задачи одного исполнителя выполняются как одна. Выключите, чтобы каждая задача плана шла отдельно.'
  },
  'settings.f.plannerRetrieval.label': { en: 'Planner sees the tool catalog', ru: 'Планировщик видит каталог инструментов' },
  'settings.f.plannerRetrieval.desc': {
    en: 'The planner looks up suitable MCP tools and names them in tasks. When off, the plan describes only the expected outcome.',
    ru: 'Планировщик ищет подходящие MCP-инструменты и называет их в задачах. Если выключено — план описывает только ожидаемый результат.'
  },
  'settings.f.plannerGraph.label': { en: 'Planner reads the research graph', ru: 'Планировщик читает граф исследований' },
  'settings.f.plannerGraph.desc': {
    en: 'The planner takes past runs into account and does not re-plan work that is already done.',
    ru: 'Планировщик учитывает прошлые запуски и не планирует заново уже сделанную работу.'
  },
  'settings.f.autoNaming.label': { en: 'Auto-name sessions', ru: 'Автоназвание сессий' },
  'settings.f.autoNaming.desc': {
    en: 'Title a new session after its first request.',
    ru: 'Придумывать название новой сессии по первому запросу.'
  },

  // Fields — Approvals
  'settings.f.hitl.label': { en: 'Ask for my approval', ru: 'Спрашивать моё подтверждение' },
  'settings.f.hitl.desc': {
    en: 'Before risky actions (outward-facing requests, irreversible commands) and at key steps, agents stop and wait for your decision in the chat.',
    ru: 'Перед рискованными действиями (внешние запросы, необратимые команды) и на ключевых шагах агенты останавливаются и ждут вашего решения в чате.'
  },
  'settings.f.workOrder.label': { en: "Show the agent's work plan", ru: 'Показывать план работы агента' },
  'settings.f.workOrder.desc': {
    en: 'Before starting, an executor agent shows its goal, steps, tools and assumptions. You can accept, correct or reject the plan.',
    ru: 'Перед началом агент-исполнитель показывает цель, шаги, инструменты и допущения. План можно принять, поправить или отклонить.'
  },
  'settings.f.hitlMode.label': { en: 'Confirmation mode', ru: 'Режим подтверждений' },
  'settings.f.hitlMode.desc': {
    en: 'How this run handles every confirmation. auto — nobody is asked and each one is approved at once, for a run nobody is sitting with. basic — you are asked and have ten minutes. debug — you are asked and the run waits, however long it takes, for when the point is to watch a neighbouring system. Silence approves in none of the three.',
    ru: 'Как прогон обращается с подтверждениями. auto — никого не спрашиваем, всё подтверждается сразу: для прогона, который некому сопровождать. basic — спрашиваем и ждём вас десять минут. debug — спрашиваем и ждём сколько угодно: когда нужно посмотреть работу соседней системы. Молчание не подтверждает ни в одном из трёх.'
  },
  'settings.f.hitlTimeout.label': { en: 'If nobody answers (legacy)', ru: 'Если никто не ответил (устаревшее)' },
  'settings.f.hitlTimeout.desc': {
    en: 'Superseded by the confirmation mode above. Still read to derive a mode for a stand configured before the mode existed: a non-positive value means "wait for me", i.e. debug.',
    ru: 'Заменено режимом подтверждений выше. Читается только чтобы вывести режим для стенда, настроенного до его появления: неположительное значение означает «ждать меня», то есть debug.'
  },
  'settings.f.hitlTimeout.envValues': {
    en: '-1 — reads as the debug mode (wait for the human)\nN > 0 — reads as the basic mode',
    ru: '-1 — читается как режим debug (ждать человека)\nN > 0 — читается как режим basic'
  },
  'settings.f.hitlMode.envValues': {
    en: 'auto — never ask\nbasic — ask, wait 10 minutes, then refuse\ndebug — ask and wait for the human',
    ru: 'auto — не спрашивать\nbasic — спросить, ждать 10 минут, затем отказать\ndebug — спросить и ждать человека'
  },
  'settings.f.workOrderVeto.label': { en: 'Search and computation plans without an answer', ru: 'Планы поиска и вычислений без ответа' },
  'settings.f.workOrderVeto.desc': {
    en: 'Work plans in which the agent only searches, reads or runs computations can start automatically after this time unless you pause them. Plans with external side effects follow the rule above.',
    ru: 'Планы работы, в которых агент только ищет, читает или выполняет вычисления, могут стартовать автоматически через это время, если вы не поставили их на паузу. Планы с внешними последствиями подчиняются правилу выше.'
  },
  'settings.f.workOrderVeto.envValues': {
    en: '-1 — wait for a human (no auto-start)\nN > 0 — start automatically after N seconds',
    ru: '-1 — ждать решения человека (без автостарта)\nN > 0 — стартовать автоматически через N секунд'
  },
  'settings.f.experimentPlanAuto.label': {
    en: 'Approve the experiment plan for me',
    ru: 'Одобрять план эксперимента за меня'
  },
  'settings.f.experimentPlanAuto.desc': {
    en: 'The experiment module shows its plan — the tasks, the tools, the estimated time — and waits for an explicit approval; an unanswered plan is never approved, the run just stops there. Turn this on for a run that has to go through without you.',
    ru: 'Модуль экспериментов показывает свой план — задачи, инструменты, оценку времени — и ждёт явного одобрения; план без ответа не одобряется, и прогон на этом заканчивается. Включите, если прогон должен пройти без вас.'
  },
  'settings.f.experimentPlanTimeout.label': {
    en: 'How long to wait for the plan decision',
    ru: 'Сколько ждать решения по плану'
  },
  'settings.f.experimentPlanTimeout.desc': {
    en: 'Seconds. If nobody answers within this window the run is paused — the plan is NOT approved, and the experiments do not start.',
    ru: 'Секунды. Если за это время никто не ответил, прогон встаёт на паузу — план НЕ одобряется и эксперименты не запускаются.'
  },
  'settings.f.experimentResultAuto.label': {
    en: 'Accept the experiment result for me',
    ru: 'Принимать результат эксперимента за меня'
  },
  'settings.f.experimentResultAuto.desc': {
    en: 'When the tasks are done the module shows what came out and asks whether to accept it or send the experiment back for a redesign. Turn this on and whatever came out is accepted.',
    ru: 'Когда задачи выполнены, модуль показывает, что получилось, и спрашивает: принять или отправить эксперимент на переделку. С включённой настройкой принимается то, что получилось.'
  },
  'settings.f.experimentResultTimeout.label': {
    en: 'How long to wait for the result decision',
    ru: 'Сколько ждать решения по результату'
  },
  'settings.f.experimentResultTimeout.desc': {
    en: 'Seconds. If nobody answers within this window the run is paused — the result is neither accepted nor sent back.',
    ru: 'Секунды. Если за это время никто не ответил, прогон встаёт на паузу — результат не принят и не отправлен на переделку.'
  },

  // Fields — Tools & code
  'settings.f.maxSearches.label': { en: 'Web searches per turn', ru: 'Поисков в сети за один ход' },
  'settings.f.maxSearches.desc': {
    en: 'How many times an agent may search the web before it must answer from what it found. 0 — web search is off.',
    ru: 'Сколько раз агент может искать в интернете, прежде чем отвечать по найденному. 0 — поиск запрещён.'
  },
  'settings.f.coderMode.label': { en: 'Where code runs', ru: 'Где выполняется код' },
  'settings.f.coderMode.desc': {
    en: 'Locally — the agent runs commands and edits files on this server. OpenHands — tasks go to an agent in a remote sandbox.',
    ru: 'Локально — агент сам запускает команды и правит файлы на этом сервере. OpenHands — задачи передаются агенту в удалённой песочнице.'
  },
  'settings.f.coderMode.opt.local': { en: 'Locally', ru: 'Локально' },
  'settings.f.coderMode.opt.openhands': { en: 'OpenHands sandbox', ru: 'Песочница OpenHands' },
  'settings.f.sandboxUrl.label': { en: 'Sandbox URL', ru: 'Адрес песочницы' },
  'settings.f.sandboxUrl.desc': {
    en: 'Server that executes code in isolation. Required in OpenHands mode.',
    ru: 'Сервер изолированного выполнения кода. Обязателен в режиме OpenHands.'
  },
  'settings.f.workspaceId.label': { en: 'Persistent workspace', ru: 'Постоянная рабочая папка' },
  'settings.f.workspaceId.desc': {
    en: 'Folder name where code is kept between tasks. Empty — a folder is created automatically.',
    ru: 'Имя папки, в которой код сохраняется между задачами. Пусто — папка создаётся автоматически.'
  },
  'settings.f.workspaceId.placeholder': { en: 'e.g. workspace_1', ru: 'например, workspace_1' },
  'settings.f.keepScore.label': { en: 'Keep threshold', ru: 'Порог отбора' },
  'settings.f.keepScore.desc': {
    en: 'A found tool is given to the agent if its relevance to the task is at least this value (0–1).',
    ru: 'Найденный инструмент передаётся агенту, если его релевантность задаче не ниже этого значения (0–1).'
  },
  'settings.f.abstainScore.label': { en: 'Hand-off threshold', ru: 'Порог передачи в CoderAgent' },
  'settings.f.abstainScore.desc': {
    en: 'If no tool reaches the keep threshold: the best is at least this value — the top two are used anyway; below it — the task goes to CoderAgent.',
    ru: 'Если ни один инструмент не прошёл порог отбора: лучший не ниже этого значения — берутся два лучших; ниже — задача передаётся CoderAgent.'
  },
  'settings.f.experimentRouteFedot.label': {
    en: 'Use FEDOT.MAS in experiments',
    ru: 'Использовать FEDOT.MAS в экспериментах'
  },
  'settings.f.experimentRouteFedot.desc': {
    en: 'Off: the experiment plan never offers FEDOT.MAS, and MCP tools are called directly by ExperimentAgent (ReAct). On: FEDOT.MAS is kept for the rare task that has to chain several tools in one search loop. An experiment module served as a separate A2A service reads only the environment variable.',
    ru: 'Выключено: план эксперимента не предлагает FEDOT.MAS, и MCP-инструменты вызывает напрямую ExperimentAgent (ReAct). Включено: FEDOT.MAS остаётся для редкой задачи, которой нужно связать несколько инструментов в одном цикле поиска. Модуль экспериментов, запущенный отдельным A2A-сервисом, читает только переменную окружения.'
  },
  'settings.f.fedotFallback.label': { en: 'FEDOT.MAS fallback', ru: 'Резервный маршрут FEDOT.MAS' },
  'settings.f.fedotFallback.desc': {
    en: 'When neither tool ranker finds a usable MCP tool, the task goes to FedotAgent instead of CoderAgent.',
    ru: 'Если ни один ранжировщик не нашёл подходящего MCP-инструмента, задача уходит FedotAgent, а не CoderAgent.'
  },
  'settings.f.fedotFallback.scopeHint': {
    en: 'The tree is rebuilt before the next request. Disabling it blocks new FEDOT.MAS calls but does not cancel one already running.',
    ru: 'Дерево пересоберётся перед следующим запросом. Выключение блокирует новые вызовы FEDOT.MAS, но не отменяет уже выполняющийся.'
  },
  'settings.f.experimentRouteFedot.scopeHint': {
    en: 'The tree is rebuilt before the next request. Disabling it blocks new experiment FEDOT.MAS calls but does not cancel one already running.',
    ru: 'Дерево пересоберётся перед следующим запросом. Выключение блокирует новые вызовы FEDOT.MAS в экспериментах, но не отменяет уже выполняющийся.'
  },
  'settings.f.experimentRouteAlembic.label': {
    en: 'Offer Alembic for repository reuse',
    ru: 'Предлагать Alembic для переиспользования репозитория'
  },
  'settings.f.experimentRouteAlembic.desc': {
    en: 'When inspected repository code already fits unchanged, ask whether to run it directly with Coder (default) or wrap it as an MCP tool. Code changes always stay on Coder.',
    ru: 'Если проверенный код репозитория подходит без изменений, спросить: запустить напрямую через Coder (по умолчанию) или обернуть в MCP-инструмент. Любые изменения кода всегда идут в Coder.'
  },

  // Fields — Models
  'settings.f.providerSort.label': { en: 'Provider priority', ru: 'Приоритет при выборе провайдера' },
  'settings.f.providerSort.desc': {
    en: 'OpenRouter picks which host runs the model. Applies only to models served through OpenRouter.',
    ru: 'OpenRouter выбирает, у какого хостера запускать модель. Действует только для моделей через OpenRouter.'
  },
  'settings.f.providerSort.opt.default': { en: 'Balanced', ru: 'Баланс' },
  'settings.f.providerSort.opt.price': { en: 'Price', ru: 'Цена' },
  'settings.f.providerSort.opt.latency': { en: 'Latency', ru: 'Отклик' },
  'settings.f.providerSort.opt.throughput': { en: 'Throughput', ru: 'Скорость' },
  'settings.f.providerOrder.label': { en: 'Preferred providers', ru: 'Предпочтительные провайдеры' },
  'settings.f.providerOrder.desc': {
    en: 'These hosts are tried first, in the order shown. Empty — any host.',
    ru: 'Эти хостеры пробуются первыми, в указанном порядке. Пусто — любые.'
  },
  'settings.f.providerOrder.placeholder': { en: 'Type a name and press Enter', ru: 'Введите название и нажмите Enter' },
  'settings.f.maxRetries.label': { en: 'Retries on failures', ru: 'Повторы при сбоях' },
  'settings.f.maxRetries.desc': {
    en: 'How many times to retry a model call after a temporary network or API error. 0 — do not retry.',
    ru: 'Сколько раз повторять запрос к модели при временной ошибке сети или API. 0 — не повторять.'
  },

  // Fields — Graphs & memory
  'settings.f.knowledgeGraph.label': { en: 'Research graph', ru: 'Граф исследований' },
  'settings.f.knowledgeGraph.desc': {
    en: 'Records how each run went so agents can build on past work, and the Research Graph view shows it. When off, nothing is recorded and agents work without history.',
    ru: 'Записывает ход каждого запуска: агенты опираются на прошлую работу, а вкладка «Граф исследований» её показывает. Если выключить — ничего не записывается, и агенты работают без истории.'
  },
  'settings.f.researchGraph.label': { en: 'Research graph', ru: 'Граф исследования' },
  'settings.f.researchGraph.desc': {
    en: 'A shared board where agents record findings, hypotheses and conclusions. When off, agents pass context only through their answers.',
    ru: 'Общая доска, куда агенты заносят находки, гипотезы и выводы. Если выключить — агенты передают контекст только через свои ответы.'
  },
  'settings.danger.session.label': { en: "Clear this session's graphs", ru: 'Очистить графы этой сессии' },
  'settings.danger.session.desc': {
    en: 'Removes the execution and research graphs of the open session. The research graph is archived first; the execution graph keeps only the agent list.',
    ru: 'Удаляет граф выполнения и граф исследования открытой сессии. Граф исследования сначала архивируется; в графе выполнения остаётся только список агентов.'
  },
  'settings.danger.session.btn': { en: 'Clear', ru: 'Очистить' },
  'settings.danger.session.confirm': {
    en: 'Clear the graphs of session "{name}"? This cannot be undone from the interface.',
    ru: 'Очистить графы сессии «{name}»? Отменить это из интерфейса нельзя.'
  },
  'settings.danger.memory.label': { en: 'Delete the shared knowledge memory', ru: 'Удалить общую память знаний' },
  'settings.danger.memory.desc': {
    en: 'There is one knowledge memory for the whole server — it disappears for every session at once. An archive is made first.',
    ru: 'Память знаний одна на весь сервер — она исчезнет во всех сессиях сразу. Перед удалением создаётся архив.'
  },
  'settings.danger.memory.btn': { en: 'Delete', ru: 'Удалить' },
  'settings.danger.memory.confirm': {
    en: 'Delete the knowledge memory for all sessions of this server?',
    ru: 'Удалить память знаний для всех сессий этого сервера?'
  },
  'settings.danger.memory.finalTitle': {
    en: 'Are you sure you want to delete the knowledge memory?',
    ru: 'Вы уверены, что хотите удалить память знаний?'
  },
  'settings.danger.memory.finalText': {
    en: 'It disappears for every session and every user of this server at once. It cannot be restored from the interface.',
    ru: 'Она исчезнет сразу во всех сессиях у всех пользователей этого сервера. Восстановить её из интерфейса нельзя.'
  },
  'settings.danger.memory.finalBtn': { en: 'Delete memory', ru: 'Удалить память' },
  'settings.danger.noSession': { en: 'Open a session first.', ru: 'Сначала откройте сессию.' },
  'settings.danger.confirmBtn': { en: 'Yes, delete', ru: 'Да, удалить' },
  'settings.danger.cancel': { en: 'Cancel', ru: 'Отмена' },
  'settings.danger.deleting': { en: 'Deleting…', ru: 'Удаление…' },
  'settings.danger.done': { en: 'Deleted: {what}.', ru: 'Удалено: {what}.' },
  'settings.danger.nothing': { en: 'nothing', ru: 'ничего' },
  'settings.danger.failed': { en: 'Deletion failed: {error}', ru: 'Ошибка удаления: {error}' },

  // Fields — System
  'settings.f.defaultUsername.label': { en: 'Default user', ru: 'Пользователь по умолчанию' },
  'settings.f.defaultUsername.desc': {
    en: 'Selected automatically when the interface opens; created if it does not exist yet.',
    ru: 'Выбирается автоматически при открытии интерфейса; если такого нет — создаётся.'
  },
  'settings.f.defaultUsername.placeholder': { en: 'not set', ru: 'не задан' },
  'settings.f.opik.label': { en: 'Opik tracing', ru: 'Трассировка в Opik' },
  'settings.f.opik.desc': {
    en: 'Send agent logs and traces to the Opik dashboard for debugging.',
    ru: 'Отправлять логи и трассы агентов в дашборд Opik для отладки.'
  },
  'settings.f.useProxy.label': { en: 'Corporate proxy', ru: 'Корпоративный прокси' },
  'settings.f.useProxy.desc': {
    en: 'Model calls go through the proxy from SERVICES__PROXY_URL.',
    ru: 'Запросы к моделям идут через прокси из SERVICES__PROXY_URL.'
  },
  'settings.f.autoClearGraph.label': { en: 'Clear graphs before a session', ru: 'Очищать графы перед сессией' },
  'settings.f.autoClearGraph.desc': {
    en: "A session's graphs are cleared before its agents are first built.",
    ru: 'Графы сессии очищаются перед первой сборкой её агентов.'
  },

  // ── Experiment plan review card (CoScientist/experiments/plan_view.py) ──
  'plan.title': { en: 'Experiment plan', ru: 'План эксперимента' },
  'plan.sidebarTitle': { en: 'Experiment plan', ru: 'План эксперимента' },
  'plan.sidebarHint': {
    en: 'Read the plan in the chat: the design matrix, then each task. Revise sends your corrections back to the planner.',
    ru: 'План — в карточке в чате: матрица плана, затем задачи. «Доработать» отправит ваши правки планировщику.'
  },
  'plan.revision': { en: 'revision {n}', ru: 'ревизия {n}' },
  'plan.tasks': { en: '{n} task(s)', ru: 'задач: {n}' },
  'plan.tasksTitle': { en: 'Tasks', ru: 'Задачи' },
  'plan.min': { en: 'min', ru: 'мин' },
  'plan.goal': { en: 'Goal', ru: 'Цель' },
  'plan.hypothesis': { en: 'Hypothesis', ru: 'Гипотеза' },
  'plan.hypotheses': { en: 'Hypotheses', ru: 'Гипотезы' },
  'plan.methods': { en: 'Methods', ru: 'Методы' },
  'plan.matrix': {
    en: 'Design matrix — hypothesis → experiment → data → baseline → metrics',
    ru: 'Матрица плана — гипотеза → эксперимент → данные → базлайн → метрики'
  },
  'plan.col.task': { en: 'Task', ru: 'Задача' },
  'plan.col.hypothesis': { en: 'Hypothesis', ru: 'Гипотеза' },
  'plan.col.question': { en: 'Question', ru: 'Вопрос' },
  'plan.col.dataset': { en: 'Dataset', ru: 'Данные' },
  'plan.col.baselines': { en: 'Baselines', ru: 'Базлайны' },
  'plan.col.metrics': { en: 'Metrics', ru: 'Метрики' },
  'plan.col.tools': { en: 'Tools', ru: 'Инструменты' },
  'plan.col.artifacts': { en: 'Analysis', ru: 'Анализ' },
  'plan.col.route': { en: 'Route', ru: 'Маршрут' },
  'plan.task.question': { en: 'Question', ru: 'Вопрос' },
  'plan.task.dataset': { en: 'Dataset', ru: 'Данные' },
  'plan.task.baselines': { en: 'Baselines', ru: 'Базлайны' },
  'plan.task.metrics': { en: 'Metrics', ru: 'Метрики' },
  'plan.task.analysis': { en: 'Analysis', ru: 'Анализ' },
  'plan.task.description': { en: 'What runs', ru: 'Что выполняется' },
  'plan.task.rationale': { en: 'Why', ru: 'Зачем' },
  'plan.task.tools': { en: 'MCP / tools', ru: 'MCP / инструменты' },
  'plan.task.repo': { en: 'Repository', ru: 'Репозиторий' },
  'plan.task.codeAssessment': { en: 'Code assessment', ru: 'Оценка кода' },
  'plan.task.params': { en: 'Launch params', ru: 'Параметры запуска' },
  'plan.task.inputs': { en: 'Inputs', ru: 'Входные данные' },
  'plan.task.criteria': { en: 'Success criteria', ru: 'Критерии успеха' },
  'plan.task.expected': { en: 'Expected artifacts', ru: 'Ожидаемые артефакты' },
  'plan.task.warnings': { en: 'Warnings', ru: 'Предупреждения' },
  'plan.task.optional': { en: 'optional', ru: 'необязательная' },
  'plan.task.after': { en: 'after', ru: 'после' },
  'plan.optionalTool': { en: 'optional', ru: 'необязательный' },
  'plan.noInputs': { en: 'no inputs — the task starts from its own launch params', ru: 'входных данных нет — задача стартует со своих параметров' },
  'plan.noTools': { en: 'no MCP tools — this route does not use them', ru: 'MCP-инструменты не используются этим маршрутом' },
  'plan.risks': { en: 'Risks', ru: 'Риски' },
  'plan.assumptions': { en: 'Assumptions', ru: 'Допущения' },
  'plan.critique': { en: 'Automatic review', ru: 'Автоматическая проверка' },
  'plan.critique.approve': {
    en: 'The deterministic critic found no blocking issue.',
    ru: 'Детерминированная проверка не нашла блокирующих проблем.'
  },
  'plan.critique.revise': {
    en: 'The deterministic critic asked for a revision.',
    ru: 'Детерминированная проверка потребовала доработки.'
  },
  'plan.expandAll': { en: 'Expand all', ru: 'Раскрыть все' },
  'plan.collapseAll': { en: 'Collapse all', ru: 'Свернуть все' },
  'plan.accept': { en: 'Approve', ru: 'Утвердить' },
  'plan.acceptWithIssues': { en: 'Approve with issues', ru: 'Согласовать с замечаниями' },
  'plan.exhausted.title': {
    en: 'Automatic revisions are exhausted',
    ru: 'Лимит автоматических доработок исчерпан'
  },
  'plan.exhausted.body': {
    en: 'No more model rewrites will run. Review this exact plan and its unresolved issues before deciding.',
    ru: 'Новых автоматических переписываний не будет. Проверьте именно эту версию плана и оставшиеся замечания перед решением.'
  },
  'plan.exhausted.previousBody': {
    en: 'The latest model answer was structurally invalid, so this is the last schema-valid plan. No more model rewrites will run; approve this exact version or reject it.',
    ru: 'Последний ответ модели был структурно неверным, поэтому показана последняя версия, прошедшая проверку схемы. Новых автоматических доработок не будет: согласуйте именно эту версию или отклоните её.'
  },
  'plan.approvalNotePlaceholder': {
    en: 'Optional note recorded with your approval',
    ru: 'Необязательный комментарий к согласованию'
  },
  'plan.revise': { en: 'Revise', ru: 'Доработать' },
  'plan.reject': { en: 'Reject', ru: 'Отклонить' },
  'plan.feedbackPlaceholder': {
    en: 'Corrections for the planner — then Revise',
    ru: 'Правки для планировщика — затем «Доработать»'
  },

  // ── Work Order cards (hitl.js) ──
  'hitl.msg.workOrder': {
    en: "Agent {agent} declares its work order. Review the plan and the assumptions.",
    ru: "Агент «{agent}» представил план работы. Проверьте план, условия и ограничения."
  },
  'hitl.msg.workOrderAmendment': {
    en: "Agent {agent} wants to amend its work order.",
    ru: "Агент «{agent}» хочет изменить свой план работы."
  },
  'hitl.via.workOrder': { en: 'work order (plan before acting)', ru: 'план работы агента (план до действий)' },
  'hitl.via.workOrderAmendment': { en: 'work order amendment', ru: 'поправка к плану работы' },
  'workOrder.title': { en: 'Work Order', ru: 'План работы агента' },
  'workOrder.amendTitle': { en: 'Work Order Amendment', ru: 'Поправка к плану работы' },
  'workOrder.noticeTitle': { en: 'Work Order (for information)', ru: 'План работы агента' },
  'workOrder.tier.read': { en: 'read', ru: 'чтение' },
  'workOrder.tier.compute': { en: 'compute', ru: 'вычисления' },
  'workOrder.tier.side_effect': { en: 'side effects', ru: 'побочные эффекты' },
  // The one-line strip a Work Order / Work Report folds into (hitl.js woStrip).
  // Short on purpose: the line also has to fit the goal or the summary.
  'woStrip.steps': { en: 'Steps', ru: 'Шаги' },
  'woStrip.assumptions': { en: 'Assumptions', ru: 'Условия' },
  'woStrip.findings': { en: 'Findings', ru: 'Находки' },
  'woStrip.artifacts': { en: 'Artifacts', ru: 'Артефакты' },
  'woStrip.changes': { en: 'Changes', ru: 'Изменения' },
  'woStrip.details': { en: 'Details', ru: 'Подробности' },
  'workOrder.goal': { en: 'Goal', ru: 'Цель' },
  'workOrder.done': { en: 'Done when', ru: 'Критерий готовности' },
  'workOrder.assumptions': { en: 'Assumptions', ru: 'Условия и ограничения' },
  'workOrder.assumptionsHint': {
    en: 'Uncheck the assumptions you reject — the agent must not rely on them.',
    ru: 'Снимите галочку с условий, которые вы отклоняете, — агент не должен на них опираться.'
  },
  'workOrder.steps': { en: 'Steps', ru: 'Шаги' },
  'workOrder.tools': { en: 'Tools', ru: 'Инструменты' },
  'workOrder.sideEffects': { en: 'Side effects', ru: 'Побочные эффекты' },
  'workOrder.expected': { en: 'Expected outcome', ru: 'Ожидаемый результат' },
  'workOrder.fallback': { en: 'If it fails', ru: 'Если не получится' },
  'workOrder.reason': { en: 'Reason', ru: 'Обоснование' },
  'workOrder.added': { en: 'Requested changes', ru: 'Запрошенные изменения' },
  'workOrder.countdown': { en: 'Starts automatically in {s} s', ru: 'Автоматический старт через {s} с' },
  'workOrder.paused': { en: 'Paused — waiting for your decision', ru: 'Пауза — ждём вашего решения' },
  'workOrder.blocking': { en: 'Waiting for your decision', ru: 'Ждём вашего решения' },
  'workOrder.btn.pause': { en: 'Pause', ru: 'Пауза' },
  'workOrder.ph.notes': {
    en: 'Notes or corrections for the agent (optional for Accept, required for Revise)',
    ru: 'Заметки или правки для агента (для «Принять» — необязательно, для «Доработать» — обязательно)'
  },
  'workOrder.approved': { en: '✓ Work order approved', ru: '✓ План работы одобрен' },
  'workOrder.rejectedAssumptions': { en: '{n} assumption(s) rejected', ru: 'отклонено условий: {n}' },
  'workOrder.deviation': { en: 'Blocked', ru: 'Заблокировано' },
  'workOrder.reason.no_work_order': { en: 'no work order declared yet', ru: 'план работы ещё не объявлен' },
  'workOrder.reason.rejected': { en: 'work order was rejected', ru: 'план работы отклонён' },
  'workOrder.reason.undeclared_tool': { en: 'tool not in the work order', ru: 'инструмента нет в плане работы' },
  'workOrder.reason.no_active_step': { en: 'no step marked in progress', ru: 'ни один шаг не отмечен как выполняемый' },
  'workStep.title': { en: 'Step check', ru: 'Проверка шага' },
  'workStep.step': { en: 'Step', ru: 'Шаг' },
  'workStep.sends': { en: 'sends', ru: 'отправит' },
  'workStep.expects': { en: 'expects', ru: 'ожидает' },
  'workStep.found': { en: 'found', ru: 'найдено' },
  'workStep.sent': { en: 'Sent', ru: 'Отправлено' },
  'workStep.expected': { en: 'Expected', ru: 'Ожидалось' },
  'workStep.foundTitle': { en: 'Found', ru: 'Найдено' },
  'workStep.calls': { en: 'Calls (recorded by the system)', ru: 'Вызовы (записаны системой)' },
  'workStep.noCalls': {
    en: 'No tool calls were recorded for this step',
    ru: 'Для этого шага не записано ни одного вызова инструмента',
  },
  'workStep.callError': { en: 'error', ru: 'ошибка' },
  'workStep.history': { en: 'Earlier rounds', ru: 'Предыдущие раунды' },
  'workStep.review.accepted': { en: 'step accepted', ru: 'шаг принят' },
  'workStep.review.revise': { en: 'sent back', ru: 'на доработке' },
  'workStep.review.rejected': { en: 'stopped', ru: 'остановлено' },
  'workStep.countdown': { en: 'Accepted automatically in {s} s', ru: 'Автоматическое принятие через {s} с' },
  'workStep.btn.redo': { en: 'Redo the step', ru: 'Переделать шаг' },
  'workStep.btn.stop': { en: 'Stop the work', ru: 'Остановить работу' },
  'workStep.ph.notes': {
    en: 'What to fix or check (required to redo the step)…',
    ru: 'Что исправить или проверить (обязательно, чтобы переделать шаг)…',
  },
  'workStep.accepted': { en: '✓ Step {step} accepted', ru: '✓ Шаг {step} принят' },
  'workStep.sentBack': { en: '↺ Step {step} sent back', ru: '↺ Шаг {step} отправлен на доработку' },
  'workStep.stopped': { en: '✗ Work stopped at step {step}', ru: '✗ Работа остановлена на шаге {step}' },

  // ── Work Report cards (hitl.js) ──
  'hitl.msg.workReport': {
    en: 'Agent {agent} reports what it did. Check the findings against its work order.',
    ru: 'Агент «{agent}» отчитался о работе. Сверьте находки с планом работы.'
  },
  'hitl.via.workReport': { en: 'work report (result after acting)', ru: 'отчёт агента (результат после работы)' },
  'workReport.title': { en: 'Work Report', ru: 'Отчёт агента' },
  'workReport.round': { en: 'round {n}', ru: 'раунд {n}' },
  'workReport.summary': { en: 'Summary', ru: 'Итог' },
  'workReport.finalAnswer': {
    en: 'The agent finished without a report — its final answer:',
    ru: 'Агент завершил работу без отчёта — его итоговый ответ:'
  },
  'workReport.findings': { en: 'Findings', ru: 'Находки' },
  'workReport.findingsHint': {
    en: 'Tick the findings that are wrong — the agent will recheck them on rework.',
    ru: 'Отметьте неверные находки — при доработке агент перепроверит их.'
  },
  'workReport.markWrong': { en: 'Wrong', ru: 'Неверно' },
  'workReport.noEvidence': { en: 'no evidence given', ru: 'нет подтверждения' },
  'workReport.confidence.high': { en: 'high confidence', ru: 'высокая уверенность' },
  'workReport.confidence.medium': { en: 'medium confidence', ru: 'средняя уверенность' },
  'workReport.confidence.low': { en: 'low confidence', ru: 'низкая уверенность' },
  'workReport.verdict.met': { en: 'met', ru: 'выполнен' },
  'workReport.verdict.partial': { en: 'partially met', ru: 'выполнен частично' },
  'workReport.verdict.not_met': { en: 'not met', ru: 'не выполнен' },
  'workReport.outcome': { en: 'Expected vs actual', ru: 'Ожидание и факт' },
  'workReport.actual': { en: 'Actual outcome', ru: 'Фактический результат' },
  'workReport.steps': { en: 'Steps: plan vs done', ru: 'Шаги: план и факт' },
  'workReport.artifacts': { en: 'Artifacts', ru: 'Артефакты' },
  'workReport.kind.file': { en: 'file', ru: 'файл' },
  'workReport.kind.dataset': { en: 'dataset', ru: 'датасет' },
  'workReport.kind.graph_node': { en: 'graph node', ru: 'узел графа' },
  'workReport.kind.link': { en: 'link', ru: 'ссылка' },
  'workReport.kind.other': { en: 'other', ru: 'другое' },
  'workReport.journal': { en: 'Journal (recorded by the system)', ru: 'Журнал (записан системой)' },
  'workReport.sideEffectsDone': { en: 'Side effects performed:', ru: 'Выполненные побочные эффекты:' },
  'workReport.amendments': { en: 'Amendments', ru: 'Поправки к плану' },
  'workReport.deviations': { en: 'Blocked calls', ru: 'Заблокированные вызовы' },
  'workReport.warn.no_report': {
    en: 'The agent finished without a report: only its answer and the journal are shown.',
    ru: 'Агент завершил работу без отчёта: показаны только его ответ и журнал.'
  },
  'workReport.warn.open_steps': { en: 'Steps not closed: {steps}', ru: 'Незакрытые шаги: {steps}' },
  'workReport.warn.done_not_met': { en: 'Done criteria: {verdict}', ru: 'Критерий готовности: {verdict}' },
  'workReport.warn.findings_without_evidence': {
    en: 'Findings without evidence: {findings}', ru: 'Находки без подтверждения: {findings}'
  },
  'workReport.warn.unreviewed_steps': {
    en: 'Steps finished without your check: {steps}',
    ru: 'Шаги, завершённые без вашей проверки: {steps}',
  },
  'workReport.warn.deviations': { en: 'Blocked calls during the run: {count}', ru: 'Заблокированных вызовов за работу: {count}' },
  'workReport.countdown': { en: 'Accepted automatically in {s} s', ru: 'Автоматическое принятие через {s} с' },
  'workReport.btn.rework': { en: 'Send back for rework', ru: 'На доработку' },
  'workReport.btn.returnParent': { en: 'Return to parent', ru: 'Вернуть родителю' },
  'workReport.ph.notes': {
    en: 'What is wrong or missing (optional for Accept, required for rework)',
    ru: 'Что неверно или чего не хватает (для «Принять» — необязательно, для доработки — обязательно)'
  },
  'workReport.accepted': { en: '✓ Work report accepted', ru: '✓ Отчёт принят' },
  'workReport.sentBack': { en: '↺ Sent back for rework', ru: '↺ Отправлено на доработку' },
  'workReport.rejected': { en: '✗ Work report rejected', ru: '✗ Отчёт отклонён' },
  'workReport.disputedCount': { en: '{n} finding(s) marked wrong', ru: 'неверных находок: {n}' },

  // ── HITL research-frame form (rendered at runtime, keyed by currentLang) ──
  'hitl.form.sidebarTitle': { en: 'Research Frame', ru: 'Рамка исследования' },
  'hitl.form.sidebarHint': {
    en: 'Fill in the form in the chat. The agent fills the empty fields itself.',
    ru: 'Заполните форму в чате. Пустые поля агент заполнит сам.'
  },
  'hitl.form.title': { en: 'Research Frame', ru: 'Рамка исследования' },
  'hitl.form.notSet': { en: 'not set', ru: 'не задано' },
  'hitl.form.placeholderFallback': {
    en: 'Leave empty and the agent will fill in a working value',
    ru: 'Оставьте пустым, чтобы агент подставил рабочее значение'
  },
  'hitl.form.save': { en: 'Save Frame', ru: 'Сохранить рамку' },
  'hitl.form.skip': { en: 'Skip (agent decides)', ru: 'Пропустить (агент решит)' },
  'hitl.form.saved': {
    en: '✓ Frame saved ({n} fields set by the operator)',
    ru: '✓ Рамка сохранена ({n} поле(й) заданы оператором)'
  },
  'hitl.form.skipped': {
    en: '→ Frame skipped — the agent will fill in the values',
    ru: '→ Рамка пропущена — агент подставит значения'
  },

  // Internal-loop review request (agent name replaces {agent}).
  'hitl.internalLoop': {
    en: "Agent {agent} proposes its result. Please review.",
    ru: "Агент «{agent}» предлагает свой результат. Проверьте его."
  },

  // ── HITL request card (built at render time from agent_name / invoked_via / trigger) ──
  'hitl.title': { en: 'Human-In-The-Loop Required', ru: 'Требуется решение человека' },
  'hitl.titleShort': { en: 'HITL Required', ru: 'Нужно решение' },
  'hitl.viaLabel': { en: 'Invoked via', ru: 'Причина вызова' },
  'hitl.msg.beforeTool': {
    en: "Agent {agent} is about to execute tool {tool}. Approve execution?",
    ru: "Агент «{agent}» собирается выполнить инструмент {tool}. Разрешить выполнение?"
  },
  'hitl.msg.afterAgent': {
    en: "Agent {agent} proposes the following output. Please review.",
    ru: "Агент «{agent}» предлагает следующий результат. Проверьте его."
  },
  'hitl.msg.beforeAgent': {
    en: "Agent {agent} is about to start. Approve?",
    ru: "Агент «{agent}» собирается начать работу. Разрешить?"
  },
  'hitl.msg.bashCommand': {
    en: "Agent {agent} wants to run a command that is outward-facing or hard to reverse. Approve execution?",
    ru: "Агент «{agent}» хочет выполнить команду с внешними или необратимыми последствиями. Разрешить выполнение?"
  },
  'hitl.via.beforeTool': {
    en: 'confirmation before running tool «{tool}»',
    ru: 'подтверждение запуска инструмента «{tool}»'
  },
  'hitl.via.afterAgent': {
    en: 'agent output review',
    ru: 'проверка результата агента'
  },
  'hitl.via.beforeAgent': {
    en: 'confirmation before agent start',
    ru: 'подтверждение запуска агента'
  },
  'hitl.via.bashCommand': {
    en: 'system command check (rule: {rule})',
    ru: 'проверка системной команды (правило: {rule})'
  },
  'hitl.via.callback': { en: 'automatic check', ru: 'автоматическая проверка' },
  'hitl.via.tool': { en: 'direct request from agent «{tool}»', ru: 'агент запрашивает подтверждение' },
  'hitl.via.internalLoop': { en: 'intermediate result review', ru: 'согласование промежуточного результата' },
  'hitl.via.requestInput': { en: 'request for user input', ru: 'запрос данных от пользователя' },
  'hitl.via.downloadCancel': { en: 'dataset download cancelled', ru: 'отмена загрузки датасета' },
  'hitl.via.unknown': { en: 'not specified', ru: 'не указан' },
  'hitl.block.toolCall': { en: 'Tool call', ru: 'Вызов инструмента' },
  'hitl.block.command': { en: 'Command', ru: 'Команда' },
  'hitl.block.userQuery': { en: 'User query', ru: 'Запрос пользователя' },
  'hitl.block.output': { en: 'Proposed output', ru: 'Предлагаемый результат' },
  'hitl.reviseEmpty': {
    en: 'Type your corrections in the field above first.',
    ru: 'Сначала напишите правки в поле выше.'
  },
  'hitl.sendUnavailable': {
    en: 'The decision was not sent: the server connection is unavailable.',
    ru: 'Решение не отправлено: нет соединения с сервером.'
  },
  'hitl.btn.accept': { en: 'Accept', ru: 'Принять' },
  'hitl.btn.acceptResult': { en: 'Accept result', ru: 'Принять результат' },
  'hitl.btn.sendRevise': { en: 'Send for revision', ru: 'Отправить на доработку' },
  'hitl.btn.rejectAsk': { en: 'Reject…', ru: 'Отклонить…' },
  'hitl.btn.rejectConfirm': { en: 'Click again to reject', ru: 'Нажмите ещё раз, чтобы отклонить' },
  'hitl.fb.label': { en: 'Corrections for the agent', ru: 'Правки для агента' },
  'hitl.fb.optional': { en: 'optional', ru: 'необязательно' },
  'hitl.fb.inputLabel': { en: 'Your answer to the agent', ru: 'Ваш ответ агенту' },
  'hitl.fb.replyLabel': { en: 'Or write your own answer', ru: 'Или напишите свой ответ' },
  'hitl.hintSubmit': { en: 'send', ru: 'отправить' },
  'hitl.btn.reject': { en: 'Reject', ru: 'Отклонить' },
  'hitl.btn.revise': { en: 'Revise', ru: 'Доработать' },
  'hitl.btn.reply': { en: 'Reply', ru: 'Ответить' },
  'hitl.btn.send': { en: 'Send', ru: 'Отправить' },
  'hitl.btn.openRoadmap': { en: 'Open Roadmap', ru: 'Открыть план' },
  'hitl.answerInChat': { en: 'Answer in the chat card.', ru: 'Ответьте в карточке в чате.' },
  'hitl.ph.input': { en: 'For example: use the 2024 data only…', ru: 'Например: используй только данные за 2024 год…' },
  'hitl.ph.reply': { en: 'Your answer to the question…', ru: 'Ваш ответ на вопрос…' },
  'hitl.ph.revise': { en: 'For example: add a source to every conclusion…', ru: 'Например: добавь источник к каждому выводу…' },

  // ── Common ──
  'common.cancel': { en: 'Cancel', ru: 'Отмена' },
  'common.close': { en: 'Close', ru: 'Закрыть' },
  'common.loading': { en: 'Loading…', ru: 'Загрузка…' },
  'common.showMore': { en: 'Show more', ru: 'Показать больше' },
  'common.showLess': { en: 'Show less', ru: 'Скрыть' },
  'common.copy': { en: 'Copy', ru: 'Копировать' },
  'common.errorPrefix': { en: 'Error: {error}', ru: 'Ошибка: {error}' },

  // ── Top bar ──
  'topbar.idle': { en: 'No active run', ru: 'Нет активного запуска' },
  'topbar.processing': { en: 'Running', ru: 'Выполняется' },
  'topbar.waiting': { en: 'Waiting for your decision', ru: 'Ждёт вашего решения' },
  'topbar.failed': { en: 'Failed', ru: 'Ошибка' },
  'topbar.offline': { en: 'No connection', ru: 'Нет связи' },
  'topbar.events': { en: '{count} events', ru: 'Событий: {count}' },
  'topbar.elapsed': { en: 'Elapsed', ru: 'Идёт' },

  // ── Tooltips (sidebar & header) ──
  'nav.registerUser': { en: 'Register user', ru: 'Зарегистрировать пользователя' },
  'nav.renameSession': { en: 'Rename session', ru: 'Переименовать сессию' },
  'nav.saveSession': { en: 'Save session to disk', ru: 'Сохранить сессию на диск' },
  'nav.exportSession': { en: 'Export session (.zip)', ru: 'Экспорт сессии (.zip)' },
  'nav.importSession': { en: 'Import session', ru: 'Импорт сессии' },
  'nav.restoreSession': { en: 'Restore saved session', ru: 'Восстановить сохранённую сессию' },
  'nav.newSession': { en: 'New session', ru: 'Новая сессия' },
  'nav.sidebarShow': { en: 'Show sidebar', ru: 'Показать панель' },
  'nav.sidebarHide': { en: 'Hide sidebar', ru: 'Скрыть панель' },
  'chat.clearView': { en: 'Clear current view (history is preserved)', ru: 'Очистить текущий вид (история сохраняется)' },
  'chat.attach': { en: 'Add context', ru: 'Добавить контекст' },
  'chat.stop': { en: 'Stop agents', ru: 'Остановить агентов' },

  // ── Chat runtime ──
  'chat.system': { en: 'System', ru: 'Система' },
  'chat.you': { en: 'You', ru: 'Вы' },
  'chat.result': { en: 'Result', ru: 'Результат' },
  'chat.showFull': { en: 'Show full output', ru: 'Показать полностью' },
  'chat.collapse': { en: 'Collapse', ru: 'Свернуть' },
  'chat.sandboxOpenActive': { en: 'Open Active CoderSandbox: {url}', ru: 'Открыть активную песочницу: {url}' },
  'chat.sandboxActive': { en: 'Sandbox active: {url}', ru: 'Песочница активна: {url}' },
  'chat.sandboxOpen': { en: 'Open CoderSandbox: {url}', ru: 'Открыть песочницу: {url}' },
  'chat.sandboxStandby': { en: 'Sandbox standby: {url}', ru: 'Песочница в ожидании: {url}' },
  'chat.detachDataset': { en: 'Detach dataset', ru: 'Открепить датасет' },

  // ── Telemetry & metrics ──
  'telemetry.waiting': { en: 'Waiting for connection…', ru: 'Ожидание подключения…' },
  'telemetry.live': { en: 'Live', ru: 'Live' },
  'metrics.noCalls': { en: 'no model calls yet', ru: 'вызовов модели пока не было' },
  'metrics.calls': { en: 'calls', ru: 'выз.' },
  'metrics.tokens': { en: 'tok', ru: 'ток.' },
  'metrics.unpriced': {
    en: '{count} call(s) on a model with no known price — total is a floor',
    ru: '{count} вызов(а) модели без известной цены — итог занижен'
  },

  // ── Activity rail runtime hints ──
  'rail.hintCalls': { en: '{count} tool call(s)', ru: 'вызовов инструментов: {count}' },
  'rail.hintDelegated': { en: 'delegated', ru: 'делегировано' },
  'rail.hintRunning': { en: 'Running', ru: 'Выполняется' },
  'rail.toolFinished': { en: '{done}/{calls} finished', ru: 'завершено {done}/{calls}' },
  'rail.toolErrors': { en: '{count} error(s)', ru: 'ошибок: {count}' },
  'rail.delegatedTitle': { en: 'Delegated', ru: 'Делегировано' },

  // ── HITL ──
  'hitl.sidebarHint': { en: 'Answer in the chat card.', ru: 'Ответьте в карточке в чате.' },
  'hitl.required': { en: 'HITL Required', ru: 'Требуется подтверждение' },
  'hitl.requiredLong': { en: 'Human-In-The-Loop Required', ru: 'Требуется подтверждение человеком' },
  'hitl.proposedOutput': { en: 'Proposed output', ru: 'Предложенный результат' },
  'hitl.accept': { en: 'Accept', ru: 'Принять' },
  'hitl.reject': { en: 'Reject', ru: 'Отклонить' },
  'hitl.revise': { en: 'Revise', ru: 'На доработку' },
  'hitl.send': { en: 'Send', ru: 'Отправить' },
  'hitl.reply': { en: 'Reply', ru: 'Ответить' },
  'hitl.openRoadmap': { en: 'Open Roadmap', ru: 'Открыть план' },
  'hitl.placeholderInput': { en: 'Enter instructions for the agent…', ru: 'Введите инструкции для агента…' },
  'hitl.placeholderAnswer': { en: 'Your answer to the question — then "Reply"', ru: 'Ваш ответ на вопрос — затем «Ответить»' },
  'hitl.placeholderRevise': { en: 'Edits for the agent — then Revise', ru: 'Правки для агента — затем «На доработку»' },
  'hitl.emptyEdits': { en: 'Enter your edits in the field above, then press Revise.', ru: 'Введите правки в поле выше, затем нажмите «На доработку».' },
  'hitl.empty': { en: '(empty)', ru: '(пусто)' },
  'hitl.inputSent': { en: '💬 HITL Input: {feedback}', ru: '💬 HITL ввод: {feedback}' },
  'hitl.approved': { en: '✓ HITL Approved', ru: '✓ HITL подтверждено' },
  'hitl.rejected': { en: '✗ HITL Rejected', ru: '✗ HITL отклонено' },
  'hitl.revisionRequested': { en: '✎ HITL Revision requested: {feedback}', ru: '✎ HITL: запрошена доработка: {feedback}' },
  'hitl.timeoutMsg': {
    en: '⏱ HITL: no answer for {seconds} s — the proposal of agent {agent} was auto-approved, the pipeline continues.',
    ru: '⏱ HITL: нет ответа {seconds} с — предложение агента «{agent}» автоподтверждено, пайплайн продолжен.'
  },
  'hitl.source.mode_auto': { en: 'decided by auto mode', ru: 'решено режимом auto' },
  'hitl.source.timeout': { en: 'review timed out', ru: 'время ожидания истекло' },
  'hitl.source.system': { en: 'system decision', ru: 'системное решение' },

  // ── WebSocket system messages ──
  'ws.datasetAttached': {
    en: 'Dataset attached: {url}\nThe coder agent will pass it to the sandbox when a step needs that data.',
    ru: 'Датасет прикреплён: {url}\nАгент Coder передаст его в песочницу, когда шагу понадобятся эти данные.'
  },
  'ws.datasetDetached': { en: 'Dataset link detached.', ru: 'Ссылка на датасет откреплена.' },
  'ws.datasetRejected': { en: 'Dataset link rejected: {message}', ru: 'Ссылка на датасет отклонена: {message}' },
  'ws.reportLangRejected': { en: 'Report language rejected: {message}', ru: 'Язык отчёта отклонён: {message}' },

  // ── Identity modal ──
  'identity.title': { en: 'Local user', ru: 'Локальный пользователь' },
  'identity.subtitle': { en: 'Users and sessions are cleared when the service restarts.', ru: 'Пользователи и сессии очищаются при перезапуске сервиса.' },
  'identity.continueAs': { en: 'Continue as', ru: 'Продолжить как' },
  'identity.open': { en: 'Open', ru: 'Открыть' },
  'identity.registerLabel': { en: 'Register a new Nick', ru: 'Регистрация нового ника' },
  'identity.register': { en: 'Register', ru: 'Зарегистрировать' },
  'identity.nickPlaceholder': { en: 'User', ru: 'Пользователь' },
  'identity.enterNick': { en: 'Enter a nick.', ru: 'Введите ник.' },
  'identity.noUsers': { en: 'No local users', ru: 'Нет локальных пользователей' },
  'identity.noSessions': { en: 'No sessions', ru: 'Нет сессий' },

  // ── Session operations ──
  'sessions.titlePrompt': { en: 'Session title:', ru: 'Название сессии:' },
  'sessions.renamePrompt': { en: 'New session title:', ru: 'Новое название сессии:' },
  'sessions.createError': { en: 'Could not create session: {error}', ru: 'Не удалось создать сессию: {error}' },
  'sessions.hiddenMark': { en: '(hidden)', ru: '(скрыта)' },
  'sessions.hideError': { en: 'Could not change session visibility: {error}', ru: 'Не удалось изменить видимость сессий: {error}' },
  'sessions.hiddenOld': { en: 'Hidden sessions: {count}. Turn on “Show hidden sessions” in the session menu to see them.', ru: 'Скрыто сессий: {count}. Их можно показать через «Показывать скрытые» в меню сессии.' },
  'sessions.nothingToHide': { en: 'No old sessions to hide.', ru: 'Нет старых сессий, которые можно скрыть.' },
  'sessions.renameError': { en: 'Could not rename session: {error}', ru: 'Не удалось переименовать сессию: {error}' },
  'sessions.initError': { en: 'Failed to initialize local sessions: {error}', ru: 'Не удалось загрузить локальные сессии: {error}' },
  'sessions.noActiveExport': { en: 'No active session to export.', ru: 'Нет активной сессии для экспорта.' },
  'sessions.exporting': { en: '📥 Exporting session…', ru: '📥 Экспорт сессии…' },
  'sessions.exported': { en: '✅ Session exported: {filename}', ru: '✅ Сессия экспортирована: {filename}' },
  'sessions.exportFailed': { en: '❌ Export failed: {error}', ru: '❌ Ошибка экспорта: {error}' },
  'sessions.noActiveSave': { en: 'No active session to save.', ru: 'Нет активной сессии для сохранения.' },
  'sessions.saving': { en: '💾 Saving session to disk…', ru: '💾 Сохранение сессии на диск…' },
  'sessions.saved': { en: '✅ Session saved: {filename}', ru: '✅ Сессия сохранена: {filename}' },
  'sessions.saveFailed': { en: '❌ Save failed: {error}', ru: '❌ Ошибка сохранения: {error}' },
  'sessions.importing': { en: '📤 Importing session from {filename}…', ru: '📤 Импорт сессии из {filename}…' },
  'sessions.imported': { en: '✅ Session imported: {title}', ru: '✅ Сессия импортирована: {title}' },
  'sessions.mcpRebuilds': { en: ' (MCP rebuilds launched)', ru: ' (запущена пересборка MCP)' },
  'sessions.importFailed': { en: '❌ Import failed: {error}', ru: '❌ Ошибка импорта: {error}' },
  'sessions.restoring': { en: '📂 Restoring session from {filename}…', ru: '📂 Восстановление сессии из {filename}…' },
  'sessions.restored': { en: '✅ Session restored: {title}', ru: '✅ Сессия восстановлена: {title}' },
  'sessions.restoreFailed': { en: '❌ Restore failed: {error}', ru: '❌ Ошибка восстановления: {error}' },
  'sessions.downloadFailed': { en: '❌ Download failed: {error}', ru: '❌ Ошибка скачивания: {error}' },
  'sessions.deleteConfirm': { en: 'Delete saved session "{filename}"?', ru: 'Удалить сохранённую сессию «{filename}»?' },
  'sessions.deleted': { en: '🗑 Saved session deleted: {filename}', ru: '🗑 Сохранённая сессия удалена: {filename}' },
  'sessions.deleteFailed': { en: '❌ Delete failed: {error}', ru: '❌ Ошибка удаления: {error}' },
  'sessions.savedBy': { en: ' · by {nick}', ru: ' · от {nick}' },
  'sessions.loadFailed': { en: 'Failed to load saved sessions: {error}', ru: 'Не удалось загрузить сохранённые сессии: {error}' },

  // ── Saved sessions modal ──
  'saved.title': { en: 'Saved Sessions', ru: 'Сохранённые сессии' },
  'saved.subtitle': { en: 'Restore or manage previously saved sessions', ru: 'Восстановление и управление сохранёнными сессиями' },
  'saved.empty': { en: 'No saved sessions found.', ru: 'Сохранённые сессии не найдены.' },
  'saved.restore': { en: 'Restore', ru: 'Восстановить' },
  'saved.download': { en: 'Download', ru: 'Скачать' },
  'saved.delete': { en: 'Delete', ru: 'Удалить' },

  // ── MCP rebuild modal ──
  'mcpRebuild.title': { en: 'MCP Servers Found', ru: 'Найдены серверы MCP' },
  'mcpRebuild.subtitle': { en: 'This session contains MCP tool server builds', ru: 'Эта сессия содержит сборки MCP-серверов' },
  'mcpRebuild.explain': {
    en: '<strong>Rebuild</strong> will launch Docker builds for each server (takes ~10–30 min per server). <strong>Skip</strong> will import the session without starting the servers — build logs and metadata will still be available.',
    ru: '<strong>Пересборка</strong> запустит Docker-сборку каждого сервера (~10–30 минут на сервер). <strong>Пропустить</strong> импортирует сессию без запуска серверов — логи сборки и метаданные останутся доступными.'
  },
  'mcpRebuild.skip': { en: 'Skip', ru: 'Пропустить' },
  'mcpRebuild.rebuild': { en: 'Rebuild', ru: 'Пересобрать' },

  // ── Dataset modal & upload widget ──
  'dataset.menuItem': { en: 'Dataset link (.zip)', ru: 'Ссылка на датасет (.zip)' },
  'dataset.title': { en: 'Dataset link (for openhands)', ru: 'Ссылка на датасет (для openhands)' },
  'dataset.desc': {
    en: 'A direct http(s) URL of a .zip archive. The coder agent sees it and sends it to the sandbox, where it is unpacked, when a step needs that data.',
    ru: 'Прямая http(s) ссылка на .zip архив. Агент Coder видит её и отправляет в песочницу, где архив распаковывается, когда шагу нужны эти данные.'
  },
  'dataset.remove': { en: 'Remove', ru: 'Удалить' },
  'dataset.attach': { en: 'Attach', ru: 'Прикрепить' },
  'dataset.errEmpty': { en: 'Enter a link to a .zip archive.', ru: 'Введите ссылку на .zip архив.' },
  'dataset.errInvalid': { en: 'That is not a valid URL.', ru: 'Это недопустимый URL.' },
  'dataset.errProtocol': { en: 'The link must be an http(s) URL.', ru: 'Ссылка должна быть http(s) URL.' },
  'dataset.errNotZip': { en: 'The link must point to a .zip archive.', ru: 'Ссылка должна указывать на .zip архив.' },
  'dataset.notConnected': { en: 'Not connected — reconnect and try again.', ru: 'Нет подключения — переподключитесь и попробуйте снова.' },
  'dataset.upload.title': { en: 'Dataset upload', ru: 'Загрузка датасета' },
  'dataset.upload.details': { en: 'Details', ru: 'Подробности' },
  'dataset.upload.filename': { en: 'Filename:', ru: 'Файл:' },
  'dataset.upload.speed': { en: 'Speed:', ru: 'Скорость:' },
  'dataset.upload.eta': { en: 'ETA:', ru: 'Осталось:' },
  'dataset.upload.status': { en: 'Status:', ru: 'Статус:' },
  'dataset.uploading': { en: 'Uploading', ru: 'Загрузка' },
  'dataset.uploadingSandbox': { en: 'Uploading dataset to sandbox…', ru: 'Загрузка датасета в песочницу…' },

  // ── Roadmap modal ──
  'roadmap.title': { en: 'Roadmap & Execution Plan', ru: 'План и этапы выполнения' },
  'roadmap.subtitle': { en: 'Execution steps for the current session', ru: 'Этапы выполнения текущей сессии' },
  'roadmap.addTask': { en: 'Add Task', ru: 'Добавить задачу' },
  'roadmap.filter.all': { en: 'All', ru: 'Все' },
  'roadmap.searchPlaceholder': { en: 'Search tasks…', ru: 'Поиск задач…' },
  'roadmap.expandAllDesc': { en: 'Expand all descriptions', ru: 'Развернуть все описания' },
  'roadmap.collapseAllDesc': { en: 'Collapse all descriptions', ru: 'Свернуть все описания' },
  'roadmap.empty.none': { en: 'No tasks found in roadmap', ru: 'В плане нет задач' },
  'roadmap.empty.noMatch': { en: 'No tasks match current filter/search', ru: 'Нет задач по текущему фильтру/поиску' },
  'roadmap.empty.hint': { en: 'PlannerAgent generates the plan, or click "Add Task" to create one manually.', ru: 'PlannerAgent создаёт план автоматически, либо нажмите «Добавить задачу», чтобы создать её вручную.' },
  'roadmap.feedbackLabel': { en: 'Feedback / Revision Instructions for Planner', ru: 'Замечания / инструкции по доработке для планировщика' },
  'roadmap.feedbackPlaceholder': { en: 'Specify what should be changed or added (e.g. Add validation against off-targets, split task 2)…', ru: 'Укажите, что изменить или добавить (например: добавить проверку по офф-таргетам, разбить задачу 2)…' },
  'roadmap.saveChanges': { en: 'Save Changes', ru: 'Сохранить изменения' },
  'roadmap.revisePlan': { en: 'Revise Plan', ru: 'Отправить на доработку' },
  'roadmap.saveConfirm': { en: 'Save & Confirm', ru: 'Сохранить и подтвердить' },
  'roadmap.status.done': { en: 'Completed', ru: 'Завершено' },
  'roadmap.status.in_progress': { en: 'In Progress', ru: 'В работе' },
  'roadmap.status.error': { en: 'Failed', ru: 'Ошибка' },
  'roadmap.status.todo': { en: 'Pending', ru: 'Ожидает' },
  'roadmap.progress': { en: '{done} of {total} completed ({percent}%)', ru: 'Завершено {done} из {total} ({percent}%)' },
  'roadmap.tasksCount': { en: '{count} task(s)', ru: 'задач: {count}' },
  'roadmap.statTotal': { en: 'Total: {count}', ru: 'Всего: {count}' },
  'roadmap.statActive': { en: 'In progress: {count}', ru: 'В работе: {count}' },
  'roadmap.statDone': { en: 'Done: {count}', ru: 'Готово: {count}' },
  'roadmap.noPrereq': { en: 'No prerequisite', ru: 'Без зависимости' },
  'roadmap.dependsOn': { en: 'Depends on: {id}', ru: 'Зависит от: {id}' },
  'roadmap.deleteTask': { en: 'Delete task', ru: 'Удалить задачу' },
  'roadmap.editTask': { en: 'Edit task', ru: 'Редактировать задачу' },
  'roadmap.moveUp': { en: 'Move up', ru: 'Переместить вверх' },
  'roadmap.moveDown': { en: 'Move down', ru: 'Переместить вниз' },
  'roadmap.toggleStatus': { en: 'Click to toggle status', ru: 'Нажмите, чтобы изменить статус' },
  'roadmap.jumpToPrereq': { en: 'Jump to prerequisite task', ru: 'Перейти к задаче-зависимости' },
  'roadmap.fieldTitle': { en: 'Task Title', ru: 'Название задачи' },
  'roadmap.titlePlaceholder': { en: 'Enter task title…', ru: 'Введите название задачи…' },
  'roadmap.fieldDesc': { en: 'Description', ru: 'Описание' },
  'roadmap.descPlaceholder': { en: 'Specific instructions or details for this task…', ru: 'Конкретные инструкции или детали для этой задачи…' },
  'roadmap.fieldNotes': { en: 'Notes / Parameters (Optional)', ru: 'Заметки / параметры (необязательно)' },
  'roadmap.notesPlaceholder': { en: 'e.g. Max iterations: 3, temperature: 0.2', ru: 'например: макс. итераций: 3, температура: 0.2' },
  'roadmap.doneEditing': { en: 'Done', ru: 'Готово' },
  'roadmap.untitled': { en: 'Untitled task', ru: 'Задача без названия' },
  'roadmap.errEmptyTitle': { en: 'Task title cannot be empty.', ru: 'Название задачи не может быть пустым.' },
  'roadmap.errNoPrereq': { en: "Prerequisite task '{id}' not found in the plan.", ru: 'Задача-зависимость «{id}» не найдена в плане.' },
  'roadmap.confirmDelete': { en: 'Delete task {id}?', ru: 'Удалить задачу {id}?' },
  'roadmap.errNoPending': { en: 'No pending roadmap confirmation request found.', ru: 'Нет ожидающего запроса на подтверждение плана.' },
  'roadmap.syncing': { en: 'Syncing roadmap…', ru: 'Синхронизация плана…' },
  'roadmap.synced': { en: 'Roadmap synced.', ru: 'План синхронизирован.' },
  'roadmap.syncError': { en: 'Error syncing roadmap: {error}', ru: 'Ошибка синхронизации плана: {error}' },
  'roadmap.savingChanges': { en: 'Saving changes…', ru: 'Сохранение изменений…' },
  'roadmap.savedOk': { en: 'Roadmap saved successfully.', ru: 'План сохранён.' },
  'roadmap.saveError': { en: 'Error saving roadmap: {error}', ru: 'Ошибка сохранения плана: {error}' },
  'roadmap.errSave': { en: 'Failed to save roadmap', ru: 'Не удалось сохранить план' },
  'roadmap.errUnknown': { en: 'Unknown error', ru: 'Неизвестная ошибка' },
  'roadmap.sendingRevision': { en: 'Sending revision request…', ru: 'Отправка запроса на доработку…' },
  'roadmap.revisionSent': { en: 'Revision requested.', ru: 'Доработка запрошена.' },
  'roadmap.sentForRevision': { en: '✗ HITL Sent for Revision', ru: '✗ HITL отправлен на доработку' },
  'roadmap.savingConfirming': { en: 'Saving and confirming plan…', ru: 'Сохранение и подтверждение плана…' },
  'roadmap.savedConfirmed': { en: 'Roadmap saved and confirmed.', ru: 'План сохранён и подтверждён.' },

  // ── Experiment viewer ──
  'experiments.title': { en: 'Experiment Viewer', ru: 'Просмотр эксперимента' },
  'experiments.subtitle': { en: 'Live Tool Activity Feed', ru: 'Поток вызовов инструментов' },
  'experiments.expandAll': { en: 'Expand all', ru: 'Развернуть всё' },
  'experiments.collapseAll': { en: 'Collapse all', ru: 'Свернуть всё' },
  'experiments.clearFeed': { en: 'Clear feed', ru: 'Очистить ленту' },
  'experiments.empty': { en: 'No tool activity yet', ru: 'Вызовов инструментов пока нет' },
  'experiments.emptyHint': { en: 'Tool calls and results from agents will appear here in real time', ru: 'Вызовы инструментов и их результаты будут появляться здесь в реальном времени' },
  'experiments.calls': { en: 'calls', ru: 'вызовов' },
  'experiments.running': { en: 'running', ru: 'в работе' },
  'experiments.failed': { en: 'failed', ru: 'с ошибкой' },
  'experiments.delegates': { en: 'delegates', ru: 'делегирует' },
  'experiments.delegatedTask': { en: 'Task and answer', ru: 'Задача и ответ' },
  'experiments.runningDots': { en: 'running…', ru: 'выполняется…' },
  'experiments.args': { en: 'Arguments', ru: 'Аргументы' },
  'experiments.output': { en: 'Output', ru: 'Результат' },
  'experiments.effectiveArgs': { en: 'Arguments as run (after callbacks)', ru: 'Аргументы при запуске (после callback-ов)' },
  'experiments.stateInputs': { en: 'Read from state', ru: 'Прочитано из state' },
  'experiments.stateBadge': { en: 'state', ru: 'state' },
  'experiments.stateBadgeTitle': { en: 'The tool also read session state', ru: 'Инструмент также читал state сессии' },
  'experiments.error': { en: 'Error', ru: 'Ошибка' },
  'experiments.callNotRecorded': { en: '(the call itself was not recorded)', ru: '(сам вызов не был записан)' },
  'experiments.noArgs': { en: '(no arguments)', ru: '(нет аргументов)' },
  'experiments.emptyValue': { en: '(empty)', ru: '(пусто)' },
  'experiments.noValue': { en: '(none)', ru: '(нет)' },
  'experiments.emptyList': { en: '(empty list)', ru: '(пустой список)' },
  'experiments.waiting': { en: 'waiting for the result…', ru: 'ожидание результата…' },
  'experiments.truncated': { en: '… truncated (server preview cap)', ru: '… обрезано (лимит предпросмотра сервера)' },
  'experiments.retry': { en: 'Retry — {error}', ru: 'Повторить — {error}' },
  'experiments.loadFailed': { en: 'failed to load full result', ru: 'не удалось загрузить полный результат' },

  // ── Standalone graph page ──
  'graph.title': { en: 'CoScientist — Research Graph', ru: 'CoScientist — Граф исследований' },
  'graph.brand': { en: 'Graph', ru: 'Граф' },
  'graph.connecting': { en: 'connecting…', ru: 'подключение…' },
  'graph.live': { en: 'live', ru: 'live' },
  'graph.refresh': { en: 'refresh', ru: 'обновить' },
  'graph.backTrace': { en: '← session trace', ru: '← трассировка сессии' },
  'graph.view.label': { en: 'view', ru: 'вид' },
  'graph.view.research': { en: 'research', ru: 'исследования' },
  'graph.view.execution': { en: 'execution log', ru: 'лог выполнения' },
  'graph.turn.study': { en: 'study', ru: 'исследование' },
  'graph.turn.request': { en: 'request', ru: 'запрос' },
  'graph.choice.live': { en: 'live', ru: 'live' },
  'graph.choice.archived': { en: 'archived', ru: 'архивное' },
  'graph.choice.study': { en: 'study', ru: 'исследование' },
  'graph.status': { en: '{nodes} nodes · {edges} edges', ru: 'узлов: {nodes} · рёбер: {edges}' },
  'graph.age.justNow': { en: ' · last written just now', ru: ' · обновлено только что' },
  'graph.age.min': { en: ' · last written {mins} min ago', ru: ' · обновлено {mins} мин назад' },
  'graph.age.hours': { en: ' · last written {hours} h ago', ru: ' · обновлено {hours} ч назад' },
  'graph.age.date': { en: ' · last written {date}', ru: ' · обновлено {date}' },
  'graph.error': { en: 'error: {error}', ru: 'ошибка: {error}' },
  'graph.noSession': { en: 'Open the graph from an active CoScientist session.', ru: 'Откройте граф из активной сессии CoScientist.' },
  'graph.callNotFound': { en: 'call {call} not found in this session', ru: 'вызов {call} не найден в этой сессии' },
  'graph.count.calls': { en: '{count} call(s)', ru: 'вызовов: {count}' },
  'graph.count.item': { en: '{count} item(s)', ru: 'элементов: {count}' },
  'graph.count.field': { en: '{count} field(s)', ru: 'полей: {count}' },
  'graph.chars': { en: '{count} chars', ru: '{count} симв.' },
  // Research node type names (fallback for data without a server type_word)
  'graph.type.researchquestion': { en: 'Question', ru: 'Вопрос' },
  'graph.type.hypothesis': { en: 'Hypothesis', ru: 'Гипотеза' },
  'graph.type.evidence': { en: 'Evidence', ru: 'Данные' },
  'graph.type.conclusion': { en: 'Conclusion', ru: 'Вывод' },
  'graph.type.verificationmethod': { en: 'Method', ru: 'Метод' },
  'graph.type.confirmationcriteria': { en: 'Criteria', ru: 'Критерии' },
  'graph.type.tool': { en: 'Tool', ru: 'Инструмент' },
  'graph.type.resource': { en: 'Resource', ru: 'Ресурс' },
  'graph.type.empiricalbase': { en: 'Data source', ru: 'Источник данных' },
  'graph.type.constraint': { en: 'Constraint', ru: 'Ограничение' },
  'graph.type.codeartifact': { en: 'Code', ru: 'Код' },
  'graph.type.generateddata': { en: 'Dataset', ru: 'Набор данных' },
  'graph.type.report': { en: 'Report', ru: 'Отчёт' },
  'graph.type.publication': { en: 'Publication', ru: 'Публикация' },
  'graph.type.spec': { en: 'Spec', ru: 'Спецификация' },
  'graph.type.efficiencyjustification': { en: 'Efficiency', ru: 'Эффективность' },
  'graph.type.costmodel': { en: 'Cost model', ru: 'Модель стоимости' },
  'graph.type.efficiencymetric': { en: 'Metric', ru: 'Метрика' },
  'graph.type.framing': { en: 'Technical specification', ru: 'Техническое задание' },
  'graph.type.outcome': { en: 'Outcome', ru: 'Итог' },
  'graph.type.planstep': { en: 'Plan step', ru: 'Шаг плана' },
  'graph.plan.title': { en: 'Research plan', ru: 'План исследования' },
  'graph.plan.experiments': { en: 'Experiment plan', ru: 'План экспериментов' },
  'graph.type.experimenttask': { en: 'Experiment task', ru: 'Задача эксперимента' },
  'graph.card.elaborates': { en: 'for step:', ru: 'к шагу:' },
  'graph.edge.elaborates': { en: 'details', ru: 'детализирует' },
  // Research edge labels
  'graph.edge.motivates': { en: 'motivates', ru: 'мотивирует' },
  'graph.edge.tested_by': { en: 'tested by', ru: 'проверяется' },
  'graph.edge.requires': { en: 'requires', ru: 'требует' },
  'graph.edge.uses': { en: 'uses', ru: 'использует' },
  'graph.edge.consumes': { en: 'consumes', ru: 'потребляет' },
  'graph.edge.produces': { en: 'produces', ru: 'производит' },
  'graph.edge.supports': { en: 'supports', ru: 'поддерживает' },
  'graph.edge.refutes': { en: 'refutes', ru: 'опровергает' },
  'graph.edge.refines': { en: 'refines', ru: 'уточняет' },
  'graph.edge.based_on': { en: 'based on', ru: 'основано на' },
  'graph.edge.determines_sufficiency': { en: 'sufficiency for', ru: 'достаточность для' },
  'graph.edge.formulated_for': { en: 'criterion for', ru: 'критерий для' },
  'graph.edge.regulates': { en: 'regulates', ru: 'регулирует' },
  'graph.edge.constrains': { en: 'constrains', ru: 'ограничивает' },
  'graph.edge.derived_from': { en: 'derived from', ru: 'получено из' },
  'graph.edge.contextualizes': { en: 'frames', ru: 'контекст для' },
  'graph.edge.defines_scope': { en: 'scope of', ru: 'область' },
  'graph.edge.relates_to': { en: 'relates to', ru: 'относится к' },
  'graph.edge.applies_to': { en: 'applies to', ru: 'применяется к' },
  'graph.edge.conditional_successor': { en: 'if refuted, continue with', ru: 'при опровержении перейти к' },
  'graph.edge.via': { en: 'via', ru: 'через' },
  // Execution node kinds (panel subtitle)
  'graph.kind.goal': { en: 'request', ru: 'запрос' },
  'graph.kind.result': { en: 'answer', ru: 'ответ' },
  'graph.kind.agent': { en: 'agent', ru: 'агент' },
  'graph.kind.agent_call': { en: 'agent', ru: 'агент' },
  'graph.kind.tool_call': { en: 'tool call', ru: 'вызов инструмента' },
  'graph.kind.decision': { en: 'decision', ru: 'решение' },
  'graph.kind.system': { en: 'system', ru: 'система' },
  // Node hover tooltip (plain text, one line per item)
  'graph.tooltip.type': { en: 'type: {kind}', ru: 'тип: {kind}' },
  'graph.tooltip.source': { en: 'source: {agent}', ru: 'источник: {agent}' },
  'graph.tooltip.status': { en: 'status: {status}', ru: 'статус: {status}' },
  'graph.tooltip.toolCalls': { en: 'tool calls: {count}', ru: 'вызовов инструментов: {count}' },
  // Detail / inspector panel
  'graph.detail.provenance': { en: 'provenance — produced by', ru: 'происхождение — получено из' },
  // Who took part in a node. A basis is not decoration: `assignee` is what the
  // plan intended and every other basis is something the system watched happen,
  // so the wording has to keep a reader from reading an intention as a fact.
  'graph.contrib.title': { en: 'participants', ru: 'участники' },
  // A node's own write-up: what it establishes, what it rests on, who took
  // part. Distinct from `graph.summary.*`, which is one AGENT's account of one
  // run — the two are different documents about different things.
  'graph.nodeReport.title': { en: 'report on this node', ru: 'отчёт по узлу' },
  'graph.nodeReport.ask': { en: 'write the report', ru: 'составить отчёт' },
  'graph.nodeReport.writing': { en: 'writing…', ru: 'пишется…' },
  'graph.nodeReport.again': { en: 'rewrite', ru: 'переписать' },
  'graph.nodeReport.stale': { en: 'the node has moved since', ru: 'узел с тех пор изменился' },
  'graph.nodeReport.auto': { en: 'write node reports', ru: 'писать отчёты по узлам' },
  'graph.basis.commit': { en: 'wrote this', ru: 'записал' },
  'graph.basis.status': { en: 'moved its status', ru: 'сменил статус' },
  'graph.basis.delegation': { en: 'delegated the work', ru: 'делегировал работу' },
  'graph.basis.provenance': { en: 'made the call behind it', ru: 'сделал вызов, давший это' },
  'graph.basis.work_order': { en: 'took the step (work order)', ru: 'взял шаг (план работы)' },
  'graph.basis.route': { en: 'ran it', ru: 'выполнил' },
  'graph.basis.assignee': { en: 'planned to — not observed', ru: 'назначен планом — не подтверждено' },
  'graph.detail.openInLog': { en: 'open this call in the execution log', ru: 'открыть этот вызов в логе выполнения' },
  'graph.detail.execLog': { en: '↗ execution log', ru: '↗ лог выполнения' },
  'graph.detail.says': { en: 'what this says', ru: 'содержание' },
  'graph.detail.details': { en: 'details', ru: 'детали' },
  'graph.detail.by': { en: 'by {agent}', ru: 'от {agent}' },
  'graph.detail.truncated': {
    en: 'recorded up to {kept} characters · {dropped} more were not kept',
    ru: 'записано {kept} символов · ещё {dropped} не сохранено'
  },
  'graph.detail.cutShort': {
    en: 'recorded value was cut short · shown up to the last complete field',
    ru: 'записанное значение обрезано · показано до последнего целого поля'
  },
  'graph.detail.arguments': { en: 'arguments', ru: 'аргументы' },
  'graph.detail.plan': { en: 'plan', ru: 'план' },
  'graph.detail.decision': { en: 'decision', ru: 'решение' },
  'graph.detail.outcome': { en: 'outcome', ru: 'итог' },
  'graph.detail.note': { en: 'note', ru: 'примечание' },
  'graph.detail.none': { en: 'none', ru: 'нет' },
  'graph.detail.result': { en: 'result', ru: 'результат' },
  'graph.detail.files': { en: 'files', ru: 'файлы' },
  'graph.detail.call': { en: 'call', ru: 'вызов' },
  'graph.detail.artifacts': { en: 'artifacts', ru: 'артефакты' },
  'graph.detail.agent': { en: 'agent', ru: 'агент' },
  'graph.detail.request': { en: 'request', ru: 'запрос' },
  'graph.detail.answer': { en: 'answer', ru: 'ответ' },
  'graph.detail.whatAsked': { en: 'what was asked', ru: 'что было запрошено' },
  'graph.detail.report': { en: 'report', ru: 'отчёт' },
  'graph.detail.task': { en: 'task', ru: 'задача' },
  'graph.detail.ioDelegation': {
    en: 'task and report come from the delegation that started this run',
    ru: 'задача и отчёт взяты из делегирования, запустившего этот запуск'
  },
  'graph.detail.ioRequest': {
    en: 'task and report come from the request this agent served and its answer',
    ru: 'задача и отчёт взяты из запроса, который обслуживал агент, и его ответа'
  },
  'graph.detail.taskReport': { en: 'task · report', ru: 'задача · отчёт' },
  'graph.detail.notRecorded': { en: 'not recorded for this agent', ru: 'для этого агента ничего не записано' },
  'graph.detail.toolCalls': { en: 'tool calls', ru: 'вызовы инструментов' },
  'graph.summary.title': { en: 'summary', ru: 'сводка' },
  'graph.tab.report': { en: 'Report', ru: 'Отчёт' },
  'graph.tab.more': { en: 'Details', ru: 'Дополнительно' },
  'graph.summary.loading': { en: 'writing the summary…', ru: 'готовлю сводку…' },
  'graph.summary.waiting': { en: 'the agent is still running; the summary is written once it finishes', ru: 'агент ещё работает; сводка появится, когда он закончит' },
  'graph.summary.now': { en: 'write it now', ru: 'написать сейчас' },
  'graph.summary.stale': { en: 'the trace has changed since this summary was written', ru: 'трассировка изменилась после того, как сводка была написана' },
  'graph.summary.again': { en: 'regenerate', ru: 'заново' },
  'graph.summary.by': { en: 'by {model}', ru: 'модель: {model}' },
  'graph.summary.failed': { en: 'could not write the summary: {error}', ru: 'не удалось подготовить сводку: {error}' },
  'graph.detail.failedCount': { en: ' · {count} failed', ru: ' · ошибок: {count}' },
  'graph.detail.noneRecorded': { en: 'none recorded', ru: 'не записано' },
  'graph.detail.empty': { en: 'nothing recorded yet', ru: 'пока ничего не записано' },
  'graph.detail.runOf': { en: 'run {run} of {runs}', ru: 'запуск {run} из {runs}' },
  'graph.detail.started': { en: 'started {at}', ru: 'начат в {at}' },

  // ── Research graph: the story a card tells ──
  // Everything below is written into the page by JS after load, so each one is
  // fetched with t(...) rather than data-i18n (applyLanguage runs once, on
  // DOMContentLoaded). The server sends CODES for these, never sentences —
  // otherwise an English reader gets Russian banners.
  'graph.why.title': { en: 'why', ru: 'почему' },
  'graph.why.missing': {
    en: 'no reason was recorded for this outcome',
    ru: 'причина этого исхода не записана'
  },
  'graph.card.criterion': { en: 'criterion:', ru: 'критерий:' },
  // What a method answers. Without it a reader counts four method cards
  // and cannot tell why there are four.
  'graph.card.tests': { en: 'tests:', ru: 'проверяет:' },
  'graph.origin.question': { en: 'the question as first stated', ru: 'исходная формулировка вопроса' },
  'graph.origin.modified': { en: 'modified: {reason}', ru: 'модификация: {reason}' },
  'graph.origin.refined': { en: 'refined: {reason}', ru: 'уточнение: {reason}' },
  'graph.origin.retried': { en: 'retried: {reason}', ru: 'повторная проверка: {reason}' },
  'graph.origin.other': { en: 'replaces an earlier hypothesis', ru: 'заменяет прежнюю гипотезу' },
  'graph.count.attached': { en: '{count} attached', ru: 'вложений: {count}' },
  'graph.chips.tools': { en: 'tools', ru: 'инструменты' },
  'graph.attach.title': { en: 'attached', ru: 'вложения' },
  'graph.history.title': { en: 'how it got here', ru: 'как дошло до этого' },
  'graph.history.initial': { en: 'created as {status}', ru: 'создан как {status}' },
  'graph.history.entry': { en: '{from} → {to} · {source}', ru: '{from} → {to} · {source}' },
  'graph.links.title': { en: 'connections', ru: 'связи' },
  'graph.edge.realises': { en: 'carries out', ru: 'выполняет шаг' },
  'graph.edge.supersedes.refuted': { en: '✗ refuted → modified', ru: '✗ опровергнута → модифицирована' },
  'graph.edge.supersedes.confirmed': { en: '✓ confirmed → refined', ru: '✓ подтверждена → уточнена' },
  'graph.edge.supersedes.inconclusive': { en: '≈ unsettled → retried', ru: '≈ без ответа → перепроверена' },
  'graph.edge.supersedes.other': { en: 'superseded by', ru: 'заменена на' },
  'graph.edge.frames': { en: 'frames', ru: 'задаёт рамку' },
  'graph.edge.concludes': { en: 'sums up', ru: 'подводит итог' },
  'graph.edge.via': { en: 'via {node}', ru: 'через {node}' },
  'graph.counters.confirmed': { en: 'confirmed', ru: 'подтверждено' },
  'graph.counters.refuted': { en: 'refuted', ru: 'опровергнуто' },
  'graph.counters.under_verification': { en: 'under verification', ru: 'на проверке' },
  'graph.counters.formulated': { en: 'formulated', ru: 'сформулировано' },
  'graph.counters.inconclusive': { en: 'inconclusive', ru: 'без ответа' },
  'graph.counters.postponed': { en: 'postponed', ru: 'отложено' },
  'graph.header.meta': {
    en: '{branches} branch(es) · {iterations} iteration(s)',
    ru: 'веток: {branches} · итераций: {iterations}'
  },
  // Shown instead of the branch tally while a study has no hypotheses to
  // branch: when the record was written, and over how long.
  'graph.header.span': {
    en: 'recorded from {from}, over {mins} min',
    ru: 'записано с {from}, за {mins} мин'
  },
  // Node attribute codes. The server sends the code (see store._fields) and the
  // panel captions it here, so one record reads correctly in both languages.
  // Keys are the attribute names from schema.NODE_TYPES.attr_docs.
  'graph.field.formulation': { en: 'Statement', ru: 'Формулировка' },
  'graph.field.rationale': { en: 'Why', ru: 'Обоснование' },
  'graph.field.priority': { en: 'Priority', ru: 'Приоритет' },
  'graph.field.content': { en: 'Finding', ru: 'Результат' },
  'graph.field.subtype': { en: 'Kind', ru: 'Вид' },
  'graph.field.reliability': { en: 'Confidence', ru: 'Достоверность' },
  'graph.field.source_ref': { en: 'Source', ru: 'Источник' },
  'graph.field.measured_on': { en: 'Measured on', ru: 'На чём измерено' },
  'graph.field.synthesis': { en: 'Conclusion', ru: 'Вывод' },
  'graph.field.validity_bounds': { en: 'Limits of validity', ru: 'Границы применимости' },
  'graph.field.new_question': { en: 'Opens next', ru: 'Открывает вопрос' },
  'graph.field.procedure': { en: 'Procedure', ru: 'Процедура' },
  'graph.field.limits': { en: 'Limits', ru: 'Ограничения' },
  'graph.field.limitations': { en: 'Known weaknesses', ru: 'Известные слабости' },
  'graph.field.threshold': { en: 'Threshold', ru: 'Порог' },
  'graph.field.tools': { en: 'Tools', ru: 'Инструменты' },
  'graph.field.how_established': { en: 'How it was established', ru: 'Как установлено' },
  'graph.field.against_criteria': { en: 'Against the criteria', ru: 'По критериям' },
  'graph.field.open_questions': { en: 'What remains', ru: 'Что осталось' },
  'graph.field.title': { en: 'Step', ru: 'Шаг' },
  'graph.field.reproducibility': { en: 'Reproducibility', ru: 'Воспроизводимость' },
  'graph.field.confirmations_needed': { en: 'Confirmations needed', ru: 'Нужно подтверждений' },
  'graph.field.metric': { en: 'Metric', ru: 'Метрика' },
  'graph.field.criterion': { en: 'Criterion', ru: 'Критерий' },
  'graph.field.value': { en: 'Value', ru: 'Значение' },
  'graph.field.name': { en: 'Name', ru: 'Название' },
  'graph.field.location': { en: 'Where', ru: 'Где' },
  'graph.field.tool_type': { en: 'Type', ru: 'Тип' },
  'graph.field.base_type': { en: 'Type', ru: 'Тип' },
  'graph.field.method_type': { en: 'Type', ru: 'Тип' },
  'graph.field.volume': { en: 'Size', ru: 'Объём' },
  'graph.field.resource_type': { en: 'Resource', ru: 'Ресурс' },
  'graph.field.remaining': { en: 'Remaining', ru: 'Осталось' },
  'graph.field.limit': { en: 'Total', ru: 'Всего' },
  // «Основание и приёмка» — поля ТЗ по ГОСТ 19.201-78. Они ложатся атрибутами
  // корневого вопроса, значит попадают в карточку «Постановка»; без записи здесь
  // читатель увидел бы «basis document».
  'graph.field.basis_document': { en: 'Basis for the work', ru: 'Основание для работы' },
  'graph.field.customer': { en: 'Customer', ru: 'Заказчик' },
  'graph.field.topic_name': { en: 'Name of the topic', ru: 'Наименование темы' },
  'graph.field.deliverables': { en: 'Documents delivered', ru: 'Отчётные документы' },
  'graph.field.stages': { en: 'Stages and deadlines', ru: 'Этапы и сроки' },
  'graph.field.acceptance': { en: 'Acceptance procedure', ru: 'Порядок приёмки' },
  'graph.field.expected_effect': { en: 'Expected effect', ru: 'Ожидаемый эффект' },
  'graph.field.domain': { en: 'Subject area', ru: 'Предметная область' },
  'graph.field.gap': { en: 'Knowledge gap', ru: 'Пробел в знаниях' },
  'graph.field.question': { en: 'Question', ru: 'Вопрос' },
  'graph.field.statement': { en: 'Statement', ru: 'Утверждение' },
  'graph.field.description': { en: 'Description', ru: 'Описание' },
  'graph.field.inputs': { en: 'Needs', ru: 'Что нужно' },
  'graph.field.outputs': { en: 'Yields', ru: 'Что даёт' },
  'graph.field.cost': { en: 'Cost', ru: 'Стоимость' },
  'graph.field.path': { en: 'File', ru: 'Файл' },
  'graph.field.uri': { en: 'Address', ru: 'Адрес' },
  'graph.field.completion_criteria': { en: 'Research completion conditions', ru: 'Условия завершения исследования' },
  'graph.field.target_setting': { en: 'Expected result', ru: 'Ожидаемый результат' },
  'graph.field.research_form': { en: 'Research format', ru: 'Формат исследования' },
  'graph.field.decomposition': { en: 'Research subtasks', ru: 'Подзадачи исследования' },
  'graph.field.assignee': { en: 'Assigned to', ru: 'Исполнитель' },
  'graph.field.plan_task_id': { en: 'Plan step', ru: 'Шаг плана' },
  'graph.field.specificity': { en: 'Research boundaries', ru: 'Границы исследования' },
  'graph.field.trl': { en: 'TRL', ru: 'УГТ' },
  'graph.field.ai_application_model': { en: 'Role of AI', ru: 'Роль ИИ в работе' },
  'graph.field.modality': { en: 'Research approach', ru: 'Подход к исследованию' },
  'graph.field.norms': { en: 'Research conduct rules', ru: 'Правила проведения исследования' },
  'graph.field.frameworks': { en: 'Theoretical models and approaches', ru: 'Теоретические модели и подходы' },
  'graph.field.method': { en: 'Method', ru: 'Метод' },
  'graph.field.not_tested_reason': { en: 'Why untested', ru: 'Почему не проверялась' },
  'graph.field.failure_reason': { en: 'Why it failed', ru: 'Почему не удалось' },
  'graph.field.result_kind': { en: 'Result type', ru: 'Характер результата' },
  'graph.field.evidence_assessment': { en: 'Reliability assessment', ru: 'Оценка достоверности' },
  'graph.field.postponed_reason': { en: 'Why postponed', ru: 'Почему отложена' },
  'graph.field.inconclusive_reason': { en: 'What stayed unsettled', ru: 'Что осталось неясным' },
  // The plan's own record, on a method and on an experiment task. Left
  // uncaptioned the panel printed the bare storage key — a Russian reader met
  // "analysis_artifacts" and "mcp_servers" in the middle of a Russian card,
  // which is most of what made the method blocks heavy to read.
  'graph.field.instruments': { en: 'Run with', ru: 'Чем выполняется' },
  'graph.field.mcp_servers': { en: 'MCP tools', ru: 'Инструменты MCP' },
  'graph.field.route': { en: 'Route', ru: 'Маршрут' },
  'graph.field.experiment_question': { en: 'Experimental question', ru: 'Вопрос эксперимента' },
  'graph.field.success_criteria': { en: 'Success criteria', ru: 'Критерии успеха' },
  'graph.field.baselines': { en: 'Compared against', ru: 'С чем сравнивается' },
  'graph.field.metrics': { en: 'Measured', ru: 'Что измеряется' },
  'graph.field.analysis_artifacts': { en: 'Analysis artifacts', ru: 'Артефакты анализа' },
  'graph.field.expected_artifacts': { en: 'Expected artifacts', ru: 'Ожидаемые артефакты' },
  'graph.field.dataset': { en: 'Data', ru: 'Данные' },
  'graph.field.depends_on': { en: 'Depends on', ru: 'Зависит от' },
  'graph.field.hypothesis_refs': { en: 'Claims tested', ru: 'Проверяемые гипотезы' },
  'graph.field.operation_ref': { en: 'Research task', ru: 'Задача исследования' },
  'graph.field.task_id': { en: 'Task', ru: 'Задача' },
  'graph.field.experiment_task_id': { en: 'Task', ru: 'Задача' },
  'graph.field.experiment_run_id': { en: 'Run', ru: 'Прогон' },
  'graph.field.plan_id': { en: 'Plan', ru: 'План' },
  'graph.field.plan_revision': { en: 'Plan revision', ru: 'Редакция плана' },
  'graph.field.result_id': { en: 'Result', ru: 'Результат' },
  'graph.field.query': { en: 'Query', ru: 'Запрос' },
  'graph.field.sources': { en: 'Sources', ru: 'Источники' },
  'graph.field.label': { en: 'Name', ru: 'Название' },
  'graph.field.notes': { en: 'Notes', ru: 'Примечания' },
  // The bands the research canvas is read down, top to bottom. A card sits in
  // the band of the stage that produced it, so the picture says where the study
  // has got to before any card is read. The hypothesis band holds the claim and
  // the bar written for it; what tests the claim stands in the band below.
  'graph.stage.framing': { en: 'Framing', ru: 'Постановка' },
  'graph.stage.literature': { en: 'Literature review', ru: 'Анализ литературы' },
  'graph.stage.hypotheses': { en: 'Hypotheses', ru: 'Гипотезы' },
  'graph.stage.experiment': { en: 'Experiments', ru: 'Эксперименты' },
  'graph.stage.report': { en: 'Report', ru: 'Отчёт' },
  'graph.stage.empty': { en: 'not reached yet', ru: 'этап не начат' },
  // What the record is missing — said out loud, because an empty canvas looked
  // the same whether nobody wrote anything or every write was refused.
  'graph.gap.no_root': { en: 'no root question', ru: 'нет корневого вопроса' },
  'graph.gap.no_frame': { en: 'the research frame was never set', ru: 'рамка исследования не задана' },
  'graph.gap.no_hypotheses': { en: 'no hypotheses yet', ru: 'гипотез пока нет' },
  'graph.gap.no_methods': { en: '{count} hypothesis(es) with no method', ru: 'гипотез без метода: {count}' },
  'graph.gap.no_evidence': { en: 'methods ran, no observations recorded', ru: 'методы есть, наблюдений нет' },
  'graph.gap.unreasoned_failures': { en: '{count} outcome(s) with no reason', ru: 'исходов без причины: {count}' },
  'graph.gap.rejected_commits': { en: '{count} write(s) refused', ru: 'записей отклонено: {count}' },
  'graph.error.http': { en: 'HTTP {status}', ru: 'HTTP {status}' },
  'graph.error.server': { en: 'server: {error}', ru: 'сервер: {error}' },
};

/** Применяет текущий язык ко всем элементам с data-i18n / data-i18n-placeholder */
function applyLanguage(lang) {
  // The server rejects a language change while a run is active, so ignore
  // the request here too. The buttons are disabled for the run anyway.
  if (lang && typeof runActive !== 'undefined' && runActive) return;
  const prevLang = currentLang;
  if (lang) currentLang = lang;
  localStorage.setItem(I18N_LANG_KEY, currentLang);
  document.documentElement.lang = currentLang;

  // The indicator builds its phrases in JS, not from data-i18n attributes.
  if (window.StatusIndicator) StatusIndicator.setLang(currentLang);

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const entry = i18n[key];
    if (entry && entry[currentLang]) el.innerHTML = entry[currentLang];
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const key = el.getAttribute('data-i18n-placeholder');
    const entry = i18n[key];
    if (entry && entry[currentLang]) el.placeholder = entry[currentLang];
  });

  document.querySelectorAll('[data-i18n-title]').forEach(el => {
    const key = el.getAttribute('data-i18n-title');
    const entry = i18n[key];
    if (entry && entry[currentLang]) el.title = entry[currentLang];
  });
  // Icon-only buttons: the name a screen reader announces.
  document.querySelectorAll('[data-i18n-aria]').forEach(el => {
    const key = el.getAttribute('data-i18n-aria');
    const entry = i18n[key];
    if (entry && entry[currentLang]) el.setAttribute('aria-label', entry[currentLang]);
  });
  if (window.PlanTracker) PlanTracker.render();
  // The rail toggle's tooltip depends on its state, so it is not a plain
  // data-i18n-title the loop above can swap.
  if (typeof applySideRailState === 'function') applySideRailState();

  // Динамические элементы статуса и пользователя
  const nicknameEl = document.getElementById('active-nickname');
  const user = (typeof activeUser !== 'undefined') ? activeUser : null;
  if (nicknameEl) {
    if (user && user.nickname) {
      nicknameEl.textContent = user.nickname;
    } else {
      const entry = i18n['nav.noUser'];
      nicknameEl.textContent = (entry && entry[currentLang]) || 'No user selected';
    }
  }
  const isWsOpen = typeof ws !== 'undefined' && ws && ws.readyState === 1;
  const connStatusEl = document.getElementById('conn-status');
  if (connStatusEl) {
    const key = isWsOpen ? 'nav.connected' : 'nav.disconnected';
    const entry = i18n[key];
    connStatusEl.textContent = (entry && entry[currentLang]) || (isWsOpen ? 'Connected' : 'Disconnected');
  }

  // Dynamic HUD labels JS writes outside the data-i18n pass.
  if (typeof renderStatusBadge === 'function') renderStatusBadge();
  if (typeof renderEventCount === 'function') renderEventCount();
  const metricsSummaryEl = document.getElementById('metrics-summary');
  if (metricsSummaryEl && metricsSummaryEl.dataset.empty === '1') {
    metricsSummaryEl.textContent = t('metrics.noCalls');
  }

  // The settings modal builds its fields (and the language switch) in JS.
  if (typeof renderSettings === 'function') renderSettings();

  // Обновляем кнопки переключателя языка
  const btnEn = document.getElementById('lang-btn-en');
  const btnRu = document.getElementById('lang-btn-ru');
  if (btnEn) {
    btnEn.classList.toggle('bg-primary', currentLang === 'en');
    btnEn.classList.toggle('text-on-primary', currentLang === 'en');
  }
  if (btnRu) {
    btnRu.classList.toggle('bg-primary', currentLang === 'ru');
    btnRu.classList.toggle('text-on-primary', currentLang === 'ru');
  }

  // HITL cards compose their header text in JS (agent / tool / trigger).
  if (typeof relocalizeHitlCards === 'function') relocalizeHitlCards();

  if (typeof renderActivityRail === 'function' && typeof activityAgents !== 'undefined' && activityAgents.size) {
    renderActivityRail();
  }

  if (typeof RunTimer !== 'undefined' && typeof RunTimer.reapplyLanguage === 'function') {
    RunTimer.reapplyLanguage();
  }

  // One control sets both the interface language and the report language.
  // Send only on an explicit change with a live session. A re-render call
  // (no argument) must not replay the message.
  if (lang && lang !== prevLang && typeof activeSession !== 'undefined' && activeSession && typeof sendReportLanguage === 'function') {
    sendReportLanguage(currentLang);
  }
}

/** Хелпер для получения перевода по ключу. Подставляет {placeholder} из vars. */
function t(key, vars = null, fallback = '') {
  if (typeof vars === 'string') { fallback = vars; vars = null; }
  const entry = i18n[key];
  let text = (entry && entry[currentLang]) || fallback || key;
  if (vars) {
    Object.keys(vars).forEach(name => {
      text = text.split('{' + name + '}').join(String(vars[name]));
    });
  }
  return text;
}
window.t = t;

// Применяем язык при загрузке страницы (после того как DOM построится)
document.addEventListener('DOMContentLoaded', () => applyLanguage());


