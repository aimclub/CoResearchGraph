    // =========================================================================
    // i18n – lightweight translation system
    // =========================================================================
    // Текущий язык: 'ru' по умолчанию (или из localStorage)
    let currentLang = localStorage.getItem(LANG_STORAGE_KEY) || 'ru';

    const i18n = {
      // ── Navigation & Sidebar ──
      'nav.agents': { en: 'Agents', ru: 'Агенты' },
      'nav.user': { en: 'User', ru: 'Пользователь' },
      'nav.session': { en: 'Session', ru: 'Сессия' },
      'nav.noUser': { en: 'No user selected', ru: 'Пользователь не выбран' },
      'nav.connected': { en: 'Connected', ru: 'Подключено' },
      'nav.disconnected': { en: 'Disconnected', ru: 'Отключено' },
      'nav.orchestrator': { en: 'ORCHESTRATOR', ru: 'ОРКЕСТРАТОР' },

      // ── Composer: report language (NOT the interface language) ──
      'composer.reportLang.ru': { en: 'Report: RU', ru: 'Отчёт: RU' },
      'composer.reportLang.en': { en: 'Report: EN', ru: 'Отчёт: EN' },

      // Agent descriptions in side nav
      'agent.OrchestratorAgent.desc': { en: 'Master Orchestrator', ru: 'Главный оркестратор' },
      'agent.PlannerAgent.desc': { en: 'Roadmap Planner', ru: 'Планировщик задач' },
      'agent.ToolsViewer.desc': { en: 'Tools Viewer', ru: 'Просмотр инструментов' },
      'agent.KnowledgeGraph.desc': { en: 'Knowledge Graph', ru: 'Граф знаний' },
      'agent.MCPBuilder.desc': { en: 'MCP Builder', ru: 'Сборщик MCP' },
      'agent.CoderSandbox.desc': { en: 'CoderSandbox', ru: 'Песочница кода' },
      'agent.__settings__.desc': { en: 'Settings', ru: 'Настройки' },

      // Chat controls & Header
      'chat.missionControl': { en: 'Mission Control', ru: 'Центр управления' },
      'chat.online': { en: 'Online', ru: 'Онлайн' },
      'chat.offline': { en: 'Offline', ru: 'Офлайн' },
      'chat.sendQuery': { en: 'Send a query to begin orchestration', ru: 'Отправьте запрос для начала работы' },
      'chat.placeholder': { en: 'Send system command…  (Enter — send, Shift+Enter — new line)', ru: 'Введите запрос… (Enter — отправить, Shift+Enter — новая строка)' },
      'telemetry.header': { en: 'Telemetry Output', ru: 'Лог телеметрии' },
      'usage.header': { en: 'Usage & Cost', ru: 'Использование и стоимость' },

      // ── Activity Rail HUD ──
      'rail.agents': { en: 'Agents', ru: 'Агенты' },
      'rail.tools': { en: 'Tools', ru: 'Инструменты' },
      'rail.standby': { en: 'Standby — awaiting tool invocation', ru: 'Ожидание вызова инструментов…' },
      'rail.noTools': { en: 'No tool calls yet', ru: 'Инструменты ещё не вызывались' },
      'rail.toggle': { en: 'Show/hide agent activity', ru: 'Показать/скрыть активность агентов' },

      // ── Settings modal header ──
      'settings.title': { en: 'Settings', ru: 'Настройки' },
      'settings.subtitle': { en: 'System Configuration', ru: 'Конфигурация системы' },

      // ── General section ──
      'settings.general': { en: 'General', ru: 'Общие' },
      'settings.general.hint': {
        en: '(Default values are set in .env or settings.py)',
        ru: '(Значения по умолчанию задаются в .env или settings.py)'
      },

      'settings.startMode.label': { en: 'Session Start Mode', ru: 'Режим запуска сессии' },
      'settings.startMode.desc': {
        en: 'Choose which agent starts the session. <strong>PlannerAgent</strong> runs PlannerAgent first, then OrchestratorAgent. <strong>OrchestratorAgent</strong> starts directly with the orchestrator. <strong>Orchestrator as a planner</strong> gives Orchestrator the plan tool and disables PlannerAgent.',
        ru: 'Выберите, какой агент запускает сессию. <strong>PlannerAgent</strong> сначала запускает PlannerAgent, затем OrchestratorAgent. <strong>OrchestratorAgent</strong> запускает оркестратор напрямую. <strong>Orchestrator as a planner</strong> даёт оркестратору инструмент планирования и отключает PlannerAgent.'
      },
      'settings.startMode.planner': { en: 'PlannerAgent', ru: 'PlannerAgent' },
      'settings.startMode.init': { en: 'PlannerAgent', ru: 'PlannerAgent' },
      'settings.startMode.orchestrator': { en: 'OrchestratorAgent', ru: 'OrchestratorAgent' },
      'settings.startMode.orchestratorPlanner': { en: 'Orchestrator as a planner', ru: 'Orchestrator в роли планировщика' },

      'settings.maxRetries.label': { en: 'Max LLM Retries', ru: 'Макс. повторов LLM' },
      'settings.maxRetries.desc': {
        en: 'Number of LLM call retry attempts on transient network or upstream API errors.',
        ru: 'Количество повторных попыток вызова LLM при временных сбоях сети или API.'
      },

      'settings.hitl.label': { en: 'HITL Enabled', ru: 'Включить HITL (подтверждения)' },
      'settings.hitl.desc': {
        en: 'Toggle Human-in-the-Loop approval for dangerous or outward-facing actions.',
        ru: 'Включить подтверждение человеком (Human-in-the-Loop) для опасных или внешних действий.'
      },
      'settings.hitlTimeout.label': { en: 'HITL Auto-Approve Timeout (s)', ru: 'Автоподтверждение HITL (сек)' },
      'settings.hitlTimeout.desc': {
        en: 'Seconds before auto-approving HITL requests (-1 for no timeout / wait for human).',
        ru: 'Секунды до автоподтверждения HITL (-1 — без тайм-аута, ждать человека).'
      },

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
        en: "Agent '{agent}' proposes its result. Please review.",
        ru: "Агент '{agent}' предлагает свой результат. Проверьте его."
      },

      'settings.usePlanner.label': { en: 'Use Planner', ru: 'Использовать планировщик' },
      'settings.usePlanner.note': {
        en: '(disabled when using PlannerAgent mode)',
        ru: '(отключено в режиме PlannerAgent)'
      },
      'settings.usePlanner.desc': {
        en: 'Whether the orchestrator delegates to PlannerAgent for task decomposition.',
        ru: 'Делегирует ли оркестратор декомпозицию задач компоненту PlannerAgent.'
      },

      'settings.contextInit.label': { en: 'Research Frame', ru: 'Рамка исследования' },
      'settings.contextInit.desc': {
        en: 'Draft research frame and seed it into the research graph before the orchestrator runs.',
        ru: 'Формировать фрейм исследования и добавлять его в граф исследований до запуска оркестратора.'
      },

      'settings.useProxy.label': { en: 'Use Corporate Proxy', ru: 'Корпоративный прокси' },
      'settings.useProxy.note': { en: '(set only in env)', ru: '(задаётся в .env)' },
      'settings.useProxy.desc': {
        en: 'Route LLM model calls through corporate proxy (SERVICES__PROXY_URL).',
        ru: 'Маршрутизировать вызовы моделей LLM через корпоративный прокси (SERVICES__PROXY_URL).'
      },

      'settings.opik.label': { en: 'Enable Opik Tracing', ru: 'Включить трассировку Opik' },
      'settings.opik.desc': {
        en: 'Send execution logs and agent traces to your Opik dashboard.',
        ru: 'Отправлять логи выполнения и трассировки агентов в дашборд Opik.'
      },

      'settings.autoNaming.label': { en: 'Auto-name Sessions', ru: 'Авто-наименование сессий' },
      'settings.autoNaming.desc': {
        en: 'Automatically generate session titles based on the first prompt.',
        ru: 'Автоматически генерировать названия сессий на основе первого запроса.'
      },

      'settings.defaultUsername.label': { en: 'Default Username', ru: 'Имя пользователя по умолчанию' },
      'settings.defaultUsername.desc': {
        en: 'Auto-selects user on startup (can also be set via COSCIENTIST_USERNAME in .env).',
        ru: 'Автоматически выбирает пользователя при старте (задаётся также через COSCIENTIST_USERNAME в .env).'
      },
      'settings.defaultUsername.placeholder': {
        en: 'e.g. COSCIENTIST_USERNAME in .env',
        ru: 'например COSCIENTIST_USERNAME в .env'
      },

      // ── Graphs section ──
      'settings.graphs': { en: 'Graphs', ru: 'Графы' },
      'settings.knowledgeGraph.label': { en: 'Knowledge Graph', ru: 'Граф знаний' },
      'settings.knowledgeGraph.desc': {
        en: 'Records the execution graph of every run and lets agents read it (<span class="font-mono">get_graph_history</span>, <span class="font-mono">get_agents_info</span>, <span class="font-mono">search_knowledge_memory</span>). When off, nothing is recorded, the Graph view stays empty, and the tools disappear from every agent and from their prompts.',
        ru: 'Записывает граф выполнения каждого запуска и позволяет агентам читать его (<span class="font-mono">get_graph_history</span>, <span class="font-mono">get_agents_info</span>, <span class="font-mono">search_knowledge_memory</span>). Когда выключен, ничего не записывается, вкладка графа пуста, а инструменты убираются у всех агентов.'
      },

      'settings.researchGraph.label': { en: 'Research Graph', ru: 'Граф исследований' },
      'settings.researchGraph.desc': {
        en: 'The typed research blackboard agents commit findings to (<span class="font-mono">research_commit</span>, <span class="font-mono">research_context_slice</span>, orchestrator triggers). When off, the whole feature — tools and prompt sections — drops out and agents pass context through their answers only.',
        ru: 'Доска исследований, куда агенты записывают результаты (<span class="font-mono">research_commit</span>, <span class="font-mono">research_context_slice</span>, триггеры оркестратора). Когда выключен, инструменты и секции промптов отключаются, контекст передается только в ответах.'
      },

      'settings.graphs.sessionNote': {
        en: 'Applies to new sessions — the agent system is built once per session.',
        ru: 'Применяется к новым сессиям — система агентов инициализируется при создании сессии.'
      },

      'settings.deleteGraph.label': { en: 'Delete Graph Data', ru: 'Удалить данные графов' },
      'settings.deleteGraph.desc': {
        en: 'Wipe what the graphs have recorded. The execution and research graphs belong to the <strong>current session</strong>; the knowledge memory is installation-wide and disappears for every session at once. The research graph and the knowledge memory are archived next to their files first; the execution graph keeps only the agent roster.',
        ru: 'Очистить записанные данными графов. Графы выполнения и исследований относятся к <strong>текущей сессии</strong>; память знаний распространяется на всю систему. Граф исследований и память знаний архивируются; граф выполнения сохраняет только список агентов.'
      },
      'settings.deleteGraph.optExecution': { en: 'Execution (session)', ru: 'Выполнение (сессия)' },
      'settings.deleteGraph.optResearch': { en: 'Research (session)', ru: 'Исследования (сессия)' },
      'settings.deleteGraph.optMemory': { en: 'Knowledge memory (global)', ru: 'Память знаний (глобальная)' },
      'settings.deleteGraph.optAll': { en: 'All of the above', ru: 'Всё вышеперечисленное' },
      'settings.deleteGraph.btn': { en: 'Delete', ru: 'Удалить' },

      'settings.autoClearGraph.label': { en: 'Auto-clear Graph Before Session', ru: 'Автоочистка графов перед сессией' },
      'settings.autoClearGraph.note': { en: '(set only in env)', ru: '(задаётся в .env)' },
      'settings.autoClearGraph.desc': {
        en: 'Automatically clear graph data before each session starts.',
        ru: 'Автоматически очищать данные графов перед началом каждой сессии.'
      },

      // ── PlannerAgent section ──
      'settings.planner.retrieval.label': { en: 'Retrieval Tools', ru: 'Инструменты поиска' },
      'settings.planner.retrieval.desc': {
        en: 'Let the planner search the MCP registry (<span class="font-mono">retrieve_tools</span>, <span class="font-mono">get_server_info</span>) before writing the roadmap. When off, it plans by outcome and never names concrete tools or server ids.',
        ru: 'Разрешить планировщику искать в реестре MCP (<span class="font-mono">retrieve_tools</span>, <span class="font-mono">get_server_info</span>) перед созданием плана. Когда выключено, планирование происходит без указания конкретных инструментов.'
      },

      'settings.planner.graph.label': { en: 'Graph Tools', ru: 'Инструменты графа' },
      'settings.planner.graph.note': { en: '(disabled — Knowledge Graph is off)', ru: '(отключено — Граф знаний выключен)' },
      'settings.planner.graph.desc': {
        en: 'Let the planner read the shared knowledge graph (history, agent roster, knowledge memory) so it does not re-plan finished work.',
        ru: 'Разрешить планировщику читать общий граф знаний (историю, список агентов, память знаний), чтобы не планировать заново выполненную работу.'
      },

      'settings.planner.critic.label': { en: 'Plan Critic', ru: 'Критик плана' },
      'settings.planner.critic.desc': {
        en: 'Have an LLM critic review the registered roadmap (assignees, coverage, dependencies) before it is executed, and send it back to the planner if it objects. Runs whether or not HITL is on, before a human sees the plan. Costs one extra LLM call per planning run.',
        ru: 'Проверять созданный план с помощью LLM-критика (исполнители, покрытие, зависимости) перед выполнением и возвращать планировщику при наличии замечаний. Добавляет 1 вызов LLM на запуск планирования.'
      },

      'settings.planner.rounds.label': { en: 'Revision Rounds', ru: 'Раунды доработки' },
      'settings.planner.rounds.desc': {
        en: 'How many times the critic may send the roadmap back. <span class="font-mono">1</span> — it gets a single say and the rewrite then stands. Each extra round is a full replan, and a critic that never approves would otherwise keep the planner going.',
        ru: 'Сколько раз критик может возвращать план на доработку. <span class="font-mono">1</span> — одна проверка, после чего версия утверждается. Каждый доп. раунд — полный переутверждённый план.'
      },

      'settings.planner.mergeTasks.label': { en: 'Merge Tasks', ru: 'Объединение задач' },
      'settings.planner.mergeTasks.desc': {
        en: 'Automatically merge consecutive tasks assigned to the same executor (CoderAgent / TaskExecutorAgent) into a single task. Turn off to keep every task the planner wrote as a separate unit of work.',
        ru: 'Автоматически объединять последовательные задачи, назначенные одному исполнителю (CoderAgent / TaskExecutorAgent), в одну задачу.'
      },

      // ── ResearchAgent section ──
      'settings.research.maxSearches.label': { en: 'Per-turn — max searches', ru: 'Макс. поисков за ход' },
      'settings.research.maxSearches.desc': {
        en: 'Maximum number of web search tool calls per agent turn. After this limit, the agent must synthesize from existing results.',
        ru: 'Максимальное количество вызовов поиска в сети за один ход агента. После превышения лимита агент должен отвечать из имеющихся данных.'
      },

      // ── HypothesesAgent section ──
      'settings.hypotheses.maxActive.label': { en: 'Max Active Hypotheses', ru: 'Макс. активных гипотез' },
      'settings.hypotheses.maxActive.desc': {
        en: 'How many hypotheses are kept as active (<span class="font-mono">formulated</span>) simultaneously for parallel verification. <span class="font-mono">1</span> — the classic "one at a time" mode: the agent picks the single best hypothesis and postpones the rest. Higher values let the orchestrator verify several branches in parallel.',
        ru: 'Сколько гипотез одновременно сохраняются активными (<span class="font-mono">formulated</span>) для параллельной проверки. <span class="font-mono">1</span> — режим "по одной": выбирается 1 лучшая гипотеза. Более высокие значения позволяют проверять несколько веток параллельно.'
      },

      // ── CoderAgent section ──
      'settings.coder.mode.label': { en: 'Coder Execution Mode', ru: 'Режим выполнения Coder' },
      'settings.coder.mode.note': { en: '(no Sandbox URL set — OpenHands mode requires a Sandbox URL)', ru: '(не задан URL песочницы — для режима OpenHands требуется URL песочницы)' },
      'settings.coder.mode.desc': {
        en: 'Choose execution mode: <strong>local</strong> uses in-process tools (<span class="font-mono">execute_bash</span>, file edits, git); <strong>openhands</strong> relays tasks to the remote OpenHands sandbox agent.',
        ru: 'Выберите режим выполнения: <strong>local</strong> использует локальные инструменты (<span class="font-mono">execute_bash</span>, правка файлов, git); <strong>openhands</strong> передаёт задачи удалённому агенту в песочнице OpenHands.'
      },

      'settings.coder.sandboxUrl.label': { en: 'Sandbox Remote URL', ru: 'Удалённый URL песочницы' },
      'settings.coder.sandboxUrl.desc': {
        en: 'The endpoint URL of the isolated code-execution sandbox server.',
        ru: 'URL-адрес изолированного сервера-песочницы для выполнения кода.'
      },

      'settings.coder.workspaceId.label': { en: 'Coder Workspace ID', ru: 'ID рабочей области Coder' },
      'settings.coder.workspaceId.desc': {
        en: 'Pin a custom persistent workspace folder name to save code state across delegations. Leave empty to auto-generate.',
        ru: 'Указать имя папки рабочей области для сохранения состояния кода между вызовами. Оставьте пустым для автогенерации.'
      },
      'settings.coder.workspaceId.placeholder': { en: 'e.g. workspace_1', ru: 'например workspace_1' },

      // ── TaskExecutorAgent section ──
      'settings.taskExec.keepScore.label': { en: 'Tool Keep Threshold', ru: 'Порог релевантности инструментов' },
      'settings.taskExec.keepScore.desc': {
        en: 'Minimum relevance score (0.0 to 1.0) for a retrieved MCP tool to be loaded into context.',
        ru: 'Минимальный балл релевантности (от 0.0 до 1.0) для загрузки найденного инструмента MCP в контекст.'
      },

      'settings.taskExec.abstainScore.label': { en: 'Tool Abstain Threshold', ru: 'Порог отказа от инструментов' },
      'settings.taskExec.abstainScore.desc': {
        en: 'Threshold below which the tool pipeline completely abstains, so the executor re-routes the task to the CoderAgent.',
        ru: 'Порог, ниже которого пайплайн инструментов отказывается от выполнения, и исполнитель перенаправляет задачу в CoderAgent.'
      },

      // ── Empty / Common sections ──
      'settings.noConfig': { en: 'No configurable parameters yet.', ru: 'Пока нет настраиваемых параметров.' },
      'settings.cancel': { en: 'Cancel', ru: 'Отмена' },
      'settings.save': { en: 'Save Settings', ru: 'Сохранить настройки' },
    };

    /** Применяет текущий язык ко всем элементам с data-i18n / data-i18n-placeholder */
    function applyLanguage(lang) {
      if (lang) currentLang = lang;
      localStorage.setItem(LANG_STORAGE_KEY, currentLang);
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

      // Динамические элементы статуса и пользователя
      const nicknameEl = document.getElementById('active-nickname');
      if (nicknameEl) {
        if (activeUser && activeUser.nickname) {
          nicknameEl.textContent = activeUser.nickname;
        } else {
          const entry = i18n['nav.noUser'];
          nicknameEl.textContent = (entry && entry[currentLang]) || 'No user selected';
        }
      }
      const isWsOpen = ws && ws.readyState === 1;
      const connStatusEl = document.getElementById('conn-status');
      if (connStatusEl) {
        const key = isWsOpen ? 'nav.connected' : 'nav.disconnected';
        const entry = i18n[key];
        connStatusEl.textContent = (entry && entry[currentLang]) || (isWsOpen ? 'Connected' : 'Disconnected');
      }
      const badgeEl = document.getElementById('active-badge');
      if (badgeEl) {
        const key = isWsOpen ? 'chat.online' : 'chat.offline';
        const entry = i18n[key];
        badgeEl.textContent = (entry && entry[currentLang]) || (isWsOpen ? 'Online' : 'Offline');
      }

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

      if (typeof renderActivityRail === 'function' && typeof activityAgents !== 'undefined' && activityAgents.size) {
        renderActivityRail();
      }
    }

    /** Хелпер для получения перевода по ключу */
    function t(key, fallback = '') {
      const entry = i18n[key];
      if (entry && entry[currentLang]) return entry[currentLang];
      return fallback || key;
    }
    window.t = t;

    // Применяем язык при загрузке страницы (после того как DOM построится)
    document.addEventListener('DOMContentLoaded', () => applyLanguage());


