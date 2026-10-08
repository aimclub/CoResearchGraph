/**
 * Live status indicator — ONE human sentence about what the system is doing.
 *
 * Why a component of its own: everything needed already arrives over the chat
 * websocket (`status`, `agent_event`, `agent_output`, `tool_activity`,
 * `hitl_request`, `final_response`), but in raw form — tool names, call ids,
 * JSON previews. The activity rail and the ToolsViewer render exactly that,
 * for a developer. A user who is not one needs the opposite: no tool names, no
 * ids, one line that says "I am searching the web" and a way to expand it.
 *
 * So this module is a *reducer*: every websocket frame is fed in, and a small
 * state machine decides which phrase is true right now. Nothing here talks to
 * the server, and nothing on the server knows about it — which is why it also
 * works on snapshot replay (a reconnect, a session switch) and in `?demo=`
 * mode, where a recorded `agent_events.json` is played back with no run at all.
 *
 * Public API (all no-ops before `mount()`):
 *   StatusIndicator.mount(el)          — attach to a container element
 *   StatusIndicator.feed(msg, quiet)   — push one websocket frame
 *   StatusIndicator.setLang('ru'|'en') — phrase language
 *   StatusIndicator.setConnected(bool) — websocket up/down
 *   StatusIndicator.markStopped()      — the user pressed Stop
 *   StatusIndicator.reset()            — session switch / clear
 *   StatusIndicator.demo(file, speed)  — replay a saved bundle's events
 */
(function () {
  'use strict';

  // ── Tuning ────────────────────────────────────────────────────────────────
  // A phrase is held at least this long even if the next tool call arrives
  // sooner — a run that fires five tools in 200 ms must not strobe.
  const MIN_HOLD_MS = 1200;
  const RENDER_THROTTLE_MS = 350;
  // How long silence must last before "working" decays into "thinking".
  const SETTLE_GRACE_MS = 900;
  // No event at all for this long *and nothing open*: stop claiming an action.
  // While a call is open, silence is not news — a sandbox task runs for forty
  // minutes without emitting anything, and that IS the state to report.
  //
  // Measured against a recorded run, the gap between a tool result and the
  // next call is 14 s at the median and 56 s at p90 — all of it the model
  // reading what came back. A 20-second threshold turned most of those gaps
  // into "Думаю над задачей"; past a minute and a half, that really is all
  // there is to say.
  const SILENCE_MS = 90000;
  // How long a step must run before the wait is worth apologising for.
  const LONG_STEP_MS = 90000;
  const DONE_LINGER_MS = 2600; // green check, then hide
  const ERROR_LINGER_MS = 8000;
  const STOPPED_LINGER_MS = 3000;
  const MAX_STEPS = 6;
  const DETAIL_LIMIT = 64;

  // ── Phrase dictionary ─────────────────────────────────────────────────────
  // Categories are deliberately coarse: the user cares about the KIND of work,
  // not which of four search backends answered.
  const CATEGORIES = {
    web_search: { icon: 'travel_explore', ru: 'Ищу информацию в интернете', en: 'Searching the web' },
    papers: { icon: 'menu_book', ru: 'Читаю научные статьи', en: 'Reading scientific papers' },
    rag: { icon: 'database', ru: 'Проверяю архивы', en: 'Searching the knowledge base' },
    code: { icon: 'terminal', ru: 'Пишу и запускаю код', en: 'Writing and running code' },
    data: { icon: 'dataset', ru: 'Обрабатываю данные', en: 'Preparing the data' },
    chem: { icon: 'science', ru: 'Считаю свойства молекул', en: 'Computing molecular properties' },
    plot: { icon: 'insert_chart', ru: 'Строю графики', en: 'Building charts' },
    graph_write: { icon: 'hub', ru: 'Дополняю граф знаний', en: 'Recording findings in the knowledge graph' },
    graph_read: { icon: 'account_tree', ru: 'Сверяюсь с картой исследования', en: 'Checking the research map' },
    mcp_build: { icon: 'construction', ru: 'Собираю новый инструмент', en: 'Building a new tool' },
    mcp_find: { icon: 'extension', ru: 'Подбираю инструменты для задачи', en: 'Picking tools for the task' },
    tasks: { icon: 'checklist', ru: 'Планирую шаги', en: 'Planning the steps' },
    files: { icon: 'description', ru: 'Работаю с документами', en: 'Working with documents' },
    tool: { icon: 'build', ru: 'Работаю с инструментом', en: 'Using a tool' },
  };

  // First match wins, so the specific patterns come before the generic ones.
  // Order matters more than it looks: `research_*` (the research-graph tools)
  // contains the substring "search", so it has to be decided long before the
  // catch-all search rule at the bottom, or half the run reads as web search.
  const RULES = [
    [/(build|check)_mcp|mcp_build|alembic/, 'mcp_build'],
    [/search_mcp|register_mcp|mcp_server|tool_retriev|rerank|prepare_tool/, 'mcp_find'],
    [/create_plan|task_status|active_tasks|roadmap|todo/, 'tasks'],
    [/chroma|embed|vector|knowledge_memory|memory|recall|retriev/, 'rag'],
    [/research_commit|graph_commit|upsert|add_node|write_graph/, 'graph_write'],
    [/research_|graph|_node|_edge/, 'graph_read'],
    [/arxiv|pubmed|openalex|semantic_scholar|scholar|paper|pico|doi|crossref/, 'papers'],
    [/rdkit|smiles|pubchem|molecul|chembl|inchi|admet|reaction/, 'chem'],
    [/plot|chart|figure|histogram|visuali/, 'plot'],
    [/sandbox|shell|bash|exec|python|jupyter|notebook|git_|npm|pip_|coder/, 'code'],
    [/dataset|csv|parquet|upload|download|fedot|automl|table/, 'data'],
    [/tavily|serp|duckduckgo|google|web_search|browse|fetch_url|crawl/, 'web_search'],
    [/file|read|write|document|report|format_results|pdf|marker|parse/, 'files'],
  ];

  // Deliberately kept out of RULES: "lookup"/"query"/"find" appear in the name
  // of half the tools ever written, so this guess is only worth making after a
  // tool's own description has had its say.
  const GENERIC_RULE = [/search|query|lookup|find|browse/, 'web_search'];

  // Agent roles, as a user would name them. Anything missing falls back to the
  // Agent roles, as a user would name them. Anything missing falls back to the
  // bare class name with the "Agent" suffix stripped.
  const AGENTS = {
    OrchestratorAgent: { ru: 'агент-координатор', en: 'orchestrator agent' },
    PlannerAgent: { ru: 'агент-планировщик', en: 'planner agent' },
    PlanningPipelineAgent: { ru: 'агент-планировщик', en: 'planner agent' },
    ContextInitAgent: { ru: 'агент рамки исследования', en: 'research frame agent' },
    ContextInitSessionAgent: { ru: 'агент рамки исследования', en: 'research frame agent' },
    HypothesesAgent: { ru: 'агент генерации гипотез', en: 'hypotheses agent' },
    ResearchAgent: { ru: 'агент-исследователь', en: 'researcher agent' },
    TaskExecutorAgent: { ru: 'агент-исполнитель', en: 'executor agent' },
    ToolPipelineAgent: { ru: 'агент подбора инструментов', en: 'tool pipeline agent' },
    CoderAgent: { ru: 'агент-инженер', en: 'engineer agent' },
    DatasetCollectorAgent: { ru: 'агент сбора данных', en: 'data collector agent' },
    MedicalAgent: { ru: 'агент медицинского анализа', en: 'medical analyst agent' },
    McpBuilderAgent: { ru: 'агент сборки инструментов', en: 'tool builder agent' },
    ToolPreparerAgent: { ru: 'агент подготовки инструментов', en: 'tool preparer agent' },
    // The tool pipeline fans out into half a dozen internal agents. Naming each
    // one tells a user nothing — they are all the same activity to them.
    ParallelToolSearcherAgent: { ru: 'агент подбора инструментов', en: 'tool search agent' },
    LocalToolsExtractorAgent: { ru: 'агент подбора инструментов', en: 'tool search agent' },
    ToolRetrieverAgent: { ru: 'агент подбора инструментов', en: 'tool search agent' },
    ToolWebSearcherAgent: { ru: 'агент подбора инструментов', en: 'tool search agent' },
    ToolReranker: { ru: 'агент подбора инструментов', en: 'tool search agent' },
    FullSetToolReranker: { ru: 'агент подбора инструментов', en: 'tool search agent' },
    WebToolsDeployerAgent: { ru: 'агент подключения инструментов', en: 'tool deployer agent' },
    ResultAggregatorAgent: { ru: 'агент составления отчёта', en: 'report writer agent' },
    ExperimentAgent: { ru: 'агент экспериментов', en: 'experiment agent' },
    FedotAgent: { ru: 'агент AutoML', en: 'AutoML agent' },
    system: { ru: 'система', en: 'system' },
  };

  // What the agent is doing with what just came back. The gap between a tool
  // result and the next call is the model generating — a median of 14 seconds
  // and up to a minute — and "thinking" wastes it: the useful thing to say is
  // what it is thinking *about*, which is whatever just finished.
  const REVIEW = {
    web_search: { ru: 'Изучаю найденное в интернете', en: 'Reading what the search returned' },
    papers: { ru: 'Разбираю найденные статьи', en: 'Going through the papers' },
    rag: { ru: 'Сверяюсь с базой знаний', en: 'Cross-checking the knowledge base' },
    code: { ru: 'Разбираю результаты запуска', en: 'Reading the run output' },
    data: { ru: 'Проверяю подготовленные данные', en: 'Checking the prepared data' },
    chem: { ru: 'Оцениваю расчёты', en: 'Weighing the computed values' },
    plot: { ru: 'Смотрю на графики', en: 'Looking at the charts' },
    graph_write: { ru: 'Свожу выводы', en: 'Tying the findings together' },
    graph_read: { ru: 'Сопоставляю с картой исследования', en: 'Matching the research map' },
    mcp_build: { ru: 'Проверяю собранный инструмент', en: 'Checking the new tool' },
    mcp_find: { ru: 'Выбираю из найденных инструментов', en: 'Choosing among the tools found' },
    tasks: { ru: 'Сверяюсь с планом', en: 'Checking the plan' },
    files: { ru: 'Читаю документы', en: 'Reading the documents' },
    tool: { ru: 'Разбираю ответ инструмента', en: 'Reading the tool answer' },
    delegation: { ru: 'Изучаю отчёт агента', en: 'Reading the agent report' },
  };

  // Phrases that are not about a tool at all.
  const PHASES = {
    starting: { icon: 'bolt', ru: 'Принимаю задачу', en: 'Picking up the task' },
    framing: { icon: 'architecture', ru: 'Строю рамку исследования', en: 'Building research frame' },
    thinking: { icon: 'neurology', ru: 'Обдумываю', en: 'Thinking it through' },
    long_thinking: { icon: 'neurology', ru: 'Думаю над задачей', en: 'Working on the task' },
    delegating: { icon: 'alt_route', ru: 'Передаю задачу', en: 'Handing over to' },
    // An AgentTool delegation stays open for as long as the subordinate works —
    // hours, in a real study. "Handing over" is true for a second; who is
    // working is true for the whole time, so that is what the line says.
    agentWorking: { icon: 'groups', ru: 'Работает %s', en: '%s is working' },
    finalizing: { icon: 'auto_awesome', ru: 'Собираю итоговый отчёт', en: 'Assembling the final report' },
    waiting: { icon: 'front_hand', ru: 'Жду вашего ответа', en: 'Waiting for your answer' },
    waiting_frame: { icon: 'front_hand', ru: 'Жду подтверждения рамки исследования', en: 'Waiting for research frame confirmation' },
    done: { icon: 'check_circle', ru: 'Готово', en: 'Done' },
    error: { icon: 'error', ru: 'Что-то пошло не так', en: 'Something went wrong' },
    stopped: { icon: 'stop_circle', ru: 'Отменено', en: 'Cancelled' },
    offline: { icon: 'cloud_off', ru: 'Соединение потеряно, переподключаюсь', en: 'Connection lost, reconnecting' },
  };

  const TEXT = {
    longRun: { ru: 'это может занять несколько минут', en: 'this may take a few minutes' },
    toolFailed: { ru: 'Инструмент не ответил, пробую иначе', en: 'A tool did not answer — trying another way' },
    step: { ru: 'Шаг %d из %d', en: 'Step %d of %d' },
    details: { ru: 'Подробнее', en: 'Details' },
    ready: { ru: 'Результат готов', en: 'Result ready' },
    total: { ru: 'Всего', en: 'Total' },
    planHeader: { ru: 'План исследования', en: 'Research plan' },
    openPlan: { ru: 'Открыть весь план', en: 'Open full roadmap' },
    recentActivity: { ru: 'Недавние действия', en: 'Recent activity' },
    openTools: { ru: 'Журнал инструментов', en: 'Tools viewer' },
    moreTasks: { ru: 'ещё %d шагов в плане', en: '%d more steps in plan' },
    reviewTool: { ru: 'Разбираю ответ инструмента «%s»', en: 'Reading the answer from tool «%s»' },
    collapse: { ru: 'Свернуть', en: 'Collapse' },
    expand: { ru: 'Развернуть', en: 'Expand' },
  };

  // Colour + iconography per phase, in the page's own Tailwind tokens.
  const TONES = {
    work: { bar: 'bg-primary', icon: 'text-primary', card: 'border-primary/25 bg-primary/[0.06]', text: 'text-on-surface' },
    wait: { bar: 'bg-tertiary', icon: 'text-tertiary', card: 'border-tertiary/40 bg-tertiary/[0.08]', text: 'text-on-surface' },
    done: { bar: 'bg-secondary', icon: 'text-secondary', card: 'border-secondary/30 bg-secondary/[0.06]', text: 'text-on-surface' },
    fail: { bar: 'bg-error', icon: 'text-error', card: 'border-error/40 bg-error/[0.08]', text: 'text-on-surface' },
    mute: { bar: 'bg-outline-variant', icon: 'text-outline-variant', card: 'border-outline-variant/25 bg-surface-container-high/40', text: 'text-on-surface-variant' },
  };

  const PHASE_TONE = {
    waiting: 'wait', waiting_frame: 'wait', done: 'done', error: 'fail', stopped: 'fail', offline: 'mute',
  };

  const EXPAND_KEY = 'coscientist.status_expanded';
  const PLAN_EXPAND_KEY = 'coscientist.status_plan_expanded';
  const ACTIVITY_EXPAND_KEY = 'coscientist.status_activity_expanded';

  // ── State ─────────────────────────────────────────────────────────────────
  let lang = 'ru';
  let root = null;          // container element
  let expanded = false;
  let planExpanded = true;
  let activityExpanded = true;
  let connected = true;
  let showAllPlanTasks = false;

  const st = newState();

  function newState() {
    return {
      phase: 'idle',
      category: null,
      agent: null,
      detail: null,
      target: null,        // agent a delegation points at
      toolName: null,      // only used by the unrecognised-tool phrase
      since: 0,            // when the current run started
      phraseSince: 0,      // when the visible phrase was last changed
      activitySince: 0,    // when the activity the phrase names began
      lastEventAt: 0,
      // Everything currently open, innermost last — tool calls AND AgentTool
      // delegations, which are calls like any other and stay open for as long
      // as the subordinate works. Keeping them here is what lets the line fall
      // back to "Работает Инженер" when a nested tool finishes, instead of
      // claiming the system is thinking while an agent runs for six hours.
      open: new Map(),     // key -> {kind, tool, agent, category, detail, target, at}
      anonSeq: 0,          // calls that arrive without a call id, closed LIFO
      // The tool that closed last, while nothing is open: what the model is
      // reading right now.
      afterglow: null,
      steps: [],           // [{key, icon, text, status}]
      stepIndex: new Map(),// call_id -> step
      // The orchestrator's plan (the task tracker) and, nested inside one of
      // its steps, the plan the agent in the sandbox keeps for itself. Both
      // are {current, done, total}; both are null until something plans.
      tasks: [],           // task-tracker items, kept so a status update lands
      plan: null,
      sandboxPlan: null,
      note: null,          // transient sub-line (a failed tool, …)
      hideAt: 0,           // when a terminal phase should disappear
    };
  }

  // ── Small helpers ─────────────────────────────────────────────────────────
  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s === null || s === undefined ? '' : String(s);
    return d.innerHTML;
  }

  function pick(entry, fallback) {
    if (!entry) return fallback || '';
    return entry[lang] || entry.en || entry.ru || fallback || '';
  }

  function agentLabel(name, capitalize = false) {
    if (!name) return '';
    const known = AGENTS[name];
    let label = known ? pick(known, name) : '';
    if (!label) {
      const bare = String(name).replace(/Agent$/, '');
      label = (lang === 'ru') ? ('агент-' + bare.toLowerCase()) : (bare + ' agent');
    }
    if (capitalize && label) {
      return label.charAt(0).toUpperCase() + label.slice(1);
    }
    return label;
  }

  /** Which category a tool call belongs to.
   *
   *  The name is tried first and is right for everything the system ships. A
   *  tool from an MCP server built at runtime is named by whatever repository
   *  it came from, though, so an unrecognised name falls back to matching the
   *  same rules against the tool's own description, which `tool_activity`
   *  carries on every call record. */
  function classify(toolName, description) {
    const name = String(toolName || '').toLowerCase();
    const text = String(description || '').toLowerCase();
    const match = (haystack) => {
      for (const [pattern, category] of RULES) {
        if (pattern.test(haystack)) return category;
      }
      return null;
    };
    return match(name)
      || (text && match(text))
      || (GENERIC_RULE[0].test(name) ? GENERIC_RULE[1] : null)
      || (text && GENERIC_RULE[0].test(text) ? GENERIC_RULE[1] : null)
      || 'tool';
  }

  /** A delegation, not tool use: `transfer_to_agent`, or an AgentTool whose
   *  tool name IS the subordinate's agent name. Mirrors the activity rail. */
  function delegationTarget(toolName, args) {
    if (toolName === 'transfer_to_agent') {
      return (args && (args.agent_name || args.agentName)) || null;
    }
    const name = String(toolName || '');
    if (AGENTS[name]) return name;
    return /Agent$/.test(name) ? name : null;
  }

  /** The one argument worth showing a human — a query, a path, a molecule.
   *
   *  Deliberately strict. `tool_activity` hands over a *string* whenever the
   *  arguments were too big to keep as structure, so a string here is a
   *  truncated JSON dump, never a sentence: showing it would put
   *  `{"nodes": [{"ref": "e1"…` under the phrase. Same for any single value
   *  that is itself serialized structure. */
  function detailOf(args) {
    if (!args || typeof args !== 'object' || Array.isArray(args)) return null;
    const keys = ['query', 'q', 'search_query', 'question', 'task', 'topic',
      'name', 'smiles', 'molecule', 'url', 'path', 'file_path', 'filename',
      'title', 'command', 'request'];
    for (const key of keys) {
      const value = args[key];
      if (typeof value !== 'string') continue;
      const text = value.trim();
      if (!text || /^[[{]/.test(text)) continue;
      return trim(text);
    }
    return null;
  }

  function trim(value) {
    const text = String(value).replace(/\s+/g, ' ').trim();
    return text.length > DETAIL_LIMIT ? text.slice(0, DETAIL_LIMIT - 1) + '…' : text;
  }

  function elapsed(ms) {
    const total = Math.max(0, Math.round(ms / 1000));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    // A delegation can be open for six hours; m:ss would read as "372:14".
    return h
      ? h + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0')
      : m + ':' + String(s).padStart(2, '0');
  }

  function stamp(timestamp) {
    if (!timestamp) return Date.now();
    const parsed = new Date(timestamp).getTime();
    return Number.isFinite(parsed) ? parsed : Date.now();
  }

  // ── Steps (the expanded view) ─────────────────────────────────────────────
  function pushStep(key, icon, text) {
    const step = { key: key, icon: icon, text: text, status: 'running' };
    st.steps.push(step);
    if (key) st.stepIndex.set(key, step);
    while (st.steps.length > MAX_STEPS) {
      const dropped = st.steps.shift();
      if (dropped.key) st.stepIndex.delete(dropped.key);
    }
    return step;
  }

  function closeStep(key, failed) {
    const step = key ? st.stepIndex.get(key) : null;
    const target = step || [...st.steps].reverse().find(s => s.status === 'running');
    if (target) target.status = failed ? 'error' : 'done';
  }

  // ── Phrase transitions ────────────────────────────────────────────────────
  let holdTimer = null;

  /** Set the visible phase, holding the previous phrase for MIN_HOLD_MS so a
   *  burst of fast tool calls reads as a sequence instead of a flicker.
   *  Terminal phases (done/error/stopped) always land immediately. */
  function setPhase(phase, patch, immediate) {
    const now = Date.now();
    // `refresh()` runs on every frame of a long-running call and would keep
    // re-announcing the same thing — re-arming the anti-flicker hold and
    // resetting the phrase clock for no reason. Saying nothing new is free.
    if (phase === st.phase
      && (patch && 'category' in patch ? patch.category : null) === st.category
      && (patch && 'detail' in patch ? patch.detail : null) === st.detail
      && (patch && 'target' in patch ? patch.target : null) === st.target
      && !holdTimer) {
      if (patch && patch.agent) st.agent = patch.agent;
      render();
      return;
    }
    const apply = () => {
      st.phase = phase;
      st.category = (patch && 'category' in patch) ? patch.category : null;
      st.detail = (patch && 'detail' in patch) ? patch.detail : null;
      st.target = (patch && 'target' in patch) ? patch.target : null;
      st.toolName = (patch && 'toolName' in patch) ? patch.toolName : null;
      if (patch && patch.agent) st.agent = patch.agent;
      st.phraseSince = Date.now();
      // The timer counts the *activity*, not the phrase: a phrase deferred by
      // the anti-flicker hold must not reset a tool's clock to zero.
      st.activitySince = (patch && patch.since) || st.phraseSince;
      render();
    };
    if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; }
    const waited = now - st.phraseSince;
    if (immediate || st.phase === 'idle' || waited >= MIN_HOLD_MS) {
      apply();
    } else {
      holdTimer = setTimeout(apply, MIN_HOLD_MS - waited);
      render(); // keep the timer/sub-line live meanwhile
    }
  }

  // ── The open-work stack ───────────────────────────────────────────────────

  /** The innermost thing currently running, or null. */
  function topWork() {
    let last = null;
    st.open.forEach(entry => { last = entry; });
    return last;
  }

  function openWork(callId, entry) {
    st.open.set(callId || ('anon:' + (++st.anonSeq)), entry);
    st.afterglow = null;
  }

  /** Close a call by its id; without one, close the innermost anonymous call. */
  function closeWork(callId) {
    if (callId && st.open.has(callId)) {
      const entry = st.open.get(callId);
      st.open.delete(callId);
      return entry;
    }
    let lastAnon = null;
    st.open.forEach((entry, key) => {
      if (String(key).startsWith('anon:')) lastAnon = key;
    });
    if (lastAnon === null) return null;
    const entry = st.open.get(lastAnon);
    st.open.delete(lastAnon);
    return entry;
  }

  /** Say whatever the stack says is true.
   *
   *  Sticky phases own the line until their own event releases it — a run that
   *  ended, or a question waiting on the user. Everything else is a pure
   *  function of what is open, which is why "thinking" is now rare: on a real
   *  run something is open 99% of the time.
   *
   *  Agents chain calls back to back, so an empty stack is confirmed after a
   *  short grace period rather than the instant a result lands. */
  const STICKY = ['idle', 'waiting', 'waiting_frame', 'done', 'error', 'stopped'];

  let settleTimer = null;

  function refresh() {
    cancelSettle();
    if (STICKY.includes(st.phase)) return;
    const work = topWork();
    if (work) { announce(work); return; }
    if (st.phase === 'starting') { announce(null); return; }
    settleTimer = setTimeout(() => {
      settleTimer = null;
      if (!topWork() && !STICKY.includes(st.phase)) announce(null);
    }, SETTLE_GRACE_MS);
  }

  function announce(work) {
    if (work && work.kind === 'delegation') {
      setPhase('delegating', {
        agent: work.agent, target: work.target, since: work.at,
      });
      return;
    }
    if (work) {
      setPhase('working', {
        category: work.category, detail: work.detail, agent: work.agent,
        toolName: work.tool, since: work.at,
      });
      return;
    }
    if (finalStage()) { setPhase('finalizing', { agent: st.agent }); return; }
    if (frameStage()) { setPhase('framing', { agent: st.agent }); return; }
    if (st.afterglow) {
      setPhase('reviewing', {
        category: st.afterglow.category, agent: st.afterglow.agent,
        toolName: st.afterglow.toolName, since: st.afterglow.at,
      });
      return;
    }
    setPhase('thinking', { agent: st.agent });
  }

  function cancelSettle() {
    if (settleTimer) { clearTimeout(settleTimer); settleTimer = null; }
  }

  function finalStage() {
    return st.agent === 'ResultAggregatorAgent';
  }

  function frameStage() {
    return st.agent === 'ContextInitAgent' || st.agent === 'ContextInitSessionAgent';
  }

  function startRun(at) {
    // A second message in the same session starts a new run: preserve existing
    // plan tasks unless a new create_plan call explicitly replaces them.
    if (['idle', 'done', 'stopped', 'error', 'waiting', 'waiting_frame'].includes(st.phase)) {
      const prevTasks = st.tasks;
      const prevPlan = st.plan;
      Object.assign(st, newState(), { since: at || Date.now() });
      if (prevTasks && prevTasks.length) {
        st.tasks = prevTasks;
        st.plan = prevPlan || summarise(prevTasks);
      }
    }
    st.lastEventAt = Date.now();
    st.hideAt = 0;
    setPhase('starting', {}, true);
  }

  function endRun(phase, linger) {
    // A failed or stopped run may still be followed by an idle `status` broadcast.
    // It must not turn an error or cancellation into a green "done".
    if (phase === 'done' && (st.phase === 'error' || st.phase === 'stopped')) return;
    cancelSettle();
    if (phase === 'done') st.note = null;
    st.open.clear();
    st.afterglow = null;
    st.steps.forEach(step => {
      if (step.status === 'running') step.status = (phase === 'done' ? 'done' : 'error');
    });
    if (phase === 'done' && st.tasks) {
      for (const task of st.tasks) {
        if (task && (DONE_STATUS.test(task.status) || RUNNING_STATUS.test(task.status))) {
          task._workFinished = true;
          task.status = 'DONE';
        }
      }
      st.plan = summarise(st.tasks);
      if (window.RoadmapModal && typeof window.RoadmapModal.updateTasks === 'function') {
        window.RoadmapModal.updateTasks(st.tasks, false);
      }
    }
    st.hideAt = Date.now() + linger;
    setPhase(phase, {}, true);
    setTimeout(() => {
      if (st.hideAt && Date.now() >= st.hideAt - 50) {
        st.phase = 'idle';
        st.hideAt = 0;
        render();
      }
    }, linger + 60);
  }

  // ── Plan progress ─────────────────────────────────────────────────────────
  // Two plans exist at once and they are not the same thing. The orchestrator's
  // task tracker plans the *study*; the agent inside the sandbox plans the job
  // it was handed, which is one step of that study. Both are read here into the
  // same shape, and the sub-line prefers whichever is more specific.

  const DONE_STATUS = /done|complete|finish/i;
  const RUNNING_STATUS = /progress|running|active|doing/i;

  function matchAgent(a, b) {
    if (!a || !b) return false;
    const s1 = String(a).toLowerCase().replace(/agent$/, '');
    const s2 = String(b).toLowerCase().replace(/agent$/, '');
    return s1 === s2;
  }

  function isAgentActive(agentName) {
    if (!agentName) return false;
    for (const entry of st.open.values()) {
      if (entry.kind === 'delegation' && matchAgent(entry.target, agentName)) return true;
      if (matchAgent(entry.agent, agentName)) return true;
    }
    if (matchAgent(st.agent, agentName) && ['working', 'thinking', 'delegating'].includes(st.phase)) {
      return true;
    }
    return false;
  }

  function isTaskDone(task) {
    if (!task) return false;
    if (task._workFinished) return true;
    const status = String(task.status || '');
    if (!DONE_STATUS.test(status)) return false;
    // If marked DONE, only consider it done if the assigned agent is not currently working
    if (task.assignee && isAgentActive(task.assignee)) {
      return false;
    }
    return true;
  }

  function isTaskActive(task) {
    if (!task) return false;
    if (isTaskDone(task)) return false;
    const status = String(task.status || '');
    if (RUNNING_STATUS.test(status)) return true;
    if (task.assignee && isAgentActive(task.assignee)) return true;
    return false;
  }

  function markAgentWorkFinished(agentName) {
    if (!agentName || !st.tasks) return;
    let changed = false;
    for (const task of st.tasks) {
      if (task && matchAgent(task.assignee, agentName)) {
        const isDoneOrActive = DONE_STATUS.test(task.status) || RUNNING_STATUS.test(task.status);
        if (isDoneOrActive && !task._workFinished) {
          task._workFinished = true;
          task.status = 'DONE';
          changed = true;
        }
      }
    }
    if (changed) {
      st.plan = summarise(st.tasks);
      if (window.RoadmapModal && typeof window.RoadmapModal.updateTasks === 'function') {
        window.RoadmapModal.updateTasks(st.tasks, false);
      }
    }
  }

  /** {total, done, current} for a list of {title, status} items. */
  function summarise(items) {
    if (!Array.isArray(items) || !items.length) return null;
    const done = items.filter(item => isTaskDone(item)).length;
    // The step a person would name if asked "what is it doing?": the one in
    // progress, or the next one waiting when the agent marks nothing as such.
    const current = items.find(item => isTaskActive(item))
      || items.find(item => !isTaskDone(item));
    const title = current && current.title ? trim(current.title) : null;
    return { total: items.length, done: done, current: title };
  }

  /** The task tracker answers with `{plan: [...]}` (a fresh plan), `{tasks:
   *  [...]}` (the whole list) or `{task: {...}}` (one item's new status). A
   *  preview under the truncation cap keeps its structure, so all three are
   *  readable straight off the broadcast. */
  function readPlan(result) {
    if (!result) return;
    if (typeof result === 'string') {
      try {
        result = JSON.parse(result);
      } catch (_) {
        return;
      }
    }
    if (!result || typeof result !== 'object') return;
    const list = Array.isArray(result.plan) ? result.plan
      : Array.isArray(result.tasks) ? result.tasks : null;
    if (list) {
      const prevMap = new Map((st.tasks || []).map(t => [t && t.id, t]));
      st.tasks = list.map(item => {
        if (!item || typeof item !== 'object') return item;
        const prev = prevMap.get(item.id) || {};
        const assignee = item.assignee !== undefined ? item.assignee : prev.assignee;
        const isDone = DONE_STATUS.test(item.status || prev.status || '');
        const workFinished = prev._workFinished || (isDone && !isAgentActive(assignee));
        return {
          id: item.id || prev.id,
          title: item.title !== undefined ? item.title : (prev.title || ''),
          description: (item.description !== undefined && item.description !== null && item.description !== '')
            ? item.description
            : (prev.description || ''),
          notes: (item.notes !== undefined && item.notes !== null && item.notes !== '')
            ? item.notes
            : (prev.notes || ''),
          status: item.status || prev.status || 'TODO',
          assignee: assignee,
          parent_id: item.parent_id !== undefined ? item.parent_id : prev.parent_id,
          _workFinished: workFinished,
        };
      });
    } else if (result.task && result.task.id) {
      // `update_task_status` reports only the item it changed.
      const updated = result.task;
      const known = st.tasks.find(item => item && item.id === updated.id);
      if (known) {
        if (updated.status !== undefined) {
          known.status = updated.status;
          if (DONE_STATUS.test(updated.status)) {
            known._workFinished = !isAgentActive(known.assignee);
          } else {
            known._workFinished = false;
          }
        }
        if (updated.title) known.title = updated.title;
        if (updated.description !== undefined && updated.description !== '') known.description = updated.description;
        if (updated.notes !== undefined && updated.notes !== '') known.notes = updated.notes;
        if (updated.assignee !== undefined) known.assignee = updated.assignee;
        if (updated.parent_id !== undefined) known.parent_id = updated.parent_id;
      } else {
        const isDone = DONE_STATUS.test(updated.status || '');
        st.tasks.push({
          id: updated.id,
          title: updated.title || '',
          description: updated.description || '',
          notes: updated.notes || '',
          status: updated.status || 'TODO',
          assignee: updated.assignee,
          parent_id: updated.parent_id,
          _workFinished: isDone && !isAgentActive(updated.assignee),
        });
      }
    } else {
      return;
    }
    st.plan = summarise(st.tasks);

    // Keep RoadmapModal in sync
    if (window.RoadmapModal && typeof window.RoadmapModal.updateTasks === 'function') {
      window.RoadmapModal.updateTasks(st.tasks, false);
    }
  }

  /** The sandbox agent's plan, in the shape its service publishes:
   *  `{revision, current, progress: {total, done}, items: [{title, status}]}`.
   *  `progress` is trusted when it is there, `items` is the fallback. */
  function readSandboxPlan(plan) {
    if (!plan || typeof plan !== 'object') return null;
    const fromItems = summarise(plan.items);
    const progress = (plan.progress && typeof plan.progress === 'object') ? plan.progress : {};
    const total = Number(progress.total);
    const done = Number(progress.done);
    const current = typeof plan.current === 'string' && plan.current.trim()
      ? trim(plan.current)
      : (fromItems && fromItems.current);
    if (!Number.isFinite(total) || total <= 0) return fromItems;
    return {
      total: total,
      done: Number.isFinite(done) ? done : 0,
      current: current || null,
    };
  }

  // ── The reducer ───────────────────────────────────────────────────────────
  function feed(msg, quiet) {
    if (!msg || !msg.type) return;
    const now = Date.now();

    switch (msg.type) {
      case 'user_message':
        startRun(stamp(msg.timestamp));
        break;

      case '__typing__':
        // `showTyping()` from the legacy call sites: a run is live, but no
        // event of its own has arrived yet (e.g. right after a reconnect).
        if (st.phase === 'idle') startRun(now);
        break;

      case 'status':
        if (msg.status === 'processing') {
          if (st.phase === 'idle') startRun(now);
        } else if (st.phase !== 'idle' && st.hideAt === 0 && st.phase !== 'stopped') {
          endRun('done', DONE_LINGER_MS);
        }
        break;

      case 'agent_event':
        st.lastEventAt = now;
        if ((st.phase === 'waiting' || st.phase === 'waiting_frame') && msg.author && msg.author !== 'user') {
          st.phase = 'thinking';
          st.phraseSince = now;
        }
        // The export replays the user's own turns as authored events too.
        if (msg.author && msg.author !== 'user') st.agent = msg.author;
        st.note = null;
        refresh();
        break;

      case 'agent_output':
        st.lastEventAt = now;
        if (st.phase === 'waiting' || st.phase === 'waiting_frame') {
          st.phase = 'thinking';
          st.phraseSince = now;
        }
        if (msg.agent) {
          st.agent = msg.agent;
          markAgentWorkFinished(msg.agent);
        }
        pushStep(null, 'task_alt', agentLabel(msg.agent) + ' — ' + pick(TEXT.ready)).status = 'done';
        // A subordinate just delivered; the caller is reading its report now.
        st.afterglow = { category: 'delegation', agent: msg.agent, at: now };
        refresh();
        break;

      case 'tool_activity':
        onToolActivity(msg, now);
        break;

      case 'sandbox_plan':
        // The agent inside the container rewrote its task list. This is the
        // only news that arrives during a sandbox run, which is the longest
        // single thing the system does.
        st.lastEventAt = now;
        if (st.phase === 'waiting' || st.phase === 'waiting_frame') {
          st.phase = 'thinking';
          st.phraseSince = now;
        }
        st.sandboxPlan = readSandboxPlan(msg.plan);
        if (msg.agent) st.agent = msg.agent;
        break;

      case 'session_snapshot':
        if (msg.active_tasks && Array.isArray(msg.active_tasks)) {
          readPlan({ tasks: msg.active_tasks });
        }
        break;

      case 'tasks_updated':
        if (msg.tasks && Array.isArray(msg.tasks)) {
          readPlan({ tasks: msg.tasks });
        }
        refresh();
        break;

      case 'hitl_request':
        st.lastEventAt = now;
        const isFrameReq = (msg.agent_name === 'ContextInitAgent' || msg.agent_name === 'ContextInitSessionAgent')
          || Boolean(msg.form)
          || frameStage();
        setPhase(isFrameReq ? 'waiting_frame' : 'waiting', { agent: msg.agent_name || st.agent }, true);
        break;

      case 'hitl_timeout':
      case 'hitl_cancelled':
      case 'hitl_response':
        if (st.phase === 'waiting' || st.phase === 'waiting_frame') {
          st.phase = 'thinking';
          st.phraseSince = now;
          refresh();
          render();
        }
        break;

      case 'final_response':
        if (msg.content === 'Stopped' || st.phase === 'stopped') {
          endRun('stopped', STOPPED_LINGER_MS);
        } else {
          endRun('done', DONE_LINGER_MS);
        }
        break;

      case 'error':
        // Deliberately not showing `msg.message`: it is a Python traceback
        // summary, and it is already posted into the chat as a system message.
        st.note = null;
        endRun('error', ERROR_LINGER_MS);
        break;

      default:
        return;
    }
    if (!quiet) render();
  }

  function onToolActivity(msg, now) {
    const tool = msg.tool;
    if (!tool) return;
    if (st.phase === 'idle') startRun(now);
    st.lastEventAt = now;
    if (st.phase === 'waiting' || st.phase === 'waiting_frame') {
      st.phase = 'thinking';
      st.phraseSince = now;
    }
    const author = msg.author || 'system';

    if (msg.phase === 'call') {
      const target = delegationTarget(tool, msg.args);
      if (target) {
        // A delegation is a call like any other: it has an id, it gets a
        // result, and it is open the whole time the subordinate works. Putting
        // it on the stack is what keeps the line honest for those hours.
        st.agent = author;
        openWork(msg.call_id, {
          kind: 'delegation', tool: tool, agent: author, target: target, at: now,
        });
        pushStep(msg.call_id, 'alt_route',
          pick(PHASES.delegating) + ': ' + agentLabel(target));
        st.note = null;
        refresh();
        return;
      }
      const category = classify(tool, msg.description);
      const detail = detailOf(msg.args);
      openWork(msg.call_id, {
        kind: 'tool', tool: tool, agent: author, category: category,
        detail: detail, at: now,
      });
      pushStep(msg.call_id, CATEGORIES[category].icon, phraseFor(category, tool));
      st.note = null;
      refresh();
      return;
    }

    // result / error
    const failed = msg.phase === 'error';
    const closed = closeWork(msg.call_id);
    if (!failed && closed && closed.kind === 'delegation' && closed.target) {
      markAgentWorkFinished(closed.target);
    }
    // What the model is about to read. Only meaningful once nothing is left
    // running — with work still open, that work is the better answer.
    if (!failed) {
      st.afterglow = {
        category: closed && closed.kind === 'delegation'
          ? 'delegation'
          : (closed ? closed.category : classify(tool, msg.description)),
        toolName: closed ? closed.tool : tool,
        agent: closed ? closed.agent : author,
        at: now,
      };
    }
    closeStep(msg.call_id, failed);
    // The sandbox plan describes a container that is now done with this task.
    if (/run_sandbox_task|check_sandbox_task/.test(String(tool))) st.sandboxPlan = null;
    if (failed) {
      // One bad tool is routine — the run continues. Say so calmly, and never
      // flip the whole indicator into an error state for it.
      st.note = pick(TEXT.toolFailed);
    } else if (/plan|task/i.test(String(tool))) {
      readPlan(msg.result);
    }
    refresh();
  }

  function phraseFor(category, tool) {
    const entry = CATEGORIES[category] || CATEGORIES.tool;
    // Last resort — an unrecognised (usually freshly built MCP) tool. Its name
    // is the only honest thing left to say, so at least make it readable.
    if (category === 'tool' && tool) {
      return pick(entry) + ' «' + String(tool).replace(/_/g, ' ') + '»';
    }
    return pick(entry);
  }

  // ── Rendering ─────────────────────────────────────────────────────────────
  let renderTimer = null;
  let lastRenderAt = 0;

  function render() {
    if (!root) return;
    const now = Date.now();
    if (now - lastRenderAt < RENDER_THROTTLE_MS) {
      if (!renderTimer) {
        renderTimer = setTimeout(() => { renderTimer = null; render(); },
          RENDER_THROTTLE_MS - (now - lastRenderAt));
      }
      return;
    }
    lastRenderAt = now;
    paint();
  }

  function view() {
    // What the user should be told right now, derived fresh on every paint so
    // silence and elapsed time can change the phrase with no new events.
    let phase = st.phase;
    if (phase !== 'idle' && !connected) phase = 'offline';
    // Silence only means "I have nothing to report" when nothing is running.
    // A sandbox task runs for forty minutes without emitting a single event,
    // and a delegation for hours: their silence is not absence of news, it IS
    // the news, and decaying it into "Думаю над задачей" is how this line came
    // to say nothing useful for 94% of a real run.
    const silent = st.lastEventAt && (Date.now() - st.lastEventAt) > SILENCE_MS;
    if (silent && !st.open.size && ['thinking', 'reviewing', 'starting'].includes(phase)) {
      phase = 'long_thinking';
    }

    let icon, text;
    if (phase === 'working') {
      const category = CATEGORIES[st.category] || CATEGORIES.tool;
      icon = category.icon;
      text = phraseFor(st.category, st.toolName);
    } else if (phase === 'delegating') {
      icon = PHASES.agentWorking.icon;
      text = pick(PHASES.agentWorking).replace('%s', agentLabel(st.target));
    } else if (phase === 'reviewing') {
      const category = CATEGORIES[st.category] || CATEGORIES.tool;
      icon = st.category === 'delegation' ? 'groups' : category.icon;
      if (st.toolName && st.category !== 'delegation') {
        const base = pick(REVIEW[st.category] || REVIEW.tool);
        text = `${base} «${st.toolName}»`;
      } else {
        text = pick(REVIEW[st.category] || REVIEW.tool);
      }
    } else {
      const entry = PHASES[phase] || PHASES.thinking;
      icon = entry.icon;
      text = pick(entry);
    }
    return { phase: phase, icon: icon, text: text };
  }

  function subLine(phase) {
    const parts = [];
    if (st.note) {
      parts.push(st.note);
    } else {
      if (st.agent && phase !== 'done' && phase !== 'stopped') parts.push(agentLabel(st.agent, true));
      const step = stepLine();
      // The step names the work better than a tool argument does, so it wins
      // the one slot they would otherwise share.
      if (!step && st.detail && phase === 'working') parts.push('«' + st.detail + '»');
      if (step) parts.push(step);
      // The apology is for the case where nothing can be said: with work open,
      // the line already names it and shows how long it has been running.
      if (!st.open.size && phase === 'long_thinking'
        && Date.now() - st.activitySince > LONG_STEP_MS) {
        parts.push(pick(TEXT.longRun));
      }
    }
    return parts.join(' · ');
  }

  // A plan title shares one line with the agent name and the phrase, so it is
  // cut shorter than a standalone detail would be.
  const STEP_TITLE_LIMIT = 44;

  /** "Шаг 2 из 5: Обучить модель" — the plan position, named.
   *
   *  The sandbox agent's plan wins when there is one: it describes the step
   *  running inside the container right now, while the task tracker describes
   *  the whole study. Its counter can go *down* — an agent may add items
   *  mid-run — which is why this is a caption and never a progress bar. */
  function stepLine() {
    const plan = st.sandboxPlan || st.plan;
    if (!plan || !plan.total) return null;
    const position = Math.min(plan.done + 1, plan.total);
    const counter = pick(TEXT.step).replace('%d', position).replace('%d', plan.total);
    if (!plan.current) return counter;
    const title = plan.current.length > STEP_TITLE_LIMIT
      ? plan.current.slice(0, STEP_TITLE_LIMIT - 1) + '…'
      : plan.current;
    return counter + ': ' + title;
  }

  function paint() {
    // Nothing to paint on yet. The app's own bootstrap runs before
    // DOMContentLoaded, so `reset()` / `feed()` legitimately arrive before
    // `mount()` — the state is kept, and the first paint after mounting shows
    // it. `render()` guards the same way.
    if (!root) return;
    if (st.tasks && st.tasks.length) {
      st.plan = summarise(st.tasks);
    }
    const current = view();
    if (current.phase === 'idle') {
      root.classList.add('hidden');
      root.innerHTML = '';
      return;
    }
    root.classList.remove('hidden');

    const tone = TONES[PHASE_TONE[current.phase] || 'work'];
    const live = ['working', 'thinking', 'starting', 'delegating', 'reviewing',
      'finalizing', 'long_thinking', 'framing'].includes(current.phase);
    const sub = subLine(current.phase);
    // The elapsed time of the *current step*, not of the run: on hour three of
    // a study the run total says nothing, while "12 мин" on this step says
    // whether to worry. The run total moves to the card's tooltip.
    const from = st.activitySince || st.since;
    const timer = from ? elapsed(Date.now() - from) : '';
    const title = st.since
      ? pick(TEXT.total) + ': ' + elapsed(Date.now() - st.since)
      : '';
    const steps = st.steps.slice().reverse();
    const activePlan = st.sandboxPlan || st.plan;
    const hasPlan = Boolean(activePlan && activePlan.total > 0);
    const planPercent = hasPlan ? Math.min(100, Math.round((activePlan.done / activePlan.total) * 100)) : 0;

    root.innerHTML = `
      <div class="si-card flex items-stretch gap-0 rounded-xl border ${tone.card} overflow-hidden transition-colors" title="${esc(title)}">
        <span class="w-1 shrink-0 ${tone.bar} ${live ? 'si-bar' : ''}"></span>
        <div class="flex-1 min-w-0 flex items-center gap-3 px-3 py-2.5">
          <span class="material-symbols-outlined text-lg ${tone.icon} ${live ? 'si-icon' : ''}">${current.icon}</span>
          <div class="flex-1 min-w-0">
            <div class="text-[13px] font-semibold leading-tight ${tone.text} ${live ? 'si-shimmer' : ''} truncate">${esc(current.text)}</div>
            ${sub ? `<div class="text-[10px] text-outline-variant leading-tight truncate mt-0.5">${esc(sub)}</div>` : ''}
            ${hasPlan ? `
            <div class="mt-1 flex items-center gap-2">
              <div class="si-mini-progress flex-1">
                <div class="si-mini-progress-fill" style="width: ${planPercent}%"></div>
              </div>
              <span class="text-[9px] font-mono text-outline-variant/80 shrink-0 tabular-nums">${activePlan.done}/${activePlan.total} (${planPercent}%)</span>
            </div>` : ''}
          </div>
          ${timer ? `<span class="text-[10px] font-mono text-outline-variant shrink-0 tabular-nums">${timer}</span>` : ''}
          <button type="button" title="${esc(pick(TEXT.details))}"
            class="si-toggle shrink-0 text-outline-variant hover:text-primary transition-colors flex items-center">
            <span class="material-symbols-outlined text-base">${expanded ? 'expand_more' : 'expand_less'}</span>
          </button>
        </div>
      </div>
      ${expanded && (steps.length || hasPlan) ? `
      <div class="mt-1.5 space-y-1.5">
        ${hasPlan ? `
        <div class="p-2.5 rounded-lg border border-outline-variant/15 bg-surface-container-lowest/80 ${planExpanded ? 'space-y-2' : ''}">
          <div class="flex items-center justify-between text-[11px] font-semibold text-on-surface cursor-pointer select-none si-plan-header">
            <div class="flex items-center gap-1.5">
              <span class="material-symbols-outlined text-sm text-primary">account_tree</span>
              <span>${esc(pick(TEXT.planHeader))}</span>
              <span class="text-[9px] font-mono font-normal text-outline-variant">(${activePlan.done} / ${activePlan.total})</span>
            </div>
            <div class="flex items-center gap-2">
              <button type="button" onclick="event.stopPropagation(); if (window.openRoadmapEditor) window.openRoadmapEditor();"
                class="text-[10px] text-primary hover:underline flex items-center gap-0.5 font-medium transition-colors">
                <span>${esc(pick(TEXT.openPlan))}</span>
                <span class="material-symbols-outlined text-[13px]">open_in_new</span>
              </button>
              <button type="button"
                class="si-toggle-plan text-outline-variant hover:text-primary transition-colors flex items-center p-0.5 rounded"
                title="${esc(pick(planExpanded ? TEXT.collapse : TEXT.expand))}">
                <span class="material-symbols-outlined text-[15px]">${planExpanded ? 'expand_less' : 'expand_more'}</span>
              </button>
            </div>
          </div>

          ${planExpanded ? `
          <div class="h-1 bg-surface-container-high rounded-full overflow-hidden">
            <div class="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-300 rounded-full" style="width: ${planPercent}%;"></div>
          </div>

          <div class="space-y-1 pt-0.5 max-h-80 overflow-y-auto pr-1">
            ${(() => {
            const list = st.tasks && st.tasks.length ? st.tasks : [];
            const displayList = showAllPlanTasks ? list : list.slice(0, 5);
            const remaining = list.length - 5;
            return displayList.map(task => {
              const isDone = isTaskDone(task);
              const isActive = isTaskActive(task);
              const norm = isDone ? 'done' : (isActive ? 'active' : 'todo');
              const icon = norm === 'done' ? 'check' : (norm === 'active' ? 'autorenew' : 'radio_button_unchecked');
              const tone = norm === 'done' ? 'text-outline-variant' : (norm === 'active' ? 'text-primary font-medium' : 'text-outline-variant');
              const spin = norm === 'active' ? 'si-icon' : '';
              const assignee = task.assignee ? agentLabel(task.assignee) : '';
              return `<div class="flex items-center gap-2 py-1 border-b border-outline-variant/10 last:border-b-0 text-[10px] ${tone}" title="${esc(task.title || '')}">
                  <span class="material-symbols-outlined text-[13px] shrink-0 ${spin}">${icon}</span>
                  <span class="font-medium flex-1 min-w-0 truncate">${task.id ? `<span class="font-mono text-[9px] opacity-75 mr-1">${esc(task.id)}</span>` : ''}${esc(task.title || '')}</span>
                  ${assignee ? `<span class="text-[8px] font-mono shrink-0 px-1 py-0.2 rounded bg-surface-container-high text-outline-variant/80">${esc(assignee)}</span>` : ''}
                </div>`;
            }).join('') + (remaining > 0 ? `
                <div class="pt-0.5 flex items-center justify-between text-[9px] font-mono">
                  <button type="button" class="si-toggle-plan-tasks text-primary/80 hover:underline">
                    ${showAllPlanTasks ? '↑ ' + (lang === 'ru' ? 'Свернуть задачи' : 'Show fewer') : '+ ' + pick(TEXT.moreTasks).replace('%d', remaining)}
                  </button>
                  <button type="button" onclick="if (window.openRoadmapEditor) window.openRoadmapEditor();" class="text-outline-variant hover:text-primary">
                    ${esc(pick(TEXT.openPlan))} →
                  </button>
                </div>` : '');
          })()}
          </div>` : ''}
        </div>` : ''}

        ${steps.length ? `
        <div class="px-3 py-2 rounded-lg border border-outline-variant/15 bg-surface-container-lowest/60 ${activityExpanded ? 'space-y-1' : ''}">
          <div class="text-[9px] font-mono text-outline-variant/70 ${activityExpanded ? 'mb-1' : ''} flex items-center justify-between cursor-pointer select-none si-activity-header">
            <div class="flex items-center gap-1">
              <span class="material-symbols-outlined text-xs">history</span>
              <span>${esc(pick(TEXT.recentActivity))}</span>
            </div>
            <div class="flex items-center gap-2">
              <button type="button" onclick="event.stopPropagation(); if (window.openToolsViewer) window.openToolsViewer();"
                class="text-[9px] text-primary hover:underline flex items-center gap-0.5 font-medium transition-colors">
                <span>${esc(pick(TEXT.openTools))}</span>
                <span class="material-symbols-outlined text-[12px]">open_in_new</span>
              </button>
              <button type="button"
                class="si-toggle-activity text-outline-variant hover:text-primary transition-colors flex items-center p-0.5 rounded"
                title="${esc(pick(activityExpanded ? TEXT.collapse : TEXT.expand))}">
                <span class="material-symbols-outlined text-[14px]">${activityExpanded ? 'expand_less' : 'expand_more'}</span>
              </button>
            </div>
          </div>
          ${activityExpanded ? steps.map(step => {
            const stepTone = step.status === 'error' ? 'text-error'
              : step.status === 'done' ? 'text-outline-variant' : 'text-primary';
            const stepIcon = step.status === 'error' ? 'close'
              : step.status === 'done' ? 'check' : step.icon;
            return `<div class="flex items-center gap-2 ${stepTone}">
              <span class="material-symbols-outlined text-[13px] ${step.status === 'running' ? 'si-icon' : ''}">${stepIcon}</span>
              <span class="text-[10px] truncate">${esc(step.text)}</span>
            </div>`;
          }).join('') : ''}
        </div>` : ''}
      </div>` : ''}`;

    const toggle = root.querySelector('.si-toggle');
    if (toggle) {
      toggle.addEventListener('click', () => {
        expanded = !expanded;
        try { localStorage.setItem(EXPAND_KEY, expanded ? 'on' : 'off'); } catch (_) { /* private mode */ }
        paint();
      });
    }

    const planHeader = root.querySelector('.si-plan-header');
    if (planHeader) {
      planHeader.addEventListener('click', (e) => {
        if (e.target.closest('button') && !e.target.closest('.si-toggle-plan')) {
          return;
        }
        planExpanded = !planExpanded;
        try { localStorage.setItem(PLAN_EXPAND_KEY, planExpanded ? 'on' : 'off'); } catch (_) {}
        paint();
      });
    }

    const activityHeader = root.querySelector('.si-activity-header');
    if (activityHeader) {
      activityHeader.addEventListener('click', (e) => {
        if (e.target.closest('button') && !e.target.closest('.si-toggle-activity')) {
          return;
        }
        activityExpanded = !activityExpanded;
        try { localStorage.setItem(ACTIVITY_EXPAND_KEY, activityExpanded ? 'on' : 'off'); } catch (_) {}
        paint();
      });
    }

    const planToggle = root.querySelector('.si-toggle-plan-tasks');
    if (planToggle) {
      planToggle.addEventListener('click', () => {
        showAllPlanTasks = !showAllPlanTasks;
        paint();
      });
    }
  }

  // The phrase can go stale on its own (silence, elapsed time), so repaint on a
  // slow tick as well as on events.
  setInterval(() => { if (st.phase !== 'idle') paint(); }, 1000);

  // ── Demo mode ─────────────────────────────────────────────────────────────
  // Replays the `agent_events.json` of a saved bundle straight into the
  // reducer, so the indicator can be developed without spending a real run.
  async function demo(filename, speed) {
    const factor = Number(speed) > 0 ? Number(speed) : 20;
    const response = await fetch('/api/saved-sessions/' + encodeURIComponent(filename) + '/events');
    if (!response.ok) throw new Error('demo: HTTP ' + response.status);
    const events = (await response.json()).events || [];
    reset();
    feed({ type: 'status', status: 'processing' });
    let previous = null;
    for (const event of events) {
      const at = stamp(event.timestamp);
      const gap = previous ? Math.min(Math.max((at - previous) / factor, 60), 2500) : 250;
      previous = at;
      await new Promise(resolve => setTimeout(resolve, gap));
      feed(event);
    }
    feed({ type: 'final_response' });
  }

  // ── Public API ────────────────────────────────────────────────────────────
  function mount(element) {
    root = element || null;
    try { expanded = localStorage.getItem(EXPAND_KEY) === 'on'; } catch (_) { expanded = false; }
    try { planExpanded = localStorage.getItem(PLAN_EXPAND_KEY) !== 'off'; } catch (_) { planExpanded = true; }
    try { activityExpanded = localStorage.getItem(ACTIVITY_EXPAND_KEY) !== 'off'; } catch (_) { activityExpanded = true; }
    paint();
    const demoFile = new URLSearchParams(location.search).get('demo');
    if (demoFile) {
      const speed = new URLSearchParams(location.search).get('demo_speed');
      demo(demoFile, speed).catch(err => console.warn('status demo failed:', err));
    }
  }

  function reset() {
    if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; }
    cancelSettle();
    Object.assign(st, newState());
    paint();
  }

  /** Every public entry point is guarded.
   *
   *  This module is an observer of someone else's event stream, called from
   *  the middle of the app's own control flow — the websocket switch, the
   *  session bootstrap, the language toggle. A fault in a status line must
   *  cost the status line and nothing else; without this, one bad frame
   *  propagates into `bootstrap()`'s catch and the user is asked to create an
   *  account by hand. Same rule the sandbox and tool-activity sinks follow. */
  function guarded(name, fn) {
    return function () {
      try {
        return fn.apply(null, arguments);
      } catch (error) {
        console.warn('StatusIndicator.' + name + ' failed:', error);
        return undefined;
      }
    };
  }

  window.StatusIndicator = {
    mount: guarded('mount', mount),
    feed: guarded('feed', feed),
    reset: guarded('reset', reset),
    demo: demo,   // async: the caller already handles its rejection
    setLang: guarded('setLang', function (value) { if (value) { lang = value; paint(); } }),
    setConnected: guarded('setConnected', function (value) { connected = !!value; render(); }),
    markStopped: guarded('markStopped', function () {
      endRun('stopped', STOPPED_LINGER_MS);
    }),
  };
})();
