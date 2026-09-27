// Drives CoScientist/web/static/status_indicator.js with a linear pipeline's
// events (stubbed DOM) and checks the line it paints. Run by
// tests/unit/test_status_indicator_stages.py: `node <this> <indicator.js>`.
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const document = {
  createElement: () => {
    let text = '';
    return { set textContent(v) { text = v; }, get innerHTML() { return esc(text); } };
  },
};
const root = {
  hidden: true,
  classList: { add() { root.hidden = true; }, remove() { root.hidden = false; } },
  innerHTML: '',
  querySelector: () => null,
};
// The plan tracker lists the stages; the indicator re-renders it on a change.
let trackerRenders = 0;
const context = {
  PlanTracker: { render() { trackerRenders += 1; } },
  document, console, URLSearchParams, setTimeout, clearTimeout,
  setInterval: () => 0,
  localStorage: { getItem: () => 'on', setItem() {} },  // expanded view on
  location: { search: '' },
};
context.window = context;
vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), context);
const SI = context.StatusIndicator;
SI.mount(root);

const STAGES = ['TZSpecAgent', 'TZQueryGenAgent', 'PlannerAgent', 'LiteratureOrchestrator', 'ReportAgent']
  .map((agent, i) => ({
    agent,
    title: ['Техническое задание', 'Поисковые запросы', 'План исследования', 'Анализ литературы', 'Итоговый отчёт'][i],
    members: agent === 'LiteratureOrchestrator' ? [agent, 'ResearchAgent'] : [agent],
  }));

function painted() {
  SI.setLang('ru');  // paints right away (render() is throttled)
  return root.hidden ? '' : root.innerHTML.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
}
const activity = (phase, author, extra) =>
  SI.feed(Object.assign({ type: 'tool_activity', phase, author }, extra || {}));
const statuses = () => SI.stages().map(stage => stage.status).join(' ');

SI.feed({ type: 'session_snapshot', pipeline_stages: STAGES, active_tasks: [] });
SI.feed({ type: 'user_message', message: 'go' });
['RootOrchestrator', 'ModuleA_TZLiterature', 'TZAgent', 'TZSpecAgent', 'TZSpecAgent_task']
  .forEach(agent => activity('agent_start', agent));
activity('call', 'TZSpecAgent_task', { tool: 'fill_tz_section', call_id: 'c1' });
let line = painted();
assert.match(line, /Этап 1 из 5 · Техническое задание/, line);
assert.match(line, /0\/5 \(0%\)/, line);
assert.strictEqual(statuses(), 'in_progress todo todo todo todo');
assert.strictEqual(SI.stages()[0].title, 'Техническое задание');
assert.strictEqual(SI.stageOf('TZSpecAgent_task'), 0);
assert.strictEqual(SI.stageOf('ResearchAgent'), 3);
// Frames inside the same stage do not re-render the tracker.
const rendersBefore = trackerRenders;
activity('call', 'TZSpecAgent_task', { tool: 'fill_tz_section', call_id: 'c1b' });
assert.strictEqual(trackerRenders, rendersBefore);

SI.feed({ type: 'hitl_request', agent_name: 'TZSpecAgent', form: { kind: 'tz' } });
line = painted();
assert.match(line, /Жду вашего ответа/, line);           // not the research frame
assert.match(line, /Этап 1 из 5 · Техническое задание/, line);

SI.feed({ type: 'hitl_response', request_id: 'r' });
activity('agent_end', 'TZSpecAgent_task');                // a worker, not the stage
assert.match(painted(), /0\/5/);
activity('agent_end', 'TZSpecAgent');
assert.match(painted(), /1\/5 \(20%\)/);
assert.strictEqual(statuses(), 'done todo todo todo todo');
assert.strictEqual(trackerRenders, rendersBefore + 1);

activity('agent_start', 'TZQueryGenAgent');
line = painted();
assert.match(line, /Этап 2 из 5 · Поисковые запросы/, line);
assert.match(line, /1\/5/, line);

// A subordinate's tool call moves forward when its agent_start was trimmed…
activity('call', 'ResearchAgent', { tool: 'tavily_search', call_id: 'c2', args: { query: 'q' } });
assert.match(painted(), /Этап 4 из 5 · Анализ литературы/);
// …but a late call from an earlier stage does not move back.
activity('call', 'TZQueryGenAgent', { tool: 'x', call_id: 'c3' });
assert.match(painted(), /Этап 4 из 5/);
assert.strictEqual(statuses(), 'done done done in_progress todo');

// A failed run marks its stage failed, and keeps it so after the card hides;
// a stopped one leaves it pending.
SI.feed({ type: 'error', message: 'boom' });
assert.strictEqual(statuses(), 'done done done error todo');
SI.feed({ type: 'status', status: 'idle' });
assert.strictEqual(statuses(), 'done done done error todo');
SI.feed({ type: 'user_message', message: 'retry' });
activity('agent_start', 'LiteratureOrchestrator');
const rendersBeforeStop = trackerRenders;
SI.markStopped();
assert.strictEqual(statuses(), 'done done done todo todo');
assert.strictEqual(trackerRenders, rendersBeforeStop + 1);

// A session switch keeps the stages (they are config), not the position.
SI.reset();
SI.feed({ type: 'user_message', message: 'again' });
activity('agent_start', 'ReportAgent');
activity('agent_end', 'ReportAgent');
SI.feed({ type: 'final_response', content: 'ok' });
assert.match(painted(), /Готово.*5\/5 \(100%\)/);
assert.strictEqual(statuses(), 'done done done done done');

// Not a linear pipeline: the old line, no stage counter.
SI.feed({ type: 'session_snapshot', pipeline_stages: [] });
SI.feed({ type: 'user_message', message: 'go' });
activity('call', 'ResearchAgent', { tool: 'tavily_search', call_id: 'c9', args: { query: 'q' } });
line = painted();
assert.doesNotMatch(line, /Этап/, line);
assert.strictEqual(SI.stages().length, 0);

// ── The ТЗ cards: counted, no tools and no agents named ────────────────────
// No ТЗ tool or ТЗ agent is ever named; in a linear pipeline no module, root
// or request either (the stage names the work).
const TZ_NAMES = /fill[ _]tz|fill[ _]agent|TZSpecAgent|tzspecagent/i;
const ALL_NAMES = /fill[ _]tz|fill[ _]agent|TZSpecAgent|tzspecagent|ModuleA|rootorchestrator|Разработать/i;
const tzSnapshot = extra => SI.feed(Object.assign({ type: 'tz_snapshot', agent: 'TZSpecAgent' }, extra));

for (const stages of [STAGES, []]) {       // a linear pipeline, and not
  const NAMES = stages.length ? ALL_NAMES : TZ_NAMES;
  SI.feed({ type: 'session_snapshot', pipeline_stages: stages, tz: null });
  SI.feed({ type: 'user_message', message: 'go' });
  activity('agent_start', 'RootOrchestrator');
  // The root hands the module over: an AgentTool, flagged so by the server
  // (the name alone does not say it is an agent). Open for the whole module.
  activity('call', 'RootOrchestrator', {
    tool: 'ModuleA_TZLiterature', call_id: 'm1', is_delegation: true,
    target_agent: 'ModuleA_TZLiterature', args: { request: 'Разработать технологию' },
  });
  ['ModuleA_TZLiterature', 'TZAgent', 'TZSpecAgent', 'TZSpecAgent_task']
    .forEach(agent => activity('agent_start', agent));
  activity('call', 'TZSpecAgent_task', { tool: 'fill_tz_section', call_id: 't1', args: { section: 'Тип задачи' } });
  line = painted();
  assert.doesNotMatch(line, NAMES, line);  // before the first count arrives

  activity('result', 'TZSpecAgent_task', { tool: 'fill_tz_section', call_id: 't1', result: { status: 'ok' } });
  tzSnapshot({ phase: 'filling', progress: { filled: 3, total: 16 } });
  activity('call', 'TZSpecAgent_limits', { tool: 'fill_tz_section', call_id: 't2', args: { section: 'Ограничения по сырью' } });
  line = painted();
  assert.match(line, /Заполнено карточек ТЗ: 3 из 16/, line);
  assert.match(line, /3\/16 \(19%\)/, line);
  assert.doesNotMatch(line, NAMES, line);
  if (stages.length) assert.match(line, /Этап 1 из 5 · Техническое задание/, line);

  tzSnapshot({ phase: 'review', progress: { filled: 16, total: 16 } });
  SI.feed({ type: 'hitl_request', agent_name: 'TZSpecAgent', form: { kind: 'tz' } });
  assert.match(painted(), /Жду вашего ответа/);

  SI.feed({ type: 'hitl_response', request_id: 'r' });
  tzSnapshot({ phase: 'agent_filling', progress: { filled: 16, total: 16 }, agent_fill: { done: 0, total: 3 } });
  activity('agent_start', 'TZSpecAgent_quality_fill');
  activity('call', 'TZSpecAgent_quality_fill', { tool: 'fill_agent_fields', call_id: 't3' });
  assert.match(painted(), /Отредактировано карточек ТЗ: 0 из 3/);
  activity('result', 'TZSpecAgent_quality_fill', { tool: 'fill_agent_fields', call_id: 't3', result: { status: 'ok' } });
  tzSnapshot({ phase: 'agent_filling', progress: { filled: 16, total: 16 }, agent_fill: { done: 2, total: 3 } });
  line = painted();
  assert.match(line, /Отредактировано карточек ТЗ: 2 из 3/, line);
  assert.doesNotMatch(line, NAMES, line);

  // The ТЗ is done and the next agent works: no card count any more.
  tzSnapshot({ phase: 'final', progress: { filled: 16, total: 16 } });
  activity('agent_end', 'TZSpecAgent');
  activity('agent_start', 'TZQueryGenAgent');
  line = painted();
  assert.doesNotMatch(line, /карточек/, line);
  if (stages.length) {
    assert.match(line, /Этап 2 из 5 · Поисковые запросы/, line);
    assert.doesNotMatch(line, NAMES, line);  // the open module call is not a "tool"
  }
  SI.feed({ type: 'final_response', content: 'ok' });
  SI.reset();
}

// A run stopped half way through the ТЗ does not leave its count behind.
SI.feed({ type: 'session_snapshot', pipeline_stages: [], tz: { type: 'tz_snapshot', phase: 'filling', agent: 'TZSpecAgent', progress: { filled: 5, total: 16 } } });
SI.feed({ type: 'user_message', message: 'other' });
activity('call', 'ResearchAgent', { tool: 'tavily_search', call_id: 'x1', args: { query: 'q' } });
assert.doesNotMatch(painted(), /карточек/);

// A durable pause is not completion, even if an in-flight tool reports later.
SI.reset();
SI.feed({type: 'user_message', message: 'pause test'});
SI.feed({type: 'run_control', state: 'paused', pause_causes: ['manual']});
SI.feed({type: 'status', status: 'paused'});
activity('result', 'ResearchAgent', {tool: 'search', call_id: 'late', result: {status: 'ok'}});
assert.match(painted(), /Жду вашего ответа/);
assert.doesNotMatch(painted(), /Исследование завершено/);
SI.feed({type: 'hitl_timeout', paused: true});
assert.match(painted(), /Жду вашего ответа/);
SI.feed({type: 'run_control', state: 'running', pause_causes: []});
assert.doesNotMatch(painted(), /Жду вашего ответа/);

console.log('ok');
