// Drives CoScientist/web/static/js/plan_tracker.js (stubbed DOM, stubbed
// status indicator and roadmap) and checks the rows it paints. Run by
// tests/unit/test_plan_tracker_stages.py: `node <this> <plan_tracker.js>`.
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const element = () => ({
  hidden: false,
  classList: {
    add(name) { if (name === 'hidden') this.owner.hidden = true; },
    remove(name) { if (name === 'hidden') this.owner.hidden = false; },
  },
  style: {}, textContent: '', innerHTML: '', scrollTop: 0, clientHeight: 100,
  querySelector: () => null,
});
const elements = {};
['plan-tracker', 'plan-tracker-list', 'plan-tracker-count', 'plan-tracker-bar'].forEach(id => {
  elements[id] = element();
  elements[id].classList.owner = elements[id];
});

let stages = [];
let tasks = [];
const STAGE_OF = { TZSpecAgent: 0, PlannerAgent: 1, LiteratureOrchestrator: 2, ResearchAgent: 2, ReportAgent: 3 };
const context = {
  document: { getElementById: id => elements[id] || null },
  CSS: { escape: s => s },
  t: (key, fallback) => fallback || key,
  escHtml: esc,
  RoadmapModal: {
    getTasks: () => tasks,
    normalizeStatus: s => (/done/i.test(s) ? 'done' : /progress/i.test(s) ? 'in_progress' : 'todo'),
  },
  StatusIndicator: {
    stages: () => stages,
    stageOf: name => (name in STAGE_OF ? STAGE_OF[name] : -1),
  },
};
context.window = context;
vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), context);
const tracker = context.PlanTracker;

/** The painted list as "indent status title" lines. */
function rows() {
  const out = [];
  const re = /<li[^>]*class="([^"]*)"[^>]*>\s*<span[^>]*>([^<]*)<\/span>\s*<span[^>]*>[^<]*<\/span>\s*<span[^>]*>([^<]*)<\/span>/g;
  let m;
  while ((m = re.exec(elements['plan-tracker-list'].innerHTML))) {
    out.push(`${/ml-5/.test(m[1]) ? '  ' : ''}${m[2]} ${m[3]}`);
  }
  return out;
}

// Nothing to show: the card stays out of the layout.
tracker.render();
assert.strictEqual(elements['plan-tracker'].hidden, true);

// A linear pipeline is a plan before any roadmap exists.
stages = [
  { agent: 'TZSpecAgent', title: 'Техническое задание', status: 'done' },
  { agent: 'PlannerAgent', title: 'План исследования', status: 'done' },
  { agent: 'LiteratureOrchestrator', title: 'Анализ литературы', status: 'in_progress' },
  { agent: 'ReportAgent', title: 'Итоговый отчёт', status: 'todo' },
];
tracker.render();
assert.strictEqual(elements['plan-tracker'].hidden, false);
assert.strictEqual(elements['plan-tracker-count'].textContent, '2/4');
assert.strictEqual(elements['plan-tracker-bar'].style.width, '50%');
assert.deepStrictEqual(rows(), [
  'check_circle Техническое задание',
  'check_circle План исследования',
  'autorenew Анализ литературы',
  'radio_button_unchecked Итоговый отчёт',
]);

// The roadmap's tasks nest under the stage of their assignee; one without an
// assignee goes with the others. The count still measures the stages.
tasks = [
  { id: 'task_1', title: 'LIT-01: betaine', status: 'DONE', assignee: 'ResearchAgent' },
  { id: 'task_2', title: 'LIT-02: salinity', status: 'IN_PROGRESS', assignee: 'ResearchAgent' },
  { id: 'task_3', title: 'LIT-03: <cost>', status: 'TODO' },
];
tracker.render();
assert.strictEqual(elements['plan-tracker-count'].textContent, '2/4');
assert.deepStrictEqual(rows(), [
  'check_circle Техническое задание',
  'check_circle План исследования',
  'autorenew Анализ литературы',
  '  check_circle LIT-01: betaine',
  '  autorenew LIT-02: salinity',
  '  radio_button_unchecked LIT-03: &lt;cost&gt;',
  'radio_button_unchecked Итоговый отчёт',
]);

// Not a linear pipeline: the plain roadmap, as before.
stages = [];
tracker.render();
assert.strictEqual(elements['plan-tracker-count'].textContent, '1/3');
assert.deepStrictEqual(rows(), [
  'check_circle LIT-01: betaine',
  'autorenew LIT-02: salinity',
  'radio_button_unchecked LIT-03: &lt;cost&gt;',
]);

console.log('ok');
