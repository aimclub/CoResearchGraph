/**
 * How much of a run does the status indicator spend on each phrase?
 *
 * A unit test proves a transition is correct. It cannot tell you that the line
 * says "Думаю над задачей" for 94% of an eight-hour study — which is what this
 * one measured, and what nothing else would have caught: every transition in
 * that run was individually right.
 *
 * The reducer is loaded unmodified and driven by a recorded event log at the
 * timestamps the events actually carry. Timers are virtualised along with the
 * clock, because the anti-flicker hold and the silence decay are exactly what
 * is under question here — a real setTimeout would never fire in a replay that
 * takes two seconds to cover eight hours.
 *
 * Usage (via the Python driver, which unpacks the bundle for you):
 *     python scripts/status_timeline.py <bundle.cossession.zip> [--lines]
 *
 * Directly, with an events JSON of the form {"events": [...]}:
 *     node scripts/status_timeline.js <status_indicator.js> <events.json>
 *     SHOW_LINES=1 node ...   # also print the rendered line once an hour
 */
'use strict';

// ── Virtual clock and timer queue ───────────────────────────────────────────
let clock = 0;
let seq = 0;
const timers = new Map();

Date.now = () => clock;
global.setTimeout = (fn, ms) => {
  const id = ++seq;
  timers.set(id, { at: clock + (ms || 0), fn, every: null });
  return id;
};
global.setInterval = (fn, ms) => {
  const id = ++seq;
  timers.set(id, { at: clock + (ms || 0), fn, every: ms || 1 });
  return id;
};
global.clearTimeout = (id) => timers.delete(id);
global.clearInterval = (id) => timers.delete(id);

function advance(to) {
  for (;;) {
    let next = null;
    for (const [id, timer] of timers) {
      if (timer.at <= to && (next === null || timer.at < next[1].at)) next = [id, timer];
    }
    if (!next) break;
    const [id, timer] = next;
    clock = timer.at;
    if (timer.every) timer.at = clock + timer.every;
    else timers.delete(id);
    timer.fn();
  }
  clock = to;
}

// ── Just enough DOM ─────────────────────────────────────────────────────────
const store = {};
global.localStorage = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
};

function fakeEl() {
  return {
    innerHTML: '',
    set textContent(value) { this.innerHTML = String(value == null ? '' : value); },
    classList: { add() {}, remove() {}, contains() { return false; } },
    querySelector() { return { addEventListener() {} }; },
  };
}

const root = fakeEl();
global.document = { createElement: () => fakeEl() };
global.location = { search: '' };
global.window = global;
global.fetch = () => Promise.reject(new Error('offline'));

// ── Replay ──────────────────────────────────────────────────────────────────
const [modulePath, eventsPath] = process.argv.slice(2);
if (!modulePath || !eventsPath) {
  console.error('usage: node status_timeline.js <status_indicator.js> <events.json>');
  process.exit(2);
}
require(require('path').resolve(modulePath));
StatusIndicator.mount(root);

const events = require(require('path').resolve(eventsPath)).events || [];
const at = (event) => (event.timestamp ? new Date(event.timestamp).getTime() : null);
const stamped = events.filter(at);
if (!stamped.length) {
  console.error('no timestamped events in that log');
  process.exit(1);
}
clock = at(stamped[0]) - 1000;

const strip = (html) => html.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
const phrase = () => {
  const m = root.innerHTML.match(/text-\[13px\][^>]*>([^<]*)</);
  return m ? m[1].trim() : '(hidden)';
};

const spent = new Map();
const samples = [];
let last = phrase();
let lastAt = clock;
let ticks = 0;
const STEP_MS = 500;
const SAMPLE_EVERY = 120;   // one rendered line per minute of run time

function sample() {
  const now = phrase();
  if (now !== last) {
    spent.set(last, (spent.get(last) || 0) + (clock - lastAt));
    last = now;
    lastAt = clock;
  }
}

for (const event of stamped) {
  const target = at(event);
  while (clock + STEP_MS < target) {
    advance(clock + STEP_MS);
    sample();
    if (++ticks % SAMPLE_EVERY === 0) samples.push([clock, strip(root.innerHTML)]);
  }
  advance(target);
  StatusIndicator.feed(event);
  sample();
}
advance(clock + 2000);
sample();
spent.set(last, (spent.get(last) || 0) + (clock - lastAt));

// ── Report ──────────────────────────────────────────────────────────────────
const total = [...spent.values()].reduce((a, b) => a + b, 0);
const started = at(stamped[0]);

if (process.env.SHOW_LINES && samples.length) {
  const step = Math.max(1, Math.floor(samples.length / 12));
  console.log('— the line, at points across the run —');
  for (let i = 0; i < samples.length; i += step) {
    const [when, line] = samples[i];
    console.log(`  +${((when - started) / 60000).toFixed(0).padStart(4)} min  ${line}`);
  }
  console.log('');
}

console.log(`run: ${(total / 60000).toFixed(0)} min, ${stamped.length} events\n`);
[...spent.entries()]
  .sort((a, b) => b[1] - a[1])
  .forEach(([text, ms]) => {
    const share = (ms / total * 100).toFixed(1).padStart(5);
    console.log(`${share}%  ${(ms / 60000).toFixed(0).padStart(4)} min  ${text}`);
  });
