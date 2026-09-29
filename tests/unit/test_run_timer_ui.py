"""Exercise the real timer, session status and budget UI with a fake clock."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_elapsed_time_survives_counter_updates_pause_and_reconnect(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    source = Path(__file__).resolve().parents[2] / "CoScientist/web/static/js"
    script = tmp_path / "timer.cjs"
    script.write_text(r'''
const assert = require('assert'), fs = require('fs'), path = require('path'), vm = require('vm');
const origin = Date.parse('2026-09-27T12:00:00Z');
let now = origin, intervalCount = 0, typingCount = 0;
const intervals = new Map(), nodes = new Map();
class Clock extends Date { static now() { return now; } }
function element() {
  const classes = new Set(['hidden']);
  return {textContent:'', innerHTML:'', dataset:{}, classList:{
    add: (...names) => names.forEach(name => classes.add(name)),
    remove: (...names) => names.forEach(name => classes.delete(name)),
    contains: name => classes.has(name),
    toggle: (name, on) => on ? classes.add(name) : classes.delete(name),
  }, addEventListener() {}, querySelector: () => null, querySelectorAll: () => []};
}
const sb = {console, Date:Clock, currentLang:'ru', runStatusVersion:-1, runActive:false,
  MutationObserver: class {observe() {}},
  currentPlannerHitlRequest:null, t: text => text, escHtml:String,
  document:{body:{dataset:{}}, addEventListener() {}, getElementById: id => {
    if (!nodes.has(id)) nodes.set(id, element()); return nodes.get(id);
  }}, localStorage:{getItem: () => null, setItem() {}},
  setInterval: callback => {const id = ++intervalCount; intervals.set(id, callback); return id;},
  clearInterval: id => intervals.delete(id),
  showTyping: () => {typingCount++;}, hideTyping() {}, resetAgents() {},
  activityMarkIdle() {}, updateRoadmapModalButtons() {},
};
sb.window = sb;
vm.createContext(sb);
for (const name of ['telemetry.js', 'sessions.js', 'run_control.js']) {
  vm.runInContext(fs.readFileSync(path.join(process.argv[2], name), 'utf8'), sb);
}
const timer = sb.RunTimer, control = sb.RunControl;
const advance = seconds => {now += seconds * 1000; [...intervals.values()].forEach(fn => fn());};
const duration = () => nodes.get('metrics-duration').textContent;
const run = {type:'run_control',run_id:'run-1',state:'running',revision:1,
  started_at:new Date(origin).toISOString(),pause_causes:[],budget:{used:0,limit:100}};
control.feed(run);
for (let calls = 1; calls <= 60; calls++) {
  advance(1);
  control.feed({...run, budget:{used:calls,limit:100}});
}
assert.equal(duration(), '01:00');
assert.equal(timer.startedAt, origin);
assert.equal(intervalCount, 1, 'Counter updates must not recreate the timer interval');
assert.equal(typingCount, 1, 'Counter updates must not replay processing lifecycle events');
sb.applyRunStatus('processing'); // A duplicate legacy status is harmless too.
assert.equal(duration(), '01:00');
assert.equal(intervalCount, 1);
const paused = {...run,state:'paused',revision:2,pause_causes:['manual']};
control.feed(paused);
advance(30);
control.feed({...run,revision:3});
assert.equal(duration(), '01:30', 'Resume uses the original start');
assert.equal(timer.startedAt, origin);

timer.reset(); // The page's session-snapshot path resets the old session UI.
const snapshot = {type:'session_snapshot',status:'paused',run_control:paused,
  run_times:{started_at:new Date(now).toISOString(),finished_at:null}};
timer.restoreFromSnapshot(snapshot);
control.feed(snapshot);
advance(10);
assert.equal(duration(), '01:40', 'Paused reconnect uses durable origin, not reload time');
assert.equal(timer.isRunning, true);
const completed = {...run,state:'completed',revision:4};
control.feed(completed);
advance(20);
control.feed(completed);
assert.equal(duration(), '01:40', 'Completed duration must stay frozen');

const finishedSnapshot = {type:'session_snapshot',status:'idle',run_control:completed,
  run_times:{started_at:run.started_at,finished_at:new Date(origin + 100000).toISOString()}};
timer.reset();
timer.restoreFromSnapshot(finishedSnapshot);
control.feed(finishedSnapshot);
assert.equal(duration(), '01:40', 'Idle status must not overwrite restored duration');
timer.start(); // A genuinely new run starts from its own origin.
advance(7);
assert.equal(duration(), '00:07');
console.log('ok');
''', encoding="utf-8")
    result = subprocess.run([node, str(script), str(source)], capture_output=True,
                            text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
