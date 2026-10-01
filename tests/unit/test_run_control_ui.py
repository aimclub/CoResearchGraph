"""The run budget lives in session spend without hiding pending decisions."""
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess

import pytest


WEB = Path(__file__).resolve().parents[2] / "CoScientist/web"


def test_budget_panel_is_inside_sidebar_spend_not_chat_composer():
    class Layout(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.locations = {}

        def handle_starttag(self, tag, attrs):
            identity = dict(attrs).get("id")
            if identity:
                self.locations.setdefault(identity, []).append([item[1] for item in self.stack])
            if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
                self.stack.append((tag, identity))

        def handle_endtag(self, tag):
            for index in range(len(self.stack) - 1, -1, -1):
                if self.stack[index][0] == tag:
                    del self.stack[index:]
                    break

    layout = Layout()
    layout.feed((WEB / "templates/index.html").read_text(encoding="utf-8"))
    panels = layout.locations["run-control-panel"]
    assert len(panels) == 1
    assert "side-rail" in panels[0] and "usage-panel" in panels[0]
    assert "chat-form" not in panels[0]
    assert "chat-form" in layout.locations["pause-run-btn"][0]
    assert "chat-form" in layout.locations["resume-run-btn"][0]


def test_counter_respects_collapsed_spend_but_reveals_new_budget_decision(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = tmp_path / "run-control.cjs"
    script.write_text(r'''
const fs = require('fs'), vm = require('vm');
const listeners = {}, posts = [];
const classList = {toggle() {}, add() {}, remove() {}};
const usage = {open: false};
const panel = {classList, innerHTML: '', querySelectorAll: () => [],
  querySelector: selector => panel.innerHTML.includes(selector.slice(1, -1))
    ? {addEventListener: (event, callback) => {listeners[selector] = callback;}} : null};
const button = () => ({classList});
const nodes = {'run-control-panel': panel, 'usage-panel': usage,
  'pause-run-btn': button(), 'resume-run-btn': button()};
const run = {type:'run_control', run_id:'r', state:'running', revision:1,
  pause_causes:[], budget:{used:1, limit:100}};
const sb = {console, currentLang:'ru', activeUser:{id:'u'}, activeSession:{id:'s'},
  document:{getElementById: id => nodes[id] || null, body:{dataset:{}}}, escHtml:String,
  crypto:{randomUUID: () => 'decision-1'}, addSystemMsg: text => {throw new Error(text);},
  fetch: async (url, options) => {posts.push({url, body:JSON.parse(options.body)});
    return {ok:true, json:async () => ({run:{...run, revision:3, budget:{used:100, limit:200}}})};},
};
sb.window = sb;
vm.createContext(sb);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), sb);
(async () => {
  sb.RunControl.feed(run);
  const normalCollapsed = !usage.open;
  const counter = panel.innerHTML.includes('1 / 100');
  const paused = {...run, revision:2, state:'paused', pause_causes:['budget_exhausted'], budget:{used:100, limit:100}};
  sb.RunControl.feed(paused);
  const decisionOpened = usage.open;
  usage.open = false; // The reader can still fold a decision they already saw.
  sb.RunControl.feed(paused);
  const sameDecisionRespectsFold = !usage.open;
  await listeners['[data-budget-continue]']();
  process.stdout.write(JSON.stringify({normalCollapsed, counter, decisionOpened, sameDecisionRespectsFold, posts}));
})().catch(error => {console.error(error); process.exitCode = 1;});
''', encoding="utf-8")
    done = subprocess.run([node, str(script), str(WEB / "static/js/run_control.js")],
                          capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert all(result[key] for key in ("normalCollapsed", "counter", "decisionOpened", "sameDecisionRespectsFold"))
    assert len(result["posts"]) == 1
    assert result["posts"][0]["url"].endswith("/runs/r/budget/decision")
    assert result["posts"][0]["body"] == {"expected_revision": 2, "decision": "continue", "decision_id": "decision-1"}


def test_scientific_pauses_have_explicit_actions_and_keep_budget_guard(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = tmp_path / "scientific-control.cjs"
    script.write_text(r'''
const fs = require('fs'), vm = require('vm');
const listeners = {}, posts = [];
const cls = () => ({hidden:false, toggle(key, value) {if(key==='hidden') this.hidden=value;},
  add(key) {if(key==='hidden') this.hidden=true;}});
const usage = {open:false};
const panel = {classList:cls(), innerHTML:'', querySelectorAll:()=>[],
  querySelector: selector => panel.innerHTML.includes(selector.slice(1,-1))
    ? {addEventListener: (_, callback) => {listeners[selector]=callback;}} : null};
const resume = {classList:cls()}, pause = {classList:cls()};
const nodes = {'run-control-panel':panel, 'usage-panel':usage, 'resume-run-btn':resume, 'pause-run-btn':pause};
let sequence=0, payload;
const sb = {console, currentLang:'ru', activeUser:{id:'u'}, activeSession:{id:'s'},
  document:{getElementById:id=>nodes[id],body:{dataset:{}}}, escHtml:String,
  crypto:{randomUUID:()=>`d-${++sequence}`}, addSystemMsg:text=>{throw new Error(text);},
  fetch:async(url, options)=>{posts.push({url,body:JSON.parse(options.body)});
    return {ok:true,json:async()=>({run:payload})};}};
sb.window=sb; vm.createContext(sb);
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),sb);
const base={type:'run_control',run_id:'r',state:'paused',revision:1,budget:{used:62,limit:100}};
const feed=(causes,decisions)=>{payload={...base,revision:++sequence,pause_causes:causes,pending_decisions:decisions};sb.RunControl.feed(payload);};
(async()=>{
  feed(['planning_review'],{planning_review:{kind:'experiment_plan_recovery',reason:'max_plan_revisions'}});
  const savedPlanResume=!resume.classList.hidden && usage.open;
  await sb.RunControl.resume();
  feed(['planning_review','budget_exhausted'],{planning_review:{kind:'experiment_plan_recovery'}});
  const independentBudgetBlocksResume=resume.classList.hidden;
  feed(['planning_review'],{planning_review:{kind:'plan_capacity_conflict',can_raise_limit:false}});
  const noImpossibleApproval=!panel.innerHTML.includes('data-approve-capacity') && resume.classList.hidden;
  feed(['planning_review'],{planning_review:{kind:'plan_capacity_conflict',can_raise_limit:true}});
  await listeners['[data-approve-capacity]']();
  feed(['scientific_work_incomplete'],{scientific_work_incomplete:{kind:'scientific_outcome',reason:'root_roadmap_incomplete',unfinished_task_ids:['TASK-1','TASK-3']}});
  await listeners['[data-outcome-continue]']();
  await listeners['[data-outcome-limited]']();
  process.stdout.write(JSON.stringify({savedPlanResume,independentBudgetBlocksResume,noImpossibleApproval,posts}));
})().catch(error=>{console.error(error);process.exitCode=1;});
''', encoding="utf-8")
    done = subprocess.run([node, str(script), str(WEB / "static/js/run_control.js")],
                          capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["savedPlanResume"]
    assert result["independentBudgetBlocksResume"]
    assert result["noImpossibleApproval"]
    assert result["posts"][0]["url"].endswith("/runs/r/resume")
    decisions = result["posts"][1:]
    assert all(row["url"].endswith("/runs/r/outcome/decision") for row in decisions)
    assert [row["body"]["decision"] for row in decisions] == ["approve_capacity", "continue", "accept_limited"]
    assert decisions[-1]["body"]["accepted_item_ids"] == ["TASK-1", "TASK-3"]
    assert all(row["body"]["confirmed"] is True for row in decisions)
