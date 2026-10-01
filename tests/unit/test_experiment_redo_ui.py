"""Client-side task selection is explicit and survives card redraws."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_result_acceptance_is_not_a_retry_and_redo_has_explicit_scope(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = Path(__file__).resolve().parents[2] / "CoScientist/web/static/js/hitl.js"
    script = tmp_path / "redo.cjs"
    script.write_text(r'''
const vm = require('vm'), fs = require('fs');
const sent = [], notes = [], feedback = {value: 'use a different dataset'}, scope = {};
const checkbox = {dataset: {redoTask: 'EXP-1'}, checked: false}, button = {};
const card = {
  querySelector: () => null,
  querySelectorAll: sel => sel === 'input[data-redo-task]:checked' ? (checkbox.checked ? [checkbox] : [])
    : sel === 'input[data-redo-task]' ? [checkbox] : [],
};
const sb = {console, currentLang: 'en', CSS: {escape: s => s},
  document: {addEventListener() {}, querySelector: () => card, getElementById: id => id.startsWith('hitl-feedback-') ? feedback
    : id.startsWith('hitl-redo-scope-') ? scope : id.startsWith('hitl-redo-') ? button : null},
  escHtml: String, escJs: String, t: s => s, addSystemMsg: s => notes.push(s),
  currentPlannerHitlRequest: null, setTimeout, clearTimeout,
};
sb.window = sb;
vm.createContext(sb);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), sb);
vm.runInContext(`hitlCards.set('r', {context:{experiment_targeted_redo:{
  task_choices:[{task_id:'EXP-1',name:'Fit'},{task_id:'EXP-2',name:'Analyze'},{task_id:'EXP-3',name:'Independent'}],
  affected_by_task:{'EXP-1':['EXP-1','EXP-2']}}}});`, sb);
sb.sendHitlResponse = payload => {sent.push(payload); return true;};
sb.respondHITLApprove('r');
sb.respondExperimentRedo('r'); // nothing selected: no packet
checkbox.checked = true;
sb.syncExperimentRedo('r');
const preview = scope.textContent;
const enabled = !button.disabled;
sb.captureHitlCardState('r');
checkbox.checked = false;
sb.restoreHitlCardState('r');
const restored = checkbox.checked;
sb.respondExperimentRedo('r');
process.stdout.write(JSON.stringify({sent, notes, preview, enabled, restored}));
''', encoding="utf-8")
    done = subprocess.run([node, str(script), str(js)], capture_output=True, text=True,
                          encoding="utf-8", timeout=30)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert len(result["sent"]) == 2
    assert result["sent"][0]["approved"] is True
    assert "selected_task_ids" not in result["sent"][0]
    assert result["sent"][1]["selected_task_ids"] == ["EXP-1"]
    assert result["sent"][1]["approved"] is False
    assert "EXP-1, EXP-2" in result["preview"]
    assert "EXP-3" not in result["preview"]
    assert result["enabled"] and result["restored"]
    assert len(result["notes"]) == 1
