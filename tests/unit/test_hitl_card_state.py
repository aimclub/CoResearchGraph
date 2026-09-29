"""Browser-state regression checks for HITL cards redrawn by localisation."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


JS = Path(__file__).resolve().parents[2] / "CoScientist" / "web" / "static" / "js"

_HARNESS = r"""
const fs = require('fs'), path = require('path'), vm = require('vm');
const dir = process.argv[2];
const cards = new Map(), sent = [], notices = [];
const classList = () => ({ add() {}, remove() {}, toggle() {}, contains: () => false });

function makeCard(rid) {
  const button = { disabled: false, classList: classList() };
  const feedback = { value: '', disabled: false, classList: classList(), focus() {} };
  const assumption = { dataset: { woAssumption: 'A1' }, checked: true, disabled: false };
  const controls = {
    dataset: {},
    querySelectorAll(sel) { return sel === 'button, textarea' ? [button, feedback] : []; },
  };
  const card = {
    dataset: { hitlCard: rid }, button, feedback, assumption, controls,
    querySelectorAll(sel) {
      if (sel === 'input[data-wo-assumption]') return [assumption];
      if (sel === 'input[data-wr-finding]') return [];
      if (sel === 'input[data-wo-assumption], input[data-wr-finding]') return [assumption];
      return [];
    },
    querySelector() { return null; },
  };
  feedback.closest = () => card;
  cards.set(rid, card);
  return card;
}

const document = {
  documentElement: { lang: 'ru', classList: classList(), setAttribute() {} },
  body: { classList: classList() },
  addEventListener() {}, createElement: () => ({ style: {}, classList: classList(), setAttribute() {}, appendChild() {} }),
  querySelector(selector) {
    const match = selector.match(/data-hitl-card="([^"]*)"/);
    return match ? (cards.get(match[1]) || null) : null;
  },
  querySelectorAll() { return []; },
  getElementById(id) {
    if (id.startsWith('hitl-controls-')) return cards.get(id.slice(14))?.controls || null;
    if (id.startsWith('hitl-feedback-')) return cards.get(id.slice(14))?.feedback || null;
    return null;
  },
};

const sb = {
  console, document, CSS: { escape: s => String(s) },
  localStorage: { getItem: () => 'ru', setItem() {} },
  setInterval: () => 0, clearInterval() {}, setTimeout: () => 0, clearTimeout() {},
  appendMsgToFeed() {}, scrollChat() {}, addTelemetry() {},
  addSystemMsg: text => notices.push(text), releasePlanGate() {}, updateRoadmapModalButtons() {},
  collapseFold() {}, documentBlock: () => '', foldable: html => html,
  renderMarkdown: s => String(s), mdInline: s => String(s), mdBlock: s => String(s),
  escHtml: s => String(s == null ? '' : s), escJs: s => String(s),
  ws: { readyState: 1, send: raw => sent.push(JSON.parse(raw)) },
};
sb.window = sb;
sb.StatusIndicator = { feed() {}, setLang() {}, agentName: name => name };

vm.createContext(sb);
vm.runInContext(fs.readFileSync(path.join(dir, 'i18n.js'), 'utf8'), sb);
vm.runInContext(fs.readFileSync(path.join(dir, 'hitl.js'), 'utf8'), sb);

// The regression concerns the state around a redraw, not HTML parsing. Mimic
// outerHTML replacement by installing a fresh set of controls on every render.
sb.renderWorkOrderCard = (_live, data) => makeCard(data.request_id);
const request = rid => ({
  request_id: rid, agent_name: 'Agent' + rid, trigger: 'work_order',
  context: { work_order: { assumptions: [{ id: 'A1', text: 'one' }], steps: [] } },
});

sb.showHITL(request('A'));
cards.get('A').feedback.value = 'keep this note';
cards.get('A').assumption.checked = false;
sb.respondWorkOrder('A', 'approve');

sb.showHITL(request('B'));
sb.respondWorkOrder('B', 'approve');
sb.relocalizeHitlCards();

const afterRedraw = {
  aDisabled: cards.get('A').button.disabled,
  bDisabled: cards.get('B').button.disabled,
  feedback: cards.get('A').feedback.value,
  assumption: cards.get('A').assumption.checked,
};
const beforeStaleClick = sent.length;
sb.respondWorkOrder('A', 'reject');
const afterStaleClick = sent.length;

sb.showHITL(request('H'), { history: true });
const historyDisabled = cards.get('H').button.disabled;
sb.showHITL(request('H'));
const redeliveryEnabled = !cards.get('H').button.disabled;

sb.showHITL(request('C'));
sb.ws = null;
const beforeOfflineClick = sent.length;
sb.respondWorkOrder('C', 'approve');

process.stdout.write(JSON.stringify({
  afterRedraw,
  stalePackets: afterStaleClick - beforeStaleClick,
  offlinePackets: sent.length - beforeOfflineClick,
  historyDisabled,
  redeliveryEnabled,
  offlineStillOpen: !cards.get('C').button.disabled && !cards.get('C').controls.dataset.answered,
  offlineNotice: notices.at(-1),
}));
"""


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = tmp_path_factory.mktemp("hitl-state") / "run.js"
    script.write_text(_HARNESS, encoding="utf-8")
    done = subprocess.run(
        [node, str(script), str(JS)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_answered_cards_keep_state_and_stay_locked_after_redraw(result):
    assert result["afterRedraw"] == {
        "aDisabled": True,
        "bDisabled": True,
        "feedback": "keep this note",
        "assumption": False,
    }


def test_stale_and_offline_clicks_do_not_send_or_fake_success(result):
    assert result["stalePackets"] == 0
    assert result["offlinePackets"] == 0
    assert result["offlineStillOpen"] is True
    assert "не отправлено" in result["offlineNotice"]


def test_history_is_locked_until_the_server_redelivers_it(result):
    assert result["historyDisabled"] is True
    assert result["redeliveryEnabled"] is True
