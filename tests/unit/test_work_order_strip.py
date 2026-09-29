"""Work Order / Work Report cards fold their lists into one narrow strip.

An operator reading a run in the chat asked for this directly: the findings,
the assumptions («Условия и ограничения») and the steps with their ticks read as
pages once a run has done a few rounds, and nobody reads them again after the
card is answered. So they now sit behind one `<details>` line that says what is
inside — counts, step progress, the verdict — with the goal or summary cut to
the width that is left.

What must NOT change with it is the mechanism the cards are answered with:
checkboxes read back by respondWorkOrder, restored on replay, step marks ticked
in place by progress notices. A closed `<details>` keeps its children in the
DOM, which is why it was chosen over the `.fold-body` clip.

The render tests run the real hitl.js + i18n.js in Node; they skip where there
is no Node. The source checks below them always run.

Run from the repo root:  pytest tests/unit/test_work_order_strip.py -q
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "CoScientist" / "web"
JS = WEB / "static" / "js"
CHAT_JS = JS / "chat.js"

_HARNESS = r"""
const fs = require('fs'), path = require('path'), vm = require('vm');
const dir = process.argv[2];
const feed = [];
const sb = {
  console, localStorage: { getItem: () => 'ru', setItem() {} }, window: {},
  document: {
    querySelector: () => null, querySelectorAll: () => [], getElementById: () => null,
    addEventListener() {}, createElement: () => ({ style: {}, setAttribute() {}, appendChild() {} }),
    documentElement: { classList: { toggle() {}, contains: () => false }, lang: 'ru', setAttribute() {} },
    body: { classList: { toggle() {}, contains: () => false, add() {}, remove() {} } },
  },
  CSS: { escape: s => s }, setInterval: () => 0, clearInterval() {},
  appendMsgToFeed: h => feed.push(h), scrollChat() {}, addTelemetry() {}, addSystemMsg() {},
  foldable: h => `<div class="fold">${h}</div>`, renderMarkdown: s => String(s),
  documentBlock: d => (d.document && d.document.artifact_id) ? '<button>doc</button>' : '',
  escHtml: s => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;'),
};
// hitl.js now renders operator/model text through the shared markdown helpers.
// The harness only needs their escaping contract, not the full marked runtime.
sb.mdInline = s => sb.escHtml(s);
sb.mdBlock = s => sb.escHtml(s);
vm.createContext(sb);
vm.runInContext(fs.readFileSync(path.join(dir, 'i18n.js'), 'utf8'), sb);
vm.runInContext(fs.readFileSync(path.join(dir, 'hitl.js'), 'utf8')
  .replace(/^\s*window\.[a-zA-Z]+\s*=.*$/gm, ''), sb);
sb.placeHitlCard = (rid, h) => feed.push(h);

const order = {
  revision: 2, tier: 'read', goal: 'Собрать <литературу> о метаболитах',
  assumptions: [1, 2, 3, 4].map(i => ({ id: 'A' + i, text: 'Условие ' + i })),
  steps: ['done', 'done', 'done', 'done', 'pending'].map((s, i) => ({ id: 'S' + (i + 1), title: 't', status: s })),
};
const report = { round: 1, done_verdict: 'partial', summary: 'Итог отчёта',
  findings: [1, 2, 3, 4, 5].map(i => ({ id: 'F' + i, text: 'f', confidence: 'high', evidence: 'e' })) };
const doc = { artifact_id: 'doc1' };
const out = {};
const take = (key, fn) => { const n = feed.length; fn(); out[key] = feed.slice(n).join(''); };

take('live_report', () => sb.renderWorkReportCard(true, { request_id: 'r1', agent_name: 'A', trigger: 'work_report',
  timeout_seconds: 600, document: doc, context: { work_order: order, work_report: report,
  warnings: [{ code: 'open_steps', steps: ['S5'] }] } }));
take('history_report', () => sb.renderWorkReportCard(false, { request_id: 'r2', agent_name: 'A', trigger: 'work_report',
  timeout_seconds: 600, document: doc, context: { work_order: order, work_report: report, warnings: [] } }));
take('live_order', () => sb.renderWorkOrderCard(true, { request_id: 'r3', agent_name: 'A', trigger: 'work_order',
  timeout_seconds: 0, context: { work_order: order } }));
take('amended_notice', () => sb.renderWorkOrderNotice({ kind: 'amended', agent_name: 'A', work_order: order,
  reason: 'r', diff: { added_tools: ['tavily_extract'], added_steps: [{ id: 'S5', title: 't' }] } }));
take('empty_compact_report', () => sb.renderWorkReportCard(true, { request_id: 'r5', agent_name: 'A',
  trigger: 'work_report', timeout_seconds: 600, document: doc,
  context: { work_order: order, work_report: { ...report, findings: [] }, warnings: [] } }));
take('empty_verdict_replay', () => sb.renderWorkOrderNotice({ kind: 'report', agent_name: 'A',
  work_order: order, work_report: { ...report, done_verdict: '' } }));

const el = c => ({ dataset: {}, checked: c });
const a = { textContent: '4' }, f = { textContent: '5' };
sb.woRecount({ querySelectorAll(sel) {
  if (sel === 'input[data-wo-assumption]') return [el(true), el(false), el(true), el(true)];
  if (sel === 'input[data-wr-finding]') return [el(false), el(true), el(false), el(false), el(true)];
  if (sel === '[data-wo-assumption-count]') return [a];
  if (sel === '[data-wr-finding-count]') return [f];
  return [];
} });
out.recount = { assumptions: a.textContent, findings: f.textContent };
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = tmp_path_factory.mktemp("strip") / "render.js"
    script.write_text(_HARNESS, encoding="utf-8")
    done = subprocess.run([node, str(script), str(JS)], capture_output=True,
                          text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _strip(html: str) -> str:
    head = html.split("<details data-wo-strip", 1)
    return head[1].split(">", 1)[0] if len(head) == 2 else ""


def _line(html: str) -> str:
    summary = html.split("<summary", 1)[1].split("</summary>", 1)[0]
    import re
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", summary.split(">", 1)[1])).strip()


# ── What the operator sees ───────────────────────────────────────────────────


def test_history_is_one_line(cards):
    """The clutter the operator complained about was answered cards: those fold."""
    assert "<details data-wo-strip" in cards["history_report"]
    assert "open" not in _strip(cards["history_report"])
    assert "open" not in _strip(cards["amended_notice"])


def test_a_live_card_opens_because_silence_approves_nothing(cards):
    """hitl/mode.py: in `basic` silence REFUSES after ten minutes, in `debug` it
    waits, and in `auto` no card is drawn. So every live card is waiting for a
    person, countdown or not, and must not hide what they are asked to judge."""
    assert "open" in _strip(cards["live_report"])
    assert "open" in _strip(cards["live_order"])


def test_report_document_button_stays_visible_outside_the_strip(cards):
    """Folding report details must not fold away the way into the Markdown.

    Both a pending report and the same report in replay/history keep their
    document button before the collapsible strip, so it is visible whether the
    strip itself is open or closed.
    """
    for key in ("live_report", "history_report", "empty_compact_report"):
        html = cards[key]
        assert html.count("<button>doc</button>") == 1, key
        if "<details data-wo-strip" in html:
            assert html.index("<button>doc</button>") < html.index("<details data-wo-strip"), key


def test_report_document_button_opens_the_shared_right_panel():
    """The real chat button must call the document viewer, not a download."""
    source = CHAT_JS.read_text(encoding="utf-8")
    block = source.split("function documentBlock(event)", 1)[1].split(
        "function thinkingBlock", 1)[0]
    assert "data-doc-id" in block
    assert "onclick=\"openDocument(" in block
    assert "t('doc.open')" in block


def test_the_line_says_what_is_inside(cards):
    line = _line(cards["history_report"])
    assert "Находки 5" in line and "Шаги 4/5" in line
    assert "Итог отчёта" in line                    # the summary, cut to fit
    amended = _line(cards["amended_notice"])
    assert "Изменения +2" in amended and "Условия 4" in amended


def test_warnings_stay_outside_the_strip(cards):
    """A warning must not wait for someone to open anything."""
    before = cards["live_report"].split("<details data-wo-strip", 1)[0]
    assert "⚠" in before


def test_no_strip_opens_onto_an_empty_box(cards):
    assert "<details data-wo-strip" not in cards["empty_compact_report"]


def test_an_empty_verdict_does_not_print_its_i18n_key(cards):
    assert "workReport.verdict." not in cards["empty_verdict_replay"]


def test_the_lead_text_is_escaped(cards):
    assert "&lt;литературу&gt;" in cards["amended_notice"]
    assert "<литературу>" not in cards["amended_notice"]


# ── What the cards are answered with ─────────────────────────────────────────


def test_the_answer_mechanism_is_inside_the_strip_and_intact(cards):
    """Checkboxes and step marks moved behind the strip, not out of the DOM."""
    order = cards["live_order"]
    assert order.count('data-wo-assumption="') == 4
    assert order.count("data-wo-step=") == 5
    assert "data-wo-progress" in order
    assert cards["live_report"].count('data-wr-finding="') == 5


def test_deviations_stay_visible(cards):
    html = cards["amended_notice"]
    assert html.index("data-wo-deviations") > html.index("</details>")


def test_the_line_counts_what_the_operator_marked(cards):
    """The line is built from the server payload, which knows nothing of ticks
    made in this tab — and once the strip folds, that line is all anyone sees."""
    assert cards["recount"] == {"assumptions": "3/4", "findings": "5 ✗2"}


# ── Source invariants (no Node needed) ───────────────────────────────────────


def _src(name: str) -> str:
    return (JS / name).read_text(encoding="utf-8")


def _body(src: str, fn: str) -> str:
    return src.split(f"function {fn}(", 1)[1].split("\nfunction ", 1)[0]


def test_answering_folds_the_strip_and_locks_its_marks():
    body = _body(_src("hitl.js"), "disableHitlControls")
    assert "details[data-wo-strip][data-auto-open]" in body
    # A timeout or a cancel never passes through respondWorkOrder.
    assert "input[data-wr-finding]" in body and "el.disabled = true" in body


def test_the_replay_puts_disputed_findings_back():
    body = _body(_src("hitl.js"), "applyHitlOutcome")
    assert "disputed_finding_ids" in body
    assert "woRecount(card)" in body


def test_checkbox_changes_recount_the_strip_immediately():
    src = _src("hitl.js")
    assert "document.addEventListener('change'" in src
    assert "input[data-wo-assumption], input[data-wr-finding]" in src
    assert "woRecount(card)" in src


def test_manual_disclosure_state_survives_redraw_and_relocalization():
    src = _src("hitl.js")
    assert "workOrderStripState" in src
    assert "details.removeAttribute('data-auto-open')" in src
    assert "workOrderStripState.set(card.dataset.hitlCard, details.open)" in src
    assert "redrawWorkOrderCards()" in _body(src, "relocalizeHitlCards")


def test_every_strip_label_has_both_languages():
    i18n = _src("i18n.js")
    for key in ("steps", "assumptions", "findings", "artifacts", "changes", "details"):
        line = next((l for l in i18n.splitlines() if f"'woStrip.{key}'" in l), "")
        assert "en:" in line and "ru:" in line, key
