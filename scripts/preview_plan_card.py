"""Render the experiment-plan review card the web UI shows, to one HTML file.

Two things this is for:

* seeing the card without a full run — the plan review is the last step of a
  planning round, and waiting for one to reach it is a slow way to check a
  wording or a column;
* showing the difference from the old view side by side. ``--before`` adds the
  Markdown-in-a-``<pre>`` panel the HITL card used before ``plan_view.py``, so
  the same plan is rendered both ways on one page.

The card is rendered by the *shipped* front-end — ``CoScientist/web/static/js``
run under node with a small DOM shim — never by a copy of it, so what this
writes is what the browser draws. Without node it writes the plan JSON and the
before panel, and says what is missing.

Usage::

    python scripts/preview_plan_card.py --before --open
    python scripts/preview_plan_card.py --plan run.json --lang en -o card.html

``--plan`` accepts an ExperimentPlan JSON, a session-state JSON (the plan is
read from ``experiment_runtime.plan``), or a plan-view JSON such as the one a
``decision`` node in the execution graph carries. With no ``--plan`` a built-in
sample plan is used.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import webbrowser
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATIC = ROOT / "CoScientist" / "web" / "static" / "js"
INDEX = ROOT / "CoScientist" / "web" / "templates" / "index.html"

# Enough DOM for the card's string builders, and nothing else: the modules it
# loads only ever touch getElementById, innerHTML and classList.
_SHIM = r"""
const fs = require('fs'), path = require('path'), vm = require('vm');
const [root, planFile, lang, outFile] = process.argv.slice(2);
const elements = new Map();
const classList = () => ({ add() {}, remove() {}, toggle() {}, contains: () => false });
const element = id => ({ id, innerHTML: '', textContent: '', style: {},
  classList: classList(), dataset: {},
  insertAdjacentHTML(_w, html) { this.innerHTML += html; },
  setAttribute() {}, removeAttribute() {}, addEventListener() {},
  querySelector: () => null, querySelectorAll: () => [], scrollTop: 0, scrollHeight: 0 });
const sandbox = {
  console,
  localStorage: { getItem: () => lang, setItem() {} },
  document: {
    // documentElement/body: the modules toggle page-level classes on load.
    documentElement: element('html'),
    body: element('body'),
    getElementById: id => (elements.has(id) ? elements : elements.set(id, element(id))).get(id),
    createElement: tag => element(tag),
    addEventListener() {}, querySelector: () => null, querySelectorAll: () => [],
  },
  // placeHitlCard escapes the request id before querying for an existing card.
  CSS: { escape: v => String(v) },
  window: {}, navigator: { language: lang }, LANG_STORAGE_KEY: 'coscientist.lang',
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
const load = rel => vm.runInContext(
  fs.readFileSync(path.join(root, rel), 'utf8'), sandbox, { filename: rel });
load('CoScientist/web/static/js/state.js');
load('CoScientist/web/static/js/i18n.js');
// chat.js drags in the whole app; the card uses two of its helpers.
vm.runInContext(`
  function scrollChat() {}
  function appendMsgToFeed(html) { document.getElementById('chat-feed').insertAdjacentHTML('x', html); }
  function addSystemMsg() {} function respondHITL() {} function respondHITLEdit() {}
  function updateRoadmapModalButtons() {}
  StatusIndicator = null;
`, sandbox);
load('CoScientist/web/static/js/hitl.js');
vm.runInContext('currentLang = ' + JSON.stringify(lang) + ';', sandbox);
sandbox.__plan = JSON.parse(fs.readFileSync(planFile, 'utf8'));
vm.runInContext(`
  showHITL({ request_id: 'preview-0', agent_name: 'ExperimentPlannerAgent',
             action_type: 'approve', message: 'Review and explicitly approve the experiment plan.',
             options: [], context: { experiment_plan: __plan } });
  __plan.tasks.forEach(t => planOpenTasks.add('preview-0:' + t.id));
  document.getElementById('plan-tasks-preview-0').innerHTML =
    __plan.tasks.map((t, i) => planTaskCard('preview-0', t, i)).join('');
`, sandbox);
// The feed holds the card with a placeholder where the task list goes; splice
// the unfolded list back in, since the shim's elements are not really nested.
const feed = sandbox.document.getElementById('chat-feed').innerHTML;
const tasks = sandbox.document.getElementById('plan-tasks-preview-0').innerHTML;
fs.writeFileSync(outFile, JSON.stringify({
  feed: feed.replace(/(<div id="plan-tasks-preview-0"[^>]*>)[\s\S]*?(<\/div>\s*<div class="mt-3")/,
                     (_m, open, tail) => open + tasks + '</div><div class="mt-3"'),
  panel: sandbox.document.getElementById('hitl-panel').innerHTML,
}), 'utf8');
"""

_PAGE = """<!doctype html>
<html lang="{lang}"><head><meta charset="utf-8">
<title>Experiment plan review — preview</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<script>{tailwind}</script>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;700&family=JetBrains+Mono&display=swap">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined">
<style>
  body {{ background:#101319; color:#e1e2eb; font-family:Inter,sans-serif; margin:0; padding:24px; }}
  .material-symbols-outlined {{ font-family:'Material Symbols Outlined'; font-weight:normal;
    font-style:normal; line-height:1; letter-spacing:normal; text-transform:none;
    display:inline-block; white-space:nowrap; word-wrap:normal; direction:ltr; }}
  h2 {{ font-family:'Space Grotesk',sans-serif; font-size:12px; letter-spacing:.18em;
    text-transform:uppercase; color:#94a3b8; margin:0 0 12px; }}
  .cols {{ display:flex; gap:28px; align-items:flex-start; }}
  .col {{ flex:1; min-width:0; }}
  .side {{ width:320px; flex:none; }}
</style></head><body>
<div class="cols">{columns}</div>
</body></html>
"""

_BEFORE_PANEL = """
<div class="my-6 relative">
  <div class="relative bg-surface-container-lowest p-6 rounded-xl border border-primary/30 shadow-2xl">
    <div class="flex items-center gap-3 mb-3">
      <div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
        <span class="material-symbols-outlined text-on-primary text-sm">ads_click</span>
      </div>
      <h3 class="font-headline font-bold text-on-surface uppercase tracking-tight">Human-In-The-Loop Required</h3>
    </div>
    <p class="text-sm text-on-surface-variant leading-relaxed pl-11">{message}</p>
    <div class="mt-3 pl-11">
      <p class="text-[10px] font-bold text-outline-variant uppercase tracking-wider mb-1">Proposed output</p>
      <pre class="font-mono text-[11px] leading-relaxed text-on-surface-variant whitespace-pre-wrap \
bg-surface-container-high p-3 rounded-lg border border-outline-variant/10 max-h-96 overflow-auto">{output}</pre>
    </div>
  </div>
</div>
"""


def _sample_plan() -> dict:
    """A plan with the shapes a reviewer has to be able to tell apart: a ready
    MCP route, a build that has no tool yet, a task that waits on both, and an
    optional one."""
    def task(tid, name, route, hypothesis, **over):
        row = {
            "id": tid, "name": name, "route": route,
            "description": "Run the bounded computation and write the metrics table.",
            "rationale": "Produces direct computational evidence for the hypothesis.",
            "design": {
                "hypothesis_ref": hypothesis,
                "experiment_question": f"Does this task produce evidence for {hypothesis}?",
                "dataset": {"name": "ChEMBL GSK-3beta actives",
                            "ref": "s3://coscientist/datasets/chembl_gsk3b.csv",
                            "notes": "4 812 rows, IC50 in nM, deduplicated by canonical SMILES."},
                "baselines": [{"name": "random sampling", "kind": "method"},
                              {"name": "published RF model", "kind": "prior_result",
                               "ref": "10.1021/acs.jcim.1c00203"}],
                "metrics": [{"name": "ROC-AUC", "direction": "maximize",
                             "threshold": 0.8, "test": "bootstrap CI"}],
                "analysis_artifacts": [{"name": "metrics_table.json", "role": "metrics_table",
                                        "prepare_via": "mcp", "path_or_tool": "metrics_table.json"}],
            },
            "mcp_servers": [{
                "name": "chem-props", "server_id": "srv-chem", "source": "registry",
                "url": "http://mcp.internal:8931/mcp", "health": "healthy",
                "tools": [{"name": "estimate_property", "required_for_task": True,
                           "description": "Compute a physicochemical property for a SMILES."}],
            }],
            "input_data": [], "launch_params": {"target": "GSK-3beta", "n_folds": "5"},
            "success_criteria": [
                {"criterion_id": f"{tid}-C1", "kind": "threshold", "metric": "ROC-AUC",
                 "operator": ">=", "target": 0.8,
                 "description": "ROC-AUC clears the bar on the held-out split.",
                 "verification": "Read the metric off metrics_table.json."},
            ],
            "expected_artifacts": [{"name": "metrics_table.json", "role": "data",
                                    "media_type": "application/json",
                                    "description": "Per-fold metrics for the model and both baselines."}],
            "est_duration_min": 25, "warnings": [], "depends_on": [], "optional": False,
        }
        row.update(over)
        return row

    build = task("EXP-2", "Build an MCP server from the docking repo", "alembic_build", "H2",
                 repo_url="https://github.com/ccsb-scripps/AutoDock-Vina",
                 post_build_route="react_tools", mcp_servers=[], est_duration_min=45,
                 code_assessment={
                     "requirement": "reuse",
                     "evidence": "The existing vina CLI covers docking unchanged.",
                     "entrypoints": ["vina --config <config>"],
                 },
                 description="Turn the docking repository into a served MCP tool the next task calls.",
                 warnings=["a full build takes tens of minutes and may fail on system libraries"])
    build["design"] = dict(build["design"],
                           experiment_question="Can the docking repository be served as a tool at all?",
                           dataset={"name": "unspecified"}, baselines=[], analysis_artifacts=[],
                           metrics=[{"name": "build_succeeded", "direction": "maximize"}])
    build["success_criteria"] = [{"criterion_id": "EXP-2-C1", "kind": "execution",
                                  "description": "The build serves an MCP endpoint.",
                                  "verification": "check_mcp_build reports a URL."}]
    build["expected_artifacts"] = [{"name": "mcp_endpoint.txt", "role": "data",
                                    "media_type": "text/plain",
                                    "description": "The URL the built server is serving on."}]

    dock = task("EXP-3", "Dock the top 200 candidates", "react_tools", "H2",
                depends_on=["EXP-1", "EXP-2"], est_duration_min=60,
                warnings=["depends on EXP-2 succeeding; no fallback docking tool is registered"],
                mcp_servers=[{"name": "autodock-built", "server_id": "srv-dock",
                              "source": "alembic", "url": "http://mcp.internal:8940/mcp",
                              "health": "unknown",
                              "tools": [{"name": "dock_ligand", "required_for_task": True,
                                         "description": "Dock one ligand into a prepared receptor."}]}],
                input_data=[
                    {"data_id": "TOP200", "kind": "task_artifact", "required": True,
                     "description": "The 200 highest-ranked molecules from the descriptor model.",
                     "source_task_id": "EXP-1", "source_artifact_id": "metrics_table.json"},
                    {"data_id": "RECEPTOR", "kind": "s3", "required": True,
                     "description": "Prepared GSK-3beta receptor structure.",
                     "bucket": "coscientist", "s3_key": "receptors/1q41_prepared.pdbqt"},
                ])
    dock["design"] = dict(dock["design"], also_tests=["H1"])

    return {
        "schema_version": "experiment-plan/1.0",
        "plan_id": "PLAN-7f3a91", "experiment_run_id": "EXRUN-7f3a91", "revision": 2,
        "source_request": "Find small-molecule inhibitors of GSK-3beta and rank them.",
        "goal": ("Rank candidate GSK-3beta inhibitors by predicted activity and confirm "
                 "the ranking beats two baselines."),
        "hypothesis": ("A descriptor model trained on ChEMBL actives ranks GSK-3beta "
                       "inhibitors better than random and better than the published RF model."),
        "hypotheses": [
            {"hypothesis_id": "H1",
             "statement": "A descriptor model ranks GSK-3beta actives above inactives with ROC-AUC >= 0.8."},
            {"hypothesis_id": "H2",
             "statement": "Docking scores add signal the descriptor model does not already carry."},
        ],
        "methods": ["descriptor modelling", "retrospective virtual screening", "molecular docking"],
        "context_digest": "chem-props MCP is registered and healthy; the docking repo is a build candidate.",
        "context_refs": ["TOOL-srv-chem", "REPO-autodock"],
        "tasks": [
            task("EXP-1", "Descriptor model on ChEMBL actives", "fedot_mas", "H1"),
            build, dock,
            task("EXP-4", "Sensitivity re-run on a held-out target", "coder", "H1",
                 optional=True, est_duration_min=15, mcp_servers=[],
                 description="Re-fit the descriptor model on a second kinase to check "
                             "the result is not target-specific."),
        ],
        "risks": ["The docking build may fail, leaving H2 untested this round.",
                  "ChEMBL IC50 values come from mixed assay conditions; the activity "
                  "threshold is a judgement call."],
        "assumptions": ["The prepared receptor is in the right protonation state for the assay pH.",
                        "A 5-fold split is enough to bound the ROC-AUC estimate."],
        "total_est_duration_min": 145,
        "created_at": "2026-07-31T18:00:00+00:00",
    }


_SAMPLE_CRITIQUE = {
    "verdict": "approve",
    "issues": [{"issue_id": "I-1", "severity": "minor", "category": "design",
                "task_id": "EXP-4",
                "message": "EXP-4 is optional and shares its metric with EXP-1.",
                "suggestion": "Keep it optional, or give it a metric of its own."}],
}


def _load(path: Path | None) -> tuple[dict, dict | None, object | None]:
    """(plan view, critique, ExperimentPlan or None) from whatever was passed."""
    from CoScientist.experiments.plan_view import plan_to_view
    from CoScientist.experiments.schemas import ExperimentPlan

    if path is None:
        plan = ExperimentPlan.model_validate(_sample_plan())
        return plan_to_view(plan, _SAMPLE_CRITIQUE), _SAMPLE_CRITIQUE, plan

    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("kind") == "experiment_plan":       # already a plan view
        return raw, raw.get("critique"), None
    critique = raw.get("experiment_plan_critique")
    if runtime := raw.get("experiment_runtime"):   # a session state dump
        raw, critique = runtime.get("plan") or {}, critique or runtime.get("critique")
    plan = ExperimentPlan.model_validate(raw)
    return plan_to_view(plan, critique), critique, plan


def _render_card(view: dict, lang: str) -> tuple[str, str] | None:
    """(chat card, sidebar) from the shipped front-end, or None without node."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shim, plan_file, out = (tmp_path / "shim.js", tmp_path / "plan.json",
                                tmp_path / "out.json")
        shim.write_text(_SHIM, encoding="utf-8")
        plan_file.write_text(json.dumps(view, ensure_ascii=False), encoding="utf-8")
        try:
            done = subprocess.run(
                ["node", str(shim), str(ROOT), str(plan_file), lang, str(out)],
                capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"node could not run ({exc}); writing the before panel only.")
            return None
        if done.returncode != 0:
            print(f"node failed:\n{done.stderr.strip()[:2000]}")
            return None
        payload = json.loads(out.read_text(encoding="utf-8"))
        return payload["feed"], payload["panel"]


def _tailwind_config() -> str:
    text = INDEX.read_text(encoding="utf-8")
    head = '<script id="tailwind-config">'
    return text.split(head, 1)[1].split("</script>", 1)[0] if head in text else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=None,
                        help="ExperimentPlan / session-state / plan-view JSON")
    parser.add_argument("--lang", default="ru", choices=("ru", "en"))
    parser.add_argument("--before", action="store_true",
                        help="also render the Markdown-in-a-<pre> panel this replaced")
    parser.add_argument("-o", "--out", type=Path, default=Path("plan_card_preview.html"))
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args()

    view, _critique, plan = _load(args.plan)
    columns = []

    if args.before:
        if plan is None:
            print("--before needs a plan, not a plan view; skipping that column.")
        else:
            from CoScientist.experiments.review import render_experiment_plan

            columns.append(
                '<div class="col"><h2>before — context.output in a &lt;pre&gt;</h2>'
                + _BEFORE_PANEL.format(
                    message="Review and explicitly approve the experiment plan.",
                    # The same language as the card beside it, or the
                    # comparison is between two different things.
                    output=escape(render_experiment_plan(plan, args.lang)))
                + "</div>")

    rendered = _render_card(view, args.lang)
    if rendered is not None:
        feed, panel = rendered
        columns.append(f'<div class="col"><h2>after — context.experiment_plan</h2>{feed}</div>')
        columns.append(f'<div class="side"><h2>sidebar</h2>{panel}</div>')
    elif not columns:
        target = args.out.with_suffix(".json")
        target.write_text(json.dumps(view, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote the plan view to {target}. Install node to render the card.")
        return 1

    args.out.write_text(
        _PAGE.format(lang=args.lang, tailwind=_tailwind_config(), columns="".join(columns)),
        encoding="utf-8")
    print(f"Wrote {args.out.resolve()}  ({args.out.stat().st_size // 1024} KB)")
    if args.open_browser:
        webbrowser.open(args.out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
