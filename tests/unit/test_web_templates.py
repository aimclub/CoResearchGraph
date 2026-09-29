"""The page templates have to be well-formed, and must not carry a second
copy of the front-end.

``index.html`` spent eleven commits on main with a botched conflict resolution
in it: a17e018 (#333) pasted 2 690 lines of the modular front-end into the body
after ``</main>``, lost the opening ``<script>`` that would have made it run,
and left a bare ``=======`` behind. The page therefore had 19 opening script
tags and 20 closing ones, and a quarter of a megabyte of JavaScript rendered as
body text — invisible only because ``body`` is ``overflow-hidden``.

Nothing failed loudly. The modules loaded at the bottom of the page still
defined everything, so the app worked; the copy was simply dead. d170e31 (#343)
then edited it, applying a fix to code that never ran.

These checks are cheap and would each have caught it on the day.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "CoScientist" / "web"
TEMPLATES = sorted((WEB / "templates").glob("*.html"))
MODULES = sorted((WEB / "static" / "js").rglob("*.js"))

CONFLICT_MARKER = re.compile(r"^(<<<<<<<|=======|>>>>>>>)\s*$", re.M)
# A function declaration at any indent. Indent is what a copy of the front-end
# pasted into a page keeps, so anchoring at column zero would miss exactly the
# thing this is looking for.
FUNCTION = re.compile(r"^\s*function\s+([A-Za-z_$][\w$]*)\s*\(", re.M)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_script_tags_are_balanced(template: Path):
    """An orphan </script> is a lost opening tag, and everything above it is
    text on the page instead of code."""
    text = _read(template)

    assert text.count("<script") == text.count("</script>"), (
        f"{template.name}: {text.count('<script')} opening vs "
        f"{text.count('</script>')} closing script tags"
    )


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_no_conflict_markers_survive(template: Path):
    text = _read(template)

    assert not CONFLICT_MARKER.findall(text), (
        f"{template.name} still carries a merge-conflict marker"
    )


def test_the_page_does_not_carry_a_second_copy_of_the_front_end():
    """index.html loads the front-end from static/js; a copy inside the page is
    either dead weight or, if it ever ran, a SyntaxError.

    Both scripts share one top-level scope, so a ``let`` declared in each is
    "Identifier has already been declared" and the modules never load — the
    copy that was there redeclared two hundred of them. A duplicated *function
    name* is the signal that survives whatever the indentation is: 101 of the
    102 functions in that block were already defined in static/js.
    """
    page = _read(WEB / "templates" / "index.html")
    module_functions = set()
    for module in MODULES:
        module_functions |= set(FUNCTION.findall(_read(module)))

    clashes = sorted(set(FUNCTION.findall(page)) & module_functions)

    assert not clashes, (
        f"index.html redefines {len(clashes)} function(s) that static/js "
        f"already defines: {', '.join(clashes[:10])}"
        + (" …" if len(clashes) > 10 else "")
    )


def test_every_module_the_page_loads_exists():
    """A renamed module leaves a 404 and a half-dead page, not an error."""
    page = _read(WEB / "templates" / "index.html")
    missing = [
        src for src in re.findall(r'<script src="(/static/[^"]+)"', page)
        if not (WEB / "static" / src[len("/static/"):]).exists()
    ]

    assert not missing, f"index.html loads files that do not exist: {missing}"


def test_the_plan_card_survives_a_language_switch():
    """Every other HITL card relocalizes through data-i18n spans.

    The experiment plan card cannot: it interpolates counts into "{n} tasks",
    builds the design-matrix headers and the per-task field names in JS, and
    none of that leaves a span for applyTranslations to swap. So it is redrawn,
    and applyTranslations has to reach the redraw.
    """
    hitl = _read(WEB / "static" / "js" / "hitl.js")
    i18n = _read(WEB / "static" / "js" / "i18n.js")

    assert "relocalizeHitlCards" in i18n, "applyTranslations must call into the cards"
    assert "redrawPlanCards()" in hitl, "relocalizeHitlCards must reach the plan card"
    body = hitl.split("function redrawPlanCards()", 1)[1].split("\nfunction ", 1)[0]
    assert "renderExperimentPlanReview" in body
    # placeHitlCard appends when it finds no card, so a stale planByRequest
    # entry would resurrect one the session has already cleared.
    assert "data-hitl-card" in body
    # Lifecycle and draft state are restored together after outerHTML replaces
    # the card; this is what keeps an answered review answered.
    assert "captureHitlCardState" in body
    assert "restoreHitlCardState" in body


def test_the_front_end_knows_the_experiment_module_agents():
    """Four agents with no label, no icon and no place in the tree read as
    blanks in the activity rail and in the agent graph.

    The live hierarchy comes from /api/agents, but the labels and icons are
    static tables, and nothing fails when a name is missing from them — the
    user just sees a bare class name, or nothing.
    """
    names = (
        "ExperimentModuleAgent",
        "ExperimentPlannerAgent",
        "ExperimentExecutorAgent",
        "ExperimentResultReviewAgent",
    )
    indicator = _read(WEB / "static" / "status_indicator.js")
    experiments = _read(WEB / "static" / "js" / "modals" / "experiments.js")
    roadmap = _read(WEB / "static" / "js" / "modals" / "roadmap.js")

    for name in names:
        assert name in indicator, f"{name}: no RU/EN label for the activity rail"
        assert name in experiments, f"{name}: unknown to the agent tree"
    assert "ExperimentModuleAgent" in roadmap, "the module has no icon"
