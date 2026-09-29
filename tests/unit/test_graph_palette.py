"""Colour has to answer one question per channel, or it answers none.

The research canvas encodes three things at once: the card's colour is WHAT the
node is, the status word's colour is WHERE the claim stands, and the link's
colour is HOW two cards relate. Each of the ways that breaks is invisible —
nothing throws, nothing looks wrong on a small graph, and the reader simply
stops being able to tell two kinds of card apart:

* a type missing from ``R_KIND`` does not lose its colour, it inherits the
  STATUS colour, which is how ``Hypothesis`` (deliberately left out so it could
  carry its verdict) came out amber next to an orange ``Evidence``;
* two types sharing a hue look like one type — tool, dataset and code were all
  ``#a9682a``, and conclusion, report and publication all ``#fb923c``;
* a type taking green or red makes an ordinary card read as a verdict.

These are cheap checks against the server's own list of node types, so a new
type cannot be added without being given a colour.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from CoScientist.graph.research.schema import NODE_TYPES

PAGE = (
    Path(__file__).resolve().parents[2]
    / "CoScientist" / "web" / "templates" / "graph.html"
)

#: The two colours the verdict owns, as the page's :root declares them.
GOOD, CRIT = "#22c55e", "#ef4444"


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _table(page: str, name: str) -> dict[str, str]:
    """The `kind: "#rrggbb"` pairs of one const table."""
    block = re.search(r"const " + name + r" = \{(.*?)\n    \};", page, re.S)
    assert block, f"the page must still declare {name}"
    return dict(re.findall(r"(\w+):\s*\"(#[0-9a-fA-F]{6})\"", block.group(1)))


def test_every_node_type_the_server_can_write_has_a_colour(page):
    kinds = _table(page, "R_KIND")
    want = {name.lower() for name in NODE_TYPES}
    assert want - set(kinds) == set(), sorted(want - set(kinds))
    # And nothing extra: a stale key is a colour nobody can see, and it hides
    # the fact that the type it was for has gone.
    assert set(kinds) - want == set(), sorted(set(kinds) - want)


def test_no_two_types_share_a_colour(page):
    kinds = _table(page, "R_KIND")
    seen: dict[str, list[str]] = {}
    for kind, colour in kinds.items():
        seen.setdefault(colour.lower(), []).append(kind)
    clashes = {c: k for c, k in seen.items() if len(k) > 1}
    assert not clashes, clashes


def test_the_two_cards_a_reader_must_tell_apart_are_not_alike(page):
    """The complaint that started this: a claim and a measurement."""
    kinds = _table(page, "R_KIND")
    assert "hypothesis" in kinds, "the central claim cannot borrow the verdict colour"
    assert kinds["hypothesis"] != kinds["evidence"]


def test_green_and_red_belong_to_the_verdict_alone(page):
    kinds = _table(page, "R_KIND")
    taken = {c.lower() for c in kinds.values()}
    assert GOOD not in taken, "a green card reads as a confirmed one"
    assert CRIT not in taken, "a red card reads as a refuted one"


def test_a_settled_claim_is_green_a_rejected_one_red_and_the_rest_neutral(page):
    good = re.search(r"const STATUS_GOOD = new Set\(\[(.*?)\]\)", page, re.S)
    bad = re.search(r"const STATUS_BAD = new Set\(\[(.*?)\]\)", page, re.S)
    assert good and bad, "the page must still declare the two verdict sets"
    positive = set(re.findall(r'"(\w+)"', good.group(1)))
    negative = set(re.findall(r'"(\w+)"', bad.group(1)))
    assert {"confirmed", "validated", "met", "approved"} <= positive
    assert {"refuted", "rejected", "failed"} <= negative
    assert not positive & negative
    # A third thing that is neither, and is not neutral either: what is
    # happening RIGHT NOW. Drawn in neutral ink it read exactly like a card
    # nobody had touched, which is how a plan step that had been running for
    # minutes came to look identical to one not yet started.
    live = re.search(r"const STATUS_LIVE = new Set\(\[(.*?)\]\)", page, re.S)
    assert live, "the page must say which statuses are work in flight"
    working = set(re.findall(r'"(\w+)"', live.group(1)))
    assert {"in_progress", "running", "under_verification"} <= working
    assert "obtained" not in working, "received evidence is not work in flight"
    assert not working & positive and not working & negative, (
        "work in flight is not a verdict either way")
    assert re.search(r"statusInk = st => STATUS_GOOD\.has\(st\) \? GOOD\s*"
                     r":\s*STATUS_BAD\.has\(st\) \? CRIT\s*"
                     r":\s*STATUS_LIVE\.has\(st\) \? WARN : INK2", page)
    # …and the rest stays neutral: a postponed branch is not bad news.
    for status in ("inconclusive", "postponed", "planned", "proposed"):
        assert f'"{status}"' not in good.group(1)
        assert f'"{status}"' not in bad.group(1)
        assert f'"{status}"' not in live.group(1)


def test_the_status_word_is_the_one_thing_on_the_card_wearing_that_colour(page):
    """vis styles four label classes; `boldital` is the only one this card does
    not already spend, so the status word is wrapped in <b><i> and that class
    is given the verdict colour."""
    assert 'boldital: { color: statusInk(n.status)' in page
    assert 'mod: "bold"' in page, "a status word is bold, not italic"
    assert '"<b>" + head + " · </b><b><i>" + visSafe(sw) + "</i></b>"' in page


def test_an_untyped_link_is_not_three_stops_off_the_background(page):
    """The canvas is #14161a, so a legible link is a light one. #4a5160 and
    #4b5563 were the two greys a reader could not follow across a study."""
    for dim in ('"#4a5160"', '"#4b5563"', '"#aeb6c2"', '"#6b7280", size: 11'):
        assert dim not in page, dim
    assert 'const EDGE_NEUTRAL = "#cbd5e1"' in page


def test_an_edge_label_is_escaped_before_it_is_made_bold(page):
    """Enabling the html parser on a label means the same rule as the cards:
    a '<' or a stray '&' in an agent-written reason must not reach it."""
    assert re.search(r"edgeLabel = v => \{ const e = visSafe\(", page)
    assert 'return e ? "<b>" + e + "</b>" : ""' in page
    # Every research edge label goes through it, including the one that
    # interpolates a model-written reason.
    assert "edge.label = edgeLabel(t('graph.edge.supersedes.'" in page
    assert "(research ? edgeLabel(rlabel) : rlabel)" in page


# ── what is happening right now ───────────────────────────────────────────────
# The second thing colour on this canvas has to answer, and the one it did not:
# which stage is going on. The plan track had no entry in the status table at
# all, so a step being worked on and a step nobody had started drew as the same
# slate bullet — and the operator's complaint was exactly that.

def test_the_canvas_knows_the_plan_track_s_own_statuses(page):
    block = re.search(r"const R_STATUS = \{(.*?)\n    \};", page, re.S)
    assert block, "the page must still declare R_STATUS"
    known = set(re.findall(r"(\w+):\s*\{", block.group(1)))
    for status in NODE_TYPES["PlanStep"].statuses:
        assert status in known, f"a plan step drawn as {status} has no colour"
    for status in NODE_TYPES["ExperimentTask"].statuses:
        assert status in known, f"an experiment task drawn as {status} has no colour"
    for status in NODE_TYPES["VerificationMethod"].statuses:
        assert status in known, f"a method drawn as {status} has no colour"


def test_the_two_node_types_a_reader_watches_are_research_nodes(page):
    """`RESEARCH_KINDS` is built from `R_LEVEL`, and the tooltip of anything
    outside it prints the raw code. The two kinds whose progress a reader
    actually follows were the two that printed «experimenttask / in_progress»
    while the card beside them said it in words."""
    block = re.search(r"const R_LEVEL = \{(.*?)\n?\s*\};", page, re.S)
    assert block, "the page must still declare R_LEVEL"
    assert "planstep" in block.group(1) and "experimenttask" in block.group(1)


def test_only_an_action_card_pulses(page):
    """A plan step and an experiment task are work somebody is doing this
    minute; a hypothesis under verification is a state. The first pair moves,
    the second does not — that distinction is the operator's, and it is the
    reason the pulse is keyed on the KIND and not on the status alone."""
    kinds = re.search(r"const PULSING_KINDS = new Set\(\[(.*?)\]\)", page, re.S)
    statuses = re.search(r"const PULSING_STATUS = new Set\(\[(.*?)\]\)", page, re.S)
    assert kinds and statuses, "the page must say what pulses and when"
    assert set(re.findall(r'"(\w+)"', kinds.group(1))) == {"planstep", "experimenttask"}
    assert set(re.findall(r'"(\w+)"', statuses.group(1))) == {"in_progress", "running"}
    assert "PULSING_KINDS.has(n.kind) && PULSING_STATUS.has(n.status)" in page


def test_the_pulse_is_painted_not_written_into_the_cards(page):
    """The poll rebuilds every node object wholesale every 1.5 s, so a colour
    an animation wrote into the DataSet is overwritten within the second — and
    each such write re-runs vis's html-label parser over agent-written text,
    the one path in this file that can take the canvas down. The halo is drawn
    on the canvas instead, and the loop stops when nothing is running."""
    assert "function drawPulse(ctx)" in page
    assert re.search(r'network\.on\("beforeDrawing".*?drawPulse\(ctx\)', page, re.S)
    loop = re.search(r"function pulseLoop\(ts\)\{(.*?)\n    \}", page, re.S)
    assert loop, "the page must still drive the pulse from its own loop"
    body = loop.group(1)
    # Nothing in the pulse touches the node store.
    assert "nodes.update" not in body
    # …and it has a way out. `pulsing` is recomputed only inside `poll`, so with
    # the live feed switched off — or with every poll throwing — nothing would
    # ever empty the set, and the loop would repaint the canvas twenty times a
    # second for as long as the tab stayed open.
    assert "pulseRunning = false" in body
    assert "!pulsing.size" in body and "checked" in body


def test_a_failed_poll_stops_stale_activity_and_late_views_are_ignored(page):
    """A dead feed cannot honestly keep saying that yesterday's task runs."""
    assert "if (pollGeneration !== studyGen) return" in page
    assert "if (pollInFlightFor === pollGeneration) return" in page
    failed = re.search(r"catch \(err\) \{(.*?)\n      \} finally", page, re.S)
    assert failed, "poll failures need their own reconciliation path"
    assert "setPulsing(new Set())" in failed.group(1)


def test_no_pictograms_on_a_card(page):
    """A method card carried five or six of them on a box 230 px wide and none
    said anything the words beside them did not. The status glyph is not one:
    ✓ ✗ ⧗ ○ is a typographic mark, and it is the one thing on the card read at
    a glance rather than word by word."""
    label = re.search(r"function researchLabel\(n\)\{(.*?)\n    \}\n", page, re.S)
    assert label, "the page must still build its card labels in researchLabel"
    body = re.sub(r"^\s*//.*$", "", label.group(1), flags=re.M)
    found = [ch for ch in body if ord(ch) >= 0x1F300 or 0x2600 <= ord(ch) <= 0x27BF]
    assert not found, f"emoji left on the card: {found}"
    assert "R_ICON" not in body, "the type icon is a pictogram too"
