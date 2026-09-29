"""Expanding a ``[[linkN]]`` reference back into something a reader can use.

Every case here comes off a live session. The registry hands the model short
references instead of four-hundred-character presigned URLs, and expands them
again on the way out. Expanding them as ``[name](url)`` is right for a
reference standing free in prose — and wrong everywhere a URL is already
expected, because markdown does not nest.
"""
from __future__ import annotations

import pytest

from CoScientist.agents.callbacks.link_registry import expand_refs

URL = (
    "http://10.32.11.45:9000/tox-bucket/tox_antitargets/fig3_tsne_f6ca9105.png"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=3600"
    "&X-Amz-Signature=8b65236f3cb0c6e35be4a0a5ba19a5015a82737ea25d3ec6de9e8a1bf"
)
REF = "[[link7f3a0000]]"
REGISTRY = {"link7f3a0000": {"url": URL, "label": "fig3_tsne_f6ca9105.png"}}


def expand(text: str) -> str:
    return expand_refs(text, REGISTRY, as_markdown=True)


def test_a_reference_in_prose_becomes_a_named_link():
    """What the markdown form is for: a sentence, not a wall of signature."""
    out = expand(f"See {REF} for the chemical space.")
    assert out == f"See [fig3_tsne_f6ca9105.png]({URL}) for the chemical space."


@pytest.mark.parametrize(
    "template",
    [
        "![t-SNE]({ref})",
        "[fig3_tsne_f6ca9105.png]({ref})",
        '[a]({ref} "title")',
        "auto <{ref}>",
        "code `{ref}` inline",
        "```json\n{{\"figure\": \"{ref}\"}}\n```",
    ],
    ids=["image", "link", "titled", "autolink", "inline-code", "fence"],
)
def test_a_reference_where_a_url_is_expected_expands_bare(template):
    """Markdown does not nest, and these four contexts already want a URL.

    Verbatim from a live session, with the markdown form applied inside an
    image destination:

        "figure_artifact": "S3: [fig3.png]([fig3.png](http://…X-Amz-…))"

    The destination is then the literal string ``[fig3.png](http://…)``, which
    is not a URL. The browser cannot fetch it and draws the alt text — which is
    exactly what the user saw: captions, and no pictures.
    """
    out = expand(template.format(ref=REF))
    assert URL in out
    # The invariant is simply that nothing nests: a destination opening onto
    # another link is the whole defect. The model's own label, where it wrote
    # one, is left exactly as it wrote it.
    assert "]([" not in out
    assert f"]({URL}" in out or f'"{URL}"' in out or f"`{URL}`" in out or f"<{URL}>" in out


def test_the_fence_ends_and_prose_resumes():
    """A closing fence must restore the named form, or the rest of a report
    that happens to quote one JSON block loses every link it had."""
    out = expand(f"```json\n{{\"a\": \"{REF}\"}}\n```\n\nand then {REF} again")
    body, tail = out.split("```\n\n")
    assert f'"a": "{URL}"' in body
    assert tail == f"and then [fig3_tsne_f6ca9105.png]({URL}) again"


def test_expansion_does_not_deepen_on_a_second_round_trip():
    """The nesting compounded one level per turn.

    ``expand_link_refs`` writes ``[name](url)`` into the agent's own event, and
    replaying that history redacts the URL back to a reference — now sitting in
    a destination. Without the context guard each turn wrapped it again.
    """
    once = expand(f"see {REF}")
    twice = expand(once.replace(URL, REF))
    assert twice == once
    assert twice.count("](") == 1


def test_an_unknown_reference_is_left_where_a_reader_can_see_it():
    """Deleting it would hand the next agent a sentence with no object."""
    assert expand("see [[link00000000]]") == "see [[link00000000]]"


def test_tool_arguments_still_get_the_bare_url():
    """A tool needs an address, not prose. ``as_markdown`` defaults to off."""
    assert expand_refs(f"{REF}", REGISTRY) == URL
