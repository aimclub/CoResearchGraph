"""What the chat renderer writes after the sanitizer must escape itself.

``renderMarkdown`` in ``chat.js`` parses the agent's markdown, sanitizes the
result with DOMPurify, and then keeps editing the HTML string: it shortens a
bare link's text to the file name and appends an ``<img>`` preview. That string
goes straight into ``innerHTML``. Nothing added after the sanitize call is ever
sanitized again, so those two steps have to escape their own output.

Two ways it did not, both reproduced by running the real function with
marked/DOMPurify stubbed out:

    https://example.org/%3Cimg%20src=x%20onerror=alert(1)%3E
      -> <a href="..."><img src=x onerror=alert(1)></a>

The href is a plain https URL, so the sanitizer passes the link; the markup
appears only afterwards, when ``decodeURIComponent`` turns the percent escapes
back into ``<`` and ``>`` and the shortened text is written raw.

    [x" onerror=alert(1) z="](https://e.org/a.png)
      -> <img src="https://e.org/a.png" alt="x" onerror=alert(1) z="" ...>

A double quote in a text node is not escaped by the sanitizer — it does not
have to be, in a text node — and the preview puts that text inside
``alt="…"``, where it closes the attribute.

And one that only broke the chat: a URL need not be valid percent-encoding, so
a bare ``https://shop.example/100%off`` threw ``URIError: URI malformed`` out of
``renderMarkdown``, which is called from inside the template literals that
build the feed.

These are structural checks on the source, the way ``test_web_templates`` pins
the front end: there is no JS runtime in this suite, and each of them would
have caught its defect on the day.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

CHAT_JS = (
    Path(__file__).resolve().parents[2]
    / "CoScientist" / "web" / "static" / "js" / "chat.js"
)


@pytest.fixture(scope="module")
def source() -> str:
    return CHAT_JS.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    """The source of one top-level ``function name(...) {...}``."""
    start = source.index(f"function {name}(")
    depth = 0
    for i in range(source.index("{", start), len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def test_the_renderer_has_an_escaper_for_text_and_one_for_attributes(source):
    text = _function_body(source, "asText")
    for char, entity in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;")):
        assert f"replace(/{char}/g, '{entity}')" in text, char
    # An attribute value taken from already-escaped HTML needs only its
    # delimiter; escaping & again would print "&amp;amp;" in the alt text.
    attr = _function_body(source, "asAttr")
    assert "replace(/\"/g, '&quot;')" in attr
    assert "&amp;" not in attr


def test_the_shortened_link_text_is_escaped(source):
    """The text is a decoded URL segment, so it can carry < and >."""
    render = _function_body(source, "renderMarkdown")
    assert "'<a href=\"' + url + '\">' + asText(short) + '</a>'" in render


def test_the_image_preview_escapes_its_alt_text(source):
    render = _function_body(source, "renderMarkdown")
    assert "alt=\"' + asAttr(label) + '\"" in render
    # And nothing else drops a bare label into an attribute.
    assert "+ label +" not in render


def test_decoding_a_url_segment_cannot_throw_out_of_the_render(source):
    """One stray '%' in a bare link blanked the message it appeared in."""
    render = _function_body(source, "renderMarkdown")
    guarded = re.search(
        r"try\s*\{[^}]*decodeURIComponent[^}]*\}\s*catch", render, re.S
    )
    assert guarded, "decodeURIComponent must be inside a try/catch"


def test_a_bare_url_still_becomes_a_short_link(source):
    """The point of the fix being pinned here: a presigned URL runs to
    hundreds of characters of X-Amz parameters, and showing it as the link
    text floods the chat. Only a link whose text IS its URL is shortened —
    text the agent wrote is left alone."""
    render = _function_body(source, "renderMarkdown")
    assert "if (text.replace(/&amp;/g, '&') !== rawUrl) return m;" in render
    assert "rawUrl.split(/[?#]/)[0].split('/').filter(Boolean).pop()" in render


def test_a_bare_artifact_path_is_linked_at_all(source):
    """GFM auto-links full URLs only, so a reminted root-relative path
    rendered as unclickable text."""
    render = _function_body(source, "renderMarkdown")
    assert r"/(^|\s)(\/api\/artifact\/[^\s)]*)/g" in render
