"""What a workspace file is, for the purpose of showing it to a person.

A sandbox workspace is a working tree: source, logs, CSVs, plots, checkpoints.
Most of it can be read on the spot and only a little of it cannot, so the panel
needs one question answered per entry — can this be shown, and as what? The
answer lives here rather than in the browser because the same table decides the
content type the bytes are served with, and the two must not drift: a file
shown as text must also be *sent* as text, or an .html left in the workspace
would run on our own origin.
"""
from __future__ import annotations

from pathlib import PurePosixPath

#: Read as text and shown as source. Extensions, without the dot.
TEXT_SUFFIXES = frozenset("""
txt text md markdown rst adoc log out err diff patch
py pyi pyx ipynb r jl m lua pl rb php go rs java kt scala swift
c h cc cpp cxx hpp hh cu sh bash zsh fish ps1 bat
js mjs cjs jsx ts tsx vue svelte css scss sass less
json jsonl ndjson yaml yml toml ini cfg conf properties env sample
csv tsv psv sql graphql proto
html htm xml xhtml rss atom tex bib cls sty
gitignore dockerignore editorconfig lock mod sum
""".split())

#: Files a working tree carries without an extension.
TEXT_NAMES = frozenset({
    "dockerfile", "makefile", "readme", "license", "licence", "notice",
    "authors", "changelog", "copying", "codeowners", "procfile", "justfile",
    "cmakelists.txt", "requirements.txt", "pipfile", "gemfile", "rakefile",
})

#: Shown in an ``<img>``. SVG is included deliberately: a script inside an SVG
#: does not run when the SVG is an image, only when it is a document.
IMAGE_TYPES = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
    "ico": "image/x-icon", "svg": "image/svg+xml", "avif": "image/avif",
    "tif": "image/tiff", "tiff": "image/tiff",
}

#: Read whole into the page, so bounded per kind: a log can be gigabytes and a
#: checkpoint routinely is. What is cut off is said so, not silently dropped.
CAPS = {"text": 2 * 1024 * 1024, "image": 25 * 1024 * 1024,
        "pdf": 25 * 1024 * 1024, "binary": 0}


def _suffix(name: str) -> str:
    return PurePosixPath(str(name or "")).suffix.lstrip(".").lower()


def kind_of(name: str) -> str:
    """One of ``text``, ``image``, ``pdf``, ``binary``."""
    base = PurePosixPath(str(name or "")).name.lower()
    suffix = _suffix(base)
    if suffix in IMAGE_TYPES:
        return "image"
    if suffix == "pdf":
        return "pdf"
    if suffix in TEXT_SUFFIXES or base in TEXT_NAMES:
        return "text"
    # A dotfile is its own suffix — .gitignore, .env — and is nearly always text.
    if not suffix and base.startswith(".") and base[1:] in TEXT_SUFFIXES:
        return "text"
    return "binary"


def media_type_of(name: str) -> str:
    """The type the bytes are served with — never one the browser will execute.

    Anything textual goes out as ``text/plain``: an .html or .js from a
    workspace is something to read, and serving it as itself would hand a
    sandbox agent's output the page's own origin.
    """
    kind = kind_of(name)
    if kind == "image":
        return IMAGE_TYPES[_suffix(name)]
    if kind == "pdf":
        return "application/pdf"
    if kind == "text":
        return "text/plain; charset=utf-8"
    return "application/octet-stream"


def cap_for(name: str) -> int:
    return CAPS[kind_of(name)]


__all__ = ["kind_of", "media_type_of", "cap_for", "CAPS",
           "TEXT_SUFFIXES", "TEXT_NAMES", "IMAGE_TYPES"]
