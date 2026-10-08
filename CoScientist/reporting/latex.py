"""Render the final markdown report to LaTeX.

Modes (see :data:`CoScientist.config.report.LATEX_MODES`):

* ``skip``       — do nothing.
* ``standalone`` — one compilable ``report.tex`` with a minimal preamble.
* ``body``       — a body-only ``report.body.tex`` fragment for pasting into
                   an existing Overleaf project.
* ``tree``       — a multi-file project under ``latex/``: ``main.tex`` that
                   ``\\input``s per-section files plus a ``references.bib``.

Uses ``pandoc`` when available (best fidelity); otherwise a small built-in
markdown→LaTeX converter that covers headings, emphasis, lists, links, images,
fenced code and pipe tables. LaTeX output is experimental (default mode is
``skip``), so the fallback aims for "compiles and is readable", not perfection.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{float}
\usepackage[margin=1in]{geometry}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
"""

_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def render_latex(
    markdown: str,
    report_dir: Path | str,
    mode: str,
    references: Optional[List[str]] = None,
) -> List[Path]:
    """Render ``markdown`` to LaTeX under ``report_dir``; return files written."""
    report_dir = Path(report_dir)
    if mode == "skip":
        return []

    body = _markdown_to_latex(markdown)

    if mode == "body":
        out = report_dir / "report.body.tex"
        out.write_text(body + "\n", encoding="utf-8")
        return [out]

    if mode == "standalone":
        out = report_dir / "report.tex"
        out.write_text(_PREAMBLE + "\n\\begin{document}\n\n" + body + "\n\n\\end{document}\n",
                       encoding="utf-8")
        return [out]

    if mode == "tree":
        return _render_tree(markdown, report_dir, references or [])

    raise ValueError(f"unknown latex mode {mode!r}")


def _render_tree(markdown: str, report_dir: Path, references: List[str]) -> List[Path]:
    latex_dir = report_dir / "latex"
    sections_dir = latex_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    # Split on level-2 headings (## ) into section files; keep any preamble text
    # before the first heading as "overview".
    parts = re.split(r"(?m)^##\s+(.+)$", markdown)
    intro = parts[0].strip()
    sections = list(zip(parts[1::2], parts[2::2]))  # (title, body)

    inputs: List[str] = []
    if intro:
        (sections_dir / "00_overview.tex").write_text(
            _markdown_to_latex(intro) + "\n", encoding="utf-8")
        inputs.append("sections/00_overview")
        written.append(sections_dir / "00_overview.tex")
    for i, (title, sec_body) in enumerate(sections, 1):
        slug = re.sub(r"[^a-z0-9]+", "_", title.strip().lower()).strip("_") or f"section_{i}"
        fname = f"{i:02d}_{slug}.tex"
        latex = f"\\section{{{_escape(title.strip())}}}\n\n" + _markdown_to_latex(sec_body)
        (sections_dir / fname).write_text(latex + "\n", encoding="utf-8")
        inputs.append(f"sections/{fname[:-4]}")
        written.append(sections_dir / fname)

    bib = latex_dir / "references.bib"
    bib.write_text(_references_to_bibtex(references), encoding="utf-8")
    written.append(bib)
    if not references:
        # Documented limitation, not a silent gap. See finalize._extract_references.
        logger.warning(
            "latex(tree): no structured references available — references.bib is a "
            "placeholder. Reliable bibliography needs paper-research to retain raw "
            "citation metadata in session state."
        )

    body_inputs = "\n".join(f"\\input{{{p}}}" for p in inputs)
    main = (
        _PREAMBLE
        + "\n\\begin{document}\n\n"
        + body_inputs
        + "\n\n\\bibliographystyle{plain}\n\\bibliography{references}\n\n"
        + "\\end{document}\n"
    )
    main_tex = latex_dir / "main.tex"
    main_tex.write_text(main, encoding="utf-8")
    written.append(main_tex)
    return written


def _references_to_bibtex(references: List[str]) -> str:
    if not references:
        return (
            "% No structured references were available for this run.\n"
            "% Populate this file (or enable citation capture in paper-research).\n"
        )
    entries = []
    for i, ref in enumerate(references, 1):
        entries.append(
            f"@misc{{ref{i},\n  note = {{{_escape(ref)}}}\n}}"
        )
    return "\n\n".join(entries) + "\n"


# ── markdown -> latex ────────────────────────────────────────────────────────

def _markdown_to_latex(markdown: str) -> str:
    if shutil.which("pandoc"):
        try:
            proc = subprocess.run(
                ["pandoc", "--from=gfm", "--to=latex", "--wrap=preserve"],
                input=markdown.encode("utf-8"),
                capture_output=True,
                timeout=60,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.decode("utf-8")
            logger.warning("latex: pandoc failed (%s); using fallback converter",
                           proc.stderr.decode("utf-8", "replace")[:200])
        except Exception as exc:  # pragma: no cover
            logger.warning("latex: pandoc error (%s); using fallback converter", exc)
    return _fallback_convert(markdown)


def _escape(text: str) -> str:
    return "".join(_SPECIALS.get(ch, ch) for ch in text)


def _inline(text: str) -> str:
    """Convert inline markdown (after escaping) to LaTeX."""
    # Protect images/links before escaping their surrounding text.
    tokens: dict = {}

    def stash(latex: str) -> str:
        key = f"\x00{len(tokens)}\x00"
        tokens[key] = latex
        return key

    text = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda m: stash(
            "\\begin{figure}[H]\\centering\\includegraphics[width=0.8\\linewidth]{"
            + m.group(2) + "}"
            + (f"\\caption{{{_escape(m.group(1))}}}" if m.group(1) else "")
            + "\\end{figure}"
        ),
        text,
    )
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: stash(f"\\href{{{m.group(2)}}}{{{_escape(m.group(1))}}}"),
        text,
    )
    text = re.sub(r"`([^`]+)`", lambda m: stash(f"\\texttt{{{_escape(m.group(1))}}}"), text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda m: stash(f"\\textbf{{{_escape(m.group(1))}}}"), text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", lambda m: stash(f"\\textit{{{_escape(m.group(1))}}}"), text)

    text = _escape(text)
    for key, latex in tokens.items():
        text = text.replace(_escape(key), latex).replace(key, latex)
    return text


def _fallback_convert(markdown: str) -> str:
    lines = markdown.splitlines()
    out: List[str] = []
    i = 0
    in_code = False
    list_open = False

    def close_list():
        nonlocal list_open
        if list_open:
            out.append("\\end{itemize}")
            list_open = False

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            close_list()
            if not in_code:
                out.append("\\begin{verbatim}")
                in_code = True
            else:
                out.append("\\end{verbatim}")
                in_code = False
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        # Pipe table: a header row followed by a separator row.
        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            close_list()
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            out.append(_convert_table(table_lines))
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            close_list()
            level = len(heading.group(1))
            cmd = {1: "section", 2: "subsection", 3: "subsubsection"}.get(level, "paragraph")
            out.append(f"\\{cmd}{{{_inline(heading.group(2).strip())}}}")
            i += 1
            continue

        bullet = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if bullet:
            if not list_open:
                out.append("\\begin{itemize}")
                list_open = True
            out.append(f"  \\item {_inline(bullet.group(1))}")
            i += 1
            continue

        if not line.strip():
            close_list()
            out.append("")
            i += 1
            continue

        close_list()
        out.append(_inline(line))
        i += 1

    close_list()
    if in_code:
        out.append("\\end{verbatim}")
    return "\n".join(out)


def _convert_table(table_lines: List[str]) -> str:
    def cells(row: str) -> List[str]:
        return [c.strip() for c in row.strip().strip("|").split("|")]

    header = cells(table_lines[0])
    body_rows = [cells(r) for r in table_lines[2:] if r.strip()]
    ncol = len(header)
    col_spec = "l" * ncol
    lines = [
        "\\begin{center}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(_inline(c) for c in header) + " \\\\",
        "\\midrule",
    ]
    for row in body_rows:
        row = (row + [""] * ncol)[:ncol]
        lines.append(" & ".join(_inline(c) for c in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    return "\n".join(lines)


__all__ = ["render_latex"]
