"""Report language: the user picks it per session in the web UI.

The Result Aggregator writes the run's final report, and its language is a
per-session choice — not a build-time constant. An agent instruction is a plain
string built once per session, so the language cannot come from settings or from
a ``<<...>>`` template sentinel. It travels as ADK session state instead:

  1. the web layer writes ``state['report_language']`` ("ru" or "en");
  2. ``inject_report_language`` (before_agent) renders the matching block into
     ``state['report_language_block']``;
  3. the prompt carries ``{report_language_block?}`` and ADK substitutes it at
     run time.

The callback injects the WHOLE language contract, not just a language name. The
report's section headings, the heading-substitution rule that constrains the
verbatim ``format_results`` output, and the entity glossary all differ by
language, so they live here — one block per language, kept parallel line for
line — and the prompt stays one readable English document.
"""
from google.adk.agents.callback_context import CallbackContext

import logging
logger = logging.getLogger(__name__)

# Set by the web layer (see web/app.py `apply_report_language`).
REPORT_LANGUAGE_STATE_KEY = "report_language"
# Rendered by this callback, read by the prompt's {report_language_block?}.
REPORT_LANGUAGE_BLOCK_STATE_KEY = "report_language_block"

REPORT_LANGUAGES = ("ru", "en")
# Every entrypoint that does not set the key — the CLI, A2A, alembic — lands
# here, which reproduces the behavior those paths had before the key existed.
DEFAULT_REPORT_LANGUAGE = "ru"


def normalize_report_language(raw) -> str:
    """Map any input to a supported language code.

    The single fallback site: a missing key, None, an empty string, and an
    unsupported code all collapse to DEFAULT_REPORT_LANGUAGE.
    """
    lang = str(raw or "").strip().lower()
    return lang if lang in REPORT_LANGUAGES else DEFAULT_REPORT_LANGUAGE


_RU_BLOCK = """### Report language

Write the entire report in Russian. These instructions are in English, the
report is not. Everything the reader sees is Russian: headings, prose, captions,
list items, and conclusions.

Heading substitutions inside the `format_results` blocks — these two, and no
others: `## Figures` becomes `## Иллюстрации`, and `## Data tables` becomes
`## Таблицы данных`.

Section headings, in this order. The role each one carries is in parentheses,
so match them to the five sections of step 3:
`## Цель` (objective), `## Подход` (approach), `## Результаты` (results),
`## Обсуждение` (discussion), `## Ограничения и дальнейшие шаги`
(limitations and next steps).

The graph, its node labels, and the digest above are written in English.
Translate their substance into Russian — never quote an English sentence into
the report. Keep these untranslated:
   - numbers, units, formulas, and dates;
   - node ids, file paths, and links;
   - citation strings, author names, and paper titles;
   - code, tool, agent, and parameter identifiers.
For a domain term whose Russian form is ambiguous, write the Russian term and
give the English one in parentheses at its first use.

Name the graph entities with the spec's Russian terms: ResearchQuestion —
исследовательский вопрос, Hypothesis — гипотеза, Evidence — свидетельство,
Conclusion — заключение, VerificationMethod — метод проверки, Constraint —
ограничение, Tool — инструмент. Report hypothesis statuses as
подтверждена / опровергнута / отложена.
"""

_EN_BLOCK = """### Report language

Write the entire report in English. These instructions are in English, and so is
the report. Everything the reader sees is English: headings, prose, captions,
list items, and conclusions.

Heading substitutions inside the `format_results` blocks — none. Keep
`## Figures` and `## Data tables` exactly as `format_results` returned them.

Section headings, in this order. The role each one carries is in parentheses,
so match them to the five sections of step 3:
`## Objective` (objective), `## Approach` (approach), `## Results` (results),
`## Discussion` (discussion), `## Limitations and next steps`
(limitations and next steps).

The graph, its node labels, and the digest above are written in English. Use
their wording directly — no translation step stands between the graph and the
report. Keep these as they are:
   - numbers, units, formulas, and dates;
   - node ids, file paths, and links;
   - citation strings, author names, and paper titles;
   - code, tool, agent, and parameter identifiers.
For a domain term with more than one accepted form, use the form the graph uses
and stay with it for the whole report.

Name the graph entities with the schema's canonical English terms:
ResearchQuestion, Hypothesis, Evidence, Conclusion, VerificationMethod,
Constraint, Tool. Report hypothesis statuses as
confirmed / refuted / postponed.
"""

_BLOCKS = {"ru": _RU_BLOCK, "en": _EN_BLOCK}


def inject_report_language(callback_context: CallbackContext):
    """before_agent: render state['report_language'] into the prompt's language block.

    The instruction carries ``{report_language_block?}`` rather than a bare
    language name, because the headings, the substitution rule, and the glossary
    all change with the language. Best-effort — a failure here falls back to the
    default language instead of breaking the run. Returns None so the agent's
    other before_agent callbacks still run.
    """
    try:
        lang = normalize_report_language(
            callback_context.state.get(REPORT_LANGUAGE_STATE_KEY)
        )
    except Exception:  # noqa: BLE001
        logger.info("inject_report_language: unreadable state, using the default")
        lang = DEFAULT_REPORT_LANGUAGE
    callback_context.state[REPORT_LANGUAGE_BLOCK_STATE_KEY] = _BLOCKS[lang]
    return None
