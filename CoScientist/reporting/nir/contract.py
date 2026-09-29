"""The NIR (GOST 7.32-2017) input contract, mirrored from the normcontrol MCP.

The authority is the server's own YAML contract, vendored at
``itmo-normcontrol/mcp_normcontrol/vendor/gost-docgen/gostdoc/contracts/
nir_report_gost_7_32_2017.yaml`` (a byte-identical copy sits in that repo's
root). Everything here is a transcription of it, and nothing else in this
package may hard-code a field name, an enum value or a limit.

Why mirror it at all instead of reading the spec over MCP: the builder has to
produce a valid document *before* the first network call, and a run where the
server is briefly unreachable should fail in our code with a readable message
rather than emit a malformed request. ``nir_report_validate`` stays the final
authority — this module only stops us from wasting a round trip on a mistake
we can catch locally.

The contract declares ``closed_world: true``. Any key the contract does not
describe is an ``unknown_value`` error, so the builder must emit exactly the
keys named here and no others.
"""
from __future__ import annotations

import re
from typing import Dict, FrozenSet, Tuple

# ── Envelope ────────────────────────────────────────────────────────────────

CONTRACT_ID = "nir-report-gost-7.32-2017"
CONTRACT_VERSION = "1.1"

#: ``values`` is ``{"contract": {...}, "document": {...}}``. A missing or
#: mismatched envelope is rejected before any field is looked at
#: (``missing_contract_reference`` / ``contract_id_mismatch``).
def envelope(document: Dict) -> Dict:
    """Wrap a built ``document`` in the contract reference the server expects."""
    return {
        "contract": {"id": CONTRACT_ID, "version": CONTRACT_VERSION},
        "document": document,
    }


# ── Modes ───────────────────────────────────────────────────────────────────

#: ``draft`` tolerates ``<...>`` placeholders and temporary pagination.
#: ``production`` additionally requires a real ``page_count`` and a complete
#: ``page_map`` — a page number per section, which only exists once the DOCX
#: has been laid out. python-docx does not paginate, so nothing on either side
#: can supply it up front; production is therefore unreachable in one pass.
MODES: Tuple[str, ...] = ("draft", "production")
DEFAULT_MODE = "draft"

#: What the renderer inserts for an unknown value in draft mode. Its presence
#: in a finished document is a warning, not an error.
PLACEHOLDER = "<...>"


# ── Document fields ─────────────────────────────────────────────────────────

#: Fields the server refuses to render without. ``title.stage`` is *conditional*
#: and is not listed: it becomes required only for an interim report (see
#: :data:`REPORT_TYPE_INTERIM`).
REQUIRED_FIELDS: Tuple[str, ...] = (
    "title.udc",
    "title.registration_nioktr",
    "title.registration_ikrbs",
    "title.approval",
    "title.research_title",
    "title.report_title",
    "title.report_type",
    "title.supervisor",
    "performers",
    "abstract.keywords",
    "abstract.text",
    "introduction",
    "sections",
    "conclusion",
    "references",
)

#: Filled from the organisation profile by the server when we omit them. We do
#: omit them: hard-coding the ITMO name here would duplicate a value the
#: contract already owns and would silently go stale if the profile changes.
DEFAULTED_FIELDS: Tuple[str, ...] = (
    "organization.parent",
    "organization.full",
    "organization.short",
    "title.document_type",
    "title.city",
    "title.year",
    "abstract.stats",
    "contents",
)

REPORT_TYPE_INTERIM = "промежуточный"
REPORT_TYPE_FINAL = "заключительный"
REPORT_TYPES: Tuple[str, ...] = (REPORT_TYPE_INTERIM, REPORT_TYPE_FINAL)

#: ``abstract.stats`` and ``contents`` accept this single value; the renderer
#: computes the statistics line and materialises the table of contents itself.
AUTO = "auto"

KEYWORDS_MIN = 5
KEYWORDS_MAX = 15
#: A recommendation in the contract, so exceeding it is a warning rather than a
#: rejection. The builder reports it and lets the author decide.
ABSTRACT_MAX_CHARS = 850

APPENDIX_STATUSES: Tuple[str, ...] = ("обязательное", "рекомендуемое", "справочное")


# ── Blocks ──────────────────────────────────────────────────────────────────

#: ``type`` discriminates the union. Closed world applies inside a block too:
#: allowed keys are exactly ``{"type"} | required | optional``.
BLOCK_FIELDS: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = {
    "paragraph": (frozenset({"text"}), frozenset({"id"})),
    "note": (frozenset({"text"}), frozenset({"id"})),
    "table": (frozenset({"id", "title", "columns", "rows"}), frozenset({"widths_mm"})),
    "figure": (frozenset({"id", "title", "path", "alt_text"}), frozenset({"width_mm"})),
    "formula": (frozenset({"id", "expression"}), frozenset({"explanation"})),
}

BLOCK_TYPES: Tuple[str, ...] = tuple(BLOCK_FIELDS)


def block_allowed_keys(block_type: str) -> FrozenSet[str]:
    """Every key a block of this type may carry, ``type`` included."""
    required, optional = BLOCK_FIELDS[block_type]
    return frozenset({"type"}) | required | optional


# ── Geometry ────────────────────────────────────────────────────────────────

#: The working width of the text area on an A4 page with the contract's
#: 30/15 mm side margins. Both a figure and a table are measured against it.
MAX_WIDTH_MM = 165
FIGURE_DEFAULT_WIDTH_MM = 150
FIGURE_MIN_WIDTH_MM = 1


# ── Identifiers ─────────────────────────────────────────────────────────────

#: Sections, blocks and appendices. Must start with a latin letter.
SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
#: References are looser: a bibliography key may start with a digit.
REFERENCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


# ── Assets ──────────────────────────────────────────────────────────────────

#: ``figure.path`` must be a key of ``assets``, and the server re-checks the
#: name against exactly this pattern. No directories, ASCII only, 128 chars.
ASSET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

#: The server decodes each asset and opens it with Pillow, demanding that the
#: real format and the file extension agree.
ASSET_FORMATS: Dict[str, FrozenSet[str]] = {
    "PNG": frozenset({".png"}),
    "JPEG": frozenset({".jpg", ".jpeg"}),
}
ASSET_EXTENSIONS: FrozenSet[str] = frozenset(
    ext for exts in ASSET_FORMATS.values() for ext in exts
)

#: ``MAX_FILE_SIZE`` on the server, measured over ``json.dumps({values, assets})``.
MAX_REQUEST_BYTES = 50 * 1024 * 1024
#: What we allow assets to take of that, leaving the document text room to grow
#: past any estimate we make of it. Base64 costs ~4 bytes per 3 of image.
ASSET_BUDGET_BYTES = 30 * 1024 * 1024


# ── Server error codes ──────────────────────────────────────────────────────

#: Retrying helps for exactly one of these: a single DOCX renders at a time
#: process-wide, and a concurrent call is refused rather than queued.
ERROR_SERVER_BUSY = "server_busy"
#: The renderer refuses to substitute a font. Without four Times New Roman
#: variants (or GOSTDOC_FONT_DIR) every render fails this way, which is a
#: deployment problem and not something a retry or a different document fixes.
ERROR_FONTS_MISSING = "fonts_missing"
ERROR_INPUT_INVALID = "input_validation_failed"
ERROR_REQUEST_TOO_LARGE = "request_too_large"
#: The DOCX was built and then could not be saved: the server's own S3 is down
#: or unconfigured. Observed on a freshly stood-up instance whose MinIO was not
#: wired, and it is worth naming, because "the document is fine, their storage
#: is not" is invisible from the code alone.
ERROR_STORAGE_FAILED = "storage_failed"

#: Codes that mean "the document is wrong"; the author must change the values.
AUTHORING_ERRORS: FrozenSet[str] = frozenset({
    ERROR_INPUT_INVALID,
    "output_validation_failed",
    "invalid_arguments",
    "invalid_asset_reference",
    "invalid_asset_name",
    "invalid_asset",
    ERROR_REQUEST_TOO_LARGE,
})


__all__ = [
    "CONTRACT_ID",
    "CONTRACT_VERSION",
    "envelope",
    "MODES",
    "DEFAULT_MODE",
    "PLACEHOLDER",
    "REQUIRED_FIELDS",
    "DEFAULTED_FIELDS",
    "REPORT_TYPE_INTERIM",
    "REPORT_TYPE_FINAL",
    "REPORT_TYPES",
    "AUTO",
    "KEYWORDS_MIN",
    "KEYWORDS_MAX",
    "ABSTRACT_MAX_CHARS",
    "APPENDIX_STATUSES",
    "BLOCK_FIELDS",
    "BLOCK_TYPES",
    "block_allowed_keys",
    "MAX_WIDTH_MM",
    "FIGURE_DEFAULT_WIDTH_MM",
    "FIGURE_MIN_WIDTH_MM",
    "SLUG_RE",
    "REFERENCE_ID_RE",
    "ASSET_NAME_RE",
    "ASSET_FORMATS",
    "ASSET_EXTENSIONS",
    "MAX_REQUEST_BYTES",
    "ASSET_BUDGET_BYTES",
    "ERROR_SERVER_BUSY",
    "ERROR_FONTS_MISSING",
    "ERROR_INPUT_INVALID",
    "ERROR_REQUEST_TOO_LARGE",
    "ERROR_STORAGE_FAILED",
    "AUTHORING_ERRORS",
]
