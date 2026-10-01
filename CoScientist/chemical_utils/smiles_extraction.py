"""Pull SMILES molecules AND reaction SMILES out of free-text RAG/literature
answers.

The paper-analysis MCP tools (`explore_scientific_database`, `explore_my_papers`)
are prompted to copy SMILES strings verbatim into their text answer (see
mcp-servers/paper-analysis-mcp-server/prompts.py, rule 7), but they hand back
plain prose — nothing marks *which* substring is a SMILES, or whether it names
a single molecule or a whole reaction (`reactants>agents>products`). This
module finds candidate tokens and keeps only the ones RDKit can actually
parse — as a molecule for `extract_smiles`, as a reaction (with at least one
reactant AND one product) for `extract_reactions` — so a stray word, number or
degenerate `A>>` fragment never gets treated as real chemistry.
"""
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdChemReactions
    RDLogger.DisableLog("rdApp.*")  # RDKit logs a parse error per rejected candidate — silence that
    _RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - rdkit is an optional/transitive dependency
    Chem = None
    rdChemReactions = None
    _RDKIT_AVAILABLE = False
    logger.warning("rdkit not installed — chemistry extraction from literature text is disabled.")

# Text is split on whitespace/quotes/markdown-table pipes; a SMILES never contains these.
_TOKEN_SPLIT_RE = re.compile(r"[\s,;|`\"'“”‘’]+")
# The characters SMILES is built from (organic subset + charges/rings/stereo).
_SMILES_CHARSET_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$:/\\%.]+$")
# Require at least one character that plain English words don't use, so bare
# words ("Instructions", "CoScientist") never reach the (expensive) RDKit check.
_SMILES_MARKER_RE = re.compile(r"[0-9@+#\[\]=]")

_MIN_LEN = 4
_MAX_LEN = 200


def _is_valid_smiles(token: str) -> bool:
    return _RDKIT_AVAILABLE and Chem.MolFromSmiles(token) is not None


def _canonical(token: str) -> str:
    if not _RDKIT_AVAILABLE:
        return token
    mol = Chem.MolFromSmiles(token)
    return Chem.MolToSmiles(mol) if mol is not None else token


def extract_smiles(text: str) -> List[str]:
    """Return the deduplicated, RDKit-canonicalized SMILES found in ``text``.

    Order is preserved (first occurrence wins); returns [] if RDKit is not
    installed or no candidate validates, never raises.
    """
    if not text or not _RDKIT_AVAILABLE:
        return []

    seen_canonical = set()
    results: List[str] = []

    for raw in _TOKEN_SPLIT_RE.split(text):
        for candidate in (raw, raw.rstrip(".")):
            if not (_MIN_LEN <= len(candidate) <= _MAX_LEN):
                continue
            if not _SMILES_MARKER_RE.search(candidate):
                continue
            if not _SMILES_CHARSET_RE.match(candidate):
                continue
            if not _is_valid_smiles(candidate):
                continue

            canonical = _canonical(candidate)
            if canonical not in seen_canonical:
                seen_canonical.add(canonical)
                results.append(canonical)
            break  # accepted — don't also try the un-stripped/stripped sibling

    return results


# ── Reactions (reactants>agents>products) ───────────────────────────────────
# Same charset as a molecule plus '>', the reaction-SMILES step separator.
_REACTION_CHARSET_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$:/\\%.>]+$")

_MIN_REACTION_LEN = 9   # shortest real one: "C>>C" territory plus some margin
_MAX_REACTION_LEN = 400  # multi-reagent steps run longer than a single molecule


def _parse_reaction(token: str):
    """Return a RDKit ChemicalReaction for a real reaction, or None.

    Rejects anything RDKit can't parse AND anything with no reactant or no
    product (e.g. "CC>>" or ">>CC" parse fine but describe no reaction).
    """
    if not _RDKIT_AVAILABLE or token.count(">") != 2:
        return None
    try:
        rxn = rdChemReactions.ReactionFromSmarts(token, useSmiles=True)
    except Exception:
        return None
    if rxn is None:
        return None
    if rxn.GetNumReactantTemplates() < 1 or rxn.GetNumProductTemplates() < 1:
        return None
    return rxn


def extract_reactions(text: str) -> List[str]:
    """Return the deduplicated, RDKit-canonicalized reaction SMILES found in
    ``text`` — each with at least one reactant and one product.

    Order is preserved (first occurrence wins); returns [] if RDKit is not
    installed or no candidate validates, never raises. Molecule SMILES never
    match here (no '>' in the organic-subset charset extract_smiles uses),
    and reaction SMILES never match extract_smiles (its charset excludes '>')
    — the two are mutually exclusive by construction.
    """
    if not text or not _RDKIT_AVAILABLE:
        return []

    seen_canonical = set()
    results: List[str] = []

    for raw in _TOKEN_SPLIT_RE.split(text):
        for candidate in (raw, raw.rstrip(".")):
            if not (_MIN_REACTION_LEN <= len(candidate) <= _MAX_REACTION_LEN):
                continue
            if not _REACTION_CHARSET_RE.match(candidate):
                continue

            rxn = _parse_reaction(candidate)
            if rxn is None:
                continue

            canonical = rdChemReactions.ReactionToSmiles(rxn)
            if canonical not in seen_canonical:
                seen_canonical.add(canonical)
                results.append(canonical)
            break  # accepted — don't also try the un-stripped/stripped sibling

    return results
