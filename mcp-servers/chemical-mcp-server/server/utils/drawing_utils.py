import io
import logging
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

logger = logging.getLogger(__name__)

_STEP_W = 900
_STEP_H = 300
_HEADER_H = 28
_MOL_SIZE = (300, 300)
_MOLS_PER_ROW = 4


def _blank_png(width: int, height: int, color: tuple = (245, 245, 245)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _reaction_png(reaction_smiles: str) -> bytes:
    """Render a single reaction SMILES to PNG bytes via RDKit."""
    try:
        rxn = AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True)
        if rxn is None:
            raise ValueError("null reaction")
        img = Draw.ReactionToImage(rxn, subImgSize=(250, _STEP_H))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        logger.warning("draw_reaction failed for %r: %s", reaction_smiles[:80], exc)
        return _blank_png(_STEP_W, _STEP_H)


def _header_strip(text: str, width: int = _STEP_W, height: int = _HEADER_H) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(210, 225, 245))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), text, fill=(20, 20, 20), font=font)
    return img


def draw_route_image(route: dict[str, Any]) -> bytes:
    """
    Draw all reaction steps of a retrosynthesis route as a vertical strip.
    Each step gets a labelled header + reaction image. Returns PNG bytes.
    """
    steps = route.get("steps", [])
    if not steps:
        return _blank_png(_STEP_W, _STEP_H)

    strips: list[Image.Image] = []
    for idx, step in enumerate(steps):
        rxn_smi = step.get("reaction_smiles", "")
        plausibility = step.get("plausibility")
        label = f"Step {idx + 1}"
        if plausibility is not None:
            label += f"   plausibility: {plausibility:.3f}"

        strips.append(_header_strip(label))

        rxn_img = Image.open(io.BytesIO(_reaction_png(rxn_smi)))
        # normalise width so all steps are the same width
        if rxn_img.width != _STEP_W:
            ratio = _STEP_W / rxn_img.width
            rxn_img = rxn_img.resize((_STEP_W, int(rxn_img.height * ratio)), Image.LANCZOS)
        strips.append(rxn_img)

    total_h = sum(s.height for s in strips)
    combined = Image.new("RGB", (_STEP_W, total_h), "white")
    y = 0
    for s in strips:
        combined.paste(s, (0, y))
        y += s.height

    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return buf.getvalue()


def draw_reactions_strip(reactions: list[tuple[str, str]]) -> bytes:
    """
    Draw a vertical strip of labelled reaction images.

    Args:
        reactions: List of (reaction_smiles, label) pairs.

    Returns:
        PNG bytes with all reactions stacked vertically.
    """
    if not reactions:
        return _blank_png(_STEP_W, _STEP_H)

    strips: list[Image.Image] = []
    for rxn_smi, label in reactions:
        strips.append(_header_strip(label))
        rxn_img = Image.open(io.BytesIO(_reaction_png(rxn_smi)))
        if rxn_img.width != _STEP_W:
            ratio = _STEP_W / rxn_img.width
            rxn_img = rxn_img.resize((_STEP_W, int(rxn_img.height * ratio)), Image.LANCZOS)
        strips.append(rxn_img)

    total_h = sum(s.height for s in strips)
    combined = Image.new("RGB", (_STEP_W, total_h), "white")
    y = 0
    for s in strips:
        combined.paste(s, (0, y))
        y += s.height

    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return buf.getvalue()


def draw_molecules_grid(
    smiles_list: list[str],
    labels: list[str] | None = None,
    mol_size: tuple[int, int] = _MOL_SIZE,
    mols_per_row: int = _MOLS_PER_ROW,
) -> bytes:
    """
    Draw a grid of molecules with optional labels. Returns PNG bytes.
    Silently skips unparseable SMILES.
    """
    mols: list[Any] = []
    valid_labels: list[str] = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_labels.append((labels[i] if labels else smi)[:50])

    if not mols:
        return _blank_png(mol_size[0], mol_size[1])

    result = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(mols_per_row, len(mols)),
        subImgSize=mol_size,
        legends=valid_labels,
        returnPNG=True,
    )
    if isinstance(result, bytes):
        return result
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()
