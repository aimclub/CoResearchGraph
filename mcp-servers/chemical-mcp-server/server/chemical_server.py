import base64
import json
import logging
import uuid
from io import StringIO
from typing import Annotated, Dict, List, Optional

import aiohttp
import pandas as pd
import pubchempy as pcp
import py3Dmol
import requests
from fastmcp import FastMCP
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import CalcMolDescriptors

from .clients.affinity_db import (
    VALID_AFFINITY_TYPES,
    fetch_affinity_bindingdb,
    fetch_chembl_data,
    fetch_uniprot_id,
)
from .clients.chemical_client import ChemServiceError
from .config import get_settings
from .ocr_pipeline import (
    extract_molecules_from_image_urls,
    extract_reactions_from_image_urls,
)
from .service_resources import chem_service, retrosynthesis_service
from .utils import vault
from .utils.drawing_utils import draw_molecules_grid, draw_reactions_strip, draw_route_image
from .utils.vault import safe_id


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("ChemTools")


@mcp.tool()
async def fetch_activity_data(
    source: str,
    protein_name: str,
    protein_id: Optional[str] = None,
    affinity_type: str = "IC50",
    cutoff: int = 10000,
    user_id: Annotated[Optional[str], "Owner of the session. The framework supplies it."] = None,
    session_id: Annotated[Optional[str], "Current session. The framework supplies it."] = None,
) -> dict:
    """
    Unified data retrieval tool for biochemical databases.

    Fetches protein-ligand interaction or activity data from BindingDB or ChEMBL
    and uploads the result as a CSV to S3.

    Args:
        source (str): Data source ("bindingdb" or "chembl").
        protein_name (str): Target protein name.
        protein_id (str, optional): Target protein id. If passed, protein_name is ignored.
        affinity_type (str, optional): Type of affinity (Ki, Kd, IC50). Defaults to "IC50".
        cutoff (int, optional): Optional threshold (nM) for BindingDB. Defaults to 10000.
        user_id: Session owner. The framework fills this in.
        session_id: Session identifier. The framework fills this in.

    Returns:
        dict: ``bucket``, ``s3_key`` and ``presigned_url`` of the CSV, plus
        ``info`` describing the dataset. On failure, ``error``.
    """
    source = source.lower().strip()
    if affinity_type not in VALID_AFFINITY_TYPES:
        return {"error": f"Invalid affinity type '{affinity_type}'. Must be one of {VALID_AFFINITY_TYPES}"}

    try:
        async with aiohttp.ClientSession() as session:
            if source == "bindingdb":
                target_id = protein_id
                if not target_id:
                    resolved_id = await fetch_uniprot_id(session, protein_name)
                    if not resolved_id:
                        return {"error": f"[BindingDB] Could not find UniProt ID for '{protein_name}'"}
                    target_id = resolved_id

                results = await fetch_affinity_bindingdb(
                    session, target_id, affinity_type, cutoff
                )
            elif source == "chembl":
                results = await fetch_chembl_data(
                    target_name=protein_name,
                    target_id=protein_id,
                    affinity_type=affinity_type,
                    session=session,
                )
            else:
                return {"error": f"Unsupported data source '{source}'. Use 'bindingdb' or 'chembl'."}

        if not isinstance(results, list):
            return {"answer": results}

        df = pd.DataFrame(results)
        if len(df) == 0:
            return {"error": "The data was not saved because it is empty."}

        buffer = StringIO()
        df.info(buf=buffer)
        # The file used to be written to a local path the caller chose. That path
        # means nothing outside this container, so the next step could never open
        # it. The CSV goes to S3 instead, under the session prefix.
        target = safe_id(protein_id or protein_name, "target")
        stored = vault.upload(
            user_id, session_id, "activity_data",
            f"{source}_{target}_{affinity_type}_{uuid.uuid4()}.csv",
            df.to_csv(index=False).encode("utf-8"),
        )
        return {**stored, "rows": len(df), "info": buffer.getvalue()}
    except Exception as e:
        return {"error": f"[fetch_activity_data] Error: {str(e)}"}


@mcp.tool()
def name2smiles(
    mol: Annotated[str, "Name of a molecule"],
):
    """
    Convert a molecule name to its SMILES representation.
    
    This method retrieves the SMILES string for a given molecule name via PubChem (pubchempy).
    
    Args:
        mol (str): The name of the molecule to convert.
    
    Returns:
        str: The SMILES string representation of the molecule if successful,
             an error message if the request fails,
             or a "couldn't obtain smiles" message if the name is invalid.
    """
    try:
        compound = pcp.get_compounds(mol, "name")
        if not compound:
            return "I've couldn't obtain smiles, the name is wrong"
        return compound[0].canonical_smiles
    except requests.RequestException as e:
        return f"Failed to execute. Error: {repr(e)}"
    except (IndexError, AttributeError):
        return "I've couldn't obtain smiles, the name is wrong"


@mcp.tool()
def smiles2name(smiles: Annotated[str, "SMILES of a molecule"]):
    """
    Converts a SMILES string representing a molecule into its IUPAC name.
    
    Args:
        smiles (str): The SMILES string of the molecule.
    
    Returns:
        str: The IUPAC name of the molecule, or an error message if the conversion fails.
    """

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{smiles}/property/IUPACName/JSON"
    )
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            data = response.json()
            properties = data.get("PropertyTable", {}).get("Properties", [])
            if not properties:
                return 'Could not find such IUPAC'  # or raise, or return a sentinel
            prop = properties[0]
            # CID 0 means PubChem couldn't resolve the SMILES
            if prop.get("CID", 0) == 0:
                return 'Could not find such IUPAC'
            return prop.get("IUPACName", 'Could not find such IUPAC')  # may still be None if name isn't available
        return "I've couldn't get iupac name"
    except requests.RequestException as e:
        return f"Failed to execute. Error: {repr(e)}"
    except (KeyError, IndexError, json.JSONDecodeError):
        return "I've couldn't get iupac name"


@mcp.tool()
def smiles2prop(
    smiles: Annotated[Optional[str], "The SMILES string of the molecule. Leave empty if providing IUPAC name."] = None,
    iupac: Annotated[Optional[str], "The IUPAC name of the molecule. (e.g., 'aspirin'). Leave empty if providing SMILES."] = None
):
    """
    Calculate molecular properties from a SMILES string or IUPAC name.
    
    Args:
        smiles (str, optional): The SMILES string of the molecule.
        iupac (str, optional): The IUPAC name of the molecule.
    
    Returns:
        CalcMolDescriptors: An object containing calculated molecular properties. 
                             Returns an error message as a string if the calculation fails.
    """

    try:
        if not smiles and not iupac:
            return "Failed to execute. Error: Either smiles or iupac must be provided."

        if iupac:
            compound = pcp.get_compounds(iupac, "name")
            if len(compound):
                smiles = compound[0].canonical_smiles

        res = CalcMolDescriptors(Chem.MolFromSmiles(smiles))
        return res
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"


@mcp.tool()
def visualize_molecule(
    smiles: Annotated[str, "SMILES of a molecule"],
    user_id: Annotated[Optional[str], "Owner of the session. The framework supplies it."] = None,
    session_id: Annotated[Optional[str], "Current session. The framework supplies it."] = None,
) -> dict:
    """
    Visualizes a molecule from its SMILES and stores the HTML file in S3.

    Args:
        smiles: SMILES string of the molecule.
        user_id: Session owner. The framework fills this in.
        session_id: Session identifier. The framework fills this in.

    Returns:
        dict: ``bucket``, ``s3_key`` and ``presigned_url`` of the HTML
        visualization. The URL expires in one hour, the object does not, so keep
        the bucket and the key. On failure, ``error``.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}

        AllChem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        view = py3Dmol.view(
            data=Chem.MolToMolBlock(mol),
            style={"stick": {}, "sphere": {"scale": 0.3}},
            width=600,
            height=400,
        )
        view.setBackgroundColor("#b8bfcc")
        view.zoomTo()

        return vault.upload(
            user_id, session_id, "molecule_visualizations",
            f"{uuid.uuid4()}.html",
            view.write_html().encode("utf-8"),
        )

    except Exception as e:
        return {"error": f"Failed to visualize molecule. Error: {repr(e)}"}


@mcp.tool()
def extract_reactions(
    image_urls: Annotated[List[str], "List of public HTTP(S) URLs of images"],
    user_id: Annotated[Optional[str], "Owner of the session. The framework supplies it."] = None,
    session_id: Annotated[Optional[str], "Current session. The framework supplies it."] = None,
) -> Dict:
    """Detect chemical reactions in images loaded by URLs (ChemService).

    Each URL is processed in turn (in memory, no local cache). Annotated JPEGs go
    to S3 under the session prefix.

    Args:
        image_urls: One or more direct image links.
        user_id: Session owner. The framework fills this in.
        session_id: Session identifier. The framework fills this in.

    Returns:
        dict: ``answer`` maps labels (from URL paths; disambiguated with ``__{index}`` on collision)
        to reaction dicts. ``metadata`` has ``annotated_images`` (each with ``bucket``,
        ``s3_key`` and ``presigned_url``), ``source_urls``, and optionally ``failed``
        (per-URL errors) if some URLs could not be processed.
    """
    try:
        response = extract_reactions_from_image_urls(image_urls, user_id, session_id)
        return response
    except ChemServiceError as e:
        logger.error("extract_reactions ChemServiceError: %s", e)
        return {"answer": f"ChemService reaction extraction failed: {e}"}
    except requests.RequestException as e:
        logger.error("extract_reactions download ERROR: %s", e)
        return {"answer": f"Could not download an image: {e}"}
    except Exception as e:
        logger.error("extract_reactions ERROR: %s", e)
        return {"answer": f"Could not extract reactions from images. Error: {e}"}


@mcp.tool()
def extract_molecules(
    image_urls: Annotated[List[str], "List of public HTTP(S) URLs of images"],
    user_id: Annotated[Optional[str], "Owner of the session. The framework supplies it."] = None,
    session_id: Annotated[Optional[str], "Current session. The framework supplies it."] = None,
) -> Dict:
    """Detect molecular structures in images loaded by URLs (ChemService).

    Each URL is processed in turn. Annotated JPEGs go to S3 under the session prefix.

    Args:
        image_urls: One or more direct image links.
        user_id: Session owner. The framework fills this in.
        session_id: Session identifier. The framework fills this in.

    Returns:
        dict: ``answer`` maps labels to ``smiles`` / ``errors``. ``metadata`` includes
        ``annotated_images`` (each with ``bucket``, ``s3_key`` and ``presigned_url``),
        ``source_urls``, and optionally ``failed``.
    """
    try:
        response = extract_molecules_from_image_urls(image_urls, user_id, session_id)
        return response
    except ChemServiceError as e:
        logger.error("extract_molecules ChemServiceError: %s", e)
        return {"answer": f"ChemService molecule extraction failed: {e}"}
    except requests.RequestException as e:
        logger.error("extract_molecules download ERROR: %s", e)
        return {"answer": f"Could not download an image: {e}"}
    except Exception as e:
        logger.error("extract_molecules ERROR: %s", e)
        return {"answer": f"Could not extract molecules from images. Error: {e}"}



@mcp.tool()
def calculate_docking(
    smiles: str,
    pdb_id: str,
    user_id: Annotated[Optional[str], "Owner of the session. The framework supplies it."] = None,
    session_id: Annotated[Optional[str], "Current session. The framework supplies it."] = None,
) -> dict:
    """
    Calculate docking score for a molecule and upload the HTML visualization to S3.

    Args:
        smiles: SMILES string of the molecule.
        pdb_id: PDB identifier for the receptor structure.
        user_id: Session owner. The framework fills this in.
        session_id: Session identifier. The framework fills this in.

    Returns:
        dict: ``answer`` with affinity and errors. When a visualization comes
        back, ``metadata.docking_html`` holds ``bucket``, ``s3_key`` and
        ``presigned_url``. Keep the bucket and the key: the URL expires in an hour.
    """
    try:
        response = chem_service.calculate_docking_score(smiles, pdb_id)
    except ChemServiceError as e:
        return {
            "answer": {"affinity": None, "errors": str(e)},
            "metadata": {},
        }

    if isinstance(response, dict) and "data" in response:
        data = response.get("data")
        errors = response.get("error")
    else:
        data = response if isinstance(response, dict) else None
        errors = response.get("error") if isinstance(response, dict) else None

    affinity = None
    docking_html: Optional[dict] = None

    if data:
        affinity = data.get("affinity")
        visualization = data.get("visualization")
        if visualization:
            if isinstance(visualization, (bytes, bytearray)):
                html_content = bytes(visualization)
            else:
                html_content = base64.b64decode(visualization)
            filename = f"docking_{pdb_id}_{uuid.uuid4()}.html"
            docking_html = vault.upload(
                user_id, session_id, "docking_results", filename, html_content,
            )

    return {
        "answer": {"affinity": affinity, "errors": errors},
        "metadata": ({"docking_html": docking_html} if docking_html else {}),
    }
    
@mcp.tool()
def retrosynthesis_tree_search(
    smiles: Annotated[str, "Target molecule SMILES"],
    mode: Annotated[str, "One of: fast, balanced, deep"] = "fast",
    user_id: Annotated[Optional[str], "Owner of the session. The framework supplies it."] = None,
    session_id: Annotated[Optional[str], "Current session. The framework supplies it."] = None,
) -> Dict:
    """
    Plan a retrosynthesis route for a target molecule.

    Use this when the user asks for possible synthetic routes or precursors
    for a target SMILES. This calls the retrosynthesis service and returns
    ASKCOS-like routes with steps, reactants, and scores.

    Args:
        smiles (str): Target molecule SMILES.
        mode (str): Search depth/quality preset ("fast", "balanced", "deep").

    Returns:
        dict: Retrosynthesis result payload with:
            - target (str | None): input target SMILES returned by ASKCOS.
            - routes (List[Dict]): list of retrosynthesis routes:
                - id (str): unique route identifier.
                - depth (int | None): longest path length in the route.
                - precursor_cost (float | None): summed precursor cost metric.
                - score (float | None): overall route score.
                - min_step_plausibility (float | None): lowest step plausibility.
                - avg_step_plausibility (float | None): average step plausibility.
                - steps (List[Dict]): ordered reaction steps:
                    - reaction_smiles (str): step reaction SMILES.
                    - mapped_smiles (str | None): atom-mapped reaction SMILES.
                    - plausibility (float | None): step plausibility score.
                    - precursor_rank (int | None): ranking of precursor set.
                    - precursor_score (float | None): model score for precursors.
                    - model_score (float | None): model score for the step.
                    - template (Dict | None): template metadata:
                        reaction_smarts (str): reaction SMARTS pattern.
                        template_rank (int | None): rank among templates.
                        num_examples (int | None): template training examples count.
                    - reactants (List[Dict]): precursor molecules:
                        smiles (str): molecule SMILES.
                        terminal (bool | None): True if purchasable/terminal.
                        buy_link (str | None): vendor link if available.
                        stoichiometry (int): reagent count (default 1).
                    - products (List[Dict]): products, same schema as reactants.
            - metadata (Dict): visualization info:
                - route_images (List[Dict]): one entry per route with:
                    - route_id (str): route identifier.
                    - bucket (str): S3 bucket.
                    - s3_key (str): S3 object key.
                    - presigned_url (str): temporary URL to view the image (1 h TTL).
        On failure returns a dict with an "answer" message.
    """
    try:
        result = retrosynthesis_service.retrosynthesis_result(smiles=smiles, mode=mode)
    except Exception as e:
        logger.error(f"retrosynthesis_tree_search ERROR: {e}")
        return {"answer": "Could not run retrosynthesis tree search."}

    metadata: Dict = {}
    try:
        route_images = []
        for i, route in enumerate(result.get("routes", [])):
            route_id = route.get("id", f"route_{i}")
            img_bytes = draw_route_image(route)
            stored = vault.upload(
                user_id, session_id, "retrosynthesis",
                f"{route_id}_{uuid.uuid4()}.png", img_bytes,
            )
            route_images.append({"route_id": route_id, **stored})
        metadata["route_images"] = route_images
    except Exception as e:
        logger.warning("retrosynthesis_tree_search: could not render images: %s", e)

    result["metadata"] = metadata
    return result

@mcp.tool()
def classify_reaction(
    smiles: Annotated[
        List[str],
        "Each entry is one full reaction SMILES: 'A.B>>C' (not separate molecules per list item)",
    ],
    num_results: Annotated[int, "Max classes per reaction (1..50)"] = 10,
) -> Dict:
    """
    Classify reaction SMILES into reaction classes.

    Use this when the user provides reaction SMILES and wants the reaction
    type/class (e.g., named reactions or class labels). Returns ASKCOS-like
    classification hits with ranks and confidence.

    Args:
        smiles (List[str]): One or more reaction strings; each is reactants>>products,
            with multiple reactants joined by "." (e.g. ["CCO.CC(=O)O>>CCOC(=O)C"]).
        num_results (int): Max number of classes per reaction (1..50).

    Returns:
        dict: ASKCOS classification payload with:
            - status_code (int): upstream status code.
            - message (str): upstream message.
            - result (List[Dict]): list of hits with:
                - rank (int): hit rank.
                - reaction_num (str): reaction identifier.
                - reaction_name (str): reaction name.
                - reaction_classnum (str): class number.
                - reaction_classname (str): class name.
                - reaction_superclassnum (str): superclass number.
                - reaction_superclassname (str): superclass name.
                - prediction_certainty (float): confidence score.
        On failure returns a dict with an "answer" message.
    """
    try:
        return retrosynthesis_service.classify_reaction_smiles(smiles=smiles, num_results=num_results)
    except Exception as e:
        logger.error(f"classify_reaction ERROR: {e}")
        return {"answer": "Could not classify reaction SMILES."}

@mcp.tool()
def forward_predict(
    smiles: Annotated[List[str], "Batch of reaction inputs (reactants)"],
    backend: Annotated[str, "One of: wldn5, graph2smiles, augmented_transformer"],
    retrosynthesis_model_name: Annotated[str, "Model name for backend"] = "pistachio",
    reagents: Annotated[str, "Reagents string"] = "",
    solvent: Annotated[str, "Solvent string"] = "",
    user_id: Annotated[Optional[str], "Owner of the session. The framework supplies it."] = None,
    session_id: Annotated[Optional[str], "Current session. The framework supplies it."] = None,
) -> Dict:
    """
    Predict reaction products from reactants (forward synthesis).

    Use this when the user provides reactants and wants predicted products.
    You can specify backend/model_name and optional reagents/solvent strings.

    Args:
        smiles (List[str]): Batch of reaction inputs (reactants).
        backend (str): One of "wldn5", "graph2smiles", "augmented_transformer".
        retrosynthesis_model_name (str): Model name for the backend (default "pistachio").
        reagents (str): Reagents string as in ASKCOS controller.
        solvent (str): Solvent string as in ASKCOS controller.

    Returns:
        dict: ASKCOS forward payload with:
            - inputs (List[str]): normalized inputs (reactants+reagents+solvent).
            - backend (str): backend identifier used.
            - model_name (str): model name used.
            - predictions (List[Dict]): predicted products:
                - smiles (str): product SMILES.
                - score (float): model probability/score.
            - metadata (Dict): visualization info:
                - predictions_image (Dict):
                    - bucket (str): S3 bucket.
                    - s3_key (str): S3 object key.
                    - presigned_url (str): temporary URL to view the grid image (1 h TTL).
                - top_reactions_image (Dict): reaction drawings for top 3 products,
                  same three fields.
        On failure returns a dict with an "answer" message.
    """
    try:
        result = retrosynthesis_service.forward_predict_products(
            smiles=smiles,
            backend=backend,
            model_name=retrosynthesis_model_name,
            reagents=reagents,
            solvent=solvent,
        )
    except Exception as e:
        logger.error(f"forward_predict ERROR: {e}")
        return {"answer": "Could not run forward prediction."}

    metadata: Dict = {}
    try:
        predictions = result.get("predictions", [])
        smiles_list = [p["smiles"] for p in predictions if p.get("smiles")]
        labels = [
            f"score: {p['score']:.3f}" if isinstance(p.get("score"), float) else ""
            for p in predictions
            if p.get("smiles")
        ]
        img_bytes = draw_molecules_grid(smiles_list, labels=labels)
        metadata["predictions_image"] = vault.upload(
            user_id, session_id, "forward_prediction",
            f"predictions_{uuid.uuid4()}.png", img_bytes,
        )
    except Exception as e:
        logger.warning("forward_predict: could not render images: %s", e)

    try:
        predictions = result.get("predictions", [])
        inputs = result.get("inputs", smiles)
        reactants_smi = ".".join(inputs) if isinstance(inputs, list) else str(inputs)
        top = [p for p in predictions if p.get("smiles")][:3]
        reactions = []
        for i, p in enumerate(top):
            rxn_smi = f"{reactants_smi}>>{p['smiles']}"
            score = p.get("score")
            label = f"Top {i + 1}"
            if isinstance(score, float):
                label += f"   score: {score:.3f}"
            reactions.append((rxn_smi, label))
        if reactions:
            rxn_img_bytes = draw_reactions_strip(reactions)
            metadata["top_reactions_image"] = vault.upload(
                user_id, session_id, "forward_prediction",
                f"top_reactions_{uuid.uuid4()}.png", rxn_img_bytes,
            )
    except Exception as e:
        logger.warning("forward_predict: could not render reaction images: %s", e)

    result["metadata"] = metadata
    return result

# Every tool that makes a figure, a table or a CSV stores it in S3. Without this
# configuration the server starts, accepts a call, and then fails inside boto3.
_REQUIRED_S3_SETTINGS = (
    ("S3__ENDPOINT_URL", "s3_endpoint_url"),
    ("S3__BUCKET_NAME", "s3_bucket_name"),
    ("S3__ACCESS_KEY", "s3_access_key"),
    ("S3__SECRET_KEY", "s3_secret_key"),
)


def _check_s3_settings(settings) -> None:
    """Stop the server when the S3 configuration is incomplete.

    A missing endpoint sends boto3 to the real AWS endpoint of the bucket name.
    On a host without that egress the call blocks until the retries run out. The
    tool returns nothing, the file reaches no bucket, and the log shows only the
    stream teardown that follows. Refuse to start instead.
    """
    missing = [name for name, attr in _REQUIRED_S3_SETTINGS if not getattr(settings, attr)]
    if missing:
        raise SystemExit(
            "chemical-mcp-server: missing S3 configuration: "
            + ", ".join(missing)
            + ". Set these in the env_file of the service. Two compose files"
            " start this server, and they read different files:"
            " mcp-servers/docker-compose.yml reads mcp-servers/.env, and"
            " mcp-servers/chemical-mcp-server/docker-compose.yml reads"
            " mcp-servers/chemical-mcp-server/.env."
        )


def main() -> None:
    """Entry point for the MCP server."""
    settings = get_settings()
    _check_s3_settings(settings)
    logger.info(
        "S3 endpoint %s, bucket %s", settings.s3_endpoint_url, settings.s3_bucket_name,
    )
    mcp.run(
        transport="http",
        host=settings.chem_mcp_host,
        port=settings.chem_mcp_port,
        path=settings.chem_mcp_path,
    )


if __name__ == "__main__":
    main()
