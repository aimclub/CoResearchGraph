from typing import List, Dict, Any
import requests
import os
import logging

from CoScientist.config import get_settings
settings = get_settings()

RETROSYNTHESIS_SERVICES_HOST = settings.hosts_ports.retrosynthesis_services_host 
RETROSYNTHESIS_SERVICES_PORT = settings.hosts_ports.retrosynthesis_services_port
RETROSYNTHESIS_SERVICES_URL = f"http://{RETROSYNTHESIS_SERVICES_HOST}:{RETROSYNTHESIS_SERVICES_PORT}"
REQUEST_TIMEOUT = 60

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrosynthesis_result(smiles: str, mode: str = "fast", max_routes: int = 5) -> Dict[str, Any]:
    """
    Proxy request to the Retrosynthesis service tree-search endpoint

    Args:
        smiles (str): Target molecule SMILES.
        mode (str): One of "fast", "balanced", "deep".
        max_routes (int): Maximum number of routes to return.
    Returns:
        response (Dict[str, Any]): Retrosynthesis result payload with:
            - target (str | None): input target SMILES returned by ASKCOS.
            - routes (List[Dict[str, Any]]): list of retrosynthesis routes:
                - id (str): unique route identifier.
                - depth (int | None): longest path length in the route.
                - precursor_cost (float | None): summed precursor cost metric.
                - score (float | None): overall route score.
                - min_step_plausibility (float | None): lowest step plausibility.
                - avg_step_plausibility (float | None): average step plausibility.
                - steps (List[Dict[str, Any]]): ordered reaction steps:
                    - reaction_smiles (str): step reaction SMILES.
                    - mapped_smiles (str | None): atom-mapped reaction SMILES.
                    - plausibility (float | None): step plausibility score.
                    - precursor_rank (int | None): ranking of precursor set.
                    - precursor_score (float | None): model score for precursors.
                    - model_score (float | None): model score for the step.
                    - template (Dict[str, Any] | None): template metadata:
                      reaction_smarts (str): reaction SMARTS pattern.
                      template_rank (int | None): rank among templates.
                      num_examples (int | None): template training examples count.
                    - reactants (List[Dict[str, Any]]): precursor molecules:
                      smiles (str): molecule SMILES.
                      terminal (bool | None): True if purchasable/terminal.
                      buy_link (str | None): vendor link if available.
                      stoichiometry (int): reagent count (default 1).
                    - products (List[Dict[str, Any]]): products, same schema
    """
    api_url = f"{RETROSYNTHESIS_SERVICES_URL}/api/v1/retrosynthesis/result"
    logger.info(f"Calling Retrosynthesis API: {api_url}")
    try:
        response = requests.post(
            api_url,
            json={"smiles": smiles},
            params={"mode": mode},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            error_msg = f"Retrosynthesis API returned status {response.status_code}: {response.text[:500]}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        json_response = response.json()
        if json_response is None:
            error_msg = "Retrosynthesis API returned None JSON response"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if isinstance(json_response, dict) and isinstance(json_response.get("routes"), list):
            json_response["routes"] = json_response["routes"][:max(0, int(max_routes))]
        return json_response
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to connect to Retrosynthesis API at {RETROSYNTHESIS_SERVICES_URL}: {str(e)}"
        logger.error(error_msg)
        raise ConnectionError(error_msg)

def classify_reaction_smiles(smiles: List[str], num_results: int = 10) -> Dict[str, Any]:
    """
    Proxy request to the ASKCOS reaction-classification endpoint.

    Args:
        smiles (List[str]): List of reaction SMILES, e.g. ["A.B>>C"].
        num_results (int): Max number of classes per reaction (1..50).
    Returns:
        response (Dict[str, Any]): ASKCOS classification payload with:
            - status_code (int): upstream status code.
            - message (str): upstream message.
            - result (List[Dict[str, Any]]): list of hits with:
                - rank (int): hit rank.
                - reaction_num (str): reaction identifier.
                - reaction_name (str): reaction name.
                - reaction_classnum (str): class number.
                - reaction_classname (str): class name.
                - reaction_superclassnum (str): superclass number.
                - reaction_superclassname (str): superclass name.
                - prediction_certainty (float): confidence score.
    """
    api_url = f"{RETROSYNTHESIS_SERVICES_URL}/api/v1/reaction-classification/classify"
    logger.info(f"Calling Reaction Classification API: {api_url}")
    try:
        response = requests.post(
            api_url,
            json={"smiles": smiles, "num_results": num_results},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            error_msg = (
                f"Reaction Classification API returned status {response.status_code}: "
                f"{response.text[:500]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        json_response = response.json()
        if json_response is None:
            error_msg = "Reaction Classification API returned None JSON response"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return json_response
    except requests.exceptions.RequestException as e:
        error_msg = (
            f"Failed to connect to Reaction Classification API at "
            f"{RETROSYNTHESIS_SERVICES_URL}: {str(e)}"
        )
        logger.error(error_msg)
        raise ConnectionError(error_msg)

def forward_predict_products(
    smiles: List[str],
    backend: str = "wldn5",
    model_name: str = "pistachio",
    reagents: str = "",
    solvent: str = "",
) -> Dict[str, Any]:
    """
    Proxy request to the ASKCOS forward prediction endpoint.

    Args:
        smiles (List[str]): Batch of reaction inputs (reactants).
        backend (str): One of "wldn5", "graph2smiles", "augmented_transformer".
        model_name (str): Model name for the backend (default "pistachio").
        reagents (str): Reagents string as in ASKCOS controller.
        solvent (str): Solvent string as in ASKCOS controller.
    Returns:
        response (Dict[str, Any]): ASKCOS forward payload with:
            - inputs (List[str]): normalized inputs (reactants+reagents+solvent).
            - backend (str): backend identifier used.
            - model_name (str): model name used.
            - predictions (List[Dict[str, Any]]): predicted products:
                - smiles (str): product SMILES.
                - score (float): model probability/score.
    """
    api_url = f"{RETROSYNTHESIS_SERVICES_URL}/api/v1/forward/predict"
    logger.info(f"Calling Forward Prediction API: {api_url}")
    try:
        response = requests.post(
            api_url,
            json={
                "smiles": smiles,
                "backend": backend,
                "model_name": model_name,
                "reagents": reagents,
                "solvent": solvent,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            error_msg = (
                f"Forward Prediction API returned status {response.status_code}: "
                f"{response.text[:500]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        json_response = response.json()
        if json_response is None:
            error_msg = "Forward Prediction API returned None JSON response"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"FORWARD PREDICTION JSON RESPONSE: {json_response}")
        return json_response
    except requests.exceptions.RequestException as e:
        error_msg = (
            f"Failed to connect to Forward Prediction API at "
            f"{RETROSYNTHESIS_SERVICES_URL}: {str(e)}"
        )
        logger.error(error_msg)
        raise ConnectionError(error_msg)

if __name__ == "__main__":
    print(retrosynthesis_result(smiles="C1CCCCC1"))
