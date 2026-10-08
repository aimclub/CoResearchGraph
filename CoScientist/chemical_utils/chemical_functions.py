import requests
from typing import List, Dict, Any, Callable
from functools import wraps
import os
import logging
import inspect

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHEM_SERVICES_HOST = os.getenv("HOSTS_PORTS__OPENCHEMIE_HOST")
CHEM_SERVICES_PORT = os.getenv("HOSTS_PORTS__OPENCHEMIE_PORT")
CHEM_SERVICES_URL = f"http://{CHEM_SERVICES_HOST}:{CHEM_SERVICES_PORT}"
REQUEST_TIMEOUT = 60


def handle_api_request(endpoint: str, file_param_name: str = None, ):
    """
    Decorator for handling requests to Chemical ToolsService API.
    
    Args:
        endpoint (str): API endpoint path (e.g., "/extract_molecules_from_figure/")
        file_param_name (str): Name of the file parameter in multipart/form-data (e.g., "image" or "pdf_file")
    
    Returns:
        A decorator that wraps a function and performs all necessary checks.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            """
            Wrapper that executes API request and handles all errors.
            
            For file uploads:
                Args: file_data (bytes): File data (image or PDF)
            
            For parameter requests:
                Args: Any parameters passed to the function (e.g., smiles, pdb_id)
            
            Returns:
                Data from the "data" field of the API response
            """
            try:
                api_url = f"{CHEM_SERVICES_URL}{endpoint}"
                logger.info(f"Calling ChemService API: {api_url}")
                if file_param_name:
                    if args:
                        file_data = args[0]
                    elif file_param_name in kwargs:
                        file_data = kwargs.pop(file_param_name)
                    else:
                        raise ValueError(f"File data must be provided as first argument or '{file_param_name}' keyword")
                    
                    response = requests.post(
                        api_url,
                        files={file_param_name: file_data},
                        timeout=REQUEST_TIMEOUT
                    )
                else:
                    params = {}
                    if args:
                        sig = inspect.signature(func)
                        param_names = list(sig.parameters.keys())
                        for i, arg in enumerate(args):
                            if i < len(param_names):
                                params[param_names[i]] = arg
                    params.update(kwargs)
                    
                    response = requests.post(
                        api_url,
                        params=params,
                        timeout=REQUEST_TIMEOUT
                    )
                if response.status_code != 200:
                    error_msg = f"ChemService API returned status {response.status_code}: {response.text[:500]}"
                    logger.error(error_msg)
                    return {'errors': error_msg}

                json_response = response.json()
                if json_response is None:
                    error_msg = "ChemService API returned None JSON response"
                    logger.error(error_msg)
                    return {'errors': error_msg}

                if "data" not in json_response:
                    error_msg = f"ChemService API response missing 'data' field. Response: {json_response}"
                    logger.error(error_msg)
                    return {'errors': error_msg}
                
                return json_response
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Failed to connect to ChemService API at {CHEM_SERVICES_URL}: {str(e)}"
                logger.error(error_msg)
                return {'errors': error_msg}
        return wrapper
    return decorator


@handle_api_request(endpoint="/extract_reactions_from_pdf/", file_param_name="pdf_file")
def extract_reactions_from_pdf(file: bytes) -> List[Dict]:
    """
    Extract reactions information from a PDF file.
    Response contains list of reactions for each page of the PDF.
    Each reaction contains list of reactants, products and conditions.
    
    Args:
        file (bytes): PDF file to extract reactions from.
    Returns:
        response (List[Dict]): List of reactions in pdf file for each page.
    Raises:
        ConnectionError: If API is unavailable or connection fails.
        ValueError: If API returns invalid response.
        RuntimeError: For unexpected errors.
    """
    pass


@handle_api_request(endpoint="/extract_reactions_from_figure/", file_param_name="image")
def extract_reactions_from_figure(image: bytes) -> List[Dict]:
    """
    Extract reactions information from an image.
    
    Response contains list of reactions on the image.
    Each reaction contains list of reactants, products and conditions.
    Args:
        image (bytes): Image to extract reactions from.
    Returns:
        response (List[Dict]): List of reactions on the image.
    """
    pass


@handle_api_request(endpoint="/extract_molecules_from_pdf/", file_param_name="pdf_file")
def extract_molecules_from_pdf(file: bytes) -> List[Dict]:
    """
    Extract molecules information from a PDF file.
    Response contains list of molecules for each page of the PDF.
    Each molecule contains bbox and smiles.
    
    Args:
        file (bytes): PDF file to extract molecules from.
    Returns:
        response (List[Dict]): List of molecules in pdf file for each page.
    """
    pass


@handle_api_request(endpoint="/extract_molecules_from_figure/", file_param_name="image")
def extract_molecules_from_figure(image: bytes) -> List[Dict]:
    """
    Extract molecules information from an image.
    Response contains list of molecules on the image.
    Each molecule contains bbox and smiles.
    
    Args:
        image (bytes): Image to extract molecules from.
    Returns:
        response (List[Dict]): List of molecules on the image.
    """
    pass


@handle_api_request(endpoint="/convert_image_to_smiles/", file_param_name="image")
def convert_image_to_smiles(image: bytes) -> str:
    """
    Convert an image to a smiles string.
    Response contains smiles string of the image.
    Args:
        image (bytes): Image to convert to smiles.
    Returns:
        response (str): SMILES string of the image.
    """
    pass

@handle_api_request(endpoint="/docking/", file_param_name=None)
def calculate_docking_score(smiles: str, pdb_id: str) -> str:
    """
    Calculate docking score for a molecule.
    Response contains docking score for the molecule.
    Args:
        smiles (str): SMILES string of the molecule.
        pdb_id (str): ID of the PDB file containing the receptor structure.
    Returns:
        response (str): Docking score for the molecule.
    """
    pass


def remove_keys(obj: Any, keys_to_remove: set[str] = {"bbox", "score"}) -> Any:
    """Processes ChemService json output to remove unnecessary keys like 'score' and 'bbox'."""
    if isinstance(obj, dict):
        for k in keys_to_remove:
            obj.pop(k, None)
        for v in obj.values():
            remove_keys(v, keys_to_remove)
    elif isinstance(obj, list):
        for item in obj:
            remove_keys(item, keys_to_remove)
    return obj


if __name__ == "__main__":
    result = calculate_docking_score(smiles="C1CCCCC1", pdb_id="5vfi")
    print(result)
