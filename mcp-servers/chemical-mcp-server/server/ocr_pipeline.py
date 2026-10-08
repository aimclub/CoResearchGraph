import uuid
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

from .service_resources import chem_service
from .utils import vault
from .utils.image_utils import download_url_to_bytes, draw_bboxes_on_image

ANNOTATED_IMAGES_FEATURE = "annotated_images"


def _label_from_image_url(url: str) -> str:
    """
    Builds a short label for an image from the last URL path.

    Args:
        url (str): URL for image.

    Returns:
        str: Unquoted filename from the path, or "image" if the path has no name.
    """
    path = urlparse(url.strip()).path
    name = path.rstrip("/").rsplit("/", 1)[-1] if path else ""
    return unquote(name) if name else "image"


def _upload_annotated_jpeg(jpeg_bytes: bytes, user_id: Optional[str],
                           session_id: Optional[str]) -> Dict:
    """
    Uploads annotated bytes under the session prefix.

    Args:
        jpeg_bytes (bytes): image data.
        user_id (str | None): session owner.
        session_id (str | None): session identifier.

    Returns:
        Dict: bucket, s3_key and presigned_url of the stored image.
    """
    return vault.upload(
        user_id, session_id, ANNOTATED_IMAGES_FEATURE,
        f"{uuid.uuid4()}.jpg", jpeg_bytes,
    )


def _normalize_figure_response(raw: Any) -> tuple[list, Optional[Any]]:
    """
    Normalizes the chemical figure API payload into recognitions and errors.

    Args:
        raw (Any): Raw response from the figure extractor (dict with "data"/"errors" or a list).

    Returns:
        tuple[list, Optional[Any]]: (recognitions list, errors or None). Empty list if shape is unknown.
    """
    if isinstance(raw, dict):
        return raw.get("data", []), raw.get("errors")
    if isinstance(raw, list):
        return raw, None
    return [], None


def extract_molecules_from_image_url(image_url: str, user_id: Optional[str] = None,
                                     session_id: Optional[str] = None) -> Dict:
    """
    Extracts molecule SMILES and bounding boxes from a single figure URL.

    Args:
        image_url (str): URL of the image to analyze.

    Returns:
        Dict: Keys "answer" (label → smiles/errors) and "metadata" (annotated_images, source_url).
    """
    label = _label_from_image_url(image_url)
    img_bytes = download_url_to_bytes(image_url)
    raw = chem_service.extract_molecules_from_figure(img_bytes)
    recognitions, errors = _normalize_figure_response(raw)

    entries: list = []
    if recognitions and isinstance(recognitions[0], dict):
        entries = recognitions[0].get("bboxes", []) or []

    bboxes: list = []
    smiles: list = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        smi = entry.get("smiles")
        if smi:
            smiles.append(smi)
            bbox = entry.get("bbox")
            if bbox is not None:
                bboxes.append(bbox)

    annotated: list[Dict] = []
    if bboxes:
        annotated_jpeg = draw_bboxes_on_image(img_bytes, {"molecules": bboxes})
        annotated.append(_upload_annotated_jpeg(annotated_jpeg, user_id, session_id))

    return {
        "answer": {label: {"smiles": smiles, "errors": errors}},
        "metadata": {
            "annotated_images": annotated,
            "source_url": image_url.strip(),
        },
    }


def extract_reactions_from_image_url(image_url: str, user_id: Optional[str] = None,
                                     session_id: Optional[str] = None) -> Dict:
    """
    Extracts reactions (reactants, products, conditions) from a single figure URL.

    Args:
        image_url (str): URL of the image to analyze.

    Returns:
        Dict: Keys "answer" (label → per-reaction structure and errors) and "metadata"
              (annotated_images, source_url).
    """
    label = _label_from_image_url(image_url)
    img_bytes = download_url_to_bytes(image_url)
    raw = chem_service.extract_reactions_from_figure(img_bytes)
    recognitions, errors = _normalize_figure_response(raw)

    reactions: list = []
    if recognitions and isinstance(recognitions[0], dict):
        reactions = recognitions[0].get("reactions", []) or []

    per_image: Dict[str, Any] = {}
    if errors is not None:
        per_image["errors"] = errors

    bboxes = {"reagents": [], "products": [], "conditions": []}
    for reaction_id, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        key = f"reaction_{reaction_id}"
        per_image[key] = {"reactants": [], "products": [], "conditions": []}
        for r in reaction.get("reactants", []) or []:
            if isinstance(r, dict) and "bbox" in r:
                bboxes["reagents"].append(r["bbox"])
            try:
                per_image[key]["reactants"].append(r["smiles"])
            except (KeyError, TypeError):
                if isinstance(r, dict):
                    per_image[key]["reactants"].append(r.get("text"))

        for p in reaction.get("products", []) or []:
            if isinstance(p, dict) and "bbox" in p:
                bboxes["products"].append(p["bbox"])
            try:
                per_image[key]["products"].append(p["smiles"])
            except (KeyError, TypeError):
                if isinstance(p, dict):
                    per_image[key]["products"].append(p.get("text"))

        for c in reaction.get("conditions", []) or []:
            if isinstance(c, dict) and "bbox" in c:
                bboxes["conditions"].append(c["bbox"])
            try:
                per_image[key]["conditions"].append(c["smiles"])
            except (KeyError, TypeError):
                if isinstance(c, dict):
                    t = c.get("text")
                    if t not in (None, [], ""):
                        per_image[key]["conditions"].append(t)

    annotated: list[Dict] = []
    if any(bboxes.values()):
        annotated_jpeg = draw_bboxes_on_image(img_bytes, bboxes)
        annotated.append(_upload_annotated_jpeg(annotated_jpeg, user_id, session_id))

    return {
        "answer": {label: per_image},
        "metadata": {
            "annotated_images": annotated,
            "source_url": image_url.strip(),
        },
    }


def extract_molecules_from_image_urls(image_urls: list[str], user_id: Optional[str] = None,
                                      session_id: Optional[str] = None) -> Dict:
    """
    Runs molecule extraction over multiple image URLs.

    Args:
        image_urls (list[str]): Non-empty URLs; blanks are skipped.

    Returns:
        Dict: "answer" is a merged mapping or an error string if all URLs failed; "metadata" includes
              annotated URLs, source URLs, and optional "failed" entries per URL.
    """
    combined: Dict[str, Any] = {}
    annotated: list[Dict] = []
    source_urls: list[str] = []
    failures: list[Dict[str, str]] = []

    urls = [u.strip() for u in image_urls if u and str(u).strip()]
    for i, url in enumerate(urls):
        try:
            one = extract_molecules_from_image_url(url, user_id, session_id)
        except Exception as e:
            failures.append({"url": url, "error": str(e)})
            continue
        for k, v in one["answer"].items():
            key = k
            if key in combined:
                key = f"{k}__{i}"
            combined[key] = v
        annotated.extend(one["metadata"].get("annotated_images", []))
        source_urls.append(one["metadata"].get("source_url", url))

    meta: Dict[str, Any] = {
        "annotated_images": annotated,
        "source_urls": source_urls,
    }
    if failures:
        meta["failed"] = failures
    if not combined and failures:
        return {"answer": "All image URLs failed molecule extraction.", "metadata": meta}
    return {"answer": combined, "metadata": meta}


def extract_reactions_from_image_urls(image_urls: list[str], user_id: Optional[str] = None,
                                      session_id: Optional[str] = None) -> Dict:
    """
    Runs reaction extraction over multiple image URLs.

    Args:
        image_urls (list[str]): Non-empty URLs; blanks are skipped.

    Returns:
        Dict: "answer" is a merged mapping or an error string if all URLs failed; "metadata" includes
              annotated URLs, source URLs, and optional "failed" entries per URL.
    """
    combined: Dict[str, Any] = {}
    annotated: list[Dict] = []
    source_urls: list[str] = []
    failures: list[Dict[str, str]] = []

    urls = [u.strip() for u in image_urls if u and str(u).strip()]
    for i, url in enumerate(urls):
        try:
            one = extract_reactions_from_image_url(url, user_id, session_id)
        except Exception as e:
            failures.append({"url": url, "error": str(e)})
            continue
        for k, v in one["answer"].items():
            key = k
            if key in combined:
                key = f"{k}__{i}"
            combined[key] = v
        annotated.extend(one["metadata"].get("annotated_images", []))
        source_urls.append(one["metadata"].get("source_url", url))

    meta: Dict[str, Any] = {
        "annotated_images": annotated,
        "source_urls": source_urls,
    }
    if failures:
        meta["failed"] = failures
    if not combined and failures:
        return {"answer": "All image URLs failed reaction extraction.", "metadata": meta}
    return {"answer": combined, "metadata": meta}