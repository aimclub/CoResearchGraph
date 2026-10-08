import io
import logging
from typing import Any

import fitz
import requests
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

BOX_COLORS = {
    "molecules": "red",
    "products": "green",
    "reagents": "blue",
    "conditions": "orange",
}
DEFAULT_BOX_COLOR = "gray"


def download_url_to_bytes(url: str, timeout: int = 120) -> bytes:
    """
    Downloads binary content (e.g. an image) from an HTTP(S) URL.

    Args:
        url (str): Full URL to fetch.
        timeout (int, optional): Request timeout in seconds. Defaults to 120.

    Returns:
        bytes: Response body from the server.
        
    """
    url = url.strip()
    if not url:
        raise ValueError("Empty image URL")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def draw_bboxes_on_image(image: bytes, bboxes: dict[str, Any]) -> bytes:
    """
    Draws axis-aligned bounding boxes on an image and returns a JPEG.

    Args:
        image (bytes): Raw image bytes, or a fitz.Pixmap (converted internally).
        bboxes (dict[str, Any]): Map of category name to a list of boxes; each box is
            [x1, y1, x2, y2] in normalized coordinates (0..1 relative to width/height).

    Returns:
        bytes: JPEG-encoded image with outlines.
    """
    if isinstance(image, fitz.Pixmap):
        image = image.tobytes("ppm")
    img = Image.open(io.BytesIO(image))
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for key, boxes in bboxes.items():
        color = BOX_COLORS.get(key, DEFAULT_BOX_COLOR)
        for bbox in boxes if isinstance(boxes, list) else [boxes]:
            x1 = bbox[0] * w
            y1 = bbox[1] * h
            x2 = bbox[2] * w
            y2 = bbox[3] * h
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    output = io.BytesIO()
    img.save(output, format="JPEG", quality=95)
    return output.getvalue()
