import base64
from io import BytesIO
from typing import BinaryIO
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from PIL import Image

from CoScientist.paper_parser.s3_connection import s3_service


def convert_to_base64(file_path):
    """
    Convert an image file to a Base64 encoded string.
    
    This method reads an image from the specified file path, encodes it as a JPEG image in memory, then converts it
    into a Base64 string representation.
    
    Args:
        file_path (str): The path to the image file.
    
    Returns:
        str: A Base64 encoded string representing the JPEG image.
    """
    if file_path.startswith("http://"):
        s3_key, bucket_name = extract_s3_bucket_and_key(file_path)
        pil_image = Image.open(BytesIO(s3_service.get_image_bytes_from_s3(s3_key, bucket_name)))
    else:
        pil_image = Image.open(file_path)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def load_image_as_binary(file_path: str) -> BytesIO:
    """
    Loads an image from a local path or S3 and returns a BytesIO object.
    
    Args:
        file_path (str): The path to the image file.
    Returns:
        BytesIO: A BytesIO object representing the image.
    """
    if file_path.startswith("http://") or file_path.startswith("https://"):
        s3_key, bucket_name = extract_s3_bucket_and_key(file_path)
        pil_image = Image.open(BytesIO(s3_service.get_image_bytes_from_s3(s3_key, bucket_name)))
        pil_image.save("tmp.jpeg", format="JPEG")
    else:
        pil_image = Image.open(file_path)
        pil_image.save("tmp.jpeg", format="JPEG")

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")

    return buffered.getvalue()


def prompt_func(data):
    """
    Creates a structured message containing text and images for use in a conversational context.
    
    This method prepares the input data into a format suitable for presenting information in a multi-modal interface,
    by converting images to data URIs that can be directly embedded in a message and combining them with the provided
    text.
    
    Args:
        data (dict): A dictionary containing the message content:
            - "text" (str): The text content of the message;
            - "image" (list): A list of base64 encoded JPEG images to include in the message.
    
    Returns:
        HumanMessage: A HumanMessage object with a structured 'content' list.
            The 'content' list contains dictionaries representing each part of the message,
            with "type" keys indicating whether it's "text" or "image_url". Image URLs
            are formatted as data URIs.
    """
    text = data["text"]
    imgs = data["image"]
    content_parts = []

    for img in imgs:
        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{img}",
        }
        # image_part = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        content_parts.append(image_part)

    text_part = {"type": "text", "text": text}
    content_parts.append(text_part)

    return HumanMessage(content=content_parts)


def extract_s3_bucket_and_key(s3_url: str):
    """
    Extracts the file key in S3 storage and the bucket name from the full file path.
    
    Args:
        s3_url: The full path to the file in S3 storage.

    Returns:
        A tuple of S3 key and bucket name.
    """
    o = urlparse(s3_url)
    bucket, key = o.path.split('/', 2)[1:]
    return key, bucket
