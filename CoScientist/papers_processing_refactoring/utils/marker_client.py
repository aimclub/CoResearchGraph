import base64
import shutil
import tempfile
import time
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from bs4 import BeautifulSoup
import requests
from PIL import Image
from pypdf import PdfReader, PdfWriter


def load_bytes(uri: str) -> bytes:
    """If uri is local path, read the file, if uri is http url, download it and return bytes
    
    Args
        uri: local path or http url
    Returns
        content of the file
    """
    if uri.startswith("http://") or uri.startswith("https://"):
        response = requests.get(uri)
        content = response.content
    else:
        content = Path(uri).read_bytes()
    return content


@dataclass
class ConvertedContent:
    images: dict[str, Image.Image]
    text: str


class MarkerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def convert(self, pdf_uri: str) -> ConvertedContent:
        content = load_bytes(pdf_uri)
        files = {"pdf_file": ("file.pdf", content, "application/pdf")}
        response = requests.post(self.base_url, files=files)

        res = response.json()
        text, images, elapsed_time = res["html"], res["images"], res["processing_time"]

        decoded_images = {}
        for name, image in images.items():
            bio = BytesIO()
            bio.write(base64.b64decode(image))
            pil_image = Image.open(bio)
            decoded_images[name] = pil_image
        return ConvertedContent(images=decoded_images, text=text)


def split_pdf(pdf_uri: str, pages_per_part: int) -> list[tuple[Path, int]]:
    """Разбивает PDF на части.

    Returns:
        Список кортежей (путь_к_файлу_части, смещение_страниц_от_начала).
    """
    content = load_bytes(pdf_uri)
    reader = PdfReader(BytesIO(content))
    total_pages = len(reader.pages)
    
    temp_dir = Path(tempfile.mkdtemp(prefix="marker_split_"))
    
    if total_pages <= pages_per_part:
        single_path = temp_dir / "full.pdf"
        single_path.write_bytes(content)
        return [(single_path, 0)]
    
    parts = []
    for start in range(0, total_pages, pages_per_part):
        writer = PdfWriter()
        end = min(start + pages_per_part, total_pages)
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])
        
        part_path = temp_dir / f"part_{start + 1}_{end}.pdf"
        with open(part_path, "wb") as f:
            writer.write(f)
        parts.append((part_path, start))
    
    return parts


def adjust_page_numbers(
        html: str, images: dict[str, Image.Image], page_offset: int
) -> tuple[str, dict[str, Image.Image]]:
    """Смещает номера страниц в названиях изображений и в HTML-тегах на заданный offset."""
    if page_offset == 0:
        return html, images

    pattern = re.compile(r"_page_(\d+)_")
    
    def replace_offset(match: re.Match) -> str:
        current_page = int(match.group(1))
        return f"_page_{current_page + page_offset}_"
    
    updated_html = pattern.sub(replace_offset, html)
    
    updated_images = {}
    for img_name, img_obj in images.items():
        new_name = pattern.sub(replace_offset, img_name)
        updated_images[new_name] = img_obj
    
    return updated_html, updated_images


def combine_html(html_parts: list[str]) -> str:
    """Объединяет несколько HTML-документов в один, склеивая содержимое тегов <body>."""
    if not html_parts:
        return ""
    if len(html_parts) == 1:
        return html_parts[0]
    
    base_soup = BeautifulSoup(html_parts[0], "html.parser")
    base_body = base_soup.find("body")
    
    # TODO: check whether there is any data loss if the chunk does not contain a body
    if not base_body:
        return html_parts[0]
    
    for html in html_parts[1:]:
        part_soup = BeautifulSoup(html, "html.parser")
        part_body = part_soup.find("body")
        if part_body:
            base_body.extend(part_body.contents)
    
    return str(base_soup)


def convert_pdf_with_splitting(
        client: MarkerClient,
        pdf_uri: str,
        pages_per_part: int = 10,
) -> ConvertedContent:
    """Обрабатывает большой PDF по частям и собирает результат в единый ConvertedContent."""
    part_entries = split_pdf(pdf_uri, pages_per_part)
    
    all_html_parts = []
    merged_images = {}
    
    try:
        for idx, (part_path, page_offset) in enumerate(part_entries):
            res = client.convert(str(part_path))
            html, images = adjust_page_numbers(res.text, res.images, page_offset)
            all_html_parts.append(html)
            merged_images.update(images)
            time.sleep(5)
        
        final_html = combine_html(all_html_parts)
        return ConvertedContent(images=merged_images, text=final_html)
    
    finally:
        if part_entries:
            shutil.rmtree(part_entries[0][0].parent, ignore_errors=True)
    