import base64
from contextlib import asynccontextmanager
import io
import logging
import os
import tempfile
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from PIL import Image

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

converter: PdfConverter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global converter
    logger.info("Models loading...")
    
    config = {
        "output_format": "html",
        # "force_ocr": False,
        # "extract_images": True,
    }
    config_parser = ConfigParser(config)
    
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )
    
    logger.info("Models are loaded.")
    yield
    
    logger.info("Shutting down...")
    del converter


app = FastAPI(
    title="Marker PDF API",
    description="API for parsing PDF to HTML with image extraction",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,  # noqa
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_base64(image: Image.Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Marker PDF API is running"}


@app.post("/convert")
async def convert_pdf(pdf_file: UploadFile = File(...)):
    """Parse PDF file to HTML with image extraction

    Args
        - pdf_file: PDF file to parse

    Returns
        JSON response with the following fields:
            - html: parsed PDF content in HTML
            - images: base64 encoded images
            - processing_time: time of parsing
    """
    start_time = time.time()
    
    try:
        logger.info(f"Received file: {pdf_file.filename}")
        pdf_content = await pdf_file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_content)
            tmp_path = tmp_file.name
        
        try:
            logger.info(f"Parsing started: {pdf_file.filename}")
            rendered = converter(tmp_path)  # noqa
            text, _, images = text_from_rendered(rendered)
            logger.info(f"Parsing finished: {pdf_file.filename}")

            images_base64 = {}
            for img_name, img in images.items():
                if isinstance(img, Image.Image):
                    images_base64[img_name] = image_to_base64(img)
            
            processing_time = time.time() - start_time
            
            return JSONResponse(content={
                "html": text,
                "images": images_base64,
                "processing_time": processing_time
            })
        
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Parsing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Parsing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1
    )