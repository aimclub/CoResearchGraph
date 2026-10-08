import os, base64, uuid
import logging
import pandas as pd
import fitz
from io import BytesIO, StringIO
from pathlib import Path
from PIL import Image
from langchain_core.messages import SystemMessage, HumanMessage
from protollm.connectors import create_llm_connector
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

from CoScientist.chemical_utils.chemical_functions import extract_molecules_from_figure
from CoScientist.chemical_utils.ocr_pipeline import render_molecule_detections
from CoScientist.paper_parser.s3_connection import S3BucketService

from prompt import extract_mol_properties_prompt

IMG_STORAGE_PATH = os.getenv("IMG_STORAGE_PATH")
DATASETS_LLM_URL = os.getenv("DATASETS_LLM_URL")

def _s3_env(primary: str, fallback: str) -> str | None:
    """S3__* is the name every server and the main app use. The bare names stay
    as a fallback for one release, so an existing deployment keeps working."""
    return os.getenv(primary) or os.getenv(fallback)


s3_service = S3BucketService(
    endpoint=_s3_env("S3__ENDPOINT_URL", "ENDPOINT_URL"),
    access_key=_s3_env("S3__ACCESS_KEY", "ACCESS_KEY"),
    secret_key=_s3_env("S3__SECRET_KEY", "SECRET_KEY"),
    bucket_name=_s3_env("S3__BUCKET_NAME", "BUCKET_NAME"),
)

mcp = FastMCP("DatasetCollection")

def convert_pdf_pages_to_images(pdf_bytes: bytes) -> list:
    """Converts PDF bytes into fitz.Pixmap images."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=500)
        images.append(pix)
    return images
        
def extract_smiles_from_images(images: list) -> list:
    """Uses OpenChemIE tool to extract molecular SMILES and ID from PDF screenshots."""
    results = []
    for img in images:
        pil_image = Image.open(BytesIO(img.tobytes("png")))
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        image_file = buffered.getvalue()
        res = extract_molecules_from_figure(image=image_file)
        results.append(res.get("data", []))
    return results

def mols_to_csv(results):
    """"Formats OpenChemIE JSONs with molecular SMILES and IDs into pandas DataFrame."""
    df = pd.DataFrame()
    
    all_refs = []
    all_smiles = []
    for res in results:
        res = res[0]
        corefs = res["corefs"]
        mols_idxs = [i[0] for i in corefs]
        refs_idxs = [i[1] for i in corefs]
        smiles = [res["bboxes"][i]["smiles"] for i in mols_idxs]
        refs = [res["bboxes"][i]["text"] for i in refs_idxs]
        for i in range(len(corefs)):
            ref = ";".join(refs[i]) if refs[i] != [] else "Unknown ID"
            all_refs.append(ref)
            all_smiles.append(smiles[i])

    df["id"] = all_refs
    df["smiles"] = all_smiles

    return df

def extract_props(model_url: str, question: str, pdfs: list) -> dict:
    """
    Queries a language model with a question and a list of PDF documents to collect a dataset of
    molecules and their properties from scientific papers.

    pdfs: list of (filename, bytes) tuples.
    """

    llm = create_llm_connector(model_url)

    content = []

    for filename, pdf_bytes in pdfs:
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        paper_part = {
            "type": "file",
            "file": {
                "filename": filename,
                "file_data": f"data:application/pdf;base64,{base64_pdf}",
            },
        }
        content.append(paper_part)

    text_part = {"type": "text", "text": f"USER QUESTION: {question}"}
    content.append(text_part)

    messages = [
        SystemMessage(content=extract_mol_properties_prompt),
        HumanMessage(content=content)
    ]

    res = llm.invoke(messages)
    clean_text = res.content.split("```csv")[-1].split("```")[0].strip()
    df = pd.read_csv(StringIO(clean_text))
    return df

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorders DataFrame columns into a consistent layout."""
    first_cols = ['id', 'smiles']
    last_cols = ['units', 'source']

    existing_first = [col for col in first_cols if col in df.columns]
    existing_last = [col for col in last_cols if col in df.columns]
    middle_cols = [col for col in df.columns if col not in existing_first + existing_last]

    new_order = existing_first + middle_cols + existing_last

    return df[new_order]


@mcp.tool()
def extract_mols_prop_dataset(
    model_url: str,
    question: str,
    s3_keys: list,
    session_id: str,
    user_id: str
) -> dict:
    """
    Extracts a dataset with molecular SMILES and properties from PDF documents
    by calling the OpenChemIE tool and quering a language model. It returns the resulting dataset
    along with the original PDF pages, annotated with bounding boxes highlighting the detected
    molecular structures.

    Args:
        model_url (str): The URL of the language model to use for querying.
        question (str): The question to ask the language model.
        s3_keys (list): A list of S3 keys pointing to PDF documents.
        session_id (str): Session ID.
        user_id (str): User ID.
    Returns:
        dict: ``answer`` describes the dataset. ``metadata.dataset`` and each entry
        of ``metadata.annotated_images`` carry ``bucket``, ``s3_key`` and
        ``presigned_url``. The bucket and the key are the durable reference. The
        URL expires in one hour.
    """
    run_id = str(uuid.uuid4())
    s3_client = s3_service.create_s3_client()
    # The ephemeral/ top segment is what the bucket lifecycle rule filters on.
    # Objects written at the old prefix matched no rule and never expired.
    images_prefix = f"ephemeral/{user_id}/{session_id}/dataset_collection/annotated_images/{run_id}"
    annotated_images = []

    all_datasets = []
    for s3_key in s3_keys:
        try:
            pdf_bytes = s3_client.get_object(Bucket=s3_service.bucket_name, Key=s3_key)["Body"].read()
            images = convert_pdf_pages_to_images(pdf_bytes)
            results = extract_smiles_from_images(images)
            rendered_files = render_molecule_detections(images, results)
            for file_name, file_bytes in rendered_files:
                image_key = f"{images_prefix}/{Path(s3_key).stem}/{file_name}"
                s3_client.upload_fileobj(BytesIO(file_bytes), s3_service.bucket_name, image_key)
                annotated_images.append({
                    "bucket": s3_service.bucket_name,
                    "s3_key": image_key,
                    "presigned_url": s3_service.generate_presigned_url(image_key, expiration=3600),
                })
            mols_df = mols_to_csv(results)
            mols_df['id'] = mols_df['id'].astype(str)
            props_df = extract_props(model_url, question, [(Path(s3_key).name, pdf_bytes)])
            props_df['id'] = props_df['id'].astype(str)
            merged_df = pd.merge(props_df, mols_df, on="id", how="inner")
            merged_df["source"] = s3_key
            all_datasets.append(merged_df)
        except Exception as e:
            logger.error(f"Error processing PDF {s3_key}: {e}")
            logger.info("Skipping this PDF and continuing with others...")
            continue
    
    if not all_datasets:
        raise ValueError("No PDFs were successfully processed")

    combined_dataset = pd.concat(all_datasets, ignore_index=True)
    final_dataset = reorder_columns(combined_dataset)
    csv_buffer = StringIO()
    final_dataset.to_csv(csv_buffer, sep="\t", index=False)

    s3_key = f"ephemeral/{user_id}/{session_id}/dataset_collection/final_dataset_{run_id}.csv"
    s3_client.upload_fileobj(BytesIO(csv_buffer.getvalue().encode("utf-8")), s3_service.bucket_name, s3_key)

    answer = f"Dataset extracted with {len(final_dataset)} molecules and properties"
    return {
        "answer": answer,
        "metadata": {
            "dataset": {
                "bucket": s3_service.bucket_name,
                "s3_key": s3_key,
                "presigned_url": s3_service.generate_presigned_url(s3_key, expiration=3600),
            },
            "annotated_images": annotated_images,
        },
    }

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=7331, path="/mcp")