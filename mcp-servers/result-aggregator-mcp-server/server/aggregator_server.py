import os
import logging
import urllib.parse
import shutil
import uuid
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResultAggregatorServer")

mcp = FastMCP("ResultAggregator")

def url_to_filename(url: str, default_ext: str = ".png") -> str:
    path_str = urllib.parse.urlparse(url).path
    filename = os.path.basename(path_str)
    if not filename or "." not in filename:
        filename = f"{uuid.uuid4().hex}{default_ext}"
    else:
        # Strip query parameters if any remained in basename
        filename = filename.split("?")[0]
    return filename

@mcp.tool()
async def format_results(
    session_id: str,
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Scans the local workspace for run results, downloads remote artifacts (images/CSVs),
    copies them to the local logs/reproduction/media/ folder for offline rendering,
    and returns formatted Markdown tables, image embeds, and a combined summary.
    """
    logger.info(f"Formatting results for session: {session_id}")
    
    media_dir = Path("logs/reproduction/media")
    media_dir.mkdir(parents=True, exist_ok=True)
    
    markdown_sections = []
    copied_files = []
    
    # 1. Download/Copy remote artifacts
    if artifacts:
        logger.info(f"Processing {len(artifacts)} artifacts...")
        for art in artifacts:
            url = art.get("url")
            if not url:
                continue
                
            # Determine if it's an image or CSV
            url_lower = url.lower()
            is_image = any(url_lower.endswith(ext) or f"{ext}?" in url_lower or f"{ext}&" in url_lower for ext in [".png", ".jpg", ".jpeg", ".svg"])
            is_csv = any(url_lower.endswith(ext) or f"{ext}?" in url_lower or f"{ext}&" in url_lower for ext in [".csv"]) or "csv" in url_lower
            
            filename = url_to_filename(url, default_ext=".png" if is_image else ".csv")
            dest_filename = f"{session_id}_{filename}"
            dest_path = media_dir / dest_filename
            
            logger.info(f"Downloading {url} to {dest_path}...")
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                dest_path.write_bytes(resp.content)
                copied_files.append(str(dest_path))
                
                # Format block
                if is_image:
                    section = f"### Figure ({art.get('tool', 'Plot')})\n\n"
                    section += f"![{art.get('tool', 'Plot')}](media/{dest_filename})\n"
                    markdown_sections.append(section)
                elif is_csv:
                    # Parse CSV and format table
                    try:
                        df = pd.read_csv(dest_path)
                        preview_df = df.head(15)
                        table_md = preview_df.to_markdown(index=False)
                        
                        section = f"### Data Table ({art.get('tool', 'Table')}) — [Download Full CSV](media/{dest_filename})\n\n"
                        section += f"{table_md}\n"
                        if len(df) > 15:
                            section += f"\n*... showing first 15 of {len(df)} rows.*\n"
                        markdown_sections.append(section)
                    except Exception as e:
                        logger.error(f"Error parsing CSV {dest_filename}: {e}")
                        markdown_sections.append(f"### Data ({art.get('tool', 'Table')})\n\n[Download CSV](media/{dest_filename})\n")
            except Exception as e:
                logger.error(f"Failed to download artifact {url}: {e}")
                
    # 2. Scan local workspace
    workspace_dir = Path("workspace") / f"ws_{session_id}"
    if workspace_dir.exists():
        logger.info(f"Scanning local workspace {workspace_dir}...")
        # Search recursively
        for root, _, files in os.walk(workspace_dir):
            for file in files:
                file_path = Path(root) / file
                file_lower = file.lower()
                dest_filename = f"{session_id}_{file}"
                dest_path = media_dir / dest_filename
                
                if any(file_lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".svg"]):
                    logger.info(f"Copying workspace image {file_path} to {dest_path}...")
                    try:
                        shutil.copy2(file_path, dest_path)
                        copied_files.append(str(dest_path))
                        section = f"### Figure ({file_path.stem})\n\n"
                        section += f"![{file_path.stem}](media/{dest_filename})\n"
                        markdown_sections.append(section)
                    except Exception as e:
                        logger.error(f"Error copying local image {file_path}: {e}")
                        
                elif file_lower.endswith(".csv"):
                    logger.info(f"Parsing workspace CSV {file_path}...")
                    try:
                        shutil.copy2(file_path, dest_path)
                        copied_files.append(str(dest_path))
                        df = pd.read_csv(file_path)
                        preview_df = df.head(15)
                        table_md = preview_df.to_markdown(index=False)
                        
                        section = f"### Data Table ({file_path.stem}) — [Download Full CSV](media/{dest_filename})\n\n"
                        section += f"{table_md}\n"
                        if len(df) > 15:
                            section += f"\n*... showing first 15 of {len(df)} rows.*\n"
                        markdown_sections.append(section)
                    except Exception as e:
                        logger.error(f"Error reading local CSV {file_path}: {e}")
                        
    combined_md = "\n\n".join(markdown_sections)
    
    return {
        "formatted_markdown": combined_md,
        "copied_files": copied_files,
        "status": "success",
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7331)
    parser.add_argument("--path", default="/mcp")
    args = parser.parse_args()
    mcp.run(transport="http", host=args.host, port=args.port, path=args.path)

if __name__ == "__main__":
    main()
