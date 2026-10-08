import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import boto3
from botocore.client import Config, ClientError
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class S3ETLArtifactStore:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
    ):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
        )

    @staticmethod
    def _prefix(article_id: str, step: str) -> str:
        return f"articles/{article_id}/{step}/"

    def step_exists(self, article_id: str, step: str) -> bool:
        resp = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self._prefix(article_id, step),
            MaxKeys=1,
        )
        return "Contents" in resp

    # ---------- HTML ----------

    def put_html(self, article_id: str, step: str, html: str) -> None:
        key = self._prefix(article_id, step) + "document.html"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=html.encode("utf-8"),
            ContentType="text/html",
        )

    def get_html(self, article_id: str, step: str) -> str | None:
        key = self._prefix(article_id, step) + "document.html"
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            logger.error(f"Error getting HTML for article {article_id}: {e}")
            return None
        return obj["Body"].read().decode("utf-8")

    # ---------- Images ----------

    def put_images(
        self,
        article_id: str,
        step: str,
        images: dict[str, Image.Image],
    ) -> None:
        for name, img in images.items():
            buf = BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            key = self._prefix(article_id, step) + f"images/{name}"
            self.client.upload_fileobj(buf, self.bucket, key)

    def list_images(self, article_id: str, step: str) -> list[str]:
        prefix = self._prefix(article_id, step) + "images/"
        resp = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
        )
        if "Contents" not in resp:
            return []
        return [Path(o["Key"]).name for o in resp["Contents"]]

    def get_image(
        self,
        article_id: str,
        step: str,
        image_name: str,
    ) -> Image.Image | None:
        key = self._prefix(article_id, step) + f"images/{image_name}"
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            logger.error(
                f"Error getting image {image_name} for article {article_id} on {step} step: {e}"
            )
            return None
        return Image.open(BytesIO(obj["Body"].read()))

    # ---------- Metadata ----------

    def put_metadata(
        self,
        article_id: str,
        step: str,
        metadata: dict[str, Any],
    ) -> None:
        key = self._prefix(article_id, step) + "meta.json"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(metadata).encode("utf-8"),
            ContentType="application/json",
        )

    def get_metadata(self, article_id: str, step: str) -> dict[str, Any] | None:
        key = self._prefix(article_id, step) + "meta.json"
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            logger.error(f"Error getting metadata for article {article_id} on {step} step: {e}")
            return None
        return json.loads(obj["Body"].read().decode("utf-8"))
    
    # ---------- Files ----------
    
    def put_file(self, article_id: str, step: str, file_name: str, data: bytes) -> None:
        key = self._prefix(article_id, step) + file_name
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
    
    def get_file(self, article_id: str, step: str, file_name: str) -> bytes | None:
        key = self._prefix(article_id, step) + file_name
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            logger.error(f"Error getting file {file_name} for article {article_id} on {step} step: {e}")
            return None
        return resp["Body"].read()

    # ---------- Delete ----------

    def delete_step(self, article_id: str, step: str) -> None:
        prefix = self._prefix(article_id, step)
        resp = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
        )
        if "Contents" not in resp:
            return
        for obj in resp["Contents"]:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=obj["Key"],
            )

    def delete_article(self, article_id: str) -> None:
        prefix = f"articles/{article_id}/"
        resp = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
        )
        if "Contents" not in resp:
            return
        for obj in resp["Contents"]:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=obj["Key"],
            )
    
    def delete_file(self, article_id: str, step: str, filename: str) -> None:
        key = self._prefix(article_id, step) + filename
        self.client.delete_object(
            Bucket=self.bucket,
            Key=key,
        )
