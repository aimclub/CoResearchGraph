import json
from io import BytesIO
from typing import Any

import boto3
from botocore.client import Config
from PIL import Image


# TODO: maybe create separate user for this bucket
class S3DomainArtifactStore:
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
    def _article_prefix(domain: str,article_id: str) -> str:
        return f"articles/{domain}/{article_id}/"

    def publish_article(
        self,
        domain: str,
        article_id: str,
        html: str,
        pdf_data: bytes,
        images: dict[str, Image.Image | None],
        metadata: dict[str, Any] | None = None,
        paper_summary: str | None = None,
    ) -> None:
        # Summary publication is disabled for the current ingestion pipeline.
        # Keep this optional branch for compatibility with explicit legacy callers.
        if paper_summary is not None:
            self.client.put_object(
                Bucket=self.bucket,
                Key=self._article_prefix(domain, article_id) + "summary.txt",
                Body=paper_summary.encode("utf-8"),
                ContentType="text/plain",
            )
        
        # HTML
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._article_prefix(domain, article_id) + "article.html",
            Body=html.encode("utf-8"),
            ContentType="text/html",
        )
        
        # Paper file
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._article_prefix(domain, article_id) + "paper.pdf",
            Body=pdf_data,
            ContentType="application/pdf",
        )

        # Images
        if any(list(images.values())):
            for name, img in images.items():
                if img:
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    key = self._article_prefix(domain, article_id) + f"images/{name}"
                    self.client.upload_fileobj(buf, self.bucket, key)

        # Metadata (optional)
        if metadata is not None:
            self.client.put_object(
                Bucket=self.bucket,
                Key=self._article_prefix(domain, article_id) + "meta.json",
                Body=json.dumps(metadata).encode("utf-8"),
                ContentType="application/json",
            )

    def get_article_html(self, domain: str, article_id: str) -> str:
        key = self._article_prefix(domain, article_id) + "article.html"
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read().decode("utf-8")

    def get_image_url(
        self,
        domain: str,
        article_id: str,
        image_name: str,
    ) -> str:
        key = self._article_prefix(domain, article_id) + f"images/{image_name}"
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=3600,
        )
    
    def download_image_from_s3(self, domain: str, article_id: str, image_name, tmp_path: str) -> None:
        key = self._article_prefix(domain, article_id) + f"images/{image_name}"
        return self.client.download_file(
            Bucket=self.bucket,
            Key=key,
            Filename=tmp_path
        )
    
    def get_image_bytes_from_s3(self, domain: str, article_id: str, image_name, ) -> bytes:
        key = self._article_prefix(domain, article_id) + f"images/{image_name}"
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response['Body'].read()

    def delete_article(self, domain: str, article_id: str) -> None:
        prefix = self._article_prefix(domain, article_id)
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
