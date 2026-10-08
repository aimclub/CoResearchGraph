import os
from io import BytesIO
from pathlib import Path

import boto3
from botocore.client import Config

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Bounds for one S3 call. The defaults are 60 s to connect, 60 s to read, and up
# to 5 attempts, so an endpoint that drops packets blocks the caller for about
# 300 s. An MCP tool runs inside that call, and the ADK client gives up at 300 s.
# The client then closes the stream, and the server logs a ClosedResourceError
# for a response nobody waits for. Fail in seconds instead, so the tool returns
# an error the agent can read.
_CONNECT_TIMEOUT_SECONDS = 5
_READ_TIMEOUT_SECONDS = 30
_MAX_ATTEMPTS = 3

_CLIENT_CONFIG = Config(
    signature_version="s3v4",
    connect_timeout=_CONNECT_TIMEOUT_SECONDS,
    read_timeout=_READ_TIMEOUT_SECONDS,
    retries={"max_attempts": _MAX_ATTEMPTS, "mode": "standard"},
)


class S3BucketService:
    """
    A class for working with S3 compatible storage.
    
        The class provides methods for loading, viewing, and deleting objects within a specific bucket.
    
        Attributes:
            - endpoint: URL of S3 compatible service
            - access_key: user's login
            - secret_key: user's password
            - bucket_name: bucket names for work
    """
    
    def __init__(
            self,
            endpoint: str,
            access_key: str,
            secret_key: str,
            bucket_name: str = "default",
    ) -> None:
        """
        Initializes the class based on the definition of the fields required for operation.
        
        Args:
            endpoint: URL of S3 compatible service
            access_key: user's login
            secret_key: user's password
            bucket_name: bucket names for work
        """
        self.bucket_name = bucket_name
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key

    def create_s3_client(self) -> boto3.client:
        """
        Creates a client for working with S3 compatible storage based on the service URL, login, and password.
        
        Returns:
            Boto3 a client for working with S3 compatible storage.
        """
        client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=_CLIENT_CONFIG,
        )
        return client
    
    def upload_file_object(
            self,
            prefix: str,
            source_file_name: str,
            file_path: str,
    ) -> None:
        """
        Uploads a file to S3 bucket with specified prefix and source file name.

        Args:
            prefix: The prefix/folder path in the S3 bucket where the file will be stored
            source_file_name: The name of the file to be stored in S3
            file_path: Local path to the file to be uploaded

        Returns:
            None
        """
        client = self.create_s3_client()
        destination_path = (Path(prefix, source_file_name)).as_posix()
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        buffer = BytesIO(content)
        client.upload_fileobj(buffer, self.bucket_name, destination_path)

    def upload_bytes(self, prefix: str, source_file_name: str, data: bytes) -> str:
        """
        Uploads bytes to S3 bucket with specified prefix and file name.

        Args:
            prefix: The prefix/folder path in the S3 bucket where the file will be stored
            source_file_name: The name of the file to be stored in S3
            data: Raw bytes to upload

        Returns:
            S3 key (path) of the uploaded object, for use with generate_presigned_url
        """
        client = self.create_s3_client()
        destination_path = (Path(prefix, source_file_name)).as_posix()
        buffer = BytesIO(data)
        client.upload_fileobj(buffer, self.bucket_name, destination_path)
        return destination_path

    def list_objects(self, prefix: str) -> list[str]:
        """
        Lists all objects in the S3 bucket with the given prefix.

        Args:
            prefix: The prefix to filter objects in the S3 bucket. If empty, lists all objects.

        Returns:
            A list of object keys (file paths) that match the prefix
        """
        client = self.create_s3_client()
        
        paginator = client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
        
        storage_content: list[str] = []
        
        for page in page_iterator:
            contents = page.get("Contents", [])
            for item in contents:
                storage_content.append(item["Key"])
        
        return storage_content
    
    def delete_file_object(self, prefix: str, source_file_name: str) -> None:
        """
        Deletes a single file from S3 bucket.

        Args:
            prefix: The prefix/folder path where the file is located
            source_file_name: The name of the file to be deleted

        Returns:
            None
        """
        client = self.create_s3_client()
        path_to_file = (Path(prefix, source_file_name)).as_posix()
        client.delete_object(Bucket=self.bucket_name, Key=path_to_file)
    
    def create_new_bucket(self, bucket_name: str) -> None:
        """
        Creates a new S3 bucket.

        Args:
            bucket_name: The name of the bucket to be created

        Returns:
            None
        """
        client = self.create_s3_client()
        try:
            client.create_bucket(Bucket=bucket_name)
        except Exception as e:
            print(e)
    
    def del_bucket(self, bucket_name: str) -> None:
        """
        Deletes an S3 bucket.

        Args:
            bucket_name: The name of the bucket to be deleted

        Returns:
            None
        """
        client = self.create_s3_client()
        try:
            client.delete_bucket(Bucket=bucket_name)
        except Exception as e:
            print(e)
    
    def generate_presigned_url(self, s3_key: str, method: str = 'get_object', expiration: int = 360) -> str:
        """
        Generates a presigned URL for accessing an S3 object.

        Args:
            s3_key: The key (path) of the S3 object
            method: HTTP method for the presigned URL (default: 'get_object')
            expiration: Time in seconds for the URL to remain valid (default: 360)

        Returns:
            A presigned URL string for accessing the S3 object
        """
        client = self.create_s3_client()
        return client.generate_presigned_url(
            method,
            Params={'Bucket': self.bucket_name, 'Key': s3_key},
            ExpiresIn=expiration
        )
    
    def download_image_from_s3(self, s3_key: str, local_path: str) -> None:
        """
        Downloads an image file from S3 to a local path.

        Args:
            s3_key: The key (path) of the image file in S3
            local_path: Local file path where the image will be saved

        Returns:
            None
        """
        client = self.create_s3_client()
        client.download_file(self.bucket_name, s3_key, local_path)
    
    def get_image_bytes_from_s3(self, s3_key: str, bucket_name: str) -> bytes:
        """
        Retrieves image content as bytes from S3.

        Args:
            s3_key: The key (path) of the image file in S3
            bucket_name: The name of the S3 bucket where the file is located

        Returns:
            The content of the image file as bytes
        """
        client = self.create_s3_client()
        response = client.get_object(Bucket=bucket_name, Key=s3_key)
        return response['Body'].read()
    
    def clean_up_by_prefix(self, prefix_to_delete: str):
        """
        Deletes all files related to one article from S3.

        Args:
            prefix_to_delete: A prefix (basically paper file name) to find and delete from storage

        Returns:
            None
        """
        client = self.create_s3_client()
        response = client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix_to_delete)
        try:
            contents = response["Contents"]
        except KeyError:
            print(f"No files found with prefix: {prefix_to_delete}")
            return
        keys_to_delete = [item["Key"] for item in contents]
        deleted_count = 0
        for file_key in keys_to_delete:
            try:
                client.delete_object(Bucket=self.bucket_name, Key=file_key)
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {file_key}: {str(e)}")
        print(f"Deleted {deleted_count} out of {len(keys_to_delete)} file(s) with prefix: {prefix_to_delete}")


s3_service = S3BucketService(
    endpoint=os.getenv("S3__ENDPOINT_URL"),
    access_key=os.getenv("S3__ACCESS_KEY"),
    secret_key=os.getenv("S3__SECRET_KEY"),
    bucket_name=os.getenv("S3__BUCKET_NAME")
)

if __name__ == "__main__":
    s3_client = s3_service.create_s3_client()
    buckets = s3_client.list_buckets()
    # for bucket in buckets["Buckets"]:
    #     print(bucket["Name"], bucket["CreationDate"])
    # objects = s3_service.list_objects(prefix="")
    # print(objects)