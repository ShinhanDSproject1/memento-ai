import boto3
import os
import uuid
from urllib.parse import quote


def upload_certification_image(png_bytes: bytes, ext: str = "png") -> str:
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")
    AWS_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
    AWS_UPLOAD_DIR = os.getenv("AWS_S3_UPLOAD_DIR", "certification")

    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    file_name = f"{uuid.uuid4()}.{ext}"
    key = f"{AWS_UPLOAD_DIR}/{file_name}"

    s3_client.put_object(
        Bucket=AWS_BUCKET_NAME,
        Key=key,
        Body=png_bytes,
        ContentType="image/png",
    )

    url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{quote(key)}"
    return url
