import argparse
import os

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


def env(name, default=""):
    return os.environ.get(name, default).strip()


def main():
    parser = argparse.ArgumentParser(description="Upload a file to Cloudflare R2.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    endpoint_url = env("R2_ENDPOINT_URL")
    access_key = env("R2_ACCESS_KEY_ID")
    secret_key = env("R2_SECRET_ACCESS_KEY")
    if not endpoint_url or not access_key or not secret_key:
        raise RuntimeError("R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY are required.")

    total_bytes = os.path.getsize(args.file)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 10, "mode": "standard"},
        ),
    )
    transfer_config = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=4,
        use_threads=True,
    )

    transferred = 0
    next_log = 512 * 1024 * 1024

    def progress(bytes_amount):
        nonlocal transferred, next_log
        transferred += bytes_amount
        if transferred >= next_log or transferred == total_bytes:
            print(
                f"[upload] {transferred / (1024 ** 3):.2f} / "
                f"{total_bytes / (1024 ** 3):.2f} GiB"
            )
            next_log += 512 * 1024 * 1024

    print(f"[upload] Uploading {args.file} to s3://{args.bucket}/{args.key}")
    client.upload_file(
        args.file,
        args.bucket,
        args.key,
        Callback=progress,
        Config=transfer_config,
    )
    head = client.head_object(Bucket=args.bucket, Key=args.key)
    size = int(head.get("ContentLength", 0))
    if size != total_bytes:
        raise RuntimeError(f"Uploaded size mismatch: local={total_bytes}, remote={size}")
    print(f"[upload] Upload complete: s3://{args.bucket}/{args.key} ({size} bytes)")


if __name__ == "__main__":
    main()
