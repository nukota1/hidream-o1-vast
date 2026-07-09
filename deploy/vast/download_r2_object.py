import argparse
import os

import boto3
from botocore.config import Config


def env(name, default=""):
    return os.environ.get(name, default).strip()


def file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Download a private Cloudflare R2 object.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-bytes", type=int, default=0)
    args = parser.parse_args()

    endpoint_url = env("R2_ENDPOINT_URL")
    access_key = env("R2_ACCESS_KEY_ID")
    secret_key = env("R2_SECRET_ACCESS_KEY")
    if not endpoint_url or not access_key or not secret_key:
        raise RuntimeError("R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY are required.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tmp_path = f"{args.output}.part"
    try:
        os.remove(tmp_path)
    except OSError:
        pass

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

    transferred = 0
    next_log = 512 * 1024 * 1024

    def progress(bytes_amount):
        nonlocal transferred, next_log
        transferred += bytes_amount
        if transferred >= next_log:
            print(f"[entrypoint] R2 download progress: {transferred / (1024 ** 3):.2f} GB")
            next_log += 512 * 1024 * 1024

    print(f"[entrypoint] Downloading R2 object s3://{args.bucket}/{args.key} to {args.output}")
    with open(tmp_path, "wb") as f:
        client.download_fileobj(args.bucket, args.key, f, Callback=progress)

    actual_bytes = file_size(tmp_path)
    if args.min_bytes and actual_bytes < args.min_bytes:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(
            f"Downloaded R2 object is smaller than expected: {actual_bytes}/{args.min_bytes} bytes"
        )

    os.replace(tmp_path, args.output)
    print(f"[entrypoint] R2 download complete: {args.output} ({actual_bytes} bytes)")


if __name__ == "__main__":
    main()
