#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lora_r2_sync import LoraR2Sync
from lora_training import LoraStore


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish or restore user-scoped LoRA assets with Cloudflare R2."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish", help="Upload one ready local LoRA.")
    publish.add_argument("--owner-id", default="local")
    publish.add_argument("--model-id", required=True)
    publish.add_argument(
        "--remote-owner-key",
        default="",
        help="Optional hashed target owner key for a one-time local-to-cloud migration.",
    )
    publish.add_argument(
        "--include-training-data",
        action="store_true",
        help="Also upload the source images and captions under dataset/.",
    )

    restore = subparsers.add_parser("restore", help="Restore all LoRAs for one owner.")
    restore.add_argument("--owner-id", default="local")
    restore.add_argument(
        "--restore-training-data",
        action="store_true",
        help="Also restore source images and captions when available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    store = LoraStore()
    sync = LoraR2Sync.from_environment(store)
    if not sync.enabled:
        raise RuntimeError("Set LORA_R2_SYNC_ENABLED=1 before running this command.")
    if args.command == "publish":
        if args.include_training_data:
            sync.include_training_data = True
        result = sync.publish_model(
            args.owner_id,
            args.model_id,
            remote_owner_key=args.remote_owner_key or None,
        )
    else:
        if args.restore_training_data:
            sync.restore_training_data = True
        result = sync.sync_owner(args.owner_id, force=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
