import hashlib
import json
import mimetypes
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from lora_training import (
    LORA_CATEGORIES,
    LORA_MODEL_TYPES,
    canonical_model_type,
    owner_storage_key,
)


TRUE_VALUES = {"1", "true", "yes", "on"}
MODEL_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
OWNER_KEY_PATTERN = re.compile(r"^(?:local|[a-f0-9]{32})$")
METADATA_MAX_BYTES = 2 * 1024 * 1024


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_prefix(value):
    parts = [part for part in str(value or "").strip().strip("/").split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError("LORA_R2_PREFIX must be a non-empty object prefix.")
    return "/".join(parts)


def safe_artifact_path(value):
    path = PurePosixPath(str(value or ""))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.parts[0] not in {"output", "dataset"}
    ):
        raise ValueError("Remote LoRA artifact path is invalid.")
    return path


class LoraR2Sync:
    """Synchronize per-user LoRA assets between local storage and R2.

    Remote metadata is uploaded last and acts as the completion marker. R2 keys
    contain only the same pseudonymous owner key used by the local LoraStore.
    """

    def __init__(
        self,
        store,
        *,
        enabled=False,
        bucket="",
        prefix="loras/v1",
        endpoint_url="",
        access_key_id="",
        secret_access_key="",
        upload_after_training=True,
        include_training_data=False,
        restore_training_data=False,
        sync_interval_seconds=60,
        max_model_bytes=4 * 1024 * 1024 * 1024,
        client=None,
    ):
        self.store = store
        self.enabled = bool(enabled)
        self.bucket = str(bucket or "").strip()
        self.prefix = safe_prefix(prefix)
        self.endpoint_url = str(endpoint_url or "").strip().rstrip("/")
        self.access_key_id = str(access_key_id or "").strip()
        self.secret_access_key = str(secret_access_key or "").strip()
        self.upload_after_training = bool(upload_after_training)
        self.include_training_data = bool(include_training_data)
        self.restore_training_data = bool(restore_training_data)
        self.sync_interval_seconds = max(0, int(sync_interval_seconds))
        self.max_model_bytes = max(1, int(max_model_bytes))
        self._client = client
        self._lock = threading.RLock()
        self._last_sync = {}

    @classmethod
    def from_environment(cls, store):
        endpoint_url = os.environ.get(
            "LORA_R2_ENDPOINT_URL",
            os.environ.get("R2_ENDPOINT_URL", ""),
        )
        access_key_id = os.environ.get(
            "LORA_R2_ACCESS_KEY_ID",
            os.environ.get("R2_ACCESS_KEY_ID", ""),
        )
        secret_access_key = os.environ.get(
            "LORA_R2_SECRET_ACCESS_KEY",
            os.environ.get("R2_SECRET_ACCESS_KEY", ""),
        )
        return cls(
            store,
            enabled=env_flag("LORA_R2_SYNC_ENABLED"),
            bucket=os.environ.get("LORA_R2_BUCKET", ""),
            prefix=os.environ.get("LORA_R2_PREFIX", "loras/v1"),
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            upload_after_training=env_flag("LORA_R2_UPLOAD_AFTER_TRAINING", True),
            include_training_data=env_flag("LORA_R2_INCLUDE_TRAINING_DATA"),
            restore_training_data=env_flag("LORA_R2_RESTORE_TRAINING_DATA"),
            sync_interval_seconds=os.environ.get("LORA_R2_SYNC_INTERVAL_SECONDS", "60"),
            max_model_bytes=os.environ.get(
                "LORA_R2_MAX_MODEL_BYTES",
                str(4 * 1024 * 1024 * 1024),
            ),
        )

    @property
    def configured(self):
        if self._client is not None:
            return bool(self.bucket)
        return all([
            self.bucket,
            self.endpoint_url,
            self.access_key_id,
            self.secret_access_key,
        ])

    def public_status(self, owner_id, result=None):
        value = {
            "enabled": self.enabled,
            "configured": self.configured,
            "upload_after_training": self.upload_after_training,
            "owner_key": owner_storage_key(owner_id),
            "bucket": self.bucket if self.enabled else "",
            "prefix": self.prefix if self.enabled else "",
        }
        if result:
            value.update(result)
        elif not self.enabled:
            value["status"] = "disabled"
        elif not self.configured:
            value["status"] = "misconfigured"
        else:
            value["status"] = "ready"
        return value

    def _require_client(self):
        if not self.enabled:
            return None
        if not self.configured:
            raise RuntimeError(
                "LoRA R2 sync requires LORA_R2_BUCKET and R2 S3 credentials."
            )
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name="auto",
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 10, "mode": "standard"},
                ),
            )
        return self._client

    def _owner_prefix(self, owner_key):
        if not OWNER_KEY_PATTERN.fullmatch(owner_key):
            raise ValueError("Invalid LoRA owner storage key.")
        return f"{self.prefix}/owners/{owner_key}/models"

    def _model_prefix(self, owner_key, model_id):
        if not MODEL_ID_PATTERN.fullmatch(str(model_id or "")):
            raise KeyError("Unknown LoRA")
        return f"{self._owner_prefix(owner_key)}/{model_id}"

    def _metadata_key(self, owner_key, model_id):
        return f"{self._model_prefix(owner_key, model_id)}/metadata.json"

    def _artifact_key(self, owner_key, model_id, relative_path):
        path = safe_artifact_path(relative_path)
        return f"{self._model_prefix(owner_key, model_id)}/{path.as_posix()}"

    def _list_metadata_keys(self, client, owner_key):
        prefix = f"{self._owner_prefix(owner_key)}/"
        continuation = None
        keys = []
        while True:
            arguments = {"Bucket": self.bucket, "Prefix": prefix}
            if continuation:
                arguments["ContinuationToken"] = continuation
            response = client.list_objects_v2(**arguments)
            for item in response.get("Contents") or []:
                key = str(item.get("Key") or "")
                relative = key[len(prefix):] if key.startswith(prefix) else ""
                match = re.fullmatch(r"([a-f0-9]{32})/metadata\.json", relative)
                if match:
                    keys.append((match.group(1), key))
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
            if not continuation:
                break
        return keys

    def _read_remote_metadata(self, client, key, expected_model_id):
        response = client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"].read(METADATA_MAX_BYTES + 1)
        if len(body) > METADATA_MAX_BYTES:
            raise ValueError("Remote LoRA metadata is too large.")
        metadata = json.loads(body.decode("utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Remote LoRA metadata must be an object.")
        if metadata.get("id") != expected_model_id:
            raise ValueError("Remote LoRA metadata ID does not match its object key.")
        if metadata.get("status") != "ready":
            raise ValueError("Remote LoRA is not ready.")
        if metadata.get("category") not in LORA_CATEGORIES:
            raise ValueError("Remote LoRA category is unsupported.")
        model_type = canonical_model_type(metadata.get("model_type"))
        if model_type not in LORA_MODEL_TYPES:
            raise ValueError("Remote LoRA model profile is unsupported.")
        metadata["model_type"] = model_type
        storage = metadata.get("remote_storage")
        if not isinstance(storage, dict) or storage.get("schema_version") != 1:
            raise ValueError("Remote LoRA storage manifest is unsupported.")
        artifacts = storage.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts or len(artifacts) > 1000:
            raise ValueError("Remote LoRA artifact manifest is invalid.")
        validated = []
        total_bytes = 0
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ValueError("Remote LoRA artifact entry is invalid.")
            path = safe_artifact_path(artifact.get("path"))
            size = int(artifact.get("size") or 0)
            sha256 = str(artifact.get("sha256") or "")
            if size <= 0 or not re.fullmatch(r"[a-f0-9]{64}", sha256):
                raise ValueError("Remote LoRA artifact checksum is invalid.")
            total_bytes += size
            validated.append({"path": path.as_posix(), "size": size, "sha256": sha256})
        if total_bytes > self.max_model_bytes:
            raise ValueError("Remote LoRA exceeds LORA_R2_MAX_MODEL_BYTES.")
        if not any(
            item["path"] == "output/pytorch_lora_weights.safetensors"
            for item in validated
        ):
            raise ValueError("Remote LoRA has no inference weight artifact.")
        storage["artifacts"] = validated
        return metadata

    def _local_artifacts(self, owner_id, model_id):
        model_root = self.store.model_root(owner_id, model_id)
        weight_path = self.store.weight_path(owner_id, model_id)
        if not weight_path.is_file() or weight_path.is_symlink():
            raise RuntimeError("Completed LoRA weight file is missing.")
        roots = [model_root / "output"]
        if self.include_training_data:
            roots.append(model_root / "dataset")
        paths = []
        for root in roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    relative = PurePosixPath(path.relative_to(model_root).as_posix())
                    safe_artifact_path(relative)
                    paths.append((path, relative))
        return paths

    def publish_model(self, owner_id, model_id, *, remote_owner_key=None):
        if not self.enabled:
            return {"status": "disabled", "uploaded": 0}
        client = self._require_client()
        owner_key = remote_owner_key or owner_storage_key(owner_id)
        if not OWNER_KEY_PATTERN.fullmatch(owner_key):
            raise ValueError("Invalid target owner storage key.")
        with self._lock:
            metadata = self.store.read(owner_id, model_id)
            if metadata.get("status") != "ready":
                raise RuntimeError("Only ready LoRAs can be published to R2.")
            artifacts = []
            total_bytes = 0
            for path, relative in self._local_artifacts(owner_id, model_id):
                size = path.stat().st_size
                total_bytes += size
                if total_bytes > self.max_model_bytes:
                    raise RuntimeError("LoRA exceeds LORA_R2_MAX_MODEL_BYTES.")
                artifact = {
                    "path": relative.as_posix(),
                    "size": size,
                    "sha256": sha256_file(path),
                }
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                client.upload_file(
                    str(path),
                    self.bucket,
                    self._artifact_key(owner_key, model_id, relative),
                    ExtraArgs={"ContentType": content_type},
                )
                artifacts.append(artifact)
            synced_at = utc_now()
            remote_storage = {
                "schema_version": 1,
                "synced_at": synced_at,
                "training_data_included": self.include_training_data,
                "artifacts": artifacts,
            }
            published = self.store.update(
                owner_id,
                model_id,
                r2_sync_status="ready",
                r2_synced_at=synced_at,
                r2_sync_error="",
                remote_storage=remote_storage,
            )
            remote_metadata = dict(published)
            if not self.include_training_data:
                remote_metadata.pop("captions", None)
            client.put_object(
                Bucket=self.bucket,
                Key=self._metadata_key(owner_key, model_id),
                Body=json.dumps(remote_metadata, ensure_ascii=False, indent=2).encode("utf-8"),
                ContentType="application/json; charset=utf-8",
            )
            self._last_sync[owner_key] = time.monotonic()
            return {
                "status": "uploaded",
                "uploaded": len(artifacts),
                "bytes": total_bytes,
                "owner_key": owner_key,
                "model_id": model_id,
            }

    def mark_publish_failed(self, owner_id, model_id, error):
        return self.store.update(
            owner_id,
            model_id,
            r2_sync_status="failed",
            r2_sync_error=str(error)[-1000:],
        )

    def _artifact_is_current(self, path, artifact):
        return (
            path.is_file()
            and path.stat().st_size == artifact["size"]
            and sha256_file(path) == artifact["sha256"]
        )

    def _download_model(self, client, owner_id, owner_key, model_id, metadata):
        try:
            local = self.store.read(owner_id, model_id)
        except KeyError:
            local = None
        if local and local.get("status") in {"queued", "training"}:
            return "skipped"

        model_root = self.store.model_root(owner_id, model_id)
        changed = False
        for artifact in metadata["remote_storage"]["artifacts"]:
            relative = safe_artifact_path(artifact["path"])
            if relative.parts[0] == "dataset" and not self.restore_training_data:
                continue
            destination = model_root.joinpath(*relative.parts)
            if self._artifact_is_current(destination, artifact):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp_path = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.part")
            try:
                client.download_file(
                    self.bucket,
                    self._artifact_key(owner_key, model_id, relative),
                    str(temp_path),
                )
                if not self._artifact_is_current(temp_path, artifact):
                    raise RuntimeError("Downloaded LoRA artifact failed checksum validation.")
                temp_path.replace(destination)
                changed = True
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        weight_path = self.store.weight_path(owner_id, model_id)
        weight_artifact = next(
            item
            for item in metadata["remote_storage"]["artifacts"]
            if item["path"] == "output/pytorch_lora_weights.safetensors"
        )
        if not self._artifact_is_current(weight_path, weight_artifact):
            raise RuntimeError("LoRA weight was not restored correctly.")
        self.store.write_metadata(owner_id, model_id, metadata)
        return "downloaded" if changed or local is None else "unchanged"

    def sync_owner(self, owner_id, *, force=False):
        if not self.enabled:
            return {"status": "disabled", "remote_models": 0, "downloaded": 0}
        client = self._require_client()
        owner_key = owner_storage_key(owner_id)
        with self._lock:
            now = time.monotonic()
            last_sync = self._last_sync.get(owner_key, 0)
            if (
                not force
                and self.sync_interval_seconds > 0
                and now - last_sync < self.sync_interval_seconds
            ):
                return {"status": "cached", "remote_models": 0, "downloaded": 0}
            downloaded = 0
            unchanged = 0
            skipped = 0
            errors = []
            metadata_keys = self._list_metadata_keys(client, owner_key)
            for model_id, key in metadata_keys:
                try:
                    metadata = self._read_remote_metadata(client, key, model_id)
                    outcome = self._download_model(
                        client,
                        owner_id,
                        owner_key,
                        model_id,
                        metadata,
                    )
                    if outcome == "downloaded":
                        downloaded += 1
                    elif outcome == "unchanged":
                        unchanged += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    errors.append({"model_id": model_id, "message": str(exc)[:500]})
            self._last_sync[owner_key] = time.monotonic()
            return {
                "status": "partial" if errors else "synced",
                "remote_models": len(metadata_keys),
                "downloaded": downloaded,
                "unchanged": unchanged,
                "skipped": skipped,
                "errors": errors,
            }
