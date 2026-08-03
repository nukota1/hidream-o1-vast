import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from lora_r2_sync import LoraR2Sync
from lora_training import (
    LoraStore,
    current_model_type,
    is_lora_compatible,
    owner_storage_key,
    validate_training_request,
)
from sdxl_janku_workflow import (
    _wait_for_model_file,
    configure_pipeline_loras,
    configure_pipeline_reference,
    generate_with_janku,
    prepare_character_reference,
)
from scripts.train_character_lora import (
    load_pixels,
    load_training_records,
    target_bucket_size,
)


def encoded_image(colour=(210, 40, 90, 255), size=(512, 512)):
    image = Image.new("RGBA", size, colour)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeR2Body(io.BytesIO):
    pass


class FakeR2Client:
    def __init__(self):
        self.objects = {}

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = bytes(Body)

    def get_object(self, *, Bucket, Key):
        return {"Body": FakeR2Body(self.objects[(Bucket, Key)])}

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        contents = [
            {"Key": key, "Size": len(value)}
            for (bucket, key), value in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def delete_objects(self, *, Bucket, Delete):
        for item in Delete.get("Objects") or []:
            self.objects.pop((Bucket, item["Key"]), None)
        return {"Deleted": Delete.get("Objects") or []}


class LoraTrainingTests(unittest.TestCase):
    zero_environment = {
        "IMAGE_MODEL_FAMILY": "animagine",
        "IMAGE_MODEL_PROFILE": "sdxl-animagine-zero",
        "ANIMAGINE_MODEL_REPO": "cagliostrolab/animagine-xl-4.0-zero",
    }

    def create_ready_model(self, store, owner_id, category="character"):
        with patch.dict(os.environ, self.zero_environment):
            model = store.create(owner_id, {
                "name": f"{category.title()} asset",
                "trigger_word": f"nkt_{category}001",
                "identity_prompt": "petite, blonde hair" if category == "character" else "",
                "category": category,
                "model_type": "sdxl-animagine-zero",
                "images": [{
                    "image": encoded_image(),
                    "caption": f"sample {category} caption",
                }],
            })
        output = store.output_path(owner_id, model["id"])
        output.mkdir(parents=True)
        (output / "pytorch_lora_weights.safetensors").write_bytes(
            f"{category}-weights".encode("utf-8")
        )
        (output / "training.json").write_text(
            json.dumps({"category": category}),
            encoding="utf-8",
        )
        return store.update(owner_id, model["id"], status="ready", progress=100)

    def test_owner_storage_key_does_not_expose_external_user_id(self):
        self.assertEqual(owner_storage_key("local"), "local")
        key = owner_storage_key("google-user-123")
        self.assertNotIn("google-user", key)
        self.assertEqual(len(key), 32)

    def test_training_request_uses_model_compatible_defaults(self):
        with patch.dict(os.environ, self.zero_environment):
            result = validate_training_request({
                "name": "Hinata",
                "trigger_word": "sakurai_hinata",
                "identity_prompt": "小柄、金髪、後頭部中央のお団子、ピンク色の瞳",
                "identity_negative_prompt": "side bun, bun beside ear",
                "category": "character",
                "model_type": "sdxl-animagine-zero",
                "images": [{"image": encoded_image()}],
            })

        self.assertEqual(result["steps"], 200)
        self.assertEqual(result["rank"], 16)
        self.assertEqual(result["resolution"], 768)
        self.assertEqual(
            result["identity_prompt"],
            "小柄、金髪、後頭部中央のお団子、ピンク色の瞳",
        )
        self.assertEqual(result["learning_rate"], 1e-4)
        self.assertFalse(result["random_flip"])
        self.assertEqual(
            result["identity_negative_prompt"],
            "side bun, bun beside ear",
        )

    def test_incompatible_model_type_is_rejected(self):
        with (
            patch.dict(os.environ, self.zero_environment),
            self.assertRaisesRegex(ValueError, "基盤モデル"),
        ):
            validate_training_request({
                "name": "Hinata",
                "trigger_word": "sakurai_hinata",
                "category": "character",
                "model_type": "sdxl-animagine-opt",
                "images": [{"image": encoded_image()}],
            })

    def test_many_images_do_not_default_to_overtraining(self):
        with patch.dict(os.environ, self.zero_environment):
            result = validate_training_request({
                "name": "Hinata",
                "trigger_word": "sakurai_hinata",
                "category": "character",
                "model_type": "sdxl-animagine-zero",
                "images": [
                    {"image": encoded_image()}
                    for _ in range(30)
                ],
            })

        self.assertEqual(result["steps"], 600)

    def test_style_training_uses_separate_defaults(self):
        with patch.dict(os.environ, self.zero_environment):
            result = validate_training_request({
                "name": "House style",
                "trigger_word": "nkt_style001",
                "category": "style",
                "model_type": "sdxl-animagine-zero",
                "images": [
                    {"image": encoded_image()}
                    for _ in range(50)
                ],
            })

        self.assertEqual(result["steps"], 500)
        self.assertEqual(result["rank"], 32)
        self.assertEqual(result["learning_rate"], 5e-5)
        self.assertTrue(result["random_flip"])
        self.assertEqual(result["recommended_weight"], 0.6)

    def test_legacy_opt_lora_is_not_compatible_with_zero(self):
        with patch.dict(os.environ, self.zero_environment):
            self.assertEqual(current_model_type(), "sdxl-animagine-zero")
            self.assertFalse(is_lora_compatible({
                "model_type": "sdxl-animagine",
            }))

    def test_store_saves_dataset_and_isolates_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LoraStore(directory)
            with patch.dict(os.environ, self.zero_environment):
                model = store.create("owner-a", {
                    "name": "Hinata",
                    "trigger_word": "sakurai_hinata",
                    "identity_prompt": "petite, blonde hair, pink eyes",
                    "category": "character",
                    "model_type": "sdxl-animagine-zero",
                    "images": [{
                        "image": encoded_image((210, 40, 90, 128)),
                        "source_id": "gallery-1",
                        "prompt": "pink short hair",
                        "caption": "pink hair, short hair, school uniform",
                    }],
                })

            self.assertEqual(model["status"], "queued")
            self.assertEqual(model["image_count"], 1)
            self.assertEqual(
                store.read("owner-a", model["id"])["identity_prompt"],
                "petite, blonde hair, pink eyes",
            )
            self.assertEqual(len(store.list("owner-a")), 1)
            self.assertEqual(store.list("owner-b"), [])
            saved = Image.open(store.dataset_path("owner-a", model["id"]) / "001.png")
            self.assertEqual(saved.mode, "RGB")
            self.assertGreater(saved.getpixel((0, 0))[0], 210)
            caption = (
                store.dataset_path("owner-a", model["id"]) / "001.txt"
            ).read_text(encoding="utf-8")
            self.assertEqual(
                caption,
                "pink hair, short hair, school uniform",
            )

    def test_r2_publish_uses_pseudonymous_owner_and_omits_dataset_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LoraStore(directory)
            model = self.create_ready_model(store, "google-user-123")
            checkpoint = store.output_path(
                "google-user-123",
                model["id"],
            ) / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "pytorch_lora_weights.safetensors").write_bytes(b"checkpoint")
            client = FakeR2Client()
            sync = LoraR2Sync(
                store,
                enabled=True,
                bucket="model-cache",
                client=client,
            )

            result = sync.publish_model("google-user-123", model["id"])

            keys = [key for bucket, key in client.objects if bucket == "model-cache"]
            self.assertEqual(result["status"], "uploaded")
            self.assertTrue(all("google-user-123" not in key for key in keys))
            self.assertTrue(any(key.endswith("/metadata.json") for key in keys))
            self.assertTrue(any(key.endswith("/pytorch_lora_weights.safetensors") for key in keys))
            self.assertTrue(any(key.endswith("/training.json") for key in keys))
            self.assertFalse(any("/checkpoint-100/" in key for key in keys))
            self.assertFalse(any("/dataset/" in key for key in keys))
            metadata_key = next(key for key in keys if key.endswith("/metadata.json"))
            remote_metadata = json.loads(client.objects[("model-cache", metadata_key)])
            self.assertNotIn("captions", remote_metadata)
            self.assertNotIn("remote_storage", store.public(store.read("google-user-123", model["id"])))

    def test_r2_republish_prunes_unreferenced_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LoraStore(directory)
            model = self.create_ready_model(store, "owner-a")
            checkpoint = store.output_path("owner-a", model["id"]) / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "pytorch_lora_weights.safetensors").write_bytes(b"checkpoint")
            client = FakeR2Client()
            sync = LoraR2Sync(
                store,
                enabled=True,
                bucket="model-cache",
                include_checkpoints=True,
                client=client,
            )
            sync.publish_model("owner-a", model["id"])
            self.assertTrue(any("/checkpoint-100/" in key for _, key in client.objects))

            sync.include_checkpoints = False
            result = sync.publish_model("owner-a", model["id"])

            self.assertEqual(result["deleted"], 1)
            self.assertEqual(result["cleanup_errors"], [])
            self.assertFalse(any("/checkpoint-100/" in key for _, key in client.objects))

    def test_r2_sync_restores_character_and_style_for_only_the_requested_owner(self):
        client = FakeR2Client()
        with tempfile.TemporaryDirectory() as source_directory:
            source = LoraStore(source_directory)
            character = self.create_ready_model(source, "owner-a", "character")
            style = self.create_ready_model(source, "owner-a", "style")
            publisher = LoraR2Sync(
                source,
                enabled=True,
                bucket="model-cache",
                client=client,
            )
            publisher.publish_model("owner-a", character["id"])
            publisher.publish_model("owner-a", style["id"])

        with tempfile.TemporaryDirectory() as target_directory:
            target = LoraStore(target_directory)
            restorer = LoraR2Sync(
                target,
                enabled=True,
                bucket="model-cache",
                sync_interval_seconds=0,
                client=client,
            )

            result = restorer.sync_owner("owner-a", force=True)

            self.assertEqual(result["status"], "synced")
            self.assertEqual(result["downloaded"], 2)
            self.assertEqual(
                {item["category"] for item in target.list("owner-a")},
                {"character", "style"},
            )
            self.assertEqual(target.list("owner-b"), [])
            for item in target.list("owner-a"):
                self.assertTrue(target.weight_path("owner-a", item["id"]).is_file())
                self.assertFalse(target.model_root("owner-a", item["id"]).joinpath("dataset").exists())

    def test_r2_publish_can_migrate_local_model_to_cloud_owner_key(self):
        client = FakeR2Client()
        cloud_owner = "cloudflare-user-a"
        cloud_owner_key = owner_storage_key(cloud_owner)
        with tempfile.TemporaryDirectory() as source_directory:
            source = LoraStore(source_directory)
            model = self.create_ready_model(source, "local")
            publisher = LoraR2Sync(
                source,
                enabled=True,
                bucket="model-cache",
                client=client,
            )
            publisher.publish_model(
                "local",
                model["id"],
                remote_owner_key=cloud_owner_key,
            )
            self.assertTrue(all(f"/owners/{cloud_owner_key}/" in key for _, key in client.objects))

        with tempfile.TemporaryDirectory() as target_directory:
            target = LoraStore(target_directory)
            restorer = LoraR2Sync(
                target,
                enabled=True,
                bucket="model-cache",
                client=client,
            )
            result = restorer.sync_owner(cloud_owner, force=True)
            self.assertEqual(result["downloaded"], 1)
            self.assertEqual(target.list(cloud_owner)[0]["id"], model["id"])

    def test_r2_can_include_and_restore_private_training_data(self):
        client = FakeR2Client()
        with tempfile.TemporaryDirectory() as source_directory:
            source = LoraStore(source_directory)
            model = self.create_ready_model(source, "owner-a")
            publisher = LoraR2Sync(
                source,
                enabled=True,
                bucket="model-cache",
                include_training_data=True,
                client=client,
            )
            publisher.publish_model("owner-a", model["id"])
            self.assertTrue(any("/dataset/001.png" in key for _, key in client.objects))
            metadata_key = next(key for _, key in client.objects if key.endswith("/metadata.json"))
            remote_metadata = json.loads(client.objects[("model-cache", metadata_key)])
            self.assertIn("captions", remote_metadata)

        with tempfile.TemporaryDirectory() as target_directory:
            target = LoraStore(target_directory)
            restorer = LoraR2Sync(
                target,
                enabled=True,
                bucket="model-cache",
                restore_training_data=True,
                client=client,
            )
            restorer.sync_owner("owner-a", force=True)
            self.assertTrue(
                target.model_root("owner-a", model["id"]).joinpath("dataset", "001.png").is_file()
            )
            self.assertTrue(
                target.model_root("owner-a", model["id"]).joinpath("dataset", "001.txt").is_file()
            )

    def test_r2_sync_rejects_corrupted_weight_without_registering_model(self):
        client = FakeR2Client()
        with tempfile.TemporaryDirectory() as source_directory:
            source = LoraStore(source_directory)
            model = self.create_ready_model(source, "owner-a")
            publisher = LoraR2Sync(
                source,
                enabled=True,
                bucket="model-cache",
                client=client,
            )
            publisher.publish_model("owner-a", model["id"])
        weight_key = next(
            key
            for bucket, key in client.objects
            if bucket == "model-cache" and key.endswith("pytorch_lora_weights.safetensors")
        )
        client.objects[("model-cache", weight_key)] = b"corrupted"

        with tempfile.TemporaryDirectory() as target_directory:
            target = LoraStore(target_directory)
            restorer = LoraR2Sync(
                target,
                enabled=True,
                bucket="model-cache",
                client=client,
            )
            result = restorer.sync_owner("owner-a", force=True)

            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["downloaded"], 0)
            self.assertEqual(len(result["errors"]), 1)
            self.assertEqual(target.list("owner-a"), [])

    def test_training_records_use_manifest_and_prefix_trigger(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            image_path = dataset / "sample.png"
            Image.new("RGB", (1024, 1536), "white").save(image_path)
            (dataset / "captions.json").write_text(
                '{"sample.png":"1girl, solo, black t-shirt, white background"}',
                encoding="utf-8",
            )

            records = load_training_records(
                dataset,
                "nkt_chr001",
                "character",
            )

            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["caption"].startswith("nkt_chr001, "))
            self.assertIn("black t-shirt", records[0]["caption"])

    def test_portrait_bucket_preserves_full_body_aspect_ratio(self):
        self.assertEqual(target_bucket_size(1024, 1536, 768), (640, 960))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "portrait.png"
            Image.new("RGB", (1024, 1536), "white").save(path)

            pixels, size = load_pixels(path, 768)

            self.assertEqual(size, (960, 640))
            self.assertEqual(tuple(pixels.shape), (1, 3, 960, 640))

    def test_pipeline_loads_character_and_style_adapters_with_independent_weights(self):
        calls = []

        class FakePipeline:
            def unload_lora_weights(self):
                calls.append(("unload",))

            def load_lora_weights(self, directory, **kwargs):
                calls.append(("load", directory, kwargs))

            def set_adapters(self, name, adapter_weights):
                calls.append(("set", name, adapter_weights))

        with tempfile.TemporaryDirectory() as directory:
            character_path = Path(directory) / "character.safetensors"
            style_path = Path(directory) / "style.safetensors"
            character_path.write_bytes(b"character")
            style_path.write_bytes(b"style")
            configure_pipeline_loras(FakePipeline(), [
                {
                    "weights_path": character_path,
                    "weight": 0.8,
                    "adapter_name": "character_asset",
                },
                {
                    "weights_path": style_path,
                    "weight": 0.55,
                    "adapter_name": "style_asset",
                },
            ])

        self.assertEqual(calls[0], ("unload",))
        self.assertEqual(calls[1][2]["weight_name"], character_path.name)
        self.assertEqual(calls[1][2]["adapter_name"], "character_asset")
        self.assertEqual(calls[2][2]["weight_name"], style_path.name)
        self.assertEqual(calls[2][2]["adapter_name"], "style_asset")
        self.assertEqual(
            calls[3],
            ("set", ["character_asset", "style_asset"], [0.8, 0.55]),
        )

    def test_reference_adapter_loads_once_and_updates_scale(self):
        calls = []

        class FakePipeline:
            def load_ip_adapter(self, repo_id, **kwargs):
                calls.append(("load", repo_id, kwargs))

            def set_ip_adapter_scale(self, weight):
                calls.append(("scale", weight))

            def unload_ip_adapter(self):
                calls.append(("unload",))

        pipeline = FakePipeline()
        configure_pipeline_reference(pipeline, enabled=True, weight=0.55)
        configure_pipeline_reference(pipeline, enabled=True, weight=0.7)
        configure_pipeline_reference(pipeline, enabled=False)

        self.assertEqual(len([call for call in calls if call[0] == "load"]), 1)
        self.assertEqual(calls[0][1], "h94/IP-Adapter")
        self.assertEqual(calls[0][2]["subfolder"], "sdxl_models")
        self.assertEqual(
            calls[0][2]["image_encoder_folder"],
            "models/image_encoder",
        )
        self.assertEqual(calls[1], ("scale", 0.55))
        self.assertEqual(calls[2], ("scale", 0.7))
        self.assertEqual(calls[3], ("unload",))
        self.assertFalse(pipeline._character_reference_adapter_loaded)

    def test_generation_passes_reference_image_to_pipeline(self):
        class FakePipeline:
            scheduler = type("Scheduler", (), {"config": {}})()

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return type("Result", (), {
                    "images": [Image.new("RGB", (32, 32), "blue")]
                })()

        settings = {
            "steps": 1,
            "sampler": "euler",
            "negative_prompt": "",
            "width": 32,
            "height": 32,
            "cfg": 1.0,
            "clip_skip": 1,
            "seed": 1,
        }
        pipeline = FakePipeline()
        reference = Image.new("RGBA", (16, 16), (210, 40, 90, 255))
        with (
            patch("sdxl_janku_workflow._set_sampler"),
            patch("sdxl_janku_workflow._generator", return_value=None),
        ):
            generate_with_janku(
                pipeline,
                "1girl, solo",
                settings,
                reference_image=reference,
            )

        self.assertEqual(pipeline.kwargs["ip_adapter_image"].mode, "RGB")

    def test_character_reference_crop_reduces_old_scene_area(self):
        reference = Image.new("RGB", (1000, 800), "white")

        cropped = prepare_character_reference(reference)

        self.assertEqual(cropped.size, (700, 640))

    def test_model_wait_reports_background_download_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "animagine.safetensors"
            marker_path = Path(f"{model_path}.download_failed")
            marker_path.write_text(
                "Animagine download failed after retries.",
                encoding="utf-8",
            )

            with patch("sdxl_janku_workflow.time.sleep") as sleep:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Animagine download failed after retries",
                ):
                    _wait_for_model_file(str(model_path), 100)

            sleep.assert_not_called()

    def test_complete_model_ignores_stale_download_failure_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "animagine.safetensors"
            model_path.write_bytes(b"complete")
            Path(f"{model_path}.download_failed").write_text(
                "stale failure",
                encoding="utf-8",
            )

            _wait_for_model_file(str(model_path), len(b"complete"))


if __name__ == "__main__":
    unittest.main()
