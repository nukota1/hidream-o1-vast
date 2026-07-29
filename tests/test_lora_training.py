import base64
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from lora_training import (
    LoraStore,
    current_model_type,
    is_lora_compatible,
    owner_storage_key,
    validate_training_request,
)
from sdxl_janku_workflow import (
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


class LoraTrainingTests(unittest.TestCase):
    zero_environment = {
        "IMAGE_MODEL_FAMILY": "animagine",
        "IMAGE_MODEL_PROFILE": "sdxl-animagine-zero",
        "ANIMAGINE_MODEL_REPO": "cagliostrolab/animagine-xl-4.0-zero",
    }

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


if __name__ == "__main__":
    unittest.main()
