import tempfile
import unittest
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as app_module
from app import (
    apply_character_constraints,
    apply_background_replacement_constraints,
    apply_event_instruction_constraints,
    apply_lora_leakage_constraints,
    apply_story_composition_defaults,
    apply_story_scene_constraints,
    apply_story_subject_constraints,
    apply_source_scene_exclusion,
    background_inpaint_prompt,
    background_prompt_requests_subject_change,
    batch_generation_seeds,
    build_consistent_story_prompt,
    prioritize_consistent_story_tags,
    prioritize_story_negative_tags,
    prepare_prompt,
)
from image_edit_workflows import (
    CHARACTER_CHROMA_RGB,
    _anime_foreground_mask,
    _automatic_background_mask,
    _edit_waifu,
    _feather_background_mask,
    _load_source_and_mask,
    extract_plain_background_mask,
    prepare_character_layer,
    restore_unmasked_pixels,
)


class BackgroundPreservationTests(unittest.TestCase):
    def test_batch_generation_seeds_keep_first_and_randomize_followups(self):
        values = iter((32, 101, 202))

        seeds = batch_generation_seeds(
            32,
            3,
            randbelow=lambda _limit: next(values),
        )

        self.assertEqual(seeds, [32, 101, 202])
        self.assertEqual(len(set(seeds)), 3)

    def test_character_asset_uses_neutral_background_without_chroma(self):
        settings = app_module.normalize_generation_settings({
            "preset": "bishoujo_game",
            "negative_prompt": "low quality",
        })
        prompt_info = {
            "prompt": "1girl, solo, pink short hair, school uniform",
            "intent_notes": "",
        }

        result = apply_character_constraints(prompt_info, settings)

        self.assertIn("simple white background", result["prompt"])
        self.assertNotIn("green background", result["prompt"])

    def test_story_prompt_keeps_identity_before_scene(self):
        settings = app_module.normalize_generation_settings({
            "preset": "bishoujo_game",
            "negative_prompt": "low quality",
        })
        character = {
            "prompt": "1girl, solo, pink short hair, blue eyes, school uniform",
            "intent_notes": "character",
            "source": "test",
        }
        scene = {
            "prompt": "peace sign, smiling, beach, blue sea, sunny sky",
            "intent_notes": "scene",
            "source": "test",
        }

        result = build_consistent_story_prompt(character, scene, settings)

        self.assertLess(
            result["prompt"].index("pink hair"),
            result["prompt"].index("beach"),
        )
        self.assertIn("short hair", result["prompt"])
        self.assertTrue(result["prompt"].startswith("1girl, solo"))
        self.assertIn("peace sign", result["prompt"])
        self.assertEqual(
            result["character_prompt"],
            "1girl, solo, pink hair, short hair, blue eyes, school uniform",
        )

    def test_free_prompt_is_kept_before_catalog_tags(self):
        settings = app_module.normalize_generation_settings({
            "preset": "bishoujo_game",
            "negative_prompt": "low quality",
        })
        refined = {
            "prompt": "1girl, raising right hand, looking at viewer",
            "intent_notes": "",
            "source": "test-refiner",
        }

        with (
            patch.object(app_module, "IMAGE_MODEL_FAMILY", "animagine"),
            patch.object(app_module._STATE["refiner"], "refine", return_value=refined),
        ):
            result = prepare_prompt(
                "右手を高く上げる",
                "t2i",
                settings,
                True,
                workflow="character",
                supplemental_prompt="blue eyes, black sailor uniform",
            )
            result = apply_character_constraints(result, settings)

        self.assertIn("raising right hand", result["prompt"])
        self.assertIn("blue eyes", result["prompt"])
        self.assertLess(
            result["prompt"].index("raising right hand"),
            result["prompt"].index("blue eyes"),
        )

    def test_compose_free_prompt_is_kept_before_catalog_tags(self):
        settings = app_module.normalize_generation_settings({
            "preset": "bishoujo_game",
            "negative_prompt": "low quality",
        })
        refined = {
            "prompt": "empty school corridor, warm sunset light",
            "intent_notes": "",
            "source": "test-refiner",
        }

        with patch.object(app_module._STATE["refiner"], "refine", return_value=refined):
            result = prepare_prompt(
                "夕方の学校の廊下",
                "edit",
                settings,
                True,
                workflow="compose_background",
                supplemental_prompt="rain, puddles",
            )

        self.assertIn("empty school corridor", result["prompt"])
        self.assertIn("rain", result["prompt"])
        self.assertLess(
            result["prompt"].index("empty school corridor"),
            result["prompt"].index("rain"),
        )

    def test_background_prompt_defaults_to_eye_level_for_standing_character(self):
        prompt = background_inpaint_prompt("rural road, rice fields, heavy rain")

        self.assertIn("eye-level view", prompt)
        self.assertIn("central perspective", prompt)
        self.assertIn("unobstructed lower center", prompt)
        self.assertIn("clear ground plane in foreground", prompt)

    def test_background_prompt_preserves_explicit_aerial_camera(self):
        prompt = background_inpaint_prompt("aerial view, rural road, rice fields")

        self.assertIn("aerial view", prompt)
        self.assertNotIn("eye-level view", prompt)

    def test_background_only_detects_requested_pose_change(self):
        self.assertTrue(background_prompt_requests_subject_change(
            "背景を海に変更。ポーズを顔元でピースに変更。"
        ))
        self.assertFalse(background_prompt_requests_subject_change(
            "背景を海に変更。"
        ))
        self.assertFalse(background_prompt_requests_subject_change(
            "人物を含めない海の背景へ変更。"
        ))

    def test_event_peace_sign_is_expanded_and_prioritized(self):
        settings = app_module.normalize_generation_settings({
            "preset": "bishoujo_game",
            "negative_prompt": "low quality",
        })
        prompt_info = {"prompt": "pink short hair, female student, sea"}

        constrained = apply_event_instruction_constraints(
            prompt_info,
            (
                "笑顔でこちらを見ている。右手でピースサインを作り、"
                "指の間から目をのぞかせる。右耳にガラスの林檎のピアス。全身。"
            ),
            settings,
        )

        self.assertTrue(constrained["prompt"].startswith(
            "full body, smile, looking at viewer, right hand, "
            "(v over eye:1.4), single glass apple earring"
        ))
        self.assertNotIn("(v:1.3)", constrained["prompt"])
        self.assertNotIn("looking through fingers", constrained["prompt"])
        self.assertIn("hands under chin", settings["negative_prompt"])
        self.assertIn("finger to lips", settings["negative_prompt"])
        self.assertIn("finger to cheek", settings["negative_prompt"])
        self.assertIn("earrings on both ears", settings["negative_prompt"])

    def test_story_scene_constraints_preserve_rural_rice_field_evening(self):
        prompt_info = {"prompt": "soft lighting"}

        constrained = apply_story_scene_constraints(
            prompt_info,
            "田舎の田んぼ道で学校の帰りの夕方。",
        )

        self.assertIn("countryside", constrained["prompt"])
        self.assertIn("rice fields", constrained["prompt"])
        self.assertIn("rural road", constrained["prompt"])
        self.assertIn("after school", constrained["prompt"])
        self.assertIn("sunset", constrained["prompt"])
        self.assertTrue(constrained["prompt"].startswith(
            "countryside, rice fields, rural road, after school, sunset"
        ))

    def test_story_subject_constraints_infer_single_girl_from_lora_captions(self):
        settings = {"negative_prompt": "low quality"}
        prompt_info = {
            "prompt": "multiple girls, petite, smile, rice fields",
            "intent_notes": "",
        }
        metadata = {
            "captions": [{
                "training_caption": "1girl, solo, blonde hair, red eyes",
            }],
        }

        constrained = apply_story_subject_constraints(
            prompt_info,
            settings,
            "笑顔でこちらを見ている。",
            metadata,
        )

        self.assertTrue(constrained["prompt"].startswith("1girl, solo"))
        self.assertNotIn("multiple girls", constrained["prompt"])
        self.assertIn("multiple girls", settings["negative_prompt"])
        self.assertIn("duplicate", settings["negative_prompt"])
        self.assertIn("blue background", settings["negative_prompt"])

    def test_story_subject_constraints_allow_explicit_group(self):
        settings = {"negative_prompt": "low quality"}
        prompt_info = {"prompt": "2girls, rice fields", "intent_notes": ""}

        constrained = apply_story_subject_constraints(
            prompt_info,
            settings,
            "二人の少女が田んぼ道を歩く。",
            None,
        )

        self.assertEqual(constrained["prompt"], "2girls, rice fields")
        self.assertNotIn("multiple girls", settings["negative_prompt"])

    def test_full_body_story_defaults_square_animagine_canvas_to_portrait(self):
        settings = {"width": 1024, "height": 1024}

        with patch.object(app_module, "IMAGE_MODEL_FAMILY", "animagine"):
            changed = apply_story_composition_defaults(
                settings,
                "全身を描く。",
            )

        self.assertTrue(changed)
        self.assertEqual((settings["width"], settings["height"]), (768, 1152))

    def test_consistent_story_tags_put_one_character_before_pose_and_scene(self):
        prompt_info = {
            "prompt": (
                "peace sign, sandy beach, pink short hair, female student, "
                "full body, 1girl, solo, masterpiece"
            )
        }

        prioritized = prioritize_consistent_story_tags(prompt_info)

        self.assertTrue(prioritized["prompt"].startswith(
            "1girl, solo, pink short hair, peace sign"
        ))
        self.assertLess(
            prioritized["prompt"].index("pink short hair"),
            prioritized["prompt"].index("peace sign"),
        )
        self.assertLess(
            prioritized["prompt"].index("full body"),
            prioritized["prompt"].index("sandy beach"),
        )

    def test_consistent_story_tags_keep_asymmetric_identity_and_body_before_outfit(self):
        prompt_info = {
            "prompt": (
                "soft anime illustration, seaside promenade, white dress, standing, "
                "petite, youthful round face, centered back hair bun, pink eyes, "
                "1girl, solo, masterpiece"
            )
        }

        prioritized = prioritize_consistent_story_tags(prompt_info)

        self.assertTrue(prioritized["prompt"].startswith(
            "1girl, solo, petite, centered back hair bun, pink eyes, youthful round face"
        ))
        self.assertLess(
            prioritized["prompt"].index("pink eyes"),
            prioritized["prompt"].index("standing"),
        )
        self.assertLess(
            prioritized["prompt"].index("standing"),
            prioritized["prompt"].index("seaside promenade"),
        )
        self.assertLess(
            prioritized["prompt"].index("seaside promenade"),
            prioritized["prompt"].index("white dress"),
        )
        self.assertLess(
            prioritized["prompt"].index("seaside promenade"),
            prioritized["prompt"].index("soft anime illustration"),
        )

    def test_story_priority_compacts_pose_and_identity_before_scene(self):
        prompt_info = {
            "prompt": (
                "countryside, rice fields, rural road, after school, sunset, "
                "full body, smile, looking at viewer, right hand, (v over eye:1.4), "
                "(v:1.3), peace sign, looking through fingers, hand beside face, "
                "single earring, glass apple earring, 1girl, solo, petite, "
                "small frame, short stature, youthful face, blonde hair, medium hair, "
                "half updo, small braided bun, back bun, white cat hairclip, red eyes, "
                "masterpiece, high score, great score, absurdres"
            )
        }

        prioritized = prioritize_consistent_story_tags(prompt_info)

        self.assertTrue(prioritized["prompt"].startswith(
            "1girl, solo, petite, blonde hair, braided back bun, red eyes, "
            "full body, smile, looking at viewer, right hand, (v over eye:1.4), "
            "single glass apple earring, countryside, rice fields, rural road, "
            "after school, sunset"
        ))
        self.assertNotIn("(v:1.3)", prioritized["prompt"])
        self.assertNotIn("looking through fingers", prioritized["prompt"])
        self.assertNotIn("hand beside face", prioritized["prompt"])

    def test_story_negative_tags_prioritize_duplicate_background_and_body_failures(self):
        settings = {
            "negative_prompt": (
                "low quality, side bun, blue background, long legs, "
                "multiple girls, bad anatomy, curvy"
            )
        }

        prioritize_story_negative_tags(settings)

        self.assertTrue(settings["negative_prompt"].startswith(
            "multiple girls, blue background, long legs, curvy, side bun, bad anatomy"
        ))

    def test_story_prompt_compacts_verbose_lora_identity(self):
        settings = app_module.normalize_generation_settings({
            "preset": "bishoujo_game",
            "negative_prompt": "low quality",
        })
        character = {
            "prompt": (
                "petite proportions, youthful face, shoulder-length blonde hair, "
                "half updo, one small braided bun centered at the back of the head, "
                "white cat-shaped hair ornament, rose crimson eyes, deep reddish-pink irises"
            ),
            "intent_notes": "",
            "source": "test",
        }
        scene = {
            "prompt": "full body, smile, rice fields, evening",
            "intent_notes": "",
            "source": "test",
        }

        result = build_consistent_story_prompt(character, scene, settings)

        self.assertIn("petite", result["prompt"])
        self.assertIn("blonde hair", result["prompt"])
        self.assertIn("medium hair", result["prompt"])
        self.assertIn("small braided bun", result["prompt"])
        self.assertIn("back bun", result["prompt"])
        self.assertIn("white cat hairclip", result["prompt"])
        self.assertIn("red eyes", result["prompt"])
        self.assertNotIn("petite proportions", result["prompt"])
        self.assertNotIn("deep reddish-pink irises", result["prompt"])

    def test_previous_scene_is_excluded_without_blocking_shared_target_tags(self):
        settings = {"negative_prompt": "low quality"}

        excluded = apply_source_scene_exclusion(
            settings,
            "school classroom, large windows, warm sunset, smiling",
            "beach, blue ocean, warm sunset, peace sign",
        )

        self.assertEqual(excluded, ["school classroom", "large windows"])
        self.assertIn("school classroom", settings["negative_prompt"])
        self.assertNotIn("smiling", settings["negative_prompt"])
        self.assertEqual(settings["negative_prompt"].count("warm sunset"), 0)

    def test_lora_dataset_constants_are_negative_unless_requested(self):
        settings = {"negative_prompt": "low quality"}
        metadata = {
            "training_leakage_tags": [
                "white background",
                "black cat-print t-shirt",
                "denim shorts",
            ]
        }

        excluded = apply_lora_leakage_constraints(
            settings,
            "白いサマードレスで海辺に立つ。black cat-print t-shirt",
            metadata,
        )

        self.assertEqual(excluded, ["white background", "denim shorts"])
        self.assertIn("white background", settings["negative_prompt"])
        self.assertNotIn(
            "black cat-print t-shirt",
            settings["negative_prompt"],
        )

    def test_ocean_background_replacement_excludes_indoor_remnants(self):
        settings = {"negative_prompt": "low quality"}
        prompt_info = {"prompt": "seaside, ocean background, daytime"}

        constrained = apply_background_replacement_constraints(
            prompt_info,
            "背景を海に変更。",
            settings,
        )

        self.assertTrue(constrained["prompt"].startswith(
            "(outdoors:1.2), sandy beach, (open ocean:1.3), (ocean horizon:1.2)"
        ))
        self.assertIn("classroom", settings["negative_prompt"])
        self.assertIn("window frame", settings["negative_prompt"])

    def test_new_story_ocean_is_prioritized_without_replacement_wording(self):
        settings = {"negative_prompt": "low quality"}
        prompt_info = {
            "prompt": "summer, seaside promenade, bright day, full body"
        }

        constrained = apply_background_replacement_constraints(
            prompt_info,
            "青い海が見える夏の遊歩道。明るい昼。",
            settings,
        )

        self.assertTrue(constrained["prompt"].startswith(
            "(outdoors:1.2), (open ocean:1.3), (ocean horizon:1.2)"
        ))
        self.assertIn("interior", settings["negative_prompt"])
        self.assertNotIn("classroom", settings["negative_prompt"])

    def test_plain_background_mask_excludes_character(self):
        source = Image.new("RGB", (32, 32), "white")
        for y in range(8, 25):
            for x in range(10, 23):
                source.putpixel((x, y), (180, 30, 60))

        mask = _automatic_background_mask(source)

        self.assertIsNotNone(mask)
        self.assertEqual(mask.getpixel((0, 0)), 255)
        self.assertEqual(mask.getpixel((16, 16)), 0)

    def test_restore_unmasked_pixels_keeps_character_exact(self):
        source = Image.new("RGB", (16, 16), "white")
        edited = Image.new("RGB", (16, 16), (20, 80, 180))
        editable_mask = Image.new("L", (16, 16), 255)
        for y in range(4, 12):
            for x in range(5, 11):
                source.putpixel((x, y), (210, 40, 70))
                editable_mask.putpixel((x, y), 0)

        restored = restore_unmasked_pixels(source, edited, editable_mask)

        self.assertEqual(restored.getpixel((8, 8)), source.getpixel((8, 8)))
        self.assertEqual(restored.getpixel((0, 0)), edited.getpixel((0, 0)))

    def test_semantic_foreground_is_inverted_into_background_mask(self):
        source = Image.new("RGB", (32, 32), "white")
        foreground = Image.new("L", (32, 32), 0)
        for y in range(8, 25):
            for x in range(10, 23):
                foreground.putpixel((x, y), 255)

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.png"
            source.save(source_path)
            with patch("image_edit_workflows._anime_foreground_mask", return_value=foreground):
                _, background = extract_plain_background_mask(source_path)

        self.assertEqual(background.getpixel((0, 0)), 255)
        self.assertEqual(background.getpixel((16, 16)), 0)

    def test_green_screen_removes_background_between_hair_strands(self):
        source = Image.new("RGB", (32, 32), CHARACTER_CHROMA_RGB)
        foreground_alpha = Image.new("L", (32, 32), 0)
        for y in range(6, 28):
            for x in range(8, 25):
                source.putpixel((x, y), (245, 245, 245))
                foreground_alpha.putpixel((x, y), 255)
        # Simulate an ISNet false positive: green background enclosed by hair
        # was classified as foreground.
        source.putpixel((16, 12), CHARACTER_CHROMA_RGB)
        foreground_alpha.putpixel((16, 12), 255)

        with patch(
            "image_edit_workflows._anime_foreground_alpha",
            return_value=foreground_alpha,
        ):
            layer, background = prepare_character_layer(source)

        self.assertEqual(background.getpixel((16, 12)), 255)
        self.assertEqual(layer.getpixel((16, 12))[3], 0)
        self.assertEqual(background.getpixel((12, 20)), 0)
        self.assertEqual(layer.getpixel((12, 20)), (245, 245, 245, 255))

    def test_soft_chroma_edge_is_decontaminated_before_compositing(self):
        source = Image.new("RGB", (12, 12), CHARACTER_CHROMA_RGB)
        foreground_alpha = Image.new("L", (12, 12), 0)
        for y in range(3, 9):
            for x in range(4, 8):
                source.putpixel((x, y), (255, 255, 255))
                foreground_alpha.putpixel((x, y), 255)
        for y in range(3, 9):
            source.putpixel((3, y), (128, 255, 128))
            source.putpixel((8, y), (128, 255, 128))
            foreground_alpha.putpixel((3, y), 128)
            foreground_alpha.putpixel((8, y), 128)

        with patch(
            "image_edit_workflows._anime_foreground_alpha",
            return_value=foreground_alpha,
        ):
            layer, background_mask = prepare_character_layer(source)

        edge_pixel = layer.getpixel((3, 5))
        self.assertAlmostEqual(edge_pixel[3], 128, delta=1)
        self.assertGreaterEqual(edge_pixel[0], 250)
        self.assertGreaterEqual(edge_pixel[1], 250)
        self.assertGreaterEqual(edge_pixel[2], 250)

        composed = restore_unmasked_pixels(
            layer,
            Image.new("RGB", source.size, (0, 0, 255)),
            background_mask,
        )
        red, green, blue = composed.getpixel((3, 5))
        self.assertLessEqual(abs(red - green), 2)
        self.assertGreater(blue, green)

    def test_semantic_confidence_never_makes_character_interior_transparent(self):
        source = Image.new("RGB", (16, 16), CHARACTER_CHROMA_RGB)
        foreground_alpha = Image.new("L", (16, 16), 0)
        for y in range(3, 13):
            for x in range(3, 13):
                source.putpixel((x, y), (245, 160, 190))
                # ISNet confidence is not physical opacity. A moderately
                # confident but solid foreground must remain fully opaque.
                foreground_alpha.putpixel((x, y), 180)
        # A closed zero-confidence false negative on a non-chroma source pixel
        # is a segmentation pinhole, not transparent character material.
        foreground_alpha.putpixel((7, 7), 0)

        with patch(
            "image_edit_workflows._anime_foreground_alpha",
            return_value=foreground_alpha,
        ):
            layer, background_mask = prepare_character_layer(source)

        self.assertEqual(layer.getpixel((8, 8))[3], 255)
        self.assertEqual(background_mask.getpixel((8, 8)), 0)
        self.assertEqual(layer.getpixel((7, 7))[3], 255)
        self.assertEqual(background_mask.getpixel((7, 7)), 0)
        self.assertLess(layer.getpixel((3, 8))[3], 255)

    def test_opaque_green_spill_is_neutralized_without_removing_pixel(self):
        source = Image.new("RGB", (12, 12), CHARACTER_CHROMA_RGB)
        foreground_alpha = Image.new("L", (12, 12), 0)
        for y in range(3, 9):
            for x in range(4, 8):
                source.putpixel((x, y), (24, 210, 84))
                foreground_alpha.putpixel((x, y), 255)

        with patch(
            "image_edit_workflows._anime_foreground_alpha",
            return_value=foreground_alpha,
        ):
            layer, background_mask = prepare_character_layer(source)

        red, green, blue, alpha = layer.getpixel((6, 6))
        self.assertEqual(alpha, 255)
        self.assertEqual(background_mask.getpixel((6, 6)), 0)
        self.assertLessEqual(green, max(red, blue))

    def test_white_external_image_does_not_enable_green_color_key(self):
        source = Image.new("RGB", (16, 16), "white")
        foreground_alpha = Image.new("L", (16, 16), 0)
        for y in range(4, 13):
            for x in range(5, 12):
                foreground_alpha.putpixel((x, y), 255)

        with patch(
            "image_edit_workflows._anime_foreground_alpha",
            return_value=foreground_alpha,
        ):
            foreground = _anime_foreground_mask(source)

        self.assertEqual(foreground.getpixel((8, 8)), 255)
        self.assertEqual(foreground.getpixel((0, 0)), 0)

    def test_transparent_png_reuses_alpha_as_background_mask(self):
        source = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        for y in range(4, 13):
            for x in range(5, 12):
                source.putpixel((x, y), (210, 40, 70, 255))

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.png"
            source.save(source_path)
            with patch(
                "image_edit_workflows._semantic_background_mask",
                side_effect=AssertionError("semantic extraction should not run"),
            ):
                loaded, background = extract_plain_background_mask(source_path)

        self.assertEqual(loaded.getpixel((8, 8)), (210, 40, 70))
        self.assertEqual(background.getpixel((0, 0)), 255)
        self.assertEqual(background.getpixel((8, 8)), 0)

    def test_full_edit_flattens_transparent_png_onto_white(self):
        source = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        for y in range(4, 13):
            for x in range(5, 12):
                source.putpixel((x, y), (210, 40, 70, 255))

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.png"
            source.save(source_path)
            loaded, mask, mask_was_explicit = _load_source_and_mask(
                source_path, None, "full"
            )

        self.assertEqual(loaded.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(loaded.getpixel((8, 8)), (210, 40, 70))
        self.assertEqual(mask.getpixel((0, 0)), 255)
        self.assertEqual(mask.getpixel((8, 8)), 255)
        self.assertFalse(mask_was_explicit)

    def test_background_feather_never_enters_foreground(self):
        background = Image.new("L", (32, 32), 255)
        for y in range(8, 25):
            for x in range(10, 23):
                background.putpixel((x, y), 0)

        feathered = _feather_background_mask(background)

        for y in range(8, 25):
            for x in range(10, 23):
                self.assertEqual(feathered.getpixel((x, y)), 0)
        self.assertEqual(feathered.getpixel((0, 0)), 255)

    def test_background_edit_stops_when_mask_cannot_be_detected(self):
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        for y in range(32):
            for x in range(32):
                pixels[y, x] = (
                    (x * 37 + y * 17) % 256,
                    (x * 11 + y * 43) % 256,
                    (x * 29 + y * 7) % 256,
                )
        source = Image.fromarray(pixels, mode="RGB")

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.png"
            source.save(source_path)

            with (
                patch("image_edit_workflows._semantic_background_mask", return_value=None),
                self.assertRaisesRegex(RuntimeError, "画像全体の編集は行いません"),
            ):
                _load_source_and_mask(source_path, None, "background")

    def test_waifu_background_edit_restores_source_subject(self):
        source = Image.new("RGB", (16, 16), "white")
        mask = Image.new("L", (16, 16), 255)
        for y in range(4, 12):
            for x in range(5, 11):
                source.putpixel((x, y), (210, 40, 70))
                mask.putpixel((x, y), 0)

        class FakePipeline:
            last_kwargs = None

            def __call__(self, **_kwargs):
                self.last_kwargs = _kwargs
                return SimpleNamespace(images=[Image.new("RGB", source.size, (20, 80, 180))])

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.png"
            mask_path = Path(directory) / "mask.png"
            source.save(source_path)
            mask.save(mask_path)

            with (
                patch("image_edit_workflows.fit_prompt_for_sdxl", side_effect=lambda _pipe, prompt: prompt),
                patch("image_edit_workflows._generator", return_value=None),
            ):
                pipeline = FakePipeline()
                result = _edit_waifu(
                    pipeline,
                    "rainy schoolyard",
                    "",
                    source_path,
                    mask_path,
                    seed=1,
                    strength=0.85,
                    callback=None,
                    edit_scope="background",
                    status_callback=None,
                )

        self.assertEqual(result.getpixel((8, 8)), source.getpixel((8, 8)))
        self.assertEqual(result.getpixel((0, 0)), (20, 80, 180))
        self.assertEqual(pipeline.last_kwargs["strength"], 1.0)

    def test_background_api_uses_inpaint_editor_instead_of_separate_t2i(self):
        source = Image.new("RGB", (16, 16), (210, 40, 70))
        mask = Image.new("L", (16, 16), 255)

        def as_base64(image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("ascii")

        class ImmediateThread:
            def __init__(self, target, daemon):
                self.target = target

            def start(self):
                self.target()

        previous_editor_pipe = app_module._STATE["editor_pipe"]
        previous_editor_id = app_module._STATE["editor_id"]
        app_module._STATE["editor_pipe"] = None
        app_module._STATE["editor_id"] = None
        try:
            with (
                patch("app.threading.Thread", ImmediateThread),
                patch("app.load_editor_pipeline", return_value=object()),
                patch("app.edit_image", return_value=Image.new("RGB", source.size, "blue")) as edit,
                patch("app.generate_with_janku", side_effect=AssertionError("background T2I must not run")),
            ):
                client = app_module.app.test_client()
                response = client.post("/api/generate/start", json={
                    "workflow": "compose",
                    "mode": "edit",
                    "prompt": "empty beach background",
                    "free_prompt": "empty beach background",
                    "refine_enabled": False,
                    "source_image": as_base64(source),
                    "background_mask_image": as_base64(mask),
                    "editor_model": "waifu_inpaint_xl",
                    "edit_scope": "background",
                    "edit_strength": 0.85,
                })
                self.assertEqual(response.status_code, 200)
                job_id = response.get_json()["job_id"]
                stream = client.get(f"/api/generate/stream/{job_id}")
                events = [
                    json.loads(line[6:])
                    for line in stream.get_data(as_text=True).splitlines()
                    if line.startswith("data: ")
                ]
        finally:
            app_module._STATE["editor_pipe"] = previous_editor_pipe
            app_module._STATE["editor_id"] = previous_editor_id

        edit.assert_called_once()
        done = next(event for event in events if event["type"] == "done")
        self.assertEqual(done["editor_model"], "waifu_inpaint_xl")

    def test_story_api_generates_once_without_mask_or_editor(self):
        class ImmediateThread:
            def __init__(self, target, daemon):
                self.target = target

            def start(self):
                self.target()

        previous_pipe = app_module._STATE["janku_pipe"]
        app_module._STATE["janku_pipe"] = None
        try:
            with (
                patch("app.threading.Thread", ImmediateThread),
                patch("app.load_janku_pipeline", return_value=object()),
                patch("app.configure_pipeline_reference") as reference,
                patch("app.configure_requested_loras"),
                patch(
                    "app.generate_with_janku",
                    return_value=Image.new("RGB", (32, 32), "blue"),
                ) as generate,
                patch(
                    "app.edit_image",
                    side_effect=AssertionError("story generation must not use an editor"),
                ),
            ):
                client = app_module.app.test_client()
                response = client.post("/api/generate/start", json={
                    "workflow": "character",
                    "mode": "t2i",
                    "generation_intent": "story_illustration",
                    "prompt": "1girl, pink short hair, school uniform",
                    "character_prompt": "1girl, pink short hair, school uniform",
                    "scene_prompt": "beach, blue sea, sunny sky",
                    "free_prompt": "1girl, pink short hair, school uniform",
                    "refine_enabled": False,
                    "width": 32,
                    "height": 32,
                    "steps": 1,
                })
                self.assertEqual(response.status_code, 200)
                job_id = response.get_json()["job_id"]
                stream = client.get(f"/api/generate/stream/{job_id}")
                events = [
                    json.loads(line[6:])
                    for line in stream.get_data(as_text=True).splitlines()
                    if line.startswith("data: ")
                ]
        finally:
            app_module._STATE["janku_pipe"] = previous_pipe

        generate.assert_called_once()
        reference.assert_called_once()
        done = next(event for event in events if event["type"] == "done")
        self.assertEqual(done["generation_intent"], "story_illustration")
        self.assertIsNone(done["background_mask"])
        self.assertFalse(done["reference_used"])

    def test_character_api_generates_batch_sequentially_with_unique_seeds(self):
        class ImmediateThread:
            def __init__(self, target, daemon):
                self.target = target

            def start(self):
                self.target()

        previous_pipe = app_module._STATE["janku_pipe"]
        app_module._STATE["janku_pipe"] = None
        try:
            with (
                patch("app.threading.Thread", ImmediateThread),
                patch("app.load_janku_pipeline", return_value=object()),
                patch("app.configure_pipeline_reference"),
                patch("app.configure_requested_loras"),
                patch(
                    "app.batch_generation_seeds",
                    return_value=[32, 101, 202],
                ),
                patch(
                    "app.generate_with_janku",
                    return_value=Image.new("RGB", (32, 32), "blue"),
                ) as generate,
            ):
                client = app_module.app.test_client()
                response = client.post("/api/generate/start", json={
                    "workflow": "character",
                    "mode": "t2i",
                    "generation_intent": "character_asset",
                    "prompt": "1girl, solo, full body",
                    "character_prompt": "1girl, solo, full body",
                    "refine_enabled": False,
                    "batch_count": 3,
                    "seed": 32,
                    "width": 512,
                    "height": 512,
                    "steps": 10,
                })
                self.assertEqual(response.status_code, 200)
                job_id = response.get_json()["job_id"]
                stream = client.get(f"/api/generate/stream/{job_id}")
                events = [
                    json.loads(line[6:])
                    for line in stream.get_data(as_text=True).splitlines()
                    if line.startswith("data: ")
                ]
        finally:
            app_module._STATE["janku_pipe"] = previous_pipe

        self.assertEqual(generate.call_count, 3)
        self.assertEqual(
            [call.args[2]["seed"] for call in generate.call_args_list],
            [32, 101, 202],
        )
        batch_images = [event for event in events if event["type"] == "batch_image"]
        self.assertEqual(len(batch_images), 3)
        self.assertEqual(
            [event["settings"]["seed"] for event in batch_images],
            [32, 101, 202],
        )
        done = next(event for event in events if event["type"] == "done")
        self.assertEqual(done["batch_count"], 3)
        self.assertNotIn("image", done)

    def test_story_api_applies_character_and_style_loras_independently(self):
        class ImmediateThread:
            def __init__(self, target, daemon):
                self.target = target

            def start(self):
                self.target()

        identity_prompt = (
            "petite proportions, youthful face, blonde hair, medium hair, "
            "half updo, back bun, blue hairclips, pink eyes"
        )
        character_metadata = {
            "id": "a" * 32,
            "name": "Hinata",
            "trigger_word": "hinata_chr",
            "identity_prompt": identity_prompt,
            "identity_negative_prompt": "side bun, bun beside ear",
            "category": "character",
            "status": "ready",
            "model_type": app_module.current_model_type(),
            "training_leakage_tags": [],
        }
        style_metadata = {
            "id": "b" * 32,
            "name": "House Style",
            "trigger_word": "vn_style",
            "identity_prompt": "",
            "category": "style",
            "status": "ready",
            "model_type": app_module.current_model_type(),
            "training_leakage_tags": [],
        }
        previous_pipe = app_module._STATE["janku_pipe"]
        app_module._STATE["janku_pipe"] = None
        try:
            with (
                patch("app.threading.Thread", ImmediateThread),
                patch.object(
                    app_module.LORA_STORE,
                    "read",
                    side_effect=lambda owner, model_id: (
                        character_metadata
                        if model_id == character_metadata["id"]
                        else style_metadata
                    ),
                ),
                patch("app.load_janku_pipeline", return_value=object()),
                patch("app.configure_pipeline_reference"),
                patch("app.configure_requested_loras") as configure_loras,
                patch(
                    "app.generate_with_janku",
                    return_value=Image.new("RGB", (32, 32), "blue"),
                ) as generate,
            ):
                client = app_module.app.test_client()
                response = client.post("/api/generate/start", json={
                    "workflow": "character",
                    "mode": "t2i",
                    "generation_intent": "story_illustration",
                    "prompt": "1girl, white sundress, full body, standing",
                    "character_prompt": "1girl, white sundress, full body, standing",
                    "scene_prompt": "beach, blue sea, sunny sky",
                    "free_prompt": "1girl, white sundress, full body, standing",
                    "character_lora_id": character_metadata["id"],
                    "character_lora_weight": 0.8,
                    "style_lora_id": style_metadata["id"],
                    "style_lora_weight": 0.55,
                    "refine_enabled": False,
                    "width": 32,
                    "height": 32,
                    "steps": 1,
                })
                self.assertEqual(response.status_code, 200)
                job_id = response.get_json()["job_id"]
                stream = client.get(f"/api/generate/stream/{job_id}")
                events = [
                    json.loads(line[6:])
                    for line in stream.get_data(as_text=True).splitlines()
                    if line.startswith("data: ")
                ]
        finally:
            app_module._STATE["janku_pipe"] = previous_pipe

        generated_prompt = generate.call_args.args[1]
        generated_tags = [
            tag.strip() for tag in generated_prompt.split(",") if tag.strip()
        ]
        self.assertLess(
            generated_tags.index("1girl"),
            generated_tags.index("hinata_chr"),
        )
        self.assertLess(
            generated_tags.index("hinata_chr"),
            generated_tags.index("vn_style"),
        )
        self.assertIn("petite", generated_prompt)
        self.assertNotIn("petite proportions", generated_prompt)
        self.assertIn("back bun", generated_prompt)
        self.assertIn("pink eyes", generated_prompt)
        generated_settings = generate.call_args.args[2]
        self.assertIn("side bun", generated_settings["negative_prompt"])
        done = next(event for event in events if event["type"] == "done")
        self.assertEqual(done["lora_identity_prompt"], identity_prompt)
        self.assertEqual(done["character_lora_id"], character_metadata["id"])
        self.assertEqual(done["style_lora_id"], style_metadata["id"])
        self.assertEqual(done["style_lora_weight"], 0.55)
        requested = configure_loras.call_args.args[1]
        self.assertEqual(requested[0]["adapter_name"], "character_asset")
        self.assertEqual(requested[1]["adapter_name"], "style_asset")


if __name__ == "__main__":
    unittest.main()
