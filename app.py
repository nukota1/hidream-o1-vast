import argparse
import base64
import gc
import io
import json
import os
import queue
import re
import tempfile
import threading
import traceback
import uuid
from datetime import datetime, timezone

import torch
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from PIL import ImageEnhance

from image_edit_workflows import EDITOR_MODELS, edit_image, editor_model_choices, load_editor_pipeline, unload_editor
from prompt_refiner import LocalPromptRefiner
from sdxl_janku_workflow import (
    fit_prompt_for_sdxl,
    generate_with_janku,
    load_janku_pipeline,
)

load_dotenv()

app = Flask(__name__)
_GEN_LOCK = threading.Lock()
_JOBS = {}
_STATE = {
    "janku_pipe": None,
    "editor_pipe": None,
    "editor_id": None,
    "refiner": LocalPromptRefiner(),
}

IMAGE_MODEL_FAMILY = os.environ.get("IMAGE_MODEL_FAMILY", "janku").strip().lower()
IMAGE_MODEL_LABEL = os.environ.get(
    "IMAGE_MODEL_LABEL",
    "Animagine XL 4.0 Opt" if IMAGE_MODEL_FAMILY == "animagine" else "JANKU v7.77",
)
APP_NAME = os.environ.get("APP_NAME", "Animagine Image Studio" if IMAGE_MODEL_FAMILY == "animagine" else "JANKU Image Studio")

BASE_NEGATIVE_PROMPT = (
    "lowres, worst quality, low quality, bad anatomy, bad hands, extra fingers, "
    "missing fingers, malformed limbs, blurry, jpeg artifacts, text, watermark, signature"
)

BACKGROUNDLESS_POSITIVE_TAGS = ("simple white background", "plain background", "isolated")
BACKGROUNDLESS_NEGATIVE_TAGS = (
    "scenery",
    "detailed background",
    "landscape",
    "outdoors",
    "indoors",
    "room",
    "street",
    "city",
    "forest",
    "sky",
    "clouds",
    "building",
    "furniture",
    "horizon",
    "road",
    "school",
    "beach",
    "mountain",
    "river",
    "garden",
    "field",
    "alley",
    "rain",
    "snow",
    "puddle",
    "sunset",
    "sunrise",
)
BACKGROUNDLESS_JAPANESE_PATTERNS = (
    r"背景\s*(?:を|は)?\s*(?:描かない|描くな|描かなくて|不要|いらない|なし|無し|省略)",
    r"(?:背景なし|背景無し|無背景|背景不要|背景は不要|背景はいらない)",
    r"(?:白背景|白い背景|単色背景|無地背景)",
)
BACKGROUNDLESS_ENGLISH_PATTERN = re.compile(
    r"\b(?:no|without)\s+(?:a\s+)?background\b|\bbackgroundless\b|\bwhite background\b|\bplain background\b",
    re.IGNORECASE,
)

STYLE_PRESETS = {
    "bishoujo_game": {
        "label": "美少女ゲーム風",
        "prompt_hint": (
            "soft luminous Japanese bishoujo game CG, gentle low-contrast pastel rendering"
        ),
        "positive_style_tags": [
            "soft luminous bishoujo game CG",
            "gentle low contrast",
            "diffuse overcast lighting",
        ],
        "negative_style_tags": [
            "harsh contrast",
            "dramatic shadows",
            "underexposed",
            "crushed blacks",
            "heavy black outlines",
            "hard cel shading",
            "sunny",
            "clear sky",
            "direct sunlight",
        ],
        "width": 832,
        "height": 1216,
        "steps": 32,
        "cfg": 5.0,
        "sampler": "euler",
        "clip_skip": 2,
        "negative_prompt": BASE_NEGATIVE_PROMPT + ", photorealistic, live action, 3d render",
        "style": {"anime_strength": 90, "line_detail": 60, "color_vividness": 65, "background_mood": 82, "photoreal_avoidance": 95},
    },
    "anime_illustration": {
        "label": "アニメイラスト",
        "prompt_hint": (
            "high-quality modern Japanese anime illustration, appealing character art, "
            "clean linework, vivid controlled colors, balanced lighting"
        ),
        "width": 832,
        "height": 1216,
        "steps": 32,
        "cfg": 5.0,
        "sampler": "euler",
        "clip_skip": 2,
        "negative_prompt": BASE_NEGATIVE_PROMPT + ", photorealistic, live action",
        "style": {"anime_strength": 88, "line_detail": 72, "color_vividness": 78, "background_mood": 55, "photoreal_avoidance": 92},
    },
    "manga": {
        "label": "漫画風",
        "prompt_hint": (
            "Japanese monochrome manga illustration, strong ink linework, expressive panel composition, "
            "screentone shading, clear black and white contrast"
        ),
        "width": 832,
        "height": 1216,
        "steps": 30,
        "cfg": 4.0,
        "sampler": "euler",
        "clip_skip": 2,
        "negative_prompt": BASE_NEGATIVE_PROMPT + ", full color, painting, photorealistic",
        "style": {"anime_strength": 75, "line_detail": 92, "color_vividness": 5, "background_mood": 62, "photoreal_avoidance": 96},
    },
    "light_novel": {
        "label": "ライトノベル挿絵風",
        "prompt_hint": (
            "Japanese light novel cover illustration, elegant anime character rendering, "
            "delicate line art, luminous color accents, refined composition"
        ),
        "width": 832,
        "height": 1216,
        "steps": 32,
        "cfg": 5.0,
        "sampler": "euler",
        "clip_skip": 2,
        "negative_prompt": BASE_NEGATIVE_PROMPT + ", photorealistic, live action, 3d render",
        "style": {"anime_strength": 88, "line_detail": 82, "color_vividness": 68, "background_mood": 66, "photoreal_avoidance": 94},
    },
    "custom": {
        "label": "カスタム",
        "prompt_hint": "",
        "width": 1024,
        "height": 1024,
        "steps": 32,
        "cfg": 5.0,
        "sampler": "euler",
        "clip_skip": 2,
        "negative_prompt": BASE_NEGATIVE_PROMPT,
        "style": {"anime_strength": 70, "line_detail": 70, "color_vividness": 65, "background_mood": 60, "photoreal_avoidance": 80},
    },
}


if IMAGE_MODEL_FAMILY == "animagine":
    animagine_negative = (
        "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, "
        "fewer digits, cropped, worst quality, low quality, low score, bad score, "
        "average score, signature, watermark, username, blurry"
    )
    STYLE_PRESETS["bishoujo_game"].update({
        "prompt_hint": (
            "luminous high-end Japanese bishoujo game CG, detailed anime illustration, "
            "soft layered shading, transparent color rendering, refined character art"
        ),
        "positive_style_tags": [
            "high-end Japanese bishoujo game CG",
            "visual novel CG",
            "detailed anime illustration",
            "anime coloring",
            "highly detailed",
            "intricate details",
            "detailed glossy eyes",
            "soft layered shading",
            "transparent color rendering",
            "luminous soft lighting",
            "delicate highlights",
            "rich vivid colors",
        ],
        "negative_style_tags": [
            "flat color",
            "flat shading",
            "muted colors",
            "dull colors",
            "underexposed",
            "crushed blacks",
        ],
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "cfg": 5.0,
        "sampler": "euler_a",
        "clip_skip": 2,
        "negative_prompt": animagine_negative,
        "style": {"anime_strength": 100, "line_detail": 100, "color_vividness": 91, "background_mood": 0, "photoreal_avoidance": 85},
    })
    STYLE_PRESETS["anime_illustration"].update({
        "width": 1024, "height": 1024,
        "steps": 28, "cfg": 5.0, "sampler": "euler_a", "negative_prompt": animagine_negative,
    })
    STYLE_PRESETS["manga"].update({
        "width": 1024, "height": 1024,
        "steps": 28, "cfg": 5.0, "sampler": "euler_a", "negative_prompt": animagine_negative + ", full color",
    })
    STYLE_PRESETS["light_novel"].update({
        "width": 1024, "height": 1024,
        "steps": 28, "cfg": 5.0, "sampler": "euler_a", "negative_prompt": animagine_negative,
    })
    STYLE_PRESETS["custom"].update({
        "steps": 28, "cfg": 5.0, "sampler": "euler_a", "negative_prompt": animagine_negative,
    })


def clamp_int(value, default, low, high):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def clamp_float(value, default, low, high):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def normalize_dimension(value, default):
    value = clamp_int(value, default, 512, 2048)
    return max(512, (value // 64) * 64)


def normalize_style_settings(settings, defaults):
    settings = settings or {}
    return {
        key: clamp_int(settings.get(key), default, 0, 100)
        for key, default in defaults.items()
    }


def normalize_generation_settings(data):
    preset_id = data.get("preset") if data.get("preset") in STYLE_PRESETS else "custom"
    preset = STYLE_PRESETS[preset_id]
    sampler = data.get("sampler", preset["sampler"])
    if sampler not in {"euler_a", "euler"}:
        sampler = preset["sampler"]
    style = normalize_style_settings(data.get("style_settings"), preset["style"])
    negative_prompt = str(data.get("negative_prompt") or preset["negative_prompt"]).strip()
    return {
        "preset": preset_id,
        "width": normalize_dimension(data.get("width"), preset["width"]),
        "height": normalize_dimension(data.get("height"), preset["height"]),
        "steps": clamp_int(data.get("steps"), preset["steps"], 10, 60),
        "cfg": clamp_float(data.get("cfg"), preset["cfg"], 1.0, 12.0),
        "sampler": sampler,
        "clip_skip": clamp_int(data.get("clip_skip"), preset["clip_skip"], 1, 4),
        "seed": clamp_int(data.get("seed"), 32, 0, 2**31 - 1),
        "negative_prompt": style_negative_prompt(
            negative_prompt,
            style,
            preset["style"],
            preset.get("negative_style_tags", []),
        ),
        "style": style,
    }


def describe_style(style):
    return (
        f"anime and visual-novel strength {style['anime_strength']}/100; "
        f"line-art detail {style['line_detail']}/100; "
        f"color vividness {style['color_vividness']}/100; "
        f"background atmosphere {style['background_mood']}/100; "
        f"avoid photorealism {style['photoreal_avoidance']}/100"
    )


def deterministic_style_hint(style):
    hints = []
    if style["anime_strength"] >= 70:
        hints.append("anime illustration")
    if style["line_detail"] >= 70:
        hints.append("clean detailed line art")
    if style["color_vividness"] >= 70:
        hints.append("vivid controlled colors")
    elif style["color_vividness"] <= 20:
        hints.append("monochrome black and white")
    if style["background_mood"] >= 65:
        hints.append("atmospheric detailed background")
    if style["photoreal_avoidance"] >= 70:
        hints.append("2D illustration, not photorealistic")
    return ", ".join(hints)


def _style_tier(value):
    if value >= 80:
        return "high"
    if value >= 55:
        return "normal"
    if value >= 30:
        return "low"
    return "minimal"


def style_adjustment_tags(style, preset_style):
    """Return only intentional deviations from the selected style preset."""
    definitions = {
        "anime_strength": {
            "high": "anime visual novel CG",
            "normal": "anime illustration",
            "low": "semi-realistic illustration",
            "minimal": "painterly illustration",
        },
        "line_detail": {
            "high": "intricate crisp line art",
            "normal": "clean line art",
            "low": "soft line art",
            "minimal": "painterly soft edges",
        },
        "color_vividness": {
            "high": "vivid saturated colors",
            "normal": "balanced colors",
            "low": "restrained colors",
            "minimal": "muted limited palette",
        },
        "background_mood": {
            "high": "detailed atmospheric background",
            "normal": "atmospheric background",
            "low": "simple background",
            "minimal": "minimal background",
        },
        "photoreal_avoidance": {
            "high": "2D anime illustration",
            "normal": "2D illustration",
            "low": "illustration",
            "minimal": "semi-realistic illustration",
        },
    }
    tags = []
    for key, tiers in definitions.items():
        selected = _style_tier(style[key])
        baseline = _style_tier(preset_style[key])
        if selected != baseline:
            tags.append(tiers[selected])
    return tags


def join_unique_tags(*groups):
    tags = []
    seen = set()
    for group in groups:
        for item in group:
            tag = str(item).strip()
            if tag and tag.lower() not in seen:
                tags.append(tag)
                seen.add(tag.lower())
    return ", ".join(tags)


def background_suppression_requested(user_prompt, mode):
    """Treat an explicit no-background request as composition, not style."""
    if mode != "t2i":
        return False
    return (
        any(re.search(pattern, user_prompt) for pattern in BACKGROUNDLESS_JAPANESE_PATTERNS)
        or bool(BACKGROUNDLESS_ENGLISH_PATTERN.search(user_prompt))
    )


def _is_background_scene_tag(tag):
    value = tag.lower().strip()
    scene_terms = (
        "background",
        "scenery",
        "landscape",
        "outdoors",
        "indoors",
        "room",
        "street",
        "city",
        "forest",
        "sky",
        "cloud",
        "building",
        "furniture",
        "horizon",
        "road",
        "school",
        "beach",
        "mountain",
        "river",
        "garden",
        "field",
        "alley",
        "rain",
        "snow",
        "puddle",
        "sunset",
        "sunrise",
        "schoolyard",
        "school yard",
        "classroom",
    )
    return any(term in value for term in scene_terms)


def apply_background_suppression(prompt_info, settings):
    """Make an explicit background ban survive both prompt refinement and SDXL priors."""
    quality_tags = ["masterpiece", "high score", "great score", "absurdres"]
    tags = [tag.strip() for tag in prompt_info["prompt"].split(",") if tag.strip()]
    quality = [tag for tag in tags if tag.lower() in {item.lower() for item in quality_tags}]
    subject_tags = [
        tag for tag in tags
        if tag.lower() not in {item.lower() for item in quality_tags}
        and not _is_background_scene_tag(tag)
    ]
    prompt_info["prompt"] = join_unique_tags(
        subject_tags,
        BACKGROUNDLESS_POSITIVE_TAGS,
        quality if quality else (quality_tags if IMAGE_MODEL_FAMILY == "animagine" else ()),
    )
    settings["negative_prompt"] = join_unique_tags(
        settings["negative_prompt"].split(","),
        BACKGROUNDLESS_NEGATIVE_TAGS,
    )
    prompt_info["intent_notes"] = (
        f"{prompt_info.get('intent_notes', '')} Background suppression applied: "
        "isolated subject on a simple white background."
    ).strip()
    return prompt_info


def style_negative_prompt(negative_prompt, style, preset_style, preset_negative_tags=()):
    """Make each style control affect diffusion conditioning, not just the refiner."""
    additions = list(preset_negative_tags)
    if _style_tier(style["anime_strength"]) != _style_tier(preset_style["anime_strength"]):
        if style["anime_strength"] < 55:
            additions.extend(["chibi", "flat cel shading"])
    if style["line_detail"] < 30:
        additions.extend(["heavy lineart", "sharp outlines"])
    elif style["line_detail"] >= 80 and preset_style["line_detail"] < 80:
        additions.extend(["soft focus", "blurry outlines"])
    if style["color_vividness"] < 30:
        additions.extend(["oversaturated", "neon colors"])
    elif style["color_vividness"] >= 80 and preset_style["color_vividness"] < 80:
        additions.extend(["monochrome", "desaturated"])
    if style["background_mood"] < 30:
        additions.extend(["detailed scenery", "busy background"])
    elif style["background_mood"] >= 80 and preset_style["background_mood"] < 80:
        additions.extend(["empty background", "plain background"])
    if style["photoreal_avoidance"] >= 55:
        additions.extend(["photorealistic", "live action", "3d render"])

    tags = []
    for tag in (item.strip() for item in [*negative_prompt.split(","), *additions] if item.strip()):
        if tag.lower() not in {item.lower() for item in tags}:
            tags.append(tag)
    return ", ".join(tags)


def apply_image_style_tone(image, settings):
    """Apply the selected presentation controls after diffusion has finished."""
    if IMAGE_MODEL_FAMILY != "janku" or settings["preset"] != "bishoujo_game":
        return image

    style = settings["style"]
    # JANKU tends toward hard shadows. The game-CG preset deliberately keeps
    # the character readable under a rainy, overcast scene.
    image = ImageEnhance.Brightness(image).enhance(1.08)
    image = ImageEnhance.Contrast(image).enhance(0.82)
    image = ImageEnhance.Color(image).enhance(0.55 + (style["color_vividness"] * 0.007))
    image = ImageEnhance.Sharpness(image).enhance(0.55 + (style["line_detail"] * 0.0075))
    return image


def prepare_prompt(user_prompt, mode, settings, refine_enabled):
    preset = STYLE_PRESETS[settings["preset"]]
    adjustment_tags = style_adjustment_tags(settings["style"], preset["style"]) if mode == "t2i" else []
    preset_style_tags = preset.get("positive_style_tags", []) if mode == "t2i" else []
    # A broad preset is useful guidance, but detailed preset prose must not
    # displace the user's concrete visual requirements in SDXL's CLIP window.
    refiner_preset_hint = preset["prompt_hint"] if mode == "t2i" else "preserve source image style"
    style_description = describe_style(settings["style"]) if mode == "t2i" else "preserve source image style; no restyling"
    contains_japanese = bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", user_prompt))
    if refine_enabled or contains_japanese:
        try:
            refined = _STATE["refiner"].refine(
                user_prompt=user_prompt,
                mode=mode,
                preset_label=preset["label"],
                preset_hint=refiner_preset_hint,
                style_description=style_description,
                enhance=refine_enabled,
            )
            if preset_style_tags or adjustment_tags:
                # Animagine follows ordered tags. User-requested composition and
                # setting must precede preset styling; quality stays last.
                if IMAGE_MODEL_FAMILY == "animagine":
                    # Animagine requires its quality tags to remain at the end.
                    quality = ["masterpiece", "high score", "great score", "absurdres"]
                    tags = [tag.strip() for tag in refined["prompt"].split(",") if tag.strip()]
                    tags = [tag for tag in tags if tag.lower() not in {item.lower() for item in quality}]
                    refined["prompt"] = join_unique_tags(
                        tags,
                        adjustment_tags,
                        preset_style_tags,
                        quality,
                    )
                else:
                    refined["prompt"] = join_unique_tags(
                        preset_style_tags,
                        adjustment_tags,
                        refined["prompt"].split(","),
                    )
                refined["intent_notes"] = (
                    f"{refined.get('intent_notes', '')} Applied style controls: "
                    f"{', '.join([*preset_style_tags, *adjustment_tags])}."
                ).strip()
            return refined
        except Exception as exc:
            print(f"[refine] Local refinement failed: {exc}")
            source = "fallback"
            notes = f"Local refinement failed; deterministic style hints were used: {exc}"
    elif not refine_enabled:
        source = "disabled"
        notes = "Prompt enhancement was disabled. The input was already English."

    # Keep the selected visual direction before the request in the fallback
    # path so it survives the SDXL CLIP token limit.
    parts = []
    parts.extend(preset_style_tags)
    if preset["prompt_hint"] and not preset_style_tags:
        parts.append(preset["prompt_hint"])
    style_hint = ", ".join([*adjustment_tags, deterministic_style_hint(settings["style"])])
    if style_hint:
        parts.append(style_hint)
    parts.append(user_prompt)
    return {"prompt": ", ".join(parts), "intent_notes": notes, "source": source}


def unload_pipeline(name):
    pipe = _STATE.get(name)
    _STATE[name] = None
    if pipe is not None:
        for method_name in ("maybe_free_model_hooks", "remove_all_hooks"):
            try:
                getattr(pipe, method_name)()
            except Exception:
                pass
        del pipe
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as exc:
            print(f"[cuda] Cache cleanup warning: {exc}")


def unload_active_editor():
    pipe = _STATE.get("editor_pipe")
    _STATE["editor_pipe"] = None
    _STATE["editor_id"] = None
    unload_editor(pipe)


def get_r2_client():
    endpoint_url = os.environ.get("R2_ENDPOINT_URL", "").rstrip("/")
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "")
    if not all([endpoint_url, access_key, secret_key]):
        raise RuntimeError("R2 storage is not configured.")

    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def save_image_to_r2(image_b64, metadata):
    bucket = os.environ.get("R2_BUCKET", "")
    if not bucket:
        raise RuntimeError("R2_BUCKET is not configured.")
    image_bytes = base64.b64decode(image_b64)
    now = datetime.now(timezone.utc)
    image_id = uuid.uuid4().hex
    prefix = f"generated/{now:%Y/%m/%d}/{image_id}"
    image_key = f"{prefix}.png"
    metadata_key = f"{prefix}.json"
    metadata = {
        **(metadata or {}),
        "saved_at": now.isoformat(),
        "image_key": image_key,
        "metadata_key": metadata_key,
    }
    client = get_r2_client()
    client.put_object(Bucket=bucket, Key=image_key, Body=image_bytes, ContentType="image/png")
    client.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )
    return {"bucket": bucket, "image_key": image_key, "metadata_key": metadata_key}


def decode_image_to_temp(image_b64, prefix):
    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",", 1)[1]
    path = os.path.join(tempfile.gettempdir(), f"{prefix}_{uuid.uuid4().hex}.png")
    with open(path, "wb") as f:
        f.write(base64.b64decode(image_b64))
    return path


@app.route("/")
def index():
    refine_default = os.environ.get("PROMPT_REFINE_DEFAULT", "1").lower() not in {"0", "false", "no", "off"}
    return render_template(
        "index.html",
        app_name=APP_NAME,
        image_model_label=IMAGE_MODEL_LABEL,
        presets=STYLE_PRESETS,
        presets_json=json.dumps(STYLE_PRESETS, ensure_ascii=False),
        editor_models=editor_model_choices(),
        editor_models_json=json.dumps(editor_model_choices(), ensure_ascii=False),
        refine_default=refine_default,
    )


@app.route("/api/save-to-r2", methods=["POST"])
def api_save_to_r2():
    data = request.get_json(force=True)
    image_b64 = str(data.get("image") or "").strip()
    if not image_b64:
        return jsonify({"error": "保存できる画像がありません。"}), 400
    try:
        return jsonify(save_image_to_r2(image_b64, data.get("metadata") or {}))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/generate/start", methods=["POST"])
def api_generate_start():
    data = request.get_json(force=True)
    prompt = str(data.get("prompt") or "").strip()
    mode = str(data.get("mode") or "t2i")
    if not prompt:
        return jsonify({"error": "プロンプトを入力してください。"}), 400
    if mode not in {"t2i", "edit"}:
        return jsonify({"error": f"Unknown mode: {mode}"}), 400
    source_image = str(data.get("source_image") or "").strip()
    if mode == "edit" and not source_image:
        return jsonify({"error": "編集元の画像が必要です。"}), 400

    settings = normalize_generation_settings(data)
    refine_enabled = bool(data.get("refine_enabled", True))
    mask_b64 = str(data.get("mask_image") or "").strip()
    edit_strength = clamp_float(data.get("edit_strength"), 0.55, 0.10, 0.95)
    editor_id = str(data.get("editor_model") or "waifu_inpaint_xl")
    if mode == "edit" and editor_id not in EDITOR_MODELS:
        return jsonify({"error": f"Unknown image editor: {editor_id}"}), 400
    job_id = uuid.uuid4().hex
    q = queue.Queue()
    _JOBS[job_id] = q

    def worker():
        temp_paths = []
        try:
            q.put({"type": "status", "phase": "refine", "message": "プロンプトを準備しています"})
            prompt_info = prepare_prompt(prompt, mode, settings, refine_enabled)
            if background_suppression_requested(prompt, mode):
                prompt_info = apply_background_suppression(prompt_info, settings)
            _STATE["refiner"].unload_if_cuda()
            q.put({
                "type": "optimized_prompt",
                "phase": "handoff",
                "prompt": prompt_info["prompt"],
                "intent_notes": prompt_info.get("intent_notes", ""),
                "source": prompt_info.get("source", ""),
            })

            def progress(step, total):
                q.put({"type": "progress", "step": step + 1, "total": total})

            with _GEN_LOCK:
                if mode == "edit":
                    source_path = decode_image_to_temp(source_image, "janku_source")
                    temp_paths.append(source_path)
                    mask_path = None
                    if mask_b64:
                        mask_path = decode_image_to_temp(mask_b64, "janku_mask")
                        temp_paths.append(mask_path)
                    q.put({
                        "type": "status",
                        "phase": "generate",
                        "message": f"{EDITOR_MODELS[editor_id]['label']} を準備しています",
                    })
                    unload_pipeline("janku_pipe")
                    if _STATE["editor_id"] != editor_id:
                        unload_active_editor()
                    if _STATE["editor_pipe"] is None:
                        def editor_status(message):
                            q.put({"type": "status", "phase": "generate", "message": message})
                        _STATE["editor_pipe"] = load_editor_pipeline(editor_id, status_callback=editor_status)
                        _STATE["editor_id"] = editor_id
                    image = edit_image(
                        editor_id,
                        _STATE["editor_pipe"],
                        prompt_info["prompt"],
                        source_path,
                        mask_path,
                        settings["seed"],
                        strength=edit_strength,
                        callback=progress,
                        status_callback=lambda message: q.put({"type": "status", "phase": "generate", "message": message}),
                    )
                else:
                    q.put({"type": "status", "phase": "generate", "message": f"{IMAGE_MODEL_LABEL}を準備しています"})
                    unload_active_editor()
                    if _STATE["janku_pipe"] is None:
                        def model_status(message):
                            q.put({"type": "status", "phase": "generate", "message": message})
                        _STATE["janku_pipe"] = load_janku_pipeline(status_callback=model_status)
                    fitted_prompt = fit_prompt_for_sdxl(_STATE["janku_pipe"], prompt_info["prompt"])
                    if fitted_prompt != prompt_info["prompt"]:
                        prompt_info["prompt"] = fitted_prompt
                        q.put({
                            "type": "status",
                            "phase": "generate",
                            "message": "重要な要素を優先し、モデルに収まる長さへプロンプトを最適化しました",
                        })
                    image = generate_with_janku(
                        _STATE["janku_pipe"],
                        prompt_info["prompt"],
                        settings,
                        callback=progress,
                    )
                    image = apply_image_style_tone(image, settings)

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            q.put({
                "type": "done",
                "image": base64.b64encode(buf.getvalue()).decode("ascii"),
                "original_prompt": prompt,
                "optimized_prompt": prompt_info["prompt"],
                "optimizer_source": prompt_info.get("source", ""),
                "intent_notes": prompt_info.get("intent_notes", ""),
                "settings": settings,
                "editor_model": editor_id if mode == "edit" else None,
                "edit_strength": edit_strength if mode == "edit" else None,
                "refine_enabled": refine_enabled,
            })
        except Exception as exc:
            traceback.print_exc()
            q.put({"type": "error", "message": str(exc)})
        finally:
            for path in temp_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/generate/stream/<job_id>")
def api_generate_stream(job_id):
    q = _JOBS.get(job_id)
    if q is None:
        return jsonify({"error": "Unknown job"}), 404

    def generate_events():
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                if item.get("type") in {"done", "error"}:
                    break
        finally:
            _JOBS.pop(job_id, None)

    return Response(
        generate_events(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main():
    parser = argparse.ArgumentParser(APP_NAME)
    parser.add_argument("--host", default=os.environ.get("APP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("APP_PORT", "7861")))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for image generation.")
    print(f"[app] Serving {APP_NAME} ({IMAGE_MODEL_LABEL}) on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
