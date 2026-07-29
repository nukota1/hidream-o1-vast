import argparse
import base64
import gc
import hmac
import io
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import traceback
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image, ImageEnhance

from image_edit_workflows import (
    EDITOR_MODELS,
    edit_image,
    editor_model_choices,
    extract_plain_background_mask,
    load_editor_pipeline,
    unload_editor,
)
from lora_training import (
    MAX_LORA_IMAGES_BY_CATEGORY,
    RECOMMENDED_IMAGE_COUNTS,
    LoraStore,
    current_model_type,
    is_lora_compatible,
    model_type_label,
)
from prompt_refiner import LocalPromptRefiner
from sdxl_janku_workflow import (
    configure_pipeline_loras,
    configure_pipeline_reference,
    fit_prompt_for_sdxl,
    generate_with_janku,
    load_janku_pipeline,
)

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_REQUEST_BYTES", str(512 * 1024 * 1024))
)
BACKEND_SHARED_SECRET = os.environ.get("BACKEND_SHARED_SECRET", "").strip()


@app.before_request
def require_backend_shared_secret():
    if not BACKEND_SHARED_SECRET:
        return None
    supplied = request.headers.get("X-Backend-Key", "")
    if not supplied or not hmac.compare_digest(supplied, BACKEND_SHARED_SECRET):
        return jsonify({"error": "Backend authentication required."}), 401
    return None


_GEN_LOCK = threading.Lock()
_JOBS = {}
_LORA_JOBS = {}
_STATE = {
    "janku_pipe": None,
    "lora_signature": (),
    "editor_pipe": None,
    "editor_id": None,
    "refiner": LocalPromptRefiner(),
}
_PROMPT_CATALOG = None

IMAGE_MODEL_FAMILY = os.environ.get("IMAGE_MODEL_FAMILY", "janku").strip().lower()
IMAGE_MODEL_LABEL = os.environ.get(
    "IMAGE_MODEL_LABEL",
    "Animagine XL 4.0 Zero" if IMAGE_MODEL_FAMILY == "animagine" else "JANKU v7.77",
)
APP_NAME = os.environ.get("APP_NAME", "Animagine Image Studio" if IMAGE_MODEL_FAMILY == "animagine" else "JANKU Image Studio")
LORA_STORE = LoraStore()

BASE_NEGATIVE_PROMPT = (
    "lowres, worst quality, low quality, bad anatomy, bad hands, extra fingers, "
    "missing fingers, malformed limbs, blurry, jpeg artifacts, text, watermark, signature"
)

PROMPT_CATALOG_PATH = Path(__file__).with_name("ai-nante-prompt-catalog.json")
WORKFLOW_CHARACTER = "character"
WORKFLOW_COMPOSE = "compose"

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
CHARACTER_REQUIRED_TAGS = (
    "solo",
    "simple white background",
    "plain background",
    "isolated",
)
CHARACTER_NEGATIVE_TAGS = (
    *BACKGROUNDLESS_NEGATIVE_TAGS,
    "detailed scenery",
    "busy background",
    "environment",
    "environmental effects",
    "gradient background",
    "gray background",
    "green background",
    "red background",
    "blue background",
    "geometric background",
    "studio backdrop",
    "floor",
    "spotlight",
    "background shadow",
    "multiple views",
    "character sheet",
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
        "average score, signature, watermark, username, blurry, malformed eyes, "
        "asymmetrical eyes, poorly drawn eyes, colored sclera, eye color spill"
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


def split_prompt_tags(prompt):
    return [tag.strip() for tag in re.split(r"[,\n]", str(prompt)) if tag.strip()]


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
        "beach",
        "mountain",
        "river",
        "garden",
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
    if any(term in value for term in scene_terms):
        return True
    return bool(re.search(
        r"\b(?:rural |country |dirt )?road\b|\b(?:open )?field\b|"
        r"\bschool(?:yard| yard| building| campus)\b",
        value,
    ))


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


def stabilize_character_eye_tags(tags):
    """Convert fragile prose-like eye effects into Animagine's stable tags."""
    eye_colors = (
        "red", "crimson", "blue", "green", "amber", "gold", "golden",
        "brown", "gray", "grey", "purple", "violet", "pink", "aqua",
        "cyan", "black", "orange", "yellow",
    )
    risky_terms = (
        "glassy", "glossy", "glass-like", "glass like", "glass refraction",
        "refraction", "transparent eyes", "multiple highlights", "many highlights",
        "detailed eyes", "highly detailed eyes", "catchlight", "vivid tone",
        "light tone layer",
    )
    normalized = []
    detected_color = ""
    eye_requested = False
    eye_insert_index = None
    for raw_tag in tags:
        tag = raw_tag.strip()
        value = tag.lower()
        if "eye" in value or "iris" in value:
            if eye_insert_index is None:
                eye_insert_index = len(normalized)
            for color in eye_colors:
                if re.search(rf"\b{re.escape(color)}\b", value):
                    detected_color = "red" if color == "crimson" else (
                        "gold" if color == "golden" else (
                            "gray" if color == "grey" else color
                        )
                    )
                    eye_requested = True
                    break
            if any(term in value for term in risky_terms):
                continue
            if re.search(r"\bcrimson(?: red)? eyes?\b", value):
                tag = "red eyes"
                value = tag
            elif re.search(r"\b(?:red|blue|green|amber|gold|brown|gray|purple|violet|pink|aqua|cyan|black|orange|yellow) irises\b", value):
                tag = re.sub(r"\birises\b", "eyes", value)
                value = tag
        if value not in {item.lower() for item in normalized}:
            normalized.append(tag)

    if eye_requested:
        color_tag = f"{detected_color} eyes" if detected_color else ""
        normalized = [
            tag for tag in normalized
            if not color_tag or tag.lower() != color_tag.lower()
        ]
        stable = [color_tag] if color_tag else []
        stable.extend(("gradient eyes", "eye highlights"))
        insert_at = min(eye_insert_index or 0, len(normalized))
        normalized[insert_at:insert_at] = stable
    return normalized


def apply_character_constraints(prompt_info, settings):
    """Keep one standalone character on a neutral background without a matte."""
    quality_tags = ["masterpiece", "high score", "great score", "absurdres"]
    preset = STYLE_PRESETS[settings["preset"]]
    style_control_tags = style_adjustment_tags(settings["style"], preset["style"])
    preset_style_tags = preset.get("positive_style_tags", [])
    style_tag_keys = {tag.lower() for tag in [*style_control_tags, *preset_style_tags]}
    blocked_character_tags = (
        "background", "scenery", "landscape", "environment", "weather", "outdoors", "indoors", "room",
    )
    tags = [tag.strip() for tag in prompt_info["prompt"].split(",") if tag.strip()]
    quality = [tag for tag in tags if tag.lower() in {item.lower() for item in quality_tags}]
    character_tags = [
        tag for tag in tags
        if tag.lower() not in {item.lower() for item in quality_tags}
        and tag.lower() not in style_tag_keys
        and not _is_background_scene_tag(tag)
        and not any(blocked in tag.lower() for blocked in blocked_character_tags)
    ]
    character_tags = stabilize_character_eye_tags(character_tags)
    # Keep a compact rendering direction, but leave the CLIP budget primarily
    # for the character facts the user has chosen or written.
    character_style_tags = [
        *style_control_tags[:2],
        *preset_style_tags[:4],
    ]
    prompt_info["prompt"] = join_unique_tags(
        CHARACTER_REQUIRED_TAGS,
        character_tags,
        character_style_tags,
        quality if quality else (quality_tags if IMAGE_MODEL_FAMILY == "animagine" else ()),
    )
    outerwear_terms = ("jacket", "coat", "cape", "cardigan", "hoodie", "blazer")
    unrequested_outerwear = () if any(
        term in prompt_info["prompt"].lower() for term in outerwear_terms
    ) else outerwear_terms
    character_negative = [
        tag.strip()
        for tag in settings["negative_prompt"].split(",")
        if tag.strip() and tag.strip().lower() != "cropped"
    ]
    settings["negative_prompt"] = join_unique_tags(
        character_negative,
        CHARACTER_NEGATIVE_TAGS,
        unrequested_outerwear,
    )
    prompt_info["intent_notes"] = (
        f"{prompt_info.get('intent_notes', '')} Character workflow enforced: "
        "single character on a simple white background; no silhouette mask is generated."
    ).strip()
    return prompt_info


def _load_prompt_catalog():
    global _PROMPT_CATALOG
    if _PROMPT_CATALOG is not None:
        return _PROMPT_CATALOG
    if not PROMPT_CATALOG_PATH.is_file():
        raise RuntimeError("Prompt catalog file is missing.")

    # Accept catalog exports from both UTF-8 and UTF-8-with-BOM editors.
    with PROMPT_CATALOG_PATH.open("r", encoding="utf-8-sig") as f:
        source = json.load(f)

    categories = []
    subcategories_by_category = {}
    records = []
    for category in source.get("categories", []):
        category_id = str(category.get("id") or "uncategorized")
        category_title = str(category.get("title") or category_id)
        before = len(records)
        article_sources = [
            ("", "", article)
            for article in category.get("articles", [])
        ]
        for subcategory in category.get("subcategories", []):
            subcategory_id = str(subcategory.get("id") or "subcategory")
            subcategory_title = str(subcategory.get("title") or subcategory_id)
            article_sources.extend(
                (subcategory_id, subcategory_title, article)
                for article in subcategory.get("articles", [])
            )
        for subcategory_id, subcategory_title, article in article_sources:
            article_id = str(article.get("id") or "article")
            for table_index, table in enumerate(article.get("tables", [])):
                table_title = str(table.get("title") or article.get("title") or category_title)
                group = " / ".join(
                    value for value in (subcategory_title, table_title) if value
                )
                for row_index, row in enumerate(table.get("rows", [])):
                    prompt = str(row.get("prompt") or "").strip()
                    if not prompt:
                        continue
                    name = str(row.get("name") or prompt).strip()
                    description = str(row.get("description") or "").strip()
                    records.append({
                        "id": f"{category_id}:{subcategory_id or 'root'}:{article_id}:{table_index}:{row_index}",
                        "category": category_id,
                        "subcategory": subcategory_id,
                        "subcategory_title": subcategory_title,
                        "name": name,
                        "prompt": prompt,
                        "description": description,
                        "group": group,
                        "search": " ".join((name, prompt, description, group)).casefold(),
                    })
        categories.append({
            "id": category_id,
            "title": category_title,
            "count": len(records) - before,
        })
        subcategories_by_category[category_id] = [
            {
                "id": str(subcategory.get("id") or "subcategory"),
                "title": str(subcategory.get("title") or subcategory.get("id") or "中分類"),
            }
            for subcategory in category.get("subcategories", [])
        ]

    footwear_pattern = re.compile(
        r"\b(?:shoes?|boots?|sneakers?|sandals?|loafers?|slippers?|pumps?|footwear)\b"
        r"|\b(?:high|low|kitten|stiletto)\s+heels?\b",
        re.IGNORECASE,
    )
    footwear_action_pattern = re.compile(
        r"^(?:removing|untying|putting on)\b", re.IGNORECASE
    )
    footwear_records = []
    for item in records:
        prompt_words = item["prompt"].casefold()
        if not footwear_pattern.search(prompt_words) or footwear_action_pattern.search(prompt_words):
            continue
        footwear_records.append({
            **item,
            "id": f"virtual-footwear:{item['id']}",
            "category": "clothing",
            "subcategory": "footwear",
            "subcategory_title": "靴・履物",
            "group": f"靴・履物 / {item['group']}",
        })

    if footwear_records and "clothing" in subcategories_by_category:
        subcategories_by_category["clothing"].append({
            "id": "footwear",
            "title": "靴・履物（横断）",
        })

    _PROMPT_CATALOG = {
        "categories": categories,
        "subcategories": subcategories_by_category,
        "records": records,
        "virtual_records": {("clothing", "footwear"): footwear_records},
    }
    return _PROMPT_CATALOG


def prompt_catalog_results(query="", category_id="", subcategory_id="", limit=96, offset=0):
    catalog = _load_prompt_catalog()
    terms = [term.casefold() for term in str(query).split() if term.strip()]
    matches = []
    total = 0
    source_records = catalog["virtual_records"].get(
        (category_id, subcategory_id), catalog["records"]
    )
    for item in source_records:
        if category_id and item["category"] != category_id:
            continue
        if subcategory_id and item["subcategory"] != subcategory_id:
            continue
        if terms and not all(term in item["search"] for term in terms):
            continue
        if total >= offset and len(matches) < limit:
            matches.append({key: value for key, value in item.items() if key != "search"})
        total += 1
    next_offset = offset + len(matches)
    return {
        "categories": catalog["categories"],
        "subcategories": catalog["subcategories"].get(category_id, []) if category_id else [],
        "items": matches,
        "total": total,
        "next_offset": next_offset,
        "has_more": next_offset < total,
    }


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


def background_inpaint_prompt(prompt):
    """Remove subject-building tags before masked background inpainting."""
    subject_terms = re.compile(
        r"\b(?:1girl|1boy|girl|boy|person|people|character|face|body|hair|eyes?|"
        r"outfit|dress|skirt|shirt|boots?|shoes?|pose|standing|sitting|solo)\b",
        re.IGNORECASE,
    )
    environment_tags = []
    for tag in re.split(r"[,\n]", prompt):
        tag = tag.strip()
        if tag and not subject_terms.search(tag):
            environment_tags.append(tag)
    layout_defaults = [
        "visual novel background",
        "clear foreground space",
        "open center foreground",
        "unobstructed lower center",
        "floor visible at bottom center",
        "empty central aisle extending to bottom edge",
    ]
    camera_terms = (
        "aerial view", "bird's-eye view", "birds-eye view", "top-down",
        "overhead view", "low angle", "high angle", "worm's-eye view",
        "worms-eye view", "dutch angle",
    )
    if not any(term in prompt.lower() for term in camera_terms):
        layout_defaults.extend([
            "eye-level view",
            "ground-level camera",
            "straight-on view",
            "central perspective",
            "horizon at mid-height",
            "clear ground plane in foreground",
        ])
    return join_unique_tags(
        environment_tags,
        (
            "detailed environment",
            "coherent perspective",
            "matching scene lighting",
            "empty background without people",
        ),
        layout_defaults,
    )


def background_prompt_requests_subject_change(prompt):
    """Return whether a background-only request also asks to change the subject."""
    value = str(prompt or "")
    japanese_subject_change = re.compile(
        r"(?:ポーズ|姿勢|表情|顔元|顔つき|髪型|衣装|服装|"
        r"ピース|手を|腕を|脚を|足を)|"
        r"(?:人物|キャラクター).{0,8}(?:変更|削除|消去|変え|直し|修正)"
    )
    english_subject_change = re.compile(
        r"\b(?:pose|posing|gesture|peace sign|facial expression|"
        r"hairstyle|outfit|clothing|raise (?:her|his|their) hand|"
        r"move (?:her|his|their) (?:hand|arm|leg))\b",
        re.IGNORECASE,
    )
    return bool(
        japanese_subject_change.search(value)
        or english_subject_change.search(value)
    )


def apply_event_character_lock(prompt_info, locked_character_prompt, settings):
    """Merge immutable character tags with separately refined event-scene tags."""
    quality_tags = ["masterpiece", "high score", "great score", "absurdres"]
    quality_keys = {tag.lower() for tag in quality_tags}
    preset = STYLE_PRESETS[settings["preset"]]
    style_keys = {
        tag.lower()
        for tag in [
            *preset.get("positive_style_tags", []),
            *style_adjustment_tags(settings["style"], preset["style"]),
        ]
    }
    layout_only = {
        "simple white background", "plain background",
        "pure white background", "white background", "isolated",
        "simple green background", "solid green background", "green background",
    }
    locked_tags = []
    for raw_tag in re.split(r"[,\n]", locked_character_prompt):
        tag = raw_tag.strip()
        value = tag.lower()
        if not tag or value in quality_keys or value in style_keys or value in layout_only:
            continue
        if _is_background_scene_tag(tag):
            continue
        locked_tags.append(tag)

    character_conflicts = re.compile(
        r"\b(?:1girl|1boy|girl|boy|person|character|face|body|skin|hair|eyes?|"
        r"dress|skirt|shirt|blouse|sweater|cardigan|jacket|coat|hoodie|uniform|"
        r"outfit|clothes?|clothing|boots?|shoes?|socks?|stockings?|gloves?|"
        r"hat|ribbon|jewelry|accessor(?:y|ies))\b",
        re.IGNORECASE,
    )
    scene_tags = []
    quality = []
    for raw_tag in re.split(r"[,\n]", prompt_info["prompt"]):
        tag = raw_tag.strip()
        if not tag:
            continue
        if tag.lower() in quality_keys:
            quality.append(tag)
        elif not character_conflicts.search(tag):
            scene_tags.append(tag)

    prompt_info["prompt"] = join_unique_tags(
        locked_tags,
        scene_tags,
        quality if quality else (quality_tags if IMAGE_MODEL_FAMILY == "animagine" else ()),
    )
    settings["negative_prompt"] = join_unique_tags(
        settings["negative_prompt"].split(","),
        ("different outfit", "changed clothes", "alternate costume"),
    )
    prompt_info["intent_notes"] = (
        f"{prompt_info.get('intent_notes', '')} Character and outfit lock applied from the "
        "character-generation prompt; white-background-only tags were removed."
    ).strip()
    return prompt_info


def build_consistent_story_prompt(character_info, scene_info, settings):
    """Merge separately refined identity and scene conditions in priority order."""
    raw_character_tags = [
        tag.strip()
        for tag in re.split(r"[,\n]", character_info["prompt"])
        if tag.strip()
    ]
    hair_pattern = re.compile(
        r"^(pink|blue|red|blonde|black|brown|white|silver|purple|green)\s+"
        r"(short|long|medium(?: length)?)\s+hair$",
        re.IGNORECASE,
    )
    character_tags = []
    for tag in raw_character_tags:
        match = hair_pattern.match(tag)
        if match:
            character_tags.extend((
                f"{match.group(1).lower()} hair",
                f"{match.group(2).lower()} hair",
            ))
        else:
            character_tags.append(tag)
    character_text = ", ".join(character_tags).lower()
    has_subject_count = bool(re.search(
        r"(?:^|,\s*)(?:[1-9](?:girl|boy)s?|multiple (?:girls|boys)|group)(?:,|$)",
        character_text,
    ))
    if not has_subject_count:
        if any(term in character_text for term in ("girl", "female", "woman")):
            character_tags = ["1girl", "solo", *character_tags]
        elif any(term in character_text for term in ("boy", "male", "man")):
            character_tags = ["1boy", "solo", *character_tags]
    locked_character_prompt = join_unique_tags(character_tags)
    merged = apply_event_character_lock(
        {
            "prompt": scene_info["prompt"],
            "intent_notes": scene_info.get("intent_notes", ""),
            "source": scene_info.get("source", ""),
        },
        locked_character_prompt,
        settings,
    )
    merged["intent_notes"] = " ".join(
        value
        for value in (
            character_info.get("intent_notes", ""),
            scene_info.get("intent_notes", ""),
            "Character identity and scene were refined separately; identity tags were placed first.",
        )
        if value
    )
    character_source = character_info.get("source", "")
    scene_source = scene_info.get("source", "")
    merged["source"] = (
        character_source
        if character_source == scene_source
        else " + ".join(value for value in (character_source, scene_source) if value)
    )
    merged["character_prompt"] = locked_character_prompt
    merged["scene_prompt"] = scene_info["prompt"]
    return merged


def prioritize_consistent_story_tags(prompt_info):
    """Keep immutable identity and requested action ahead of optional styling."""
    tags = [
        tag.strip()
        for tag in re.split(r"[,\n]", prompt_info["prompt"])
        if tag.strip()
    ]

    def priority(tag):
        value = tag.lower()
        if re.fullmatch(r"(?:[1-9](?:girl|boy)s?|solo|multiple (?:girls|boys)|group)", value):
            return 0
        if any(term in value for term in (
            "hair", "bun", "updo", "ponytail", "twintail", "braid",
            "eyes", "eye color", "face", "facial", "skin", "freckles", "mole",
            "petite", "short stature", "small frame", "body type", "proportions", "youthful",
            "tall stature", "hair clip", "hairpin",
        )):
            return 1
        if any(term in value for term in (
            "peace sign", "v sign", "(v:", "v over eye", "hand beside face", "two fingers",
            "one hand raised", "standing", "sitting", "kneeling", "crouching",
            "lying", "running", "jumping", "waving", "looking at viewer",
            "full body", "upper body", "portrait", "from behind", "back view",
            "profile", "smile", "expression", "arms ", "hands ",
        )):
            return 2
        if any(term in value for term in (
            "female student", "male student", "school uniform",
            "uniform", "outfit", "dress", "shirt", "skirt", "jacket", "ribbon",
            "shorts", "pants", "shoes", "boots", "socks", "stockings",
            "accessory",
        )):
            return 3
        if any(term in value for term in (
            "visual novel", "anime illustration", "lineart", "line art",
            "shading", "rendering", "lighting", "pastel colors",
            "vivid colors", "highly detailed", "intricate details",
        )):
            return 5
        if any(term in value for term in (
            "masterpiece", "high score", "great score", "absurdres",
        )):
            return 6
        return 4

    prompt_info["prompt"] = join_unique_tags(
        sorted(tags, key=priority),
    )
    return prompt_info


def apply_source_scene_exclusion(settings, source_scene_prompt, target_scene_prompt):
    """Keep a reference image's old setting from overpowering the new scene."""
    source_tags = [
        tag.strip()
        for tag in re.split(r"[,\n]", str(source_scene_prompt or ""))
        if tag.strip()
    ]
    target_tags = {
        tag.strip().lower()
        for tag in re.split(r"[,\n]", str(target_scene_prompt or ""))
        if tag.strip()
    }
    protected_terms = (
        "girl",
        "boy",
        "character",
        "hair",
        "eyes",
        "face",
        "skin",
        "body",
        "outfit",
        "uniform",
        "dress",
        "shirt",
        "skirt",
        "jacket",
        "shoes",
        "boots",
        "accessory",
        "ribbon",
        "smile",
        "smiling",
        "expression",
        "looking",
        "pose",
        "hand",
        "standing",
        "sitting",
        "full body",
        "upper body",
        "masterpiece",
        "high score",
        "great score",
        "absurdres",
    )
    exclusions = [
        tag
        for tag in source_tags
        if tag.lower() not in target_tags
        and not any(term in tag.lower() for term in protected_terms)
    ]
    if exclusions:
        settings["negative_prompt"] = join_unique_tags(
            settings["negative_prompt"].split(","),
            exclusions,
        )
    return exclusions


def apply_lora_leakage_constraints(settings, requested_prompt, lora_metadata):
    """Suppress dataset constants unless the current request explicitly asks for them."""
    raw_tags = (lora_metadata or {}).get("training_leakage_tags") or []
    if isinstance(raw_tags, str):
        raw_tags = split_prompt_tags(raw_tags)
    fixed_exclusions = split_prompt_tags(
        (lora_metadata or {}).get("identity_negative_prompt") or ""
    )
    raw_tags = [*raw_tags, *fixed_exclusions]
    requested = str(requested_prompt or "").lower()
    exclusions = []
    for value in raw_tags:
        tag = str(value or "").strip()
        if not tag or tag.lower() in requested:
            continue
        exclusions.append(tag)
    if exclusions:
        settings["negative_prompt"] = join_unique_tags(
            split_prompt_tags(settings.get("negative_prompt", "")),
            exclusions,
        )
    return exclusions


def apply_background_replacement_constraints(prompt_info, original_prompt, settings):
    """Strengthen explicit whole-background replacements for tag-based SDXL."""
    value = str(original_prompt or "")
    replaces_background = bool(re.search(
        r"(?:背景|background).{0,24}(?:変更|変えて|置き換|replace|change)",
        value,
        re.IGNORECASE,
    ))
    requests_ocean = bool(re.search(
        r"(?:海|海辺|浜辺|ビーチ|\bocean\b|\bsea\b|\bbeach\b)",
        value,
        re.IGNORECASE,
    ))
    if not requests_ocean:
        return prompt_info

    existing_tags = [
        tag.strip()
        for tag in re.split(r"[,\n]", prompt_info["prompt"])
        if tag.strip()
    ]
    ocean_tags = (
        "(outdoors:1.2)",
        "(open ocean:1.3)",
        "(ocean horizon:1.2)",
        "blue sea visible in background",
        "blue sky",
    )
    if replaces_background:
        ocean_tags = (
            "(outdoors:1.2)",
            "sandy beach",
            "(open ocean:1.3)",
            "(ocean horizon:1.2)",
            "blue sea visible in background",
            "blue sky",
        )
    prompt_info["prompt"] = join_unique_tags(ocean_tags, existing_tags)
    settings["negative_prompt"] = join_unique_tags(
        settings["negative_prompt"].split(","),
        (
            "indoors",
            "interior",
        ),
    )
    if replaces_background:
        settings["negative_prompt"] = join_unique_tags(
            settings["negative_prompt"].split(","),
            (
                "classroom",
                "school desk",
                "window",
                "window frame",
            ),
        )
        prompt_info["intent_notes"] = (
            f"{prompt_info.get('intent_notes', '')} Explicit ocean background replacement "
            "was expanded into an outdoor beach setting and indoor remnants were excluded."
        ).strip()
    else:
        prompt_info["intent_notes"] = (
            f"{prompt_info.get('intent_notes', '')} The requested ocean was prioritized "
            "as a visible outdoor background."
        ).strip()
    return prompt_info


def apply_event_instruction_constraints(prompt_info, original_prompt, settings):
    """Stabilize explicit event-CG gestures that a single refined tag may miss."""
    value = str(original_prompt or "")
    requests_peace_sign = bool(re.search(
        r"(?:ピース|Vサイン|\bpeace sign\b|\bv sign\b)",
        value,
        re.IGNORECASE,
    ))
    if not requests_peace_sign:
        return prompt_info

    existing_tags = [
        tag.strip()
        for tag in re.split(r"[,\n]", prompt_info["prompt"])
        if tag.strip()
    ]
    prompt_info["prompt"] = join_unique_tags(
        (
            "(v over eye:1.4)",
            "(v:1.3)",
            "peace sign",
            "hand beside face",
        ),
        existing_tags,
    )
    settings["negative_prompt"] = join_unique_tags(
        settings["negative_prompt"].split(","),
        (
            "hands under chin",
            "both hands on cheeks",
            "clasped hands",
            "hidden hands",
            "shushing",
            "finger to lips",
            "single raised finger",
            "index finger raised",
            "finger to cheek",
            "fingers to cheeks",
            "poking cheeks",
            "hands on cheeks",
        ),
    )
    prompt_info["intent_notes"] = (
        f"{prompt_info.get('intent_notes', '')} Explicit face-level peace-sign gesture "
        "was expanded into model-facing pose tags."
    ).strip()
    return prompt_info


def prepare_prompt(
    user_prompt,
    mode,
    settings,
    refine_enabled,
    workflow=WORKFLOW_CHARACTER,
    supplemental_prompt="",
):
    preset = STYLE_PRESETS[settings["preset"]]
    adjustment_tags = style_adjustment_tags(settings["style"], preset["style"]) if mode == "t2i" else []
    preset_style_tags = preset.get("positive_style_tags", []) if mode == "t2i" else []
    supplemental_tags = split_prompt_tags(supplemental_prompt)
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
                workflow=workflow,
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
                        supplemental_tags,
                        adjustment_tags,
                        preset_style_tags,
                        quality,
                    )
                else:
                    refined["prompt"] = join_unique_tags(
                        refined["prompt"].split(","),
                        supplemental_tags,
                        adjustment_tags,
                        preset_style_tags,
                    )
                refined["intent_notes"] = (
                    f"{refined.get('intent_notes', '')} Applied style controls: "
                    f"{', '.join([*preset_style_tags, *adjustment_tags])}."
                ).strip()
            elif supplemental_tags:
                refined["prompt"] = join_unique_tags(
                    refined["prompt"].split(","),
                    supplemental_tags,
                )
            return refined
        except Exception as exc:
            print(f"[refine] Local refinement failed: {exc}")
            source = "fallback"
            notes = f"Local refinement failed; deterministic style hints were used: {exc}"
    elif not refine_enabled:
        source = "disabled"
        notes = "Prompt enhancement was disabled. The input was already English."

    if IMAGE_MODEL_FAMILY == "animagine" and mode == "t2i":
        quality = ["masterpiece", "high score", "great score", "absurdres"]
        user_tags = [
            tag.strip()
            for tag in re.split(r"[,\n]", user_prompt)
            if tag.strip() and tag.strip().lower() not in {item.lower() for item in quality}
        ]
        prompt = join_unique_tags(
            user_tags,
            supplemental_tags,
            adjustment_tags,
            preset_style_tags,
            quality,
        )
        return {"prompt": prompt, "intent_notes": notes, "source": source}

    parts = [user_prompt]
    if supplemental_prompt:
        parts.append(supplemental_prompt)
    parts.extend(preset_style_tags)
    if preset["prompt_hint"] and not preset_style_tags:
        parts.append(preset["prompt_hint"])
    style_hint = ", ".join([*adjustment_tags, deterministic_style_hint(settings["style"])])
    if style_hint:
        parts.append(style_hint)
    return {"prompt": ", ".join(parts), "intent_notes": notes, "source": source}


def unload_pipeline(name):
    pipe = _STATE.get(name)
    _STATE[name] = None
    if name == "janku_pipe":
        _STATE["lora_signature"] = ()
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


def decode_image_payload(image_b64):
    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("参照画像を読み込めませんでした。") from exc


def request_owner_id():
    return str(request.headers.get("X-App-User-Id") or "local").strip() or "local"


def public_lora(metadata):
    item = LORA_STORE.public(metadata)
    item["compatible"] = is_lora_compatible(metadata)
    return item


def configure_requested_loras(pipe, requested):
    signature = tuple(
        (
            item["metadata"]["id"],
            round(float(item["weight"]), 3),
            item["adapter_name"],
        )
        for item in requested
        if item.get("metadata")
    )
    if _STATE.get("lora_signature") == signature:
        return
    adapters = []
    for item in requested:
        metadata = item.get("metadata")
        if not metadata:
            continue
        adapters.append({
            "weights_path": LORA_STORE.weight_path(
                metadata["_owner_id"],
                metadata["id"],
            ),
            "weight": round(float(item["weight"]), 3),
            "adapter_name": item["adapter_name"],
        })
    configure_pipeline_loras(pipe, adapters)
    _STATE["lora_signature"] = signature


def read_requested_lora(owner_id, model_id, category):
    if not model_id:
        return None
    try:
        metadata = LORA_STORE.read(owner_id, model_id)
    except KeyError as exc:
        raise ValueError("選択したLoRAが見つかりません。") from exc
    if metadata.get("status") != "ready":
        raise RuntimeError(
            "選択したLoRAはまだ学習中、または学習に失敗しています。"
        )
    if not is_lora_compatible(metadata):
        raise TypeError(
            "選択したLoRAは現在の基盤モデルと互換性がありません。"
        )
    if metadata.get("category") != category:
        raise ValueError(
            f"選択したLoRAは{category}カテゴリではありません。"
        )
    metadata["_owner_id"] = owner_id
    return metadata


def insert_lora_triggers(prompt, character_trigger="", style_trigger=""):
    """Follow Animagine ordering: subject, character identity, then style."""
    tags = [tag.strip() for tag in str(prompt or "").split(",") if tag.strip()]
    subject_tags = []
    remaining = []
    for tag in tags:
        if not remaining and tag.lower() in {"1girl", "1boy", "1other", "solo"}:
            subject_tags.append(tag)
        else:
            remaining.append(tag)
    return join_unique_tags(
        subject_tags,
        (character_trigger,) if character_trigger else (),
        (style_trigger,) if style_trigger else (),
        remaining,
    )


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


@app.route("/api/prompt-catalog")
def api_prompt_catalog():
    query = str(request.args.get("q") or "").strip()
    category_id = str(request.args.get("category") or "").strip()
    subcategory_id = str(request.args.get("subcategory") or "").strip()
    limit = clamp_int(request.args.get("limit"), 96, 12, 120)
    offset = clamp_int(request.args.get("offset"), 0, 0, 100000)
    try:
        return jsonify(prompt_catalog_results(query, category_id, subcategory_id, limit, offset))
    except Exception as exc:
        return jsonify({"error": f"Prompt catalog could not be loaded: {exc}"}), 500


@app.route("/api/lora/models")
def api_lora_models():
    owner_id = request_owner_id()
    return jsonify({
        "items": [public_lora(item) for item in LORA_STORE.list(owner_id)],
        "current_model_type": current_model_type(),
        "current_model_label": model_type_label(),
        "cuda_available": torch.cuda.is_available(),
        "recommended_image_counts": RECOMMENDED_IMAGE_COUNTS,
        "max_image_counts": MAX_LORA_IMAGES_BY_CATEGORY,
    })


@app.route("/api/lora/train/start", methods=["POST"])
def api_lora_train_start():
    owner_id = request_owner_id()
    try:
        metadata = LORA_STORE.create(owner_id, request.get_json(force=True) or {})
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"学習データを準備できませんでした: {exc}"}), 500

    job_id = uuid.uuid4().hex
    q = queue.Queue()
    _LORA_JOBS[job_id] = q

    def worker():
        model_id = metadata["id"]
        recent_output = deque(maxlen=16)
        try:
            LORA_STORE.update(owner_id, model_id, status="training", progress=0, error="")
            q.put({
                "type": "status",
                "phase": "training",
                "message": "推論モデルを解放し、LoRA学習を準備しています",
            })
            with _GEN_LOCK:
                _STATE["refiner"].unload_if_cuda()
                unload_pipeline("janku_pipe")
                unload_active_editor()
                command = LORA_STORE.training_command(owner_id, model_id)
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                last_progress = -1
                for raw_line in process.stdout or ():
                    line = raw_line.strip()
                    if not line:
                        continue
                    recent_output.append(line)
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event_type = event.get("type")
                    if event_type == "progress":
                        step = int(event.get("step") or 0)
                        total = max(1, int(event.get("total") or metadata["steps"]))
                        percent = min(99, round(step / total * 100))
                        if percent != last_progress:
                            LORA_STORE.update(
                                owner_id,
                                model_id,
                                progress=percent,
                                last_loss=event.get("loss"),
                            )
                            last_progress = percent
                        q.put(event)
                    elif event_type == "status":
                        q.put({
                            "type": "status",
                            "phase": "training",
                            "message": str(event.get("message") or "LoRAを学習しています"),
                        })
                return_code = process.wait()
                weight_path = LORA_STORE.weight_path(owner_id, model_id)
                if return_code != 0 or not weight_path.is_file():
                    tail = "\n".join(recent_output)
                    raise RuntimeError(
                        "LoRA学習プロセスが失敗しました。"
                        + (f"\n{tail[-2000:]}" if tail else "")
                    )
            completed = LORA_STORE.update(
                owner_id,
                model_id,
                status="ready",
                progress=100,
                completed_at=datetime.now(timezone.utc).isoformat(),
                error="",
            )
            q.put({"type": "done", "model": public_lora(completed)})
        except Exception as exc:
            traceback.print_exc()
            failed = LORA_STORE.update(
                owner_id,
                model_id,
                status="failed",
                error=str(exc)[-2000:],
            )
            q.put({
                "type": "error",
                "message": str(exc),
                "model": public_lora(failed),
            })
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id, "model": public_lora(metadata)})


@app.route("/api/lora/train/stream/<job_id>")
def api_lora_train_stream(job_id):
    q = _LORA_JOBS.get(job_id)
    if q is None:
        return jsonify({"error": "Unknown LoRA training job"}), 404

    def generate_events():
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            _LORA_JOBS.pop(job_id, None)

    return Response(
        generate_events(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/generate/start", methods=["POST"])
def api_generate_start():
    data = request.get_json(force=True)
    free_prompt = str(data.get("free_prompt") or "").strip()[:5000]
    catalog_prompt = str(data.get("catalog_prompt") or "").strip()[:5000]
    prompt = str(
        data.get("prompt")
        or "\n".join(value for value in (free_prompt, catalog_prompt) if value)
    ).strip()
    mode = str(data.get("mode") or "t2i")
    workflow = str(data.get("workflow") or WORKFLOW_CHARACTER)
    character_prompt = str(data.get("character_prompt") or "").strip()[:5000]
    scene_prompt = str(data.get("scene_prompt") or "").strip()[:5000]
    source_scene_prompt = str(data.get("source_scene_prompt") or "").strip()[:5000]
    generation_intent = str(data.get("generation_intent") or "").strip()
    if not generation_intent:
        if mode == "edit":
            generation_intent = "manual_edit"
        elif workflow == WORKFLOW_COMPOSE:
            generation_intent = "consistent_regeneration"
        elif scene_prompt:
            generation_intent = "story_illustration"
        else:
            generation_intent = "character_asset"
    if generation_intent not in {
        "character_asset",
        "story_illustration",
        "consistent_regeneration",
        "manual_edit",
    }:
        return jsonify({"error": f"Unknown generation intent: {generation_intent}"}), 400
    if not prompt:
        return jsonify({"error": "プロンプトを入力してください。"}), 400
    if mode not in {"t2i", "edit"}:
        return jsonify({"error": f"Unknown mode: {mode}"}), 400
    if workflow not in {WORKFLOW_CHARACTER, WORKFLOW_COMPOSE}:
        return jsonify({"error": f"Unknown workflow: {workflow}"}), 400
    if workflow == WORKFLOW_CHARACTER and mode != "t2i":
        return jsonify({"error": "Character workflow must use text-to-image generation."}), 400
    if workflow == WORKFLOW_COMPOSE and mode not in {"edit", "t2i"}:
        return jsonify({"error": "Compose workflow must use image editing or integrated generation."}), 400
    if workflow == WORKFLOW_CHARACTER and not character_prompt:
        character_prompt = prompt
    if mode == "t2i" and generation_intent in {
        "story_illustration",
        "consistent_regeneration",
    } and not character_prompt:
        return jsonify({"error": "保持するキャラクター定義を入力してください。"}), 400
    if generation_intent == "story_illustration" and not scene_prompt:
        return jsonify({"error": "一枚絵として生成する背景・シーンを入力してください。"}), 400
    source_image = str(data.get("source_image") or "").strip()
    if mode == "edit" and not source_image:
        return jsonify({"error": "編集元の画像が必要です。"}), 400
    reference_image_b64 = str(data.get("reference_image") or "").strip()
    reference_strength = clamp_float(
        data.get("reference_strength"),
        float(os.environ.get("IP_ADAPTER_DEFAULT_SCALE", "0.25")),
        0.0,
        1.0,
    )

    settings = normalize_generation_settings(data)
    refine_enabled = bool(data.get("refine_enabled", True))
    lock_character_outfit = bool(data.get(
        "lock_character_outfit",
        generation_intent in {"story_illustration", "consistent_regeneration"},
    ))
    locked_character_prompt = str(
        data.get("locked_character_prompt") or character_prompt
    ).strip()[:5000]
    mask_b64 = str(data.get("mask_image") or "").strip()
    background_mask_b64 = str(data.get("background_mask_image") or "").strip()
    edit_strength = clamp_float(data.get("edit_strength"), 0.55, 0.10, 0.95)
    edit_scope = str(data.get("edit_scope") or "background")
    if edit_scope not in {"background", "full"}:
        return jsonify({"error": f"Unknown edit scope: {edit_scope}"}), 400
    if (
        workflow == WORKFLOW_COMPOSE
        and mode == "edit"
        and edit_scope == "background"
        and background_prompt_requests_subject_change(prompt)
    ):
        return jsonify({
            "error": (
                "「立ち絵を保持して背景だけ変更」では人物ピクセルを固定するため、"
                "ポーズ・表情・衣装は変更できません。"
                "「ポーズ・構図を含めてイベントCGを再生成」を選択してください。"
            )
        }), 400
    editor_id = str(data.get("editor_model") or "waifu_inpaint_xl")
    if mode == "edit" and editor_id not in EDITOR_MODELS:
        return jsonify({"error": f"Unknown image editor: {editor_id}"}), 400
    owner_id = request_owner_id()
    selected_character_lora = None
    selected_style_lora = None
    character_lora_id = str(
        data.get("character_lora_id") or data.get("lora_id") or ""
    ).strip()
    style_lora_id = str(data.get("style_lora_id") or "").strip()
    character_lora_weight = clamp_float(
        data.get("character_lora_weight", data.get("lora_weight")),
        0.8,
        0.0,
        1.5,
    )
    style_lora_weight = clamp_float(
        data.get("style_lora_weight"),
        0.6,
        0.0,
        1.5,
    )
    if mode == "t2i":
        try:
            selected_character_lora = read_requested_lora(
                owner_id,
                character_lora_id,
                "character",
            )
            selected_style_lora = read_requested_lora(
                owner_id,
                style_lora_id,
                "style",
            )
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
    job_id = uuid.uuid4().hex
    q = queue.Queue()
    _JOBS[job_id] = q

    def worker():
        temp_paths = []
        generated_background_mask = ""
        try:
            q.put({"type": "status", "phase": "refine", "message": "プロンプトを準備しています"})
            lora_identity_prompt = (
                str(selected_character_lora.get("identity_prompt") or "").strip()
                if selected_character_lora
                else ""
            )
            effective_character_prompt = character_prompt
            if lora_identity_prompt and mode == "t2i":
                effective_character_prompt = "\n".join(
                    value
                    for value in (lora_identity_prompt, character_prompt)
                    if value
                )
            priority_prompt = free_prompt or prompt
            supplemental_prompt = catalog_prompt if free_prompt else ""
            optimized_character_prompt = ""
            optimized_scene_prompt = ""
            uses_separated_story_prompt = (
                mode == "t2i"
                and generation_intent in {
                    "story_illustration",
                    "consistent_regeneration",
                }
            )
            if (
                lora_identity_prompt
                and mode == "t2i"
                and not uses_separated_story_prompt
            ):
                priority_prompt = "\n".join(
                    value
                    for value in (lora_identity_prompt, priority_prompt)
                    if value
                )
            if uses_separated_story_prompt:
                character_supplemental = (
                    supplemental_prompt if workflow == WORKFLOW_CHARACTER else ""
                )
                scene_supplemental = (
                    supplemental_prompt if workflow == WORKFLOW_COMPOSE else ""
                )
                scene_request = (
                    scene_prompt
                    if generation_intent == "story_illustration"
                    else priority_prompt
                )
                character_info = prepare_prompt(
                    effective_character_prompt,
                    "t2i",
                    settings,
                    refine_enabled,
                    workflow="story_character",
                    supplemental_prompt=character_supplemental,
                )
                scene_info = prepare_prompt(
                    scene_request,
                    "t2i",
                    settings,
                    refine_enabled,
                    workflow="story_scene",
                    supplemental_prompt=scene_supplemental,
                )
                prompt_info = build_consistent_story_prompt(
                    character_info,
                    scene_info,
                    settings,
                )
                optimized_character_prompt = character_info["prompt"]
                optimized_scene_prompt = scene_info["prompt"]
                if (
                    generation_intent == "consistent_regeneration"
                    and source_scene_prompt
                ):
                    excluded_scene_tags = apply_source_scene_exclusion(
                        settings,
                        source_scene_prompt,
                        optimized_scene_prompt,
                    )
                    if excluded_scene_tags:
                        prompt_info["intent_notes"] = (
                            f"{prompt_info.get('intent_notes', '')} "
                            "Previous-scene tags were added to the negative prompt so the "
                            "reference image supplies character identity without restoring "
                            "its old background."
                        ).strip()
            else:
                prompt_workflow = (
                    "compose_background"
                    if workflow == WORKFLOW_COMPOSE and edit_scope == "background"
                    else workflow
                )
                prompt_info = prepare_prompt(
                    priority_prompt,
                    mode,
                    settings,
                    refine_enabled,
                    workflow=prompt_workflow,
                    supplemental_prompt=supplemental_prompt,
                )
            if workflow == WORKFLOW_CHARACTER and not uses_separated_story_prompt:
                prompt_info = apply_character_constraints(prompt_info, settings)
            elif (
                workflow == WORKFLOW_COMPOSE
                and mode == "t2i"
                and not uses_separated_story_prompt
                and lock_character_outfit
                and locked_character_prompt
            ):
                prompt_info = apply_event_character_lock(
                    prompt_info, locked_character_prompt, settings
                )
            elif (
                workflow == WORKFLOW_COMPOSE
                and mode == "edit"
                and edit_scope == "background"
            ):
                prompt_info["prompt"] = background_inpaint_prompt(prompt_info["prompt"])
                prompt_info["intent_notes"] = (
                    f"{prompt_info.get('intent_notes', '')} Background-only inpainting: "
                    "subject-building tags were removed to prevent duplicate characters."
                ).strip()
            elif background_suppression_requested(prompt, mode):
                prompt_info = apply_background_suppression(prompt_info, settings)
            if mode == "t2i" and generation_intent in {
                "story_illustration",
                "consistent_regeneration",
            }:
                prompt_info = apply_background_replacement_constraints(
                    prompt_info,
                    "\n".join(
                        value
                        for value in (scene_prompt, prompt)
                        if value
                    ),
                    settings,
                )
                prompt_info = apply_event_instruction_constraints(
                    prompt_info,
                    "\n".join(
                        value
                        for value in (character_prompt, scene_prompt, prompt)
                        if value
                    ),
                    settings,
                )
                prompt_info = prioritize_consistent_story_tags(prompt_info)
            if (
                selected_character_lora or selected_style_lora
            ) and mode == "t2i":
                prompt_info["prompt"] = insert_lora_triggers(
                    prompt_info["prompt"],
                    (
                        selected_character_lora["trigger_word"]
                        if selected_character_lora
                        else ""
                    ),
                    (
                        selected_style_lora["trigger_word"]
                        if selected_style_lora
                        else ""
                    ),
                )
                request_text = "\n".join(
                    value
                    for value in (character_prompt, scene_prompt, prompt)
                    if value
                )
                leakage_exclusions = []
                if selected_character_lora:
                    leakage_exclusions.extend(apply_lora_leakage_constraints(
                        settings,
                        request_text,
                        selected_character_lora,
                    ))
                    prompt_info["intent_notes"] = (
                        f"{prompt_info.get('intent_notes', '')} "
                        f"Character LoRA: {selected_character_lora['name']} "
                        f"({selected_character_lora['trigger_word']}, "
                        f"weight {character_lora_weight:.2f})."
                    ).strip()
                    if lora_identity_prompt:
                        prompt_info["intent_notes"] = (
                            f"{prompt_info['intent_notes']} "
                            "The LoRA fixed identity profile was applied before scene styling."
                        )
                if selected_style_lora:
                    leakage_exclusions.extend(apply_lora_leakage_constraints(
                        settings,
                        request_text,
                        selected_style_lora,
                    ))
                    prompt_info["intent_notes"] = (
                        f"{prompt_info.get('intent_notes', '')} "
                        f"Style LoRA: {selected_style_lora['name']} "
                        f"({selected_style_lora['trigger_word']}, "
                        f"weight {style_lora_weight:.2f})."
                    ).strip()
                if leakage_exclusions:
                    prompt_info["intent_notes"] = (
                        f"{prompt_info['intent_notes']} "
                        "Training-set constants suppressed: "
                        f"{', '.join(dict.fromkeys(leakage_exclusions))}."
                    )
            if reference_image_b64 and mode == "t2i":
                prompt_info["intent_notes"] = (
                    f"{prompt_info.get('intent_notes', '')} "
                    f"Character reference image applied at weight {reference_strength:.2f}; "
                    "the reference supplies appearance while text controls requested changes."
                ).strip()
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
                    scene_mask_path = None
                    if background_mask_b64:
                        scene_mask_path = decode_image_to_temp(background_mask_b64, "janku_background_mask")
                        temp_paths.append(scene_mask_path)
                    source = Image.open(source_path).convert("RGB")
                    plain_background_mask = None
                    if edit_scope == "background" and not mask_path and not scene_mask_path:
                        source, plain_background_mask = extract_plain_background_mask(source_path)
                    if scene_mask_path:
                        plain_background_mask = Image.open(scene_mask_path).convert("L").resize(source.size)
                    effective_mask_path = mask_path
                    if edit_scope == "background" and effective_mask_path is None and scene_mask_path:
                        effective_mask_path = scene_mask_path
                    reusable_mask = None
                    if edit_scope == "background":
                        if effective_mask_path:
                            reusable_mask = Image.open(effective_mask_path).convert("L").resize(source.size)
                        else:
                            reusable_mask = plain_background_mask
                        if reusable_mask is None:
                            raise RuntimeError(
                                "無地背景を自動抽出できませんでした。人物を保護するため処理を中止しました。"
                                "白い領域を背景にした編集マスクを指定してください。"
                            )
                        if effective_mask_path is None:
                            effective_mask_path = os.path.join(
                                tempfile.gettempdir(), f"janku_semantic_mask_{uuid.uuid4().hex}.png"
                            )
                            reusable_mask.save(effective_mask_path)
                            temp_paths.append(effective_mask_path)
                    if reusable_mask is not None:
                        mask_buffer = io.BytesIO()
                        reusable_mask.save(mask_buffer, format="PNG")
                        generated_background_mask = base64.b64encode(mask_buffer.getvalue()).decode("ascii")
                    if edit_scope == "background":
                        q.put({
                            "type": "status",
                            "phase": "generate",
                            "message": (
                                f"{EDITOR_MODELS[editor_id]['label']} で人物を保護しながら"
                                "背景をinpaintしています"
                            ),
                        })
                        unload_pipeline("janku_pipe")
                        if _STATE["editor_id"] != editor_id:
                            unload_active_editor()
                        if _STATE["editor_pipe"] is None:
                            def editor_status(message):
                                q.put({
                                    "type": "status",
                                    "phase": "generate",
                                    "message": message,
                                })
                            _STATE["editor_pipe"] = load_editor_pipeline(
                                editor_id,
                                status_callback=editor_status,
                            )
                            _STATE["editor_id"] = editor_id
                        image = edit_image(
                            editor_id,
                            _STATE["editor_pipe"],
                            prompt_info["prompt"],
                            settings["negative_prompt"],
                            source_path,
                            effective_mask_path,
                            settings["seed"],
                            strength=edit_strength,
                            callback=progress,
                            status_callback=lambda message: q.put({
                                "type": "status",
                                "phase": "generate",
                                "message": message,
                            }),
                            edit_scope=edit_scope,
                        )
                    else:
                        q.put({
                            "type": "status",
                            "phase": "generate",
                            "message": f"{EDITOR_MODELS[editor_id]['label']} で元画像になじむように編集しています",
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
                            settings["negative_prompt"],
                            source_path,
                            effective_mask_path,
                            settings["seed"],
                            strength=edit_strength,
                            callback=progress,
                            status_callback=lambda message: q.put({"type": "status", "phase": "generate", "message": message}),
                            edit_scope=edit_scope,
                        )
                else:
                    q.put({"type": "status", "phase": "generate", "message": f"{IMAGE_MODEL_LABEL}を準備しています"})
                    unload_active_editor()
                    if _STATE["janku_pipe"] is None:
                        def model_status(message):
                            q.put({"type": "status", "phase": "generate", "message": message})
                        _STATE["janku_pipe"] = load_janku_pipeline(status_callback=model_status)
                    reference_image = (
                        decode_image_payload(reference_image_b64)
                        if reference_image_b64
                        else None
                    )
                    configure_pipeline_reference(
                        _STATE["janku_pipe"],
                        enabled=reference_image is not None,
                        weight=reference_strength,
                        status_callback=lambda message: q.put({
                            "type": "status",
                            "phase": "generate",
                            "message": message,
                        }),
                    )
                    configure_requested_loras(
                        _STATE["janku_pipe"],
                        [
                            {
                                "metadata": selected_character_lora,
                                "weight": character_lora_weight,
                                "adapter_name": "character_asset",
                            },
                            {
                                "metadata": selected_style_lora,
                                "weight": style_lora_weight,
                                "adapter_name": "style_asset",
                            },
                        ],
                    )
                    # Keep the ordered user facts first and reserve room for the
                    # model's quality suffix. The pipeline still uses 77-token
                    # CLIP windows even when its long-prompt helper is enabled.
                    fitted_prompt = fit_prompt_for_sdxl(_STATE["janku_pipe"], prompt_info["prompt"])
                    if fitted_prompt != prompt_info["prompt"]:
                        prompt_info["prompt"] = fitted_prompt
                        q.put({
                            "type": "status",
                            "phase": "generate",
                            "message": "重要な要素と品質タグを優先し、モデルに収まる長さへ最適化しました",
                        })
                    image = generate_with_janku(
                        _STATE["janku_pipe"],
                        prompt_info["prompt"],
                        settings,
                        callback=progress,
                        reference_image=reference_image,
                    )
                    image = apply_image_style_tone(image, settings)

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            q.put({
                "type": "done",
                "image": base64.b64encode(buf.getvalue()).decode("ascii"),
                "original_prompt": prompt,
                "free_prompt": free_prompt or None,
                "catalog_prompt": catalog_prompt or None,
                "optimized_prompt": prompt_info["prompt"],
                "optimizer_source": prompt_info.get("source", ""),
                "intent_notes": prompt_info.get("intent_notes", ""),
                "settings": settings,
                "image_model_profile": current_model_type(),
                "image_model_label": IMAGE_MODEL_LABEL,
                "generation_intent": generation_intent,
                "character_prompt": character_prompt or None,
                "scene_prompt": (
                    scene_prompt
                    if generation_intent == "story_illustration"
                    else (prompt if generation_intent == "consistent_regeneration" else None)
                ),
                "optimized_character_prompt": optimized_character_prompt or None,
                "optimized_scene_prompt": optimized_scene_prompt or None,
                "reference_used": bool(reference_image_b64 and mode == "t2i"),
                "reference_strength": (
                    reference_strength
                    if reference_image_b64 and mode == "t2i"
                    else None
                ),
                "source_scene_prompt": source_scene_prompt or None,
                "editor_model": editor_id if mode == "edit" else None,
                "edit_strength": edit_strength if mode == "edit" else None,
                "edit_scope": edit_scope if mode == "edit" else None,
                "background_mask": generated_background_mask or None,
                # Legacy Character-LoRA fields are retained for existing
                # gallery records and Cloudflare clients.
                "lora_id": (
                    selected_character_lora["id"]
                    if selected_character_lora
                    else None
                ),
                "lora_name": (
                    selected_character_lora["name"]
                    if selected_character_lora
                    else None
                ),
                "lora_trigger_word": (
                    selected_character_lora["trigger_word"]
                    if selected_character_lora
                    else None
                ),
                "lora_identity_prompt": (
                    selected_character_lora.get("identity_prompt")
                    if selected_character_lora
                    else None
                ),
                "lora_weight": (
                    character_lora_weight if selected_character_lora else None
                ),
                "character_lora_id": (
                    selected_character_lora["id"]
                    if selected_character_lora
                    else None
                ),
                "character_lora_name": (
                    selected_character_lora["name"]
                    if selected_character_lora
                    else None
                ),
                "character_lora_trigger_word": (
                    selected_character_lora["trigger_word"]
                    if selected_character_lora
                    else None
                ),
                "character_lora_weight": (
                    character_lora_weight if selected_character_lora else None
                ),
                "style_lora_id": (
                    selected_style_lora["id"] if selected_style_lora else None
                ),
                "style_lora_name": (
                    selected_style_lora["name"] if selected_style_lora else None
                ),
                "style_lora_trigger_word": (
                    selected_style_lora["trigger_word"]
                    if selected_style_lora
                    else None
                ),
                "style_lora_weight": (
                    style_lora_weight if selected_style_lora else None
                ),
                "applied_loras": [
                    {
                        "id": metadata["id"],
                        "name": metadata["name"],
                        "category": category,
                        "trigger_word": metadata["trigger_word"],
                        "weight": weight,
                        "model_type": metadata["model_type"],
                    }
                    for metadata, category, weight in (
                        (
                            selected_character_lora,
                            "character",
                            character_lora_weight,
                        ),
                        (
                            selected_style_lora,
                            "style",
                            style_lora_weight,
                        ),
                    )
                    if metadata
                ],
                "refine_enabled": refine_enabled,
                "workflow": workflow,
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
