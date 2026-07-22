import argparse
import base64
import gc
import hmac
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
    normalize_plain_background,
    unload_editor,
)
from prompt_refiner import LocalPromptRefiner
from sdxl_janku_workflow import (
    fit_prompt_for_sdxl,
    generate_with_janku,
    load_janku_pipeline,
)

load_dotenv()

app = Flask(__name__)
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
_STATE = {
    "janku_pipe": None,
    "editor_pipe": None,
    "editor_id": None,
    "refiner": LocalPromptRefiner(),
}
_PROMPT_CATALOG = None

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
)
CHARACTER_NEGATIVE_TAGS = (
    *BACKGROUNDLESS_NEGATIVE_TAGS,
    "detailed scenery",
    "busy background",
    "environment",
    "environmental effects",
    "gradient background",
    "gray background",
    "coloured background",
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
    for raw_tag in tags:
        tag = raw_tag.strip()
        value = tag.lower()
        if "eye" in value or "iris" in value:
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
        normalized = [*stable, *normalized]
    return normalized


def apply_character_constraints(prompt_info, settings):
    """Keep the character workflow to one character on white without forcing a pose."""
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
        "single character on a plain white background; requested pose and framing preserved."
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
    return join_unique_tags(
        environment_tags,
        (
            "detailed environment",
            "coherent perspective",
            "matching scene lighting",
            "empty background without people",
        ),
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


def prepare_prompt(user_prompt, mode, settings, refine_enabled, workflow=WORKFLOW_CHARACTER):
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

    if IMAGE_MODEL_FAMILY == "animagine" and mode == "t2i":
        quality = ["masterpiece", "high score", "great score", "absurdres"]
        user_tags = [
            tag.strip()
            for tag in re.split(r"[,\n]", user_prompt)
            if tag.strip() and tag.strip().lower() not in {item.lower() for item in quality}
        ]
        prompt = join_unique_tags(
            user_tags,
            adjustment_tags,
            preset_style_tags,
            quality,
        )
        return {"prompt": prompt, "intent_notes": notes, "source": source}

    parts = [user_prompt]
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


@app.route("/api/generate/start", methods=["POST"])
def api_generate_start():
    data = request.get_json(force=True)
    prompt = str(data.get("prompt") or "").strip()
    mode = str(data.get("mode") or "t2i")
    workflow = str(data.get("workflow") or WORKFLOW_CHARACTER)
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
    source_image = str(data.get("source_image") or "").strip()
    if mode == "edit" and not source_image:
        return jsonify({"error": "編集元の画像が必要です。"}), 400

    settings = normalize_generation_settings(data)
    refine_enabled = bool(data.get("refine_enabled", True))
    lock_character_outfit = bool(data.get("lock_character_outfit", False))
    locked_character_prompt = str(data.get("locked_character_prompt") or "").strip()[:5000]
    mask_b64 = str(data.get("mask_image") or "").strip()
    background_mask_b64 = str(data.get("background_mask_image") or "").strip()
    edit_strength = clamp_float(data.get("edit_strength"), 0.55, 0.10, 0.95)
    edit_scope = str(data.get("edit_scope") or "background")
    if edit_scope not in {"background", "full"}:
        return jsonify({"error": f"Unknown edit scope: {edit_scope}"}), 400
    editor_id = str(data.get("editor_model") or "waifu_inpaint_xl")
    if mode == "edit" and editor_id not in EDITOR_MODELS:
        return jsonify({"error": f"Unknown image editor: {editor_id}"}), 400
    job_id = uuid.uuid4().hex
    q = queue.Queue()
    _JOBS[job_id] = q

    def worker():
        temp_paths = []
        generated_background_mask = ""
        try:
            q.put({"type": "status", "phase": "refine", "message": "プロンプトを準備しています"})
            prompt_workflow = (
                ("event_scene" if lock_character_outfit and locked_character_prompt else "event_cg")
                if workflow == WORKFLOW_COMPOSE and mode == "t2i"
                else (
                    "compose_background"
                    if workflow == WORKFLOW_COMPOSE and edit_scope == "background"
                    else workflow
                )
            )
            prompt_info = prepare_prompt(
                prompt, mode, settings, refine_enabled, workflow=prompt_workflow
            )
            if workflow == WORKFLOW_CHARACTER:
                prompt_info = apply_character_constraints(prompt_info, settings)
            elif (
                workflow == WORKFLOW_COMPOSE
                and mode == "t2i"
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
                    if reusable_mask is not None:
                        mask_buffer = io.BytesIO()
                        reusable_mask.save(mask_buffer, format="PNG")
                        generated_background_mask = base64.b64encode(mask_buffer.getvalue()).decode("ascii")
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
                    )
                    image = apply_image_style_tone(image, settings)
                    if workflow == WORKFLOW_CHARACTER:
                        image = normalize_plain_background(image)

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
                "edit_scope": edit_scope if mode == "edit" else None,
                "background_mask": generated_background_mask or None,
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
