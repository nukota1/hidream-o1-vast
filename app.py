"""Minimal, elegant Flask app for HiDream-O1-Image.

Features
--------
- Text-to-image, image editing, and multi-reference subject-driven generation
- Standalone prompt refinement (local Gemma backend or OpenAI-compatible API)
- Server-Sent Events for real-time per-step progress
- Single-file: HTML / CSS / JS all embedded below

Run
---
    python app.py --model_path /path/to/HiDream-O1-Image --model_type full
"""

import argparse
import base64
import io
import json
import os
import queue
import tempfile
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

import torch
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request
from PIL import Image
from transformers import AutoProcessor

load_dotenv()

from models.pipeline import DEFAULT_TIMESTEPS, generate_image
from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from prompt_agent import (
    build_local_agent,
    rewrite_prompt_api,
    rewrite_prompt_local,
)


# ── Globals ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
_GEN_LOCK = threading.Lock()
_STATE = {
    "model": None,
    "processor": None,
    "model_type": "full",
    "agent": None,
}
_JOBS = {}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
AUTO_PROMPT_OPTIMIZE = os.environ.get("AUTO_PROMPT_OPTIMIZE", "1").lower() not in {
    "0", "false", "no", "off"
}

O1_PROMPT_OPTIMIZER_SYSTEM = """
You are a prompt director for HiDream-O1-Image-Dev.
Your job is to convert the user's request, in any language, into one clear English
image-generation prompt that HiDream-O1 can follow.

Rules:
- Preserve the user's intent, subject, composition, clothing, colors, mood, and location.
- Do not invent unrelated places, props, seasons, clothing, species, brands, text, or logos.
- If the user asks for anime, manga, bishoujo game, visual novel, game CG, or illustration,
  make the style explicit: "2D Japanese visual novel game CG illustration, not photorealistic,
  not a real-life photo".
- If the user asks for photo/realistic, keep it photographic. Otherwise choose the style implied
  by the user's wording and keep it explicit.
- Convert abstract or Japanese-specific terms into visible English visual details.
- Turn long bullet lists into a natural, dense single paragraph.
- Keep fine character details, camera direction, pose, weather, background, and material details.
- For O1, use direct visual language and avoid tag soup.
- Output JSON only, with keys "prompt" and "intent_notes".
""".strip()


def _add_special_tokens(tokenizer):
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def _get_tokenizer(processor):
    from transformers import PreTrainedTokenizerBase
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def load_image_model(model_path):
    print(f"[app] Loading checkpoint from {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path)
    dtype_name = os.environ.get("HIDREAM_TORCH_DTYPE", "auto").lower()
    if dtype_name == "auto":
        device_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ""
        dtype_name = "float16" if "v100" in device_name or "tesla v100" in device_name else "bfloat16"
    dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported HIDREAM_TORCH_DTYPE: {dtype_name}")
    print(f"[app] Using torch dtype: {dtype}")
    # NOTE: torch_dtype = torch.float32 will generate more detailed images but with more memory usage
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map="cuda"
    ).eval()
    _add_special_tokens(_get_tokenizer(processor))
    return processor, model


def strip_thinking(text):
    while "<think>" in text and "</think>" in text:
        before, rest = text.split("<think>", 1)
        _, after = rest.split("</think>", 1)
        text = before + after
    return text.strip()


def parse_json_object(text):
    text = strip_thinking(text)
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return json.loads(candidate)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("No JSON object found in optimizer output.")


def fallback_o1_prompt(user_prompt):
    return (
        "Create a 2D Japanese visual novel game CG illustration, not photorealistic, "
        "not a real-life photo. Follow the user's request exactly and preserve every "
        "character, clothing, pose, weather, background, color, and material detail. "
        "Render clean anime line art, expressive eyes, detailed hair, crisp character "
        "design, and a coherent composition. User request: "
        + user_prompt.replace("\n", " ")
    )


def clamp_int(value, default=50, low=0, high=100):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def normalize_style_settings(settings):
    settings = settings or {}
    return {
        "anime_strength": clamp_int(settings.get("anime_strength"), 80),
        "line_detail": clamp_int(settings.get("line_detail"), 70),
        "color_vividness": clamp_int(settings.get("color_vividness"), 65),
        "background_mood": clamp_int(settings.get("background_mood"), 60),
        "photoreal_avoidance": clamp_int(settings.get("photoreal_avoidance"), 85),
    }


def describe_style_settings(settings):
    settings = normalize_style_settings(settings)
    return (
        "User style preference sliders, 0 to 100:\n"
        f"- Anime / visual novel style strength: {settings['anime_strength']}\n"
        f"- Clean line art and fine detail strength: {settings['line_detail']}\n"
        f"- Vivid color and eye-catching rendering strength: {settings['color_vividness']}\n"
        f"- Nostalgic / atmospheric background strength: {settings['background_mood']}\n"
        f"- Avoid photorealism and live-action photography strength: {settings['photoreal_avoidance']}\n"
        "Reflect these preferences in the English prompt without contradicting the user's requested content."
    )


def post_ollama_chat(messages, timeout=180):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 1400,
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_openai_compatible_chat(messages, timeout=180):
    base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model_name = os.environ.get("OPENAI_MODEL", "")
    if not all([base_url, api_key, model_name]):
        raise RuntimeError("OpenAI-compatible prompt optimizer is not configured.")

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1400,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://127.0.0.1:7861",
            "X-Title": "HiDream-O1 Local Prompt Refiner",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_chat_completion(result):
    choices = result.get("choices") or []
    if not choices:
        raise ValueError("No choices returned from prompt optimizer.")
    return choices[0].get("message", {}).get("content", "")


def optimize_prompt_for_o1(user_prompt, mode="t2i", style_settings=None):
    if not AUTO_PROMPT_OPTIMIZE:
        return {
            "prompt": user_prompt,
            "intent_notes": "Automatic O1 prompt optimization is disabled.",
            "source": "disabled",
        }

    user_message = (
        f"Mode: {mode}\n"
        "Rewrite the following request for HiDream-O1-Image. "
        "Respect the user's intended art style and visual priorities.\n\n"
        f"{describe_style_settings(style_settings)}\n\n"
        f"{user_prompt}"
    )
    try:
        result = post_openai_compatible_chat(
            [
                {"role": "system", "content": O1_PROMPT_OPTIMIZER_SYSTEM},
                {"role": "user", "content": user_message},
            ]
        )
        raw = parse_chat_completion(result)
        parsed = parse_json_object(raw)
        optimized = str(parsed.get("prompt", "")).strip()
        if len(optimized) < 40:
            raise ValueError("Optimizer returned a prompt that is too short.")
        return {
            "prompt": optimized,
            "intent_notes": str(parsed.get("intent_notes", "")).strip(),
            "source": os.environ.get("OPENAI_MODEL", "openai-compatible"),
        }
    except Exception as api_exc:
        print(f"[prompt] OpenAI-compatible optimization failed, falling back to Ollama: {api_exc}")
    try:
        result = post_ollama_chat(
            [
                {"role": "system", "content": O1_PROMPT_OPTIMIZER_SYSTEM},
                {"role": "user", "content": user_message},
            ]
        )
        raw = result.get("message", {}).get("content", "")
        parsed = parse_json_object(raw)
        optimized = str(parsed.get("prompt", "")).strip()
        if len(optimized) < 40:
            raise ValueError("Optimizer returned a prompt that is too short.")
        return {
            "prompt": optimized,
            "intent_notes": str(parsed.get("intent_notes", "")).strip(),
            "source": "ollama",
        }
    except Exception as exc:
        print(f"[prompt] O1 prompt optimization failed: {exc}")
        return {
            "prompt": fallback_o1_prompt(user_prompt),
            "intent_notes": f"Fallback prompt wrapper used because optimization failed: {exc}",
            "source": "fallback",
        }


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
    client.put_object(
        Bucket=bucket,
        Key=image_key,
        Body=image_bytes,
        ContentType="image/png",
    )
    client.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )
    return {
        "bucket": bucket,
        "image_key": image_key,
        "metadata_key": metadata_key,
        "endpoint": os.environ.get("R2_ENDPOINT_URL", ""),
    }


# ── HTML ─────────────────────────────────────────────────────────────────────

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>HiDream-O1-Image{% if model_type == 'dev' %}-Dev{% endif %}</title>
<style>
  :root {
    --bg: #fbfbfd;
    --panel: #ffffff;
    --panel-2: #f5f5f7;
    --border: rgba(0, 0, 0, 0.08);
    --border-strong: rgba(0, 0, 0, 0.14);
    --text: #1d1d1f;
    --muted: #86868b;
    --accent: #1d1d1f;
    --accent-soft: #f0f0f2;
    --blue: #0071e3;
    --blue-hover: #0077ed;
    --purple: #8a5cf6;
    --pink: #ff6ea1;
    --mint: #22c9a4;
    --gradient: linear-gradient(135deg, #8a5cf6 0%, #0071e3 45%, #22c9a4 100%);
    --gradient-warm: linear-gradient(135deg, #ff6ea1 0%, #8a5cf6 100%);
    --danger: #d70015;
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.06);
    --radius: 12px;
    --radius-sm: 8px;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
      "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 32px;
    background: rgba(251, 251, 253, 0.85);
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 10;
  }
  .brand {
    font-weight: 600; font-size: 17px; letter-spacing: -0.01em;
    background: var(--gradient);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
  }
  .topbar-meta { color: var(--muted); font-size: 12px; }
  .layout {
    display: grid;
    grid-template-columns: 400px 1fr;
    gap: 24px;
    padding: 24px 32px 40px;
    max-width: 1600px; margin: 0 auto;
  }
  .sidebar, .canvas {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
  }
  .sidebar {
    padding: 24px; max-height: calc(100vh - 130px);
    overflow-y: auto; align-self: start;
  }
  .canvas {
    min-height: calc(100vh - 130px);
    padding: 28px; display: flex; flex-direction: column;
  }
  h2 {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted); margin: 0 0 12px;
  }
  .group { margin-bottom: 22px; }
  .group:last-of-type { margin-bottom: 0; }
  label {
    display: block; font-size: 12px; font-weight: 500;
    color: var(--text); margin-bottom: 8px;
  }
  input[type=text], input[type=number], input[type=password],
  textarea, select {
    width: 100%;
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    font-size: 13px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
    -webkit-appearance: none;
  }
  input:focus, textarea:focus, select:focus {
    border-color: var(--blue);
    box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.15);
  }
  textarea { resize: vertical; min-height: 96px; line-height: 1.5; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .tabs {
    display: flex; gap: 0; margin-bottom: 22px;
    background: var(--panel-2);
    border-radius: 10px; padding: 3px;
  }
  .tab {
    flex: 1; text-align: center; padding: 8px 10px;
    font-size: 12.5px; font-weight: 500;
    color: var(--muted); border-radius: 7px; cursor: pointer;
    user-select: none; transition: all 0.2s;
  }
  .tab.active {
    background: var(--panel); color: var(--text);
    box-shadow: var(--shadow-sm);
  }
  .tab:hover:not(.active) { color: var(--text); }
  details {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px 14px; margin-bottom: 16px;
    background: var(--panel-2);
    transition: background 0.15s;
  }
  details summary {
    cursor: pointer; color: var(--text);
    font-size: 12px; font-weight: 500;
    outline: none; list-style: none;
    display: flex; align-items: center; justify-content: space-between;
  }
  details summary::after {
    content: "⌄"; color: var(--muted); font-size: 14px;
    transition: transform 0.2s;
  }
  details[open] summary { margin-bottom: 14px; }
  details[open] summary::after { transform: rotate(180deg); }
  details summary::-webkit-details-marker { display: none; }
  .file-input {
    border: 1.5px dashed var(--border-strong);
    border-radius: var(--radius-sm);
    padding: 16px; text-align: center; color: var(--muted);
    cursor: pointer; transition: all 0.15s;
    font-size: 12.5px;
  }
  .file-input:hover {
    border-color: var(--blue); color: var(--blue);
    background: rgba(0, 113, 227, 0.03);
  }
  .file-input input { display: none; }
  .thumbs { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  .thumb {
    width: 64px; height: 64px; border-radius: var(--radius-sm);
    overflow: hidden; border: 1px solid var(--border);
    position: relative; box-shadow: var(--shadow-sm);
  }
  .thumb img { width: 100%; height: 100%; object-fit: cover; }
  .thumb .x {
    position: absolute; top: 4px; right: 4px;
    width: 18px; height: 18px; border-radius: 50%;
    background: rgba(0,0,0,0.65); color: #fff;
    font-size: 11px; line-height: 18px; text-align: center;
    cursor: pointer; backdrop-filter: blur(4px);
  }
  button {
    font-family: inherit; cursor: pointer;
    transition: all 0.15s; -webkit-appearance: none;
  }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-primary {
    width: 100%; background: var(--gradient); color: #fff;
    border: none; border-radius: 980px; padding: 12px 20px;
    font-size: 14px; font-weight: 600;
    letter-spacing: -0.01em;
    box-shadow: 0 4px 14px rgba(138, 92, 246, 0.25);
  }
  .btn-primary:hover:not(:disabled) {
    box-shadow: 0 6px 20px rgba(138, 92, 246, 0.35);
    transform: translateY(-1px);
  }
  .btn-secondary {
    width: 100%; background: var(--panel); color: var(--text);
    border: 1px solid var(--border-strong); border-radius: 980px;
    padding: 11px 20px; font-size: 13px; font-weight: 500;
    margin-bottom: 10px;
  }
  .btn-secondary:hover:not(:disabled) {
    background: var(--panel-2); border-color: var(--text);
  }
  .btn-link {
    background: none; border: none; color: var(--blue);
    padding: 4px 0; font-size: 12px; font-weight: 500;
  }
  .btn-link:hover:not(:disabled) { text-decoration: underline; }
  .canvas-empty {
    flex: 1; display: flex; align-items: center; justify-content: center;
    color: var(--muted); font-size: 14px;
  }
  .canvas-image {
    flex: 1; display: flex; align-items: center; justify-content: center;
    background: var(--panel-2);
    border-radius: var(--radius);
    overflow: hidden; padding: 16px; min-height: 400px;
  }
  .canvas-image img {
    max-width: 100%; max-height: calc(100vh - 220px);
    border-radius: var(--radius-sm);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
  }
  .meta {
    margin-top: 18px; padding: 16px 18px;
    background: var(--panel-2);
    border-radius: var(--radius-sm);
    font-size: 13px; color: var(--text);
  }
  .meta .label {
    color: var(--muted); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    font-size: 10.5px; margin-bottom: 6px;
  }
  .meta pre {
    margin: 0 0 14px; white-space: pre-wrap; word-break: break-word;
    color: var(--text); font-family: inherit; line-height: 1.55;
  }
  .meta pre:last-child { margin-bottom: 0; }
  .progress {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    color: var(--muted); padding: 24px;
    position: relative;
  }
  .progress-preview {
    width: 100%; max-width: 520px; aspect-ratio: 1;
    border-radius: var(--radius);
    background: var(--panel-2);
    overflow: hidden; position: relative;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.08),
                0 0 0 1px var(--border);
  }
  .progress-preview::before {
    content: ""; position: absolute; inset: 0;
    background: var(--gradient);
    opacity: 0.08; pointer-events: none;
    mix-blend-mode: screen;
  }
  .progress-preview img {
    width: 100%; height: 100%; object-fit: cover;
    display: block; transition: filter 0.4s ease;
  }
  .progress-preview.empty { display: flex;
    align-items: center; justify-content: center; }
  .progress-preview.empty::after {
    content: ""; width: 80px; height: 80px;
    border-radius: 50%; background: var(--gradient);
    filter: blur(28px); opacity: 0.85;
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes pulse {
    0%, 100% { transform: scale(0.85); opacity: 0.65; }
    50% { transform: scale(1.2); opacity: 1; }
  }
  .progress-meta {
    display: flex; align-items: center; justify-content: space-between;
    width: 100%; max-width: 520px; margin-top: 20px;
    font-variant-numeric: tabular-nums;
  }
  .progress-label {
    font-size: 13px; font-weight: 600; color: var(--text);
    letter-spacing: -0.01em;
  }
  .progress-step {
    font-size: 12px; color: var(--muted);
    font-feature-settings: "tnum";
  }
  .process-list {
    width: 100%; max-width: 520px; margin-top: 14px;
    display: grid; gap: 8px;
  }
  .process-step {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 11px; border: 1px solid var(--border);
    border-radius: var(--radius-sm); background: var(--panel);
    color: var(--muted); font-size: 12.5px;
  }
  .process-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--border-strong); flex: 0 0 auto;
  }
  .process-step.active {
    color: var(--text); border-color: rgba(0, 113, 227, 0.45);
    background: rgba(0, 113, 227, 0.04);
  }
  .process-step.active .process-dot {
    background: var(--blue);
    box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.12);
  }
  .process-step.done { color: var(--text); }
  .process-step.done .process-dot { background: var(--mint); }
  .progress-bar {
    width: 100%; max-width: 520px; height: 3px; border-radius: 999px;
    background: var(--panel-2); overflow: hidden; margin-top: 10px;
  }
  .progress-bar-fill {
    height: 100%; background: var(--gradient);
    border-radius: 999px; width: 0%;
    transition: width 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .refine-preview {
    margin-top: 12px; padding: 12px 14px;
    background: rgba(0, 113, 227, 0.06);
    border: 1px solid rgba(0, 113, 227, 0.2);
    border-radius: var(--radius-sm);
    font-size: 12.5px;
  }
  .refine-preview .label {
    color: var(--blue); font-weight: 600;
    font-size: 10.5px; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 6px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .refine-preview pre {
    margin: 0; white-space: pre-wrap; word-break: break-word;
    font-family: inherit; line-height: 1.5;
  }
  .refine-actions { display: flex; gap: 8px; margin-top: 10px; }
  .refine-actions button { flex: 1; padding: 7px 10px; font-size: 12px; }
  .range-row {
    display: grid; grid-template-columns: 1fr auto; gap: 10px;
    align-items: center; margin-bottom: 14px;
  }
  .range-row label { margin: 0; }
  .range-row .value {
    color: var(--muted); font-size: 12px; min-width: 32px;
    text-align: right; font-variant-numeric: tabular-nums;
  }
  input[type="range"] {
    grid-column: 1 / -1; width: 100%; accent-color: var(--blue);
    padding: 0; border: none; background: transparent;
  }
  .err {
    color: var(--danger); margin-top: 12px; font-size: 12.5px;
    padding: 8px 12px; background: rgba(215, 0, 21, 0.06);
    border-radius: var(--radius-sm);
  }
  .toggle-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 4px 0; margin-bottom: 12px;
  }
  .toggle-row label { margin: 0; font-weight: 500; }
  .toggle-row .hint { color: var(--muted); font-size: 11px; margin-top: 2px; }
  .divider {
    height: 1px; background: var(--border);
    margin: 22px 0;
  }
  .spinner-inline {
    width: 14px; height: 14px; display: inline-block;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff; border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: -2px; margin-right: 8px;
  }
  .spinner-blue {
    border: 2px solid rgba(0, 113, 227, 0.2);
    border-top-color: var(--blue);
  }
  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; }
    .sidebar { max-height: none; }
  }
</style>
</head>
<body>
  <div class="topbar">
    <div>
      <span class="brand">HiDream-O1-Image{% if model_type == 'dev' %}-Dev{% endif %}</span>
    </div>
    <span class="topbar-meta"></span>
  </div>

  <div class="layout">
    <aside class="sidebar">
      <div class="tabs" id="tabs">
        <div class="tab active" data-mode="t2i">テキスト生成</div>
        <div class="tab" data-mode="edit">画像編集</div>
        <div class="tab" data-mode="subject">参照生成</div>
      </div>

      <div class="group">
        <label>プロンプト</label>
        <textarea id="prompt" placeholder="描きたい内容を日本語で入力してください"></textarea>
      </div>

      <div class="group" id="refs-group" style="display:none">
        <label id="refs-label">参照画像</label>
        <label class="file-input">
          <input id="refs" type="file" accept="image/*" multiple />
          <span>画像を選択</span>
        </label>
        <div class="thumbs" id="thumbs"></div>
        <label id="keep-aspect-row" style="display:none; margin-top:10px; font-weight:400; text-transform:none; letter-spacing:0; cursor:pointer;">
          <input id="keep-aspect" type="checkbox" style="vertical-align:middle; margin-right:6px;" />
          参照画像の縦横比を維持する
        </label>
        <div id="edit-scheduler-row" style="display:none; margin-top:12px;">
          <label>スケジューラー</label>
          <select id="edit-scheduler">
            <option value="flow_match" selected>flow_match（標準）</option>
            <option value="flash">flash</option>
          </select>
        </div>
      </div>

      <details open>
        <summary>画風調整</summary>
        <div class="range-row">
          <label for="anime-strength">アニメ・美少女ゲーム風</label>
          <span class="value" id="anime-strength-value">80</span>
          <input id="anime-strength" type="range" min="0" max="100" value="80" />
        </div>
        <div class="range-row">
          <label for="line-detail">線画・描き込み</label>
          <span class="value" id="line-detail-value">70</span>
          <input id="line-detail" type="range" min="0" max="100" value="70" />
        </div>
        <div class="range-row">
          <label for="color-vividness">色の鮮やかさ</label>
          <span class="value" id="color-vividness-value">65</span>
          <input id="color-vividness" type="range" min="0" max="100" value="65" />
        </div>
        <div class="range-row">
          <label for="background-mood">背景の雰囲気</label>
          <span class="value" id="background-mood-value">60</span>
          <input id="background-mood" type="range" min="0" max="100" value="60" />
        </div>
        <div class="range-row">
          <label for="photoreal-avoidance">実写化を避ける強さ</label>
          <span class="value" id="photoreal-avoidance-value">85</span>
          <input id="photoreal-avoidance" type="range" min="0" max="100" value="85" />
        </div>
      </details>

      <details id="refine-section">
        <summary>プロンプト調整</summary>
        <label>処理方式</label>
        <select id="refine-backend">
          <option value="api">OpenRouter / Gemma</option>
          <option value="local">ローカル Gemma</option>
        </select>
        <div id="api-fields" style="margin-top: 12px">
          <label>Base URL</label>
          <input id="api-base" type="text" autocomplete="off" name="hd-base-url"
                 placeholder="https://api.openai.com/v1" value="{{ env_base_url }}" />
          <label style="margin-top:10px">APIキー</label>
          <input id="api-key" type="password" autocomplete="new-password" name="hd-api-key"
                 placeholder="サーバー設定済み。空欄のままで利用できます" value="" />
          <label style="margin-top:10px">モデル</label>
          <input id="api-model" type="text" autocomplete="off" name="hd-model"
                 placeholder="gpt-4o-mini" value="{{ env_model }}" />
        </div>
        <button class="btn-secondary" id="refine-btn" style="margin-top: 14px; margin-bottom: 0">
          プロンプトを調整
        </button>
        <div id="refine-preview" class="refine-preview" style="display:none"></div>
      </details>

      <details>
        <summary>生成設定</summary>
        <div class="row" style="margin-bottom: 12px">
          <div>
            <label>幅</label>
            <input id="width" type="number" value="2048" step="64" min="512" />
          </div>
          <div>
            <label>高さ</label>
            <input id="height" type="number" value="2048" step="64" min="512" />
          </div>
        </div>
        <label>シード</label>
        <input id="seed" type="number" value="32" />
      </details>

      <button class="btn-primary" id="go">生成する</button>
      <div class="err" id="err" style="display:none"></div>
    </aside>

    <main class="canvas">
      <div class="canvas-empty" id="empty">
        <div style="text-align: center">
          <div style="font-size: 32px; opacity: 0.3; margin-bottom: 8px">◍</div>
          <div>生成した画像がここに表示されます</div>
        </div>
      </div>
      <div class="progress" id="progress" style="display:none">
        <div class="progress-preview empty" id="progress-preview">
          <img id="progress-img" style="display:none" />
        </div>
        <div class="progress-meta">
          <span class="progress-label" id="progress-text">準備中</span>
          <span class="progress-step" id="progress-sub">—</span>
        </div>
        <div class="process-list" id="process-list">
          <div class="process-step" data-step="refine"><span class="process-dot"></span><span>日本語プロンプトを解析して英語化</span></div>
          <div class="process-step" data-step="handoff"><span class="process-dot"></span><span>最適化プロンプトをHiDream-O1へ送信</span></div>
          <div class="process-step" data-step="generate"><span class="process-dot"></span><span>画像を生成</span></div>
          <div class="process-step" data-step="done"><span class="process-dot"></span><span>結果をウェブ画面へ返却</span></div>
        </div>
        <div class="progress-bar"><div class="progress-bar-fill" id="progress-fill"></div></div>
      </div>
      <div class="canvas-image" id="out" style="display:none">
        <img id="img" />
      </div>
      <div class="meta" id="meta" style="display:none"></div>
      <div class="meta" id="after-actions" style="display:none">
        <button class="btn-secondary" id="save-r2-btn" style="margin-bottom: 12px">R2に画像を保存</button>
        <div id="save-r2-result" style="color:var(--muted); font-size:12.5px; margin-bottom:14px"></div>
        <div class="label">AIに修正を依頼</div>
        <textarea id="edit-prompt" placeholder="例：表情を少し笑顔にして、背景の雨を強くしてください"></textarea>
        <button class="btn-primary" id="edit-go" style="margin-top: 12px">この画像を修正する</button>
        <div id="edit-chat" style="margin-top:12px; color:var(--muted); font-size:12.5px"></div>
      </div>
    </main>
  </div>

<script>
const $ = (id) => document.getElementById(id);
const MODEL_TYPE = "{{ model_type }}";
let mode = "t2i";
let refFiles = [];
let lastRefined = null;
let originalPrompt = null;
let lastImageB64 = "";
let lastOriginalPrompt = "";
let lastOptimizedPrompt = "";
let lastOptimizerSource = "";

const styleControls = [
  ["anime-strength", "anime_strength"],
  ["line-detail", "line_detail"],
  ["color-vividness", "color_vividness"],
  ["background-mood", "background_mood"],
  ["photoreal-avoidance", "photoreal_avoidance"],
];

function syncStyleControl(id) {
  const input = $(id);
  const value = $(id + "-value");
  if (input && value) value.textContent = input.value;
}

function getStyleSettings() {
  const settings = {};
  styleControls.forEach(([id, key]) => {
    const input = $(id);
    settings[key] = input ? parseInt(input.value) : 50;
  });
  return settings;
}

styleControls.forEach(([id]) => {
  const input = $(id);
  if (input) {
    syncStyleControl(id);
    input.oninput = () => syncStyleControl(id);
  }
});


document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    mode = t.dataset.mode;
    const refsGroup = $("refs-group");
    const refineSection = $("refine-section");
    const keepAspectRow = $("keep-aspect-row");
    const editSchedRow = $("edit-scheduler-row");
    if (mode === "t2i") {
      refsGroup.style.display = "none";
      refFiles = []; renderThumbs();
      refineSection.style.display = "";
      keepAspectRow.style.display = "none";
      $("keep-aspect").checked = false;
      editSchedRow.style.display = "none";
    } else {
      refsGroup.style.display = "";
      $("refs-label").textContent = mode === "edit"
        ? "Source image (1)"
        : "Reference images (2+)";
      $("refs").multiple = mode !== "edit";
      // Refine only available for T2I per design spec.
      refineSection.style.display = "none";
      refineSection.removeAttribute("open");
      clearRefinePreview();
      // `keep_original_aspect` only applies when there is exactly one ref image.
      keepAspectRow.style.display = mode === "edit" ? "" : "none";
      if (mode !== "edit") $("keep-aspect").checked = false;
      // Editing scheduler selector only applies to the Dev model + Edit tab.
      editSchedRow.style.display = (mode === "edit" && MODEL_TYPE === "dev") ? "" : "none";
    }
  };
});

$("refs").onchange = (e) => {
  const files = Array.from(e.target.files);
  refFiles = mode === "edit" ? files.slice(0, 1) : refFiles.concat(files);
  renderThumbs();
  e.target.value = "";
};

function renderThumbs() {
  const c = $("thumbs"); c.innerHTML = "";
  refFiles.forEach((f, i) => {
    const url = URL.createObjectURL(f);
    const el = document.createElement("div");
    el.className = "thumb";
    el.innerHTML = `<img src="${url}" /><div class="x" data-i="${i}">×</div>`;
    el.querySelector(".x").onclick = () => {
      refFiles.splice(i, 1); renderThumbs();
    };
    c.appendChild(el);
  });
}

$("refine-backend").onchange = (e) => {
  $("api-fields").style.display = e.target.value === "api" ? "" : "none";
};

function fileToB64(f) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result.split(",")[1]);
    r.onerror = rej;
    r.readAsDataURL(f);
  });
}

function showErr(msg) {
  const e = $("err"); e.textContent = msg; e.style.display = msg ? "" : "none";
}

function setProcessStep(activeStep) {
  const order = ["refine", "handoff", "generate", "done"];
  const activeIndex = order.indexOf(activeStep);
  document.querySelectorAll(".process-step").forEach((el) => {
    const step = el.dataset.step;
    const idx = order.indexOf(step);
    el.classList.toggle("active", step === activeStep);
    el.classList.toggle("done", activeIndex >= 0 && idx >= 0 && idx < activeIndex);
  });
}

function clearRefinePreview() {
  lastRefined = null; originalPrompt = null;
  $("refine-preview").style.display = "none";
  $("refine-preview").innerHTML = "";
}

function renderRefinePreview(refined) {
  lastRefined = refined;
  const html = `
    <div class="label">
      <span>Refined Prompt</span>
      <span style="font-weight: 400; color: var(--muted); text-transform: none; letter-spacing: 0">Preview</span>
    </div>
    <pre>${escapeHtml(refined.prompt)}</pre>
    <div class="refine-actions">
      <button class="btn-secondary" id="refine-apply" style="margin: 0">Use This Prompt</button>
      <button class="btn-secondary" id="refine-discard" style="margin: 0">Discard</button>
    </div>
  `;
  const box = $("refine-preview");
  box.innerHTML = html; box.style.display = "";
  $("refine-apply").onclick = () => {
    $("prompt").value = refined.prompt;
    clearRefinePreview();
  };
  $("refine-discard").onclick = () => clearRefinePreview();
}

$("refine-btn").onclick = async () => {
  const prompt = $("prompt").value.trim();
  if (!prompt) { showErr("Please enter a prompt to refine."); return; }
  showErr("");
  const btn = $("refine-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-inline spinner-blue"></span>Refining…';
  try {
    const body = {
      prompt,
      backend: $("refine-backend").value,
      api: {
        base_url: $("api-base").value.trim(),
        api_key: $("api-key").value.trim(),
        model: $("api-model").value.trim(),
      },
    };
    const r = await fetch("/api/refine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "Refine failed");
    renderRefinePreview(data);
  } catch (e) {
    showErr(e.message);
  } finally {
    btn.disabled = false; btn.textContent = "Refine Prompt";
  }
};

$("go").onclick = async () => {
  const prompt = $("prompt").value.trim();
  if (!prompt) { showErr("Please enter a prompt."); return; }
  if (mode === "edit" && refFiles.length !== 1) {
    showErr("Edit mode requires exactly one source image."); return;
  }
  if (mode === "subject" && refFiles.length < 2) {
    showErr("Subject mode requires at least two reference images."); return;
  }
  showErr("");
  const btn = $("go"); btn.disabled = true;
  btn.innerHTML = '<span class="spinner-inline"></span>Generating…';
  $("empty").style.display = "none";
  $("out").style.display = "none";
  $("meta").style.display = "none";
  $("after-actions").style.display = "none";
  $("save-r2-result").textContent = "";
  $("progress").style.display = "";
  $("progress-text").textContent = "Preparing";
  $("progress-sub").textContent = "Optimizing prompt";
  $("progress-fill").style.width = "0%";
  $("progress-img").style.display = "none";
  $("progress-img").removeAttribute("src");
  $("progress-preview").classList.add("empty");
  setProcessStep("refine");
  let optimizedPrompt = "";
  let optimizerNotes = "";
  let optimizerSource = "";

  try {
    const refs_b64 = await Promise.all(refFiles.map(fileToB64));
    const keepAspect = mode === "edit" && $("keep-aspect").checked && refFiles.length === 1;
    const editingScheduler = (mode === "edit" && MODEL_TYPE === "dev")
      ? $("edit-scheduler").value : null;
    const startResp = await fetch("/api/generate/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode, prompt,
        width: parseInt($("width").value),
        height: parseInt($("height").value),
        seed: parseInt($("seed").value),
        style_settings: getStyleSettings(),
        refs_b64,
        keep_original_aspect: keepAspect,
        editing_scheduler: editingScheduler,
      }),
    });
    const startData = await startResp.json();
    if (!startResp.ok) throw new Error(startData.error || "Failed to start");
    const jobId = startData.job_id;

    await new Promise((resolve, reject) => {
      const es = new EventSource(`/api/generate/stream/${jobId}`);
      es.onmessage = (ev) => {
        const d = JSON.parse(ev.data);
        if (d.type === "status") {
          $("progress-text").textContent = "Preparing";
          $("progress-sub").textContent = d.message || "Optimizing prompt";
          if (d.phase) setProcessStep(d.phase);
        } else if (d.type === "optimized_prompt") {
          optimizedPrompt = d.prompt || "";
          optimizerNotes = d.intent_notes || "";
          optimizerSource = d.source || "";
          $("progress-text").textContent = "Prompt optimized";
          $("progress-sub").textContent = "Starting HiDream-O1 generation";
          setProcessStep("handoff");
        } else if (d.type === "progress") {
          setProcessStep("generate");
          const pct = Math.round((d.step / d.total) * 100);
          $("progress-text").textContent = `Generating · ${pct}%`;
          $("progress-sub").textContent = `Step ${d.step} / ${d.total}`;
          $("progress-fill").style.width = pct + "%";
          if (d.preview) {
            $("progress-img").src = "data:image/jpeg;base64," + d.preview;
            $("progress-img").style.display = "";
            $("progress-preview").classList.remove("empty");
          }
        } else if (d.type === "done") {
          $("img").src = "data:image/png;base64," + d.image;
          setProcessStep("done");
          lastImageB64 = d.image;
          lastOriginalPrompt = d.original_prompt || prompt;
          optimizedPrompt = d.optimized_prompt || optimizedPrompt;
          optimizerNotes = d.intent_notes || optimizerNotes;
          optimizerSource = d.optimizer_source || optimizerSource;
          lastOptimizedPrompt = optimizedPrompt;
          lastOptimizerSource = optimizerSource;
          $("progress").style.display = "none";
          $("out").style.display = "";
          $("after-actions").style.display = "";
          $("meta").style.display = "";
          $("meta").innerHTML = `
            <details>
              <summary>Prompt sent to HiDream-O1</summary>
              <div style="white-space:pre-wrap; margin-top:8px">${escapeHtml(optimizedPrompt)}</div>
              ${optimizerSource ? `<div style="margin-top:8px; color:var(--muted)">Refiner: ${escapeHtml(optimizerSource)}</div>` : ""}
              ${optimizerNotes ? `<div style="margin-top:8px; color:var(--muted)">${escapeHtml(optimizerNotes)}</div>` : ""}
            </details>
          `;
          es.close(); resolve();
        } else if (d.type === "error") {
          es.close(); reject(new Error(d.message));
        }
      };
      es.onerror = () => { es.close(); reject(new Error("Stream connection lost")); };
    });
  } catch (e) {
    showErr(e.message);
    $("progress").style.display = "none";
    $("empty").style.display = "";
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate";
  }
};

$("save-r2-btn").onclick = async () => {
  if (!lastImageB64) { showErr("保存できる画像がありません。"); return; }
  const btn = $("save-r2-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-inline spinner-blue"></span>R2に保存中';
  $("save-r2-result").textContent = "";
  try {
    const r = await fetch("/api/save-to-r2", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: lastImageB64,
        original_prompt: lastOriginalPrompt,
        optimized_prompt: lastOptimizedPrompt,
        optimizer_source: lastOptimizerSource,
        style_settings: getStyleSettings(),
      }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "R2保存に失敗しました");
    $("save-r2-result").textContent = `保存しました: ${data.bucket}/${data.image_key}`;
  } catch (e) {
    showErr(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "R2に画像を保存";
  }
};

$("edit-go").onclick = async () => {
  const instruction = $("edit-prompt").value.trim();
  if (!lastImageB64) { showErr("修正元の画像がありません。"); return; }
  if (!instruction) { showErr("修正したい内容を入力してください。"); return; }

  const btn = $("edit-go");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-inline"></span>修正中';
  showErr("");
  $("edit-chat").innerHTML += `<div><strong>あなた:</strong> ${escapeHtml(instruction)}</div>`;
  $("empty").style.display = "none";
  $("out").style.display = "none";
  $("meta").style.display = "none";
  $("after-actions").style.display = "none";
  $("progress").style.display = "";
  $("progress-text").textContent = "修正準備中";
  $("progress-sub").textContent = "修正指示を英語化しています";
  $("progress-fill").style.width = "0%";
  $("progress-img").style.display = "none";
  $("progress-img").removeAttribute("src");
  $("progress-preview").classList.add("empty");
  setProcessStep("refine");

  let optimizedPrompt = "";
  let optimizerNotes = "";
  let optimizerSource = "";
  const editOriginalPrompt = `${lastOriginalPrompt || ""}\n\n修正指示: ${instruction}`.trim();

  try {
    const startResp = await fetch("/api/generate/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: "edit",
        prompt: instruction,
        width: parseInt($("width").value),
        height: parseInt($("height").value),
        seed: parseInt($("seed").value),
        refs_b64: [lastImageB64],
        keep_original_aspect: true,
        editing_scheduler: "flow_match",
        style_settings: getStyleSettings(),
      }),
    });
    const startData = await startResp.json();
    if (!startResp.ok) throw new Error(startData.error || "修正を開始できませんでした");

    await new Promise((resolve, reject) => {
      const es = new EventSource(`/api/generate/stream/${startData.job_id}`);
      es.onmessage = (ev) => {
        const d = JSON.parse(ev.data);
        if (d.type === "status") {
          $("progress-text").textContent = "修正準備中";
          $("progress-sub").textContent = d.message || "修正指示を処理しています";
          if (d.phase) setProcessStep(d.phase);
        } else if (d.type === "optimized_prompt") {
          optimizedPrompt = d.prompt || "";
          optimizerNotes = d.intent_notes || "";
          optimizerSource = d.source || "";
          $("progress-text").textContent = "修正指示を最適化しました";
          $("progress-sub").textContent = "HiDream-O1の画像編集へ送信します";
          setProcessStep("handoff");
        } else if (d.type === "progress") {
          setProcessStep("generate");
          const pct = Math.round((d.step / d.total) * 100);
          $("progress-text").textContent = `画像を修正中 · ${pct}%`;
          $("progress-sub").textContent = `ステップ ${d.step} / ${d.total}`;
          $("progress-fill").style.width = pct + "%";
          if (d.preview) {
            $("progress-img").src = "data:image/jpeg;base64," + d.preview;
            $("progress-img").style.display = "";
            $("progress-preview").classList.remove("empty");
          }
        } else if (d.type === "done") {
          $("img").src = "data:image/png;base64," + d.image;
          setProcessStep("done");
          lastImageB64 = d.image;
          lastOriginalPrompt = editOriginalPrompt;
          lastOptimizedPrompt = d.optimized_prompt || optimizedPrompt;
          lastOptimizerSource = d.optimizer_source || optimizerSource;
          $("progress").style.display = "none";
          $("out").style.display = "";
          $("after-actions").style.display = "";
          $("meta").style.display = "";
          $("meta").innerHTML = `
            <details>
              <summary>HiDream-O1へ送信した修正プロンプト</summary>
              <div style="white-space:pre-wrap; margin-top:8px">${escapeHtml(lastOptimizedPrompt)}</div>
              ${lastOptimizerSource ? `<div style="margin-top:8px; color:var(--muted)">Refiner: ${escapeHtml(lastOptimizerSource)}</div>` : ""}
              ${optimizerNotes ? `<div style="margin-top:8px; color:var(--muted)">${escapeHtml(optimizerNotes)}</div>` : ""}
            </details>
          `;
          $("edit-chat").innerHTML += `<div><strong>AI:</strong> 修正画像を生成しました。</div>`;
          $("edit-prompt").value = "";
          es.close(); resolve();
        } else if (d.type === "error") {
          es.close(); reject(new Error(d.message));
        }
      };
      es.onerror = () => { es.close(); reject(new Error("修正処理の接続が切れました")); };
    });
  } catch (e) {
    showErr(e.message);
    $("progress").style.display = "none";
    $("out").style.display = "";
    $("after-actions").style.display = "";
  } finally {
    btn.disabled = false;
    btn.textContent = "この画像を修正する";
  }
};

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  })[c]);
}
</script>
</body>
</html>
"""


# ── Routes ───────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    def _env(*keys, default=""):
        for k in keys:
            v = os.environ.get(k)
            if v:
                return v
        return default
    return render_template_string(
        INDEX_HTML,
        model_type=_STATE["model_type"],
        env_base_url=_env("OPENAI_BASE_URL", ),
        env_model=_env("OPENAI_MODEL", ),
    )


@app.route("/api/refine", methods=["POST"])
def api_refine():
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400
    backend = data.get("backend", "local")
    api_cfg = data.get("api") or {}

    try:
        if backend == "local":
            if _STATE["agent"] is None:
                model_id = os.environ.get("HIDREAM_AGENT_MODEL", "google/gemma-4-31B-it")
                _STATE["agent"] = build_local_agent(model_id)
            refined = rewrite_prompt_local(*_STATE["agent"], prompt)
        elif backend == "api":
            base_url = api_cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")
            api_key = api_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
            model_name = api_cfg.get("model") or os.environ.get("OPENAI_MODEL", "")
            if not all([base_url, api_key, model_name]):
                return jsonify({"error": "API requires base_url, api_key, model"}), 400
            refined = rewrite_prompt_api(
                prompt,
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
            )
        else:
            return jsonify({"error": f"Unknown backend: {backend}"}), 400
        return jsonify(refined)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/save-to-r2", methods=["POST"])
def api_save_to_r2():
    data = request.get_json(force=True)
    image_b64 = (data.get("image") or "").strip()
    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",", 1)[1]
    if not image_b64:
        return jsonify({"error": "No image to save."}), 400

    metadata = {
        "original_prompt": data.get("original_prompt", ""),
        "optimized_prompt": data.get("optimized_prompt", ""),
        "optimizer_source": data.get("optimizer_source", ""),
        "style_settings": normalize_style_settings(data.get("style_settings") or {}),
    }
    try:
        return jsonify(save_image_to_r2(image_b64, metadata))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate/start", methods=["POST"])
def api_generate_start():
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    mode = data.get("mode", "t2i")
    width = int(data.get("width", 2048))
    height = int(data.get("height", 2048))
    seed = int(data.get("seed", 32))
    style_settings = normalize_style_settings(data.get("style_settings") or {})
    refs_b64 = data.get("refs_b64") or []
    keep_original_aspect = bool(data.get("keep_original_aspect", False))
    editing_scheduler = data.get("editing_scheduler") or "flow_match"
    if editing_scheduler not in ("flow_match", "flash"):
        return jsonify({"error": f"Unknown editing_scheduler: {editing_scheduler}"}), 400

    if mode == "edit" and len(refs_b64) != 1:
        return jsonify({"error": "Edit mode requires exactly one reference image"}), 400
    if mode == "subject" and len(refs_b64) < 2:
        return jsonify({"error": "Subject mode requires at least two reference images"}), 400
    if keep_original_aspect and len(refs_b64) != 1:
        keep_original_aspect = False

    job_id = uuid.uuid4().hex
    q = queue.Queue()
    _JOBS[job_id] = q

    def worker():
        tmp_paths = []
        try:
            q.put({
                "type": "status",
                "phase": "refine",
                "message": "Refining Japanese prompt into O1-friendly English",
            })
            prompt_info = optimize_prompt_for_o1(prompt, mode=mode, style_settings=style_settings)
            optimized_prompt = prompt_info["prompt"]
            q.put({
                "type": "optimized_prompt",
                "phase": "handoff",
                "prompt": optimized_prompt,
                "intent_notes": prompt_info.get("intent_notes", ""),
                "source": prompt_info.get("source", ""),
            })

            for b64 in refs_b64:
                raw = base64.b64decode(b64)
                path = os.path.join(tempfile.gettempdir(), f"hidream_{uuid.uuid4().hex}.png")
                with open(path, "wb") as f:
                    f.write(raw)
                tmp_paths.append(path)

            def cb(step, total, get_preview=None):
                msg = {"type": "progress", "step": step + 1, "total": total}
                # Only send a preview image at the 1/4, 1/2, and 3/4 milestones
                # to avoid flooding the SSE stream and keep UI progress in sync.
                milestones = {total // 4, total // 2, (3 * total) // 4}
                want_preview = (
                    get_preview is not None
                    and (step + 1) in milestones
                )
                if want_preview:
                    try:
                        img = get_preview()
                        # Downscale to keep payload tiny — full image is sent at the end.
                        max_side = 384
                        w, h = img.size
                        if max(w, h) > max_side:
                            scale = max_side / max(w, h)
                            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                        buf2 = io.BytesIO()
                        img.save(buf2, format="JPEG", quality=72, optimize=True)
                        msg["preview"] = base64.b64encode(buf2.getvalue()).decode("ascii")
                    except Exception:
                        pass
                q.put(msg)

            with _GEN_LOCK:
                if _STATE["model_type"] == "full":
                    kwargs = dict(
                        num_inference_steps=50,
                        guidance_scale=5.0,
                        shift=3.0,
                        timesteps_list=None,
                        scheduler_name="default",
                    )
                elif mode == "edit" and editing_scheduler == "flow_match":
                    kwargs = dict(
                        num_inference_steps=28,
                        guidance_scale=0.0,
                        shift=1.0,
                        timesteps_list=DEFAULT_TIMESTEPS,
                        scheduler_name="flow_match",
                    )
                else:
                    kwargs = dict(
                        num_inference_steps=28,
                        guidance_scale=0.0,
                        shift=1.0,
                        timesteps_list=DEFAULT_TIMESTEPS,
                        scheduler_name="flash",
                        noise_scale_start=7.5,
                        noise_scale_end=7.5,
                        noise_clip_std=2.5,
                    )
                image = generate_image(
                    model=_STATE["model"],
                    processor=_STATE["processor"],
                    prompt=optimized_prompt,
                    ref_image_paths=tmp_paths if tmp_paths else None,
                    height=height,
                    width=width,
                    seed=seed,
                    keep_original_aspect=keep_original_aspect,
                    callback=cb,
                    **kwargs,
                )
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            q.put({
                "type": "done",
                "image": base64.b64encode(buf.getvalue()).decode("ascii"),
                "original_prompt": prompt,
                "optimized_prompt": optimized_prompt,
                "optimizer_source": prompt_info.get("source", ""),
                "intent_notes": prompt_info.get("intent_notes", ""),
            })
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            for p in tmp_paths:
                try: os.remove(p)
                except OSError: pass
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/generate/stream/<job_id>")
def api_generate_stream(job_id):
    q = _JOBS.get(job_id)
    if q is None:
        return jsonify({"error": "Unknown job"}), 404

    def gen():
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") in ("done", "error"):
                    break
        finally:
            _JOBS.pop(job_id, None)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Entrypoint ───────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser("HiDream-O1-Image Flask app")
    p.add_argument("--model_path", type=str,
                   default=os.environ.get("HIDREAM_MODEL_PATH"),
                   help="Path to HiDream-O1-Image checkpoint directory. "
                        "Defaults to $HIDREAM_MODEL_PATH from .env.")
    p.add_argument("--model_type", type=str,
                   default=os.environ.get("HIDREAM_MODEL_TYPE", "full"),
                   choices=["full", "dev"])
    p.add_argument("--host", type=str,
                   default=os.environ.get("HIDREAM_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("HIDREAM_PORT", "7860")))
    args = p.parse_args()

    if not args.model_path:
        p.error("--model_path is required (or set HIDREAM_MODEL_PATH in .env)")

    assert torch.cuda.is_available(), "CUDA is required for inference."
    processor, model = load_image_model(args.model_path)
    _STATE["processor"] = processor
    _STATE["model"] = model
    _STATE["model_type"] = args.model_type

    print(f"[app] Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
