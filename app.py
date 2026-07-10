import argparse
import base64
import gc
import io
import json
import os
import queue
import tempfile
import threading
import uuid
from datetime import datetime, timezone

import torch
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from prompt_refiner import LocalPromptRefiner
from sdxl_janku_workflow import (
    edit_with_qwen_image_edit,
    generate_with_janku,
    load_janku_pipeline,
    load_qwen_edit_pipeline,
)

load_dotenv()

app = Flask(__name__)
_GEN_LOCK = threading.Lock()
_JOBS = {}
_STATE = {
    "janku_pipe": None,
    "qwen_edit_pipe": None,
    "refiner": LocalPromptRefiner(),
}

BASE_NEGATIVE_PROMPT = (
    "lowres, worst quality, low quality, bad anatomy, bad hands, extra fingers, "
    "missing fingers, malformed limbs, blurry, jpeg artifacts, text, watermark, signature"
)

STYLE_PRESETS = {
    "bishoujo_game": {
        "label": "美少女ゲーム風",
        "prompt_hint": (
            "2D Japanese bishoujo visual novel game CG, polished character design, "
            "clean expressive eyes, finely rendered hair, crisp line art, detailed scenic background"
        ),
        "width": 1216,
        "height": 832,
        "steps": 28,
        "cfg": 4.5,
        "sampler": "euler_a",
        "clip_skip": 2,
        "negative_prompt": BASE_NEGATIVE_PROMPT + ", photorealistic, live action, 3d render",
        "style": {"anime_strength": 90, "line_detail": 80, "color_vividness": 72, "background_mood": 68, "photoreal_avoidance": 95},
    },
    "anime_illustration": {
        "label": "アニメイラスト",
        "prompt_hint": (
            "high-quality modern Japanese anime illustration, appealing character art, "
            "clean linework, vivid controlled colors, balanced lighting"
        ),
        "width": 832,
        "height": 1216,
        "steps": 28,
        "cfg": 4.5,
        "sampler": "euler_a",
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
        "steps": 28,
        "cfg": 4.5,
        "sampler": "euler_a",
        "clip_skip": 2,
        "negative_prompt": BASE_NEGATIVE_PROMPT + ", photorealistic, live action, 3d render",
        "style": {"anime_strength": 88, "line_detail": 82, "color_vividness": 68, "background_mood": 66, "photoreal_avoidance": 94},
    },
    "custom": {
        "label": "カスタム",
        "prompt_hint": "",
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "cfg": 4.5,
        "sampler": "euler_a",
        "clip_skip": 2,
        "negative_prompt": BASE_NEGATIVE_PROMPT,
        "style": {"anime_strength": 70, "line_detail": 70, "color_vividness": 65, "background_mood": 60, "photoreal_avoidance": 80},
    },
}


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
    return {
        "preset": preset_id,
        "width": normalize_dimension(data.get("width"), preset["width"]),
        "height": normalize_dimension(data.get("height"), preset["height"]),
        "steps": clamp_int(data.get("steps"), preset["steps"], 10, 60),
        "cfg": clamp_float(data.get("cfg"), preset["cfg"], 1.0, 12.0),
        "sampler": sampler,
        "clip_skip": clamp_int(data.get("clip_skip"), preset["clip_skip"], 1, 4),
        "seed": clamp_int(data.get("seed"), 32, 0, 2**31 - 1),
        "negative_prompt": str(data.get("negative_prompt") or preset["negative_prompt"]).strip(),
        "style": normalize_style_settings(data.get("style_settings"), preset["style"]),
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


def prepare_prompt(user_prompt, mode, settings, refine_enabled):
    preset = STYLE_PRESETS[settings["preset"]]
    if refine_enabled:
        try:
            return _STATE["refiner"].refine(
                user_prompt=user_prompt,
                mode=mode,
                preset_label=preset["label"],
                preset_hint=preset["prompt_hint"],
                style_description=describe_style(settings["style"]),
            )
        except Exception as exc:
            print(f"[refine] Local refinement failed: {exc}")
            source = "fallback"
            notes = f"Local refinement failed; deterministic style hints were used: {exc}"
    else:
        source = "disabled"
        notes = "Prompt refinement was disabled by the user."

    parts = [user_prompt]
    if preset["prompt_hint"]:
        parts.append(preset["prompt_hint"])
    style_hint = deterministic_style_hint(settings["style"])
    if style_hint:
        parts.append(style_hint)
    return {"prompt": ", ".join(parts), "intent_notes": notes, "source": source}


def unload_pipeline(name):
    _STATE[name] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        presets=STYLE_PRESETS,
        presets_json=json.dumps(STYLE_PRESETS, ensure_ascii=False),
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
    job_id = uuid.uuid4().hex
    q = queue.Queue()
    _JOBS[job_id] = q

    def worker():
        temp_paths = []
        try:
            q.put({"type": "status", "phase": "refine", "message": "プロンプトを準備しています"})
            prompt_info = prepare_prompt(prompt, mode, settings, refine_enabled)
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
                    q.put({"type": "status", "phase": "generate", "message": "Qwen Image Editを準備しています"})
                    unload_pipeline("janku_pipe")
                    if _STATE["qwen_edit_pipe"] is None:
                        _STATE["qwen_edit_pipe"] = load_qwen_edit_pipeline()
                    image = edit_with_qwen_image_edit(
                        _STATE["qwen_edit_pipe"],
                        prompt_info["prompt"],
                        source_path,
                        mask_path,
                        settings["seed"],
                        callback=progress,
                    )
                else:
                    q.put({"type": "status", "phase": "generate", "message": "JANKU v7.77を準備しています"})
                    unload_pipeline("qwen_edit_pipe")
                    if _STATE["janku_pipe"] is None:
                        def model_status(message):
                            q.put({"type": "status", "phase": "generate", "message": message})
                        _STATE["janku_pipe"] = load_janku_pipeline(status_callback=model_status)
                    image = generate_with_janku(
                        _STATE["janku_pipe"],
                        prompt_info["prompt"],
                        settings,
                        callback=progress,
                    )

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
                "refine_enabled": refine_enabled,
            })
        except Exception as exc:
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
    parser = argparse.ArgumentParser("JANKU Image Studio")
    parser.add_argument("--host", default=os.environ.get("APP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("APP_PORT", "7861")))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for image generation.")
    print(f"[app] Serving JANKU Image Studio on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
