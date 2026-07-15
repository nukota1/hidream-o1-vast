import gc
import os
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from sdxl_janku_workflow import fit_prompt_for_sdxl


EDITOR_MODELS = {
    "waifu_inpaint_xl": {
        "label": "Waifu-Inpaint-XL（標準）",
        "description": "アニメ・Illustrious XL向けの編集です。マスクなしでは元画像を強く保持します。狙った箇所を大きく変える場合は、白を修正範囲にしたマスクを使います。",
        "kind": "waifu_inpaint",
        "repo_env": "WAIFU_INPAINT_MODEL",
        "default_repo": "ShinoharaHare/Waifu-Inpaint-XL",
    },
    "flux_kontext_dev": {
        "label": "FLUX.1-Kontext-dev",
        "description": "指示文による画像編集です。現在のワークフローではマスクを使用しません。",
        "kind": "flux_kontext",
        "repo_env": "FLUX_KONTEXT_MODEL",
        "default_repo": "black-forest-labs/FLUX.1-Kontext-dev",
    },
    "hidream_o1_image": {
        "label": "HiDream-O1-Image",
        "description": "HiDreamのフルモデルによる指示ベースの画像編集です。現在のワークフローではマスクを使用しません。",
        "kind": "hidream_cli",
        "repo_env": "HIDREAM_O1_IMAGE_MODEL",
        "default_repo": "HiDream-ai/HiDream-O1-Image",
    },
}


def editor_model_choices():
    return {key: {"label": value["label"], "description": value["description"]} for key, value in EDITOR_MODELS.items()}


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype():
    # Waifu-Inpaint-XL is an SDXL checkpoint, and its published FP16 weights
    # are more reliable than automatic BF16 on recent consumer GPUs.
    return torch.float16 if _device() == "cuda" else torch.float32


def _generator(seed):
    return torch.Generator(device=_device()).manual_seed(int(seed))


def _enable_memory_savers(pipe):
    for method_name in ("enable_vae_slicing", "enable_vae_tiling"):
        try:
            getattr(pipe, method_name)()
        except (AttributeError, RuntimeError):
            pass


def unload_editor(pipe):
    if pipe is not None:
        for method_name in ("maybe_free_model_hooks", "remove_all_hooks"):
            try:
                getattr(pipe, method_name)()
            except (AttributeError, RuntimeError):
                pass
        del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_editor_pipeline(editor_id, status_callback=None):
    config = EDITOR_MODELS[editor_id]
    repo_id = os.environ.get(config["repo_env"], config["default_repo"]).strip()
    kind = config["kind"]
    if status_callback:
        status_callback(f"{config['label']} を準備しています。初回のみモデルをダウンロードします")

    if kind == "waifu_inpaint":
        from diffusers import StableDiffusionXLInpaintPipeline

        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(repo_id, torch_dtype=_dtype())
        pipe = pipe.to(_device())
        _enable_memory_savers(pipe)
        return pipe

    if kind == "flux_kontext":
        from diffusers import FluxKontextPipeline

        pipe = FluxKontextPipeline.from_pretrained(repo_id, torch_dtype=_dtype())
        if _device() == "cuda":
            # FLUX is deliberately lazy-loaded and offloaded. It must not share
            # VRAM with JANKU or the local prompt refiner.
            pipe.enable_model_cpu_offload(device="cuda")
        else:
            pipe = pipe.to(_device())
        _enable_memory_savers(pipe)
        return pipe

    if kind == "hidream_cli":
        return {"repo_id": repo_id, "kind": kind}

    raise RuntimeError(f"Unsupported image editor: {editor_id}")


def _automatic_background_mask(source):
    """Return a flood-filled mask for a near-uniform background, if present."""
    pixels = np.asarray(source, dtype=np.int16)
    height, width, _ = pixels.shape
    edge_pixels = np.concatenate((
        pixels[0:2].reshape(-1, 3),
        pixels[max(0, height - 2):height].reshape(-1, 3),
        pixels[:, 0:2].reshape(-1, 3),
        pixels[:, max(0, width - 2):width].reshape(-1, 3),
    ))
    background_colour = np.median(edge_pixels, axis=0)
    # The flood-fill prevents light clothing within the silhouette from being
    # selected as background merely because it is close to white.
    candidates = np.max(np.abs(pixels - background_colour), axis=2) <= 34
    background = np.zeros((height, width), dtype=bool)
    pending = deque()

    for x in range(width):
        for y in (0, height - 1):
            if candidates[y, x] and not background[y, x]:
                background[y, x] = True
                pending.append((y, x))
    for y in range(1, height - 1):
        for x in (0, width - 1):
            if candidates[y, x] and not background[y, x]:
                background[y, x] = True
                pending.append((y, x))

    while pending:
        y, x = pending.popleft()
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= next_y < height
                and 0 <= next_x < width
                and candidates[next_y, next_x]
                and not background[next_y, next_x]
            ):
                background[next_y, next_x] = True
                pending.append((next_y, next_x))

    if background.mean() < 0.12:
        return None
    return Image.fromarray((background * 255).astype(np.uint8), mode="L")


def extract_plain_background_mask(image_path):
    """Load a source image and return its reusable plain-background mask."""
    source = Image.open(image_path).convert("RGB")
    return source, _automatic_background_mask(source)


def normalize_plain_background(image):
    """Replace only edge-connected plain background pixels with pure white."""
    source = image.convert("RGB")
    background_mask = _automatic_background_mask(source)
    if background_mask is None:
        return source
    result = Image.new("RGB", source.size, color="white")
    result.paste(source, (0, 0), ImageOps.invert(background_mask))
    return result


def _load_source_and_mask(image_path, mask_path, edit_scope, status_callback=None):
    source = Image.open(image_path).convert("RGB")
    if mask_path:
        mask = Image.open(mask_path).convert("L").resize(source.size)
        return source, mask, True
    if edit_scope == "background":
        mask = _automatic_background_mask(source)
        if mask is not None:
            if status_callback:
                status_callback("立ち絵の無地背景を抽出しています")
            return source, mask, True
    # Pose changes and arbitrary source images need full-image editing. It is
    # paired with a low denoising strength to retain the source when possible.
    if status_callback:
        status_callback("画像全体を編集対象として準備しています")
    return source, Image.new("L", source.size, color=255), False


def _step_callback(callback, total):
    def on_step_end(_pipe, step, _timestep, callback_kwargs):
        if callback:
            callback(step, total)
        return callback_kwargs

    return on_step_end


def _edit_waifu(pipe, prompt, image_path, mask_path, seed, strength, callback, edit_scope, status_callback):
    source, mask, has_edit_mask = _load_source_and_mask(
        image_path,
        mask_path,
        edit_scope,
        status_callback=status_callback,
    )
    steps = int(os.environ.get("WAIFU_INPAINT_STEPS", "28"))
    cfg = float(os.environ.get("WAIFU_INPAINT_CFG", "5.0"))
    if strength is None:
        strength_name = "WAIFU_INPAINT_MASKED_STRENGTH" if has_edit_mask else "WAIFU_INPAINT_UNMASKED_STRENGTH"
        strength_default = "0.85" if has_edit_mask else "0.55"
        strength = float(os.environ.get(strength_name, strength_default))
    final_prompt = fit_prompt_for_sdxl(
        pipe,
        prompt + ", same character, same face, preserve identity, preserve unrequested details",
    )
    result = pipe(
        prompt=final_prompt,
        image=source,
        mask_image=mask,
        num_inference_steps=steps,
        guidance_scale=cfg,
        strength=strength,
        height=source.height,
        width=source.width,
        generator=_generator(seed),
        callback_on_step_end=_step_callback(callback, steps),
    )
    return result.images[0]


def _edit_flux_kontext(pipe, prompt, image_path, _mask_path, seed, callback):
    source = Image.open(image_path).convert("RGB")
    steps = int(os.environ.get("FLUX_KONTEXT_STEPS", "28"))
    guidance = float(os.environ.get("FLUX_KONTEXT_GUIDANCE", "2.5"))
    result = pipe(
        image=source,
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=_generator(seed),
        callback_on_step_end=_step_callback(callback, steps),
    )
    return result.images[0]


def _hidream_model_path(repo_id, status_callback):
    model_path = Path(os.environ.get("HIDREAM_O1_IMAGE_PATH", "/models/HiDream-O1-Image"))
    if model_path.is_dir() and any(model_path.iterdir()):
        return model_path
    if status_callback:
        status_callback("HiDream-O1-Image をダウンロードしています。初回は時間がかかります")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=str(model_path))
    return model_path


def _edit_hidream(pipe, prompt, image_path, _mask_path, seed, status_callback):
    source_dir = Path(os.environ.get("HIDREAM_O1_RUNTIME", "/opt/hidream-o1-image"))
    inference_script = source_dir / "inference.py"
    if not inference_script.is_file():
        raise RuntimeError("HiDream-O1-Image runtime is missing from this container image.")
    model_path = _hidream_model_path(pipe["repo_id"], status_callback)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_file:
        output_path = output_file.name
    try:
        if status_callback:
            status_callback("HiDream-O1-Image で編集しています")
        command = [
            sys.executable,
            str(inference_script),
            "--model_path",
            str(model_path),
            "--prompt",
            prompt,
            "--ref_images",
            image_path,
            "--output_image",
            output_path,
            "--keep_original_aspect",
            "--seed",
            str(seed),
        ]
        completed = subprocess.run(
            command,
            cwd=str(source_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown error").strip()
            raise RuntimeError(f"HiDream-O1-Image editing failed: {detail[-2000:]}")
        return Image.open(output_path).convert("RGB").copy()
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass


def edit_image(
    editor_id,
    pipe,
    prompt,
    image_path,
    mask_path,
    seed,
    strength=None,
    callback=None,
    status_callback=None,
    edit_scope="background",
):
    kind = EDITOR_MODELS[editor_id]["kind"]
    if kind == "waifu_inpaint":
        return _edit_waifu(
            pipe,
            prompt,
            image_path,
            mask_path,
            seed,
            strength,
            callback,
            edit_scope,
            status_callback,
        )
    if kind == "flux_kontext":
        return _edit_flux_kontext(pipe, prompt, image_path, mask_path, seed, callback)
    if kind == "hidream_cli":
        return _edit_hidream(pipe, prompt, image_path, mask_path, seed, status_callback)
    raise RuntimeError(f"Unsupported image editor: {editor_id}")
