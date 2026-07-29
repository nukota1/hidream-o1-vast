import gc
import os
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter, ImageOps
from scipy.ndimage import binary_fill_holes, distance_transform_edt

from sdxl_janku_workflow import fit_prompt_for_sdxl


EDITOR_MODELS = {
    "waifu_inpaint_xl": {
        "label": "Waifu-Inpaint-XL（標準）",
        "description": "アニメ・Illustrious XL向けの人物を含む全体・局所修正です。白いマスク領域を編集します。",
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

_ANIME_SEGMENTATION_SESSION = None
CHARACTER_CHROMA_RGB = (0, 255, 0)


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


def _edge_background_colour(source):
    pixels = np.asarray(source.convert("RGB"), dtype=np.int16)
    height, width, _ = pixels.shape
    edge_pixels = np.concatenate((
        pixels[0:2].reshape(-1, 3),
        pixels[max(0, height - 2):height].reshape(-1, 3),
        pixels[:, 0:2].reshape(-1, 3),
        pixels[:, max(0, width - 2):width].reshape(-1, 3),
    ))
    return np.median(edge_pixels, axis=0)


def _verified_chroma_colour(source):
    """Return the edge colour only when it is a recognizable green screen."""
    background_colour = _edge_background_colour(source)
    red, green, blue = (int(value) for value in background_colour)
    if green < 100 or green < red + 35 or green < blue + 35:
        return None
    return background_colour


def _anime_foreground_alpha(source):
    """Return the original soft anime foreground confidence mask."""
    global _ANIME_SEGMENTATION_SESSION

    from rembg import new_session, remove

    if _ANIME_SEGMENTATION_SESSION is None:
        model_name = os.environ.get("ANIME_SEGMENTATION_MODEL", "isnet-anime").strip()
        _ANIME_SEGMENTATION_SESSION = new_session(model_name)
    return remove(
        source.convert("RGB"),
        session=_ANIME_SEGMENTATION_SESSION,
        only_mask=True,
    ).convert("L")


def _chroma_background_pixels(source):
    """Return pixels matching an edge-connected green-screen background."""
    pixels = np.asarray(source.convert("RGB"), dtype=np.int16)
    background_colour = _verified_chroma_colour(source)
    if background_colour is None:
        return None
    tolerance = int(os.environ.get("CHARACTER_CHROMA_TOLERANCE", "56"))
    tolerance = max(8, min(120, tolerance))
    return np.max(np.abs(pixels - background_colour), axis=2) <= tolerance


def _soft_character_matte(source):
    """Build a matte whose semantic confidence softens only the silhouette edge."""
    semantic = np.asarray(_anime_foreground_alpha(source), dtype=np.float32) / 255.0
    chroma_colour = _verified_chroma_colour(source)
    threshold = int(os.environ.get("ANIME_SEGMENTATION_THRESHOLD", "96"))
    threshold = max(1, min(254, threshold))
    transparent_cutoff = float(
        os.environ.get("CHARACTER_MATTE_TRANSPARENT_CUTOFF", "0.015")
    )
    opaque_cutoff = float(
        os.environ.get("CHARACTER_MATTE_OPAQUE_CUTOFF", "0.985")
    )
    transparent_cutoff = max(0.0, min(0.25, transparent_cutoff))
    opaque_cutoff = max(0.75, min(1.0, opaque_cutoff))
    foreground = semantic >= (threshold / 255.0)
    edge_alpha = semantic

    if chroma_colour is None:
        background_colour = _edge_background_colour(source)
    else:
        pixels = np.asarray(source.convert("RGB"), dtype=np.float32)
        background = np.asarray(chroma_colour, dtype=np.float32)
        background_dominance = max(
            1.0,
            float(background[1] - max(background[0], background[2])),
        )
        green_dominance = pixels[:, :, 1] - np.maximum(
            pixels[:, :, 0], pixels[:, :, 2]
        )
        chroma_alpha = 1.0 - np.clip(
            green_dominance / background_dominance, 0.0, 1.0
        )
        tolerance = int(os.environ.get("CHARACTER_CHROMA_TOLERANCE", "56"))
        tolerance = max(8, min(120, tolerance))
        near_background = (
            np.max(np.abs(pixels - background), axis=2) <= tolerance
        )
        # Exact screen-colour pixels must stay background even when ISNet
        # classifies a closed gap between hair strands as foreground.
        foreground &= ~near_background
        # Recover enclosed false-negative pinholes only when the source colour
        # is clearly not chroma. Real green gaps between hair strands remain
        # background, while skin, hair, and clothing specks become opaque.
        enclosed_holes = binary_fill_holes(foreground) & ~foreground
        foreground |= (
            enclosed_holes
            & ~near_background
            & (chroma_alpha >= opaque_cutoff)
        )
        edge_alpha = np.minimum(semantic, chroma_alpha)
        background_colour = chroma_colour

    # ISNet output is a confidence map, not physical opacity. Treating values
    # such as 180/255 as alpha makes solid hair, skin, and clothing translucent.
    # Make the classified foreground opaque and retain confidence-based alpha
    # only in the one-pixel rings touching the silhouette.
    edge_width = float(os.environ.get("CHARACTER_MATTE_EDGE_WIDTH", "1.0"))
    edge_width = max(0.5, min(2.0, edge_width))
    inside_distance = distance_transform_edt(foreground)
    outside_distance = distance_transform_edt(~foreground)
    inner_edge = foreground & (inside_distance <= edge_width)
    outer_edge = (
        ~foreground
        & (outside_distance <= edge_width)
        & (edge_alpha > transparent_cutoff)
    )
    alpha = foreground.astype(np.float32)
    alpha[inner_edge] = edge_alpha[inner_edge]
    alpha[outer_edge] = edge_alpha[outer_edge]
    alpha = np.where(alpha <= transparent_cutoff, 0.0, alpha)
    alpha = np.where(alpha >= opaque_cutoff, 1.0, alpha)
    return alpha, background_colour


def _decontaminate_background_colour(source, alpha, background_colour):
    """Recover straight foreground RGB from pixels blended with a solid screen."""
    pixels = np.asarray(source.convert("RGB"), dtype=np.float32)
    background = np.asarray(background_colour, dtype=np.float32).reshape(1, 1, 3)
    alpha_3d = alpha[:, :, None]
    safe_alpha = np.maximum(alpha_3d, 1.0 / 255.0)
    foreground = (
        pixels - ((1.0 - alpha_3d) * background)
    ) / safe_alpha
    foreground = np.clip(foreground, 0.0, 255.0)
    # Generated green screens can cast bright green reflections into dark hair
    # and anti-aliased line art. Remove only the excess green above the stronger
    # of red/blue; neutral and yellow character colours remain unchanged.
    red, green, blue = (float(value) for value in background.reshape(3))
    if green >= red + 35.0 and green >= blue + 35.0:
        despill_strength = float(
            os.environ.get("CHARACTER_CHROMA_DESPILL_STRENGTH", "1.0")
        )
        despill_strength = max(0.0, min(1.0, despill_strength))
        non_green = np.maximum(foreground[:, :, 0], foreground[:, :, 2])
        green_excess = np.maximum(0.0, foreground[:, :, 1] - non_green)
        foreground[:, :, 1] -= green_excess * despill_strength
    # Fully transparent RGB is irrelevant. Keeping it black avoids carrying the
    # chroma screen into exported PNGs or later resampling operations.
    foreground[alpha <= 0.0] = 0.0
    return Image.fromarray(
        np.rint(foreground).astype(np.uint8),
        mode="RGB",
    )


def _anime_foreground_mask(source):
    """Segment anime foreground and remove known green-screen pixels globally."""
    foreground_alpha = _anime_foreground_alpha(source)
    threshold = int(os.environ.get("ANIME_SEGMENTATION_THRESHOLD", "96"))
    threshold = max(1, min(254, threshold))
    foreground_pixels = np.asarray(foreground_alpha) >= threshold
    chroma_background = _chroma_background_pixels(source)
    if chroma_background is not None:
        # Color-keying is enabled only when the image edges are recognizably
        # green. This removes background trapped between hair strands while
        # external white-background images remain semantic-only.
        foreground_pixels &= ~chroma_background
    foreground_ratio = float(foreground_pixels.mean())
    if foreground_ratio < 0.01 or foreground_ratio > 0.95:
        raise RuntimeError(
            "人物領域を安全に抽出できませんでした。白い領域を背景にした編集マスクを指定してください。"
        )
    return Image.fromarray((foreground_pixels * 255).astype(np.uint8), mode="L")


def _semantic_background_mask(source):
    """Return an editable background mask from anime-aware segmentation."""
    return ImageOps.invert(_anime_foreground_mask(source))


def _meaningful_source_alpha(image):
    if "A" not in image.getbands():
        return None
    alpha = image.getchannel("A").convert("L")
    return alpha if alpha.getextrema() != (255, 255) else None


def _flatten_transparent_source(image, background_rgb=(255, 255, 255)):
    """Flatten an RGBA source for editors that require an opaque RGB image."""
    alpha = _meaningful_source_alpha(image)
    if alpha is None:
        return image.convert("RGB")
    flattened = Image.new("RGB", image.size, background_rgb)
    flattened.paste(image.convert("RGB"), (0, 0), alpha)
    return flattened


def extract_plain_background_mask(image_path):
    """Load a source image and return an anime-aware reusable background mask."""
    loaded = Image.open(image_path)
    source_alpha = _meaningful_source_alpha(loaded)
    source = loaded.convert("RGB")
    if source_alpha is not None:
        return source, ImageOps.invert(source_alpha)
    mask = _semantic_background_mask(source)
    return source, _feather_background_mask(mask)


def prepare_character_layer(image):
    """Convert a generated green-screen character into a reusable RGBA layer."""
    source = image.convert("RGB")
    alpha, background_colour = _soft_character_matte(source)
    foreground_rgb = _decontaminate_background_colour(
        source, alpha, background_colour
    )
    foreground_alpha = Image.fromarray(
        np.rint(alpha * 255.0).astype(np.uint8),
        mode="L",
    )
    transparent_layer = foreground_rgb.convert("RGBA")
    transparent_layer.putalpha(foreground_alpha)
    background_mask = ImageOps.invert(foreground_alpha)
    return transparent_layer, background_mask


def _feather_background_mask(mask):
    """Feather only outside the subject while keeping every subject pixel fixed."""
    mask = mask.convert("L")
    feathered = mask.filter(ImageFilter.GaussianBlur(radius=0.6))
    return ImageChops.multiply(feathered, mask)


def restore_unmasked_pixels(source, edited, editable_mask):
    """Restore source pixels outside an edit mask, blending only its soft edge."""
    source = source.convert("RGB")
    edited = edited.convert("RGB").resize(source.size)
    editable_mask = editable_mask.convert("L").resize(source.size)
    preservation_mask = ImageOps.invert(editable_mask)
    return Image.composite(source, edited, preservation_mask)


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
    loaded = Image.open(image_path)
    source_alpha = _meaningful_source_alpha(loaded)
    source = (
        _flatten_transparent_source(loaded)
        if edit_scope == "full"
        else loaded.convert("RGB")
    )
    if mask_path:
        mask = Image.open(mask_path).convert("L").resize(source.size)
        return source, mask, True
    if edit_scope == "background":
        if source_alpha is not None:
            if status_callback:
                status_callback("透明PNGのアルファから背景マスクを復元しています")
            return source, ImageOps.invert(source_alpha), True
        mask = _semantic_background_mask(source)
        if mask is not None:
            if status_callback:
                status_callback("アニメ人物を識別し、背景マスクを作成しています")
            return source, _feather_background_mask(mask), True
        raise RuntimeError(
            "無地背景を自動抽出できませんでした。人物を保護するため画像全体の編集は行いません。"
            "白い領域を背景にした編集マスクを指定してください。"
        )
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


def _edit_waifu(
    pipe,
    prompt,
    negative_prompt,
    image_path,
    mask_path,
    seed,
    strength,
    callback,
    edit_scope,
    status_callback,
):
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
    if edit_scope == "background":
        # A transparent character layer is flattened to black before it enters
        # the SDXL pipeline. Partial denoising can retain that blank latent and
        # produce a black background. Fully re-noise the masked background;
        # the original subject is restored pixel-for-pixel after decoding.
        strength = float(
            os.environ.get("WAIFU_INPAINT_BACKGROUND_STRENGTH", "1.0")
        )
        final_prompt = fit_prompt_for_sdxl(
            pipe,
            prompt
            + ", detailed environment only, coherent perspective, matching scene lighting, "
            + "natural contact with the existing foreground subject, no additional characters",
        )
        final_negative = (
            "additional person, second person, duplicate person, duplicate character, "
            "extra character, background character, crowd, face in background, body in background, "
            "floating person, pasted character, collage, cutout, mismatched lighting, "
            + (negative_prompt or "")
        )
    else:
        final_prompt = fit_prompt_for_sdxl(
            pipe,
            prompt + ", same character, same face, preserve identity, preserve unrequested details",
        )
        final_negative = negative_prompt or ""
    result = pipe(
        prompt=final_prompt,
        negative_prompt=final_negative,
        image=source,
        mask_image=mask,
        num_inference_steps=steps,
        guidance_scale=cfg,
        strength=strength,
        height=source.height,
        width=source.width,
        generator=_generator(seed),
        callback_on_step_end=_step_callback(callback, steps),
    ).images[0]
    if edit_scope == "background":
        # Diffusion and VAE decoding can alter even nominally masked-out pixels.
        # Put the original subject back after generation; only the soft mask
        # boundary remains blended so lighting and contours can meet the scene.
        result = restore_unmasked_pixels(source, result, mask)
    return result


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
    negative_prompt,
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
            negative_prompt,
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
