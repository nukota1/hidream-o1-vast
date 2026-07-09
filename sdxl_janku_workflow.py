import os
import time
import urllib.request

import torch
from PIL import Image


def _torch_dtype():
    dtype_name = os.environ.get("SDXL_TORCH_DTYPE", os.environ.get("HIDREAM_TORCH_DTYPE", "auto")).lower()
    if dtype_name == "auto":
        device_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ""
        dtype_name = "float16" if "v100" in device_name or "tesla v100" in device_name else "bfloat16"
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(dtype_name, torch.float16)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _negative_prompt():
    return os.environ.get(
        "SDXL_NEGATIVE_PROMPT",
        "lowres, worst quality, low quality, bad anatomy, bad hands, extra fingers, "
        "missing fingers, blurry, jpeg artifacts, watermark, signature, text",
    )


def _min_janku_bytes():
    try:
        return int(os.environ.get("JANKU_MODEL_MIN_BYTES", "6000000000"))
    except ValueError:
        return 6000000000


def _is_complete_file(path, min_bytes):
    if not path or not os.path.isfile(path):
        return False
    if min_bytes <= 0:
        return True
    actual_bytes = os.path.getsize(path)
    if actual_bytes < min_bytes:
        print(f"[sdxl] Incomplete model file detected: {path} ({actual_bytes}/{min_bytes} bytes)")
        return False
    return True


def _download_url(url, tmp_path, token):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    try:
        return urllib.request.urlopen(req, timeout=3600)
    except Exception:
        if not token:
            raise
        sep = "&" if "?" in url else "?"
        return urllib.request.urlopen(f"{url}{sep}token={token}", timeout=3600)


def _download_if_needed(path, url):
    if not path or not url:
        return
    min_bytes = _min_janku_bytes()
    if _is_complete_file(path, min_bytes):
        return
    if os.path.isfile(path):
        os.remove(path)
    tmp_path = f"{path}.part"
    wait_seconds = int(os.environ.get("SDXL_DOWNLOAD_WAIT_SECONDS", "7200"))
    waited = 0
    while os.path.isfile(tmp_path) and not os.path.isfile(path):
        if waited == 0:
            print(f"[sdxl] Waiting for background download: {path}")
        if waited >= wait_seconds:
            raise RuntimeError(f"Timed out waiting for background download: {path}")
        time.sleep(5)
        waited += 5
    if _is_complete_file(path, min_bytes):
        return
    if os.path.isfile(path):
        os.remove(path)
    print(f"[sdxl] Downloading model to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    token = os.environ.get("CIVITAI_TOKEN", "")
    response = _download_url(url, tmp_path, token)
    try:
        with response:
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        if not _is_complete_file(tmp_path, min_bytes):
            raise RuntimeError(f"Downloaded model is smaller than expected: {tmp_path}")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    print(f"[sdxl] Download complete: {path}")


def _load_single_file_or_repo(pipeline_cls, model_ref, dtype):
    if not model_ref:
        raise RuntimeError("Model path/repo is not configured.")
    kwargs = {"torch_dtype": dtype, "use_safetensors": True}
    if os.path.isfile(model_ref):
        pipe = pipeline_cls.from_single_file(model_ref, **kwargs)
    elif model_ref.endswith((".safetensors", ".ckpt")):
        raise RuntimeError(f"Model file does not exist: {model_ref}")
    else:
        pipe = pipeline_cls.from_pretrained(model_ref, **kwargs)
    pipe = pipe.to(_device())
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe


def load_janku_pipeline():
    from diffusers import AutoPipelineForText2Image

    model_path = os.environ.get("JANKU_MODEL_PATH", "")
    _download_if_needed(model_path, os.environ.get("JANKU_MODEL_URL", ""))
    model_ref = model_path or os.environ.get("JANKU_MODEL_REPO")
    print(f"[sdxl] Loading JANKU text-to-image model: {model_ref}")
    return _load_single_file_or_repo(AutoPipelineForText2Image, model_ref, _torch_dtype())


def load_qwen_edit_pipeline():
    from diffusers import QwenImageEditPlusPipeline

    model_ref = os.environ.get("QWEN_IMAGE_EDIT_MODEL_REPO", "Qwen/Qwen-Image-Edit-2511")
    print(f"[qwen-edit] Loading image edit model: {model_ref}")
    pipe = QwenImageEditPlusPipeline.from_pretrained(model_ref, torch_dtype=_torch_dtype())
    pipe = pipe.to(_device())
    try:
        pipe.set_progress_bar_config(disable=None)
    except Exception:
        pass
    return pipe


def _generator(seed):
    return torch.Generator(device=_device()).manual_seed(int(seed))


def generate_with_janku(pipe, prompt, width, height, seed, callback=None):
    steps = int(os.environ.get("JANKU_STEPS", "28"))
    guidance_scale = float(os.environ.get("JANKU_CFG_SCALE", "5"))
    image = pipe(
        prompt=prompt,
        negative_prompt=_negative_prompt(),
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=_generator(seed),
    ).images[0]
    if callback:
        callback(steps - 1, steps)
    return image


def _full_mask(size):
    return Image.new("L", size, 255)


def edit_with_qwen_image_edit(pipe, prompt, image_path, mask_path, width, height, seed, callback=None):
    source = Image.open(image_path).convert("RGB")
    images = [source]
    edit_prompt = prompt
    if mask_path:
        mask = Image.open(mask_path).convert("RGB")
        images.append(mask)
        edit_prompt = (
            prompt
            + "\nUse the second image as an edit mask: white areas are editable, black areas must be preserved."
        )
    steps = int(os.environ.get("QWEN_IMAGE_EDIT_STEPS", "40"))
    true_cfg_scale = float(os.environ.get("QWEN_IMAGE_EDIT_TRUE_CFG_SCALE", "4.0"))
    guidance_scale = float(os.environ.get("QWEN_IMAGE_EDIT_GUIDANCE_SCALE", "1.0"))
    negative_prompt = os.environ.get("QWEN_IMAGE_EDIT_NEGATIVE_PROMPT", " ")
    result = pipe(
        image=images,
        prompt=edit_prompt,
        generator=torch.Generator(device=_device()).manual_seed(int(seed)),
        true_cfg_scale=true_cfg_scale,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
    )
    if callback:
        callback(steps - 1, steps)
    return result.images[0]
