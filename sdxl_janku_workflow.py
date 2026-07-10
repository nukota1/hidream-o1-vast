import os
import time

import torch
from PIL import Image


def _torch_dtype():
    dtype_name = os.environ.get("MODEL_TORCH_DTYPE", "auto").lower()
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


def _min_janku_bytes():
    try:
        return int(os.environ.get("JANKU_MODEL_MIN_BYTES", "6900000000"))
    except ValueError:
        return 6900000000


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


def _format_bytes(num_bytes):
    if num_bytes >= 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.0f} MB"
    return f"{num_bytes} bytes"


def _download_status_message(path, tmp_path, min_bytes, waited):
    current_path = path if os.path.isfile(path) else tmp_path
    actual_bytes = os.path.getsize(current_path) if os.path.isfile(current_path) else 0
    target = f" / at least {_format_bytes(min_bytes)}" if min_bytes > 0 else ""
    return (
        f"Waiting for JANKU model download: {_format_bytes(actual_bytes)}{target} "
        f"({waited}s elapsed)"
    )


def _r2_config():
    bucket = os.environ.get("JANKU_R2_BUCKET", "").strip()
    key = os.environ.get("JANKU_R2_KEY", "").strip()
    endpoint_url = os.environ.get("R2_ENDPOINT_URL", "").strip()
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    if not bucket or not key:
        return None
    missing = [
        name
        for name, value in (
            ("R2_ENDPOINT_URL", endpoint_url),
            ("R2_ACCESS_KEY_ID", access_key),
            ("R2_SECRET_ACCESS_KEY", secret_key),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing R2 settings for JANKU model download: {', '.join(missing)}")
    return bucket, key, endpoint_url, access_key, secret_key


def _download_from_r2(path, tmp_path, min_bytes, status_callback=None):
    import boto3
    from botocore.config import Config

    bucket, key, endpoint_url, access_key, secret_key = _r2_config()
    if status_callback:
        status_callback(f"Downloading JANKU model from R2: s3://{bucket}/{key}")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 10, "mode": "standard"},
        ),
    )

    transferred = 0
    next_status = 512 * 1024 * 1024

    def progress(bytes_amount):
        nonlocal transferred, next_status
        transferred += bytes_amount
        if status_callback and transferred >= next_status:
            status_callback(
                f"Downloading JANKU model from R2: {_format_bytes(transferred)}"
            )
            next_status += 512 * 1024 * 1024

    with open(tmp_path, "wb") as f:
        client.download_fileobj(bucket, key, f, Callback=progress)

    if not _is_complete_file(tmp_path, min_bytes):
        raise RuntimeError(f"Downloaded R2 model is smaller than expected: {tmp_path}")
    os.replace(tmp_path, path)
    if status_callback:
        status_callback("JANKU model download from R2 complete; loading model")


def _download_if_needed(path, status_callback=None):
    if not path:
        return
    if _r2_config() is None:
        raise RuntimeError("JANKU_R2_BUCKET and JANKU_R2_KEY are required.")
    min_bytes = _min_janku_bytes()
    if _is_complete_file(path, min_bytes):
        return
    if os.path.isfile(path):
        os.remove(path)
    tmp_path = f"{path}.part"
    wait_seconds = int(os.environ.get("MODEL_DOWNLOAD_WAIT_SECONDS", "7200"))
    waited = 0
    while os.path.isfile(tmp_path) and not os.path.isfile(path):
        if waited == 0:
            print(f"[sdxl] Waiting for background download: {path}")
        if status_callback and waited % 10 == 0:
            status_callback(_download_status_message(path, tmp_path, min_bytes, waited))
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
    if status_callback:
        status_callback("Downloading JANKU model from R2 before generation")
    try:
        _download_from_r2(path, tmp_path, min_bytes, status_callback=status_callback)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _load_single_file_or_repo(pipeline_cls, model_ref, dtype, single_file_pipeline_cls=None):
    if not model_ref:
        raise RuntimeError("Model path/repo is not configured.")
    kwargs = {"torch_dtype": dtype, "use_safetensors": True}
    if os.path.isfile(model_ref):
        cls = single_file_pipeline_cls or pipeline_cls
        pipe = cls.from_single_file(model_ref, **kwargs)
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


def load_janku_pipeline(status_callback=None):
    from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline

    model_path = os.environ.get("JANKU_MODEL_PATH", "")
    _download_if_needed(model_path, status_callback=status_callback)
    model_ref = model_path
    print(f"[sdxl] Loading JANKU text-to-image model: {model_ref}")
    if status_callback:
        status_callback("Loading JANKU model into GPU memory")
    return _load_single_file_or_repo(
        AutoPipelineForText2Image,
        model_ref,
        _torch_dtype(),
        single_file_pipeline_cls=StableDiffusionXLPipeline,
    )


def load_qwen_edit_pipeline():
    from diffusers import QwenImageEditPlusPipeline

    model_ref = os.environ.get("QWEN_IMAGE_EDIT_MODEL", "Qwen/Qwen-Image-Edit-2511")
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


def _set_sampler(pipe, sampler):
    from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler

    scheduler_cls = EulerAncestralDiscreteScheduler if sampler == "euler_a" else EulerDiscreteScheduler
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, timestep_spacing="linspace")


def generate_with_janku(pipe, prompt, settings, callback=None):
    steps = int(settings["steps"])
    _set_sampler(pipe, settings["sampler"])

    def on_step_end(_pipe, step, _timestep, callback_kwargs):
        if callback:
            callback(step, steps)
        return callback_kwargs

    image = pipe(
        prompt=prompt,
        negative_prompt=settings["negative_prompt"],
        width=int(settings["width"]),
        height=int(settings["height"]),
        num_inference_steps=steps,
        guidance_scale=float(settings["cfg"]),
        clip_skip=int(settings["clip_skip"]),
        generator=_generator(settings["seed"]),
        callback_on_step_end=on_step_end,
    ).images[0]
    return image


def edit_with_qwen_image_edit(pipe, prompt, image_path, mask_path, seed, callback=None):
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

    def on_step_end(_pipe, step, _timestep, callback_kwargs):
        if callback:
            callback(step, steps)
        return callback_kwargs

    result = pipe(
        image=images,
        prompt=edit_prompt,
        generator=torch.Generator(device=_device()).manual_seed(int(seed)),
        true_cfg_scale=true_cfg_scale,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        callback_on_step_end=on_step_end,
    )
    return result.images[0]
