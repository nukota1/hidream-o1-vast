import os

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

    model_ref = os.environ.get("JANKU_MODEL_PATH") or os.environ.get("JANKU_MODEL_REPO")
    print(f"[sdxl] Loading JANKU text-to-image model: {model_ref}")
    return _load_single_file_or_repo(AutoPipelineForText2Image, model_ref, _torch_dtype())


def load_inpaint_pipeline():
    from diffusers import AutoPipelineForInpainting

    model_ref = os.environ.get("SDXL_INPAINT_MODEL_PATH") or os.environ.get("SDXL_INPAINT_MODEL_REPO")
    print(f"[sdxl] Loading inpaint model: {model_ref}")
    return _load_single_file_or_repo(AutoPipelineForInpainting, model_ref, _torch_dtype())


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


def inpaint_with_sdxl(pipe, prompt, image_path, mask_path, width, height, seed, callback=None):
    source = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L") if mask_path else _full_mask(source.size)
    steps = int(os.environ.get("SDXL_INPAINT_STEPS", "28"))
    guidance_scale = float(os.environ.get("SDXL_INPAINT_CFG_SCALE", "5"))
    strength = float(os.environ.get("SDXL_INPAINT_STRENGTH", "0.75"))
    result = pipe(
        prompt=prompt,
        negative_prompt=_negative_prompt(),
        image=source,
        mask_image=mask,
        width=width,
        height=height,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=_generator(seed),
    ).images[0]
    if callback:
        callback(steps - 1, steps)
    return result
