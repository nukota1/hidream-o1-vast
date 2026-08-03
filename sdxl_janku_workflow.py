import os
import re
import time
from pathlib import Path

import torch


def _torch_dtype():
    dtype_name = os.environ.get("MODEL_TORCH_DTYPE", "auto").lower()
    if dtype_name == "auto":
        # Animagine's official single-file LPW example uses FP16. In practice
        # it also avoids VAE colour corruption seen with automatic BF16 on
        # recent consumer GPUs.
        if _image_model_family() == "animagine":
            dtype_name = "float16"
        else:
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


def _image_model_family():
    return os.environ.get("IMAGE_MODEL_FAMILY", "janku").strip().lower()


def _animagine_model_path():
    return os.environ.get(
        "ANIMAGINE_MODEL_PATH",
        "/models/checkpoints/animagine-xl-4.0-zero.safetensors",
    )


def _animagine_min_bytes():
    try:
        return int(os.environ.get("ANIMAGINE_MODEL_MIN_BYTES", "6900000000"))
    except ValueError:
        return 6900000000


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


def _wait_for_model_file(path, min_bytes, status_callback=None):
    """Wait for the entrypoint's background Hugging Face download."""
    wait_seconds = int(os.environ.get("MODEL_DOWNLOAD_WAIT_SECONDS", "7200"))
    failure_marker = f"{path}.download_failed"
    waited = 0
    while not _is_complete_file(path, min_bytes):
        if os.path.isfile(failure_marker):
            try:
                detail = Path(failure_marker).read_text(encoding="utf-8").strip()[:1000]
            except (OSError, UnicodeError):
                detail = ""
            message = "Image model download failed during startup. Check the Vast.ai instance log."
            if detail:
                message = f"{message} {detail}"
            raise RuntimeError(message)
        if waited >= wait_seconds:
            raise RuntimeError(f"Timed out waiting for model download: {path}")
        if status_callback and waited % 10 == 0:
            actual = os.path.getsize(path) if os.path.isfile(path) else 0
            status_callback(
                f"Waiting for image model download: {_format_bytes(actual)} / at least {_format_bytes(min_bytes)}"
            )
        time.sleep(5)
        waited += 5


def _load_single_file_or_repo(
    pipeline_cls,
    model_ref,
    dtype,
    single_file_pipeline_cls=None,
    single_file_kwargs=None,
):
    if not model_ref:
        raise RuntimeError("Model path/repo is not configured.")
    kwargs = {"torch_dtype": dtype, "use_safetensors": True}
    if os.path.isfile(model_ref):
        cls = single_file_pipeline_cls or pipeline_cls
        pipe = cls.from_single_file(model_ref, **kwargs, **(single_file_kwargs or {}))
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

    if _image_model_family() == "animagine":
        model_path = _animagine_model_path()
        _wait_for_model_file(model_path, _animagine_min_bytes(), status_callback=status_callback)
        model_label = os.environ.get(
            "IMAGE_MODEL_LABEL",
            "Animagine XL 4.0 Zero",
        ).strip()
        model_config = os.environ.get(
            "ANIMAGINE_MODEL_CONFIG",
            "cagliostrolab/animagine-xl-4.0-zero",
        ).strip()
        print(f"[sdxl] Loading {model_label} model: {model_path}")
        if status_callback:
            status_callback(f"Loading {model_label} into GPU memory")
        return _load_single_file_or_repo(
            AutoPipelineForText2Image,
            model_path,
            _torch_dtype(),
            single_file_pipeline_cls=StableDiffusionXLPipeline,
            single_file_kwargs={
                "config": model_config,
                "custom_pipeline": "lpw_stable_diffusion_xl",
                "add_watermarker": False,
            },
        )

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


def _generator(seed):
    return torch.Generator(device=_device()).manual_seed(int(seed))


def prepare_character_reference(image):
    """Focus IP-Adapter on a typical centered ADV character, not the old scene."""
    reference = image.convert("RGB")
    enabled = os.environ.get(
        "IP_ADAPTER_CHARACTER_CROP",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return reference
    width, height = reference.size
    if width < 64 or height < 64:
        return reference
    return reference.crop((
        int(width * 0.15),
        int(height * 0.05),
        int(width * 0.85),
        int(height * 0.85),
    ))


def _set_sampler(pipe, sampler):
    from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler

    scheduler_cls = EulerAncestralDiscreteScheduler if sampler == "euler_a" else EulerDiscreteScheduler
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, timestep_spacing="linspace")


def fit_prompt_for_sdxl(pipe, prompt):
    """Keep comma-separated priorities inside both SDXL CLIP context windows."""
    tokenizers = [tokenizer for tokenizer in (getattr(pipe, "tokenizer", None), getattr(pipe, "tokenizer_2", None)) if tokenizer]
    if not tokenizers:
        return prompt

    def fits(candidate):
        for tokenizer in tokenizers:
            token_ids = tokenizer(candidate, truncation=False, verbose=False).input_ids
            if len(token_ids) > tokenizer.model_max_length:
                return False
        return True

    quality_tags = ("masterpiece", "high score", "great score", "absurdres")
    parts = [item.strip() for item in prompt.split(",") if item.strip()]
    quality = [part for part in parts if part.lower() in quality_tags]
    content = [part for part in parts if part.lower() not in quality_tags]
    quality_suffix = ", ".join(quality)

    accepted = []
    for part in content:
        candidate = ", ".join([*accepted, part, quality_suffix]).strip(", ")
        if fits(candidate):
            accepted.append(part)
        else:
            continue
    compact = ", ".join([*accepted, *quality])
    if compact:
        print("[sdxl] Prompt was compacted to fit the SDXL CLIP context window")
        return compact

    # A disabled refiner or malformed non-tag prompt may contain no commas.
    # Token-truncate it instead of returning an overlong string unchanged.
    compact = prompt
    for tokenizer in tokenizers:
        token_ids = tokenizer(
            compact,
            truncation=True,
            max_length=tokenizer.model_max_length,
            verbose=False,
        ).input_ids
        compact = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    print("[sdxl] Non-tag prompt was token-truncated to fit the CLIP context window")
    return compact


def generate_with_janku(
    pipe,
    prompt,
    settings,
    callback=None,
    reference_image=None,
):
    steps = int(settings["steps"])
    _set_sampler(pipe, settings["sampler"])

    def on_step_end(_pipe, step, _timestep, callback_kwargs):
        if callback:
            callback(step, steps)
        return callback_kwargs

    pipeline_args = {
        "prompt": prompt,
        "negative_prompt": settings["negative_prompt"],
        "width": int(settings["width"]),
        "height": int(settings["height"]),
        "num_inference_steps": steps,
        "guidance_scale": float(settings["cfg"]),
        "clip_skip": int(settings["clip_skip"]),
        "generator": _generator(settings["seed"]),
        "callback_on_step_end": on_step_end,
    }
    if reference_image is not None:
        pipeline_args["ip_adapter_image"] = prepare_character_reference(reference_image)
    image = pipe(
        **pipeline_args,
    ).images[0]
    return image


def configure_pipeline_reference(
    pipe,
    enabled=False,
    weight=0.25,
    status_callback=None,
):
    """Configure one SDXL IP-Adapter reference without coupling it to LoRA."""
    loaded = bool(getattr(pipe, "_character_reference_adapter_loaded", False))
    if not enabled:
        if loaded:
            pipe.unload_ip_adapter()
            pipe._character_reference_adapter_loaded = False
        return

    if not loaded:
        repo_id = os.environ.get("IP_ADAPTER_MODEL", "h94/IP-Adapter").strip()
        subfolder = os.environ.get("IP_ADAPTER_SUBFOLDER", "sdxl_models").strip()
        weight_name = os.environ.get(
            "IP_ADAPTER_WEIGHT_NAME",
            "ip-adapter-plus_sdxl_vit-h.safetensors",
        ).strip()
        image_encoder_folder = os.environ.get(
            "IP_ADAPTER_IMAGE_ENCODER_FOLDER",
            "models/image_encoder",
        ).strip()
        if status_callback:
            status_callback(
                "キャラクター参照アダプターを準備しています。初回のみダウンロードします"
            )
        pipe.load_ip_adapter(
            repo_id,
            subfolder=subfolder,
            weight_name=weight_name,
            image_encoder_folder=image_encoder_folder,
        )
        pipe._character_reference_adapter_loaded = True

    pipe.set_ip_adapter_scale(max(0.0, min(1.0, float(weight))))


def configure_pipeline_loras(pipe, adapters=None):
    """Load independent Character and Style LoRAs on the shared SDXL pipeline."""
    try:
        pipe.unload_lora_weights()
    except (AttributeError, RuntimeError, ValueError):
        pass
    active = [adapter for adapter in (adapters or []) if adapter.get("weights_path")]
    if not active:
        return
    adapter_names = []
    adapter_weights = []
    for index, adapter in enumerate(active):
        path = Path(adapter["weights_path"])
        if not path.is_file():
            raise RuntimeError(f"LoRA weights do not exist: {path}")
        requested_name = str(
            adapter.get("adapter_name") or f"user_adapter_{index}"
        )
        adapter_name = re.sub(r"[^A-Za-z0-9_-]+", "_", requested_name)
        if adapter_name in adapter_names:
            adapter_name = f"{adapter_name}_{index}"
        pipe.load_lora_weights(
            str(path.parent),
            weight_name=path.name,
            adapter_name=adapter_name,
        )
        adapter_names.append(adapter_name)
        adapter_weights.append(float(adapter.get("weight", 1.0)))
    try:
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
    except TypeError:
        pipe.set_adapters(adapter_names, adapter_weights)


def configure_pipeline_lora(pipe, weights_path=None, weight=0.8):
    """Backward-compatible single-adapter wrapper."""
    configure_pipeline_loras(
        pipe,
        [{
            "weights_path": weights_path,
            "weight": weight,
            "adapter_name": "character_asset",
        }] if weights_path else [],
    )
