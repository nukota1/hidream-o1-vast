import os
import sys
import json
import math
from collections import defaultdict

import torch
from PIL import Image
from safetensors.torch import safe_open


def _add_repo_path(env_name, default_path):
    repo_dir = os.environ.get(env_name, default_path)
    if repo_dir and os.path.isdir(repo_dir) and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    return repo_dir


def _torch_dtype():
    dtype_name = os.environ.get("HIDREAM_TORCH_DTYPE", "auto").lower()
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
    }.get(dtype_name, torch.bfloat16)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_i1_pipeline():
    _add_repo_path("HIDREAM_I1_REPO_DIR", "/workspace/third_party/HiDream-I1")
    try:
        from hi_diffusers import HiDreamImagePipeline
        from hi_diffusers import HiDreamImageTransformer2DModel
        from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
        from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "HiDream-I1 code is not available. Set HIDREAM_I1_REPO_DIR or let the Vast entrypoint clone it."
        ) from exc

    model_id = os.environ.get("HIDREAM_I1_MODEL_REPO", "HiDream-ai/HiDream-I1-Dev")
    llm_id = os.environ.get("HIDREAM_I1_LLM_REPO", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    dtype = _torch_dtype()
    scheduler = FlashFlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=float(os.environ.get("HIDREAM_I1_SHIFT", "6.0")),
        use_dynamic_shifting=False,
    )
    print(f"[i1] Loading generation model: {model_id}")
    print(f"[i1] Loading text encoder: {llm_id}")
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(llm_id, use_fast=False)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        llm_id,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=dtype,
    ).to(_device())
    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(_device())
    pipe = HiDreamImagePipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=dtype,
    ).to(_device(), dtype)
    pipe.transformer = transformer
    return pipe


def _resize_e11_image(pil_image, image_size=1024):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    m = 16
    width, height = pil_image.width, pil_image.height
    s_max = image_size * image_size
    scale = math.sqrt(s_max / (width * height))
    new_sizes = [
        (round(width * scale) // m * m, round(height * scale) // m * m),
        (round(width * scale) // m * m, math.floor(height * scale) // m * m),
        (math.floor(width * scale) // m * m, round(height * scale) // m * m),
        (math.floor(width * scale) // m * m, math.floor(height * scale) // m * m),
    ]
    new_sizes = sorted(new_sizes, key=lambda x: x[0] * x[1], reverse=True)
    for new_size in new_sizes:
        if new_size[0] * new_size[1] <= s_max:
            break
    s1 = width / new_size[0]
    s2 = height / new_size[1]
    if s1 < s2:
        pil_image = pil_image.resize([new_size[0], round(height / s1)], resample=Image.BICUBIC)
        top = (round(height / s1) - new_size[1]) // 2
        pil_image = pil_image.crop((0, top, new_size[0], top + new_size[1]))
    else:
        pil_image = pil_image.resize([round(width / s2), new_size[1]], resample=Image.BICUBIC)
        left = (round(width / s2) - new_size[0]) // 2
        pil_image = pil_image.crop((left, 0, left + new_size[0], new_size[1]))
    return pil_image


def _load_sharded_safetensors(directory):
    with open(f"{directory}/diffusion_pytorch_model.safetensors.index.json", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    shards = defaultdict(list)
    for name, file_name in weight_map.items():
        shards[file_name].append(name)
    state_dict = {}
    for file_name, names in shards.items():
        with safe_open(f"{directory}/{file_name}", framework="pt", device="cpu") as f:
            state_dict.update({name: f.get_tensor(name) for name in names})
    return state_dict


def load_e11_pipeline():
    _add_repo_path("HIDREAM_E11_REPO_DIR", "/workspace/third_party/HiDream-E1")
    try:
        from huggingface_hub import snapshot_download
        from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
        from diffusers import HiDreamImageTransformer2DModel
        from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "HiDream-E1.1 code is not available. Set HIDREAM_E11_REPO_DIR or let the Vast entrypoint clone it."
        ) from exc

    base_id = os.environ.get("HIDREAM_E11_BASE_REPO", "HiDream-ai/HiDream-I1-Full")
    edit_id = os.environ.get("HIDREAM_E11_MODEL_REPO", "HiDream-ai/HiDream-E1-1")
    llm_id = os.environ.get("HIDREAM_E11_LLM_REPO", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    dtype = _torch_dtype()
    print(f"[e1.1] Loading editing base model: {base_id}")
    print(f"[e1.1] Loading editing weights: {edit_id}")
    print(f"[e1.1] Loading text encoder: {llm_id}")
    edit_path = edit_id if os.path.isdir(edit_id) else snapshot_download(edit_id)
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(llm_id)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        llm_id,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=dtype,
    )
    transformer = HiDreamImageTransformer2DModel.from_pretrained(base_id, subfolder="transformer")
    transformer.max_seq = 8192
    src_dict = transformer.state_dict()
    edit_dict = _load_sharded_safetensors(os.path.join(edit_path, "transformer"))
    reload_keys = {"editing": src_dict, "refine": edit_dict}
    transformer.load_state_dict(edit_dict, strict=True)
    pipe = HiDreamImageEditingPipeline.from_pretrained(
        base_id,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=dtype,
        transformer=transformer,
    ).to(_device(), dtype)
    return pipe, reload_keys


def generate_with_i1(pipe, prompt, width, height, seed, callback=None):
    generator = torch.Generator(device=_device()).manual_seed(seed)
    steps = int(os.environ.get("HIDREAM_I1_STEPS", "28"))
    guidance_scale = float(os.environ.get("HIDREAM_I1_GUIDANCE_SCALE", "0"))
    kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "num_inference_steps": steps,
        "generator": generator,
    }
    image = pipe(**kwargs).images[0]
    if callback:
        callback(steps - 1, steps)
    return image


def edit_with_e11(pipe_bundle, prompt, image_path, width, height, seed, callback=None):
    pipe, reload_keys = pipe_bundle
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size
    image_size = int(os.environ.get("HIDREAM_E11_IMAGE_SIZE", "1024"))
    image = _resize_e11_image(original_image, image_size=image_size)
    generator = torch.Generator(device=_device()).manual_seed(seed)
    steps = int(os.environ.get("HIDREAM_E11_STEPS", "28"))
    guidance_scale = float(os.environ.get("HIDREAM_E11_GUIDANCE_SCALE", "3"))
    image_guidance_scale = float(os.environ.get("HIDREAM_E11_IMAGE_GUIDANCE_SCALE", "1.5"))
    refine_strength = float(os.environ.get("HIDREAM_E11_REFINE_STRENGTH", "0.3"))
    negative_prompt = os.environ.get("HIDREAM_E11_NEGATIVE_PROMPT", "low quality, blurry, distorted")
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": image,
        "guidance_scale": guidance_scale,
        "image_guidance_scale": image_guidance_scale,
        "num_inference_steps": steps,
        "generator": generator,
        "refine_strength": refine_strength,
        "reload_keys": reload_keys,
        "clip_cfg_norm": True,
    }
    result = pipe(**kwargs)
    if callback:
        callback(steps - 1, steps)
    return result.images[0].resize(original_size)
