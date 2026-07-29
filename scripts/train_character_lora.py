#!/usr/bin/env python3
import argparse
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


def emit(event_type, **values):
    print(json.dumps({"type": event_type, **values}, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an SDXL LoRA adapter.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trigger-word", required=True)
    parser.add_argument("--category", default="character")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--random-flip", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def category_prompt(category):
    return {
        "character": "1girl, solo, character",
        "style": "anime illustration style",
        "pose": "anime character pose",
        "background": "anime visual novel background",
    }.get(category, "1girl, solo, character")


def load_caption_manifest(dataset):
    path = dataset / "captions.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("captions.json must be an object keyed by image filename.")
    return {
        str(name): str(caption).strip()
        for name, caption in value.items()
        if str(caption).strip()
    }


def load_training_records(dataset, trigger_word, category):
    manifest = load_caption_manifest(dataset)
    base_prompt = f"{trigger_word}, {category_prompt(category)}"
    records = []
    for path in sorted(
        candidate for candidate in dataset.iterdir()
        if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ):
        sidecar = path.with_suffix(".txt")
        caption = (
            sidecar.read_text(encoding="utf-8").strip()
            if sidecar.is_file()
            else manifest.get(path.name, "")
        )
        caption = caption or base_prompt
        caption_tags = {
            tag.strip().lower()
            for tag in caption.split(",")
            if tag.strip()
        }
        if trigger_word.lower() not in caption_tags:
            caption = f"{trigger_word}, {caption}"
        records.append({"path": path, "caption": caption})
    return records


def target_bucket_size(width, height, resolution):
    aspect = max(1.0 / 4.0, min(4.0, width / max(1, height)))
    target_area = resolution * resolution
    target_width = max(256, int(round(math.sqrt(target_area * aspect) / 64.0) * 64))
    target_height = max(256, int(round(math.sqrt(target_area / aspect) / 64.0) * 64))
    return target_width, target_height


def load_pixels(path, resolution, random_flip=False):
    with Image.open(path) as loaded:
        image = ImageOps.exif_transpose(loaded).convert("RGB")
    width, height = image.size
    target_width, target_height = target_bucket_size(width, height, resolution)
    scale = max(target_width / width, target_height / height)
    resized = image.resize(
        (
            max(target_width, round(width * scale)),
            max(target_height, round(height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - target_width) // 2)
    top = max(0, (resized.height - target_height) // 2)
    image = resized.crop((
        left,
        top,
        left + target_width,
        top + target_height,
    ))
    if random_flip and random.random() < 0.5:
        image = ImageOps.mirror(image)
    pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    return tensor, (target_height, target_width)


def encode_prompt(pipe, prompt, device):
    prompt_embeds_list = []
    pooled_prompt_embeds = None
    for tokenizer, text_encoder in (
        (pipe.tokenizer, pipe.text_encoder),
        (pipe.tokenizer_2, pipe.text_encoder_2),
    ):
        text_input_ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        with torch.no_grad():
            encoded = text_encoder(
                text_input_ids,
                output_hidden_states=True,
                return_dict=False,
            )
        pooled_prompt_embeds = encoded[0]
        prompt_embeds_list.append(encoded[-1][-2])
    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def main():
    args = parse_args()
    dataset = Path(args.dataset)
    output = Path(args.output)
    records = load_training_records(dataset, args.trigger_word, args.category)
    if not records:
        raise RuntimeError("No training images were found.")
    if args.dry_run:
        bucket_sizes = set()
        for record in records:
            with Image.open(record["path"]) as image:
                bucket_sizes.add(
                    target_bucket_size(*image.size, args.resolution)
                )
        output.mkdir(parents=True, exist_ok=True)
        (output / "dry-run.json").write_text(
            json.dumps({
                "image_count": len(records),
                "trigger_word": args.trigger_word,
                "steps": args.steps,
                "rank": args.rank,
                "resolution": args.resolution,
                "caption_count": len({
                    record["caption"] for record in records
                }),
                "bucket_sizes": sorted(bucket_sizes),
                "random_flip": args.random_flip,
                "learning_rate": args.learning_rate,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        emit("done", dry_run=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires a CUDA GPU.")

    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    weight_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    load_kwargs = {
        "torch_dtype": weight_dtype,
        "use_safetensors": True,
        "add_watermarker": False,
    }
    emit("status", message="SDXL基盤モデルを読み込んでいます")
    if Path(args.base_model).is_file():
        if args.config:
            load_kwargs["config"] = args.config
        pipe = StableDiffusionXLPipeline.from_single_file(
            args.base_model,
            **load_kwargs,
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.base_model,
            **load_kwargs,
        )
    pipe.to(device)
    unet = pipe.unet
    vae = pipe.vae
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    for module in (unet, vae, pipe.text_encoder, pipe.text_encoder_2):
        module.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    for parameter in unet.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)
    try:
        unet.enable_gradient_checkpointing()
    except Exception:
        pass

    prompt_conditions = {}
    for prompt in sorted({record["caption"] for record in records}):
        prompt_conditions[prompt] = encode_prompt(pipe, prompt, device)
    del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
    gc.collect()
    torch.cuda.empty_cache()

    trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=weight_dtype == torch.float16)
    unet.train()
    vae.eval()
    emit(
        "status",
        message=(
            f"{len(records)}枚・{len(prompt_conditions)}種類の"
            "キャプションからLoRA学習を開始します"
        ),
    )

    def save_adapter(destination):
        destination.mkdir(parents=True, exist_ok=True)
        state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unet)
        )
        StableDiffusionXLPipeline.save_lora_weights(
            destination,
            unet_lora_layers=state_dict,
            safe_serialization=True,
        )

    for step in range(args.steps):
        record = records[step % len(records)]
        pixels, (target_height, target_width) = load_pixels(
            record["path"],
            args.resolution,
            random_flip=args.random_flip,
        )
        pixels = pixels.to(device=device, dtype=weight_dtype)
        prompt_embeds, pooled_prompt_embeds = prompt_conditions[record["caption"]]
        time_ids = torch.tensor(
            [[
                target_height,
                target_width,
                0,
                0,
                target_height,
                target_width,
            ]],
            device=device,
            dtype=pooled_prompt_embeds.dtype,
        )
        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (1,),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=weight_dtype):
            model_pred = unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": time_ids,
                },
                return_dict=False,
            )[0]
            target = (
                noise_scheduler.get_velocity(latents, noise, timesteps)
                if noise_scheduler.config.prediction_type == "v_prediction"
                else noise
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 0 or (step + 1) % max(1, args.steps // 100) == 0:
            emit(
                "progress",
                step=step + 1,
                total=args.steps,
                loss=round(float(loss.detach().item()), 6),
            )
        if (
            args.save_every > 0
            and (step + 1) % args.save_every == 0
            and (step + 1) < args.steps
        ):
            emit("status", message=f"{step + 1} stepの中間重みを保存しています")
            save_adapter(output / f"checkpoint-{step + 1}")

    emit("status", message="LoRA重みを保存しています")
    save_adapter(output)
    (output / "training.json").write_text(
        json.dumps({
            "trigger_word": args.trigger_word,
            "category": args.category,
            "image_count": len(records),
            "steps": args.steps,
            "rank": args.rank,
            "resolution": args.resolution,
            "caption_count": len(prompt_conditions),
            "random_flip": args.random_flip,
            "learning_rate": args.learning_rate,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    emit("done", weight_file="pytorch_lora_weights.safetensors")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("error", message=str(exc))
        raise
