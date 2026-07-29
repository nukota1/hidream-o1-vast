#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdxl_janku_workflow import (
    configure_pipeline_loras,
    fit_prompt_for_sdxl,
    generate_with_janku,
    load_janku_pipeline,
)


DEFAULT_PROMPT = (
    "1girl, solo, nkt_chr001, petite proportions, youthful face, blonde hair, "
    "pink eyes, medium hair, small centered back hair bun, "
    "white cat-shaped hair ornament near bangs, red school blazer, white blouse, "
    "plaid pleated skirt, "
    "standing at a seaside train station, ocean, sunset, waving, full body, "
    "visual novel event cg, masterpiece, high score, great score, absurdres"
)

DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, "
    "fewer digits, cropped, worst quality, low quality, watermark, signature, "
    "multiple girls, black t-shirt, cat print, denim shorts, white background"
)


def parse_lora(value):
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("LoRA must use LABEL=/path/to/weights.")
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "-", label.strip()).strip("-")
    if not safe_label:
        raise argparse.ArgumentTypeError("LoRA label is empty.")
    return safe_label, Path(path.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic baseline and LoRA comparison images."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora", action="append", type=parse_lora, default=[])
    parser.add_argument("--weight", type=float, default=0.8)
    parser.add_argument("--style-lora", type=Path)
    parser.add_argument("--style-weight", type=float, default=0.6)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1152)
    args = parser.parse_args()

    missing = [str(path) for _, path in args.lora if not path.is_file()]
    if args.style_lora and not args.style_lora.is_file():
        missing.append(str(args.style_lora))
    if missing:
        raise FileNotFoundError(f"Missing LoRA weights: {', '.join(missing)}")

    args.output.mkdir(parents=True, exist_ok=True)
    pipe = load_janku_pipeline(status_callback=print)
    prompt = fit_prompt_for_sdxl(pipe, args.prompt)
    settings = {
        "steps": args.steps,
        "sampler": "euler_a",
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "cfg": 5.0,
        "clip_skip": 2,
        "seed": args.seed,
    }
    variants = [("baseline", [])]
    if args.style_lora:
        variants.append(("style-only", [{
            "weights_path": args.style_lora,
            "weight": args.style_weight,
            "adapter_name": "style_asset",
        }]))
    for label, weights in args.lora:
        adapters = [{
            "weights_path": weights,
            "weight": args.weight,
            "adapter_name": "character_asset",
        }]
        if args.style_lora:
            adapters.append({
                "weights_path": args.style_lora,
                "weight": args.style_weight,
                "adapter_name": "style_asset",
            })
        variants.append((label, adapters))

    results = []
    for label, adapters in variants:
        print(f"[evaluation] generating {label}", flush=True)
        configure_pipeline_loras(pipe, adapters)
        image = generate_with_janku(pipe, prompt, settings)
        output_path = args.output / f"{label}.png"
        image.save(output_path, format="PNG")
        results.append({
            "label": label,
            "adapters": [
                {
                    "path": str(adapter["weights_path"]),
                    "weight": adapter["weight"],
                    "name": adapter["adapter_name"],
                }
                for adapter in adapters
            ],
            "output": str(output_path),
        })

    (args.output / "evaluation.json").write_text(
        json.dumps({
            "prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "steps": args.steps,
            "width": args.width,
            "height": args.height,
            "weight": args.weight,
            "style_weight": args.style_weight,
            "results": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[evaluation] results={args.output}", flush=True)


if __name__ == "__main__":
    main()
