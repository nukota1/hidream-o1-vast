import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path


def backend_headers():
    headers = {}
    shared_secret = os.environ.get("BACKEND_SHARED_SECRET", "").strip()
    if shared_secret:
        headers["X-Backend-Key"] = shared_secret
    return headers


def post_json(url, payload):
    headers = {
        "Content-Type": "application/json",
        **backend_headers(),
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test event-CG generation or background inpainting.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:7861")
    parser.add_argument("--workflow", choices=("character", "compose"), default="compose")
    parser.add_argument(
        "--method",
        choices=("integrated", "inpaint"),
        default="inpaint",
        help="Inpaint preserves the source subject; integrated regenerates a coherent event CG.",
    )
    parser.add_argument("--prompt", help="Override the built-in smoke-test prompt.")
    parser.add_argument("--catalog-prompt", default="", help="Optional catalog tags sent separately from free input.")
    parser.add_argument("--scene-prompt", default="", help="Optional scene input for one-pass story illustration generation.")
    parser.add_argument("--lora-id", default="", help="Optional registered Character LoRA id.")
    parser.add_argument("--lora-weight", type=float, default=0.8)
    parser.add_argument("--reference-image", type=Path, help="Optional IP-Adapter character reference.")
    parser.add_argument("--reference-strength", type=float, default=0.25)
    parser.add_argument("--mask-output", type=Path, help="Write the reusable background mask when returned.")
    parser.add_argument("--metadata-output", type=Path, help="Write the final SSE result metadata as JSON.")
    parser.add_argument("--background-mask-input", type=Path, help="Reuse a previously returned background mask.")
    parser.add_argument("--refine", action="store_true", help="Enable local prompt refinement.")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--locked-character-prompt",
        help="Lock these optimized character and outfit tags during integrated generation.",
    )
    args = parser.parse_args()

    integrated_prompt = args.prompt or (
        "1girl, solo, full body, slender six-head-tall proportions, shoulder-length blonde "
        "wavy hair, crimson red eyes, white long dress, brown ankle boots, holding a transparent "
        "umbrella with cat patterns, rural Japanese country road, rice fields, heavy rain, "
        "puddles, nostalgic evening atmosphere, coherent perspective, wet ground reflections"
    )
    payload = {
        "workflow": args.workflow,
        "mode": "t2i" if args.method == "integrated" else "edit",
        "prompt": integrated_prompt,
        "free_prompt": integrated_prompt,
        "catalog_prompt": args.catalog_prompt,
        "refine_enabled": args.refine,
        "preset": "bishoujo_game",
        "width": args.width,
        "height": args.height,
        "steps": 28,
        "cfg": 5.0,
        "sampler": "euler_a",
        "clip_skip": 2,
        "seed": args.seed,
        "negative_prompt": "low quality, worst quality, text, watermark",
        "style_settings": {
            "anime_strength": 100,
            "line_detail": 100,
            "color_vividness": 91,
            "photoreal_avoidance": 85,
        },
    }
    if args.lora_id:
        payload.update({
            "lora_id": args.lora_id,
            "lora_weight": args.lora_weight,
        })
    if args.reference_image:
        payload.update({
            "reference_image": base64.b64encode(
                args.reference_image.read_bytes()
            ).decode("ascii"),
            "reference_strength": args.reference_strength,
        })
    if args.workflow == "character":
        payload["mode"] = "t2i"
        payload["character_prompt"] = integrated_prompt
        if args.scene_prompt:
            payload["scene_prompt"] = args.scene_prompt
            payload["generation_intent"] = "story_illustration"
    elif args.method == "inpaint":
        background_prompt = args.prompt or (
            "empty rural Japanese country road, rice fields, heavy rain, puddles, "
            "nostalgic evening atmosphere, coherent perspective, wet ground reflections"
        )
        payload.update(
            {
                "prompt": background_prompt,
                "free_prompt": background_prompt,
                "catalog_prompt": args.catalog_prompt,
                "source_image": base64.b64encode(args.source.read_bytes()).decode("ascii"),
                "background_mask_image": (
                    base64.b64encode(args.background_mask_input.read_bytes()).decode("ascii")
                    if args.background_mask_input
                    else ""
                ),
                "editor_model": "waifu_inpaint_xl",
                "edit_scope": "background",
                "edit_strength": 0.85,
            }
        )
    elif args.locked_character_prompt:
        payload.update(
            {
                "lock_character_outfit": True,
                "locked_character_prompt": args.locked_character_prompt,
            }
        )
    start = post_json(f"{args.base_url}/api/generate/start", payload)
    stream_url = f"{args.base_url}/api/generate/stream/{start['job_id']}"
    stream_request = urllib.request.Request(
        stream_url,
        headers=backend_headers(),
        method="GET",
    )
    with urllib.request.urlopen(stream_request, timeout=900) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "status":
                print(event.get("message", ""), flush=True)
            elif event["type"] == "error":
                raise RuntimeError(event["message"])
            elif event["type"] == "done":
                args.output.write_bytes(base64.b64decode(event["image"]))
                if args.mask_output and event.get("background_mask"):
                    args.mask_output.write_bytes(base64.b64decode(event["background_mask"]))
                if args.metadata_output:
                    metadata = {key: value for key, value in event.items() if key not in {"image", "background_mask"}}
                    args.metadata_output.write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                print(f"prompt={event['optimized_prompt']}")
                print(f"output={args.output}")
                return
    raise RuntimeError("Generation stream ended without an image.")


if __name__ == "__main__":
    main()
