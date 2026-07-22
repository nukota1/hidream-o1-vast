import argparse
import base64
import json
import urllib.request
from pathlib import Path


def post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
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
        default="integrated",
        help="Integrated generates one coherent image; inpaint keeps the source subject.",
    )
    parser.add_argument("--prompt", help="Override the built-in smoke-test prompt.")
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
    if args.workflow == "character":
        payload["mode"] = "t2i"
    elif args.method == "inpaint":
        payload.update(
            {
                "prompt": (
                    "empty rural Japanese country road, rice fields, heavy rain, puddles, "
                    "nostalgic evening atmosphere, coherent perspective, wet ground reflections"
                ),
                "source_image": base64.b64encode(args.source.read_bytes()).decode("ascii"),
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
    with urllib.request.urlopen(stream_url, timeout=900) as response:
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
                print(f"prompt={event['optimized_prompt']}")
                print(f"output={args.output}")
                return
    raise RuntimeError("Generation stream ended without an image.")


if __name__ == "__main__":
    main()
