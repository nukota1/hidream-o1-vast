#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdxl_janku_workflow import configure_pipeline_loras, load_janku_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Load the active SDXL pipeline and attach an existing LoRA."
    )
    parser.add_argument("weights")
    parser.add_argument("--weight", type=float, default=0.8)
    parser.add_argument("--style-weights", default="")
    parser.add_argument("--style-weight", type=float, default=0.6)
    args = parser.parse_args()

    pipe = load_janku_pipeline(status_callback=print)
    adapters = [{
        "weights_path": args.weights,
        "weight": args.weight,
        "adapter_name": "character_asset",
    }]
    if args.style_weights:
        adapters.append({
            "weights_path": args.style_weights,
            "weight": args.style_weight,
            "adapter_name": "style_asset",
        })
    configure_pipeline_loras(pipe, adapters)
    adapters = (
        pipe.get_active_adapters()
        if hasattr(pipe, "get_active_adapters")
        else [adapter["adapter_name"] for adapter in adapters]
    )
    print(f"active_adapters={adapters}")


if __name__ == "__main__":
    main()
