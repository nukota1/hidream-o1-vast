# JANKU Image Studio

Japanese web application for anime image generation on a rented GPU server.

## Runtime

- Text-to-image: JANKU v7.77 / Illustrious XL
- Image editing: Qwen/Qwen-Image-Edit-2511
- Prompt refinement: Qwen/Qwen3.5-9B on CPU, optional per request
- Model cache: private Cloudflare R2 for JANKU
- Generated image storage: private Cloudflare R2
- Backend and UI: Flask
- Public gateway: Cloudflare Worker reverse proxy
- Container registry: GitHub Container Registry

ComfyUI and HiDream O1/I1/E1.1 are not part of the current runtime.

## JANKU defaults

The v7.77 model guidance recommends Euler or Euler A, 25-30 steps, CFG 3-5,
and Clip Skip 2. The application default is:

```text
Sampler: Euler A
Steps: 28
CFG: 4.5
Clip Skip: 2
```

The web UI provides presets for bishoujo visual-novel CG, anime illustration,
manga, light-novel illustration, and custom settings. Every value can be changed
under `詳細設定`.

## Vast.ai

Use `Docker ENTRYPOINT` launch mode and expose port `7861`.

```text
ghcr.io/nukota1/hidream-o1-image:latest
```

Copy the variables from `deploy/vast/env.vast.example` into the Vast.ai template.
Do not put model weights or secrets in Git.

JANKU is downloaded from:

```text
s3://ai-model-cache/models/JANKUTrainedChenkinNoobai_v777.safetensors
```

Qwen Image Edit and the prompt refiner are downloaded from Hugging Face.

## Storage

Generated images are saved only when the user presses the R2 save button:

```text
generated/YYYY/MM/DD/<uuid>.png
generated/YYYY/MM/DD/<uuid>.json
```

The JSON object contains the original prompt, final prompt, selected preset,
refinement state, generation settings, and object keys.

## Build

Pushing `main` triggers `.github/workflows/build-ghcr.yml` and publishes both
`latest` and commit-SHA tags to GHCR.
