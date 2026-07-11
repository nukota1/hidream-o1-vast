# JANKU Image Studio

Japanese web application for anime image generation. It runs locally with Docker
for development and on a rented GPU server for production.

## Runtime

- Text-to-image: JANKU v7.77 / Illustrious XL
- Default image editing: ShinoharaHare/Waifu-Inpaint-XL
- Optional image editing: FLUX.1-Kontext-dev and HiDream-O1-Image, loaded only when selected
- Prompt refinement: Qwen/Qwen3.5-9B on CPU, optional per request
- Model cache: private Cloudflare R2 for JANKU
- Generated image storage: private Cloudflare R2
- Backend and UI: Flask
- Public gateway: Cloudflare Worker reverse proxy
- Container registry: GitHub Container Registry

ComfyUI and HiDream O1/I1/E1.1 are not part of the current runtime.

## Local development (RTX 5090)

Use the local Docker Compose configuration for normal development and testing.
It uses the same application image as production, exposes the app at
`http://127.0.0.1:7861`, and keeps all downloaded models in the named Docker
volume `janku-models-local`. Recreating the container does not download models
again.

Waifu-Inpaint-XL is the default editor and fits alongside this application's
single-active-model design on a 32GB GPU. JANKU is unloaded before an edit model
loads, and the editor is unloaded before the next JANKU generation. Reserve at
least 80GB of Docker disk for JANKU, Waifu-Inpaint-XL, and the prompt refiner.
FLUX.1-Kontext-dev and HiDream-O1-Image are intentionally not downloaded until
they are chosen in the web UI; reserve substantially more disk when using either.

1. Start Docker Desktop.
2. Create `.env` from `.env.example` and set the R2 credentials.
3. Create `.env.local` from `.env.local.example` to change non-secret local
   settings when needed.
4. Run the following in PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\local.ps1 start
```

The first startup downloads JANKU from R2, Waifu-Inpaint-XL, and Qwen3.5 from
Hugging Face. It starts the web app immediately while these models are
cached in the background. Check readiness and download progress with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\local.ps1 logs
```

Other commands are `stop`, `restart`, `status`, and `build`.

## JANKU defaults

The v7.77 model guidance recommends Euler or Euler A, 25-30 steps, CFG 3-5,
and Clip Skip 2. The application default is:

```text
Sampler: Euler
Steps: 32
CFG: 5.0
Clip Skip: 2
```

The web UI provides presets for bishoujo visual-novel CG, anime illustration,
manga, light-novel illustration, and custom settings. Every value can be changed
under `詳細設定`.

Japanese input is always translated into compact Illustrious/Danbooru tags.
Turning off `構図と表現をAIで補完` disables inferred embellishment, but does
not disable the required Japanese-to-English tag conversion.

## Vast.ai

Use `Docker ENTRYPOINT` launch mode and expose port `7861`.

Use at least 100GB for Vast.ai `Disk Space (Container + Volume)` with the
default editor. Increase this before selecting FLUX.1-Kontext-dev or
HiDream-O1-Image, because those optional models are downloaded into `/models`.

```text
ghcr.io/nukota1/hidream-o1-image:latest
```

Copy the variables from `deploy/vast/env.vast.example` into the Vast.ai template.
Do not put model weights or secrets in Git.

JANKU is downloaded from:

```text
s3://ai-model-cache/models/JANKUTrainedChenkinNoobai_v777.safetensors
```

Waifu-Inpaint-XL and the prompt refiner are prefetched from Hugging Face.
FLUX.1-Kontext-dev and HiDream-O1-Image are downloaded only when selected in
the image-editing model list.

## Image editing models

The image editing panel offers three workflows:

1. **Waifu-Inpaint-XL**: default anime inpainting workflow. Uploading a mask
   edits white areas and preserves black areas. Without a mask, it uses a
   balanced whole-image edit. The web UI provides an edit-strength slider:
   around 40 favors the source image, 55 is balanced, and 65 or more favors the
   requested change. Use a mask for a large, localized change while preserving
   the rest of the character.
2. **FLUX.1-Kontext-dev**: instruction-based image editing workflow. It is
   downloaded only after selection; its current workflow does not use a mask.
3. **HiDream-O1-Image**: instruction-based editing with the official full
   HiDream model. It is downloaded only after selection; its current workflow
   does not use a mask.

Before the first Waifu or FLUX download, log into Hugging Face, accept each
model's access conditions, create a read token, and set `HF_TOKEN`. A missing
approval or token is reported in the app instead of preventing the server from
starting.

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
